"""
Single-image inference for GeoGuessr AI.

Loads a trained checkpoint and predicts the geographic location of
a street-level image provided as a file path or URL.

Outputs:
  - Predicted lat/lng (regression head)
  - Top-5 candidate H3 cells with confidence scores
  - Estimated country name (via reverse geocoding)

Usage:
    python src/inference/predict.py --checkpoint checkpoints/best_model.pth \
        --image path/to/street.jpg

    python src/inference/predict.py --checkpoint checkpoints/best_model.pth \
        --image https://example.com/street.jpg \
        --true_lat 48.8566 --true_lng 2.3522
"""

import argparse
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import requests
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_val_transform
from src.models.classifier import build_model
from src.models.losses import haversine_distance
from src.training.config import PROCESSED_DIR


def load_image_from_path(path: Path) -> Image.Image:
    """Load image from local file path."""
    return Image.open(path).convert("RGB")


def load_image_from_url(url: str, timeout: int = 15) -> Image.Image:
    """
    Download and load an image from a URL.

    We validate that the content type is an image to avoid processing
    HTML error pages or other non-image responses.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError(f"URL does not point to an image (Content-Type: {content_type})")

    return Image.open(io.BytesIO(response.content)).convert("RGB")


def load_image(source: str) -> Image.Image:
    """
    Load an image from either a file path or URL.

    Auto-detects based on whether source starts with http/https.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_image_from_url(source)
    return load_image_from_path(Path(source))


def reverse_geocode_country(lat: float, lng: float) -> str:
    """
    Estimates country name from coordinates using the Nominatim geocoding API.

    Uses a free public API with attribution. Falls back gracefully if the
    API is unavailable (e.g., offline or rate limited).
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat, "lon": lng,
            "format": "json", "zoom": 3,  # zoom=3 returns country level
        }
        headers = {"User-Agent": "GeoGuessrAI/1.0 (educational project)"}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("address", {}).get("country", "Unknown")
    except Exception:
        return "Unknown (geocoding unavailable)"


class GeoPredictor:
    """
    Inference engine for geographic location prediction.

    Loads the model once and exposes a predict() method for single-image inference.
    Designed to be used by both the CLI script and the Gradio demo.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        processed_dir: Path = PROCESSED_DIR,
        device: Optional[torch.device] = None,
        image_size: int = 224,
    ):
        """
        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            processed_dir: Directory containing label_encoder.pkl.
            device: Inference device. Defaults to CUDA if available.
            image_size: Input image size (must match training size).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.image_size = image_size

        # ── Load label encoder ─────────────────────────────────────────────
        label_encoder_path = processed_dir / "label_encoder.pkl"
        with open(label_encoder_path, "rb") as f:
            self.label_map = pickle.load(f)

        self.idx_to_cell = {
            int(k): v for k, v in self.label_map["idx_to_cell"].items()
        }
        self.cell_centers = self.label_map["cell_centers"]
        self.num_classes = self.label_map["num_classes"]

        # ── Load model ─────────────────────────────────────────────────────
        self.model = build_model(
            num_cells=self.num_classes,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.model.eval()

        # ── Image preprocessing ────────────────────────────────────────────
        self.transform = get_val_transform(image_size)

        print(f"GeoPredictor ready on {device}")

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        """
        Predict the geographic location of a street-level image.

        Args:
            image: PIL Image in RGB format.
            top_k: Number of top candidate cells to return.

        Returns:
            dict with:
                pred_lat, pred_lng: Regression-head coordinates (degrees)
                top_k_cells: List of {cell_id, confidence, lat, lng, country}
                country: Estimated country name for the top prediction
        """
        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            cell_logits, pred_coords = self.model(tensor)

        # Regression coordinates
        pred_lat = pred_coords[0, 0].item()
        pred_lng = pred_coords[0, 1].item()

        # Classification: top-K cells with confidence scores
        probs = F.softmax(cell_logits[0], dim=0)
        top_probs, top_indices = probs.topk(top_k)

        top_k_cells = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            cell_id = self.idx_to_cell[idx]
            lat, lng = self.cell_centers[cell_id]
            top_k_cells.append({
                "cell_id": cell_id,
                "confidence": round(prob, 4),
                "cell_center_lat": round(lat, 4),
                "cell_center_lng": round(lng, 4),
                "rank": len(top_k_cells) + 1,
            })

        # Reverse geocode the top prediction
        country = reverse_geocode_country(pred_lat, pred_lng)

        return {
            "pred_lat": round(pred_lat, 4),
            "pred_lng": round(pred_lng, 4),
            "country": country,
            "top_k_cells": top_k_cells,
        }

    def predict_with_error(
        self, image: Image.Image, true_lat: float, true_lng: float, top_k: int = 5
    ) -> dict:
        """
        Predict and compute error against known ground truth.

        Returns all prediction fields plus distance_km.
        """
        result = self.predict(image, top_k=top_k)

        pred_coords = torch.tensor([[result["pred_lat"], result["pred_lng"]]])
        true_coords = torch.tensor([[true_lat, true_lng]])
        distance = haversine_distance(pred_coords, true_coords, reduction="none")[0].item()

        result["true_lat"] = true_lat
        result["true_lng"] = true_lng
        result["distance_km"] = round(distance, 1)

        return result


def main():
    parser = argparse.ArgumentParser(description="Predict geographic location from street image")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "best_model.pth",
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to image file or image URL"
    )
    parser.add_argument("--true_lat", type=float, default=None, help="Ground truth latitude")
    parser.add_argument("--true_lng", type=float, default=None, help="Ground truth longitude")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    # ── Load image ─────────────────────────────────────────────────────────
    print(f"Loading image: {args.image}")
    image = load_image(args.image)

    # ── Build predictor ────────────────────────────────────────────────────
    predictor = GeoPredictor(
        checkpoint_path=args.checkpoint,
        processed_dir=args.processed_dir,
    )

    # ── Predict ────────────────────────────────────────────────────────────
    if args.true_lat is not None and args.true_lng is not None:
        result = predictor.predict_with_error(image, args.true_lat, args.true_lng, args.top_k)
    else:
        result = predictor.predict(image, args.top_k)

    # ── Display results ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"  Predicted location: {result['pred_lat']:.4f}°, {result['pred_lng']:.4f}°")
    print(f"  Estimated country:  {result['country']}")

    if "distance_km" in result:
        print(f"  Ground truth:       {result['true_lat']:.4f}°, {result['true_lng']:.4f}°")
        print(f"  Error distance:     {result['distance_km']:.1f} km")

    print(f"\n  Top-{args.top_k} candidate cells:")
    for cell in result["top_k_cells"]:
        print(
            f"    #{cell['rank']}: {cell['cell_id']} "
            f"(conf={cell['confidence']:.3f}, "
            f"center={cell['cell_center_lat']:.2f}°, {cell['cell_center_lng']:.2f}°)"
        )
    print("=" * 50)

    # Output as JSON for programmatic use
    print("\nJSON output:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
