"""
Gradio web demo for GeoGuessr AI.

Provides an interactive interface for geographic location prediction:
  - Upload a street-level image or paste a URL
  - See a world map with the predicted location pin
  - View top-5 country predictions with confidence bars
  - Optionally provide ground truth to see error distance
  - Example images from 5 different continents

Usage:
    python app/demo.py                         # Default port 7860
    python app/demo.py --share                 # Public share link
    python app/demo.py --checkpoint path/to/checkpoint.pth
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Gracefully handle missing dependencies at import time
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Run: pip install gradio")

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from PIL import Image
from src.training.config import PROCESSED_DIR, OUTPUTS_DIR

EXAMPLES_DIR = PROJECT_ROOT / "examples"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"

# Global predictor — loaded once at startup
_predictor = None


def get_predictor(checkpoint_path: Path):
    """
    Lazy-loads the GeoPredictor singleton.

    We load the model once and reuse it for all requests. Loading at startup
    rather than on first request avoids a long lag on the first prediction.
    """
    global _predictor
    if _predictor is None:
        from src.inference.predict import GeoPredictor
        _predictor = GeoPredictor(
            checkpoint_path=checkpoint_path,
            processed_dir=PROCESSED_DIR,
        )
    return _predictor


def predict_from_image(
    image: Image.Image,
    true_lat: Optional[float],
    true_lng: Optional[float],
    checkpoint_path: Path,
) -> tuple:
    """
    Core prediction function called by Gradio.

    Returns:
        map_html: HTML string of the folium map
        confidence_data: List of [country/cell, confidence_pct] for bar chart
        summary_text: Human-readable prediction summary
    """
    if image is None:
        return None, None, "Please upload an image first."

    try:
        predictor = get_predictor(checkpoint_path)

        # Run prediction
        if true_lat is not None and true_lng is not None and true_lat != 0 and true_lng != 0:
            result = predictor.predict_with_error(image, true_lat, true_lng, top_k=5)
        else:
            result = predictor.predict(image, top_k=5)

        # ── Build folium map ───────────────────────────────────────────────
        from src.inference.visualize import create_prediction_map
        map_path = OUTPUTS_DIR / "demo_map.html"
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        create_prediction_map(
            pred_lat=result["pred_lat"],
            pred_lng=result["pred_lng"],
            true_lat=result.get("true_lat"),
            true_lng=result.get("true_lng"),
            top_k_cells=result["top_k_cells"],
            country=result["country"],
            output_path=map_path,
        )

        # Read HTML for embedding
        map_html = map_path.read_text(encoding="utf-8")

        # ── Confidence chart data ──────────────────────────────────────────
        # Show top-5 cells with geographic labels
        confidence_data = []
        for cell in result["top_k_cells"]:
            # Use cell center lat/lng as label since we don't have country per cell
            label = f"#{cell['rank']}: ({cell['cell_center_lat']:.1f}°, {cell['cell_center_lng']:.1f}°)"
            confidence_data.append([label, round(cell["confidence"] * 100, 2)])

        # ── Summary text ───────────────────────────────────────────────────
        summary = build_summary_text(result)

        return map_html, confidence_data, summary

    except FileNotFoundError as e:
        error_msg = (
            f"Model checkpoint not found: {e}\n\n"
            "Please train the model first:\n"
            "  python src/training/train.py"
        )
        return None, None, error_msg
    except Exception as e:
        return None, None, f"Prediction failed: {e}"


def build_summary_text(result: dict) -> str:
    """Formats the prediction result as a human-readable summary."""
    lines = [
        f"**Predicted Location**",
        f"📍 Coordinates: {result['pred_lat']:.4f}°, {result['pred_lng']:.4f}°",
        f"🌍 Country: {result['country']}",
    ]

    if "distance_km" in result:
        dist = result["distance_km"]
        # GeoGuessr-style scoring tier
        if dist < 25:
            tier = "🏆 City level!"
        elif dist < 200:
            tier = "✅ Region level"
        elif dist < 500:
            tier = "🗺️ Country level"
        else:
            tier = "🌐 Continent level"

        lines.extend([
            f"\n**Ground Truth Comparison**",
            f"📌 True: {result['true_lat']:.4f}°, {result['true_lng']:.4f}°",
            f"📏 Error: **{dist:.1f} km** — {tier}",
        ])

    lines.append(f"\n**Top-5 Candidate Cells**")
    for cell in result["top_k_cells"]:
        lines.append(
            f"  #{cell['rank']}: Cell {cell['cell_id'][:8]}... "
            f"(conf: {cell['confidence']:.1%})"
        )

    return "\n".join(lines)


def build_gradio_app(checkpoint_path: Path) -> gr.Blocks:
    """
    Constructs the Gradio UI.

    Layout:
      Left column:  Image upload + coordinates input + submit button + examples
      Right column: World map + confidence bars + summary text
    """
    with gr.Blocks(
        title="GeoGuessr AI",
        theme=gr.themes.Soft(),
        css="""
            .map-container { min-height: 450px; }
            .title-block { text-align: center; padding: 20px; }
        """,
    ) as demo:

        # ── Title ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="title-block">
            <h1>🌍 GeoGuessr AI</h1>
            <p style="color: #666; font-size: 16px;">
                Upload a street-level photo and AI will predict where in the world it was taken.
            </p>
        </div>
        """)

        with gr.Row():
            # ── Left column: inputs ────────────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Street-level Image",
                    height=300,
                )

                with gr.Accordion("Optional: Provide Ground Truth", open=False):
                    gr.Markdown("Enter true coordinates to see how accurate the prediction was.")
                    with gr.Row():
                        true_lat_input = gr.Number(
                            label="True Latitude", value=None, precision=4
                        )
                        true_lng_input = gr.Number(
                            label="True Longitude", value=None, precision=4
                        )

                predict_btn = gr.Button("🔍 Predict Location", variant="primary", size="lg")

                # ── Example images ─────────────────────────────────────────
                # These examples span different continents for a global demo.
                # Replace with actual downloaded example images.
                example_files = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
                if example_files:
                    gr.Examples(
                        examples=[[str(f)] for f in example_files[:5]],
                        inputs=[image_input],
                        label="Example images (click to load)",
                    )
                else:
                    gr.Markdown(
                        "_Add street-level images to the `examples/` directory to show demos here._"
                    )

            # ── Right column: outputs ──────────────────────────────────────
            with gr.Column(scale=2):
                map_output = gr.HTML(
                    label="Predicted Location",
                    elem_classes=["map-container"],
                    value="<div style='text-align:center; padding:100px; color:#999;'>"
                          "Upload an image and click Predict</div>",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        confidence_output = gr.Dataframe(
                            headers=["Location", "Confidence (%)"],
                            label="Top-5 Candidates",
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        summary_output = gr.Markdown(
                            label="Summary",
                            value="*Prediction results will appear here.*",
                        )

        # ── Button click handler ───────────────────────────────────────────
        predict_btn.click(
            fn=lambda img, lat, lng: predict_from_image(img, lat, lng, checkpoint_path),
            inputs=[image_input, true_lat_input, true_lng_input],
            outputs=[map_output, confidence_output, summary_output],
            show_progress=True,
        )

        # ── About section ──────────────────────────────────────────────────
        with gr.Accordion("About this model", open=False):
            gr.Markdown("""
            **GeoGuessr AI** predicts geographic locations from street-level images.

            **Architecture:** EfficientNet-B4 backbone (pretrained on ImageNet) with:
            - Classification head → predicts which H3 hexagonal cell the image is from
            - Regression head → predicts exact lat/lng coordinates

            **Training data:** Geotagged street-level images from Mapillary API v4,
            partitioned into H3 hexagonal cells at resolution 3 (~7000 cells globally).

            **Accuracy tiers** (after full training):
            - 🌐 Continent level (1500km+): most images
            - 🗺️ Country level (< 500km): ~40-60% of images
            - ✅ Region level (< 200km): ~25-40%
            - 🏆 City level (< 25km): ~5-15%

            *Note: Performance varies heavily by training data coverage.*
            """)

    return demo


def main():
    parser = argparse.ArgumentParser(description="GeoGuessr AI Gradio Demo")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint"
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    if not GRADIO_AVAILABLE:
        print("Install gradio with: pip install gradio>=4.0.0")
        sys.exit(1)

    print(f"Starting GeoGuessr AI demo on port {args.port}...")
    print(f"Checkpoint: {args.checkpoint}")

    demo = build_gradio_app(checkpoint_path=args.checkpoint)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
