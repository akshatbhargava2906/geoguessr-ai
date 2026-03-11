"""
Evaluation script for GeoGuessr AI.

Runs the trained model on the held-out test set and reports:
  - Country-level accuracy (distance < 500km)
  - Region-level accuracy  (distance < 200km)
  - City-level accuracy    (distance < 25km)
  - Median and mean Haversine distance
  - Top-1 and Top-5 cell classification accuracy

These thresholds roughly correspond to GeoGuessr's scoring tiers:
  500km = continent/country, 200km = region, 25km = city neighborhood

Usage:
    python src/training/evaluate.py --checkpoint checkpoints/best_model.pth
    python src/training/evaluate.py --checkpoint checkpoints/best_model.pth --split val
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GeoDataset, get_val_transform
from src.models.classifier import build_model
from src.models.losses import haversine_distance
from src.training.config import Config, PROCESSED_DIR


# GeoGuessr-inspired accuracy thresholds (in km)
ACCURACY_THRESHOLDS = {
    "country_level (500km)": 500.0,
    "region_level  (200km)": 200.0,
    "city_level     (25km)": 25.0,
}


def get_cell_center_coords(
    cell_indices: torch.Tensor, label_map: dict, device: torch.device
) -> torch.Tensor:
    """
    Converts predicted cell indices to their geographic center coordinates.

    Used as a baseline: if the regression head were absent, what would the
    accuracy be from cell centroids alone?
    """
    idx_to_cell = label_map["idx_to_cell"]
    cell_centers = label_map["cell_centers"]

    coords = []
    for idx in cell_indices.cpu().tolist():
        cell_id = idx_to_cell[str(idx)]
        lat, lng = cell_centers[cell_id]
        coords.append([lat, lng])

    return torch.tensor(coords, dtype=torch.float32, device=device)


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    label_map: dict,
    device: torch.device,
    use_regression_head: bool = True,
) -> dict:
    """
    Runs model on a dataloader and collects predictions.

    Args:
        use_regression_head: If True, use regression coordinates for distance metrics.
                             If False, use cell centroid coordinates (ablation baseline).
    """
    model.eval()
    all_pred_logits = []
    all_pred_coords = []
    all_true_labels = []
    all_true_coords = []

    for images, labels, lats, lngs in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        true_coords = torch.stack([lats, lngs], dim=1).float().to(device)

        with autocast(enabled=device.type == "cuda"):
            cell_logits, pred_coords = model(images)

        if not use_regression_head:
            # Ablation: use cell center instead of regression head
            pred_cell_indices = cell_logits.argmax(dim=1)
            pred_coords = get_cell_center_coords(pred_cell_indices, label_map, device)

        all_pred_logits.append(cell_logits.cpu())
        all_pred_coords.append(pred_coords.cpu())
        all_true_labels.append(labels.cpu())
        all_true_coords.append(true_coords.cpu())

    pred_logits = torch.cat(all_pred_logits, dim=0)
    pred_coords = torch.cat(all_pred_coords, dim=0)
    true_labels = torch.cat(all_true_labels, dim=0)
    true_coords = torch.cat(all_true_coords, dim=0)

    return {
        "pred_logits": pred_logits,
        "pred_coords": pred_coords,
        "true_labels": true_labels,
        "true_coords": true_coords,
    }


def compute_metrics(results: dict) -> dict:
    """
    Computes all evaluation metrics from collected predictions.

    Returns a dict suitable for pretty-printing and JSON export.
    """
    pred_logits = results["pred_logits"]
    pred_coords = results["pred_coords"]
    true_labels = results["true_labels"]
    true_coords = results["true_coords"]

    n = len(true_labels)

    # ── Distance metrics (regression head) ────────────────────────────────
    distances_km = haversine_distance(pred_coords, true_coords, reduction="none").numpy()

    # ── Threshold accuracy metrics ─────────────────────────────────────────
    threshold_metrics = {}
    for name, threshold_km in ACCURACY_THRESHOLDS.items():
        accuracy = (distances_km < threshold_km).mean()
        threshold_metrics[name] = float(accuracy)

    # ── Cell classification metrics ────────────────────────────────────────
    pred_top1 = pred_logits.argmax(dim=1)
    acc_top1 = (pred_top1 == true_labels).float().mean().item()

    _, top5 = pred_logits.topk(5, dim=1)
    acc_top5 = top5.eq(true_labels.view(-1, 1).expand_as(top5)).any(dim=1).float().mean().item()

    return {
        "n_samples": n,
        **threshold_metrics,
        "median_haversine_km": float(np.median(distances_km)),
        "mean_haversine_km": float(np.mean(distances_km)),
        "p25_haversine_km": float(np.percentile(distances_km, 25)),
        "p75_haversine_km": float(np.percentile(distances_km, 75)),
        "cell_acc_top1": acc_top1,
        "cell_acc_top5": acc_top5,
    }


def print_summary_table(metrics: dict, split: str = "test") -> None:
    """Prints a clean formatted summary table to stdout."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS  ({split} set, n={metrics['n_samples']:,})")
    print("=" * 60)

    print("\n  Geographic Distance Accuracy:")
    print(f"  {'Metric':<35} {'Value':>10}")
    print("  " + "-" * 45)
    for name in ACCURACY_THRESHOLDS:
        val = metrics[name]
        print(f"  {name:<35} {val:>9.2%}")

    print("\n  Distance Errors (regression head):")
    print(f"  {'Metric':<35} {'Value':>10}")
    print("  " + "-" * 45)
    print(f"  {'Median Haversine distance':<35} {metrics['median_haversine_km']:>8.1f} km")
    print(f"  {'Mean Haversine distance':<35} {metrics['mean_haversine_km']:>8.1f} km")
    print(f"  {'25th percentile (km)':<35} {metrics['p25_haversine_km']:>8.1f} km")
    print(f"  {'75th percentile (km)':<35} {metrics['p75_haversine_km']:>8.1f} km")

    print("\n  Cell Classification Accuracy:")
    print(f"  {'Metric':<35} {'Value':>10}")
    print("  " + "-" * 45)
    print(f"  {'Top-1 cell accuracy':<35} {metrics['cell_acc_top1']:>9.2%}")
    print(f"  {'Top-5 cell accuracy':<35} {metrics['cell_acc_top5']:>9.2%}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoGuessr AI model")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--processed_dir", type=Path, default=PROCESSED_DIR
    )
    parser.add_argument(
        "--batch_size", type=int, default=64
    )
    parser.add_argument(
        "--no_regression", action="store_true",
        help="Ablation: use cell centroids instead of regression head"
    )
    parser.add_argument(
        "--output_json", type=Path, default=None,
        help="Save metrics to JSON file"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load label encoder ─────────────────────────────────────────────────
    label_encoder_path = args.processed_dir / "label_encoder.pkl"
    with open(label_encoder_path, "rb") as f:
        label_map = pickle.load(f)
    num_classes = label_map["num_classes"]

    # ── Load model ─────────────────────────────────────────────────────────
    model = build_model(
        num_cells=num_classes,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # ── Build dataset and loader ───────────────────────────────────────────
    csv_path = args.processed_dir / f"{args.split}.csv"
    dataset = GeoDataset(csv_path, split=args.split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    print(f"Evaluating on {len(dataset):,} {args.split} samples...")

    # ── Run evaluation ─────────────────────────────────────────────────────
    results = run_evaluation(
        model, loader, label_map, device,
        use_regression_head=not args.no_regression
    )
    metrics = compute_metrics(results)

    # ── Display results ────────────────────────────────────────────────────
    print_summary_table(metrics, split=args.split)

    # ── Save to JSON ───────────────────────────────────────────────────────
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
