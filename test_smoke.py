"""
End-to-end smoke test for GeoGuessr AI pipeline.

Runs the full pipeline on 100 synthetic images to verify that nothing is
broken before investing time in real data collection and training.

Tests:
  1. Synthetic data generation (collect.py)
  2. H3 cell assignment and preprocessing (preprocess.py)
  3. Dataset loading and augmentation (dataset.py)
  4. Model instantiation (classifier.py)
  5. Forward pass with correct tensor shapes
  6. Loss calculation (losses.py)
  7. Two training epochs (train.py)
  8. Evaluation on test split (evaluate.py)
  9. Single-image inference (predict.py)
  10. Map visualization (visualize.py)

Usage:
    python test_smoke.py
    python test_smoke.py --n_images 50   # Faster (fewer images, fewer cells)
    python test_smoke.py --keep          # Don't clean up test data afterward
"""

import argparse
import json
import pickle
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Test constants ─────────────────────────────────────────────────────────────
N_IMAGES = 100          # Enough to form a few H3 cells after filtering
MIN_IMAGES_PER_CELL = 3  # Very low threshold so smoke test doesn't filter everything
H3_RESOLUTION = 2        # Very coarse (122 cells globally) → guaranteed overlap


# ── Colorful terminal output ────────────────────────────────────────────────────
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def pass_msg(test_name: str) -> str:
    return f"{Colors.GREEN}✓ PASS{Colors.RESET}  {test_name}"


def fail_msg(test_name: str, error: str) -> str:
    return f"{Colors.RED}✗ FAIL{Colors.RESET}  {test_name}\n       {error}"


def section(title: str) -> None:
    print(f"\n{Colors.BLUE}{Colors.BOLD}── {title} ──{Colors.RESET}")


def run_smoke_tests(n_images: int = N_IMAGES, keep_data: bool = False) -> bool:
    """
    Runs all smoke tests in a temporary directory.

    Returns True if all tests pass, False if any fail.
    """
    results = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="geoguessr_smoke_"))
    print(f"\n{Colors.BOLD}GeoGuessr AI — Smoke Test{Colors.RESET}")
    print(f"Temp directory: {tmp_dir}")
    print(f"Images: {n_images}, H3 resolution: {H3_RESOLUTION}, min images/cell: {MIN_IMAGES_PER_CELL}")

    try:
        # ── 1. Data generation ─────────────────────────────────────────────
        section("1. Synthetic Data Generation")
        try:
            from src.data.collect import collect_synthetic

            raw_dir = tmp_dir / "data" / "raw"
            metadata_path = raw_dir / "metadata.csv"
            n = collect_synthetic(n_images, raw_dir, metadata_path)

            assert n == n_images, f"Expected {n_images}, got {n}"
            assert metadata_path.exists(), "metadata.csv not created"

            import pandas as pd
            df = pd.read_csv(metadata_path)
            assert len(df) == n_images, f"CSV has {len(df)} rows, expected {n_images}"
            assert all(c in df.columns for c in ["image_id", "lat", "lng", "filepath"])

            print(pass_msg(f"Generated {n_images} synthetic images + metadata.csv"))
            results.append(True)
        except Exception as e:
            print(fail_msg("Synthetic data generation", str(e)))
            traceback.print_exc()
            results.append(False)

        # ── 2. Preprocessing ───────────────────────────────────────────────
        section("2. H3 Cell Assignment & Dataset Split")
        processed_dir = tmp_dir / "data" / "processed"
        try:
            from src.data.preprocess import (
                assign_h3_cells,
                filter_sparse_cells,
                build_label_encoder,
                split_dataset,
                get_cell_center,
            )
            import pandas as pd
            import json, pickle

            df = pd.read_csv(metadata_path)
            df = assign_h3_cells(df, resolution=H3_RESOLUTION)

            assert "h3_cell" in df.columns
            assert df["h3_cell"].notna().all()
            print(f"  Unique cells: {df['h3_cell'].nunique()}")

            df = filter_sparse_cells(df, min_images=MIN_IMAGES_PER_CELL)
            assert len(df) > 0, "All cells were filtered out — lower MIN_IMAGES_PER_CELL"

            cell_to_idx, idx_to_cell = build_label_encoder(df["h3_cell"].tolist())
            df["label"] = df["h3_cell"].map(cell_to_idx)
            num_classes = len(cell_to_idx)
            print(f"  Classes after filtering: {num_classes}")

            cell_centers = {cell: get_cell_center(cell) for cell in cell_to_idx}

            train_df, val_df, test_df = split_dataset(df, train_ratio=0.8, val_ratio=0.1)

            processed_dir.mkdir(parents=True, exist_ok=True)

            # Fix filepaths to be absolute for the temp dir
            for split_df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
                # Ensure filepath is relative to project root equivalent in temp dir
                split_df = split_df.copy()
                # Replace project root prefix with tmp_dir
                split_df["filepath"] = split_df["filepath"].apply(
                    lambda p: str(tmp_dir / "data" / "raw" / Path(p).name)
                )
                split_df.to_csv(processed_dir / f"{name}.csv", index=False)

            label_map = {
                "cell_to_idx": cell_to_idx,
                "idx_to_cell": {str(k): v for k, v in idx_to_cell.items()},
                "cell_centers": {k: list(v) for k, v in cell_centers.items()},
                "num_classes": num_classes,
                "h3_resolution": H3_RESOLUTION,
            }
            with open(processed_dir / "label_encoder.pkl", "wb") as f:
                pickle.dump(label_map, f)
            with open(processed_dir / "label_map.json", "w") as f:
                json.dump(label_map, f, indent=2)

            print(pass_msg(f"Preprocessing complete: {num_classes} classes, "
                           f"{len(train_df)}/{len(val_df)}/{len(test_df)} splits"))
            results.append(True)
        except Exception as e:
            print(fail_msg("Preprocessing", str(e)))
            traceback.print_exc()
            results.append(False)

        # ── 3. Dataset loading ─────────────────────────────────────────────
        section("3. Dataset + DataLoader")
        try:
            from torch.utils.data import DataLoader

            # Temporarily monkey-patch PROJECT_ROOT in dataset.py
            import src.data.dataset as ds_module
            original_root = ds_module.PROJECT_ROOT
            ds_module.PROJECT_ROOT = tmp_dir

            train_csv = processed_dir / "train.csv"

            # Re-read the train CSV and fix paths to be absolute
            import pandas as pd
            train_df_check = pd.read_csv(train_csv)

            dataset = ds_module.GeoDataset(
                csv_path=train_csv,
                split="train",
                label_encoder_path=processed_dir / "label_encoder.pkl",
            )

            # Override df with absolute paths
            import os
            dataset.df["filepath"] = dataset.df["filepath"].apply(
                lambda p: str(tmp_dir / "data" / "raw" / Path(p).name)
            )

            assert len(dataset) > 0, "Dataset is empty"

            loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True, num_workers=0)
            batch = next(iter(loader))
            images, labels, lats, lngs = batch

            assert images.shape[1:] == (3, 224, 224), f"Wrong image shape: {images.shape}"
            assert labels.dtype == torch.int64 or labels.dtype == torch.long
            assert lats.shape == (images.shape[0],)

            ds_module.PROJECT_ROOT = original_root
            print(pass_msg(f"Dataset: {len(dataset)} samples, batch shape {images.shape}"))
            results.append(True)
        except Exception as e:
            print(fail_msg("Dataset loading", str(e)))
            traceback.print_exc()
            results.append(False)

        # ── 4 & 5. Model instantiation + forward pass ──────────────────────
        section("4–5. Model Instantiation + Forward Pass")
        try:
            from src.models.classifier import GeoClassifier

            num_classes = label_map["num_classes"]
            device = torch.device("cpu")  # Smoke test always on CPU

            # Use a tiny model variant for speed
            model = GeoClassifier(
                num_cells=num_classes,
                backbone_name="efficientnet_b0",  # Smaller for speed
                pretrained=False,  # Skip downloading weights in smoke test
            ).to(device)

            # Forward pass with a fake batch
            dummy_input = torch.randn(2, 3, 224, 224)
            cell_logits, pred_coords = model(dummy_input)

            assert cell_logits.shape == (2, num_classes), f"Wrong logits shape: {cell_logits.shape}"
            assert pred_coords.shape == (2, 2), f"Wrong coords shape: {pred_coords.shape}"
            assert pred_coords[:, 0].abs().max() <= 90.0, "Lat out of range"
            assert pred_coords[:, 1].abs().max() <= 180.0, "Lng out of range"

            params = model.count_parameters()
            print(pass_msg(
                f"Model OK: {params['total']:,} params, "
                f"logits {cell_logits.shape}, coords {pred_coords.shape}"
            ))
            results.append(True)
        except Exception as e:
            print(fail_msg("Model instantiation/forward pass", str(e)))
            traceback.print_exc()
            results.append(False)

        # ── 6. Loss calculation ────────────────────────────────────────────
        section("6. Loss Functions")
        try:
            from src.models.losses import CombinedGeoLoss, haversine_distance

            criterion = CombinedGeoLoss(lambda_regression=0.1)

            # Dummy data
            bs = 4
            logits = torch.randn(bs, num_classes)
            pred_coords = torch.tensor([[40.0, -74.0], [51.5, -0.1], [-33.9, 151.2], [35.7, 139.7]])
            true_labels = torch.randint(0, num_classes, (bs,))
            true_coords = torch.tensor([[40.7, -74.0], [48.9, 2.3], [-33.9, 151.2], [37.6, 126.9]])

            loss, loss_dict = criterion(logits, pred_coords, true_labels, true_coords)

            assert loss.item() > 0, "Loss should be positive"
            assert not torch.isnan(loss), "Loss is NaN"
            assert "ce" in loss_dict and "haversine_km" in loss_dict

            # Test standalone Haversine
            dist = haversine_distance(pred_coords[:1], true_coords[:1], reduction="none")
            assert dist.shape == (1,)
            assert dist.item() >= 0

            print(pass_msg(
                f"Losses OK: CE={loss_dict['ce']:.3f}, "
                f"Haversine={loss_dict['haversine_km']:.1f}km"
            ))
            results.append(True)
        except Exception as e:
            print(fail_msg("Loss functions", str(e)))
            traceback.print_exc()
            results.append(False)

        # ── 7. Mini training run (2 epochs) ───────────────────────────────
        section("7. Mini Training Run (2 epochs)")
        try:
            from src.models.losses import CombinedGeoLoss
            from src.data.dataset import GeoDataset

            # Reuse model from test 4
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = CombinedGeoLoss()

            # Build minimal dataset loaders
            from torch.utils.data import DataLoader

            def make_loader(split: str):
                ds = GeoDataset(
                    processed_dir / f"{split}.csv",
                    split=split,
                    label_encoder_path=processed_dir / "label_encoder.pkl",
                )
                # Fix paths
                ds.df["filepath"] = ds.df["filepath"].apply(
                    lambda p: str(tmp_dir / "data" / "raw" / Path(p).name)
                )
                return DataLoader(ds, batch_size=min(4, len(ds)), num_workers=0)

            train_loader = make_loader("train")
            val_loader = make_loader("val")

            for epoch in range(2):
                model.train()
                for images, labels, lats, lngs in train_loader:
                    true_coords = torch.stack([lats, lngs], dim=1).float()
                    optimizer.zero_grad()
                    cell_logits, pred_coords = model(images)
                    loss, _ = criterion(cell_logits, pred_coords, labels, true_coords)
                    loss.backward()
                    optimizer.step()

            # Save checkpoint
            ckpt_dir = tmp_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / "smoke_model.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": 1,
                "num_classes": num_classes,
            }, ckpt_path)

            print(pass_msg("2-epoch training run completed, checkpoint saved"))
            results.append(True)
        except Exception as e:
            print(fail_msg("Mini training run", str(e)))
            traceback.print_exc()
            results.append(False)
            ckpt_path = None

        # ── 8. Evaluation ──────────────────────────────────────────────────
        section("8. Evaluation Metrics")
        try:
            from src.training.evaluate import compute_metrics

            # Synthetic predictions/labels for metric computation
            n_test = 20
            fake_pred_logits = torch.randn(n_test, num_classes)
            fake_pred_coords = torch.tensor([[random_lat_lng() for _ in range(2)] for _ in range(n_test)])
            fake_true_labels = torch.randint(0, num_classes, (n_test,))
            fake_true_coords = torch.tensor([[random_lat_lng() for _ in range(2)] for _ in range(n_test)])
            # Fix coord ranges
            fake_pred_coords[:, 0] = fake_pred_coords[:, 0] * 90
            fake_pred_coords[:, 1] = fake_pred_coords[:, 1] * 180
            fake_true_coords[:, 0] = fake_true_coords[:, 0] * 90
            fake_true_coords[:, 1] = fake_true_coords[:, 1] * 180

            metrics = compute_metrics({
                "pred_logits": fake_pred_logits,
                "pred_coords": fake_pred_coords,
                "true_labels": fake_true_labels,
                "true_coords": fake_true_coords,
            })

            required_keys = ["median_haversine_km", "mean_haversine_km", "cell_acc_top1", "cell_acc_top5"]
            for key in required_keys:
                assert key in metrics, f"Missing metric: {key}"

            print(pass_msg(
                f"Metrics computed: median={metrics['median_haversine_km']:.0f}km, "
                f"top1={metrics['cell_acc_top1']:.2%}"
            ))
            results.append(True)
        except Exception as e:
            print(fail_msg("Evaluation metrics", str(e)))
            traceback.print_exc()
            results.append(False)

        # ── 9. Inference ───────────────────────────────────────────────────
        section("9. Single-Image Inference")
        try:
            if ckpt_path is None or not ckpt_path.exists():
                raise RuntimeError("No checkpoint available (training test failed)")

            from src.inference.predict import GeoPredictor
            from PIL import Image as PILImage
            import numpy as np

            predictor = GeoPredictor(
                checkpoint_path=ckpt_path,
                processed_dir=processed_dir,
                device=torch.device("cpu"),
            )

            # Override model with the smoke model (efficientnet_b0, not b4)
            predictor.model = model
            predictor.model.eval()

            # Use a random test image
            test_image = PILImage.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            result = predictor.predict(test_image, top_k=3)

            assert "pred_lat" in result
            assert "pred_lng" in result
            assert -90 <= result["pred_lat"] <= 90
            assert -180 <= result["pred_lng"] <= 180
            assert len(result["top_k_cells"]) == 3

            print(pass_msg(
                f"Inference OK: ({result['pred_lat']:.2f}°, {result['pred_lng']:.2f}°)"
            ))
            results.append(True)
        except Exception as e:
            print(fail_msg("Single-image inference", str(e)))
            traceback.print_exc()
            results.append(False)

        # ── 10. Visualization ──────────────────────────────────────────────
        section("10. Map Visualization")
        try:
            from src.inference.visualize import create_prediction_map

            map_output = tmp_dir / "test_map.html"
            create_prediction_map(
                pred_lat=48.85,
                pred_lng=2.35,
                true_lat=51.50,
                true_lng=-0.12,
                output_path=map_output,
            )

            assert map_output.exists(), "Map HTML file not created"
            content = map_output.read_text()
            assert "folium" in content or "leaflet" in content, "Map doesn't contain leaflet/folium JS"

            print(pass_msg(f"Map saved ({map_output.stat().st_size:,} bytes)"))
            results.append(True)
        except Exception as e:
            print(fail_msg("Map visualization", str(e)))
            traceback.print_exc()
            results.append(False)

    finally:
        if not keep_data:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\nTemp directory cleaned up.")
        else:
            print(f"\nTemp data kept at: {tmp_dir}")

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    total = len(results)

    print(f"\n{'=' * 50}")
    print(f"SMOKE TEST SUMMARY: {n_pass}/{total} passed")

    if n_fail == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED ✓{Colors.RESET}")
        print("The pipeline is ready for real training.")
    else:
        print(f"{Colors.RED}{Colors.BOLD}{n_fail} TEST(S) FAILED ✗{Colors.RESET}")
        print("Fix the failing tests before running real training.")
    print('=' * 50 + '\n')

    return n_fail == 0


def random_lat_lng() -> float:
    """Helper: returns a random value in [-1, 1] for synthetic coords."""
    import random
    return random.uniform(-1, 1)


def main():
    parser = argparse.ArgumentParser(description="Run GeoGuessr AI smoke tests")
    parser.add_argument("--n_images", type=int, default=N_IMAGES, help="Number of synthetic images")
    parser.add_argument("--keep", action="store_true", help="Keep temp data after test")
    args = parser.parse_args()

    success = run_smoke_tests(n_images=args.n_images, keep_data=args.keep)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
