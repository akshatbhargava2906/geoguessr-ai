"""
Geographic cell partitioning and dataset preprocessing.

Converts raw (image, lat, lng) triples into a classification problem by:
  1. Assigning each image to an H3 hexagonal cell (resolution 3 ≈ 7000 cells globally)
  2. Filtering sparse cells (< MIN_IMAGES_PER_CELL)
  3. Encoding cell IDs to integer class indices
  4. Splitting into train/val/test sets
  5. Saving processed manifests + label encoder to data/processed/

H3 resolution 3 gives cells with edge length ~59km, which is a good trade-off:
  - Coarse enough to have many training images per cell
  - Fine enough to distinguish continental regions
  - Can be refined at inference time using the regression head

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --resolution 4 --min_images 20
"""

import argparse
import json
import pickle
from pathlib import Path

import h3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")  # Non-interactive backend for headless environments

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_CSV = RAW_DATA_DIR / "metadata.csv"

# Default H3 resolution.
# Resolution 3: ~7k cells, edge ~59km — good for continent/country-level training
# Resolution 4: ~41k cells, edge ~22km — better for city-level, needs more data
DEFAULT_RESOLUTION = 3
MIN_IMAGES_PER_CELL = 50


def assign_h3_cells(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """
    Maps each (lat, lng) pair to an H3 hexagonal cell index.

    H3 is Uber's hierarchical hexagonal grid. We use it instead of a
    rectangular grid because hexagons have uniform distance to all neighbors,
    reducing edge artifacts in the classification loss.
    """
    df = df.copy()
    df["h3_cell"] = df.apply(
        lambda row: h3.latlng_to_cell(row["lat"], row["lng"], resolution), axis=1
    )
    return df


def filter_sparse_cells(df: pd.DataFrame, min_images: int) -> pd.DataFrame:
    """
    Removes cells with fewer than `min_images` samples.

    Sparse cells cause class imbalance and the model can't generalize
    from too few examples. Filtering here is cleaner than weighted sampling.
    """
    cell_counts = df["h3_cell"].value_counts()
    valid_cells = cell_counts[cell_counts >= min_images].index
    filtered = df[df["h3_cell"].isin(valid_cells)].copy()

    n_removed = len(df) - len(filtered)
    n_cells_removed = len(cell_counts) - len(valid_cells)
    print(
        f"Filtered {n_removed} images from {n_cells_removed} sparse cells "
        f"(< {min_images} images). Kept {len(valid_cells)} cells."
    )
    return filtered


def build_label_encoder(cells: list[str]) -> tuple[dict, dict]:
    """
    Creates bidirectional mappings between H3 cell IDs and integer class indices.

    Returns:
        cell_to_idx: {"8828308281fffff": 0, ...}
        idx_to_cell: {0: "8828308281fffff", ...}
    """
    sorted_cells = sorted(set(cells))  # Sort for reproducibility
    cell_to_idx = {cell: idx for idx, cell in enumerate(sorted_cells)}
    idx_to_cell = {idx: cell for cell, idx in cell_to_idx.items()}
    return cell_to_idx, idx_to_cell


def get_cell_center(cell_id: str) -> tuple[float, float]:
    """Returns the (lat, lng) of an H3 cell's geographic center."""
    lat, lng = h3.cell_to_latlng(cell_id)
    return lat, lng


def split_dataset(
    df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/10/10 train/val/test split.

    Stratification ensures each cell appears in all three splits,
    preventing zero-shot cells in val/test.
    We check if a cell has enough samples for stratification (needs ≥3 per class).
    """
    # Cells that appear rarely might not support stratification
    # Fall back to non-stratified for very small cells
    cell_counts = df["h3_cell"].value_counts()
    stratifiable = cell_counts[cell_counts >= 3].index
    df_strat = df[df["h3_cell"].isin(stratifiable)]
    df_no_strat = df[~df["h3_cell"].isin(stratifiable)]

    num_classes = df_strat["h3_cell"].nunique() if len(df_strat) > 0 else 0
    test_size = 1 - train_ratio - val_ratio
    can_stratify = (
        len(df_strat) > 0
        and int(len(df_strat) * test_size) >= num_classes
        and int(len(df_strat) * test_size * (train_ratio / (train_ratio + val_ratio))) >= num_classes
    )

    if can_stratify:
        train_val, test = train_test_split(
            df_strat, test_size=test_size,
            stratify=df_strat["h3_cell"], random_state=42
        )
        relative_val = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val, test_size=relative_val,
            stratify=train_val["h3_cell"], random_state=42
        )
    else:
        train_val, test = train_test_split(df, test_size=max(1, int(len(df) * test_size)), random_state=42)
        train, val = train_test_split(train_val, test_size=max(1, int(len(train_val) * val_ratio / (train_ratio + val_ratio))), random_state=42)

    # Append non-stratifiable rows to train only
    if len(df_no_strat) > 0:
        train = pd.concat([train, df_no_strat]).sample(frac=1, random_state=42)

    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


def plot_coverage_map(df: pd.DataFrame, output_path: Path) -> None:
    """
    Saves a scatter plot of image locations as a geographic coverage map.

    Useful for quickly verifying that the dataset has global coverage
    and identifying geographic blind spots.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(df["lng"], df["lat"], s=0.5, alpha=0.3, c="steelblue")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Dataset Geographic Coverage ({len(df):,} images)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Coverage map saved to {output_path}")


def plot_cell_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """
    Saves a histogram of images per cell.

    Helps visualize class imbalance. A long right tail is expected
    (some dense regions like Europe have many more images).
    """
    counts = df["h3_cell"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(counts.values, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(counts.median(), color="red", linestyle="--", label=f"Median: {counts.median():.0f}")
    ax.set_xlabel("Images per cell")
    ax.set_ylabel("Number of cells")
    ax.set_title("Distribution of Images per H3 Cell")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Cell distribution plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw dataset: assign H3 cells, filter, split, encode labels"
    )
    parser.add_argument(
        "--resolution", type=int, default=DEFAULT_RESOLUTION,
        help=f"H3 resolution (default: {DEFAULT_RESOLUTION})"
    )
    parser.add_argument(
        "--min_images", type=int, default=MIN_IMAGES_PER_CELL,
        help=f"Minimum images per cell to keep (default: {MIN_IMAGES_PER_CELL})"
    )
    parser.add_argument(
        "--metadata", type=Path, default=METADATA_CSV,
        help="Path to metadata.csv from data collection"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=PROCESSED_DATA_DIR,
        help="Directory to save processed splits"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load raw metadata ──────────────────────────────────────────────────
    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {args.metadata}\n"
            "Run src/data/collect.py first to generate data."
        )

    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata)
    print(f"Loaded {len(df):,} images.")

    # Drop rows with missing coordinates (corrupted or incomplete downloads)
    df = df.dropna(subset=["lat", "lng", "filepath"])
    print(f"After dropping NaN coordinates: {len(df):,} images.")

    # ── Assign H3 cells ────────────────────────────────────────────────────
    print(f"Assigning H3 cells at resolution {args.resolution}...")
    df = assign_h3_cells(df, resolution=args.resolution)

    n_cells_before = df["h3_cell"].nunique()
    print(f"Total unique cells before filtering: {n_cells_before:,}")

    # ── Filter sparse cells ────────────────────────────────────────────────
    df = filter_sparse_cells(df, min_images=args.min_images)
    n_cells_after = df["h3_cell"].nunique()
    print(f"Cells after filtering: {n_cells_after:,}")

    # ── Build label encoder ────────────────────────────────────────────────
    cell_to_idx, idx_to_cell = build_label_encoder(df["h3_cell"].tolist())
    df["label"] = df["h3_cell"].map(cell_to_idx)

    # Precompute cell centers for inference (centroid prediction fallback)
    cell_centers = {
        cell: get_cell_center(cell) for cell in cell_to_idx.keys()
    }

    # ── Split dataset ──────────────────────────────────────────────────────
    train_df, val_df, test_df = split_dataset(df)

    # ── Save processed splits ──────────────────────────────────────────────
    train_df.to_csv(args.output_dir / "train.csv", index=False)
    val_df.to_csv(args.output_dir / "val.csv", index=False)
    test_df.to_csv(args.output_dir / "test.csv", index=False)

    # Save label encoder as JSON (human-readable) and pickle (fast load)
    label_map = {
        "cell_to_idx": cell_to_idx,
        "idx_to_cell": {str(k): v for k, v in idx_to_cell.items()},  # JSON needs str keys
        "cell_centers": {k: list(v) for k, v in cell_centers.items()},
        "num_classes": len(cell_to_idx),
        "h3_resolution": args.resolution,
    }
    with open(args.output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    with open(args.output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_map, f)

    # ── Save dataset stats ─────────────────────────────────────────────────
    stats = {
        "total_images": len(df),
        "num_cells": n_cells_after,
        "train_images": len(train_df),
        "val_images": len(val_df),
        "test_images": len(test_df),
        "h3_resolution": args.resolution,
        "min_images_per_cell": args.min_images,
        "images_per_cell_median": float(df["h3_cell"].value_counts().median()),
        "images_per_cell_mean": float(df["h3_cell"].value_counts().mean()),
    }
    with open(args.output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Visualizations ─────────────────────────────────────────────────────
    plot_coverage_map(df, args.output_dir / "coverage_map.png")
    plot_cell_distribution(df, args.output_dir / "cell_distribution.png")

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"  Total images:          {len(df):>8,}")
    print(f"  H3 resolution:         {args.resolution:>8}")
    print(f"  Valid cells (classes): {n_cells_after:>8,}")
    print(f"  Train / Val / Test:    {len(train_df):>5,} / {len(val_df):>4,} / {len(test_df):>4,}")
    print(f"  Median imgs/cell:      {stats['images_per_cell_median']:>8.1f}")
    print(f"  Mean imgs/cell:        {stats['images_per_cell_mean']:>8.1f}")
    print(f"\n  Outputs saved to: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
