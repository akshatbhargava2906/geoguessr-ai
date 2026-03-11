"""
PyTorch Dataset for GeoGuessr AI.

Loads preprocessed (image, cell_label, lat, lng) samples and applies
augmentations appropriate for street-level photography.

Key design decisions:
- RandomHorizontalFlip is included (mirrors are valid scenes)
- RandomVerticalFlip is excluded (street images always have sky up, road down)
- ColorJitter simulates different lighting / camera conditions
- Raw coordinates are returned alongside the class label so we can compute
  the Haversine regression loss and evaluate in km, not just cell accuracy
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ImageNet statistics — required when using ImageNet-pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Training augmentation pipeline for street-level images.

    RandomResizedCrop simulates different zoom/crop scenarios (camera zoom varies).
    ColorJitter handles illumination differences (morning/night/overcast).
    No vertical flip because the street/sky orientation is a strong geographic cue.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,  # Slight hue shift for color cast simulation
        ),
        # Occasional Gaussian blur simulates motion blur / low res cameras
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """
    Validation/test transform: deterministic resize + normalize only.

    No augmentation at test time ensures reproducible evaluation metrics.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class GeoDataset(Dataset):
    """
    Dataset of street-level images mapped to geographic H3 cell classes.

    Each sample returns:
        image_tensor: (3, H, W) float32 tensor, ImageNet-normalized
        label:        int64, H3 cell class index
        lat:          float32, true latitude
        lng:          float32, true longitude

    Including raw coordinates alongside the class label is critical for
    computing the Haversine regression loss during training and for
    evaluating in km (not just cell accuracy) during evaluation.
    """

    def __init__(
        self,
        csv_path: Path,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        label_encoder_path: Optional[Path] = None,
    ):
        """
        Args:
            csv_path: Path to train.csv / val.csv / test.csv
            split: One of "train", "val", "test" — controls augmentation
            image_size: Target image size for transforms
            transform: Override default transforms (useful for custom augmentation)
            label_encoder_path: Path to label_encoder.pkl. Defaults to processed dir.
        """
        self.csv_path = Path(csv_path)
        self.split = split
        self.image_size = image_size

        # Load the manifest (filepath, label, lat, lng per row)
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {self.csv_path}\n"
                "Run src/data/preprocess.py first."
            )
        self.df = pd.read_csv(self.csv_path)

        # Load the label encoder for inverse lookups during inference
        if label_encoder_path is None:
            label_encoder_path = PROCESSED_DIR / "label_encoder.pkl"
        if label_encoder_path.exists():
            with open(label_encoder_path, "rb") as f:
                self.label_map = pickle.load(f)
        else:
            self.label_map = None

        # Select appropriate transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transform(image_size)
        else:
            self.transform = get_val_transform(image_size)

        # Track corrupt images so we skip them on subsequent epochs
        self._corrupt_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float, float]:
        """
        Loads and returns a single sample.

        Corrupt image handling:
            - On first encounter: log a warning, return a zero tensor
            - Subsequent encounters: return zero tensor immediately
            This avoids crashing the DataLoader while still training on
            valid data. The zero tensor will produce a valid but uninformative
            loss contribution.
        """
        if idx in self._corrupt_indices:
            return self._zero_sample(idx)

        row = self.df.iloc[idx]

        # Build the absolute path from the stored relative path
        img_path = PROJECT_ROOT / row["filepath"]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except (FileNotFoundError, OSError, Exception) as e:
            # Log only first occurrence to avoid flooding stdout
            print(f"[Dataset] Warning: corrupt/missing image at index {idx}: {img_path} — {e}")
            self._corrupt_indices.add(idx)
            return self._zero_sample(idx)

        label = int(row["label"])
        lat = float(row["lat"])
        lng = float(row["lng"])

        return image, label, lat, lng

    def _zero_sample(self, idx: int) -> tuple[torch.Tensor, int, float, float]:
        """
        Returns a zero-filled placeholder for corrupt/missing images.

        The label is 0 (first class) and coordinates are (0, 0) which is
        in the ocean — fine as a fallback, the zero image gives near-zero
        features so the loss contribution is minimal.
        """
        zero_image = torch.zeros(3, self.image_size, self.image_size)
        row = self.df.iloc[idx]
        return zero_image, int(row.get("label", 0)), float(row.get("lat", 0.0)), float(row.get("lng", 0.0))

    def get_num_classes(self) -> int:
        """Returns total number of geographic cell classes."""
        return self.df["label"].nunique()

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes inverse-frequency class weights for weighted CE loss.

        Useful when geographic imbalance is severe (Europe >> Antarctica).
        Not used by default but available for experimentation.
        """
        counts = self.df["label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights = weights / weights.sum()  # Normalize to sum to 1
        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    processed_dir: Path = PROCESSED_DIR,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Convenience function to build all three DataLoaders at once.

    Returns (train_loader, val_loader, test_loader).
    Validation and test loaders use shuffle=False for reproducibility.
    """
    train_dataset = GeoDataset(processed_dir / "train.csv", split="train", image_size=image_size)
    val_dataset = GeoDataset(processed_dir / "val.csv", split="val", image_size=image_size)
    test_dataset = GeoDataset(processed_dir / "test.csv", split="test", image_size=image_size)

    # Pin memory accelerates CPU→GPU transfers on systems with CUDA
    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for stable BatchNorm
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for faster eval (no gradients)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"DataLoaders ready: {len(train_dataset)} train / "
          f"{len(val_dataset)} val / {len(test_dataset)} test samples")

    return train_loader, val_loader, test_loader
