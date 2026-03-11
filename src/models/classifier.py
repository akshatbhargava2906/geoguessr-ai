"""
GeoGuessr AI full model: backbone + classification head + regression head.

Architecture overview:
                    ┌─────────────────────────┐
    Image (3,224,224) │   EfficientNet-B4        │  features (1792,)
                    │   (pretrained backbone)   │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
    ┌─────────▼──────────┐           ┌──────────▼─────────┐
    │  Classification    │           │   Regression Head  │
    │  Dropout(0.4)      │           │   Linear → 2       │
    │  Linear → 512      │           │   tanh scaled       │
    │  ReLU              │           │   → [lat, lng]     │
    │  Dropout(0.2)      │           └────────────────────┘
    │  Linear → N_cells  │
    └────────────────────┘

The dual-head design allows us to use both:
  1. Cell classification: coarse but robust (handles image ambiguity)
  2. Coordinate regression: fine-grained within-cell localization

At inference time, we combine both: use the classification head to
identify the top-K candidate cells, then use the regression head to
pinpoint the location within the predicted region.
"""

import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.models.backbone import GeoBackbone

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GeoClassifier(nn.Module):
    """
    Full geographic prediction model.

    Inputs:  (B, 3, 224, 224) image batch
    Outputs:
        cell_logits:  (B, num_cells) — raw logits for H3 cell classification
        pred_coords:  (B, 2)         — predicted [lat, lng] in degrees
    """

    def __init__(
        self,
        num_cells: int,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout_rate_1: float = 0.4,
        dropout_rate_2: float = 0.2,
        hidden_dim: int = 512,
    ):
        """
        Args:
            num_cells: Number of H3 geographic cell classes (output size).
            backbone_name: timm model name for the encoder backbone.
            pretrained: Whether to load ImageNet pretrained weights.
            dropout_rate_1: Dropout before first linear layer (higher = more regularization).
            dropout_rate_2: Dropout between hidden layers.
            hidden_dim: Width of the intermediate layer in the classification head.
        """
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone = GeoBackbone(model_name=backbone_name, pretrained=pretrained)
        feat_dim = self.backbone.feature_dim

        # ── Classification head ───────────────────────────────────────────
        # Dropout(0.4) → Linear(feat, 512) → ReLU → Dropout(0.2) → Linear(512, N)
        # The two-layer head gives more capacity than a single linear layer,
        # while still being much smaller than the backbone.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate_1),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate_2),
            nn.Linear(hidden_dim, num_cells),
        )

        # ── Regression head ───────────────────────────────────────────────
        # Direct lat/lng prediction from backbone features.
        # tanh outputs in [-1, 1], scaled to:
        #   lat: [-90, 90] (multiply by 90)
        #   lng: [-180, 180] (multiply by 180)
        # Using tanh + scaling is numerically better than directly predicting
        # degrees, as tanh saturates and prevents runaway coordinate predictions.
        self.regressor = nn.Linear(feat_dim, 2)  # 2 outputs: [lat, lng]

        # Initialize regression head weights small to avoid large initial haversine loss
        nn.init.xavier_uniform_(self.regressor.weight, gain=0.01)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: extract features, classify cell, regress coordinates.

        Args:
            x: (B, 3, H, W) ImageNet-normalized image batch

        Returns:
            cell_logits: (B, num_cells) unnormalized class scores
            pred_coords: (B, 2) predicted coordinates [lat, lng] in degrees
        """
        # Extract backbone features: (B, feat_dim)
        features = self.backbone(x)

        # Classification: (B, num_cells)
        cell_logits = self.classifier(features)

        # Regression: tanh → scale to coordinate ranges
        raw_coords = self.regressor(features)          # (B, 2), unbounded
        pred_coords = torch.tanh(raw_coords)           # (B, 2), range [-1, 1]

        # Scale each dimension to its valid range
        lat = pred_coords[:, 0:1] * 90.0    # [-90, 90]
        lng = pred_coords[:, 1:2] * 180.0   # [-180, 180]
        pred_coords = torch.cat([lat, lng], dim=1)  # (B, 2)

        return cell_logits, pred_coords

    def freeze_backbone(self) -> None:
        """Delegate to backbone freeze for warm-up training."""
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        """Delegate to backbone unfreeze for fine-tuning."""
        self.backbone.unfreeze()

    def count_parameters(self) -> dict[str, int]:
        """Returns parameter counts by component for logging."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = (
            sum(p.numel() for p in self.classifier.parameters())
            + sum(p.numel() for p in self.regressor.parameters())
        )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone": backbone_params,
            "heads": head_params,
            "total": backbone_params + head_params,
            "trainable": trainable,
        }


def build_model(
    num_cells: int,
    backbone_name: str = "efficientnet_b4",
    pretrained: bool = True,
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> GeoClassifier:
    """
    Factory function: creates and optionally loads a model from checkpoint.

    Args:
        num_cells: Number of geographic cell classes.
        backbone_name: timm backbone model name.
        pretrained: Use ImageNet weights for new models.
        checkpoint_path: If provided, load weights from this .pth file.
        device: Target device. Defaults to CUDA if available, else CPU.

    Returns:
        GeoClassifier model on the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GeoClassifier(
        num_cells=num_cells,
        backbone_name=backbone_name,
        pretrained=pretrained and (checkpoint_path is None),
    )

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Support both raw state_dict and wrapped checkpoint dicts
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        if "epoch" in checkpoint:
            print(f"  Resumed from epoch {checkpoint['epoch']}")
        if "val_haversine_km" in checkpoint:
            print(f"  Best val Haversine: {checkpoint['val_haversine_km']:.1f} km")

    model = model.to(device)

    # Log parameter counts
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    return model
