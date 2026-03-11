"""
Loss functions for geographic prediction.

Two losses are combined during training:

1. CrossEntropyLoss — for cell classification
   Standard for multi-class problems. Encourages the model to assign
   probability mass to the correct H3 cell.

2. HaversineLoss — for coordinate regression
   Measures great-circle distance between predicted and true coordinates.
   Using actual km distance (not MSE in degrees) is important because
   1 degree of latitude ≠ 1 degree of longitude at different latitudes.

Combined loss = CE_loss + lambda * haversine_loss
where lambda (lambda_regression in config) balances the two objectives.

The Haversine regression head learns to pinpoint the exact location
within a cell, while the classification head handles which cell to predict.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Earth's mean radius in kilometers
EARTH_RADIUS_KM = 6371.0


class HaversineLoss(nn.Module):
    """
    Mean Haversine distance loss in kilometers.

    Haversine formula computes the great-circle distance between two points
    on a sphere given their latitudes and longitudes. This is the correct
    distance metric for geographic coordinates — unlike Euclidean or MSE
    loss which doesn't account for Earth's curvature or the non-uniform
    degree-to-km conversion at different latitudes.

    Input conventions:
        pred_coords: (B, 2) tensor of [lat, lng] in degrees
        true_coords: (B, 2) tensor of [lat, lng] in degrees
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: "mean" or "none". "mean" returns a scalar loss;
                       "none" returns per-sample distances for analysis.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self, pred_coords: torch.Tensor, true_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Haversine distance between predicted and true coordinates.

        Args:
            pred_coords: (B, 2) — predicted [lat, lng] in degrees
            true_coords: (B, 2) — ground truth [lat, lng] in degrees

        Returns:
            Scalar mean distance in km (or per-sample if reduction="none")
        """
        return haversine_distance(pred_coords, true_coords, reduction=self.reduction)


def haversine_distance(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Differentiable Haversine distance calculation.

    The formula:
        a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlng/2)
        c = 2 * atan2(√a, √(1−a))
        d = R * c

    We clamp `a` to [0, 1] before sqrt to avoid NaN gradients at a=1
    (antipodal points) or a<0 from floating point errors.

    Args:
        pred_coords: (B, 2) tensor [lat_deg, lng_deg]
        true_coords: (B, 2) tensor [lat_deg, lng_deg]
        reduction: "mean", "sum", or "none"

    Returns:
        Distance in km, scalar or (B,) depending on reduction
    """
    # Convert degrees to radians
    deg2rad = math.pi / 180.0
    pred_rad = pred_coords * deg2rad
    true_rad = true_coords * deg2rad

    lat1 = pred_rad[:, 0]
    lng1 = pred_rad[:, 1]
    lat2 = true_rad[:, 0]
    lng2 = true_rad[:, 1]

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    # Haversine formula
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2) ** 2
    )

    # Clamp to avoid NaN at antipodal points (numerical stability)
    a = torch.clamp(a, min=0.0, max=1.0)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distances = EARTH_RADIUS_KM * c

    if reduction == "mean":
        return distances.mean()
    elif reduction == "sum":
        return distances.sum()
    else:  # "none"
        return distances


class CombinedGeoLoss(nn.Module):
    """
    Combined classification + regression loss for geographic prediction.

    The model simultaneously predicts:
      - Which H3 cell the image is from (classification)
      - The exact lat/lng within that cell (regression)

    The two losses are trained jointly with a lambda weighting:
        total = CE(cell_logits, cell_labels) + lambda * Haversine(pred_coords, true_coords)

    Lambda is typically small (0.1) because:
      - CE loss magnitude is ~log(num_classes) ≈ 8 for 3000 classes
      - Haversine loss magnitude is ~500–2000 km at the start of training
      - Without lambda scaling, the regression loss would dominate
    """

    def __init__(self, lambda_regression: float = 0.1, label_smoothing: float = 0.1):
        """
        Args:
            lambda_regression: Weight for the Haversine regression loss.
            label_smoothing: Smoothing for CrossEntropyLoss. Helps prevent
                             overconfidence on the training set.
        """
        super().__init__()
        self.lambda_regression = lambda_regression
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.haversine_loss = HaversineLoss(reduction="mean")

    def forward(
        self,
        cell_logits: torch.Tensor,
        pred_coords: torch.Tensor,
        true_labels: torch.Tensor,
        true_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            cell_logits: (B, num_classes) raw logits from classification head
            pred_coords: (B, 2) predicted [lat, lng] from regression head
            true_labels: (B,) ground truth cell class indices
            true_coords: (B, 2) ground truth [lat, lng] in degrees

        Returns:
            total_loss: scalar tensor for backprop
            loss_dict: {"ce": float, "haversine_km": float, "total": float}
                       for logging
        """
        ce = self.ce_loss(cell_logits, true_labels)
        haversine = self.haversine_loss(pred_coords, true_coords)
        total = ce + self.lambda_regression * haversine

        loss_dict = {
            "ce": ce.item(),
            "haversine_km": haversine.item(),
            "total": total.item(),
        }
        return total, loss_dict
