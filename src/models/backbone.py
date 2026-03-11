"""
EfficientNet-B4 backbone for geographic feature extraction.

We use timm (PyTorch Image Models) to load a pretrained EfficientNet-B4.
EfficientNet-B4 is chosen because:
  - Strong ImageNet accuracy (83% top-1) relative to parameter count
  - Compound scaling balances depth/width/resolution efficiently
  - 19M parameters — heavy enough to learn fine-grained scene cues,
    but trainable on a single GPU in reasonable time

The backbone is used as a feature extractor only; the classification
and regression heads are in classifier.py.

Warm-up training:
  During the first `freeze_epochs` epochs, we freeze the backbone weights
  and only train the newly initialized heads. This prevents the randomly
  initialized heads from destroying the pretrained features in early training.
"""

import timm
import torch
import torch.nn as nn


class GeoBackbone(nn.Module):
    """
    Pretrained EfficientNet-B4 feature extractor.

    Usage:
        backbone = GeoBackbone(model_name="efficientnet_b4", pretrained=True)
        features = backbone(images)  # (B, feature_dim)
        head = nn.Linear(backbone.feature_dim, num_classes)
    """

    def __init__(self, model_name: str = "efficientnet_b4", pretrained: bool = True):
        """
        Args:
            model_name: Any timm model name (e.g. "efficientnet_b4", "convnext_base").
                        EfficientNet-B4 is the default and recommended choice.
            pretrained: Load ImageNet pretrained weights. Set False only for ablations.
        """
        super().__init__()

        # `num_classes=0` removes the original classification head,
        # giving us raw feature vectors instead of ImageNet logits.
        # `global_pool="avg"` applies global average pooling to collapse
        # spatial dimensions: (B, C, H, W) → (B, C).
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # No classification head
            global_pool="avg",   # Global average pooling
        )

        # timm exposes num_features for the feature vector dimensionality
        self.feature_dim = self.encoder.num_features

        print(f"Backbone: {model_name} | Feature dim: {self.feature_dim} | "
              f"Pretrained: {pretrained}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of images.

        Args:
            x: (B, 3, H, W) ImageNet-normalized image batch

        Returns:
            features: (B, feature_dim) float32 feature vectors
        """
        return self.encoder(x)

    def freeze(self) -> None:
        """
        Freeze all backbone parameters.

        Called during warm-up epochs so only the heads are updated.
        Freezing prevents gradient flow into pretrained weights early in
        training when the loss is large and gradients are noisy.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Backbone frozen (warm-up mode)")

    def unfreeze(self) -> None:
        """
        Unfreeze all backbone parameters for end-to-end fine-tuning.

        Called after warm-up. At this point, the heads are better initialized
        and the gradients are more meaningful, making fine-tuning stable.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Backbone unfrozen (fine-tuning mode)")

    def count_trainable_params(self) -> int:
        """Returns number of parameters currently requiring gradients."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
