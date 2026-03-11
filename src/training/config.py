"""
Central configuration for GeoGuessr AI training.

All hyperparameters live here. This module is imported by train.py, evaluate.py,
and any script that needs training settings — there are NO circular imports because
config.py imports nothing from the project itself.

To run a custom experiment, either:
  1. Edit this file directly (good for one-off runs)
  2. Override via command-line args in train.py (recommended for sweeps)
  3. Subclass Config and override specific fields
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Path constants ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass
class Config:
    """
    Training hyperparameter configuration.

    Using a dataclass instead of a dict ensures:
      - Type hints for IDE support
      - Default values with easy overrides
      - Serialization to JSON for experiment tracking
    """

    # ── Data ──────────────────────────────────────────────────────────────
    processed_dir: Path = PROCESSED_DIR
    image_size: int = 224
    num_workers: int = 4

    # ── Model ─────────────────────────────────────────────────────────────
    backbone: str = "efficientnet_b4"

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Warm-up: freeze backbone for first N epochs while heads learn
    # This prevents the randomly initialized heads from corrupting pretrained features
    freeze_epochs: int = 5

    # Loss weighting: CE + lambda * Haversine
    # lambda is small because Haversine magnitudes (~500km) >> CE (~8 nats)
    lambda_regression: float = 0.1

    # Label smoothing prevents overconfidence (common in large-class problems)
    label_smoothing: float = 0.1

    # ── Optimizer / Scheduler ─────────────────────────────────────────────
    # OneCycleLR finds the learning rate peak automatically and includes
    # cosine annealing — generally outperforms fixed LR + manual scheduling
    max_lr: float = 1e-3       # Peak LR for OneCycleLR (10× base LR is a good starting point)
    pct_start: float = 0.3     # Fraction of cycle spent increasing LR

    # ── Early stopping ────────────────────────────────────────────────────
    # Stop if val Haversine doesn't improve for this many epochs
    patience: int = 5

    # ── Checkpointing ─────────────────────────────────────────────────────
    checkpoint_dir: Path = CHECKPOINTS_DIR
    checkpoint_name: str = "best_model.pth"

    # ── Experiment tracking ────────────────────────────────────────────────
    # wandb is optional — the training script will skip it gracefully
    # if the package isn't installed or WANDB_API_KEY isn't set
    use_wandb: bool = False
    wandb_project: str = "geoguessr-ai"
    wandb_entity: Optional[str] = None   # Set to your wandb username/org

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int = 42
    log_interval: int = 50    # Log every N batches within an epoch

    def __post_init__(self):
        """Create output directories at config load time."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize config to a plain dict (for wandb logging)."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }

    @property
    def best_checkpoint_path(self) -> Path:
        """Full path to the best model checkpoint file."""
        return self.checkpoint_dir / self.checkpoint_name


# Default config instance — import this directly in other modules:
#   from src.training.config import Config
cfg = Config()
