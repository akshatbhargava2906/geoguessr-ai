"""
Training entry point for GeoGuessr AI.

Implements:
  - AdamW optimizer + OneCycleLR scheduler
  - Mixed precision training (torch.cuda.amp)
  - Warm-up backbone freezing for first N epochs
  - Early stopping on validation Haversine distance
  - Comprehensive logging (train/val loss, accuracy @1/5/10, median km)
  - Optional wandb experiment tracking
  - Checkpoint saving (best model only to save disk space)

Usage:
    # Default config:
    python src/training/train.py

    # Custom hyperparameters:
    python src/training/train.py --batch_size 64 --lr 5e-5 --epochs 100

    # With wandb:
    python src/training/train.py --use_wandb

    # Resume from checkpoint:
    python src/training/train.py --resume checkpoints/best_model.pth
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Ensure project root is on path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_dataloaders
from src.models.classifier import build_model
from src.models.losses import CombinedGeoLoss, haversine_distance
from src.training.config import Config


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Computes top-K cell classification accuracy.

    Top-K accuracy is more informative than top-1 for geographic prediction
    because adjacent cells are often correct too — the model may predict the
    right neighborhood in the wrong hex.
    """
    _, top_k = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = top_k.eq(labels.view(-1, 1).expand_as(top_k))
    return correct.any(dim=1).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: CombinedGeoLoss,
    scaler: GradScaler,
    device: torch.device,
    config: Config,
    epoch: int,
) -> dict:
    """
    Runs one full training epoch.

    Returns a dict with mean losses and accuracy metrics for logging.
    """
    model.train()
    total_ce = 0.0
    total_haversine = 0.0
    total_loss = 0.0
    total_correct_1 = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]", leave=False)
    for batch_idx, (images, labels, lats, lngs) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        true_coords = torch.stack([lats, lngs], dim=1).float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # set_to_none is faster than zero_grad()

        # Mixed precision forward pass
        with autocast(enabled=device.type == "cuda"):
            cell_logits, pred_coords = model(images)
            loss, loss_dict = criterion(cell_logits, pred_coords, labels, true_coords)

        # Scaled backward pass (handles fp16 gradient underflow)
        scaler.scale(loss).backward()
        # Gradient clipping prevents exploding gradients after backbone unfreeze
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # OneCycleLR updates every batch, not every epoch
        scheduler.step()

        # Accumulate metrics
        bs = images.size(0)
        total_ce += loss_dict["ce"] * bs
        total_haversine += loss_dict["haversine_km"] * bs
        total_loss += loss_dict["total"] * bs
        total_correct_1 += (cell_logits.argmax(1) == labels).sum().item()
        total_samples += bs

        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.3f}",
                "hav": f"{loss_dict['haversine_km']:.0f}km",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

    return {
        "train/ce_loss": total_ce / total_samples,
        "train/haversine_km": total_haversine / total_samples,
        "train/total_loss": total_loss / total_samples,
        "train/acc_top1": total_correct_1 / total_samples,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: CombinedGeoLoss,
    device: torch.device,
    label_map: dict,
) -> dict:
    """
    Evaluates model on validation set.

    Computes:
      - CE loss and total loss
      - Top-1, Top-5, Top-10 cell accuracy
      - Median and mean Haversine distance in km
    """
    model.eval()
    total_ce = 0.0
    total_loss = 0.0
    all_distances = []
    all_logits = []
    all_labels = []

    for images, labels, lats, lngs in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        true_coords = torch.stack([lats, lngs], dim=1).float().to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            cell_logits, pred_coords = model(images)
            loss, loss_dict = criterion(cell_logits, pred_coords, labels, true_coords)

        # Per-sample Haversine distances for median computation
        distances = haversine_distance(pred_coords, true_coords, reduction="none")
        all_distances.append(distances.cpu())
        all_logits.append(cell_logits.cpu())
        all_labels.append(labels.cpu())

        bs = images.size(0)
        total_ce += loss_dict["ce"] * bs
        total_loss += loss_dict["total"] * bs

    n = sum(l.size(0) for l in all_labels)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_distances = torch.cat(all_distances, dim=0)

    # For H3 cell accuracy: the regression head predicts within-cell location,
    # so cell accuracy is our primary classification metric
    acc1 = accuracy_at_k(all_logits, all_labels, k=1)
    acc5 = accuracy_at_k(all_logits, all_labels, k=5)
    acc10 = accuracy_at_k(all_logits, all_labels, k=10)

    return {
        "val/ce_loss": total_ce / n,
        "val/total_loss": total_loss / n,
        "val/acc_top1": acc1,
        "val/acc_top5": acc5,
        "val/acc_top10": acc10,
        "val/haversine_median_km": all_distances.median().item(),
        "val/haversine_mean_km": all_distances.mean().item(),
    }


def try_init_wandb(config: Config) -> bool:
    """
    Attempts to initialize wandb. Returns True on success.

    Graceful failure: if wandb isn't installed or WANDB_API_KEY isn't set,
    we just skip logging rather than crashing the training run.
    """
    if not config.use_wandb:
        return False
    try:
        import wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=config.to_dict(),
        )
        print("wandb initialized successfully")
        return True
    except Exception as e:
        print(f"wandb initialization failed (skipping): {e}")
        return False


def log_metrics(metrics: dict, wandb_active: bool, step: int) -> None:
    """Log metrics to stdout (always) and wandb (if active)."""
    if wandb_active:
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            pass


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: Config,
    label_map: dict,
) -> None:
    """Saves a full training checkpoint including model + optimizer state."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_haversine_km": metrics.get("val/haversine_median_km", float("inf")),
        "val_acc_top1": metrics.get("val/acc_top1", 0.0),
        "config": config.to_dict(),
        "num_classes": label_map["num_classes"],
    }
    torch.save(checkpoint, config.best_checkpoint_path)
    print(f"  Saved checkpoint to {config.best_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GeoGuessr AI model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (unused, for compatibility)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--freeze_epochs", type=int, default=None)
    parser.add_argument("--lambda_regression", type=float, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--processed_dir", type=Path, default=None)
    args = parser.parse_args()

    # ── Build config (override defaults with CLI args) ─────────────────────
    config = Config()
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.lr is not None: config.lr = args.lr
    if args.epochs is not None: config.epochs = args.epochs
    if args.freeze_epochs is not None: config.freeze_epochs = args.freeze_epochs
    if args.lambda_regression is not None: config.lambda_regression = args.lambda_regression
    if args.use_wandb: config.use_wandb = True
    if args.processed_dir is not None: config.processed_dir = args.processed_dir

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU will be very slow. A GPU is strongly recommended.")

    # ── Load label encoder ─────────────────────────────────────────────────
    label_encoder_path = config.processed_dir / "label_encoder.pkl"
    if not label_encoder_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {label_encoder_path}\n"
            "Run src/data/preprocess.py first."
        )
    with open(label_encoder_path, "rb") as f:
        label_map = pickle.load(f)
    num_classes = label_map["num_classes"]
    print(f"Number of geographic cell classes: {num_classes}")

    # ── Build dataloaders ──────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        processed_dir=config.processed_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
    )

    # ── Build model ────────────────────────────────────────────────────────
    model = build_model(
        num_cells=num_classes,
        backbone_name=config.backbone,
        pretrained=True,
        checkpoint_path=args.resume,
        device=device,
    )

    # ── Loss function ──────────────────────────────────────────────────────
    criterion = CombinedGeoLoss(
        lambda_regression=config.lambda_regression,
        label_smoothing=config.label_smoothing,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    # AdamW decouples weight decay from the gradient update — important for
    # transformers and modern CNN architectures
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    # OneCycleLR: linearly warm up LR then cosine anneal.
    # This is the most reliable scheduler for fine-tuning pretrained models.
    total_steps = len(train_loader) * config.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        total_steps=total_steps,
        pct_start=config.pct_start,
        anneal_strategy="cos",
    )

    # ── Mixed precision scaler ─────────────────────────────────────────────
    # GradScaler is a no-op on CPU but needed for CUDA fp16
    scaler = GradScaler(enabled=device.type == "cuda")

    # ── wandb ─────────────────────────────────────────────────────────────
    wandb_active = try_init_wandb(config)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_haversine = float("inf")
    epochs_without_improvement = 0
    start_time = time.time()

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Backbone will be frozen for the first {config.freeze_epochs} epoch(s).\n")

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # ── Warm-up backbone freezing ──────────────────────────────────────
        if epoch == 0:
            model.freeze_backbone()
        elif epoch == config.freeze_epochs:
            model.unfreeze_backbone()
            print(f"\nEpoch {epoch+1}: Backbone unfrozen. Starting full fine-tuning.")

        # ── Train one epoch ────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device, config, epoch
        )

        # ── Validate ───────────────────────────────────────────────────────
        val_metrics = validate(model, val_loader, criterion, device, label_map)

        # ── Log metrics ────────────────────────────────────────────────────
        all_metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        log_metrics(all_metrics, wandb_active, step=epoch + 1)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch+1:3d}/{config.epochs} ({elapsed:.0f}s) | "
            f"Loss: {train_metrics['train/total_loss']:.4f} | "
            f"Val Loss: {val_metrics['val/total_loss']:.4f} | "
            f"Acc @1/5/10: {val_metrics['val/acc_top1']:.3f}/{val_metrics['val/acc_top5']:.3f}/{val_metrics['val/acc_top10']:.3f} | "
            f"Median Hav: {val_metrics['val/haversine_median_km']:.0f}km"
        )

        # ── Early stopping & checkpoint ────────────────────────────────────
        val_haversine = val_metrics["val/haversine_median_km"]
        if val_haversine < best_val_haversine:
            best_val_haversine = val_haversine
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, config, label_map)
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{config.patience} epochs.")

        if epochs_without_improvement >= config.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    # ── Final summary ──────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes.")
    print(f"Best validation Haversine distance: {best_val_haversine:.1f} km")
    print(f"Best checkpoint saved to: {config.best_checkpoint_path}")

    if wandb_active:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
