# GeoGuessr AI

A GeoGuessr-style machine learning system that predicts geographic locations from street-level images. Given a photo, the model predicts where in the world it was taken.

## Architecture

```
                         ┌─────────────────────────────────────────┐
  Street Image           │           GeoGuessr AI Model            │
  (3 × 224 × 224)  ───▶  │                                         │
                         │   ┌─────────────────────────┐           │
                         │   │  EfficientNet-B4         │          │
                         │   │  (pretrained ImageNet)   │          │
                         │   │  feature_dim = 1792      │          │
                         │   └──────────┬──────────────┘           │
                         │              │ features (1792,)         │
                         │    ┌─────────┴──────────┐               │
                         │    │                    │               │
                         │  ┌─▼──────────┐  ┌─────▼──────────┐     │
                         │  │Classification│  │  Regression    │   │
                         │  │    Head      │  │    Head        │   │
                         │  │              │  │                │   │
                         │  │ Dropout(0.4) │  │ Linear → 2     │   │
                         │  │ Linear→512   │  │ tanh × [90,180]│   │
                         │  │ ReLU         │  └────────┬───────┘   │
                         │  │ Dropout(0.2) │           │           │
                         │  │ Linear→N     │     [lat, lng]        │
                         │  └──────┬───────┘                       │
                         │         │                               │
                         │    Top-K H3 cells                       │
                         └─────────────────────────────────────────┘

  Loss = CrossEntropy(cell_logits, true_cell) + λ × Haversine(pred_coords, true_coords)
```

## Project Structure

```
GeoGuessrAi/
├── src/
│   ├── data/
│   │   ├── collect.py        # Mapillary API v4 data collection + synthetic fallback
│   │   ├── preprocess.py     # H3 cell assignment, filtering, train/val/test split
│   │   └── dataset.py        # PyTorch Dataset with augmentations
│   ├── models/
│   │   ├── backbone.py       # EfficientNet-B4 feature extractor (timm)
│   │   ├── classifier.py     # Full model: backbone + dual prediction heads
│   │   └── losses.py         # Haversine loss + combined CE + regression loss
│   ├── training/
│   │   ├── config.py         # All hyperparameters (no circular imports)
│   │   ├── train.py          # Training loop with AMP, OneCycleLR, early stopping
│   │   └── evaluate.py       # Test-set evaluation with detailed metrics
│   └── inference/
│       ├── predict.py        # Single-image inference from file path or URL
│       └── visualize.py      # Folium interactive map generation
├── app/
│   └── demo.py               # Gradio web interface
├── data/
│   ├── raw/                  # Downloaded images + metadata.csv
│   └── processed/            # Train/val/test CSVs + label encoder
├── checkpoints/              # Saved model weights
├── outputs/                  # Generated maps and visualizations
├── examples/                 # Example images for the demo
├── test_smoke.py             # End-to-end smoke test (100 synthetic images)
└── requirements.txt
```

## Setup

### 1. Install Dependencies

```bash
# Python 3.10+ required
pip install -r requirements.txt

# Verify installation
python -c "import torch; import timm; import h3; print('OK')"
```

### 2. API Keys

**Mapillary API** (for real data collection):
1. Create an account at [mapillary.com](https://www.mapillary.com)
2. Go to Settings → Developer → Generate token
3. Export: `export MAPILLARY_TOKEN=your_token_here`

**Weights & Biases** (optional, for experiment tracking):
```bash
pip install wandb
wandb login
```

## Pipeline

### Step 1 — Collect Data

```bash
# Real data from Mapillary:
python src/data/collect.py --token $MAPILLARY_TOKEN --max_images 10000

# Synthetic data (no token needed, for testing):
python src/data/collect.py --max_images 1000 --synthetic

# Custom region (e.g. Europe only):
python src/data/collect.py --token $MAPILLARY_TOKEN \
    --bbox 35.0 -10.0 71.0 40.0 --max_images 50000
```

Output: `data/raw/` with JPEG images and `data/raw/metadata.csv`

### Step 2 — Preprocess

```bash
python src/data/preprocess.py

# Custom resolution (4 = finer cells, needs more data):
python src/data/preprocess.py --resolution 4 --min_images 20
```

Output: `data/processed/` with train/val/test CSVs, label encoder, and coverage maps.

### Step 3 — Train

```bash
# Default config (50 epochs, EfficientNet-B4):
python src/training/train.py

# Custom hyperparameters:
python src/training/train.py --batch_size 64 --lr 5e-5 --epochs 100

# With wandb tracking:
python src/training/train.py --use_wandb

# Resume from checkpoint:
python src/training/train.py --resume checkpoints/best_model.pth
```

Best checkpoint saved to `checkpoints/best_model.pth`.

### Step 4 — Evaluate

```bash
python src/training/evaluate.py --checkpoint checkpoints/best_model.pth
```

Example output:
```
============================================================
  EVALUATION RESULTS  (test set, n=1,000)
============================================================

  Geographic Distance Accuracy:
  Metric                               Value
  ---------------------------------------------
  country_level (500km)               62.30%
  region_level  (200km)               38.70%
  city_level     (25km)               12.40%

  Distance Errors (regression head):
  Metric                               Value
  ---------------------------------------------
  Median Haversine distance            285.0 km
  Mean Haversine distance              621.3 km
  ...
```

### Step 5 — Predict Single Image

```bash
# From file:
python src/inference/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/street.jpg

# From URL with ground truth:
python src/inference/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --image https://example.com/street.jpg \
    --true_lat 48.8566 --true_lng 2.3522
```

### Step 6 — Run Demo

```bash
python app/demo.py
# Open http://localhost:7860 in your browser

# Public share link:
python app/demo.py --share
```

### Step 7 — Visualize Predictions

```bash
python src/inference/visualize.py \
    --pred_lat 48.85 --pred_lng 2.35 \
    --true_lat 51.50 --true_lng -0.12
# Opens outputs/prediction_map.html
```

## Smoke Test

Verify the full pipeline works before real training:

```bash
python test_smoke.py
```

This generates 100 synthetic images, runs the full pipeline (preprocess → train 2 epochs → evaluate → predict), and reports pass/fail for each stage.

## Expected Performance Benchmarks

Performance depends heavily on training data size and geographic coverage.

| Training Images | Median Error | Country Acc (500km) | Region Acc (200km) |
|----------------|-------------|--------------------|--------------------|
| 10,000         | ~1,500 km   | ~35%               | ~18%               |
| 100,000        | ~500 km     | ~55%               | ~30%               |
| 1,000,000+     | ~150 km     | ~70%               | ~50%               |

For comparison, top human GeoGuessr players achieve ~150-300km median error.
The IM2GPS paper (2009) baseline was ~700km median on global test sets.

## Hyperparameters

Key settings in `src/training/config.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `backbone` | `efficientnet_b4` | Any timm model works |
| `batch_size` | 32 | Increase if you have >8GB VRAM |
| `lr` | 1e-4 | Base LR for AdamW |
| `max_lr` | 1e-3 | Peak LR for OneCycleLR |
| `epochs` | 50 | Early stopping at patience=5 |
| `freeze_epochs` | 5 | Backbone frozen for warm-up |
| `lambda_regression` | 0.1 | Weight for Haversine loss |
| `image_size` | 224 | Input resolution |

## Limitations

- **Data bias**: Performance is much better in regions with dense Mapillary coverage (Europe, North America) than sparse regions (Central Africa, remote Asia).
- **Ambiguous scenes**: Indoor-like or featureless outdoor scenes are fundamentally difficult — even humans struggle with these in GeoGuessr.
- **H3 cell resolution**: At resolution 3, each cell is ~59km across. The regression head refines within a cell but can't exceed the training data density.
- **No temporal modeling**: The model sees a single frame. GeoGuessr allows looking in all directions; a multi-view ensemble would improve accuracy.
- **Synthetic data**: The fallback synthetic images have no realistic geographic features. Train on real Mapillary data for meaningful performance.

## License

MIT License. The model architecture uses EfficientNet-B4 (Apache 2.0 via timm) and H3 (Apache 2.0 via Uber). Training data from Mapillary is subject to their [terms of service](https://www.mapillary.com/terms).
