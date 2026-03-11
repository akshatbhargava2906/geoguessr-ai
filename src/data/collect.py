"""
Data collection script for GeoGuessr AI.

Downloads geotagged street-level images from the Mapillary API v4.
Falls back to synthetic data generation when no API token is provided,
making the entire pipeline testable without credentials.

Usage:
    # Real data with Mapillary:
    python src/data/collect.py --token YOUR_TOKEN --max_images 5000

    # Synthetic data (no token needed):
    python src/data/collect.py --max_images 100 --synthetic
"""

import argparse
import csv
import io
import math
import os
import random
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# Project root is two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_CSV = RAW_DATA_DIR / "metadata.csv"

# Mapillary API constants
MAPILLARY_BASE_URL = "https://graph.mapillary.com"
MAPILLARY_TILE_URL = "https://tiles.mapillary.com"

# Rate limiting: Mapillary free tier allows ~100 req/min
REQUEST_DELAY_SECONDS = 0.7  # ~85 requests/min, safe margin


def rate_limited_get(url: str, params: dict, token: str, retries: int = 3) -> Optional[dict]:
    """
    Makes a rate-limited GET request to the Mapillary API.

    We sleep between requests to avoid hitting the 100 req/min ceiling.
    Retries on transient network errors with exponential backoff.
    """
    headers = {"Authorization": f"OAuth {token}"}
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                # Rate limited — back off longer
                wait = 60 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
            elif response.status_code in (500, 502, 503):
                wait = 5 * (2 ** attempt)
                print(f"Server error {response.status_code}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.RequestException as e:
            wait = 5 * (2 ** attempt)
            print(f"Network error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None


def grid_sample_bbox(
    min_lat: float, min_lng: float, max_lat: float, max_lng: float, grid_steps: int = 10
) -> Iterator[Tuple[float, float, float, float]]:
    """
    Divides a bounding box into a grid of sub-bboxes and yields each cell.

    Using a grid ensures geographic diversity — simply querying the full bbox
    would return images clustered in popular areas only.
    """
    lat_step = (max_lat - min_lat) / grid_steps
    lng_step = (max_lng - min_lng) / grid_steps

    for i in range(grid_steps):
        for j in range(grid_steps):
            cell_min_lat = min_lat + i * lat_step
            cell_max_lat = cell_min_lat + lat_step
            cell_min_lng = min_lng + j * lng_step
            cell_max_lng = cell_min_lng + lng_step
            yield cell_min_lat, cell_min_lng, cell_max_lat, cell_max_lng


def fetch_images_in_bbox(
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    token: str,
    limit: int = 100,
) -> list[dict]:
    """
    Queries Mapillary API v4 for images within a bounding box.

    The fields parameter controls what metadata we get back.
    We request thumb_256_url for bandwidth efficiency.
    """
    bbox_str = f"{min_lng},{min_lat},{max_lng},{max_lat}"
    params = {
        "fields": "id,thumb_256_url,geometry,captured_at",
        "bbox": bbox_str,
        "limit": limit,
    }
    data = rate_limited_get(f"{MAPILLARY_BASE_URL}/images", params, token)
    if data is None or "data" not in data:
        return []
    return data["data"]


def download_image(url: str, save_path: Path) -> bool:
    """
    Downloads a single image from URL and saves as JPEG.

    Returns True on success, False on any failure.
    We catch everything here because a single bad image shouldn't crash the run.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)
        img.save(save_path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def collect_from_mapillary(
    token: str,
    bbox: Tuple[float, float, float, float],
    max_images: int,
    output_dir: Path,
    metadata_path: Path,
) -> int:
    """
    Main collection loop: queries Mapillary grid → downloads images → writes CSV.

    Returns the number of successfully collected images.
    """
    min_lat, min_lng, max_lat, max_lng = bbox
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = 0
    seen_ids = set()

    # Open CSV in append mode so we can resume interrupted runs
    csv_exists = metadata_path.exists()
    csv_file = open(metadata_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=["image_id", "lat", "lng", "filepath"])
    if not csv_exists:
        writer.writeheader()

    grid_cells = list(grid_sample_bbox(min_lat, min_lng, max_lat, max_lng, grid_steps=10))
    random.shuffle(grid_cells)  # Shuffle for diversity when stopping early

    with tqdm(total=max_images, desc="Collecting images") as pbar:
        for cell_bbox in grid_cells:
            if collected >= max_images:
                break

            images = fetch_images_in_bbox(*cell_bbox, token=token, limit=100)

            for img_meta in images:
                if collected >= max_images:
                    break

                img_id = img_meta.get("id")
                if img_id in seen_ids:
                    continue
                seen_ids.add(img_id)

                # Extract coordinates from GeoJSON geometry
                geometry = img_meta.get("geometry", {})
                if geometry.get("type") != "Point":
                    continue
                lng, lat = geometry["coordinates"]  # GeoJSON is [lng, lat]

                thumb_url = img_meta.get("thumb_256_url")
                if not thumb_url:
                    continue

                save_path = output_dir / f"{img_id}.jpg"
                if save_path.exists():
                    # Resume: skip already downloaded images
                    seen_ids.add(img_id)
                    collected += 1
                    pbar.update(1)
                    continue

                success = download_image(thumb_url, save_path)
                if success:
                    writer.writerow({
                        "image_id": img_id,
                        "lat": lat,
                        "lng": lng,
                        "filepath": str(save_path.relative_to(PROJECT_ROOT)),
                    })
                    csv_file.flush()
                    collected += 1
                    pbar.update(1)

    csv_file.close()
    return collected


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------

def generate_synthetic_image(lat: float, lng: float) -> Image.Image:
    """
    Creates a fake 256x256 'street scene' for pipeline testing.

    The image encodes rough geographic info as HSV color so different
    regions produce visually distinct images, making the smoke test
    meaningful (the model can learn *something* even on fake data).
    """
    # Map lat/lng to HSV color space for geographic variation
    hue = int(((lng + 180) / 360) * 179)          # longitude → hue
    saturation = int(((lat + 90) / 180) * 200 + 55)  # latitude → saturation
    value = random.randint(120, 200)               # random brightness

    # Create base color block
    img_array = np.full((256, 256, 3), [hue, saturation, value], dtype=np.uint8)

    # Add some noise to simulate texture
    noise = np.random.randint(-20, 20, (256, 256, 3), dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array, mode="RGB")


def collect_synthetic(
    max_images: int,
    output_dir: Path,
    metadata_path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> int:
    """
    Generates synthetic dataset with random global lat/lng points.

    Useful for:
    - Testing the full pipeline without Mapillary credentials
    - Smoke tests in CI/CD
    - Verifying preprocessing/training code correctness
    """
    if bbox is None:
        # Default: sample globally, but bias toward populated areas
        bbox = (-60.0, -180.0, 75.0, 180.0)

    min_lat, min_lng, max_lat, max_lng = bbox
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_id", "lat", "lng", "filepath"])
        writer.writeheader()

        for i in tqdm(range(max_images), desc="Generating synthetic images"):
            lat = random.uniform(min_lat, max_lat)
            lng = random.uniform(min_lng, max_lng)

            img = generate_synthetic_image(lat, lng)
            img_id = f"synthetic_{i:06d}"
            save_path = output_dir / f"{img_id}.jpg"
            img.save(save_path, "JPEG", quality=85)

            writer.writerow({
                "image_id": img_id,
                "lat": lat,
                "lng": lng,
                "filepath": str(save_path.relative_to(PROJECT_ROOT)),
            })

    print(f"Generated {max_images} synthetic images in {output_dir}")
    return max_images


def main():
    parser = argparse.ArgumentParser(
        description="Collect geotagged street-level images for GeoGuessr AI training"
    )
    parser.add_argument("--token", type=str, default=None, help="Mapillary API v4 access token")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LAT", "MIN_LNG", "MAX_LAT", "MAX_LNG"),
        default=[-60.0, -180.0, 75.0, 180.0],
        help="Bounding box for collection (default: global)",
    )
    parser.add_argument("--max_images", type=int, default=10000, help="Maximum images to collect")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data instead of downloading from Mapillary",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory to save images and metadata",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    metadata_path = output_dir / "metadata.csv"

    if args.synthetic or args.token is None:
        if args.token is None and not args.synthetic:
            print("No Mapillary token provided. Falling back to synthetic data generation.")
            print("Use --token YOUR_TOKEN to collect real data from Mapillary.")
        n = collect_synthetic(
            max_images=args.max_images,
            output_dir=output_dir,
            metadata_path=metadata_path,
            bbox=tuple(args.bbox),
        )
    else:
        n = collect_from_mapillary(
            token=args.token,
            bbox=tuple(args.bbox),
            max_images=args.max_images,
            output_dir=output_dir,
            metadata_path=metadata_path,
        )

    print(f"\nCollection complete. {n} images saved to {output_dir}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
