"""
Geographic prediction visualization using folium.

Generates an interactive HTML map showing:
  - Predicted location (red pin)
  - True location (green pin, if provided)
  - Great circle line between them
  - Distance annotation
  - Top-K candidate cells as blue markers

Usage:
    python src/inference/visualize.py \
        --pred_lat 48.85 --pred_lng 2.35 \
        --true_lat 51.50 --true_lng -0.12 \
        --output outputs/prediction_map.html
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import folium
    from folium.plugins import AntPath
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Pure-Python Haversine distance calculation for visualization use.

    Duplicates the PyTorch version but without tensor overhead,
    since visualization runs outside the training loop.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0, 1 - a)))


def create_prediction_map(
    pred_lat: float,
    pred_lng: float,
    true_lat: Optional[float] = None,
    true_lng: Optional[float] = None,
    top_k_cells: Optional[list[dict]] = None,
    country: str = "Unknown",
    output_path: Optional[Path] = None,
) -> str:
    """
    Creates an interactive folium map for a single prediction.

    Map features:
      - Red marker: predicted location (regression head output)
      - Green marker: true location (if provided)
      - Orange dashed arc: great circle path between prediction and truth
      - Blue markers: top-K classification candidate cells (if provided)
      - Distance annotation in the corner

    Args:
        pred_lat, pred_lng: Predicted coordinates
        true_lat, true_lng: Ground truth coordinates (optional)
        top_k_cells: List of candidate cells from classification head
        country: Estimated country for the title
        output_path: Where to save the HTML file. Defaults to outputs/prediction_map.html.

    Returns:
        Path to the saved HTML file as a string.
    """
    if not FOLIUM_AVAILABLE:
        raise ImportError(
            "folium is required for visualization. Install with: pip install folium"
        )

    if output_path is None:
        output_path = OUTPUTS_DIR / "prediction_map.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Center map between prediction and truth ────────────────────────────
    if true_lat is not None and true_lng is not None:
        center_lat = (pred_lat + true_lat) / 2
        center_lng = (pred_lng + true_lng) / 2
        # Compute distance for title
        dist_km = haversine_km(pred_lat, pred_lng, true_lat, true_lng)
    else:
        center_lat, center_lng = pred_lat, pred_lng
        dist_km = None

    # ── Create map ─────────────────────────────────────────────────────────
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=4,
        tiles="CartoDB positron",  # Clean, uncluttered basemap
    )

    # ── Predicted location (red pin) ───────────────────────────────────────
    folium.Marker(
        location=[pred_lat, pred_lng],
        popup=folium.Popup(
            f"<b>Prediction</b><br>"
            f"Lat: {pred_lat:.4f}°<br>"
            f"Lng: {pred_lng:.4f}°<br>"
            f"Country: {country}",
            max_width=200,
        ),
        tooltip=f"Predicted: {pred_lat:.2f}°, {pred_lng:.2f}°",
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(m)

    # ── True location (green pin) ──────────────────────────────────────────
    if true_lat is not None and true_lng is not None:
        folium.Marker(
            location=[true_lat, true_lng],
            popup=folium.Popup(
                f"<b>Ground Truth</b><br>"
                f"Lat: {true_lat:.4f}°<br>"
                f"Lng: {true_lng:.4f}°",
                max_width=200,
            ),
            tooltip=f"True: {true_lat:.2f}°, {true_lng:.2f}°",
            icon=folium.Icon(color="green", icon="map-marker", prefix="fa"),
        ).add_to(m)

        # ── Great circle line ──────────────────────────────────────────────
        # Use AntPath for an animated dashed line effect
        try:
            AntPath(
                locations=[[pred_lat, pred_lng], [true_lat, true_lng]],
                color="#e74c3c",
                weight=3,
                opacity=0.8,
                tooltip=f"Distance: {dist_km:.1f} km",
                dash_array=[10, 20],
            ).add_to(m)
        except Exception:
            # Fallback to static polyline if AntPath fails
            folium.PolyLine(
                locations=[[pred_lat, pred_lng], [true_lat, true_lng]],
                color="#e74c3c",
                weight=3,
                opacity=0.8,
                tooltip=f"Distance: {dist_km:.1f} km",
                dash_array="10 20",
            ).add_to(m)

        # ── Distance annotation (floating text box) ────────────────────────
        dist_html = f"""
        <div style="
            position: fixed; bottom: 30px; left: 30px;
            background-color: white; padding: 10px 15px;
            border: 2px solid #e74c3c; border-radius: 8px;
            font-family: Arial, sans-serif; font-size: 14px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3); z-index: 1000;
        ">
            <b>Prediction Error</b><br>
            🔴 Predicted: {pred_lat:.2f}°, {pred_lng:.2f}°<br>
            🟢 True: {true_lat:.2f}°, {true_lng:.2f}°<br>
            📏 Distance: <b>{dist_km:.1f} km</b>
        </div>
        """
        m.get_root().html.add_child(folium.Element(dist_html))

    # ── Top-K candidate cells (blue circles) ──────────────────────────────
    if top_k_cells:
        for cell in top_k_cells:
            lat = cell.get("cell_center_lat", cell.get("lat"))
            lng = cell.get("cell_center_lng", cell.get("lng"))
            confidence = cell.get("confidence", 0)
            rank = cell.get("rank", "?")

            if lat is None or lng is None:
                continue

            # Scale circle size by confidence
            radius = max(30000, confidence * 500000)

            folium.CircleMarker(
                location=[lat, lng],
                radius=8,
                color="#3498db",
                fill=True,
                fill_color="#3498db",
                fill_opacity=0.4 * confidence + 0.1,
                popup=folium.Popup(
                    f"<b>Candidate #{rank}</b><br>"
                    f"Cell: {cell.get('cell_id', 'N/A')}<br>"
                    f"Confidence: {confidence:.3f}<br>"
                    f"Center: {lat:.2f}°, {lng:.2f}°",
                    max_width=200,
                ),
                tooltip=f"#{rank}: {confidence:.3f}",
            ).add_to(m)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; top: 10px; right: 10px;
        background-color: white; padding: 10px;
        border: 1px solid #ccc; border-radius: 5px;
        font-family: Arial, sans-serif; font-size: 13px;
        box-shadow: 2px 2px 4px rgba(0,0,0,0.2); z-index: 1000;
    ">
        <b>GeoGuessr AI</b><br>
        <span style="color:red">●</span> Predicted location<br>
        <span style="color:green">●</span> True location<br>
        <span style="color:#3498db">●</span> Candidate cells
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Save map ───────────────────────────────────────────────────────────
    m.save(str(output_path))
    print(f"Interactive map saved to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive prediction map"
    )
    parser.add_argument("--pred_lat", type=float, required=True)
    parser.add_argument("--pred_lng", type=float, required=True)
    parser.add_argument("--true_lat", type=float, default=None)
    parser.add_argument("--true_lng", type=float, default=None)
    parser.add_argument("--country", type=str, default="Unknown")
    parser.add_argument(
        "--output", type=Path,
        default=OUTPUTS_DIR / "prediction_map.html"
    )
    args = parser.parse_args()

    create_prediction_map(
        pred_lat=args.pred_lat,
        pred_lng=args.pred_lng,
        true_lat=args.true_lat,
        true_lng=args.true_lng,
        country=args.country,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
