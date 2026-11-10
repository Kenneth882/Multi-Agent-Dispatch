"""
ride-share-marl/pipeline/zone_grid.py
---------------------------------------
Builds a geographically correct mapping from NYC TLC taxi zone LocationIDs
to (row, col) cells in a 10x10 simulation grid covering Manhattan.

Methodology
-----------
1. Load Manhattan zones from the official TLC taxi-zone shapefile using the
   'borough' field — no hardcoded LocationID lists.
2. The shapefile is already in EPSG:2263 (NY State Plane, feet), a projected
   CRS suitable for spatial calculations. No reprojection is needed, but the
   CRS is verified and documented.
3. A 10x10 grid is constructed over the Manhattan bounding box (with a small
   buffer so boundary zones fall inside a cell, not on an edge).
4. Each zone is assigned to a cell using its representative_point() — the
   guaranteed-interior point of the polygon.

IMPORTANT: Representative points approximate the geographic center of a taxi
zone. They are NOT actual GPS pickup/drop-off coordinates. Trip positions
within a zone are approximated by the zone's representative point for
simulation purposes.

Row 0 = southernmost strip of Manhattan (Battery Park area).
Row 9 = northernmost strip (Inwood/Washington Heights).
Col 0 = westernmost strip (Hudson River side).
Col 9 = easternmost strip (East River / Harlem River side).

Invalid or unknown zone IDs
---------------------------
zone_to_grid() raises KeyError for any zone ID not in the mapping.
Callers must handle this explicitly. The sentinel (0, 0) is never returned
silently because it would corrupt simulation statistics.

Usage
-----
    from zone_grid import build_zone_grid_map, zone_to_grid, save_zone_grid_map

    zone_map = build_zone_grid_map(shapefile_path)
    row, col = zone_to_grid(161, zone_map)   # Midtown Center -> some cell
    save_zone_grid_map(zone_map, output_path)
    visualize_zone_grid(shapefile_path, zone_map, viz_path)
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import SHAPEFILE, ZONE_GRID_MAP, GRID_VIZ, PROCESSED_DIR

log = logging.getLogger(__name__)

GRID_SIZE = 10
EXPECTED_CRS = "EPSG:2263"
# Small fractional buffer so zones whose rep-point lands exactly on the
# grid boundary are still assigned to a valid cell.
GRID_BUFFER_FRACTION = 1e-6


# ---------------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------------

def build_zone_grid_map(
    shapefile_path: Path = SHAPEFILE,
    grid_size: int = GRID_SIZE,
) -> dict[int, tuple[int, int]]:
    """
    Build a {LocationID: (row, col)} mapping for all Manhattan taxi zones.

    The grid origin (row=0, col=0) is the south-west corner of the Manhattan
    bounding box. Row increases northward; col increases eastward.

    Args:
        shapefile_path: Path to taxi_zones.shp.
        grid_size: Number of cells per side (default 10).

    Returns:
        dict mapping int LocationID -> (int row, int col).

    Raises:
        FileNotFoundError: if shapefile_path does not exist.
        ValueError: if the shapefile has unexpected columns or CRS.
    """
    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Shapefile not found: {shapefile_path}\n"
            f"Place taxi_zones.shp (and .dbf/.prj/.shx) in: {shapefile_path.parent}"
        )

    gdf = gpd.read_file(str(shapefile_path))

    # Validate expected columns
    for col in ("borough", "LocationID", "geometry"):
        if col not in gdf.columns:
            raise ValueError(f"Shapefile missing expected column: '{col}'")

    # Verify CRS — EPSG:2263 is NY State Plane (feet), already projected
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS defined.")
    if str(gdf.crs.to_epsg()) != "2263":
        log.warning(
            "Shapefile CRS is %s, expected EPSG:2263. Reprojecting.",
            gdf.crs
        )
        gdf = gdf.to_crs("EPSG:2263")

    manhattan = gdf[gdf["borough"] == "Manhattan"].copy()
    if manhattan.empty:
        raise ValueError("No zones with borough='Manhattan' found in shapefile.")

    log.info("Building zone grid map for %d Manhattan zones.", len(manhattan))

    # Bounding box of all Manhattan zones
    minx, miny, maxx, maxy = manhattan.total_bounds
    # Add tiny buffer so boundary representative points land inside a cell
    buf_x = (maxx - minx) * GRID_BUFFER_FRACTION
    buf_y = (maxy - miny) * GRID_BUFFER_FRACTION
    minx -= buf_x
    miny -= buf_y
    maxx += buf_x
    maxy += buf_y

    cell_w = (maxx - minx) / grid_size
    cell_h = (maxy - miny) / grid_size

    zone_map: dict[int, tuple[int, int]] = {}

    for _, zone in manhattan.iterrows():
        loc_id = int(zone["LocationID"])
        rep = zone.geometry.representative_point()

        col = int((rep.x - minx) / cell_w)
        row = int((rep.y - miny) / cell_h)

        # Clamp to [0, grid_size-1] — handles any floating-point edge cases
        col = max(0, min(grid_size - 1, col))
        row = max(0, min(grid_size - 1, row))

        zone_map[loc_id] = (row, col)

    log.info("Zone grid map built: %d zones mapped.", len(zone_map))
    return zone_map


def zone_to_grid(zone_id: int, zone_map: dict) -> tuple[int, int]:
    """
    Return (row, col) for a zone ID.

    Raises:
        KeyError: if zone_id is not in zone_map. Never returns a silent
                  fallback — callers must handle unknown zones explicitly.
    """
    key = int(zone_id)
    if key not in zone_map:
        raise KeyError(
            f"Zone ID {zone_id} is not in the zone grid map. "
            f"Valid IDs: {sorted(zone_map.keys())}"
        )
    return zone_map[key]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_zone_grid_map(
    zone_map: dict[int, tuple[int, int]],
    output_path: Path = ZONE_GRID_MAP,
) -> None:
    """Save zone_map as JSON: {str(LocationID): [row, col]}."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): list(v) for k, v in zone_map.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
    log.info("Zone grid map saved to %s", output_path)


def load_zone_grid_map(
    input_path: Path = ZONE_GRID_MAP,
) -> dict[int, tuple[int, int]]:
    """Load zone_map from JSON. Returns {int LocationID: (row, col)}."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Zone grid map not found: {input_path}\n"
            f"Run: python ride-share-marl/pipeline/zone_grid.py"
        )
    with open(input_path) as f:
        raw = json.load(f)
    return {int(k): tuple(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_zone_grid(
    shapefile_path: Path = SHAPEFILE,
    zone_map: dict | None = None,
    output_path: Path = GRID_VIZ,
    grid_size: int = GRID_SIZE,
) -> None:
    """
    Save a PNG showing Manhattan taxi zones coloured by grid cell assignment,
    with the 10x10 grid overlaid and zone labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap

    if zone_map is None:
        zone_map = build_zone_grid_map(shapefile_path, grid_size)

    gdf = gpd.read_file(str(shapefile_path))
    manhattan = gdf[gdf["borough"] == "Manhattan"].copy()
    manhattan["loc_id"] = manhattan["LocationID"].astype(int)
    manhattan["row"] = manhattan["loc_id"].map(lambda z: zone_map.get(z, (-1, -1))[0])
    manhattan["col"] = manhattan["loc_id"].map(lambda z: zone_map.get(z, (-1, -1))[1])
    manhattan["cell"] = manhattan["row"] * grid_size + manhattan["col"]

    minx, miny, maxx, maxy = manhattan.total_bounds
    cell_w = (maxx - minx) / grid_size
    cell_h = (maxy - miny) / grid_size

    fig, ax = plt.subplots(figsize=(8, 14))

    cmap = get_cmap("tab20", grid_size * grid_size)
    norm = Normalize(vmin=0, vmax=grid_size * grid_size - 1)
    manhattan.plot(
        ax=ax,
        column="cell",
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.75,
    )

    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(miny + i * cell_h, color="navy", linewidth=0.6, alpha=0.5)
        ax.axvline(minx + i * cell_w, color="navy", linewidth=0.6, alpha=0.5)

    # Label grid cells (row, col)
    for r in range(grid_size):
        for c in range(grid_size):
            cx = minx + (c + 0.5) * cell_w
            cy = miny + (r + 0.5) * cell_h
            ax.text(cx, cy, f"({r},{c})", fontsize=4, ha="center", va="center",
                    color="navy", alpha=0.6)

    # Label each zone with its LocationID and representative point
    for _, zone in manhattan.iterrows():
        rep = zone.geometry.representative_point()
        ax.annotate(
            str(int(zone["LocationID"])),
            xy=(rep.x, rep.y),
            fontsize=4.5,
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
        )

    ax.set_title(
        "Manhattan Taxi Zones → 10×10 Simulation Grid\n"
        "(cell labels = (row,col); zone labels = LocationID;\n"
        " rep. points approximate zone centers, not GPS coordinates)",
        fontsize=9,
    )
    ax.set_axis_off()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Grid visualization saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Build and save the geographic zone-to-grid mapping."
    )
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip generating the visualization PNG.")
    args = parser.parse_args()

    zone_map = build_zone_grid_map()
    save_zone_grid_map(zone_map)

    print(f"\nZone grid map: {len(zone_map)} Manhattan zones mapped to 10x10 grid")
    print(f"Saved to: {ZONE_GRID_MAP}")

    # Print a sample
    print("\nSample mappings (LocationID -> (row, col) -> zone name):")
    gdf = gpd.read_file(str(SHAPEFILE))
    name_map = dict(zip(gdf["LocationID"].astype(int), gdf["zone"]))
    sample_ids = sorted(zone_map.keys())[:10]
    for loc_id in sample_ids:
        r, c = zone_map[loc_id]
        print(f"  Zone {loc_id:>3} ({name_map.get(loc_id, '?'):<30}) -> row={r}, col={c}")

    # Show grid occupancy
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c in zone_map.values():
        grid[r, c] += 1
    print(f"\nGrid cell occupancy (zones per cell):")
    print(grid)
    print(f"Cells with >=1 zone: {(grid > 0).sum()} / {GRID_SIZE * GRID_SIZE}")
    print(f"Max zones in one cell: {grid.max()}")

    if not args.no_viz:
        visualize_zone_grid(zone_map=zone_map)
        print(f"\nVisualization saved to: {GRID_VIZ}")
