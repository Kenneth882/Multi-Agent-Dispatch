"""
ride-share-marl/pipeline/demand_generator.py
---------------------------------------------
Derives passenger spawn weights and demand snapshots from processed TLC data.

SP2 change: zone-to-grid mapping is now geographically correct.
  - Uses zone_grid.build_zone_grid_map() which assigns each Manhattan taxi
    zone to a 10x10 grid cell based on its representative_point() in the
    EPSG:2263 projected CRS.
  - Invalid or unknown zone IDs raise KeyError — never silently mapped to (0,0).
  - spawn_weights.npy is invalidated if zone_grid_map.json is newer, forcing
    regeneration with the correct geographic mapping.

SP1 compatibility: get_spawn_weights(), get_demand_snapshot(), and
get_hourly_spawn_rate() preserve their original signatures.
build_zone_grid_map() is kept as a thin wrapper for backward compatibility
but now delegates to zone_grid.build_zone_grid_map().
"""

import os
import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import MANHATTAN_TRIPS, SPAWN_WEIGHTS, ZONE_GRID_MAP, SHAPEFILE

from zone_grid import (
    build_zone_grid_map as _geo_build_zone_grid_map,
    load_zone_grid_map,
    zone_to_grid,
    save_zone_grid_map,
    GRID_SIZE,
)

log = logging.getLogger(__name__)

DATA_PATH = str(MANHATTAN_TRIPS)
CACHE_PATH = str(SPAWN_WEIGHTS)


# ---------------------------------------------------------------------------
# SP1 compatibility shim
# ---------------------------------------------------------------------------

def build_zone_grid_map(grid_size: int = GRID_SIZE) -> dict:
    """
    SP1 compatibility wrapper.

    Returns the geographic zone-to-grid mapping (delegates to zone_grid.py).
    The grid_size argument is accepted for API compatibility but must equal
    GRID_SIZE (10); other values raise ValueError.
    """
    if grid_size != GRID_SIZE:
        raise ValueError(
            f"grid_size must be {GRID_SIZE}; got {grid_size}. "
            "The geographic mapping is fixed to the 10x10 Manhattan grid."
        )
    if ZONE_GRID_MAP.exists():
        return load_zone_grid_map(ZONE_GRID_MAP)
    zone_map = _geo_build_zone_grid_map(SHAPEFILE, GRID_SIZE)
    save_zone_grid_map(zone_map, ZONE_GRID_MAP)
    return zone_map


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

def _spawn_weights_are_stale() -> bool:
    """
    Return True if spawn_weights.npy should be regenerated.

    Stale conditions:
      - spawn_weights.npy does not exist
      - zone_grid_map.json is newer than spawn_weights.npy (mapping changed)
    """
    cache = Path(CACHE_PATH)
    if not cache.exists():
        return True
    if ZONE_GRID_MAP.exists() and ZONE_GRID_MAP.stat().st_mtime > cache.stat().st_mtime:
        log.info(
            "spawn_weights.npy is older than zone_grid_map.json — regenerating."
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Public API (SP1-compatible signatures)
# ---------------------------------------------------------------------------

def get_spawn_weights(
    grid_size: int = GRID_SIZE,
    cache_path: str = CACHE_PATH,
) -> np.ndarray:
    """
    Return a (10, 10) array of passenger spawn probabilities.

    Probabilities are derived from pickup frequency per grid cell across all
    Manhattan trips in manhattan_trips.parquet. The array sums to 1.0.

    Trips with zone IDs not in the geographic mapping are skipped with a
    warning rather than silently mapped to (0, 0).

    The cache is invalidated if zone_grid_map.json is newer than the cached
    .npy file, ensuring the correct geographic mapping is always used.
    """
    if not _spawn_weights_are_stale() and cache_path == CACHE_PATH:
        weights = np.load(cache_path)
        assert weights.shape == (grid_size, grid_size), (
            f"Cached spawn_weights shape {weights.shape} != ({grid_size},{grid_size}). "
            "Delete data/processed/spawn_weights.npy and re-run."
        )
        return weights

    zone_map = build_zone_grid_map(grid_size)
    df = pd.read_parquet(DATA_PATH, columns=["PULocationID"])

    weight_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    skipped = 0

    for zone_id in df["PULocationID"]:
        try:
            r, c = zone_to_grid(int(zone_id), zone_map)
            weight_grid[r, c] += 1
        except KeyError:
            skipped += 1

    if skipped > 0:
        log.warning(
            "Skipped %d trips with zone IDs not in the Manhattan mapping "
            "(%.2f%% of total).",
            skipped, 100 * skipped / len(df),
        )

    total = weight_grid.sum()
    if total == 0:
        raise RuntimeError(
            "No valid trips found for spawn weight calculation. "
            "Check that manhattan_trips.parquet contains Manhattan zone IDs."
        )

    weight_grid /= total

    assert weight_grid.shape == (grid_size, grid_size)
    assert abs(weight_grid.sum() - 1.0) < 1e-9, "Spawn weights do not sum to 1."
    assert (weight_grid >= 0).all(), "Spawn weights contain negative values."

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, weight_grid)
    log.info("Spawn weights saved to %s", cache_path)
    return weight_grid


def get_demand_snapshot(
    hour: int,
    day_of_week: int,
    n_samples: int = 10,
) -> pd.DataFrame:
    """
    Return a sample of trip requests for a given hour and day of week.

    Trips with zone IDs not in the geographic mapping are excluded.
    Returns a DataFrame with columns:
        pickup_row, pickup_col, dropoff_row, dropoff_col, trip_distance
    """
    df = pd.read_parquet(DATA_PATH)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    subset = df[
        (df["tpep_pickup_datetime"].dt.hour == hour) &
        (df["tpep_pickup_datetime"].dt.dayofweek == day_of_week)
    ]

    if len(subset) == 0:
        log.warning("No trips found for hour=%d, day=%d", hour, day_of_week)
        return pd.DataFrame()

    zone_map = build_zone_grid_map()
    sample = subset.sample(n=min(n_samples * 5, len(subset))).copy()

    rows = []
    for _, trip in sample.iterrows():
        try:
            pu_r, pu_c = zone_to_grid(int(trip["PULocationID"]), zone_map)
            do_r, do_c = zone_to_grid(int(trip["DOLocationID"]), zone_map)
        except KeyError:
            continue
        rows.append({
            "pickup_row": pu_r,
            "pickup_col": pu_c,
            "dropoff_row": do_r,
            "dropoff_col": do_c,
            "trip_distance": trip.get("trip_distance", np.nan),
        })
        if len(rows) >= n_samples:
            break

    if not rows:
        log.warning("No valid trips after zone filtering for hour=%d, day=%d", hour, day_of_week)
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


def get_hourly_spawn_rate() -> dict:
    """
    Return average trip count per hour across all days.
    Preserves SP1 signature.
    """
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.hour
    hourly_rate = df.groupby("hour").size() / df["hour"].nunique()
    return hourly_rate.to_dict()


# ---------------------------------------------------------------------------
# CLI (SP1 __main__ preserved)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== Demand Snapshot (Monday 9am) ===")
    requests = get_demand_snapshot(hour=9, day_of_week=0, n_samples=10)
    print(requests)

    print("\n=== Spawn Weight Grid ===")
    weights = get_spawn_weights()
    print(f"Grid shape: {weights.shape}")
    print(f"Sum: {weights.sum():.6f}")
    print(f"Hottest cell: {np.unravel_index(weights.argmax(), weights.shape)}")
    print(f"Max spawn probability: {weights.max():.4f}")
    print(f"Non-zero cells: {(weights > 0).sum()} / {GRID_SIZE * GRID_SIZE}")

    print("\n=== Hourly Spawn Rates ===")
    rates = get_hourly_spawn_rate()
    for hour, rate in sorted(rates.items()):
        print(f"  Hour {hour:02d}: {rate:.1f} avg trips")
