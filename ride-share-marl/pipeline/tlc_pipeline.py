"""
ride-share-marl/pipeline/tlc_pipeline.py
-----------------------------------------
Unified TLC data pipeline for Senior Project 2.

Supports:
  - Yellow Taxi Trip Records (January 2023)
  - High Volume For-Hire Vehicle (HVFHV) Trip Records (January 2023)
    - HV0003 = Uber, HV0005 = Lyft only

Output schema (combined_trips.parquet):
  trip_id                     str       UUID per row
  provider                    str       "yellow" | "uber" | "lyft"
  request_datetime            datetime  actual for HVFHV; synthetic or NaT for yellow
  request_datetime_is_synthetic bool    True when request_datetime was not observed
  pickup_datetime             datetime
  dropoff_datetime            datetime
  PULocationID                int
  DOLocationID                int
  trip_duration_seconds       float
  trip_distance               float     NaN when not available

Yellow taxi request_datetime policy
------------------------------------
Yellow taxi records have no request timestamp. The SyntheticRequestPolicy enum
controls what is written into request_datetime:

  NONE              request_datetime = NaT, request_datetime_is_synthetic = True
  UNIFORM_0_5MIN    request_datetime = pickup_datetime - Uniform(0, 300 seconds)
                    request_datetime_is_synthetic = True

The field request_datetime_is_synthetic is ALWAYS True for yellow taxi rows
regardless of policy. Never treat synthetic times as observed data.

CLI usage:
  python ride-share-marl/pipeline/tlc_pipeline.py [--policy none|uniform_0_5min]
"""

import sys
import uuid
import argparse
import logging
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import (
    PARQUET_RAW, HVFHV_RAW, SHAPEFILE,
    PROCESSED_DIR, COMBINED_TRIPS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HVFHV_UBER = "HV0003"
HVFHV_LYFT = "HV0005"
PROVIDER_MAP = {HVFHV_UBER: "uber", HVFHV_LYFT: "lyft"}

MAX_TRIP_SECONDS = 10 * 3600   # 10 hours — trips longer than this are invalid
MIN_TRIP_SECONDS = 1            # trips with zero or negative duration are invalid

YELLOW_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
]

HVFHV_COLUMNS = [
    "hvfhs_license_num",
    "request_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
]

OUTPUT_COLUMNS = [
    "trip_id",
    "provider",
    "request_datetime",
    "request_datetime_is_synthetic",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_duration_seconds",
    "trip_distance",
]


class SyntheticRequestPolicy(Enum):
    NONE = "none"
    UNIFORM_0_5MIN = "uniform_0_5min"


# ---------------------------------------------------------------------------
# Zone loading
# ---------------------------------------------------------------------------

def load_manhattan_zones(shapefile_path: Path) -> set:
    """Return the set of LocationIDs whose borough is Manhattan."""
    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Shapefile not found: {shapefile_path}\n"
            f"Place taxi_zones.shp (and .dbf/.prj/.shx) in: {shapefile_path.parent}"
        )
    zones = gpd.read_file(str(shapefile_path))
    if "borough" not in zones.columns or "LocationID" not in zones.columns:
        raise ValueError("Shapefile missing expected columns: 'borough', 'LocationID'")
    manhattan_ids = set(zones.loc[zones["borough"] == "Manhattan", "LocationID"].tolist())
    log.info("Loaded %d Manhattan zone IDs from shapefile.", len(manhattan_ids))
    return manhattan_ids


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list, source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{source}] Missing required columns: {missing}")


def _filter_manhattan(df: pd.DataFrame, zones: set, source: str) -> tuple[pd.DataFrame, int]:
    before = len(df)
    mask = df["PULocationID"].isin(zones) & df["DOLocationID"].isin(zones)
    df = df[mask].copy()
    removed = before - len(df)
    log.info("[%s] Manhattan filter removed %d rows (%d remain).", source, removed, len(df))
    return df, removed


def _drop_null_locations(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna(subset=["PULocationID", "DOLocationID"])
    removed = before - len(df)
    if removed:
        log.info("[%s] Dropped %d rows with null location IDs.", source, removed)
    return df, removed


def _validate_timestamps(df: pd.DataFrame, pickup_col: str, dropoff_col: str,
                          source: str) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")
    df = df.dropna(subset=[pickup_col, dropoff_col])
    duration = (df[dropoff_col] - df[pickup_col]).dt.total_seconds()
    df = df[(duration >= MIN_TRIP_SECONDS) & (duration <= MAX_TRIP_SECONDS)].copy()
    removed = before - len(df)
    if removed:
        log.info("[%s] Dropped %d rows with invalid timestamps or durations.", source, removed)
    return df, removed


def _assign_trip_ids(n: int) -> list:
    return [str(uuid.uuid4()) for _ in range(n)]


# ---------------------------------------------------------------------------
# Yellow taxi processor
# ---------------------------------------------------------------------------

def process_yellow(
    parquet_path: Path,
    manhattan_zones: set,
    policy: SyntheticRequestPolicy = SyntheticRequestPolicy.NONE,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load and process yellow taxi trips.

    Returns (standardized_df, stats_dict).
    request_datetime_is_synthetic is always True for yellow taxi rows.
    """
    source = "yellow"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Yellow taxi file not found: {parquet_path}\n"
            f"Download from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page\n"
            f"and place at: {parquet_path}"
        )

    log.info("[%s] Reading parquet (column-selective)...", source)
    df = pd.read_parquet(str(parquet_path), columns=YELLOW_COLUMNS, engine="pyarrow")
    log.info("[%s] Raw rows: %d", source, len(df))
    stats = {"raw_rows": len(df), "removed_null_location": 0,
             "removed_timestamp": 0, "removed_non_manhattan": 0}

    _validate_columns(df, YELLOW_COLUMNS, source)

    df, n = _drop_null_locations(df, source)
    stats["removed_null_location"] = n

    df, n = _validate_timestamps(df, "tpep_pickup_datetime", "tpep_dropoff_datetime", source)
    stats["removed_timestamp"] = n

    df, n = _filter_manhattan(df, manhattan_zones, source)
    stats["removed_non_manhattan"] = n

    # Synthetic request_datetime
    if policy == SyntheticRequestPolicy.UNIFORM_0_5MIN:
        if rng is None:
            rng = np.random.default_rng()
        offsets = rng.integers(0, 300, size=len(df))  # seconds, [0, 300)
        request_dt = df["tpep_pickup_datetime"] - pd.to_timedelta(offsets, unit="s")
    else:
        request_dt = pd.NaT

    duration = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()

    out = pd.DataFrame({
        "trip_id": _assign_trip_ids(len(df)),
        "provider": "yellow",
        "request_datetime": request_dt,
        "request_datetime_is_synthetic": True,
        "pickup_datetime": df["tpep_pickup_datetime"].values,
        "dropoff_datetime": df["tpep_dropoff_datetime"].values,
        "PULocationID": df["PULocationID"].values,
        "DOLocationID": df["DOLocationID"].values,
        "trip_duration_seconds": duration.values,
        "trip_distance": df["trip_distance"].values,
    })

    stats["final_rows"] = len(out)
    log.info("[%s] Final rows: %d", source, len(out))
    return out, stats


# ---------------------------------------------------------------------------
# HVFHV processor
# ---------------------------------------------------------------------------

def process_hvfhv(
    parquet_path: Path,
    manhattan_zones: set,
) -> tuple[pd.DataFrame, dict]:
    """
    Load and process HVFHV trips (Uber HV0003 and Lyft HV0005 only).

    Returns (standardized_df, stats_dict).
    request_datetime is the actual observed request time from the TLC record.
    """
    source = "hvfhv"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"HVFHV file not found: {parquet_path}\n"
            f"Download from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page\n"
            f"and place at: {parquet_path}"
        )

    log.info("[%s] Reading parquet (column-selective)...", source)
    df = pd.read_parquet(str(parquet_path), columns=HVFHV_COLUMNS, engine="pyarrow")
    log.info("[%s] Raw rows: %d", source, len(df))
    stats = {"raw_rows": len(df), "removed_provider_filter": 0,
             "removed_null_location": 0, "removed_timestamp": 0,
             "removed_non_manhattan": 0}

    _validate_columns(df, HVFHV_COLUMNS, source)

    # Provider filter: keep only Uber and Lyft
    before = len(df)
    df = df[df["hvfhs_license_num"].isin(PROVIDER_MAP)].copy()
    stats["removed_provider_filter"] = before - len(df)
    log.info("[%s] Provider filter (Uber+Lyft only) removed %d rows (%d remain).",
             source, stats["removed_provider_filter"], len(df))

    df, n = _drop_null_locations(df, source)
    stats["removed_null_location"] = n

    df, n = _validate_timestamps(df, "pickup_datetime", "dropoff_datetime", source)
    stats["removed_timestamp"] = n

    # Validate request_datetime separately — coerce but do not drop rows for it
    df["request_datetime"] = pd.to_datetime(df["request_datetime"], errors="coerce")

    df, n = _filter_manhattan(df, manhattan_zones, source)
    stats["removed_non_manhattan"] = n

    duration = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()

    out = pd.DataFrame({
        "trip_id": _assign_trip_ids(len(df)),
        "provider": df["hvfhs_license_num"].map(PROVIDER_MAP).values,
        "request_datetime": df["request_datetime"].values,
        "request_datetime_is_synthetic": False,
        "pickup_datetime": df["pickup_datetime"].values,
        "dropoff_datetime": df["dropoff_datetime"].values,
        "PULocationID": df["PULocationID"].values,
        "DOLocationID": df["DOLocationID"].values,
        "trip_duration_seconds": duration.values,
        "trip_distance": df["trip_miles"].values,
    })

    stats["final_rows"] = len(out)
    log.info("[%s] Final rows: %d", source, len(out))
    return out, stats


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    yellow_path: Path = PARQUET_RAW,
    hvfhv_path: Path = HVFHV_RAW,
    shapefile_path: Path = SHAPEFILE,
    output_path: Path = COMBINED_TRIPS,
    policy: SyntheticRequestPolicy = SyntheticRequestPolicy.NONE,
    skip_missing: bool = False,
) -> pd.DataFrame:
    """
    Run the full pipeline. Returns the combined DataFrame.

    Args:
        skip_missing: if True, silently skip a provider whose file is absent
                      (useful for running with only one file present).
                      Missing files are always reported.
    """
    manhattan_zones = load_manhattan_zones(shapefile_path)

    frames = []
    all_stats = {}

    # Yellow taxi
    try:
        df_yellow, stats_y = process_yellow(yellow_path, manhattan_zones, policy=policy)
        frames.append(df_yellow)
        all_stats["yellow"] = stats_y
    except FileNotFoundError as e:
        if skip_missing:
            log.warning(str(e))
        else:
            raise

    # HVFHV
    try:
        df_hvfhv, stats_h = process_hvfhv(hvfhv_path, manhattan_zones)
        frames.append(df_hvfhv)
        all_stats["hvfhv"] = stats_h
    except FileNotFoundError as e:
        if skip_missing:
            log.warning(str(e))
        else:
            raise

    if not frames:
        raise RuntimeError("No data loaded — both input files are missing.")

    combined = pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(str(output_path), index=False, engine="pyarrow")

    _print_summary(combined, all_stats, output_path)
    return combined


# ---------------------------------------------------------------------------
# CLI summary
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame, stats: dict, output_path: Path) -> None:
    print("\n" + "=" * 60)
    print("TLC Pipeline Summary")
    print("=" * 60)

    provider_counts = df["provider"].value_counts().to_dict()
    print(f"\nProvider counts in output:")
    for provider, count in sorted(provider_counts.items()):
        print(f"  {provider:<10} {count:>10,}")
    print(f"  {'TOTAL':<10} {len(df):>10,}")

    for source, s in stats.items():
        print(f"\n[{source}] Row accounting:")
        print(f"  Raw rows read:            {s.get('raw_rows', 0):>10,}")
        if "removed_provider_filter" in s:
            print(f"  Removed (provider filter):{s['removed_provider_filter']:>10,}")
        print(f"  Removed (null locations): {s.get('removed_null_location', 0):>10,}")
        print(f"  Removed (bad timestamps): {s.get('removed_timestamp', 0):>10,}")
        print(f"  Removed (non-Manhattan):  {s.get('removed_non_manhattan', 0):>10,}")
        print(f"  Final rows:               {s.get('final_rows', 0):>10,}")

    print(f"\nOutput saved to: {output_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TLC pipeline: process yellow taxi and HVFHV trip records."
    )
    parser.add_argument(
        "--policy",
        choices=[p.value for p in SyntheticRequestPolicy],
        default=SyntheticRequestPolicy.NONE.value,
        help=(
            "Synthetic request-time policy for yellow taxi rows. "
            "'none' = NaT (default). "
            "'uniform_0_5min' = pickup_datetime minus Uniform(0, 300s). "
            "Always marked request_datetime_is_synthetic=True."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip providers whose raw files are absent instead of raising an error.",
    )
    args = parser.parse_args()

    run_pipeline(
        policy=SyntheticRequestPolicy(args.policy),
        skip_missing=args.skip_missing,
    )
