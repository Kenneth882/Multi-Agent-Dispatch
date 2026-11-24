"""Validate TLC passenger demand and prepare January 18 simulation scenarios."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from pyarrow.parquet import ParquetFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ride-share-marl" / "pipeline"))

from config.paths import HVFHV_RAW, RAW_DIR, RESULTS_DIR, SHAPEFILE, ZONE_GRID_MAP
from tlc_pipeline import MAX_TRIP_SECONDS, MIN_TRIP_SECONDS, PROVIDER_MAP
from zone_grid import GRID_SIZE, build_zone_grid_map

LOOKUP = RAW_DIR / "taxi_zone_lookup.csv"
HVFHV_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-01.parquet"
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
SHAPEFILE_URL = "https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip"
DAY = pd.Timestamp("2023-01-18")
WINDOWS = {"midday": (11, 14), "peak": (17, 20)}
TIMES = ["request_datetime", "pickup_datetime", "dropoff_datetime"]
LOCATIONS = ["PULocationID", "DOLocationID"]
LICENSE = "hvfhs_license_num"
REQUIRED = [LICENSE, *TIMES, *LOCATIONS]
STAGES = [
    "original_monthly_records", "required_values_present", "parseable_timestamps",
    "january_18_requests", "uber_or_lyft", "manhattan_pickup", "manhattan_dropoff",
    "valid_timestamp_order", "valid_trip_duration", "unique_records",
    "valid_pickup_and_dropoff_grid",
]


def load_manhattan_lookup(path):
    lookup = pd.read_csv(path, usecols=["LocationID", "Borough"])
    ids = lookup.loc[lookup["Borough"].eq("Manhattan"), "LocationID"]
    if ids.empty or ids.isna().any():
        raise ValueError("The taxi zone lookup must contain Manhattan LocationIDs.")
    return set(ids.astype(int))


def valid_cell(cell):
    return (
        isinstance(cell, (tuple, list)) and len(cell) == 2
        and all(type(value) is int and 0 <= value < GRID_SIZE for value in cell)
    )


def validate_requests(path, manhattan, zone_map, batch_size=65_536):
    """Scan bounded batches and retain only valid-day candidates for exact deduplication."""
    counts = Counter(dict.fromkeys(STAGES, 0))
    source = ParquetFile(path)
    missing = set(REQUIRED) - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing HVFHV columns: {sorted(missing)}")
    candidates = []
    for batch in source.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        counts["original_monthly_records"] += len(frame)
        frame = frame.dropna(subset=REQUIRED)
        counts["required_values_present"] += len(frame)
        times = frame[TIMES].apply(pd.to_datetime, errors="coerce", format="mixed")
        mask = times.notna().all(axis=1)
        counts["parseable_timestamps"] += mask.sum()
        mask &= times["request_datetime"].between(
            DAY, DAY + pd.Timedelta(days=1), inclusive="left"
        )
        counts["january_18_requests"] += mask.sum()
        mask &= frame[LICENSE].isin(PROVIDER_MAP)
        counts["uber_or_lyft"] += mask.sum()
        for location, stage in zip(LOCATIONS, ("manhattan_pickup", "manhattan_dropoff")):
            mask &= frame[location].isin(manhattan)
            counts[stage] += mask.sum()
        mask &= times["request_datetime"].le(times["pickup_datetime"])
        counts["valid_timestamp_order"] += mask.sum()
        duration = (times["dropoff_datetime"] - times["pickup_datetime"]).dt.total_seconds()
        mask &= duration.between(MIN_TRIP_SECONDS, MAX_TRIP_SECONDS)
        counts["valid_trip_duration"] += mask.sum()
        if mask.any():
            candidates.append(frame.loc[mask])

    if candidates:
        requests = pd.concat(candidates, ignore_index=True).drop_duplicates()[REQUIRED].copy()
    else:
        requests = pd.DataFrame(columns=REQUIRED)
    counts["unique_records"] = len(requests)
    mapping_counts = Counter()
    keep = pd.Series(True, index=requests.index, dtype=bool)
    usable = {zone: cell for zone, cell in zone_map.items() if valid_cell(cell)}
    for location, endpoint in zip(LOCATIONS, ("pickup", "dropoff")):
        missing = ~requests[location].isin(zone_map)
        invalid = ~missing & ~requests[location].isin(usable)
        mapping_counts[f"{endpoint}_missing_mapping"] = int(missing.sum())
        mapping_counts[f"{endpoint}_invalid_mapping"] = int(invalid.sum())
        keep &= ~(missing | invalid)
    requests = requests.loc[keep].copy()
    counts["valid_pickup_and_dropoff_grid"] = len(requests)
    mapping_counts["requests_rejected_for_mapping"] = int((~keep).sum())
    requests["provider"] = requests[LICENSE].map(PROVIDER_MAP)
    requests["request_datetime"] = pd.to_datetime(requests["request_datetime"], format="mixed")
    for location, endpoint in zip(LOCATIONS, ("pickup", "dropoff")):
        cells = requests[location].map(usable)
        for axis, name in enumerate(("row", "col")):
            requests[f"{endpoint}_{name}"] = cells.map(lambda cell: cell[axis]).astype(int)
    return requests, counts, mapping_counts


def prepare_scenario(requests, window, seed, max_requests=None):
    start, end = (DAY + pd.Timedelta(hours=hour) for hour in window)
    timestamps = requests["request_datetime"]
    scenario = requests.loc[timestamps.between(start, end, inclusive="left")].copy()
    if max_requests is not None and len(scenario) > max_requests:
        scenario = scenario.sample(n=max_requests, random_state=seed)
    scenario = scenario.sort_values("request_datetime", kind="stable")
    scenario["request_time_seconds"] = (
        (scenario["request_datetime"] - start).dt.total_seconds()
    )
    return scenario.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trips", type=Path, default=HVFHV_RAW)
    parser.add_argument("--lookup", type=Path, default=LOOKUP)
    parser.add_argument("--report", type=Path, default=RESULTS_DIR / "dataset_validation_2023-01-18.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-requests", type=int, help="Optional sample size per window.")
    parser.add_argument("--scenario-dir", type=Path, help="Write midday.parquet and peak.parquet here.")
    args = parser.parse_args()
    if args.max_requests is not None and args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")

    report = [
        ("method", "request_date", str(DAY.date()), "NYC local request time"),
        ("method", "duration_seconds", f"{MIN_TRIP_SECONDS} to {MAX_TRIP_SECONDS}", "Inclusive pickup-to-dropoff duration"),
        ("method", "deduplication", "exact source rows", "Includes optional source columns"),
    ]
    missing = []
    for path, url in ((args.trips, HVFHV_URL), (args.lookup, LOOKUP_URL), (SHAPEFILE, SHAPEFILE_URL)):
        if not path.is_file():
            missing.append(path)
            try:
                display_path = path.relative_to(ROOT).as_posix()
            except ValueError:
                display_path = str(path)
            report.append(("missing_input", display_path, "missing", url))

    if SHAPEFILE.is_file():
        zone_map = build_zone_grid_map(SHAPEFILE)
        report.extend([
            ("geography", "zones_mapped", len(zone_map), "TLC polygon representative points in EPSG:2263"),
            ("geography", "invalid_cells", sum(not valid_cell(c) for c in zone_map.values()), "Both axes must be integers in [0, 9]"),
            ("geography", "occupied_cells", len(set(zone_map.values())), "10x10 grid"),
        ])
        if ZONE_GRID_MAP.is_file():
            cached = json.loads(ZONE_GRID_MAP.read_text(encoding="utf-8"))
            expected = {str(zone): list(cell) for zone, cell in zone_map.items()}
            report.append(("geography", "saved_map_matches_rebuild", cached == expected, "data/processed/zone_grid_map.json"))
    if args.lookup.is_file() and SHAPEFILE.is_file():
        manhattan = load_manhattan_lookup(args.lookup)
        for name, zones in (
            ("lookup_zones_missing_from_geometry", manhattan - zone_map.keys()),
            ("geometry_zones_missing_from_lookup", zone_map.keys() - manhattan),
        ):
            report.append(("geography", name, len(zones), ", ".join(map(str, sorted(zones)))))
    if missing:
        report.append(("status", "trip_validation", "blocked", "Trip metrics unavailable until required input files are present"))
    else:
        requests, stages, mappings = validate_requests(args.trips, manhattan, zone_map)
        report.extend(("filter", name, count, "Records remaining") for name, count in stages.items())
        report.extend(("mapping", name, count, "") for name, count in mappings.items())
        report.append(("result", "final_valid_passenger_requests", len(requests), "All January 18 hours"))
        for provider in PROVIDER_MAP.values():
            report.append(("provider", provider, int(requests["provider"].eq(provider).sum()), "Final valid requests"))
        hours = requests["request_datetime"].dt.hour.value_counts()
        for hour in range(24):
            report.append(("hourly_demand", f"{hour:02d}:00", int(hours.get(hour, 0)), "NYC local time"))
        for label, window in WINDOWS.items():
            full = prepare_scenario(requests, window, args.seed)
            sampling = args.max_requests is not None and len(full) > args.max_requests
            sampled = prepare_scenario(requests, window, args.seed, args.max_requests) if sampling else full
            reproducible = "not sampled"
            if sampling:
                repeat = prepare_scenario(requests, window, args.seed, args.max_requests)
                reproducible = sampled.equals(repeat)
            if args.scenario_dir is not None:
                args.scenario_dir.mkdir(parents=True, exist_ok=True)
                sampled.to_parquet(args.scenario_dir / f"{label}.parquet", index=False)
            report.extend([
                ("scenario", f"{label}_available_requests", len(full), f"[{window[0]}:00, {window[1]}:00)"),
                ("scenario", f"{label}_selected_requests", len(sampled), f"seed={args.seed}"),
                ("scenario", f"{label}_fixed_seed_repeat_matches", reproducible, f"seed={args.seed}"),
            ])
        report.append(("status", "trip_validation", "complete", ""))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "metric", "value", "detail"])
        writer.writerows(report)
    print(f"Validation report saved to {args.report}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
