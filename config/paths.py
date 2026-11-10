from pathlib import Path

# Project root is the parent of this file's directory
ROOT = Path(__file__).resolve().parent.parent

# Raw data
RAW_DIR = ROOT / "data" / "raw"
PARQUET_RAW = RAW_DIR / "yellow_tripdata_2023-01.parquet"
SHAPEFILE = RAW_DIR / "taxi_zones" / "taxi_zones.shp"

# Processed data
PROCESSED_DIR = ROOT / "data" / "processed"
MANHATTAN_TRIPS = PROCESSED_DIR / "manhattan_trips.parquet"
SPAWN_WEIGHTS = PROCESSED_DIR / "spawn_weights.npy"

# Results
RESULTS_DIR = ROOT / "results"
