from pathlib import Path

# Project root is the parent of this file's directory
ROOT = Path(__file__).resolve().parent.parent

# Raw data
RAW_DIR = ROOT / "data" / "raw"
PARQUET_RAW = RAW_DIR / "yellow_tripdata_2023-01.parquet"
HVFHV_RAW = RAW_DIR / "fhvhv_tripdata_2023-01.parquet"
SHAPEFILE = RAW_DIR / "taxi_zones" / "taxi_zones.shp"

# Processed data
PROCESSED_DIR = ROOT / "data" / "processed"
MANHATTAN_TRIPS = PROCESSED_DIR / "manhattan_trips.parquet"       # SP1 yellow-only output
COMBINED_TRIPS = PROCESSED_DIR / "combined_trips.parquet"         # SP2 yellow + HVFHV output
SPAWN_WEIGHTS = PROCESSED_DIR / "spawn_weights.npy"
ZONE_GRID_MAP = PROCESSED_DIR / "zone_grid_map.json"   # LocationID -> (row, col)
GRID_VIZ = PROCESSED_DIR / "zone_grid_visualization.png"

# Results
RESULTS_DIR = ROOT / "results"
