import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import PARQUET_RAW, SHAPEFILE, MANHATTAN_TRIPS, PROCESSED_DIR

import pandas as pd
import geopandas as gpd

def load_and_filter(parquet_path, shapefile_path):
    
    zones = gpd.read_file(shapefile_path)
    manhattan_zones = zones[zones["borough"] == "Manhattan"]["LocationID"].tolist()

    df = pd.read_parquet(parquet_path)

    # Filter to Manhattan only
    df = df[
        df["PULocationID"].isin(manhattan_zones) &
        df["DOLocationID"].isin(manhattan_zones)
    ]

    df = df[[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "fare_amount"
    ]].dropna()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(MANHATTAN_TRIPS), index=False)
    print(f"Saved {len(df)} Manhattan trips")
    print(df.head(3))
    return df

if __name__ == "__main__":
    load_and_filter(str(PARQUET_RAW), str(SHAPEFILE))