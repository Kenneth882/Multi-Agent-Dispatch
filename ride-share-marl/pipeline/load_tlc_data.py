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

    df.to_parquet("data/processed/manhattan_trips.parquet", index=False)
    print(f"Saved {len(df)} Manhattan trips")
    print(df.head(3))
    return df

if __name__ == "__main__":
    load_and_filter(
        "data/raw/yellow_tripdata_2023-01.parquet",
        "data/raw/taxi_zones/taxi_zones.shp"
    )