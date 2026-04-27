import pandas as pd
import numpy as np
import os

GRID_SIZE = 10

def zone_to_grid(zone_id, zone_grid_map):
    """Map a TLC zone ID to a grid cell using precomputed mapping."""
    return zone_grid_map.get(int(zone_id), (0, 0))

def build_zone_grid_map(grid_size=GRID_SIZE):
    """
    Build a mapping from TLC zone ID to grid cell.
    Manhattan has 69 zones, we distribute them across the grid
    based on their zone ID range.
    """
    # Manhattan zone IDs range roughly from 1 to 263
    # We map them to a 10x10 grid based on position
    zone_map = {}
    manhattan_zone_ids = list(range(1, 264))
    
    for i, zone_id in enumerate(manhattan_zone_ids):
        row = (i // grid_size) % grid_size
        col = i % grid_size
        zone_map[zone_id] = (row, col)
    
    return zone_map


def get_demand_snapshot(hour, day_of_week, n_samples=10):
    """Sample n ride requests from a given hour and day."""
    df = pd.read_parquet("data/processed/manhattan_trips.parquet")
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    subset = df[
        (df["tpep_pickup_datetime"].dt.hour == hour) &
        (df["tpep_pickup_datetime"].dt.dayofweek == day_of_week)
    ]

    if len(subset) == 0:
        print(f"No trips found for hour={hour}, day={day_of_week}")
        return pd.DataFrame()

    sample = subset.sample(n=min(n_samples, len(subset))).copy()
    zone_map = build_zone_grid_map()

    sample["pickup_row"] = sample["PULocationID"].apply(lambda z: zone_to_grid(z, zone_map)[0])
    sample["pickup_col"] = sample["PULocationID"].apply(lambda z: zone_to_grid(z, zone_map)[1])
    sample["dropoff_row"] = sample["DOLocationID"].apply(lambda z: zone_to_grid(z, zone_map)[0])
    sample["dropoff_col"] = sample["DOLocationID"].apply(lambda z: zone_to_grid(z, zone_map)[1])

    return sample[[
        "pickup_row", "pickup_col",
        "dropoff_row", "dropoff_col",
        "trip_distance"
    ]].reset_index(drop=True)


def get_spawn_weights(grid_size=GRID_SIZE, cache_path="data/processed/spawn_weights.npy"):
    """
    Compute probability of passenger spawning in each grid cell
    based on historical pickup frequency from TLC data.
    """
    if os.path.exists(cache_path):
        return np.load(cache_path)

    df = pd.read_parquet("data/processed/manhattan_trips.parquet")
    zone_map = build_zone_grid_map()

    rows = df["PULocationID"].map(lambda z: zone_map.get(int(z), (0, 0))[0]).to_numpy()
    cols = df["PULocationID"].map(lambda z: zone_map.get(int(z), (0, 0))[1]).to_numpy()

    weight_grid = np.zeros((grid_size, grid_size))
    np.add.at(weight_grid, (rows, cols), 1)
    weight_grid = weight_grid / weight_grid.sum()

    np.save(cache_path, weight_grid)
    return weight_grid


def get_hourly_spawn_rate():
    """Returns average number of ride requests per hour."""
    df = pd.read_parquet("data/processed/manhattan_trips.parquet")
    df["hour"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.hour
    hourly_rate = df.groupby("hour").size() / df["hour"].nunique()
    return hourly_rate.to_dict()


if __name__ == "__main__":
    print("=== Demand Snapshot (Monday 9am) ===")
    requests = get_demand_snapshot(hour=9, day_of_week=0, n_samples=10)
    print(requests)

    print("\n=== Spawn Weight Grid ===")
    weights = get_spawn_weights()
    print(f"Grid shape: {weights.shape}")
    print(f"Hottest cell: {np.unravel_index(weights.argmax(), weights.shape)}")
    print(f"Max spawn probability: {weights.max():.4f}")

    print("\n=== Hourly Spawn Rates ===")
    rates = get_hourly_spawn_rate()
    for hour, rate in sorted(rates.items()):
        print(f"  Hour {hour:02d}: {rate:.1f} avg trips")