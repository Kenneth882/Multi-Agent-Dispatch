"""
This is one small build for the High Volume veichles meaning uber and lyft. It reads the January 2023 FHV trip data and prepares two scenarios for Wednesday, January 18th, 2023: midday (11:00-14:00) and peak (17:00-20:00). 
The output includes filtered trip data for each scenario and a CSV file with hourly request counts for the entire day.

Reads:
    data/raw/fhvhv_tripdata_2023-01.parquet
    data/raw/taxi_zone_lookup.csv
    This is for the yellow taxis which we had last semester and the uber and lyft additions.
Writes:
    data/processed/scenarios/wed_2023-01-18_midday.parquet   (11:00-14:00)
    data/processed/scenarios/wed_2023-01-18_peak.parquet     (17:00-20:00)
    data/processed/scenarios/wed_2023-01-18_hourly_counts.csv


"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "scenarios"

TRIPS_FILE = RAW_DIR / "fhvhv_tripdata_2023-01.parquet"
ZONES_FILE = RAW_DIR / "taxi_zone_lookup.csv"

# --- Settings ----------------------------------------------------------------
DATE = "2023-01-18"  # Wednesday on January 18,2023
SCENARIOS = {
    "midday": ("11:00", "14:00"),
    "peak": ("17:00", "20:00"),
}
PROVIDERS = {"HV0003": "Uber", "HV0005": "Lyft"}
ISLAND_ZONES = {103, 104, 105}  
MAX_TRIP_SECONDS = 3 * 60 * 60

COLUMNS = [
    "hvfhs_license_num",
    "request_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "congestion_surcharge",
]


def manhattan_zone_ids() -> set[int]:
    zones = pd.read_csv(ZONES_FILE)
    ids = set(zones.loc[zones["Borough"] == "Manhattan", "LocationID"].astype(int))
    return ids - ISLAND_ZONES


def load_day() -> pd.DataFrame:
    """Read only Uber/Lyft rows requested on DATE, without loading the whole month."""
    dataset = ds.dataset(TRIPS_FILE, format="parquet")
    ts_type = dataset.schema.field("request_datetime").type
    start = pa.scalar(datetime.fromisoformat(DATE), type=ts_type)
    end = pa.scalar(datetime.fromisoformat(DATE) + pd.Timedelta(days=1), type=ts_type)

    flt = (
        ds.field("hvfhs_license_num").isin(list(PROVIDERS))
        & (ds.field("request_datetime") >= start)
        & (ds.field("request_datetime") < end)
    )
    return dataset.to_table(columns=COLUMNS, filter=flt).to_pandas()


def clean(df: pd.DataFrame, manhattan: set[int]) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["request_datetime", "pickup_datetime", "dropoff_datetime",
                           "PULocationID", "DOLocationID"])
    df = df[df["PULocationID"].isin(manhattan) & df["DOLocationID"].isin(manhattan)]
    df = df[(df["pickup_datetime"] >= df["request_datetime"])
            & (df["dropoff_datetime"] > df["pickup_datetime"])]
    df = df[(df["trip_miles"] > 0) & (df["trip_time"] > 0)
            & (df["trip_time"] <= MAX_TRIP_SECONDS)]
    df = df[df["base_passenger_fare"] >= 0]
    df = df.drop_duplicates()

    df = df.assign(
        provider=df["hvfhs_license_num"].map(PROVIDERS),
        wait_seconds=(df["pickup_datetime"] - df["request_datetime"]).dt.total_seconds(),
    ).sort_values("request_datetime").reset_index(drop=True)

    print(f"Cleaning: {n0:,} rows -> {len(df):,} Manhattan-only rows")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Fraction of trips to keep per scenario (e.g. 0.1). Default: all.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for f in (TRIPS_FILE, ZONES_FILE):
        if not f.exists():
            raise SystemExit(f"Missing {f.relative_to(PROJECT_ROOT)} - download it from the TLC trip record page.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = clean(load_day(), manhattan_zone_ids())

    # Hourly demand for the whole day: use this to confirm the peak window.
    hourly = df.groupby(df["request_datetime"].dt.hour).size().rename("requests")
    hourly.index.name = "hour"
    hourly.to_csv(OUT_DIR / f"wed_{DATE}_hourly_counts.csv")
    print("\nRequests per hour:")
    print(hourly.to_string())
    print(f"Busiest hour: {hourly.idxmax()}:00\n")

    for name, (t0, t1) in SCENARIOS.items():
        start, end = pd.Timestamp(f"{DATE} {t0}"), pd.Timestamp(f"{DATE} {t1}")
        s = df[(df["request_datetime"] >= start) & (df["request_datetime"] < end)]
        if args.sample_frac < 1.0:
            s = s.sample(frac=args.sample_frac, random_state=args.seed).sort_values("request_datetime")
        s = s.reset_index(drop=True)

        out = OUT_DIR / f"wed_{DATE}_{name}.parquet"
        s.to_parquet(out, index=False)
        print(f"{name:>7} {t0}-{t1}: {len(s):,} trips "
              f"(Uber {(s['provider'] == 'Uber').sum():,}, Lyft {(s['provider'] == 'Lyft').sum():,}), "
              f"median wait {s['wait_seconds'].median():.0f}s -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()