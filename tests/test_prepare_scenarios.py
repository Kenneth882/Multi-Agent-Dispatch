import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_scenarios
from prepare_scenarios import (
    DAY, WINDOWS, load_manhattan_lookup, prepare_scenario, validate_requests,
)


def record(hour=11, **changes):
    requested = DAY + pd.Timedelta(hours=hour)
    pickup = requested + pd.Timedelta(minutes=2)
    return {
        "hvfhs_license_num": "HV0003",
        "request_datetime": str(requested),
        "pickup_datetime": str(pickup),
        "dropoff_datetime": str(pickup + pd.Timedelta(minutes=5)),
        "PULocationID": 1,
        "DOLocationID": 2,
        "optional_source_field": "original",
        **changes,
    }


def test_filter_counts_and_exact_duplicates_across_batches(tmp_path):
    lookup = tmp_path / "lookup.csv"
    lookup.write_text("LocationID,Borough\n1,Manhattan\n2,Manhattan\n3,Queens\n4,Manhattan\n5,Manhattan\n")
    parquet = tmp_path / "trips.parquet"
    pd.DataFrame([
        record(),
        record(17, hvfhs_license_num="HV0005"),
        record(hvfhs_license_num="HV0004"),
        record(PULocationID=3),
        record(DOLocationID=3),
        record(DOLocationID=None),
        record(pickup_datetime="malformed"),
        record(request_datetime="2023-01-17 11:00:00"),
        record(pickup_datetime="2023-01-18 10:59:00"),
        record(dropoff_datetime="2023-01-18 11:02:00"),
        record(dropoff_datetime="2023-01-18 23:30:00"),
        record(PULocationID=4),
        record(DOLocationID=5),
        record(),
        record(optional_source_field="distinct source row"),
    ]).to_parquet(parquet, index=False)

    requests, counts, mappings = validate_requests(
        parquet, load_manhattan_lookup(lookup),
        {1: (0, 9), 2: (9, 0), 5: (10, 0)}, batch_size=3,
    )
    assert list(counts.values()) == [15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 3]
    assert mappings == {
        "pickup_missing_mapping": 1, "pickup_invalid_mapping": 0,
        "dropoff_missing_mapping": 0, "dropoff_invalid_mapping": 1,
        "requests_rejected_for_mapping": 2,
    }
    assert requests["provider"].value_counts().to_dict() == {"uber": 2, "lyft": 1}
    assert requests[["pickup_row", "pickup_col", "dropoff_row", "dropoff_col"]].values.tolist() == [
        [0, 9, 9, 0], [0, 9, 9, 0], [0, 9, 9, 0],
    ]


def test_scenario_boundaries_and_actual_seeded_sampling():
    minutes = [659, 660, 661, 662, 839, 840, 1019, 1020, 1022, 1199, 1200]
    requests = pd.DataFrame({
        "request_datetime": DAY + pd.to_timedelta(minutes, unit="min"),
        "id": minutes,
    })
    midday = prepare_scenario(requests, WINDOWS["midday"], seed=42)
    peak = prepare_scenario(requests, WINDOWS["peak"], seed=42)
    assert midday["id"].tolist() == [660, 661, 662, 839]
    assert peak["id"].tolist() == [1020, 1022, 1199]
    assert peak["request_time_seconds"].tolist() == [0, 120, 10740]
    for window in WINDOWS.values():
        sample = prepare_scenario(requests, window, seed=42, max_requests=2)
        repeat = prepare_scenario(requests, window, seed=42, max_requests=2)
        assert len(sample) == 2
        assert sample["id"].is_unique
        pd.testing.assert_frame_equal(sample, repeat)


def test_no_valid_requests_retains_scenario_schema(tmp_path):
    parquet = tmp_path / "trips.parquet"
    pd.DataFrame([record(hvfhs_license_num="HV0004")]).to_parquet(parquet, index=False)
    requests, counts, _ = validate_requests(parquet, {1, 2}, {})
    assert counts["valid_pickup_and_dropoff_grid"] == 0
    scenario = prepare_scenario(requests, WINDOWS["midday"], seed=42)
    assert scenario.empty
    assert {"request_time_seconds", "pickup_row", "dropoff_col"} <= set(scenario)


def test_prepared_offsets_preserve_subsecond_arrivals():
    start = DAY + pd.Timedelta(hours=11)
    requests = pd.DataFrame({"request_datetime": [start + pd.Timedelta(seconds=60.5)]})
    scenario = prepare_scenario(requests, WINDOWS["midday"], seed=42)
    assert scenario["request_time_seconds"].tolist() == [60.5]


def test_report_reconciles_counts_and_checks_sampling(tmp_path, monkeypatch):
    lookup = tmp_path / "lookup.csv"
    lookup.write_text("LocationID,Borough\n1,Manhattan\n2,Manhattan\n")
    parquet = tmp_path / "trips.parquet"
    pd.DataFrame([
        record(11), record(12), record(17, hvfhs_license_num="HV0005"),
        record(18, hvfhs_license_num="HV0005"), record(20),
    ]).to_parquet(parquet, index=False)
    shapefile = tmp_path / "taxi_zones.shp"
    shapefile.touch()
    report = tmp_path / "validation.csv"
    monkeypatch.setattr(prepare_scenarios, "SHAPEFILE", shapefile)
    monkeypatch.setattr(prepare_scenarios, "ZONE_GRID_MAP", tmp_path / "absent.json")
    monkeypatch.setattr(prepare_scenarios, "build_zone_grid_map", lambda _: {1: (0, 1), 2: (4, 5)})
    monkeypatch.setattr(sys, "argv", [
        "prepare_scenarios.py", "--trips", str(parquet), "--lookup", str(lookup),
        "--report", str(report), "--max-requests", "1",
        "--scenario-dir", str(tmp_path / "scenarios"),
    ])
    assert prepare_scenarios.main() == 0
    with report.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {row["metric"]: row["value"] for row in rows}
    hourly = [int(row["value"]) for row in rows if row["category"] == "hourly_demand"]
    assert len(hourly) == 24
    assert sum(hourly) == int(values["final_valid_passenger_requests"]) == 5
    assert int(values["uber"]) + int(values["lyft"]) == 5
    for label in WINDOWS:
        exported = pd.read_parquet(tmp_path / "scenarios" / f"{label}.parquet")
        assert len(exported) == 1
        assert {"request_datetime", "request_time_seconds", "PULocationID", "DOLocationID"} <= set(exported)
        assert values[f"{label}_available_requests"] == "2"
        assert values[f"{label}_selected_requests"] == "1"
        assert values[f"{label}_fixed_seed_repeat_matches"] == "True"
