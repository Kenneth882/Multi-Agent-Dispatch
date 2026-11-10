"""
tests/test_tlc_pipeline.py
---------------------------
Unit tests for ride-share-marl/pipeline/tlc_pipeline.py.
All tests use synthetic in-memory DataFrames; no real TLC files are required.
Run from project root: pytest tests/test_tlc_pipeline.py -v
"""
import sys
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ride-share-marl" / "pipeline"))

import tlc_pipeline as _tlc_mod
from tlc_pipeline import (
    _filter_manhattan,
    _drop_null_locations,
    _validate_timestamps,
    process_yellow,
    process_hvfhv,
    load_manhattan_zones,
    SyntheticRequestPolicy,
    OUTPUT_COLUMNS,
    HVFHV_UBER,
    HVFHV_LYFT,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MANHATTAN_ZONES = {1, 2, 3, 4, 5}
NON_MANHATTAN = {99, 100}

T0 = pd.Timestamp("2023-01-15 09:00:00")
T1 = pd.Timestamp("2023-01-15 09:20:00")
T_REQUEST = pd.Timestamp("2023-01-15 08:58:00")


def _yellow_df(n=3, pu=1, do=2, pickup=T0, dropoff=T1, distance=2.5):
    """Minimal valid yellow taxi DataFrame."""
    return pd.DataFrame({
        "tpep_pickup_datetime": [pickup] * n,
        "tpep_dropoff_datetime": [dropoff] * n,
        "PULocationID": [pu] * n,
        "DOLocationID": [do] * n,
        "trip_distance": [distance] * n,
    })


def _hvfhv_df(n=3, license_num=HVFHV_UBER, pu=1, do=2,
              request=T_REQUEST, pickup=T0, dropoff=T1, miles=2.5):
    """Minimal valid HVFHV DataFrame."""
    return pd.DataFrame({
        "hvfhs_license_num": [license_num] * n,
        "request_datetime": [request] * n,
        "pickup_datetime": [pickup] * n,
        "dropoff_datetime": [dropoff] * n,
        "PULocationID": [pu] * n,
        "DOLocationID": [do] * n,
        "trip_miles": [miles] * n,
    })


def _run_yellow(df, policy=SyntheticRequestPolicy.NONE, rng=None):
    """Run process_yellow with filesystem mocked out."""
    fake_path = Path("/fake/yellow.parquet")
    with patch.object(_tlc_mod.pd, "read_parquet", return_value=df), \
         patch.object(_tlc_mod.Path, "exists", return_value=True):
        return process_yellow(fake_path, MANHATTAN_ZONES, policy=policy, rng=rng)


def _run_hvfhv(df):
    """Run process_hvfhv with filesystem mocked out."""
    fake_path = Path("/fake/hvfhv.parquet")
    with patch.object(_tlc_mod.pd, "read_parquet", return_value=df), \
         patch.object(_tlc_mod.Path, "exists", return_value=True):
        return process_hvfhv(fake_path, MANHATTAN_ZONES)


# ---------------------------------------------------------------------------
# _filter_manhattan
# ---------------------------------------------------------------------------

class TestFilterManhattan:
    def test_keeps_both_in_manhattan(self):
        df = pd.DataFrame({"PULocationID": [1, 2], "DOLocationID": [2, 3]})
        out, removed = _filter_manhattan(df, MANHATTAN_ZONES, "test")
        assert len(out) == 2
        assert removed == 0

    def test_removes_pu_outside(self):
        df = pd.DataFrame({"PULocationID": [99, 1], "DOLocationID": [2, 2]})
        out, removed = _filter_manhattan(df, MANHATTAN_ZONES, "test")
        assert len(out) == 1
        assert removed == 1

    def test_removes_do_outside(self):
        df = pd.DataFrame({"PULocationID": [1, 1], "DOLocationID": [2, 99]})
        out, removed = _filter_manhattan(df, MANHATTAN_ZONES, "test")
        assert len(out) == 1
        assert removed == 1

    def test_removes_both_outside(self):
        df = pd.DataFrame({"PULocationID": [99], "DOLocationID": [100]})
        out, removed = _filter_manhattan(df, MANHATTAN_ZONES, "test")
        assert len(out) == 0
        assert removed == 1

    def test_empty_input(self):
        df = pd.DataFrame({"PULocationID": [], "DOLocationID": []})
        out, removed = _filter_manhattan(df, MANHATTAN_ZONES, "test")
        assert len(out) == 0
        assert removed == 0


# ---------------------------------------------------------------------------
# _drop_null_locations
# ---------------------------------------------------------------------------

class TestDropNullLocations:
    def test_no_nulls(self):
        df = pd.DataFrame({"PULocationID": [1, 2], "DOLocationID": [3, 4]})
        out, removed = _drop_null_locations(df, "test")
        assert len(out) == 2
        assert removed == 0

    def test_null_pu(self):
        df = pd.DataFrame({"PULocationID": [None, 2], "DOLocationID": [3, 4]})
        out, removed = _drop_null_locations(df, "test")
        assert len(out) == 1
        assert removed == 1

    def test_null_do(self):
        df = pd.DataFrame({"PULocationID": [1, 2], "DOLocationID": [None, 4]})
        out, removed = _drop_null_locations(df, "test")
        assert len(out) == 1
        assert removed == 1

    def test_both_null(self):
        df = pd.DataFrame({"PULocationID": [None], "DOLocationID": [None]})
        out, removed = _drop_null_locations(df, "test")
        assert len(out) == 0
        assert removed == 1


# ---------------------------------------------------------------------------
# _validate_timestamps
# ---------------------------------------------------------------------------

class TestValidateTimestamps:
    def test_valid_trip(self):
        df = pd.DataFrame({
            "pickup": [T0],
            "dropoff": [T1],
        })
        out, removed = _validate_timestamps(df, "pickup", "dropoff", "test")
        assert len(out) == 1
        assert removed == 0

    def test_zero_duration_removed(self):
        df = pd.DataFrame({"pickup": [T0], "dropoff": [T0]})
        out, removed = _validate_timestamps(df, "pickup", "dropoff", "test")
        assert len(out) == 0
        assert removed == 1

    def test_negative_duration_removed(self):
        df = pd.DataFrame({"pickup": [T1], "dropoff": [T0]})
        out, removed = _validate_timestamps(df, "pickup", "dropoff", "test")
        assert len(out) == 0
        assert removed == 1

    def test_over_10h_removed(self):
        long_dropoff = T0 + pd.Timedelta(hours=11)
        df = pd.DataFrame({"pickup": [T0], "dropoff": [long_dropoff]})
        out, removed = _validate_timestamps(df, "pickup", "dropoff", "test")
        assert len(out) == 0
        assert removed == 1

    def test_unparseable_timestamp_removed(self):
        df = pd.DataFrame({"pickup": ["not-a-date"], "dropoff": [T1]})
        out, removed = _validate_timestamps(df, "pickup", "dropoff", "test")
        assert len(out) == 0
        assert removed == 1


# ---------------------------------------------------------------------------
# process_yellow
# ---------------------------------------------------------------------------

class TestProcessYellow:
    def test_output_schema(self):
        out, _ = _run_yellow(_yellow_df())
        assert list(out.columns) == OUTPUT_COLUMNS

    def test_provider_is_yellow(self):
        out, _ = _run_yellow(_yellow_df())
        assert (out["provider"] == "yellow").all()

    def test_request_datetime_is_synthetic_always_true(self):
        out, _ = _run_yellow(_yellow_df())
        assert out["request_datetime_is_synthetic"].all()

    def test_policy_none_gives_nat(self):
        out, _ = _run_yellow(_yellow_df(), policy=SyntheticRequestPolicy.NONE)
        assert out["request_datetime"].isna().all()

    def test_policy_uniform_gives_timestamps(self):
        rng = np.random.default_rng(42)
        out, _ = _run_yellow(_yellow_df(), policy=SyntheticRequestPolicy.UNIFORM_0_5MIN, rng=rng)
        assert not out["request_datetime"].isna().all()
        assert (out["request_datetime"] <= out["pickup_datetime"]).all()

    def test_non_manhattan_filtered(self):
        out, stats = _run_yellow(_yellow_df(pu=99, do=100))
        assert len(out) == 0
        assert stats["removed_non_manhattan"] == 3

    def test_trip_duration_computed(self):
        out, _ = _run_yellow(_yellow_df())
        expected = (T1 - T0).total_seconds()
        assert (out["trip_duration_seconds"] == expected).all()

    def test_trip_ids_are_unique(self):
        out, _ = _run_yellow(_yellow_df(n=5))
        assert out["trip_id"].nunique() == 5

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="yellow"):
            process_yellow(Path("/nonexistent/yellow.parquet"), MANHATTAN_ZONES)

    def test_missing_column_raises(self):
        df = _yellow_df().drop(columns=["trip_distance"])
        with patch.object(_tlc_mod.pd, "read_parquet", return_value=df), \
             patch.object(_tlc_mod.Path, "exists", return_value=True):
            with pytest.raises(ValueError, match="trip_distance"):
                process_yellow(Path("/fake/yellow.parquet"), MANHATTAN_ZONES)


# ---------------------------------------------------------------------------
# process_hvfhv
# ---------------------------------------------------------------------------

class TestProcessHvfhv:
    def test_output_schema(self):
        out, _ = _run_hvfhv(_hvfhv_df())
        assert list(out.columns) == OUTPUT_COLUMNS

    def test_uber_kept(self):
        out, _ = _run_hvfhv(_hvfhv_df(license_num=HVFHV_UBER))
        assert (out["provider"] == "uber").all()

    def test_lyft_kept(self):
        out, _ = _run_hvfhv(_hvfhv_df(license_num=HVFHV_LYFT))
        assert (out["provider"] == "lyft").all()

    def test_other_provider_filtered(self):
        out, stats = _run_hvfhv(_hvfhv_df(license_num="HV0002"))
        assert len(out) == 0
        assert stats["removed_provider_filter"] == 3

    def test_mixed_providers(self):
        df = pd.concat([
            _hvfhv_df(n=2, license_num=HVFHV_UBER),
            _hvfhv_df(n=2, license_num=HVFHV_LYFT),
            _hvfhv_df(n=1, license_num="HV0002"),
        ], ignore_index=True)
        out, stats = _run_hvfhv(df)
        assert len(out) == 4
        assert stats["removed_provider_filter"] == 1

    def test_request_datetime_is_synthetic_false(self):
        out, _ = _run_hvfhv(_hvfhv_df())
        assert not out["request_datetime_is_synthetic"].any()

    def test_request_datetime_preserved(self):
        out, _ = _run_hvfhv(_hvfhv_df())
        assert not out["request_datetime"].isna().all()

    def test_non_manhattan_filtered(self):
        out, stats = _run_hvfhv(_hvfhv_df(pu=99, do=100))
        assert len(out) == 0
        assert stats["removed_non_manhattan"] == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="fhvhv"):
            process_hvfhv(Path("/nonexistent/fhvhv.parquet"), MANHATTAN_ZONES)

    def test_missing_column_raises(self):
        df = _hvfhv_df().drop(columns=["trip_miles"])
        with patch.object(_tlc_mod.pd, "read_parquet", return_value=df), \
             patch.object(_tlc_mod.Path, "exists", return_value=True):
            with pytest.raises(ValueError, match="trip_miles"):
                process_hvfhv(Path("/fake/hvfhv.parquet"), MANHATTAN_ZONES)


# ---------------------------------------------------------------------------
# Output schema completeness
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_all_output_columns_present_yellow(self):
        out, _ = _run_yellow(_yellow_df())
        for col in OUTPUT_COLUMNS:
            assert col in out.columns, f"Missing column: {col}"

    def test_all_output_columns_present_hvfhv(self):
        out, _ = _run_hvfhv(_hvfhv_df())
        for col in OUTPUT_COLUMNS:
            assert col in out.columns, f"Missing column: {col}"
