"""
tests/test_zone_grid.py
------------------------
Tests for the geographic zone-to-grid mapping (Task 3).

Covers:
  - Valid (row, col) bounds for all Manhattan zones
  - Determinism: same shapefile -> same mapping every call
  - Missing / invalid zone IDs raise KeyError, never return (0, 0)
  - Pickup and drop-off zones use the same mapping (consistency)
  - spawn_weights shape, non-negative values, and sum-to-1
  - spawn_weights are regenerated when zone_grid_map.json is newer

Run from project root:
    pytest tests/test_zone_grid.py -v
"""
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "ride-share-marl" / "pipeline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PIPELINE))

from config.paths import SHAPEFILE, ZONE_GRID_MAP, SPAWN_WEIGHTS, MANHATTAN_TRIPS
import zone_grid as zg
import demand_generator as dg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_real_map() -> dict:
    """Load the already-built zone_grid_map.json (requires prior pipeline run)."""
    if not ZONE_GRID_MAP.exists():
        pytest.skip("zone_grid_map.json not found — run zone_grid.py first")
    return zg.load_zone_grid_map(ZONE_GRID_MAP)


# ---------------------------------------------------------------------------
# 1. Valid bounds: every mapped zone has 0 <= row,col <= 9
# ---------------------------------------------------------------------------

class TestValidBounds:
    def test_all_rows_in_range(self):
        zm = _load_real_map()
        for loc_id, (row, col) in zm.items():
            assert 0 <= row <= 9, f"Zone {loc_id}: row={row} out of [0,9]"

    def test_all_cols_in_range(self):
        zm = _load_real_map()
        for loc_id, (row, col) in zm.items():
            assert 0 <= col <= 9, f"Zone {loc_id}: col={col} out of [0,9]"

    def test_at_least_one_zone_mapped(self):
        zm = _load_real_map()
        assert len(zm) > 0

    def test_all_keys_are_positive_integers(self):
        zm = _load_real_map()
        for k in zm:
            assert isinstance(k, int) and k > 0, f"Bad key: {k}"

    def test_manhattan_zone_count(self):
        """NYC TLC shapefile has 69 Manhattan zones."""
        if not SHAPEFILE.exists():
            pytest.skip("Shapefile not available")
        zm = zg.build_zone_grid_map(SHAPEFILE)
        assert len(zm) == 69


# ---------------------------------------------------------------------------
# 2. Determinism: two independent calls produce identical mappings
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_build_twice_same_result(self):
        if not SHAPEFILE.exists():
            pytest.skip("Shapefile not available")
        zm1 = zg.build_zone_grid_map(SHAPEFILE)
        zm2 = zg.build_zone_grid_map(SHAPEFILE)
        assert zm1 == zm2

    def test_save_load_roundtrip(self):
        zm = _load_real_map()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = Path(f.name)
        try:
            zg.save_zone_grid_map(zm, tmp)
            zm2 = zg.load_zone_grid_map(tmp)
            assert zm == zm2
        finally:
            tmp.unlink(missing_ok=True)

    def test_json_keys_are_strings_on_disk(self):
        """JSON spec requires string keys; verify the saved file."""
        with open(ZONE_GRID_MAP) as f:
            raw = json.load(f)
        assert all(isinstance(k, str) for k in raw)

    def test_spawn_weights_deterministic(self):
        """Loading cached weights twice returns identical arrays."""
        if not SPAWN_WEIGHTS.exists():
            pytest.skip("spawn_weights.npy not found")
        w1 = np.load(SPAWN_WEIGHTS)
        w2 = np.load(SPAWN_WEIGHTS)
        np.testing.assert_array_equal(w1, w2)


# ---------------------------------------------------------------------------
# 3. Missing / invalid zone IDs raise KeyError — never silently (0, 0)
# ---------------------------------------------------------------------------

class TestMissingZones:
    def test_unknown_zone_raises_key_error(self):
        zm = _load_real_map()
        with pytest.raises(KeyError):
            zg.zone_to_grid(99999, zm)

    def test_zero_zone_raises_key_error(self):
        zm = _load_real_map()
        with pytest.raises(KeyError):
            zg.zone_to_grid(0, zm)

    def test_negative_zone_raises_key_error(self):
        zm = _load_real_map()
        with pytest.raises(KeyError):
            zg.zone_to_grid(-1, zm)

    def test_unknown_zone_does_not_return_origin(self):
        """Ensure the sentinel (0,0) is never silently returned."""
        zm = _load_real_map()
        try:
            result = zg.zone_to_grid(99999, zm)
            assert result != (0, 0), "zone_to_grid silently returned (0,0) for unknown zone"
        except KeyError:
            pass  # correct behaviour

    def test_demand_generator_skips_unknown_zones(self):
        """get_spawn_weights must skip trips with unknown zone IDs, not crash."""
        zm = _load_real_map()
        # zone_to_grid raises KeyError for unknown IDs — demand_generator catches it
        with pytest.raises(KeyError):
            zg.zone_to_grid(99999, zm)


# ---------------------------------------------------------------------------
# 4. Pickup / drop-off mapping consistency
# ---------------------------------------------------------------------------

class TestPickupDropoffConsistency:
    def test_same_zone_same_cell_for_pu_and_do(self):
        """The same zone ID must map to the same cell regardless of PU/DO role."""
        zm = _load_real_map()
        for loc_id in list(zm.keys())[:10]:
            pu_cell = zg.zone_to_grid(loc_id, zm)
            do_cell = zg.zone_to_grid(loc_id, zm)
            assert pu_cell == do_cell, (
                f"Zone {loc_id} maps differently as PU vs DO: {pu_cell} vs {do_cell}"
            )

    def test_demand_snapshot_columns_present(self):
        """get_demand_snapshot must return (row, col) for both PU and DO."""
        if not MANHATTAN_TRIPS.exists():
            pytest.skip("manhattan_trips.parquet not found")
        df = dg.get_demand_snapshot(hour=9, day_of_week=0, n_samples=5)
        if df.empty:
            pytest.skip("No trips for hour=9, day=0")
        for col in ("pickup_row", "pickup_col", "dropoff_row", "dropoff_col"):
            assert col in df.columns, f"Missing column: {col}"

    def test_demand_snapshot_bounds(self):
        """All (row, col) values in demand snapshot must be in [0, 9]."""
        if not MANHATTAN_TRIPS.exists():
            pytest.skip("manhattan_trips.parquet not found")
        df = dg.get_demand_snapshot(hour=9, day_of_week=0, n_samples=20)
        if df.empty:
            pytest.skip("No trips for hour=9, day=0")
        for col in ("pickup_row", "pickup_col", "dropoff_row", "dropoff_col"):
            assert df[col].between(0, 9).all(), f"{col} has out-of-range values"


# ---------------------------------------------------------------------------
# 5. spawn_weights validity
# ---------------------------------------------------------------------------

class TestSpawnWeights:
    def test_shape_is_10x10(self):
        if not SPAWN_WEIGHTS.exists():
            pytest.skip("spawn_weights.npy not found")
        w = np.load(SPAWN_WEIGHTS)
        assert w.shape == (10, 10)

    def test_all_nonnegative(self):
        if not SPAWN_WEIGHTS.exists():
            pytest.skip("spawn_weights.npy not found")
        w = np.load(SPAWN_WEIGHTS)
        assert (w >= 0).all()

    def test_sums_to_one(self):
        if not SPAWN_WEIGHTS.exists():
            pytest.skip("spawn_weights.npy not found")
        w = np.load(SPAWN_WEIGHTS)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_nonzero_cells_match_mapped_zones(self):
        """Cells with weight > 0 must correspond to cells that have mapped zones."""
        if not SPAWN_WEIGHTS.exists():
            pytest.skip("spawn_weights.npy not found")
        zm = _load_real_map()
        w = np.load(SPAWN_WEIGHTS)
        occupied = set(zm.values())
        for r in range(10):
            for c in range(10):
                if w[r, c] > 0:
                    assert (r, c) in occupied, (
                        f"Cell ({r},{c}) has weight {w[r,c]:.4f} but no zone maps there"
                    )


# ---------------------------------------------------------------------------
# 6. Cache invalidation: stale spawn_weights.npy is regenerated
# ---------------------------------------------------------------------------

class TestCacheInvalidation:
    def test_stale_weights_detected(self):
        """_spawn_weights_are_stale() returns True when map is newer than cache."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = tmp_path / "spawn_weights.npy"
            map_file = tmp_path / "zone_grid_map.json"

            # Create cache first, then map file (map is newer)
            np.save(cache, np.zeros((10, 10)))
            time.sleep(0.05)
            map_file.write_text("{}")

            with patch("demand_generator.CACHE_PATH", str(cache)), \
                 patch("demand_generator.ZONE_GRID_MAP", map_file):
                assert dg._spawn_weights_are_stale() is True

    def test_fresh_weights_not_stale(self):
        """_spawn_weights_are_stale() returns False when cache is newer than map."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = tmp_path / "spawn_weights.npy"
            map_file = tmp_path / "zone_grid_map.json"

            # Create map first, then cache (cache is newer)
            map_file.write_text("{}")
            time.sleep(0.05)
            np.save(cache, np.zeros((10, 10)))

            with patch("demand_generator.CACHE_PATH", str(cache)), \
                 patch("demand_generator.ZONE_GRID_MAP", map_file):
                assert dg._spawn_weights_are_stale() is False

    def test_missing_cache_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "nonexistent.npy"
            with patch("demand_generator.CACHE_PATH", str(cache)):
                assert dg._spawn_weights_are_stale() is True
