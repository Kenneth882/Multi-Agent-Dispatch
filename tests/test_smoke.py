"""
Smoke tests for Senior Project 2 setup.
Run from project root: pytest tests/test_smoke.py -v
"""
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_config_paths_importable():
    from config.paths import ROOT, PROCESSED_DIR, RESULTS_DIR, SPAWN_WEIGHTS
    assert ROOT.exists(), f"Project root not found: {ROOT}"
    assert (ROOT / "config").exists()
    assert (ROOT / "matching").exists()
    assert (ROOT / "tests").exists()


def test_config_paths_types():
    from config import paths
    from pathlib import Path
    assert isinstance(paths.ROOT, Path)
    assert isinstance(paths.SPAWN_WEIGHTS, Path)
    assert isinstance(paths.MANHATTAN_TRIPS, Path)


def test_env_package_importable():
    from env.ride_share_env import RideShareEnv, GRID_SIZE, NUM_AGENTS, MAX_STEPS
    assert GRID_SIZE == 10
    assert NUM_AGENTS == 20
    assert MAX_STEPS == 200


def test_matching_package_importable():
    import matching
    from matching.base import BaseMatcher


def test_models_package_importable():
    from models.baseline_agent import SingleAgentWrapper


def test_explanations_package_importable():
    from explanations.explain import generate_step_explanations
    from explanations.consistency_check import ConsistencyChecker


def test_directory_structure():
    expected = ["env", "models", "experiments", "explanations",
                "matching", "tests", "config", "results", "data"]
    for d in expected:
        assert (ROOT / d).is_dir(), f"Missing directory: {d}"


def test_env_reset_runs():
    from env.ride_share_env import RideShareEnv
    env = RideShareEnv()
    obs, info = env.reset(seed=0)
    assert len(obs) == 20
    assert all(v.shape == (202,) for v in obs.values())


def test_base_matcher_is_abstract():
    from matching.base import BaseMatcher
    import pytest
    with pytest.raises(TypeError):
        BaseMatcher()
