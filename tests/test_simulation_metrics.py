import math
import sys
import tracemalloc
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import matching_env
from env.matching_env import MatchingEnv, SimConfig
from matching import CentralizedBatchMatcher, GreedyMatcher, RegionalBiddingMatcher, SparseMatcher


def test_metrics_reconcile_completed_expired_and_unfinished_requests():
    trips = pd.DataFrame([
        (0, 1, 0, 2, 0),
        (0, 9, 9, 9, 8),
        (100, 2, 0, 9, 0),
        (250, 3, 0, 3, 2),
    ], columns=["request_time_seconds", "pickup_row", "pickup_col", "dropoff_row", "dropoff_col"])
    env = MatchingEnv(GreedyMatcher(), SimConfig(n_drivers=1, request_timeout=180), trips_df=trips)
    env.reset()
    env.drivers["d0"].position = (0, 0)
    for _ in range(5):
        env.step()

    expected = {
        "n_requests_total": 4,
        "n_assignments": 2,
        "n_pickups": 2,
        "n_completed": 1,
        "n_expired": 1,
        "n_unfinished": 2,
        "assignment_rate": 0.5,
        "pickup_rate": 0.5,
        "trip_completion_rate": 0.25,
        "avg_request_to_assignment_seconds": (60 + 140) / 2,
        "avg_request_to_pickup_seconds": (120 + 140) / 2,
        "avg_assignment_to_pickup_seconds": (60 + 0) / 2,
        "avg_pickup_distance_cells": (1 + 0) / 2,
        "vehicle_utilization": (60 + 60) / 300,
        "avg_idle_seconds_per_driver": 120,
        "avg_completed_trips_per_driver": 1,
        "candidate_pairs_considered": 2,
        "avg_wait_seconds": 120,
    }
    metrics = env.metrics()
    assert {key: metrics[key] for key in expected} == expected
    assert metrics["n_requests_total"] == metrics["n_completed"] + metrics["n_expired"] + metrics["n_unfinished"]


@pytest.mark.parametrize(("matcher", "evaluations"), [
    (GreedyMatcher(), 3),
    (CentralizedBatchMatcher(), 4),
    (SparseMatcher(max_distance=1), 2),
    (RegionalBiddingMatcher(cross_region=False), 2),
])
def test_pair_count_measures_evaluations(matcher, evaluations):
    env = MatchingEnv(matcher, SimConfig(n_drivers=2))
    env.reset()
    env.drivers["d0"].position = (0, 0)
    env.drivers["d1"].position = (9, 9)
    env._create_request((0, 0), (0, 0))
    env._create_request((9, 9), (9, 9))
    env.step()
    assert env.metrics()["candidate_pairs_considered"] == evaluations


def test_empty_metrics_and_wall_clock_timing(monkeypatch):
    now = 0

    class TimedMatcher:
        def match(self, drivers, requests, context):
            nonlocal now
            now += 0.25
            return []

    monkeypatch.setattr(matching_env, "perf_counter", lambda: now)
    env = MatchingEnv(TimedMatcher(), SimConfig(n_drivers=0))
    empty = env.run(0)
    assert math.isnan(empty.performance["avg_matching_execution_seconds_per_timestep"])
    result = env.run(2)
    for key in ("assignment_rate", "pickup_rate", "trip_completion_rate", "vehicle_utilization",
                "avg_request_to_pickup_seconds", "avg_idle_seconds_per_driver", "avg_completed_trips_per_driver"):
        assert math.isnan(result.metrics[key])
    assert result.metrics["n_requests_total"] == 0
    assert result.performance == {
        "total_matching_execution_seconds": 0.5,
        "avg_matching_execution_seconds_per_timestep": 0.25,
        "total_simulation_execution_seconds": 0.5,
        "peak_matching_memory_bytes": None,
        "matching_memory_profiled_timesteps": 0,
    }


def test_memory_peak_captures_temporary_allocation_and_preserves_existing_tracer():
    class AllocatingMatcher:
        def match(self, drivers, requests, context):
            allocation = bytearray(65_536)
            assert len(allocation) == 65_536
            return []

    env = MatchingEnv(AllocatingMatcher(), profile_matching_memory=True)
    if not tracemalloc.is_tracing():
        result = env.run(2)
        assert result.performance["matching_memory_profiled_timesteps"] == 2
        assert result.performance["peak_matching_memory_bytes"] >= 65_536
        assert not tracemalloc.is_tracing()

    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    try:
        assert env.run(1).performance["peak_matching_memory_bytes"] is None
        assert tracemalloc.is_tracing()
    finally:
        if not already_tracing:
            tracemalloc.stop()
