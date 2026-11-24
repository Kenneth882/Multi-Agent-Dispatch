import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.matching_env import Driver, DriverStatus, MatchingContext, PassengerRequest, RequestStatus, SimConfig
from experiments import compare_matchers as comparison


def test_batch_can_match_two_when_greedy_uses_the_only_feasible_driver():
    drivers = {
        "a": Driver("a", (0, 1)),
        "b": Driver("b", (0, 3)),
        "busy": Driver("busy", (0, 0), status=DriverStatus.ON_TRIP),
    }
    requests = {
        "r1": PassengerRequest("r1", "test", (0, 2), (0, 2), 0),
        "r2": PassengerRequest("r2", "test", (0, 0), (0, 0), 1),
        "r3": PassengerRequest("r3", "test", (9, 9), (9, 9), 1),
    }
    context = MatchingContext(60, drivers, requests, SimConfig())
    greedy = comparison.make_matcher("greedy", 1)
    batch = comparison.make_matcher("centralized_batch", 1)
    inputs = (context.eligible_drivers(), context.waiting_requests(), context)
    assert greedy.match(*inputs) == [("a", "r1")]
    assert set(batch.match(*inputs)) == {("a", "r2"), ("b", "r1")}
    assert requests["r3"].status == RequestStatus.WAITING


@pytest.mark.parametrize("algorithm", comparison.ALGORITHMS)
def test_all_algorithms_share_pickup_cutoff_and_replay(algorithm):
    trips = pd.DataFrame([{
        "request_time_seconds": 0, "pickup_row": 0, "pickup_col": 0,
        "dropoff_row": 1, "dropoff_col": 1, "provider": "synthetic",
    }])
    result = comparison.run_matcher(algorithm, 0, SimConfig(n_drivers=1), 42, trips, 1)
    assert result["n_requests_total"] == result["n_unfinished"] == 1
    assert result["n_assignments"] == 0


def test_cli_shares_sample_and_configuration_across_algorithms(tmp_path, monkeypatch):
    calls = []

    def record(algorithm, cutoff, config, seed, trips, steps, measure_memory):
        calls.append((algorithm, cutoff, config, seed, trips, steps, measure_memory))
        return {}

    monkeypatch.setattr(comparison, "run_matcher", record)
    monkeypatch.setattr(comparison, "write_comparisons", lambda results, output, synthetic: None)
    comparison.main([
        "--synthetic", "--algorithms", "greedy", "centralized_batch",
        "--fleet-sizes", "20", "--sampling-rates", "5", "--seeds", "42", "43",
        "--pickup-cutoffs", "3", "--duration-minutes", "10", "--output-dir", str(tmp_path),
    ])
    assert len(calls) == 4
    for first, second in zip(calls[::2], calls[1::2]):
        assert (first[0], second[0]) == ("greedy", "centralized_batch")
        assert first[1:4] == second[1:4]
        assert first[4] is second[4]
        assert len(first[4]) == 10
        assert first[5:] == second[5:]
    assert {call[3] for call in calls} == {42, 43}
