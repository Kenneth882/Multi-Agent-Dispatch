import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.matching_env import Driver, MatchingContext, MatchingEnv, PassengerRequest, SimConfig
from matching import RegionalBiddingMatcher


def match(driver_positions, request_positions, **options):
    drivers = [Driver(f"d{i}", pos) for i, pos in enumerate(driver_positions)]
    requests = [PassengerRequest(f"r{i}", "test", pos, pos, 0) for i, pos in enumerate(request_positions)]
    context = MatchingContext(60, {}, {}, SimConfig())
    result = RegionalBiddingMatcher(**options).match(drivers, requests, context)
    assert len(result) == len({d for d, _ in result}) == len({r for _, r in result})
    return result, context.matcher_metrics


def test_competing_bid_displaces_driver_who_rebids_on_alternative():
    driver_positions = [(0, 1), (0, 2)]
    request_positions = [(0, 2), (0, 0)]
    options = {"cross_region": False, "max_pickup_distance": 1}
    result, metrics = match(driver_positions, request_positions, **options)
    assert result == [("d0", "r1"), ("d1", "r0")]
    assert metrics["regional_displacements"] == 1
    assert metrics["regional_bids"] == metrics["regional_auction_rounds"] == 3
    assert metrics["regional_iteration_limit_hits"] == 0
    assert metrics["regional_communication_events"] == 0
    assert match(driver_positions, request_positions, mode="greedy", **options)[0] == [("d0", "r0")]


def test_broker_preserves_local_awards_and_only_exchanges_leftovers():
    drivers = [(0, 4), (1, 4)]
    requests = [(0, 4), (1, 5)]
    local, local_metrics = match(drivers, requests, cross_region=False, max_pickup_distance=1)
    broker, broker_metrics = match(drivers, requests, cross_region=True, max_pickup_distance=1)
    assert local == [("d0", "r0")]
    assert local_metrics["regional_cross_assignments"] == 0
    assert broker == [("d0", "r0"), ("d1", "r1")]
    assert broker_metrics["regional_cross_assignments"] == 1
    assert broker_metrics["regional_communication_events"] == 4  # two submissions and two notifications


def test_broker_respects_cutoff_and_custom_partition():
    drivers, requests = [(0, 3)], [(0, 4)]
    result, metrics = match(drivers, requests, n_regions=3, cross_region=True, max_pickup_distance=0)
    assert result == []
    assert metrics["regional_cross_assignments"] == metrics["regional_bids"] == 0
    assert metrics["regional_communication_events"] == 2
    assert match(drivers, requests, n_regions=3, cross_region=False)[0] == []
    assert match(drivers, requests, n_regions=1, cross_region=False)[0] == [("d0", "r0")]


@pytest.mark.parametrize(("drivers", "requests", "count"), [
    ([(0, 0), (0, 1)], [(0, 0)], 1),
    ([(0, 0)], [(0, 1), (0, 2)], 1),
    ([], [(0, 0)], 0),
    ([(0, 0)], [], 0),
])
def test_unequal_partitions_terminate_with_unique_assignments(drivers, requests, count):
    result, metrics = match(drivers, requests, cross_region=False)
    assert len(result) == count
    assert metrics["regional_iteration_limit_hits"] == 0


def test_iteration_cap_returns_only_current_awards():
    drivers, requests = [(0, 0), (0, 1)], [(0, 0), (0, 1)]
    result, metrics = match(drivers, requests, cross_region=False, max_iterations=1)
    assert result == [("d0", "r0")]
    assert metrics["regional_auction_rounds"] == metrics["regional_bids"] == 1
    assert metrics["regional_iteration_limit_hits"] == 1


@pytest.mark.parametrize("options", [{"epsilon": 0}, {"epsilon": float("inf")}, {"max_iterations": 0}, {"mode": "other"}])
def test_invalid_auction_configuration(options):
    with pytest.raises(ValueError):
        RegionalBiddingMatcher(**options)


def test_environment_aggregates_and_resets_auction_metrics():
    env = MatchingEnv(RegionalBiddingMatcher(), SimConfig(n_drivers=1), profile_matching_memory=True)
    env.reset()
    env.drivers["d0"].position = (0, 0)
    env._create_request((0, 0), (0, 0))
    env.step()
    env._create_request((0, 0), (0, 0))
    env.step()
    assert env.metrics()["regional_bids"] == 2
    assert env.metrics()["regional_auction_rounds"] == 2
    assert env.performance_metrics()["total_matching_execution_seconds"] >= 0
    assert env.performance_metrics()["peak_matching_memory_bytes"] > 0
    assert env.run(1).metrics["regional_bids"] == 0
