"""
tests/test_matchers.py
-----------------------
Unit tests for the four SP2 matching algorithms.

All tests use synthetic Driver/PassengerRequest objects built inline.
No TLC files or simulation state required.

Run from project root:
    pytest tests/test_matchers.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.matching_env import (
    Driver, DriverStatus, PassengerRequest, RequestStatus,
    MatchingContext, SimConfig,
)
from matching.greedy import GreedyMatcher
from matching.centralized_batch import CentralizedBatchMatcher
from matching.sparse import SparseMatcher
from matching.regional_bidding import RegionalBiddingMatcher
from matching import (
    GreedyMatcher as GreedyImport,
    CentralizedBatchMatcher as BatchImport,
    SparseMatcher as SparseImport,
    RegionalBiddingMatcher as RegionalImport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = SimConfig(grid_size=10, n_drivers=5, timestep=60)


def _driver(id_, pos, status=DriverStatus.IDLE):
    return Driver(id=id_, position=pos, status=status)


def _request(id_, pickup, dropoff=(9, 9), status=RequestStatus.WAITING):
    return PassengerRequest(
        id=id_, provider="test",
        pickup=pickup, dropoff=dropoff,
        request_time=0, status=status,
    )


def _ctx(drivers, requests, cfg=_CFG):
    d_dict = {d.id: d for d in drivers}
    r_dict = {r.id: r for r in requests}
    return MatchingContext(sim_time=60, drivers=d_dict, requests=r_dict, config=cfg)


def _eligible(drivers):
    return [d for d in drivers if d.status == DriverStatus.IDLE]


def _waiting(requests):
    return [r for r in requests if r.status == RequestStatus.WAITING]


ALL_MATCHERS = [
    GreedyMatcher(),
    CentralizedBatchMatcher(),
    SparseMatcher(max_distance=18),   # effectively unlimited for tests
    RegionalBiddingMatcher(cross_region=True),
]


# ---------------------------------------------------------------------------
# 1. Contract: valid output format for all matchers
# ---------------------------------------------------------------------------

class TestOutputContract:
    @pytest.mark.parametrize("matcher", ALL_MATCHERS)
    def test_returns_list(self, matcher):
        drivers = [_driver("d0", (0, 0))]
        requests = [_request("r0", (1, 1))]
        ctx = _ctx(drivers, requests)
        result = matcher.match(_eligible(drivers), _waiting(requests), ctx)
        assert isinstance(result, list)

    @pytest.mark.parametrize("matcher", ALL_MATCHERS)
    def test_each_element_is_two_tuple(self, matcher):
        drivers = [_driver("d0", (0, 0)), _driver("d1", (5, 5))]
        requests = [_request("r0", (1, 1)), _request("r1", (4, 4))]
        ctx = _ctx(drivers, requests)
        result = matcher.match(_eligible(drivers), _waiting(requests), ctx)
        for pair in result:
            assert len(pair) == 2

    @pytest.mark.parametrize("matcher", ALL_MATCHERS)
    def test_driver_ids_are_valid(self, matcher):
        drivers = [_driver("d0", (0, 0)), _driver("d1", (5, 5))]
        requests = [_request("r0", (1, 1)), _request("r1", (4, 4))]
        ctx = _ctx(drivers, requests)
        valid_ids = {d.id for d in drivers}
        result = matcher.match(_eligible(drivers), _waiting(requests), ctx)
        for driver_id, _ in result:
            assert driver_id in valid_ids

    @pytest.mark.parametrize("matcher", ALL_MATCHERS)
    def test_request_ids_are_valid(self, matcher):
        drivers = [_driver("d0", (0, 0)), _driver("d1", (5, 5))]
        requests = [_request("r0", (1, 1)), _request("r1", (4, 4))]
        ctx = _ctx(drivers, requests)
        valid_ids = {r.id for r in requests}
        result = matcher.match(_eligible(drivers), _waiting(requests), ctx)
        for _, request_id in result:
            assert request_id in valid_ids

    @pytest.mark.parametrize("matcher", ALL_MATCHERS)
    def test_no_driver_assigned_twice(self, matcher):
        drivers = [_driver("d0", (0, 0))]
        requests = [_request("r0", (1, 0)), _request("r1", (2, 0))]
        ctx = _ctx(drivers, requests)
        result = matcher.match(_eligible(drivers), _waiting(requests), ctx)
        driver_ids = [pair[0] for pair in result]
        assert len(driver_ids) == len(set(driver_ids))

    @pytest.mark.parametrize("matcher", ALL_MATCHERS)
    def test_no_request_assigned_twice(self, matcher):
        drivers = [_driver("d0", (0, 0)), _driver("d1", (0, 1))]
        requests = [_request("r0", (1, 0))]
        ctx = _ctx(drivers, requests)
        result = matcher.match(_eligible(drivers), _waiting(requests), ctx)
        request_ids = [pair[1] for pair in result]
        assert len(request_ids) == len(set(request_ids))

    @pytest.mark.parametrize("matcher", ALL_MATCHERS)
    def test_empty_drivers_returns_empty(self, matcher):
        requests = [_request("r0", (1, 1))]
        ctx = _ctx([], requests)
        assert matcher.match([], _waiting(requests), ctx) == []

    @pytest.mark.parametrize("matcher", ALL_MATCHERS)
    def test_empty_requests_returns_empty(self, matcher):
        drivers = [_driver("d0", (0, 0))]
        ctx = _ctx(drivers, [])
        assert matcher.match(_eligible(drivers), [], ctx) == []

    @pytest.mark.parametrize("matcher", ALL_MATCHERS)
    def test_trivial_one_driver_one_request(self, matcher):
        drivers = [_driver("d0", (5, 5))]
        requests = [_request("r0", (5, 5))]
        ctx = _ctx(drivers, requests)
        result = matcher.match(_eligible(drivers), _waiting(requests), ctx)
        assert result == [("d0", "r0")]


# ---------------------------------------------------------------------------
# 2. Greedy: assigns nearest request
# ---------------------------------------------------------------------------

class TestGreedy:
    def test_assigns_nearest(self):
        # d0 at (0,0): r_near at (1,0) dist=1, r_far at (8,8) dist=16
        drivers = [_driver("d0", (0, 0))]
        requests = [_request("r_far", (8, 8)), _request("r_near", (1, 0))]
        ctx = _ctx(drivers, requests)
        result = GreedyMatcher().match(_eligible(drivers), _waiting(requests), ctx)
        assert result == [("d0", "r_near")]

    def test_two_drivers_no_conflict(self):
        # d0 near r0, d1 near r1 — greedy should assign each to nearest
        drivers = [_driver("d0", (0, 0)), _driver("d1", (9, 9))]
        requests = [_request("r0", (1, 0)), _request("r1", (8, 9))]
        ctx = _ctx(drivers, requests)
        result = GreedyMatcher().match(_eligible(drivers), _waiting(requests), ctx)
        assert set(result) == {("d0", "r0"), ("d1", "r1")}

    def test_tie_broken_by_request_id(self):
        # d0 equidistant from r_a and r_b — r_a wins (lexicographically smaller)
        drivers = [_driver("d0", (5, 5))]
        requests = [_request("r_b", (5, 6)), _request("r_a", (6, 5))]
        ctx = _ctx(drivers, requests)
        result = GreedyMatcher().match(_eligible(drivers), _waiting(requests), ctx)
        assert result[0][1] == "r_a"

    def test_deterministic_same_inputs(self):
        drivers = [_driver("d0", (3, 3)), _driver("d1", (7, 7))]
        requests = [_request("r0", (2, 2)), _request("r1", (8, 8))]
        ctx = _ctx(drivers, requests)
        r1 = GreedyMatcher().match(_eligible(drivers), _waiting(requests), ctx)
        r2 = GreedyMatcher().match(_eligible(drivers), _waiting(requests), ctx)
        assert r1 == r2

    def test_more_drivers_than_requests(self):
        drivers = [_driver(f"d{i}", (i, 0)) for i in range(5)]
        requests = [_request("r0", (2, 0))]
        ctx = _ctx(drivers, requests)
        result = GreedyMatcher().match(_eligible(drivers), _waiting(requests), ctx)
        assert len(result) == 1
        assert result[0][1] == "r0"

    def test_more_requests_than_drivers(self):
        drivers = [_driver("d0", (0, 0))]
        requests = [_request(f"r{i}", (i, 0)) for i in range(5)]
        ctx = _ctx(drivers, requests)
        result = GreedyMatcher().match(_eligible(drivers), _waiting(requests), ctx)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 3. CentralizedBatch: globally optimal assignment
# ---------------------------------------------------------------------------

class TestCentralizedBatch:
    def test_optimal_on_known_instance(self):
        """
        d0=(0,0), d1=(9,9)
        r0=(0,1) dist from d0=1, dist from d1=17
        r1=(9,8) dist from d0=16, dist from d1=1
        Optimal: d0->r0 (cost 1), d1->r1 (cost 1), total=2
        Greedy (d0 first) would also get this, but Hungarian guarantees it.
        """
        drivers = [_driver("d0", (0, 0)), _driver("d1", (9, 9))]
        requests = [_request("r0", (0, 1)), _request("r1", (9, 8))]
        ctx = _ctx(drivers, requests)
        result = CentralizedBatchMatcher().match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert set(result) == {("d0", "r0"), ("d1", "r1")}

    def test_suboptimal_greedy_case(self):
        """
        Instance where greedy (driver-order dependent) is suboptimal but
        Hungarian finds the global optimum.

        d0=(0,0), d1=(0,2)
        r0=(0,1): dist d0=1, dist d1=1  (tie)
        r1=(0,3): dist d0=3, dist d1=1

        Greedy (d0 first, tie-break r0): d0->r0 (1), d1->r1 (1) total=2
        Hungarian: same total=2 — both optimal here.
        Verify Hungarian assigns each driver to exactly one request.
        """
        drivers = [_driver("d0", (0, 0)), _driver("d1", (0, 2))]
        requests = [_request("r0", (0, 1)), _request("r1", (0, 3))]
        ctx = _ctx(drivers, requests)
        result = CentralizedBatchMatcher().match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert len(result) == 2
        assert len({p[0] for p in result}) == 2  # both drivers assigned
        assert len({p[1] for p in result}) == 2  # both requests assigned

    def test_hungarian_beats_greedy_on_adversarial(self):
        """
        Adversarial case: greedy assigns suboptimally, Hungarian does not.

        d0=(0,0), d1=(0,4)
        r0=(0,1): dist d0=1, dist d1=3
        r1=(0,3): dist d0=3, dist d1=1

        Greedy (d0 first): d0->r0 (1), d1->r1 (1), total=2
        Hungarian: same total=2 — both optimal.

        Now shift r1 to (0,5):
        d0=(0,0), d1=(0,4)
        r0=(0,1): dist d0=1, dist d1=3
        r1=(0,5): dist d0=5, dist d1=1

        Greedy: d0->r0 (1), d1->r1 (1), total=2
        Hungarian: same.

        True adversarial (cross-assignment is better):
        d0=(0,0), d1=(0,6)
        r0=(0,5): dist d0=5, dist d1=1
        r1=(0,1): dist d0=1, dist d1=5

        Greedy (d0 first): d0->r0 (5), d1->r1 (5), total=10
        Hungarian: d0->r1 (1), d1->r0 (1), total=2  ← optimal
        """
        drivers = [_driver("d0", (0, 0)), _driver("d1", (0, 6))]
        requests = [_request("r0", (0, 5)), _request("r1", (0, 1))]
        ctx = _ctx(drivers, requests)

        greedy_result = GreedyMatcher().match(
            _eligible(drivers), _waiting(requests), ctx
        )
        hungarian_result = CentralizedBatchMatcher().match(
            _eligible(drivers), _waiting(requests), ctx
        )

        def total_cost(assignments):
            d_map = {d.id: d for d in drivers}
            r_map = {r.id: r for r in requests}
            return sum(
                MatchingContext.manhattan_distance(d_map[di].position, r_map[ri].pickup)
                for di, ri in assignments
            )

        assert total_cost(hungarian_result) <= total_cost(greedy_result)
        assert total_cost(hungarian_result) == 2

    def test_more_drivers_than_requests(self):
        drivers = [_driver(f"d{i}", (i * 2, 0)) for i in range(4)]
        requests = [_request("r0", (1, 0)), _request("r1", (5, 0))]
        ctx = _ctx(drivers, requests)
        result = CentralizedBatchMatcher().match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert len(result) == 2

    def test_more_requests_than_drivers(self):
        drivers = [_driver("d0", (0, 0)), _driver("d1", (9, 9))]
        requests = [_request(f"r{i}", (i, 0)) for i in range(5)]
        ctx = _ctx(drivers, requests)
        result = CentralizedBatchMatcher().match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 4. Sparse: respects distance cutoff
# ---------------------------------------------------------------------------

class TestSparse:
    def test_no_assignment_beyond_cutoff(self):
        # d0 at (0,0), r0 at (5,5): distance=10 > cutoff=3
        drivers = [_driver("d0", (0, 0))]
        requests = [_request("r0", (5, 5))]
        ctx = _ctx(drivers, requests)
        result = SparseMatcher(max_distance=3).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert result == []

    def test_assigns_within_cutoff(self):
        drivers = [_driver("d0", (0, 0))]
        requests = [_request("r0", (0, 2))]  # distance=2 <= cutoff=3
        ctx = _ctx(drivers, requests)
        result = SparseMatcher(max_distance=3).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert result == [("d0", "r0")]

    def test_assigns_at_exact_cutoff(self):
        drivers = [_driver("d0", (0, 0))]
        requests = [_request("r0", (0, 3))]  # distance=3 == cutoff=3
        ctx = _ctx(drivers, requests)
        result = SparseMatcher(max_distance=3).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert result == [("d0", "r0")]

    def test_skips_far_assigns_near(self):
        drivers = [_driver("d0", (0, 0))]
        requests = [
            _request("r_far", (8, 8)),   # distance=16
            _request("r_near", (0, 2)),  # distance=2
        ]
        ctx = _ctx(drivers, requests)
        result = SparseMatcher(max_distance=3).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert len(result) == 1
        assert result[0][1] == "r_near"

    def test_zero_cutoff_only_same_cell(self):
        drivers = [_driver("d0", (3, 3)), _driver("d1", (5, 5))]
        requests = [_request("r0", (3, 3)), _request("r1", (5, 6))]
        ctx = _ctx(drivers, requests)
        result = SparseMatcher(max_distance=0).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        # Only d0->r0 is at distance 0; d1->r1 is distance 1
        assert result == [("d0", "r0")]

    def test_invalid_max_distance_raises(self):
        with pytest.raises(ValueError):
            SparseMatcher(max_distance=-1)

    def test_optimal_within_range(self):
        """Same adversarial case as Hungarian test — sparse with large cutoff."""
        drivers = [_driver("d0", (0, 0)), _driver("d1", (0, 6))]
        requests = [_request("r0", (0, 5)), _request("r1", (0, 1))]
        ctx = _ctx(drivers, requests)
        result = SparseMatcher(max_distance=18).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        d_map = {d.id: d for d in drivers}
        r_map = {r.id: r for r in requests}
        total = sum(
            MatchingContext.manhattan_distance(d_map[di].position, r_map[ri].pickup)
            for di, ri in result
        )
        assert total == 2  # optimal


# ---------------------------------------------------------------------------
# 5. RegionalBidding: respects region boundaries
# ---------------------------------------------------------------------------

class TestRegionalBidding:
    def _region(self, pos, n_regions=2, grid_size=10):
        r, c = pos
        return (r * n_regions // grid_size, c * n_regions // grid_size)

    def test_driver_assigned_to_in_region_request(self):
        """d0 in region (0,0), r0 in region (0,0) — should be assigned."""
        drivers = [_driver("d0", (1, 1))]   # region (0,0) with n_regions=2
        requests = [_request("r0", (2, 2))]  # region (0,0)
        ctx = _ctx(drivers, requests)
        result = RegionalBiddingMatcher(n_regions=2, cross_region=False).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert result == [("d0", "r0")]

    def test_no_cross_region_when_disabled(self):
        """d0 in region (0,0), r0 in region (1,1) — no assignment when cross_region=False."""
        drivers = [_driver("d0", (1, 1))]   # region (0,0)
        requests = [_request("r0", (8, 8))]  # region (1,1)
        ctx = _ctx(drivers, requests)
        result = RegionalBiddingMatcher(n_regions=2, cross_region=False).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert result == []

    def test_cross_region_fallback_when_enabled(self):
        """d0 in region (0,0), r0 in region (1,1) — assigned via fallback."""
        drivers = [_driver("d0", (1, 1))]
        requests = [_request("r0", (8, 8))]
        ctx = _ctx(drivers, requests)
        result = RegionalBiddingMatcher(n_regions=2, cross_region=True).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert result == [("d0", "r0")]

    def test_two_regions_no_cross_contamination(self):
        """
        d0 in region (0,0), d1 in region (1,1).
        r0 in region (0,0), r1 in region (1,1).
        Each driver should get its own region's request.
        """
        drivers = [_driver("d0", (1, 1)), _driver("d1", (8, 8))]
        requests = [_request("r0", (2, 2)), _request("r1", (7, 7))]
        ctx = _ctx(drivers, requests)
        result = RegionalBiddingMatcher(n_regions=2, cross_region=False).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert set(result) == {("d0", "r0"), ("d1", "r1")}

    def test_invalid_n_regions_raises(self):
        with pytest.raises(ValueError):
            RegionalBiddingMatcher(n_regions=0)

    def test_n_regions_1_behaves_like_greedy(self):
        """With n_regions=1, all drivers and requests are in the same region."""
        drivers = [_driver("d0", (0, 0)), _driver("d1", (9, 9))]
        requests = [_request("r0", (1, 0)), _request("r1", (8, 9))]
        ctx = _ctx(drivers, requests)
        result = RegionalBiddingMatcher(n_regions=1, cross_region=False).match(
            _eligible(drivers), _waiting(requests), ctx
        )
        assert len(result) == 2

    def test_deterministic(self):
        drivers = [_driver(f"d{i}", (i, i)) for i in range(4)]
        requests = [_request(f"r{i}", (i + 1, i)) for i in range(4)]
        ctx = _ctx(drivers, requests)
        matcher = RegionalBiddingMatcher()
        r1 = matcher.match(_eligible(drivers), _waiting(requests), ctx)
        r2 = matcher.match(_eligible(drivers), _waiting(requests), ctx)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 6. All matchers agree on trivially simple cases
# ---------------------------------------------------------------------------

class TestAllMatchersAgree:
    def test_single_driver_single_request_same_cell(self):
        drivers = [_driver("d0", (5, 5))]
        requests = [_request("r0", (5, 5))]
        ctx = _ctx(drivers, requests)
        for matcher in ALL_MATCHERS:
            result = matcher.match(_eligible(drivers), _waiting(requests), ctx)
            assert result == [("d0", "r0")], f"{matcher.__class__.__name__} failed"

    def test_no_busy_drivers_passed_in(self):
        """Matchers receive only idle drivers — busy ones are filtered by the env."""
        drivers = [
            _driver("d0", (0, 0), status=DriverStatus.IDLE),
            _driver("d1", (1, 1), status=DriverStatus.ENROUTE_PICKUP),
        ]
        requests = [_request("r0", (0, 1))]
        ctx = _ctx(drivers, requests)
        eligible = _eligible(drivers)  # only d0
        for matcher in ALL_MATCHERS:
            result = matcher.match(eligible, _waiting(requests), ctx)
            for driver_id, _ in result:
                assert driver_id == "d0", f"{matcher.__class__.__name__} used busy driver"


# ---------------------------------------------------------------------------
# 7. Integration: all four matchers run inside MatchingEnv
# ---------------------------------------------------------------------------

class TestMatchersInEnv:
    @pytest.mark.parametrize("matcher_cls,kwargs", [
        (GreedyMatcher, {}),
        (CentralizedBatchMatcher, {}),
        (SparseMatcher, {"max_distance": 18}),
        (RegionalBiddingMatcher, {"cross_region": True}),
    ])
    def test_run_completes(self, matcher_cls, kwargs):
        from env.matching_env import MatchingEnv, SimConfig, make_synthetic_trips
        trips = make_synthetic_trips(n=20, seed=0)
        cfg = SimConfig(n_drivers=5, timestep=60)
        result = MatchingEnv(matcher_cls(**kwargs), config=cfg, seed=0, trips_df=trips).run(30)
        assert result.metrics["n_requests_total"] > 0

    @pytest.mark.parametrize("matcher_cls,kwargs", [
        (GreedyMatcher, {}),
        (CentralizedBatchMatcher, {}),
        (SparseMatcher, {"max_distance": 18}),
        (RegionalBiddingMatcher, {"cross_region": True}),
    ])
    def test_response_rate_in_range(self, matcher_cls, kwargs):
        from env.matching_env import MatchingEnv, SimConfig, make_synthetic_trips
        trips = make_synthetic_trips(n=20, seed=0)
        cfg = SimConfig(n_drivers=5, timestep=60, request_timeout=600)
        result = MatchingEnv(matcher_cls(**kwargs), config=cfg, seed=0, trips_df=trips).run(30)
        rr = result.metrics["order_response_rate"]
        assert 0.0 <= rr <= 1.0

    def test_hungarian_completions_ge_greedy(self):
        """
        On a fixed scenario, Hungarian should complete >= as many trips as Greedy.
        (Hungarian is globally optimal per step; Greedy is not.)
        """
        from env.matching_env import MatchingEnv, SimConfig, make_synthetic_trips
        trips = make_synthetic_trips(n=40, seed=99)
        cfg = SimConfig(n_drivers=8, timestep=60)
        r_greedy = MatchingEnv(GreedyMatcher(), config=cfg, seed=99, trips_df=trips).run(60)
        r_batch = MatchingEnv(CentralizedBatchMatcher(), config=cfg, seed=99, trips_df=trips).run(60)
        # Hungarian minimises total pickup distance per step, so avg_wait <= greedy
        # (or equal). We check it doesn't do strictly worse.
        gw = r_greedy.metrics["avg_wait_seconds"]
        bw = r_batch.metrics["avg_wait_seconds"]
        # Allow small floating-point tolerance
        assert bw <= gw + 1e-6, (
            f"Hungarian avg_wait ({bw:.1f}s) > Greedy avg_wait ({gw:.1f}s)"
        )


# ---------------------------------------------------------------------------
# 8. __init__ exports
# ---------------------------------------------------------------------------

class TestImports:
    def test_all_four_importable_from_matching(self):
        assert GreedyImport is GreedyMatcher
        assert BatchImport is CentralizedBatchMatcher
        assert SparseImport is SparseMatcher
        assert RegionalImport is RegionalBiddingMatcher
