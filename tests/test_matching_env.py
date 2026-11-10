"""
tests/test_matching_env.py
---------------------------
Unit tests for env/matching_env.py (SP2 matching simulation).

No TLC files required — all tests use synthetic scenarios.

Run from project root:
    pytest tests/test_matching_env.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.matching_env import (
    SimConfig,
    DriverStatus,
    RequestStatus,
    Driver,
    PassengerRequest,
    MatchingContext,
    MatchingEnv,
    StepResult,
    SimResult,
    make_synthetic_trips,
    _step_toward,
)
from matching.base import BaseMatcher


# ---------------------------------------------------------------------------
# Test matchers
# ---------------------------------------------------------------------------

class NullMatcher(BaseMatcher):
    """Never assigns anyone."""
    def match(self, eligible_drivers, waiting_requests, context):
        return []


class GreedyNearestMatcher(BaseMatcher):
    """Assigns each idle driver to the nearest waiting request."""
    def match(self, eligible_drivers, waiting_requests, context):
        assignments = []
        used = set()
        for driver in eligible_drivers:
            best, best_d = None, float("inf")
            for req in waiting_requests:
                if req.id in used:
                    continue
                d = context.manhattan_distance(driver.position, req.pickup)
                if d < best_d:
                    best_d = d
                    best = req
            if best:
                assignments.append((driver.id, best.id))
                used.add(best.id)
        return assignments


class DoubleMatcher(BaseMatcher):
    """Tries to assign the same driver twice — should be rejected."""
    def match(self, eligible_drivers, waiting_requests, context):
        if not eligible_drivers or len(waiting_requests) < 2:
            return []
        d = eligible_drivers[0]
        return [
            (d.id, waiting_requests[0].id),
            (d.id, waiting_requests[1].id),
        ]


class DoubleRequestMatcher(BaseMatcher):
    """Tries to assign the same request to two drivers — should be rejected."""
    def match(self, eligible_drivers, waiting_requests, context):
        if len(eligible_drivers) < 2 or not waiting_requests:
            return []
        r = waiting_requests[0]
        return [
            (eligible_drivers[0].id, r.id),
            (eligible_drivers[1].id, r.id),
        ]


class BusyDriverMatcher(BaseMatcher):
    """
    Tries to assign a driver that is already ENROUTE_PICKUP.
    Passes all drivers (not just eligible) to simulate a buggy matcher.
    """
    def __init__(self, env_ref):
        self._env = env_ref

    def match(self, eligible_drivers, waiting_requests, context):
        if not waiting_requests:
            return []
        # Pick the first non-idle driver directly from env state
        busy = [d for d in self._env.drivers.values()
                if d.status != DriverStatus.IDLE]
        if not busy or not waiting_requests:
            return []
        return [(busy[0].id, waiting_requests[0].id)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(matcher=None, n_drivers=3, timestep=60, timeout=None, seed=0):
    cfg = SimConfig(n_drivers=n_drivers, timestep=timestep, request_timeout=timeout)
    trips = make_synthetic_trips(n=10, seed=seed)
    return MatchingEnv(matcher or NullMatcher(), config=cfg, seed=seed, trips_df=trips)


def _env_with_one_request(pickup=(5, 5), dropoff=(7, 7), driver_pos=(5, 5)):
    """Minimal env: 1 driver placed at driver_pos, 1 pre-injected request."""
    cfg = SimConfig(n_drivers=1, timestep=60)
    env = MatchingEnv(NullMatcher(), config=cfg, seed=0, trips_df=None)
    env.reset()
    # Override driver position
    env.drivers["d0"].position = driver_pos
    # Manually inject a request at t=0
    env.sim_time = 0
    req = env._create_request(pickup=pickup, dropoff=dropoff)
    return env, req


# ---------------------------------------------------------------------------
# 1. Full lifecycle: IDLE -> ENROUTE_PICKUP -> ON_TRIP -> IDLE
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_driver_starts_idle(self):
        env = _make_env()
        env.reset()
        for d in env.drivers.values():
            assert d.status == DriverStatus.IDLE

    def test_assignment_sets_enroute(self):
        env, req = _env_with_one_request(pickup=(5, 5), driver_pos=(5, 5))
        env.matcher = GreedyNearestMatcher()
        env.step()
        d = env.drivers["d0"]
        # Zero-distance: driver already at pickup, so pickup happens this step
        assert d.status in (DriverStatus.ON_TRIP, DriverStatus.IDLE)

    def test_enroute_then_pickup_then_complete(self):
        """Driver at (0,0), pickup at (2,0), dropoff at (2,2). speed=1."""
        env, req = _env_with_one_request(pickup=(2, 0), dropoff=(2, 2), driver_pos=(0, 0))
        env.matcher = GreedyNearestMatcher()

        # Step 1: assign -> driver moves toward (2,0)
        env.step()
        assert env.drivers["d0"].status == DriverStatus.ENROUTE_PICKUP
        assert req.status == RequestStatus.ASSIGNED

        # Steps 2-3: driver reaches pickup at (2,0)
        env.step()
        env.step()
        assert env.drivers["d0"].status == DriverStatus.ON_TRIP
        assert req.status == RequestStatus.PICKED_UP
        assert req.actual_simulated_pickup_time is not None

        # Steps 4-5: driver reaches dropoff at (2,2)
        env.step()
        env.step()
        assert env.drivers["d0"].status == DriverStatus.IDLE
        assert req.status == RequestStatus.COMPLETED
        assert req.completion_time is not None

    def test_zero_distance_pickup(self):
        """Driver already at pickup cell — pickup happens in the same step as assignment."""
        env, req = _env_with_one_request(pickup=(3, 3), dropoff=(3, 3), driver_pos=(3, 3))
        env.matcher = GreedyNearestMatcher()
        env.step()
        # Same cell for pickup and dropoff: should complete in one step
        assert req.status in (RequestStatus.PICKED_UP, RequestStatus.COMPLETED)

    def test_zero_distance_dropoff(self):
        """Pickup == dropoff: driver completes trip in one step after pickup."""
        env, req = _env_with_one_request(pickup=(4, 4), dropoff=(4, 4), driver_pos=(4, 4))
        env.matcher = GreedyNearestMatcher()
        env.step()
        assert req.status == RequestStatus.COMPLETED

    def test_driver_returns_to_idle_after_completion(self):
        env, req = _env_with_one_request(pickup=(1, 0), dropoff=(1, 1), driver_pos=(0, 0))
        env.matcher = GreedyNearestMatcher()
        for _ in range(10):
            env.step()
            if req.status == RequestStatus.COMPLETED:
                break
        assert env.drivers["d0"].status == DriverStatus.IDLE
        assert env.drivers["d0"].assigned_request_id is None

    def test_events_recorded_in_order(self):
        env, req = _env_with_one_request(pickup=(1, 0), dropoff=(1, 1), driver_pos=(0, 0))
        env.matcher = GreedyNearestMatcher()
        for _ in range(10):
            env.step()
            if req.status == RequestStatus.COMPLETED:
                break
        event_types = [e["event"] for e in env._events]
        # request_arrival was recorded at t=0 before steps
        assert "request_arrival" in event_types
        assert "assignment" in event_types
        assert "pickup" in event_types
        assert "completion" in event_types
        # Times must be non-decreasing
        times = [e["sim_time"] for e in env._events]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# 2. Busy driver not reassigned
# ---------------------------------------------------------------------------

class TestBusyDriver:
    def test_busy_driver_rejected(self):
        env, req = _env_with_one_request(pickup=(2, 0), dropoff=(5, 0), driver_pos=(0, 0))
        env.matcher = GreedyNearestMatcher()
        env.step()  # assigns d0 -> req, d0 is now ENROUTE_PICKUP

        # Inject a second request
        req2 = env._create_request(pickup=(3, 0), dropoff=(6, 0))

        # Try to assign the busy driver via BusyDriverMatcher
        env.matcher = BusyDriverMatcher(env)
        result = env.step()
        assert result.assignments_made == 0
        assert req2.status == RequestStatus.WAITING

    def test_only_idle_drivers_in_eligible(self):
        env, req = _env_with_one_request(pickup=(2, 0), dropoff=(5, 0), driver_pos=(0, 0))
        env.matcher = GreedyNearestMatcher()
        env.step()  # d0 is now ENROUTE_PICKUP

        ctx = MatchingContext(env.sim_time, env.drivers, env.requests, env.config)
        eligible = ctx.eligible_drivers()
        assert all(d.status == DriverStatus.IDLE for d in eligible)


# ---------------------------------------------------------------------------
# 3. Duplicate assignment prevention
# ---------------------------------------------------------------------------

class TestDuplicateAssignment:
    def test_driver_assigned_twice_rejected(self):
        cfg = SimConfig(n_drivers=2, timestep=60)
        env = MatchingEnv(DoubleMatcher(), config=cfg, seed=0, trips_df=None)
        env.reset()
        env.drivers["d0"].position = (0, 0)
        env.drivers["d1"].position = (9, 9)
        env._create_request(pickup=(0, 1), dropoff=(5, 5))
        env._create_request(pickup=(0, 2), dropoff=(5, 5))
        result = env.step()
        # Only one assignment should succeed for d0
        assert result.assignments_made == 1

    def test_request_assigned_twice_rejected(self):
        cfg = SimConfig(n_drivers=2, timestep=60)
        env = MatchingEnv(DoubleRequestMatcher(), config=cfg, seed=0, trips_df=None)
        env.reset()
        env.drivers["d0"].position = (0, 0)
        env.drivers["d1"].position = (1, 0)
        env._create_request(pickup=(0, 1), dropoff=(5, 5))
        result = env.step()
        # Only one driver should get the request
        assert result.assignments_made == 1


# ---------------------------------------------------------------------------
# 4. Request timeout / expiration
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_request_expires_after_timeout(self):
        cfg = SimConfig(n_drivers=1, timestep=60, request_timeout=120)
        env = MatchingEnv(NullMatcher(), config=cfg, seed=0, trips_df=None)
        env.reset()
        env.drivers["d0"].position = (9, 9)  # far from any request
        req = env._create_request(pickup=(0, 0), dropoff=(1, 1))
        assert req.timeout == 120  # 0 + 120

        env.step()  # t=60: not yet expired
        assert req.status == RequestStatus.WAITING

        env.step()  # t=120: expires
        assert req.status == RequestStatus.EXPIRED

    def test_expired_request_not_assigned(self):
        cfg = SimConfig(n_drivers=1, timestep=60, request_timeout=60)
        env = MatchingEnv(GreedyNearestMatcher(), config=cfg, seed=0, trips_df=None)
        env.reset()
        req = env._create_request(pickup=(0, 0), dropoff=(1, 1))
        # Expire it manually before step
        req.status = RequestStatus.EXPIRED
        result = env.step()
        assert result.assignments_made == 0

    def test_no_timeout_request_never_expires(self):
        cfg = SimConfig(n_drivers=1, timestep=60, request_timeout=None)
        env = MatchingEnv(NullMatcher(), config=cfg, seed=0, trips_df=None)
        env.reset()
        req = env._create_request(pickup=(9, 9), dropoff=(0, 0))
        for _ in range(100):
            env.step()
        assert req.status == RequestStatus.WAITING

    def test_expiration_event_recorded(self):
        cfg = SimConfig(n_drivers=1, timestep=60, request_timeout=60)
        env = MatchingEnv(NullMatcher(), config=cfg, seed=0, trips_df=None)
        env.reset()
        env._create_request(pickup=(9, 9), dropoff=(0, 0))
        env.step()  # t=60: expires
        expiry_events = [e for e in env._events if e["event"] == "expiration"]
        assert len(expiry_events) == 1


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_driver_positions(self):
        env1 = _make_env(seed=99)
        env2 = _make_env(seed=99)
        env1.reset()
        env2.reset()
        assert {d.id: d.position for d in env1.drivers.values()} == \
               {d.id: d.position for d in env2.drivers.values()}

    def test_different_seeds_different_positions(self):
        env1 = _make_env(seed=1)
        env2 = _make_env(seed=2)
        env1.reset()
        env2.reset()
        pos1 = {d.id: d.position for d in env1.drivers.values()}
        pos2 = {d.id: d.position for d in env2.drivers.values()}
        assert pos1 != pos2

    def test_same_seed_same_run_metrics(self):
        trips = make_synthetic_trips(n=20, seed=5)
        cfg = SimConfig(n_drivers=4, timestep=60)
        r1 = MatchingEnv(GreedyNearestMatcher(), config=cfg, seed=5, trips_df=trips).run(30)
        r2 = MatchingEnv(GreedyNearestMatcher(), config=cfg, seed=5, trips_df=trips).run(30)
        assert r1.metrics == r2.metrics

    def test_reset_restores_initial_state(self):
        env = _make_env(seed=7)
        env.reset()
        pos_before = {d.id: d.position for d in env.drivers.values()}
        for _ in range(5):
            env.step()
        env.reset()
        pos_after = {d.id: d.position for d in env.drivers.values()}
        assert pos_before == pos_after


# ---------------------------------------------------------------------------
# 6. Synthetic scenario (no TLC files)
# ---------------------------------------------------------------------------

class TestSyntheticScenario:
    def test_run_completes_without_tlc_files(self):
        trips = make_synthetic_trips(n=15, seed=0)
        cfg = SimConfig(n_drivers=3, timestep=60)
        result = MatchingEnv(GreedyNearestMatcher(), config=cfg, seed=0, trips_df=trips).run(60)
        assert isinstance(result, SimResult)
        assert result.metrics["n_requests_total"] > 0

    def test_null_matcher_zero_completions(self):
        trips = make_synthetic_trips(n=10, seed=0)
        cfg = SimConfig(n_drivers=3, timestep=60)
        result = MatchingEnv(NullMatcher(), config=cfg, seed=0, trips_df=trips).run(60)
        assert result.metrics["n_completed"] == 0

    def test_greedy_matcher_completes_some(self):
        trips = make_synthetic_trips(n=20, seed=0)
        cfg = SimConfig(n_drivers=5, timestep=60)
        result = MatchingEnv(GreedyNearestMatcher(), config=cfg, seed=0, trips_df=trips).run(60)
        assert result.metrics["n_completed"] > 0

    def test_metrics_response_rate_in_range(self):
        trips = make_synthetic_trips(n=20, seed=0)
        cfg = SimConfig(n_drivers=5, timestep=60, request_timeout=300)
        result = MatchingEnv(GreedyNearestMatcher(), config=cfg, seed=0, trips_df=trips).run(60)
        rr = result.metrics["order_response_rate"]
        assert 0.0 <= rr <= 1.0

    def test_spawn_weights_shape_not_required(self):
        """MatchingEnv must work without spawn_weights.npy."""
        trips = make_synthetic_trips(n=5, seed=0)
        cfg = SimConfig(n_drivers=2, timestep=60)
        env = MatchingEnv(NullMatcher(), config=cfg, seed=0, trips_df=trips)
        env.reset()
        env.step()  # must not raise


# ---------------------------------------------------------------------------
# 7. MatchingContext
# ---------------------------------------------------------------------------

class TestMatchingContext:
    def test_eligible_drivers_only_idle(self):
        env, req = _env_with_one_request(pickup=(2, 0), dropoff=(5, 0), driver_pos=(0, 0))
        env.matcher = GreedyNearestMatcher()
        env.step()  # d0 is now ENROUTE_PICKUP
        ctx = MatchingContext(env.sim_time, env.drivers, env.requests, env.config)
        assert all(d.status == DriverStatus.IDLE for d in ctx.eligible_drivers())

    def test_waiting_requests_excludes_assigned(self):
        env, req = _env_with_one_request(pickup=(2, 0), dropoff=(5, 0), driver_pos=(0, 0))
        env.matcher = GreedyNearestMatcher()
        env.step()
        ctx = MatchingContext(env.sim_time, env.drivers, env.requests, env.config)
        assert req not in ctx.waiting_requests()

    def test_manhattan_distance(self):
        assert MatchingContext.manhattan_distance((0, 0), (3, 4)) == 7
        assert MatchingContext.manhattan_distance((5, 5), (5, 5)) == 0
        assert MatchingContext.manhattan_distance((0, 9), (9, 0)) == 18

    def test_travel_time_proportional_to_distance(self):
        cfg = SimConfig(timestep=60, driver_speed=1)
        ctx = MatchingContext(0, {}, {}, cfg)
        assert ctx.travel_time((0, 0), (3, 0)) == 3 * 60
        assert ctx.travel_time((0, 0), (0, 0)) == 0


# ---------------------------------------------------------------------------
# 8. _step_toward utility
# ---------------------------------------------------------------------------

class TestStepToward:
    def test_moves_row_first(self):
        assert _step_toward((0, 0), (3, 3), 1) == (1, 0)

    def test_reaches_target_exactly(self):
        assert _step_toward((2, 2), (2, 2), 1) == (2, 2)

    def test_speed_2_moves_two_cells(self):
        assert _step_toward((0, 0), (5, 5), 2) == (2, 0)

    def test_speed_larger_than_distance_clamps(self):
        assert _step_toward((0, 0), (1, 1), 10) == (1, 1)

    def test_negative_direction(self):
        assert _step_toward((5, 5), (3, 3), 1) == (4, 5)


# ---------------------------------------------------------------------------
# 9. StepResult fields
# ---------------------------------------------------------------------------

class TestStepResult:
    def test_step_returns_step_result(self):
        env = _make_env()
        env.reset()
        result = env.step()
        assert isinstance(result, StepResult)
        assert result.sim_time == env.config.timestep

    def test_sim_time_increments(self):
        env = _make_env()
        env.reset()
        for i in range(1, 4):
            r = env.step()
            assert r.sim_time == i * env.config.timestep

    def test_active_drivers_nonnegative(self):
        env = _make_env()
        env.reset()
        for _ in range(5):
            r = env.step()
            assert r.active_drivers >= 0
            assert r.waiting_requests >= 0


# ---------------------------------------------------------------------------
# 10. make_synthetic_trips
# ---------------------------------------------------------------------------

class TestMakeSyntheticTrips:
    def test_returns_dataframe(self):
        df = make_synthetic_trips(n=10)
        assert len(df) == 10

    def test_required_columns_present(self):
        df = make_synthetic_trips(n=5)
        for col in ("request_time_seconds", "pickup_row", "pickup_col",
                    "dropoff_row", "dropoff_col", "provider"):
            assert col in df.columns

    def test_positions_in_bounds(self):
        df = make_synthetic_trips(n=50, grid_size=10)
        for col in ("pickup_row", "pickup_col", "dropoff_row", "dropoff_col"):
            assert df[col].between(0, 9).all()

    def test_sorted_by_request_time(self):
        df = make_synthetic_trips(n=20)
        assert list(df["request_time_seconds"]) == sorted(df["request_time_seconds"])

    def test_deterministic(self):
        df1 = make_synthetic_trips(n=10, seed=42)
        df2 = make_synthetic_trips(n=10, seed=42)
        assert df1.equals(df2)
