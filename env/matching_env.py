"""
env/matching_env.py
--------------------
Senior Project 2 — discrete-time ride-sharing matching simulation.

This file is independent of env/ride_share_env.py (the SP1 PettingZoo/PPO
environment). It does not import from it and does not affect any existing
PPO experiments.

Architecture overview
---------------------
  SimConfig           -- immutable simulation parameters
  DriverStatus        -- enum: IDLE | ENROUTE_PICKUP | ON_TRIP
  RequestStatus       -- enum: WAITING | ASSIGNED | PICKED_UP | COMPLETED | EXPIRED
  Driver              -- mutable driver state (dataclass)
  PassengerRequest    -- mutable request state (dataclass)
  MatchingContext     -- read-only view passed to matchers each step
  MatchingEnv         -- simulation engine
  StepResult          -- per-step summary returned by step()
  SimResult           -- full run summary returned by run()

Driver positions are simulated (seeded random initial placement).
They are NOT directly observed from TLC data.

Taxi-zone representative points approximate trip positions within a zone.
They are NOT actual GPS pickup/drop-off coordinates.

Matching interface (matching/base.py):
  match(eligible_drivers, waiting_requests, context) -> list[(driver_id, request_id)]
"""

from __future__ import annotations

import sys
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import MANHATTAN_TRIPS, ZONE_GRID_MAP

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimConfig:
    """Immutable simulation parameters."""
    grid_size: int = 10
    n_drivers: int = 20
    timestep: int = 60          # seconds per simulation step
    driver_speed: int = 1       # grid cells per step (Manhattan distance)
    request_timeout: Optional[int] = None  # seconds; None = no timeout


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DriverStatus(Enum):
    IDLE = auto()
    ENROUTE_PICKUP = auto()
    ON_TRIP = auto()


class RequestStatus(Enum):
    WAITING = auto()
    ASSIGNED = auto()
    PICKED_UP = auto()
    COMPLETED = auto()
    EXPIRED = auto()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Driver:
    """
    Mutable driver state.

    position is (row, col) on the grid.
    Driver starting positions are simulated (seeded RNG), not from TLC data.
    """
    id: str
    position: tuple[int, int]
    status: DriverStatus = DriverStatus.IDLE
    assigned_request_id: Optional[str] = None
    status_since: int = 0           # sim_time when status last changed


@dataclass
class PassengerRequest:
    """
    Mutable passenger request state.

    pickup and dropoff are (row, col) grid cells derived from taxi-zone
    representative points — they approximate zone centers, not GPS coordinates.
    """
    id: str
    provider: str
    pickup: tuple[int, int]
    dropoff: tuple[int, int]
    request_time: int               # sim_time (seconds) when request arrived
    status: RequestStatus = RequestStatus.WAITING
    assigned_driver_id: Optional[str] = None
    assignment_time: Optional[int] = None
    actual_simulated_pickup_time: Optional[int] = None
    completion_time: Optional[int] = None
    timeout: Optional[int] = None   # absolute sim_time when request expires


# ---------------------------------------------------------------------------
# Matching context (read-only view for matchers)
# ---------------------------------------------------------------------------

class MatchingContext:
    """
    Read-only snapshot of simulation state passed to matchers.

    Matchers must not modify drivers or requests directly.
    """

    def __init__(
        self,
        sim_time: int,
        drivers: dict[str, Driver],
        requests: dict[str, PassengerRequest],
        config: SimConfig,
    ) -> None:
        self._sim_time = sim_time
        self._drivers = drivers
        self._requests = requests
        self.config = config

    @property
    def sim_time(self) -> int:
        return self._sim_time

    def eligible_drivers(self) -> list[Driver]:
        """Return idle drivers available for assignment."""
        return [d for d in self._drivers.values() if d.status == DriverStatus.IDLE]

    def waiting_requests(self) -> list[PassengerRequest]:
        """Return unassigned, non-expired requests."""
        return [
            r for r in self._requests.values()
            if r.status == RequestStatus.WAITING
        ]

    @staticmethod
    def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        """Manhattan (L1) distance between two grid cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def travel_time(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Estimated travel time in seconds between two grid cells."""
        steps = -(-self.manhattan_distance(a, b) // self.config.driver_speed)  # ceil div
        return steps * self.config.timestep


# ---------------------------------------------------------------------------
# Step and run results
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    sim_time: int
    assignments_made: int
    pickups: int
    completions: int
    expirations: int
    active_drivers: int
    waiting_requests: int


@dataclass
class SimResult:
    events: list[dict]
    metrics: dict
    config: SimConfig


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

class MatchingEnv:
    """
    Discrete-time ride-sharing matching simulation.

    Each step:
      1. Inject new requests whose request_time <= sim_time
      2. Expire timed-out waiting requests
      3. Call matcher.match(eligible, waiting, context) -> assignments
      4. Validate and apply assignments
      5. Move each driver one step toward its target
      6. Process arrivals (pickup / drop-off)
      7. Record events and return StepResult

    Args:
        matcher:   Any object with match(eligible, waiting, context) method.
        config:    SimConfig (defaults to SimConfig()).
        seed:      Integer seed for reproducible driver placement and synthetic
                   request generation. Driver positions are simulated, not from data.
        trips_df:  Optional DataFrame with columns [request_time_seconds,
                   pickup_row, pickup_col, dropoff_row, dropoff_col, provider].
                   If None, a tiny synthetic scenario is used (no TLC files needed).
    """

    def __init__(
        self,
        matcher,
        config: SimConfig = SimConfig(),
        seed: int = 42,
        trips_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.matcher = matcher
        self.config = config
        self._seed = seed
        self._trips_df = trips_df
        self._rng = np.random.default_rng(seed)

        self.sim_time: int = 0
        self.drivers: dict[str, Driver] = {}
        self.requests: dict[str, PassengerRequest] = {}
        self._pending_trips: list[dict] = []   # sorted by request_time_seconds
        self._events: list[dict] = []
        self._request_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset simulation to t=0. Reproducible with the same seed."""
        self._rng = np.random.default_rng(self._seed)
        self.sim_time = 0
        self._events = []
        self._request_counter = 0

        # Driver starting positions are simulated (seeded RNG).
        # They are NOT directly observed from TLC data.
        self.drivers = {
            f"d{i}": Driver(
                id=f"d{i}",
                position=(
                    int(self._rng.integers(0, self.config.grid_size)),
                    int(self._rng.integers(0, self.config.grid_size)),
                ),
            )
            for i in range(self.config.n_drivers)
        }

        self.requests = {}
        self._pending_trips = self._prepare_pending_trips()

    def step(self) -> StepResult:
        """Advance simulation by one timestep. Call reset() first."""
        self.sim_time += self.config.timestep

        # 1. Inject new requests
        self._inject_requests()

        # 2. Expire timed-out requests
        n_expired = self._expire_requests()

        # 3. Match
        context = MatchingContext(self.sim_time, self.drivers, self.requests, self.config)
        eligible = context.eligible_drivers()
        waiting = context.waiting_requests()

        raw_assignments = self.matcher.match(eligible, waiting, context)
        n_assigned = self._apply_assignments(raw_assignments)

        # 4. Move drivers and process arrivals
        n_pickups, n_completions = self._move_and_process()

        result = StepResult(
            sim_time=self.sim_time,
            assignments_made=n_assigned,
            pickups=n_pickups,
            completions=n_completions,
            expirations=n_expired,
            active_drivers=sum(
                1 for d in self.drivers.values() if d.status != DriverStatus.IDLE
            ),
            waiting_requests=sum(
                1 for r in self.requests.values()
                if r.status == RequestStatus.WAITING
            ),
        )
        return result

    def run(self, n_steps: int) -> SimResult:
        """Run for n_steps steps and return a SimResult with metrics."""
        self.reset()
        for _ in range(n_steps):
            self.step()
        return SimResult(
            events=list(self._events),
            metrics=self.metrics(),
            config=self.config,
        )

    def metrics(self) -> dict:
        """Compute KPIs from the event log."""
        completed = [
            r for r in self.requests.values()
            if r.status == RequestStatus.COMPLETED
        ]
        all_terminal = [
            r for r in self.requests.values()
            if r.status in (RequestStatus.COMPLETED, RequestStatus.EXPIRED)
        ]
        arrived = [
            r for r in self.requests.values()
            if r.status != RequestStatus.WAITING  # at least assigned
        ]

        # Average waiting time: request_time -> actual_simulated_pickup_time
        wait_times = [
            r.actual_simulated_pickup_time - r.request_time
            for r in completed
            if r.actual_simulated_pickup_time is not None
        ]
        avg_wait = float(np.mean(wait_times)) if wait_times else float("nan")

        # Order response rate: completed / (completed + expired)
        n_completed = len(completed)
        n_expired = sum(1 for r in self.requests.values() if r.status == RequestStatus.EXPIRED)
        total_terminal = n_completed + n_expired
        response_rate = n_completed / total_terminal if total_terminal > 0 else float("nan")

        # Vehicle utilization: fraction of driver-steps spent ON_TRIP
        on_trip_events = [e for e in self._events if e["event"] == "pickup"]
        completion_events = [e for e in self._events if e["event"] == "completion"]
        total_on_trip_steps = sum(
            (e["sim_time"] - _find_pickup_time(e["request_id"], on_trip_events))
            for e in completion_events
            if _find_pickup_time(e["request_id"], on_trip_events) is not None
        )
        total_driver_time = self.config.n_drivers * self.sim_time
        utilization = (
            total_on_trip_steps / total_driver_time if total_driver_time > 0 else float("nan")
        )

        # Average detour ratio: actual trip distance / direct pickup->dropoff distance
        detour_ratios = []
        for r in completed:
            direct = MatchingContext.manhattan_distance(r.pickup, r.dropoff)
            if direct > 0 and r.assigned_driver_id is not None:
                assign_ev = next(
                    (e for e in self._events
                     if e["event"] == "assignment" and e["request_id"] == r.id),
                    None,
                )
                if assign_ev is not None:
                    driver_pos = assign_ev.get("driver_position")
                    if driver_pos is not None:
                        actual = (
                            MatchingContext.manhattan_distance(driver_pos, r.pickup) + direct
                        )
                        detour_ratios.append(actual / direct)
        avg_detour = float(np.mean(detour_ratios)) if detour_ratios else float("nan")

        return {
            "avg_wait_seconds": avg_wait,
            "order_response_rate": response_rate,
            "vehicle_utilization": utilization,
            "avg_detour_ratio": avg_detour,
            "n_completed": n_completed,
            "n_expired": n_expired,
            "n_requests_total": len(self.requests),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_pending_trips(self) -> list[dict]:
        """Convert trips_df to a sorted list of pending trip dicts."""
        if self._trips_df is None:
            return []
        required = {"request_time_seconds", "pickup_row", "pickup_col",
                    "dropoff_row", "dropoff_col"}
        missing = required - set(self._trips_df.columns)
        if missing:
            raise ValueError(f"trips_df missing columns: {missing}")
        trips = self._trips_df.to_dict("records")
        trips.sort(key=lambda t: t["request_time_seconds"])
        return trips

    def _inject_requests(self) -> None:
        """Inject all pending trips whose request_time <= sim_time."""
        while self._pending_trips and \
                self._pending_trips[0]["request_time_seconds"] <= self.sim_time:
            t = self._pending_trips.pop(0)
            self._create_request(
                pickup=(int(t["pickup_row"]), int(t["pickup_col"])),
                dropoff=(int(t["dropoff_row"]), int(t["dropoff_col"])),
                provider=str(t.get("provider", "unknown")),
            )

    def _create_request(
        self,
        pickup: tuple[int, int],
        dropoff: tuple[int, int],
        provider: str = "synthetic",
    ) -> PassengerRequest:
        self._request_counter += 1
        rid = f"r{self._request_counter}"
        timeout = (
            self.sim_time + self.config.request_timeout
            if self.config.request_timeout is not None
            else None
        )
        req = PassengerRequest(
            id=rid,
            provider=provider,
            pickup=pickup,
            dropoff=dropoff,
            request_time=self.sim_time,
            timeout=timeout,
        )
        self.requests[rid] = req
        self._record("request_arrival", request_id=rid,
                     pickup=pickup, dropoff=dropoff, provider=provider)
        return req

    def _expire_requests(self) -> int:
        """Expire WAITING requests that have passed their timeout."""
        n = 0
        for req in self.requests.values():
            if (req.status == RequestStatus.WAITING
                    and req.timeout is not None
                    and self.sim_time >= req.timeout):
                req.status = RequestStatus.EXPIRED
                self._record("expiration", request_id=req.id)
                n += 1
        return n

    def _apply_assignments(self, raw: list[tuple[str, str]]) -> int:
        """
        Validate and apply matcher assignments.

        Rules enforced:
          - Driver must be IDLE
          - Request must be WAITING
          - No driver or request assigned twice in the same batch
        Invalid pairs are silently skipped with a warning.
        """
        assigned_drivers: set[str] = set()
        assigned_requests: set[str] = set()
        n = 0

        for driver_id, request_id in (raw or []):
            driver = self.drivers.get(driver_id)
            req = self.requests.get(request_id)

            if driver is None:
                log.warning("Assignment ignored: unknown driver %s", driver_id)
                continue
            if req is None:
                log.warning("Assignment ignored: unknown request %s", request_id)
                continue
            if driver.status != DriverStatus.IDLE:
                log.warning("Assignment ignored: driver %s is not IDLE (%s)",
                            driver_id, driver.status)
                continue
            if req.status != RequestStatus.WAITING:
                log.warning("Assignment ignored: request %s is not WAITING (%s)",
                            request_id, req.status)
                continue
            if driver_id in assigned_drivers:
                log.warning("Assignment ignored: driver %s assigned twice in one step", driver_id)
                continue
            if request_id in assigned_requests:
                log.warning("Assignment ignored: request %s assigned twice in one step", request_id)
                continue

            # Apply
            driver.status = DriverStatus.ENROUTE_PICKUP
            driver.assigned_request_id = request_id
            driver.status_since = self.sim_time
            req.status = RequestStatus.ASSIGNED
            req.assigned_driver_id = driver_id
            req.assignment_time = self.sim_time

            assigned_drivers.add(driver_id)
            assigned_requests.add(request_id)
            n += 1

            self._record("assignment",
                         driver_id=driver_id, request_id=request_id,
                         driver_position=driver.position,
                         pickup=req.pickup)
        return n

    def _move_and_process(self) -> tuple[int, int]:
        """
        Move each non-idle driver one step toward its target.
        Process pickups and drop-offs that occur this step.
        Returns (n_pickups, n_completions).
        """
        n_pickups = 0
        n_completions = 0

        for driver in self.drivers.values():
            if driver.status == DriverStatus.IDLE:
                continue

            req = self.requests.get(driver.assigned_request_id)
            if req is None:
                # Orphaned driver — reset to idle (should not happen)
                driver.status = DriverStatus.IDLE
                driver.assigned_request_id = None
                continue

            if driver.status == DriverStatus.ENROUTE_PICKUP:
                target = req.pickup
                driver.position = _step_toward(
                    driver.position, target, self.config.driver_speed
                )
                if driver.position == target:
                    # Pickup
                    driver.status = DriverStatus.ON_TRIP
                    driver.status_since = self.sim_time
                    req.status = RequestStatus.PICKED_UP
                    req.actual_simulated_pickup_time = self.sim_time
                    self._record("pickup",
                                 driver_id=driver.id, request_id=req.id,
                                 position=driver.position)
                    n_pickups += 1
                    # Zero-distance trip: pickup == dropoff, complete immediately
                    if driver.position == req.dropoff:
                        driver.status = DriverStatus.IDLE
                        driver.assigned_request_id = None
                        driver.status_since = self.sim_time
                        req.status = RequestStatus.COMPLETED
                        req.completion_time = self.sim_time
                        self._record("completion",
                                     driver_id=driver.id, request_id=req.id,
                                     position=driver.position)
                        n_completions += 1

            elif driver.status == DriverStatus.ON_TRIP:
                target = req.dropoff
                driver.position = _step_toward(
                    driver.position, target, self.config.driver_speed
                )
                if driver.position == target:
                    # Drop-off
                    driver.status = DriverStatus.IDLE
                    driver.assigned_request_id = None
                    driver.status_since = self.sim_time
                    req.status = RequestStatus.COMPLETED
                    req.completion_time = self.sim_time
                    self._record("completion",
                                 driver_id=driver.id, request_id=req.id,
                                 position=driver.position)
                    n_completions += 1

        return n_pickups, n_completions

    def _record(self, event: str, **kwargs) -> None:
        self._events.append({"event": event, "sim_time": self.sim_time, **kwargs})


# ---------------------------------------------------------------------------
# Movement utility
# ---------------------------------------------------------------------------

def _step_toward(
    pos: tuple[int, int],
    target: tuple[int, int],
    speed: int,
) -> tuple[int, int]:
    """
    Move pos toward target by up to `speed` steps (Manhattan).
    Moves row-first, then column. Returns new position.
    """
    row, col = pos
    tr, tc = target
    remaining = speed

    dr = tr - row
    if dr != 0:
        move = min(abs(dr), remaining) * (1 if dr > 0 else -1)
        row += move
        remaining -= abs(move)

    if remaining > 0:
        dc = tc - col
        if dc != 0:
            move = min(abs(dc), remaining) * (1 if dc > 0 else -1)
            col += move

    return (row, col)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _find_pickup_time(request_id: str, pickup_events: list[dict]) -> Optional[int]:
    for e in pickup_events:
        if e["request_id"] == request_id:
            return e["sim_time"]
    return None


# ---------------------------------------------------------------------------
# Dataset loader (optional — requires processed TLC data)
# ---------------------------------------------------------------------------

def load_trips_for_simulation(
    trips_path: Path = MANHATTAN_TRIPS,
    zone_map_path: Path = ZONE_GRID_MAP,
    sim_start: pd.Timestamp = pd.Timestamp("2023-01-09 09:00:00"),  # Monday 9am
    sim_duration_seconds: int = 3600,
    max_requests: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a window of TLC trips and convert to simulation request format.

    Returns a DataFrame with columns:
        request_time_seconds  -- seconds since sim_start (int)
        pickup_row, pickup_col, dropoff_row, dropoff_col  -- grid cells (int)
        provider              -- str

    Taxi-zone representative points approximate trip positions within a zone.
    They are NOT actual GPS pickup/drop-off coordinates.

    Requires:
        data/processed/manhattan_trips.parquet
        data/processed/zone_grid_map.json
    """
    import json

    if not trips_path.exists():
        raise FileNotFoundError(
            f"Processed trips not found: {trips_path}\n"
            "Run: python ride-share-marl/pipeline/load_tlc_data.py"
        )
    if not zone_map_path.exists():
        raise FileNotFoundError(
            f"Zone grid map not found: {zone_map_path}\n"
            "Run: python ride-share-marl/pipeline/zone_grid.py"
        )

    with open(zone_map_path) as f:
        raw_map = json.load(f)
    zone_map = {int(k): tuple(v) for k, v in raw_map.items()}

    df = pd.read_parquet(trips_path)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    sim_end = sim_start + pd.Timedelta(seconds=sim_duration_seconds)
    window = df[
        (df["tpep_pickup_datetime"] >= sim_start) &
        (df["tpep_pickup_datetime"] < sim_end)
    ].copy()

    if max_requests is not None:
        window = window.head(max_requests)

    rows = []
    for _, trip in window.iterrows():
        pu_id = int(trip["PULocationID"])
        do_id = int(trip["DOLocationID"])
        if pu_id not in zone_map or do_id not in zone_map:
            continue
        pu_r, pu_c = zone_map[pu_id]
        do_r, do_c = zone_map[do_id]
        t_sec = int(
            (trip["tpep_pickup_datetime"] - sim_start).total_seconds()
        )
        rows.append({
            "request_time_seconds": max(0, t_sec),
            "pickup_row": pu_r,
            "pickup_col": pu_c,
            "dropoff_row": do_r,
            "dropoff_col": do_c,
            "provider": "yellow",
        })

    return pd.DataFrame(rows).sort_values("request_time_seconds").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic scenario (no TLC files required)
# ---------------------------------------------------------------------------

def make_synthetic_trips(
    n: int = 30,
    grid_size: int = 10,
    duration_seconds: int = 3600,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate a tiny synthetic trip table for testing without TLC files.

    Driver positions are simulated; trip positions are random grid cells.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "request_time_seconds": int(rng.integers(0, duration_seconds)),
            "pickup_row": int(rng.integers(0, grid_size)),
            "pickup_col": int(rng.integers(0, grid_size)),
            "dropoff_row": int(rng.integers(0, grid_size)),
            "dropoff_col": int(rng.integers(0, grid_size)),
            "provider": "synthetic",
        })
    if not rows:
        return pd.DataFrame(columns=[
            "request_time_seconds", "pickup_row", "pickup_col",
            "dropoff_row", "dropoff_col", "provider",
        ])
    return pd.DataFrame(rows).sort_values("request_time_seconds").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI — sample simulation trace
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from matching.base import BaseMatcher

    class NearestMatcher(BaseMatcher):
        """Greedy nearest-driver test matcher (not the SP2 Greedy algorithm)."""
        def match(self, eligible_drivers, waiting_requests, context):
            assignments = []
            used_requests = set()
            for driver in eligible_drivers:
                best_req = None
                best_dist = float("inf")
                for req in waiting_requests:
                    if req.id in used_requests:
                        continue
                    d = context.manhattan_distance(driver.position, req.pickup)
                    if d < best_dist:
                        best_dist = d
                        best_req = req
                if best_req is not None:
                    assignments.append((driver.id, best_req.id))
                    used_requests.add(best_req.id)
            return assignments

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = SimConfig(n_drivers=5, timestep=60, request_timeout=600)
    trips = make_synthetic_trips(n=20, seed=7)
    env = MatchingEnv(NearestMatcher(), config=cfg, seed=42, trips_df=trips)

    print("=== Sample Simulation Trace (5 drivers, 20 synthetic requests, 60 steps) ===\n")
    env.reset()
    for step_num in range(1, 61):
        result = env.step()
        if any([result.assignments_made, result.pickups,
                result.completions, result.expirations]):
            print(
                f"t={result.sim_time:>5}s | "
                f"assigned={result.assignments_made} "
                f"pickups={result.pickups} "
                f"completions={result.completions} "
                f"expired={result.expirations} | "
                f"active={result.active_drivers} "
                f"waiting={result.waiting_requests}"
            )

    m = env.metrics()
    print("\n=== Metrics ===")
    print(f"  Completed:          {m['n_completed']}")
    print(f"  Expired:            {m['n_expired']}")
    print(f"  Total requests:     {m['n_requests_total']}")
    print(f"  Avg wait:           {m['avg_wait_seconds']:.1f}s")
    print(f"  Response rate:      {m['order_response_rate']:.2%}")
    print(f"  Vehicle util:       {m['vehicle_utilization']:.2%}")
    print(f"  Avg detour ratio:   {m['avg_detour_ratio']:.2f}")
