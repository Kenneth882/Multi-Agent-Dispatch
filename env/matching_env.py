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
import tracemalloc
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from math import nan
from pathlib import Path
from time import perf_counter
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
    request_time: float             # elapsed seconds when the request arrived
    status: RequestStatus = RequestStatus.WAITING
    assigned_driver_id: Optional[str] = None
    assignment_time: Optional[int] = None
    actual_simulated_pickup_time: Optional[int] = None
    completion_time: Optional[int] = None
    timeout: Optional[float] = None  # absolute sim_time when request expires


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
        self.candidate_pairs_considered = 0
        self.matcher_metrics: Counter[str] = Counter()

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

    def pickup_distance(self, driver: Driver, request: PassengerRequest) -> int:
        """Record one driver-request evaluation and return pickup distance."""
        self.candidate_pairs_considered += 1
        return self.manhattan_distance(driver.position, request.pickup)


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
    performance: dict = field(default_factory=dict)


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
                   If None, no arrivals are scheduled. Use make_synthetic_trips
                   to generate a scenario without TLC data.
        profile_matching_memory: Trace allocations during each matcher call.
    """

    def __init__(
        self,
        matcher,
        config: SimConfig = SimConfig(),
        seed: int = 42,
        trips_df: Optional[pd.DataFrame] = None,
        *,
        profile_matching_memory: bool = False,
    ) -> None:
        self.matcher = matcher
        self.config = config
        self._seed = seed
        self._trips_df = trips_df
        self._rng = np.random.default_rng(seed)
        self._profile_matching_memory = profile_matching_memory

        self.sim_time: int = 0
        self.drivers: dict[str, Driver] = {}
        self.requests: dict[str, PassengerRequest] = {}
        self._pending_trips: deque[dict] = deque()
        self._events: list[dict] = []
        self._request_counter: int = 0
        self._candidate_pairs = 0
        self._matcher_metrics: Counter[str] = Counter()
        self._matching_calls = 0
        self._matching_seconds = 0.0
        self._simulation_seconds = 0.0
        self._profiled_calls = 0
        self._matching_peak_bytes = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset simulation to t=0. Reproducible with the same seed."""
        started = perf_counter()
        self._rng = np.random.default_rng(self._seed)
        self.sim_time = 0
        self._events = []
        self._request_counter = 0
        self._candidate_pairs = 0
        self._matcher_metrics.clear()
        self._matching_calls = 0
        self._matching_seconds = 0.0
        self._profiled_calls = 0
        self._matching_peak_bytes = 0

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
        self._inject_requests()
        self._simulation_seconds = perf_counter() - started

    def step(self) -> StepResult:
        """Advance simulation by one timestep. Call reset() first."""
        started = perf_counter()
        self.sim_time += self.config.timestep

        # 1. Inject new requests
        self._inject_requests()

        # 2. Expire timed-out requests
        n_expired = self._expire_requests()

        # 3. Match
        context = MatchingContext(self.sim_time, self.drivers, self.requests, self.config)
        eligible = context.eligible_drivers()
        waiting = context.waiting_requests()

        raw_assignments = self._measure_match(eligible, waiting, context)
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
        self._simulation_seconds += perf_counter() - started
        return result

    def run(self, n_steps: int) -> SimResult:
        """Run for n_steps steps and return a SimResult with metrics."""
        started = perf_counter()
        self.reset()
        for _ in range(n_steps):
            self.step()
        self._simulation_seconds = perf_counter() - started
        return SimResult(
            events=list(self._events),
            metrics=self.metrics(),
            config=self.config,
            performance=self.performance_metrics(),
        )

    def performance_metrics(self) -> dict:
        """Wall-clock timings and optional traced allocation peak."""
        return {
            "total_matching_execution_seconds": self._matching_seconds,
            "avg_matching_execution_seconds_per_timestep": (
                self._matching_seconds / self._matching_calls if self._matching_calls else nan
            ),
            "total_simulation_execution_seconds": self._simulation_seconds,
            "peak_matching_memory_bytes": (
                self._matching_peak_bytes
                if self._matching_calls and self._profiled_calls == self._matching_calls
                else None
            ),
            "matching_memory_profiled_timesteps": self._profiled_calls,
        }

    def _measure_match(self, eligible, waiting, context):
        # tracemalloc is process-wide. Leave an existing tracing session alone.
        trace = self._profile_matching_memory and not tracemalloc.is_tracing()
        if trace:
            tracemalloc.start()
        started = perf_counter()
        try:
            return self.matcher.match(eligible, waiting, context)
        finally:
            elapsed = perf_counter() - started
            if trace:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self._matching_peak_bytes = max(self._matching_peak_bytes, peak)
                self._profiled_calls += 1
            self._matching_seconds += elapsed
            self._matching_calls += 1
            self._candidate_pairs += context.candidate_pairs_considered
            self._matcher_metrics.update(context.matcher_metrics)

    def metrics(self) -> dict:
        """Compute deterministic outcome and workload metrics."""
        requests = self.requests.values()
        assigned = [r for r in requests if r.assignment_time is not None]
        picked_up = [r for r in requests if r.actual_simulated_pickup_time is not None]
        completed = [
            r for r in requests
            if r.status == RequestStatus.COMPLETED
        ]
        n_requests = len(requests)
        n_completed = len(completed)
        n_expired = sum(r.status == RequestStatus.EXPIRED for r in requests)

        def ratio(numerator, denominator):
            return numerator / denominator if denominator else nan

        total_on_trip_time = sum(
            (r.completion_time if r.completion_time is not None else self.sim_time)
            - r.actual_simulated_pickup_time
            for r in picked_up
        )
        total_enroute_time = sum(
            (r.actual_simulated_pickup_time if r.actual_simulated_pickup_time is not None else self.sim_time)
            - r.assignment_time
            for r in assigned
        )
        total_driver_time = self.config.n_drivers * self.sim_time

        pickup_distances = {
            e["request_id"]: MatchingContext.manhattan_distance(e["driver_position"], e["pickup"])
            for e in self._events if e["event"] == "assignment"
        }
        completed_wait = []
        detour_ratios = []
        for r in completed:
            if r.actual_simulated_pickup_time is not None:
                completed_wait.append(r.actual_simulated_pickup_time - r.request_time)
            direct = MatchingContext.manhattan_distance(r.pickup, r.dropoff)
            if direct > 0 and r.id in pickup_distances:
                detour_ratios.append((pickup_distances[r.id] + direct) / direct)
        traveled_pickup_distances = [pickup_distances[r.id] for r in picked_up if r.id in pickup_distances]

        return {
            **self._matcher_metrics,
            "n_requests_total": n_requests,
            "n_assignments": len(assigned),
            "n_pickups": len(picked_up),
            "n_completed": n_completed,
            "n_expired": n_expired,
            "n_unfinished": n_requests - n_completed - n_expired,
            "assignment_rate": ratio(len(assigned), n_requests),
            "pickup_rate": ratio(len(picked_up), n_requests),
            "trip_completion_rate": ratio(n_completed, n_requests),
            "avg_request_to_assignment_seconds": ratio(
                sum(r.assignment_time - r.request_time for r in assigned), len(assigned)
            ),
            "avg_request_to_pickup_seconds": ratio(
                sum(r.actual_simulated_pickup_time - r.request_time for r in picked_up), len(picked_up)
            ),
            "avg_assignment_to_pickup_seconds": ratio(
                sum(r.actual_simulated_pickup_time - r.assignment_time for r in picked_up), len(picked_up)
            ),
            "avg_pickup_distance_cells": ratio(sum(traveled_pickup_distances), len(traveled_pickup_distances)),
            "vehicle_utilization": ratio(total_on_trip_time, total_driver_time),
            "avg_idle_seconds_per_driver": ratio(
                total_driver_time - total_on_trip_time - total_enroute_time, self.config.n_drivers
            ),
            "avg_completed_trips_per_driver": ratio(n_completed, self.config.n_drivers),
            "candidate_pairs_considered": self._candidate_pairs,
            "avg_wait_seconds": ratio(sum(completed_wait), len(completed_wait)),
            "order_response_rate": ratio(n_completed, n_completed + n_expired),
            "avg_detour_ratio": ratio(sum(detour_ratios), len(detour_ratios)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_pending_trips(self) -> deque[dict]:
        """Queue pending trips in chronological order."""
        if self._trips_df is None:
            return deque()
        required = {"request_time_seconds", "pickup_row", "pickup_col",
                    "dropoff_row", "dropoff_col"}
        missing = required - set(self._trips_df.columns)
        if missing:
            raise ValueError(f"trips_df missing columns: {missing}")
        trips = self._trips_df.to_dict("records")
        trips.sort(key=lambda t: t["request_time_seconds"])
        return deque(trips)

    def _inject_requests(self) -> None:
        """Inject all pending trips whose request_time <= sim_time."""
        while self._pending_trips and \
                self._pending_trips[0]["request_time_seconds"] <= self.sim_time:
            t = self._pending_trips.popleft()
            self._create_request(
                pickup=(int(t["pickup_row"]), int(t["pickup_col"])),
                dropoff=(int(t["dropoff_row"]), int(t["dropoff_col"])),
                provider=str(t.get("provider", "unknown")),
                request_time=float(t["request_time_seconds"]),
            )

    def _create_request(
        self,
        pickup: tuple[int, int],
        dropoff: tuple[int, int],
        provider: str = "synthetic",
        request_time: Optional[float] = None,
    ) -> PassengerRequest:
        if request_time is None:
            request_time = self.sim_time
        self._request_counter += 1
        rid = f"r{self._request_counter}"
        timeout = (
            request_time + self.config.request_timeout
            if self.config.request_timeout is not None
            else None
        )
        req = PassengerRequest(
            id=rid,
            provider=provider,
            pickup=pickup,
            dropoff=dropoff,
            request_time=request_time,
            timeout=timeout,
        )
        self.requests[rid] = req
        self._record("request_arrival", request_id=rid,
                     pickup=pickup, dropoff=dropoff, provider=provider,
                     request_time=request_time)
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
                if driver.status_since == self.sim_time and driver.position != target:
                    continue  # travel starts after the assignment boundary
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
# Dataset loader (optional — requires processed TLC data)
# ---------------------------------------------------------------------------

def load_trips_for_simulation(
    trips_path: Path = MANHATTAN_TRIPS,
    zone_map_path: Path = ZONE_GRID_MAP,
    sim_start: Optional[pd.Timestamp] = None,
    sim_duration_seconds: Optional[int] = None,
    max_requests: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a window of TLC trips and convert to simulation request format.

    Returns a DataFrame with columns:
        request_time_seconds  -- elapsed seconds since the scenario or window start
        pickup_row, pickup_col, dropoff_row, dropoff_col  -- grid cells (int)
        provider              -- str

    Taxi-zone representative points approximate trip positions within a zone.
    They are NOT actual GPS pickup/drop-off coordinates.

    Accepts prepared scenarios, unified TLC parquet, and legacy yellow parquet.
    Prepared scenarios carry relative offsets that identify the original window.
    Other files default to Jan 9 at 09:00 for one hour. Only yellow records
    can use pickup time as an arrival proxy when request time is unavailable.
    """
    import json

    if not trips_path.exists():
        raise FileNotFoundError(
            f"Processed trips not found: {trips_path}\n"
            "Run load_tlc_data.py for yellow-only data or tlc_pipeline.py for unified data."
        )
    if not zone_map_path.exists():
        raise FileNotFoundError(
            f"Zone grid map not found: {zone_map_path}\n"
            "Run: python ride-share-marl/pipeline/zone_grid.py"
        )

    with open(zone_map_path) as f:
        raw_map = json.load(f)
    zone_map = {
        int(k): tuple(v) for k, v in raw_map.items()
        if isinstance(v, list) and len(v) == 2
        and all(type(cell) is int and 0 <= cell < 10 for cell in v)
    }

    df = pd.read_parquet(trips_path)
    prepared = {"request_datetime", "request_time_seconds"}.issubset(df.columns)
    if "request_datetime" in df.columns:
        request_at = pd.to_datetime(df["request_datetime"], errors="coerce", format="mixed")
        if "pickup_datetime" in df.columns:
            yellow_pickup = pd.to_datetime(df["pickup_datetime"], errors="coerce", format="mixed")
            request_at = request_at.fillna(yellow_pickup.where(df["provider"].eq("yellow")))
    else:
        request_at = pd.to_datetime(df["tpep_pickup_datetime"])
        df["provider"] = "yellow"

    if sim_start is None:
        sim_start = pd.Timestamp("2023-01-09 09:00:00")
        if prepared:
            offsets = pd.to_numeric(df["request_time_seconds"], errors="coerce")
            origins = (
                request_at.dt.floor("s") - pd.to_timedelta(np.floor(offsets), unit="s")
            ).dropna().unique()
            if len(origins) > 1:
                raise ValueError("Prepared scenario has inconsistent request time offsets")
            if len(origins) == 1:
                sim_start = origins[0]
    sim_start = pd.Timestamp(sim_start)
    if sim_duration_seconds is None and not prepared:
        sim_duration_seconds = 3600
    in_window = request_at >= sim_start
    if sim_duration_seconds is not None:
        sim_end = sim_start + pd.Timedelta(seconds=sim_duration_seconds)
        in_window &= request_at < sim_end
    mapped = df["PULocationID"].isin(zone_map) & df["DOLocationID"].isin(zone_map)
    eligible = in_window & mapped
    window = df.loc[eligible].assign(arrival_time=request_at.loc[eligible])
    window = window.sort_values("arrival_time", kind="stable")

    if max_requests is not None:
        window = window.head(max_requests)

    rows = []
    for _, trip in window.iterrows():
        pu_id = int(trip["PULocationID"])
        do_id = int(trip["DOLocationID"])
        pu_r, pu_c = zone_map[pu_id]
        do_r, do_c = zone_map[do_id]
        t_sec = (trip["arrival_time"] - sim_start) / pd.Timedelta(seconds=1)
        rows.append({
            "request_time_seconds": max(0, t_sec),
            "pickup_row": pu_r,
            "pickup_col": pu_c,
            "dropoff_row": do_r,
            "dropoff_col": do_c,
            "provider": trip["provider"],
        })

    return pd.DataFrame(rows, columns=[
        "request_time_seconds", "pickup_row", "pickup_col",
        "dropoff_row", "dropoff_col", "provider",
    ])


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
