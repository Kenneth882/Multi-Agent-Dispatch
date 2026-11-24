import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from env.matching_env import (
    DriverStatus, MatchingEnv, RequestStatus, SimConfig, load_trips_for_simulation,
)
from prepare_scenarios import DAY, WINDOWS, prepare_scenario


def test_prepared_scenario_lifecycle(tmp_path):
    start = DAY + pd.Timedelta(hours=17)
    timestamps = start + pd.to_timedelta([0, 60.5, 3660], unit="s")
    requests = pd.DataFrame({
        "request_datetime": timestamps,
        "pickup_datetime": timestamps + pd.Timedelta(minutes=35),
        "dropoff_datetime": timestamps + pd.Timedelta(minutes=70),
        "PULocationID": [1, 2, 2],
        "DOLocationID": [2, 1, 2],
        "provider": ["uber", "lyft", "lyft"],
        "pickup_row": [9, 9, 9],
    })
    scenario = prepare_scenario(requests, WINDOWS["peak"], seed=42)
    path = tmp_path / "peak.parquet"
    scenario.to_parquet(path, index=False)
    zones = tmp_path / "map.json"
    zones.write_text(json.dumps({1: [2, 0], 2: [2, 2]}), encoding="utf-8")
    trips = load_trips_for_simulation(path, zones)
    assert trips["request_time_seconds"].tolist() == [0, 60.5, 3660]
    assert trips["pickup_row"].tolist() == [2, 2, 2]

    class RepeatAssignments:
        def match(self, drivers, requests, context):
            if not requests:
                return []
            assignment = ("d0", requests[0].id)
            return [assignment, assignment]

    env = MatchingEnv(RepeatAssignments(), SimConfig(n_drivers=1, request_timeout=120), trips_df=trips)
    env.reset()
    driver = env.drivers["d0"]
    driver.position = (0, 0)
    first = env.requests["r1"]
    assert driver.status == DriverStatus.IDLE
    assert first.status == RequestStatus.WAITING
    assert len(env.requests) == 1

    step = env.step()  # 60 s
    assert step.assignments_made == 1
    assert len(env.requests) == 1
    assert first.status == RequestStatus.ASSIGNED
    assert driver.status == DriverStatus.ENROUTE_PICKUP
    assert driver.position == (0, 0)

    step = env.step()  # 120 s
    second = env.requests["r2"]
    assert second.request_time == 60.5
    assert second.status == RequestStatus.WAITING
    assert step.assignments_made == 0
    assert driver.position == (1, 0)
    assert env.step().pickups == 1  # 180 s
    assert driver.status == DriverStatus.ON_TRIP
    assert first.status == RequestStatus.PICKED_UP
    assert first.actual_simulated_pickup_time == 180
    assert second.status == RequestStatus.WAITING
    assert env.step().expirations == 1  # 240 s
    assert second.status == RequestStatus.EXPIRED
    assert env.step().completions == 1  # 300 s
    assert first.status == RequestStatus.COMPLETED
    assert first.completion_time == 300
    assert driver.status == DriverStatus.IDLE
    assert driver.assigned_request_id is None
    assert env.metrics()["avg_wait_seconds"] == 180

    while env.sim_time < 3600:
        env.step()
    assert len(env.requests) == 2
    step = env.step()  # 3660 s
    assert (step.assignments_made, step.pickups, step.completions) == (1, 1, 1)
    assert env.requests["r3"].status == RequestStatus.COMPLETED
    assert driver.status == DriverStatus.IDLE


def test_missing_hvf_request_time_does_not_fall_back_to_pickup(tmp_path):
    start = DAY + pd.Timedelta(hours=11)
    path = tmp_path / "unified.parquet"
    pd.DataFrame({
        "request_datetime": [pd.NaT, pd.NaT, start + pd.Timedelta(minutes=2)],
        "pickup_datetime": [start + pd.Timedelta(minutes=10)] * 3,
        "PULocationID": [1, 1, 1], "DOLocationID": [2, 2, 2],
        "provider": ["yellow", "uber", "lyft"],
    }).to_parquet(path, index=False)
    zones = tmp_path / "map.json"
    zones.write_text(json.dumps({1: [0, 0], 2: [0, 1]}), encoding="utf-8")
    trips = load_trips_for_simulation(path, zones, sim_start=start)
    assert trips["provider"].tolist() == ["lyft", "yellow"]
    assert trips["request_time_seconds"].tolist() == [120, 600]
