"""Compare fleet sizes and demand samples on full January 18 HVFHV scenarios."""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from prepare_scenarios import ROOT, WINDOWS, prepare_scenario
from config.paths import PROCESSED_DIR, RESULTS_DIR
from env.matching_env import DriverStatus, MatchingEnv, RequestStatus, SimConfig
from matching import GreedyMatcher

SEED = 42
FLEET_SIZES = (20, 40)
SAMPLING_PERCENTAGES = (1, 2, 5)
TIMESTEP = 60
TIMEOUT = 600


def audit_run(env, events, expected_requests):
    """Reconcile request states and replay driver reservations from the log."""
    states = Counter(request.status for request in env.requests.values())
    totals = Counter(event["event"] for event in events)
    active = {}
    assigned = set()
    transporting = set()
    for event in events:
        kind = event["event"]
        if kind not in {"assignment", "pickup", "completion"}:
            continue
        driver, request = event["driver_id"], event["request_id"]
        if kind == "assignment":
            if driver in active or request in assigned:
                raise RuntimeError("Overlapping driver assignment or duplicate request assignment")
            active[driver] = request
            assigned.add(request)
        elif active.get(driver) != request:
            raise RuntimeError("Pickup or completion without a matching driver reservation")
        elif kind == "pickup":
            if driver in transporting:
                raise RuntimeError("Driver picked up more than one request without completing a trip")
            transporting.add(driver)
        else:
            if driver not in transporting:
                raise RuntimeError("Completion occurred before pickup")
            transporting.remove(driver)
            del active[driver]

    expected_totals = {
        "request_arrival": expected_requests,
        "assignment": states[RequestStatus.ASSIGNED] + states[RequestStatus.PICKED_UP] + states[RequestStatus.COMPLETED],
        "pickup": states[RequestStatus.PICKED_UP] + states[RequestStatus.COMPLETED],
        "completion": states[RequestStatus.COMPLETED],
        "expiration": states[RequestStatus.EXPIRED],
    }
    if len(env.requests) != expected_requests or any(
        totals[event] != count for event, count in expected_totals.items()
    ):
        raise RuntimeError("Request counts do not reconcile with the event log")
    busy = {d.id: d.assigned_request_id for d in env.drivers.values() if d.status != DriverStatus.IDLE}
    if active != busy or transporting != {d.id for d in env.drivers.values() if d.status == DriverStatus.ON_TRIP}:
        raise RuntimeError("Final driver states do not match the event log")
    return states, totals


def run_configuration(sample, fleet_size, minutes):
    config = SimConfig(grid_size=10, n_drivers=fleet_size, timestep=TIMESTEP, request_timeout=TIMEOUT)
    env = MatchingEnv(GreedyMatcher(), config, seed=SEED, trips_df=sample)
    result = env.run(minutes * 60 // TIMESTEP)
    states, events = audit_run(env, result.events, len(sample))
    waiting = states[RequestStatus.WAITING]
    enroute = states[RequestStatus.ASSIGNED]
    on_trip = states[RequestStatus.PICKED_UP]
    return {
        "total_passenger_requests": len(sample),
        "requests_per_minute": len(sample) / minutes,
        "successful_assignments": events["assignment"],
        "completed_trips": states[RequestStatus.COMPLETED],
        "expired_requests": states[RequestStatus.EXPIRED],
        "requests_remaining": waiting + enroute + on_trip,
        "waiting_at_end": waiting,
        "assigned_enroute_at_end": enroute,
        "on_trip_at_end": on_trip,
        "picked_up_requests": result.metrics["n_pickups"],
        "average_wait_seconds": result.metrics["avg_request_to_pickup_seconds"],
        "driver_utilization": result.metrics["vehicle_utilization"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", type=Path, default=PROCESSED_DIR / "scenarios")
    parser.add_argument("--validation-report", type=Path, default=RESULTS_DIR / "dataset_validation_2023-01-18.csv")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "demand_calibration_2023-01-18.csv")
    args = parser.parse_args()
    report = pd.read_csv(args.validation_report).set_index("metric")["value"]
    if report.get("trip_validation") != "complete":
        parser.error("Generate a complete January 18 validation report and full scenario exports first.")
    day_total = int(report["final_valid_passenger_requests"])
    day_uber, day_lyft = int(report["uber"]), int(report["lyft"])
    if day_total != day_uber + day_lyft or day_total != sum(int(report[f"{h:02d}:00"]) for h in range(24)):
        raise ValueError("Provider and hourly demand counts must reconcile with daily demand")

    rows = []
    for label, window in WINDOWS.items():
        source = pd.read_parquet(args.scenario_dir / f"{label}.parquet")
        minutes = (window[1] - window[0]) * 60
        full = prepare_scenario(source, window, SEED)
        hourly_count = sum(int(report[f"{h:02d}:00"]) for h in range(*window))
        if len(full) != len(source) or len(full) != hourly_count:
            raise ValueError(f"{label} export must contain the full validated scenario")
        busiest_minute_count = full["request_datetime"].dt.floor("min").value_counts().max()
        for percentage in SAMPLING_PERCENTAGES:
            sample_size = (len(full) * percentage + 50) // 100
            sample = prepare_scenario(full, window, SEED, max_requests=sample_size)
            for fleet_size in FLEET_SIZES:
                metrics = run_configuration(sample, fleet_size, minutes)
                rows.append({
                    "scenario": label,
                    "fleet_size": fleet_size,
                    "demand_sample_percent": percentage,
                    "seed": SEED,
                    "matcher": "greedy",
                    "grid_size": 10,
                    "timestep_seconds": TIMESTEP,
                    "request_timeout_seconds": TIMEOUT,
                    "window_minutes": minutes,
                    "end_policy": "stop_at_window_end",
                    "full_day_requests": day_total,
                    "full_day_uber_requests": day_uber,
                    "full_day_lyft_requests": day_lyft,
                    "full_scenario_requests": len(full),
                    "full_scenario_requests_per_minute": len(full) / minutes,
                    "full_scenario_busiest_minute_requests": busiest_minute_count,
                    **metrics,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.6f")
    print(f"Saved {len(rows)} audited configurations to {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")


if __name__ == "__main__":
    main()
