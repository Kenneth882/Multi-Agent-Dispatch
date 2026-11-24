"""Benchmark exact sparse and dense matching on identical single-step inputs."""

import argparse
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
import tracemalloc

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import RESULTS_DIR
from env.matching_env import Driver, MatchingContext, PassengerRequest, SimConfig, load_trips_for_simulation
from matching import CentralizedBatchMatcher, SparseMatcher

SEED = 42
SYNTHETIC_CASES = (
    (20, 120, 1), (40, 250, 3),
    (200, 1_000, 1), (200, 1_000, 18),
    (600, 2_500, 1), (600, 2_500, 3),
)


def benchmark(matcher, drivers, requests, config, repeats):
    def match():
        context = MatchingContext(60, {}, {}, config)
        started = perf_counter()
        pairs = matcher.match(drivers, requests, context)
        elapsed = perf_counter() - started
        return pairs, context.candidate_pairs_considered, elapsed

    pairs, candidates, _ = match()  # warm up before timing
    durations = [match()[2] for _ in range(repeats)]
    context = MatchingContext(60, {}, {}, config)
    tracemalloc.start()
    try:
        replay = matcher.match(drivers, requests, context)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if replay != pairs or context.candidate_pairs_considered != candidates:
        raise RuntimeError("Traced replay changed match results")

    by_driver = {driver.id: driver for driver in drivers}
    by_request = {request.id: request for request in requests}
    distances = [
        MatchingContext.manhattan_distance(by_driver[d].position, by_request[r].pickup)
        for d, r in pairs
    ]
    if len(pairs) != len({d for d, _ in pairs}) or len(pairs) != len({r for _, r in pairs}):
        raise RuntimeError("Duplicate assignments")
    return {
        "candidate_pairs": candidates,
        "matching_median_ms": median(durations) * 1_000,
        "peak_traced_bytes": peak,
        "successful_assignments": len(pairs),
        "total_pickup_distance_cells": sum(distances),
    }, distances


def compare(source, driver_cells, request_cells, cutoff, repeats):
    drivers = [Driver(f"d{i}", tuple(cell)) for i, cell in enumerate(driver_cells)]
    requests = [
        PassengerRequest(f"r{i}", source, tuple(cell), tuple(cell), 0)
        for i, cell in enumerate(request_cells)
    ]
    config = SimConfig(n_drivers=len(drivers))
    outcomes = []
    for algorithm, matcher in (
        ("centralized_batch", CentralizedBatchMatcher(cutoff)),
        ("sparse", SparseMatcher(cutoff)),
    ):
        metrics, distances = benchmark(matcher, drivers, requests, config, repeats)
        if any(distance > cutoff for distance in distances):
            raise RuntimeError(f"{algorithm} returned an infeasible assignment")
        outcomes.append({
            "source": source,
            "algorithm": algorithm,
            "drivers": len(drivers),
            "requests": len(requests),
            "cutoff_cells": cutoff,
            "seed": SEED,
            "timing_repeats": repeats,
            **metrics,
        })
    for key in ("successful_assignments", "total_pickup_distance_cells"):
        if outcomes[0][key] != outcomes[1][key]:
            raise RuntimeError(f"Dense and sparse objectives differ for {source}: {key}")
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tlc-scenario", type=Path, help="Include the busiest arrival minute from a prepared TLC scenario.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "sparse_matching_benchmark.csv")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if tracemalloc.is_tracing():
        parser.error("Run without an existing allocation-tracing session.")

    rows = []
    if args.tlc_scenario:
        trips = load_trips_for_simulation(args.tlc_scenario)
        if trips.empty:
            parser.error("The TLC scenario contains no arrivals.")
        minutes = (trips["request_time_seconds"] // 60).astype(int)
        busiest = minutes.value_counts().idxmax()
        request_cells = trips.loc[minutes.eq(busiest), ["pickup_row", "pickup_col"]].to_numpy().tolist()
        for fleet_size, cutoff in ((20, 1), (40, 3), (40, 18)):
            drivers = np.random.default_rng(SEED).integers(0, 10, size=(fleet_size, 2)).tolist()
            source = f"tlc_{args.tlc_scenario.stem}_minute_{busiest}_simulated_drivers"
            rows.extend(compare(source, drivers, request_cells, cutoff, args.repeats))

    for fleet_size, requests, cutoff in SYNTHETIC_CASES:
        rng = np.random.default_rng(SEED)
        driver_cells = rng.integers(0, 10, size=(fleet_size, 2)).tolist()
        request_cells = rng.integers(0, 10, size=(requests, 2)).tolist()
        rows.extend(compare("synthetic_uniform_grid", driver_cells, request_cells, cutoff, args.repeats))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.6f")
    print(f"Saved {len(rows)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
