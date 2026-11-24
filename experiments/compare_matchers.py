"""Run paired SP2 matcher experiments and generate preliminary comparisons."""

import argparse
from itertools import product
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import PROCESSED_DIR, RESULTS_DIR, ZONE_GRID_MAP
from env.matching_env import MatchingEnv, SimConfig, load_trips_for_simulation, make_synthetic_trips
from matching import CentralizedBatchMatcher, GreedyMatcher, RegionalBiddingMatcher, SparseMatcher

GRID_SIZE = 10
ALGORITHMS = ("greedy", "centralized_batch", "sparse", "regional")
DISPLAY = {
    "greedy": "Greedy",
    "centralized_batch": "Centralized batch",
    "sparse": "Sparse",
    "regional": "Regional",
}
COMPARISONS = {
    "n_assignments": "Successful assignments",
    "n_completed": "Completed trips",
    "n_expired": "Expired requests",
    "avg_request_to_pickup_seconds": "Request to pickup (s)",
    "avg_assignment_to_pickup_seconds": "Assignment to pickup (s)",
    "vehicle_utilization": "Driver utilization (%)",
    "total_matching_execution_seconds": "Matching wall time (ms)",
    "peak_matching_memory_bytes": "Peak traced memory (KiB)",
}
SCALES = {
    "vehicle_utilization": 100,
    "total_matching_execution_seconds": 1000,
    "peak_matching_memory_bytes": 1 / 1024,
}


def make_matcher(algorithm, cutoff, regional_options=None):
    if algorithm == "greedy":
        return GreedyMatcher(pickup_radius=cutoff)
    if algorithm == "centralized_batch":
        return CentralizedBatchMatcher(max_pickup_distance=cutoff)
    if algorithm == "sparse":
        distance = 2 * (GRID_SIZE - 1) if cutoff is None else cutoff
        return SparseMatcher(max_distance=distance)
    if algorithm == "regional":
        return RegionalBiddingMatcher(max_pickup_distance=cutoff, **(regional_options or {}))
    raise ValueError(f"Unknown algorithm: {algorithm}")


def sample_requests(trips, rate, seed):
    size = int(len(trips) * rate / 100 + 0.5)
    if size == len(trips):
        return trips
    return trips.sample(n=size, random_state=seed).sort_values(
        "request_time_seconds", kind="stable"
    ).reset_index(drop=True)


def run_matcher(algorithm, cutoff, config, seed, trips, steps, measure_memory=True, *, regional_options=None):
    def run(profile):
        return MatchingEnv(
            make_matcher(algorithm, cutoff, regional_options), config, seed, trips,
            profile_matching_memory=profile,
        ).run(steps)

    result = run(False)
    performance = result.performance.copy()
    if measure_memory:
        profiled = run(True)
        if result.metrics != profiled.metrics or result.events != profiled.events:
            raise RuntimeError(f"Memory replay changed {algorithm} outcomes")
        for key in ("peak_matching_memory_bytes", "matching_memory_profiled_timesteps"):
            performance[key] = profiled.performance[key]

    metrics = result.metrics
    if (
        metrics["n_requests_total"] != len(trips)
        or metrics["n_completed"] + metrics["n_expired"] + metrics["n_unfinished"] != len(trips)
    ):
        raise RuntimeError(f"Request counts did not reconcile for {algorithm}")
    return {**metrics, **performance}


def write_comparisons(results, output, synthetic):
    label = "synthetic runner test" if synthetic else "processed TLC scenarios"
    title = f"Preliminary SP2 matcher comparison | {label}"
    results.to_csv(output / "metrics.csv", index=False, float_format="%.9g")

    table = [f"# {title}", "", "Wall times come from unprofiled runs. Memory peaks come from separate replays.", ""]
    headings = ["Scenario", "Fleet", "Sample %", "Seed", "Cutoff (cells)", "Algorithm", *COMPARISONS.values()]
    table.extend(["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"])
    for row in results.to_dict("records"):
        values = [row[key] for key in ("scenario", "fleet_size", "sampling_percent", "seed", "pickup_cutoff_cells")]
        values.append(DISPLAY[row["algorithm"]])
        for key in COMPARISONS:
            value = row[key]
            if pd.isna(value):
                values.append("—")
            elif key in SCALES or key.startswith("avg_"):
                values.append(f"{value * SCALES.get(key, 1):.2f}")
            else:
                values.append(str(value))
        table.append("| " + " | ".join(map(str, values)) + " |")
    (output / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    for index, (scenario, scenario_rows) in enumerate(results.groupby("scenario", sort=False), start=1):
        figure, axes = plt.subplots(4, 2, figsize=(13, 12), layout="constrained")
        figure.suptitle(f"{title}\n{scenario} · bars show means across selected seeds")
        for ax, (metric, heading) in zip(axes.flat, COMPARISONS.items()):
            groups = ["fleet_size", "sampling_percent", "pickup_cutoff_cells"]
            values = scenario_rows.pivot_table(
                index=groups, columns="algorithm", values=metric, aggfunc="mean", dropna=False,
            ).reindex(columns=list(scenario_rows["algorithm"].unique()))
            ax.set_title(heading)
            if values.notna().to_numpy().any():
                values.index = [
                    f"{fleet} / {rate:g}%\n{cutoff} cells" if cutoff != "none" else f"{fleet} / {rate:g}%\nno cutoff"
                    for fleet, rate, cutoff in values.index
                ]
                (values.rename(columns=DISPLAY) * SCALES.get(metric, 1)).plot.bar(
                    ax=ax, rot=0, legend=False, width=0.75,
                )
                ax.set_xlabel("")
                ax.grid(axis="y", alpha=0.25)
                if metric == "n_assignments":
                    ax.legend()
            else:
                ax.text(0.5, 0.5, "No measurement", ha="center", transform=ax.transAxes)
                ax.set_xticks([])
        figure.supxlabel("Fleet size / sampled demand · pickup cutoff")
        figure.savefig(output / f"scenario_{index}_comparison.png", dpi=150)
        plt.close(figure)


def parse_cutoff(text):
    if text.lower() == "none":
        return None
    try:
        cutoff = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a nonnegative integer or 'none'.") from exc
    if cutoff < 0:
        raise argparse.ArgumentTypeError("Pickup cutoff must be nonnegative.")
    return cutoff


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS[:2])
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--scenarios", nargs="+", type=Path, help="Prepared scenario parquet files.")
    sources.add_argument("--synthetic", action="store_true", help="Run a labeled synthetic runner test.")
    parser.add_argument("--zone-map", type=Path, default=ZONE_GRID_MAP)
    parser.add_argument("--fleet-sizes", nargs="+", type=int, default=[20, 40])
    parser.add_argument("--sampling-rates", nargs="+", type=float, default=[1, 2, 5], help="Percent of each window.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--pickup-cutoffs", nargs="+", type=parse_cutoff, default=[None], help="Manhattan cells or 'none'.")
    parser.add_argument("--duration-minutes", type=int, default=180)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--regional-mode", choices=("auction", "greedy"), default="auction")
    parser.add_argument("--regional-scope", choices=("local", "broker"), default="broker")
    parser.add_argument("--regional-divisions", type=int, default=2)
    parser.add_argument("--auction-epsilon", type=float, default=0.5)
    parser.add_argument("--auction-max-iterations", type=int, default=10_000)
    parser.add_argument("--no-memory", action="store_true", help="Skip separate allocation-tracing runs.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if (
        args.duration_minutes <= 0 or args.timeout_seconds < 0
        or any(fleet <= 0 for fleet in args.fleet_sizes)
        or any(not 0 <= rate <= 100 for rate in args.sampling_rates)
        or any(not 0 <= seed < 2**32 for seed in args.seeds)
    ):
        parser.error("Use positive duration and fleet sizes, a nonnegative timeout, rates from 0 to 100, and 32-bit unsigned seeds.")
    default_name = "synthetic_matcher_test" if args.synthetic else "preliminary_matcher_comparison"
    output = args.output_dir or RESULTS_DIR / default_name
    output.mkdir(parents=True, exist_ok=True)
    regional_options = {
        "mode": args.regional_mode,
        "cross_region": args.regional_scope == "broker",
        "n_regions": args.regional_divisions,
        "epsilon": args.auction_epsilon,
        "max_iterations": args.auction_max_iterations,
    }
    if "regional" in args.algorithms:
        try:
            make_matcher("regional", None, regional_options)
        except ValueError as exc:
            parser.error(str(exc))

    rows = []
    scenarios = args.scenarios or [None]
    scenario_stems = [scenario.stem for scenario in args.scenarios or []]
    for scenario in scenarios:
        if scenario is None:
            source = make_synthetic_trips(n=200, seed=42, duration_seconds=args.duration_minutes * 60)
            label = "synthetic"
        else:
            source = load_trips_for_simulation(
                scenario, args.zone_map, sim_duration_seconds=args.duration_minutes * 60,
            )
            label = scenario.stem if scenario_stems.count(scenario.stem) == 1 else scenario.as_posix()
        for rate, seed in product(args.sampling_rates, args.seeds):
            trips = sample_requests(source, rate, seed)
            for fleet, cutoff in product(args.fleet_sizes, args.pickup_cutoffs):
                config = SimConfig(grid_size=GRID_SIZE, n_drivers=fleet, timestep=60, request_timeout=args.timeout_seconds)
                for algorithm in args.algorithms:
                    options = {"regional_options": regional_options} if algorithm == "regional" else {}
                    rows.append({
                        "scenario": label,
                        "scenario_file": scenario.as_posix() if scenario else "",
                        "data_source": "synthetic" if args.synthetic else "processed_tlc",
                        "result_stage": "preliminary",
                        "algorithm": algorithm,
                        "fleet_size": fleet,
                        "sampling_percent": rate,
                        "seed": seed,
                        "pickup_cutoff_cells": "none" if cutoff is None else str(cutoff),
                        "grid_size": GRID_SIZE,
                        "duration_minutes": args.duration_minutes,
                        "timestep_seconds": config.timestep,
                        "request_timeout_seconds": args.timeout_seconds,
                        "end_policy": "stop_at_duration",
                        "available_requests": len(source),
                        **({f"regional_{key}": value for key, value in regional_options.items()} if options else {}),
                        **run_matcher(
                            algorithm, cutoff, config, seed, trips, args.duration_minutes,
                            not args.no_memory, **options,
                        ),
                    })
    write_comparisons(pd.DataFrame(rows), output, args.synthetic)
    print(f"Saved {len(rows)} preliminary comparisons to {output}")


if __name__ == "__main__":
    main()
