# Multi-Agent-Dispatch

## Project Scope

**Senior Project 2** compares greedy, centralized batch, sparse, and regional auction dispatch on a 10×10 Manhattan grid using actual NYC TLC request times and simulated driver positions. See the [third progress update](PROGRESS_UPDATE_3.md) for the current evidence and remaining work.

**Senior Project 1** implements an explainable ride-sharing dispatcher using multi-agent reinforcement learning. Its original research question is:

> How does the reliability of automated explanations of dispatch decisions impact the performance of a MARL-based ride-sharing dispatcher?

We simulate a ride-sharing environment (similar to Uber/Lyft) on a 10×10 grid representing Manhattan, where 20–40 driver agents learn to pick up passengers efficiently. A second model adds an explainability layer and is compared against the baseline on both efficiency and explanation consistency.

---

## Datasets

| Input | Expected location | Use |
|---|---|---|
| [January 2023 HVFHV records](https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-01.parquet) | `data/raw/fhvhv_tripdata_2023-01.parquet` | Primary SP2 Uber/Lyft input |
| [Official taxi zone lookup](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv) | `data/raw/taxi_zone_lookup.csv` | Checks Manhattan pickup and drop-off zones |
| [Taxi zone shapefile](https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip) | `data/raw/taxi_zones/taxi_zones.shp` alongside `.shx`, `.dbf`, and `.prj` | Geographic grid mapping |
| [January 2023 Yellow Taxi records](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet) | `data/raw/yellow_tripdata_2023-01.parquet` | Legacy simulation and full-suite data fixtures |

Keep raw files, processed data, virtual environments, and scratch experiment runs outside version control. Save scratch runs under `data/processed/` and retain only the small reports and figures used as evidence under `results/`. Keep credentials in local environment configuration.

---

## Project Structure

```
config/paths.py                  # Repository-relative data paths
data/raw/                        # Local TLC inputs
data/processed/                  # Local grid, parquet exports, and audit runs
env/matching_env.py              # SP2 simulation and evaluation metrics
env/ride_share_env.py            # SP1 reinforcement learning simulation
matching/                        # Common interface and four SP2 matchers
ride-share-marl/pipeline/        # Geographic mapping and TLC processing
scripts/prepare_scenarios.py      # Validation report and window exports
scripts/calibrate_demand.py       # SP2 fleet and demand calibration
experiments/compare_matchers.py   # SP2 paired full-window runs
experiments/benchmark_sparse.py   # SP2 single-step sparse/dense benchmark
experiments/run_simulation.py     # SP1 example simulation
models/, explanations/           # SP1 training and explainability support
tests/, results/                 # Tests and selected experiment outputs
```

---

## Setup and Legacy Simulation

**1. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**2. Add raw data to `data/raw/`**
- `yellow_tripdata_2023-01.parquet`
- `taxi_zones/taxi_zones.shp` (and associated shapefile files)

**3. Process the dataset**
```bash
python ride-share-marl/pipeline/load_tlc_data.py
```

**4. Verify demand generation**
```bash
python ride-share-marl/pipeline/demand_generator.py
```

**5. Run the Senior Project 1 simulation or train the PPO models**
```bash
python experiments/run_simulation.py
python experiments/train_baseline.py
python experiments/train_explainable.py
```

---

## Senior Project 1 Evaluation Metrics

**Efficiency (primary)**
| Metric | Description |
|---|---|
| Average Waiting Time | Mean time from passenger spawn to pickup |
| Order Response Rate | Fraction of passengers successfully picked up |
| Vehicle Utilization Rate | Fraction of time drivers are carrying a passenger |
| Average Detour Ratio | Extra distance driven vs. optimal route |

**Explainability**
| Metric | Description |
|---|---|
| Explanation Consistency | Same scenario → same explanation across runs |

Metrics are adopted from DualG-MARL (Sha et al., 2026) and CoopRide (Wang et al., 2025).

---

## Team Roles

| Member | Responsibilities |
|---|---|
| **Sajid Patwary** | Research, evaluation metrics, experiment analysis, final report |
| **Kenneth Romero Linares** | Environment setup, dataset preprocessing, model implementation |

Both members collaborate on testing, debugging, and writing the final report.

---

## References

- Sha et al. (2026). DualG-MARL. *Scientific Reports*. https://www.nature.com/articles/s41598-026-35004-8
- Wang et al. (2025). CoopRide. *ACM KDD*. https://dl.acm.org/doi/10.1145/3690624.3709205 — [GitHub](https://github.com/tsinghua-fib-lab/CoopRide)

---

## Senior Project 2

Compare four ride-sharing matching algorithms on the same 10×10 Manhattan simulation.

| Algorithm | Description |
|---|---|
| Greedy | Oldest requests get first choice of the nearest available driver |
| Centralized Batch | Exact feasible assignment per step using SciPy's dense solver. Maximizes assignment count, then minimizes pickup distance. |
| Sparse Matching | Grid-indexed feasible pairs solved with sparse minimum-cost matching |
| Regional Agent Bidding | Regional price auctions with optional brokered cross-region fallback |

### Prepare data and run tests

Use the same Python interpreter for installation and execution. Python 3.13 was used for verification on Windows. If the IDE reports missing imports for `pyarrow` or `gymnasium`, select the interpreter used for the requirements install.

After placing the inputs listed above, run these commands from the repository root to rebuild the grid, export the primary SP2 scenarios, and prepare the legacy fixtures used by the full test suite.

```bash
python ride-share-marl/pipeline/zone_grid.py
python scripts/prepare_scenarios.py --scenario-dir data/processed/scenarios
python ride-share-marl/pipeline/load_tlc_data.py
python ride-share-marl/pipeline/demand_generator.py
python -m pytest -q -rs
```

The primary SP2 path writes its validation CSV and the two January 18 scenario parquets. For the optional unified monthly Yellow Taxi and HVFHV parquet, run `python ride-share-marl/pipeline/tlc_pipeline.py`. The script writes `data/processed/combined_trips.parquet` and supports `--skip-missing` for partial input. January 18 experiments use the scenario exports.

### Run the four existing matchers on the same TLC window

```python
from config.paths import COMBINED_TRIPS
from env.matching_env import MatchingEnv, SimConfig, load_trips_for_simulation
from matching import CentralizedBatchMatcher, GreedyMatcher, RegionalBiddingMatcher, SparseMatcher

trips = load_trips_for_simulation(COMBINED_TRIPS)
config = SimConfig(request_timeout=600)
for matcher in (GreedyMatcher(), CentralizedBatchMatcher(), SparseMatcher(), RegionalBiddingMatcher()):
    result = MatchingEnv(matcher, config=config, seed=42, trips_df=trips).run(60)
    print(type(matcher).__name__, result.metrics)
```

The unified parquet's default window is January 9, 2023, from 9 to 10 a.m. Yellow Taxi records can use the pickup time as an arrival proxy. Uber and Lyft require the observed request timestamp. Driver starting positions and routes are simulated on the grid.

### Validate January 18 HVFHV scenarios

Place the January 2023 HVFHV parquet in `data/raw/fhvhv_tripdata_2023-01.parquet` and the official lookup in `data/raw/taxi_zone_lookup.csv`. The validator also uses the taxi-zone shapefile described above. [January 2023 HVFHV parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-01.parquet) · [TLC taxi zone lookup CSV](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv)

```bash
python scripts/prepare_scenarios.py
```

The report is written to `results/dataset_validation_2023-01-18.csv`. The script scans the monthly file in batches and retains only qualifying January 18 records for exact deduplication. It uses New York local request time to count hourly demand and select the half-open midday `[11:00, 14:00)` and peak `[17:00, 20:00)` windows. Required fields are the provider license, request/pickup/drop-off timestamps, and both zone IDs. Trip duration is computed from pickup and drop-off timestamps using the existing TLC pipeline limits.

Pass `--max-requests N --seed 42` to cap each window with reproducible sampling without replacement. Without a cap, the report counts every valid request in each window. A missing input produces a partial report and exit code 1.

### Run a prepared Uber/Lyft scenario

From the repository root, use the same TLC inputs listed above.

```bash
python ride-share-marl/pipeline/zone_grid.py --no-viz
python scripts/prepare_scenarios.py --scenario-dir data/processed/scenarios --max-requests 100 --seed 42
python -c "from pathlib import Path; from env.matching_env import MatchingEnv, SimConfig, load_trips_for_simulation; from matching import GreedyMatcher; trips = load_trips_for_simulation(Path('data/processed/scenarios/midday.parquet')); result = MatchingEnv(GreedyMatcher(), SimConfig(request_timeout=600), seed=42, trips_df=trips).run(210); print(result.metrics)"
python -m pytest -q
```

Use `peak.parquet` for the 5–8 p.m. window. The example runs the three-hour window and 30 additional steps for drivers to finish remaining trips. The loader uses the scenario's original window start even when sampling omits the first requests. Both endpoints are resolved through `zone_grid_map.json`. Requests at the window start appear at reset. Later requests become eligible at the first simulation tick at or after their arrival. The first matching tick is 60 seconds by default.

### January 18 demand calibration

The official January 2023 HVFHV file yields **166,986** valid January 18 Manhattan requests under the current validation rules, including 132,861 Uber and 34,125 Lyft. That is 152 above the midterm estimate of 166,834. Midday had 20,530 requests (114.06 per minute), and the evening peak had 37,144 (206.36 per minute on average, 260 in its busiest minute). [Dataset validation CSV](results/dataset_validation_2023-01-18.csv)

Run the calibration from the repository root with full scenario exports.

```bash
python scripts/prepare_scenarios.py --scenario-dir data/processed/scenarios
python scripts/calibrate_demand.py
```

The [calibration CSV](results/demand_calibration_2023-01-18.csv) compares 20 and 40 drivers at 1%, 2%, and 5% demand in both windows. Each sample uses seed 42 and is shared across the two fleets at its scenario and rate. The runner uses greedy matching on a 10×10 grid with 60-second steps. A request expires after 600 seconds while waiting for assignment. Every run stops after exactly 180 steps, with no drain period. Remaining requests include those still waiting, assigned and en route to pickup, or on a trip. Average wait includes every passenger picked up by the cutoff, and utilization is the fraction of fleet time spent carrying passengers.

Use the following evening-peak configurations for initial comparisons at a fixed fleet size.

| Demand | Drivers | Sample | Requests/min | Greedy outcomes |
|---|---:|---:|---:|---|
| Low | 40 | 1% | 2.06 | 369 completed, 0 expired, 2 unfinished |
| Moderate | 40 | 2% | 4.13 | 733 completed, 0 expired, 10 unfinished |
| High | 40 | 5% | 10.32 | 1,375 completed, 381 expired, 101 unfinished |

The moderate configuration roughly doubles passenger-carrying utilization from 11.5% to 23.1% versus low demand. The high configuration exposes capacity pressure, with 20.5% of requests expiring. At 20 drivers and 5% peak demand, 63.1% expired, making that combination useful as an overload test. The same 40-driver fleet handled all sampled 5% midday requests without expirations, so the demand labels depend on the window.

These results describe one date, seed, and baseline matcher. Sampling reduces demand without estimating the real fleet size. Grid travel uses simulated distances and driver positions. Wait averages exclude riders who never reach pickup, and the fixed cutoff leaves some trips unfinished.

### SP2 evaluation metrics

`MatchingEnv.run(steps)` returns deterministic outcomes and candidate counts in `result.metrics`, and wall-clock and memory measurements in `result.performance`. For a simulation driven step by step, use `env.metrics()` and `env.performance_metrics()`.

| Metric keys | Definition and units |
|---|---|
| `n_requests_total`, `n_assignments`, `n_pickups` | Counts of arrivals admitted by the cutoff, successful assignments after validation, and simulated pickups. |
| `n_completed`, `n_expired`, `n_unfinished` | Counts of completed trips, expired waiting requests, and pending requests at the cutoff. Unfinished equals admitted minus completed minus expired and includes waiting, en-route-to-pickup, and on-trip requests. Future scheduled arrivals are outside these counts. |
| `assignment_rate`, `pickup_rate`, `trip_completion_rate` | Assignments, pickups, and completions respectively, each divided by admitted requests. These are fractions from 0 to 1. |
| `avg_request_to_assignment_seconds` | Mean assignment timestamp minus scenario arrival timestamp for assigned requests. |
| `avg_request_to_pickup_seconds`, `avg_assignment_to_pickup_seconds` | Mean simulated pickup timestamp minus arrival or assignment timestamp respectively, over all pickups, including trips still in progress. |
| `avg_pickup_distance_cells` | Mean Manhattan distance in grid cells from the driver's assignment position to the pickup cell, over all pickups. |
| `vehicle_utilization` | Occupied driver-seconds divided by drivers × elapsed simulation seconds. An ongoing trip contributes occupied time up to the cutoff. |
| `avg_idle_seconds_per_driver` | Fleet time minus occupied time and time traveling to pickups, divided by the driver count. Pickup travel still in progress contributes up to the cutoff. |
| `avg_completed_trips_per_driver` | Completed trips divided by all simulated drivers. |
| `candidate_pairs_considered` | Total driver–request pairs scored or emitted across matching calls. Dense matching counts every pair it scores, including cutoff rejections. Sparse matching emits only eligible graph edges and counts each one once. Private unmatched edges and internal solver operations are excluded. Custom matchers can call `context.pickup_distance(driver, request)` to count evaluations. |
| `total_matching_execution_seconds`, `avg_matching_execution_seconds_per_timestep` | Wall-clock duration inside `match()` summed across steps, and divided by the number of steps, including empty steps. Assignment validation and application occur outside this timing. |
| `total_simulation_execution_seconds` | Wall-clock duration of reset through the last step, including matching and environment work. Input loading and final metric aggregation are outside this timing. Manual stepping sums the active reset and step durations. |
| `peak_matching_memory_bytes`, `matching_memory_profiled_timesteps` | Largest per-call traced allocation peak in bytes, and the count of calls profiled. Enable with `MatchingEnv(..., profile_matching_memory=True)`. |

Undefined averages and rates use `NaN`, empty counts use zero, and an unavailable memory measurement uses `None`. For compatibility, `avg_wait_seconds` still averages completed trips only, and `order_response_rate` still divides completed trips by completed plus expired. Use the new metrics for comparisons with explicit cohorts.

Timing uses Python's monotonic `perf_counter()`. Compare runtimes with memory profiling off, then repeat the same inputs with profiling on for memory. Tracing adds overhead to the measured call. [Python timing documentation](https://docs.python.org/3/library/time.html#time.perf_counter)

Memory uses process-wide `tracemalloc` around each matcher call. It measures the peak of live allocations traced during that call, including temporary Python objects and NumPy allocations registered with the tracer. SciPy or other native work buffers can bypass tracing and be missed. The measurement excludes pre-existing allocations and does not represent process RSS or total native memory. Use a serial benchmark without unrelated allocating threads. A pre-existing tracing session is left alone, and an incomplete set of measurements yields `None` for the peak. [Python allocation tracing](https://docs.python.org/3/library/tracemalloc.html) · [NumPy allocation tracking](https://numpy.org/doc/stable/reference/c-api/data_memory.html)

### Preliminary SP2 matcher comparison

The commands below regenerate the real January 18 midday and peak comparisons. Export the full windows so the runner can independently sample 1%, 2%, and 5% demand without replacement. Each algorithm starts with a fresh matcher and environment. Within each configuration, the same sample and seed give every algorithm identical arrivals and initial driver positions.

```bash
python scripts/prepare_scenarios.py --scenario-dir data/processed/scenarios
python experiments/compare_matchers.py --scenarios data/processed/scenarios/midday.parquet data/processed/scenarios/peak.parquet --algorithms greedy centralized_batch --fleet-sizes 20 40 --sampling-rates 1 2 5 --seeds 42 --pickup-cutoffs 18 --duration-minutes 180 --timeout-seconds 600 --output-dir results/preliminary_matcher_comparison
```

The shared 18-cell limit admits every possible pickup distance on the 10×10 grid, matching the calibration setup. Runs stop at 180 steps without a drain period and use a 600-second timeout while waiting for assignment. Average wait means request to simulated pickup. Average pickup time means assignment to simulated pickup. Both are measured over picked-up passengers. Timing comes from an unprofiled run and peak traced memory from a fresh replay of the same configuration. The runner checks that the replay produces identical events and outcomes.

The [metrics CSV](results/preliminary_matcher_comparison/metrics.csv) contains every standardized metric and the settings for each run. The [comparison table](results/preliminary_matcher_comparison/comparison.md) and the [midday](results/preliminary_matcher_comparison/scenario_1_comparison.png) and [peak](results/preliminary_matcher_comparison/scenario_2_comparison.png) charts summarize the paired runs. Charts show means across selected seeds. With seed 42, the algorithms completed equal numbers of trips in eight configurations. Centralized batch completed more in the other four, including 1,789 versus 1,375 trips at 5% peak demand with 40 drivers. These are preliminary results from one day and one seed.

For a follow-up with multiple seeds and a tighter shared cutoff, use a separate output directory to retain the preliminary results. Regional uses the auction with brokered fallback by default.

```bash
python experiments/compare_matchers.py --scenarios data/processed/scenarios/midday.parquet data/processed/scenarios/peak.parquet --algorithms greedy centralized_batch sparse regional --fleet-sizes 20 40 --sampling-rates 1 2 5 --seeds 42 43 44 --pickup-cutoffs 3 --duration-minutes 180 --output-dir results/matcher_followup
```

If TLC scenario files are unavailable, the following command exercises the runner and labels its outputs as synthetic.

```bash
python experiments/compare_matchers.py --synthetic --algorithms greedy centralized_batch sparse regional --fleet-sizes 2 --sampling-rates 5 --seeds 42 43 --pickup-cutoffs 3 --duration-minutes 10 --no-memory
```

### Exact sparse matching and preliminary benchmarks

The sparse matcher buckets grid positions by cell and enumerates only the Manhattan neighborhood inside the cutoff. Vertices in the same cell reuse the same neighbor list. The graph uses compressed sparse row (CSR) storage throughout. Its smaller partition forms the rows, and each row gets one private unmatched column so the solver can cover every row even in disconnected graphs. A real edge of distance `d` has cost `d + 1`, which preserves zero-distance matches. The dummy cost exceeds the largest possible total real-edge cost, so minimizing cost first maximizes the number of real assignments and then minimizes their total pickup distance. Dummy matches are discarded. The unweighted sparse maximum matching routine can maximize count alone. The weighted LAPJVsp routine satisfies both objectives in one solve. [SciPy sparse weighted matching](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.min_weight_full_bipartite_matching.html)

For `D` drivers, `R` requests, `E` eligible pairs, and `k = min(D,R)`, the CSR graph stores `E + k` edges and uses `O(D + R + E)` space including grid buckets and solver workspace. Let `U` be the number of occupied row-side cells, `B` the maximum number of cells in a clipped cutoff neighborhood, and `e_c` the number of eligible columns for row cell `c`. Graph construction takes `O(D log D + R log R + UB + Σ(e_c log(e_c + 1)) + E)` time, including sorting. The sparse shortest-augmenting-path solver performs up to `k` augmentations. A conservative bound is `O(k(E + (D + R)^2))` because one search can scan column arrays repeatedly. For broad cutoffs or heavily co-located demand, `E` can approach `D × R`, erasing the storage advantage. Equal-cost solutions can select different individual pairs and lead to different later simulation states. [SciPy implementation](https://github.com/scipy/scipy/blob/main/scipy/sparse/csgraph/_matching.pyx)

Run the benchmark from the repository root after exporting the full peak scenario.

```bash
python experiments/benchmark_sparse.py --tlc-scenario data/processed/scenarios/peak.parquet --repeats 5
```

The [benchmark CSV](results/sparse_matching_benchmark.csv) compares identical single-step inputs. The TLC cases use 260 actual pickup cells from the busiest evening arrival minute and seed-42 simulated driver positions. The larger workloads use uniform synthetic grid positions. Wall time is the median of five untraced calls after warmup. Allocation peaks come from a separate traced call and can miss native work buffers. These are single-step measurements.

Both solvers returned the same assignment count and total pickup distance in all nine cases. Selected results below were measured with SciPy 1.17.1 and NumPy 2.4.4. Values are shown as dense → sparse.

| Input | Drivers × requests | Cutoff | Candidate pairs | Median wall time | Peak traced memory | Matched / total pickup cells |
|---|---:|---:|---:|---:|---:|---:|
| TLC peak minute | 40 × 260 | 3 | 10,400 → 2,138 | 4.31 → 1.78 ms | 166 → 75 KiB | 33 / 52 |
| TLC peak minute | 40 × 260 | 18 | 10,400 → 10,400 | 5.43 → 7.47 ms | 166 → 282 KiB | 40 / 93 |
| Synthetic uniform | 600 × 2,500 | 1 | 1,500,000 → 69,042 | 459.68 → 11.37 ms | 22.92 → 1.85 MiB | 600 / 0 |

The tight-cutoff gains depend on low edge density. At cutoff 18 both tested workloads used more traced memory with sparse matching, and the runtime direction differed between the TLC and synthetic cases. The larger uniform workloads contain enough co-located requests to give each driver a zero-distance match, so they primarily test graph construction and storage at scale.

### Regional price auction

`RegionalBiddingMatcher` defaults to a 2×2 partition and sequential bidding. A driver's region comes from its current position, and a request's region comes from its pickup. For each auction, a real request has valuation `reward - Manhattan pickup distance` if it meets the cutoff. The common reward is `1 + k × (max_eligible_distance + epsilon)`, where `k` is the smaller of the number of drivers with eligible choices and the number of requests. Idle has value zero. Prices and valuations are in grid-cell cost units. They are internal scores.

An unassigned driver compares valuations minus current prices. If its best surplus is positive, it bids the current price plus the difference between its best and second-best surplus plus `epsilon`. The idle option also competes for second place. Each accepted bid takes ownership of one request, and a displaced driver rejoins the queue. Driver and request IDs break ties deterministically. A driver with no positive surplus exits for that auction. Prices reset every timestep. Each auction ends when the queue empties or reaches `max_iterations`, which defaults to 10,000 bidder turns per region or broker phase. At the cap, current awards stand and unassigned requests remain available to the simulator. This is a fixed-epsilon implementation without epsilon scaling. A cap or the two-phase regional policy can produce different outcomes from a global optimum. [Bertsekas, *The Auction Algorithm for the Assignment Problem*](https://www.mit.edu/~dimitrib/TheAuctionAP.pdf)

- **Strict local** with `cross_region=False`. Each regional auction receives only its eligible drivers and waiting requests, and maintains its own prices and awards. The simulator partitions the inputs in one process. Local auction decisions require no cross-region data exchange.
- **Brokered fallback** with `cross_region=True`. Local awards are finalized first. Participating regions then expose leftover driver IDs and positions and request IDs and pickup cells to a common broker. The broker auctions only pairs spanning different regions and returns the awards to both regions. It has shared visibility into the leftovers and uses centralized coordination.
- **Greedy prototype** with `mode="greedy"`. This retains driver-first local nearest-request matching with optional immediate global fallback. Its fallback has no modeled message exchange.

The simulator sums `regional_bids`, `regional_auction_rounds` (one queued-driver turn, including turns that exit without bidding), `regional_displacements`, `regional_iteration_limit_hits`, and `regional_cross_assignments` across steps. `regional_communication_events` counts one logical submission per participating region when the broker opens and two award notifications per cross-region assignment. These counters model message counts without measuring bytes, latency, or network traffic. Runtime and traced memory use the standard matching instrumentation.

The following small synthetic example exercises the auction and records counters in `metrics.csv`. Replace `broker` with `local` for strict local matching or `auction` with `greedy` for the prototype. Use a different output directory for each run.

```bash
python experiments/compare_matchers.py --synthetic --algorithms regional --regional-mode auction --regional-scope broker --regional-divisions 2 --auction-epsilon 0.5 --auction-max-iterations 10000 --fleet-sizes 2 --sampling-rates 5 --seeds 42 --pickup-cutoffs 3 --duration-minutes 10 --output-dir data/processed/regional_auction_smoke
```
