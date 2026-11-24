# Research Progress Update #3

**November 23, 2026 · Senior Project 2**

## 1. Work completed this cycle

| Goal from Update #2 | Status and evidence |
|---|---|
| Validate the NYC TLC datasets | **Completed for this study.** Processed the official January 2023 HVFHV file and verified the Manhattan lookup and geographic grid for the January 18 scenarios. |
| Test and improve the simulation | **Completed for the current simulation.** Integrated prepared Uber/Lyft requests, tested driver and passenger states, and checked that demand and outcomes reconcile. |
| Refine greedy and centralized batch | **Completed for the initial comparison.** Reviewed feasibility, unmatched requests, and unavailable drivers. Added a case where centralized batch finds two feasible pairs and greedy finds one. |
| Improve sparse matching | **Completed for the current implementation.** Replaced dense cutoff storage with a spatially indexed CSR graph and checked the objective against dense matching. |
| Begin regional auction bidding | **Partially completed.** Implemented and tested prices, displacement, rebidding, and an optional cross-region broker. Real-data scaling and protocol evaluation remain open. |
| Begin performance evaluation | **Partially completed.** Ran demand calibration, paired greedy/batch experiments, and single-step sparse benchmarks. A multi-seed comparison of all four algorithms remains open. |

## 2. Dataset validation

The January 2023 HVFHV input contained **18,479,031** rows. The script scanned it in batches, checked required fields and timestamps, selected January 18 by New York local request time, and filtered both pickup and drop-off against the official TLC taxi-zone lookup.

| Filtering stage | Records remaining |
|---|---:|
| Original rows, required fields present, parseable timestamps | 18,479,031 |
| January 18 requests | 554,171 |
| Uber or Lyft | 554,171 |
| Manhattan pickup | 237,043 |
| Manhattan pickup and drop-off | 169,099 |
| Request at or before pickup | 166,986 |
| Valid duration, exact duplicate removal, valid grid endpoints | **166,986** |

The final count was **132,861 Uber** and **34,125 Lyft** requests. The total is 152 above the earlier estimate of 166,834. Midday (11 a.m.–2 p.m.) contained **20,530** requests, averaging **114.06/min**. Evening peak (5–8 p.m.) contained **37,144**, averaging **206.36/min** and reaching **260** in its busiest minute.

The rebuilt map matched the saved map and the official lookup's Manhattan zone IDs. The **69 zones** occupied **30 grid cells**. The pipeline reported zero missing or invalid grid mappings for retained requests. Full hourly and stage counts are in the [dataset validation CSV](results/dataset_validation_2023-01-18.csv).

## 3. Simulation improvements

- The SP2 environment loads prepared scenario parquets and also supports legacy Yellow Taxi input. Uber/Lyft arrivals use the observed request timestamp. Simulated driver movement determines pickup, trip completion, and waiting times.
- Tests cover idle, en route, and occupied driver states, and waiting, assigned, picked up, completed, and expired passenger states. Assignment checks reject duplicate requests and unavailable drivers.
- Standardized metrics cover matching outcomes, passenger wait and pickup distance, fleet utilization, candidate pairs, wall time, and optional traced allocation peaks. The fixed experiment cutoff records unfinished requests separately from expired requests.
- The final review fixed fractional-second truncation in prepared scenario offsets and added a regression test. Both published January 18 windows had whole-second request timestamps, so experiment inputs were unaffected. The SP1 simulation entry point was preserved.

## 4. Matching algorithm improvements

| Algorithm | Current implementation |
|---|---|
| Greedy | Processes requests in arrival order and selects the closest feasible idle driver. Requests with no feasible driver remain waiting. |
| Centralized batch | Uses a dense per-step assignment solver. It maximizes feasible assignment count, then minimizes total pickup distance on the 10×10 grid. Infeasible solver entries are discarded. |
| Sparse | Generates feasible neighbors from grid buckets, stores real edges and private unmatched edges in CSR, and solves the same two objectives. Zero-distance and disconnected cases are covered by tests. |
| Regional auction | Uses a configurable partition, defaulting to 2×2. Drivers bid against request prices and can displace earlier owners. Strict local auctions, centralized fallback on leftovers, and the original greedy regional prototype are selectable. |

A deterministic regional example produced **3 bids, 3 bidder turns, and 1 displacement** and matched both requests. A second example preserved a local award and added **1 cross-region match** with **4 modeled communication events**. A separate 10-request synthetic runner check produced **2 assignments, 2 bids, and 16 modeled communication events**. These checks establish basic behavior and metric export.

## 5. Automated tests and reproducibility

The full suite finished with **250 passed, 0 failed, and 0 skipped**. `python -m pip check` reported no broken requirements. The review used Python 3.13.12 on Windows with SciPy 1.17.1, NumPy 2.4.4, pandas 3.0.2, and PyArrow 24.0.0.

The audit rebuilt the scenarios from the official raw file and reran the published configurations in an ignored local directory. Dataset validation and demand calibration CSVs reproduced byte for byte. Across all 24 paired experiment runs, workload and outcome metrics reproduced exactly. Across all 18 sparse benchmark rows, candidate counts, assignment counts, and total pickup distance reproduced exactly. Wall-clock times and allocation peaks are measurements from individual runs and can vary.

## 6. Preliminary experimental results

The calibration used **12** greedy configurations across midday and peak, **20 and 40 drivers**, and **1%, 2%, and 5%** samples. All runs used seed **42**, the 10×10 grid, 60-second steps, a 600-second waiting timeout, and a fixed 180-step cutoff with no drain period. The calibration reconciled request counts and found no overlapping driver assignments. For peak with 40 drivers, the low, moderate, and high samples averaged **2.06**, **4.13**, and **10.32** requests/min, respectively. At 5% peak demand, **381 of 1,857** expired with 40 drivers, versus **1,171** with 20 drivers. [Calibration CSV](results/demand_calibration_2023-01-18.csv)

The paired greedy/batch comparison used the same 12 workload configurations and an 18-cell shared pickup limit, which admits every distance on the grid. The matchers completed equal numbers in eight pairs. Batch completed more in the other four. The table shows **5% peak demand and 40 drivers**.

| Metric | Greedy | Centralized batch |
|---|---:|---:|
| Successful assignments | 1,411 | 1,822 |
| Completed trips | 1,375 | 1,789 |
| Expired requests | 381 | 30 |
| Unfinished requests | 101 | 38 |
| Average request-to-pickup time | 633.0 s | 159.3 s |
| Average assignment-to-pickup time | 114.2 s | 45.0 s |
| Driver utilization | 42.2% | 54.4% |
| Total matching wall time | 5.11 ms | 28.50 ms |
| Peak traced allocation | 6.83 KiB | 18.42 KiB |

[Full comparison CSV](results/preliminary_matcher_comparison/metrics.csv) · [Comparison table](results/preliminary_matcher_comparison/comparison.md) · [Midday chart](results/preliminary_matcher_comparison/scenario_1_comparison.png) · [Peak chart](results/preliminary_matcher_comparison/scenario_2_comparison.png)

Sparse and dense returned equal assignment counts and total pickup distance in all nine single-step cases. For the TLC peak-minute snapshot with 40 simulated drivers, 260 requests, and a 3-cell cutoff, both found **33 assignments totaling 52 pickup cells**. Sparse reduced candidate pairs from **10,400 to 2,138**, median wall time from **4.31 to 1.78 ms**, and traced peak from **166 to 75 KiB**. At an 18-cell cutoff on the same snapshot, sparse took **7.47 ms** versus **5.43 ms** and used **282 KiB** versus **166 KiB**. For the larger **synthetic** case with 600 drivers, 2,500 requests, and cutoff 1, median time was **11.37 ms** sparse versus **459.68 ms** dense. [Sparse benchmark CSV](results/sparse_matching_benchmark.csv)

## 7. Remaining issues and limitations

- **Limited experimental coverage.** Real-data comparisons use one date, two windows, and one seed. The archived paired comparison covers greedy and centralized batch. Sparse evidence consists of single-step snapshots and synthetic workloads. Regional evidence consists of deterministic and synthetic checks.
- **Simulation approximations.** TLC zone polygons map to grid cells, and driver positions, travel, and fleet sizes are simulated. Sampling changes the workload. Fixed cutoffs leave some requests unfinished, and mean waits only include passengers picked up before the cutoff.
- **Computational measurement.** Traced allocations can omit native SciPy buffers. Timing is sensitive to hardware and system load. Sparse storage loses its advantage as more pairs become eligible.
- **Regional policy.** Fixed epsilon and iteration caps can leave feasible matches unfilled. The broker sees leftover drivers and requests and centrally resolves cross-region awards. Communication metrics count logical messages, with no network latency or traffic measurement.
- **Verified scope.** The study was reproduced from the streaming HVFHV scenario preparation path. The optional mixed-provider monthly export was outside this reproduction run. The review found no failing automated tests after the timestamp fix.

## 8. Recommended work for the next two weeks

| When | Planned task |
|---|---|
| Week 1 | Run paired comparisons of all four matchers across at least three seeds. Keep driver initialization and sampled requests identical within each comparison. Include the calibrated 1%, 2%, and 5% demand levels and compare a restrictive cutoff with the 18-cell baseline. |
| Week 1 | Compare strict local auction, brokered auction, and the greedy regional prototype. Record displacements, cap hits, unmatched outcomes, and communication events. |
| Week 2 | Repeat on additional TLC dates and report variation across seeds and dates. Use paired differences for assignments, completion, expiration, and wait. |
| Week 2 | Repeat runtime measurements on a quiet machine and add a separate process-level memory benchmark for native allocation coverage. Examine how sparse performance changes with eligible edge density. |

See the [README](README.md) for data locations, setup, and reproduction commands. The review outputs used to verify reproducibility are under ignored local paths.
