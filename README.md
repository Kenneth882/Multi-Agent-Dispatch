# Multi-Agent-Dispatch

## Project Goal

This project implements an **Explainable Multi-Agent Ride-Sharing Dispatcher** using Multi-Agent Reinforcement Learning (MARL). The core research question is:

> How does the reliability of automated explanations of dispatch decisions impact the performance of a MARL-based ride-sharing dispatcher?

We simulate a ride-sharing environment (similar to Uber/Lyft) on a 10×10 grid representing Manhattan, where 20–40 driver agents learn to pick up passengers efficiently. A second model adds an explainability layer and is compared against the baseline on both efficiency and explanation consistency.

---

## Dataset

- **NYC TLC Yellow Taxi Trip Records — January 2023**
  - Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
  - Filtered to Manhattan-only trips using the TLC Taxi Zone shapefile
  - Used to derive passenger spawn probabilities and hourly demand rates for the simulation

- **TLC Taxi Zone Shapefile**
  - Source: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
  - Used to map zone IDs to grid cells

Raw data files are not committed to this repo. Place them in `data/raw/` before running.

---

## Project Structure

```
Multi-Agent-Dispatch/
├── data/
│   ├── raw/                  # TLC parquet + taxi zone shapefile (not committed)
│   └── processed/            # Filtered Manhattan trips (not committed)
├── env/                      # PettingZoo simulation environment
├── models/                   # Baseline and explainable MARL agents
├── experiments/              # Training and evaluation scripts
├── explanations/             # Explanation generation and consistency checks
├── results/                  # Saved metrics, plots, logs
├── notebooks/                # Exploratory analysis and visualizations
├── ride-share-marl/
│   └── pipeline/
│       ├── load_tlc_data.py  # Filters TLC data to Manhattan
│       └── demand_generator.py # Derives spawn weights and hourly rates
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
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

**5. Run the simulation** *(coming soon)*
```bash
python experiments/run_baseline.py
python experiments/run_explainable.py
```

---

## Evaluation Metrics

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
