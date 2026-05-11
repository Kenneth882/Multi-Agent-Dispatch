import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
WEIGHTS = [0.01, 0.05, 0.1, 0.5]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_all_results():
    baseline = load_json(f"{RESULTS_DIR}/baseline_results.json")["averages"]

    explainable = {}
    for w in WEIGHTS:
        path = f"{RESULTS_DIR}/explainable_results_w{w}.json"
        if os.path.exists(path):
            explainable[w] = load_json(path)["averages"]

    return baseline, explainable


def plot_tradeoff(baseline, explainable):
    weights = sorted(explainable.keys())
    metrics = ["response_rate", "utilization", "consistency", "detour_ratio"]
    labels  = ["Order Response Rate", "Vehicle Utilization", "Explanation Consistency", "Detour Ratio"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Baseline vs. Explainable Model — Metric Comparison", fontsize=14)

    for ax, metric, label in zip(axes.flatten(), metrics, labels):
        baseline_val = baseline[metric]
        exp_vals = [explainable[w][metric] for w in weights]

        ax.axhline(y=baseline_val, color="steelblue", linestyle="--", linewidth=1.5, label="Baseline")
        ax.plot(weights, exp_vals, color="darkorange", marker="o", linewidth=2, label="Explainable")
        ax.set_title(label)
        ax.set_xlabel("Consistency Reward Weight")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/comparison_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


def plot_spawn_weights():
    weights_path = "data/processed/spawn_weights.npy"
    if not os.path.exists(weights_path):
        print("Spawn weights not found, skipping heatmap.")
        return

    weights = np.load(weights_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weights, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Spawn Probability")
    ax.set_title("Passenger Spawn Probability Heatmap\n(derived from NYC TLC Jan 2023)")
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/spawn_heatmap.png"
    plt.savefig(out_path, dpi=150)
    print(f"Heatmap saved to {out_path}")
    plt.show()


def print_summary_table(baseline, explainable):
    weights = sorted(explainable.keys())
    print("\n--- Summary Table ---")
    print(f"{'Model':<22} {'Pickups':>8} {'Avg Wait':>10} {'Resp Rate':>11} {'Utilization':>12} {'Detour':>8} {'Consistency':>13}")
    print("-" * 90)

    b = baseline
    print(f"{'Baseline':<22} {b['avg_pickups']:>8.1f} {b['avg_wait']:>10.2f} {b['response_rate']:>10.2%} {b['utilization']:>11.2%} {b['detour_ratio']:>8.4f} {b['consistency']:>12.2%}")

    for w in weights:
        e = explainable[w]
        print(f"{'Explainable w='+str(w):<22} {e['avg_pickups']:>8.1f} {e['avg_wait']:>10.2f} {e['response_rate']:>10.2%} {e['utilization']:>11.2%} {e['detour_ratio']:>8.4f} {e['consistency']:>12.2%}")


if __name__ == "__main__":
    baseline, explainable = load_all_results()

    if not explainable:
        print("No explainable results found. Run eval_explainable.py for each weight first.")
    else:
        print_summary_table(baseline, explainable)
        plot_tradeoff(baseline, explainable)

    plot_spawn_weights()
