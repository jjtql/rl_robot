#!/usr/bin/env python3
"""Recompute ICCC results with crossed train/evaluation-seed bootstrap.

The evaluation design is balanced but crossed: learned policies use three
training seeds, all methods use three held-out environment seeds, and every
train/evaluation cell contains five episodes.  This script preserves those
dependencies and pairs environment/episode resamples across methods.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
EVAL_ROOT = ROOT / "data" / "eval"
OUTPUT_ROOT = Path(__file__).resolve().parent / "tables"

STAGES = ("multi_low", "multi_realistic", "multi_hard", "multi_extreme")
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = (100, 101, 102)

METHODS = {
    "ours": {
        "kind": "learned",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_seed{train_seed}",
    },
    "no_attention": {
        "kind": "learned",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_attention_seed{train_seed}",
    },
    "no_carry": {
        "kind": "learned",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_carry_seed{train_seed}",
    },
    "no_prediction": {
        "kind": "learned",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_pred_seed{train_seed}",
    },
    "no_service_reward": {
        "kind": "learned",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_latency_reward_seed{train_seed}",
    },
    "no_residual": {
        "kind": "learned",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_residual_seed{train_seed}",
    },
    "vanilla_lstm_ppo": {
        "kind": "learned",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_vanilla_lstm_ppo_seed{train_seed}",
    },
    "horizon2": {"kind": "baseline", "policy": "horizon2"},
    "horizon3": {"kind": "baseline", "policy": "horizon3"},
    "nearest": {"kind": "baseline", "policy": "nearest"},
    "oldest": {"kind": "baseline", "policy": "oldest"},
    "distance_age": {"kind": "baseline", "policy": "distance_age"},
    "risk_aware": {"kind": "baseline", "policy": "risk_aware"},
    "dynamic_weighted": {"kind": "baseline", "policy": "dynamic_weighted"},
    "aco_tsp": {"kind": "baseline", "policy": "aco_tsp"},
    "planner_ensemble": {"kind": "baseline", "policy": "planner_ensemble"},
}

METRICS = {
    "coverage_rate": "higher",
    "cover_latency_seconds": "lower",
    "cover_latency_p90_seconds": "lower",
    "strict_response_sla_success_rate": "higher",
    "active_steam_mean": "lower",
    "action_delta_mean": "lower",
}

COMPARATORS = (
    "horizon2",
    "risk_aware",
    "no_residual",
    "vanilla_lstm_ppo",
    "no_attention",
    "no_carry",
    "no_prediction",
    "no_service_reward",
)

ROBUST_METHODS = {
    "ours": {
        "kind": "learned",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_seed{train_seed}",
    },
    "horizon2": {"kind": "baseline", "policy": "horizon2"},
    "dynamic_weighted": {"kind": "baseline", "policy": "dynamic_weighted"},
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode_path(method, stage, train_seed, eval_seed):
    spec = METHODS[method]
    if spec["kind"] == "baseline":
        return EVAL_ROOT / "baselines" / f"{spec['policy']}_{stage}_seed{eval_seed}.csv"
    run = spec["run"].format(train_seed=train_seed)
    return EVAL_ROOT / "main_ablation" / run / f"ppo_{stage}_seed{eval_seed}.csv"


def load_grid(method, stage, metric):
    spec = METHODS[method]
    train_seeds = TRAIN_SEEDS if spec["kind"] == "learned" else (None,)
    grid = {}
    for train_seed in train_seeds:
        for eval_seed in EVAL_SEEDS:
            path = episode_path(method, stage, train_seed, eval_seed)
            if not path.is_file():
                raise FileNotFoundError(f"Missing evaluation file: {path}")
            rows = read_rows(path)
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            if values.size != 5:
                raise ValueError(f"Expected 5 episodes in {path}, found {values.size}")
            grid[(train_seed, eval_seed)] = values
    return grid


def load_density_high_grid(method, stage, metric):
    spec = ROBUST_METHODS[method]
    train_seeds = TRAIN_SEEDS if spec["kind"] == "learned" else (None,)
    grid = {}
    for train_seed in train_seeds:
        for eval_seed in EVAL_SEEDS:
            if spec["kind"] == "baseline":
                path = (
                    EVAL_ROOT
                    / "robustness"
                    / "density_high"
                    / "baselines"
                    / f"{spec['policy']}_{stage}_seed{eval_seed}.csv"
                )
            else:
                run = spec["run"].format(train_seed=train_seed)
                path = (
                    EVAL_ROOT
                    / "robustness"
                    / "density_high"
                    / run
                    / f"ppo_{stage}_seed{eval_seed}.csv"
                )
            if not path.is_file():
                raise FileNotFoundError(f"Missing robustness evaluation file: {path}")
            rows = read_rows(path)
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            if values.size != 5:
                raise ValueError(f"Expected 5 episodes in {path}, found {values.size}")
            grid[(train_seed, eval_seed)] = values
    return grid


def point_mean(grid):
    return float(np.mean(np.concatenate(list(grid.values()))))


def variance_components(grid):
    train_keys = sorted({key[0] for key in grid if key[0] is not None})
    eval_keys = sorted({key[1] for key in grid})
    train_means = []
    for train_seed in train_keys:
        values = np.concatenate([grid[(train_seed, eval_seed)] for eval_seed in eval_keys])
        train_means.append(float(values.mean()))
    eval_means = []
    for eval_seed in eval_keys:
        values = np.concatenate([values for (train_seed, seed), values in grid.items() if seed == eval_seed])
        eval_means.append(float(values.mean()))
    train_sd = float(np.std(train_means, ddof=1)) if len(train_means) > 1 else 0.0
    eval_sd = float(np.std(eval_means, ddof=1)) if len(eval_means) > 1 else 0.0
    return train_sd, eval_sd


def grid_array(grid):
    baseline = all(key[0] is None for key in grid)
    if baseline:
        return np.stack([grid[(None, eval_seed)] for eval_seed in EVAL_SEEDS], axis=0), True
    return np.stack(
        [
            np.stack([grid[(train_seed, eval_seed)] for eval_seed in EVAL_SEEDS], axis=0)
            for train_seed in TRAIN_SEEDS
        ],
        axis=0,
    ), False


def draw_means(array, baseline, train_indices, eval_indices, episode_indices):
    if baseline:
        sampled = array[eval_indices[:, :, None], episode_indices]
        return sampled.mean(axis=(1, 2))
    sampled = array[
        train_indices[:, :, None, None],
        eval_indices[:, None, :, None],
        episode_indices[:, None, :, :],
    ]
    return sampled.mean(axis=(1, 2, 3))


def bootstrap_method(grid, rng, resamples):
    array, baseline = grid_array(grid)
    train_indices = rng.integers(0, len(TRAIN_SEEDS), size=(resamples, len(TRAIN_SEEDS)))
    eval_indices = rng.integers(0, len(EVAL_SEEDS), size=(resamples, len(EVAL_SEEDS)))
    episode_indices = rng.integers(0, 5, size=(resamples, len(EVAL_SEEDS), 5))
    return draw_means(array, baseline, train_indices, eval_indices, episode_indices)


def bootstrap_delta(main_grid, other_grid, rng, resamples):
    main_array, main_baseline = grid_array(main_grid)
    other_array, other_baseline = grid_array(other_grid)
    train_indices = rng.integers(0, len(TRAIN_SEEDS), size=(resamples, len(TRAIN_SEEDS)))
    eval_indices = rng.integers(0, len(EVAL_SEEDS), size=(resamples, len(EVAL_SEEDS)))
    episode_indices = rng.integers(0, 5, size=(resamples, len(EVAL_SEEDS), 5))
    main_draws = draw_means(main_array, main_baseline, train_indices, eval_indices, episode_indices)
    other_draws = draw_means(other_array, other_baseline, train_indices, eval_indices, episode_indices)
    return main_draws - other_draws


def two_sided_bootstrap_p(deltas):
    nonpositive = (np.count_nonzero(deltas <= 0.0) + 1.0) / (deltas.size + 1.0)
    nonnegative = (np.count_nonzero(deltas >= 0.0) + 1.0) / (deltas.size + 1.0)
    return float(min(1.0, 2.0 * min(nonpositive, nonnegative)))


def add_holm(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["other_method"], row["metric"])].append((float(row["p_value"]), index))
    for indexed in groups.values():
        indexed.sort()
        running = 0.0
        total = len(indexed)
        for rank, (p_value, row_index) in enumerate(indexed):
            adjusted = min(1.0, (total - rank) * p_value)
            running = max(running, adjusted)
            rows[row_index]["p_value_holm_4_stages"] = running
    return rows


def summarize_methods(rng, resamples):
    rows = []
    for method in METHODS:
        for stage in STAGES:
            for metric, direction in METRICS.items():
                grid = load_grid(method, stage, metric)
                draws = bootstrap_method(grid, rng, resamples)
                train_sd, eval_sd = variance_components(grid)
                rows.append(
                    {
                        "method": method,
                        "stage": stage,
                        "metric": metric,
                        "direction": direction,
                        "mean": point_mean(grid),
                        "ci95_low": float(np.percentile(draws, 2.5)),
                        "ci95_high": float(np.percentile(draws, 97.5)),
                        "between_train_seed_sd": train_sd,
                        "between_eval_seed_sd": eval_sd,
                        "train_seed_count": 0 if METHODS[method]["kind"] == "baseline" else len(TRAIN_SEEDS),
                        "eval_seed_count": len(EVAL_SEEDS),
                        "episodes_per_cell": 5,
                    }
                )
    return rows


def compare_to_main(rng, resamples):
    rows = []
    for other in COMPARATORS:
        for stage in STAGES:
            for metric, direction in METRICS.items():
                main_grid = load_grid("ours", stage, metric)
                other_grid = load_grid(other, stage, metric)
                deltas = bootstrap_delta(main_grid, other_grid, rng, resamples)
                delta = point_mean(main_grid) - point_mean(other_grid)
                rows.append(
                    {
                        "main_method": "ours",
                        "other_method": other,
                        "stage": stage,
                        "metric": metric,
                        "direction": direction,
                        "main_mean": point_mean(main_grid),
                        "other_mean": point_mean(other_grid),
                        "delta_main_minus_other": delta,
                        "delta_ci95_low": float(np.percentile(deltas, 2.5)),
                        "delta_ci95_high": float(np.percentile(deltas, 97.5)),
                        "p_value": two_sided_bootstrap_p(deltas),
                        "p_value_holm_4_stages": "",
                        "main_better": (delta > 0.0) if direction == "higher" else (delta < 0.0),
                    }
                )
    return add_holm(rows)


def write_service_table(summary_rows):
    index = {(row["method"], row["stage"], row["metric"]): row for row in summary_rows}
    rows = []
    for method in ("ours", "horizon2", "risk_aware", "no_residual", "vanilla_lstm_ppo"):
        for stage in STAGES:
            out = {"method": method, "stage": stage}
            for metric in METRICS:
                row = index[(method, stage, metric)]
                out[f"{metric}_mean"] = row["mean"]
                out[f"{metric}_ci95_low"] = row["ci95_low"]
                out[f"{metric}_ci95_high"] = row["ci95_high"]
            rows.append(out)
    write_rows(OUTPUT_ROOT / "service_metrics_hierarchical.csv", rows)


def write_density_high_table(rng, resamples):
    rows = []
    for method in ROBUST_METHODS:
        for stage in ("multi_hard", "multi_extreme"):
            for metric, direction in METRICS.items():
                grid = load_density_high_grid(method, stage, metric)
                draws = bootstrap_method(grid, rng, resamples)
                train_sd, eval_sd = variance_components(grid)
                rows.append(
                    {
                        "condition": "density_high",
                        "method": method,
                        "stage": stage,
                        "metric": metric,
                        "direction": direction,
                        "mean": point_mean(grid),
                        "ci95_low": float(np.percentile(draws, 2.5)),
                        "ci95_high": float(np.percentile(draws, 97.5)),
                        "between_train_seed_sd": train_sd,
                        "between_eval_seed_sd": eval_sd,
                    }
                )
    write_rows(OUTPUT_ROOT / "density_high_hierarchical.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    summary_rows = summarize_methods(rng, args.resamples)
    comparison_rows = compare_to_main(rng, args.resamples)
    write_rows(OUTPUT_ROOT / "hierarchical_method_summary.csv", summary_rows)
    write_rows(OUTPUT_ROOT / "hierarchical_comparisons.csv", comparison_rows)
    write_service_table(summary_rows)
    write_density_high_table(rng, args.resamples)
    print(f"Wrote {len(summary_rows)} method rows and {len(comparison_rows)} comparison rows to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
