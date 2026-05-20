import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_METRICS = [
    "episode_reward",
    "coverage_rate",
    "missed_count",
    "cover_latency",
    "target_distance",
    "selected_target_risk_score",
    "height_uniformity",
    "overfill_penalty",
    "action_delta_mean",
    "action_l2_mean",
]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate paper tables and simple plots from evaluation/training CSV files.")
    parser.add_argument("inputs", nargs="+", help="Input CSV files, such as all_rows.csv or episodes.csv.")
    parser.add_argument("--output-dir", default="runs/plots")
    parser.add_argument("--group-by", default="method,policy,stage,max_steams")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--plot", action="store_true", help="Write PNG bar plots when matplotlib is available.")
    return parser


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def group_rows(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        groups[key].append(row)
    return groups


def summarize(groups, group_keys, metrics):
    out_rows = []
    for key, rows in sorted(groups.items()):
        out = {group_key: value for group_key, value in zip(group_keys, key)}
        out["episodes"] = len(rows)
        for metric in metrics:
            values = np.array([safe_float(row.get(metric)) for row in rows], dtype=np.float64)
            values = values[~np.isnan(values)]
            out[f"{metric}_mean"] = float(values.mean()) if values.size else np.nan
            out[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out_rows.append(out)
    return out_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_label(row, group_keys):
    return " | ".join(str(row.get(group_key, "")) for group_key in group_keys if row.get(group_key, "") != "")


def write_plots(output_dir, summary_rows, group_keys, metrics):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; wrote tables only.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [make_label(row, group_keys) for row in summary_rows]
    x = np.arange(len(summary_rows))
    width = 0.72

    for metric in metrics:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        means = np.array([safe_float(row.get(mean_key)) for row in summary_rows], dtype=np.float64)
        stds = np.array([safe_float(row.get(std_key)) for row in summary_rows], dtype=np.float64)
        valid = ~np.isnan(means)
        if not valid.any():
            continue

        fig_width = max(8.0, 0.55 * len(summary_rows))
        fig, ax = plt.subplots(figsize=(fig_width, 4.8))
        ax.bar(x[valid], means[valid], width=width, yerr=stds[valid], capsize=3)
        ax.set_ylabel(metric)
        ax.set_xticks(x[valid])
        ax.set_xticklabels([labels[i] for i in x[valid]], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.png", dpi=160)
        plt.close(fig)


def main():
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    group_keys = parse_csv_list(args.group_by)
    metrics = parse_csv_list(args.metrics)
    rows = read_rows(args.inputs)
    groups = group_rows(rows, group_keys)
    summary_rows = summarize(groups, group_keys, metrics)
    write_csv(output_dir / "summary.csv", summary_rows)
    if args.plot:
        write_plots(output_dir, summary_rows, group_keys, metrics)
    print(f"Wrote {len(summary_rows)} grouped rows to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
