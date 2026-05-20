import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_METRICS = [
    "episode_reward",
    "coverage_rate",
    "success_count",
    "spawned_count",
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
    parser = argparse.ArgumentParser(description="Aggregate paper evaluation CSV files.")
    parser.add_argument("inputs", nargs="+", help="Input CSV files.")
    parser.add_argument("--output", default="runs/eval/summary.csv")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    return parser


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


def main():
    args = build_arg_parser().parse_args()
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    rows = read_rows(args.inputs)
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("method", ""), row.get("policy", ""), row.get("stage", ""), row.get("max_steams", ""))
        grouped[key].append(row)

    out_rows = []
    for (method, policy, stage, max_steams), group in sorted(grouped.items()):
        out = {
            "method": method,
            "policy": policy,
            "stage": stage,
            "max_steams": max_steams,
            "episodes": len(group),
        }
        for metric in metrics:
            values = np.array([safe_float(row.get(metric)) for row in group], dtype=np.float64)
            values = values[~np.isnan(values)]
            out[f"{metric}_mean"] = float(values.mean()) if values.size else np.nan
            out[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out_rows.append(out)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else ["method", "policy", "stage", "max_steams", "episodes"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {output} with {len(out_rows)} grouped rows.")


if __name__ == "__main__":
    main()
