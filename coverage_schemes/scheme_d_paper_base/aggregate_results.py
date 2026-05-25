import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_STEP_SECONDS = 0.002


DEFAULT_METRICS = [
    "episode_reward",
    "coverage_rate",
    "success_count",
    "spawned_count",
    "missed_count",
    "cover_latency",
    "cover_latency_seconds",
    "per_point_cover_speed",
    "covered_per_second",
    "spawned_per_second",
    "covered_per_100_steps",
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


def derived_float(row, key, default=np.nan):
    value = safe_float(row.get(key))
    return default if np.isnan(value) else value


def add_derived_metrics(row):
    out = dict(row)
    steps = derived_float(out, "steps", 0.0)
    success_count = derived_float(out, "success_count", 0.0)
    spawned_count = derived_float(out, "spawned_count", 0.0)
    cover_latency_steps = derived_float(out, "cover_latency", 0.0)
    step_seconds = derived_float(out, "step_seconds", DEFAULT_STEP_SECONDS)
    if step_seconds <= 0.0:
        step_seconds = DEFAULT_STEP_SECONDS

    episode_time_seconds = steps * step_seconds
    cover_latency_seconds = cover_latency_steps * step_seconds
    derived = {
        "step_seconds": step_seconds,
        "episode_time_seconds": episode_time_seconds,
        "cover_latency_seconds": cover_latency_seconds,
        "per_point_cover_speed": (
            1.0 / cover_latency_seconds
            if success_count > 0.0 and cover_latency_seconds > 0.0
            else 0.0
        ),
        "covered_per_second": (
            success_count / episode_time_seconds if episode_time_seconds > 0.0 else 0.0
        ),
        "spawned_per_second": (
            spawned_count / episode_time_seconds if episode_time_seconds > 0.0 else 0.0
        ),
        "covered_per_100_steps": success_count * 100.0 / steps if steps > 0.0 else 0.0,
    }
    for key, value in derived.items():
        existing = safe_float(out.get(key))
        if np.isnan(existing):
            out[key] = value
    return out


def main():
    args = build_arg_parser().parse_args()
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    rows = [add_derived_metrics(row) for row in read_rows(args.inputs)]
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
