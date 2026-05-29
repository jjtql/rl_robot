import argparse
import csv
import re
import subprocess
from pathlib import Path

import numpy as np


NUMERIC_CKPT_RE = re.compile(r"scheme_d_paper_base_(\d+)_full\.pt$")
TAGGED_CKPT_RE = re.compile(r"scheme_d_paper_base_(.+)_full\.pt$")


def parse_csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def checkpoint_tag(path):
    name = path.name
    numeric = NUMERIC_CKPT_RE.match(name)
    if numeric:
        return numeric.group(1)
    tagged = TAGGED_CKPT_RE.match(name)
    if tagged:
        return tagged.group(1)
    return path.stem


def checkpoint_sort_key(path):
    tag = checkpoint_tag(path)
    if tag.isdigit():
        return (0, int(tag), tag)
    if tag == "latest":
        return (2, 10**9, tag)
    return (1, 0, tag)


def discover_checkpoints(run_dir, tags=None, stride=100, include_stage_ends=True, include_latest=True):
    checkpoint_dir = Path(run_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    paths_by_tag = {checkpoint_tag(path): path for path in checkpoint_dir.glob("scheme_d_paper_base_*_full.pt")}
    if tags:
        selected = []
        for tag in tags:
            path = paths_by_tag.get(tag)
            if path is None and tag.isdigit():
                path = checkpoint_dir / f"scheme_d_paper_base_{tag}_full.pt"
            if path is None or not path.exists():
                raise FileNotFoundError(f"Checkpoint tag not found in {checkpoint_dir}: {tag}")
            selected.append(path)
        return sorted(selected, key=checkpoint_sort_key)

    selected = []
    for tag, path in paths_by_tag.items():
        if tag.isdigit() and int(tag) % int(stride) == 0:
            selected.append(path)
        elif include_stage_ends and tag.endswith("_end"):
            selected.append(path)
        elif include_latest and tag == "latest":
            selected.append(path)
    return sorted(selected, key=checkpoint_sort_key)


def read_summary(summary_path):
    with Path(summary_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_checkpoint(rows, checkpoint, tag):
    numeric_metrics = [
        "episode_reward_mean",
        "coverage_rate_mean",
        "missed_count_mean",
        "cover_latency_seconds_mean",
        "cover_latency_p90_seconds_mean",
        "response_sla_success_rate_mean",
        "active_steam_mean_mean",
        "covered_per_second_mean",
        "covered_per_100_steps_mean",
        "target_distance_mean",
        "material_quality_loss_mean",
    ]
    groups = {}
    for row in rows:
        key = (row["policy"], row["stage"])
        groups.setdefault(key, []).append(row)

    out = []
    for (policy, stage), group_rows in sorted(groups.items()):
        summary = {
            "checkpoint_tag": tag,
            "checkpoint": str(checkpoint),
            "policy": policy,
            "stage": stage,
            "seeds": len(group_rows),
            "episodes_per_seed": group_rows[0].get("episodes", ""),
        }
        for metric in numeric_metrics:
            values = np.array([float(row.get(metric, 0.0)) for row in group_rows], dtype=np.float64)
            summary[f"{metric}_across_seeds"] = float(values.mean()) if values.size else 0.0
            summary[f"{metric}_std_across_seeds"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out.append(summary)
    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate multiple saved PPO checkpoints and rank held-out performance.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing checkpoints/.")
    parser.add_argument("--checkpoint-tags", help="Comma-separated tags such as 300,500,700,900,latest,multi_hard_end.")
    parser.add_argument("--stride", type=int, default=100, help="Auto-select numeric checkpoints divisible by this value.")
    parser.add_argument("--no-stage-ends", action="store_true", help="Do not auto-include *_end_full.pt checkpoints.")
    parser.add_argument("--no-latest", action="store_true", help="Do not auto-include latest_full.pt.")
    parser.add_argument("--policies", default="ppo", help="Policies passed to run_matrix. Use ppo for checkpoint selection.")
    parser.add_argument("--stages", default="multi_hard")
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--decision-dt-seconds", type=float, help="Pass --decision-dt-seconds to run_matrix.")
    parser.add_argument("--response-sla-seconds", type=float, help="Pass --response-sla-seconds to run_matrix.")
    parser.add_argument("--demo-mode", action="store_true", help="Pass --demo-mode to run_matrix for long continuous-session evaluation.")
    parser.add_argument("--device", help="Device override passed to run_matrix.")
    parser.add_argument(
        "--rank-objective",
        choices=["coverage", "latency"],
        default="coverage",
        help="Rank checkpoints by coverage-first legacy score or latency/SLA-first response score.",
    )
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--output-dir", help="Output root. Defaults to runs/matrix/checkpoint_sweeps/<run>.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir)
    tags = parse_csv_list(args.checkpoint_tags) if args.checkpoint_tags else None
    checkpoints = discover_checkpoints(
        run_dir,
        tags=tags,
        stride=args.stride,
        include_stage_ends=not args.no_stage_ends,
        include_latest=not args.no_latest,
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints selected from {run_dir}")

    output_root = Path(args.output_dir or f"runs/matrix/checkpoint_sweeps/{run_dir.name}")
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for checkpoint in checkpoints:
        tag = checkpoint_tag(checkpoint)
        out_dir = output_root / tag
        command = [
            args.python,
            "-m",
            "coverage_schemes.scheme_d_paper_base.run_matrix",
            "--policies",
            args.policies,
            "--stages",
            args.stages,
            "--seeds",
            args.seeds,
            "--episodes",
            str(args.episodes),
            "--steps",
            str(args.steps),
            "--model",
            str(checkpoint),
            "--output-dir",
            str(out_dir),
        ]
        if args.device:
            command.extend(["--device", args.device])
        if args.decision_dt_seconds is not None:
            command.extend(["--decision-dt-seconds", str(args.decision_dt_seconds)])
        if args.response_sla_seconds is not None:
            command.extend(["--response-sla-seconds", str(args.response_sla_seconds)])
        if args.demo_mode:
            command.append("--demo-mode")
        print(f"\n=== checkpoint {tag} ===", flush=True)
        subprocess.run(command, check=True)
        all_rows.extend(summarize_checkpoint(read_summary(out_dir / "summary.csv"), checkpoint, tag))

    summary_path = output_root / "checkpoint_summary.csv"
    write_csv(summary_path, all_rows)

    print(f"\nWrote checkpoint summary to {summary_path}", flush=True)
    for stage in parse_csv_list(args.stages):
        candidates = [
            row for row in all_rows
            if row["policy"] == "ppo" and row["stage"] == stage
        ]
        if args.rank_objective == "latency":
            candidates.sort(
                key=lambda row: (
                    float(row["response_sla_success_rate_mean_across_seeds"]),
                    -float(row["cover_latency_p90_seconds_mean_across_seeds"]),
                    float(row["covered_per_second_mean_across_seeds"]),
                    -float(row["missed_count_mean_across_seeds"]),
                    float(row["coverage_rate_mean_across_seeds"]),
                ),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda row: (
                    float(row["coverage_rate_mean_across_seeds"]),
                    float(row["episode_reward_mean_across_seeds"]),
                    -float(row["cover_latency_seconds_mean_across_seeds"]),
                ),
                reverse=True,
            )
        if candidates:
            print(f"\nTop PPO checkpoints for {stage} ({args.rank_objective}):", flush=True)
            for row in candidates[:5]:
                print(
                    f"{row['checkpoint_tag']:>18} | "
                    f"Cov {float(row['coverage_rate_mean_across_seeds']):.3f} "
                    f"+/- {float(row['coverage_rate_mean_std_across_seeds']):.3f} | "
                    f"R {float(row['episode_reward_mean_across_seeds']):.1f} | "
                    f"Lat {float(row['cover_latency_seconds_mean_across_seeds']):.3f}s | "
                    f"P90 {float(row['cover_latency_p90_seconds_mean_across_seeds']):.3f}s | "
                    f"SLA {float(row['response_sla_success_rate_mean_across_seeds']):.2f} | "
                    f"Rate {float(row['covered_per_second_mean_across_seeds']):.2f}/s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
