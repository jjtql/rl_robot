#!/usr/bin/env python3
"""Export the minimal real-rollout dataset consumed by the static web UI."""

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "runs" / "v12_small_paper_suite" / "visuals"
DEST_ROOT = ROOT / "system_ui" / "data"
ASSET_ROOT = ROOT / "system_ui" / "assets"
STAGES = ("multi_low", "multi_realistic", "multi_hard", "multi_extreme")
SOURCES = {
    "ours": SOURCE_ROOT / "thermal_lstm_spawnhist_latency_v12_fast_seed0",
    "horizon2": SOURCE_ROOT / "horizon2",
    "no_attention": SOURCE_ROOT / "ablations" / "no_attention",
    "no_carry": SOURCE_ROOT / "ablations" / "no_carry",
    "no_prediction": SOURCE_ROOT / "ablations" / "no_prediction",
    "no_service_reward": SOURCE_ROOT / "ablations" / "no_service_reward",
    "no_residual": SOURCE_ROOT / "ablations" / "no_residual",
}
FIELDS = (
    "step",
    "reward",
    "cover_x",
    "cover_y",
    "selected_target_id",
    "selected_target_x",
    "selected_target_y",
    "selected_target_risk_score",
    "route_confidence",
    "route_stagnation_score",
    "coverage_rate",
    "effective_coverage_rate",
    "cover_latency_seconds",
    "cover_latency_p90_seconds",
    "response_sla_success_rate",
    "strict_response_sla_success_rate",
    "success_count",
    "spawned_count",
    "missed_count",
    "steam_count",
    "pending_steam_count",
    "burst_lull_phase",
    "active_steams",
)


def export_csv(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{source} is missing fields: {', '.join(missing)}")
        with destination.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=FIELDS)
            writer.writeheader()
            for row in reader:
                writer.writerow({field: row.get(field, "") for field in FIELDS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEST_ROOT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    for name, default_source in SOURCES.items():
        source = source_root / default_source.relative_to(SOURCE_ROOT)
        for stage in STAGES:
            source_dir = source / stage
            destination_dir = args.dest_root / name / stage
            export_csv(source_dir / "trajectory.csv", destination_dir / "trajectory.csv")
            summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
            destination_dir.mkdir(parents=True, exist_ok=True)
            (destination_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "runs" / "v12_small_paper_suite" / "paper_figures" / "trajectory_comparison_multi_realistic.png",
        ASSET_ROOT / "trajectory_comparison_multi_realistic.png",
    )
    shutil.copy2(
        ROOT / "runs" / "v12_small_paper_suite" / "paper_figures" / "mujoco_environment_snapshot.png",
        ASSET_ROOT / "mujoco_environment_snapshot.png",
    )
    print(f"Exported {len(SOURCES) * len(STAGES)} rollouts to {args.dest_root}")


if __name__ == "__main__":
    main()
