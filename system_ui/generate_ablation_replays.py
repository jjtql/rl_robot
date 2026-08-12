#!/usr/bin/env python3
"""Generate reproducible MuJoCo rollout traces used by the system UI."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISUAL_ROOT = ROOT / "runs" / "v12_small_paper_suite" / "visuals"
TRAIN_ROOT = ROOT / "runs" / "v12_fast_paper_suite_20260529_175323" / "train"
MAIN_CONFIG = TRAIN_ROOT / "thermal_lstm_spawnhist_latency_v12_fast_seed0" / "config.json"

STAGES = ("multi_low", "multi_realistic", "multi_hard", "multi_extreme")
VARIANTS = {
    "ours": {
        "policy": "ppo",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_seed0",
        "output": VISUAL_ROOT / "thermal_lstm_spawnhist_latency_v12_fast_seed0",
    },
    "horizon2": {
        "policy": "horizon2",
        "config": MAIN_CONFIG,
        "output": VISUAL_ROOT / "horizon2",
    },
    "no_attention": {
        "policy": "ppo",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_attention_seed0",
        "output": VISUAL_ROOT / "ablations" / "no_attention",
    },
    "no_carry": {
        "policy": "ppo",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_carry_seed0",
        "output": VISUAL_ROOT / "ablations" / "no_carry",
    },
    "no_prediction": {
        "policy": "ppo",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_pred_seed0",
        "output": VISUAL_ROOT / "ablations" / "no_prediction",
    },
    "no_service_reward": {
        "policy": "ppo",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_latency_reward_seed0",
        "output": VISUAL_ROOT / "ablations" / "no_service_reward",
    },
    "no_residual": {
        "policy": "ppo",
        "run": "thermal_lstm_spawnhist_latency_v12_fast_no_residual_seed0",
        "output": VISUAL_ROOT / "ablations" / "no_residual",
    },
}

EXPECTED_CONFIG = {
    "ours": {"use_lstm": True, "residual_policy": True},
    "no_attention": {"use_steam_attention": False},
    "no_carry": {"carry_lstm_state_across_chunks": False},
    "no_prediction": {"pred_coef": 0.0},
    "no_service_reward": {"latency_first_reward": False},
    "no_residual": {"residual_policy": False},
}


def parse_csv(value, available):
    if value == "all":
        return list(available)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"Unknown values: {', '.join(unknown)}")
    return selected


def checkpoint_for(spec):
    run = spec.get("run")
    if not run:
        return None
    return TRAIN_ROOT / run / "checkpoints" / "scheme_d_paper_base_latest_full.pt"


def config_for(spec):
    if spec.get("config"):
        return Path(spec["config"])
    run = spec.get("run")
    return TRAIN_ROOT / run / "config.json" if run else None


def verify_variant(name, spec):
    config_path = config_for(spec)
    if not config_path or not config_path.is_file():
        raise FileNotFoundError(f"{name}: missing config {config_path}")
    checkpoint = checkpoint_for(spec)
    if spec["policy"] == "ppo" and (not checkpoint or not checkpoint.is_file()):
        raise FileNotFoundError(f"{name}: missing checkpoint {checkpoint}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_CONFIG.get(name, {}).items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"{name}: expected {key}={expected!r}, got {actual!r}")


def build_command(name, stage, steps, seed):
    spec = VARIANTS[name]
    output_dir = Path(spec["output"]) / stage
    command = [
        sys.executable,
        "-m",
        "coverage_schemes.scheme_d_paper_base.visualize_episode",
        "--policy",
        spec["policy"],
        "--stage",
        stage,
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--demo-mode",
        "--output-dir",
        str(output_dir),
    ]
    checkpoint = checkpoint_for(spec)
    if checkpoint:
        command.extend(("--model", str(checkpoint)))
    else:
        command.extend(("--config", str(config_for(spec))))
    return command, output_dir


def generate_one(name, stage, steps, seed, force, env):
    command, output_dir = build_command(name, stage, steps, seed)
    trajectory = output_dir / "trajectory.csv"
    summary = output_dir / "summary.json"
    if not force and trajectory.is_file() and summary.is_file():
        return name, stage, "SKIP", "already exists"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(f"{name}/{stage} failed:\n{tail}")
    if not trajectory.is_file() or not summary.is_file():
        raise RuntimeError(f"{name}/{stage} finished without replay files")
    result = json.loads(summary.read_text(encoding="utf-8"))
    detail = (
        f"coverage={result['coverage_rate']:.3f}, "
        f"latency={result['cover_latency_seconds']:.2f}s, "
        f"p90={result['cover_latency_p90_seconds']:.2f}s"
    )
    return name, stage, "DONE", detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all", help="Comma-separated keys or all.")
    parser.add_argument("--stages", default="all", help="Comma-separated stage names or all.")
    parser.add_argument("--steps", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", help="Optional physical GPU list, e.g. 5.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    variants = parse_csv(args.variants, VARIANTS)
    stages = parse_csv(args.stages, STAGES)
    for name in variants:
        verify_variant(name, VARIANTS[name])

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    tasks = [(name, stage) for name in variants for stage in stages]
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(generate_one, name, stage, args.steps, args.seed, args.force, env): (name, stage)
            for name, stage in tasks
        }
        for future in as_completed(futures):
            name, stage = futures[future]
            try:
                _, _, status, detail = future.result()
                print(f"[{status}] {name}/{stage}: {detail}", flush=True)
            except Exception as error:
                failures.append((name, stage, str(error)))
                print(f"[FAIL] {name}/{stage}: {error}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} replay task(s) failed")


if __name__ == "__main__":
    main()
