import argparse
import csv
import json
import shlex
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import deep_update, load_config, save_config, set_global_seeds
from .env import ShangZengEnv
from .eval import configure_env_from_config, coverage_timing_metrics, resolve_metric_step_seconds
from .policies import build_policy, checkpoint_config


MAIN_METHOD = "thermal_lstm_spawnhist_latency_v12_fast"
ABLATION_METHODS = [
    MAIN_METHOD,
    "thermal_lstm_spawnhist_latency_v12_fast_no_pred",
    "thermal_lstm_spawnhist_latency_v12_fast_no_carry",
    "thermal_lstm_spawnhist_latency_v12_fast_no_latency_reward",
    "thermal_lstm_spawnhist_latency_v12_fast_no_attention",
    "thermal_lstm_spawnhist_latency_v12_fast_no_residual",
    "thermal_lstm_spawnhist_latency_v12_fast_vanilla_lstm_ppo",
]
BASELINE_POLICIES = [
    "nearest",
    "oldest",
    "distance_age",
    "risk_aware",
    "dynamic_weighted",
    "horizon2",
    "horizon3",
    "aco_tsp",
    "planner_ensemble",
]
PAPER_METRICS = [
    "coverage_rate",
    "effective_coverage_rate",
    "cover_latency_seconds",
    "cover_latency_p90_seconds",
    "cover_latency_p95_seconds",
    "cover_latency_max_seconds",
    "response_sla_success_rate",
    "strict_response_sla_success_rate",
    "active_steam_mean",
    "active_steam_max",
    "pending_steam_count",
    "oldest_active_age_max_seconds",
    "action_delta_mean",
    "action_l2_mean",
    "episode_reward",
    "missed_count",
    "covered_per_second",
    "full_session_terminal_clear",
]
SIGNIFICANCE_METRICS = [
    "coverage_rate",
    "effective_coverage_rate",
    "cover_latency_seconds",
    "cover_latency_p90_seconds",
    "response_sla_success_rate",
    "strict_response_sla_success_rate",
    "active_steam_mean",
    "pending_steam_count",
    "action_delta_mean",
]
LOWER_IS_BETTER = {
    "cover_latency_seconds",
    "cover_latency_p90_seconds",
    "active_steam_mean",
    "pending_steam_count",
    "action_delta_mean",
}
SWEEP_LATENCY = {
    "low": 18.0,
    "mid": 30.0,
    "high": 42.0,
}
SWEEP_BACKLOG = {
    "low": 0.10,
    "mid": 0.20,
    "high": 0.32,
}
ROBUSTNESS_SCENARIOS = {
    "density_low": {
        "thermal_background_weight": 0.035,
        "thermal_hotspot_strength": 3.0,
    },
    "density_high": {
        "thermal_background_weight": 0.075,
        "thermal_hotspot_strength": 4.5,
    },
    "action_delay_1": {
        "action_delay_steps": 1,
    },
    "action_noise_003": {
        "action_noise_std": 0.03,
    },
    "hotspot_drift": {
        "thermal_drift_std": 0.008,
    },
    "domain_randomization_010": {
        "domain_randomization": True,
        "domain_randomization_scale": 0.10,
    },
}


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_csv(value):
    return [int(item) for item in parse_csv(value)]


def shell_join(command):
    return " ".join(shlex.quote(str(item)) for item in command)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def metric_mean(row, metric):
    return as_float(row.get(f"{metric}_mean"))


def permutation_p_value(a, b, rng, resamples):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 1.0
    observed = abs(float(a.mean() - b.mean()))
    combined = np.concatenate([a, b])
    count = 0
    for _ in range(max(int(resamples), 1)):
        permuted = rng.permutation(combined)
        diff = abs(float(permuted[: a.size].mean() - permuted[a.size :].mean()))
        if diff >= observed - 1e-12:
            count += 1
    return float((count + 1) / (max(int(resamples), 1) + 1))


def bootstrap_delta_ci(a, b, rng, resamples):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0
    deltas = []
    for _ in range(max(int(resamples), 1)):
        a_sample = rng.choice(a, size=a.size, replace=True)
        b_sample = rng.choice(b, size=b.size, replace=True)
        deltas.append(float(a_sample.mean() - b_sample.mean()))
    return (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


def add_holm_correction(rows):
    indexed = []
    for idx, row in enumerate(rows):
        p_value = as_float(row.get("p_value"))
        if p_value is not None:
            indexed.append((p_value, idx))
    indexed.sort()
    total = len(indexed)
    running = 0.0
    for rank, (p_value, idx) in enumerate(indexed):
        adjusted = min((total - rank) * p_value, 1.0)
        running = max(running, adjusted)
        rows[idx]["p_value_holm"] = running
    return rows


def strip_seed(run_name):
    marker = "_seed"
    if marker not in run_name:
        return run_name, ""
    method, seed = run_name.rsplit(marker, 1)
    return method, seed


class SuiteRunner:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.run_root)
        self.train_root = self.root / "train"
        self.eval_root = self.root / "eval"
        self.tables_dir = self.root / "paper_tables"
        self.visuals_dir = self.root / "visuals"
        self.runtime_dir = self.root / "runtime"
        self.manifest = {
            "suite": "v12_small_paper_supplemental",
            "run_root": str(self.root),
            "commands": [],
            "notes": [],
        }
        self.seeds = parse_int_csv(args.seeds)
        self.eval_seeds = parse_int_csv(args.eval_seeds)
        if args.quick:
            self.seeds = self.seeds[:1]
            self.eval_seeds = self.eval_seeds[:1]
            self.args.eval_episodes = min(self.args.eval_episodes, 1)
            self.args.eval_steps = min(self.args.eval_steps, 800)
            self.args.stages = "multi_hard,multi_extreme"
            self.args.robust_stages = "multi_hard"
            self.args.train_jobs = min(self.args.train_jobs, 1)

    def run_command(self, label, command):
        record = {
            "label": label,
            "command": shell_join(command),
        }
        self.manifest["commands"].append(record)
        print()
        print(f">>> [{label}] {record['command']}", flush=True)
        if self.args.dry_run:
            return
        subprocess.run(command, check=True)

    def train_command(self, methods, run_dir, output_name, extra_flags=None):
        command = [
            self.args.python,
            "-m",
            "coverage_schemes.scheme_d_paper_base.run_training_suite",
            "--methods",
            ",".join(methods),
            "--seeds",
            ",".join(str(seed) for seed in self.seeds),
            "--run-dir",
            str(run_dir),
            "--output",
            str(run_dir / output_name),
            "--execute",
            "--jobs",
            str(self.args.train_jobs),
            "--device",
            self.args.device,
        ]
        if self.args.train_stage_episodes:
            command.extend(["--stage-episodes", self.args.train_stage_episodes])
        if self.args.quick:
            command.extend(
                [
                    "--stage-episodes",
                    "multi_low:3,multi_realistic:3,multi_hard:4,multi_extreme:2",
                    "--episode-steps",
                    "600",
                    "--bc-episodes",
                    "8",
                    "--bc-epochs",
                    "2",
                    "--bc-stage-episodes",
                    "multi_low:2,multi_realistic:2,multi_hard:2,multi_extreme:2",
                    "--save-interval",
                    "20",
                ]
            )
        if extra_flags:
            command.extend(extra_flags)
        return command

    def train_main_and_ablations(self):
        if self.args.skip_train:
            print("skip_train is set; training main and ablations is skipped.")
            return
        run_dir = self.train_root / "main_ablation"
        command = self.train_command(ABLATION_METHODS, run_dir, "train_commands.txt")
        self.run_command("train_main_ablation", command)

    def sweep_combos(self):
        if self.args.sweep_mode == "none":
            return []
        combos = []
        for latency_name, latency_gain in SWEEP_LATENCY.items():
            for backlog_name, backlog_gain in SWEEP_BACKLOG.items():
                combos.append((latency_name, backlog_name, latency_gain, backlog_gain))
        if self.args.quick or self.args.sweep_mode == "compact":
            combos = [
                ("low", "low", SWEEP_LATENCY["low"], SWEEP_BACKLOG["low"]),
                ("mid", "mid", SWEEP_LATENCY["mid"], SWEEP_BACKLOG["mid"]),
                ("high", "high", SWEEP_LATENCY["high"], SWEEP_BACKLOG["high"]),
            ]
        return combos

    def train_reward_sweep(self):
        if self.args.skip_train or self.args.skip_sweep:
            print("training reward sweep is skipped.")
            return
        for latency_name, backlog_name, latency_gain, backlog_gain in self.sweep_combos():
            combo = f"lat_{latency_name}_backlog_{backlog_name}"
            run_dir = self.train_root / "reward_sweep" / combo
            extra = [
                "--cover-latency-penalty-gain",
                str(latency_gain),
                "--backlog-penalty-gain",
                str(backlog_gain),
            ]
            command = self.train_command([MAIN_METHOD], run_dir, "train_commands.txt", extra_flags=extra)
            self.run_command(f"train_sweep_{combo}", command)

    def checkpoint_path(self, method, seed, run_dir=None):
        base = Path(run_dir) if run_dir is not None else self.train_root / "main_ablation"
        return base / f"{method}_seed{seed}" / "checkpoints" / "scheme_d_paper_base_latest_full.pt"

    def run_config_path(self, method, seed, run_dir=None):
        base = Path(run_dir) if run_dir is not None else self.train_root / "main_ablation"
        return base / f"{method}_seed{seed}" / "config.json"

    def first_main_checkpoint(self):
        for seed in self.seeds:
            checkpoint = self.checkpoint_path(MAIN_METHOD, seed)
            if checkpoint.exists() or self.args.dry_run:
                return checkpoint, seed
        raise FileNotFoundError(f"No main v12 checkpoint found under {self.train_root / 'main_ablation'}")

    def reference_config_path(self):
        checkpoint, seed = self.first_main_checkpoint()
        config_path = self.run_config_path(MAIN_METHOD, seed)
        if config_path.exists():
            return config_path
        generated_path = self.root / "reference_v12_config.json"
        if self.args.dry_run:
            return generated_path
        config = deep_update(load_config(None), checkpoint_config(checkpoint))
        save_config(config, generated_path)
        return generated_path

    def matrix_command(self, output_dir, policies, model, stages=None, seeds=None, extra_flags=None):
        command = [
            self.args.python,
            "-m",
            "coverage_schemes.scheme_d_paper_base.run_matrix",
            "--policies",
            ",".join(policies),
            "--stages",
            stages or self.args.stages,
            "--seeds",
            seeds or ",".join(str(seed) for seed in self.eval_seeds),
            "--episodes",
            str(self.args.eval_episodes),
            "--steps",
            str(self.args.eval_steps),
            "--model",
            str(model),
            "--device",
            self.args.device,
            "--decision-dt-seconds",
            str(self.args.decision_dt_seconds),
            "--output-dir",
            str(output_dir),
        ]
        if self.args.demo_mode:
            command.append("--demo-mode")
        if self.args.response_sla_seconds is not None:
            command.extend(["--response-sla-seconds", str(self.args.response_sla_seconds)])
        if extra_flags:
            command.extend(extra_flags)
        return command

    def eval_baselines(self):
        if self.args.skip_eval:
            print("skip_eval is set; baseline evaluation is skipped.")
            return
        checkpoint, _ = self.first_main_checkpoint()
        output_dir = self.eval_root / "baselines"
        command = self.matrix_command(output_dir, BASELINE_POLICIES, checkpoint)
        self.run_command("eval_baselines", command)

    def eval_main_and_ablations(self):
        if self.args.skip_eval:
            print("skip_eval is set; PPO evaluation is skipped.")
            return
        for method in ABLATION_METHODS:
            for seed in self.seeds:
                checkpoint = self.checkpoint_path(method, seed)
                if not checkpoint.exists() and not self.args.dry_run:
                    if self.args.fail_on_missing:
                        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                    print(f"Skipping missing checkpoint: {checkpoint}")
                    continue
                output_dir = self.eval_root / "main_ablation" / f"{method}_seed{seed}"
                command = self.matrix_command(output_dir, ["ppo"], checkpoint)
                self.run_command(f"eval_{method}_seed{seed}", command)

    def eval_reward_sweep(self):
        if self.args.skip_eval or self.args.skip_sweep:
            print("reward sweep evaluation is skipped.")
            return
        for latency_name, backlog_name, _, _ in self.sweep_combos():
            combo = f"lat_{latency_name}_backlog_{backlog_name}"
            run_dir = self.train_root / "reward_sweep" / combo
            for seed in self.seeds:
                checkpoint = self.checkpoint_path(MAIN_METHOD, seed, run_dir=run_dir)
                if not checkpoint.exists() and not self.args.dry_run:
                    if self.args.fail_on_missing:
                        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                    print(f"Skipping missing checkpoint: {checkpoint}")
                    continue
                output_dir = self.eval_root / "reward_sweep" / combo / f"{MAIN_METHOD}_seed{seed}"
                command = self.matrix_command(output_dir, ["ppo"], checkpoint)
                self.run_command(f"eval_sweep_{combo}_seed{seed}", command)

    def robustness_flags(self, overrides):
        flags = []
        for key, value in overrides.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    flags.append(flag)
            else:
                flags.extend([flag, str(value)])
        return flags

    def eval_robustness(self):
        if self.args.skip_eval or self.args.skip_robustness:
            print("robustness evaluation is skipped.")
            return
        checkpoint, _ = self.first_main_checkpoint()
        stages = self.args.robust_stages
        for scenario, overrides in ROBUSTNESS_SCENARIOS.items():
            flags = self.robustness_flags(overrides)
            output_dir = self.eval_root / "robustness" / scenario / "baselines"
            command = self.matrix_command(
                output_dir,
                ["horizon2", "dynamic_weighted"],
                checkpoint,
                stages=stages,
                extra_flags=flags,
            )
            self.run_command(f"robust_{scenario}_baselines", command)
            for seed in self.seeds:
                model = self.checkpoint_path(MAIN_METHOD, seed)
                if not model.exists() and not self.args.dry_run:
                    if self.args.fail_on_missing:
                        raise FileNotFoundError(f"Missing checkpoint: {model}")
                    continue
                output_dir = self.eval_root / "robustness" / scenario / f"{MAIN_METHOD}_seed{seed}"
                command = self.matrix_command(
                    output_dir,
                    ["ppo"],
                    model,
                    stages=stages,
                    extra_flags=flags,
                )
                self.run_command(f"robust_{scenario}_ppo_seed{seed}", command)

    def visualize(self):
        if self.args.skip_visuals:
            print("visualization is skipped.")
            return
        checkpoint, seed = self.first_main_checkpoint()
        config_path = self.reference_config_path()
        for stage in parse_csv(self.args.visual_stages):
            ppo_output = self.visuals_dir / f"{MAIN_METHOD}_seed{seed}" / stage
            ppo_command = [
                self.args.python,
                "-m",
                "coverage_schemes.scheme_d_paper_base.visualize_episode",
                "--policy",
                "ppo",
                "--model",
                str(checkpoint),
                "--config",
                str(config_path),
                "--stage",
                stage,
                "--steps",
                str(self.args.eval_steps),
                "--seed",
                str(self.args.visual_seed),
                "--output-dir",
                str(ppo_output),
            ]
            if self.args.demo_mode:
                ppo_command.append("--demo-mode")
            self.run_command(f"visual_ppo_{stage}", ppo_command)

            base_output = self.visuals_dir / "horizon2" / stage
            base_command = [
                self.args.python,
                "-m",
                "coverage_schemes.scheme_d_paper_base.visualize_episode",
                "--policy",
                "horizon2",
                "--config",
                str(config_path),
                "--stage",
                stage,
                "--steps",
                str(self.args.eval_steps),
                "--seed",
                str(self.args.visual_seed),
                "--output-dir",
                str(base_output),
            ]
            if self.args.demo_mode:
                base_command.append("--demo-mode")
            self.run_command(f"visual_horizon2_{stage}", base_command)

    def runtime_one(self, policy_name, model_path, stage, seed, steps):
        config = load_config(None)
        if model_path:
            config = deep_update(config, checkpoint_config(model_path))
        config["device"] = self.args.device
        config["decision_dt_seconds"] = self.args.decision_dt_seconds
        if self.args.response_sla_seconds is not None:
            config["response_sla_seconds"] = self.args.response_sla_seconds
        set_global_seeds(seed)
        env = ShangZengEnv(
            model_path=config["model_path"],
            max_episode_steps=steps,
            target_selector=config.get("target_selector", "risk_aware"),
            steam_attention_observation=config.get("use_steam_attention", False),
            spawn_history_observation=config.get("use_spawn_history_observation", False),
            thermal_context_observation=config.get("use_thermal_context_observation", False),
            route_summary_observation=config.get("use_route_summary_observation", False),
            material_map_observation=config.get("use_material_map", False),
            attention_steam_count=config.get("attention_steam_count", 6),
            attention_steam_dim=config.get("attention_steam_dim", 8),
        )
        configure_env_from_config(env, config)
        env.configure_curriculum(stage)
        env.max_episode_steps = steps
        if self.args.demo_mode:
            env.target_success_count = max(env.target_success_count, steps)
            env.target_coverage = 1.0
        metric_step_seconds = resolve_metric_step_seconds(
            env,
            config=config,
            override=self.args.decision_dt_seconds,
        )
        policy = build_policy(
            policy_name,
            env,
            model_path=str(model_path) if policy_name == "ppo" else None,
            deterministic=True,
            config_override=config,
        )
        obs, info = env.reset(seed=seed)
        policy.reset()
        act_ms = []
        step_ms = []
        total_reward = 0.0
        for step in range(steps):
            t0 = time.perf_counter()
            action = policy.act(env, obs)
            t1 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            t2 = time.perf_counter()
            act_ms.append((t1 - t0) * 1000.0)
            step_ms.append((t2 - t1) * 1000.0)
            total_reward += float(reward)
            if self.args.demo_mode and terminated:
                terminated = False
            if terminated or truncated:
                break
        actual_steps = step + 1 if steps > 0 else 0
        success_count = int(info.get("success_count", 0))
        spawned_count = int(info.get("spawned_count", 0))
        timing = coverage_timing_metrics(
            steps=actual_steps,
            success_count=success_count,
            spawned_count=spawned_count,
            cover_latency_steps=float(info.get("cover_latency", 0.0)),
            step_seconds=metric_step_seconds,
        )
        return {
            "policy": policy_name,
            "stage": stage,
            "seed": seed,
            "steps": actual_steps,
            "act_ms_mean": float(np.mean(act_ms)) if act_ms else 0.0,
            "act_ms_p90": float(np.percentile(act_ms, 90)) if act_ms else 0.0,
            "env_step_ms_mean": float(np.mean(step_ms)) if step_ms else 0.0,
            "env_step_ms_p90": float(np.percentile(step_ms, 90)) if step_ms else 0.0,
            "coverage_rate": float(info.get("coverage_rate", 0.0)),
            "effective_coverage_rate": float(info.get("effective_coverage_rate", 0.0)),
            "cover_latency_seconds": timing["cover_latency_seconds"],
            "cover_latency_p90_seconds": float(info.get("cover_latency_p90", 0.0)) * metric_step_seconds,
            "response_sla_success_rate": float(info.get("response_sla_success_rate", 0.0)),
            "active_steam_mean": float(info.get("active_steam_mean", 0.0)),
            "pending_steam_count": int(info.get("pending_steam_count", 0)),
            "episode_reward": total_reward,
        }

    def runtime(self):
        if self.args.skip_runtime:
            print("runtime efficiency evaluation is skipped.")
            return
        if self.args.dry_run:
            self.manifest["notes"].append("runtime would measure ppo, horizon2, and dynamic_weighted act/env-step ms")
            return
        checkpoint, _ = self.first_main_checkpoint()
        rows = []
        for stage in parse_csv(self.args.runtime_stages):
            for seed in self.eval_seeds:
                rows.append(self.runtime_one("ppo", checkpoint, stage, seed, self.args.runtime_steps))
                rows.append(self.runtime_one("horizon2", checkpoint, stage, seed, self.args.runtime_steps))
                rows.append(self.runtime_one("dynamic_weighted", checkpoint, stage, seed, self.args.runtime_steps))
        write_csv(self.runtime_dir / "runtime_results.csv", rows)
        print(f"Wrote runtime results to {self.runtime_dir / 'runtime_results.csv'}", flush=True)

    def collect_eval_rows(self):
        rows = []
        for summary_path in sorted(self.eval_root.glob("**/summary.csv")):
            rel = summary_path.relative_to(self.eval_root)
            parts = rel.parts
            for row in read_csv(summary_path):
                out = {
                    "source_summary": str(summary_path),
                    "table": parts[0] if parts else "unknown",
                    "variant": "",
                    "train_method": "",
                    "train_seed": "",
                    "sweep": "",
                    "scenario": "",
                }
                if parts[0] == "baselines":
                    out["variant"] = row.get("policy", "")
                elif parts[0] == "main_ablation" and len(parts) >= 2:
                    method, train_seed = strip_seed(parts[1])
                    out["variant"] = method
                    out["train_method"] = method
                    out["train_seed"] = train_seed
                elif parts[0] == "reward_sweep" and len(parts) >= 3:
                    method, train_seed = strip_seed(parts[2])
                    out["variant"] = parts[1]
                    out["sweep"] = parts[1]
                    out["train_method"] = method
                    out["train_seed"] = train_seed
                elif parts[0] == "robustness" and len(parts) >= 3:
                    out["scenario"] = parts[1]
                    if parts[2] == "baselines":
                        out["variant"] = row.get("policy", "")
                    else:
                        method, train_seed = strip_seed(parts[2])
                        out["variant"] = method
                        out["train_method"] = method
                        out["train_seed"] = train_seed
                else:
                    out["variant"] = row.get("policy", "")
                out.update(row)
                rows.append(out)
        return rows

    def aggregate_rows(self, rows, group_fields):
        grouped = defaultdict(list)
        for row in rows:
            key = tuple(row.get(field, "") for field in group_fields)
            grouped[key].append(row)
        out_rows = []
        for key, group in sorted(grouped.items()):
            out = {field: value for field, value in zip(group_fields, key)}
            out["n"] = len(group)
            out["eval_seeds"] = ",".join(sorted({str(row.get("seed", "")) for row in group if row.get("seed", "")}))
            out["train_seeds"] = ",".join(
                sorted({str(row.get("train_seed", "")) for row in group if row.get("train_seed", "")})
            )
            for metric in PAPER_METRICS:
                source_field = f"{metric}_mean"
                values = [as_float(row.get(source_field)) for row in group]
                mean, std = mean_std(values)
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
            out_rows.append(out)
        return out_rows

    def samples_for(self, rows, metric):
        samples = [metric_mean(row, metric) for row in rows]
        return [sample for sample in samples if sample is not None]

    def significance_rows(self, rows):
        rng = np.random.default_rng(int(self.args.significance_seed))
        out_rows = []
        stages = sorted({row.get("stage", "") for row in rows if row.get("stage", "")})
        max_steams_values = sorted({row.get("max_steams", "") for row in rows if row.get("max_steams", "")})
        for stage in stages:
            for max_steams in max_steams_values:
                main_rows = [
                    row
                    for row in rows
                    if row.get("table") == "main_ablation"
                    and row.get("variant") == MAIN_METHOD
                    and row.get("stage") == stage
                    and row.get("max_steams") == max_steams
                ]
                if not main_rows:
                    continue
                comparisons = []
                baseline_variants = sorted(
                    {
                        row.get("variant", "")
                        for row in rows
                        if row.get("table") == "baselines"
                        and row.get("stage") == stage
                        and row.get("max_steams") == max_steams
                    }
                )
                for variant in baseline_variants:
                    comparisons.append(
                        (
                            "baseline",
                            variant,
                            [
                                row
                                for row in rows
                                if row.get("table") == "baselines"
                                and row.get("variant") == variant
                                and row.get("stage") == stage
                                and row.get("max_steams") == max_steams
                            ],
                        )
                    )
                ablation_variants = sorted(
                    {
                        row.get("variant", "")
                        for row in rows
                        if row.get("table") == "main_ablation"
                        and row.get("variant") != MAIN_METHOD
                        and row.get("stage") == stage
                        and row.get("max_steams") == max_steams
                    }
                )
                for variant in ablation_variants:
                    comparisons.append(
                        (
                            "ablation",
                            variant,
                            [
                                row
                                for row in rows
                                if row.get("table") == "main_ablation"
                                and row.get("variant") == variant
                                and row.get("stage") == stage
                                and row.get("max_steams") == max_steams
                            ],
                        )
                    )
                for comparison_group, variant, other_rows in comparisons:
                    if not other_rows:
                        continue
                    for metric in SIGNIFICANCE_METRICS:
                        main_samples = self.samples_for(main_rows, metric)
                        other_samples = self.samples_for(other_rows, metric)
                        if not main_samples or not other_samples:
                            continue
                        main_mean = float(np.mean(main_samples))
                        other_mean = float(np.mean(other_samples))
                        delta = main_mean - other_mean
                        ci_low, ci_high = bootstrap_delta_ci(
                            main_samples,
                            other_samples,
                            rng,
                            self.args.significance_resamples,
                        )
                        p_value = permutation_p_value(
                            main_samples,
                            other_samples,
                            rng,
                            self.args.significance_resamples,
                        )
                        lower_is_better = metric in LOWER_IS_BETTER
                        main_better = delta < 0.0 if lower_is_better else delta > 0.0
                        out_rows.append(
                            {
                                "comparison_group": comparison_group,
                                "main_variant": MAIN_METHOD,
                                "other_variant": variant,
                                "stage": stage,
                                "max_steams": max_steams,
                                "metric": metric,
                                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                                "main_n": len(main_samples),
                                "other_n": len(other_samples),
                                "main_mean": main_mean,
                                "other_mean": other_mean,
                                "delta_main_minus_other": delta,
                                "bootstrap_ci95_low": ci_low,
                                "bootstrap_ci95_high": ci_high,
                                "p_value": p_value,
                                "main_better": bool(main_better),
                            }
                        )
        return add_holm_correction(out_rows)

    def write_tables(self):
        if self.args.dry_run:
            self.write_manifest()
            return
        rows = self.collect_eval_rows()
        write_csv(self.tables_dir / "all_eval_summary_rows.csv", rows)
        baseline_rows = [row for row in rows if row.get("table") == "baselines"]
        ablation_rows = [row for row in rows if row.get("table") == "main_ablation"]
        main_rows = baseline_rows + [row for row in ablation_rows if row.get("variant") == MAIN_METHOD]
        sweep_rows = [row for row in rows if row.get("table") == "reward_sweep"]
        robustness_rows = [row for row in rows if row.get("table") == "robustness"]
        write_csv(
            self.tables_dir / "baseline_results.csv",
            self.aggregate_rows(baseline_rows, ["variant", "policy", "stage", "max_steams"]),
        )
        write_csv(
            self.tables_dir / "main_results.csv",
            self.aggregate_rows(main_rows, ["variant", "policy", "stage", "max_steams"]),
        )
        write_csv(
            self.tables_dir / "ablation_results.csv",
            self.aggregate_rows(ablation_rows, ["variant", "policy", "stage", "max_steams"]),
        )
        write_csv(
            self.tables_dir / "tradeoff_sweep.csv",
            self.aggregate_rows(sweep_rows, ["sweep", "variant", "policy", "stage", "max_steams"]),
        )
        write_csv(
            self.tables_dir / "robustness_results.csv",
            self.aggregate_rows(robustness_rows, ["scenario", "variant", "policy", "stage", "max_steams"]),
        )
        write_csv(self.tables_dir / "significance_tests.csv", self.significance_rows(rows))
        self.write_manifest()
        print(f"Wrote paper tables to {self.tables_dir}", flush=True)

    def write_manifest(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2, sort_keys=True)

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.train_main_and_ablations()
        self.train_reward_sweep()
        self.eval_baselines()
        self.eval_main_and_ablations()
        self.eval_reward_sweep()
        self.eval_robustness()
        self.visualize()
        self.runtime()
        self.write_tables()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="One-click V12 supplemental suite for small-paper experiments."
    )
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--run-root", default="runs/v12_small_paper_suite")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-jobs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-stage-episodes", help="Optional training curriculum override.")
    parser.add_argument("--eval-seeds", default="100,101,102")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=3200)
    parser.add_argument("--stages", default="multi_low,multi_realistic,multi_hard,multi_extreme")
    parser.add_argument("--decision-dt-seconds", type=float, default=0.05)
    parser.add_argument("--response-sla-seconds", type=float)
    parser.add_argument("--demo-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep-mode", choices=["full", "compact", "none"], default="full")
    parser.add_argument("--robust-stages", default="multi_hard,multi_extreme")
    parser.add_argument("--visual-stages", default="multi_realistic,multi_hard,multi_extreme")
    parser.add_argument("--visual-seed", type=int, default=100)
    parser.add_argument("--runtime-stages", default="multi_hard,multi_extreme")
    parser.add_argument("--runtime-steps", type=int, default=1600)
    parser.add_argument("--significance-resamples", type=int, default=10000)
    parser.add_argument("--significance-seed", type=int, default=20260610)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-visuals", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--fail-on-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true", help="Smoke-sized suite for checking wiring.")
    parser.add_argument("--dry-run", action="store_true", help="Print and record commands without executing them.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    SuiteRunner(args).run()


if __name__ == "__main__":
    main()
