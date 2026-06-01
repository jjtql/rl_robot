import argparse
import csv
from pathlib import Path

import numpy as np

from .config import deep_update, load_config, set_global_seeds
from .env import ShangZengEnv
from .eval import configure_env_from_config, evaluate_episode, resolve_metric_step_seconds
from .policies import build_policy, checkpoint_config


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run a matrix of paper evaluations.")
    parser.add_argument("--config", help="Optional JSON config.")
    parser.add_argument("--model", help="Checkpoint for PPO policy.")
    parser.add_argument("--model-path", help="MuJoCo XML path.")
    parser.add_argument("--device", help="PPO device override: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--target-selector", choices=["nearest", "risk_aware"], help="Target features/reward-shaping selector.")
    parser.add_argument("--max-steams", help="Comma-separated active steam capacities for generalization tests, e.g. 3,4,6.")
    parser.add_argument("--policies", default="random,nearest,oldest,distance_age,risk_aware,dynamic_weighted,horizon2,horizon3,deadline_horizon2,aco_tsp,planner_ensemble,ppo")
    parser.add_argument("--stages", default="multi_low,multi_realistic")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--decision-dt-seconds", type=float, help="Real high-level control period used to report time metrics.")
    parser.add_argument("--response-sla-seconds", type=float, help="Override response SLA in real high-level control seconds.")
    parser.add_argument("--output-dir", default="runs/matrix")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--demo-mode", action="store_true")
    return parser


def summarize(rows):
    metrics = [
        "episode_reward",
        "coverage_rate",
        "missed_count",
        "cover_latency",
        "cover_latency_seconds",
        "cover_latency_p50_seconds",
        "cover_latency_p90_seconds",
        "cover_latency_p95_seconds",
        "cover_latency_max_seconds",
        "response_sla_success_rate",
        "active_steam_mean",
        "active_steam_max",
        "oldest_active_age_max_seconds",
        "per_point_cover_speed",
        "covered_per_second",
        "spawned_per_second",
        "covered_per_100_steps",
        "target_distance",
        "selected_target_risk_score",
        "selected_target_thermal_score",
        "route_confidence",
        "route_stagnation_score",
        "height_uniformity",
        "overfill_penalty",
        "material_hole_loss",
        "material_tv_loss",
        "material_quality_loss",
    ]
    out = {}
    for metric in metrics:
        values = np.array([row[metric] for row in rows], dtype=np.float64)
        out[f"{metric}_mean"] = float(values.mean()) if values.size else 0.0
        out[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    if args.model:
        config = deep_update(config, checkpoint_config(args.model))
    if args.model_path:
        config["model_path"] = args.model_path
    if args.device:
        config["device"] = args.device
    if args.target_selector is not None:
        config["target_selector"] = args.target_selector
    if args.decision_dt_seconds is not None:
        config["decision_dt_seconds"] = args.decision_dt_seconds
    if args.response_sla_seconds is not None:
        config["response_sla_seconds"] = args.response_sla_seconds

    policies = parse_csv_list(args.policies)
    if "ppo" in policies and not args.model:
        raise ValueError("--model is required when --policies includes ppo")
    stages = parse_csv_list(args.stages)
    seeds = [int(item) for item in parse_csv_list(args.seeds)]
    max_steams_values = [int(item) for item in parse_csv_list(args.max_steams)] if args.max_steams else [None]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary_rows = []

    for policy_name in policies:
        for stage in stages:
            for max_steams in max_steams_values:
                for seed in seeds:
                    set_global_seeds(seed)
                    env = ShangZengEnv(
                        model_path=config["model_path"],
                        max_episode_steps=args.steps,
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
                    if max_steams is not None:
                        env.max_steams = max_steams
                    env.max_episode_steps = args.steps
                    if args.demo_mode:
                        env.target_success_count = max(env.target_success_count, args.steps)
                        env.target_coverage = 1.0
                    metric_step_seconds = resolve_metric_step_seconds(
                        env,
                        config=config,
                        override=args.decision_dt_seconds,
                    )

                    policy = build_policy(
                        policy_name,
                        env,
                        model_path=args.model,
                        deterministic=not args.stochastic,
                        config_override=config,
                    )
                    policy_rows = []
                    for episode in range(args.episodes):
                        ep_seed = seed * 10_000 + episode
                        metrics = evaluate_episode(
                            env,
                            policy,
                            ep_seed,
                            args.steps,
                            demo_mode=args.demo_mode,
                            metric_step_seconds=metric_step_seconds,
                        )
                        row = {
                            "policy": policy_name,
                            "method": policy.name,
                            "stage": stage,
                            "max_steams": env.max_steams,
                            "seed": seed,
                            "episode_seed": ep_seed,
                            "episode": episode + 1,
                            "checkpoint": str(args.model or ""),
                        }
                        row.update(metrics)
                        policy_rows.append(row)
                        all_rows.append(row)
                    steam_tag = f"_max{env.max_steams}" if max_steams is not None else ""
                    per_run_path = output_dir / f"{policy_name}_{stage}{steam_tag}_seed{seed}.csv"
                    write_csv(per_run_path, policy_rows)
                    summary = {
                        "policy": policy_name,
                        "stage": stage,
                        "max_steams": env.max_steams,
                        "seed": seed,
                        "episodes": len(policy_rows),
                    }
                    summary.update(summarize(policy_rows))
                    summary_rows.append(summary)
                    print(
                        f"{policy_name} | {stage} | max_steams {env.max_steams} | seed {seed} | "
                        f"R:{summary['episode_reward_mean']:.2f} | Cov:{summary['coverage_rate_mean']:.2f} | "
                        f"Lat:{summary['cover_latency_seconds_mean']:.3f}s | "
                        f"P90:{summary['cover_latency_p90_seconds_mean']:.3f}s | "
                        f"SLA:{summary['response_sla_success_rate_mean']:.2f} | "
                        f"Rate:{summary['covered_per_second_mean']:.2f}/s | "
                        f"Miss:{summary['missed_count_mean']:.2f}",
                        flush=True,
                    )

    write_csv(output_dir / "all_rows.csv", all_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    print(f"Wrote {len(all_rows)} rows and {len(summary_rows)} summary rows to {output_dir}")


if __name__ == "__main__":
    main()
