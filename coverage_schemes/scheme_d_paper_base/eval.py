import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import deep_update, load_config, set_global_seeds
from .env import ShangZengEnv
from .policies import build_policy, checkpoint_config


def env_step_seconds(env):
    try:
        step_seconds = float(env.model.opt.timestep)
    except (AttributeError, TypeError, ValueError):
        step_seconds = 1.0
    return step_seconds if step_seconds > 0.0 else 1.0


def resolve_metric_step_seconds(env, config=None, override=None):
    for value in (override, (config or {}).get("decision_dt_seconds")):
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return value
    return env_step_seconds(env)


def coverage_timing_metrics(steps, success_count, spawned_count, cover_latency_steps, step_seconds):
    episode_time_seconds = float(steps) * float(step_seconds)
    cover_latency_seconds = float(cover_latency_steps) * float(step_seconds)
    return {
        "step_seconds": float(step_seconds),
        "episode_time_seconds": episode_time_seconds,
        "cover_latency_seconds": cover_latency_seconds,
        "per_point_cover_speed": (
            1.0 / cover_latency_seconds if success_count > 0 and cover_latency_seconds > 0.0 else 0.0
        ),
        "covered_per_second": (
            float(success_count) / episode_time_seconds if episode_time_seconds > 0.0 else 0.0
        ),
        "spawned_per_second": (
            float(spawned_count) / episode_time_seconds if episode_time_seconds > 0.0 else 0.0
        ),
        "covered_per_100_steps": float(success_count) * 100.0 / float(steps) if steps > 0 else 0.0,
    }


def configure_env_from_config(env, config):
    env.set_target_selector(config.get("target_selector", "risk_aware"))
    env.potential_shaping_enabled = bool(config.get("potential_shaping", True))
    env.best_progress_enabled = bool(config.get("best_progress_reward", True))
    env.cover_reward_scale = float(config.get("cover_reward_scale", env.cover_reward_scale))
    env.quick_cover_bonus_scale = float(config.get("quick_cover_bonus_scale", env.quick_cover_bonus_scale))
    env.precision_bonus_scale = float(config.get("precision_bonus_scale", env.precision_bonus_scale))
    env.potential_gain_scale = float(config.get("potential_gain_scale", env.potential_gain_scale))
    env.best_progress_gain_scale = float(config.get("best_progress_gain_scale", env.best_progress_gain_scale))
    env.active_steam_penalty_scale = float(config.get("active_steam_penalty_scale", env.active_steam_penalty_scale))
    env.age_penalty_scale = float(config.get("age_penalty_scale", env.age_penalty_scale))
    env.material_observation_enabled = bool(config.get("material_observation", True))
    env.material_tv_reward_enabled = bool(config.get("material_tv_reward", False))
    env.material_tv_reward_gain = float(config.get("material_tv_reward_gain", env.material_tv_reward_gain))
    env.action_penalty_enabled = bool(config.get("action_smoothing_penalty", True))
    env.latency_first_reward_enabled = bool(config.get("latency_first_reward", False))
    env.cover_latency_penalty_gain = float(config.get("cover_latency_penalty_gain", env.cover_latency_penalty_gain))
    env.oldest_active_penalty_gain = float(config.get("oldest_active_penalty_gain", env.oldest_active_penalty_gain))
    env.backlog_penalty_gain = float(config.get("backlog_penalty_gain", env.backlog_penalty_gain))
    env.response_sla_steps = int(config.get("response_sla_steps", env.response_sla_steps))
    env.response_sla_bonus = float(config.get("response_sla_bonus", env.response_sla_bonus))
    env.response_sla_miss_penalty = float(config.get("response_sla_miss_penalty", env.response_sla_miss_penalty))
    env.action_delay_steps = int(config.get("action_delay_steps", 0))
    env.action_noise_std = float(config.get("action_noise_std", 0.0))
    env.domain_randomization_enabled = bool(config.get("domain_randomization", False))
    env.domain_randomization_scale = float(config.get("domain_randomization_scale", env.domain_randomization_scale))
    env.thermal_spawn_enabled = bool(config.get("thermal_spawn", True))
    env.thermal_hotspot_count = int(config.get("thermal_hotspot_count", env.thermal_hotspot_count))
    env.thermal_hotspot_sigma = float(config.get("thermal_hotspot_sigma", env.thermal_hotspot_sigma))
    env.thermal_hotspot_strength = float(config.get("thermal_hotspot_strength", env.thermal_hotspot_strength))
    env.thermal_background_weight = float(config.get("thermal_background_weight", env.thermal_background_weight))
    env.thermal_drift_std = float(config.get("thermal_drift_std", env.thermal_drift_std))
    env.thermal_refresh_probability = float(config.get("thermal_refresh_probability", env.thermal_refresh_probability))
    env.thermal_lifetime_steps = int(config.get("thermal_lifetime_steps", env.thermal_lifetime_steps))
    env.thermal_recent_spawn_radius = float(config.get("thermal_recent_spawn_radius", env.thermal_recent_spawn_radius))
    env.thermal_recent_spawn_suppression = float(
        config.get("thermal_recent_spawn_suppression", env.thermal_recent_spawn_suppression)
    )
    env.thermal_recent_spawn_memory = int(config.get("thermal_recent_spawn_memory", env.thermal_recent_spawn_memory))
    env.burst_lull_spawn_enabled = bool(config.get("burst_lull_spawn", False))
    env.burst_lull_lull_steps = int(config.get("burst_lull_lull_steps", env.burst_lull_lull_steps))
    env.burst_lull_charge_steps = int(config.get("burst_lull_charge_steps", env.burst_lull_charge_steps))
    env.burst_lull_sparse_threshold = int(
        config.get("burst_lull_sparse_threshold", env.burst_lull_sparse_threshold)
    )
    env.burst_lull_burst_min = int(config.get("burst_lull_burst_min", env.burst_lull_burst_min))
    env.burst_lull_burst_max = int(config.get("burst_lull_burst_max", env.burst_lull_burst_max))
    env.burst_lull_burst_interval_steps = int(
        config.get("burst_lull_burst_interval_steps", env.burst_lull_burst_interval_steps)
    )
    env.burst_lull_trickle_probability = float(
        config.get("burst_lull_trickle_probability", env.burst_lull_trickle_probability)
    )
    env.burst_lull_initial_burst = bool(config.get("burst_lull_initial_burst", env.burst_lull_initial_burst))
    env.spawn_history_observation_enabled = bool(
        config.get("use_spawn_history_observation", env.spawn_history_observation_enabled)
    )
    env.thermal_context_observation_enabled = bool(config.get("use_thermal_context_observation", False))
    env.route_summary_observation_enabled = bool(config.get("use_route_summary_observation", False))
    env.steam_attention_observation_enabled = bool(
        config.get("use_steam_attention", env.steam_attention_observation_enabled)
    )
    env.material_map_observation_enabled = bool(config.get("use_material_map", env.material_map_observation_enabled))
    env.attention_steam_count = int(config.get("attention_steam_count", env.attention_steam_count))
    env.attention_steam_dim = int(config.get("attention_steam_dim", env.attention_steam_dim))
    env.refresh_observation_space()
    return env


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Headless evaluation for paper experiments.")
    parser.add_argument("--config", help="Optional JSON config.")
    parser.add_argument("--model", help="Checkpoint for PPO policy.")
    parser.add_argument("--policy", default="ppo", choices=["ppo", "random", "nearest", "oldest", "distance_age", "risk_aware", "dynamic_weighted", "horizon2", "horizon3", "aco_tsp", "planner_ensemble"])
    parser.add_argument("--method", help="Method name written to CSV.")
    parser.add_argument("--stage", default="multi_realistic", choices=["single_easy", "single_precision", "multi_low", "multi_realistic", "multi_hard", "multi_extreme"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs/eval/results.csv")
    parser.add_argument("--model-path", help="MuJoCo XML path.")
    parser.add_argument("--device", help="PPO device override: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--target-selector", choices=["nearest", "risk_aware"], help="Target features/reward-shaping selector.")
    parser.add_argument("--max-steams", type=int, help="Override active steam capacity after curriculum configuration.")
    parser.add_argument("--decision-dt-seconds", type=float, help="Real high-level control period used to report time metrics.")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--demo-mode", action="store_true", help="Keep spawning after success instead of normal termination.")
    return parser


def evaluate_episode(env, policy, seed, max_steps, demo_mode=False, metric_step_seconds=None):
    obs, info = env.reset(seed=seed)
    policy.reset()
    total_reward = 0.0
    previous_missed = info.get("missed_count", 0)
    action_delta = []
    action_l2 = []
    previous_action = np.zeros(env.action_space.shape[0], dtype=np.float32)

    for step in range(max_steps):
        action = policy.act(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action_delta.append(float(np.linalg.norm(action - previous_action)))
        action_l2.append(float(np.dot(action, action)))
        previous_action = action.copy()

        if hasattr(policy, "reset_recurrent"):
            reset_for_cover = info.get("covered", False) and getattr(policy, "recurrent_reset_on_cover", True)
            reset_for_miss = (
                info.get("missed_count", 0) > previous_missed
                and getattr(policy, "recurrent_reset_on_miss", True)
            )
            if reset_for_cover or reset_for_miss:
                policy.reset_recurrent()
        previous_missed = info.get("missed_count", 0)

        if demo_mode and terminated:
            terminated = False
        if terminated or truncated:
            break

    steps = step + 1
    success_count = int(info.get("success_count", 0))
    spawned_count = int(info.get("spawned_count", 0))
    cover_latency_steps = float(info.get("cover_latency", 0.0))
    sim_step_seconds = env_step_seconds(env)
    step_seconds = sim_step_seconds if metric_step_seconds is None else float(metric_step_seconds)
    timing = coverage_timing_metrics(
        steps=steps,
        success_count=success_count,
        spawned_count=spawned_count,
        cover_latency_steps=cover_latency_steps,
        step_seconds=step_seconds,
    )
    step_seconds = timing["step_seconds"]

    metrics = {
        "sim_step_seconds": float(sim_step_seconds),
        "decision_dt_seconds": float(step_seconds),
        "steps": steps,
        "episode_reward": float(total_reward),
        "coverage_rate": float(info.get("coverage_rate", 0.0)),
        "success_count": success_count,
        "spawned_count": spawned_count,
        "missed_count": int(info.get("missed_count", 0)),
        "cover_latency": cover_latency_steps,
        "last_cover_latency": float(info.get("last_cover_latency", 0.0)),
        "cover_latency_p50": float(info.get("cover_latency_p50", 0.0)),
        "cover_latency_p90": float(info.get("cover_latency_p90", 0.0)),
        "cover_latency_p95": float(info.get("cover_latency_p95", 0.0)),
        "cover_latency_max": float(info.get("cover_latency_max", 0.0)),
        "cover_latency_p50_seconds": float(info.get("cover_latency_p50", 0.0)) * step_seconds,
        "cover_latency_p90_seconds": float(info.get("cover_latency_p90", 0.0)) * step_seconds,
        "cover_latency_p95_seconds": float(info.get("cover_latency_p95", 0.0)) * step_seconds,
        "cover_latency_max_seconds": float(info.get("cover_latency_max", 0.0)) * step_seconds,
        "response_sla_steps": int(info.get("response_sla_steps", 0)),
        "response_sla_seconds": float(info.get("response_sla_steps", 0)) * step_seconds,
        "response_sla_success_count": int(info.get("response_sla_success_count", 0)),
        "response_sla_miss_count": int(info.get("response_sla_miss_count", 0)),
        "response_sla_success_rate": float(info.get("response_sla_success_rate", 0.0)),
        "active_steam_mean": float(info.get("active_steam_mean", 0.0)),
        "active_steam_max": int(info.get("active_steam_max", 0)),
        "oldest_active_age": float(info.get("oldest_active_age", 0.0)),
        "oldest_active_age_max": float(info.get("oldest_active_age_max", 0.0)),
        "oldest_active_age_max_seconds": float(info.get("oldest_active_age_max", 0.0)) * step_seconds,
        "target_distance": float(info.get("target_distance", 0.0)),
        "target_selector": str(info.get("target_selector", "")),
        "spawn_history_observation_enabled": bool(info.get("spawn_history_observation_enabled", False)),
        "thermal_context_observation_enabled": bool(info.get("thermal_context_observation_enabled", False)),
        "route_summary_observation_enabled": bool(info.get("route_summary_observation_enabled", False)),
        "steam_attention_observation_enabled": bool(info.get("steam_attention_observation_enabled", False)),
        "material_map_observation_enabled": bool(info.get("material_map_observation_enabled", False)),
        "selected_target_id": int(info.get("selected_target_id", -1)),
        "selected_target_x": float(info.get("selected_target_x", 0.0)),
        "selected_target_y": float(info.get("selected_target_y", 0.0)),
        "selected_target_distance": float(info.get("selected_target_distance", 0.0)),
        "nearest_target_distance": float(info.get("nearest_target_distance", 0.0)),
        "selected_target_age_score": float(info.get("selected_target_age_score", 0.0)),
        "selected_target_distance_score": float(info.get("selected_target_distance_score", 0.0)),
        "selected_target_material_score": float(info.get("selected_target_material_score", 0.0)),
        "selected_target_reachability_score": float(info.get("selected_target_reachability_score", 0.0)),
        "selected_target_thermal_score": float(info.get("selected_target_thermal_score", 0.0)),
        "selected_target_risk_score": float(info.get("selected_target_risk_score", 0.0)),
        "steps_since_cover": int(info.get("steps_since_cover", 0)),
        "route_active_density": float(info.get("route_active_density", 0.0)),
        "route_max_age_score": float(info.get("route_max_age_score", 0.0)),
        "route_mean_age_score": float(info.get("route_mean_age_score", 0.0)),
        "route_nearest_distance_score": float(info.get("route_nearest_distance_score", 0.0)),
        "route_target_thermal_score": float(info.get("route_target_thermal_score", 0.0)),
        "route_confidence": float(info.get("route_confidence", 0.0)),
        "route_spawn_ready": float(info.get("route_spawn_ready", 0.0)),
        "route_stagnation_score": float(info.get("route_stagnation_score", 0.0)),
        "potential_shaping_enabled": bool(info.get("potential_shaping_enabled", True)),
        "best_progress_enabled": bool(info.get("best_progress_enabled", True)),
        "latency_first_reward_enabled": bool(info.get("latency_first_reward_enabled", False)),
        "material_observation_enabled": bool(info.get("material_observation_enabled", True)),
        "material_tv_reward_enabled": bool(info.get("material_tv_reward_enabled", False)),
        "action_penalty_enabled": bool(info.get("action_penalty_enabled", True)),
        "action_delay_steps": int(info.get("action_delay_steps", 0)),
        "action_noise_std": float(info.get("action_noise_std", 0.0)),
        "domain_randomization_enabled": bool(info.get("domain_randomization_enabled", False)),
        "spawn_burst_probability": float(info.get("spawn_burst_probability", 0.0)),
        "burst_lull_spawn_enabled": bool(info.get("burst_lull_spawn_enabled", False)),
        "burst_lull_phase": str(info.get("burst_lull_phase", "")),
        "burst_lull_lull_remaining": int(info.get("burst_lull_lull_remaining", 0)),
        "burst_lull_charge": int(info.get("burst_lull_charge", 0)),
        "burst_lull_charge_score": float(info.get("burst_lull_charge_score", 0.0)),
        "burst_lull_pending_count": int(info.get("burst_lull_pending_count", 0)),
        "burst_lull_next_spawn_delay": int(info.get("burst_lull_next_spawn_delay", 0)),
        "burst_lull_burst_interval_steps": int(info.get("burst_lull_burst_interval_steps", 0)),
        "last_burst_spawn_count": int(info.get("last_burst_spawn_count", 0)),
        "thermal_spawn_enabled": bool(info.get("thermal_spawn_enabled", False)),
        "thermal_hotspot_count": int(info.get("thermal_hotspot_count", 0)),
        "thermal_background_weight": float(info.get("thermal_background_weight", 0.0)),
        "thermal_hotspot_strength": float(info.get("thermal_hotspot_strength", 0.0)),
        "thermal_peak_x": float(info.get("thermal_peak_x", 0.0)),
        "thermal_peak_y": float(info.get("thermal_peak_y", 0.0)),
        "thermal_peak_score": float(info.get("thermal_peak_score", 0.0)),
        "mean_material_height": float(info.get("mean_material_height", 0.0)),
        "max_material_height": float(info.get("max_material_height", 0.0)),
        "height_uniformity": float(info.get("height_uniformity", 0.0)),
        "overfill_penalty": float(info.get("overfill_penalty", 0.0)),
        "material_hole_loss": float(info.get("material_hole_loss", 0.0)),
        "material_tv_loss": float(info.get("material_tv_loss", 0.0)),
        "material_quality_loss": float(info.get("material_quality_loss", 0.0)),
        "action_delta_mean": float(np.mean(action_delta)) if action_delta else 0.0,
        "action_l2_mean": float(np.mean(action_l2)) if action_l2 else 0.0,
    }
    metrics.update(timing)
    return metrics


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    if args.policy == "ppo" and args.model:
        config = deep_update(config, checkpoint_config(args.model))
    if args.model_path:
        config["model_path"] = args.model_path
    if args.device:
        config["device"] = args.device
    if args.target_selector is not None:
        config["target_selector"] = args.target_selector
    if args.decision_dt_seconds is not None:
        config["decision_dt_seconds"] = args.decision_dt_seconds
    set_global_seeds(args.seed)

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
    env.configure_curriculum(args.stage)
    if args.max_steams is not None:
        env.max_steams = int(args.max_steams)
    env.max_episode_steps = args.steps
    if args.demo_mode:
        env.target_success_count = max(env.target_success_count, args.steps)
        env.target_coverage = 1.0
    metric_step_seconds = resolve_metric_step_seconds(env, config=config, override=args.decision_dt_seconds)

    policy = build_policy(
        args.policy,
        env,
        model_path=args.model,
        deterministic=not args.stochastic,
        config_override=config,
    )
    method = args.method or policy.name
    rows = []
    for episode in range(args.episodes):
        ep_seed = args.seed + episode
        metrics = evaluate_episode(
            env,
            policy,
            ep_seed,
            args.steps,
            demo_mode=args.demo_mode,
            metric_step_seconds=metric_step_seconds,
        )
        row = {
            "method": method,
            "policy": args.policy,
            "checkpoint": str(args.model or ""),
            "seed": args.seed,
            "episode_seed": ep_seed,
            "episode": episode + 1,
            "stage": args.stage,
            "max_steams": env.max_steams,
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"Eval {episode + 1}/{args.episodes} | {method} | "
            f"stage:{args.stage} | R:{metrics['episode_reward']:.1f} | "
            f"Cov:{metrics['coverage_rate']:.2f} | "
            f"Lat:{metrics['cover_latency_seconds']:.3f}s | "
            f"P90:{metrics['cover_latency_p90_seconds']:.3f}s | "
            f"SLA:{metrics['response_sla_success_rate']:.2f} | "
            f"Rate:{metrics['covered_per_second']:.2f}/s | "
            f"Miss:{metrics['missed_count']}",
            flush=True,
        )

    write_rows(args.output, rows)
    summary = {
        "episodes": len(rows),
        "coverage_mean": float(np.mean([row["coverage_rate"] for row in rows])) if rows else 0.0,
        "cover_latency_seconds_mean": float(np.mean([row["cover_latency_seconds"] for row in rows])) if rows else 0.0,
        "cover_latency_p90_seconds_mean": float(np.mean([row["cover_latency_p90_seconds"] for row in rows])) if rows else 0.0,
        "response_sla_success_rate_mean": float(np.mean([row["response_sla_success_rate"] for row in rows])) if rows else 0.0,
        "active_steam_mean": float(np.mean([row["active_steam_mean"] for row in rows])) if rows else 0.0,
        "covered_per_second_mean": float(np.mean([row["covered_per_second"] for row in rows])) if rows else 0.0,
        "per_point_cover_speed_mean": float(np.mean([row["per_point_cover_speed"] for row in rows])) if rows else 0.0,
        "missed_mean": float(np.mean([row["missed_count"] for row in rows])) if rows else 0.0,
        "reward_mean": float(np.mean([row["episode_reward"] for row in rows])) if rows else 0.0,
        "output": str(args.output),
    }
    with Path(args.output).with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
