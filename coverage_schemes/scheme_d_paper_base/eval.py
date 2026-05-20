import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import deep_update, load_config, set_global_seeds
from .env import ShangZengEnv
from .policies import build_policy, checkpoint_config


def configure_env_from_config(env, config):
    env.set_target_selector(config.get("target_selector", "risk_aware"))
    env.potential_shaping_enabled = bool(config.get("potential_shaping", True))
    env.best_progress_enabled = bool(config.get("best_progress_reward", True))
    env.material_observation_enabled = bool(config.get("material_observation", True))
    env.material_tv_reward_enabled = bool(config.get("material_tv_reward", False))
    env.material_tv_reward_gain = float(config.get("material_tv_reward_gain", env.material_tv_reward_gain))
    env.action_penalty_enabled = bool(config.get("action_smoothing_penalty", True))
    env.action_delay_steps = int(config.get("action_delay_steps", 0))
    env.action_noise_std = float(config.get("action_noise_std", 0.0))
    env.domain_randomization_enabled = bool(config.get("domain_randomization", False))
    env.domain_randomization_scale = float(config.get("domain_randomization_scale", env.domain_randomization_scale))
    return env


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Headless evaluation for paper experiments.")
    parser.add_argument("--config", help="Optional JSON config.")
    parser.add_argument("--model", help="Checkpoint for PPO policy.")
    parser.add_argument("--policy", default="ppo", choices=["ppo", "random", "nearest", "oldest", "distance_age", "risk_aware", "dynamic_weighted", "horizon2", "horizon3", "aco_tsp"])
    parser.add_argument("--method", help="Method name written to CSV.")
    parser.add_argument("--stage", default="multi_realistic", choices=["single_easy", "single_precision", "multi_low", "multi_realistic", "multi_hard", "multi_extreme"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs/eval/results.csv")
    parser.add_argument("--model-path", help="MuJoCo XML path.")
    parser.add_argument("--target-selector", choices=["nearest", "risk_aware"], help="Target features/reward-shaping selector.")
    parser.add_argument("--max-steams", type=int, help="Override active steam capacity after curriculum configuration.")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--demo-mode", action="store_true", help="Keep spawning after success instead of normal termination.")
    return parser


def evaluate_episode(env, policy, seed, max_steps, demo_mode=False):
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

        if hasattr(policy, "reset_recurrent") and (
            info.get("covered", False) or info.get("missed_count", 0) > previous_missed
        ):
            policy.reset_recurrent()
        previous_missed = info.get("missed_count", 0)

        if demo_mode and terminated:
            terminated = False
        if terminated or truncated:
            break

    return {
        "steps": step + 1,
        "episode_reward": float(total_reward),
        "coverage_rate": float(info.get("coverage_rate", 0.0)),
        "success_count": int(info.get("success_count", 0)),
        "spawned_count": int(info.get("spawned_count", 0)),
        "missed_count": int(info.get("missed_count", 0)),
        "cover_latency": float(info.get("cover_latency", 0.0)),
        "last_cover_latency": float(info.get("last_cover_latency", 0.0)),
        "target_distance": float(info.get("target_distance", 0.0)),
        "target_selector": str(info.get("target_selector", "")),
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
        "selected_target_risk_score": float(info.get("selected_target_risk_score", 0.0)),
        "potential_shaping_enabled": bool(info.get("potential_shaping_enabled", True)),
        "best_progress_enabled": bool(info.get("best_progress_enabled", True)),
        "material_observation_enabled": bool(info.get("material_observation_enabled", True)),
        "material_tv_reward_enabled": bool(info.get("material_tv_reward_enabled", False)),
        "action_penalty_enabled": bool(info.get("action_penalty_enabled", True)),
        "action_delay_steps": int(info.get("action_delay_steps", 0)),
        "action_noise_std": float(info.get("action_noise_std", 0.0)),
        "domain_randomization_enabled": bool(info.get("domain_randomization_enabled", False)),
        "spawn_burst_probability": float(info.get("spawn_burst_probability", 0.0)),
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
    if args.target_selector is not None:
        config["target_selector"] = args.target_selector
    set_global_seeds(args.seed)

    env = ShangZengEnv(
        model_path=config["model_path"],
        max_episode_steps=args.steps,
        target_selector=config.get("target_selector", "risk_aware"),
        steam_attention_observation=config.get("use_steam_attention", False),
        material_map_observation=config.get("use_material_map", False),
    )
    configure_env_from_config(env, config)
    env.configure_curriculum(args.stage)
    if args.max_steams is not None:
        env.max_steams = int(args.max_steams)
    env.max_episode_steps = args.steps
    if args.demo_mode:
        env.target_success_count = max(env.target_success_count, args.steps)
        env.target_coverage = 1.0

    policy = build_policy(args.policy, env, model_path=args.model, deterministic=not args.stochastic)
    method = args.method or policy.name
    rows = []
    for episode in range(args.episodes):
        ep_seed = args.seed + episode
        metrics = evaluate_episode(env, policy, ep_seed, args.steps, demo_mode=args.demo_mode)
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
            f"Cov:{metrics['coverage_rate']:.2f} | Miss:{metrics['missed_count']}",
            flush=True,
        )

    write_rows(args.output, rows)
    summary = {
        "episodes": len(rows),
        "coverage_mean": float(np.mean([row["coverage_rate"] for row in rows])) if rows else 0.0,
        "missed_mean": float(np.mean([row["missed_count"] for row in rows])) if rows else 0.0,
        "reward_mean": float(np.mean([row["episode_reward"] for row in rows])) if rows else 0.0,
        "output": str(args.output),
    }
    with Path(args.output).with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
