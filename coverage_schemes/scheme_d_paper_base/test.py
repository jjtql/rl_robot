import argparse
import time
from collections import deque

import mujoco.viewer

from .config import deep_update, load_config, set_global_seeds
from .env import ShangZengEnv
from .eval import configure_env_from_config
from .policies import build_policy, checkpoint_config


POLICY_CHOICES = (
    "ppo",
    "random",
    "nearest",
    "oldest",
    "distance_age",
    "risk_aware",
    "dynamic_weighted",
    "horizon2",
    "horizon3",
    "aco_tsp",
    "planner_ensemble",
)


def load_viewer_config(config_path=None, model_path=None, policy_name="ppo"):
    config = load_config(config_path)
    if policy_name == "ppo" and model_path:
        config = deep_update(config, checkpoint_config(model_path))
    return config


def test_model(
    model_path=None,
    policy_name="ppo",
    config_path=None,
    stage="single_easy",
    target_selector="risk_aware",
    model_xml_path=None,
    max_steams=None,
    seed=0,
    deterministic=True,
    max_steps=10000,
    summary_interval=50,
    demo_mode=True,
    sleep_seconds=0.02,
):
    config = load_viewer_config(config_path=config_path, model_path=model_path, policy_name=policy_name)
    if model_xml_path:
        config["model_path"] = model_xml_path
    if target_selector:
        config["target_selector"] = target_selector
    set_global_seeds(seed)

    env = ShangZengEnv(
        model_path=config["model_path"],
        max_episode_steps=max_steps if demo_mode else min(max_steps, int(config.get("episode_steps", 800))),
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
        env.max_steams = int(max_steams)
    if demo_mode:
        env.max_episode_steps = max_steps
        env.target_success_count = max(env.target_success_count, max_steps)
        env.target_coverage = 1.0

    policy = build_policy(
        policy_name,
        env,
        model_path=model_path,
        deterministic=deterministic,
        config_override=config,
    )

    print(f"Policy: {policy_name} ({getattr(policy, 'name', policy_name)})")
    print(f"Model: {model_path or '-'}")
    print(f"Stage: {stage} | max_steams:{env.max_steams} | cover_radius:{env.cover_radius}")
    print(f"Target selector: {env.target_selector}")
    print(f"Steam attention: {config.get('use_steam_attention', False)}")
    print(f"Burst-lull spawn: {env.burst_lull_spawn_enabled}")
    print(f"Latency-first reward: {env.latency_first_reward_enabled} | SLA steps:{env.response_sla_steps}")
    print(f"Demo mode: {'on' if demo_mode else 'off'} | max_steps:{max_steps}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, info = env.reset(seed=seed)
        policy.reset()
        previous_missed = info.get("missed_count", 0)
        recent_events = deque(maxlen=max(summary_interval, 1) * 5)

        for step in range(max_steps):
            action = policy.act(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)

            reset_for_cover = info.get("covered", False) and getattr(policy, "recurrent_reset_on_cover", True)
            reset_for_miss = (
                info.get("missed_count", 0) > previous_missed
                and getattr(policy, "recurrent_reset_on_miss", True)
            )
            if reset_for_cover or reset_for_miss:
                if hasattr(policy, "reset_recurrent"):
                    policy.reset_recurrent()
            previous_missed = info.get("missed_count", 0)
            recent_events.append(1 if info.get("covered", False) else 0)

            viewer.sync()
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))

            if step % summary_interval == 0:
                dist = info.get("target_distance", 0.0)
                recent_covers = sum(recent_events)
                print(
                    f"step:{step:5d} | R:{reward:+.2f} | dist:{dist:.3f} | "
                    f"cov:{info['coverage_rate']:.2f} | recent_cover:{recent_covers} | "
                    f"covered:{info.get('success_count', 0)}/{info.get('spawned_count', 0)} | "
                    f"steam:{info['steam_count']} | "
                    f"lat:{info.get('cover_latency', 0):.0f} | "
                    f"p90:{info.get('cover_latency_p90', 0):.0f} | "
                    f"sla:{info.get('response_sla_success_rate', 0):.2f}"
                )

            if demo_mode and terminated:
                terminated = False

            if terminated or truncated:
                print(
                    f"\n{'=' * 40}\n"
                    f"Episode end | {'SUCCESS' if terminated else 'TIMEOUT'}\n"
                    f"  Covered: {info.get('success_count', 0)}/{info.get('spawned_count', 0)}\n"
                    f"  Coverage: {info['coverage_rate']:.2f}\n"
                    f"  Latency: mean={info.get('cover_latency', 0):.0f}, "
                    f"p90={info.get('cover_latency_p90', 0):.0f}, "
                    f"sla={info.get('response_sla_success_rate', 0):.2f}\n"
                    f"{'=' * 40}\n"
                )
                obs, info = env.reset(seed=seed + step + 1)
                policy.reset()


def main():
    parser = argparse.ArgumentParser(description="Open a live MuJoCo viewer for PPO or rule-policy testing.")
    parser.add_argument("--model", help="PPO checkpoint. Required when --policy ppo.")
    parser.add_argument("--policy", default="ppo", choices=POLICY_CHOICES)
    parser.add_argument("--config", help="Optional JSON config.")
    parser.add_argument(
        "--stage",
        default="single_easy",
        choices=["single_easy", "single_precision", "multi_low", "multi_realistic", "multi_hard", "multi_extreme"],
    )
    parser.add_argument("--target-selector", default="risk_aware", choices=["nearest", "risk_aware"])
    parser.add_argument("--model-path", help="MuJoCo XML path.")
    parser.add_argument("--max-steams", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.02, help="Viewer delay per step. Use 0 for fastest playback.")
    parser.add_argument("--episode-reset", action="store_true", help="Reset on normal episode success/timeout.")
    args = parser.parse_args()
    if args.policy == "ppo" and not args.model:
        parser.error("--model is required when --policy ppo")

    test_model(
        model_path=args.model,
        policy_name=args.policy,
        config_path=args.config,
        stage=args.stage,
        target_selector=args.target_selector,
        model_xml_path=args.model_path,
        max_steams=args.max_steams,
        seed=args.seed,
        deterministic=not args.stochastic,
        max_steps=args.steps,
        summary_interval=args.interval,
        demo_mode=not args.episode_reset,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
