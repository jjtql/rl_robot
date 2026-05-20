import argparse
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch

from .algo import PPOAgent
from .env import ShangZengEnv


def load_checkpoint(agent, model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        agent.model.load_state_dict(checkpoint["model"])
        if "icm" in checkpoint:
            agent.icm.load_state_dict(checkpoint["icm"])
    else:
        agent.model.load_state_dict(checkpoint)
    return checkpoint


def checkpoint_config(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint.get("config", {})
    return {}


def test_model(
    model_path,
    stage="single_easy",
    target_selector="risk_aware",
    max_steams=None,
    deterministic=True,
    max_steps=10000,
    summary_interval=50,
    demo_mode=True,
):
    checkpoint_cfg = checkpoint_config(model_path)
    env = ShangZengEnv(
        max_episode_steps=max_steps if demo_mode else 400,
        target_selector=target_selector,
        steam_attention_observation=checkpoint_cfg.get("use_steam_attention", False),
        material_map_observation=checkpoint_cfg.get("use_material_map", False),
    )
    env.configure_curriculum(stage)
    if max_steams is not None:
        env.max_steams = int(max_steams)
    if demo_mode:
        # Demo mode should keep spawning new steam points instead of resetting as
        # soon as the curriculum success condition is reached.
        env.max_episode_steps = max_steps
        env.target_success_count = max(env.target_success_count, max_steps)
        env.target_coverage = 1.0

    obs_dim = env.observation_space.shape[0]
    agent = PPOAgent(
        obs_dim,
        env.action_space.shape[0],
        seq_len=env.max_episode_steps,
        use_lstm=checkpoint_cfg.get("use_lstm", True),
        use_steam_attention=checkpoint_cfg.get("use_steam_attention", False),
        use_material_map=checkpoint_cfg.get("use_material_map", False),
    )

    print(f"Model: {model_path}")
    print(f"Stage: {stage} | steams:{env.max_steams} | cover_radius:{env.cover_radius}")
    print(f"Target selector: {env.target_selector}")
    print(f"Steam attention: {checkpoint_cfg.get('use_steam_attention', False)}")
    print(f"Material map: {checkpoint_cfg.get('use_material_map', False)}")
    print(f"Demo mode: {'on' if demo_mode else 'off'} | max_steps:{max_steps}")
    load_checkpoint(agent, model_path)
    agent.model.eval()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        s, info = env.reset()
        hx, cx = None, None
        previous_missed = info.get("missed_count", 0)
        recent_events = deque(maxlen=max(summary_interval, 1) * 5)

        for step in range(max_steps):
            with torch.no_grad():
                action, _, _, _, hx, cx = agent.select_action(s, hx, cx, deterministic=deterministic)

            s, r, terminated, truncated, info = env.step(action)
            if info.get("covered", False) or info.get("missed_count", 0) > previous_missed:
                hx, cx = None, None
            previous_missed = info.get("missed_count", 0)
            recent_events.append(1 if info.get("covered", False) else 0)
            viewer.sync()
            time.sleep(0.02)

            if step % summary_interval == 0:
                dist = info.get('target_distance', 0)
                recent_covers = sum(recent_events)
                print(
                    f"step:{step:5d} | R:{r:+.2f} | dist:{dist:.3f} | "
                    f"cov:{info['coverage_rate']:.2f} | recent_cover:{recent_covers} | "
                    f"covered:{info.get('success_count', 0)}/{info.get('spawned_count', 0)} | "
                    f"steam:{info['steam_count']}"
                )

            if demo_mode and terminated:
                terminated = False

            if terminated or truncated:
                print(
                    f"\n{'='*40}\n"
                    f"Episode end | {'SUCCESS' if terminated else 'TIMEOUT'}\n"
                    f"  Covered: {info.get('success_count', 0)}/{info.get('spawned_count', 0)}\n"
                    f"  Coverage: {info['coverage_rate']:.2f}\n"
                    f"{'='*40}\n"
                )
                hx, cx = None, None
                s, info = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="scheme_d_fixed_latest_full.pt")
    parser.add_argument("--stage", default="single_easy", choices=["single_easy", "single_precision", "multi_low", "multi_realistic", "multi_hard", "multi_extreme"])
    parser.add_argument("--target-selector", default="risk_aware", choices=["nearest", "risk_aware"])
    parser.add_argument("--max-steams", type=int)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--episode-reset", action="store_true", help="Reset on normal episode success/timeout.")
    args = parser.parse_args()
    test_model(
        args.model,
        stage=args.stage,
        target_selector=args.target_selector,
        max_steams=args.max_steams,
        deterministic=not args.stochastic,
        max_steps=args.steps,
        summary_interval=args.interval,
        demo_mode=not args.episode_reset,
    )
