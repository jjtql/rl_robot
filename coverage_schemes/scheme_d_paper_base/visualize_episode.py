import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import deep_update, load_config, set_global_seeds
from .env import ShangZengEnv
from .eval import configure_env_from_config
from .policies import build_policy, checkpoint_config


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Record one qualitative trajectory and material heatmap.")
    parser.add_argument("--config", help="Optional JSON config.")
    parser.add_argument("--policy", default="risk_aware", choices=["ppo", "random", "nearest", "oldest", "distance_age", "risk_aware", "dynamic_weighted", "horizon2", "horizon3", "deadline_horizon2", "deadline_rescue_horizon2", "slack_horizon2", "corridor_waypoint", "sla_route_ensemble", "sticky_sla_ensemble", "aco_tsp", "planner_ensemble"])
    parser.add_argument("--model", help="Checkpoint for PPO policy.")
    parser.add_argument("--stage", default="multi_realistic", choices=["single_easy", "single_precision", "multi_low", "multi_realistic", "multi_hard", "multi_extreme"])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/qualitative")
    parser.add_argument("--model-path", help="MuJoCo XML path.")
    parser.add_argument("--target-selector", choices=["nearest", "risk_aware"])
    parser.add_argument("--max-steams", type=int)
    parser.add_argument("--demo-mode", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    return parser


def steam_snapshot(env):
    parts = []
    for steam in env.steams:
        parts.append(
            f"{int(steam.get('id', -1))}:{float(steam['pos'][0]):.5f}:"
            f"{float(steam['pos'][1]):.5f}:{float(steam['age']):.1f}"
        )
    return ";".join(parts)


def record_row(step, reward, env, info):
    metric_step_seconds = float(getattr(env, "metric_step_seconds", 1.0) or 1.0)
    return {
        "step": step,
        "reward": float(reward),
        "cover_x": float(env.cover_center[0]),
        "cover_y": float(env.cover_center[1]),
        "selected_target_id": int(info.get("selected_target_id", -1)),
        "selected_target_x": float(info.get("selected_target_x", 0.0)),
        "selected_target_y": float(info.get("selected_target_y", 0.0)),
        "selected_target_risk_score": float(info.get("selected_target_risk_score", 0.0)),
        "selected_target_age_score": float(info.get("selected_target_age_score", 0.0)),
        "selected_target_distance_score": float(info.get("selected_target_distance_score", 0.0)),
        "selected_target_material_score": float(info.get("selected_target_material_score", 0.0)),
        "selected_target_reachability_score": float(info.get("selected_target_reachability_score", 0.0)),
        "selected_target_thermal_score": float(info.get("selected_target_thermal_score", 0.0)),
        "route_confidence": float(info.get("route_confidence", 0.0)),
        "route_stagnation_score": float(info.get("route_stagnation_score", 0.0)),
        "coverage_rate": float(info.get("coverage_rate", 0.0)),
        "effective_coverage_rate": float(info.get("effective_coverage_rate", 0.0)),
        "cover_latency_seconds": float(
            info.get("cover_latency_seconds", float(info.get("cover_latency", 0.0)) * metric_step_seconds)
        ),
        "cover_latency_p90_seconds": float(
            info.get("cover_latency_p90_seconds", float(info.get("cover_latency_p90", 0.0)) * metric_step_seconds)
        ),
        "response_sla_success_rate": float(info.get("response_sla_success_rate", 0.0)),
        "strict_response_sla_success_rate": float(info.get("strict_response_sla_success_rate", 0.0)),
        "success_count": int(info.get("success_count", 0)),
        "spawned_count": int(info.get("spawned_count", 0)),
        "missed_count": int(info.get("missed_count", 0)),
        "steam_count": int(info.get("steam_count", 0)),
        "pending_steam_count": int(info.get("pending_steam_count", 0)),
        "active_steam_mean": float(info.get("active_steam_mean", 0.0)),
        "oldest_active_age_max_seconds": float(
            info.get(
                "oldest_active_age_max_seconds",
                float(info.get("oldest_active_age_max", 0.0)) * metric_step_seconds,
            )
        ),
        "full_session_terminal_clear": bool(info.get("full_session_terminal_clear", False)),
        "burst_lull_phase": str(info.get("burst_lull_phase", "")),
        "burst_lull_charge_score": float(info.get("burst_lull_charge_score", 0.0)),
        "burst_lull_lull_remaining": int(info.get("burst_lull_lull_remaining", 0)),
        "burst_lull_pending_count": int(info.get("burst_lull_pending_count", 0)),
        "burst_lull_next_spawn_delay": int(info.get("burst_lull_next_spawn_delay", 0)),
        "last_burst_spawn_count": int(info.get("last_burst_spawn_count", 0)),
        "mean_material_height": float(info.get("mean_material_height", 0.0)),
        "height_uniformity": float(info.get("height_uniformity", 0.0)),
        "overfill_penalty": float(info.get("overfill_penalty", 0.0)),
        "active_steams": steam_snapshot(env),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir, env, rows):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; wrote CSV/JSON only.")
        return

    cover_xy = np.array([[row["cover_x"], row["cover_y"]] for row in rows], dtype=np.float32)
    target_xy = np.array([[row["selected_target_x"], row["selected_target_y"]] for row in rows], dtype=np.float32)
    risk = np.array([row["selected_target_risk_score"] for row in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    fig.subplots_adjust(left=0.11, right=0.82, bottom=0.11, top=0.90)
    circle = plt.Circle(env.pot_center[:2], env.pot_radius, fill=False, linestyle="--", linewidth=1.2, color="0.35")
    ax.add_patch(circle)
    if cover_xy.size:
        ax.plot(cover_xy[:, 0], cover_xy[:, 1], color="#2563eb", linewidth=1.8, label="cover path")
        ax.scatter(cover_xy[0, 0], cover_xy[0, 1], color="#16a34a", s=46, edgecolor="white", linewidth=0.6, zorder=4, label="start")
        ax.scatter(cover_xy[-1, 0], cover_xy[-1, 1], color="#dc2626", s=46, edgecolor="white", linewidth=0.6, zorder=4, label="end")
    if target_xy.size:
        scatter = ax.scatter(
            target_xy[:, 0],
            target_xy[:, 1],
            c=risk,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            s=20,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.25,
            label="selected target",
        )
        color_ax = fig.add_axes([0.855, 0.15, 0.035, 0.68])
        fig.colorbar(scatter, cax=color_ax, label="risk score")
    ax.set_aspect("equal", adjustable="box")
    margin = float(env.pot_radius) * 1.08
    ax.set_xlim(float(env.pot_center[0]) - margin, float(env.pot_center[0]) + margin)
    ax.set_ylim(float(env.pot_center[1]) - margin, float(env.pot_center[1]) + margin)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Cover Trajectory and Selected Targets")
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    fig.savefig(output_dir / "trajectory.png", dpi=170)
    fig.savefig(output_dir / "trajectory.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    extent = [
        float(env.grid_world_xy[..., 0].min()),
        float(env.grid_world_xy[..., 0].max()),
        float(env.grid_world_xy[..., 1].min()),
        float(env.grid_world_xy[..., 1].max()),
    ]
    image = ax.imshow(env.material_height.T, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    fig.colorbar(image, ax=ax, label="material height")
    if cover_xy.size:
        ax.plot(cover_xy[:, 0], cover_xy[:, 1], color="white", linewidth=1.5, alpha=0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Final Material Height")
    fig.tight_layout()
    fig.savefig(output_dir / "material_heatmap.png", dpi=170)
    plt.close(fig)

    steps = np.array([row["step"] for row in rows], dtype=np.float32)
    active = np.array([row["steam_count"] for row in rows], dtype=np.float32)
    pending = np.array([row["pending_steam_count"] for row in rows], dtype=np.float32)
    coverage = np.array([row["coverage_rate"] for row in rows], dtype=np.float32)
    effective = np.array([row["effective_coverage_rate"] for row in rows], dtype=np.float32)
    latency = np.array([row["cover_latency_seconds"] for row in rows], dtype=np.float32)
    p90 = np.array([row["cover_latency_p90_seconds"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
    axes[0].plot(steps, active, color="#1f77b4", label="active")
    axes[0].plot(steps, pending, color="#ff7f0e", label="pending/uncleared")
    axes[0].set_ylabel("steam count")
    axes[0].legend(loc="upper right")

    axes[1].plot(steps, coverage, color="#2ca02c", label="raw coverage")
    axes[1].plot(steps, effective, color="#9467bd", label="effective coverage")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("rate")
    axes[1].legend(loc="lower right")

    axes[2].plot(steps, latency, color="#d62728", label="mean latency")
    axes[2].plot(steps, p90, color="#8c564b", label="p90 latency")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("seconds")
    axes[2].legend(loc="upper left")

    fig.suptitle("Backlog, Coverage, and Response Latency")
    fig.tight_layout()
    fig.savefig(output_dir / "backlog_latency.png", dpi=170)
    plt.close(fig)


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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    policy = build_policy(
        args.policy,
        env,
        model_path=args.model,
        deterministic=not args.stochastic,
        config_override=config,
    )
    obs, info = env.reset(seed=args.seed)
    policy.reset()
    rows = [record_row(0, 0.0, env, info)]
    total_reward = 0.0

    for step in range(1, args.steps + 1):
        action = policy.act(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rows.append(record_row(step, reward, env, info))
        if args.demo_mode and terminated:
            terminated = False
        if terminated or truncated:
            break

    write_csv(output_dir / "trajectory.csv", rows)
    metric_step_seconds = float(getattr(env, "metric_step_seconds", 1.0) or 1.0)
    summary = {
        "policy": args.policy,
        "stage": args.stage,
        "seed": args.seed,
        "steps": len(rows) - 1,
        "total_reward": float(total_reward),
        "coverage_rate": float(info.get("coverage_rate", 0.0)),
        "effective_coverage_rate": float(info.get("effective_coverage_rate", 0.0)),
        "cover_latency_seconds": float(
            info.get("cover_latency_seconds", float(info.get("cover_latency", 0.0)) * metric_step_seconds)
        ),
        "cover_latency_p90_seconds": float(
            info.get("cover_latency_p90_seconds", float(info.get("cover_latency_p90", 0.0)) * metric_step_seconds)
        ),
        "response_sla_success_rate": float(info.get("response_sla_success_rate", 0.0)),
        "strict_response_sla_success_rate": float(info.get("strict_response_sla_success_rate", 0.0)),
        "success_count": int(info.get("success_count", 0)),
        "spawned_count": int(info.get("spawned_count", 0)),
        "missed_count": int(info.get("missed_count", 0)),
        "pending_steam_count": int(info.get("pending_steam_count", 0)),
        "full_session_terminal_clear": bool(info.get("full_session_terminal_clear", False)),
        "target_selector": env.target_selector,
        "max_steams": env.max_steams,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_plots(output_dir, env, rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
