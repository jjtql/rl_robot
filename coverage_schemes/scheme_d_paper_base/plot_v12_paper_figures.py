import argparse
import csv
import os
from pathlib import Path

import numpy as np


STAGE_ORDER = ["multi_low", "multi_realistic", "multi_hard", "multi_extreme"]
STAGE_LABELS = {
    "multi_low": "Low",
    "multi_realistic": "Realistic",
    "multi_hard": "Hard",
    "multi_extreme": "Extreme",
}
OURS = "thermal_lstm_spawnhist_latency_v12_fast"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pick_row(rows, variant, policy, stage):
    for row in rows:
        if row.get("variant") == variant and row.get("policy") == policy and row.get("stage") == stage:
            return row
    raise KeyError(f"Missing row for variant={variant}, policy={policy}, stage={stage}")


def parse_trajectory_snapshot(value):
    points = []
    for part in str(value or "").split(";"):
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 4:
            continue
        try:
            points.append(
                {
                    "id": int(fields[0]),
                    "x": float(fields[1]),
                    "y": float(fields[2]),
                    "age": float(fields[3]),
                }
            )
        except ValueError:
            continue
    return points


def trajectory_rows(path):
    rows = read_csv(path)
    return rows


def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=240)
    fig.savefig(output_dir / f"{stem}.pdf")


def plot_main_coverage(tables_dir, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate paper figures") from exc

    rows = read_csv(tables_dir / "main_results.csv")
    labels = [STAGE_LABELS[stage] for stage in STAGE_ORDER]
    ours_mean = []
    ours_std = []
    horizon_mean = []
    horizon_std = []
    for stage in STAGE_ORDER:
        ours = pick_row(rows, OURS, "ppo", stage)
        horizon = pick_row(rows, "horizon2", "horizon2", stage)
        ours_mean.append(as_float(ours.get("coverage_rate_mean")))
        ours_std.append(as_float(ours.get("coverage_rate_std")))
        horizon_mean.append(as_float(horizon.get("coverage_rate_mean")))
        horizon_std.append(as_float(horizon.get("coverage_rate_std")))

    x = np.arange(len(STAGE_ORDER), dtype=np.float64)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    ax.bar(
        x - width / 2.0,
        horizon_mean,
        width,
        yerr=horizon_std,
        capsize=4,
        label="Horizon-2",
        color="#6b7280",
        edgecolor="#374151",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2.0,
        ours_mean,
        width,
        yerr=ours_std,
        capsize=4,
        label="Ours",
        color="#2f80ed",
        edgecolor="#174a8b",
        linewidth=0.8,
    )
    ax.set_ylim(0.72, 0.96)
    ax.set_ylabel("Coverage rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_title("Coverage Across Curriculum Stages")
    for idx, (base, ours_value) in enumerate(zip(horizon_mean, ours_mean)):
        delta = (ours_value - base) * 100.0
        delta_label = "tied" if abs(delta) < 0.05 else f"{delta:+.1f} pts"
        ax.text(
            idx,
            max(base, ours_value) + 0.012,
            delta_label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )
    fig.tight_layout()
    save_figure(fig, output_dir, "main_coverage_ours_vs_horizon2")
    plt.close(fig)


def trajectory_arrays(rows):
    cover_xy = np.array(
        [[as_float(row.get("cover_x")), as_float(row.get("cover_y"))] for row in rows],
        dtype=np.float64,
    )
    target_rows = [
        row for row in rows
        if as_float(row.get("selected_target_id"), -1.0) >= 0.0
    ]
    target_xy = np.array(
        [
            [as_float(row.get("selected_target_x")), as_float(row.get("selected_target_y"))]
            for row in target_rows
        ],
        dtype=np.float64,
    )
    risk = np.array(
        [as_float(row.get("selected_target_risk_score")) for row in target_rows],
        dtype=np.float64,
    )
    return cover_xy, target_xy, risk


def draw_trajectory_axis(ax, rows, title):
    from matplotlib.patches import Circle

    center = np.array([1.8, 0.0], dtype=np.float64)
    radius = 0.8
    cover_xy, target_xy, risk = trajectory_arrays(rows)
    ax.add_patch(
        Circle(
            center,
            radius,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            color="#4b5563",
        )
    )
    if cover_xy.size:
        ax.plot(
            cover_xy[:, 0],
            cover_xy[:, 1],
            color="#2563eb",
            linewidth=1.8,
            label="cover path",
        )
        ax.scatter(
            cover_xy[0, 0],
            cover_xy[0, 1],
            s=46,
            color="#16a34a",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label="start",
        )
        ax.scatter(
            cover_xy[-1, 0],
            cover_xy[-1, 1],
            s=46,
            color="#dc2626",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label="end",
        )
    scatter = None
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
    margin = radius * 1.08
    ax.set_xlim(center[0] - margin, center[0] + margin)
    ax.set_ylim(center[1] - margin, center[1] + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(alpha=0.16, linewidth=0.6)
    return scatter


def plot_standardized_trajectory(rows, output_dir, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate paper figures") from exc

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    fig.subplots_adjust(left=0.11, right=0.82, bottom=0.11, top=0.90)
    scatter = draw_trajectory_axis(ax, rows, title)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    if scatter is not None:
        color_ax = fig.add_axes([0.855, 0.15, 0.035, 0.68])
        fig.colorbar(scatter, cax=color_ax, label="risk score")
    save_figure(fig, output_dir, "trajectory_standardized")
    plt.close(fig)


def plot_trajectory_comparison(run_root, output_dir, stage):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate paper figures") from exc

    ours_dir = run_root / "visuals" / f"{OURS}_seed0" / stage
    horizon_dir = run_root / "visuals" / "horizon2" / stage
    ours_rows = trajectory_rows(ours_dir / "trajectory.csv")
    horizon_rows = trajectory_rows(horizon_dir / "trajectory.csv")
    stage_label = STAGE_LABELS[stage]

    plot_standardized_trajectory(
        horizon_rows,
        horizon_dir,
        f"Horizon-2 - {stage_label}",
    )
    plot_standardized_trajectory(
        ours_rows,
        ours_dir,
        f"Ours - {stage_label}",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.17, top=0.88, wspace=0.14)
    scatter = draw_trajectory_axis(axes[0], horizon_rows, "Horizon-2")
    draw_trajectory_axis(axes[1], ours_rows, "Ours")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.48, 0.01),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    if scatter is not None:
        color_ax = fig.add_axes([0.925, 0.20, 0.018, 0.62])
        fig.colorbar(scatter, cax=color_ax, label="risk score")
    fig.suptitle(f"Trajectory Comparison: {stage_label}", fontsize=14)
    save_figure(fig, output_dir, f"trajectory_comparison_{stage}")
    plt.close(fig)


def plot_environment_overview(run_root, output_dir, stage="multi_hard"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrowPatch
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate paper figures") from exc

    traj_path = (
        run_root
        / "visuals"
        / OURS
        .replace("thermal_lstm_spawnhist_latency_v12_fast", "thermal_lstm_spawnhist_latency_v12_fast_seed0")
        / stage
        / "trajectory.csv"
    )
    if not traj_path.exists():
        traj_path = run_root / "visuals" / "thermal_lstm_spawnhist_latency_v12_fast_seed0" / stage / "trajectory.csv"
    rows = trajectory_rows(traj_path)
    if not rows:
        raise FileNotFoundError(f"No trajectory rows found at {traj_path}")

    cover_xy = np.array([[as_float(row["cover_x"]), as_float(row["cover_y"])] for row in rows], dtype=np.float64)
    target_xy = np.array(
        [[as_float(row["selected_target_x"]), as_float(row["selected_target_y"])] for row in rows],
        dtype=np.float64,
    )
    risk = np.array([as_float(row.get("selected_target_risk_score")) for row in rows], dtype=np.float64)
    snapshot_index = min(max(len(rows) // 2, 0), len(rows) - 1)
    active_points = parse_trajectory_snapshot(rows[snapshot_index].get("active_steams", ""))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    center = cover_xy.mean(axis=0)
    radius = max(np.linalg.norm(cover_xy - center, axis=1).max(), 1.1)
    ax.add_patch(Circle(center, radius, fill=False, linestyle="--", linewidth=1.1, color="#4b5563"))
    ax.plot(cover_xy[:, 0], cover_xy[:, 1], color="#2563eb", linewidth=2.0, label="cover trajectory")
    ax.scatter(cover_xy[0, 0], cover_xy[0, 1], s=58, color="#16a34a", edgecolor="white", linewidth=0.8, label="start")
    ax.scatter(cover_xy[-1, 0], cover_xy[-1, 1], s=58, color="#dc2626", edgecolor="white", linewidth=0.8, label="end")
    if target_xy.size:
        step = max(len(target_xy) // 160, 1)
        scatter = ax.scatter(
            target_xy[::step, 0],
            target_xy[::step, 1],
            c=risk[::step],
            cmap="inferno",
            s=20,
            alpha=0.70,
            label="selected thermal spots",
        )
        fig.colorbar(scatter, ax=ax, fraction=0.044, pad=0.02, label="risk score")
    if active_points:
        ages = np.array([point["age"] for point in active_points], dtype=np.float64)
        age_scale = ages / max(float(ages.max()), 1.0)
        ax.scatter(
            [point["x"] for point in active_points],
            [point["y"] for point in active_points],
            s=70 + 110 * age_scale,
            color="#f97316",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.88,
            label="active steam spots",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Workspace and Thermal-Spot Service")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    ax.set_axis_off()
    boxes = [
        ("Thermal spawn field", 0.18, 0.78, "#f97316"),
        ("Horizon-2 planner", 0.50, 0.78, "#6b7280"),
        ("LSTM-PPO residual", 0.50, 0.48, "#2563eb"),
        ("ABB cover action", 0.80, 0.62, "#16a34a"),
        ("Coverage / latency metrics", 0.50, 0.20, "#7c3aed"),
    ]
    for text, x, y, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.38,rounding_size=0.08",
                "facecolor": color,
                "edgecolor": "none",
                "alpha": 0.95,
            },
        )
    arrows = [
        ((0.30, 0.78), (0.40, 0.78)),
        ((0.50, 0.70), (0.50, 0.56)),
        ((0.62, 0.56), (0.72, 0.62)),
        ((0.72, 0.62), (0.62, 0.72)),
        ((0.80, 0.54), (0.56, 0.28)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.3,
                color="#111827",
                alpha=0.82,
            )
        )
    ax.text(
        0.50,
        0.02,
        "MuJoCo ABB workspace with burst-lull thermal targets and planner-guided residual control",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
        wrap=True,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "mujoco_task_overview")
    plt.close(fig)


def plot_mujoco_environment_snapshot(
    output_dir,
    stage="multi_hard",
    steps=80,
    camera_azimuth=150.0,
    camera_elevation=-21.0,
    camera_distance=4.5,
    camera_lookat=(1.05, 0.0, 1.0),
):
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib and mujoco are required to render the environment snapshot") from exc

    from .config import load_config, set_global_seeds
    from .env import ShangZengEnv
    from .eval import configure_env_from_config
    from .policies import build_policy

    set_global_seeds(7)
    config = load_config(None)
    env = ShangZengEnv(
        model_path=config["model_path"],
        max_episode_steps=max(int(steps) + 1, 100),
        target_selector="risk_aware",
    )
    configure_env_from_config(env, config)
    env.configure_curriculum(stage)
    env.burst_lull_spawn_enabled = True
    env.burst_lull_lull_steps = 70
    env.burst_lull_charge_steps = 110
    env.burst_lull_sparse_threshold = 2
    env.burst_lull_burst_min = 3
    env.burst_lull_burst_max = 5
    env.burst_lull_burst_interval_steps = 4
    env.burst_lull_trickle_probability = 0.004
    env.thermal_hotspot_strength = 3.8
    env.thermal_background_weight = 0.055
    env.thermal_recent_spawn_memory = 24
    obs, _ = env.reset(seed=7)
    policy = build_policy("horizon2", env, config_override=config)
    policy.reset()
    for _ in range(max(int(steps), 0)):
        action = policy.act(env, obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth = float(camera_azimuth)
    camera.elevation = float(camera_elevation)
    camera.distance = float(camera_distance)
    camera.lookat[:] = np.asarray(camera_lookat, dtype=np.float64)
    renderer.update_scene(env.data, camera=camera)
    image = renderer.render()
    renderer.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_dir / "mujoco_environment_snapshot.png", image)
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title("MuJoCo Thermal-Spot Coverage Environment", pad=8)
    fig.tight_layout(pad=0.1)
    fig.savefig(output_dir / "mujoco_environment_snapshot.pdf")
    plt.close(fig)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate V12 paper-ready figures.")
    parser.add_argument("--run-root", default="runs/v12_small_paper_suite")
    parser.add_argument("--output-dir", default="runs/v12_small_paper_suite/paper_figures")
    parser.add_argument("--environment-stage", default="multi_hard", choices=STAGE_ORDER)
    parser.add_argument(
        "--trajectory-stages",
        default="multi_realistic,multi_hard,multi_extreme",
        help="Comma-separated stages for standardized trajectory comparisons.",
    )
    parser.add_argument("--skip-mujoco-snapshot", action="store_true")
    parser.add_argument("--camera-azimuth", type=float, default=150.0)
    parser.add_argument("--camera-elevation", type=float, default=-21.0)
    parser.add_argument("--camera-distance", type=float, default=4.5)
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(1.05, 0.0, 1.0),
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    plot_main_coverage(run_root / "paper_tables", output_dir)
    for stage in [value.strip() for value in args.trajectory_stages.split(",") if value.strip()]:
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown trajectory stage: {stage}")
        plot_trajectory_comparison(run_root, output_dir, stage)
    plot_environment_overview(run_root, output_dir, stage=args.environment_stage)
    if not args.skip_mujoco_snapshot:
        plot_mujoco_environment_snapshot(
            output_dir,
            stage=args.environment_stage,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
            camera_distance=args.camera_distance,
            camera_lookat=args.camera_lookat,
        )
    print(f"Wrote paper figures to {output_dir}")


if __name__ == "__main__":
    main()
