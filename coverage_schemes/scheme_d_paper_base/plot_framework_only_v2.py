import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon


INK = "#243447"
MUTED = "#5F6B7A"
BLUE = "#0072B2"
BLUE_FILL = "#F2F7FB"
ORANGE = "#B66A00"
ORANGE_FILL = "#FFF8E8"
GREEN = "#007A5E"
GREEN_FILL = "#EFF8F4"
PURPLE = "#9B4D77"
PURPLE_FILL = "#FAF2F6"
GRAY = "#536273"
GRAY_FILL = "#F7F8FA"
PANEL_EDGE = "#707B89"


def panel(ax, x, y, width, height, label, title):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.003,rounding_size=0.003",
            linewidth=0.85,
            edgecolor=PANEL_EDGE,
            facecolor="white",
            transform=ax.transAxes,
            zorder=0,
        )
    )
    ax.text(
        x + 0.012,
        y + height - 0.020,
        f"({label}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        zorder=5,
    )


def box(ax, x, y, width, height, title, body, face, edge, title_size=6.8, body_size=5.9):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.003,rounding_size=0.002",
            linewidth=0.95,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.70,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="semibold",
        color=INK,
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=body_size,
        color=INK,
        linespacing=1.08,
        zorder=3,
    )


def arrow(ax, points, color=INK, dashed=False, linewidth=1.0):
    for start, end in zip(points, points[1:]):
        if abs(start[0] - end[0]) > 1e-9 and abs(start[1] - end[1]) > 1e-9:
            raise ValueError(f"Arrow segment is not orthogonal: {start} -> {end}")
    linestyle = "--" if dashed else "-"
    start_px = np.asarray(ax.transAxes.transform(points[-2]), dtype=np.float64)
    tip_px = np.asarray(ax.transAxes.transform(points[-1]), dtype=np.float64)
    direction = tip_px - start_px
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    base_center_px = tip_px - direction * 5.8
    half_width_px = 3.7
    triangle_px = np.stack(
        [
            tip_px,
            base_center_px + normal * half_width_px,
            base_center_px - normal * half_width_px,
        ]
    )
    inverse = ax.transAxes.inverted()
    base_center = tuple(inverse.transform(base_center_px))
    triangle = inverse.transform(triangle_px)
    line_points = list(points[:-1]) + [base_center]

    ax.plot(
        [point[0] for point in line_points],
        [point[1] for point in line_points],
        transform=ax.transAxes,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        dash_capstyle="butt",
        dash_joinstyle="miter",
        zorder=4,
    )
    ax.add_patch(
        Polygon(
            triangle,
            closed=True,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            zorder=5,
        )
    )


def connector(ax, points, color=INK, dashed=False, linewidth=1.0):
    for start, end in zip(points, points[1:]):
        if abs(start[0] - end[0]) > 1e-9 and abs(start[1] - end[1]) > 1e-9:
            raise ValueError(f"Connector segment is not orthogonal: {start} -> {end}")
    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        transform=ax.transAxes,
        color=color,
        linewidth=linewidth,
        linestyle="--" if dashed else "-",
        solid_capstyle="butt",
        solid_joinstyle="miter",
        dash_capstyle="butt",
        dash_joinstyle="miter",
        zorder=4,
    )


def build(snapshot, output_dir):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.16, 4.65), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    panel(ax, 0.015, 0.355, 0.190, 0.615, "a", "Dynamic task")
    panel(ax, 0.220, 0.355, 0.765, 0.615, "b", "Planner-guided residual controller")
    panel(ax, 0.015, 0.035, 0.970, 0.275, "c", "Experimental verification")

    image_ax = fig.add_axes([0.028, 0.625, 0.164, 0.250])
    image_ax.imshow(plt.imread(snapshot))
    image_ax.set_axis_off()
    box(
        ax,
        0.035,
        0.450,
        0.150,
        0.110,
        "Persistent service",
        "burst-lull targets\npersist until covered",
        GRAY_FILL,
        GRAY,
        body_size=5.7,
    )

    block_w = 0.115
    block_h = 0.130
    top_y = 0.700
    lower_y = 0.475
    columns = [0.245, 0.390, 0.535, 0.680, 0.825]
    observation_x = 0.255
    observation_w = 0.105
    branch_x = 0.375

    box(ax, observation_x, 0.590, observation_w, block_h, "Observation", "robot + spots\nhistory + context", GRAY_FILL, GRAY, body_size=5.7)
    box(ax, columns[1], top_y, block_w, block_h, "Horizon-2", "base action $u_t^{base}$", ORANGE_FILL, ORANGE)
    box(
        ax,
        columns[2],
        top_y,
        block_w,
        block_h,
        "Phase fusion",
        "$u^{raw}=u^{base}+\\beta k u^{res}$",
        GREEN_FILL,
        GREEN,
        title_size=6.2,
        body_size=5.0,
    )
    box(ax, columns[3], top_y, block_w, block_h, "Action shield", "clip | guard\nprogress", GRAY_FILL, GRAY, body_size=5.4)
    box(ax, columns[4], top_y, block_w, block_h, "Action", "$[\\Delta x,\\Delta y,s]$", GREEN_FILL, GREEN, body_size=5.6)
    box(ax, columns[1], lower_y, block_w, block_h, "Spot attention", "8 spots $\\rightarrow$ context", BLUE_FILL, BLUE, body_size=5.5)
    box(
        ax,
        columns[2],
        lower_y,
        block_w,
        block_h,
        "LSTM-PPO",
        "memory + PPO heads\nresidual $u_t^{res}$",
        BLUE_FILL,
        BLUE,
        body_size=5.4,
    )
    box(
        ax,
        columns[3],
        lower_y,
        block_w,
        block_h,
        "PPO objective",
        "clipped PPO loss\nvalue + prediction",
        PURPLE_FILL,
        PURPLE,
        title_size=6.0,
        body_size=5.0,
    )
    box(
        ax,
        columns[4],
        lower_y,
        block_w,
        block_h,
        "Service feedback",
        "coverage | latency\nbacklog | smoothness",
        PURPLE_FILL,
        PURPLE,
        title_size=5.9,
        body_size=5.0,
    )

    gap = 0.0040
    connector(ax, [(0.192 + gap, 0.750), (0.230, 0.750)], color=INK, linewidth=1.0)
    connector(ax, [(0.185 + gap, 0.505), (0.230, 0.505)], color=INK, linewidth=1.0)
    connector(ax, [(0.230, 0.505), (0.230, 0.750)], color=INK, linewidth=1.0)
    arrow(
        ax,
        [(0.230, 0.655), (observation_x - gap, 0.655)],
        color=INK,
        linewidth=1.0,
    )
    connector(
        ax,
        [(observation_x + observation_w + gap, 0.655), (branch_x, 0.655)],
        color=INK,
        linewidth=1.0,
    )
    connector(
        ax,
        [(branch_x, lower_y + block_h / 2), (branch_x, top_y + block_h / 2)],
        color=INK,
        linewidth=1.0,
    )
    arrow(ax, [(branch_x, top_y + block_h / 2), (columns[1] - gap, top_y + block_h / 2)])
    arrow(ax, [(branch_x, lower_y + block_h / 2), (columns[1] - gap, lower_y + block_h / 2)])
    arrow(ax, [(columns[1] + block_w + gap, top_y + block_h / 2), (columns[2] - gap, top_y + block_h / 2)])
    arrow(ax, [(columns[1] + block_w + gap, lower_y + block_h / 2), (columns[2] - gap, lower_y + block_h / 2)])
    arrow(ax, [(columns[2] + block_w / 2, lower_y + block_h + gap), (columns[2] + block_w / 2, top_y - gap)])
    arrow(ax, [(columns[2] + block_w + gap, top_y + block_h / 2), (columns[3] - gap, top_y + block_h / 2)])
    arrow(ax, [(columns[3] + block_w + gap, top_y + block_h / 2), (columns[4] - gap, top_y + block_h / 2)])

    arrow(
        ax,
        [
            (columns[4] + block_w / 2, top_y - gap),
            (columns[4] + block_w / 2, lower_y + block_h + gap),
        ],
        color=PURPLE,
        dashed=True,
    )
    arrow(
        ax,
        [
            (columns[4] - gap, lower_y + block_h / 2),
            (columns[3] + block_w + gap, lower_y + block_h / 2),
        ],
        color=PURPLE,
        dashed=True,
    )
    arrow(
        ax,
        [
            (columns[3] - gap, lower_y + block_h / 2),
            (columns[2] + block_w + gap, lower_y + block_h / 2),
        ],
        color=PURPLE,
        dashed=True,
    )

    ax.text(0.966, 0.947, "solid: online   dashed: training", transform=ax.transAxes, ha="right", va="top", fontsize=5.7, color=MUTED)

    y = 0.075
    height = 0.160
    experiment_w = 0.210
    experiment_x = [0.035, 0.272, 0.509, 0.746]
    box(ax, experiment_x[0], y, experiment_w, height, "Stages", "Low / Realistic / Hard / Extreme\n$N_{max}=3 / 4 / 6 / 8$", GRAY_FILL, PANEL_EDGE)
    box(ax, experiment_x[1], y, experiment_w, height, "Independent seeds", "train: 0--2 | held-out: 100--102\n5 episodes $\\times$ 3200 steps", GRAY_FILL, PANEL_EDGE, body_size=5.7)
    box(ax, experiment_x[2], y, experiment_w, height, "Comparisons", "Horizon-2 + 8 baselines\n5 ablations + reward sweep", GRAY_FILL, PANEL_EDGE)
    box(ax, experiment_x[3], y, experiment_w, height, "Evidence", "coverage mean/std + bootstrap CI\nlatency | p90 | SLA | backlog", GRAY_FILL, PANEL_EDGE)
    for left, right in zip(experiment_x, experiment_x[1:]):
        arrow(
            ax,
            [(left + experiment_w + gap, y + height / 2), (right - gap, y + height / 2)],
            color=PANEL_EDGE,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "ieee_framework_overview_final"
    fig.savefig(base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".png"), dpi=320, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate a standalone IEEE framework figure.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    build(Path(args.snapshot), Path(args.output_dir))


if __name__ == "__main__":
    main()
