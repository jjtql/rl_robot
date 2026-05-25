import argparse
import subprocess
from pathlib import Path


METHODS = {
    "attention_map_tv": [
        "--target-selector",
        "risk_aware",
        "--bc-policy",
        "risk_aware",
        "--steam-attention",
        "--material-map",
        "--material-tv-reward",
    ],
    "attention_map": ["--target-selector", "risk_aware", "--bc-policy", "risk_aware", "--steam-attention", "--material-map"],
    "attention": ["--target-selector", "risk_aware", "--bc-policy", "risk_aware", "--steam-attention"],
    "attention_stable_v2": [
        "--target-selector",
        "risk_aware",
        "--bc-policy",
        "risk_aware",
        "--steam-attention",
        "--ppo-lr",
        "3e-5",
        "--ppo-epochs",
        "2",
        "--ppo-clip",
        "0.08",
        "--ppo-value-clip",
        "0.08",
        "--ppo-entropy-start",
        "0.0015",
        "--ppo-entropy-end",
        "0.0003",
        "--ppo-entropy-decay-steps",
        "1200000",
        "--bc-supervised-coef",
        "0.05",
        "--bc-supervised-min-coef",
        "0.01",
        "--bc-supervised-decay-steps",
        "1200000",
        "--bc-episodes",
        "100",
        "--bc-epochs",
        "12",
        "--bc-stage-episodes",
        "single_easy:10,multi_low:40,multi_realistic:50",
    ],
    "attention_map_tv_stable_v2": [
        "--target-selector",
        "risk_aware",
        "--bc-policy",
        "risk_aware",
        "--steam-attention",
        "--material-map",
        "--material-tv-reward",
        "--ppo-lr",
        "3e-5",
        "--ppo-epochs",
        "2",
        "--ppo-clip",
        "0.08",
        "--ppo-value-clip",
        "0.08",
        "--ppo-entropy-start",
        "0.0015",
        "--ppo-entropy-end",
        "0.0003",
        "--ppo-entropy-decay-steps",
        "1200000",
        "--bc-supervised-coef",
        "0.05",
        "--bc-supervised-min-coef",
        "0.01",
        "--bc-supervised-decay-steps",
        "1200000",
        "--bc-episodes",
        "100",
        "--bc-epochs",
        "12",
        "--bc-stage-episodes",
        "single_easy:10,multi_low:40,multi_realistic:50",
    ],
    "attention_multisteam_bc_v3": [
        "--target-selector",
        "risk_aware",
        "--bc-policy",
        "risk_aware",
        "--steam-attention",
        "--stage-episodes",
        "multi_low:500,multi_realistic:500",
        "--no-action-smoothing-penalty",
        "--ppo-lr",
        "1e-5",
        "--ppo-epochs",
        "1",
        "--ppo-clip",
        "0.05",
        "--ppo-value-clip",
        "0.05",
        "--ppo-entropy-start",
        "0.0005",
        "--ppo-entropy-end",
        "0.0001",
        "--ppo-entropy-decay-steps",
        "1200000",
        "--bc-supervised-coef",
        "0.12",
        "--bc-supervised-min-coef",
        "0.08",
        "--bc-supervised-decay-steps",
        "1200000",
        "--bc-episodes",
        "160",
        "--bc-epochs",
        "16",
        "--bc-stage-episodes",
        "single_easy:5,multi_low:65,multi_realistic:90",
    ],
    "attention_map_tv_multisteam_bc_v3": [
        "--target-selector",
        "risk_aware",
        "--bc-policy",
        "risk_aware",
        "--steam-attention",
        "--material-map",
        "--material-tv-reward",
        "--material-tv-reward-gain",
        "2.0",
        "--stage-episodes",
        "multi_low:500,multi_realistic:500",
        "--no-action-smoothing-penalty",
        "--ppo-lr",
        "1e-5",
        "--ppo-epochs",
        "1",
        "--ppo-clip",
        "0.05",
        "--ppo-value-clip",
        "0.05",
        "--ppo-entropy-start",
        "0.0005",
        "--ppo-entropy-end",
        "0.0001",
        "--ppo-entropy-decay-steps",
        "1200000",
        "--bc-supervised-coef",
        "0.12",
        "--bc-supervised-min-coef",
        "0.08",
        "--bc-supervised-decay-steps",
        "1200000",
        "--bc-episodes",
        "160",
        "--bc-epochs",
        "16",
        "--bc-stage-episodes",
        "single_easy:5,multi_low:65,multi_realistic:90",
    ],
    "planner_residual_attention_v4": [
        "--target-selector",
        "risk_aware",
        "--bc-policy",
        "horizon2",
        "--steam-attention",
        "--residual-policy",
        "--residual-base-policy",
        "horizon2",
        "--residual-beta",
        "0.20",
        "--stage-episodes",
        "multi_low:350,multi_realistic:350,multi_hard:400",
        "--no-action-smoothing-penalty",
        "--ppo-lr",
        "8e-6",
        "--ppo-epochs",
        "1",
        "--ppo-clip",
        "0.04",
        "--ppo-value-clip",
        "0.04",
        "--ppo-entropy-start",
        "0.0004",
        "--ppo-entropy-end",
        "0.00008",
        "--ppo-entropy-decay-steps",
        "1200000",
        "--bc-supervised-coef",
        "0.18",
        "--bc-supervised-min-coef",
        "0.12",
        "--bc-supervised-decay-steps",
        "1200000",
        "--bc-episodes",
        "180",
        "--bc-epochs",
        "12",
        "--bc-stage-episodes",
        "multi_low:50,multi_realistic:60,multi_hard:70",
    ],
    "planner_residual_map_tv_v4": [
        "--target-selector",
        "risk_aware",
        "--bc-policy",
        "horizon2",
        "--steam-attention",
        "--material-map",
        "--material-tv-reward",
        "--material-tv-reward-gain",
        "2.0",
        "--residual-policy",
        "--residual-base-policy",
        "horizon2",
        "--residual-beta",
        "0.20",
        "--stage-episodes",
        "multi_low:300,multi_realistic:350,multi_hard:450",
        "--action-delay-steps",
        "1",
        "--action-noise-std",
        "0.025",
        "--domain-randomization",
        "--domain-randomization-scale",
        "0.10",
        "--no-action-smoothing-penalty",
        "--ppo-lr",
        "8e-6",
        "--ppo-epochs",
        "1",
        "--ppo-clip",
        "0.04",
        "--ppo-value-clip",
        "0.04",
        "--ppo-entropy-start",
        "0.0004",
        "--ppo-entropy-end",
        "0.00008",
        "--ppo-entropy-decay-steps",
        "1200000",
        "--bc-supervised-coef",
        "0.18",
        "--bc-supervised-min-coef",
        "0.12",
        "--bc-supervised-decay-steps",
        "1200000",
        "--bc-episodes",
        "180",
        "--bc-epochs",
        "12",
        "--bc-stage-episodes",
        "multi_low:45,multi_realistic:60,multi_hard:75",
    ],
    "risk_aware": ["--target-selector", "risk_aware", "--bc-policy", "risk_aware"],
    "nearest": ["--target-selector", "nearest", "--bc-policy", "nearest"],
    "no_bc": ["--target-selector", "risk_aware", "--no-bc"],
    "no_lstm": ["--target-selector", "risk_aware", "--no-lstm"],
    "no_potential": ["--target-selector", "risk_aware", "--no-potential-shaping"],
    "no_best_progress": ["--target-selector", "risk_aware", "--no-best-progress"],
    "no_material_obs": ["--target-selector", "risk_aware", "--no-material-observation"],
    "no_action_penalty": ["--target-selector", "risk_aware", "--no-action-smoothing-penalty"],
    "no_curriculum": ["--target-selector", "risk_aware", "--no-curriculum", "--flat-stage", "multi_realistic"],
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate or execute multi-seed PPO training commands.")
    parser.add_argument("--python", default="mujoco_rl_env/bin/python")
    parser.add_argument("--run-dir", default="runs/scheme_d_paper_suite")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--methods", default="risk_aware,nearest,no_bc,no_lstm,no_potential")
    parser.add_argument("--stage-episodes", help="Optional curriculum override, e.g. multi_realistic:250.")
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--update-episodes", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--bc-episodes", type=int)
    parser.add_argument("--bc-epochs", type=int)
    parser.add_argument("--bc-stage-episodes")
    parser.add_argument("--ppo-lr", type=float)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument("--ppo-clip", type=float)
    parser.add_argument("--ppo-value-clip", type=float)
    parser.add_argument("--ppo-entropy-start", type=float)
    parser.add_argument("--ppo-entropy-end", type=float)
    parser.add_argument("--ppo-entropy-decay-steps", type=int)
    parser.add_argument("--ppo-max-grad-norm", type=float)
    parser.add_argument("--bc-supervised-coef", type=float)
    parser.add_argument("--bc-supervised-min-coef", type=float)
    parser.add_argument("--bc-supervised-decay-steps", type=int)
    parser.add_argument("--action-delay-steps", type=int)
    parser.add_argument("--action-noise-std", type=float)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--domain-randomization-scale", type=float)
    parser.add_argument("--residual-beta", type=float)
    parser.add_argument("--device", help="Training device override: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--output", default="runs/scheme_d_paper_suite/commands.txt")
    parser.add_argument("--execute", action="store_true", help="Run commands sequentially. Intended for long monitored jobs.")
    return parser


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def command_for(args, method, seed):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    run_name = f"{method}_seed{seed}"
    command = [
        args.python,
        "-m",
        "coverage_schemes.scheme_d_paper_base.train",
        "--run-dir",
        args.run_dir,
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--headless",
    ]
    command.extend(METHODS[method])
    if args.stage_episodes:
        command.extend(["--stage-episodes", args.stage_episodes])
    for attr, flag in (
        ("episode_steps", "--episode-steps"),
        ("update_episodes", "--update-episodes"),
        ("save_interval", "--save-interval"),
        ("bc_episodes", "--bc-episodes"),
        ("bc_epochs", "--bc-epochs"),
        ("bc_stage_episodes", "--bc-stage-episodes"),
        ("ppo_lr", "--ppo-lr"),
        ("ppo_epochs", "--ppo-epochs"),
        ("ppo_clip", "--ppo-clip"),
        ("ppo_value_clip", "--ppo-value-clip"),
        ("ppo_entropy_start", "--ppo-entropy-start"),
        ("ppo_entropy_end", "--ppo-entropy-end"),
        ("ppo_entropy_decay_steps", "--ppo-entropy-decay-steps"),
        ("ppo_max_grad_norm", "--ppo-max-grad-norm"),
        ("bc_supervised_coef", "--bc-supervised-coef"),
        ("bc_supervised_min_coef", "--bc-supervised-min-coef"),
        ("bc_supervised_decay_steps", "--bc-supervised-decay-steps"),
        ("action_delay_steps", "--action-delay-steps"),
        ("action_noise_std", "--action-noise-std"),
        ("domain_randomization_scale", "--domain-randomization-scale"),
        ("residual_beta", "--residual-beta"),
        ("device", "--device"),
    ):
        value = getattr(args, attr)
        if value is not None:
            command.extend([flag, str(value)])
    if args.domain_randomization:
        command.append("--domain-randomization")
    return command


def shell_join(command):
    return " ".join(command)


def main():
    args = build_arg_parser().parse_args()
    seeds = [int(item) for item in parse_csv(args.seeds)]
    methods = parse_csv(args.methods)
    commands = [command_for(args, method, seed) for method in methods for seed in seeds]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for command in commands:
            handle.write(shell_join(command) + "\n")
    print(f"Wrote {len(commands)} commands to {output}")

    if args.execute:
        for idx, command in enumerate(commands, start=1):
            print(f"[{idx}/{len(commands)}] {shell_join(command)}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
