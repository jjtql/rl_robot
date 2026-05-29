import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .algo import PPOAgent
from .config import load_config, save_config, set_global_seeds
from .env import ShangZengEnv
from .policies import (
    action_toward_steam,
    build_base_policy,
    build_residual_base_policy,
    combine_residual_action,
    residual_beta_for_env,
    shield_residual_action,
    select_distance_age_steam,
    select_nearest_steam,
    select_risk_aware_steam,
)


BC_POLICY_CHOICES = (
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
RESIDUAL_GLUE_CHOICES = ("fixed", "phase_aware")
TARGET_SELECTOR_CHOICES = ("nearest", "risk_aware")
STAGE_CHOICES = ("single_easy", "single_precision", "multi_low", "multi_realistic", "multi_hard", "multi_extreme")


def residual_beta_at_step(config, total_steps):
    base_beta = float(config.get("residual_beta", 0.25))
    warmup_steps = int(config.get("residual_beta_warmup_steps", 0) or 0)
    if warmup_steps <= 0:
        return base_beta
    start = config.get("residual_beta_start")
    end = config.get("residual_beta_end")
    start = base_beta if start is None else float(start)
    end = base_beta if end is None else float(end)
    progress = min(max(float(total_steps) / max(warmup_steps, 1), 0.0), 1.0)
    return float(start + progress * (end - start))


def new_memory():
    return {
        "s": [],
        "s_next": [],
        "a": [],
        "r": [],
        "lp": [],
        "v": [],
        "d": [],
        "reset": [],
        "pred_target": [],
        "pred_mask": [],
        "bc_a": [],
        "bc_mask": [],
    }


def select_expert_steam(env, policy):
    if policy == "nearest":
        return select_nearest_steam(env)
    if policy == "oldest":
        return max(env.steams, key=lambda item: item["age"]) if env.steams else None
    if policy == "distance_age":
        return select_distance_age_steam(env)
    if policy == "risk_aware":
        return select_risk_aware_steam(env)
    raise ValueError(f"Unknown BC expert policy: {policy}")


def expert_action(env, policy="risk_aware"):
    if policy in ("dynamic_weighted", "horizon2", "horizon3", "aco_tsp", "planner_ensemble"):
        controller = build_base_policy(policy)
        return controller.act(env, None)
    return action_toward_steam(env, select_expert_steam(env, policy))


def collect_behavior_clone_data(env, episodes, max_steps, seed, expert_policy="risk_aware", stage_plan=None):
    states = []
    actions = []
    if stage_plan is None:
        stage_plan = [{"name": "single_easy", "episodes": int(episodes)}]
    stage_plan = [
        {"name": str(stage["name"]), "episodes": int(stage["episodes"])}
        for stage in stage_plan
        if int(stage["episodes"]) > 0
    ]

    previous_stage = getattr(env, "curriculum_stage", "single_easy")
    previous_max_steps = env.max_episode_steps
    previous_target_success_count = env.target_success_count
    previous_target_coverage = env.target_coverage
    stage_samples = {}
    stage_episodes = {}
    episode_offset = 0

    try:
        for stage in stage_plan:
            stage_name = stage["name"]
            env.configure_curriculum(stage_name)
            env.max_episode_steps = max_steps
            env.target_success_count = max_steps + 1
            env.target_coverage = 1.0
            stage_samples[stage_name] = 0
            stage_episodes[stage_name] = stage["episodes"]

            controller = build_base_policy(expert_policy)
            for _ in range(stage["episodes"]):
                s, info = env.reset(seed=seed + 10_000 + episode_offset)
                controller.reset()
                previous_missed = info.get("missed_count", 0)
                episode_offset += 1
                for _ in range(max_steps):
                    a = controller.act(env, s)
                    states.append(s)
                    actions.append(a)
                    stage_samples[stage_name] += 1
                    s, _, _, truncated, info = env.step(a)
                    if info.get("covered", False) or info.get("missed_count", 0) > previous_missed:
                        previous_missed = info.get("missed_count", 0)
                    if truncated:
                        break
    finally:
        env.configure_curriculum(previous_stage)
        env.max_episode_steps = previous_max_steps
        env.target_success_count = previous_target_success_count
        env.target_coverage = previous_target_coverage

    stats = {
        "policy": expert_policy,
        "episodes": episode_offset,
        "samples": len(states),
        "stage_episodes": stage_episodes,
        "stage_samples": stage_samples,
    }
    return states, actions, stats


def allocate_stage_episodes(stage_plan, total_episodes):
    total_episodes = int(total_episodes)
    if total_episodes <= 0:
        return []
    if not stage_plan:
        return [{"name": "single_easy", "episodes": total_episodes}]

    cleaned = [
        {"name": str(stage["name"]), "episodes": max(int(stage["episodes"]), 0)}
        for stage in stage_plan
    ]
    weight_sum = sum(stage["episodes"] for stage in cleaned)
    if weight_sum <= 0:
        return [{"name": "single_easy", "episodes": total_episodes}]

    raw_counts = [total_episodes * stage["episodes"] / weight_sum for stage in cleaned]
    counts = [int(np.floor(value)) for value in raw_counts]
    remainder = total_episodes - sum(counts)
    order = sorted(range(len(cleaned)), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    return [
        {"name": stage["name"], "episodes": count}
        for stage, count in zip(cleaned, counts)
        if count > 0
    ]


def resolve_bc_stage_plan(config):
    return allocate_stage_episodes(config.get("bc_stages"), int(config.get("bc_episodes", 0)))


def parse_stage_episodes(value):
    if not value:
        return None
    stages = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, episodes = item.split(":", 1)
        episodes = int(episodes)
        if episodes < 0:
            raise ValueError(f"Episode count must be non-negative: {item}")
        stages.append({"name": name.strip(), "episodes": episodes})
    return stages


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the paper-ready scheme_d baseline.")
    parser.add_argument("--config", help="Optional JSON config override.")
    parser.add_argument("--run-dir", default="runs/scheme_d_paper_base", help="Root output directory.")
    parser.add_argument("--run-name", help="Run name. Defaults to timestamp plus seed.")
    parser.add_argument("--model-path", help="MuJoCo XML path.")
    parser.add_argument("--seed", type=int, help="Global seed.")
    parser.add_argument("--episode-steps", type=int, help="Episode length override.")
    parser.add_argument("--update-episodes", type=int, help="PPO update interval in episodes.")
    parser.add_argument("--save-interval", type=int, help="Checkpoint save interval in episodes.")
    parser.add_argument("--ppo-lr", type=float, help="PPO learning rate.")
    parser.add_argument("--ppo-epochs", type=int, help="PPO epochs per update.")
    parser.add_argument("--ppo-clip", type=float, help="PPO policy clip range.")
    parser.add_argument("--ppo-value-clip", type=float, help="PPO value clip range.")
    parser.add_argument("--ppo-reward-scale", type=float, help="Scale applied to rewards before GAE.")
    parser.add_argument("--ppo-reward-clip", type=float, help="Absolute reward clip after scaling.")
    parser.add_argument("--ppo-entropy-start", type=float, help="Initial entropy coefficient.")
    parser.add_argument("--ppo-entropy-end", type=float, help="Final entropy coefficient.")
    parser.add_argument("--ppo-entropy-decay-steps", type=int, help="Steps over which entropy coefficient is annealed.")
    parser.add_argument("--ppo-max-grad-norm", type=float, help="PPO gradient clipping norm.")
    parser.add_argument("--pred-coef", type=float, help="Auxiliary next-steam prediction loss coefficient.")
    parser.add_argument("--prediction-horizon-steps", type=int, help="Retroactively label this many pre-spawn steps for prediction.")
    parser.add_argument("--bc-supervised-coef", type=float, help="Auxiliary behavior-cloning loss coefficient during PPO.")
    parser.add_argument("--bc-supervised-min-coef", type=float, help="Minimum auxiliary BC loss coefficient after decay.")
    parser.add_argument("--bc-supervised-decay-steps", type=int, help="Steps over which auxiliary BC loss decays.")
    parser.add_argument("--target-selector", choices=TARGET_SELECTOR_CHOICES, help="Target features/reward-shaping selector.")
    parser.add_argument("--steam-attention", action="store_true", help="Use a steam-set attention encoder before LSTM PPO.")
    parser.add_argument("--attention-steam-count", type=int, help="Number of active steam points exposed to the attention encoder.")
    parser.add_argument("--spawn-history-observation", action="store_true", help="Append recent steam spawn history features for LSTM prediction.")
    parser.add_argument("--thermal-context-observation", action="store_true", help="Append thermal hotspot and spawn-pressure context features.")
    parser.add_argument("--route-summary-observation", action="store_true", help="Append planner-style route summary features.")
    parser.add_argument("--material-map", action="store_true", help="Append a compact material/frontier map and use the attention-map PPO encoder.")
    parser.add_argument("--material-tv-reward", action="store_true", help="Reward reductions in material hole/TV/overfill loss.")
    parser.add_argument("--material-tv-reward-gain", type=float, help="Gain for material TV/quality shaping.")
    parser.add_argument("--no-lstm", action="store_true", help="Use a feed-forward PPO policy instead of LSTM PPO.")
    parser.add_argument("--no-curriculum", action="store_true", help="Train only one flat stage instead of the curriculum.")
    parser.add_argument("--flat-stage", choices=STAGE_CHOICES, help="Stage used by --no-curriculum.")
    parser.add_argument("--no-potential-shaping", action="store_true", help="Disable distance-potential reward shaping.")
    parser.add_argument("--no-best-progress", action="store_true", help="Disable per-steam best-progress reward.")
    parser.add_argument("--cover-reward-scale", type=float, help="Multiplier for the per-steam cover reward after curriculum setup.")
    parser.add_argument("--quick-cover-bonus-scale", type=float, help="Multiplier for the quick-cover latency bonus.")
    parser.add_argument("--precision-bonus-scale", type=float, help="Multiplier for the within-radius precision bonus.")
    parser.add_argument("--potential-gain-scale", type=float, help="Multiplier for distance-potential shaping after curriculum setup.")
    parser.add_argument("--best-progress-gain-scale", type=float, help="Multiplier for per-steam best-progress shaping.")
    parser.add_argument("--active-steam-penalty-scale", type=float, help="Multiplier for active-steam count penalty.")
    parser.add_argument("--age-penalty-scale", type=float, help="Multiplier for active-steam age penalty.")
    parser.add_argument("--no-material-observation", action="store_true", help="Zero material and material-risk observation features.")
    parser.add_argument("--no-action-smoothing-penalty", action="store_true", help="Disable action delta/L2 reward penalty.")
    parser.add_argument("--latency-first-reward", action="store_true", help="Prioritize response latency, SLA, and backlog in reward shaping.")
    parser.add_argument("--decision-dt-seconds", type=float, help="Real high-level control period used for training-time latency/SLA metrics.")
    parser.add_argument("--cover-latency-penalty-gain", type=float, help="Per-cover latency penalty gain used by --latency-first-reward.")
    parser.add_argument("--oldest-active-penalty-gain", type=float, help="Per-step penalty for the oldest active steam age.")
    parser.add_argument("--backlog-penalty-gain", type=float, help="Per-step penalty for active steam backlog.")
    parser.add_argument("--response-sla-steps", type=int, help="Latency target in simulator steps for response-speed metrics and reward.")
    parser.add_argument("--response-sla-seconds", type=float, help="Latency target in real high-level control seconds; converted to steps using --decision-dt-seconds.")
    parser.add_argument("--response-sla-bonus", type=float, help="Bonus for covering before response SLA.")
    parser.add_argument("--response-sla-miss-penalty", type=float, help="Penalty for covering after response SLA.")
    parser.add_argument("--continuous-session-chunks", type=int, help="Number of episode windows kept inside one physical session.")
    parser.add_argument("--carry-lstm-state-across-chunks", action="store_true", help="Carry LSTM hidden state across continuous-session chunks.")
    parser.add_argument("--lstm-sequence-chunks", type=int, help="Number of episode windows per PPO recurrent training sequence.")
    parser.add_argument("--action-delay-steps", type=int, help="Apply an N-step command delay during training.")
    parser.add_argument("--action-noise-std", type=float, help="Stddev of Gaussian action noise inside the environment.")
    parser.add_argument("--domain-randomization", action="store_true", help="Randomize motion/material parameters at episode reset.")
    parser.add_argument("--domain-randomization-scale", type=float, help="Relative randomization scale.")
    parser.add_argument("--no-thermal-spawn", action="store_true", help="Disable high-temperature-region driven steam spawning.")
    parser.add_argument("--thermal-hotspot-count", type=int, help="Number of latent high-temperature regions.")
    parser.add_argument("--thermal-hotspot-sigma", type=float, help="Spatial radius of each high-temperature region.")
    parser.add_argument("--thermal-hotspot-strength", type=float, help="Spawn weighting strength for high-temperature regions.")
    parser.add_argument("--thermal-background-weight", type=float, help="Non-hot-region spawn floor to keep background steam events.")
    parser.add_argument("--thermal-drift-std", type=float, help="Per-step random-walk drift of high-temperature regions.")
    parser.add_argument("--thermal-refresh-probability", type=float, help="Per-step probability of refreshing each high-temperature region.")
    parser.add_argument("--thermal-lifetime-steps", type=int, help="Nominal lifetime of a high-temperature region.")
    parser.add_argument("--thermal-recent-spawn-radius", type=float, help="Radius used to suppress repeated spawns at the same location.")
    parser.add_argument("--thermal-recent-spawn-suppression", type=float, help="Strength of recent-spawn suppression.")
    parser.add_argument("--thermal-recent-spawn-memory", type=int, help="Number of recent spawn locations used for suppression.")
    parser.add_argument("--burst-lull-spawn", action="store_true", help="Use thermal burst/lull spawn cycles instead of immediate force-spawn.")
    parser.add_argument("--burst-lull-lull-steps", type=int, help="No-new-burst lull length after coverage events.")
    parser.add_argument("--burst-lull-charge-steps", type=int, help="Accumulation steps required before the next thermal burst.")
    parser.add_argument("--burst-lull-sparse-threshold", type=int, help="Active-steam count at or below which a charged burst may fire.")
    parser.add_argument("--burst-lull-burst-min", type=int, help="Minimum steam points spawned in a thermal burst.")
    parser.add_argument("--burst-lull-burst-max", type=int, help="Maximum steam points spawned in a thermal burst.")
    parser.add_argument("--burst-lull-burst-interval-steps", type=int, help="Step gap between individual steam spawns inside one burst.")
    parser.add_argument("--burst-lull-trickle-probability", type=float, help="Small sparse-spawn probability during lull after partial charge.")
    parser.add_argument("--residual-policy", action="store_true", help="Train the actor as a residual around a rule/planner base controller.")
    parser.add_argument("--residual-base-policy", choices=BC_POLICY_CHOICES, help="Base controller for residual PPO.")
    parser.add_argument("--residual-beta", type=float, help="Scale applied to the learned residual action.")
    parser.add_argument("--residual-beta-start", type=float, help="Initial residual action scale for release schedules.")
    parser.add_argument("--residual-beta-end", type=float, help="Final residual action scale after the release warmup.")
    parser.add_argument("--residual-beta-warmup-steps", type=int, help="Steps over which residual beta increases from start to end.")
    parser.add_argument("--no-residual-guard", action="store_true", help="Disable residual direction guard.")
    parser.add_argument("--residual-action-shield", action="store_true", help="Keep residual actions from undoing planner progress.")
    parser.add_argument("--residual-glue", choices=RESIDUAL_GLUE_CHOICES, help="How planner and LSTM residual actions are combined.")
    parser.add_argument("--residual-sparse-base-policy", choices=BC_POLICY_CHOICES, help="Base controller used in sparse/lull phases.")
    parser.add_argument("--residual-dense-base-policy", choices=BC_POLICY_CHOICES, help="Base controller used in dense/burst phases.")
    parser.add_argument("--residual-phase-sparse-threshold", type=int, help="Active-steam count treated as sparse for phase-aware residual glue.")
    parser.add_argument("--residual-phase-dense-threshold", type=int, help="Active-steam count treated as dense for phase-aware residual glue.")
    parser.add_argument("--residual-sparse-beta-scale", type=float, help="Residual beta multiplier for sparse visible-target phases.")
    parser.add_argument("--residual-lull-beta-scale", type=float, help="Residual beta multiplier during explicit burst/lull quiet phases.")
    parser.add_argument("--residual-charging-beta-scale", type=float, help="Residual beta multiplier while spawn charge accumulates.")
    parser.add_argument("--residual-dense-beta-scale", type=float, help="Residual beta multiplier when many targets are active.")
    parser.add_argument("--residual-burst-beta-scale", type=float, help="Residual beta multiplier while a sequential burst is being emitted.")
    parser.add_argument("--stagnation-recovery-steps", type=int, help="Steps without coverage before the shield becomes stricter.")
    parser.add_argument("--keep-lstm-state-on-cover", action="store_true", help="Do not reset recurrent state when a steam is covered.")
    parser.add_argument("--keep-lstm-state-on-miss", action="store_true", help="Do not reset recurrent state when a steam is missed.")
    parser.add_argument("--device", help="Training device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--bc-episodes", type=int, help="Behavior cloning warm-start episodes.")
    parser.add_argument("--bc-epochs", type=int, help="Behavior cloning epochs.")
    parser.add_argument("--bc-policy", choices=BC_POLICY_CHOICES, help="Rule expert used for behavior cloning.")
    parser.add_argument(
        "--bc-stage-episodes",
        help="BC collection mix, e.g. single_easy:10,multi_low:20,multi_realistic:20.",
    )
    parser.add_argument("--stage-episodes", help="Override curriculum, e.g. single_easy:2,multi_low:2.")
    parser.add_argument("--no-bc", action="store_true", help="Disable behavior cloning warm start.")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib training plots.")
    parser.add_argument("--headless", action="store_true", help="Force headless mode.")
    return parser


def apply_cli_overrides(config, args):
    for key in (
        "model_path",
        "seed",
        "episode_steps",
        "update_episodes",
        "save_interval",
        "bc_epochs",
        "ppo_lr",
        "ppo_epochs",
        "ppo_clip",
        "ppo_value_clip",
        "ppo_reward_scale",
        "ppo_reward_clip",
        "ppo_entropy_start",
        "ppo_entropy_end",
        "ppo_entropy_decay_steps",
        "ppo_max_grad_norm",
        "pred_coef",
        "prediction_horizon_steps",
        "bc_supervised_coef",
        "bc_supervised_min_coef",
        "bc_supervised_decay_steps",
        "action_delay_steps",
        "action_noise_std",
        "domain_randomization_scale",
        "thermal_hotspot_count",
        "thermal_hotspot_sigma",
        "thermal_hotspot_strength",
        "thermal_background_weight",
        "thermal_drift_std",
        "thermal_refresh_probability",
        "thermal_lifetime_steps",
        "thermal_recent_spawn_radius",
        "thermal_recent_spawn_suppression",
        "thermal_recent_spawn_memory",
        "burst_lull_lull_steps",
        "burst_lull_charge_steps",
        "burst_lull_sparse_threshold",
        "burst_lull_burst_min",
        "burst_lull_burst_max",
        "burst_lull_burst_interval_steps",
        "burst_lull_trickle_probability",
        "attention_steam_count",
        "residual_base_policy",
        "residual_glue",
        "residual_sparse_base_policy",
        "residual_dense_base_policy",
        "residual_phase_sparse_threshold",
        "residual_phase_dense_threshold",
        "residual_sparse_beta_scale",
        "residual_lull_beta_scale",
        "residual_charging_beta_scale",
        "residual_dense_beta_scale",
        "residual_burst_beta_scale",
        "residual_beta",
        "residual_beta_start",
        "residual_beta_end",
        "residual_beta_warmup_steps",
        "stagnation_recovery_steps",
        "cover_reward_scale",
        "quick_cover_bonus_scale",
        "precision_bonus_scale",
        "potential_gain_scale",
        "best_progress_gain_scale",
        "active_steam_penalty_scale",
        "age_penalty_scale",
        "cover_latency_penalty_gain",
        "oldest_active_penalty_gain",
        "backlog_penalty_gain",
        "decision_dt_seconds",
        "response_sla_steps",
        "response_sla_seconds",
        "response_sla_bonus",
        "response_sla_miss_penalty",
        "continuous_session_chunks",
        "lstm_sequence_chunks",
        "device",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.target_selector is not None:
        config["target_selector"] = args.target_selector
    if args.steam_attention:
        config["use_steam_attention"] = True
        config["use_lstm"] = True
    if args.spawn_history_observation:
        config["use_spawn_history_observation"] = True
        config["use_lstm"] = True
    if args.thermal_context_observation:
        config["use_thermal_context_observation"] = True
        config["use_lstm"] = True
    if args.route_summary_observation:
        config["use_route_summary_observation"] = True
        config["use_lstm"] = True
    if args.material_map:
        config["use_material_map"] = True
        config["use_steam_attention"] = True
        config["use_lstm"] = True
    if args.material_tv_reward:
        config["material_tv_reward"] = True
    if args.material_tv_reward_gain is not None:
        config["material_tv_reward_gain"] = args.material_tv_reward_gain
    if args.no_lstm:
        config["use_lstm"] = False
    if config.get("use_material_map", False):
        config["use_steam_attention"] = True
    if config.get("use_steam_attention", False):
        config["use_lstm"] = True
    if args.flat_stage is not None:
        config["flat_stage"] = args.flat_stage
    if args.no_potential_shaping:
        config["potential_shaping"] = False
    if args.no_best_progress:
        config["best_progress_reward"] = False
    if args.no_material_observation:
        config["material_observation"] = False
    if args.no_action_smoothing_penalty:
        config["action_smoothing_penalty"] = False
    if args.latency_first_reward:
        config["latency_first_reward"] = True
    if args.carry_lstm_state_across_chunks:
        config["carry_lstm_state_across_chunks"] = True
    if args.domain_randomization:
        config["domain_randomization"] = True
    if args.no_thermal_spawn:
        config["thermal_spawn"] = False
    if args.burst_lull_spawn:
        config["burst_lull_spawn"] = True
    if args.residual_policy:
        config["residual_policy"] = True
    if args.no_residual_guard:
        config["residual_guard"] = False
    if args.residual_action_shield:
        config["residual_action_shield"] = True
    if args.keep_lstm_state_on_cover:
        config["recurrent_reset_on_cover"] = False
    if args.keep_lstm_state_on_miss:
        config["recurrent_reset_on_miss"] = False
    if args.bc_policy is not None:
        config["bc_policy"] = args.bc_policy
    bc_stage_override = parse_stage_episodes(args.bc_stage_episodes)
    if bc_stage_override is not None:
        config["bc_stages"] = bc_stage_override
        if args.bc_episodes is None:
            config["bc_episodes"] = sum(stage["episodes"] for stage in bc_stage_override)
    if args.bc_episodes is not None:
        config["bc_episodes"] = args.bc_episodes
    if args.no_bc:
        config["bc_warm_start"] = False
        config["bc_episodes"] = 0
    if args.plot:
        config["plot"] = True
        config["headless"] = False
    if args.headless:
        config["headless"] = True
        config["plot"] = False
    stage_override = parse_stage_episodes(args.stage_episodes)
    if stage_override is not None:
        config["curriculum"] = stage_override
    if args.no_curriculum:
        config["use_curriculum"] = False
        total_episodes = sum(stage["episodes"] for stage in config["curriculum"])
        config["curriculum"] = [{"name": config.get("flat_stage", "multi_realistic"), "episodes": total_episodes}]
    return config


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
    env.configure_response_timing(
        decision_dt_seconds=config.get("decision_dt_seconds"),
        response_sla_seconds=config.get("response_sla_seconds"),
        response_sla_steps=env.response_sla_steps,
    )
    config["response_sla_steps"] = int(env.response_sla_steps)
    config["response_sla_seconds"] = float(env.response_sla_seconds)
    config["decision_dt_seconds"] = float(env.metric_step_seconds)
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


class RunLogger:
    def __init__(self, run_path):
        self.run_path = Path(run_path)
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.episode_csv = self.run_path / "episodes.csv"
        self.events_jsonl = self.run_path / "events.jsonl"
        self._csv_file = self.episode_csv.open("w", newline="", encoding="utf-8")
        self._writer = None

    def log_event(self, event, payload):
        row = {"time": datetime.now().isoformat(timespec="seconds"), "event": event}
        row.update(payload)
        with self.events_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def log_episode(self, row):
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        self._csv_file.close()


def save_checkpoint(agent, ep, config, run_path):
    checkpoint_dir = Path(run_path) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prefix = checkpoint_dir / config["checkpoint_prefix"]
    torch.save(agent.model.state_dict(), f"{prefix}_{ep}.pth")
    torch.save(
        {
            "model": agent.model.state_dict(),
            "optimizer": agent.opt.state_dict(),
            "icm": agent.icm.state_dict(),
            "icm_optimizer": agent.icm_opt.state_dict(),
            "episode": ep,
            "config": config,
            "obs_dim": agent.s_dim,
            "action_dim": agent.a_dim,
            "seq_len": agent.seq_len,
            "use_lstm": agent.use_lstm,
            "use_steam_attention": agent.use_steam_attention,
            "use_material_map": agent.use_material_map,
            "base_obs_dim": agent.base_obs_dim,
            "attention_steam_count": agent.attention_steam_count,
            "attention_steam_dim": agent.attention_steam_dim,
            "device": str(agent.device),
            "train_steps": agent.train_steps,
        },
        f"{prefix}_{ep}_full.pt",
    )


def maybe_init_plot(enabled):
    if not enabled:
        return None, None
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    return plt, ax


def update_plot(plt, ax, history, stage_coverages):
    if plt is None or ax is None:
        return
    ax[0, 0].cla()
    ax[0, 0].plot(history["reward"])
    ax[0, 0].set_title("Reward")
    ax[0, 1].cla()
    ax[0, 1].plot(history["coverage"])
    ax[0, 1].set_title("Coverage")
    ax[1, 0].cla()
    ax[1, 0].plot(history["target_distance"])
    ax[1, 0].set_title("Target Distance")
    ax[1, 1].cla()
    if stage_coverages:
        ax[1, 1].bar(range(len(stage_coverages[-50:])), stage_coverages[-50:])
    ax[1, 1].set_title("Recent Coverage (per ep)")
    plt.tight_layout()
    plt.pause(0.01)


def train(config, run_path):
    set_global_seeds(int(config["seed"]))
    env = ShangZengEnv(
        model_path=config["model_path"],
        max_episode_steps=config["episode_steps"],
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
    obs_dim = env.observation_space.shape[0]
    config["base_obs_dim"] = int(env.base_obs_dim)
    recurrent_sequence_steps = int(config["episode_steps"]) * max(int(config.get("lstm_sequence_chunks", 1)), 1)
    config["recurrent_sequence_steps"] = int(recurrent_sequence_steps)
    agent = PPOAgent(
        obs_dim,
        env.action_space.shape[0],
        seq_len=recurrent_sequence_steps,
        use_lstm=config.get("use_lstm", True),
        use_steam_attention=config.get("use_steam_attention", False),
        use_material_map=config.get("use_material_map", False),
        base_obs_dim=config.get("base_obs_dim", 35),
        attention_steam_count=config.get("attention_steam_count", 6),
        attention_steam_dim=config.get("attention_steam_dim", 8),
        lr=config.get("ppo_lr", 1e-4),
        ppo_epochs=config.get("ppo_epochs", 4),
        clip_param=config.get("ppo_clip", 0.12),
        value_clip_param=config.get("ppo_value_clip", 0.12),
        reward_scale=config.get("ppo_reward_scale", 0.02),
        reward_clip=config.get("ppo_reward_clip", 4.0),
        entropy_coef_start=config.get("ppo_entropy_start", 0.003),
        entropy_coef_end=config.get("ppo_entropy_end", 0.0005),
        entropy_decay_steps=config.get("ppo_entropy_decay_steps", 240000),
        max_grad_norm=config.get("ppo_max_grad_norm", 0.35),
        pred_coef=config.get("pred_coef", 0.0),
        bc_supervised_coef=config.get("bc_supervised_coef", 0.03),
        bc_supervised_min_coef=config.get("bc_supervised_min_coef", 0.0),
        bc_supervised_decay_steps=config.get("bc_supervised_decay_steps", 240000),
        device=config.get("device", "auto"),
    )
    config["device"] = str(agent.device)
    if not config.get("action_smoothing_penalty", True):
        agent.smooth_coef = 0.0
        agent.action_l2_coef = 0.0
    update_timestep = max(config["update_episodes"] * config["episode_steps"], recurrent_sequence_steps)

    logger = RunLogger(run_path)
    save_config(config, Path(run_path) / "config.json")
    logger.log_event("start", {"config": config})

    memory = new_memory()
    total_steps = 0
    smooth_reward = -100.0
    last_update_stats = {}
    history = defaultdict(list)
    plt, ax = maybe_init_plot(config.get("plot", False))
    start_time = time.time()

    if config.get("bc_warm_start", False) and config.get("bc_episodes", 0) > 0:
        bc_stage_plan = resolve_bc_stage_plan(config)
        bc_policy = config.get("bc_policy", "risk_aware")
        stage_label = ",".join(f"{stage['name']}:{stage['episodes']}" for stage in bc_stage_plan)
        print(f"\nCollecting {bc_policy} rule-policy warm-start data ({stage_label})...", flush=True)
        bc_states, bc_actions, bc_collect_stats = collect_behavior_clone_data(
            env,
            episodes=config["bc_episodes"],
            max_steps=min(config["episode_steps"], 500),
            seed=int(config["seed"]),
            expert_policy=bc_policy,
            stage_plan=bc_stage_plan,
        )
        if config.get("residual_policy", False):
            clone_actions = np.zeros_like(np.asarray(bc_actions, dtype=np.float32))
        else:
            clone_actions = bc_actions
        bc_stats = agent.behavior_clone(bc_states, clone_actions, epochs=config["bc_epochs"])
        logger.log_event("behavior_clone", {**bc_collect_stats, **bc_stats})
        print(
            f"Warm start samples: {len(bc_states)} | "
            f"BC loss: {bc_stats.get('bc_loss', 0.0):.6f}",
            flush=True,
        )
        save_checkpoint(agent, "bc_ready", config, run_path)

    save_checkpoint(agent, "latest", config, run_path)

    global_ep = 0
    total_curriculum_episodes = sum(stage["episodes"] for stage in config["curriculum"])

    try:
        for stage in config["curriculum"]:
            env.configure_curriculum(stage["name"])
            if config.get("continuous_training", False):
                env.max_episode_steps = config["episode_steps"]
                env.target_success_count = config["continuous_success_count"]
                env.target_coverage = 1.0
            print(
                f"\n{'='*50}\n"
                f"Stage: {stage['name']}\n"
                f"Episodes: {stage['episodes']}\n"
                f"Max steams: {env.max_steams}\n"
                f"Cover radius: {env.cover_radius}\n"
                f"Cover reward: {env.cover_reward}\n"
                f"Time penalty: {env.time_penalty}\n"
                f"Potential gain: {env.potential_gain}\n"
                f"Reward scales: cover={env.cover_reward_scale}, quick={env.quick_cover_bonus_scale}, "
                f"precision={env.precision_bonus_scale}, potential={env.potential_gain_scale}, "
                f"best_progress={env.best_progress_gain_scale}\n"
                f"Target selector: {env.target_selector}\n"
                f"LSTM: {agent.use_lstm}\n"
                f"Steam attention: {agent.use_steam_attention}\n"
                f"Attention steam count: {getattr(agent, 'attention_steam_count', 0)}\n"
                f"Spawn history obs: {config.get('use_spawn_history_observation', False)}\n"
                f"Thermal context obs: {config.get('use_thermal_context_observation', False)}\n"
                f"Route summary obs: {config.get('use_route_summary_observation', False)}\n"
                f"Material map: {agent.use_material_map}\n"
                f"Device: {agent.device}\n"
                f"Prediction coef/horizon: {agent.pred_coef}/{config.get('prediction_horizon_steps', 0)}\n"
                f"Material TV reward: {env.material_tv_reward_enabled}\n"
                f"Residual policy: {config.get('residual_policy', False)}\n"
                f"Residual base: {config.get('residual_base_policy', 'risk_aware')}\n"
                f"Residual glue: {config.get('residual_glue', 'fixed')} "
                f"(sparse={config.get('residual_sparse_base_policy', 'horizon2')}, "
                f"dense={config.get('residual_dense_base_policy', 'dynamic_weighted')}, "
                f"beta scales sparse/lull/charging/dense/burst="
                f"{config.get('residual_sparse_beta_scale', 1.0)}/"
                f"{config.get('residual_lull_beta_scale', 1.0)}/"
                f"{config.get('residual_charging_beta_scale', 1.0)}/"
                f"{config.get('residual_dense_beta_scale', 1.0)}/"
                f"{config.get('residual_burst_beta_scale', 1.0)})\n"
                f"Residual shield: {config.get('residual_action_shield', False)} "
                f"(recovery={config.get('stagnation_recovery_steps', 180)})\n"
                f"Residual beta: {config.get('residual_beta', 0.25)} "
                f"(schedule {config.get('residual_beta_start')} -> {config.get('residual_beta_end')} "
                f"over {config.get('residual_beta_warmup_steps', 0)} steps)\n"
                f"Latency-first reward: {env.latency_first_reward_enabled} "
                f"(sla={env.response_sla_steps} steps/{env.response_sla_seconds:.2f}s, "
                f"dt={env.metric_step_seconds:.3f}s, latency_gain={env.cover_latency_penalty_gain}, "
                f"oldest_gain={env.oldest_active_penalty_gain}, backlog_gain={env.backlog_penalty_gain})\n"
                f"Continuous session chunks: {config.get('continuous_session_chunks', 1)} "
                f"(carry_lstm={config.get('carry_lstm_state_across_chunks', False)}, "
                f"seq_steps={config.get('recurrent_sequence_steps', config['episode_steps'])})\n"
                f"Action delay/noise: {env.action_delay_steps}/{env.action_noise_std}\n"
                f"Domain randomization: {env.domain_randomization_enabled}\n"
                f"Thermal spawn: {env.thermal_spawn_enabled} "
                f"(hotspots={env.thermal_hotspot_count}, bg={env.thermal_background_weight}, "
                f"strength={env.thermal_hotspot_strength})\n"
                f"Burst-lull spawn: {env.burst_lull_spawn_enabled} "
                f"(lull={env.burst_lull_lull_steps}, charge={env.burst_lull_charge_steps}, "
                f"threshold={env.burst_lull_sparse_threshold}, "
                f"burst={env.burst_lull_burst_min}-{env.burst_lull_burst_max}, "
                f"interval={env.burst_lull_burst_interval_steps})\n"
                f"{'='*50}",
                flush=True,
            )

            stage_rewards = []
            stage_coverages = []
            residual_base_controller = build_residual_base_policy(config)
            bc_controller = build_base_policy(config.get("bc_policy", "risk_aware"))
            session_chunks = max(int(config.get("continuous_session_chunks", 1)), 1)
            carry_session_hidden = (
                bool(config.get("carry_lstm_state_across_chunks", False))
                and bool(agent.use_lstm)
                and session_chunks > 1
            )
            session_hx, session_cx = None, None

            for ep_in_stage in range(stage["episodes"]):
                chunk_index = ep_in_stage % session_chunks
                new_session = chunk_index == 0
                last_session_chunk = chunk_index == session_chunks - 1 or ep_in_stage == stage["episodes"] - 1
                if new_session:
                    s, info = env.reset(seed=int(config["seed"]) + global_ep)
                    residual_base_controller.reset()
                    bc_controller.reset()
                    session_hx, session_cx = None, None
                else:
                    s, info = env.start_new_chunk()
                    if not carry_session_hidden:
                        residual_base_controller.reset()
                        bc_controller.reset()
                        session_hx, session_cx = None, None
                ep_r = 0.0
                ep_covered = 0
                ep_action_delta = []
                ep_action_l2 = []
                ep_hidden_resets = 0
                ep_cover_resets = 0
                ep_cover_keeps = 0
                ep_miss_resets = 0
                ep_spawn_events = 0
                ep_residual_modes = defaultdict(int)
                ep_residual_base_counts = defaultdict(int)
                ep_residual_betas = []
                pred_supervised_indices = set()
                hx, cx = (session_hx, session_cx) if carry_session_hidden and not new_session else (None, None)
                previous_missed = info.get("missed_count", 0)
                previous_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                previous_env_action = np.asarray(getattr(env, "last_action", previous_action), dtype=np.float32).copy()
                episode_memory_indices = []

                for _ in range(config["episode_steps"]):
                    hidden_reset = float(hx is None or cx is None)
                    if hidden_reset > 0.5:
                        ep_hidden_resets += 1
                    bc_action = bc_controller.act(env, s)
                    raw_a, lp, v, pred_xy, hx_n, cx_n = agent.select_action(s, hx, cx)
                    scheduled_residual_beta = residual_beta_at_step(config, total_steps)
                    current_residual_beta = scheduled_residual_beta
                    if config.get("residual_policy", False):
                        base_action = residual_base_controller.act(env, s)
                        current_residual_beta = residual_beta_for_env(config, env, scheduled_residual_beta)
                        ep_residual_modes[getattr(residual_base_controller, "last_mode", "fixed")] += 1
                        ep_residual_base_counts[
                            getattr(
                                residual_base_controller,
                                "last_policy_name",
                                config.get("residual_base_policy", "risk_aware"),
                            )
                        ] += 1
                        ep_residual_betas.append(float(current_residual_beta))
                        a = combine_residual_action(
                            base_action,
                            raw_a,
                            beta=current_residual_beta,
                            guard=config.get("residual_guard", True),
                        )
                        if config.get("residual_action_shield", False):
                            a = shield_residual_action(
                                env,
                                base_action,
                                a,
                                previous_action=previous_env_action,
                                stagnation_steps=getattr(env, "steps_since_cover", 0),
                                recovery_steps=config.get("stagnation_recovery_steps", 180),
                            )
                        bc_action_for_loss = np.zeros_like(raw_a, dtype=np.float32)
                    else:
                        a = raw_a
                        bc_action_for_loss = bc_action
                    env.set_prediction_marker(pred_xy)
                    s_n, r_ext, terminated, truncated, info = env.step(a)

                    memory["s"].append(s)
                    memory["s_next"].append(s_n)
                    memory["a"].append(raw_a)
                    memory["r"].append(r_ext)
                    memory["lp"].append(lp)
                    memory["v"].append(v)
                    continues_after_chunk = (
                        carry_session_hidden
                        and truncated
                        and not terminated
                        and not last_session_chunk
                    )
                    memory["d"].append(float(terminated or (truncated and not continues_after_chunk)))
                    memory["reset"].append(hidden_reset)
                    memory["pred_target"].append(info["pred_target"])
                    memory["pred_mask"].append(info["pred_mask"])
                    memory["bc_a"].append(bc_action_for_loss)
                    memory["bc_mask"].append(1.0 if env.steams else 0.0)
                    episode_memory_indices.append(len(memory["pred_target"]) - 1)

                    if info.get("pred_mask", 0.0) > 0.0:
                        ep_spawn_events += int(info.get("spawned_this_step", 1))
                    if config.get("pred_coef", 0.0) > 0.0 and info.get("pred_mask", 0.0) > 0.0:
                        horizon = max(int(config.get("prediction_horizon_steps", 0)), 1)
                        pred_target = np.asarray(info["pred_target"], dtype=np.float32).copy()
                        for mem_idx in episode_memory_indices[-horizon:]:
                            memory["pred_target"][mem_idx] = pred_target
                            memory["pred_mask"][mem_idx] = 1.0
                            pred_supervised_indices.add(mem_idx)

                    ep_action_delta.append(float(np.linalg.norm(a - previous_env_action)))
                    ep_action_l2.append(float(np.dot(a, a)))
                    previous_action = raw_a.copy()
                    previous_env_action = a.copy()

                    s = s_n
                    hx, cx = hx_n, cx_n
                    ep_r += r_ext
                    total_steps += 1

                    if info.get("covered", False):
                        ep_covered += 1
                        if config.get("recurrent_reset_on_cover", True):
                            hx, cx = None, None
                            ep_cover_resets += 1
                        else:
                            ep_cover_keeps += 1
                    elif info.get("missed_count", 0) > previous_missed:
                        if config.get("recurrent_reset_on_miss", True):
                            hx, cx = None, None
                            ep_miss_resets += 1
                    previous_missed = info.get("missed_count", 0)

                    if total_steps % update_timestep == 0:
                        _, _, last_v, _, _, _ = agent.select_action(s, hx, cx, deterministic=True)
                        memory["v"].append(last_v)
                        last_update_stats = agent.update(memory)
                        memory = new_memory()
                        episode_memory_indices = []
                        logger.log_event(
                            "ppo_update",
                            {
                                "total_steps": total_steps,
                                "residual_beta": residual_beta_at_step(config, total_steps),
                                **last_update_stats,
                            },
                        )

                    if terminated or truncated:
                        break

                if carry_session_hidden and not last_session_chunk and not terminated:
                    session_hx, session_cx = hx, cx
                else:
                    session_hx, session_cx = None, None

                global_ep += 1
                smooth_reward = 0.95 * smooth_reward + 0.05 * ep_r
                ep_spawned = info.get("spawned_count", 0)
                ep_coverage = info.get("coverage_rate", ep_covered / max(ep_spawned, 1))

                stage_rewards.append(ep_r)
                stage_coverages.append(ep_coverage)
                history["reward"].append(smooth_reward)
                history["coverage"].append(ep_coverage)
                history["target_distance"].append(float(info.get("target_distance", 0.0)))

                row = {
                    "method": config["method"],
                    "seed": int(config["seed"]),
                    "global_episode": global_ep,
                    "stage": stage["name"],
                    "stage_episode": ep_in_stage + 1,
                    "total_steps": total_steps,
                    "episode_reward": float(ep_r),
                    "smooth_reward": float(smooth_reward),
                    "coverage_rate": float(ep_coverage),
                    "chunk_success_count": int(ep_covered),
                    "success_count": int(info.get("success_count", 0)),
                    "spawned_count": int(ep_spawned),
                    "missed_count": int(info.get("missed_count", 0)),
                    "sim_step_seconds": float(info.get("sim_step_seconds", 0.0)),
                    "decision_dt_seconds": float(info.get("decision_dt_seconds", 0.0)),
                    "cover_latency": float(info.get("cover_latency", 0.0)),
                    "cover_latency_seconds": float(info.get("cover_latency_seconds", 0.0)),
                    "last_cover_latency": float(info.get("last_cover_latency", 0.0)),
                    "last_cover_latency_seconds": float(info.get("last_cover_latency_seconds", 0.0)),
                    "cover_latency_p50": float(info.get("cover_latency_p50", 0.0)),
                    "cover_latency_p50_seconds": float(info.get("cover_latency_p50_seconds", 0.0)),
                    "cover_latency_p90": float(info.get("cover_latency_p90", 0.0)),
                    "cover_latency_p90_seconds": float(info.get("cover_latency_p90_seconds", 0.0)),
                    "cover_latency_p95": float(info.get("cover_latency_p95", 0.0)),
                    "cover_latency_p95_seconds": float(info.get("cover_latency_p95_seconds", 0.0)),
                    "cover_latency_max": float(info.get("cover_latency_max", 0.0)),
                    "cover_latency_max_seconds": float(info.get("cover_latency_max_seconds", 0.0)),
                    "response_sla_steps": int(info.get("response_sla_steps", 0)),
                    "response_sla_seconds": float(info.get("response_sla_seconds", 0.0)),
                    "response_sla_success_count": int(info.get("response_sla_success_count", 0)),
                    "response_sla_miss_count": int(info.get("response_sla_miss_count", 0)),
                    "response_sla_success_rate": float(info.get("response_sla_success_rate", 0.0)),
                    "active_steam_mean": float(info.get("active_steam_mean", 0.0)),
                    "active_steam_max": int(info.get("active_steam_max", 0)),
                    "oldest_active_age": float(info.get("oldest_active_age", 0.0)),
                    "oldest_active_age_seconds": float(info.get("oldest_active_age_seconds", 0.0)),
                    "oldest_active_age_max": float(info.get("oldest_active_age_max", 0.0)),
                    "oldest_active_age_max_seconds": float(info.get("oldest_active_age_max_seconds", 0.0)),
                    "target_distance": float(info.get("target_distance", 0.0)),
                    "target_selector": str(info.get("target_selector", "")),
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
                    "use_lstm": bool(agent.use_lstm),
                    "use_steam_attention": bool(agent.use_steam_attention),
                    "attention_steam_count": int(getattr(agent, "attention_steam_count", 0)),
                    "use_spawn_history_observation": bool(config.get("use_spawn_history_observation", False)),
                    "use_thermal_context_observation": bool(config.get("use_thermal_context_observation", False)),
                    "use_route_summary_observation": bool(config.get("use_route_summary_observation", False)),
                    "use_material_map": bool(agent.use_material_map),
                    "device": str(agent.device),
                    "residual_policy": bool(config.get("residual_policy", False)),
                    "residual_base_policy": str(config.get("residual_base_policy", "")),
                    "residual_glue": str(config.get("residual_glue", "fixed")),
                    "residual_sparse_base_policy": str(config.get("residual_sparse_base_policy", "")),
                    "residual_dense_base_policy": str(config.get("residual_dense_base_policy", "")),
                    "residual_beta": float(residual_beta_at_step(config, total_steps)),
                    "residual_effective_beta_mean": float(np.mean(ep_residual_betas)) if ep_residual_betas else 0.0,
                    "residual_fixed_steps": int(ep_residual_modes.get("fixed", 0)),
                    "residual_lull_steps": int(ep_residual_modes.get("lull", 0)),
                    "residual_sparse_steps": int(ep_residual_modes.get("sparse", 0)),
                    "residual_charging_steps": int(ep_residual_modes.get("charging", 0)),
                    "residual_dense_steps": int(ep_residual_modes.get("dense", 0)),
                    "residual_burst_steps": int(ep_residual_modes.get("burst", 0)),
                    "residual_mid_steps": int(ep_residual_modes.get("mid", 0)),
                    "residual_horizon2_base_steps": int(ep_residual_base_counts.get("horizon2", 0)),
                    "residual_dynamic_base_steps": int(ep_residual_base_counts.get("dynamic_weighted", 0)),
                    "residual_guard": bool(config.get("residual_guard", True)),
                    "residual_action_shield": bool(config.get("residual_action_shield", False)),
                    "stagnation_recovery_steps": int(config.get("stagnation_recovery_steps", 180)),
                    "recurrent_reset_on_cover": bool(config.get("recurrent_reset_on_cover", True)),
                    "recurrent_reset_on_miss": bool(config.get("recurrent_reset_on_miss", True)),
                    "continuous_session_chunks": int(session_chunks),
                    "continuous_session_chunk_index": int(chunk_index + 1),
                    "continuous_session_new": bool(new_session),
                    "carry_lstm_state_across_chunks": bool(carry_session_hidden),
                    "recurrent_sequence_steps": int(config.get("recurrent_sequence_steps", config["episode_steps"])),
                    "recurrent_hidden_resets": int(ep_hidden_resets),
                    "recurrent_cover_resets": int(ep_cover_resets),
                    "recurrent_cover_keeps": int(ep_cover_keeps),
                    "recurrent_miss_resets": int(ep_miss_resets),
                    "prediction_supervised_steps": int(len(pred_supervised_indices)),
                    "prediction_spawn_events": int(ep_spawn_events),
                    "steps_since_cover": int(info.get("steps_since_cover", 0)),
                    "route_summary_observation_enabled": bool(info.get("route_summary_observation_enabled", False)),
                    "route_active_density": float(info.get("route_active_density", 0.0)),
                    "route_max_age_score": float(info.get("route_max_age_score", 0.0)),
                    "route_mean_age_score": float(info.get("route_mean_age_score", 0.0)),
                    "route_nearest_distance_score": float(info.get("route_nearest_distance_score", 0.0)),
                    "route_target_thermal_score": float(info.get("route_target_thermal_score", 0.0)),
                    "route_confidence": float(info.get("route_confidence", 0.0)),
                    "route_spawn_ready": float(info.get("route_spawn_ready", 0.0)),
                    "route_stagnation_score": float(info.get("route_stagnation_score", 0.0)),
                    "material_map_observation_enabled": bool(info.get("material_map_observation_enabled", False)),
                    "potential_shaping_enabled": bool(info.get("potential_shaping_enabled", True)),
                    "best_progress_enabled": bool(info.get("best_progress_enabled", True)),
                    "latency_first_reward_enabled": bool(info.get("latency_first_reward_enabled", False)),
                    "cover_latency_penalty_gain": float(info.get("cover_latency_penalty_gain", 0.0)),
                    "oldest_active_penalty_gain": float(info.get("oldest_active_penalty_gain", 0.0)),
                    "backlog_penalty_gain": float(info.get("backlog_penalty_gain", 0.0)),
                    "material_observation_enabled": bool(info.get("material_observation_enabled", True)),
                    "material_tv_reward_enabled": bool(info.get("material_tv_reward_enabled", False)),
                    "material_tv_reward_gain": float(info.get("material_tv_reward_gain", 0.0)),
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
                    "height_uniformity": float(info.get("height_uniformity", 0.0)),
                    "overfill_penalty": float(info.get("overfill_penalty", 0.0)),
                    "material_hole_loss": float(info.get("material_hole_loss", 0.0)),
                    "material_tv_loss": float(info.get("material_tv_loss", 0.0)),
                    "material_quality_loss": float(info.get("material_quality_loss", 0.0)),
                    "action_delta_mean": float(np.mean(ep_action_delta)) if ep_action_delta else 0.0,
                    "action_l2_mean": float(np.mean(ep_action_l2)) if ep_action_l2 else 0.0,
                    "actor_loss": float(last_update_stats.get("actor_loss", 0.0)),
                    "critic_loss": float(last_update_stats.get("critic_loss", 0.0)),
                    "entropy": float(last_update_stats.get("entropy", 0.0)),
                    "elapsed_seconds": float(time.time() - start_time),
                }
                logger.log_episode(row)

                if global_ep % 10 == 0 or stage["episodes"] <= 5:
                    print(
                        f"Ep:{global_ep}/{total_curriculum_episodes} | "
                        f"{stage['name']}({ep_in_stage+1}/{stage['episodes']}) | "
                        f"R:{ep_r:.1f} | Smooth:{smooth_reward:.1f} | "
                        f"Cov:{ep_coverage:.2f} ({info.get('success_count', ep_covered)}/{ep_spawned}) | "
                        f"Dist:{info.get('target_distance', 0):.3f} | "
                        f"Lat:{info.get('cover_latency_seconds', 0):.2f}s "
                        f"P90:{info.get('cover_latency_p90_seconds', 0):.2f}s "
                        f"SLA:{info.get('response_sla_success_rate', 0):.2f}",
                        flush=True,
                    )

                update_plot(plt, ax, history, stage_coverages)

                if global_ep % config["save_interval"] == 0:
                    save_checkpoint(agent, global_ep, config, run_path)
                    save_checkpoint(agent, "latest", config, run_path)

            if stage_rewards:
                print(
                    f"\n--- {stage['name']} done --- "
                    f"Avg R(last 50): {np.mean(stage_rewards[-50:]):.1f} "
                    f"Avg Cov(last 50): {np.mean(stage_coverages[-50:]):.2f}",
                    flush=True,
                )

            save_checkpoint(agent, f"{stage['name']}_end", config, run_path)
            save_checkpoint(agent, "latest", config, run_path)

        save_checkpoint(agent, total_curriculum_episodes, config, run_path)
        save_checkpoint(agent, "latest", config, run_path)
        logger.log_event("done", {"episodes": global_ep, "steps": total_steps, "elapsed_seconds": time.time() - start_time})
        print(f"\nDone! Episodes: {global_ep}, Steps: {total_steps}, Run: {run_path}", flush=True)
    finally:
        logger.close()


def main():
    args = build_arg_parser().parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{config['method']}_seed{config['seed']}_{timestamp}"
    run_path = Path(args.run_dir) / run_name
    train(config, run_path)


if __name__ == "__main__":
    main()
