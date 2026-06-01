import copy
import json
import random
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CONFIG = {
    "method": "scheme_d_paper_base",
    "model_path": str(REPO_ROOT / "o1.xml"),
    "episode_steps": 800,
    "update_episodes": 2,
    "save_interval": 50,
    "checkpoint_prefix": "scheme_d_paper_base",
    "target_selector": "risk_aware",
    "use_lstm": True,
    "use_steam_attention": False,
    "attention_steam_count": 6,
    "attention_steam_dim": 8,
    "use_spawn_history_observation": False,
    "use_thermal_context_observation": False,
    "use_route_summary_observation": False,
    "use_material_map": False,
    "use_curriculum": True,
    "flat_stage": "multi_realistic",
    "potential_shaping": True,
    "best_progress_reward": True,
    "cover_reward_scale": 1.0,
    "quick_cover_bonus_scale": 1.0,
    "precision_bonus_scale": 1.0,
    "potential_gain_scale": 1.0,
    "best_progress_gain_scale": 1.0,
    "active_steam_penalty_scale": 1.0,
    "age_penalty_scale": 1.0,
    "material_observation": True,
    "material_tv_reward": False,
    "material_tv_reward_gain": 6.0,
    "action_smoothing_penalty": True,
    "latency_first_reward": False,
    "decision_dt_seconds": None,
    "response_sla_seconds": None,
    "cover_latency_penalty_gain": 14.0,
    "oldest_active_penalty_gain": 0.08,
    "backlog_penalty_gain": 0.04,
    "response_sla_steps": 120,
    "response_sla_bonus": 6.0,
    "response_sla_miss_penalty": 8.0,
    "continuous_session_chunks": 1,
    "carry_lstm_state_across_chunks": False,
    "lstm_sequence_chunks": 1,
    "action_delay_steps": 0,
    "action_noise_std": 0.0,
    "domain_randomization": False,
    "domain_randomization_scale": 0.12,
    "thermal_spawn": True,
    "thermal_hotspot_count": 3,
    "thermal_hotspot_sigma": 0.22,
    "thermal_hotspot_strength": 1.8,
    "thermal_background_weight": 0.28,
    "thermal_drift_std": 0.006,
    "thermal_refresh_probability": 0.0025,
    "thermal_lifetime_steps": 520,
    "thermal_recent_spawn_radius": 0.18,
    "thermal_recent_spawn_suppression": 0.55,
    "thermal_recent_spawn_memory": 12,
    "burst_lull_spawn": False,
    "burst_lull_lull_steps": 80,
    "burst_lull_charge_steps": 120,
    "burst_lull_sparse_threshold": 2,
    "burst_lull_burst_min": 3,
    "burst_lull_burst_max": 5,
    "burst_lull_burst_interval_steps": 4,
    "burst_lull_trickle_probability": 0.004,
    "burst_lull_initial_burst": True,
    "residual_policy": False,
    "residual_base_policy": "risk_aware",
    "residual_beta": 0.25,
    "residual_beta_start": None,
    "residual_beta_end": None,
    "residual_beta_warmup_steps": 0,
    "residual_guard": True,
    "residual_action_shield": False,
    "residual_glue": "fixed",
    "residual_sparse_base_policy": "horizon2",
    "residual_dense_base_policy": "dynamic_weighted",
    "residual_phase_sparse_threshold": None,
    "residual_phase_dense_threshold": None,
    "residual_sparse_beta_scale": 1.25,
    "residual_lull_beta_scale": 1.35,
    "residual_charging_beta_scale": 1.0,
    "residual_dense_beta_scale": 0.35,
    "residual_burst_beta_scale": 0.25,
    "residual_emergency_beta_scale": 1.0,
    "residual_emergency_age_ratio": 1.0,
    "stagnation_recovery_steps": 180,
    "recurrent_reset_on_cover": True,
    "recurrent_reset_on_miss": True,
    "device": "auto",
    "ppo_lr": 1e-4,
    "ppo_epochs": 4,
    "ppo_clip": 0.12,
    "ppo_value_clip": 0.12,
    "ppo_reward_scale": 0.02,
    "ppo_reward_clip": 4.0,
    "ppo_entropy_start": 0.003,
    "ppo_entropy_end": 0.0005,
    "ppo_entropy_decay_steps": 240000,
    "ppo_max_grad_norm": 0.35,
    "pred_coef": 0.0,
    "prediction_horizon_steps": 80,
    "bc_supervised_coef": 0.03,
    "bc_supervised_min_coef": 0.0,
    "bc_supervised_decay_steps": 240000,
    "continuous_training": True,
    "continuous_success_count": 9999,
    "bc_warm_start": True,
    "bc_episodes": 60,
    "bc_epochs": 10,
    "bc_policy": "risk_aware",
    "bc_stages": [
        {"name": "single_easy", "episodes": 15},
        {"name": "multi_low", "episodes": 25},
        {"name": "multi_realistic", "episodes": 20},
    ],
    "seed": 0,
    "headless": True,
    "plot": False,
    "curriculum": [
        {"name": "single_easy", "episodes": 500},
        {"name": "single_precision", "episodes": 350},
        {"name": "multi_low", "episodes": 400},
        {"name": "multi_realistic", "episodes": 250},
    ],
}


def config_copy():
    return copy.deepcopy(DEFAULT_CONFIG)


def deep_update(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None):
    config = config_copy()
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            config = deep_update(config, json.load(handle))
    return config


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
