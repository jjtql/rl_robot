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
    "use_material_map": False,
    "use_curriculum": True,
    "flat_stage": "multi_realistic",
    "potential_shaping": True,
    "best_progress_reward": True,
    "material_observation": True,
    "material_tv_reward": False,
    "material_tv_reward_gain": 6.0,
    "action_smoothing_penalty": True,
    "action_delay_steps": 0,
    "action_noise_std": 0.0,
    "domain_randomization": False,
    "domain_randomization_scale": 0.12,
    "residual_policy": False,
    "residual_base_policy": "risk_aware",
    "residual_beta": 0.25,
    "residual_guard": True,
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
