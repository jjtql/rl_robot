import numpy as np
import torch
from itertools import permutations

from .algo import PPOAgent


def load_checkpoint(agent, model_path):
    checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
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


class RandomPolicy:
    name = "random"

    def reset(self):
        pass

    def act(self, env, obs):
        return env.action_space.sample().astype(np.float32)


class NearestSteamPolicy:
    name = "nearest_rule"

    def reset(self):
        pass

    def act(self, env, obs):
        return action_toward_steam(env, select_nearest_steam(env))


class OldestSteamPolicy:
    name = "oldest_rule"

    def reset(self):
        pass

    def act(self, env, obs):
        if not env.steams:
            return np.zeros(env.action_space.shape[0], dtype=np.float32)
        return action_toward_steam(env, max(env.steams, key=lambda item: item["age"]))


class DistanceAgePolicy:
    name = "distance_age_rule"

    def reset(self):
        pass

    def act(self, env, obs):
        steam = select_distance_age_steam(env)
        return action_toward_steam(env, steam)


class RiskAwarePolicy:
    name = "risk_aware_rule"

    def reset(self):
        pass

    def act(self, env, obs):
        steam = select_risk_aware_steam(env)
        return action_toward_steam(env, steam)


class DynamicWeightedGreedyPolicy:
    name = "dynamic_weighted_greedy"

    def reset(self):
        pass

    def act(self, env, obs):
        steam = select_dynamic_weighted_steam(env)
        return action_toward_steam(env, steam)


class RecedingHorizonPolicy:
    def __init__(self, horizon=2):
        self.horizon = int(horizon)
        self.name = f"horizon{self.horizon}_planner"

    def reset(self):
        pass

    def act(self, env, obs):
        steam = select_receding_horizon_steam(env, horizon=self.horizon)
        return action_toward_steam(env, steam)


class DeadlineHorizonPolicy:
    def __init__(self, horizon=2, rescue=False):
        self.horizon = int(horizon)
        self.rescue = bool(rescue)
        prefix = "deadline_rescue" if self.rescue else "deadline"
        self.name = f"{prefix}_horizon{self.horizon}_planner"

    def reset(self):
        pass

    def act(self, env, obs):
        steam = select_deadline_horizon_steam(env, horizon=self.horizon, rescue=self.rescue)
        return action_toward_steam(env, steam)


class SlackHorizonPolicy:
    def __init__(self, horizon=2):
        self.horizon = int(horizon)
        self.name = f"slack_horizon{self.horizon}_planner"

    def reset(self):
        pass

    def act(self, env, obs):
        steam = select_slack_horizon_steam(env, horizon=self.horizon)
        return action_toward_steam(env, steam)


class CorridorWaypointPolicy:
    name = "corridor_waypoint_planner"

    def reset(self):
        pass

    def act(self, env, obs):
        steam = select_corridor_waypoint_steam(env)
        return action_toward_steam(env, steam)


class SlaRouteEnsemblePolicy:
    name = "sla_route_ensemble_planner"

    def reset(self):
        pass

    def act(self, env, obs):
        steam = select_sla_route_ensemble_steam(env)
        return action_toward_steam(env, steam)


class StickySlaRouteEnsemblePolicy:
    name = "sticky_sla_route_ensemble_planner"

    def __init__(self, switch_margin=0.18, stagnation_patience=70):
        self.switch_margin = float(switch_margin)
        self.stagnation_patience = int(stagnation_patience)
        self.target_id = None
        self.last_distance = None
        self.stagnation_steps = 0

    def reset(self):
        self.target_id = None
        self.last_distance = None
        self.stagnation_steps = 0

    def _active_target(self, env):
        if self.target_id is None:
            return None
        return next((steam for steam in env.steams if steam.get("id") == self.target_id), None)

    def _arrival_ratio(self, env, steam):
        if steam is None:
            return 0.0
        sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)
        travel_steps = _travel_steps_between(env, env.cover_center, steam["pos"][:2])
        return float((steam["age"] + travel_steps) / max(sla_steps, 1))

    def _update_stagnation(self, env, steam):
        if steam is None:
            self.last_distance = None
            self.stagnation_steps = 0
            return
        dist = float(np.linalg.norm(steam["pos"][:2] - env.cover_center))
        if self.last_distance is not None and dist >= self.last_distance - 0.003:
            self.stagnation_steps += 1
        else:
            self.stagnation_steps = 0
        self.last_distance = dist

    def act(self, env, obs):
        return self.act_with_gate(env, obs, None)

    def act_with_gate(self, env, obs, gate_action=None):
        if not env.steams:
            gate = np.zeros(3, dtype=np.float32) if gate_action is None else np.asarray(gate_action, dtype=np.float32)
            hotspot = getattr(env, "_dominant_thermal_hotspot", lambda: None)()
            phase = str(getattr(env, "burst_lull_phase", "") or "")
            if hotspot is not None and gate.shape[0] >= 3 and gate[2] > 0.10 and phase in ("lull", "charging", "burst"):
                return action_toward_xy(env, hotspot["xy"])
            self.reset()
            return np.zeros(env.action_space.shape[0], dtype=np.float32)

        gate = np.zeros(3, dtype=np.float32) if gate_action is None else np.asarray(gate_action, dtype=np.float32)
        lock_bias = float(np.clip(gate[0], -1.0, 1.0)) if gate.shape[0] > 0 else 0.0
        rescue_bias = float(np.clip(gate[1], -1.0, 1.0)) if gate.shape[0] > 1 else 0.0
        hotspot_bias = float(np.clip(gate[2], -1.0, 1.0)) if gate.shape[0] > 2 else 0.0
        switch_margin = float(np.clip(self.switch_margin + 0.12 * lock_bias, 0.04, 0.36))
        urgency_floor = float(np.clip(0.76 - 0.10 * rescue_bias, 0.60, 0.88))

        current = self._active_target(env)
        self._update_stagnation(env, current)
        candidate = select_sla_route_ensemble_steam(env)

        if current is not None and candidate is not None and self.stagnation_steps < self.stagnation_patience:
            current_ratio = self._arrival_ratio(env, current)
            candidate_ratio = self._arrival_ratio(env, candidate)
            same_target = candidate.get("id") == current.get("id")
            urgent_switch = candidate_ratio >= max(urgency_floor, current_ratio + switch_margin)
            if same_target or not urgent_switch:
                action = action_toward_steam(env, current)
                return maybe_blend_toward_hotspot(env, action, current, hotspot_bias)

        target = candidate if candidate is not None else current
        self.target_id = target.get("id") if target is not None else None
        self.last_distance = None
        self.stagnation_steps = 0
        action = action_toward_steam(env, target)
        return maybe_blend_toward_hotspot(env, action, target, hotspot_bias)


class AcoTspPolicy:
    name = "aco_tsp_planner"

    def __init__(self, ants=12, iterations=3, alpha=1.0, beta=2.0):
        self.ants = int(ants)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.route = []
        self.route_ids = set()

    def reset(self):
        self.route = []
        self.route_ids = set()

    def act(self, env, obs):
        active_ids = {steam.get("id") for steam in env.steams}
        if not self.route or not self.route_ids.issubset(active_ids):
            self.route = plan_aco_tsp_route(
                env,
                ants=self.ants,
                iterations=self.iterations,
                alpha=self.alpha,
                beta=self.beta,
            )
            self.route_ids = {steam.get("id") for steam in self.route}
        while self.route and self.route[0].get("id") not in active_ids:
            self.route.pop(0)
        steam = self.route[0] if self.route else select_receding_horizon_steam(env, horizon=2)
        return action_toward_steam(env, steam)


class PlannerEnsemblePolicy:
    name = "planner_ensemble"

    def __init__(self, recovery_patience=120):
        self.recovery_patience = int(recovery_patience)
        self.last_target_id = None
        self.last_target_distance = None
        self.stagnation_steps = 0

    def reset(self):
        self.last_target_id = None
        self.last_target_distance = None
        self.stagnation_steps = 0

    def _update_stagnation(self, env):
        if self.last_target_id is None:
            self.stagnation_steps = 0
            self.last_target_distance = None
            return
        target = next((steam for steam in env.steams if steam.get("id") == self.last_target_id), None)
        if target is None:
            self.stagnation_steps = 0
            self.last_target_distance = None
            return
        dist = float(np.linalg.norm(target["pos"][:2] - env.cover_center))
        if self.last_target_distance is not None and dist >= self.last_target_distance - 0.004:
            self.stagnation_steps += 1
        else:
            self.stagnation_steps = 0
        self.last_target_distance = dist

    def _candidate_targets(self, env):
        candidates = [
            ("horizon2", select_receding_horizon_steam(env, horizon=2)),
            ("dynamic", select_dynamic_weighted_steam(env)),
            ("risk", select_risk_aware_steam(env)),
            ("nearest", select_nearest_steam(env)),
        ]
        if len(env.steams) >= 3:
            candidates.append(("horizon3", select_receding_horizon_steam(env, horizon=3)))

        unique = []
        seen = set()
        for source, steam in candidates:
            if steam is None:
                continue
            steam_id = steam.get("id")
            key = steam_id if steam_id is not None else id(steam)
            if key in seen:
                continue
            seen.add(key)
            unique.append((source, steam))
        return unique

    def _score_candidate(self, env, steam, source):
        center = np.asarray(env.cover_center, dtype=np.float32)
        xy = steam["pos"][:2]
        dist = float(np.linalg.norm(xy - center))
        travel_steps = max(1.0, np.ceil(max(dist - env.cover_radius, 0.0) / max(env.track_step_size, 1e-6)))
        age_score = float(np.clip((steam["age"] + travel_steps) / max(env.max_steam_age, 1), 0.0, 1.35))
        risk_score = score_steam(env, steam, center)
        thermal_score = thermal_score_at_xy(env, xy)
        dist_norm = float(np.clip(dist / max(env.pot_radius, 1e-6), 0.0, 1.5))
        route_bonus = 0.05 if source.startswith("horizon") else 0.0
        if source == "dynamic":
            route_bonus += 0.025
        if source == "nearest" and self.stagnation_steps >= self.recovery_patience:
            route_bonus += 0.22

        switch_penalty = 0.0
        steam_id = steam.get("id")
        if self.last_target_id is not None and steam_id != self.last_target_id:
            switch_penalty = 0.10
            if self.stagnation_steps >= self.recovery_patience:
                switch_penalty *= 0.25
        same_target_bonus = 0.04 if steam_id == self.last_target_id else 0.0

        return float(
            1.05 * risk_score
            + 0.32 * age_score
            + 0.16 * thermal_score
            + route_bonus
            + same_target_bonus
            - 0.10 * dist_norm
            - 0.018 * travel_steps
            - switch_penalty
        )

    def act(self, env, obs):
        if not env.steams:
            self.last_target_id = None
            self.last_target_distance = None
            self.stagnation_steps = 0
            return np.zeros(env.action_space.shape[0], dtype=np.float32)

        self._update_stagnation(env)
        best_steam = None
        best_score = -np.inf
        for source, steam in self._candidate_targets(env):
            score = self._score_candidate(env, steam, source)
            if score > best_score:
                best_score = score
                best_steam = steam

        if best_steam is None:
            best_steam = select_receding_horizon_steam(env, horizon=2)
        self.last_target_id = best_steam.get("id") if best_steam is not None else None
        self.last_target_distance = (
            float(np.linalg.norm(best_steam["pos"][:2] - env.cover_center)) if best_steam is not None else None
        )
        return action_toward_steam(env, best_steam)


def residual_glue_mode(env, config):
    if str(config.get("residual_glue", "fixed")) != "phase_aware":
        return "fixed"

    active_count = len(getattr(env, "steams", []) or [])
    max_steams = max(int(getattr(env, "max_steams", max(active_count, 1))), 1)
    sparse_threshold = config.get("residual_phase_sparse_threshold")
    if sparse_threshold is None:
        sparse_threshold = getattr(env, "burst_lull_sparse_threshold", config.get("burst_lull_sparse_threshold", 2))
    sparse_threshold = max(int(sparse_threshold), 0)

    dense_threshold = config.get("residual_phase_dense_threshold")
    if dense_threshold is None:
        dense_threshold = max(sparse_threshold + 2, int(np.ceil(0.55 * max_steams)))
    dense_threshold = max(int(dense_threshold), sparse_threshold + 1)

    phase = str(getattr(env, "burst_lull_phase", "") or "")
    pending_count = int(getattr(env, "burst_lull_pending_count", 0) or 0)

    if phase == "burst" or pending_count > 0:
        return "burst"
    if phase == "dense" or active_count >= dense_threshold:
        return "dense"
    if phase == "lull":
        return "lull"
    if active_count <= sparse_threshold:
        return "sparse"
    if phase == "charging":
        return "charging"
    return "mid"


def residual_beta_for_env(config, env, scheduled_beta):
    if str(config.get("residual_glue", "fixed")) != "phase_aware":
        return float(scheduled_beta)
    mode = residual_glue_mode(env, config)
    scale_by_mode = {
        "lull": float(config.get("residual_lull_beta_scale", 1.35)),
        "sparse": float(config.get("residual_sparse_beta_scale", 1.25)),
        "charging": float(config.get("residual_charging_beta_scale", 1.0)),
        "dense": float(config.get("residual_dense_beta_scale", 0.35)),
        "burst": float(config.get("residual_burst_beta_scale", 0.25)),
        "mid": 1.0,
        "fixed": 1.0,
    }
    beta = float(scheduled_beta) * scale_by_mode.get(mode, 1.0)
    emergency_scale = float(config.get("residual_emergency_beta_scale", 1.0))
    emergency_threshold = float(config.get("residual_emergency_age_ratio", 1.0))
    if emergency_scale < 1.0 and max_deadline_arrival_ratio(env) >= emergency_threshold:
        beta *= max(emergency_scale, 0.0)
    return float(np.clip(beta, 0.0, 1.0))


class PhaseAwareResidualBasePolicy:
    def __init__(self, config):
        self.config = dict(config)
        self.sparse_policy_name = self.config.get("residual_sparse_base_policy") or self.config.get(
            "residual_base_policy", "horizon2"
        )
        self.dense_policy_name = self.config.get("residual_dense_base_policy") or self.config.get(
            "residual_base_policy", "horizon2"
        )
        self.mid_policy_name = self.config.get("residual_base_policy", "horizon2")
        self.controllers = {}
        for name in {self.sparse_policy_name, self.dense_policy_name, self.mid_policy_name}:
            self.controllers[name] = build_base_policy(name)
        self.last_mode = "fixed"
        self.last_policy_name = self.mid_policy_name
        self.name = f"phase_aware_{self.sparse_policy_name}_{self.dense_policy_name}"

    def reset(self):
        self.last_mode = "fixed"
        self.last_policy_name = self.mid_policy_name
        for controller in self.controllers.values():
            controller.reset()

    def _policy_name_for_mode(self, mode):
        if mode in ("lull", "sparse", "charging"):
            return self.sparse_policy_name
        if mode in ("burst", "dense"):
            return self.dense_policy_name
        return self.mid_policy_name

    def act(self, env, obs):
        mode = residual_glue_mode(env, self.config)
        policy_name = self._policy_name_for_mode(mode)
        self.last_mode = mode
        self.last_policy_name = policy_name
        return self.controllers[policy_name].act(env, obs)

    def act_with_gate(self, env, obs, gate_action=None):
        mode = residual_glue_mode(env, self.config)
        policy_name = self._policy_name_for_mode(mode)
        self.last_mode = mode
        self.last_policy_name = policy_name
        controller = self.controllers[policy_name]
        if hasattr(controller, "act_with_gate"):
            return controller.act_with_gate(env, obs, gate_action)
        return controller.act(env, obs)


class PPOPolicy:
    name = "ppo"

    def __init__(self, env, model_path, deterministic=True, config_override=None):
        config = checkpoint_config(model_path)
        if config_override:
            config = {**config, **{key: value for key, value in config_override.items() if value is not None}}
        self.config = config
        self.agent = PPOAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            seq_len=env.max_episode_steps,
            use_lstm=config.get("use_lstm", True),
            use_steam_attention=config.get("use_steam_attention", False),
            use_material_map=config.get("use_material_map", False),
            base_obs_dim=getattr(env, "base_obs_dim", config.get("base_obs_dim", 35)),
            attention_steam_count=config.get("attention_steam_count", getattr(env, "attention_steam_count", 6)),
            attention_steam_dim=config.get("attention_steam_dim", getattr(env, "attention_steam_dim", 8)),
            device=config.get("device", "auto"),
        )
        load_checkpoint(self.agent, model_path)
        self.agent.model.eval()
        self.deterministic = deterministic
        self.hx = None
        self.cx = None
        self.residual_policy = bool(config.get("residual_policy", False))
        scheduled_beta = config.get("residual_beta_end")
        warmup_steps = int(config.get("residual_beta_warmup_steps", 0) or 0)
        if scheduled_beta is not None and warmup_steps > 0:
            self.residual_beta = float(scheduled_beta)
        else:
            self.residual_beta = float(config.get("residual_beta", 0.25))
        self.residual_guard = bool(config.get("residual_guard", True))
        self.residual_combine_mode = str(config.get("residual_combine_mode", "add"))
        self.residual_min_alignment = float(config.get("residual_min_alignment", 0.15))
        self.residual_action_shield = bool(config.get("residual_action_shield", False))
        self.stagnation_recovery_steps = int(config.get("stagnation_recovery_steps", 180))
        self.recurrent_reset_on_cover = bool(config.get("recurrent_reset_on_cover", True))
        self.recurrent_reset_on_miss = bool(config.get("recurrent_reset_on_miss", True))
        self.residual_base_policy_name = config.get("residual_base_policy", "risk_aware")
        self.residual_base_policy = build_residual_base_policy(config) if self.residual_policy else None
        self.previous_env_action = None
        if self.residual_policy:
            glue = str(config.get("residual_glue", "fixed"))
            if glue == "phase_aware":
                self.name = "phase_aware_residual_ppo"
            else:
                self.name = f"residual_{self.residual_base_policy_name}_ppo"

    def reset(self):
        self.hx = None
        self.cx = None
        self.previous_env_action = None
        if self.residual_base_policy is not None:
            self.residual_base_policy.reset()

    def reset_recurrent(self):
        self.hx = None
        self.cx = None

    def act(self, env, obs):
        with torch.no_grad():
            action, _, _, _, self.hx, self.cx = self.agent.select_action(
                obs,
                self.hx,
                self.cx,
                deterministic=self.deterministic,
            )
        if self.residual_policy:
            if self.residual_combine_mode == "gate":
                action = planner_gate_action(env, self.residual_base_policy, obs, action)
            else:
                base_action = self.residual_base_policy.act(env, obs)
                action = combine_residual_action(
                    base_action,
                    action,
                    beta=residual_beta_for_env(self.config, env, self.residual_beta),
                    guard=self.residual_guard,
                    min_alignment=self.residual_min_alignment,
                    mode=self.residual_combine_mode,
                )
            if self.residual_action_shield and self.residual_combine_mode != "gate":
                action = shield_residual_action(
                    env,
                    base_action,
                    action,
                    previous_action=self.previous_env_action,
                    stagnation_steps=getattr(env, "steps_since_cover", 0),
                    recovery_steps=self.stagnation_recovery_steps,
                    min_progress_ratio=float(self.config.get("residual_pathbend_min_progress_ratio", 0.35)),
                    guarded_progress_ratio=float(self.config.get("residual_pathbend_guarded_progress_ratio", 0.50)),
                    allow_backtrack_steps=float(self.config.get("residual_pathbend_allow_backtrack_steps", 0.0)),
                )
            self.previous_env_action = action.copy()
        return action.astype(np.float32)


def build_base_policy(kind):
    if kind == "random":
        return RandomPolicy()
    if kind == "nearest":
        return NearestSteamPolicy()
    if kind == "oldest":
        return OldestSteamPolicy()
    if kind == "distance_age":
        return DistanceAgePolicy()
    if kind == "risk_aware":
        return RiskAwarePolicy()
    if kind == "dynamic_weighted":
        return DynamicWeightedGreedyPolicy()
    if kind == "horizon2":
        return RecedingHorizonPolicy(horizon=2)
    if kind == "horizon3":
        return RecedingHorizonPolicy(horizon=3)
    if kind == "deadline_horizon2":
        return DeadlineHorizonPolicy(horizon=2)
    if kind == "deadline_rescue_horizon2":
        return DeadlineHorizonPolicy(horizon=2, rescue=True)
    if kind == "slack_horizon2":
        return SlackHorizonPolicy(horizon=2)
    if kind == "corridor_waypoint":
        return CorridorWaypointPolicy()
    if kind == "sla_route_ensemble":
        return SlaRouteEnsemblePolicy()
    if kind == "sticky_sla_ensemble":
        return StickySlaRouteEnsemblePolicy()
    if kind == "aco_tsp":
        return AcoTspPolicy()
    if kind == "planner_ensemble":
        return PlannerEnsemblePolicy()
    raise ValueError(f"Unknown base policy: {kind}")


def build_residual_base_policy(config):
    if str(config.get("residual_glue", "fixed")) == "phase_aware":
        return PhaseAwareResidualBasePolicy(config)
    return build_base_policy(config.get("residual_base_policy", "risk_aware"))


def combine_residual_action(
    base_action,
    residual_action,
    beta=0.25,
    guard=True,
    min_alignment=0.15,
    mode="add",
):
    base_action = np.asarray(base_action, dtype=np.float32)
    residual_action = np.asarray(residual_action, dtype=np.float32)
    beta = float(np.clip(beta, 0.0, 1.0))
    if str(mode) == "blend":
        candidate = ((1.0 - beta) * base_action + beta * residual_action).astype(np.float32)
    else:
        candidate = (base_action + beta * residual_action).astype(np.float32)
    candidate = np.clip(candidate, -1.0, 1.0).astype(np.float32)
    if guard:
        base_xy = base_action[:2]
        cand_xy = candidate[:2]
        base_norm = float(np.linalg.norm(base_xy))
        cand_norm = float(np.linalg.norm(cand_xy))
        if base_norm > 1e-6 and cand_norm > 1e-6:
            alignment = float(np.dot(base_xy, cand_xy) / (base_norm * cand_norm + 1e-8))
            if alignment < min_alignment:
                return base_action.astype(np.float32)
        elif base_norm > 1e-6 and cand_norm <= 1e-6:
            return base_action.astype(np.float32)
    return candidate


def planner_gate_action(env, base_controller, obs, gate_action):
    if base_controller is None:
        return np.clip(np.asarray(gate_action, dtype=np.float32), -1.0, 1.0).astype(np.float32)
    if hasattr(base_controller, "act_with_gate"):
        return base_controller.act_with_gate(env, obs, gate_action)
    return base_controller.act(env, obs)


def planner_gate_bc_target(env):
    gate = np.zeros(3, dtype=np.float32)
    max_ratio = max_deadline_arrival_ratio(env)
    active_count = len(getattr(env, "steams", []) or [])
    sparse_threshold = int(getattr(env, "burst_lull_sparse_threshold", 2))
    phase = str(getattr(env, "burst_lull_phase", "") or "")

    # Gate 0: lock bias. Positive keeps the current target; negative makes
    # urgent switches easier. This teaches anti-jitter unless the queue is old.
    gate[0] = float(np.clip(0.55 - 1.75 * max(max_ratio - 0.52, 0.0), -0.75, 0.65))

    # Gate 1: rescue bias. Positive lowers the sticky switch threshold.
    gate[1] = float(np.clip((max_ratio - 0.58) / 0.24, -0.60, 1.0))

    # Gate 2: thermal anticipation. Only encourage hotspot drift in sparse,
    # low-risk lulls; otherwise keep planner motion target-centric.
    if active_count <= sparse_threshold and phase in ("lull", "charging") and max_ratio < 0.62:
        gate[2] = 0.75
    elif active_count == 0 and phase in ("lull", "charging", "burst"):
        gate[2] = 0.45
    else:
        gate[2] = -0.35
    return gate.astype(np.float32)


def shield_residual_action(
    env,
    base_action,
    candidate_action,
    previous_action=None,
    stagnation_steps=0,
    recovery_steps=180,
    min_progress_margin=0.002,
    min_progress_ratio=0.35,
    guarded_progress_ratio=0.50,
    allow_backtrack_steps=0.0,
):
    base_action = np.asarray(base_action, dtype=np.float32)
    candidate_action = np.asarray(candidate_action, dtype=np.float32)
    if env is None or not getattr(env, "steams", None):
        return np.clip(candidate_action, -1.0, 1.0).astype(np.float32)

    base_xy = base_action[:2]
    base_norm = float(np.linalg.norm(base_xy))
    target = None
    if base_norm > 1e-6:
        best_alignment_score = -np.inf
        base_dir = base_xy / base_norm
        for steam in env.steams:
            vec = steam["pos"][:2] - env.cover_center
            dist = float(np.linalg.norm(vec))
            if dist <= 1e-6:
                alignment = 1.0
            else:
                alignment = float(np.dot(base_dir, vec / dist))
            alignment_score = alignment + 0.12 * score_steam(env, steam) - 0.08 * dist / max(env.pot_radius, 1e-6)
            if alignment_score > best_alignment_score:
                best_alignment_score = alignment_score
                target = steam
    if target is None:
        target = select_receding_horizon_steam(env, horizon=2)
    if target is None:
        return np.clip(candidate_action, -1.0, 1.0).astype(np.float32)

    center = np.asarray(env.cover_center, dtype=np.float32)
    target_xy = target["pos"][:2]
    current_dist = float(np.linalg.norm(target_xy - center))
    step_scale = max(float(getattr(env, "track_step_size", 0.04)), 1e-6)

    def projected_progress(action):
        action = np.asarray(action, dtype=np.float32)
        xy = action[:2]
        norm = float(np.linalg.norm(xy))
        if norm > 1.0:
            xy = xy / norm
        speed_scale = 0.65 + 0.35 * ((float(action[2]) + 1.0) * 0.5) if action.shape[0] > 2 else 1.0
        projected_xy = center + xy * step_scale * speed_scale
        return current_dist - float(np.linalg.norm(target_xy - projected_xy))

    base_progress = projected_progress(base_action)
    candidate_progress = projected_progress(candidate_action)
    recovery_mode = int(stagnation_steps) >= int(recovery_steps)
    backtrack_allowance = max(float(allow_backtrack_steps), 0.0) * step_scale
    required_progress = max(-backtrack_allowance, float(min_progress_ratio) * base_progress)

    if candidate_progress + min_progress_margin < required_progress:
        guarded = 0.70 * base_action + 0.30 * candidate_action
        guarded_progress = projected_progress(guarded)
        guarded_required = max(-backtrack_allowance, float(guarded_progress_ratio) * base_progress)
        if guarded_progress + min_progress_margin < guarded_required:
            return np.clip(base_action, -1.0, 1.0).astype(np.float32)
        return np.clip(guarded, -1.0, 1.0).astype(np.float32)

    if recovery_mode and candidate_progress + min_progress_margin < base_progress:
        return np.clip(base_action, -1.0, 1.0).astype(np.float32)

    if previous_action is not None:
        previous_action = np.asarray(previous_action, dtype=np.float32)
        prev_xy = previous_action[:2]
        cand_xy = candidate_action[:2]
        prev_norm = float(np.linalg.norm(prev_xy))
        cand_norm = float(np.linalg.norm(cand_xy))
        if prev_norm > 1e-6 and cand_norm > 1e-6:
            turn_alignment = float(np.dot(prev_xy, cand_xy) / (prev_norm * cand_norm + 1e-8))
            if turn_alignment < -0.55 and candidate_progress < base_progress:
                return np.clip(0.55 * base_action + 0.45 * candidate_action, -1.0, 1.0).astype(np.float32)

    return np.clip(candidate_action, -1.0, 1.0).astype(np.float32)


def select_nearest_steam(env):
    if hasattr(env, "select_nearest_steam"):
        return env.select_nearest_steam()
    if not env.steams:
        return None
    return min(env.steams, key=lambda item: np.linalg.norm(item["pos"][:2] - env.cover_center))


def select_distance_age_steam(env):
    if not env.steams:
        return None
    best_steam = None
    best_score = -np.inf
    for steam in env.steams:
        dist = float(np.linalg.norm(steam["pos"][:2] - env.cover_center))
        dist_score = 1.0 - np.clip(dist / max(env.pot_radius, 1e-6), 0.0, 1.0)
        age_score = np.clip(steam["age"] / max(env.max_steam_age, 1), 0.0, 1.0)
        score = 0.55 * age_score + 0.45 * dist_score
        if score > best_score:
            best_score = score
            best_steam = steam
    return best_steam


def select_risk_aware_steam(env):
    """Select a target using persistent-steam risk features.

    Steam points do not time out in the target task. Age is therefore a
    persistence/neglect signal, not an expiry countdown.
    """
    if hasattr(env, "select_risk_aware_steam"):
        return env.select_risk_aware_steam()
    if not env.steams:
        return None
    best_steam = None
    best_score = -np.inf
    for steam in env.steams:
        dist = float(np.linalg.norm(steam["pos"][:2] - env.cover_center))
        dist_score = 1.0 - np.clip(dist / max(env.pot_radius, 1e-6), 0.0, 1.0)
        # No automatic timeout: older steam means it has been neglected longer,
        # not that it is about to disappear.
        age_score = np.clip(steam["age"] / max(env.max_steam_age, 1), 0.0, 1.0)

        cell_xy = steam["pos"][:2]
        material_score = 0.0
        if material_scoring_enabled(env):
            grid_dist = np.linalg.norm(env.grid_world_xy - cell_xy, axis=-1)
            local_gap = float(np.mean(np.maximum(env.target_layer_height - env.material_height, 0.0) * np.exp(-grid_dist ** 2 / (2.0 * env.deposit_sigma ** 2))))
            material_score = np.clip(local_gap / max(env.target_layer_height, 1e-6), 0.0, 1.0)

        reach_offset = np.linalg.norm(cell_xy - env.home_ee_pos[:2]) if env.home_ee_pos is not None else 0.0
        reach_score = 1.0 - np.clip(reach_offset / max(env.max_target_offset, 1e-6), 0.0, 1.0)

        thermal_score = thermal_score_at_xy(env, cell_xy)

        if material_scoring_enabled(env):
            score = (
                0.35 * age_score
                + 0.25 * dist_score
                + 0.15 * material_score
                + 0.10 * reach_score
                + 0.15 * thermal_score
            )
        else:
            score = (
                0.40 * age_score
                + 0.30 * dist_score
                + 0.10 * reach_score
                + 0.20 * thermal_score
            )
        if score > best_score:
            best_score = score
            best_steam = steam
    return best_steam


def thermal_score_at_xy(env, xy):
    scorer = getattr(env, "_thermal_score_at_xy", None)
    if scorer is None:
        return 0.0
    try:
        return float(np.clip(scorer(np.asarray(xy, dtype=np.float32)), 0.0, 1.0))
    except Exception:
        return 0.0


def material_scoring_enabled(env):
    return bool(getattr(env, "material_observation_enabled", True))


def select_dynamic_weighted_steam(env):
    if not env.steams:
        return None
    density = np.clip(len(env.steams) / max(env.max_steams, 1), 0.0, 1.0)
    spawn_pressure = float(np.clip(env.spawn_probability * max(env.spawn_cooldown_steps, 1), 0.0, 1.0))
    if material_scoring_enabled(env):
        material_pressure = np.clip(env.material_quality_loss, 0.0, 1.5) / 1.5
        weights = {
            "age": 0.28 + 0.16 * density,
            "distance": 0.34 - 0.10 * density,
            "material": 0.16 + 0.14 * material_pressure,
            "reachability": 0.10 - 0.04 * material_pressure,
            "thermal": 0.12 + 0.10 * max(density, spawn_pressure),
        }
    else:
        weights = {
            "age": 0.34 + 0.16 * density,
            "distance": 0.38 - 0.08 * density,
            "material": 0.0,
            "reachability": 0.10,
            "thermal": 0.18 + 0.08 * max(density, spawn_pressure),
        }
    return max(env.steams, key=lambda steam: score_steam(env, steam, env.cover_center, weights))


def score_steam(env, steam, center_xy=None, weights=None):
    if center_xy is None:
        center_xy = env.cover_center
    if hasattr(env, "_steam_risk_components"):
        components = env._steam_risk_components(steam, center_xy)
        age_score = components["selected_target_age_score"]
        distance_score = components["selected_target_distance_score"]
        material_score = components["selected_target_material_score"]
        reachability_score = components["selected_target_reachability_score"]
        thermal_score = components.get("selected_target_thermal_score", thermal_score_at_xy(env, steam["pos"][:2]))
    else:
        dist = float(np.linalg.norm(steam["pos"][:2] - center_xy))
        distance_score = 1.0 - np.clip(dist / max(env.pot_radius, 1e-6), 0.0, 1.0)
        age_score = np.clip(steam["age"] / max(env.max_steam_age, 1), 0.0, 1.0)
        material_score = 0.0
        reachability_score = 1.0
        thermal_score = thermal_score_at_xy(env, steam["pos"][:2])
    if not material_scoring_enabled(env):
        material_score = 0.0
    weights = weights or (
        {
            "age": 0.35,
            "distance": 0.25,
            "material": 0.15,
            "reachability": 0.10,
            "thermal": 0.15,
        }
        if material_scoring_enabled(env)
        else {
            "age": 0.40,
            "distance": 0.30,
            "material": 0.0,
            "reachability": 0.10,
            "thermal": 0.20,
        }
    )
    return float(
        weights.get("age", 0.0) * age_score
        + weights.get("distance", 0.0) * distance_score
        + weights.get("material", 0.0) * material_score
        + weights.get("reachability", 0.0) * reachability_score
        + weights.get("thermal", 0.0) * thermal_score
    )


def select_receding_horizon_steam(env, horizon=2):
    if not env.steams:
        return None
    active = list(env.steams)
    if len(active) == 1:
        return active[0]
    horizon = max(1, min(int(horizon), len(active), 4))
    best_route = None
    best_score = -np.inf
    for route in permutations(active, horizon):
        score = score_target_route(env, route)
        if score > best_score:
            best_score = score
            best_route = route
    return best_route[0] if best_route else select_dynamic_weighted_steam(env)


def score_target_route(env, route):
    center = np.asarray(env.cover_center, dtype=np.float32).copy()
    elapsed = 0.0
    total = 0.0
    covered_ids = set()
    for rank, steam in enumerate(route):
        xy = steam["pos"][:2]
        dist = float(np.linalg.norm(xy - center))
        travel_steps = max(1.0, np.ceil(max(dist - env.cover_radius, 0.0) / max(env.track_step_size, 1e-6)))
        arrival_age = float(steam["age"] + elapsed + travel_steps)
        urgency = np.clip(arrival_age / max(env.max_steam_age, 1), 0.0, 1.5)
        local_score = score_steam(env, steam, center)
        latency_penalty = 0.018 * travel_steps + 0.050 * max(urgency - 1.0, 0.0)
        discount = 0.82 ** rank
        total += discount * (local_score + 0.45 * urgency - latency_penalty)
        center = xy.copy()
        elapsed += travel_steps
        covered_ids.add(steam.get("id"))

    leftovers = [steam for steam in env.steams if steam.get("id") not in covered_ids]
    if leftovers:
        leftover_age = np.mean([steam["age"] + elapsed for steam in leftovers])
        total -= 0.08 * np.clip(leftover_age / max(env.max_steam_age, 1), 0.0, 2.0)
    return float(total)


def select_deadline_horizon_steam(env, horizon=2, rescue=False):
    if not env.steams:
        return None
    active = list(env.steams)
    if len(active) == 1:
        return active[0]
    if rescue:
        rescue_target = select_deadline_rescue_steam(env, active)
        if rescue_target is not None:
            return rescue_target
    horizon = max(1, min(int(horizon), len(active), 4))
    best_route = None
    best_score = -np.inf
    for route in permutations(active, horizon):
        score = score_deadline_target_route(env, route)
        if score > best_score:
            best_score = score
            best_route = route
    return best_route[0] if best_route else max(active, key=lambda steam: steam["age"])


def select_deadline_rescue_steam(env, active=None, threshold=0.65):
    active = list(env.steams if active is None else active)
    if not active:
        return None
    center = np.asarray(env.cover_center, dtype=np.float32)
    sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)
    rescue_candidates = []
    for steam in active:
        xy = steam["pos"][:2]
        dist = float(np.linalg.norm(xy - center))
        travel_steps = max(1.0, np.ceil(max(dist - env.cover_radius, 0.0) / max(env.track_step_size, 1e-6)))
        arrival_ratio = float((steam["age"] + travel_steps) / max(sla_steps, 1))
        age_ratio = float(steam["age"] / max(sla_steps, 1))
        travel_ratio = float(np.clip(travel_steps / max(sla_steps, 1), 0.0, 1.0))
        if arrival_ratio < threshold and age_ratio < threshold:
            continue
        rescue_score = (
            2.40 * min(arrival_ratio, 2.0)
            + 1.10 * min(age_ratio, 2.0)
            + 0.15 * thermal_score_at_xy(env, xy)
            - 0.30 * travel_ratio
        )
        rescue_candidates.append((rescue_score, steam))
    if not rescue_candidates:
        return None
    return max(rescue_candidates, key=lambda item: item[0])[1]


def _travel_steps_between(env, start_xy, target_xy):
    dist = float(np.linalg.norm(np.asarray(target_xy, dtype=np.float32) - np.asarray(start_xy, dtype=np.float32)))
    return max(1.0, np.ceil(max(dist - env.cover_radius, 0.0) / max(env.track_step_size, 1e-6)))


def select_slack_horizon_steam(env, horizon=2):
    if not getattr(env, "steams", None):
        return None
    active = list(env.steams)
    if len(active) == 1:
        return active[0]

    center = np.asarray(env.cover_center, dtype=np.float32)
    sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)
    critical = []
    for steam in active:
        xy = steam["pos"][:2]
        travel_steps = _travel_steps_between(env, center, xy)
        arrival_ratio = float((steam["age"] + travel_steps) / max(sla_steps, 1))
        age_ratio = float(steam["age"] / max(sla_steps, 1))
        slack_steps = float(sla_steps - steam["age"] - travel_steps)
        if arrival_ratio >= 0.62 or slack_steps <= max(35.0, 0.16 * sla_steps):
            travel_ratio = float(np.clip(travel_steps / max(sla_steps, 1), 0.0, 1.0))
            critical_score = (
                3.80 * min(arrival_ratio, 2.0)
                + 1.65 * min(age_ratio, 2.0)
                + 0.20 * thermal_score_at_xy(env, xy)
                + 0.08 * score_steam(env, steam, center)
                - 0.28 * travel_ratio
                - 0.004 * travel_steps
            )
            critical.append((critical_score, steam))
    if critical:
        return max(critical, key=lambda item: item[0])[1]

    horizon = max(1, min(int(horizon), len(active), 4))
    best_route = None
    best_score = -np.inf
    for route in permutations(active, horizon):
        score = score_slack_target_route(env, route)
        if score > best_score:
            best_score = score
            best_route = route
    return best_route[0] if best_route else select_deadline_horizon_steam(env, horizon=2, rescue=True)


def score_slack_target_route(env, route):
    center = np.asarray(env.cover_center, dtype=np.float32).copy()
    elapsed = 0.0
    total = 0.0
    covered_ids = set()
    sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)

    for rank, steam in enumerate(route):
        xy = steam["pos"][:2]
        travel_steps = _travel_steps_between(env, center, xy)
        arrival_age = float(steam["age"] + elapsed + travel_steps)
        arrival_ratio = float(arrival_age / max(sla_steps, 1))
        age_ratio = float(steam["age"] / max(sla_steps, 1))
        travel_ratio = float(np.clip(travel_steps / max(sla_steps, 1), 0.0, 1.0))
        slack_ratio = float(1.0 - arrival_ratio)
        deadline_pressure = float(np.clip((arrival_ratio - 0.44) / 0.34, 0.0, 2.0))
        late_pressure = float(np.clip(arrival_ratio - 1.0, 0.0, 2.5))
        local_score = score_steam(env, steam, center)
        thermal_score = thermal_score_at_xy(env, xy)
        discount = 0.84 ** rank
        total += discount * (
            0.50 * local_score
            + 1.70 * deadline_pressure
            + 0.80 * np.clip(age_ratio, 0.0, 1.5)
            + 0.20 * thermal_score
            + 0.16 * np.clip(slack_ratio, 0.0, 1.0)
            - 3.10 * late_pressure
            - 0.40 * travel_ratio
            - 0.006 * travel_steps
        )
        center = np.asarray(xy, dtype=np.float32).copy()
        elapsed += travel_steps
        covered_ids.add(steam.get("id"))

    leftovers = [steam for steam in env.steams if steam.get("id") not in covered_ids]
    if leftovers:
        leftover_ratios = []
        for steam in leftovers:
            xy = steam["pos"][:2]
            travel_steps = _travel_steps_between(env, center, xy)
            leftover_ratios.append(float((steam["age"] + elapsed + travel_steps) / max(sla_steps, 1)))
        max_leftover = max(leftover_ratios)
        mean_leftover = float(np.mean(leftover_ratios))
        total -= (
            1.70 * np.clip((max_leftover - 0.62) / 0.30, 0.0, 2.0)
            + 0.58 * np.clip((mean_leftover - 0.50) / 0.45, 0.0, 2.0)
        )
    return float(total)


def select_corridor_waypoint_steam(env):
    if not getattr(env, "steams", None):
        return None
    active = list(env.steams)
    if len(active) == 1:
        return active[0]

    primary = select_deadline_horizon_steam(env, horizon=2, rescue=True)
    if primary is None:
        return select_dynamic_weighted_steam(env)

    center = np.asarray(env.cover_center, dtype=np.float32)
    primary_xy = np.asarray(primary["pos"][:2], dtype=np.float32)
    route_vec = primary_xy - center
    route_len = float(np.linalg.norm(route_vec))
    if route_len <= 1e-6:
        return primary

    route_dir = route_vec / route_len
    sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)
    direct_primary_steps = _travel_steps_between(env, center, primary_xy)
    direct_primary_ratio = float((primary["age"] + direct_primary_steps) / max(sla_steps, 1))
    if direct_primary_ratio >= 0.92:
        return primary

    corridor_width = max(float(getattr(env, "cover_radius", 0.1)) * 1.65, 0.14)
    best_candidate = None
    best_score = -np.inf
    primary_id = primary.get("id")
    max_extra_steps = max(12.0, 0.13 * sla_steps)

    for steam in active:
        if steam.get("id") == primary_id:
            continue
        steam_xy = np.asarray(steam["pos"][:2], dtype=np.float32)
        rel = steam_xy - center
        projection = float(np.dot(rel, route_dir))
        if projection < -0.04 or projection > route_len + corridor_width:
            continue
        closest = center + route_dir * np.clip(projection, 0.0, route_len)
        lateral = float(np.linalg.norm(steam_xy - closest))
        if lateral > corridor_width:
            continue

        to_candidate = _travel_steps_between(env, center, steam_xy)
        candidate_to_primary = _travel_steps_between(env, steam_xy, primary_xy)
        extra_steps = float(to_candidate + candidate_to_primary - direct_primary_steps)
        if extra_steps > max_extra_steps:
            continue

        primary_after_ratio = float((primary["age"] + to_candidate + candidate_to_primary) / max(sla_steps, 1))
        if primary_after_ratio > 0.98:
            continue

        candidate_age_ratio = float((steam["age"] + to_candidate) / max(sla_steps, 1))
        route_fraction = float(np.clip(projection / max(route_len, 1e-6), 0.0, 1.0))
        lateral_penalty = lateral / max(corridor_width, 1e-6)
        thermal_score = thermal_score_at_xy(env, steam_xy)
        local_score = score_steam(env, steam, center)
        deadline_slack = 1.0 - primary_after_ratio
        score = (
            0.95 * local_score
            + 0.85 * np.clip(candidate_age_ratio, 0.0, 1.5)
            + 0.22 * thermal_score
            + 0.18 * route_fraction
            + 0.22 * np.clip(deadline_slack, 0.0, 1.0)
            - 0.38 * lateral_penalty
            - 0.010 * max(extra_steps, 0.0)
        )
        if score > best_score:
            best_score = float(score)
            best_candidate = steam

    if best_candidate is None:
        return primary

    primary_score = score_steam(env, primary, center) + 0.40 * np.clip(direct_primary_ratio, 0.0, 1.5)
    if best_score < primary_score + 0.05:
        return primary
    return best_candidate


def _route_opportunity_score(env, target_xy, skip_id=None):
    center = np.asarray(env.cover_center, dtype=np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32)
    route_vec = target_xy - center
    route_len = float(np.linalg.norm(route_vec))
    if route_len <= 1e-6:
        return 0.0
    route_dir = route_vec / route_len
    corridor_width = max(float(getattr(env, "cover_radius", 0.1)) * 1.75, 0.15)
    direct_steps = _travel_steps_between(env, center, target_xy)
    sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)
    max_extra_steps = max(10.0, 0.12 * sla_steps)
    score = 0.0
    for steam in getattr(env, "steams", []) or []:
        if steam.get("id") == skip_id:
            continue
        xy = np.asarray(steam["pos"][:2], dtype=np.float32)
        rel = xy - center
        projection = float(np.dot(rel, route_dir))
        if projection < -0.04 or projection > route_len + corridor_width:
            continue
        closest = center + route_dir * np.clip(projection, 0.0, route_len)
        lateral = float(np.linalg.norm(xy - closest))
        if lateral > corridor_width:
            continue
        extra_steps = _travel_steps_between(env, center, xy) + _travel_steps_between(env, xy, target_xy) - direct_steps
        if extra_steps > max_extra_steps:
            continue
        age_ratio = float(np.clip(steam["age"] / max(sla_steps, 1), 0.0, 1.5))
        thermal_score = thermal_score_at_xy(env, xy)
        score += (1.0 - lateral / max(corridor_width, 1e-6)) * (0.45 + 0.70 * age_ratio + 0.25 * thermal_score)
    return float(score)


def select_sla_route_ensemble_steam(env):
    if not getattr(env, "steams", None):
        return None
    active = list(env.steams)
    if len(active) == 1:
        return active[0]

    center = np.asarray(env.cover_center, dtype=np.float32)
    sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)
    best_steam = None
    best_score = -np.inf
    for steam in active:
        xy = np.asarray(steam["pos"][:2], dtype=np.float32)
        travel_steps = _travel_steps_between(env, center, xy)
        arrival_ratio = float((steam["age"] + travel_steps) / max(sla_steps, 1))
        age_ratio = float(steam["age"] / max(sla_steps, 1))
        travel_ratio = float(np.clip(travel_steps / max(sla_steps, 1), 0.0, 1.0))
        urgency = float(np.clip((arrival_ratio - 0.36) / 0.44, 0.0, 2.2))
        near_late_bonus = float(np.clip((arrival_ratio - 0.70) / 0.25, 0.0, 1.8))
        opportunity = _route_opportunity_score(env, xy, skip_id=steam.get("id"))
        risk_score = score_steam(env, steam, center)
        thermal_score = thermal_score_at_xy(env, xy)

        leftovers = [item for item in active if item.get("id") != steam.get("id")]
        if leftovers:
            future_ratios = [
                float((item["age"] + travel_steps + _travel_steps_between(env, xy, item["pos"][:2])) / max(sla_steps, 1))
                for item in leftovers
            ]
            future_max = max(future_ratios)
            future_mean = float(np.mean(future_ratios))
        else:
            future_max = 0.0
            future_mean = 0.0
        future_penalty = (
            1.25 * np.clip((future_max - 0.82) / 0.28, 0.0, 2.0)
            + 0.38 * np.clip((future_mean - 0.62) / 0.38, 0.0, 2.0)
        )

        score = (
            1.65 * urgency
            + 1.05 * near_late_bonus
            + 0.62 * np.clip(age_ratio, 0.0, 1.5)
            + 0.50 * risk_score
            + 0.35 * np.clip(opportunity, 0.0, 3.0)
            + 0.14 * thermal_score
            - future_penalty
            - 0.28 * travel_ratio
            - 0.004 * travel_steps
        )
        if score > best_score:
            best_score = float(score)
            best_steam = steam

    return best_steam if best_steam is not None else select_slack_horizon_steam(env, horizon=2)


def max_deadline_arrival_ratio(env):
    if not getattr(env, "steams", None):
        return 0.0
    center = np.asarray(env.cover_center, dtype=np.float32)
    sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)
    ratios = []
    for steam in env.steams:
        xy = steam["pos"][:2]
        dist = float(np.linalg.norm(xy - center))
        travel_steps = max(1.0, np.ceil(max(dist - env.cover_radius, 0.0) / max(env.track_step_size, 1e-6)))
        ratios.append(float((steam["age"] + travel_steps) / max(sla_steps, 1)))
    return max(ratios) if ratios else 0.0


def score_deadline_target_route(env, route):
    center = np.asarray(env.cover_center, dtype=np.float32).copy()
    elapsed = 0.0
    total = 0.0
    covered_ids = set()
    sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)
    max_age = max(int(getattr(env, "max_steam_age", sla_steps)), 1)
    weights = {
        "age": 0.58,
        "distance": 0.14,
        "material": 0.0,
        "reachability": 0.10,
        "thermal": 0.18,
    }
    for rank, steam in enumerate(route):
        xy = steam["pos"][:2]
        dist = float(np.linalg.norm(xy - center))
        travel_steps = max(1.0, np.ceil(max(dist - env.cover_radius, 0.0) / max(env.track_step_size, 1e-6)))
        arrival_age = float(steam["age"] + elapsed + travel_steps)
        age_ratio = float(np.clip(arrival_age / max(sla_steps, 1), 0.0, 2.5))
        raw_age_ratio = float(np.clip(arrival_age / max(max_age, 1), 0.0, 1.5))
        deadline_pressure = float(np.clip((age_ratio - 0.50) / 0.50, 0.0, 1.6))
        late_pressure = float(np.clip(age_ratio - 1.0, 0.0, 2.0))
        travel_ratio = float(np.clip(travel_steps / max(sla_steps, 1), 0.0, 1.0))
        local_score = score_steam(env, steam, center, weights)
        thermal_score = thermal_score_at_xy(env, xy)
        discount = 0.86 ** rank
        total += discount * (
            0.55 * local_score
            + 0.70 * raw_age_ratio
            + 1.35 * deadline_pressure
            + 1.10 * late_pressure
            + 0.12 * thermal_score
            - 0.34 * travel_ratio
            - 0.006 * travel_steps
        )
        center = xy.copy()
        elapsed += travel_steps
        covered_ids.add(steam.get("id"))

    leftovers = [steam for steam in env.steams if steam.get("id") not in covered_ids]
    if leftovers:
        leftover_ratios = np.array(
            [(steam["age"] + elapsed) / max(sla_steps, 1) for steam in leftovers],
            dtype=np.float32,
        )
        oldest_leftover = float(np.max(leftover_ratios))
        mean_leftover = float(np.mean(leftover_ratios))
        starvation_pressure = np.clip((oldest_leftover - 0.55) / 0.45, 0.0, 2.0)
        total -= 0.80 * starvation_pressure + 0.22 * np.clip(mean_leftover, 0.0, 2.0)
    return float(total)


def plan_aco_tsp_route(env, ants=12, iterations=3, alpha=1.0, beta=2.0):
    steams = list(env.steams)
    n = len(steams)
    if n <= 1:
        return steams
    positions = np.array([steam["pos"][:2] for steam in steams], dtype=np.float32)
    pheromone = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(pheromone, 0.0)
    rng = getattr(env, "rng", np.random.default_rng())
    best_order = None
    best_cost = np.inf

    urgency = np.array([0.35 + score_steam(env, steam, env.cover_center) for steam in steams], dtype=np.float64)
    for _ in range(max(int(iterations), 1)):
        routes = []
        for _ in range(max(int(ants), 1)):
            unvisited = list(range(n))
            center = np.asarray(env.cover_center, dtype=np.float32)
            order = []
            while unvisited:
                if not order:
                    dists = np.array([np.linalg.norm(positions[idx] - center) for idx in unvisited], dtype=np.float64)
                    heuristic = urgency[unvisited] / np.maximum(dists, 1e-3)
                else:
                    last = order[-1]
                    dists = np.array([np.linalg.norm(positions[idx] - positions[last]) for idx in unvisited], dtype=np.float64)
                    heuristic = (pheromone[last, unvisited] ** alpha) * (
                        (urgency[unvisited] / np.maximum(dists, 1e-3)) ** beta
                    )
                probs = heuristic / max(float(heuristic.sum()), 1e-12)
                choice_pos = int(rng.choice(len(unvisited), p=probs))
                order.append(unvisited.pop(choice_pos))
            cost = route_cost(env, steams, order)
            routes.append((order, cost))
            if cost < best_cost:
                best_cost = cost
                best_order = order
        pheromone *= 0.70
        for order, cost in routes:
            deposit = 1.0 / max(abs(cost), 1e-6)
            for i, j in zip(order[:-1], order[1:]):
                pheromone[i, j] += deposit
                pheromone[j, i] += deposit
    return [steams[idx] for idx in best_order] if best_order else steams


def route_cost(env, steams, order):
    center = np.asarray(env.cover_center, dtype=np.float32)
    cost = 0.0
    elapsed = 0.0
    for idx in order:
        steam = steams[idx]
        xy = steam["pos"][:2]
        dist = float(np.linalg.norm(xy - center))
        travel_steps = max(1.0, np.ceil(max(dist - env.cover_radius, 0.0) / max(env.track_step_size, 1e-6)))
        urgency = np.clip((steam["age"] + elapsed + travel_steps) / max(env.max_steam_age, 1), 0.0, 1.5)
        cost += travel_steps - 10.0 * urgency - 4.0 * score_steam(env, steam, center)
        center = xy.copy()
        elapsed += travel_steps
    return float(cost)


def action_toward_steam(env, steam):
    if steam is None:
        return np.zeros(env.action_space.shape[0], dtype=np.float32)
    return action_toward_xy(env, steam["pos"][:2])


def action_toward_xy(env, xy):
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    if xy is None:
        return action
    vec = np.asarray(xy, dtype=np.float32)[:2] - env.cover_center
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        action[:2] = vec / norm
    action[2] = 1.0
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def maybe_blend_toward_hotspot(env, action, target, hotspot_bias):
    if hotspot_bias <= 0.35 or target is None:
        return action
    active_count = len(getattr(env, "steams", []) or [])
    phase = str(getattr(env, "burst_lull_phase", "") or "")
    if active_count > max(2, int(getattr(env, "burst_lull_sparse_threshold", 2))) and phase not in ("lull", "charging"):
        return action
    sla_steps = max(int(getattr(env, "response_sla_steps", 0) or getattr(env, "max_steam_age", 1)), 1)
    target_ratio = float((target["age"] + _travel_steps_between(env, env.cover_center, target["pos"][:2])) / max(sla_steps, 1))
    if target_ratio >= 0.65:
        return action
    hotspot = getattr(env, "_dominant_thermal_hotspot", lambda: None)()
    if hotspot is None:
        return action
    hotspot_action = action_toward_xy(env, hotspot["xy"])
    blend = float(np.clip(0.18 * hotspot_bias, 0.0, 0.18))
    mixed = (1.0 - blend) * np.asarray(action, dtype=np.float32) + blend * hotspot_action
    return np.clip(mixed, env.action_space.low, env.action_space.high).astype(np.float32)


def build_policy(kind, env, model_path=None, deterministic=True, config_override=None):
    if kind in (
        "random",
        "nearest",
        "oldest",
        "distance_age",
        "risk_aware",
        "dynamic_weighted",
        "horizon2",
        "horizon3",
        "deadline_horizon2",
        "deadline_rescue_horizon2",
        "slack_horizon2",
        "corridor_waypoint",
        "sla_route_ensemble",
        "sticky_sla_ensemble",
        "aco_tsp",
        "planner_ensemble",
    ):
        return build_base_policy(kind)
    if kind == "ppo":
        if not model_path:
            raise ValueError("--model is required for --policy ppo")
        return PPOPolicy(env, model_path, deterministic=deterministic, config_override=config_override)
    raise ValueError(f"Unknown policy: {kind}")
