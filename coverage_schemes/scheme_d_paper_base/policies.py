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


class PPOPolicy:
    name = "ppo"

    def __init__(self, env, model_path, deterministic=True):
        config = checkpoint_config(model_path)
        self.agent = PPOAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            seq_len=env.max_episode_steps,
            use_lstm=config.get("use_lstm", True),
            use_steam_attention=config.get("use_steam_attention", False),
            use_material_map=config.get("use_material_map", False),
            base_obs_dim=getattr(env, "base_obs_dim", config.get("base_obs_dim", 35)),
            device=config.get("device", "auto"),
        )
        load_checkpoint(self.agent, model_path)
        self.agent.model.eval()
        self.deterministic = deterministic
        self.hx = None
        self.cx = None
        self.residual_policy = bool(config.get("residual_policy", False))
        self.residual_beta = float(config.get("residual_beta", 0.25))
        self.residual_guard = bool(config.get("residual_guard", True))
        self.residual_base_policy_name = config.get("residual_base_policy", "risk_aware")
        self.residual_base_policy = build_base_policy(self.residual_base_policy_name) if self.residual_policy else None
        if self.residual_policy:
            self.name = f"residual_{self.residual_base_policy_name}_ppo"

    def reset(self):
        self.hx = None
        self.cx = None
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
            base_action = self.residual_base_policy.act(env, obs)
            action = combine_residual_action(
                base_action,
                action,
                beta=self.residual_beta,
                guard=self.residual_guard,
            )
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
    if kind == "aco_tsp":
        return AcoTspPolicy()
    raise ValueError(f"Unknown base policy: {kind}")


def combine_residual_action(base_action, residual_action, beta=0.25, guard=True, min_alignment=0.15):
    base_action = np.asarray(base_action, dtype=np.float32)
    residual_action = np.asarray(residual_action, dtype=np.float32)
    candidate = np.clip(base_action + float(beta) * residual_action, -1.0, 1.0).astype(np.float32)
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
        grid_dist = np.linalg.norm(env.grid_world_xy - cell_xy, axis=-1)
        local_gap = float(np.mean(np.maximum(env.target_layer_height - env.material_height, 0.0) * np.exp(-grid_dist ** 2 / (2.0 * env.deposit_sigma ** 2))))
        material_score = np.clip(local_gap / max(env.target_layer_height, 1e-6), 0.0, 1.0)

        reach_offset = np.linalg.norm(cell_xy - env.home_ee_pos[:2]) if env.home_ee_pos is not None else 0.0
        reach_score = 1.0 - np.clip(reach_offset / max(env.max_target_offset, 1e-6), 0.0, 1.0)

        score = 0.40 * age_score + 0.30 * dist_score + 0.20 * material_score + 0.10 * reach_score
        if score > best_score:
            best_score = score
            best_steam = steam
    return best_steam


def select_dynamic_weighted_steam(env):
    if not env.steams:
        return None
    density = np.clip(len(env.steams) / max(env.max_steams, 1), 0.0, 1.0)
    material_pressure = np.clip(env.material_quality_loss, 0.0, 1.5) / 1.5
    weights = {
        "age": 0.30 + 0.20 * density,
        "distance": 0.38 - 0.12 * density,
        "material": 0.18 + 0.18 * material_pressure,
        "reachability": 0.14 - 0.06 * material_pressure,
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
    else:
        dist = float(np.linalg.norm(steam["pos"][:2] - center_xy))
        distance_score = 1.0 - np.clip(dist / max(env.pot_radius, 1e-6), 0.0, 1.0)
        age_score = np.clip(steam["age"] / max(env.max_steam_age, 1), 0.0, 1.0)
        material_score = 0.0
        reachability_score = 1.0
    weights = weights or {"age": 0.40, "distance": 0.30, "material": 0.20, "reachability": 0.10}
    return float(
        weights["age"] * age_score
        + weights["distance"] * distance_score
        + weights["material"] * material_score
        + weights["reachability"] * reachability_score
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
    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    if steam is None:
        return action
    vec = steam["pos"][:2] - env.cover_center
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        action[:2] = vec / norm
    action[2] = 1.0
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def build_policy(kind, env, model_path=None, deterministic=True):
    if kind in ("random", "nearest", "oldest", "distance_age", "risk_aware", "dynamic_weighted", "horizon2", "horizon3", "aco_tsp"):
        return build_base_policy(kind)
    if kind == "ppo":
        if not model_path:
            raise ValueError("--model is required for --policy ppo")
        return PPOPolicy(env, model_path, deterministic=deterministic)
    raise ValueError(f"Unknown policy: {kind}")
