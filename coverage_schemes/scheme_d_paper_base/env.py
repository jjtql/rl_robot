from collections import deque
from pathlib import Path

try:
    import gymnasium as gym
except ModuleNotFoundError:
    try:
        import gym
    except ModuleNotFoundError:
        gym = None
import mujoco
import numpy as np


if gym is None:
    class _Env:
        def reset(self, seed=None, options=None):
            return None

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self):
            sample = np.random.uniform(self.low, self.high).astype(self.dtype)
            return sample

    class _Spaces:
        Box = _Box

    class _GymFallback:
        Env = _Env
        spaces = _Spaces()

    gym = _GymFallback()

spaces = gym.spaces


def default_model_path():
    return Path(__file__).resolve().parents[2] / "o1.xml"


class ShangZengEnv(gym.Env):
    """
    Steam-covering task: the end effector must move over active steam points.

    The reward is event-driven with bounded potential shaping. Progress rewards
    cannot be repeatedly farmed by moving away and back because each steam point
    tracks its own best potential reached so far. Steam points persist until
    covered; max_steam_age is only a normalization scale, not an expiry timer.
    """

    def __init__(
        self,
        model_path=None,
        max_episode_steps=400,
        target_selector="risk_aware",
        steam_attention_observation=False,
        spawn_history_observation=False,
        thermal_context_observation=False,
        material_map_observation=False,
        attention_steam_count=6,
        attention_steam_dim=8,
    ):
        super().__init__()
        if model_path is None:
            model_path = default_model_path()
        model_path = Path(model_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.model_path = str(model_path)

        self.max_episode_steps = max_episode_steps
        self.pot_origin = np.array([1.8, 0.0, 0.0], dtype=np.float32)
        self.pot_center = np.array([1.8, 0.0, 0.1], dtype=np.float32)
        self.pot_radius = 0.8
        self.max_steams = 3
        self.max_steam_age = 220
        self.steam_timeout_enabled = False
        self.spawn_probability = 0.012
        self.spawn_cooldown_steps = 55
        self.steam_speed = 0.0
        self.initial_spawn_count = 1
        self.thermal_spawn_enabled = True
        self.thermal_hotspot_count = 3
        self.thermal_hotspot_sigma = 0.22
        self.thermal_hotspot_strength = 1.8
        self.thermal_background_weight = 0.28
        self.thermal_drift_std = 0.006
        self.thermal_refresh_probability = 0.0025
        self.thermal_lifetime_steps = 520
        self.thermal_recent_spawn_radius = 0.18
        self.thermal_recent_spawn_suppression = 0.55
        self.thermal_recent_spawn_memory = 12
        self.target_selector = "risk_aware"
        self.set_target_selector(target_selector)
        self.base_obs_dim = 35
        self.spawn_history_obs_dim = 10
        self.thermal_context_obs_dim = 8
        self.attention_steam_count = int(attention_steam_count)
        self.attention_steam_dim = int(attention_steam_dim)
        if self.attention_steam_dim != 8:
            raise ValueError("attention_steam_dim must be 8 for the current steam feature schema")
        self.spawn_history_observation_enabled = bool(spawn_history_observation)
        self.thermal_context_observation_enabled = bool(thermal_context_observation)
        self.steam_attention_observation_enabled = bool(steam_attention_observation)
        self.material_map_observation_enabled = bool(material_map_observation)
        self.material_map_channels = 3

        # Motion parameters
        self.track_step_size = 0.04
        self.return_gain = 0.75
        self.max_correction = 0.036
        self.action_smoothing = 0.7
        self.action_delay_steps = 0
        self.action_noise_std = 0.0
        self.action_delay_buffer = deque()
        self.domain_randomization_enabled = False
        self.domain_randomization_scale = 0.12
        self.max_target_offset = 0.72
        self.cover_radius = 0.2  # 大半径，容易覆盖
        self.spawn_reach_margin = 0.04
        self.spawn_burst_probability = 0.0
        self.spawn_burst_min = 2
        self.spawn_burst_max = 3
        self.target_coverage = 0.8
        self.target_success_count = 8
        self.orientation_gain = 0.28
        self.orientation_limit = 0.24
        self.posture_return_gain = 0.06

        # Event reward + bounded potential shaping. Keep the scale PPO-friendly:
        # one successful cover should dominate shaping, but not explode value loss.
        self.time_penalty = 0.03
        self.active_steam_penalty = 0.01
        self.age_penalty_gain = 0.04
        self.cover_reward = 70.0
        self.quick_cover_bonus = 18.0
        self.precision_bonus = 12.0
        self.success_bonus = 30.0
        self.miss_penalty = 28.0
        self.potential_gain = 9.0
        self.potential_gamma = 0.985
        self.potential_delta_clip = 1.5
        self.best_progress_gain = 0.75
        self.action_delta_penalty_gain = 0.025
        self.action_l2_penalty_gain = 0.004
        self.potential_shaping_enabled = True
        self.best_progress_enabled = True
        self.material_observation_enabled = True
        self.action_penalty_enabled = True
        self.material_tv_reward_enabled = False
        self.material_tv_reward_gain = 6.0
        self.material_quality_delta_clip = 0.5
        self.material_hole_weight = 1.0
        self.material_tv_weight = 0.35
        self.material_overfill_weight = 0.6

        # Material grid
        self.grid_size = 9
        self.target_layer_height = 0.16
        self.max_layer_height = 0.22
        self.deposit_rate = 0.0022
        self.cover_deposit_rate = 0.006
        self.deposit_sigma = 0.18
        self.material_base_z = 0.055
        self._setup_material_grid()

        # Action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.refresh_observation_space()

        # State variables
        self.steams = []
        self.steam_history = deque(maxlen=16)
        self.thermal_hotspots = []
        self.recent_spawn_positions = deque(maxlen=self.thermal_recent_spawn_memory)
        self.last_spawn_xy = None
        self.spawned_this_step_positions = []
        self.last_steam_center = None
        self.steam_trend = np.zeros(2, dtype=np.float32)
        self.rng = np.random.default_rng()
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_ctrl = np.zeros(6, dtype=np.float32)
        self.prev_ee_pos = None
        self.target_qpos = None
        self.home_qpos = None
        self.home_ee_pos = None
        self.home_ee_mat = None
        self.target_ee_pos = None
        self.cover_center = self.pot_center[:2].copy()
        self.prev_cover_center = self.pot_center[:2].copy()
        self.prev_target_dist = None
        self.next_steam_id = 0
        self.steam_best_potential = {}
        self.last_reward_terms = {}
        self.last_selected_target = self._empty_target_selection()
        self.material_height = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.material = 1.0
        self.last_info = {}
        self._stage_nominal_params = {}
        self._reset_stats()
        self._cache_material_geoms()
        self._cache_steam_sites()
        self.configure_curriculum("single_easy")

    def refresh_observation_space(self):
        if int(self.attention_steam_dim) != 8:
            raise ValueError("attention_steam_dim must be 8 for the current steam feature schema")
        self.base_obs_dim = 35
        if self.spawn_history_observation_enabled:
            self.base_obs_dim += self.spawn_history_obs_dim
        if self.thermal_context_observation_enabled:
            self.base_obs_dim += self.thermal_context_obs_dim
        self.obs_dim = self.base_obs_dim
        if self.steam_attention_observation_enabled:
            self.obs_dim += self.attention_steam_count * self.attention_steam_dim
        if self.material_map_observation_enabled:
            self.obs_dim += self.material_map_channels * self.grid_size * self.grid_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def configure_curriculum(self, stage):
        if stage == "single_easy":
            self.max_steams = 1
            self.max_steam_age = 360
            self.spawn_probability = 0.0
            self.spawn_cooldown_steps = 999999
            self.initial_spawn_count = 1
            self.target_success_count = 1
            self.target_coverage = 1.0
            self.cover_radius = 0.28
            self.time_penalty = 0.02
            self.cover_reward = 75.0
            self.quick_cover_bonus = 20.0
            self.potential_gain = 10.0
            self.spawn_burst_probability = 0.0
            self.spawn_burst_min = 2
            self.spawn_burst_max = 3

        elif stage == "single_precision":
            self.max_steams = 1
            self.max_steam_age = 320
            self.spawn_probability = 0.0
            self.spawn_cooldown_steps = 999999
            self.initial_spawn_count = 1
            self.target_success_count = 2
            self.target_coverage = 1.0
            self.cover_radius = 0.2
            self.time_penalty = 0.025
            self.cover_reward = 72.0
            self.quick_cover_bonus = 18.0
            self.potential_gain = 9.0
            self.spawn_burst_probability = 0.0
            self.spawn_burst_min = 2
            self.spawn_burst_max = 3

        elif stage == "multi_low":
            self.max_steams = 3
            self.max_steam_age = 260
            self.spawn_probability = 0.006
            self.spawn_cooldown_steps = 80
            self.initial_spawn_count = 1
            self.target_success_count = 3
            self.target_coverage = 0.8
            self.cover_radius = 0.17
            self.time_penalty = 0.03
            self.cover_reward = 68.0
            self.quick_cover_bonus = 16.0
            self.potential_gain = 8.0
            self.spawn_burst_probability = 0.0
            self.spawn_burst_min = 2
            self.spawn_burst_max = 3

        elif stage == "multi_realistic":
            self.max_steams = 4
            self.max_steam_age = 220
            self.spawn_probability = 0.01
            self.spawn_cooldown_steps = 65
            self.initial_spawn_count = 1
            self.target_success_count = 4
            self.target_coverage = 0.8
            self.cover_radius = 0.14
            self.time_penalty = 0.035
            self.cover_reward = 64.0
            self.quick_cover_bonus = 14.0
            self.potential_gain = 7.5
            self.spawn_burst_probability = 0.0
            self.spawn_burst_min = 2
            self.spawn_burst_max = 3

        elif stage == "multi_hard":
            self.max_steams = 6
            self.max_steam_age = 200
            self.spawn_probability = 0.018
            self.spawn_cooldown_steps = 48
            self.initial_spawn_count = 2
            self.target_success_count = 6
            self.target_coverage = 0.82
            self.cover_radius = 0.12
            self.time_penalty = 0.045
            self.cover_reward = 62.0
            self.quick_cover_bonus = 12.0
            self.potential_gain = 6.5
            self.spawn_burst_probability = 0.18
            self.spawn_burst_min = 2
            self.spawn_burst_max = 3

        elif stage == "multi_extreme":
            self.max_steams = 8
            self.max_steam_age = 180
            self.spawn_probability = 0.026
            self.spawn_cooldown_steps = 36
            self.initial_spawn_count = 3
            self.target_success_count = 8
            self.target_coverage = 0.82
            self.cover_radius = 0.10
            self.time_penalty = 0.055
            self.cover_reward = 60.0
            self.quick_cover_bonus = 10.0
            self.potential_gain = 5.8
            self.spawn_burst_probability = 0.30
            self.spawn_burst_min = 2
            self.spawn_burst_max = 4

        else:
            raise ValueError(f"Unknown curriculum stage: {stage}")
        self.curriculum_stage = stage
        self._save_stage_nominal_params()

    def _save_stage_nominal_params(self):
        self._stage_nominal_params = {
            "track_step_size": float(self.track_step_size),
            "max_correction": float(self.max_correction),
            "cover_radius": float(self.cover_radius),
            "deposit_rate": float(self.deposit_rate),
            "cover_deposit_rate": float(self.cover_deposit_rate),
            "spawn_probability": float(self.spawn_probability),
        }

    def _restore_stage_nominal_params(self):
        for key, value in self._stage_nominal_params.items():
            setattr(self, key, value)

    def _apply_domain_randomization(self):
        self._restore_stage_nominal_params()
        if not self.domain_randomization_enabled:
            return
        scale = float(max(self.domain_randomization_scale, 0.0))
        if scale <= 0.0:
            return
        for key in ("track_step_size", "max_correction", "deposit_rate", "cover_deposit_rate"):
            nominal = float(self._stage_nominal_params.get(key, getattr(self, key)))
            factor = float(self.rng.uniform(1.0 - scale, 1.0 + scale))
            setattr(self, key, max(nominal * factor, 1e-6))
        radius_nominal = float(self._stage_nominal_params.get("cover_radius", self.cover_radius))
        radius_factor = float(self.rng.uniform(1.0 - 0.5 * scale, 1.0 + 0.5 * scale))
        self.cover_radius = max(radius_nominal * radius_factor, 0.04)

    def _setup_material_grid(self):
        half_width = self.pot_radius - 0.08
        coords = np.linspace(-half_width, half_width, self.grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords, indexing="ij")
        self.grid_local_xy = np.stack([xx, yy], axis=-1).astype(np.float32)
        self.grid_world_xy = self.grid_local_xy + self.pot_origin[:2]
        self.grid_mask = (np.linalg.norm(self.grid_local_xy, axis=-1) <= half_width).astype(np.float32)
        self.cell_half_size = float((coords[1] - coords[0]) * 0.42) if self.grid_size > 1 else 0.05

    def _cache_material_geoms(self):
        self.material_geom_ids = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                try:
                    row.append(self.model.geom(f"mat_cell_{i}_{j}").id)
                except KeyError:
                    row.append(-1)
            self.material_geom_ids.append(row)

    def _cache_steam_sites(self):
        self.steam_site_ids = []
        for i in range(16):
            try:
                self.steam_site_ids.append(self.model.site(f"s{i}").id)
            except KeyError:
                break

    def _reset_stats(self):
        self.step_count = 0
        self.steps_since_spawn = self.spawn_cooldown_steps
        self.spawned_count = 0
        self.covered_count = 0
        self.missed_count = 0
        self.total_cover_latency = 0.0
        self.last_cover_latency = 0.0
        self.prev_layer_progress = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        self.home_qpos = np.array([0.0, -0.2, 0.8, -1.4, 0.2, 0.0], dtype=np.float32)
        self.target_qpos = self.home_qpos.copy()
        self.data.qpos[:6] = self.home_qpos
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = self.home_qpos
        mujoco.mj_forward(self.model, self.data)

        self.home_ee_pos = self.data.site_xpos[self.model.site("ee_site").id].copy().astype(np.float32)
        self.home_ee_mat = self.data.site_xmat[self.model.site("ee_site").id].reshape(3, 3).copy().astype(np.float32)
        self.target_ee_pos = self.home_ee_pos.copy()
        self.prev_ee_pos = self.home_ee_pos.copy()
        self.cover_center = self.target_ee_pos[:2].copy()
        self.prev_cover_center = self.cover_center.copy()
        self.prev_target_dist = None
        self.material_height.fill(0.0)
        self.steams = []
        self.steam_history.clear()
        self.recent_spawn_positions = deque(maxlen=max(int(self.thermal_recent_spawn_memory), 1))
        self.last_spawn_xy = None
        self.spawned_this_step_positions = []
        self.last_steam_center = None
        self.steam_trend = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(3, dtype=np.float32)
        self.action_delay_buffer.clear()
        for _ in range(max(int(self.action_delay_steps), 0)):
            self.action_delay_buffer.append(np.zeros(3, dtype=np.float32))
        self.last_ctrl = self.data.ctrl[:6].copy()
        self.material = 1.0
        self.next_steam_id = 0
        self.steam_best_potential = {}
        self.last_reward_terms = {}
        self._reset_stats()
        self._apply_domain_randomization()
        self._initialize_thermal_hotspots()

        for _ in range(self.initial_spawn_count):
            self._spawn_steam_from_material()
        self.spawned_this_step_positions = []
        self._update_steam_trend()
        self._sync_steam_sites()
        self._update_material_visualization()
        self._hide_prediction_marker()
        self.set_cover_marker()

        obs = self._get_obs()
        self.last_info = self._make_info(0.0, False)
        return obs, self.last_info.copy()

    def step(self, action):
        self.step_count += 1
        self.spawned_this_step_positions = []
        raw_action = self._preprocess_action(action)
        prev_action = self.last_action.copy()
        smooth_action = self.action_smoothing * prev_action + (1.0 - self.action_smoothing) * raw_action
        self.last_action = smooth_action.copy()

        self._update_steam_motion()
        missed_now = self._remove_expired_steams()
        self._update_thermal_hotspots()
        self.steps_since_spawn += 1
        self.prev_cover_center = self.cover_center.copy()
        prev_target_dist, _, _ = self._target_steam_metrics(self.cover_center)
        prev_material_quality_loss = self.material_quality_loss

        # Apply action
        move_xy = smooth_action[:2]
        norm = np.linalg.norm(move_xy)
        if norm > 1.0:
            move_xy = move_xy / norm
        speed_scale = 0.65 + 0.35 * ((smooth_action[2] + 1.0) * 0.5)
        self.target_ee_pos[:2] += move_xy * self.track_step_size * speed_scale
        self._constrain_target_xy()
        self.target_ee_pos[2] = self.home_ee_pos[2]
        self.cover_center = self.target_ee_pos[:2].astype(np.float32)

        # Apply physics
        ee_pos = self.data.site_xpos[self.model.site("ee_site").id].copy()
        correction = np.clip((self.target_ee_pos - ee_pos) * self.return_gain, -self.max_correction, self.max_correction)
        correction[2] = np.clip(self.home_ee_pos[2] - ee_pos[2], -self.max_correction, self.max_correction)
        self.data.qfrc_applied[:] = self.data.qfrc_bias
        self._apply_cartesian_correction(correction, self._orientation_error_vector())
        mujoco.mj_step(self.model, self.data)

        # Get new state
        ee_pos_new = self.data.site_xpos[self.model.site("ee_site").id].copy()
        self.cover_center = ee_pos_new[:2].astype(np.float32)
        self._deposit_material(self.cover_center, self.deposit_rate)

        # Compute current distance to target and task reward.
        curr_target_dist, _, _ = self._target_steam_metrics(self.cover_center)
        action_delta = float(np.linalg.norm(smooth_action - prev_action))
        reward, covered = self._compute_reward(
            prev_target_dist=prev_target_dist,
            curr_target_dist=curr_target_dist,
            action=smooth_action,
            action_delta=action_delta,
            missed_now=missed_now,
            prev_material_quality_loss=prev_material_quality_loss,
        )

        terminated = self.covered_count >= self.target_success_count and self.coverage_rate >= self.target_coverage

        # Spawn new steam
        should_force_spawn = len(self.steams) == 0
        should_random_spawn = (
            len(self.steams) < self.max_steams
            and self.steps_since_spawn >= self.spawn_cooldown_steps
            and self.rng.random() < self.spawn_probability
        )
        if not terminated and (should_force_spawn or should_random_spawn):
            spawn_count = 1
            if should_random_spawn and self.rng.random() < self.spawn_burst_probability:
                spawn_count = int(self.rng.integers(self.spawn_burst_min, self.spawn_burst_max + 1))
            for _ in range(spawn_count):
                if len(self.steams) >= self.max_steams:
                    break
                self._spawn_steam_from_material()

        self._update_steam_trend()
        self._update_material()
        self._sync_steam_sites()
        self._update_material_visualization()
        self.set_cover_marker()

        truncated = self.step_count >= self.max_episode_steps
        obs = self._get_obs()

        ee_velocity = float(np.linalg.norm(ee_pos_new - self.prev_ee_pos) / max(self.model.opt.timestep, 1e-6))
        self.prev_ee_pos = ee_pos_new.copy()

        self.last_info = self._make_info(
            ee_velocity,
            covered,
            curr_target_dist if curr_target_dist is not None else 0.0,
        )
        return obs, float(reward), terminated, truncated, self.last_info.copy()

    def _preprocess_action(self, action):
        raw_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delay_steps = max(int(self.action_delay_steps), 0)
        if delay_steps > 0:
            self.action_delay_buffer.append(raw_action.copy())
            raw_action = self.action_delay_buffer.popleft()
        noise_std = float(max(self.action_noise_std, 0.0))
        if noise_std > 0.0:
            raw_action = raw_action + self.rng.normal(0.0, noise_std, size=raw_action.shape).astype(np.float32)
        return np.clip(raw_action, -1.0, 1.0).astype(np.float32)

    @property
    def coverage_rate(self):
        if self.spawned_count == 0:
            return 0.0
        return self.covered_count / self.spawned_count

    @property
    def average_cover_latency(self):
        if self.covered_count == 0:
            return 0.0
        return self.total_cover_latency / self.covered_count

    @property
    def valid_material(self):
        return self.material_height[self.grid_mask > 0]

    @property
    def mean_material_height(self):
        valid = self.valid_material
        return float(valid.mean()) if valid.size else 0.0

    @property
    def max_material_height(self):
        valid = self.valid_material
        return float(valid.max()) if valid.size else 0.0

    @property
    def height_uniformity(self):
        valid = self.valid_material
        if valid.size == 0:
            return 0.0
        return float(valid.std() / (self.target_layer_height + 1e-8))

    @property
    def layer_progress(self):
        return float(np.clip(self.mean_material_height / self.target_layer_height, 0.0, 1.5))

    @property
    def overfill_penalty(self):
        valid = self.valid_material
        if valid.size == 0:
            return 0.0
        return float(np.maximum(valid - self.target_layer_height, 0.0).mean() / (self.target_layer_height + 1e-8))

    @property
    def material_hole_loss(self):
        valid = self.valid_material
        if valid.size == 0:
            return 0.0
        gap = np.maximum(self.target_layer_height - valid, 0.0) / (self.target_layer_height + 1e-8)
        return float(gap.mean())

    @property
    def material_tv_loss(self):
        h = self.material_height / (self.target_layer_height + 1e-8)
        mask = self.grid_mask > 0
        dx_mask = mask[1:, :] & mask[:-1, :]
        dy_mask = mask[:, 1:] & mask[:, :-1]
        values = []
        if np.any(dx_mask):
            values.append(np.abs(h[1:, :] - h[:-1, :])[dx_mask])
        if np.any(dy_mask):
            values.append(np.abs(h[:, 1:] - h[:, :-1])[dy_mask])
        if not values:
            return 0.0
        return float(np.concatenate(values).mean())

    @property
    def material_quality_loss(self):
        return float(
            self.material_hole_weight * self.material_hole_loss
            + self.material_tv_weight * self.material_tv_loss
            + self.material_overfill_weight * self.overfill_penalty
        )

    def set_target_selector(self, selector):
        if selector not in ("nearest", "risk_aware"):
            raise ValueError(f"Unknown target selector: {selector}")
        self.target_selector = selector

    def _empty_target_selection(self):
        return {
            "target_selector": self.target_selector,
            "selected_target_id": -1,
            "selected_target_x": 0.0,
            "selected_target_y": 0.0,
            "selected_target_distance": 0.0,
            "nearest_target_distance": 0.0,
            "selected_target_age_score": 0.0,
            "selected_target_distance_score": 0.0,
            "selected_target_material_score": 0.0,
            "selected_target_reachability_score": 0.0,
            "selected_target_thermal_score": 0.0,
            "selected_target_risk_score": 0.0,
        }

    def _steam_risk_components(self, steam, center_xy=None):
        center_xy = self.cover_center if center_xy is None else np.asarray(center_xy, dtype=np.float32)
        cell_xy = steam["pos"][:2]
        dist = float(np.linalg.norm(cell_xy - center_xy))
        distance_score = float(1.0 - np.clip(dist / max(self.pot_radius, 1e-6), 0.0, 1.0))
        # Age is a persistent neglect signal. It does not imply timeout expiry.
        age_score = float(np.clip(steam["age"] / max(self.max_steam_age, 1), 0.0, 1.0))

        grid_dist = np.linalg.norm(self.grid_world_xy - cell_xy, axis=-1)
        local_gap = float(np.mean(
            np.maximum(self.target_layer_height - self.material_height, 0.0)
            * np.exp(-(grid_dist ** 2) / (2.0 * self.deposit_sigma ** 2))
        ))
        material_score = float(np.clip(local_gap / max(self.target_layer_height, 1e-6), 0.0, 1.0))
        thermal_score = float(np.clip(self._thermal_score_at_xy(cell_xy), 0.0, 1.0))

        reach_offset = float(np.linalg.norm(cell_xy - self.home_ee_pos[:2])) if self.home_ee_pos is not None else 0.0
        reachability_score = float(1.0 - np.clip(reach_offset / max(self.max_target_offset, 1e-6), 0.0, 1.0))

        risk_score = float(
            0.35 * age_score
            + 0.25 * distance_score
            + 0.15 * material_score
            + 0.10 * reachability_score
            + 0.15 * thermal_score
        )
        return {
            "selected_target_id": int(steam.get("id", -1)),
            "selected_target_x": float(cell_xy[0]),
            "selected_target_y": float(cell_xy[1]),
            "selected_target_distance": dist,
            "selected_target_age_score": age_score,
            "selected_target_distance_score": distance_score,
            "selected_target_material_score": material_score,
            "selected_target_reachability_score": reachability_score,
            "selected_target_thermal_score": thermal_score,
            "selected_target_risk_score": risk_score,
        }

    def select_nearest_steam(self, center_xy=None):
        if not self.steams:
            return None
        center_xy = self.cover_center if center_xy is None else np.asarray(center_xy, dtype=np.float32)
        return min(self.steams, key=lambda steam: np.linalg.norm(steam["pos"][:2] - center_xy))

    def select_risk_aware_steam(self, center_xy=None):
        if not self.steams:
            return None
        center_xy = self.cover_center if center_xy is None else np.asarray(center_xy, dtype=np.float32)
        return max(self.steams, key=lambda steam: self._steam_risk_components(steam, center_xy)["selected_target_risk_score"])

    def _target_steam_metrics(self, center_xy, selector=None):
        if not self.steams:
            info = self._empty_target_selection()
            return None, None, info

        selector = selector or self.target_selector
        center_xy = np.asarray(center_xy, dtype=np.float32)
        nearest = self.select_nearest_steam(center_xy)
        nearest_dist = float(np.linalg.norm(nearest["pos"][:2] - center_xy)) if nearest is not None else 0.0

        if selector == "nearest":
            target = nearest
        elif selector == "risk_aware":
            target = self.select_risk_aware_steam(center_xy)
        else:
            raise ValueError(f"Unknown target selector: {selector}")

        info = self._empty_target_selection()
        info.update(self._steam_risk_components(target, center_xy))
        info["target_selector"] = selector
        info["nearest_target_distance"] = nearest_dist
        return info["selected_target_distance"], target["pos"][:2].copy().astype(np.float32), info

    def _steam_attention_features(self):
        features = []
        # Sorted only for determinism; the attention model uses shared per-item
        # embeddings plus pooling, so it does not rely on this order.
        ranked_steams = sorted(
            self.steams,
            key=lambda steam: self._steam_risk_components(steam)["selected_target_risk_score"],
            reverse=True,
        )
        for i in range(self.attention_steam_count):
            if i < len(ranked_steams):
                steam = ranked_steams[i]
                rel_xy = (steam["pos"][:2] - self.cover_center) / self.pot_radius
                risk = self._steam_risk_components(steam)
                material_score = risk["selected_target_material_score"] if self.material_observation_enabled else 0.0
                risk_score = (
                    risk["selected_target_risk_score"]
                    if self.material_observation_enabled
                    else float(
                        0.40 * risk["selected_target_age_score"]
                        + 0.30 * risk["selected_target_distance_score"]
                        + 0.10 * risk["selected_target_reachability_score"]
                        + 0.20 * risk["selected_target_thermal_score"]
                    )
                )
                features.extend([
                    float(rel_xy[0]),
                    float(rel_xy[1]),
                    float(risk["selected_target_distance"] / max(self.pot_radius, 1e-6)),
                    float(risk["selected_target_age_score"]),
                    float(material_score),
                    float(risk["selected_target_reachability_score"]),
                    float(risk_score),
                    1.0,
                ])
            else:
                features.extend([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return np.asarray(features, dtype=np.float32)

    def _material_map_features(self):
        if not self.material_observation_enabled:
            return np.zeros(self.material_map_channels * self.grid_size * self.grid_size, dtype=np.float32)

        height = self.material_height / (self.target_layer_height + 1e-8)
        gap = np.clip(1.0 - height, 0.0, 1.5) * self.grid_mask
        overfill = np.clip(height - 1.0, 0.0, 1.0) * self.grid_mask

        padded = np.pad(height, 1, mode="edge")
        neighbor_mean = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) * 0.25
        frontier = np.clip(np.abs(height - neighbor_mean), 0.0, 1.5) * self.grid_mask

        material_map = np.stack([gap, overfill, frontier], axis=0).astype(np.float32)
        return material_map.reshape(-1)

    def _spawn_history_features(self):
        if not self.spawn_history_observation_enabled:
            return np.zeros(0, dtype=np.float32)

        if self.last_spawn_xy is None:
            last_rel = np.zeros(2, dtype=np.float32)
        else:
            last_rel = (np.asarray(self.last_spawn_xy, dtype=np.float32) - self.cover_center) / self.pot_radius

        recent_count = len(self.recent_spawn_positions)
        if recent_count:
            recent = np.asarray(self.recent_spawn_positions, dtype=np.float32)
            centroid = recent.mean(axis=0)
            centroid_rel = (centroid - self.cover_center) / self.pot_radius
            spread = float(np.mean(np.linalg.norm(recent - centroid, axis=1)) / max(self.pot_radius, 1e-6))
        else:
            centroid_rel = np.zeros(2, dtype=np.float32)
            spread = 0.0

        cooldown = max(int(self.spawn_cooldown_steps), 1)
        recent_capacity = max(int(self.thermal_recent_spawn_memory), 1)
        return np.array([
            float(last_rel[0]),
            float(last_rel[1]),
            float(centroid_rel[0]),
            float(centroid_rel[1]),
            float(np.clip(spread, 0.0, 1.0)),
            float(self.steam_trend[0]),
            float(self.steam_trend[1]),
            float(np.clip(self.steps_since_spawn / cooldown, 0.0, 1.0)),
            float(np.clip(len(self.steams) / max(self.max_steams, 1), 0.0, 1.0)),
            float(np.clip(recent_count / recent_capacity, 0.0, 1.0)),
        ], dtype=np.float32)

    def _thermal_context_features(self, target_xy=None):
        if not self.thermal_context_observation_enabled:
            return np.zeros(0, dtype=np.float32)

        dominant_hotspot = self._dominant_thermal_hotspot()
        if dominant_hotspot is None:
            hotspot_rel = np.zeros(2, dtype=np.float32)
            hotspot_score = 0.0
        else:
            hotspot_xy = np.asarray(dominant_hotspot["xy"], dtype=np.float32)
            hotspot_rel = (hotspot_xy - self.cover_center) / max(self.pot_radius, 1e-6)
            hotspot_score = self._thermal_score_at_xy(hotspot_xy)

        cover_heat = self._thermal_score_at_xy(self.cover_center)
        target_heat = self._thermal_score_at_xy(target_xy) if target_xy is not None else 0.0
        cooldown = max(int(self.spawn_cooldown_steps), 1)
        spawn_ready = float(np.clip(self.steps_since_spawn / cooldown, 0.0, 1.0))
        active_room = float(np.clip(1.0 - len(self.steams) / max(self.max_steams, 1), 0.0, 1.0))
        spawn_pressure = float(np.clip(self.spawn_probability * cooldown * spawn_ready * active_room, 0.0, 1.0))
        if self.spawn_burst_max >= self.spawn_burst_min:
            burst_mean = 0.5 * (self.spawn_burst_min + self.spawn_burst_max)
        else:
            burst_mean = 1.0
        burst_pressure = float(np.clip(self.spawn_burst_probability * max(burst_mean - 1.0, 0.0), 0.0, 1.0))

        return np.array([
            float(hotspot_rel[0]),
            float(hotspot_rel[1]),
            float(np.clip(hotspot_score, 0.0, 1.0)),
            float(np.clip(cover_heat, 0.0, 1.0)),
            float(np.clip(target_heat, 0.0, 1.0)),
            spawn_ready,
            spawn_pressure,
            burst_pressure,
        ], dtype=np.float32)

    def _get_obs(self):
        """
        精简观测，只保留核心信息：
        - 末端位置误差 (3)
        - 选中蒸汽相对方向 (2) + 距离 (1)
        - 朝选中蒸汽速度 (1)
        - 选中蒸汽风险分量 (5)
        - 其他蒸汽相对方向 (2*2) + 距离 (2)
        - 覆盖率 (1)
        - 动作 (3)
        - 关节位置 (6)
        - 材料状态 (4)
        """
        qpos = self.data.qpos[:6].copy()
        ee_pos = self.data.site_xpos[self.model.site("ee_site").id].copy()
        target_error = ee_pos - self.target_ee_pos

        target_dist, target_xy, target_info = self._target_steam_metrics(self.cover_center)
        self.last_selected_target = target_info

        # 目标方向和距离
        target_dir = np.zeros(2, dtype=np.float32)
        target_dist_norm = 1.0  # 归一化距离，1.0表示最远
        if target_xy is not None:
            target_vec = target_xy - self.cover_center
            dist = np.linalg.norm(target_vec)
            if dist > 1e-6:
                target_dir = (target_vec / dist).astype(np.float32)
            target_dist_norm = float(dist / self.pot_radius)

        # 朝目标速度
        vel_toward_target = 0.0
        if target_xy is not None:
            move_vec = self.cover_center - self.prev_cover_center
            target_vec = target_xy - self.cover_center
            target_norm = np.linalg.norm(target_vec)
            if target_norm > 1e-6:
                vel_toward_target = float(np.dot(move_vec, target_vec / target_norm) / self.pot_radius)

        observed_material_score = target_info["selected_target_material_score"] if self.material_observation_enabled else 0.0
        observed_risk_score = (
            target_info["selected_target_risk_score"]
            if self.material_observation_enabled
            else float(
                0.40 * target_info["selected_target_age_score"]
                + 0.30 * target_info["selected_target_distance_score"]
                + 0.10 * target_info["selected_target_reachability_score"]
                + 0.20 * target_info["selected_target_thermal_score"]
            )
        )
        target_risk_feats = np.array([
            target_info["selected_target_age_score"],
            target_info["selected_target_distance_score"],
            observed_material_score,
            target_info["selected_target_reachability_score"],
            observed_risk_score,
        ], dtype=np.float32)

        # 所有蒸汽的相对位置（最多3个）
        steam_feats = []
        sorted_steams = sorted(self.steams, key=lambda s: np.linalg.norm(self.cover_center - s["pos"][:2]))
        for i in range(3):
            if i < len(sorted_steams):
                steam = sorted_steams[i]
                rel_xy = (steam["pos"][:2] - self.cover_center) / self.pot_radius
                dist = np.linalg.norm(steam["pos"][:2] - self.cover_center) / self.pot_radius
                steam_feats.extend([rel_xy[0], rel_xy[1], dist])
            else:
                steam_feats.extend([0.0, 0.0, 1.0])

        material_obs = np.array([
            self.mean_material_height / self.target_layer_height,
            self.max_material_height / self.target_layer_height,
            self.height_uniformity,
            self.overfill_penalty,
        ], dtype=np.float32)
        if not self.material_observation_enabled:
            material_obs.fill(0.0)

        base_obs = np.concatenate([
            target_error.astype(np.float32),  # 3
            target_dir,  # 2
            np.array([target_dist_norm, vel_toward_target], dtype=np.float32),  # 2
            target_risk_feats,  # 5
            np.asarray(steam_feats, dtype=np.float32),  # 9
            np.array([self.coverage_rate], dtype=np.float32),  # 1
            self.last_action.astype(np.float32),  # 3
            qpos,  # 6
            material_obs,  # 4
        ])
        if self.spawn_history_observation_enabled:
            base_obs = np.concatenate([base_obs, self._spawn_history_features()])
        if self.thermal_context_observation_enabled:
            base_obs = np.concatenate([base_obs, self._thermal_context_features(target_xy)])
        if self.steam_attention_observation_enabled:
            obs = np.concatenate([base_obs, self._steam_attention_features()])
        else:
            obs = base_obs
        if self.material_map_observation_enabled:
            obs = np.concatenate([obs, self._material_map_features()])
        return obs.astype(np.float32)

    def _constrain_target_xy(self):
        offset = self.target_ee_pos[:2] - self.pot_center[:2]
        offset_norm = np.linalg.norm(offset)
        if offset_norm > self.pot_radius - 0.08:
            self.target_ee_pos[:2] = self.pot_center[:2] + offset / (offset_norm + 1e-8) * (self.pot_radius - 0.08)
        home_offset = self.target_ee_pos[:2] - self.home_ee_pos[:2]
        home_offset_norm = np.linalg.norm(home_offset)
        if home_offset_norm > self.max_target_offset:
            self.target_ee_pos[:2] = self.home_ee_pos[:2] + home_offset / (home_offset_norm + 1e-8) * self.max_target_offset

    def _apply_cartesian_correction(self, delta_pos, delta_rot):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.model.site("ee_site").id)
        J = np.vstack((jacp[:, :6], jacr[:, :6]))
        delta_twist = np.zeros(6, dtype=np.float32)
        delta_twist[:3] = delta_pos
        delta_twist[3:] = np.clip(delta_rot * self.orientation_gain, -self.orientation_limit, self.orientation_limit)
        lambda_sq = 5e-2
        try:
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6))
        except np.linalg.LinAlgError:
            J_pinv = np.linalg.pinv(J)
        posture_delta = np.zeros(6, dtype=np.float32)
        posture_delta[3:] = (self.home_qpos[3:] - self.data.qpos[3:6]) * self.posture_return_gain
        nullspace = np.eye(6, dtype=np.float32) - J_pinv @ J
        next_ctrl = self.data.ctrl[:6] + J_pinv @ delta_twist + nullspace @ posture_delta
        next_ctrl = 0.85 * self.data.ctrl[:6] + 0.15 * next_ctrl
        ctrl_range = self.model.actuator_ctrlrange[:6]
        self.data.ctrl[:6] = np.clip(next_ctrl, ctrl_range[:, 0], ctrl_range[:, 1])

    def _deposit_material(self, xy, amount):
        xy = np.asarray(xy, dtype=np.float32)
        dist2 = np.sum((self.grid_world_xy - xy) ** 2, axis=-1)
        kernel = np.exp(-dist2 / (2.0 * self.deposit_sigma ** 2)) * self.grid_mask
        self.material_height += amount * kernel.astype(np.float32)
        self.material_height = np.clip(self.material_height, 0.0, self.max_layer_height)

    def _distance_potential(self, dist):
        if dist is None:
            return 0.0
        # 0 far away, 1 at/inside cover radius; quadratic near the target keeps
        # gradients useful without making a mid-distance reward plateau.
        reach = max(self.pot_radius - self.cover_radius, 1e-6)
        normalized = np.clip((dist - self.cover_radius) / reach, 0.0, 1.0)
        return float((1.0 - normalized) ** 2)

    def _compute_reward(
        self,
        prev_target_dist,
        curr_target_dist,
        action,
        action_delta,
        missed_now,
        prev_material_quality_loss=None,
    ):
        terms = {
            "time": -self.time_penalty,
            "active": -self.active_steam_penalty * len(self.steams),
            "age": 0.0,
            "potential": 0.0,
            "best_progress": 0.0,
            "material_tv": 0.0,
            "cover": 0.0,
            "miss": -self.miss_penalty * missed_now,
            "action": 0.0,
            "success": 0.0,
        }
        if self.steams:
            mean_age = float(np.mean([steam["age"] for steam in self.steams]))
            age_ratio = min(mean_age / max(self.max_steam_age, 1), 1.0)
            terms["age"] = -self.age_penalty_gain * age_ratio

        if self.action_penalty_enabled:
            terms["action"] = -self.action_delta_penalty_gain * action_delta - self.action_l2_penalty_gain * float(np.dot(action, action))

        if self.potential_shaping_enabled and prev_target_dist is not None and curr_target_dist is not None:
            prev_phi = self._distance_potential(prev_target_dist)
            curr_phi = self._distance_potential(curr_target_dist)
            delta_phi = np.clip(
                self.potential_gamma * curr_phi - prev_phi,
                -self.potential_delta_clip,
                self.potential_delta_clip,
            )
            terms["potential"] = self.potential_gain * float(delta_phi)

        covered = False
        remaining = []
        for steam in self.steams:
            dist = float(np.linalg.norm(steam["pos"][:2] - self.cover_center))
            phi = self._distance_potential(dist)
            steam_id = steam["id"]
            best_phi = self.steam_best_potential.get(steam_id, 0.0)
            if self.best_progress_enabled and phi > best_phi:
                terms["best_progress"] += self.best_progress_gain * (phi - best_phi)
                self.steam_best_potential[steam_id] = phi

            if dist <= self.cover_radius:
                covered = True
                self.covered_count += 1
                latency = float(steam["age"])
                self.last_cover_latency = latency
                self.total_cover_latency += latency
                quickness = 1.0 - min(latency / max(self.max_steam_age, 1), 1.0)
                precision = 1.0 - min(dist / max(self.cover_radius, 1e-6), 1.0)
                terms["cover"] += self.cover_reward + self.quick_cover_bonus * quickness + self.precision_bonus * precision
                self._deposit_material(steam["pos"][:2], self.cover_deposit_rate)
                self.steam_best_potential.pop(steam_id, None)
            else:
                remaining.append(steam)
        self.steams = remaining

        if self.covered_count >= self.target_success_count and self.coverage_rate >= self.target_coverage:
            terms["success"] = self.success_bonus

        if self.material_tv_reward_enabled and prev_material_quality_loss is not None:
            quality_delta = float(prev_material_quality_loss - self.material_quality_loss)
            quality_delta = float(np.clip(quality_delta, -self.material_quality_delta_clip, self.material_quality_delta_clip))
            terms["material_tv"] = self.material_tv_reward_gain * quality_delta

        self.last_reward_terms = {key: float(value) for key, value in terms.items()}
        return float(sum(terms.values())), covered

    def _reachable_grid_mask(self):
        mask = self.grid_mask.copy()
        if self.home_ee_pos is not None:
            reach_radius = max(0.05, self.max_target_offset - self.spawn_reach_margin)
            reach_dist = np.linalg.norm(self.grid_world_xy - self.home_ee_pos[:2], axis=-1)
            mask *= (reach_dist <= reach_radius).astype(np.float32)
        return mask.astype(np.float32)

    def _sample_thermal_center(self):
        mask = self._reachable_grid_mask()
        if float(mask.sum()) <= 0.0:
            xy = self.pot_center[:2].copy()
        else:
            probs = mask.reshape(-1) / float(mask.sum())
            flat_idx = int(self.rng.choice(mask.size, p=probs))
            i, j = np.unravel_index(flat_idx, mask.shape)
            xy = self.grid_world_xy[i, j] + self.rng.normal(0.0, self.cell_half_size * 0.8, size=2)
        return self._project_xy_into_spawn_workspace(xy)

    def _new_thermal_hotspot(self):
        lifetime = max(int(self.thermal_lifetime_steps), 1)
        return {
            "xy": self._sample_thermal_center(),
            "age": int(self.rng.integers(0, max(lifetime // 3, 1))),
            "lifetime": int(max(1, lifetime * self.rng.uniform(0.75, 1.25))),
            "amp": float(self.rng.uniform(0.75, 1.0)),
        }

    def _initialize_thermal_hotspots(self):
        self.thermal_hotspots = []
        if not self.thermal_spawn_enabled:
            return
        for _ in range(max(int(self.thermal_hotspot_count), 0)):
            self.thermal_hotspots.append(self._new_thermal_hotspot())

    def _update_thermal_hotspots(self):
        if not self.thermal_spawn_enabled:
            self.thermal_hotspots = []
            return

        target_count = max(int(self.thermal_hotspot_count), 0)
        while len(self.thermal_hotspots) < target_count:
            self.thermal_hotspots.append(self._new_thermal_hotspot())
        if len(self.thermal_hotspots) > target_count:
            self.thermal_hotspots = self.thermal_hotspots[:target_count]

        drift_std = max(float(self.thermal_drift_std), 0.0)
        refresh_probability = max(float(self.thermal_refresh_probability), 0.0)
        updated = []
        for hotspot in self.thermal_hotspots:
            hotspot["age"] = int(hotspot.get("age", 0)) + 1
            expired = hotspot["age"] >= int(hotspot.get("lifetime", self.thermal_lifetime_steps))
            refreshed = self.rng.random() < refresh_probability
            if expired or refreshed:
                updated.append(self._new_thermal_hotspot())
                continue
            if drift_std > 0.0:
                hotspot["xy"] = self._project_xy_into_spawn_workspace(
                    np.asarray(hotspot["xy"], dtype=np.float32)
                    + self.rng.normal(0.0, drift_std, size=2)
                )
            updated.append(hotspot)
        self.thermal_hotspots = updated

    def _thermal_field(self):
        if not self.thermal_spawn_enabled or not self.thermal_hotspots:
            return np.zeros_like(self.grid_mask, dtype=np.float32)
        sigma = max(float(self.thermal_hotspot_sigma), 1e-4)
        field = np.zeros_like(self.grid_mask, dtype=np.float32)
        for hotspot in self.thermal_hotspots:
            xy = np.asarray(hotspot["xy"], dtype=np.float32)
            dist2 = np.sum((self.grid_world_xy - xy) ** 2, axis=-1)
            field += float(hotspot.get("amp", 1.0)) * np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)
        field *= self.grid_mask
        peak = float(field.max())
        if peak > 1e-8:
            field /= peak
        return field.astype(np.float32)

    def _thermal_score_at_xy(self, xy):
        if not self.thermal_spawn_enabled or not self.thermal_hotspots:
            return 0.0
        xy = np.asarray(xy, dtype=np.float32)
        sigma = max(float(self.thermal_hotspot_sigma), 1e-4)
        score = 0.0
        for hotspot in self.thermal_hotspots:
            dist2 = float(np.sum((xy - np.asarray(hotspot["xy"], dtype=np.float32)) ** 2))
            score += float(hotspot.get("amp", 1.0)) * float(np.exp(-dist2 / (2.0 * sigma ** 2)))
        return float(np.clip(score, 0.0, 1.0))

    def _recent_spawn_suppression_field(self):
        if not self.recent_spawn_positions:
            return np.ones_like(self.grid_mask, dtype=np.float32)
        radius = max(float(self.thermal_recent_spawn_radius), 1e-4)
        suppression = float(np.clip(self.thermal_recent_spawn_suppression, 0.0, 0.95))
        field = np.ones_like(self.grid_mask, dtype=np.float32)
        for xy in self.recent_spawn_positions:
            dist2 = np.sum((self.grid_world_xy - np.asarray(xy, dtype=np.float32)) ** 2, axis=-1)
            field *= 1.0 - suppression * np.exp(-dist2 / (2.0 * radius ** 2)).astype(np.float32)
        return np.clip(field, 0.05, 1.0).astype(np.float32)

    def _dominant_thermal_hotspot(self):
        if not self.thermal_hotspots:
            return None
        return max(self.thermal_hotspots, key=lambda hotspot: float(hotspot.get("amp", 1.0)))

    def _spawn_steam_from_material(self):
        scores = self._steam_spawn_scores()
        total = float(scores.sum())
        if total <= 1e-8:
            return
        flat_idx = int(self.rng.choice(scores.size, p=(scores.reshape(-1) / total)))
        i, j = np.unravel_index(flat_idx, scores.shape)
        xy = self.grid_world_xy[i, j] + self.rng.normal(0.0, self.cell_half_size * 0.6, size=2)
        xy = self._project_xy_into_spawn_workspace(xy)
        vel = self.rng.normal(0.0, self.steam_speed * 0.4, size=2)
        vel = self._clip_velocity(vel)
        steam_id = self.next_steam_id
        self.next_steam_id += 1
        dist = float(np.linalg.norm(xy - self.cover_center))
        self.steam_best_potential[steam_id] = self._distance_potential(dist)
        self.steams.append({
            "id": steam_id,
            "pos": np.array([xy[0], xy[1], 0.1], dtype=np.float32),
            "age": 0.0,
            "vel": vel.astype(np.float32),
        })
        self.spawned_count += 1
        self.last_spawn_xy = xy.astype(np.float32)
        self.spawned_this_step_positions.append(xy.astype(np.float32))
        self.recent_spawn_positions.append(xy.astype(np.float32))
        self.steps_since_spawn = 0

    def _steam_spawn_scores(self):
        h = self.material_height
        low_layer = np.clip(1.0 - h / (self.target_layer_height + 1e-8), 0.0, 1.0)
        padded = np.pad(h, 1, mode="edge")
        neighbor_mean = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) * 0.25
        local_gap = np.abs(h - neighbor_mean) / (self.target_layer_height + 1e-8)
        scores = (0.15 + low_layer + 0.35 * local_gap) * self.grid_mask
        if self.thermal_spawn_enabled:
            heat = self._thermal_field()
            background = float(np.clip(self.thermal_background_weight, 0.02, 1.0))
            strength = max(float(self.thermal_hotspot_strength), 0.0)
            scores *= background + strength * heat
        if self.home_ee_pos is not None:
            reach_radius = max(0.05, self.max_target_offset - self.spawn_reach_margin)
            reach_dist = np.linalg.norm(self.grid_world_xy - self.home_ee_pos[:2], axis=-1)
            scores *= (reach_dist <= reach_radius).astype(np.float32)
        for steam in self.steams:
            dist2 = np.sum((self.grid_world_xy - steam["pos"][:2]) ** 2, axis=-1)
            scores *= 1.0 - 0.75 * np.exp(-dist2 / (2.0 * 0.12 ** 2))
        if self.thermal_spawn_enabled:
            scores *= self._recent_spawn_suppression_field()
        return np.maximum(scores, 0.0).astype(np.float32)

    def _update_steam_motion(self):
        for steam in self.steams:
            steam["age"] += 1.0
            steam["vel"][:] = 0.0

    def _remove_expired_steams(self):
        # In the target task, steam points do not disappear automatically. They
        # remain active until the cover marker reaches them. Keeping this method
        # makes the step loop compatible with older experiments while disabling
        # timeout-based misses by default.
        if not self.steam_timeout_enabled:
            return 0
        if not self.steams:
            return 0
        alive = []
        missed = 0
        for steam in self.steams:
            if steam["age"] > self.max_steam_age:
                missed += 1
                self.missed_count += 1
                self.steam_best_potential.pop(steam.get("id"), None)
            else:
                alive.append(steam)
        self.steams = alive
        return missed

    def _nearest_steam_metrics(self, center_xy):
        if not self.steams:
            return None, None
        center_xy = np.asarray(center_xy, dtype=np.float32)
        distances = [float(np.linalg.norm(steam["pos"][:2] - center_xy)) for steam in self.steams]
        idx = int(np.argmin(distances))
        return distances[idx], self.steams[idx]["pos"][:2].copy().astype(np.float32)

    def _orientation_error_vector(self):
        current_mat = self.data.site_xmat[self.model.site("ee_site").id].reshape(3, 3)
        return 0.5 * (
            np.cross(current_mat[:, 0], self.home_ee_mat[:, 0])
            + np.cross(current_mat[:, 1], self.home_ee_mat[:, 1])
            + np.cross(current_mat[:, 2], self.home_ee_mat[:, 2])
        ).astype(np.float32)

    def _orientation_error_angle(self):
        current_mat = self.data.site_xmat[self.model.site("ee_site").id].reshape(3, 3)
        rot_err = self.home_ee_mat @ current_mat.T
        trace_val = np.clip((np.trace(rot_err) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.arccos(trace_val))

    def _wrist_deviation(self):
        diff = self.data.qpos[3:6] - self.home_qpos[3:]
        weights = np.array([1.0, 1.2, 1.4], dtype=np.float32)
        return float(np.linalg.norm(diff * weights))

    def _project_xy_into_pot(self, xy):
        xy = np.asarray(xy, dtype=np.float32)
        offset = xy - self.pot_center[:2]
        dist = np.linalg.norm(offset)
        max_radius = self.pot_radius - 0.05
        if dist > max_radius:
            xy = self.pot_center[:2] + offset / (dist + 1e-8) * max_radius
        return xy.astype(np.float32)

    def _project_xy_into_spawn_workspace(self, xy):
        xy = self._project_xy_into_pot(xy)
        if self.home_ee_pos is None:
            return xy
        reach_radius = max(0.05, self.max_target_offset - self.spawn_reach_margin)
        for _ in range(3):
            home_offset = xy - self.home_ee_pos[:2]
            home_dist = np.linalg.norm(home_offset)
            if home_dist > reach_radius:
                xy = self.home_ee_pos[:2] + home_offset / (home_dist + 1e-8) * reach_radius
            xy = self._project_xy_into_pot(xy)
        return xy.astype(np.float32)

    def _clip_velocity(self, vel):
        vel = np.asarray(vel, dtype=np.float32)
        speed = np.linalg.norm(vel)
        if speed > self.steam_speed:
            vel = vel / (speed + 1e-8) * self.steam_speed
        return vel.astype(np.float32)

    def _update_steam_trend(self):
        if self.steams:
            center = np.mean([steam["pos"][:2] for steam in self.steams], axis=0)
        else:
            center = self.pot_center[:2].copy()
        if self.last_steam_center is None:
            self.steam_trend = np.zeros(2, dtype=np.float32)
        else:
            self.steam_trend = ((center - self.last_steam_center) / self.pot_radius).astype(np.float32)
        self.last_steam_center = center.astype(np.float32)
        self.steam_history.append(center.astype(np.float32))

    def _sync_steam_sites(self):
        for i, site_id in enumerate(self.steam_site_ids):
            if i < len(self.steams):
                pos = self.steams[i]["pos"]
                self.model.site_pos[site_id] = [pos[0] - self.pot_origin[0], pos[1] - self.pot_origin[1], pos[2]]
                self.model.site_rgba[site_id] = [1.0, 0.0, 0.0, 1.0]
            else:
                self.model.site_rgba[site_id] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

    def _update_material(self):
        self.material = max(0.0, 1.0 - min(self.layer_progress, 1.0))

    def _update_material_visualization(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                geom_id = self.material_geom_ids[i][j]
                if geom_id < 0:
                    continue
                height = float(self.material_height[i, j]) if self.grid_mask[i, j] > 0 else 0.0
                local_xy = self.grid_local_xy[i, j]
                self.model.geom_pos[geom_id] = [local_xy[0], local_xy[1], self.material_base_z + height]
                radius = self.cell_half_size * (0.12 + 0.18 * np.clip(height / self.target_layer_height, 0.0, 1.0))
                self.model.geom_size[geom_id] = [max(radius, 0.004), 0.0, 0.0]
                alpha = float(np.clip(0.05 + height / self.target_layer_height, 0.0, 0.65))
                self.model.geom_rgba[geom_id] = [0.55, 0.32, 0.12, alpha]
        mujoco.mj_forward(self.model, self.data)

    def _hide_prediction_marker(self):
        """隐藏蓝色预测球"""
        try:
            site_id = self.model.site("pred_site").id
            self.model.site_rgba[site_id] = [0.0, 0.6, 1.0, 0.0]  # 完全透明
        except KeyError:
            pass

    def set_prediction_marker(self, pred_xy):
        """不显示预测标记"""
        self._hide_prediction_marker()

    def set_cover_marker(self):
        try:
            site_id = self.model.site("spray_site").id
        except KeyError:
            return
        self.model.site_pos[site_id] = [self.cover_center[0] - self.pot_origin[0], self.cover_center[1] - self.pot_origin[1], 0.11]
        self.model.site_rgba[site_id] = [0.1, 1.0, 0.2, 0.7]
        mujoco.mj_forward(self.model, self.data)

    def normalize_xy(self, xy):
        return ((np.asarray(xy, dtype=np.float32) - self.pot_center[:2]) / self.pot_radius).astype(np.float32)

    def denormalize_prediction(self, pred_xy):
        pred_xy = np.clip(np.asarray(pred_xy, dtype=np.float32), -1.0, 1.0)
        return (self.pot_center[:2] + pred_xy * self.pot_radius).astype(np.float32)

    def _make_info(
        self,
        ee_velocity=0.0,
        covered=False,
        target_distance=None,
    ):
        selected_target = self.last_selected_target.copy()
        if target_distance is None:
            target_distance = selected_target["selected_target_distance"]
        spawned_count = len(self.spawned_this_step_positions)
        if spawned_count:
            spawned_xy = np.mean(self.spawned_this_step_positions, axis=0).astype(np.float32)
            pred_target = self.normalize_xy(spawned_xy)
            pred_mask = 1.0
        else:
            spawned_xy = np.zeros(2, dtype=np.float32)
            pred_target = np.zeros(2, dtype=np.float32)
            pred_mask = 0.0
        dominant_hotspot = self._dominant_thermal_hotspot()
        if dominant_hotspot is None:
            thermal_xy = np.zeros(2, dtype=np.float32)
            thermal_peak_score = 0.0
        else:
            thermal_xy = np.asarray(dominant_hotspot["xy"], dtype=np.float32)
            thermal_peak_score = self._thermal_score_at_xy(thermal_xy)
        return {
            "coverage_rate": float(self.coverage_rate),
            "cover_latency": float(self.average_cover_latency),
            "last_cover_latency": float(self.last_cover_latency),
            "ee_velocity": float(ee_velocity),
            "target_distance": float(target_distance),
            "target_selector": selected_target["target_selector"],
            "selected_target_id": int(selected_target["selected_target_id"]),
            "selected_target_x": float(selected_target["selected_target_x"]),
            "selected_target_y": float(selected_target["selected_target_y"]),
            "selected_target_distance": float(selected_target["selected_target_distance"]),
            "nearest_target_distance": float(selected_target["nearest_target_distance"]),
            "selected_target_age_score": float(selected_target["selected_target_age_score"]),
            "selected_target_distance_score": float(selected_target["selected_target_distance_score"]),
            "selected_target_material_score": float(selected_target["selected_target_material_score"]),
            "selected_target_reachability_score": float(selected_target["selected_target_reachability_score"]),
            "selected_target_thermal_score": float(selected_target["selected_target_thermal_score"]),
            "selected_target_risk_score": float(selected_target["selected_target_risk_score"]),
            "spray_radius": float(self.cover_radius),
            "mean_material_height": float(self.mean_material_height),
            "max_material_height": float(self.max_material_height),
            "height_uniformity": float(self.height_uniformity),
            "layer_progress": float(self.layer_progress),
            "overfill_penalty": float(self.overfill_penalty),
            "material_hole_loss": float(self.material_hole_loss),
            "material_tv_loss": float(self.material_tv_loss),
            "material_quality_loss": float(self.material_quality_loss),
            "success_count": int(self.covered_count),
            "missed_count": int(self.missed_count),
            "spawned_count": int(self.spawned_count),
            "steam_count": int(len(self.steams)),
            "steam_timeout_enabled": bool(self.steam_timeout_enabled),
            "spawn_history_observation_enabled": bool(self.spawn_history_observation_enabled),
            "thermal_context_observation_enabled": bool(self.thermal_context_observation_enabled),
            "steam_attention_observation_enabled": bool(self.steam_attention_observation_enabled),
            "material_map_observation_enabled": bool(self.material_map_observation_enabled),
            "potential_shaping_enabled": bool(self.potential_shaping_enabled),
            "best_progress_enabled": bool(self.best_progress_enabled),
            "material_observation_enabled": bool(self.material_observation_enabled),
            "material_tv_reward_enabled": bool(self.material_tv_reward_enabled),
            "material_tv_reward_gain": float(self.material_tv_reward_gain),
            "action_penalty_enabled": bool(self.action_penalty_enabled),
            "action_delay_steps": int(self.action_delay_steps),
            "action_noise_std": float(self.action_noise_std),
            "domain_randomization_enabled": bool(self.domain_randomization_enabled),
            "domain_randomization_scale": float(self.domain_randomization_scale),
            "spawn_burst_probability": float(self.spawn_burst_probability),
            "thermal_spawn_enabled": bool(self.thermal_spawn_enabled),
            "thermal_hotspot_count": int(len(self.thermal_hotspots)),
            "thermal_background_weight": float(self.thermal_background_weight),
            "thermal_hotspot_strength": float(self.thermal_hotspot_strength),
            "thermal_peak_x": float(thermal_xy[0]),
            "thermal_peak_y": float(thermal_xy[1]),
            "thermal_peak_score": float(thermal_peak_score),
            "spawned_this_step": int(spawned_count),
            "spawned_this_step_x": float(spawned_xy[0]),
            "spawned_this_step_y": float(spawned_xy[1]),
            "material": float(self.material),
            "covered": bool(covered),
            "reward_terms": self.last_reward_terms.copy(),
            "pred_target": pred_target,
            "pred_mask": pred_mask,
        }
