import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


def resolve_device(device=None):
    requested = "auto" if device is None else str(device)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({requested}) but torch.cuda.is_available() is false")
    return torch.device(requested)


class ACModel_LSTM(nn.Module):
    """
    Actor-Critic model with LSTM for temporal reasoning.
    
    Improvements:
    - Wider network (256 -> 256) with better initialization
    - Separate feature extractors for different obs components
    - Better initial log_std for exploration
    """

    def __init__(self, s_dim, a_dim, hidden_dim=256, pred_dim=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, a_dim)
        # Start with high std for exploration
        self.log_std = nn.Parameter(torch.full((1, a_dim), -0.8))
        self.v = nn.Linear(hidden_dim, 1)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pred_dim),
            nn.Tanh(),
        )

        # Better weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Smaller gain for policy output
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)

    def forward(self, s, hidden=None):
        x = self.feat(s)
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(x, hidden)
        else:
            lstm_out, hidden_out = self.lstm(x)

        mu = torch.tanh(self.mu(lstm_out))
        std = torch.exp(self.log_std).expand_as(mu) + 1e-5
        val = self.v(lstm_out)
        pred_xy = self.pred(lstm_out)
        return mu, std, val, pred_xy, hidden_out


class ACModel_MLP(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim=256, pred_dim=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, a_dim)
        self.log_std = nn.Parameter(torch.full((1, a_dim), -0.8))
        self.v = nn.Linear(hidden_dim, 1)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pred_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)

    def forward(self, s, hidden=None):
        x = self.feat(s)
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std).expand_as(mu) + 1e-5
        val = self.v(x)
        pred_xy = self.pred(x)
        return mu, std, val, pred_xy, (None, None)


class ACModel_AttentionLSTM(nn.Module):
    def __init__(
        self,
        s_dim,
        a_dim,
        hidden_dim=256,
        pred_dim=2,
        base_obs_dim=35,
        steam_count=6,
        steam_dim=8,
        material_map_channels=0,
        material_map_size=9,
    ):
        super().__init__()
        self.base_obs_dim = int(base_obs_dim)
        self.steam_count = int(steam_count)
        self.steam_dim = int(steam_dim)
        self.material_map_channels = int(material_map_channels)
        self.material_map_size = int(material_map_size)
        self.material_map_dim = self.material_map_channels * self.material_map_size * self.material_map_size
        self.use_material_map = self.material_map_channels > 0
        expected_dim = self.base_obs_dim + self.steam_count * self.steam_dim + self.material_map_dim
        if s_dim != expected_dim:
            raise ValueError(f"Attention model expects obs dim {expected_dim}, got {s_dim}")

        steam_embed_dim = hidden_dim // 2
        self.base_feat = nn.Sequential(
            nn.Linear(self.base_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.steam_embed = nn.Sequential(
            nn.Linear(self.steam_dim, steam_embed_dim),
            nn.LayerNorm(steam_embed_dim),
            nn.ReLU(),
        )
        self.steam_attn = nn.MultiheadAttention(steam_embed_dim, num_heads=4, batch_first=True)
        self.steam_out = nn.Sequential(
            nn.Linear(steam_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        if self.use_material_map:
            self.material_map_encoder = nn.Sequential(
                nn.Conv2d(self.material_map_channels, 16, kernel_size=3, padding=1),
                nn.GroupNorm(4, 16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.GroupNorm(4, 32),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * self.material_map_size * self.material_map_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
        fuse_inputs = 3 if self.use_material_map else 2
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * fuse_inputs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, a_dim)
        self.log_std = nn.Parameter(torch.full((1, a_dim), -0.8))
        self.v = nn.Linear(hidden_dim, 1)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pred_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)

    def _encode(self, s):
        base = s[..., :self.base_obs_dim]
        steam_start = self.base_obs_dim
        steam_end = steam_start + self.steam_count * self.steam_dim
        steam_flat = s[..., steam_start:steam_end]
        steam = steam_flat.view(*s.shape[:-1], self.steam_count, self.steam_dim)
        steam_mask = steam[..., -1] <= 0.0

        batch_shape = steam.shape[:-2]
        steam_2d = steam.reshape(-1, self.steam_count, self.steam_dim)
        mask_2d = steam_mask.reshape(-1, self.steam_count)
        emb = self.steam_embed(steam_2d)
        all_padded = mask_2d.all(dim=1)
        safe_mask = mask_2d.clone()
        if all_padded.any():
            safe_mask[all_padded, 0] = False
        attn_out, _ = self.steam_attn(emb, emb, emb, key_padding_mask=safe_mask)
        valid = (~mask_2d).float().unsqueeze(-1)
        pooled = (attn_out * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pooled = pooled.view(*batch_shape, -1)

        base_feat = self.base_feat(base)
        steam_feat = self.steam_out(pooled)
        features = [base_feat, steam_feat]
        if self.use_material_map:
            map_flat = s[..., steam_end:]
            map_2d = map_flat.reshape(-1, self.material_map_channels, self.material_map_size, self.material_map_size)
            map_feat = self.material_map_encoder(map_2d).view(*batch_shape, -1)
            features.append(map_feat)
        return self.fuse(torch.cat(features, dim=-1))

    def forward(self, s, hidden=None):
        x = self._encode(s)
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(x, hidden)
        else:
            lstm_out, hidden_out = self.lstm(x)
        mu = torch.tanh(self.mu(lstm_out))
        std = torch.exp(self.log_std).expand_as(mu) + 1e-5
        val = self.v(lstm_out)
        pred_xy = self.pred(lstm_out)
        return mu, std, val, pred_xy, hidden_out


class PPOAgent:
    """
    PPO agent with LSTM for temporal reasoning.
    
    Improvements:
    - Higher entropy coefficient for exploration (0.002 -> 0.01)
    - Learning rate scheduling (warmup + decay)
    - Gradual entropy annealing
    - Better GAE computation
    """

    def __init__(
        self,
        s_dim,
        a_dim,
        seq_len=400,
        hidden_dim=256,
        use_lstm=True,
        use_steam_attention=False,
        use_material_map=False,
        base_obs_dim=35,
        attention_steam_count=6,
        attention_steam_dim=8,
        lr=1e-4,
        ppo_epochs=4,
        clip_param=0.12,
        value_clip_param=0.12,
        reward_scale=0.02,
        reward_clip=4.0,
        entropy_coef_start=0.003,
        entropy_coef_end=0.0005,
        entropy_decay_steps=240000,
        max_grad_norm=0.35,
        pred_coef=0.0,
        bc_supervised_coef=0.03,
        bc_supervised_min_coef=0.0,
        bc_supervised_decay_steps=240000,
        device="auto",
    ):
        self.device = resolve_device(device)
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.seq_len = seq_len
        self.use_steam_attention = bool(use_steam_attention)
        self.use_material_map = bool(use_material_map)
        self.use_lstm = bool(use_lstm) or self.use_steam_attention or self.use_material_map
        self.base_obs_dim = int(base_obs_dim)
        self.attention_steam_count = int(attention_steam_count)
        self.attention_steam_dim = int(attention_steam_dim)
        if self.use_material_map:
            self.use_steam_attention = True
        if self.use_steam_attention:
            material_map_channels = 3 if self.use_material_map else 0
            self.model = ACModel_AttentionLSTM(
                s_dim,
                a_dim,
                hidden_dim=hidden_dim,
                base_obs_dim=self.base_obs_dim,
                steam_count=self.attention_steam_count,
                steam_dim=self.attention_steam_dim,
                material_map_channels=material_map_channels,
            )
        else:
            model_cls = ACModel_LSTM if self.use_lstm else ACModel_MLP
            self.model = model_cls(s_dim, a_dim, hidden_dim=hidden_dim)
        self.model.to(self.device)

        # Learning rate with warmup capability
        self.base_lr = float(lr)
        self.opt = optim.Adam(self.model.parameters(), lr=self.base_lr, eps=1e-5)

        # PPO hyperparameters
        self.ppo_epochs = int(ppo_epochs)
        self.clip_param = float(clip_param)
        self.value_clip_param = float(value_clip_param)
        self.reward_scale = float(reward_scale)
        self.reward_clip = float(reward_clip)
        self.entropy_coef = float(entropy_coef_start)
        self.entropy_coef_start = float(entropy_coef_start)
        self.entropy_coef_end = float(entropy_coef_end)
        self.entropy_decay_steps = int(entropy_decay_steps)
        self.value_coef = 0.5
        self.max_grad_norm = float(max_grad_norm)

        # GAE parameters
        self.gamma = 0.985
        self.lamda = 0.95

        # Auxiliary loss coefficients
        self.pred_coef = float(pred_coef)
        self.smooth_coef = 0.02
        self.action_l2_coef = 0.002
        self.bc_supervised_coef_start = float(bc_supervised_coef)
        self.bc_supervised_min_coef = float(bc_supervised_min_coef)
        self.bc_supervised_decay_steps = int(bc_supervised_decay_steps)

        # ICM (disabled for now)
        self.icm_reward_coef = 0.0
        self.icm_loss_coef = 0.0
        self.icm = ICMModel(s_dim, a_dim, hidden_dim=hidden_dim).to(self.device)
        self.icm_opt = optim.Adam(self.icm.parameters(), lr=1e-4)

        # Training step counter
        self.train_steps = 0

    def _get_entropy_coef(self):
        """Anneal entropy coefficient over training."""
        progress = min(self.train_steps / self.entropy_decay_steps, 1.0)
        return self.entropy_coef_start + progress * (self.entropy_coef_end - self.entropy_coef_start)

    def _get_bc_supervised_coef(self):
        if self.bc_supervised_coef_start <= 0:
            return 0.0
        progress = min(self.train_steps / max(self.bc_supervised_decay_steps, 1), 1.0)
        decayed = self.bc_supervised_coef_start * (1.0 - progress)
        return max(self.bc_supervised_min_coef, decayed)

    def _forward_recurrent(self, states, reset_masks):
        if not self.use_lstm:
            return self.model(states, hidden=None)

        outputs = []
        values = []
        preds = []
        hx = None
        cx = None
        for t in range(states.size(1)):
            reset_t = reset_masks[:, t].view(1, -1, 1)
            if hx is not None and cx is not None:
                keep = 1.0 - reset_t
                hx = hx * keep
                cx = cx * keep
            mu_t, std_t, val_t, pred_t, (hx, cx) = self.model(states[:, t:t + 1], (hx, cx) if hx is not None else None)
            outputs.append((mu_t, std_t))
            values.append(val_t)
            preds.append(pred_t)
        mu = torch.cat([item[0] for item in outputs], dim=1)
        std = torch.cat([item[1] for item in outputs], dim=1)
        b_values = torch.cat(values, dim=1)
        pred_xy = torch.cat(preds, dim=1)
        return mu, std, b_values, pred_xy, (hx, cx)

    def select_action(self, s, hx=None, cx=None, deterministic=False):
        s = torch.tensor(s, dtype=torch.float32, device=self.device).view(1, 1, -1)
        with torch.no_grad():
            hidden_in = (hx, cx) if self.use_lstm and hx is not None and cx is not None else None
            mu, std, val, pred_xy, (hx_n, cx_n) = self.model(s, hidden_in)
            dist = Normal(mu, std)
            action = mu if deterministic else dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum().item()
        return (
            action.detach().cpu().numpy().flatten(),
            log_prob,
            val.item(),
            pred_xy.detach().cpu().numpy().flatten(),
            hx_n,
            cx_n,
        )

    def intrinsic_reward(self, s, s_next, a):
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).view(1, 1, -1)
        s_nt = torch.tensor(s_next, dtype=torch.float32, device=self.device).view(1, 1, -1)
        a_t = torch.tensor(a, dtype=torch.float32, device=self.device).view(1, 1, -1)
        with torch.no_grad():
            r_int, _, _ = self.icm(s_t, s_nt, a_t)
        return float(r_int.mean().item())

    def behavior_clone(self, states, actions, epochs=8, batch_size=256):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.size == 0:
            return {"bc_loss": 0.0}

        last_loss = 0.0
        n = states.shape[0]
        for _ in range(epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                s_t = torch.tensor(states[batch_idx], dtype=torch.float32, device=self.device).view(-1, 1, self.s_dim)
                a_t = torch.tensor(actions[batch_idx], dtype=torch.float32, device=self.device).view(-1, 1, self.a_dim)
                mu, _, _, _, _ = self.model(s_t, hidden=None)
                bc_loss = F.mse_loss(mu, a_t)

                self.opt.zero_grad()
                bc_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                last_loss = float(bc_loss.item())
        return {"bc_loss": last_loss}

    def update(self, memory):
        usable_steps = (len(memory["s"]) // self.seq_len) * self.seq_len
        if usable_steps == 0:
            return {}

        num_episodes = usable_steps // self.seq_len
        states = torch.tensor(np.array(memory["s"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, -1)
        next_states = torch.tensor(np.array(memory["s_next"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, -1)
        actions = torch.tensor(np.array(memory["a"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, -1)
        old_log_probs = torch.tensor(np.array(memory["lp"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, 1)
        reset_masks = torch.tensor(np.array(memory["reset"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, 1)
        pred_targets = torch.tensor(np.array(memory["pred_target"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, -1)
        pred_masks = torch.tensor(np.array(memory["pred_mask"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, 1)
        bc_actions = torch.tensor(np.array(memory["bc_a"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, -1)
        bc_masks = torch.tensor(np.array(memory["bc_mask"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, 1)

        rewards = np.asarray(memory["r"][:usable_steps], dtype=np.float32)
        rewards = np.clip(rewards * self.reward_scale, -self.reward_clip, self.reward_clip).tolist()
        values = memory["v"][:usable_steps + 1]
        dones = memory["d"][:usable_steps]

        # Compute GAE
        returns = []
        advantages = []
        gae = 0.0
        for i in reversed(range(usable_steps)):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lamda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, 1)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, 1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        current_entropy_coef = self._get_entropy_coef()
        current_bc_coef = self._get_bc_supervised_coef()

        last_stats = {}
        for epoch in range(self.ppo_epochs):
            mu, std, b_values, pred_xy, _ = self._forward_recurrent(states, reset_masks)
            dist = Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - current_entropy_coef * entropy
            old_values = torch.tensor(np.array(memory["v"][:usable_steps]), dtype=torch.float32, device=self.device).view(num_episodes, self.seq_len, 1)
            clipped_values = old_values + torch.clamp(b_values - old_values, -self.value_clip_param, self.value_clip_param)
            critic_loss_unclipped = F.smooth_l1_loss(b_values, returns, reduction="none")
            critic_loss_clipped = F.smooth_l1_loss(clipped_values, returns, reduction="none")
            critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

            pred_error = F.mse_loss(pred_xy, pred_targets, reduction="none").mean(dim=-1, keepdim=True)
            pred_loss = (pred_error * pred_masks).sum() / (pred_masks.sum() + 1e-8)
            if actions.size(1) > 1:
                action_smooth_loss = F.mse_loss(actions[:, 1:], actions[:, :-1])
            else:
                action_smooth_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            action_l2_loss = actions.pow(2).mean()
            bc_supervised_error = F.mse_loss(mu, bc_actions, reduction="none").mean(dim=-1, keepdim=True)
            bc_supervised_loss = (bc_supervised_error * bc_masks).sum() / (bc_masks.sum() + 1e-8)

            self.opt.zero_grad()
            total_loss = (
                actor_loss
                + self.value_coef * critic_loss
                + self.pred_coef * pred_loss
                + self.smooth_coef * action_smooth_loss
                + self.action_l2_coef * action_l2_loss
                + current_bc_coef * bc_supervised_loss
            )
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()

            last_stats = {
                "actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "pred_loss": float(pred_loss.item()),
                "pred_coef": float(self.pred_coef),
                "smooth_loss": float(action_smooth_loss.item()),
                "l2_loss": float(action_l2_loss.item()),
                "bc_supervised_loss": float(bc_supervised_loss.item()),
                "bc_supervised_coef": float(current_bc_coef),
                "smooth_coef": float(self.smooth_coef),
                "action_l2_coef": float(self.action_l2_coef),
                "entropy": float(entropy.item()),
                "entropy_coef": float(current_entropy_coef),
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
            }

        self.train_steps += usable_steps

        if self.icm_loss_coef > 0 or self.icm_reward_coef > 0:
            r_int, inv_loss, fwd_loss = self.icm(states.detach(), next_states.detach(), actions.detach())
            icm_loss = inv_loss + self.icm_loss_coef * fwd_loss
            self.icm_opt.zero_grad()
            icm_loss.backward()
            nn.utils.clip_grad_norm_(self.icm.parameters(), 0.5)
            self.icm_opt.step()
            last_stats.update({
                "icm_loss": float(icm_loss.item()),
                "intrinsic_reward": float(r_int.mean().item()),
            })
        else:
            last_stats.update({
                "icm_loss": 0.0,
                "intrinsic_reward": 0.0,
            })
        return last_stats


class ICMModel(nn.Module):
    """Intrinsic Curiosity Module for exploration."""

    def __init__(self, s_dim, a_dim, hidden_dim=256):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, a_dim),
        )
        self.forward_net = nn.Sequential(
            nn.Linear(hidden_dim + a_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, s, s_next, a):
        f_s = self.feature_net(s)
        f_s_next = self.feature_net(s_next)
        pred_a = self.inverse_net(torch.cat([f_s, f_s_next], dim=-1))
        inv_loss = F.mse_loss(pred_a, a)
        pred_f_s_next = self.forward_net(torch.cat([f_s, a], dim=-1))
        forward_error = F.mse_loss(pred_f_s_next, f_s_next.detach(), reduction="none").mean(-1)
        intrinsic_reward = 0.5 * forward_error
        fwd_loss = forward_error.mean()
        return intrinsic_reward, inv_loss, fwd_loss
