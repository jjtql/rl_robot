---
name: rl-robot-future-directions
description: Plan credible next steps for the /Data2/jj/rl_robot paper pipeline, including weak and strong baselines, ablations, publication positioning, horizon2/LSTM framing, formal evaluation matrices, and future method ideas after the thermal LSTM spawn-history horizon2 residual PPO run. Use when Codex is asked what to train next, how to write the paper, what claims are safe, or how to turn current results into a publishable experiment plan.
---

# RL Robot Future Directions

## Core Positioning

Use this skill to keep the paper plan honest and publishable.

Current defensible contribution:

```text
A task-specific persistent multi-steam coverage controller that combines
thermal spawn modeling, spawn-history observation, steam-set attention,
LSTM memory, and residual PPO over a two-step receding-horizon planner.
```

Avoid these claims unless future experiments justify them:

- Do not claim receding-horizon planning itself is new.
- Do not claim broad RL superiority over planners.
- Do not omit `horizon2` from comparisons when the method uses it as BC expert or residual base.
- Do not call the method paper-ready from training curves alone.

For detailed next steps and experiment designs, read `references/future_plan.md`.

## Minimum Evidence Ladder

For a light applied paper, the minimum comparison set should be:

1. Simple rules: `nearest`, `oldest`
2. Stronger rules: `distance_age`, `risk_aware`, `dynamic_weighted`
3. Planner baseline: `horizon2`
4. Full method: thermal spawn-history attention LSTM residual PPO

Add ablations if time allows:

- no LSTM
- no spawn-history observation
- no steam attention
- no residual PPO, horizon2 only
- no BC warm start
- residual base `risk_aware` instead of `horizon2`

## Claim Rules

Use safe claims when evidence is limited:

- "improves on moderate multi-steam settings" if `multi_realistic` remains better than horizon2 across seeds.
- "matches or slightly improves planner coverage while retaining learnable residual control" if hard-stage gains stay small.
- "does not yet solve extreme spawn pressure" if `multi_extreme` remains worse than planner/rule baselines.

Avoid:

- "SOTA"
- "novel receding-horizon network"
- "RL outperforms all baselines"
- "robust generalization" without held-out seeds, spawn intensities, and max-steam tests.

## Evaluation Priority

Before writing the results section, run a formal matrix with at least 5 seeds for key methods and report mean/std.

Prioritize:

- `multi_realistic` as the main publishable setting.
- `multi_hard` as stress testing.
- `multi_extreme` as limitation or future work unless improved.
- Metrics: coverage, reward, success/covered count, latency, material quality loss, material hole loss, material TV loss, overfill, action smoothness.

Treat single-steam settings as curriculum or sanity checks, not the main contribution.

## Next Method Ideas

Good next experiments:

- If v6 stays mid, try `thermal_lstm_spawnhist_thermal_v7` before any bigger beta sweep: thermal-aware target scoring, `horizon2` residual base/BC expert, 8-steam attention, and the same spawn-history prediction loss.
- Use the v6 memory candidate before more beta-only tuning: `horizon3` residual base/BC expert, 8-steam attention, thermal context observation, keep LSTM state across cover events, and restore spawn prediction loss.
- Tune residual strength: `residual_beta` in `0.1,0.2,0.3`.
- Compare residual bases: `horizon2`, `dynamic_weighted`, `risk_aware`.
- Add map-centric state: compact material map or coverage-hole features.
- Add reward terms for material quality, holes, or total variation.
- Use horizon planner as an explicit baseline and BC teacher, not as hidden magic.
- Evaluate generalization with different `max_steams`, thermal spawn parameters, and held-out seeds.

When the user wants a quick "water journal" route, prefer a narrow applied framing with transparent baselines over hiding strong baselines. The paper can still be modest; it should not be brittle.

Current hard/extreme warning:

- Do not frame `thermal_lstm_spawnhist_release_v5` as a hard/extreme improvement; held-out quick eval did not support that claim.
- Treat `thermal_lstm_spawnhist_memory_v6` as the next diagnostic experiment for whether LSTM memory and spawn prediction can combine with a stronger planner base.
