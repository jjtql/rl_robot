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
- Metrics: coverage, reward, success/covered count, and latency first. Treat material quality/loss as secondary diagnostics unless a method explicitly optimizes material deposition.

Treat single-steam settings as curriculum or sanity checks, not the main contribution.

Checkpoint selection:

- Treat saved checkpoints as candidates; do not default to `latest`.
- If early or middle checkpoints beat the final policy on held-out seeds, report the selected checkpoint protocol explicitly and use the same protocol for ablations.
- Prefer checkpoint sweep on `multi_realistic` and `multi_hard` before formal full-matrix evaluation.

## Current Method Flow

Current active direction: `thermal_lstm_spawnhist_glue_v10`.

The design goal is not to replace LSTM-PPO. The design goal is to change how the surrounding planner pieces are glued to LSTM-PPO:

- `horizon2` remains the sparse/lull short-horizon route selector and BC teacher.
- `dynamic_weighted` can be used as the dense/burst base because quick held-out v9 checks showed it was stronger on `multi_hard`.
- LSTM-PPO remains a residual learner, not the whole planner.
- Spawn-history prediction is the auxiliary task that should make LSTM memory useful.
- Steam attention handles variable target sets up to 8 active steams.
- Thermal context and route-summary features give the recurrent policy compact signals about spawn pressure, route ambiguity, and stagnation.
- Phase-aware residual glue changes the base controller and residual freedom by environment phase:
  - `lull`/`sparse`/`charging`: use the sparse base, normally `horizon2`, and allow larger residual beta so LSTM memory can move toward predicted hot regions.
  - `dense`/`burst`: use the dense base, normally `dynamic_weighted`, and strongly shrink residual beta so PPO cannot damage strong visible-target routing.
  - `mid`: fall back to the configured residual base, normally `horizon2`.
- The residual action shield remains active to keep PPO exploration from undoing planner progress.
- Material observation/reward is disabled in v9/v10 because previous runs suggested material-quality terms were diluting the coverage objective on `multi_hard` and `multi_extreme`.
- Burst/lull steam generation creates the intended complementarity: rapid one-by-one thermal bursts give the visible-target base controller useful targets to route through, while quiet sparse intervals expose the LSTM's short-term spawn prediction value.

What to look for:

- A real improvement should beat `horizon2`, `horizon3`, `dynamic_weighted`, and `planner_ensemble` on held-out `multi_hard`, not just improve training-window last50.
- If v9 improves `multi_hard` but not `multi_extreme`, frame extreme as a limitation instead of forcing a strong claim.
- If PPO only matches `horizon2`, the honest claim is that learned residual control preserves planner-level coverage while adding memory/prediction machinery, not that RL is superior.
- If early checkpoints beat `latest`, use checkpoint sweep as the selection protocol and apply it consistently across ablations.

## Next Method Ideas

Good next experiments:

- Current implemented candidate: `thermal_lstm_spawnhist_glue_v10`. It keeps the LSTM-PPO core but changes residual composition from one fixed planner base to phase-aware glue. Sparse/lull phases use `horizon2` with larger residual freedom; dense/burst phases use `dynamic_weighted` with small residual freedom. It also gives `multi_extreme` more curriculum budget than v9 and slightly strengthens the spawn prediction auxiliary loss.
- Completed diagnostic: `thermal_lstm_spawnhist_cover_v9`. It dropped material-quality learning and used burst/lull spawn cycles, but held-out quick eval showed latest PPO mostly matched `horizon2`: about `0.736/0.621/0.626` on `multi_realistic/multi_hard/multi_extreme` across train seeds 0/1, versus `horizon2` about `0.748/0.628/0.622`. Treat v9 as stable but not a breakthrough.
- Treat `thermal_lstm_spawnhist_ensemble_v8` as a negative diagnostic: held-out ckpt900 hard eval did not clearly beat `horizon2`, `horizon3`, `dynamic_weighted`, or `planner_ensemble`.
- If v6 stays mid, try `thermal_lstm_spawnhist_thermal_v7` before any bigger beta sweep: thermal-aware target scoring, `horizon2` residual base/BC expert, 8-steam attention, and the same spawn-history prediction loss.
- Use the v6 memory candidate before more beta-only tuning: `horizon3` residual base/BC expert, 8-steam attention, thermal context observation, keep LSTM state across cover events, and restore spawn prediction loss.
- Tune residual strength: `residual_beta` in `0.1,0.2,0.3`.
- Compare residual bases: `horizon2`, `dynamic_weighted`, `risk_aware`.
- Compare v8 against `horizon2`, `horizon3`, `dynamic_weighted`, and `planner_ensemble` as standalone policies before claiming PPO adds value.
- Add map-centric state only if it directly describes uncovered steam pressure or coverage holes. Avoid material map/loss terms until coverage is stable.
- Use horizon planner as an explicit baseline and BC teacher, not as hidden magic.
- Evaluate generalization with different `max_steams`, thermal spawn parameters, and held-out seeds.

When the user wants a quick "water journal" route, prefer a narrow applied framing with transparent baselines over hiding strong baselines. The paper can still be modest; it should not be brittle.

Current hard/extreme warning:

- Do not frame `thermal_lstm_spawnhist_release_v5` as a hard/extreme improvement; held-out quick eval did not support that claim.
- Treat `thermal_lstm_spawnhist_memory_v6` as the next diagnostic experiment for whether LSTM memory and spawn prediction can combine with a stronger planner base.
