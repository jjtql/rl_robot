---
name: rl-robot-project-state
description: Use for continuing /Data2/jj/rl_robot work after context reset: current LSTM-PPO method state, v9 results, v10 phase-aware residual glue, training/eval commands, repo boundaries, and what claims are safe.
---

# RL Robot Project State

## Ground Rules

Work in `/Data2/jj/rl_robot`.

- Active method code lives in `coverage_schemes/scheme_d_paper_base`.
- Do not edit `coverage_schemes/scheme_d_fixed copy` unless explicitly asked.
- Keep run outputs, checkpoints, `.pt`, `.pth`, and large `runs/` artifacts out of git unless the user explicitly asks.
- Prefer `.venv/bin/python` for local commands.
- `horizon2` is not a neural model. It is the local deterministic receding-horizon planner implemented in `coverage_schemes/scheme_d_paper_base/policies.py` via `RecedingHorizonPolicy(horizon=2)` and `select_receding_horizon_steam(env, horizon=2)`.
- The LSTM-PPO core should not be replaced unless the user explicitly changes that constraint. Other glue, observations, rewards, planners, spawn logic, scripts, and docs may be changed.

## Current Evidence

Recent completed v9 run:

```text
runs/scheme_d_paper_suite/thermal_lstm_spawnhist_cover_v9_seed0
runs/scheme_d_paper_suite/thermal_lstm_spawnhist_cover_v9_seed1
```

v9 design:

- LSTM-PPO residual policy over `horizon2`
- steam attention count `8`
- spawn-history, thermal-context, and route-summary observations enabled
- keep LSTM state on cover
- material observation/reward disabled
- burst/lull thermal spawn: after cover, quiet period; after charge and sparse active set, a thermal burst emits points one-by-one
- spawn prediction auxiliary loss enabled: `pred_coef=0.04`, `prediction_horizon_steps=120`

v9 training completed, stable but not a breakthrough:

| Stage | seed0 last50 | seed1 last50 |
| --- | ---: | ---: |
| multi_low | 0.752 | 0.774 |
| multi_realistic | 0.699 | 0.683 |
| multi_hard | 0.618 | 0.652 |
| multi_extreme | 0.611 | 0.594 |

v9 held-out quick eval, seeds 100/101/102:

| Method | realistic | hard | extreme |
| --- | ---: | ---: | ---: |
| horizon2 | 0.748 | 0.628 | 0.622 |
| dynamic_weighted | 0.790 | 0.684 | 0.564 |
| PPO latest seed0 | 0.749 | 0.616 | 0.605 |
| PPO latest seed1 | 0.723 | 0.625 | 0.647 |
| PPO latest avg | 0.736 | 0.621 | 0.626 |

Interpretation:

- v9 proved the mechanism runs, but PPO mostly matched `horizon2`.
- `dynamic_weighted` was strongest on held-out `multi_hard`, so hard-stage routing needs better glue rather than only more beta.
- Early/mid checkpoints did not turn v9 into a clear held-out breakthrough.
- Safe claim: stable planner-level residual LSTM-PPO with memory/prediction machinery.
- Unsafe claim: RL clearly beats strong planners on hard/extreme.

## Current Candidate

Current next method: `thermal_lstm_spawnhist_glue_v10`.

Goal: keep LSTM-PPO unchanged, but change planner/RL composition so each component is used where it is strongest.

Core idea:

- sparse / lull / charging phases: use `horizon2` base, allow more residual freedom so LSTM memory can move toward likely future hot regions
- dense / burst phases: use `dynamic_weighted` base, shrink residual freedom so PPO does not damage strong visible-target routing
- mid phases: fall back to configured base, normally `horizon2`

Implemented in:

- `coverage_schemes/scheme_d_paper_base/policies.py`
  - `residual_glue_mode`
  - `residual_beta_for_env`
  - `PhaseAwareResidualBasePolicy`
  - `build_residual_base_policy`
- `coverage_schemes/scheme_d_paper_base/train.py`
  - new CLI args for `--residual-glue phase_aware`
  - rollout uses phase-aware base and effective beta
  - episode CSV logs phase counts and base-policy counts
- `coverage_schemes/scheme_d_paper_base/run_training_suite.py`
  - preset `thermal_lstm_spawnhist_glue_v10`

v10 preset highlights:

- residual glue: `phase_aware`
- sparse base: `horizon2`
- dense base: `dynamic_weighted`
- residual beta schedule: `0.04 -> 0.24` over `600000` steps
- phase beta scales:
  - sparse `1.25`
  - lull `1.35`
  - charging `1.0`
  - dense `0.35`
  - burst `0.25`
- curriculum: `multi_low:80,multi_realistic:150,multi_hard:720,multi_extreme:200`
- prediction: `pred_coef=0.05`, `prediction_horizon_steps=140`
- BC expert remains `horizon2`, with residual BC target as zero residual

Smoke verification already passed:

- `python3 -m py_compile ...`
- v10 command generation
- CPU smoke train through BC, rollout, PPO update, checkpoint save
- smoke CSV showed phase-aware counters and effective beta working

Latest pushed commit:

```text
47b11f2 Add phase-aware residual glue for RL robot
origin/main
```

## Commands

Train v10 two seeds:

```bash
cd /Data2/jj/rl_robot
scripts/train_memory_v10_glue_seeds01.sh
```

Quick eval v10:

```bash
cd /Data2/jj/rl_robot
scripts/eval_v10_glue_quick.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_glue_v10_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  runs/matrix/thermal_lstm_spawnhist_glue_v10_quick
```

Checkpoint sweep:

```bash
cd /Data2/jj/rl_robot
scripts/eval_checkpoint_sweep.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_glue_v10_seed0 \
  --checkpoint-tags 500,700,900,1100,latest \
  --stages multi_hard,multi_extreme \
  --seeds 100,101,102 \
  --episodes 3 \
  --device cpu
```

If CUDA is unavailable during eval, pass `--device cpu`; do not modify the checkpoint.

## How To Inspect Results

When asked to inspect a run:

1. Read `config.json`, `episodes.csv`, and `events.jsonl`.
2. Separate training last50 from held-out eval.
3. Compare against at least `horizon2` and `dynamic_weighted` under the same checkpoint config.
4. Report coverage first, then reward and latency.
5. Do not default to `latest_full.pt`; use checkpoint sweep for hard/extreme claims.

Helpful parsing targets:

- last50 coverage by stage
- best50 hard window and nearest checkpoint
- held-out seeds 100/101/102
- phase-aware diagnostics in v10:
  - `residual_effective_beta_mean`
  - `residual_lull_steps`
  - `residual_sparse_steps`
  - `residual_dense_steps`
  - `residual_burst_steps`
  - `residual_horizon2_base_steps`
  - `residual_dynamic_base_steps`

## Next Decisions

If v10 beats `horizon2` but not `dynamic_weighted` on hard:

- keep phase-aware glue
- tune dense base and beta scales
- compare dense base `dynamic_weighted` vs `planner_ensemble`

If v10 beats hard but not extreme:

- increase extreme curriculum or make burst/lull harder earlier
- run checkpoint sweep before declaring failure

If v10 still only matches horizon2:

- stop adding residual beta
- consider making PPO re-rank targets instead of modifying actions, while keeping LSTM-PPO as the learning core

Always keep claims narrow and evidence-driven.
