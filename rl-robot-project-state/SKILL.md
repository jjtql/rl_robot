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

Current next method: `thermal_lstm_spawnhist_latency_v11`.

Goal: keep LSTM-PPO unchanged, but make the training/evaluation objective match the real ShangZeng process: a continuous roughly two-hour covering task where response time, response speed, backlog, and smoothness matter more than raw coverage percentage.

Important interpretation change:

- `coverage_rate` is still logged, but it is no longer the primary success metric.
- Primary metrics should be:
  - average cover latency
  - p90/p95/max cover latency
  - `response_sla_success_rate`
  - `covered_per_second` / `per_point_cover_speed`
  - `active_steam_mean` and `oldest_active_age_max`
  - `action_delta_mean` / `action_l2_mean` for smoothness
- For held-out checkpoint selection, use latency-first ranking before making hard/extreme claims.

v11 design:

- `thermal_lstm_spawnhist_latency_v11`
- LSTM-PPO core remains unchanged.
- `horizon2` remains the residual planner base; no `horizon3`.
- 4 episode windows form one continuous physical session:
  - `continuous_session_chunks=4`
  - `carry_lstm_state_across_chunks=True`
  - `lstm_sequence_chunks=4`
- Environment state, active steam points, thermal hotspots, and material state continue across those chunks.
- LSTM hidden state is carried across chunks unless a true new session starts.
- PPO recurrent sequence length becomes `episode_steps * lstm_sequence_chunks`, so the update sees longer temporal structure than a single 800-step chunk.
- Reward is latency-first:
  - lower cover-reward dominance
  - stronger quick-cover bonus
  - explicit cover-latency penalty
  - SLA bonus/miss penalty
  - oldest-active-steam and backlog penalties
  - action smoothing penalty is kept enabled

Implemented in:

- `coverage_schemes/scheme_d_paper_base/env.py`
  - response latency percentiles and SLA metrics
  - active backlog and oldest active steam metrics
  - `start_new_chunk()` for continuous sessions without physical reset
  - optional latency-first reward terms
- `coverage_schemes/scheme_d_paper_base/train.py`
  - CLI flags for latency reward and continuous chunks
  - LSTM hidden carry across chunks
  - longer PPO recurrent sequence support
  - episode CSV logs response/SLA/backlog/session fields
- `coverage_schemes/scheme_d_paper_base/eval.py`
  - latency/SLA/backlog metrics in evaluation CSV and summary
- `coverage_schemes/scheme_d_paper_base/run_matrix.py`
  - summary includes p90 latency and SLA metrics
- `coverage_schemes/scheme_d_paper_base/checkpoint_sweep.py`
  - `--rank-objective latency` ranks by SLA, p90 latency, speed, misses, then coverage
- `coverage_schemes/scheme_d_paper_base/run_training_suite.py`
  - preset `thermal_lstm_spawnhist_latency_v11`
  - v11 ablation presets:
    - `thermal_lstm_spawnhist_latency_v11_no_pred`
    - `thermal_lstm_spawnhist_latency_v11_no_carry`
    - `thermal_lstm_spawnhist_latency_v11_no_latency_reward`
    - `thermal_lstm_spawnhist_latency_v11_no_attention`
    - `thermal_lstm_spawnhist_latency_v11_no_residual`
- `scripts/train_memory_v11_latency_seeds01.sh`
- `scripts/eval_v11_latency_quick.sh`
- `scripts/train_v11_ablation_commands.sh`
- `scripts/eval_v11_real_time_quick.sh`
- `scripts/run_v11_full_paper_suite.sh`
- `coverage_schemes/scheme_d_paper_base/collect_eval_summaries.py`

Extra v11 experiment support:

- `eval.py`, `run_matrix.py`, and `checkpoint_sweep.py` support `--decision-dt-seconds`.
- Use `--decision-dt-seconds 0.05` when reporting real high-level robot decision time.
- CSV keeps both `sim_step_seconds` and `decision_dt_seconds`.
- `coverage_schemes/scheme_d_paper_base/test.py` now uses `build_policy()` and checkpoint config, so the MuJoCo viewer path matches v11 residual PPO, burst/lull spawn, thermal context, and latency metrics.
- `scripts/run_v11_full_paper_suite.sh` is the preferred one-command entry point for full paper experiments.

One-command paper suite:

```bash
cd /Data2/jj/rl_robot
scripts/run_v11_full_paper_suite.sh
```

Default suite behavior:

- trains full v11 plus all v11 ablations,
- uses seeds `0,1,2`,
- runs two training jobs concurrently,
- evaluates latest checkpoints for every PPO run,
- evaluates `horizon2`, `dynamic_weighted`, and `planner_ensemble` baselines,
- uses held-out eval seeds `100,101,102`,
- uses `multi_low,multi_realistic,multi_hard,multi_extreme`,
- uses 3200-step demo-mode windows,
- reports real high-level control time with `DECISION_DT_SECONDS=0.05`,
- writes `combined_summary.csv` and `paper_summary.csv`.

Useful suite switches:

```bash
TRAIN_JOBS=3 scripts/run_v11_full_paper_suite.sh
SKIP_TRAIN=1 RUN_DIR=runs/v11_paper_suite/train scripts/run_v11_full_paper_suite.sh
SKIP_EVAL=1 scripts/run_v11_full_paper_suite.sh
RUN_SIM_TIME_EVAL=1 scripts/run_v11_full_paper_suite.sh
RUN_CHECKPOINT_SWEEP=1 scripts/run_v11_full_paper_suite.sh
```

Latest completed v11 training:

| Stage | seed0 last50 | seed1 last50 |
| --- | ---: | ---: |
| multi_low | 0.862 | 0.864 |
| multi_realistic | 0.820 | 0.826 |
| multi_hard | 0.779 | 0.782 |
| multi_extreme | 0.766 | 0.743 |

Held-out quick eval, seeds 100/101/102, 3 episodes per seed, 3200-step demo-mode:

| Stage | Method | Coverage | Mean Latency | P90 Latency | SLA |
| --- | --- | ---: | ---: | ---: | ---: |
| multi_realistic | horizon2 | 0.895 | 0.681s | 1.208s | 0.177 |
| multi_realistic | PPO avg | 0.911 | 0.643s | 1.224s | 0.245 |
| multi_hard | horizon2 | 0.860 | 0.790s | 1.420s | 0.163 |
| multi_hard | PPO avg | 0.861 | 0.798s | 1.547s | 0.169 |
| multi_extreme | horizon2 | 0.814 | 0.880s | 1.704s | 0.175 |
| multi_extreme | PPO avg | 0.855 | 0.906s | 1.660s | 0.144 |

Interpretation:

- Realistic: PPO is modestly better on coverage, mean latency, and SLA, but p90 is not clearly better.
- Hard: PPO is tied with horizon2.
- Extreme: PPO improves coverage robustness and p90 slightly, but SLA is worse.
- This is usable for a cautious engineering/application paper, not for a strong "latency solved" claim.

Paper-safe framing:

- Position the method as planner-guided recurrent residual PPO for continuous steam-point coverage.
- Lead with continuous operation, burst/lull generation, latency distribution, backlog, and smoothness.
- Claim robustness under extreme burst/lull only where the table supports it.
- Do not use coverage alone as the headline metric.
- Do not hide the simulator timebase: MuJoCo XML uses `0.002 s/step`; real control reporting should use `--decision-dt-seconds`, e.g. `0.05`.

Old v10 candidate remains useful context:

Core idea:

- sparse / lull / charging phases: use `horizon2` base, allow more residual freedom so LSTM memory can move toward likely future hot regions
- dense / burst phases: use `dynamic_weighted` base, shrink residual freedom so PPO does not damage strong visible-target routing
- mid phases: fall back to configured base, normally `horizon2`

Implemented in v10:

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

## Commands

Run full v11 paper suite:

```bash
cd /Data2/jj/rl_robot
scripts/run_v11_full_paper_suite.sh
```

Train v11 two seeds:

```bash
cd /Data2/jj/rl_robot
scripts/train_memory_v11_latency_seeds01.sh
```

Quick long-window eval v11:

```bash
cd /Data2/jj/rl_robot
scripts/eval_v11_latency_quick.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_latency_v11_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  runs/matrix/thermal_lstm_spawnhist_latency_v11_quick
```

Real high-level control-time eval v11:

```bash
cd /Data2/jj/rl_robot
DECISION_DT_SECONDS=0.05 scripts/eval_v11_real_time_quick.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_latency_v11_seed1/checkpoints/scheme_d_paper_base_latest_full.pt \
  runs/matrix/thermal_lstm_spawnhist_latency_v11_real_time_quick
```

Generate v11 ablation commands:

```bash
cd /Data2/jj/rl_robot
scripts/train_v11_ablation_commands.sh
```

Run v11 ablations:

```bash
cd /Data2/jj/rl_robot
SEEDS=0,1,2 scripts/train_v11_ablation_commands.sh --execute --jobs 2
```

Latency-first checkpoint sweep:

```bash
cd /Data2/jj/rl_robot
scripts/eval_checkpoint_sweep.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_latency_v11_seed0 \
  --checkpoint-tags 500,700,900,1100,latest \
  --stages multi_hard,multi_extreme \
  --seeds 100,101,102 \
  --episodes 3 \
  --steps 3200 \
  --demo-mode \
  --decision-dt-seconds 0.05 \
  --rank-objective latency \
  --device cpu
```

Live MuJoCo viewer, after X11 forwarding or remote desktop is available:

```bash
cd /Data2/jj/rl_robot
.venv/bin/python -m coverage_schemes.scheme_d_paper_base.test \
  --policy ppo \
  --model runs/scheme_d_paper_suite/thermal_lstm_spawnhist_latency_v11_seed1/checkpoints/scheme_d_paper_base_latest_full.pt \
  --stage multi_extreme \
  --seed 100 \
  --steps 3200 \
  --interval 100 \
  --sleep 0.02
```

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
4. For v11 and later, report latency/SLA/speed first, then coverage and reward.
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
