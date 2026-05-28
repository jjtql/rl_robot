# Latency-First Continuous Session V11

## Goal

The real ShangZeng process is a long continuous operation, not a set of isolated short coverage games. A better training target is therefore response quality over time:

- how fast each steam point is covered,
- whether coverage stays inside a response SLA,
- whether active steam backlog accumulates,
- whether the oldest active steam point is left waiting too long,
- whether robot motion remains smooth enough for deployment.

Raw coverage rate is still logged, but it is no longer the primary objective for v11.

## Constraints

- Keep the LSTM-PPO core.
- Keep `horizon2` as the planner scaffold for this version.
- Do not switch to `horizon3`.
- Improve the glue, reward, metrics, session semantics, and evaluation around the existing LSTM-PPO pipeline.

## Implemented Design

### Continuous Training Windows

The environment now supports `start_new_chunk()`, which starts a new training window without resetting the physical session. This keeps:

- active steam points,
- thermal hotspots,
- material state,
- spawn/lull/burst state,
- accumulated response metrics.

The v11 preset uses:

```text
continuous_session_chunks = 4
carry_lstm_state_across_chunks = true
lstm_sequence_chunks = 4
episode_steps = 800
```

So one simulated session spans four 800-step windows, while PPO trains on a 3200-step recurrent sequence.

### LSTM State Handling

At a true new session, LSTM hidden state is reset. Across chunks inside one session, hidden state is carried forward. This makes the recurrent memory match the actual task more closely: the process is continuous and history should matter.

The existing `--keep-lstm-state-on-cover` behavior remains enabled so covering one steam point does not erase recent spawn/hotspot context.

### Latency-First Reward

The v11 reward keeps coverage rewards but reduces their dominance and adds explicit response-pressure terms:

- per-cover latency penalty,
- SLA bonus for fast responses,
- SLA miss penalty for late responses,
- per-step oldest-active-steam penalty,
- per-step backlog penalty,
- stronger quick-cover reward,
- action smoothing penalty kept enabled.

This changes the learning signal from "eventually cover many points" to "cover points quickly while keeping the field under control."

### Metrics

Environment, training CSV, eval CSV, and matrix summaries now log:

- `cover_latency_p50`,
- `cover_latency_p90`,
- `cover_latency_p95`,
- `cover_latency_max`,
- seconds variants in eval,
- `response_sla_success_rate`,
- `response_sla_success_count`,
- `response_sla_miss_count`,
- `active_steam_mean`,
- `active_steam_max`,
- `oldest_active_age`,
- `oldest_active_age_max`,
- `oldest_active_age_max_seconds`.

Training output also prints p90 latency and SLA rate.

### Checkpoint Selection

Checkpoint sweep now supports:

```bash
--rank-objective latency
```

Latency ranking sorts by:

1. higher SLA success rate,
2. lower p90 cover latency,
3. higher covered-per-second,
4. fewer misses,
5. higher coverage rate.

This avoids selecting a checkpoint that has acceptable coverage but poor response time.

## Main Preset

The main method is:

```text
thermal_lstm_spawnhist_latency_v11
```

Key settings:

```text
residual_base_policy = horizon2
residual_sparse_base_policy = horizon2
residual_dense_base_policy = horizon2
residual_glue = phase_aware
residual_beta_start = 0.04
residual_beta_end = 0.26
residual_beta_warmup_steps = 650000
pred_coef = 0.05
prediction_horizon_steps = 150
response_sla_steps = 110
continuous_session_chunks = 4
lstm_sequence_chunks = 4
```

The phase-aware residual scales still give LSTM-PPO more freedom in sparse/lull phases and less freedom during dense/burst phases, but every phase uses `horizon2` as the base planner in v11.

## Files Changed

- `coverage_schemes/scheme_d_paper_base/env.py`
- `coverage_schemes/scheme_d_paper_base/train.py`
- `coverage_schemes/scheme_d_paper_base/eval.py`
- `coverage_schemes/scheme_d_paper_base/run_matrix.py`
- `coverage_schemes/scheme_d_paper_base/checkpoint_sweep.py`
- `coverage_schemes/scheme_d_paper_base/run_training_suite.py`
- `coverage_schemes/scheme_d_paper_base/config.py`
- `scripts/train_memory_v11_latency_seeds01.sh`
- `scripts/eval_v11_latency_quick.sh`
- `rl-robot-project-state/SKILL.md`

## Training Command

```bash
cd /Data2/jj/rl_robot
scripts/train_memory_v11_latency_seeds01.sh
```

This launches seeds 0 and 1 concurrently.

## Quick Evaluation

```bash
cd /Data2/jj/rl_robot
scripts/eval_v11_latency_quick.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_latency_v11_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  runs/matrix/thermal_lstm_spawnhist_latency_v11_quick
```

The quick eval uses a longer 3200-step demo-mode window so response metrics are measured over a more realistic continuous interval.

## Latency-First Checkpoint Sweep

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
  --rank-objective latency \
  --device cpu
```

## Verification Already Run

The implementation was checked with:

```bash
python3 -m py_compile \
  coverage_schemes/scheme_d_paper_base/env.py \
  coverage_schemes/scheme_d_paper_base/train.py \
  coverage_schemes/scheme_d_paper_base/eval.py \
  coverage_schemes/scheme_d_paper_base/run_matrix.py \
  coverage_schemes/scheme_d_paper_base/checkpoint_sweep.py \
  coverage_schemes/scheme_d_paper_base/run_training_suite.py
```

A CPU smoke train completed 160 steps, triggered one PPO update, wrote checkpoints, and logged the new latency/SLA fields.

A one-row eval matrix also completed and wrote p90/SLA summary columns.

## How To Judge V11

For v11, report results in this order:

1. `response_sla_success_rate`
2. `cover_latency_p90_seconds`
3. `cover_latency_seconds`
4. `covered_per_second`
5. `active_steam_mean`
6. `oldest_active_age_max_seconds`
7. `action_delta_mean`
8. `coverage_rate`
9. reward

The method is only a real improvement if it improves response latency/SLA without creating unacceptable backlog, misses, or motion roughness.
