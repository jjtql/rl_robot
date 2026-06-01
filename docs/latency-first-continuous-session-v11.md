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
- `coverage_schemes/scheme_d_paper_base/collect_eval_summaries.py`
- `coverage_schemes/scheme_d_paper_base/run_training_suite.py`
- `coverage_schemes/scheme_d_paper_base/config.py`
- `scripts/train_memory_v11_latency_seeds01.sh`
- `scripts/eval_v11_latency_quick.sh`
- `scripts/train_v11_ablation_commands.sh`
- `scripts/eval_v11_real_time_quick.sh`
- `scripts/run_v11_full_paper_suite.sh`
- `scripts/run_v12_fast_paper_suite.sh`
- `rl-robot-project-state/SKILL.md`

## Fast-Response V12

Before the final paper run, use the fast-response preset instead of plain v11:

```bash
cd /Data2/jj/rl_robot
scripts/run_v12_fast_paper_suite.sh
```

The v12-fast preset keeps the LSTM-PPO core and `horizon2` residual base, but changes the training objective to match real response time:

- `decision_dt_seconds = 0.05`
- `response_sla_seconds = 4.0`
- internal SLA becomes `80` steps because `4.0 / 0.05 = 80`
- training CSV logs both step metrics and second metrics
- training console prints latency and p90 latency in seconds
- oldest-active penalty is normalized by SLA instead of max steam age
- latency/SLA reward pressure is stronger
- dense/burst residual freedom is lower so PPO is less likely to slow down the strong visible-target planner

V12-fast methods:

| Method | Purpose |
| --- | --- |
| `thermal_lstm_spawnhist_latency_v12_fast` | Full fast-response method |
| `thermal_lstm_spawnhist_latency_v12_fast_no_pred` | Remove auxiliary spawn prediction |
| `thermal_lstm_spawnhist_latency_v12_fast_no_carry` | Remove continuous recurrent carry |
| `thermal_lstm_spawnhist_latency_v12_fast_no_latency_reward` | Remove latency-first reward |
| `thermal_lstm_spawnhist_latency_v12_fast_no_attention` | Remove steam attention |
| `thermal_lstm_spawnhist_latency_v12_fast_no_residual` | Remove residual planner glue |

Main changed settings relative to v11:

```text
decision_dt_seconds = 0.05
response_sla_seconds = 4.0
cover_latency_penalty_gain = 30.0
response_sla_bonus = 20.0
response_sla_miss_penalty = 30.0
oldest_active_penalty_gain = 0.45
backlog_penalty_gain = 0.20
quick_cover_bonus_scale = 3.00
residual_beta_end = 0.20
residual_dense_beta_scale = 0.40
residual_burst_beta_scale = 0.25
stagnation_recovery_steps = 90
```

Use v11 as the conservative baseline and v12-fast as the previous strict-4s SLA diagnostic. V13 deadline-aware glue is the current candidate for reducing the 30-40 s latency tail.

## One-Command Paper Suite

Use this as the conservative v11 experiment entry point:

```bash
cd /Data2/jj/rl_robot
scripts/run_v11_full_paper_suite.sh
```

For the current deadline-aware run, prefer:

```bash
cd /Data2/jj/rl_robot
scripts/run_v13_deadline_paper_suite.sh
```

For the previous strict-4s fast-response diagnostic:

```bash
cd /Data2/jj/rl_robot
scripts/run_v12_fast_paper_suite.sh
```

The shared suite runner defaults to:

- the method list exported by the wrapper script,
- seeds `0,1,2`,
- two concurrent training jobs,
- held-out latest-checkpoint evaluation for every trained PPO run,
- the baseline policies exported by the wrapper script,
- stages `multi_low,multi_realistic,multi_hard,multi_extreme`,
- eval seeds `100,101,102`,
- 3 eval episodes per seed,
- 3200-step demo-mode windows,
- real control-time reporting with `DECISION_DT_SECONDS=0.05`.

Main outputs:

```text
runs/<suite_name>/train/
runs/<suite_name>/eval/latest_real_time/combined_summary.csv
runs/<suite_name>/eval/latest_real_time/paper_summary.csv
```

Common variants:

```bash
# More or fewer parallel training jobs.
TRAIN_JOBS=3 scripts/run_v13_deadline_paper_suite.sh

# Only evaluate existing runs, no training.
SKIP_TRAIN=1 RUN_DIR=runs/v13_deadline_paper_suite/train scripts/run_v13_deadline_paper_suite.sh

# Train only.
SKIP_EVAL=1 scripts/run_v13_deadline_paper_suite.sh

# Include simulator-time tables in addition to real decision-time tables.
RUN_SIM_TIME_EVAL=1 scripts/run_v13_deadline_paper_suite.sh

# Override the simulator-time reporting step if the XML timestep changes.
RUN_SIM_TIME_EVAL=1 SIM_DT_SECONDS=0.002 scripts/run_v13_deadline_paper_suite.sh

# Also sweep saved checkpoints for the full v13 method.
RUN_CHECKPOINT_SWEEP=1 scripts/run_v13_deadline_paper_suite.sh

# Re-score the same trained suite with a stricter SLA table.
EVAL_RESPONSE_SLA_SECONDS=10 SKIP_TRAIN=1 RUN_DIR=runs/v13_deadline_paper_suite/train EVAL_DIR=runs/v13_deadline_paper_suite/eval_sla10 scripts/run_v13_deadline_paper_suite.sh
```

For a cheaper dry run:

```bash
cd /Data2/jj/rl_robot
METHODS=thermal_lstm_spawnhist_latency_v11 \
SEEDS=99 \
SKIP_TRAIN=1 \
RUN_DIR=runs/scheme_d_paper_suite \
EVAL_DIR=runs/smoke/v11_full_paper_suite_script \
EVAL_STAGES=multi_low \
EVAL_SEEDS=1 \
EVAL_EPISODES=1 \
EVAL_STEPS=5 \
BASELINE_POLICIES=horizon2 \
scripts/run_v11_full_paper_suite.sh
```

The dry run is only for script verification; do not use its numbers in the paper.

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

## Real-Time Reporting

MuJoCo uses `0.002 s/step` in the current XML. That is the simulator integration step, not necessarily the real high-level robot control period. For paper tables, use an explicit decision period whenever the claim is about real response time:

```bash
cd /Data2/jj/rl_robot
DECISION_DT_SECONDS=0.05 scripts/eval_v11_real_time_quick.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_latency_v11_seed1/checkpoints/scheme_d_paper_base_latest_full.pt \
  runs/matrix/thermal_lstm_spawnhist_latency_v11_real_time_quick
```

The same option can be passed directly:

```bash
.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --model runs/scheme_d_paper_suite/thermal_lstm_spawnhist_latency_v11_seed1/checkpoints/scheme_d_paper_base_latest_full.pt \
  --policies horizon2,dynamic_weighted,planner_ensemble,ppo \
  --stages multi_realistic,multi_hard,multi_extreme \
  --seeds 100,101,102 \
  --episodes 3 \
  --steps 3200 \
  --demo-mode \
  --decision-dt-seconds 0.05 \
  --output-dir runs/matrix/thermal_lstm_spawnhist_latency_v11_real_time_quick
```

When `--decision-dt-seconds 0.05` is used, latency fields such as `cover_latency_p90_seconds` and `response_sla_seconds` are reported in real high-level control time. The CSV also keeps `sim_step_seconds`, so the paper can state both the simulator step and the decision-time reporting convention.

Do not compare a `0.002 s/step` table against a `0.05 s/decision` table. They answer different questions.

## Ablation Suite

The main paper needs actual ablations, not only an ablation plan. The current runnable v11 ablation methods are:

| Method | Purpose |
| --- | --- |
| `thermal_lstm_spawnhist_latency_v11` | Full method |
| `thermal_lstm_spawnhist_latency_v11_no_pred` | Remove auxiliary spawn prediction |
| `thermal_lstm_spawnhist_latency_v11_no_carry` | Remove continuous recurrent carry across chunks |
| `thermal_lstm_spawnhist_latency_v11_no_latency_reward` | Remove latency-first reward switch |
| `thermal_lstm_spawnhist_latency_v11_no_attention` | Remove steam attention observation |
| `thermal_lstm_spawnhist_latency_v11_no_residual` | Remove planner residual glue |

Generate the full command list:

```bash
cd /Data2/jj/rl_robot
scripts/train_v11_ablation_commands.sh
```

Run the ablations:

```bash
cd /Data2/jj/rl_robot
SEEDS=0,1,2 scripts/train_v11_ablation_commands.sh --execute --jobs 2
```

For a smaller first pass:

```bash
cd /Data2/jj/rl_robot
METHODS=thermal_lstm_spawnhist_latency_v11,thermal_lstm_spawnhist_latency_v11_no_pred,thermal_lstm_spawnhist_latency_v11_no_carry \
SEEDS=0,1 \
scripts/train_v11_ablation_commands.sh --execute --jobs 2
```

After training, evaluate each checkpoint with the same held-out seeds, stages, steps, demo-mode flag, and decision-time convention. The paper table should report at least:

1. SLA success rate
2. p90 cover latency
3. mean cover latency
4. covered per second
5. active steam mean/max
6. oldest active age max
7. action smoothness
8. coverage rate

## Current V11 Evidence

Completed v11 training was stable and much stronger than v9 in training coverage, but the held-out story is mixed.

Training last50 coverage:

| Stage | seed0 | seed1 |
| --- | ---: | ---: |
| multi_low | 0.862 | 0.864 |
| multi_realistic | 0.820 | 0.826 |
| multi_hard | 0.779 | 0.782 |
| multi_extreme | 0.766 | 0.743 |

Held-out quick eval used seeds `100,101,102`, 3 episodes per seed, 3200 steps, demo-mode. PPO values below average seed0 and seed1 checkpoints; horizon2 is the deterministic planner baseline.

| Stage | Method | Coverage | Mean Latency | P90 Latency | SLA |
| --- | --- | ---: | ---: | ---: | ---: |
| multi_realistic | horizon2 | 0.895 | 0.681s | 1.208s | 0.177 |
| multi_realistic | PPO avg | 0.911 | 0.643s | 1.224s | 0.245 |
| multi_hard | horizon2 | 0.860 | 0.790s | 1.420s | 0.163 |
| multi_hard | PPO avg | 0.861 | 0.798s | 1.547s | 0.169 |
| multi_extreme | horizon2 | 0.814 | 0.880s | 1.704s | 0.175 |
| multi_extreme | PPO avg | 0.855 | 0.906s | 1.660s | 0.144 |

Interpretation:

- Realistic: PPO is modestly better on coverage, mean latency, and SLA, but p90 latency is not clearly better.
- Hard: PPO is essentially tied with horizon2 and does not justify a strong claim.
- Extreme: PPO improves coverage robustness and p90 latency slightly, but SLA is worse.

This supports a cautious engineering/application paper, not a top-venue claim that LSTM-PPO solves latency-first coverage.

## Paper-Safe Framing

Safe claims:

- A planner-guided recurrent residual PPO framework can be trained stably for continuous ShangZeng-style steam coverage.
- Continuous session training, spawn-history observation, and latency-aware metrics better match the real task than short isolated episodes.
- In extreme burst/lull settings, the recurrent residual policy can improve coverage robustness over a strong horizon2 planner.
- The current method exposes the tension between high coverage and strict response-time SLA.

Unsafe claims:

- Do not claim that v11 solves real-time response.
- Do not claim broad superiority over horizon2 on all stages.
- Do not use coverage alone as the headline metric.
- Do not hide the 0.002s simulator step when discussing real deployment time.

Recommended title direction:

```text
Planner-Guided Recurrent Residual PPO for Continuous Steam-Point Coverage in ShangZeng Manipulation
```

Better abstract logic:

1. Introduce the real task as continuous steam-point response under burst/lull generation.
2. Explain why pure receding-horizon planning is strong when many visible points exist but weak for memory-driven sparse phases.
3. Propose LSTM-PPO as a residual policy over horizon2, with spawn history, thermal context, phase-aware residual scaling, and latency-aware training.
4. Evaluate coverage, latency distribution, SLA success, backlog, and smoothness.
5. State the limitation honestly: strict SLA remains hard, especially under real decision-time scaling.

## Viewer Check

The live MuJoCo viewer requires a graphical display. On a remote server, run it through X11 forwarding or a remote desktop session. A missing display gives:

```text
GLFWError: X11: The DISPLAY environment variable is missing
ERROR: could not initialize GLFW
```

Example viewer command after X11 is available:

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

The viewer script now loads checkpoint config, burst/lull spawn, latency reward settings, residual PPO glue, and live latency/SLA metrics instead of running a mismatched older PPO path.

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

Additional checks after the paper/experiment update:

```bash
python3 -m py_compile \
  coverage_schemes/scheme_d_paper_base/test.py \
  coverage_schemes/scheme_d_paper_base/eval.py \
  coverage_schemes/scheme_d_paper_base/run_matrix.py \
  coverage_schemes/scheme_d_paper_base/checkpoint_sweep.py \
  coverage_schemes/scheme_d_paper_base/run_training_suite.py \
  coverage_schemes/scheme_d_paper_base/config.py \
  coverage_schemes/scheme_d_paper_base/collect_eval_summaries.py

bash -n \
  scripts/eval_v11_real_time_quick.sh \
  scripts/train_v11_ablation_commands.sh \
  scripts/run_v11_full_paper_suite.sh \
  scripts/run_v12_fast_paper_suite.sh

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_training_suite \
  --methods thermal_lstm_spawnhist_latency_v11_no_pred,thermal_lstm_spawnhist_latency_v11_no_carry,thermal_lstm_spawnhist_latency_v11_no_latency_reward,thermal_lstm_spawnhist_latency_v11_no_attention,thermal_lstm_spawnhist_latency_v11_no_residual \
  --seeds 0 \
  --output runs/smoke/v11_ablation_commands_check.txt

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies horizon2 \
  --stages multi_low \
  --seeds 1 \
  --episodes 1 \
  --steps 20 \
  --decision-dt-seconds 0.05 \
  --demo-mode \
  --output-dir runs/smoke/v11_decision_dt_eval
```

The one-command suite was smoke-tested with a 5-step, one-seed evaluation-only run that generated both `combined_summary.csv` and `paper_summary.csv`.

The v12-fast timebase update was smoke-tested with:

- command generation for all six v12-fast methods,
- an 8-step CPU train using `response_sla_seconds=4.0`,
- verification that the saved config contains `decision_dt_seconds=0.05`, `response_sla_seconds=4.0`, and `response_sla_steps=80`,
- a 5-step wrapper eval through `scripts/run_v12_fast_paper_suite.sh`.

## How To Judge V11/V12

For v11/v12, report results in this order:

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

## V13 Deadline-Aware Planner Glue

The v12-fast run showed that a 4 s SLA target is too strict for the current task definition and that the long tail is mainly a queueing/starvation problem: some steam points remain active for tens of seconds while the controller keeps chasing easier visible targets.

V13 keeps the LSTM-PPO core unchanged and changes only the planner glue around it:

- new baseline/BC/residual base policy: `deadline_horizon2`,
- same two-step route enumeration as `horizon2`,
- much stronger age/deadline pressure in the route score,
- SLA target moved to the process-reasonable range: `response_sla_seconds = 15.0`,
- evaluation can still be repeated at 10 s, 15 s, and 20 s using `EVAL_RESPONSE_SLA_SECONDS`.

The route score estimates the age at arrival:

```text
arrival_age_i = age_i + elapsed_route_steps + travel_steps_i
deadline_ratio_i = arrival_age_i / response_sla_steps
```

Targets close to the deadline receive a sharply larger score, while leaving old targets outside the planned route receives an explicit starvation penalty. The intended behavior is:

- dense/burst phase: still cover efficiently with a horizon2 route,
- queue pressure high: oldest near-deadline targets dominate over nearest targets,
- sparse/lull phase: LSTM memory and spawn prediction can bias the residual toward likely upcoming hotspots,
- while moving toward an old point, the two-step route may still cover a newer point if it does not make the old point miss the deadline.

Implemented entry point:

```bash
cd /Data2/jj/rl_robot
scripts/run_v13_deadline_paper_suite.sh
```

Default v13 methods:

| Method | Purpose |
| --- | --- |
| `thermal_lstm_spawnhist_latency_v13_deadline` | Full deadline-aware residual LSTM-PPO |
| `thermal_lstm_spawnhist_latency_v13_deadline_horizon2_base` | Same SLA/reward, but ordinary `horizon2` BC/base |
| `thermal_lstm_spawnhist_latency_v13_deadline_no_pred` | Remove auxiliary spawn prediction |
| `thermal_lstm_spawnhist_latency_v13_deadline_no_carry` | Remove continuous recurrent carry |
| `thermal_lstm_spawnhist_latency_v13_deadline_no_latency_reward` | Remove latency-first reward |
| `thermal_lstm_spawnhist_latency_v13_deadline_no_attention` | Remove steam attention observation |
| `thermal_lstm_spawnhist_latency_v13_deadline_no_residual` | Remove planner residual glue |

Default v13 baselines:

```text
deadline_horizon2,horizon2,oldest,dynamic_weighted,planner_ensemble
```

To re-score an already trained suite with a different SLA threshold:

```bash
EVAL_RESPONSE_SLA_SECONDS=10 SKIP_TRAIN=1 RUN_DIR=runs/<suite>/train EVAL_DIR=runs/<suite>/eval_sla10 scripts/run_v13_deadline_paper_suite.sh
EVAL_RESPONSE_SLA_SECONDS=20 SKIP_TRAIN=1 RUN_DIR=runs/<suite>/train EVAL_DIR=runs/<suite>/eval_sla20 scripts/run_v13_deadline_paper_suite.sh
```

## V14 Deadline Rescue Diagnostic

V14 is a diagnostic extension for the case where v13 improves old-point backlog but does not reduce p90 latency enough.

New policy:

```text
deadline_rescue_horizon2
```

It behaves like `deadline_horizon2` during normal routing, but if any active steam point has predicted arrival age above about `0.65 * response_sla_steps`, it enters rescue mode and directly targets the most urgent old point. This turns the previous soft age weight into a hard anti-starvation rule.

New preset:

```text
thermal_lstm_spawnhist_latency_v14_rescue
```

Main differences from v13:

```text
bc_policy = deadline_rescue_horizon2
residual_base_policy = deadline_rescue_horizon2
residual_sparse_base_policy = deadline_rescue_horizon2
residual_dense_base_policy = deadline_rescue_horizon2
residual_beta = 0.12
residual_emergency_age_ratio = 0.65
residual_emergency_beta_scale = 0.05
```

The important design intent is that LSTM-PPO can still do residual refinement in normal phases, but when a target is close to starving, the planner almost fully takes over.
