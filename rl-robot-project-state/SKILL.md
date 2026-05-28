---
name: rl-robot-project-state
description: Work with the current /Data2/jj/rl_robot MuJoCo RL project state, including training runs, checkpoints, nohup logs, evaluation summaries, repository conventions, and the thermal LSTM spawn-history horizon2 residual PPO experiment. Use when Codex needs to inspect rl_robot results, explain what was trained, compare quick evaluations, decide what artifacts to commit, or continue from the latest project snapshot.
---

# RL Robot Project State

## Core Context

Use this skill inside `/Data2/jj/rl_robot`.

Important project boundaries:

- Treat `coverage_schemes/scheme_d_fixed copy` as the verified original baseline; do not edit it unless explicitly asked.
- Make active method work in `coverage_schemes/scheme_d_paper_base`.
- Keep generated run outputs under `runs/`; weights and large checkpoints are intentionally ignored by git.
- Prefer `.venv/bin/python` in this machine. If a task references old docs using `mujoco_rl_env/bin/python`, verify which environment exists before running.
- Existing broader workflow skill: `rl-robot-paper-continuation`. Use this skill for the latest run snapshot and local artifact decisions.

## Current Main Run

Historical reference run:

```text
runs/scheme_d_paper_suite/thermal_lstm_spawnhist_v3_horizon2_seed0
```

It finished with 1250 episodes and 1,000,000 steps.

The method is best described as:

```text
thermal spawn + spawn-history observation + steam attention + LSTM PPO
+ residual action refinement on a horizon2 planner base
```

Do not describe `horizon2` as an imported neural network. In this project it is a local deterministic receding-horizon planner implemented in `coverage_schemes/scheme_d_paper_base/policies.py` through `RecedingHorizonPolicy(horizon=2)` and `select_receding_horizon_steam(env, horizon=2)`.

Use this framing:

- `horizon2` handles explicit short-term target sequencing.
- LSTM handles learned memory, partial history, spawn patterns, and residual dynamics.
- The novelty claim must be task-specific and narrow; receding-horizon planning itself is not new.

For the detailed result snapshot, read `references/current_run_thermal_spawnhist_v3.md`.

## Latest Result Snapshot

The later `thermal_lstm_spawnhist_release_v5` runs completed for seed0/seed1. They were stable and the release schedule worked, but held-out quick evaluation did not show a breakthrough over `horizon2` or old v3:

- `multi_realistic`: roughly unchanged versus v3/horizon2.
- `multi_hard` and `multi_extreme`: training windows looked slightly better in places, but held-out quick eval was weaker than hoped.
- Important interpretation: simply increasing residual freedom is not enough; the LSTM needs a clearer memory/prediction job and the planner base needs to be stronger than `horizon2`.

The current next implementation candidate is `thermal_lstm_spawnhist_memory_v6`:

- residual base and BC expert: `horizon3`
- attention steam count: `8`
- keep recurrent state on cover: enabled
- thermal context observation: enabled
- spawn-history prediction loss: `pred_coef=0.04`
- residual beta schedule: `0.06 -> 0.30` over `650000` steps

Early partial v6 evidence was only modest: hard-stage training last-50 reached about `0.60`, but quick held-out checks did not beat `horizon2`. The later candidate `thermal_lstm_spawnhist_thermal_v7` keeps LSTM-PPO but makes the surrounding planner thermal-aware:

- thermal score is folded into target/routing risk scores
- residual base and BC expert use `horizon2`
- residual beta uses `0.06 -> 0.25`

The latest completed diagnostic candidate was `thermal_lstm_spawnhist_ensemble_v8`, intended to test whether components around LSTM-PPO can combine better than plain `horizon2`:

- residual base and BC expert: `planner_ensemble`
- base planner ensemble scores `horizon2`, `horizon3`, `dynamic_weighted`, `risk_aware`, and nearest-recovery candidates
- residual action shield: enabled, with stagnation recovery around `160` steps
- route-summary observation: enabled, adding active density, age pressure, distance pressure, target thermal score, route confidence, spawn readiness, and stagnation score
- spawn-history prediction loss remains enabled: `pred_coef=0.04`
- residual beta schedule is slower/smaller: `0.05 -> 0.22` over `700000` steps

Held-out quick eval around ckpt900 on `multi_hard` showed PPO did not clearly beat `horizon2`, `horizon3`, `dynamic_weighted`, or `planner_ensemble`. Treat v8 as a negative/diagnostic result, not a main improvement.

The completed `thermal_lstm_spawnhist_cover_v9` candidate deliberately dropped material-quality learning and focused the policy on covering steam points while using a burst/lull steam-generation regime:

- residual base and BC expert: `horizon2`
- burst/lull spawn enabled: after coverage, the environment enters a short quiet period; once spawn charge accumulates and active steam is sparse, a thermal-hotspot-driven burst emits several new points one-by-one at a short interval
- no material observation, no material map, no material TV/quality reward
- when material observation is disabled, risk/planner scoring ignores material score and uses age, distance, reachability, and thermal/spawn signals
- route-summary, thermal context, spawn-history observation, and spawn-history prediction loss remain enabled
- coverage-first reward profile increases cover/quick-cover reward, reduces precision and distance-shaping pressure, and increases active/aged-steam pressure slightly
- residual beta schedule is `0.05 -> 0.22` over `500000` steps, with a lighter residual-to-zero BC auxiliary loss (`0.04 -> 0.0` over `300000` steps)
- curriculum puts most budget on `multi_hard`: `multi_low:80,multi_realistic:170,multi_hard:850,multi_extreme:50`

v9 result snapshot:

- Training completed for seed0/seed1. Final last50 coverage was about `0.61/0.59` on `multi_extreme` and `0.62/0.65` on `multi_hard`.
- Held-out quick eval showed latest PPO mostly matched `horizon2` rather than beating it: PPO averaged about `0.736/0.621/0.626` on `multi_realistic/multi_hard/multi_extreme`, while `horizon2` was about `0.748/0.628/0.622` under the same v9 config.
- `dynamic_weighted` was stronger on held-out `multi_hard` at about `0.684`, which is the main clue for the next glue design.
- Early/middle checkpoints did not turn v9 into a clear held-out breakthrough.

The current next candidate is `thermal_lstm_spawnhist_glue_v10`, which keeps LSTM-PPO but changes the structure around it:

- residual glue: `phase_aware`
- sparse/lull/charging base: `horizon2`
- dense/burst base: `dynamic_weighted`
- sparse/lull residual beta is amplified so LSTM memory can act when few targets are visible
- dense/burst residual beta is shrunk so visible-target routing is protected
- `multi_extreme` receives more training budget than v9: `multi_low:80,multi_realistic:150,multi_hard:720,multi_extreme:200`
- spawn prediction is slightly stronger: `pred_coef=0.05`, `prediction_horizon_steps=140`

## Current Training Flow

The current main workflow is `thermal_lstm_spawnhist_glue_v10`. Its purpose is diagnostic and practical: keep the LSTM-PPO core, but change planner/RL composition so each component is used where it is strongest.

End-to-end flow:

1. Environment stage setup chooses a curriculum stage such as `multi_low`, `multi_realistic`, `multi_hard`, or `multi_extreme`.
2. The environment spawns persistent steam targets through burst/lull cycles. Thermal hotspots bias where burst targets appear, and a burst is queued as rapid sequential spawns rather than all targets appearing in one step.
3. Observation construction gives PPO the base robot/target state plus optional structured memory inputs:
   - steam-set attention over up to 8 active targets
   - spawn-history observation for recent spawn location/trend/cooldown
   - thermal-context observation for hotspot/spawn pressure
   - route-summary observation for active density, age pressure, nearest distance pressure, target thermal score, route confidence, spawn readiness, and stagnation
4. For v9, material observation is disabled. Material map and material TV/quality reward are not part of the training objective.
5. The behavior-cloning warm start still uses `horizon2` as the expert. Because training is residual PPO, the cloned action target is zero residual rather than duplicating the planner action.
6. During PPO rollout, phase-aware residual glue chooses the base controller:
   - sparse/lull/charging phases use `horizon2`
   - dense/burst phases use `dynamic_weighted`
   - mid phases fall back to `horizon2`
7. The LSTM-PPO actor outputs only a residual action. The residual beta is scheduled globally and then scaled by phase: sparse/lull gets more freedom, dense/burst gets less freedom.
8. The residual guard and action shield prevent the learned residual from undoing planner progress. When coverage stagnates, the shield falls back toward the planner direction.
9. The auxiliary spawn prediction loss stays enabled and is slightly stronger in v10 (`pred_coef=0.05`, `prediction_horizon_steps=140`) so the LSTM is pressured to use spawn history instead of only reacting to visible targets.
10. The reward is coverage-first: cover reward and quick-cover bonus are scaled up, precision and distance shaping are scaled down, and active/aged-steam penalties are scaled up. This makes hard-stage coverage more important than smooth material deposition.
11. Dense burst phases are intentionally favorable to `horizon2`, while sparse/lull phases are where LSTM memory should help by moving toward likely future thermal-spawn regions before many targets are visible.

In short: v10 no longer asks one fixed planner base to do everything. `dynamic_weighted` protects dense hard-stage routing, `horizon2` handles sparse/lull route selection, and LSTM-PPO gets the most freedom exactly when memory and spawn prediction should matter.

Important checkpoint-selection note:

- Do not assume `latest_full.pt` is the best checkpoint. Several runs show better held-out behavior at early/middle checkpoints than at the final stable policy.
- Use `scripts/eval_checkpoint_sweep.sh` to rank checkpoints by held-out coverage/reward/latency before selecting a model for paper claims.

## Working Procedure

When asked to inspect results:

1. Read `config.json`, `episodes.csv`, and `events.jsonl` from the run directory.
2. Separate training-window statistics from held-out evaluation results.
3. Prefer held-out `run_matrix.py` or `eval.py` results before making paper claims.
4. Report coverage, reward, latency, material losses, and the comparison to `horizon2`, `risk_aware`, and `dynamic_weighted`.
5. State uncertainty clearly when an evaluation is quick or low-sample.

When asked about upload/commit decisions:

- Include small text logs if the user asks for result logs.
- Do not add `.pt`, `.pth`, checkpoint directories, or large `runs/` artifacts unless explicitly requested.
- Check suspicious files with `file`, `wc -c`, and a short `sed` preview before staging.
- `nohup.out` in the current root is ASCII text, not a weight file; it records an argparse error from a malformed train command.

## Useful Commands

Inspect the latest run:

```bash
sed -n '1,220p' runs/scheme_d_paper_suite/thermal_lstm_spawnhist_v3_horizon2_seed0/config.json
tail -20 runs/scheme_d_paper_suite/thermal_lstm_spawnhist_v3_horizon2_seed0/events.jsonl
```

Run a quick held-out matrix:

```bash
.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies risk_aware,dynamic_weighted,horizon2,ppo \
  --stages multi_low,multi_realistic,multi_hard,multi_extreme \
  --seeds 100,101,102 \
  --episodes 5 \
  --steps 800 \
  --model runs/scheme_d_paper_suite/thermal_lstm_spawnhist_v3_horizon2_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  --output-dir runs/matrix/thermal_lstm_spawnhist_v3_horizon2_seed0_quick_eval
```

Sweep saved checkpoints in a run:

```bash
scripts/eval_checkpoint_sweep.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_cover_v9_seed0 \
  --checkpoint-tags 300,500,700,900,latest \
  --stages multi_hard \
  --seeds 100,101,102 \
  --episodes 5
```

Train the current v8 two-seed candidate:

```bash
scripts/train_memory_v8_ensemble_seeds01.sh
```

Train the current v9 coverage-first two-seed candidate:

```bash
scripts/train_memory_v9_cover_seeds01.sh
```

Train the current v10 phase-aware glue two-seed candidate:

```bash
scripts/train_memory_v10_glue_seeds01.sh
```

Run a quick v10 held-out comparison:

```bash
scripts/eval_v10_glue_quick.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_glue_v10_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  runs/matrix/thermal_lstm_spawnhist_glue_v10_quick
```

Run a quick v9 held-out comparison against strong non-PPO planners:

```bash
scripts/eval_v9_cover_quick.sh \
  runs/scheme_d_paper_suite/thermal_lstm_spawnhist_cover_v9_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  runs/matrix/thermal_lstm_spawnhist_cover_v9_quick
```

If local CUDA is unavailable and checkpoint loading ignores `--device cpu`, create a temporary CPU copy outside the repo or under `/tmp`; leave the original checkpoint untouched.
