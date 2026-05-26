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

The latest completed run is:

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

If local CUDA is unavailable and checkpoint loading ignores `--device cpu`, create a temporary CPU copy outside the repo or under `/tmp`; leave the original checkpoint untouched.
