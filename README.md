# Planner-Guided Residual RL For Persistent Multi-Steam Coverage

This repository contains the runnable core code for the MuJoCo reinforcement-learning paper pipeline in `rl_robot`.

The main method is a planner-guided residual policy for persistent dynamic steam/defect coverage:

- persistent steam targets that remain active until covered,
- risk-aware target features using age, distance, material gap, and reachability,
- unordered steam-set attention for variable active targets,
- material-map and material-quality reward variants,
- strong rule/planner baselines including dynamic weighted greedy, receding-horizon planning, and ACO/TSP-style routing,
- residual PPO around a strong base controller, usually `horizon2`.

## Repository Contents

```text
coverage_schemes/scheme_d_paper_base/  Core environment, PPO, policies, training, evaluation
meshes/                                Robot mesh files used by the MuJoCo XML
o1.xml                                 Default MuJoCo robot/task model
scripts/                              Reproducible command wrappers
docs/                                 Project and method notes
```

The latest implementation plan is documented in `docs/latency-first-continuous-session-v11.md`.

Generated runs, checkpoints, virtual environments, and local editor files are intentionally excluded.

## ICCC 2026 Major Revision

The five-page revised manuscript, reviewer response, corrected hierarchical statistics, and episode-level evaluation data are available in:

```text
paper/iccc2026_major_revision/
```

Recompute the reported hierarchical tables with:

```bash
python paper/iccc2026_major_revision/recompute_hierarchical_stats.py
```

The interactive real-rollout viewer is self-contained and does not require trained weights:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
# open http://127.0.0.1:8765/system_ui/
```

See `system_ui/README.md` for remote access and replay-generation instructions, and `docs/model-weights.md` for the checkpoint publication policy.

## Environment

Python 3.8+ is recommended. The original experiments used a local virtual environment named `mujoco_rl_env`.

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install PyTorch from the official PyTorch selector first, then install the remaining requirements.

## Smoke Test

Run a short headless training job:

```bash
bash scripts/smoke_train.sh
```

Expected behavior:

- it creates `runs/smoke_train/github_smoke`,
- it writes `config.json`, `episodes.csv`, `events.jsonl`, and checkpoints,
- it finishes in a few minutes on a normal CPU.

## Main v4 Training Command

Generate the 5-seed commands for the paper candidate:

```bash
python -m coverage_schemes.scheme_d_paper_base.run_training_suite \
  --methods planner_residual_attention_v4,planner_residual_map_tv_v4 \
  --seeds 0,1,2,3,4 \
  --output runs/scheme_d_paper_suite/planner_residual_v4_commands.txt
```

Run the generated commands only when you have enough time. The full 10-run matrix can take many hours.

For a single seed:

```bash
bash scripts/train_v4_seed0.sh
```

## Evaluation

After training a checkpoint, evaluate against strong baselines:

```bash
bash scripts/eval_v4_seed0.sh
```

The evaluation matrix includes:

- `nearest`
- `oldest`
- `distance_age`
- `risk_aware`
- `dynamic_weighted`
- `horizon2`
- `horizon3`
- `aco_tsp`
- `ppo`

Use `multi_low`, `multi_realistic`, and `multi_hard` for the main paper comparison.

## Aggregate Results

```bash
python -m coverage_schemes.scheme_d_paper_base.aggregate_results \
  runs/matrix/planner_residual_attention_v4_seed0_eval/all_rows.csv \
  --output runs/matrix/planner_residual_attention_v4_seed0_eval/aggregate_summary.csv
```

## Paper Readiness Criteria

Do not claim the method is paper-ready until the repository has reproducible artifacts showing:

- at least 5 seeds for the main methods,
- strong rule/planner baselines on identical seeds and stages,
- ablations for attention, material map/TV reward, BC, and residual control,
- mean/std tables for coverage, reward, latency, missed count, material losses, and action metrics,
- qualitative trajectory/material visualizations.

## Notes

This code release does not include trained checkpoints. Checkpoints are large generated artifacts and should be uploaded separately as GitHub Releases, Zenodo artifacts, or another model-storage channel if needed.
