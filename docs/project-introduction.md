# Project Introduction

## Project Goal

`rl_robot` is a MuJoCo-based reinforcement-learning project for robotic coverage control. The simulated robot must move a cover/nozzle marker over a circular workspace and cover active steam points that represent dynamic under-covered or defect-like material regions.

The current goal is to turn the validated baseline into a small applied RL paper. The work is not a general-purpose RL algorithm claim. It is an applied robotics/RL workflow around dynamic multi-target coverage, reproducible evaluation, and careful comparison against strong rule baselines.

## Task Definition

The environment contains:

- An industrial manipulator model loaded from repo-root `o1.xml`.
- A circular pot/workspace.
- A moving cover marker controlled by a continuous action.
- Steam points that appear over under-covered material or target regions.
- A material-height grid used to measure uniformity, holes, and overfill.

The action is a 3-D continuous vector:

- planar motion x,
- planar motion y,
- speed or coverage-intensity scaling.

Steam points persist until physically covered. They do not time out automatically. This is central to the paper: age is a neglect/persistence feature, not an expiration countdown.

## Main Package

The active package is:

```text
coverage_schemes/scheme_d_paper_base
```

Key files:

- `env.py`: MuJoCo/Gymnasium environment, curriculum stages, steam spawning, observations, rewards, material metrics.
- `algo.py`: PPO agent, LSTM actor-critic, attention-LSTM actor-critic, BC auxiliary loss, PPO update logic.
- `train.py`: training entrypoint with CLI overrides, BC warm start, CSV/JSONL logs, checkpoints.
- `eval.py`: headless evaluation for rule policies and PPO checkpoints.
- `policies.py`: rule baselines and PPO checkpoint loading.
- `run_matrix.py`: multi-policy, multi-stage, multi-seed evaluation matrix.
- `aggregate_results.py`: grouped mean/std summaries.
- `plot_results.py`: summary tables and optional plots.
- `visualize_episode.py`: qualitative rollout, trajectory, material heatmap, and JSON summary.
- `run_training_suite.py`: reproducible training command generator and method presets.
- `PAPER_DRAFT.md`: current manuscript skeleton.
- `PAPER_TODO.md`: current project status and command examples.

The original validated baseline is:

```text
coverage_schemes/scheme_d_fixed copy
```

Do not modify it unless explicitly asked.

## Curriculum Stages

The task uses staged environments:

- `single_easy`: one steam, larger cover radius, easier sanity stage.
- `single_precision`: one steam, smaller cover radius, more precise control.
- `multi_low`: multiple active steam points with lower density; important for paper evaluation.
- `multi_realistic`: denser/more realistic multi-steam setting; most important paper stage.

For paper evidence, focus on `multi_low` and `multi_realistic`.

## Metrics

Primary metrics:

- `episode_reward`
- `coverage_rate`
- `success_count`
- `spawned_count`
- `missed_count`
- `cover_latency`
- `last_cover_latency`
- `target_distance`

Target-selection diagnostics:

- `selected_target_id`
- `selected_target_distance`
- `nearest_target_distance`
- `selected_target_age_score`
- `selected_target_distance_score`
- `selected_target_material_score`
- `selected_target_reachability_score`
- `selected_target_risk_score`

Material metrics:

- `mean_material_height`
- `height_uniformity`
- `overfill_penalty`
- `material_hole_loss`
- `material_tv_loss`
- `material_quality_loss`

Action behavior metrics:

- `action_delta_mean`
- `action_l2_mean`

Report mean and standard deviation across seeds and episodes. Do not hand-copy one-off terminal lines into paper tables.

## Reproducibility Pattern

Use one run directory per method/seed. Each training run writes:

- `config.json`
- `episodes.csv`
- `events.jsonl`
- `checkpoints/*.pt`

Each evaluation matrix writes:

- per-policy/stage/seed CSV files,
- `all_rows.csv`,
- `summary.csv`,
- optional `aggregate_summary.csv`.

Always aggregate before judging a method.
