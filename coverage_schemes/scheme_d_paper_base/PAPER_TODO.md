# scheme_d_paper_base Paper Pipeline

This package is the clean paper baseline copied from `coverage_schemes/scheme_d_fixed copy`.
Keep the original validated scheme unchanged.

## Current Status

- Baseline source copied into a valid Python package name.
- Environment model path is configurable and defaults to repo-root `o1.xml`.
- Steam points persist until covered; they do not disappear automatically when old.
- Training supports CLI overrides, seed setup, headless mode, per-run config, CSV episode logs, JSONL event logs, and run-local checkpoints.
- Behavior cloning can warm-start from configurable rule experts and collect data across single and multi-steam stages.
- PPO observations now expose risk-aware selected-target features: target id, distance, age score, distance score, material-gap score, reachability score, and final risk score.
- Training has ablation switches for no BC, no curriculum, no LSTM, no potential shaping, no best-progress reward, no material observation, and no action smoothing/L2 penalty.
- Evaluation runs without MuJoCo viewer and supports PPO checkpoints plus rule baselines.
- Evaluation matrix supports `--max-steams 3,4,6` style generalization tests and aggregation groups by `max_steams`.
- `plot_results.py` generates grouped summary tables and optional PNG metric plots from evaluation or training CSV files.
- `visualize_episode.py` records qualitative trajectory CSV, trajectory plot, material heatmap, and summary JSON for rule or PPO policies.
- `PAPER_DRAFT.md` contains the current methods/results skeleton and reproducibility commands.
- `run_training_suite.py` writes or executes multi-seed PPO training commands for the main method and ablations.
- `--steam-attention` enables the attention-LSTM PPO variant with per-steam set features while leaving existing 35-D checkpoints compatible.

## Smoke Commands

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.train \
  --run-dir runs/smoke_train \
  --run-name smoke \
  --seed 5 \
  --episode-steps 20 \
  --update-episodes 1 \
  --save-interval 1 \
  --stage-episodes single_easy:1,multi_low:1 \
  --target-selector risk_aware \
  --bc-policy risk_aware \
  --bc-stage-episodes single_easy:1,multi_low:1 \
  --bc-epochs 1 \
  --headless

mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.eval \
  --policy nearest \
  --stage multi_low \
  --episodes 2 \
  --steps 20 \
  --seed 3 \
  --output runs/smoke_eval/nearest.csv

mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.aggregate_results \
  runs/smoke_eval/nearest.csv \
  --output runs/smoke_eval/summary.csv

mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.plot_results \
  runs/smoke_eval/nearest.csv \
  --output-dir runs/smoke_eval/plots \
  --plot

mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.visualize_episode \
  --policy risk_aware \
  --stage multi_realistic \
  --steps 80 \
  --seed 0 \
  --output-dir runs/qualitative/risk_aware_seed0

mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.run_training_suite \
  --methods attention,risk_aware,nearest,no_bc,no_lstm,no_potential,no_material_obs \
  --seeds 0,1,2,3,4 \
  --output runs/scheme_d_paper_suite/commands.txt
```

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies nearest,oldest,distance_age,risk_aware \
  --stages multi_low,multi_realistic \
  --seeds 0,1,2 \
  --episodes 10 \
  --steps 800 \
  --max-steams 3,4,6 \
  --output-dir runs/matrix/rule_baselines
```

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.train \
  --run-dir runs/ablations \
  --run-name no_lstm_no_shaping_seed0 \
  --seed 0 \
  --target-selector nearest \
  --stage-episodes multi_realistic:250 \
  --no-bc \
  --no-lstm \
  --no-potential-shaping \
  --no-best-progress \
  --no-material-observation \
  --no-action-smoothing-penalty \
  --headless
```

## Hourly Monitored Training

```bash
python3 /home/jiangjian/.codex/skills/rl-paper-pipeline/scripts/monitor_run.py \
  --name scheme_d_paper_base_seed0 \
  --cwd /mnt/d/robot_scit/rl_robot \
  --log-dir runs/monitors \
  --heartbeat-seconds 3600 \
  --post-command "mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.eval --policy ppo --model runs/scheme_d_paper_base/seed0/checkpoints/scheme_d_paper_base_latest_full.pt --stage multi_realistic --episodes 30 --steps 800 --seed 100 --output runs/eval/scheme_d_paper_base_seed0_multi_realistic.csv" \
  -- mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.train \
    --run-dir runs/scheme_d_paper_base \
    --run-name seed0 \
    --seed 0 \
    --headless
```

## Baselines To Run

- `--policy random`
- `--policy nearest`
- `--policy oldest`
- `--policy distance_age`
- `--policy risk_aware`
- `--policy ppo --model <checkpoint>`

Use `multi_low` and `multi_realistic` as the main paper settings because the risk-aware contribution requires multiple active steam points.

## Risk-Aware PPO Step

Completed: PPO can use `target_selector=risk_aware` to observe the same target semantics as the risk-aware rule expert. The nearest-target selector remains available as an ablation through `--target-selector nearest`. Training and evaluation CSVs log selected target id, age score, distance score, material score, reachability score, and final risk score so the learned policy can be compared against the rule selector.

## Next Implementation Phase

1. Run at least 5 seeds for key methods: attention PPO, nearest PPO, risk-aware PPO, no-BC PPO, no-LSTM PPO, no-potential PPO, and rule baselines.
2. Run generalization tests with `max_steams` 3, 4, and 6 using the same checkpoints.
3. Populate `PAPER_DRAFT.md` with full multi-seed results and final qualitative figures.
