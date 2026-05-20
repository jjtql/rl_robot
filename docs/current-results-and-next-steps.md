# Current Results And Next Steps

## Current Status As Of 2026-05-18

The project has a mature experiment infrastructure but the learned PPO method is not paper-ready yet.

Implemented:

- Clean package `coverage_schemes/scheme_d_paper_base`.
- Headless training/evaluation.
- CSV/JSONL logging and run-local configs.
- Rule baselines and PPO checkpoint evaluation.
- Matrix evaluation and aggregation.
- Steam-set attention PPO.
- Material-map and material-TV variants.
- BC warm start and auxiliary BC loss.
- Stable-training knobs:
  - `--bc-supervised-min-coef`,
  - `--bc-supervised-decay-steps`,
  - lower PPO learning rate/clip/epochs,
  - `--no-action-smoothing-penalty`.
- v3 multi-steam-first presets in `run_training_suite.py`.

## Important Negative Results

### `attention_stable_seed0`

Formal evaluation of `latest` showed the PPO policy did not beat rules:

- PPO `multi_low` coverage around 0.922.
- PPO `multi_realistic` coverage around 0.732.
- Strong rules reached around 0.943 on `multi_low` and around 0.824 on `multi_realistic`.

Diagnosis:

- PPO drifted away from the BC/risk-aware expert.
- Earlier checkpoint probes were sometimes better than `latest`.
- BC auxiliary loss decayed to zero too early.

### `attention_stable_seed0` `multi_low_end`

Formal evaluation still did not beat rules:

- PPO `multi_low` coverage around 0.890.
- PPO `multi_realistic` coverage around 0.769.
- Rule baselines remained stronger.

### `attention_stable_v2_seed0`

This run kept a nonzero BC anchor but still underperformed in training:

- `single_easy`: mean coverage about 0.878.
- `single_precision`: mean coverage about 0.842.
- `multi_low`: mean coverage about 0.791.
- `multi_realistic`: mean coverage about 0.687.

Diagnosis:

- v2 still spent 850 episodes on single-steam stages.
- Multi-steam behavior remained weak.
- The retained BC coefficient was too weak by the end (`0.01`) relative to the need to preserve rule-like target pursuit.

## Current Best Next Experiment

Use `planner_residual_attention_v4` first, then `planner_residual_map_tv_v4`.

Key changes:

- use strong online baselines: `dynamic_weighted`, `horizon2`, `horizon3`, `aco_tsp`,
- train on `multi_low`, `multi_realistic`, and the harder `multi_hard`,
- execute a planner/rule base action plus a bounded learned residual,
- use `horizon2` as the default residual base and BC source,
- keep a high zero-residual BC anchor so PPO does not destroy the base controller,
- optionally add material map/TV and mild action delay/noise/domain randomization in `planner_residual_map_tv_v4`.

Smoke checks passed on 2026-05-18:

- `python3 -m compileall coverage_schemes/scheme_d_paper_base`
- `run_matrix` on `multi_hard` with `nearest,risk_aware,dynamic_weighted,horizon2,aco_tsp`
- one-episode residual training smoke: `runs/smoke_train/planner_residual_attention_v4_smoke`
- one-episode residual evaluation smoke: `runs/smoke_eval/planner_residual_attention_v4_smoke.csv`
- command generation: `runs/scheme_d_paper_suite/planner_residual_v4_commands.txt`

## Generate v4 Training Commands

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.run_training_suite \
  --methods planner_residual_attention_v4,planner_residual_map_tv_v4 \
  --seeds 0,1,2,3,4 \
  --output runs/scheme_d_paper_suite/planner_residual_v4_commands.txt
```

## v4 Evaluation Command

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies nearest,oldest,distance_age,risk_aware,dynamic_weighted,horizon2,horizon3,aco_tsp,ppo \
  --model runs/scheme_d_paper_suite/planner_residual_attention_v4_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  --stages multi_low,multi_realistic,multi_hard \
  --seeds 100,101,102 \
  --episodes 10 \
  --steps 800 \
  --output-dir runs/matrix/planner_residual_attention_v4_seed0_eval
```

Aggregate:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.aggregate_results \
  runs/matrix/planner_residual_attention_v4_seed0_eval/all_rows.csv \
  --output runs/matrix/planner_residual_attention_v4_seed0_eval/aggregate_summary.csv
```

## If v4 Still Fails

Do not blindly increase PPO capacity. Follow this order:

1. Compare `ppo` against its own base planner `horizon2` on the exact same seeds.
2. If `ppo` is worse, reduce `residual_beta` from `0.20` to `0.10` and keep the residual guard enabled.
3. If PPO updates still damage the base, lower `ppo_lr` below `8e-6`, reduce `ppo_clip` below `0.04`, or increase `bc_supervised_min_coef`.
4. If all methods fail on `multi_hard`, make the hard stage less severe before claiming a learning problem.
5. If all baselines are similar on easy stages, emphasize `multi_hard`, material metrics, latency, and generalization rather than only coverage.

## Future Paper Work

Required before writing strong claims:

- Run rule/planner-baseline matrix with at least 5 seeds.
- Run v4 or successor method with at least 5 seeds.
- Run ablations:
  - no BC,
  - no attention,
  - no material map,
  - no TV reward,
  - no LSTM,
  - no potential shaping,
  - no action penalty.
- Run `--max-steams 3,4,6` generalization.
- Generate plots from final CSVs.
- Generate qualitative trajectories and material heatmaps.
- Update `PAPER_DRAFT.md` only from aggregated results.

## Paper Positioning

If the final learned method beats rules:

- claim attention-based risk-aware PPO improves multi-steam coverage and material diagnostics over selected rule and PPO baselines.

If the final learned method does not beat rules:

- write a narrower paper or report section about reproducible RL pipeline construction, strong rule baselines, and the difficulty of improving over risk-aware heuristics in persistent target coverage.

Do not overclaim.
