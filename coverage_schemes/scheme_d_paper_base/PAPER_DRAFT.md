# Risk-Aware Steam Coverage With Persistent Targets

## Abstract

We study a MuJoCo robotic coverage task in which active steam points indicate under-covered material regions. Steam points persist until physically covered by the green cover marker; they do not expire with age. The proposed method uses risk-aware target selection, behavior-cloning warm start, and PPO fine-tuning to improve multi-steam coverage under material uniformity constraints.

## Problem Setting

The robot controls a 3D action vector for planar coverage motion and speed scaling. At each step, the environment tracks active steam points, material height on a grid, cover latency, coverage rate, and action smoothness. A steam point is removed only when the cover marker reaches it. The `max_steam_age` value is used only to normalize age-related observations and rewards.

Main evaluation stages:

- `multi_low`: low-density multi-steam setting.
- `multi_realistic`: higher-density multi-steam setting.
- Generalization: override `max_steams` with 3, 4, and 6.

## Method

The risk-aware selector scores each active steam point with:

- persistent age score,
- distance score from the cover marker,
- local material-gap score,
- reachability score from the home end-effector position.

The PPO observation includes the selected target direction, normalized target distance, velocity toward the selected target, and selected-target risk features. The selector can be switched between `risk_aware` and `nearest` for ablation with `--target-selector`.

The attention variant keeps the same PPO and LSTM training loop, but adds a steam-set attention encoder before the LSTM. When `--steam-attention` is enabled, the observation appends up to six active steam items, each represented by relative position, distance, persistent age score, material-gap score, reachability score, hand-crafted risk score, and an active mask. Shared item embeddings, multi-head attention, and masked pooling produce a permutation-invariant steam context that is fused with the base observation before the recurrent actor-critic.

Training uses optional behavior cloning from rule experts before PPO. The default warm start uses the risk-aware rule expert and collects demonstrations across single and multi-steam stages.

## Baselines And Ablations

Rule baselines:

- random,
- nearest,
- oldest,
- distance-age,
- risk-aware.

PPO variants:

- risk-aware PPO,
- attention risk-aware PPO,
- nearest-target PPO,
- no behavior cloning,
- no curriculum,
- no LSTM,
- no potential shaping,
- no best-progress reward,
- no material observation,
- no action smoothing/L2 penalty.

## Metrics

Report mean and standard deviation across seeds:

- episode reward,
- coverage rate,
- success count,
- spawned count,
- missed count,
- cover latency,
- target distance,
- selected target risk score,
- material height uniformity,
- overfill penalty,
- action delta mean,
- action L2 mean.

## Reproducibility Commands

Train a risk-aware PPO seed:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.train \
  --run-dir runs/scheme_d_paper_base \
  --run-name attention_seed0 \
  --seed 0 \
  --target-selector risk_aware \
  --bc-policy risk_aware \
  --steam-attention \
  --headless
```

Evaluate a checkpoint:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.eval \
  --config runs/scheme_d_paper_base/risk_aware_seed0/config.json \
  --policy ppo \
  --model runs/scheme_d_paper_base/risk_aware_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  --stage multi_realistic \
  --episodes 30 \
  --steps 800 \
  --seed 100 \
  --output runs/eval/risk_aware_seed0_multi_realistic.csv
```

Run rule baseline generalization:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies nearest,oldest,distance_age,risk_aware \
  --stages multi_low,multi_realistic \
  --seeds 0,1,2,3,4 \
  --episodes 30 \
  --steps 800 \
  --max-steams 3,4,6 \
  --output-dir runs/matrix/rule_generalization
```

Generate plots:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.plot_results \
  runs/matrix/rule_generalization/all_rows.csv \
  --output-dir runs/plots/rule_generalization \
  --plot
```

Record a qualitative rollout:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.visualize_episode \
  --policy ppo \
  --model runs/scheme_d_paper_base/risk_aware_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  --config runs/scheme_d_paper_base/risk_aware_seed0/config.json \
  --stage multi_realistic \
  --steps 800 \
  --seed 200 \
  --output-dir runs/qualitative/risk_aware_seed0
```

## Results Tables

### Main Multi-Seed Results

| Method | Stage | Seeds | Coverage Rate | Missed Count | Cover Latency | Reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Risk-aware PPO | multi_low | TBD | TBD | TBD | TBD | TBD |
| Risk-aware PPO | multi_realistic | TBD | TBD | TBD | TBD | TBD |
| Attention risk-aware PPO | multi_low | TBD | TBD | TBD | TBD | TBD |
| Attention risk-aware PPO | multi_realistic | TBD | TBD | TBD | TBD | TBD |
| Nearest PPO | multi_low | TBD | TBD | TBD | TBD | TBD |
| Nearest PPO | multi_realistic | TBD | TBD | TBD | TBD | TBD |

### Ablation Results

| Variant | Stage | Coverage Rate | Missed Count | Uniformity | Action Delta | Reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full risk-aware PPO | multi_realistic | TBD | TBD | TBD | TBD | TBD |
| No BC | multi_realistic | TBD | TBD | TBD | TBD | TBD |
| No LSTM | multi_realistic | TBD | TBD | TBD | TBD | TBD |
| No potential shaping | multi_realistic | TBD | TBD | TBD | TBD | TBD |
| No material observation | multi_realistic | TBD | TBD | TBD | TBD | TBD |

### Generalization Results

| Method | Stage | Max Steams | Coverage Rate | Missed Count | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Risk-aware PPO | multi_realistic | 3 | TBD | TBD | TBD |
| Risk-aware PPO | multi_realistic | 4 | TBD | TBD | TBD |
| Risk-aware PPO | multi_realistic | 6 | TBD | TBD | TBD |

## Current Gaps

- Full multi-seed PPO training has not been run.
- Long-horizon evaluation tables are not populated.
- Paper figures should be regenerated from final run directories, not smoke runs.
