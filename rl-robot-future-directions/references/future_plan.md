# Future Plan For rl_robot

## Paper Strategy

A realistic paper angle:

```text
Learning residual control for persistent multi-steam coverage under thermal
spawn dynamics, using spawn-history observation and LSTM attention over active
steam points, anchored by a two-step receding-horizon planner.
```

This is a safer framing than claiming that `horizon2` itself is the invention.

Suggested contribution statements:

- Task formulation: persistent multi-steam coverage with material-quality diagnostics and thermal spawn dynamics.
- Method: attention-LSTM residual PPO over a deterministic two-step receding-horizon base.
- Evidence: improved or matched planner coverage on moderate multi-steam stages, with limitations under extreme spawn pressure.

## Formal Evaluation Matrix

Use at least 5 seeds for the important rows before writing final claims.

Recommended policies:

- `nearest`
- `oldest`
- `distance_age`
- `risk_aware`
- `dynamic_weighted`
- `horizon2`
- `ppo` full method

Recommended stages:

- `multi_low`
- `multi_realistic`
- `multi_hard`
- `multi_extreme`

Recommended evaluation command shape:

```bash
.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies nearest,oldest,distance_age,risk_aware,dynamic_weighted,horizon2,ppo \
  --stages multi_low,multi_realistic,multi_hard,multi_extreme \
  --seeds 100,101,102,103,104 \
  --episodes 30 \
  --steps 800 \
  --model runs/scheme_d_paper_suite/thermal_lstm_spawnhist_v3_horizon2_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  --output-dir runs/matrix/thermal_lstm_spawnhist_v3_horizon2_seed0_formal_eval
```

If GPU is unavailable locally and checkpoint config forces CUDA, create CPU checkpoint copies under `/tmp` for evaluation. Do not overwrite original checkpoints.

## Main Tables

Good result tables:

1. Coverage and reward by stage and method.
2. Latency and covered count by stage and method.
3. Material quality, hole loss, TV loss, and overfill by stage and method.
4. Ablation table on `multi_realistic` and `multi_hard`.
5. Generalization table for `max_steams=3,4,6` if available.

Plot ideas:

- Coverage curves from training.
- Held-out coverage bar chart with standard deviation.
- Latency distribution by method.
- Material loss comparison.

## Ablations

Priority order:

1. Full method vs `horizon2` only.
2. Full method vs no LSTM.
3. Full method vs no spawn-history observation.
4. Full method vs no steam attention.
5. Full method vs no BC warm start.
6. Full method vs residual base `risk_aware` or `dynamic_weighted`.

If time is limited, run only the first three but be conservative in claims.

## Training Variants To Try

Potential useful variants:

- `residual_beta=0.1`: more conservative residual, may preserve horizon2 strengths.
- `residual_beta=0.3`: more learning freedom, may help hard stages but can hurt stability.
- Shorter or softer extreme curriculum: current `multi_extreme` performance suggests the method struggles there.
- More material-aware reward: use material hole loss or TV loss if quality metrics matter in the paper.
- Stronger held-out randomization: vary thermal hotspot count, lifetime, and suppression.

## Writing Guidance

Safe title patterns:

- "Residual Receding-Horizon Reinforcement Learning for Persistent Multi-Steam Coverage"
- "Attention-LSTM Residual Control for Thermal Multi-Steam Coverage"

Safe abstract language:

- "We study..."
- "We propose a task-specific residual controller..."
- "The learned residual policy improves moderate multi-steam coverage in held-out evaluation..."
- "Extreme spawn pressure remains challenging..."

Reviewer-risk language to avoid:

- "no baseline"
- "only reports our success rate"
- "horizon2 is a new neural network"
- "complete solution"
- "state of the art"

If the current results are used without more training, write the paper as a small applied feasibility study, not a strong algorithmic paper.
