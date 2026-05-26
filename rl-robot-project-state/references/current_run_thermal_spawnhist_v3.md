# Current Run: thermal_lstm_spawnhist_v3_horizon2_seed0

## Location

```text
runs/scheme_d_paper_suite/thermal_lstm_spawnhist_v3_horizon2_seed0
```

Completed:

- Episodes: 1250
- Steps: 1,000,000
- Elapsed time: about 11,472.6 seconds
- Final event: `done`

## Configuration Snapshot

Main settings from `config.json`:

- `use_steam_attention: true`
- `use_lstm: true`
- `thermal_spawn: true`
- `use_spawn_history_observation: true`
- `residual_policy: true`
- `residual_base_policy: horizon2`
- `residual_beta: 0.2`
- `bc_policy: horizon2`
- `bc_supervised_coef: 0.14`
- `bc_supervised_min_coef: 0.05`
- `bc_supervised_decay_steps: 900000`
- `episode_steps: 800`
- `device: cuda`

Curriculum:

- `multi_low`: 150 episodes
- `multi_realistic`: 250 episodes
- `multi_hard`: 700 episodes
- `multi_extreme`: 150 episodes

Behavior cloning warm start:

- BC episodes: 220
- BC epochs: 12
- BC stages: `multi_low:30`, `multi_realistic:50`, `multi_hard:100`, `multi_extreme:40`

## Training-Window Statistics

These are from `episodes.csv` and should not be treated as held-out test results.

| Stage | Episodes | All Coverage | Last 50 Coverage | Best 50 Coverage |
| --- | ---: | ---: | ---: | ---: |
| multi_low | 150 | 0.793 | 0.799 | 0.809 |
| multi_realistic | 250 | 0.727 | 0.730 | 0.754 |
| multi_hard | 700 | 0.582 | 0.565 | 0.623 |
| multi_extreme | 150 | 0.493 | 0.501 | 0.508 |

Notable windows:

- `multi_realistic` best-50 coverage: about 0.754 around global episodes 305-354.
- `multi_hard` best-50 coverage: about 0.623 around global episodes 611-660.
- `multi_hard` best-50 reward: about 337.2 around global episodes 852-901.

## PPO Training Signals

From `events.jsonl`:

- PPO updates: 625
- No crash recorded
- Final entropy: about 0.619
- Final BC supervised coefficient: 0.05
- Final critic loss: about 0.022
- Actor loss stays very small near the end

Interpretation: training is stable, but the residual PPO is modest. The policy appears anchored strongly to the horizon2 base rather than learning a radically different controller.

## Quick Held-Out Evaluation

Quick matrix output:

```text
runs/matrix/thermal_lstm_spawnhist_v3_horizon2_seed0_quick_eval/aggregate_summary_all_material.csv
```

Setup:

- Policies: `risk_aware`, `dynamic_weighted`, `horizon2`, `ppo`
- Stages: `multi_low`, `multi_realistic`, `multi_hard`, `multi_extreme`
- Seeds: 100, 101, 102
- Episodes per seed: 5
- Total episodes per policy-stage: 15
- Steps: 800

Coverage and reward means:

| Stage | PPO Coverage | Horizon2 Coverage | Best Rule Coverage | PPO Reward | Horizon2 Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi_low | 0.960 | 0.960 | 0.960 | 260.8 | 265.3 |
| multi_realistic | 0.828 | 0.819 | 0.803 | 282.9 | 270.2 |
| multi_hard | 0.600 | 0.595 | 0.551 | 296.8 | 301.7 |
| multi_extreme | 0.476 | 0.491 | 0.497 | 298.2 | 312.2 |

Material quality loss means:

| Stage | PPO | Horizon2 |
| --- | ---: | ---: |
| multi_realistic | 0.588 | 0.586 |
| multi_hard | 0.513 | 0.509 |
| multi_extreme | 0.545 | 0.538 |

## Interpretation

This run is useful but not yet enough for a strong method claim.

Positive signals:

- Clear small win on `multi_realistic` coverage and reward over horizon2 in the quick held-out matrix.
- Slight coverage edge over horizon2 on `multi_hard`.
- Latest checkpoint looked best among quick hard-stage checks for held-out coverage/reward.

Weak signals:

- `multi_extreme` is worse than horizon2 and dynamic weighted rules.
- Material losses are not clearly improved over horizon2.
- Gains are small and based on only 15 held-out episodes per policy-stage.

Recommended paper language:

- Say the current method shows feasibility and modest gains on moderate multi-steam settings.
- Do not claim broad superiority over planners yet.
- Always compare against `horizon2` because the learned policy uses it as its residual base and BC expert.
