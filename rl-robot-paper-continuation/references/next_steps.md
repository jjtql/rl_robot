# Next Steps For The rl_robot Paper Pipeline

## Current Artifacts

- Original verified baseline: `coverage_schemes/scheme_d_fixed copy`
- Working paper package: `coverage_schemes/scheme_d_paper_base`
- Local continuation skill: `rl-robot-paper-continuation`
- Global monitor skill/script: `/home/jiangjian/.codex/skills/rl-paper-pipeline/scripts/monitor_run.py`
- Python environment: `mujoco_rl_env/bin/python`

## What Is Already Done

- Clean baseline package created.
- Configurable model path and seed setup added.
- Headless training with CSV/JSONL logs added.
- Headless evaluation added.
- Rule policies added: random, nearest, oldest, distance-age, risk-aware.
- Steam points persist until covered; there is no automatic timeout disappearance.
- PPO checkpoint evaluation added.
- Matrix runner added.
- Result aggregation added.
- Monitor script tested with train then post-eval.
- Attention risk-aware PPO has been implemented as an experimental variant:
  - `env.py` exposes a full steam-set observation when `steam_attention_observation=True`.
  - `algo.py` includes `ACModel_AttentionLSTM`.
  - `train.py --steam-attention` enables the attention encoder and stores the flag in checkpoints.
  - `run_matrix.py --max-steams 3,4,6` supports active-steam generalization tests.

## Immediate Next Steps

1. Run the rule baseline matrix:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies nearest,oldest,distance_age,risk_aware \
  --stages multi_low,multi_realistic \
  --seeds 0,1,2,3,4 \
  --episodes 30 \
  --steps 800 \
  --output-dir runs/matrix/rule_baselines
```

2. Start monitored risk-aware PPO baseline seed 0:

```bash
python3 /home/jiangjian/.codex/skills/rl-paper-pipeline/scripts/monitor_run.py \
  --name ppo_baseline_seed0 \
  --cwd /mnt/d/robot_scit/rl_robot \
  --log-dir runs/monitors \
  --heartbeat-seconds 3600 \
  --tail-lines 10 \
  --post-command "mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.eval --policy ppo --model runs/scheme_d_paper_base/seed0/checkpoints/scheme_d_paper_base_latest_full.pt --stage multi_realistic --episodes 30 --steps 800 --seed 100 --output runs/eval/ppo_baseline_seed0_multi_realistic.csv" \
  -- mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.train \
    --run-dir runs/scheme_d_paper_base \
    --run-name seed0 \
    --seed 0 \
    --headless
```

3. Start monitored attention risk-aware PPO seed 0:

```bash
python3 /home/jiangjian/.codex/skills/rl-paper-pipeline/scripts/monitor_run.py \
  --name attention_ppo_seed0 \
  --cwd /mnt/d/robot_scit/rl_robot \
  --log-dir runs/monitors \
  --heartbeat-seconds 3600 \
  --tail-lines 10 \
  --post-command "mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.eval --policy ppo --model runs/scheme_d_attention/seed0/checkpoints/scheme_d_paper_base_latest_full.pt --stage multi_realistic --episodes 30 --steps 800 --seed 100 --output runs/eval/attention_ppo_seed0_multi_realistic.csv" \
  -- mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.train \
    --run-dir runs/scheme_d_attention \
    --run-name seed0 \
    --seed 0 \
    --steam-attention \
    --headless
```

4. Repeat monitored training for seeds 1-4 after seed 0 is healthy.

5. Evaluate all PPO checkpoints using `eval.py` or `run_matrix.py`.

6. Add ablations only after the baseline training/evaluation pipeline is stable.

## Second-Stage Method Direction

The paper contribution should focus on attention-based risk-aware PPO for multi-steam coverage. The final experiments should emphasize `multi_low` and `multi_realistic`; single-steam stages are not enough to prove the method.

Risk-aware score components:

- distance to current cover center
- steam age ratio
- persistent age or neglect urgency
- material height gap around the steam
- reachability from home/workspace constraints

Attention method status:

- Keep the hand-crafted risk-aware selector as a baseline and behavior-cloning expert.
- Use the attention steam encoder as the main learned priority mechanism.
- Compare attention PPO against current risk-aware PPO, nearest-target PPO, and rule policies under identical seeds and episode budgets.
- Treat improved multi-steam coverage, lower latency, stable material uniformity, and lower overfill as the main evidence.

Suggested claim:

> An attention-based risk-aware PPO policy improves multi-steam coverage by learning priority over an unordered steam set while retaining risk/material diagnostics, improving coverage stability compared with nearest-target, oldest-target, hand-crafted risk-aware, and non-attention PPO baselines.

## Stronger Baselines And Future Optimization

Add these only after the current attention PPO pipeline is reproducible:

- Receding-horizon planner: plan short target/action sequences using distance, age, material gap, reachability, active-steam count, and predicted spawn pressure. This is the strongest rule-style baseline to answer reviewer concerns.
- SAC or TD3 baseline: compare continuous-control algorithms using the same observations, stages, seeds, and evaluation metrics. Add them after PPO logging and evaluation are stable.
- Stronger map state: add a compact egocentric material/frontier map instead of relying only on global material summary statistics. Keep the current steam-set features for target priority.
- Coverage-hole reward: test total-variation or local-uniformity penalties to reduce uncovered holes and overfill.
- Attention ablations: no attention, attention without material features, attention with nearest target selector, attention without BC, and attention without LSTM.
- Generalization: evaluate `max_steams=3,4,6`, different spawn intensities, and held-out random seeds.

## Paper Minimum Bar

- Full method beats random and simple rule policies on multi-steam settings.
- Full method beats the hand-crafted risk-aware rule and current non-attention PPO before making a method contribution claim.
- Full method beats at least two ablations.
- Results use at least 5 seeds for key methods.
- Evaluation reports mean/std.
- Training and evaluation commands are reproducible.
- Claims stay narrower than the evidence.
