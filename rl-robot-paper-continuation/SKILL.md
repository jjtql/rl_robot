---
name: rl-robot-paper-continuation
description: Continue the rl_robot MuJoCo reinforcement-learning paper pipeline from the current scheme_d_paper_base state. Use when working in /mnt/d/robot_scit/rl_robot on training, hourly monitoring, post-training evaluation, baseline matrices, ablations, attention-based risk-aware PPO, stronger rule/planner/SAC/TD3 baselines, multi-steam generalization, result aggregation, or paper-ready next steps for the coverage_schemes/scheme_d_paper_base package.
---

# RL Robot Paper Continuation

## Project Context

Use this skill only inside the `rl_robot` project. The validated original baseline is `coverage_schemes/scheme_d_fixed copy`; keep it unchanged. The paper-ready working package is `coverage_schemes/scheme_d_paper_base`.

Always run Python through the project virtual environment:

```bash
mujoco_rl_env/bin/python
```

For a detailed checklist, read `references/next_steps.md`.

## Current State

The first paper-pipeline phase is already implemented:

- `scheme_d_paper_base` is a clean package copied from the verified `scheme_d_fixed copy`.
- `env.py` supports configurable model paths and defaults to repo-root `o1.xml`.
- Steam points persist until covered. Do not design methods around timeout-based disappearance or miss-on-age behavior.
- `train.py` supports CLI overrides, fixed seeds, headless training, run-local configs, CSV episode logs, JSONL events, and checkpoints.
- `eval.py` runs headless evaluation for PPO checkpoints and rule policies.
- `policies.py` includes `random`, `nearest`, `oldest`, `distance_age`, `risk_aware`, and `ppo`.
- `run_matrix.py` runs multi-policy, multi-stage, multi-seed evaluation matrices.
- `aggregate_results.py` aggregates CSV results into mean/std tables.
- `PAPER_TODO.md` in `scheme_d_paper_base` documents smoke commands and monitoring commands.
- Attention PPO is already implemented as an experimental variant:
  - `env.py` enables full steam-set observation with `steam_attention_observation=True`.
  - The attention set has `attention_steam_count=6` and `attention_steam_dim=8`.
  - Per-steam features are relative xy, normalized distance, persistent age score, material-gap score, reachability score, risk score, and a valid mask.
  - `algo.py` includes `ACModel_AttentionLSTM`, using shared per-steam embeddings, multi-head attention, pooled set features, and LSTM PPO.
  - `train.py --steam-attention` enables the attention-LSTM policy and records `use_steam_attention` in checkpoints.
  - Default config keeps `use_steam_attention=False` so older 35-D PPO checkpoints remain compatible.
- `run_matrix.py --max-steams 3,4,6` supports multi-steam generalization tests.

Known validation already performed:

- `mujoco_rl_env/bin/python -m compileall coverage_schemes/scheme_d_paper_base`
- short train smoke test
- short eval tests for rule policies and PPO checkpoints
- short matrix run
- monitor-run smoke test with post-training eval

## Execution Rules

Do not edit `coverage_schemes/scheme_d_fixed copy` unless explicitly asked. Make new work in `coverage_schemes/scheme_d_paper_base` or a new experimental package.

Use `multi_low` and `multi_realistic` as the main paper settings. Single-steam stages are for curriculum and sanity checks only; the risk-aware contribution requires multiple active steam points.

Respect the no-timeout task assumption: steam points do not vanish automatically. Age is a persistence/neglect feature, not an expiration countdown.

Keep outputs under `runs/`. Prefer one run directory per seed, method, and stage.

Position the paper contribution narrowly as attention-based risk-aware PPO for dynamic multi-steam coverage. Do not claim broad SOTA unless the method beats stronger baselines and ablations. Treat the hand-crafted risk score as a rule baseline and optional BC expert, not as the final method.

When comparing methods, use this ladder before writing claims:

1. Simple rules: `random`, `nearest`, `oldest`.
2. Stronger hand-crafted rules: `distance_age`, `risk_aware`.
3. Current PPO variants: nearest-target PPO, risk-aware-target PPO.
4. Main method: attention risk-aware PPO with full steam-set observation.
5. Strong engineering baselines when time allows: receding-horizon planner, SAC, TD3.

When training for real, use the monitor script so hourly summaries are compact and full logs stay on disk:

```bash
python3 /home/jiangjian/.codex/skills/rl-paper-pipeline/scripts/monitor_run.py \
  --name paper_seed0 \
  --cwd /mnt/d/robot_scit/rl_robot \
  --log-dir runs/monitors \
  --heartbeat-seconds 3600 \
  --tail-lines 10 \
  --post-command "mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.eval --policy ppo --model runs/scheme_d_paper_base/seed0/checkpoints/scheme_d_paper_base_latest_full.pt --stage multi_realistic --episodes 30 --steps 800 --seed 100 --output runs/eval/paper_seed0_multi_realistic.csv" \
  -- mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.train \
    --run-dir runs/scheme_d_paper_base \
    --run-name seed0 \
    --seed 0 \
    --headless
```

## Next Work Order

1. Run rule-baseline matrices on `multi_low` and `multi_realistic`:
   - `nearest`
   - `oldest`
   - `distance_age`
   - `risk_aware`
2. Train and evaluate current PPO baselines:
   - nearest-target PPO using `--target-selector nearest`
   - risk-aware-target PPO using `--target-selector risk_aware`
3. Train and evaluate attention risk-aware PPO:
   - use `--steam-attention`
   - keep `--target-selector risk_aware` unless explicitly testing the attention-only ablation
   - compare against the same rule and PPO baselines with identical stages, seeds, and episode budgets
4. Run ablations:
   - no behavior cloning
   - no curriculum
   - no LSTM
   - no potential shaping
   - no best-progress anti-farming
   - no material observation
   - no action smoothing penalty
   - no steam-set attention
   - target selector nearest with attention enabled
5. Run generalization tests:
   - `max_steams=3,4,6`
   - train on the default setting, evaluate on larger active-steam counts before claiming robustness
6. Add a receding-horizon planner baseline if attention PPO only slightly beats simple rules:
   - score short action/target sequences with distance, age, material gap, reachability, active-steam count, and predicted spawn pressure
   - report whether attention PPO beats this planner or only beats naive rules
7. Add SAC or TD3 if time allows:
   - use the same observation toggles and evaluation matrix
   - keep PPO as the main method only if it remains competitive after reasonable tuning
8. Run at least 5 seeds for the important methods before making paper claims.

## Method Upgrade Notes

Use the following research directions to guide future code changes:

- Stronger state representation: move beyond a single selected target by exposing the complete steam set plus material state. The current attention variant already exposes the steam set; a later version can add a compact egocentric material/frontier map and coverage-hole or total-variation style reward terms.
- Set/attention encoding: treat active steam points as an unordered set. Prefer shared per-item encoders plus Deep Sets or Set Transformer style pooling over manually sorting targets as a learned priority mechanism.
- Risk-aware learning: keep distance, persistent age, material gap, and reachability as logged features. Use them for rule baselines, BC experts, diagnostics, and ablations, but let the attention policy learn priority instead of hard-coding all target selection.
- Planner baseline: implement a receding-horizon planner before claiming large method gains if RL only beats nearest/oldest. This is the strongest practical rule baseline for reviewers.
- Strong RL baselines: add SAC/TD3 only after the current PPO/attention pipeline is reproducible; otherwise training instability will obscure the method comparison.
- Literature positioning: cite coverage RL work with map/frontier representations and coverage-hole penalties, Deep Sets/Set Transformer for unordered target sets, continuous-control baselines such as SAC/TD3, and additive-manufacturing path-planning baselines. Verify exact citations before adding them to the paper.

## Minimal Commands

Rule baseline matrix:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies nearest,oldest,distance_age,risk_aware \
  --stages multi_low,multi_realistic \
  --seeds 0,1,2,3,4 \
  --episodes 30 \
  --steps 800 \
  --output-dir runs/matrix/rule_baselines
```

Short PPO smoke training:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.train \
  --run-dir runs/smoke_train \
  --run-name smoke \
  --seed 0 \
  --episode-steps 20 \
  --stage-episodes single_easy:1,multi_low:1 \
  --no-bc \
  --headless
```

Attention PPO smoke training:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.train \
  --run-dir runs/smoke_train \
  --run-name attention_smoke \
  --seed 0 \
  --episode-steps 20 \
  --stage-episodes single_easy:1,multi_low:1 \
  --steam-attention \
  --no-bc \
  --headless
```

Training-suite command generation:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.run_training_suite \
  --methods attention,risk_aware,nearest,no_bc,no_lstm,no_potential,no_material_obs \
  --seeds 0,1,2,3,4 \
  --output runs/scheme_d_paper_suite/commands.txt
```

Generalization matrix:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies nearest,oldest,distance_age,risk_aware \
  --stages multi_low,multi_realistic \
  --seeds 0,1,2,3,4 \
  --episodes 30 \
  --steps 800 \
  --max-steams 3,4,6 \
  --output-dir runs/matrix/rule_baselines_max_steams
```

Aggregate CSV files:

```bash
mujoco_rl_env/bin/python -m coverage_schemes.scheme_d_paper_base.aggregate_results \
  runs/matrix/rule_baselines/all_rows.csv \
  --output runs/matrix/rule_baselines/aggregate_summary.csv
```
