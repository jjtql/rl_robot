# Method And Innovation

## Current Innovation Angle

The intended paper contribution after the v4 upgrade is:

```text
Planner-guided risk-aware set-attention residual RL for persistent dynamic multi-steam coverage.
```

The core idea is not simply "use PPO". The project combines:

- persistent target semantics where steam points stay until physically covered,
- risk-aware target scoring using distance, age, material state, and reachability,
- behavior cloning from strong rule experts,
- PPO fine-tuning for continuous robot control,
- unordered steam-set attention for variable multi-target reasoning,
- material-quality metrics and optional material-map/TV shaping.
- receding-horizon and ACO/TSP-style planner baselines,
- residual PPO that learns a bounded correction around a strong base action.

The claim must remain narrow until experiments beat strong baselines. If the learned policy only matches or loses to rules, frame the work as a diagnostic/engineering result and continue method development.

## Risk-Aware Target Reasoning

The environment can select a target steam point with a hand-crafted risk-aware score. The score combines:

- persistent age score: older active steam has been neglected longer,
- distance score: closer targets can often be covered faster,
- local material-gap score: regions with material deficit are prioritized,
- reachability score: target positions nearer feasible/home workspace are preferred.

The risk-aware selector is used in three roles:

1. Strong rule baseline.
2. Behavior-cloning expert.
3. Diagnostic feature source logged to CSV.

Do not treat the hand-crafted selector itself as the final learned contribution. It is a baseline and scaffold for learning.

## Planner-Guided Residual PPO

Direct-action PPO has repeatedly underperformed strong rules. The v4 direction changes the policy interface:

```text
executed_action = base_action + residual_beta * network_residual
```

The base action can be `risk_aware`, `dynamic_weighted`, `horizon2`, `horizon3`, or `aco_tsp`. The current main setting uses `horizon2` because it is strong enough to be reviewer-credible but still cheap enough for online control.

Residual PPO uses:

- `--residual-policy`
- `--residual-base-policy horizon2`
- `--residual-beta 0.20`
- a default residual direction guard to avoid moving strongly against the base controller,
- zero-residual BC anchoring so PPO starts from the planner behavior instead of relearning low-level motion from scratch.

This gives the paper a clearer novelty story: planning prior + set attention + bounded learned residual under material-aware persistent coverage.

## Behavior Cloning And PPO

Training supports rule-policy behavior cloning before PPO. The BC warm start collects demonstrations across configurable stages, then PPO fine-tunes the policy.

Important lessons from current experiments:

- A low BC loss alone does not guarantee PPO will beat rules.
- If `bc_supervised_coef` decays to zero too early, PPO can drift away from the expert and lose multi-steam performance.
- Even when BC is retained, spending most training on single-steam stages can hurt the final multi-steam policy.
- `attention_stable_v2` kept a BC anchor but still underperformed because the curriculum spent 850 episodes on single-steam stages before reaching multi-steam.

The current code supports:

- `--bc-supervised-coef`
- `--bc-supervised-min-coef`
- `--bc-supervised-decay-steps`
- `--bc-stage-episodes`
- `--no-action-smoothing-penalty`

Use these knobs to prevent PPO drift and emphasize multi-steam behavior.

## Attention Steam Encoder

When `--steam-attention` is enabled, the observation appends an unordered steam set:

- up to 6 steam slots,
- 8 features per steam,
- valid mask for active/inactive slots.

Per-steam features include:

- relative x/y,
- normalized distance,
- persistent age score,
- material-gap score,
- reachability score,
- risk score,
- active mask.

`ACModel_AttentionLSTM` embeds steam items with a shared encoder, applies multi-head attention, masks inactive entries, pools valid steam features, fuses with the 35-D base observation, and feeds an LSTM actor-critic.

This is the main representational innovation because it avoids treating active steam points as a single selected target or a brittle fixed ordering.

## Attention-Map-TV Candidate

`--material-map` appends a compact 3-channel material/frontier map:

- normalized underfilled material gap,
- normalized overfill,
- local frontier/height variation.

`--material-tv-reward` adds shaping from improvement in material quality:

- material hole loss,
- total-variation loss,
- overfill loss.

Use these variants:

- `attention`: attention over steam set only.
- `attention_map`: attention plus material map representation.
- `attention_map_tv`: attention plus material map plus TV/material-quality reward.

`attention_map_tv` is a promising candidate only if it improves material metrics without harming coverage. It must be compared against `attention_map` and `attention`.

## Known Failure Modes

### PPO Drift From Expert

Observed in `attention_stable_seed0`:

- `latest` checkpoint underperformed rules.
- `bc_ready` and some middle checkpoints were better than `latest`.
- Original BC auxiliary coefficient decayed to zero around 241,600 steps.

Mitigation:

- keep a nonzero `bc_supervised_min_coef`,
- reduce PPO learning rate and clip,
- reduce PPO epochs,
- evaluate multiple checkpoints, not only `latest`.

### Over-Curriculum On Single Steam

Observed in `attention_stable_v2_seed0`:

- `single_easy` and `single_precision` were stable,
- `multi_low` averaged about 0.791 train coverage,
- `multi_realistic` averaged about 0.687 train coverage.

Mitigation:

- start training directly on `multi_low` and `multi_realistic`,
- keep BC demonstrations mostly from multi-steam stages,
- evaluate `attention_multisteam_bc_v3`.

### Action Penalty Fighting Rule-Like Motion

Rule policies output direct saturated actions toward targets. PPO variants with action smoothing/L2 penalties may learn less decisive motion and higher target distance.

Mitigation:

- test `--no-action-smoothing-penalty`,
- compare action L2/action delta against rules,
- do not assume lower action magnitude is better if coverage drops.

## Method Comparison Ladder

Use this comparison order:

1. Rule policies: `random`, `nearest`, `oldest`, `distance_age`, `risk_aware`.
2. PPO baselines: `nearest`, `risk_aware`.
3. Attention PPO: `attention`.
4. Stable attention PPO: `attention_multisteam_bc_v3`.
5. Map/material variants: `attention_map`, `attention_map_tv`, `attention_map_tv_multisteam_bc_v3`.
6. Ablations: no BC, no LSTM, no material observation, no potential shaping, no best-progress, no action penalty.
7. Stronger future baselines: receding-horizon planner, SAC, TD3.
8. v4 baselines: `dynamic_weighted`, `horizon2`, `horizon3`, `aco_tsp`.
9. Main v4 methods: `planner_residual_attention_v4`, `planner_residual_map_tv_v4`.

Paper claims should only compare methods with identical seeds, stages, episode budgets, and step budgets.
