# Experiment Version History

Last updated: 2026-06-04

This document consolidates the main historical versions of the ShangZeng steam coverage project. It is intended to preserve what each version changed, what it proved, what it failed to prove, and which results are safe to use in the paper.

The core constraint across these versions is unchanged:

```text
Keep the LSTM-PPO core. Planner, reward, observation, residual glue, spawn process, metrics, and evaluation protocol may change.
```

## Current Interpretation

The task should not be treated as many isolated short coverage games. A realistic ShangZeng run is a long physical session:

```text
thermal accumulation -> burst -> lull -> accumulation -> burst -> lull -> ...
terminal lull -> final drain/clear
```

Therefore the project now separates two ideas that were previously mixed together:

- final coverage: whether the controller eventually covers generated steam points,
- timely service: whether each steam point is covered soon after it appears.

Final coverage can become high if the terminal drain is long enough. The harder and more meaningful metric is timely service.

## Metric Definitions

| Metric | Meaning | Main use |
| --- | --- | --- |
| `coverage_rate` / `Cov` | `covered_count / spawned_count` | Eventual coverage. Useful but no longer sufficient. |
| `response_sla_success_rate` / `SLA` | SLA-covered points divided by covered points | Helpful diagnostic, but can look good when uncovered points remain. |
| `strict_response_sla_success_rate` | SLA-covered points divided by spawned points | Strict deadline service metric. |
| `effective_coverage_rate` / `Eff` | Alias of strict SLA success rate | Main latency-aware paper metric. |
| `cover_latency_seconds` | Mean age at coverage, in seconds | Average response time for covered points. |
| `cover_latency_p90_seconds` | P90 age at coverage, in seconds | Tail response time. |
| `pending_steam_count` / `Act` | Spawned minus covered minus missed | Remaining active backlog at the end. |
| `full_session_terminal_clear` / `Clear` | Whether terminal drain ended with zero active steam | Full-session completion quality. |

Important latency rule:

```text
latency_i = cover_time_i - spawn_time_i
```

Quiet waiting before a steam point appears is not counted as that point's latency. A high P90 means points had already appeared and were left waiting.

## Version Timeline

### Pre-v11: Coverage-First Residual PPO

Early versions focused mostly on coverage. The main shape was already present:

- LSTM-PPO policy,
- behavior cloning warm start,
- planner/rule guidance,
- steam attention,
- residual action around a stronger hand-written base.

A key negative lesson was that a learned policy does not automatically beat a strong planner. When the planner already sees current active steam points and plans well, PPO can easily damage a good base action unless the residual is carefully bounded.

Representative earlier observations:

| Stage | Release seed0 | Release seed1 | Interpretation |
| --- | ---: | ---: | --- |
| multi_low | about 0.797 | about 0.816 | Stable, near ceiling. |
| multi_realistic | about 0.728 | about 0.731 | Little improvement. |
| multi_hard | about 0.578 | about 0.589 | Small improvement. |
| multi_extreme | about 0.497 | about 0.490 | No clear improvement. |

Held-out quick eval from that era showed the same pattern: low stages could look good, but hard/extreme were not convincingly better than horizon2 or old v3 PPO.

Main lesson:

```text
More residual freedom alone is not enough. The LSTM needs predictive supervision and a task structure where history matters.
```

### V6/V9 Direction: Spawn History, Thermal Context, and Burst-Lull

This direction added or emphasized:

- keeping LSTM hidden state after coverage events,
- attention over up to 8 steam points,
- spawn-history observation,
- thermal-context observation,
- route-summary observation,
- auxiliary spawn prediction loss (`pred_coef` restored around `0.04`),
- burst-lull generation so steam appears in thermal bursts rather than uniformly.

The user requirement behind this direction was correct: the LSTM should matter most when steam is sparse, waiting, accumulating, or emerging from a remembered hot region. Horizon planning is strongest when many current active targets are already visible.

Observed V9-style result from training logs:

```text
multi_extreme last50 coverage around 0.59-0.61
```

This was not enough for a paper claim. It proved the mechanism was plausible, not that performance was strong.

### V11: Latency-First Continuous Session

Main preset:

```text
thermal_lstm_spawnhist_latency_v11
```

Main changes:

- introduced latency-first reward,
- logged latency/P90/SLA/backlog diagnostics,
- used continuous physical sessions split into chunks,
- carried LSTM hidden state across chunks,
- kept horizon2 as the residual scaffold,
- used phase-aware residual beta,
- used burst-lull thermal spawning.

Representative key settings:

```text
continuous_session_chunks = 4
carry_lstm_state_across_chunks = true
lstm_sequence_chunks = 4
response_sla_steps = 110
pred_coef = 0.05
prediction_horizon_steps = 150
residual_base_policy = horizon2
residual_glue = phase_aware
```

Main lesson:

```text
The codebase began measuring the right thing, but the physical-session definition was still too short and did not yet model the final no-steam terminal phase.
```

See also: `docs/latency-first-continuous-session-v11.md`.

### V12 Fast: Strict Real-Time SLA Diagnostic

Main preset:

```text
thermal_lstm_spawnhist_latency_v12_fast
```

Main change from V11:

- converted evaluation to a real high-level control period,
- set `decision_dt_seconds = 0.05`,
- used a strict `response_sla_seconds = 4.0`,
- increased latency and quick-cover pressure,
- reduced residual freedom in dense/burst phases.

Representative previous held-out result:

| Method | Stage | Cov | Comment |
| --- | --- | ---: | --- |
| horizon2 | hard | about 0.860 | Strong planner baseline. |
| V12 PPO | hard | about 0.871 | Slight coverage gain. |
| horizon2 | extreme | about 0.814 | Strong but not ceiling. |
| V12 PPO | extreme | about 0.866 | Best earlier coverage story. |

But P90 latency remained high:

```text
hard P90 roughly high-30s seconds
extreme P90 roughly low-40s seconds
```

Main lesson:

```text
V12 remained the strongest coverage-oriented paper candidate, but not a latency breakthrough.
```

### V13 Deadline: Oldest-First / Deadline Planner Glue

Main preset:

```text
thermal_lstm_spawnhist_latency_v13_deadline
```

Main change:

- changed base planner and BC expert toward `deadline_horizon2`,
- used 15 s SLA instead of strict 4 s,
- increased age and oldest-active pressure,
- reduced residual beta.

Training summary from historical logs:

| Stage | V13 full Cov | V13 Lat | V13 P90 | V13 SLA |
| --- | ---: | ---: | ---: | ---: |
| hard | about 0.778 | about 18.31s | about 34.38s | about 0.472 |
| extreme | about 0.726 | about 22.70s | about 42.04s | about 0.414 |

Important clarification:

```text
thermal_lstm_spawnhist_latency_v13_deadline_horizon2_base is not pure horizon2.
It is still a trained residual PPO method using horizon2 as the base planner.
```

Main lesson:

```text
Deadline-aware service helped some diagnostics, but did not clearly beat V12 as the main coverage story.
```

### V14 Rescue

Main preset:

```text
thermal_lstm_spawnhist_latency_v14_rescue
```

Main change:

- changed the base to `deadline_rescue_horizon2`,
- damped residual authority sharply,
- added emergency beta damping when a point approached SLA pressure,
- increased age pressure.

Intent:

```text
Protect old steam points from being starved by the learned residual.
```

Outcome:

```text
Useful as a safety direction, but too conservative for a strong learning claim.
```

### V15 Pathbend

Main preset:

```text
thermal_lstm_spawnhist_latency_v15_pathbend
```

Main change:

- allowed the residual to bend the base path more strongly,
- used blend mode,
- relaxed residual alignment,
- added residual supervised BC toward a secondary expert,
- introduced path-bend shield parameters.

Intent:

```text
Do not only go to the nearest/oldest target. Let LSTM slightly bend the route to pass through likely hot regions or nearby emerging points.
```

Outcome:

```text
Conceptually important, but it risked damaging the strong base planner and did not become the main candidate.
```

### V16 Corridor and V17 Corridor Base

Main presets:

```text
thermal_lstm_spawnhist_latency_v16_corridor
thermal_lstm_spawnhist_latency_v17_corridor_base
```

Main change:

- introduced `corridor_waypoint` as BC or base policy,
- tried to make the learned correction behave like corridor-aware route bending,
- V17 made corridor waypoint the actual residual base and reduced residual authority.

Outcome:

```text
Helpful for understanding route-bending, but not the cleanest final paper route.
```

### V18 Slack EDF

Main preset:

```text
thermal_lstm_spawnhist_latency_v18_slack_edf
```

Main change:

- used `slack_horizon2`, an EDF/deadline-slack style planner,
- tried to balance deadline urgency and route cost.

Outcome:

```text
Good service-scheduling idea, but still not enough by itself to solve the latency tail.
```

### V20/V21 Sticky SLA and Safe Sticky

Main presets:

```text
thermal_lstm_spawnhist_latency_v20_sticky_sla
thermal_lstm_spawnhist_latency_v21_sticky_safe
```

Main change:

- used `sticky_sla_ensemble`,
- tried to avoid target switching and planner dithering,
- V21 removed aggressive supervised/pathbend machinery and reduced residual beta.

Outcome:

```text
The safer sticky version protected planner behavior, but also reduced the learning contribution.
```

### V22/V23 RL Gate

Main presets:

```text
thermal_lstm_spawnhist_latency_v22_rl_gate
thermal_lstm_spawnhist_latency_v23_rl_gate_bc
```

Main change:

- used gate-style residual combination,
- made PPO decide how to blend/select planner behavior rather than directly pushing a large residual,
- V23 restored supervised BC pressure for the gate.

Outcome:

```text
Interesting structurally, but not yet the main paper result. It is a possible future direction if residual action learning remains unstable.
```

### V24 Urgency and V25 Route Urgency Safe

Main presets:

```text
thermal_lstm_spawnhist_latency_v24_urgency
thermal_lstm_spawnhist_latency_v24_urgency_chunk2
thermal_lstm_spawnhist_latency_v25_route_urgency_safe
thermal_lstm_spawnhist_latency_v25_route_urgency_safe_chunk2
```

Main changes:

- added EDF-style urgency scoring,
- optionally sorted attention/compact observations by urgency,
- added urgency candidate filtering for horizon planning,
- tried low-frequency PPO decision caching (`decision_interval = 2`) with emergency replanning.

Small smoke lesson:

```text
Urgency/EDF can improve pure planner service behavior in some short probes,
but directly injecting it into PPO training can confuse coverage learning.
```

This is why V24/V25 are not final candidates by themselves.

### V26 Latency Cap Diagnostic

This was a command-level variant based on V12-fast, not a named committed preset at first.

Key idea:

- set SLA around 15 s,
- increase latency and age pressure,
- reduce dense/burst residual freedom,
- keep horizon2 scaffold.

Representative smoke:

| Stage | Cov trend | Lat/P90 trend | Interpretation |
| --- | --- | --- | --- |
| low/realistic | moderate coverage | latency often under 10-16s | Useful diagnostic. |
| hard | some episodes reached high Cov | P90 around 20s in good episodes | Promising but unstable. |
| extreme | Cov around mid 0.6 in smoke | P90 sometimes under 30s | Not enough for final claim. |

Main lesson:

```text
Latency can be made to look better only by exposing the coverage/backlog trade-off.
```

### V27 Full-Session Terminal Drain

Main implementation change:

- added full-session spawn limit,
- added terminal lull / drain period,
- added strict deadline metrics,
- added `pending_steam_count`, `effective_coverage_rate`, and `full_session_terminal_clear`,
- exposed full-session options in train/eval/suite scripts.

New CLI options include:

```text
--full-session-spawn-steps
--full-session-spawn-seconds
--full-session-drain-steps
--full-session-drain-seconds
--full-session-end-on-clear
```

Important distinction:

```text
A training episode can be a chunk. A physical ShangZeng session can span many chunks.
```

Short-drain smoke result:

```text
Latency looked good, but `Act` often remained nonzero.
```

Long-drain smoke result:

```text
Hard could often clear fully, but P90 stayed high.
Extreme still left backlog in difficult cases.
```

Main lesson:

```text
Final coverage and timely coverage must be reported separately.
```

### V28 Multicycle Full Session

This is the current realistic evaluation protocol.

Command shape:

```text
episode_steps = 400
continuous_session_chunks = 8
carry_lstm_state_across_chunks = true
lstm_sequence_chunks = 8
full_session_spawn_seconds = 120.0
full_session_drain_seconds = 40.0
response_sla_seconds = 15.0
```

So one physical session is:

```text
8 chunks * 400 steps * 0.05 s = 160 s
first 120 s: burst-lull steam generation
last 40 s: terminal drain with no new steam
```

Training completed for seeds 0, 1, and 2 in:

```text
runs/v28_multicycle_full_session/train
```

#### Training Log Summary

The table below aggregates the last 5 physical sessions per seed.

| Stage | Cov | Eff@15s | Avg Lat | P90 | Clear | Pending |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| multi_low | 0.951 | 0.637 | 12.37s | 23.01s | 0.867 | 0.33 |
| multi_realistic | 0.952 | 0.469 | 16.08s | 29.31s | 0.733 | 0.60 |
| multi_hard | 0.919 | 0.406 | 19.88s | 41.12s | 0.400 | 1.27 |
| multi_extreme | 0.930 | 0.394 | 20.38s | 38.46s | 0.400 | 1.27 |

#### Held-Out Quick Eval

Held-out eval compared the trained V28 PPO checkpoints against pure horizon2 under the same full-session protocol.

| Method | Stage | Cov | Eff@15s | Avg Lat | P90 | Clear | Pending |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| horizon2 | hard | 0.978 | 0.333 | 21.46s | 38.52s | 0.60 | 0.50 |
| V28 PPO | hard | 0.987 | 0.450 | 20.91s | 42.43s | 0.73 | 0.27 |
| horizon2 | extreme | 0.983 | 0.292 | 27.16s | 49.00s | 0.80 | 0.30 |
| V28 PPO | extreme | 0.938 | 0.318 | 24.23s | 46.43s | 0.40 | 1.00 |

Interpretation:

```text
V28 improves hard-stage deadline-effective coverage and reduces pending backlog.
V28 does not solve extreme. Extreme has slightly better latency but worse coverage and clear rate than horizon2.
```

## Implementation Map

Current relevant implementation changes live in:

| File | Role |
| --- | --- |
| `coverage_schemes/scheme_d_paper_base/config.py` | Default flags for urgency, decision caching, and full-session drain. |
| `coverage_schemes/scheme_d_paper_base/env.py` | Burst-lull process, full-session terminal lull, strict SLA metrics, urgency scoring. |
| `coverage_schemes/scheme_d_paper_base/policies.py` | Planner/residual policies, urgency/EDF variants, PPO action caching hooks. |
| `coverage_schemes/scheme_d_paper_base/train.py` | CLI flags, full-session training configuration, CSV logging, session chunk handling. |
| `coverage_schemes/scheme_d_paper_base/eval.py` | Full-session eval flags and strict metric output. |
| `coverage_schemes/scheme_d_paper_base/run_training_suite.py` | Named historical presets and command generation. |

## Current Paper-Safe Claims

Safe claims:

1. The problem should be formulated as persistent service, not one-shot coverage.
2. A planner-guided LSTM-PPO residual controller can preserve high final coverage under long multicycle sessions.
3. In `multi_hard`, V28 improves strict deadline-effective coverage over horizon2 under held-out quick eval.
4. The new full-session protocol exposes an important coverage-latency trade-off that short fixed-window episodes hid.

Claims to avoid:

1. Do not claim that latency is solved.
2. Do not claim that PPO dominates horizon2 in extreme scenes.
3. Do not use raw coverage alone as the main success metric.
4. Do not imply that `SLA` alone is enough; use `Eff@15s` because it includes uncovered generated points.

## Recommended Current Command

For the current realistic full-session run:

```bash
cd /Data2/jj/rl_robot

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_training_suite   --run-dir runs/v28_multicycle_full_session/train   --methods thermal_lstm_spawnhist_latency_v12_fast   --seeds 0,1,2   --episode-steps 400   --continuous-session-chunks 8   --carry-lstm-state-across-chunks   --lstm-sequence-chunks 8   --stage-episodes multi_low:80,multi_realistic:160,multi_hard:640,multi_extreme:160   --update-episodes 2   --decision-dt-seconds 0.05   --response-sla-seconds 15.0   --full-session-spawn-seconds 120.0   --full-session-drain-seconds 40.0   --full-session-end-on-clear   --cover-latency-penalty-gain 18.0   --oldest-active-penalty-gain 0.55   --backlog-penalty-gain 0.22   --quick-cover-bonus-scale 2.8   --age-penalty-scale 2.6   --output runs/v28_multicycle_full_session/train_commands.txt   --execute --jobs 2
```

## Next Version Direction

The strongest next version should not simply increase residual beta. The bottleneck is queueing under dense/extreme bursts.

Recommended next direction:

```text
Keep LSTM-PPO.
Keep planner residual scaffold.
Train and evaluate on full multicycle sessions.
Add an explicit service scheduler objective that optimizes Eff@15s and P90, not only final coverage.
```

Concrete ideas:

- Add a stricter reward term for `pending_steam_count` at terminal drain.
- Add per-step penalty for active points older than SLA, not just oldest-active mean pressure.
- Add a route-level objective that rewards covering any steam point along the route, not only the selected target.
- Add an ablation table reporting `Cov`, `Eff@15s`, `P90`, `Clear`, and `Pending` for every component.
- Consider a gate policy that selects between horizon2, deadline_horizon2, and slack/EDF planner modes, while LSTM-PPO learns only the gate/residual correction.

## Bottom Line

The historical record is clear:

```text
V12 is the strongest earlier coverage story.
V13-V25 explored deadline, rescue, path-bending, corridor, sticky, gate, and urgency glue.
V27 fixed the episode/session mismatch.
V28 is the most realistic protocol and gives a defensible hard-stage improvement, but extreme latency remains unsolved.
```
