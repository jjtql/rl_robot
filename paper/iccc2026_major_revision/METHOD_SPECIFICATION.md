# Verified Method Specification for IC069 Revision

This document records the implementation that generated the V12 paper data. It is the source for the revised equations, pseudocode, and parameter tables. Values come from the seed-0 V12 configuration and the shared implementation; seed 1 and seed 2 use the same settings except for the random seed.

## 1. Task and Timing

- Simulator: MuJoCo digital twin with an ABB manipulator and a planar coverage center over the vessel workspace.
- High-level action: `u=[dx,dy,s]` in `[-1,1]^3`; the planar direction is normalized and `s` controls the speed scale.
- Decision interval: `0.05 s`.
- Nominal planar step: `0.04` workspace units before speed scaling.
- A spot is served when the coverage center is within the stage-specific cover radius.
- Spots persist until covered. `max_steam_age` is an age-normalization constant unless the separately disabled timeout mode is enabled.
- Raw coverage: covered spots / spawned spots at episode end. Uncleared active spots remain in the denominator.
- Covered-only SLA rate: covered within 4 s / covered spots.
- Strict SLA rate: covered within 4 s / spawned spots.
- Mean and p90 response latency use only covered spots because uncleared spots have no completion time; strict SLA and pending backlog expose the uncleared tail.
- Smoothness diagnostic: mean per-step action change `mean(||u_t-u_(t-1)||_2)`.

## 2. Four Dynamic Stages

| Parameter | Low | Realistic | Hard | Extreme |
|---|---:|---:|---:|---:|
| Max active spots | 3 | 4 | 6 | 8 |
| Age normalization | 260 | 220 | 200 | 180 |
| Nominal spawn probability | 0.006 | 0.010 | 0.018 | 0.026 |
| Spawn cooldown | 80 | 65 | 48 | 36 |
| Initial spots | 1 | 1 | 2 | 3 |
| Cover radius | 0.17 | 0.14 | 0.12 | 0.10 |
| Time penalty | 0.030 | 0.035 | 0.045 | 0.055 |
| Base cover reward before scale | 68 | 64 | 62 | 60 |
| Quick-cover reward before scale | 16 | 14 | 12 | 10 |
| Potential gain before scale | 8.0 | 7.5 | 6.5 | 5.8 |

The V12 arrival generator additionally uses three drifting thermal hotspots and repeated burst/lull cycles:

- thermal hotspot count `3`, sigma `0.22`, strength `3.8`, background weight `0.055`;
- hotspot drift standard deviation `0.003`, lifetime `960` steps, refresh probability `0.0025`;
- lull `70` steps, charge `110` steps, sparse threshold `2`;
- each burst schedules `3--5` spots, emitted one by one every `4` steps;
- trickle probability `0.005`, with an initial burst enabled.

## 3. Risk-Aware Target Score

Material observation is disabled in V12. For active spot `i`:

```text
phi_age   = clip(age_i / A_stage, 0, 1)
phi_dist  = 1 - clip(||p_i-c|| / R_pot, 0, 1)
phi_reach = 1 - clip(||p_i-p_home|| / R_max, 0, 1)
phi_therm = clip(thermal_field(p_i), 0, 1)
q_i       = 0.40*phi_age + 0.30*phi_dist
          + 0.10*phi_reach + 0.20*phi_therm
```

The risk-aware heuristic selects the first active spot attaining the maximum score and outputs the normalized planar direction toward it with speed action `s=1`. The stage pattern need not be monotonic because max active count, cover radius, spawn rate, initial load, and age normalization all change together.

## 4. Horizon-2 Planner

At each decision step, Horizon-2 enumerates all ordered routes of length `min(2,N_t)`. For route `(i_1,i_2)` it simulates travel from the current coverage center, using

```text
travel_steps = max(1, ceil(max(distance-cover_radius,0)/0.04)).
arrival_age  = current_age + elapsed_travel + travel_steps.
urgency      = clip(arrival_age/A_stage, 0, 1.5).
```

The route score is

```text
sum_j 0.82^j * [q_ij + 0.45*urgency_ij
                 - 0.018*travel_steps_ij
                 - 0.050*max(urgency_ij-1,0)]
- 0.08*clip(mean_leftover_arrival_age/A_stage,0,2).
```

The first target on the highest-scoring route is executed. Strict `>` comparison preserves the first enumerated route on an exact tie. The command is the normalized direction to that target with `s=1`, clipped to the action bounds. Horizon-3 uses the same score and a route length of three.

## 5. Observation and Network

- Base observation dimension: `61`.
- Active-spot set: at most `8` items, each with `8` features; padding mask is the last item feature.
- Total observation dimension: `61 + 8*8 = 125`.
- Base branch: Linear `61 -> 256`, LayerNorm, ReLU.
- Shared spot embedding: Linear `8 -> 128`, LayerNorm, ReLU.
- Set encoder: four-head self-attention at width `128`, padding mask, valid-item mean pooling.
- Spot output: Linear `128 -> 256`, LayerNorm, ReLU.
- Fusion: Linear `512 -> 256`, LayerNorm, ReLU, Linear `256 -> 256`, LayerNorm, ReLU.
- Temporal model: one-layer LSTM, input/hidden width `256`.
- Actor mean: Linear `256 -> 3` plus tanh.
- State-independent log standard deviation initialized to `-0.8` per action dimension.
- Critic: Linear `256 -> 1`.
- Auxiliary prediction head: Linear `256 -> 128`, ReLU, Linear `128 -> 2`, tanh.
- Orthogonal initialization; actor output gain `0.01`.

The recurrent state is retained across the four 800-step chunks of a 3,200-step continuous sequence. It is retained after coverage, reset after a miss, and reset at a new independent sequence/episode.

## 6. Residual Composition and Shield

Training-time residual scale:

```text
beta(step) = 0.03 + clip(step/700000,0,1)*(0.20-0.03).
```

At deployment the final scheduled value `0.20` is used, multiplied by the phase scale:

| Phase | Lull | Sparse | Charging | Mid | Dense | Burst |
|---|---:|---:|---:|---:|---:|---:|
| Scale | 1.35 | 1.25 | 1.00 | 1.00 | 0.40 | 0.25 |

All main-configuration phases use Horizon-2 as the base planner. The raw command is

```text
u_candidate = clip(u_H2 + beta*k_phase*u_res, -1, 1).
```

The direction guard returns the base command if planar cosine alignment between base and candidate is below `0.15`. The progress shield projects one step toward a base-aligned active target:

- required candidate progress: at least `0.35` of base progress, with margin `0.002`;
- guarded blend: `0.70*u_base + 0.30*u_candidate`;
- guarded blend must retain at least `0.50` of base progress;
- a reverse turn with cosine below `-0.55` and worse progress is replaced by `0.55*u_base + 0.45*u_candidate`;
- after `90` no-cover steps, a candidate worse than base progress is rejected;
- final action is clipped to `[-1,1]^3`.

## 7. Exact Reward Implementation

The implementation uses more terms than the simplified six-term equation in the submitted paper. For each step:

```text
R = R_time + R_active + R_mean_age + R_potential + R_best_progress
  + R_cover + R_latency + R_oldest + R_backlog + R_SLA
  + R_action_delta + R_action_L2 + R_success + R_miss.
```

Main V12 coefficients:

- stage time penalty: shown in the stage table;
- active count penalty: `0.01*2.4 = 0.024` per active spot;
- normalized mean-age penalty: `0.04*2.8 = 0.112`;
- cover reward: stage base value times `0.70`;
- quick-cover reward: stage base value times `3.0`;
- precision reward: `12*0.65 = 7.8`;
- distance-potential gain: stage value times `0.65`, discount `0.985`, delta clipped to `[-1.5,1.5]`;
- best-progress gain: `0.75*0.45 = 0.3375`;
- covered-point latency penalty gain: `30.0`, normalized by the 80-step SLA and clipped at ratio 2;
- oldest-active penalty gain: `0.45`, normalized by 80 SLA steps and clipped at ratio 2;
- backlog penalty gain: `0.20`, normalized by stage max active count and clipped at ratio 2;
- in-SLA bonus gain: `20.0`; late-SLA penalty gain: `30.0`;
- action-delta penalty gain: `0.025`; action-L2 penalty gain: `0.004`;
- success bonus: `30.0`; miss penalty: `28.0` (timeout misses are disabled in the target protocol).

The PPO input reward is multiplied by `0.02` and clipped to `[-4,4]`.

## 8. PPO, BC, and Training Budget

| Item | Value |
|---|---:|
| Training seeds | 0, 1, 2 |
| Curriculum episodes | 80 / 160 / 720 / 192 |
| Total episodes | 1,152 |
| Steps per chunk | 800 |
| Total decision steps | 921,600 |
| PPO update interval | 4 chunks = 3,200 steps |
| PPO epochs per update | 1 |
| Learning rate | `1e-5` |
| PPO clip / value clip | `0.05 / 0.05` |
| GAE gamma / lambda | `0.985 / 0.95` |
| Value coefficient | `0.5` |
| Max gradient norm | `0.35` |
| Entropy coefficient | `8e-4 -> 1.8e-4` over 1,000,000 steps |
| Auxiliary prediction coefficient | `0.05` |
| Prediction horizon | 150 steps |
| PPO action-smooth loss coefficient | `0.02` |
| PPO action-L2 loss coefficient | `0.002` |
| BC teacher | Horizon-2 |
| BC dataset | 200 episodes: 20/40/100/40 by stage |
| BC epochs | 12 |
| PPO-time BC coefficient | `0.03 -> 0` over 260,000 steps |

The reported model is the final/latest checkpoint after the fixed curriculum. Held-out evaluation seeds are not used for checkpoint selection.

## 9. Variant Component Matrix

| Variant | H2 base in action | Residual action | H2 BC warm start | Attention | Carry 3200 | Prediction | Service reward | Shield | PPO action meaning |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Main | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Bounded correction |
| No attention | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Bounded correction |
| No carry | Yes | Yes | Yes | Yes | No (800) | Yes | Yes | Yes | Bounded correction |
| No prediction | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes | Bounded correction |
| No service reward | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes | Bounded correction |
| No residual | No | No | Yes | Yes | Yes | Yes | Yes | No | Absolute action |
| Vanilla LSTM-PPO | No | No | No | Yes | Yes | No | Yes | No | Absolute action |
| Horizon-2 | Yes | No learned branch | N/A | N/A | N/A | N/A | N/A | N/A | Planner action |
| Risk-aware | No | No learned branch | N/A | N/A | N/A | N/A | N/A | N/A | Greedy heuristic action |

Interpretation constraint: Main vs No residual is the matched comparison that isolates the planner-residual action interface. No residual vs Vanilla simultaneously changes BC warm start and prediction, so their gap cannot be assigned to residualization.
