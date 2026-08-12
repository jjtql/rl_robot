# ICCC 2026 IC069 Major-Revision Status

## Version Boundary

- Submitted manuscript: `/Data2/jj/rl_robot/paper.pdf`, 5 pages.
- Final revised manuscript: `paper/iccc2026_major_revision/paper.pdf`, exactly 5 pages.
- Final source: `paper/iccc2026_major_revision/main.tex`.
- Reviewer response: `RESPONSE_TO_REVIEWER.md`.
- All reported numerical results are drawn from completed local result files; pilot runs are excluded from the reported evidence.

## Reviewer-Issue Matrix

| ID | Reviewer concern | Resolution in the five-page revision | Status |
|---|---|---|---|
| R1 | ICCC positioning | Track 6 primary; Tracks 2 and 4 secondary | Complete |
| R2 | Concrete physical setting | ABB/MuJoCo vessel, external target localization, persistent service action, burst-lull arrivals, simulation scope | Complete |
| R3 | Testable research questions | Three RQs added at the end of the Introduction | Complete |
| R4 | Shielding literature | Alshiekh et al. added; local filter distinguished from formal shielding guarantees | Complete |
| R5 | Constrained RL context | Achiam et al. added; scalar reward explicitly distinguished from CMDP/CPO | Complete |
| R6 | Dynamic routing context | Bertsimas and van Ryzin 1991/1993 added | Complete |
| R7 | Load-bearing parameters | Method text and Tables I--II report network, reward, residual, shield, stage, PPO, BC, prediction, timing, and budget values | Complete |
| R8 | Exact Horizon-2 definition | Ordered route enumeration, score, discount, leftover term, and receding-horizon action defined | Complete |
| R9 | Exact risk-aware definition | Normalized features and weights defined; risk-aware included in the full baseline table | Complete |
| R10 | Exact component matrix | Table V(b) reports action semantics and auxiliary components | Complete |
| R11 | Incorrect Vanilla attribution | Historical Vanilla result removed from causal argument; matched No-residual comparison used | Addressed by revised attribution |
| R12 | Fair direct-RL baseline | No universal direct-RL failure claim is made; the confounded historical run is excluded from causal evidence | Partially addressed by claim restriction; independent full-budget retuning remains open |
| R13 | Budgets and action interfaces | Fixed 921,600-step budget, architecture, recurrent protocol, and matched action-interface difference disclosed | Complete |
| R14 | Crossed experimental design | Three train seeds x three held-out environment seeds x five episodes per cell stated | Complete |
| R15 | Hierarchical uncertainty | 20,000-draw crossed bootstrap with paired scenarios and Holm correction | Complete |
| R16 | Baseline uncertainty | Hierarchical intervals reported for the primary comparison, matched ablation, risk-aware analysis, and density shift | Complete within page limit |
| R17 | SLA and smoothness | Coverage, mean/p90 latency, strict SLA, backlog, and action change reported | Complete |
| R18 | Metric denominators | Spawned, covered-only, all-spawned SLA, backlog, and uncleared-target behavior explicitly defined | Complete |
| R19 | Planner mismatch | Deterministic H3 capacity comparison added; no horizon-independent residual claim is made | Partially addressed: matched H3-residual retraining remains open |
| R20 | Unseen condition | Higher-density Hard/Extreme evaluation added without retraining | Complete |
| R21 | Repeated interpretation | Extreme-stage discussion compressed and bounded | Complete |
| R22 | Layout meta-commentary | Removed | Complete |
| R23 | Signed `-0.000` | Replaced by `<0.001` and described as tied | Complete |
| R24 | Bounded claims | Abstract and conclusion state coverage orientation, latency cost, and lack of universal superiority | Complete |
| R25 | Hardware scope | Simulation-only limitation retained; no deployment evidence is fabricated | Complete |

## Corrected Primary Findings

### Ours versus Horizon-2

| Stage | Coverage difference | Hierarchical 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| Low | -0.002 | [-0.018, 0.016] | 1.000 |
| Realistic | +0.005 | [-0.018, 0.034] | 1.000 |
| Hard | <0.001 | [-0.040, 0.033] | 1.000 |
| Extreme | +0.038 | [0.003, 0.074] | 0.135 |

The Extreme interval is positive before multiplicity correction, but the four-stage corrected test is not significant.

### Full versus Matched No-residual

| Stage | Coverage difference | Hierarchical 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| Low | +0.079 | [0.004, 0.202] | 0.029 |
| Realistic | +0.124 | [0.036, 0.223] | 0.003 |
| Hard | +0.144 | [0.037, 0.268] | 0.011 |
| Extreme | +0.168 | [0.050, 0.285] | 0.003 |

This is the principal supported mechanism result. It supports the complete planner-residual action interface relative to the matched absolute-action controller; it does not prove that each auxiliary module is independently beneficial.

## Remaining Experiment

A separately trained H3-residual controller with the H3 planner used consistently as behavior-cloning teacher and online base remains uncompleted. The current H3 row evaluates deterministic planner capacity only. The five-page manuscript states this directly and does not claim horizon-independent transfer.

## Claim Boundary

- Supported: the matched planner-residual interface outperforms the matched absolute-action PPO controller under the reported protocol.
- Supported: the clearest numerical Ours-versus-H2 coverage separation occurs in Extreme.
- Supported: this Extreme coverage--service trade-off has higher coverage together with worse latency, strict SLA, and backlog.
- Not supported: universal superiority over H2, H3, risk-aware, or all heuristics.
- Not supported: direct LSTM-PPO is inherently ineffective.
- Not supported: each attention, carry, prediction, and service component independently improves performance.
- Not supported: simulation results establish real-robot deployment readiness.
