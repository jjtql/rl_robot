# Response to the Reviewer for Paper IC069

**Revised title:** Coverage-Oriented Planner-Guided Memory Residual Reinforcement Learning for Robotic Thermal-Spot Coverage

We thank the reviewer for the detailed major-revision comments. We agree that the submitted manuscript did not sufficiently specify the controller, did not preserve the crossed dependence of the evaluation design, and attributed too much meaning to the historical Vanilla LSTM-PPO comparison. The revised manuscript remains exactly five IEEE conference pages, but repeated discussion has been compressed so that the required definitions, statistics, baselines, and limitations can be included.

## Summary of Revisions

1. The Introduction now identifies ICCC Track 6 as the primary track, relates the task to steam-guided pot loading in solid-state Baijiu distillation, states the external-localization assumption, and gives three explicit research questions.
2. The related-work discussion now includes stochastic dynamic routing, formal shielding, constrained policy optimization, and recent residual-control references.
3. The paper now defines all metric denominators, the risk-aware score, the Horizon-2/Horizon-3 objectives, network dimensions, residual schedule, shield, reward coefficients, training budget, curriculum, and evaluation protocol.
4. The primary uncertainty analysis is now a 20,000-draw crossed hierarchical bootstrap over training seeds, evaluation seeds, and episodes, with paired scenarios and Holm correction across stages.
5. The historical Vanilla LSTM-PPO result is no longer used for causal attribution. The mechanism claim is based on a matched absolute-action ablation that retains the observation, LSTM, H2 warm start, prediction objective, reward, curriculum, and training budget.
6. The revised results report coverage, covered-only mean and p90 latency, strict SLA over all spawned spots, active backlog, and action smoothness.
7. The claims have been narrowed: the method is not presented as uniformly superior or faster than Horizon-2. The principal supported mechanism result is the benefit of the planner-residual action interface relative to the matched absolute-action controller.

## 1. Scope, ICCC Positioning, and Research Questions

**Reviewer comment:** Explicitly position the work within ICCC, instantiate the physical process, and state testable research questions.

**Response:** The revised Introduction identifies Track 6 (Robotics) as the primary track, with secondary links to Track 2 (AI for Engineering) and Track 4 (Machine Learning). It now relates the task to pot loading in solid-state Baijiu distillation, where fermented grain is deposited in response to observed steam-breakthrough locations. The simulated ABB manipulator services externally localized targets over the vessel. The study isolates online routing and motion-command generation; thermal-image detection, granular-material dynamics, layer-uniformity assessment, and distillation-quality validation are explicitly outside its scope.

The Introduction ends with three questions:

- **RQ1:** Does a bounded planner-residual action interface improve a fixed planner relative to a matched absolute-action PPO controller?
- **RQ2:** How does the method compare with alternative planner and heuristic priors?
- **RQ3:** What coverage-service trade-off is created in latency, strict SLA, backlog, and smoothness?

## 2. Missing Literature and Technical Positioning

**Reviewer comment:** Add the relevant work on shielding, constrained reinforcement learning, and stochastic dynamic routing.

**Response:** The revision cites Alshiekh et al. for shielding, Achiam et al. for Constrained Policy Optimization, and Bertsimas and van Ryzin for stochastic dynamic routing. The text also clarifies two boundaries. First, our post-policy progress check is a local action filter and is not claimed to provide a formal temporal-logic safety guarantee. Second, latency and backlog are scalarized reward preferences; the current method is not a CMDP/CPO algorithm.

## 3. Reproducibility and Presentation

**Reviewer comment:** Define Horizon-2 and risk-aware precisely, report load-bearing parameters, add a component matrix, remove layout commentary, and correct the signed zero.

**Response:** Section II, Tables I--II, and Table V(b) now report:

- the risk score and weights: age 0.40, closeness 0.30, reachability 0.10, and thermal context 0.20 when material observation is disabled; the enabled-material branch uses age 0.35, closeness 0.25, material 0.15, reachability 0.10, and thermal context 0.15;
- the material-gap definition over reachable grid cells, including the target layer height and deposition-kernel width;
- the ordered Horizon-$L$ route enumeration, simulated travel-step score, discount, leftover-age term, and first-action execution rule;
- the 125-dimensional observation, eight target slots, width-128 target embedding, four-head attention, and width-256 LSTM;
- the residual schedule from 0.03 to 0.20 over 700,000 steps and the six phase multipliers;
- the shield alignment threshold, progress margin and ratios, reverse-turn threshold, stagnation threshold, and guarded blend;
- all load-bearing reward coefficients, action interval, SLA, stage settings, PPO settings, behavior-cloning warm start, prediction coefficient, training budget, and held-out protocol;
- the fixed 3,200-decision-step evaluation window, disabled timeout and terminal-drain behavior, and the prespecified terminal-model selection rule.

Table V(b) gives the exact action interface and component status of Full, No residual, No attention, No carry, No prediction, and No service. Non-substantive layout discussion has been removed. The Hard-stage difference is printed as `$<0.001$`, not `-0.000`.

The revised protocol also distinguishes the MuJoCo integration interval
($0.002$ s) from the high-level decision and metric interval ($0.05$ s),
documents the Gaussian thermal-score normalization, and states that optional
urgency augmentation is not used. The success bonus and inactive timeout-miss
term are reported separately. All learned results use the terminal model after
the prespecified 1,152-episode curriculum; held-out evaluation data do not
participate in model selection.

## 4. Planner Mismatch and Horizon Sensitivity

**Reviewer comment:** Test whether the learned result depends on the Horizon-2 planner and include a lower-cost planner-horizon analysis.

**Response:** The revision adds Horizon-3 to the deterministic baseline table using the same route objective with $L=3$. H3 obtains coverage 0.917, 0.927, 0.860, and 0.841 from Low to Extreme, compared with H2 values 0.919, 0.896, 0.870, and 0.816. This establishes that longer lookahead is not uniformly better and provides the requested planning-horizon sensitivity analysis.

The interpretation is deliberately restricted. The learned residual is conditioned on H2 actions; therefore, the H3 comparison evaluates deterministic planner capacity and does not establish residual transfer across base planners. The manuscript makes no claim of planning-horizon-independent residual gains. A cross-planner transfer study would require separately trained, otherwise matched residual controllers and is identified as future work rather than evidence required for the present H2-conditioned claim.

## 5. Contribution Attribution

**Reviewer comment:** The relationship among Vanilla LSTM-PPO, No residual, and Full was unclear, so the source of the gain could not be identified.

**Response:** We agree. The historical Vanilla run simultaneously removed residual composition, H2 behavior cloning, and the prediction objective. It is therefore excluded from causal claims. The revised causal comparison is Full versus a matched No-residual controller. The latter retains the same observation, recurrent encoder, H2 behavior-cloning warm start, auxiliary prediction loss, service reward, curriculum, optimizer settings, and 921,600-step budget, but emits an absolute PPO action and has no planner base or progress shield.

The Full-minus-No-residual coverage effects are:

| Stage | Difference | Hierarchical 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| Low | +0.079 | [0.004, 0.202] | 0.029 |
| Realistic | +0.124 | [0.036, 0.223] | 0.003 |
| Hard | +0.144 | [0.037, 0.268] | 0.011 |
| Extreme | +0.168 | [0.050, 0.285] | 0.003 |

Accordingly, the paper attributes the supported mechanism-level result to the complete planner-residual action interface. Attention, recurrent carry, prediction, and service shaping are described as supporting choices with mixed stage-wise ablation effects, not as four independently proven innovations.

## 6. Baselines, Statistical Dependence, and Metrics

### 6.1 Baseline fairness

**Reviewer comment:** Retune the direct PPO baseline or substantially weaken the conclusion drawn from it.

**Response:** The historical direct LSTM-PPO result is retained as a descriptive diagnostic with its uncertainty interval, but it is not used in the causal comparison or principal conclusion. The revised paper makes no claim that direct PPO or LSTM necessarily fails. The matched No-residual experiment is used because it controls the observation, warm start, auxiliary objective, reward, curriculum, and budget. A full-budget, independently tuned Vanilla protocol was not completed; consequently, the Vanilla result is excluded from causal comparisons and is interpreted only descriptively. This limitation is now stated explicitly rather than being treated as evidence against direct PPO.

The deterministic comparison set has also been broadened to Horizon-2, Horizon-3, nearest, oldest, distance-age, risk-aware, dynamic-weighted, ACO-TSP, and planner ensemble. This table shows that risk-aware reaches 0.863 Extreme coverage, slightly above our 0.855, which prevents a claim of universal dominance.

### 6.2 Hierarchical uncertainty

**Reviewer comment:** Preserve dependence among training seeds, evaluation seeds, and repeated episodes rather than treating all episodes as independent.

**Response:** Each learned method has three training seeds crossed with three held-out environment seeds and five episodes per cell. The revised analysis uses 20,000 crossed hierarchical bootstrap draws. Training labels, evaluation labels, and episodes within selected cells are resampled at their corresponding levels. Paired methods reuse the same held-out scenarios, while deterministic planners omit the training axis. Holm correction is applied across the four stages within each predeclared comparison family.

For Ours versus Horizon-2, the corrected coverage results are:

| Stage | Ours minus H2 | Hierarchical 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| Low | -0.002 | [-0.018, 0.016] | 1.000 |
| Realistic | +0.005 | [-0.018, 0.034] | 1.000 |
| Hard | <0.001 | [-0.040, 0.033] | 1.000 |
| Extreme | +0.038 | [0.003, 0.074] | 0.135 |

The Extreme interval is positive before multiplicity correction, but the four-stage Holm-adjusted test is not significant. The Abstract, Results, Discussion, and Conclusion now state this directly.

### 6.3 Denominators and service diagnostics

**Reviewer comment:** Define latency/SLA denominators and report service and smoothness metrics, not coverage alone.

**Response:** The revised paper defines coverage as covered/spawned, including active uncleared spots in the denominator. Mean and p90 latency condition on covered spots. Strict SLA is the number covered within 4 s divided by all spawned spots, so uncleared spots count as failures. Backlog is the time-average active count, and smoothness is the mean per-step Euclidean action change. Evaluation uses a fixed 3,200-decision-step window; timeout removal and terminal drain-to-clear are disabled, so these are fixed-horizon persistent-service results rather than complete-session clearance guarantees.

The service table shows that the Extreme coverage increase over H2 is accompanied by mean latency 23.88 s versus 21.38 s, p90 latency 46.01 s versus 39.92 s, backlog 3.81 versus 3.55, strict SLA 0.103 versus 0.128, and action change 0.0289 versus 0.0277. We therefore report a coverage--service trade-off with coverage as the primary objective, not a latency advantage.

### 6.4 Unseen condition

**Reviewer comment:** Add targeted validation beyond the original condition.

**Response:** We added an unseen higher-density evaluation without retraining. Ours/H2 coverage is 0.868/0.888 in Hard and 0.850/0.825 in Extreme. The hierarchical intervals overlap, so the result is reported as robustness evidence rather than superiority.

## 7. Five-Page Revision

**Reviewer comment:** Use space for reproducibility and validation rather than repeated interpretation.

**Response:** The revised manuscript is exactly five IEEE conference pages, as required for this version. We removed repeated Extreme-stage discussion, shortened generic PPO background, combined the mechanism, component, and service evidence into one full-width table, and retained only a representative qualitative rollout. This made room for exact definitions, corrected uncertainty, the expanded baseline set, component accounting, and service diagnostics without exceeding five pages.

## Scope of the Horizon Analysis

The added Horizon-3 experiment addresses planning-horizon sensitivity under an otherwise identical deterministic objective. It is not a cross-planner residual-transfer experiment. Accordingly, the revised manuscript confines its mechanism claim to the H2-conditioned controller and identifies matched cross-planner residual training as future work. This scope distinction prevents the deterministic H3 comparison from being interpreted as evidence of horizon-independent transfer.
