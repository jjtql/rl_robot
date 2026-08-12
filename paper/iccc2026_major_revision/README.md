# ICCC 2026 IC069 Major Revision

This directory contains the five-page revised manuscript, reviewer response, figures, and the episode-level data used by the corrected hierarchical analysis.

## Files

- `paper.pdf`: final five-page revision.
- `main.tex`: manuscript source.
- `RESPONSE_TO_REVIEWER.md`: point-by-point response.
- `REVISION_MATRIX.md`: completion status and remaining limitation.
- `METHOD_SPECIFICATION.md`: implementation details supporting reproducibility.
- `data/eval/`: episode-level held-out evaluation CSVs used by the analysis.
- `tables/`: generated hierarchical summaries and comparisons.
- `recompute_hierarchical_stats.py`: crossed train-seed/evaluation-seed bootstrap.
- `submitted_paper.pdf`: original submitted manuscript.
- `reviewer_form.pdf`: source major-revision review.

## Compile

With a TeX Live installation containing `latexmk`:

```bash
cd paper/iccc2026_major_revision
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The checked-in `paper.pdf` is exactly five pages. The revised source states
that the MuJoCo integrator uses `0.002 s`, while task actions and service
metrics use a `0.05 s` decision interval. The evaluated protocol disables urgency augmentation;
the reported risk score and Horizon-2 objective therefore use the normalized
age, distance, reachability, and thermal terms specified in the source.

## Recompute the statistical tables

From the repository root:

```bash
python paper/iccc2026_major_revision/recompute_hierarchical_stats.py
```

The script performs 20,000 crossed bootstrap draws over training seeds, held-out environment seeds, and episodes. It writes the results to `tables/`.

## Claim boundary

The principal supported mechanism result is the comparison between the full method and the matched absolute-action No-residual controller. Ours versus Horizon-2 is statistically indistinguishable in Low, Realistic, and Hard; the Extreme difference is positive before multiplicity correction but is not significant after four-stage Holm correction. The Extreme coverage--service trade-off also has higher latency, lower strict SLA, and higher backlog than Horizon-2.

The historical Vanilla LSTM-PPO result is retained as a descriptive diagnostic with an uncertainty interval. A full-budget independent hyperparameter retuning of that baseline was not completed, so the result is excluded from causal comparisons and is not used to support a failure claim about direct PPO.

The Horizon-3 comparison addresses deterministic planning-horizon sensitivity under the same route objective. It does not establish cross-planner residual transfer, and the manuscript confines its mechanism claim to the H2-conditioned controller.

## Checkpoints

The manuscript source, statistics, and web replay do not require trained weights. Checkpoints are generated artifacts and remain outside Git. See `docs/model-weights.md` for the recommended GitHub Release or Zenodo publication process.
