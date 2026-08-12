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

The checked-in `paper.pdf` is exactly five pages.

## Recompute the statistical tables

From the repository root:

```bash
python paper/iccc2026_major_revision/recompute_hierarchical_stats.py
```

The script performs 20,000 crossed bootstrap draws over training seeds, held-out environment seeds, and episodes. It writes the results to `tables/`.

## Claim boundary

The strongest supported mechanism result is Full versus the matched absolute-action No-residual controller. Ours versus Horizon-2 is tied in Low, Realistic, and Hard; the Extreme difference is positive before multiplicity correction but is not significant after four-stage Holm correction. The Extreme coverage operating point also has worse latency, strict SLA, and backlog than Horizon-2.

The remaining major-revision experiment is a separately trained, otherwise matched H3-residual controller. The current H3 result evaluates deterministic planner capacity only, so the manuscript does not claim planner-horizon independence.

## Checkpoints

The manuscript source, statistics, and web replay do not require trained weights. Checkpoints are generated artifacts and remain outside Git. See `docs/model-weights.md` for the recommended GitHub Release or Zenodo publication process.
