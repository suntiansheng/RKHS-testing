# CaSiNo application

The main paper compares two outcomes (mean dyad satisfaction and the indicator
that both participants report maximum satisfaction) under two covariate sets:
36 baseline survey summaries, and those summaries plus 128 normalized signed
hash coordinates of opening text. The analysis uses 232 participant-disjoint
dialogues, comprising 73 aligned-priority and 159 maximal-integrative-potential
dialogues. The sample identifier hash is checked in `analyze.frozen_main_sample`.

## Refit the manuscript analysis

```sh
python application/download_data.py
python application/joint_analysis.py --results-dir runs/casino-joint --workers 2
python application/plot_joint.py --results-dir runs/casino-joint --output-dir runs/casino-joint/figures
python application/word_analysis.py --output-dir runs/casino-words
```

The downloader checks the public ZIP checksum and extracted corpus members.
`joint_analysis.py` fits NORD and five-fold CF-NORD using ridge, Lasso and
random-forest nuisance estimates, with 5,000 Gaussian multipliers. Ridge uses
bandwidths 0.5, 1 and 2; the other learners use bandwidth 1. Ten alternative
outcome-blind participant selections use ridge at bandwidth 1. Additional
sensitivities select disjoint dialogues before restricting the regime groups
and omit the three age-related features. The complete analysis has 136 tests.

Five independent shuffled folds are shared across outcomes and covariate sets.
Nuisance tuning uses training data only. The joint runner can fit every nuisance
from a fresh checkout. An optional `--baseline-results PATH` reuses validated
main-sample predictions from `baseline_analysis.py`; it is not required.

The result directory contains `regression_tests.csv`, `analysis_summary.json`,
`analysis_plan.json`, sample-count and overlap summaries, and per-job designs,
predictions and completion records. Source and protocol changes require a new
output directory. The four PDFs produced by `plot_joint.py` form the rows of the
main application figure. Replotting the supplied compact archive instead of
refitting is available through `python reproduce_figures.py`.

## Opening-word contrasts

`word_analysis.py` uses the same verified main sample and existing tokenizer,
minimum-document-frequency rule and Jeffreys-smoothed document-presence log
odds. It performs 2,000 within-regime bootstrap resamples and plots six terms
from each direction, selected by absolute standardized contrast. These are
descriptive summaries of opening text; they do not measure causal word effects.
The output includes all word statistics and the supplementary figure.

## Additional analysis scripts

`analyze.py` preserves the original text-only analysis; `baseline_analysis.py`
preserves the companion baseline analysis and standardized contrasts. These
scripts can regenerate their intermediate results independently:

```sh
python application/analyze.py --results-dir runs/casino-text
python application/baseline_analysis.py --results-dir runs/casino-baseline
python application/null_controls.py --data-dir runs/casino-text --output-dir runs/casino-null
```

The final manuscript figures use `joint_analysis.py` and `word_analysis.py`.
Raw dialogues, participant records, fitted predictions and model files are
excluded from version control. The distributed `results/casino/` tables contain
aggregate test results and word summaries only.
