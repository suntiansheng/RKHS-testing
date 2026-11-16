# CaSiNo application

The manuscript compares mean dyad satisfaction and the indicator that both
participants give the highest satisfaction rating. Each outcome is tested
using 36 baseline survey features and then those features plus 128 normalized
signed-hash coordinates of opening text. The reference sample contains 232
participant-disjoint dialogues: 73 aligned-priority and 159 maximal-potential
negotiations. Its identifier checksum is checked by `analyze.frozen_main_sample`.

Run from the repository root:

```sh
python application/download_data.py
python application/joint_analysis.py --results-dir runs/casino-joint --workers 2
python application/plot_joint.py --results-dir runs/casino-joint --output-dir runs/casino-joint/figures
python application/word_analysis.py --output-dir runs/casino-words
```

The downloader verifies the public ZIP checksum and extracted corpus files.
`joint_analysis.py` fits NORD and five-fold CF-NORD with ridge, Lasso and random
forests, using 5,000 centered Gaussian multipliers. Ridge is evaluated at
bandwidths 0.5, 1 and 2; the other learners use bandwidth 1. The five shuffled
folds are shared across outcomes and covariate sets, with nuisance tuning
confined to each training sample. The kernel uses the fixed feature scales.

Ten alternative participant orderings use ridge at bandwidth 1 for both
outcomes and covariate sets. The runner also saves age-omission and
select-before-restriction checks. These additional checks bring the total to
136 tests. The result directory contains `regression_tests.csv`, sample and
overlap summaries, metadata, and per-job designs and nuisance predictions.
Repeat the command to resume. Source or protocol changes require a new output
directory. The optional `--baseline-results PATH` reuses validated predictions
from `baseline_analysis.py`; a fresh run needs no saved predictions.

`plot_joint.py` writes the four rows of the main application figure. To render
the distributed aggregate results without refitting, run
`python reproduce_figures.py`.

`word_analysis.py` uses the same reference sample. It counts each word once
per dialogue, uses Jeffreys-smoothed log odds and 2,000 within-group bootstrap
resamples, and plots six words per direction selected by absolute standardized
contrast. The saved table contains all retained words. These intervals are
pointwise and do not adjust for word selection.

The following companion analyses are also available:

```sh
python application/analyze.py --results-dir runs/casino-text
python application/baseline_analysis.py --results-dir runs/casino-baseline
python application/null_controls.py --data-dir runs/casino-text --output-dir runs/casino-null
```

`analyze.py` runs the earlier text-only analysis. `baseline_analysis.py` adds
baseline standardized contrasts; see [its notes](baseline_analysis.md).
The manuscript figures use `joint_analysis.py` and `word_analysis.py`.
Raw dialogues, participant records, fitted predictions and model files are
excluded from version control. `results/casino/` contains aggregate test and
word-frequency summaries.
