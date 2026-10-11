# NORD and CF-NORD with regularized learners

The supplementary simulation in `v10.tex` compares full-sample NORD with
CF-NORD using two folds, sample sizes 120 and 160, and 20 covariates. Outcome
learners are additive cubic splines, random forests, Lasso, and a neural
network. Their fitting and tuning specifications, and the spline-logistic
propensity estimator, are defined in `simulation/simcore.py` and
`simulation/configs/main.json`. Neural networks run on the CPU.

The design is additive, with amplitudes 0, 0.8, 1.2, 1.6, 2, 2.5, 3.2, 4, and 6.
The Gaussian kernel has fixed squared bandwidth 40. Calibration uses 999
centered Gaussian multipliers, a plus-one p-value, and nominal level 0.05.
There are 2,000 null replications and 1,000 replications per positive amplitude.
The seed is 2026100117.

Both procedures use all observations in the statistic. NORD trains on the
full sample; each CF-NORD fit trains on 60 or 80 observations. Independent
random permutations assign balanced folds. Data, folds, and multiplier draws
are shared across methods, learners, and amplitudes within a replication.
Preprocessing and tuning use each fit's training sample only. Known-propensity
controls retain the fitted outcomes and replace fitted propensities by their
true values. One shared oracle uses both true nuisance functions.

## Reproduce the manuscript figures

Run from the repository root after installing `requirements.txt`:

```sh
python experiments/plot_crossfit_manuscript.py
```

The command reads the five compact tables and metadata files in
`results/crossfit_comparison/n120_n160_k2/` and writes the two manuscript figures
to `results/figures/`. It verifies paired differences and their Monte Carlo
intervals, Wilson intervals, and the 34-cell Holm null screen. This verifies
the reported summaries; individual replication records are produced by the
full simulation below.

The 34 null cells comprise 16 estimated-propensity procedures, 16
known-propensity procedures, and two oracles. Two-sided exact binomial tests
receive Holm adjustment within this family. All curves are retained. The
screen flags CF-NORD with splines at both sample sizes and with the neural
network at n=120. Alternative rejection rates for these configurations are
not comparisons at equal size. Error bars are pointwise Monte Carlo intervals.

## Run the simulation

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/crossfit_comparison.py \
  --output runs/crossfit_comparison/n120_n160_k2 \
  --sample-sizes 120,160 --workers 8 --write-raw

python experiments/plot_crossfit_manuscript.py \
  --run runs/crossfit_comparison/n120_n160_k2 \
  --output results/figures/crossfit_rerun
```

The complete run has 4,000 tasks, 340,000 result rows, and 306 cells. Each task
saves an atomic compressed CSV shard and a checkpoint containing its checksum,
configuration fingerprint, and fitting diagnostics. Repeating the same command
resumes verified checkpoints. Source or configuration changes require a new
output directory. Use `--max-tasks 2` with a separate output directory for a
short execution check; an incomplete run is not a final power estimate.

Prediction errors are evaluated on 10,000 independent covariates. Fold-specific
errors are weighted by evaluation-fold size. Outputs include `summary.csv`,
`paired_differences.csv`, `null_size.csv`, `validation.json`, and `raw.csv.gz`.
To summarize a partially completed run:

```sh
python experiments/crossfit_comparison.py \
  --output runs/crossfit_comparison/n120_n160_k2 --summarize-only --write-raw
```

## Optional n=200 comparison

The n=200 sensitivity experiment is not required for the two manuscript
figures. To reproduce it and generate combined figures:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/crossfit_comparison.py \
  --output runs/crossfit_comparison/n200_k2 \
  --sample-sizes 200 --workers 8 --write-raw
python experiments/publish_crossfit_sample_sizes.py
```

The publisher requires the complete raw files, checkpoints, and shards from
both runs. It verifies 6,000 checkpoints and 510,000 result rows, preserves the
34-cell and 17-cell screens, and adds a separate 51-cell screen for combined
figures. Outputs go to `results/figures/crossfit_sample_sizes/`. It does not
require any manuscript files or internal planning documents.

## Tests

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m unittest discover -s tests -p test_crossfit_comparison.py
```

These checks cover multiplier centering, score-error variance, independent
balanced folds, the upper-tail convention for zero scores, and damaged
checkpoints.
