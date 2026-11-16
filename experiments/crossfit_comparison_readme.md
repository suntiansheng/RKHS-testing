# Additional NORD and CF-NORD comparison

This earlier experiment compares NORD with two-fold CF-NORD using four outcome
learners at `n=120,160` and `p=20`. It is retained as an additional study.
The final supplement reports the spline-only experiment with two response
models; see [the spline comparison](spline_crossfit_readme.md).

The archived study uses the additive design, amplitudes
`0, 0.8, 1.2, 1.6, 2, 2.5, 3.2, 4, 6`, 2,000 null replications and 1,000
replications at each positive amplitude. The Gaussian kernel has squared
bandwidth 40. Calibration uses 999 centered Gaussian multipliers and nominal
level 0.05. The seed is 2026100117. Outcome and propensity fitting follow
`simulation/configs/main.json`.

Both procedures test all observations. NORD trains on the full sample;
CF-NORD trains on half the sample using independent balanced folds. Data,
folds and multipliers are paired across learners and amplitudes within each
replication. Known-propensity controls retain fitted outcomes, and the oracle
uses both true nuisance functions.

To plot the archive, run from the repository root:

```sh
python experiments/plot_crossfit_manuscript.py
```

Despite its historical filename, this plotter reads the additional study in
`results/crossfit_comparison/n120_n160_k2/`. It writes rejection-rate ratios,
pointwise paired Monte Carlo intervals and a null-size figure. The 34 null
cells comprise 16 estimated-propensity procedures, 16 known-propensity
procedures and two oracles. Null intervals use the Wilson method.

For `R` paired replications, let `N_full` and `N_cf` be the rejection counts,
and let `b` and `c` count rejections by NORD only and CF-NORD only. The ratio
interval is `ratio * exp(+/- 1.95996 * s)`, with
`s^2 = R/(R-1) * (b+c)/(N_full*N_cf)`. Both counts must be positive; no
pseudocount is added. The exported `crossfit_rejection_rate_ratios.csv`
contains all 144 comparisons, including known-propensity controls.

To rerun:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/crossfit_comparison.py \
  --output runs/crossfit_comparison/n120_n160_k2 \
  --sample-sizes 120,160 --workers 8 --write-raw
python experiments/plot_crossfit_manuscript.py \
  --run runs/crossfit_comparison/n120_n160_k2 \
  --output runs/crossfit_comparison/n120_n160_k2/figures
```

The full run has 4,000 tasks, 340,000 result rows and 306 cells. Checkpoints
include source/configuration fingerprints and shard checksums. Repeating the
command resumes verified checkpoints; changed specifications require a new
output directory. Use `--max-tasks 2` in a separate directory for an execution
check, or `--summarize-only --write-raw` to summarize an existing run.

Prediction errors use 10,000 independent covariates. Outputs include
`summary.csv`, `paired_differences.csv`, `null_size.csv`, `validation.json`
and `raw.csv.gz`.

```sh
python -m unittest discover -s tests -p 'test_crossfit*.py'
```
