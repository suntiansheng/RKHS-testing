# GCM comparison

This experiment compares NORD with the scalar generalised covariance measure
(GCM) of Shah and Peters (2020), equation (3). It uses the four main simulation
structures, n=400, p=20 and 50, four outcome learners, twelve amplitudes, and
300 replications per cell. GCM and NORD use exactly the same fitted residual
products, without inverse-propensity weights.

GCM multiplies the residual-product mean by `sqrt(n)`, divides by its empirical
standard deviation, and uses a two-sided standard-normal calibration. Saved standard deviations
have denominator n-1; `simulation/gcm.py` converts them to denominator n for
the test. A zero or nonfinite score variance is reported as an error.
NORD uses 200 centered Gaussian multiplier draws and a plus-one p-value.

## Included results

The `results/gcm_*.csv` files contain the 384 complete simulation cells,
the paired NORD estimates, null rejection rates, and paired rejection-rate
differences. NORD curves in the main figures use these matched fits. The
single-split, Yan, and oracle curves use their original archived summaries.

The 32 GCM null rates range from 0.0267 to 0.0833. Paired confidence intervals
are pointwise Monte Carlo intervals.

## Recompute the comparison

Run from the repository root after installing `requirements.txt`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/gcm_comparison.py --phase null --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/gcm_comparison.py --phase power --workers 8
python experiments/publish_gcm_comparison.py
```

The runner uses one numerical thread per CPU process. It reuses propensity
fits for each data draw because the fitting specification does not depend on
the outcome learner, signal structure, or amplitude. Nuisance specifications,
seeds, and kernel bandwidth selection match `simulation/configs/main.json`.
Small numerical differences across package versions or numerical libraries
can affect rejection counts, particularly for neural networks.

Outputs go to `runs/gcm_comparison/`. Each completed job has a compressed shard
and a checkpoint checksum. Repeating a command resumes verified checkpoints;
source or configuration changes require a new output directory. Use
`--rep-stop 2 --output runs/gcm_check` for a short execution check. Summarize
an existing run with `--summarize-only`.

The full experiment has 1,200 checkpoints and 115,200 matched fitted-score
records. The publisher requires the complete run, verifies source and result
hashes, reconstructs GCM p-values from saved moments, checks rejection
indicators and cell counts, and writes the `results/gcm_*.csv` tables. Large
per-replication files and checkpoints are generated locally rather than stored
in the source repository.

## Signal cancellation diagnostic

Reproduce the diagnostic reported in the manuscript with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/gcm_signal_diagnostic.py
```

The command uses the compact protocol and calibration files included in
`results/gcm_comparison/`. Use `--run runs/gcm_comparison` to inspect a newly
computed run, or `--output PATH` to save the diagnostic elsewhere.

The diagnostic is the absolute weighted signal mean divided by the weighted
absolute signal mean, with weight e(X)(1-e(X)). It uses the 100,000 calibration
covariates and their negatives, checks the reference signal scales, and fits
no learners. The standard error treats antithetic pairs as independent units.
The ratio is approximately 0.00045 and 0.00053 for the hierarchical signal at
p=20 and 50, versus 0.80 and 0.93 for the pairwise-product signal. This describes
population-moment cancellation; it is not a power calculation.

## Tests

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m unittest discover -s simulation/tests -p test_gcm.py
```
