# Spline regression and cross-fitting

This is the supplementary comparison of NORD and two-fold CF-NORD at
`n=120,160`, with `p=20` and additive cubic splines. The two response models are

```
(i)  Y = b(X) + a D f(X) + epsilon
(ii) Y = b(X) + a (D - e(X)) f(X) + epsilon.
```

Both have regression contrast `a f(X)`. The pooled regression in model (ii)
is `b(X)` at every amplitude. The covariates, propensity, additive functions
and spline specifications are those of the main regression experiment.
The Gaussian kernel has fixed squared bandwidth 40. Calibration uses 999
centered Gaussian multipliers, a plus-one p-value and level 0.05.

The independent pilot uses 150 replications per cell, seed 2026110601 and
amplitudes `0.1,0.2,0.3,0.5,0.8,1.2,1.6,2,2.5,3.2`. It selects amplitudes for
which NORD's model (i) power with estimated propensity lies between 0.10 and
0.85 at both sample sizes. This gives `2,2.5,3.2`. Confirmation uses seed
2026110602, 5,000 null replications and 3,000 replications per positive amplitude.
It also includes `a=1.2` and a five-fold sensitivity at `n=160`, `a=0,1.2,2.5`.

NORD and CF-NORD test all observations. Independent balanced random permutations
assign the outer folds. Preprocessing, tuning and nuisance fitting use each
training sample only. Data, folds and centered multiplier draws are shared
across procedures and amplitudes within a replication. Known-propensity
controls reuse the outcome fits. Prediction errors use 1,000 independent
reference covariates. At the null the two response models coincide, so one
shared set of null results is retained.

Run from the repository root:

```sh
python experiments/spline_crossfit_plot.py
python experiments/spline_crossfit_report.py
```

These commands read the compact archives in `results/spline_crossfit/`.
The plots use raw rejection rates and exact pointwise binomial intervals for
null size. The report checks the pilot selection and exports the twelve
primary paired comparisons and prediction-error comparisons. Differences in
raw power should be considered alongside the observed null rejection rates.

To refit the pilot and confirmation experiments:

```sh
python experiments/spline_crossfit.py --config simulation/configs/spline_crossfit_pilot.json --output runs/spline-pilot --workers 4
python experiments/spline_crossfit.py --config simulation/configs/spline_crossfit_confirmation.json --output runs/spline-confirmation --workers 4
python experiments/spline_crossfit_report.py --pilot runs/spline-pilot --run runs/spline-confirmation --output runs/spline-confirmation/report
python experiments/spline_crossfit_plot.py --run runs/spline-confirmation --output runs/spline-confirmation/figures
```

Each replication writes an atomic checkpoint. Repeating the command resumes
completed work; changed source, configuration or library versions require a new
output directory. `--summarize-only --output PATH` rebuilds summaries from a
run's checkpoints. Large checkpoint and replication-level files are generated
locally rather than included in the compact archive.

The archived provenance records the source hashes at the time of the original
run. The packaged runner uses the same numerical routines from
`simulation/simcore.py`; source paths and output handling have been updated.

```sh
python -m unittest discover -s tests -p test_spline_crossfit.py
```
