# Nearest-neighbour interpolation experiment

This experiment compares full-sample NORD, five-fold CF-NORD, and an oracle
using n=400 observations and two Gaussian covariates. The outcome learner uses
20 nearest neighbours with inverse-distance weights. At training covariates,
it averages responses only over observations at zero distance, so the
full-sample outcome residuals are exactly zero. Every method uses the true
propensity to isolate outcome fitting. The Gaussian kernel has bandwidth one.

`results/interpolation.csv` contains 500 replications at each amplitude in
`0, 0.25, 0.5, 1`, with 199 centered Gaussian multiplier draws. These are also
the runner defaults.

To reproduce the archived experiment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/interpolation.py \
  --replications 500 --bootstrap-draws 199 --output runs/interpolation
```

The command writes `raw.csv`, `summary.csv`, and `config.json`. The figure
builder reads the supplied archive; reruns write to the requested output
directory. Within each replication, data, folds, and multipliers
are shared across methods and amplitudes. The plus-one upper-tail p-value is
one for the interpolating full-sample learner under both the null and
alternatives.

For a short execution check, use `--replications 2 --bootstrap-draws 19` with
another output directory. Run the numerical checks with:

```sh
python -m unittest discover -s tests -p test_interpolation.py
```
