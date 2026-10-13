# Conditional average treatment effects

This experiment uses the existing sparse hierarchical regression construction
with n=400 and p=20. The Gaussian covariates, propensity, baseline, treatment
effect, reference normalization, amplitude grid, nuisance learners, and
calibration settings match the main experiment. Only the conditional noise
standard deviation changes: it is 1 in the control arm and 1.5 in the treatment
arm. The configuration is `simulation/configs/cate.json`.

The CATE is `a * f_N(X)`. The hierarchical signal is odd and the covariate law
is centrally symmetric, so the population average treatment effect is zero
at every amplitude. At amplitude zero, the conditional means agree while the
outcome variances differ between treatment arms. This experiment therefore
assesses global CATE testing with heterogeneous effects and unequal variances.

## Run

From the repository root, using the installed Python 3.12 environment:

```sh
python experiments/cate_simulation.py --jobs 8
```

The runner starts independent serial processes, each with one numerical
thread. Every process uses `simulation/run_simulation.py` and its existing
estimators and tests. The default grid has 180 resumable shards, each containing
100 learner/data tasks. The source, configuration, interpreter, and shard grid
are recorded and checked on resume. Repeat the same command to resume.

For a small execution check in a separate directory:

```sh
python experiments/cate_simulation.py --output runs/cate_execution_check --deltas 0,1.2 --rep-stop 2 --jobs 5
```

Changing the shard size or requested grid requires a separate output directory.
Do not edit the scientific source or configuration during a run. The original
simulation configurations omit `treatment_noise_scales` and preserve their
original response generation exactly.

## Protocol and outputs

There are 300 replications for each of the 12 original amplitudes, with 200
centered Gaussian multiplier draws for NORD, single-split, and oracle. Yan
retains its normal calibration. Nuisance learners are cubic splines, random
forest, Lasso, and the original DNN. All tuning settings are unchanged.

The three fitted tests are run with each learner. The oracle uses the true
nuisances once per dataset and appears identically in all four learner panels.
The complete run has 18,000 learner/data tasks and 46,800 test records, forming
156 distinct amplitude/test/learner cells. The null comprises 13 distinct
cells. Noise scales, seeds, fit diagnostics, errors, and p-values are retained
in each shard's `raw_results.csv` and `dgp_diagnostics.csv`.

Summarize the complete or partial run with:

```sh
python experiments/cate_report.py --inputs runs/cate_hierarchical/shards/* --output runs/cate_hierarchical/report
```

The report includes rejection rates, Monte Carlo intervals, a null-size table,
and a four-panel power figure. Combined replication-level results are saved
as `combined_raw_results.csv.gz`, with an archive checksum in the report
manifest. Failed and missing runs are reported separately.

The seed base is 8012026, as in the original simulations. Thus calibration
constants and underlying covariates, assignments, and standard normal noise
are exactly paired with the current main design. Full-sample NORD and oracle
also share their kernel and multiplier seeds. The sample-selected kernel
bandwidth and fitted learners are the same empirical implementations as in
the main simulation; no general validity or minimax conclusion follows from
this one design.
