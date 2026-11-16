# Conditional average treatment effects

The CATE experiment uses the hierarchical design with `n=400`, `p=20`, and
configuration `simulation/configs/cate.json`:

```
Y = b_N(X) + D a f_N(X) + (1 + 0.5 D) epsilon,
epsilon ~ N(0,1),
mu(X) = b_N(X) + a e(X) f_N(X).
```

The signal is odd under the symmetric Gaussian covariate law, so the average
treatment effect is zero at every amplitude. At zero amplitude, the conditional
means agree and the treatment-arm variances are 1 and 2.25.

Run these commands from the repository root:

```sh
python experiments/cate_simulation.py --output runs/cate --jobs 8
python experiments/cate_report.py --inputs runs/cate/shards/* --output runs/cate/report
```

For an execution check, use a separate output directory:

```sh
python experiments/cate_simulation.py --output runs/cate-check --deltas 0,1.2 --rep-stop 2 --jobs 5
```

The grid has 300 replications at each amplitude
`0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.5, 4, 6, 20`. Outcome learners are
cubic splines, random forest, Lasso and DNN, with all 20 covariates supplied to
each learner. Propensity uses spline logistic regression on the first two
coordinates, clipped to `[0.01, 0.99]`. NORD, single splitting and oracle use
200 centered Gaussian multipliers; Yan uses normal calibration. The single
split is treatment-stratified, uses approximately half the observations for
training, and is not swapped.

Covariates, treatment assignments and standard normal errors are shared across
amplitudes and learners within each replication. The seed base is 8012026.
NORD and oracle share the full-sample kernel and multipliers. The oracle is
computed once per dataset and repeated across learner panels. The manuscript
reports 13 distinct test–learner combinations: NORD, single split and Yan for
each learner, plus oracle.

The configuration also computes two-fold CF-NORD as an additional diagnostic.
Independent balanced folds are shared across amplitudes and learners. Each
nuisance fit uses only its training fold; the test pools all 400 held-out
scores and shares the full-sample kernel and multipliers with NORD. Including
these diagnostics gives 17 combinations, 204 amplitude cells and 61,200 test
records. The full run has 18,000 learner/data tasks in 180 shards.

Each worker uses one numerical thread. Repeating the command resumes completed
shards; changes to source, configuration, interpreter or shard grid require a
new output directory.

The report writes rejection rates, Monte Carlo intervals, null-size summaries
and paired rejection differences. `simulation_cate_hierarchical.pdf` is the
manuscript figure. `simulation_cate_hierarchical_crossfit.pdf` is an additional
figure with CF-NORD. `paired_differences.csv` compares NORD with single split;
`paired_crossfit_differences.csv` compares NORD with CF-NORD. Replication-level
records are saved in `combined_raw_results.csv.gz`, with checksums in
`report_manifest.json`. Failed and missing replications remain in the report.
