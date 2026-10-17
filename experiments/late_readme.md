# Conditional local average treatment effects

This experiment uses the existing sparse hierarchical baseline `b_N` and signal
`f_N`, Gaussian covariates with `n=400`, `p=20`, and the same reference-sample
normalization. The five tests, four nuisance learners, tuning settings,
amplitude grid, and calibration budgets match the CATE experiment. The
configuration is `simulation/configs/late.json`.

## Instrument and potential outcomes

Let `A | X ~ Bernoulli(pi(X))`, using the existing propensity function, and
let `U ~ Uniform(0,1)` be independent of `X` and instrument `A`. Define

```
D(0) = 1{U <= 0.2},   D(1) = 1{U <= 0.8},
C = 1{0.2 < U <= 0.8},   D = D(A),
Y(d) = b_N(X) + 2(U - 0.5) + d[C a f_N(X) + (1-C)] + (1 + 0.5d) epsilon_d.
```

The two potential-outcome errors are independent standard normals and are
independent of `(X,U,A)`. The observed response is `Y=Y(D)`. The code draws one
independent standard normal per unit and scales it by observed treatment;
this has exactly the observed-data law of `epsilon_D`. The instrument is
conditionally independent of all potential outcomes and treatment choices;
exclusion and monotonicity hold by construction. Principal strata have
probabilities 0.2 (always-takers), 0.6 (compliers), and 0.2 (never-takers).
Observed treatment is endogenous because `U` affects both treatment and outcome.

The conditional LATE is `a f_N(X)`, and the first stage equals 0.6 at every
`X`. Hence

```
E[Y | A=1,X] - E[Y | A=0,X] = 0.6 a f_N(X),
mu(X) = E[Y | X] = b_N(X) + 0.2 + 0.6 pi(X) a f_N(X).
```

Testing equality of the two instrument-specific outcome regressions therefore
tests whether conditional LATE is zero almost surely. NORD, CF-NORD and single
split use score `(Y - mu_hat(X)) * (A - pi_hat(X))`; Yan fits the two
instrument-specific outcome regressions. In the shared simulation interface, the field `d` stores
the instrument for this design; actual treatment counts are retained in the
data-generation diagnostics. Tests do not group observations
by actual treatment or estimate a Wald ratio. The oracle uses the displayed
`mu(X)` and true instrument propensity.

The hierarchical signal is odd under a centrally symmetric covariate law.
Because compliance is independent of `X`, the population LATE is zero at
every amplitude. The full-population CATE is `0.4 + 0.6 a f_N(X)` and the ATE
is 0.4. Thus the global conditional-LATE null differs from the zero-ATE null.
The fixed positive first stage avoids weak-instrument complications; this
experiment does not assess procedures for weak instruments.

## Run

From the repository root:

```sh
python experiments/late_simulation.py --output runs/late_hierarchical_crossfit --jobs 8
```

The default run has 180 resumable shards of 100 learner/data tasks, for 18,000
tasks and 61,200 test records. Each process uses one numerical thread and
runs the existing estimators and tests. The runner records source hashes,
configuration, interpreter, and shard grid, and executes frozen source copies
in `protocol_sources/`. Repeat the same command to resume. A changed source,
configuration, or shard grid requires a separate output directory.

A small execution check, without claiming final empirical size or power, is:

```sh
python experiments/late_simulation.py --output runs/late_execution_check --deltas 0,1.2 --rep-stop 2 --jobs 5
```

There are 300 replications per amplitude and 200 centered Gaussian multiplier
draws for NORD, CF-NORD, single-split, and oracle; Yan retains its normal
calibration.
The amplitude grid is `0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.5, 4, 6, 20`.
The fitted methods use cubic splines, random forest, Lasso, and the original
DNN. Instrument propensity is estimated using the same spline logistic
procedure. The oracle is computed once per dataset and shown in every panel,
giving 17 distinct test–learner combinations and 204 amplitude cells.

CF-NORD uses two balanced folds formed by an independent random permutation,
shared across learners and amplitudes within each replication. Nuisance fitting,
preprocessing and tuning use only the training fold. The test pools all 400
held-out scores and uses the same full-sample kernel and multiplier draws as
NORD and oracle.

Seed base 8012026 and the reference calibration match the existing experiments.
Within this IV design, covariates, instrument assignments, principal strata,
actual treatment, and standard normal errors are paired across amplitudes
and learners. The design-specific seed key keeps LATE draws separate from
CATE draws. NORD, CF-NORD and oracle share kernel and multiplier seeds.

## Summaries and figures

```sh
python experiments/late_report.py --inputs runs/late_hierarchical_crossfit/shards/* --output runs/late_hierarchical_crossfit/report
```

The report verifies matching raw records and instrument-group diagnostics,
then writes rejection rates, a null-size table, and two four-panel figures.
`simulation_late_hierarchical.pdf` retains NORD, single-split, Yan and oracle
for the main paper; `simulation_late_hierarchical_crossfit.pdf` adds CF-NORD
for the supplement. Both figures use the CATE figure style.

`paired_differences.csv` reports
NORD-minus-single-split rejection rates; `paired_crossfit_differences.csv`
reports NORD-minus-CF-NORD rejection rates, both with paired Monte Carlo
intervals. Pointwise binomial intervals are saved in the summary tables. Failed and missing runs are reported separately.

The report preserves `combined_raw_results.csv.gz` and
`combined_dgp_diagnostics.csv.gz`, with deterministic gzip timestamps and
SHA-256 checksums in `report_manifest.json`. Diagnostics include principal
stratum counts, actual treatment counts within instrument arms, empirical
first stages, and LATE and reduced-form signal norms. The manifest records
input checksums and hashes of both report modules. Partial reports include
every planned cell and are clearly marked incomplete. This single design
assesses empirical calibration and power; it does not establish minimax
optimality or general validity for all fitted learners.
