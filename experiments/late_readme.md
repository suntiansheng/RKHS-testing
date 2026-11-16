# Conditional local average treatment effects

The LATE experiment uses the hierarchical design with `n=400`, `p=20`, and
configuration `simulation/configs/late.json`. Let
`A | X ~ Bernoulli(pi(X))` and let `V ~ Uniform(0,1)` be independent of the
covariates and instrument randomization. Define

```
D(0) = 1{V <= 0.2},  D(1) = 1{V <= 0.8},  D = D(A),
C = 1{0.2 < V <= 0.8},
Y(d) = b_N(X) + 2(V - 0.5) + d[C a f_N(X) + (1-C)] + (1 + 0.5d) epsilon_d.
```

The potential-outcome errors are independent standard normals, independent of
`(X,V,A)`. The observed response is `Y=Y(D)`; the code draws one standard normal
per unit to generate its observed-data law. There are 20% always-takers, 60%
compliers and 20% never-takers. The shared `V` makes treatment endogenous.

The conditional LATE is `a f_N(X)`, with first stage 0.6. Thus

```
E[Y | A=1,X] - E[Y | A=0,X] = 0.6 a f_N(X),
mu(X) = b_N(X) + 0.2 + 0.6 pi(X) a f_N(X).
```

All tests compare instrument groups. The shared simulation field `d` stores
`A`; actual treatment and principal-stratum counts are saved in the diagnostics.
NORD estimates the pooled mean and instrument propensity. The oracle uses
the displayed mean and true propensity. The odd signal has population mean
zero, so the marginal LATE is zero at every amplitude, while the population
ATE is 0.4.

Run from the repository root:

```sh
python experiments/late_simulation.py --output runs/late --jobs 8
python experiments/late_report.py --inputs runs/late/shards/* --output runs/late/report
```

For an execution check, use a separate output directory:

```sh
python experiments/late_simulation.py --output runs/late-check --deltas 0,1.2 --rep-stop 2 --jobs 5
```

The amplitude grid, 300 replications per amplitude, learners and tuning settings
match the [CATE experiment](cate_readme.md). NORD, single split and oracle use
200 centered Gaussian multipliers; Yan uses normal calibration. The single
split is stratified by instrument assignment and is not swapped. The manuscript
reports 13 distinct test–learner combinations, including one oracle per dataset.

The configuration also computes two-fold CF-NORD as an additional diagnostic,
giving 17 combinations, 204 amplitude cells and 61,200 test records. Its balanced
folds come from an independent permutation. Nuisance fitting and tuning use
only training folds, and the statistic pools all 400 held-out scores.
NORD, CF-NORD and oracle share their full-sample kernel and multipliers.

Within each replication, covariates, instrument assignments, compliance
variables and standard normal errors are shared across amplitudes and learners.
The seed base is 8012026; the design-specific seed keeps LATE draws separate
from CATE draws. The full run has 18,000 learner/data tasks in 180 shards.
Workers use one numerical thread. Repeating a command resumes completed shards;
source, configuration, interpreter or shard-grid changes require a new output
directory. The runner executes source snapshots saved in `protocol_sources/`.

The report validates the instrument-group diagnostics and writes rejection
rates, Monte Carlo intervals, null-size summaries and paired differences.
`simulation_late_hierarchical.pdf` is the manuscript figure;
`simulation_late_hierarchical_crossfit.pdf` is an additional CF-NORD figure.
`paired_differences.csv` compares NORD with single split, and
`paired_crossfit_differences.csv` compares NORD with CF-NORD. Combined test and
data-generation records are saved as compressed CSVs, with checksums in
`report_manifest.json`. Failed and missing replications remain in the report.
