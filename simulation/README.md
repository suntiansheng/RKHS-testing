# NORD regression-comparison simulations

Run from this directory after installing the repository requirements:

```sh
python run_simulation.py --config configs/smoke.json --output runs/smoke --device cpu
python summarize.py --inputs runs/smoke --output runs/smoke/summary
python -m unittest discover -s tests
```

`configs/main.json` runs NORD, cross-fitted NORD, residual, Yan, and oracle
tests. `configs/single_split.json` runs the orthogonal test on a held-out half
of each observed group. Both retain the original five-scenario grids with
300 replications, 200 multiplier draws, total sample size 400, and dimensions
20 and 50. `configs/smoke.json` covers every method and learner on a smaller
grid; it checks execution, not size or power.

The manuscript displays four scenarios: `additive`, `product`, `composition`,
and `neural_composition` (the sparse hierarchical alternative). The retained
`friedman` scenario and additional methods belong to the original null-size
screening families. Removing them changes those families. The manuscript's
NORD and GCM curves use a separate matched rerun with shared nuisance fits;
see [the GCM experiment instructions](../experiments/gcm_comparison_readme.md).
The archived single-split, Yan, and oracle curves remain the comparators for
that rerun. The supplementary interpolation and small-sample cross-fitting
studies have their own runners under `../experiments/`.

Outcome learners are cubic-spline GAM, random forest, Lasso, and a neural
network. Propensity is estimated by spline logistic regression using the first
two covariates and clipped to `[0.01, 0.99]`. The neural
network chooses its training duration on a validation set, then refits on all
available training observations. The oracle test uses the true nuisances.

The pooled Gaussian design has covariance `Sigma[j,k] = 0.5**abs(j-k)`, random
group counts, and propensity in `(0.35, 0.70)`. Responses follow
`Y = b(X) + D * delta * f(X) + epsilon`, with independent standard normal
errors; `delta` is the manuscript's signal amplitude `a`. Nonadditive
scenarios are calibrated to the additive baseline
mean and standard deviation, and perturbation L2 norm. Data draws are shared
across scenarios, signal levels, and learners using stable seeds. Seeded CPU
runs are reproducible within a fixed numerical environment; accelerator and
library changes can affect floating-point results.

The RKHS statistic uses the scores `(Y - mu_hat(X)) * (D - e_hat(X))` without
subtracting their empirical mean. The Gaussian-kernel bandwidth squared is
the median positive squared pairwise distance in the testing sample.
Gaussian multipliers are centered
within each draw; p-values use the plus-one correction and rejection occurs
when the p-value is at most the test level. Yan's statistic uses separate sample bandwidths and
paired second moments. Single-split bandwidths and scores use only the test
half. Cross-fitting uses stratified folds.

Runs checkpoint atomically and resume on the same command. Configuration,
source, and task-grid hashes prevent incompatible resumes. `--retry-failures`
reruns failed tasks; an interrupted multi-file checkpoint reruns tasks absent
from either checkpoint file. Use separate output directories for separate
grids. Select subsets with `--learners`, `--scenarios`, `--dimensions`,
`--deltas`, `--rep-start`, and `--rep-stop`; preview with `--dry-run`.
CPU learners support `--workers`; runs containing the neural network require
`--workers 1`. Set `--device cpu` for CPU execution.

The summary combines nonoverlapping runs, reports Monte Carlo uncertainty,
failures, group counts, and nuisance diagnostics, and writes PNG/PDF power
curves. Holm adjustment keeps the 170 main null cells and 40 single-split null
cells in separate families. A production size gate requires the entire
corresponding family and 300 valid replications per cell at level 0.05.
Incomplete families remain `not_evaluated`. Primary tables and figures omit
curves that fail a complete size gate; diagnostic figures retain them. Learners
are summarized separately.

To reproduce the complete screening families, run the null cells before the
alternatives, preserving separate output directories:

```sh
python run_simulation.py --config configs/main.json --output runs/main-null --deltas 0 --device cpu
python run_simulation.py --config configs/single_split.json --output runs/split-null --deltas 0 --device cpu
python summarize.py --inputs runs/main-null --output runs/main-null/summary
python summarize.py --inputs runs/split-null --output runs/split-null/summary
```

Use the corresponding configuration with
`--deltas 0.1,0.2,0.3,0.5,0.8,1.2,1.6,2.5,4,6,20` for the alternatives, with
`runs/main-alternatives` and `runs/split-alternatives` as output directories.
Then combine each null run with its alternatives:

```sh
python summarize.py --inputs runs/main-null runs/main-alternatives --output runs/main-summary
python summarize.py --inputs runs/split-null runs/split-alternatives --output runs/split-summary
```

These commands produce new summaries and retain all original screening cells.
To render the manuscript's main power curves from the supplied summaries,
run `python plot_results.py` from the repository root; no model fitting is
needed for that step.
