# RKHS two-sample simulations

Run from this directory after installing the repository requirements:

```sh
python run_simulation.py --config configs/smoke.json --output runs/smoke
python summarize.py --inputs runs/smoke --output runs/smoke/summary
python -m unittest discover -s tests
```

`configs/main.json` runs the full-sample orthogonal, cross-fit, residual, Yan,
and oracle tests. `configs/single_split.json` runs the orthogonal test on a
held-out half of each observed group. Both use 300 replications, 200 bootstrap
draws, total sample size 400, five regression scenarios, and dimensions 20 and
50. `configs/smoke.json` covers every method and learner on a smaller grid.

Outcome learners are cubic-spline GAM, random forest, Lasso, and a neural
network. Propensity is estimated by spline logistic regression. The neural
network chooses its training duration on a validation set, then refits on all
available training observations. The oracle test uses the true nuisances.

The pooled Gaussian design has random group counts and propensity in
`[0.35, 0.70]`. Nonadditive scenarios are calibrated to the additive baseline
mean and standard deviation, and perturbation L2 norm. Data draws are shared
across scenarios, signal levels, and learners using stable seeds. Seeded CPU
runs are reproducible within a fixed numerical environment; accelerator and
library changes can affect floating-point results.

The RKHS statistic uses uncentered scores. Gaussian multipliers are centered
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
