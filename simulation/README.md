# NORD regression-comparison simulations

Run from this directory after installing the repository requirements:

```sh
python run_simulation.py --config configs/smoke.json --output runs/smoke --device cpu
python summarize.py --inputs runs/smoke --output runs/smoke/summary
python -m unittest discover -s tests
```

`configs/main.json` retains the original regression grid; `configs/single_split.json`
runs the orthogonal test on a held-out half of each observed group. Both use
300 replications, 200 multiplier draws, total sample size 400, and dimensions
20 and 50. To run the four final-manuscript designs, select
`--scenarios additive,product,composition,neural_composition`. The small smoke
configuration checks execution, not size or power.

The manuscript's NORD and GCM curves use a matched rerun with shared nuisance
fits; see [the GCM instructions](../experiments/gcm_comparison_readme.md).
The archived single-split, Yan and oracle curves provide the other comparisons.
The supplementary interpolation and small-sample cross-fitting studies have
separate runners under `../experiments/`.

The additional CATE experiment uses the existing hierarchical design with
treatment-arm noise standard deviations 1 and 1.5. It compares NORD,
single-split, Yan, and oracle with all four outcome learners; see
[the CATE instructions](../experiments/cate_readme.md).

The `late_iv` design adds a valid binary instrument, endogenous treatment,
and a constant 0.6 first stage. All tests use the instrument as the group
variable to test the zero conditional-LATE null. The hierarchical signal,
learners, and calibration budgets match CATE; see [the LATE
instructions](../experiments/late_readme.md) for the construction.

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

The summary combines nonoverlapping runs and reports rejection rates, Monte
Carlo uncertainty, failures, group counts and nuisance diagnostics. Each learner
has its own panel. `rejection_summary.csv` and `primary_rejection_summary.csv`
contain the same complete results; the latter filename is retained for existing
workflows. Historical size-check fields remain in the tables for provenance.

Run the complete selected grids from this directory:

```sh
python run_simulation.py --config configs/main.json --output runs/main --scenarios additive,product,composition,neural_composition --device cpu
python run_simulation.py --config configs/single_split.json --output runs/single-split --scenarios additive,product,composition,neural_composition --device cpu
python summarize.py --inputs runs/main --output runs/main/summary
python summarize.py --inputs runs/single-split --output runs/single-split/summary
```

Fresh runs write their own tables and diagnostic figures. To render the final
manuscript figures from the supplied compact archives, run
`python reproduce_figures.py` from the repository root.
