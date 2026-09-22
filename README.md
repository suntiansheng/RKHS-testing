# RKHS regression testing

Code for testing equality of two regression functions with orthogonal RKHS
scores. The statistic uses `(Y - mu(X)) * (D - e(X))`, a Gaussian kernel, and
empirically centered Gaussian multipliers. Rejection uses a plus-one p-value.

The repository contains:

- `simulation/`: full-sample, single-split, and cross-fitted tests; residual,
  Yan, and oracle comparisons; spline, forest, Lasso, and neural-network fits.
- `experiments/interpolation.py`: the nearest-neighbour interpolation experiment.
- `application/`: the CaSiNo satisfaction analysis and within-group null diagnostics.
- `results/` and `plot_results.py`: archived simulation summaries and figure generation.

## Setup

Use Python 3.12 and install the dependencies in a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands below from the repository root. For reproducible CPU execution,
set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. The code was checked with
NumPy 1.26.4, SciPy 1.13.0, pandas 2.3.1, scikit-learn 1.6.1, Matplotlib 3.9.2,
PyTorch 2.2.2, and threadpoolctl 3.6.0. Random seeds are fixed; floating-point
results and fitted forests or networks can vary across library versions and
hardware. The archived summaries retain the reported results.

## Simulations

A small run exercises every method and learner:

```sh
python simulation/run_simulation.py --config simulation/configs/smoke.json --output runs/smoke --device cpu
python simulation/summarize.py --inputs runs/smoke --output runs/smoke/summary
```

For the full grids, run the null cells first:

```sh
python simulation/run_simulation.py --config simulation/configs/main.json --output runs/main-null --deltas 0 --device cpu
python simulation/summarize.py --inputs runs/main-null --output runs/main-null/summary
python simulation/run_simulation.py --config simulation/configs/single_split.json --output runs/split-null --deltas 0 --device cpu
python simulation/summarize.py --inputs runs/split-null --output runs/split-null/summary
```

The main and single-split configurations use 300 replications per cell and
200 multiplier draws. Their 170 and 40 null cells form separate Holm families,
including the undisplayed Friedman scenario. Inspect the null summaries before
running alternatives. A partial family is left `not_evaluated`; curves failing
a complete size screen are excluded from primary power displays. The small
smoke run checks execution only.

To run alternatives, use the corresponding configuration with
`--deltas 0.1,0.2,0.3,0.5,0.8,1.2,1.6,2.5,4,6,20` and a separate output directory.
Supply both null and alternative directories to `summarize.py --inputs`.
Existing runs resume when the same command is repeated. See
[simulation/README.md](simulation/README.md) for subset and device options.

The interpolation study uses known propensity, 500 replications, five folds,
and 199 multiplier draws:

```sh
python experiments/interpolation.py --output runs/interpolation
```

All 6,000 retained rows reproduce the archived interpolation run exactly in
the tested environment. Use `--replications 2 --bootstrap-draws 19` for a quick
execution check.

## Archived figures

```sh
python plot_results.py
```

This writes five PDF figures and size, endpoint, and matched-comparison tables
to `results/figures/`. It reads the supplied summaries without fitting models
or recomputing size-screen decisions. The two summary CSVs preserve their
original screening families; `interpolation.csv` contains the 12 reported
interpolation cells. Wilson intervals are pointwise Monte Carlo intervals.

## CaSiNo application

Download the public corpus, verify its checksum, and run the analysis:

```sh
python application/download_data.py
python application/analyze.py --results-dir runs/casino
python application/plot.py --results-dir runs/casino
python application/null_controls.py --data-dir runs/casino --output-dir runs/casino-null
```

The analysis compares aligned and maximal-potential preference groups using
a participant-disjoint sample and 128 text-hash features from the first six
natural turns. Outcomes are mean satisfaction and joint extreme satisfaction.
It fits ridge, Lasso, and random-forest nuisances with full-sample and five-fold
evaluation. Default bootstrap budgets are 5,000 for the tests, 10,000 for
descriptive contrasts, and 2,000 for word contrasts. These can be changed with
the corresponding CLI options. The null diagnostic uses 500 random-label
assignments per original group and 999 multipliers; it is conditional on the
saved sample, rather than a new sample of dialogues.

Raw dialogue data and generated outputs are excluded from version control.
The data source is the [ConvoKit CaSiNo corpus](https://convokit.cornell.edu/documentation/casino.html),
introduced by [Chawla et al. (2021)](https://aclanthology.org/2021.naacl-main.254/).
The original application environment used Python 3.9.6, NumPy 2.0.2,
pandas 2.3.3, SciPy 1.13.1, scikit-learn 1.6.1, and Matplotlib 3.9.4;
refitting in another environment can slightly change forest propensity fits.

## Checks

```sh
python -m unittest discover -s tests
python -m unittest discover -s simulation/tests
python -m unittest discover -s application/tests
```

Application corpus and numerical replay checks run when `CASINO_CORPUS_DIR`
and `CASINO_RESULTS_DIR` point to a verified corpus and saved analysis outputs.
The remaining checks run without downloaded data. Tests cover multiplier
centering, sample separation, learner fitting, deterministic seeds, resume
behavior, and size-screening families.
