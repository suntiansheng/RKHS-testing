# NORD: tests for equality of regression functions

Simulation code accompanying *Neyman-Orthogonal Tests for Equality of Regression
Functions*. NORD uses the residual product `(Y - mu(X)) * (D - e(X))`, an RKHS
norm statistic, and centered Gaussian multiplier calibration with a plus-one
p-value. This package reproduces the simulation figures in `v10.tex`.

## Installation

Use Python 3.12. The archived experiments used Python 3.12.2 and the versions
pinned below:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Run the following commands from the repository root. `requirements.txt` gives
broader supported dependency ranges; the pinned file is preferable for
reproduction. CPU runs use fixed seeds. Refitting can still differ across
hardware and numerical libraries, especially for forests and neural networks.

## Reproduce the manuscript figures

```sh
python reproduce_figures.py
```

This checks the supplied archive hashes and writes all seven simulation PDFs,
three summary tables, and a reproduction report to `results/figures/v10/`.
Use `--output PATH` to choose another directory. The five filenames ending in
`_v9.pdf` are retained because these are the filenames used by `v10.tex`.
No model fitting, downloaded data, or external manuscript directory is needed.

The figures cover the four main regression designs, the interpolation example,
and the two comparisons between NORD and two-fold CF-NORD at sample sizes 120
and 160. The main curves use matched NORD/GCM reruns and archived single-split,
Yan, and oracle results. See [the archive description](results/README.md) for
input files, screening families, and the interpolation draw-count discrepancy.

## Run the simulations

The three experiments have different calibration budgets:

| Experiment | Sample size | Replications per cell | Multiplier draws |
| --- | --- | --- | --- |
| Main regression designs and matched GCM comparison | 400 | 300 | 200 |
| Interpolating nearest-neighbour regression | 400 | 500 | 199 in the archive; 200 stated in v10 |
| Regularized NORD versus two-fold CF-NORD | 120, 160 | 2,000 under the null; 1,000 under alternatives | 999 |

A short execution check covers every main method and learner:

```sh
python simulation/run_simulation.py --config simulation/configs/smoke.json --output runs/smoke --device cpu
python simulation/summarize.py --inputs runs/smoke --output runs/smoke/summary
```

The full main and single-split grids, configuration options, and resume behavior
are documented in [simulation/README.md](simulation/README.md). The archived
main configuration retains an additional Friedman design and procedures needed
to preserve its original 170-cell null-screening family.

For the matched GCM comparison:

```sh
python experiments/gcm_comparison.py --phase null --workers 4
python experiments/gcm_comparison.py --phase power --workers 4
python experiments/publish_gcm_comparison.py
```

See [the GCM instructions](experiments/gcm_comparison_readme.md) for checkpoint
validation, the paired NORD comparison, and the signal-cancellation diagnostic.

For the regularized cross-fit comparison:

```sh
python experiments/crossfit_comparison.py --output runs/crossfit_comparison/n120_n160_k2 --sample-sizes 120,160 --workers 4 --write-raw
python experiments/plot_crossfit_manuscript.py --run runs/crossfit_comparison/n120_n160_k2 --output runs/crossfit_comparison/figures
```

See [the cross-fit instructions](experiments/crossfit_comparison_readme.md) for
shared draws, nuisance diagnostics, and the 34-cell null screen. Flagged curves
remain visible and are marked in the figures.

To rerun the archived interpolation experiment:

```sh
python experiments/interpolation.py --output runs/interpolation --bootstrap-draws 199
```

To use the 200 draws stated in `v10.tex`, run it separately with
`--bootstrap-draws 200 --output runs/interpolation-200`. The supplied interpolation
summary and figure retain the original 199-draw experiment. They are not presented
as results from a 200-draw rerun.

Complete runs require substantially more computation than the smoke check.
Raw draws, checkpoints, downloaded data, and generated figures are excluded
from version control; compact summaries are included.

## Tests

```sh
python -m unittest discover -s simulation/tests
python -m unittest discover -s tests
python -m unittest discover -s application/tests
```

The simulation tests cover score construction, multiplier centering, sample
separation, learner fitting, seeds, checkpoint integrity, and size-screening
families. The small smoke run checks execution, not empirical calibration.

## CaSiNo application

The existing application scripts are also available:

```sh
python application/download_data.py
python application/analyze.py --results-dir runs/casino
python application/plot.py --results-dir runs/casino
python application/null_controls.py --data-dir runs/casino --output-dir runs/casino-null
```

The download script verifies the public corpus checksum. Application tests
requiring the corpus or saved outputs use `CASINO_CORPUS_DIR` and
`CASINO_RESULTS_DIR`; other tests run without downloaded data. Raw dialogue
data are not included in this repository. The source is the
[ConvoKit CaSiNo corpus](https://convokit.cornell.edu/documentation/casino-corpus.html),
introduced by [Chawla et al. (2021)](https://aclanthology.org/2021.naacl-main.254/).
