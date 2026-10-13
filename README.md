# NORD: tests for equality of regression functions

Code accompanying *Neyman-Orthogonal Tests for Equality of Regression Functions*.
NORD uses the residual product `(Y - mu(X)) * (D - e(X))`, an RKHS norm statistic,
and centered Gaussian multiplier calibration with a plus-one p-value. This
repository reproduces the figures in the final JRSSB main paper and supplement.

## Installation

The archived experiments used Python 3.12.2 and the dependency versions in
`requirements-reproduce.txt`:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Run the commands below from the repository root. `requirements.txt` lists
broader supported versions. Refitting may give small numerical differences
across hardware or library versions, especially for forests and neural networks.

## Reproduce every manuscript figure

```sh
python reproduce_figures.py
```

This verifies the supplied result-table hashes and writes all **14 figure PDFs**,
summary tables and a reproduction manifest to `results/figures/jrssb/`.
Use `--output PATH` to choose another directory. No model fitting, corpus download
or external manuscript directory is needed. The filenames match the submission;
older suffixes such as `_v9` are retained as asset names.

| Manuscript content | Supplied figure assets |
| --- | --- |
| Main regression experiment and three supplementary alternatives | `simulation_hierarchical_v9.pdf`, `simulation_{additive,product,composition}_compact_v9.pdf` |
| Supplementary interpolation and regularized cross-fitting experiments | `simulation_crossfit_interpolation_v9.pdf`, `simulation_crossfit_additive_{null_size,power_difference}.pdf` |
| CATE and conditional LATE experiments | `simulation_{cate,late}_hierarchical.pdf` |
| Four rows of the main CaSiNo comparison | `casino_baseline_cate.pdf`, `casino_baseline_max_satisfaction.pdf`, `casino_results_v9.pdf`, `casino_results_max_satisfaction.pdf` |
| Supplementary opening-word contrasts | `casino_word_contrasts_v5.pdf` |

The main CaSiNo figure combines four assets in LaTeX. The repository generates
each source asset. [results/README.md](results/README.md) describes the archived
inputs, their provenance and the distinction between plotting and refitting.

## Run the simulations

| Experiment | Sample size | Replications per cell | Gaussian multiplier draws |
| --- | --- | --- | --- |
| Main regression designs and matched GCM comparison | 400 | 300 | 200 |
| Hierarchical CATE | 400 | 300 | 200 |
| Hierarchical conditional LATE | 400 | 300 | 200 |
| Interpolating nearest-neighbour regression | 400 | 500 | 199 |
| NORD versus two-fold CF-NORD | 120, 160 | 2,000 null; 1,000 alternative | 999 |

A short execution check covers all main methods and learners:

```sh
python simulation/run_simulation.py --config simulation/configs/smoke.json --output runs/smoke --device cpu
python simulation/summarize.py --inputs runs/smoke --output runs/smoke/summary
```

See [simulation/README.md](simulation/README.md) for the full regression runs,
configuration options and checkpointing. The matched GCM comparison uses:

```sh
python experiments/gcm_comparison.py --phase null --workers 4
python experiments/gcm_comparison.py --phase power --workers 4
python experiments/publish_gcm_comparison.py
```

[The GCM instructions](experiments/gcm_comparison_readme.md) also cover the
paired NORD comparison and signal-cancellation diagnostic.

For CATE and conditional LATE:

```sh
python experiments/cate_simulation.py --jobs 8
python experiments/cate_report.py --inputs runs/cate_hierarchical/shards/* --output runs/cate_hierarchical/report
python experiments/late_simulation.py --jobs 8
python experiments/late_report.py --inputs runs/late_hierarchical/shards/* --output runs/late_hierarchical/report
```

See the [CATE](experiments/cate_readme.md) and [LATE](experiments/late_readme.md)
instructions for the data-generating processes, paired seeds and oracle.

For the two supplementary cross-fitting experiments:

```sh
python experiments/interpolation.py --output runs/interpolation --bootstrap-draws 199
python experiments/crossfit_comparison.py --output runs/crossfit_comparison/n120_n160_k2 --sample-sizes 120,160 --workers 4 --write-raw
python experiments/plot_crossfit_manuscript.py --run runs/crossfit_comparison/n120_n160_k2 --output runs/crossfit_comparison/figures
```

See the [interpolation](experiments/interpolation_readme.md) and
[cross-fit](experiments/crossfit_comparison_readme.md) instructions for details.
Full experiments require substantially more computation than the smoke run.

## Run the CaSiNo application

```sh
python application/download_data.py
python application/joint_analysis.py --results-dir runs/casino-joint --workers 2
python application/plot_joint.py --results-dir runs/casino-joint --output-dir runs/casino-joint/figures
python application/word_analysis.py --output-dir runs/casino-words
```

The joint analysis fits the four comparisons in the main paper: two outcomes
(mean satisfaction and joint maximum satisfaction), each with baseline or
baseline-plus-opening-text covariates. It includes nuisance-learner, bandwidth,
participant-selection and age-coding sensitivities. The word analysis produces
the descriptive supplementary plot. See [application/README.md](application/README.md).

The downloader verifies the public corpus checksum. Raw dialogues, participant
records, fitted predictions and replication-level simulation archives are not
included. The data source is the [ConvoKit CaSiNo corpus](https://convokit.cornell.edu/documentation/casino-corpus.html),
introduced by [Chawla et al. (2021)](https://aclanthology.org/2021.naacl-main.254/).

## Tests

```sh
python -m unittest discover -s simulation/tests
python -m unittest discover -s tests
python -m unittest discover -s application/tests
```

Tests cover scores, multiplier centering, sample separation, learner fitting,
seeds, checkpoint integrity and archive consistency. Corpus-dependent tests use
`CASINO_CORPUS_DIR` and `CASINO_RESULTS_DIR`; other tests require no download.
