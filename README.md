# NORD: tests for equality of regression functions

Code accompanying *Neyman-Orthogonal Tests for Equality of Regression Functions*.
NORD combines the residual product `(Y - mu(X)) * (D - e(X))` with an RKHS norm
and centered Gaussian multiplier calibration. Tests use a plus-one p-value.

## Installation

The archived experiments used Python 3.12.2. Install their dependency versions:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Run commands from the repository root. `requirements.txt` lists compatible
version ranges. Refitting forests and neural networks may produce small
numerical differences across platforms or library versions.

## Reproduce the manuscript figures

```sh
python reproduce_figures.py
```

The command verifies the archived table checksums and writes the 16 figure PDFs,
summary tables, and an output manifest to `results/figures/jrssb/`.
Use `--output PATH` to change the destination. It requires no model fitting or
corpus download. The table below locates each result in the paper; figure and
section numbers prefixed with S refer to the supplement.

| Experiment | Location in the paper |
| --- | --- |
| Main regression comparison | Figures 1–2, Section 5.2 |
| Random-forest comparisons with PCM | Figures S4–S7, Section S5.2 |
| Interpolation | Figure S1, Section S4.4 |
| Spline NORD versus CF-NORD | Figures S2–S3, Section S4.5 |
| CATE and conditional LATE | Figure 3, Section 5.3; Figure 4, Section 5.4 |
| CaSiNo comparisons | Figure 5, Section 6 |
| Opening-word contrasts | Figure S8, Section S6.2.3 |

Figure 5 combines four source assets, one for each row (i)–(iv).
[results/README.md](results/README.md) describes the supplied tables and provenance.
All reported rejection curves are retained, including procedures with elevated
null rejection rates. PCM's null rates prevent comparison at a common type-I error.

## Run the simulations

| Experiment | Sample size | Replications per cell | Multiplier draws |
| --- | --- | --- | --- |
| Regression designs and matched GCM comparison | 400 | 300 | 200 |
| Random-forest PCM, one and six splits | 400 | 300 | Normal calibration |
| Hierarchical CATE and conditional LATE | 400 | 300 | 200 |
| Interpolating nearest-neighbour regression | 400 | 500 | 199 |
| Spline NORD versus CF-NORD | 120, 160 | 5,000 null; 3,000 alternative | 999 |

For a short execution check:

```sh
python simulation/run_simulation.py --config simulation/configs/smoke.json --output runs/smoke --device cpu
python simulation/summarize.py --inputs runs/smoke --output runs/smoke/summary
```

Full regression and PCM commands are in [simulation/README.md](simulation/README.md).
PCM requires R with `ranger` and `jsonlite`. Its upstream implementation is pinned
and checked before execution.

The other studies have separate runners and instructions:

- [Matched NORD and GCM](experiments/gcm_comparison_readme.md)
- [CATE](experiments/cate_readme.md) and [conditional LATE](experiments/late_readme.md)
- [Interpolation](experiments/interpolation_readme.md)
- [Spline cross-fitting comparison](experiments/spline_crossfit_readme.md), including the independent pilot and five-fold sensitivity

The CATE and LATE manuscript figures contain NORD, single split, Yan, and a shared
oracle: 13 distinct test-learner combinations. The supplied runs also contain
two-fold CF-NORD as an additional diagnostic, giving 17 combinations in each
archive. Earlier exploratory cross-fitting studies remain available under
`experiments/`; their results are separate from the reported spline study.

## CaSiNo application

```sh
python application/download_data.py
python application/joint_analysis.py --results-dir runs/casino-joint --workers 2
python application/plot_joint.py --results-dir runs/casino-joint --output-dir runs/casino-joint/figures
python application/word_analysis.py --output-dir runs/casino-words
```

The analysis compares aligned and maximal-integrative-potential groups using a
participant-disjoint sample. It tests mean satisfaction and joint maximum
satisfaction with 36 baseline covariates or 164 baseline-plus-text covariates.
It also reports learner, bandwidth, participant-selection, and age-coding checks.
See [application/README.md](application/README.md).

The downloader verifies the public [ConvoKit CaSiNo corpus](https://convokit.cornell.edu/documentation/casino-corpus.html),
introduced by [Chawla et al. (2021)](https://aclanthology.org/2021.naacl-main.254/).
Raw dialogues, participant records, fitted predictions, and replication-level
simulation archives are excluded from version control.

## Tests

```sh
python -m unittest discover -s simulation/tests
python -m unittest discover -s tests
python -m unittest discover -s application/tests
```

Tests cover score definitions, multiplier centering, sample separation, fitting,
seeds, checkpoint integrity, native PCM agreement, and archived numerical claims.
Corpus-dependent checks use `CASINO_CORPUS_DIR` and `CASINO_RESULTS_DIR`.
