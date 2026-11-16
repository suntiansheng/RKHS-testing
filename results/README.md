# Archived results

`python reproduce_figures.py` renders the 16 figure assets referenced by
`main.tex` and `supplement.tex`. It checks SHA-256 hashes in `manifest.json`
before reading the tables. Plotting uses completed runs; it does not refit
learners or repeat Monte Carlo experiments.

| Files | Contents |
| --- | --- |
| `frozen_summary.csv`, `single_split_summary.csv` | Original regression and single-split summaries |
| `gcm_summary.csv`, `gcm_paired_nord_summary.csv` | Matched GCM and NORD runs, each with 384 cells and 300 replications per cell |
| `gcm_paired_differences.csv`, `gcm_comparison/` | Paired rejection differences, protocol, and calibration |
| `pcm_summary.csv`, `pcm_provenance.json` | Original one- and six-split R PCM results, including the four reported designs and an additional Friedman design |
| `interpolation.csv` | Twelve cells with 500 replications and 199 multiplier draws each |
| `spline_crossfit/` | Independent pilot, selected amplitudes, and the spline confirmation study with 5,000 null and 3,000 alternative replications |
| `cate_hierarchical/`, `late_hierarchical/` | Each design's 204 cells and 17 null rates, including additional CF-NORD diagnostics |
| `casino/` | Joint-application tests, sample counts, selection summaries, software metadata, and aggregate word counts |
| `crossfit_comparison/n120_n160_k2/` | Earlier exploratory four-learner study; not used for the manuscript's spline figures |

## Numerical provenance

The regression figures use matched NORD/GCM runs and archived single-split,
Yan, and oracle curves. The null-size boxplot contains 32 cells per procedure:
four designs, two dimensions, and four outcome learners. Supplementary PCM
figures use only random-forest panels and retain both PCM curves despite their
elevated null rejection rates. The PCM summary includes all completed cells;
figure builders select the four reported designs.

The spline archive contains both response models: the original additive model
and the model with pooled regression fixed across amplitudes. It includes the
independent pilot, confirmation comparisons with estimated and known propensity,
five-fold sensitivity, and independent-reference prediction diagnostics.

CATE and LATE each have 13 manuscript test-learner combinations: NORD,
single split, and Yan with four learners, plus one oracle reused across panels.
The archives also include two-fold CF-NORD, giving 17 combinations. Those
additional results are available through the experiment report scripts.

CaSiNo tables come from the completed `casino-baseline-plus-text-v1` analysis.
`casino/analysis_plan.json` records the data and historical source hashes;
`casino/analysis_summary.json` records completion and software versions.
The word table contains aggregate counts and bootstrap summaries from the same
participant-disjoint sample, without dialogue text or participant records.

Historical source hashes describe the code used for those runs and remain
unchanged. `manifest.json` records checksums for the files distributed here.
Rendered PDF metadata may differ across runs; checksums of generated outputs
are recorded separately in `reproduction.json`.

## Recompute results

The root and experiment READMEs give rerun commands. Fresh simulations write to
`runs/`, separately from the supplied archives. Replication-level inputs are
excluded from this compact distribution; full reruns regenerate checkpoints,
diagnostics, and reports.
