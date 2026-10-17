# Archived results

`python reproduce_figures.py` renders all 16 figure assets in the final JRSSB
main paper and supplement. It verifies SHA-256 hashes in `manifest.json` before
reading the supplied tables. This path regenerates figures from completed runs;
it does not refit learners or rerun Monte Carlo experiments.

| Files | Contents |
| --- | --- |
| `frozen_summary.csv`, `single_split_summary.csv` | Original regression-test and single-split summaries |
| `gcm_summary.csv`, `gcm_paired_nord_summary.csv` | Matched GCM and NORD summaries, each with 384 cells and 300 replications per cell |
| `gcm_paired_differences.csv` | Paired NORD-minus-GCM rejection differences |
| `gcm_signal_diagnostic.csv`, `gcm_signal_diagnostic_metadata.json`, `gcm_comparison/` | Signal-cancellation calculations, protocol and reference-sample calibration |
| `interpolation.csv` | Twelve cells with 500 replications and 199 multiplier draws each |
| `crossfit_comparison/n120_n160_k2/` | Configuration, completion record, 306 summary cells, 144 paired differences and 34 null rates |
| `cate_hierarchical/`, `late_hierarchical/` | Each experiment's 204 rejection-rate cells, 17 null rates, paired NORD-minus-single-split and NORD-minus-CF-NORD differences, and run validation |
| `casino/` | The 136 joint-application tests, sample counts, selection-overlap summaries, run metadata and word-frequency summaries |

## Figure provenance

The additive, product, composition and hierarchical figures use matched NORD/GCM
reruns. Single-split, Yan and oracle curves come from the original simulations.
NORD and GCM share fitted scores; the original single-split comparison was run
separately. CATE and LATE each compute one oracle per dataset, reused across the
four learner panels. Each design includes NORD, two-fold CF-NORD, single-split
and Yan for each of the four learners. The main assets
`simulation_{cate,late}_hierarchical.pdf` retain the four original tests: NORD,
single-split, Yan and oracle. The supplementary assets
`simulation_{cate,late}_hierarchical_crossfit.pdf` add CF-NORD. CF-NORD
pools all held-out scores and shares the full-sample kernel and multiplier draws with NORD and oracle. Its
paired comparisons are in `paired_crossfit_differences.csv`; the single-split
comparisons remain in `paired_differences.csv`. The interpolating-learner archive
uses 199 multipliers; the main regression, CATE and LATE experiments use 200,
and the regularized cross-fitting comparison uses 999.

The CaSiNo comparison tables come from the completed `casino-baseline-plus-text-v1`
analysis. `casino/analysis_plan.json` records its data and source hashes;
`casino/analysis_summary.json` records completion and software versions. The
word table is the completed opening-text analysis on the same participant-disjoint
main sample. It contains aggregate counts and bootstrap summaries, not dialogue
text or participant-level data. The archived scripts used to produce the runs
predate packaging changes, so their historical source hashes are retained as
provenance rather than treated as current-source checksums.

The archives retain older diagnostic columns and additional configurations from
the original runs. Figure builders select the designs and procedures reported
in the final manuscript. No model fitting or manuscript files are needed.

## Recompute results

The root and experiment READMEs give full rerun commands. Fresh simulations write
to `runs/`; keep these separate from the supplied archives. The CATE and LATE
report manifests describe their original replication-level inputs, which are
excluded from this compact distribution. Full reruns regenerate those inputs,
checkpoints, diagnostic records and reports. `manifest.json` is the integrity
record for files actually distributed here.
