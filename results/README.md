# Simulation archives

These files are the inputs to `python reproduce_figures.py`. The command checks
their SHA-256 hashes against `manifest.json` before rendering the seven
simulation figures used by `v10.tex`. All files are compact summaries or run
specifications; raw Monte Carlo draws and fitted models are not included.

| Files | Contents |
| --- | --- |
| `frozen_summary.csv` | Original main grid, including the 170-cell null-screening family and undisplayed configurations |
| `single_split_summary.csv` | Single-split experiment and its separate 40-cell null family |
| `gcm_summary.csv`, `gcm_paired_nord_summary.csv` | Matched GCM and NORD results: 384 cells per procedure, 300 replications per cell |
| `gcm_null_size.csv`, `gcm_main_null_audit.csv` | GCM's 32-cell null screen and the refreshed 170-cell NORD family |
| `gcm_paired_differences.csv` | Paired NORD-minus-GCM rejection differences and Monte Carlo intervals |
| `gcm_signal_diagnostic.csv`, `gcm_signal_diagnostic_metadata.json` | True-signal cancellation diagnostic and calculation metadata |
| `gcm_comparison/` | Protocol and reference-sample calibration used by the signal diagnostic |
| `interpolation.csv` | Twelve interpolation cells, with 500 replications each |
| `crossfit_comparison/n120_n160_k2/` | Configuration, completion record, 306 summary cells, 144 paired comparisons, and 34 null cells |

## Figure inputs

The additive, product, composition, and hierarchical figures use the matched
NORD/GCM summaries. Single-split, Yan, and oracle curves come from the original
archives. NORD and GCM therefore share fitted scores; the single-split comparison
was run separately. These numerical comparisons should not be described as
paired across every method.

Null screens use two-sided exact binomial tests with Holm correction at level
0.05. The 170-cell main, 40-cell single-split, and 32-cell GCM families remain
separate. Curves failing these complete screens are omitted from the four main
power figures. The two regularized cross-fit figures retain all cells and mark
the failures of their separate 34-cell screen. Monte Carlo intervals are
pointwise. Passing a screen does not establish population size control.

## Interpolation multiplier count

The archived interpolation run and its supplied figure used 199 Gaussian
multiplier draws. The interpolation paragraph in `v10.tex` states 200. This
package preserves the archived results and documents the difference.

`experiments/interpolation.py --bootstrap-draws 199` uses the archived setting;
`--bootstrap-draws 200` uses the manuscript's stated setting. Use a separate
output directory when changing the draw count. The main regression grids use
200 draws and the regularized cross-fit comparison uses 999 draws.

## Recomputing results

The experiment READMEs give commands for refitting and checkpoint validation.
Fresh simulations write to `runs/`; they do not replace the supplied archives
unless a publication command is explicitly run. The archive manifest records
the distributed results, so changed summary files must be checked before
updating their hashes. Plot freshly recomputed results through the corresponding
experiment plotter while keeping the supplied archive intact.
