"""Validate and plot the n=120,160 extension beside the completed n=200 study.

The combined figures use archived simulation results and the prespecified joint
51-cell null screen. The original run-level screens are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
from io_utils import atomic_write_csv, atomic_write_json, file_hash
import crossfit_comparison as experiment
import plot_crossfit_comparison as plots


def require(condition, message):
    if not condition:
        raise ValueError(message)


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(
        values[order] * (len(values) - np.arange(len(values)))))
    return adjusted


def validate_run(directory):
    metadata = json.loads((directory / "config.json").read_text())
    config = metadata["config"]
    require(config["null_replications"] == 2000 and config["power_replications"] == 1000
            and config["bootstrap_draws"] == 999 and config["reference_n"] == 10000
            and config["learners"] == list(experiment.LEARNERS)
            and not config["five_fold_sensitivity"], "Unexpected study scope")
    validation = json.loads((directory / "validation.json").read_text())
    require(validation["complete"], f"Incomplete run: {directory}")
    for key, source in [("driver", ROOT / "experiments/crossfit_comparison.py"),
                        ("simcore", ROOT / "simulation/simcore.py")]:
        require(file_hash(source) == metadata["source_hashes"][key],
                f"Simulation source changed: {source}")
    completed = experiment.read_completed(directory, config, metadata["fingerprint"])
    expected_tasks = len(config["sample_sizes"]) * 2000
    require(len(completed) == expected_tasks, "Missing or corrupt checkpoints")
    require(set(completed) == {(n, rep) for n in config["sample_sizes"] for rep in range(2000)},
            "Replication identifiers do not cover the simulation grid")
    fits = warnings = fallbacks = 0
    for (n, rep), record in completed.items():
        expected_fits = len(experiment.task_amplitudes(config, rep)) * 4 * 3
        require(len(record["fitting_diagnostics"]) == expected_fits, "Missing outcome fit")
        fit_keys = {(fit["amplitude"], fit["learner"], fit["method"], fit["fold"])
                    for fit in record["fitting_diagnostics"]}
        expected_keys = {(a, learner, method, fold)
                         for a in experiment.task_amplitudes(config, rep)
                         for learner in experiment.LEARNERS
                         for method, fold in [("full", 0), ("cf2", 0), ("cf2", 1)]}
        require(fit_keys == expected_keys, "Incomplete nuisance-fitting grid")
        for fit in record["fitting_diagnostics"]:
            size = n if fit["method"] == "full" else n // 2
            require(fit["method"] in ["full", "cf2"], "Unexpected fit method")
            require(fit["n_train"] == fit["n_test"] == size, "Incorrect fitting sample size")
            fits += 1
            warnings += fit.get("convergence_warnings", 0)
            fallbacks += int(fit.get("fallback_used", False))

    raw = pd.read_csv(directory / "raw.csv.gz")
    keys = ["n", "p", "scenario", "learner", "method", "folds", "propensity_mode", "amplitude", "alpha"]
    require(len(raw) == len(config["sample_sizes"]) * 170000, "Unexpected raw row count")
    require(not raw.duplicated(keys + ["rep"]).any(), "Duplicate result rows")
    require(bool((raw.score_n == raw.n).all()), "A statistic dropped observations")
    require(bool((raw.reject == (raw.p_value <= raw.alpha).astype(int)).all()),
            "Rejection decisions differ from p-values")
    required = ["p_value", "statistic", "critical_value", "independent_mu_mse",
                "independent_e_mse", "score_error_R", "score_error_Q", "score_error_energy"]
    require(bool(np.isfinite(raw[required]).all().all()), "Nonfinite simulation result")
    require(bool(raw.p_value.between(.001, 1).all()), "Invalid plus-one p-value")
    np.testing.assert_allclose(raw.p_value * 1000, np.round(raw.p_value * 1000), atol=1e-10)
    np.testing.assert_allclose(raw.score_error_Q,
        np.maximum(raw.score_error_energy - raw.score_error_R**2 / raw.n, 0), atol=1e-10)

    summary = pd.read_csv(directory / "summary.csv")
    grouped = raw.groupby(keys).reject.agg(["size", "sum", "mean"]).sort_index()
    saved = summary.set_index(keys).sort_index()
    require(grouped.index.equals(saved.index), "Summary cells differ from raw results")
    np.testing.assert_array_equal(grouped["size"], saved.replications)
    np.testing.assert_array_equal(grouped["sum"], saved.rejections)
    np.testing.assert_allclose(grouped["mean"], saved.rejection_rate, atol=1e-14)
    diagnostic_columns = ["independent_mu_mse", "independent_e_mse", "sqrt_n_rmse_product",
                          "score_error_R", "score_error_Q", "score_error_energy"]
    means = raw.groupby(keys)[diagnostic_columns].mean().sort_index()
    np.testing.assert_allclose(means, saved[diagnostic_columns], atol=1e-12)
    require(bool(raw.loc[raw.propensity_mode.eq("known"), "independent_e_mse"].eq(0).all()),
            "Known-propensity control has nonzero propensity error")
    targets = np.where(summary.amplitude.eq(0), 2000, 1000)
    np.testing.assert_array_equal(summary.replications, targets)

    differences = pd.read_csv(directory / "paired_differences.csv")
    pair_keys = ["n", "learner", "propensity_mode", "amplitude"]
    expected_pairs = []
    for values, group in raw[raw.learner.ne("oracle")].groupby(pair_keys):
        wide = group.pivot(index="rep", columns="method", values="reject")
        require(not wide[["full", "cf2"]].isna().any().any(), "Unpaired rejection indicators")
        difference = wide.full - wide.cf2
        expected_pairs.append(dict(zip(pair_keys, values)) | dict(
            full_minus_crossfit=difference.mean(),
            paired_mc_se=difference.std(ddof=1) / np.sqrt(len(difference))))
    expected = pd.DataFrame(expected_pairs).set_index(pair_keys).sort_index()
    paired = differences.set_index(pair_keys).sort_index()
    require(expected.index.equals(paired.index), "Saved paired comparison cells differ")
    np.testing.assert_allclose(expected, paired[expected.columns], atol=1e-14)

    null = pd.read_csv(directory / "null_size.csv")
    exact = [binomtest(int(s), int(r), .05).pvalue for s, r in zip(null.rejections, null.replications)]
    np.testing.assert_allclose(null.binomial_p_value, exact, atol=1e-14)
    np.testing.assert_allclose(null.holm_adjusted_p_value, holm(exact), atol=1e-14)
    null = null.rename(columns={"holm_adjusted_p_value": "run_holm_adjusted_p_value",
                                "size_screen": "run_size_screen"})
    null["run_null_family_cells"] = len(null)
    details = dict(run=Path(os.path.relpath(directory, ROOT)).as_posix(), checkpoints=len(completed),
                   result_rows=len(raw), cells=len(summary), outcome_fits=fits,
                   convergence_warnings=warnings, fallback_fits=fallbacks)
    return config, summary, differences, null, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, default=ROOT / "runs/crossfit_comparison/n120_n160_k2")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/crossfit_comparison/n200_k2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/crossfit_sample_sizes")
    args = parser.parse_args()
    sources = [args.extension.resolve(), args.reference.resolve()]
    results = [validate_run(source) for source in sources]
    configs = [result[0] for result in results]
    for key in configs[0]:
        if key != "sample_sizes":
            require(configs[0][key] == configs[1][key], f"Unmatched experiment specification: {key}")
    require(configs[0]["sample_sizes"] == [120, 160] and configs[1]["sample_sizes"] == [200],
            "Unexpected comparison grid")
    config = configs[0] | {"sample_sizes": [120, 160, 200]}
    summary = pd.concat([result[1] for result in results], ignore_index=True)
    differences = pd.concat([result[2] for result in results], ignore_index=True)
    null = pd.concat([result[3] for result in results], ignore_index=True)
    null["holm_adjusted_p_value"] = holm(null.binomial_p_value)
    null["size_screen"] = np.where(null.holm_adjusted_p_value <= .05, "flagged", "not_flagged")
    require(len(summary) == 459 and len(null) == 51, "Incomplete combined grid")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in [("summary.csv", summary), ("paired_differences.csv", differences),
                             ("null_size.csv", null)]:
        atomic_write_csv(output / filename, frame)

    # Reuse the figure builder without changing the historical reproduction code.
    original_save = plots.save
    def save_with_manuscript_names(figure, destination, name):
        flagged = null[null.propensity_mode.eq("estimated")
                       & null.method.isin(["full", "cf2"]) & null.size_screen.eq("flagged")]
        if name.endswith(("estimated_propensity", "power_difference")) and len(flagged):
            for row, n in enumerate(sizes):
                for col, (learner, _) in enumerate(plots.LEARNERS):
                    if (flagged.n.eq(n) & flagged.learner.eq(learner)).any():
                        axis = figure.axes[row * 4 + col]
                        if not axis.get_title().endswith("*"):
                            axis.set_title(axis.get_title() + "*", pad=5)
            figure.text(.5, -.025, "* Null size flagged by the 51-cell Holm screen; all curves retained.",
                        ha="center", fontsize=7, color=".3")
        if name.endswith("null_size") and null.size_screen.eq("flagged").any():
            figure.text(.5, -.015, "* Null size flagged by the 51-cell Holm screen.",
                        ha="center", fontsize=7, color=".3")
        for label in figure.findobj(Text):
            text = label.get_text()
            text = text.replace("Full sample - two-fold cross-fit", "NORD - CF-NORD")
            text = text.replace("Full sample", "NORD").replace("Two-fold cross-fit", "CF-NORD (K=2)")
            label.set_text(text)
        if name.endswith("null_size"):
            figure.subplots_adjust(bottom=.13, top=.94, hspace=.70)
        original_save(figure, destination, name.replace("crossfit_additive", "crossfit_sample_sizes"))
    plots.save = save_with_manuscript_names
    plots.style()
    sizes = config["sample_sizes"]
    plots.power(summary, null, sizes, "estimated", output)
    plots.power(summary, null, sizes, "known", output)
    plots.paired_gain(differences, sizes, output)
    plots.outcome_error(summary, sizes, output)
    plots.null_size(null, sizes, output)

    validation = dict(complete=True, runs=[result[4] for result in results],
                       combined_cells=459, combined_null_cells=51,
                       combined_null_flags=int(null.size_screen.eq("flagged").sum()))
    plots.write_report(summary, null, differences, validation, config, output)
    report = (output / "report.md").read_text()
    report = report.replace("Full sample", "NORD").replace("Cross-fit", "CF-NORD")
    report = report.replace("**Four-learner full-sample versus two-fold cross-fit simulation**",
                            "**NORD and CF-NORD: sample-size comparison**")
    report = report.replace("prespecified two-sided Holm screen", "prespecified combined 51-cell two-sided Holm screen")
    flagged = null[null.size_screen.eq("flagged")].sort_values(["n", "learner", "method"])
    calibration_note = "\nCalibration findings:\n\n"
    for row in flagged.itertuples():
        method = "NORD" if row.method == "full" else "CF-NORD"
        learner = dict(plots.LEARNERS).get(row.learner, row.learner)
        calibration_note += (f"- {method}, {learner}, n={row.n}, {row.propensity_mode} propensity: "
                             f"null rejection rate {row.rejection_rate:.4f}; joint Holm p-value "
                             f"{row.holm_adjusted_p_value:.4f}.\n")
    calibration_note += ("\nThese flagged curves are retained as small-sample stress results. "
                         "Their alternative rejection rates do not provide a comparison at equal "
                         "test size. Both the original 34-cell extension screen and the combined "
                         "51-cell screen flag the same three CF-NORD cells. No NORD or known-propensity "
                         "cell is flagged; this is an empirical screen, not a proof of size control. "
                         "For splines at n=160, known-propensity null rates are 0.0525 for NORD "
                         "and 0.0530 for CF-NORD; at a=4 their powers are 0.631 and 0.486 "
                         "(paired difference 0.145, pointwise 95% interval [0.117, 0.173]). "
                         "This control provides a clearer comparison of the cost of outcome fitting.\n\n")
    report = report.replace("Null rejection rates with estimated propensity:",
                            calibration_note + "Null rejection rates with estimated propensity:")
    report += ("\nThe two new sample sizes use a 34-cell null screen in their original run; "
               "the existing n=200 run retains its original 17-cell screen. The combined table "
               "also supplies the 51-cell screen used for figure annotations. Original screen "
               "columns remain in `null_size.csv`. The two new sizes were fixed before "
               "computing their results; the timing batch was retained without changing the design.\n")
    report += "\nPower at a=3.2 with estimated propensity:\n\n"
    report += "| n | Learner | NORD | CF-NORD | Difference | Paired 95% interval |\n|---:|---|---:|---:|---:|---|\n"
    selected = differences[differences.propensity_mode.eq("estimated") & differences.amplitude.eq(3.2)]
    for n in sizes:
        for learner, label in plots.LEARNERS:
            row = selected[selected.n.eq(n) & selected.learner.eq(learner)].iloc[0]
            report += (f"| {n} | {label} | {row.full_power:.3f} | {row.crossfit_power:.3f} | "
                       f"{row.full_minus_crossfit:+.3f} | [{row.ci_lower:+.3f}, {row.ci_upper:+.3f}] |\n")
    spline = differences[differences.propensity_mode.eq("estimated")
                         & differences.learner.eq("gam") & differences.amplitude.eq(2.5)].set_index("n")
    report += "\nInterpretation across sample sizes:\n\n"
    report += ("At the common amplitude a=2.5, the estimated-propensity spline power gaps "
               "(NORD minus CF-NORD) are " + ", ".join(
                   f"{spline.loc[n, 'full_minus_crossfit']:.3f} at n={n}" for n in sizes) +
               ". Reducing n does not enlarge the spline gap at this amplitude. "
               "The curves should also be read over the full amplitude grid: smaller samples "
               "shift useful power comparisons toward stronger signals.\n")
    report += "\nFor example, the complete n=160 comparison at a=4 is:\n\n"
    report += "| Learner | NORD | CF-NORD | Difference | Paired 95% interval |\n|---|---:|---:|---:|---|\n"
    selected = differences[differences.propensity_mode.eq("estimated")
                           & differences.amplitude.eq(4) & differences.n.eq(160)].set_index("learner")
    for learner, label in plots.LEARNERS:
        row = selected.loc[learner]
        report += (f"| {label} | {row.full_power:.3f} | {row.crossfit_power:.3f} | "
                   f"{row.full_minus_crossfit:+.3f} | [{row.ci_lower:+.3f}, {row.ci_upper:+.3f}] |\n")
    report += ("\nThe prediction diagnostics show that cross-fitting increases nuisance error, "
               "but reducing n also worsens the full-sample fit and reduces the observations "
               "available to both statistics. A larger training-size penalty therefore need "
               "not yield a larger power gap. These simulations support a learner-dependent "
               "benefit from full-sample fitting, not a monotone or universal advantage as n "
               "decreases. The n=200 experiment remains a useful reference; both smaller "
               "sample sizes are retained as sensitivity results.\n")
    (output / "report.md").write_text(report)
    hashes = {}
    for source in sources:
        for name in ["config.json", "validation.json", "raw.csv.gz", "summary.csv", "paired_differences.csv", "null_size.csv"]:
            hashes[Path(os.path.relpath(source / name, ROOT)).as_posix()] = file_hash(source / name)
    for source in [Path(__file__).resolve(), ROOT / "experiments/plot_crossfit_comparison.py"]:
        hashes[str(source.relative_to(ROOT))] = file_hash(source)
    validation.update(config=config, source_sha256=hashes,
                      outputs_sha256={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.suffix in [".csv", ".pdf", ".png", ".md"]},
                      package_versions=experiment.core.package_versions())
    atomic_write_json(output / "publication_validation.json", validation)
    print(json.dumps({"output": str(output), "complete": True, "cells": 459,
                      "null_cells": 51, "null_flags": validation["combined_null_flags"],
                      "runs": validation["runs"]}, indent=2))


if __name__ == "__main__":
    main()
