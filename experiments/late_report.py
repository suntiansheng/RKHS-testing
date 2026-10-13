#!/usr/bin/env python3
"""Summarize the conditional LATE simulation and paired rejection-rate differences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import cate_report as shared
from cate_report import (AMPLITUDES, LEARNERS, METHODS, METHOD_LABELS, LEARNER_LABELS,
                         CELLS, REPLICATIONS, LEVEL, atomic_write_gzip_csv,
                         paired_differences, atomic_write_csv, atomic_write_json,
                         file_hash, _canonical, _interval_text)

LATE_OPTIONS = {"always_taker_probability": .2, "complier_probability": .6,
                "latent_outcome_coefficient": 2., "noncomplier_treatment_effect": 1.}
DIAGNOSTIC_COUNTS = ("n0", "n1", "n_always_takers", "n_compliers", "n_never_takers",
                     "n_treated", "n_treated_instrument0", "n_treated_instrument1")


def validate_config(config: dict) -> None:
    """Require the original tuning and grid plus the fixed IV construction."""
    reference = json.loads((ROOT / "simulation/configs/cate.json").read_text())
    reference.update(designs=["late_iv"], late_options=LATE_OPTIONS)
    for key, value in reference.items():
        if key != "name" and config.get(key) != value:
            raise ValueError(f"LATE configuration requires {key}={value!r}")


def validate_results(raw: pd.DataFrame, config: dict) -> None:
    shared.validate_results(raw, config)


def validate_diagnostics(diagnostic: pd.DataFrame, raw: pd.DataFrame) -> None:
    required = {"task_key", "design", "scenario", "p", "delta", "rep", "learner",
                "group_variable", "first_stage_population", "first_stage_empirical",
                "late_empirical_l2", "reduced_form_empirical_l2", *DIAGNOSTIC_COUNTS,
                *LATE_OPTIONS}
    if required - set(diagnostic.columns):
        raise ValueError(f"Missing IV diagnostic columns: {sorted(required-set(diagnostic.columns))}")
    if diagnostic[list(required)].isna().any().any():
        raise ValueError("Missing IV diagnostics")
    if diagnostic.task_key.duplicated().any():
        raise ValueError("Duplicate IV diagnostic task keys")
    if set(diagnostic.task_key) != set(raw.task_key):
        raise ValueError("IV diagnostics do not match the observed test tasks")
    if not diagnostic.group_variable.eq("instrument").all():
        raise ValueError("LATE tests must group by instrument, not observed treatment")
    for key, value in {"design": "late_iv", "scenario": "neural_composition", "p": 20,
                       "first_stage_population": .6, **LATE_OPTIONS}.items():
        if not diagnostic[key].eq(value).all():
            raise ValueError(f"IV diagnostic {key} is inconsistent with the LATE design")
    aligned = raw.drop_duplicates("task_key").merge(diagnostic, on="task_key", suffixes=("_raw", "_iv"))
    for key in ("design", "scenario", "p", "delta", "rep", "learner", "n0", "n1"):
        if not aligned[f"{key}_raw"].eq(aligned[f"{key}_iv"]).all():
            raise ValueError(f"IV diagnostic {key} disagrees with the test records")
    counts = diagnostic[list(DIAGNOSTIC_COUNTS)]
    if (not np.isfinite(counts).all().all() or not counts.eq(counts.astype(int)).all().all()
            or not counts.ge(0).all().all() or not counts.le(400).all().all()):
        raise ValueError("Invalid IV principal-stratum or arm counts")
    if (not diagnostic.n0.add(diagnostic.n1).eq(400).all()
            or not diagnostic.n_always_takers.add(diagnostic.n_compliers).add(diagnostic.n_never_takers).eq(400).all()
            or not diagnostic.n_treated_instrument0.add(diagnostic.n_treated_instrument1).eq(diagnostic.n_treated).all()
            or not diagnostic.n_treated_instrument0.le(diagnostic.n0).all()
            or not diagnostic.n_treated_instrument1.le(diagnostic.n1).all()):
        raise ValueError("Inconsistent IV principal-stratum or arm counts")
    empirical_stage = (diagnostic.n_treated_instrument1 / diagnostic.n1
                       - diagnostic.n_treated_instrument0 / diagnostic.n0)
    if not np.allclose(diagnostic.first_stage_empirical, empirical_stage):
        raise ValueError("Empirical first stages disagree with treatment counts")
    if (not np.isfinite(diagnostic.late_empirical_l2).all()
            or not diagnostic.late_empirical_l2.ge(0).all()
            or not np.allclose(diagnostic.reduced_form_empirical_l2,
                               .6 * diagnostic.late_empirical_l2)):
        raise ValueError("Reduced-form signal must be 0.6 times the conditional LATE signal")
    if diagnostic.groupby("rep")[list(DIAGNOSTIC_COUNTS)].nunique().gt(1).any().any():
        raise ValueError("IV draws differ within matched replication groups")


def load_inputs(inputs: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    frames, diagnostics, provenance = [], [], []
    reference, config = None, None
    for directory in inputs:
        directory = Path(directory).resolve()
        raw_path = directory / "raw_results.csv"
        metadata_path = directory / "metadata.json"
        diagnostic_path = directory / "dgp_diagnostics.csv"
        calibration_path = directory / "calibrations.json"
        metadata = json.loads(metadata_path.read_text())
        current_config = metadata["config"]
        validate_config(current_config)
        digest = hashlib.sha256(_canonical(current_config).encode()).hexdigest()
        if metadata.get("config_hash") != digest:
            raise ValueError(f"Configuration hash mismatch in {directory}")
        signature = (digest, metadata.get("scientific_core_sha256"), metadata.get("runner_sha256"),
                     _canonical(json.loads(calibration_path.read_text())))
        if not signature[1] or not signature[2]:
            raise ValueError(f"Missing simulation source hashes in {directory}")
        if reference is not None and signature != reference:
            raise ValueError("Input runs have inconsistent configurations, calibration, or source code")
        reference, config = signature, current_config
        frames.append(pd.read_csv(raw_path))
        diagnostics.append(pd.read_csv(diagnostic_path))
        provenance.append({"directory": os.path.relpath(directory, ROOT), "raw_sha256": file_hash(raw_path),
                           "metadata_sha256": file_hash(metadata_path),
                           "diagnostics_sha256": file_hash(diagnostic_path),
                           "calibration_sha256": file_hash(calibration_path)})
    if not frames:
        raise ValueError("Supply at least one run directory")
    raw = pd.concat(frames, ignore_index=True)
    diagnostic = pd.concat(diagnostics, ignore_index=True)
    validate_results(raw, config)
    validate_diagnostics(diagnostic, raw)
    return raw, diagnostic, config, provenance


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = shared.summarize(raw)
    summary.insert(0, "design", "late_iv")
    return summary


def screen_size(summary: pd.DataFrame) -> pd.DataFrame:
    null = shared.screen_size(summary)
    null["family"] = "late_13_cells"
    return null


def plot_power(summary: pd.DataFrame, null: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.linewidth": .6, "lines.linewidth": 1.2})
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.5), sharex=True, sharey=True)
    for ax, (learner, title) in zip(axes.flat, LEARNERS):
        for method, label, color, marker, style in METHODS:
            selected_learner = "oracle" if method == "oracle_score" else learner
            selected = summary[summary.learner.eq(selected_learner) & summary.method.eq(method)].sort_values("delta")
            ax.plot(selected.delta, selected.rejection_rate, color=color, marker=marker,
                    linestyle=style, markersize=3, alpha=.94)
        ax.set_title(title + r", $p=20$", pad=5)
        ax.set_xlim(-.3, 20.3)
        ax.set_ylim(-.025, 1.035)
        ax.set_xticks((0, 6, 12, 20))
        ax.set_yticks((0, .5, 1))
        ax.axhline(LEVEL, color=".55", lw=.6, ls=":")
        ax.grid(axis="y", color=".92", lw=.5)
    fig.supylabel("Empirical rejection rate", x=.012, fontsize=9)
    fig.supxlabel(r"Signal amplitude $a$", y=.165, fontsize=9)
    handles = [Line2D([0], [0], color=color, marker=marker, linestyle=style,
                      markersize=4, label=label) for _, label, color, marker, style in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .015),
               ncol=4, frameon=False, handlelength=2.3, columnspacing=1.8)
    fig.subplots_adjust(left=.075, right=.995, bottom=.36, top=.85, wspace=.16, hspace=.39)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"simulation_late_hierarchical.{extension}",
                    bbox_inches="tight", pad_inches=.04, dpi=220)
    plt.close(fig)


def write_report(summary: pd.DataFrame, null: pd.DataFrame, diagnostic: pd.DataFrame,
                 output: Path, *, figure_output: Path | None) -> None:
    complete = bool(summary.complete.all())
    draws = diagnostic.drop_duplicates("rep")
    lines = ["# Conditional LATE simulation", "",
             "Status: **" + ("complete" if complete else "INCOMPLETE") + "**.", "",
             "One hierarchical-signal experiment uses n = 400, p = 20, 300 planned replications per amplitude, "
             "nominal level 0.05, and 200 centered Gaussian multiplier draws. Yan uses its normal calibration.", "",
             "The instrument is A | X ~ Bernoulli(pi(X)). Independently of (X,A), U is uniform on [0,1], "
             "D(0) = 1{U <= 0.2}, D(1) = 1{U <= 0.8}, and C = 1{0.2 < U <= 0.8}. "
             "Observed treatment is D = D(A), and "
             "Y(d) = b_N(X) + 2(U - 0.5) + d[C a f_N(X) + (1-C)] + (1 + 0.5d) epsilon_d. "
             "The potential-outcome errors are independent standard normals, independent of (X,U,A). "
             "The signal and its reference-sample normalization are unchanged from the hierarchical regression setting.", "",
             "The conditional LATE is a f_N(X); the instrument first stage is 0.6, and the conditional "
             "reduced-form contrast is 0.6 a f_N(X). Its zero restriction is equivalent to the global "
             "zero conditional-LATE null in this design. The population LATE is zero at every amplitude. "
             "In contrast, the full-population CATE is 0.4 + 0.6 a f_N(X), and the ATE is 0.4. "
             "Treatment is endogenous through U. All tests compare instrument groups using (Y,X,A). "
             "NORD and single split use score (Y - mu_hat(X))(A - pi_hat(X)); Yan fits the two "
             "instrument-specific outcome regressions. No procedure groups on observed D or estimates a Wald ratio. "
             "The oracle uses mu(X) = b_N(X) + 0.2 + 0.6 pi(X) a f_N(X) and the true instrument propensity.", "",
             f"Completed cells: {int(summary.complete.sum())}/{len(summary)}; "
             f"valid test results: {int(summary.n_valid.sum()):,}/{len(summary)*REPLICATIONS:,}; "
             f"failed results: {int(summary.n_failed.sum()):,}; missing results: {int(summary.n_missing.sum()):,}.", "",
             "The 156 distinct amplitude–test–learner cells comprise 12 amplitudes times 13 combinations. "
             "The oracle is computed once per dataset and repeated in all four learner panels; these repetitions "
             "are not additional replications or additional size tests.", "",
             f"The {len(draws)} distinct observed replication draws have mean empirical instrument first stage "
             f"{draws.first_stage_empirical.mean():.3f} "
             f"(range {draws.first_stage_empirical.min():.3f} to {draws.first_stage_empirical.max():.3f}) "
             f"and mean complier fraction {(draws.n_compliers/400).mean():.3f}. "
             "These diagnostics count each replication once across amplitudes and learners.", "",
             "## Null rejection rates", "",
             "Intervals are pointwise 95% Clopper–Pearson binomial intervals, conditional on successful runs. "
             "Failed and missing runs are reported separately and never counted as nonrejections.", "",
             "| Learner | Test | Rejection rate [95% interval] | Valid | Failed | Missing |",
             "|---|---|---:|---:|---:|---:|"]
    for row in null.itertuples():
        lines.append(f"| {LEARNER_LABELS[row.learner]} | {METHOD_LABELS[row.method]} | {_interval_text(row)} "
                     f"| {row.n_valid} | {row.n_failed} | {row.n_missing} |")
    lines += [""]
    if not null.family_complete.all():
        lines += ["The null results are incomplete; the table records the available replications.", ""]
    lines += ["## Power and paired comparisons", "",
              "The numeric amplitude grid is 0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.5, 4, 6, 20. "
              "Pointwise 95% exact binomial intervals are available in "
              "`rejection_summary.csv` and the null table above.", "",
              "`rejection_summary.csv` contains every planned cell, including missing cells. "
              "`paired_differences.csv` reports NORD minus single-split rejection rates using only matched successful "
              "replications. Its pointwise normal Monte Carlo intervals use the sample variance of paired rejection "
              "differences and are not adjusted over amplitudes or learners. "
              "`combined_raw_results.csv.gz` and `combined_dgp_diagnostics.csv.gz` retain test-level results and "
              "principal-stratum, treatment, and first-stage diagnostics; their checksums are in `report_manifest.json`.", ""]
    if figure_output is not None:
        image = Path(os.path.relpath(figure_output / "simulation_late_hierarchical.png", output))
        lines += [f"![Conditional LATE rejection curves]({image.as_posix()})", ""]
    else:
        lines += ["Figures were not generated (`--skip-plots`).", ""]
    lines += ["This fixed-first-stage experiment evaluates empirical calibration and power. It does not address "
              "weak instruments, verify sufficient nuisance assumptions for every learner, or establish minimax optimality.", ""]
    (output / "late_report.md").write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure-output", type=Path,
                        help="optional figure directory (default: --output)")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    raw, diagnostic, config, provenance = load_inputs(args.inputs)
    summary = summarize(raw)
    null = screen_size(summary)
    summary = summary.merge(null[["learner", "method", "size_screen", "flag_direction"]],
                            on=["learner", "method"], how="left", validate="many_to_one")
    args.output.mkdir(parents=True, exist_ok=True)
    archives = {}
    for name, frame in (("combined_raw_results", raw), ("combined_dgp_diagnostics", diagnostic)):
        path = args.output / f"{name}.csv.gz"
        atomic_write_gzip_csv(path, frame)
        archives[name] = {"path": path.name, "sha256": file_hash(path),
                          "bytes": path.stat().st_size, "compression": "gzip; mtime=0"}
    atomic_write_csv(args.output / "rejection_summary.csv", summary)
    atomic_write_csv(args.output / "null_size.csv", null)
    atomic_write_csv(args.output / "paired_differences.csv", paired_differences(raw))
    atomic_write_csv(args.output / "failed_results.csv", raw[raw.status.eq("failed")])
    atomic_write_json(args.output / "report_manifest.json", {
        "complete": bool(summary.complete.all()), "null_family_complete": bool(null.family_complete.all()),
        "expected_cells": len(summary), "completed_cells": int(summary.complete.sum()),
        "expected_test_results": len(summary)*REPLICATIONS, "valid_test_results": int(summary.n_valid.sum()),
        "failed_test_results": int(summary.n_failed.sum()), "missing_test_results": int(summary.n_missing.sum()),
        "size_family": "late_13_cells", "size_flags": int(null.size_screen.eq("flagged").sum()),
        "oracle_counted_once": True, "all_planned_curves_retained": True, **archives,
        "interval_method": "pointwise 95% Clopper-Pearson", "inputs": provenance,
        "configuration": config, "report_script_sha256": file_hash(Path(__file__)),
        "shared_report_script_sha256": file_hash(Path(shared.__file__)),
        "reference_config_sha256": file_hash(ROOT / "simulation/configs/cate.json")})
    figure_output = None if args.skip_plots else args.figure_output or args.output
    if figure_output is not None:
        figure_output.mkdir(parents=True, exist_ok=True)
        plot_power(summary, null, figure_output)
    write_report(summary, null, diagnostic, args.output, figure_output=figure_output)
    print(f"{'Complete' if summary.complete.all() else 'INCOMPLETE'} LATE report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
