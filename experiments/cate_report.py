#!/usr/bin/env python3
"""Summarize the CATE simulation and paired rejection-rate differences."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
from io_utils import atomic_write_csv, atomic_write_json, file_hash

AMPLITUDES = (0., .1, .2, .3, .5, .8, 1.2, 1.6, 2.5, 4., 6., 20.)
LEARNERS = (("gam", "Spline"), ("rf", "Random forest"),
            ("lasso", "Lasso"), ("dnn", "DNN"))
METHODS = (
    ("orthogonal_nosplit", "NORD", "#0072B2", "o", "-"),
    ("orthogonal_single_split", "Single split", "#D55E00", "s", "-."),
    ("yan", "Yan", "#009E73", "^", ":"),
    ("oracle_score", "Oracle", "#333333", None, (0, (5, 2))),
)
METHOD_LABELS = {row[0]: row[1] for row in METHODS}
LEARNER_LABELS = dict(LEARNERS) | {"oracle": "Shared oracle"}
CELLS = tuple((learner, method[0]) for learner, _ in LEARNERS for method in METHODS[:3]) + (
    ("oracle", "oracle_score"),)
REPLICATIONS = 300
LEVEL = .05
CELL_KEYS = ["learner", "method", "delta"]


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic_write_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a reproducible compressed CSV and atomically replace its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as binary:
            with gzip.GzipFile(filename="", mode="wb", fileobj=binary,
                               compresslevel=6, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                    frame.to_csv(stream, index=False)
            binary.flush()
            os.fsync(binary.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_config(config: dict) -> None:
    """Validate the CATE design and tuning, including incomplete shards."""
    expected = {"scenarios": ["neural_composition"], "dimensions": [20],
                "n": 400, "rho": .5, "sigma": 1., "alpha": LEVEL,
                "bootstrap_draws": 200, "replications": REPLICATIONS,
                "target_pi": .5, "calibration_n": 100000,
                "propensity_mode": "estimated", "designs": ["pooled_random_shift"],
                "treatment_noise_scales": [1., 1.5]}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"CATE configuration requires {key}={value!r}")
    if tuple(float(x) for x in config.get("deltas", [])) != AMPLITUDES:
        raise ValueError("The CATE amplitude grid does not match the prespecified grid")
    if set(config.get("learners", [])) != {"gam", "rf", "lasso", "dnn", "oracle"}:
        raise ValueError("The CATE configuration must include four learners and the shared oracle")
    if set(config.get("methods", [])) != set(METHOD_LABELS):
        raise ValueError("The CATE configuration must contain the four specified tests")
    if len(config.get("designs", [])) != 1:
        raise ValueError("The CATE report accepts one design")


def validate_results(raw: pd.DataFrame, config: dict) -> None:
    required = {"task_key", "design", "scenario", "p", "n", "rho", "sigma", "delta",
                "rep", "learner", "method", "alpha", "bootstrap_draws", "dgp_seed",
                "fit_seed", "bootstrap_seed", "status", "reject", "p_value", "n0", "n1"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw-result columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("No simulation results were supplied")
    if raw[list(required - {"reject", "p_value"})].isna().any().any():
        raise ValueError("Required result identifiers or status values are missing")
    for key in (["task_key", "method"], CELL_KEYS + ["rep"]):
        if raw.duplicated(key).any():
            raise ValueError(f"Duplicate result rows for {key}")
    for key, value in {"design": config["designs"][0], "scenario": "neural_composition",
                       "p": 20, "n": 400, "rho": .5, "sigma": 1., "alpha": LEVEL,
                       "bootstrap_draws": 200}.items():
        if not raw[key].eq(value).all():
            raise ValueError(f"Result column {key} is inconsistent with the CATE design")
    if not raw.delta.isin(AMPLITUDES).all():
        raise ValueError("Unexpected signal amplitudes")
    if not raw.rep.between(1, REPLICATIONS).all() or not raw.rep.eq(raw.rep.astype(int)).all():
        raise ValueError("Replication indices must be integers from 1 to 300")
    if not set(raw[["learner", "method"]].itertuples(index=False, name=None)) <= set(CELLS):
        raise ValueError("Unexpected learner/test combination; the oracle must occur only once")
    if not raw.status.isin(["ok", "failed"]).all():
        raise ValueError("Unknown result status")
    if not raw.n0.add(raw.n1).eq(400).all():
        raise ValueError("Treatment-arm sizes do not sum to 400")
    valid = raw[raw.status.eq("ok")]
    if (not np.isfinite(valid.p_value).all() or not valid.p_value.between(0, 1).all()
            or not valid.reject.isin([0, 1]).all()
            or not valid.reject.eq(valid.p_value.le(LEVEL).astype(int)).all()):
        raise ValueError("Invalid p-values or inconsistent rejection decisions")
    failed = raw[raw.status.eq("failed")]
    if failed[["reject", "p_value"]].notna().any().any():
        raise ValueError("Failed results must not contain rejection decisions or p-values")
    for keys, columns in [(["rep"], ["dgp_seed", "n0", "n1"]),
                          (["rep", "delta"], ["bootstrap_seed"]),
                          (["rep", "delta", "learner"], ["fit_seed", "task_key"])]:
        if raw.groupby(keys)[columns].nunique().gt(1).any().any():
            raise ValueError(f"Datasets or seeds differ within matched groups {keys}")
    seeds = raw.groupby("rep").dgp_seed.first()
    if seeds.duplicated().any():
        raise ValueError("Independent replications reuse a data-generation seed")


def load_inputs(inputs: Sequence[Path]) -> tuple[pd.DataFrame, dict, list[dict]]:
    frames, provenance = [], []
    reference = None
    config = None
    for directory in inputs:
        directory = Path(directory).resolve()
        raw_path, metadata_path = directory / "raw_results.csv", directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        current_config = metadata["config"]
        validate_config(current_config)
        digest = hashlib.sha256(_canonical(current_config).encode()).hexdigest()
        if metadata.get("config_hash") != digest:
            raise ValueError(f"Configuration hash mismatch in {directory}")
        calibration = json.loads((directory / "calibrations.json").read_text())
        signature = (digest, metadata.get("scientific_core_sha256"),
                     metadata.get("runner_sha256"), _canonical(calibration))
        if not signature[1] or not signature[2]:
            raise ValueError(f"Missing simulation source hashes in {directory}")
        if reference is not None and signature != reference:
            raise ValueError("Input runs have inconsistent configurations, calibration, or source code")
        reference, config = signature, current_config
        frames.append(pd.read_csv(raw_path))
        provenance.append({"directory": os.path.relpath(directory, ROOT), "raw_sha256": file_hash(raw_path),
                           "metadata_sha256": file_hash(metadata_path),
                           "calibration_sha256": file_hash(directory / "calibrations.json")})
    if not frames:
        raise ValueError("Supply at least one run directory")
    raw = pd.concat(frames, ignore_index=True)
    validate_results(raw, config)
    return raw, config, provenance


def exact_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return math.nan, math.nan
    interval = binomtest(successes, trials).proportion_ci(confidence_level=.95, method="exact")
    return float(interval.low), float(interval.high)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    groups = {key: group for key, group in raw.groupby(CELL_KEYS)}
    rows = []
    for learner, method in CELLS:
        for amplitude in AMPLITUDES:
            group = groups.get((learner, method, amplitude), raw.iloc[:0])
            valid = group[group.status.eq("ok")]
            n_valid = len(valid)
            hits = int(valid.reject.sum())
            lower, upper = exact_interval(hits, n_valid)
            rows.append({"scenario": "neural_composition", "n": 400, "p": 20,
                         "learner": learner, "method": method, "delta": amplitude,
                         "alpha": LEVEL, "bootstrap_draws": 200, "n_expected": REPLICATIONS,
                         "n_observed": len(group), "n_valid": n_valid,
                         "n_failed": int(group.status.eq("failed").sum()),
                         "n_missing": REPLICATIONS - len(group), "rejections": hits,
                         "rejection_rate": hits / n_valid if n_valid else math.nan,
                         "ci_lower": lower, "ci_upper": upper,
                         "monte_carlo_se": math.sqrt(hits/n_valid*(1-hits/n_valid)/n_valid)
                         if n_valid else math.nan,
                         "complete": n_valid == REPLICATIONS})
    return pd.DataFrame(rows)


def screen_size(summary: pd.DataFrame) -> pd.DataFrame:
    null = summary[summary.delta.eq(0)].copy().reset_index(drop=True)
    if set(null[["learner", "method"]].itertuples(index=False, name=None)) != set(CELLS) or len(null) != 13:
        raise ValueError("The size-screen family must contain exactly 13 distinct null cells")
    complete = (null.n_valid.eq(REPLICATIONS) & null.n_failed.eq(0) & null.n_missing.eq(0)).all()
    null["family"] = "cate_13_cells"
    null["family_complete"] = bool(complete)
    null["exact_binomial_p_value"] = np.nan
    null["holm_adjusted_p_value"] = np.nan
    null["size_screen"] = "not_evaluated"
    null["flag_direction"] = ""
    if complete:
        p = np.array([binomtest(int(k), REPLICATIONS, LEVEL, alternative="two-sided").pvalue
                      for k in null.rejections])
        order = np.argsort(p, kind="stable")
        adjusted = np.minimum(1., np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
        null["exact_binomial_p_value"] = p
        null.loc[order, "holm_adjusted_p_value"] = adjusted
        flag = null.holm_adjusted_p_value.le(LEVEL)
        null["size_screen"] = np.where(flag, "flagged", "not_flagged")
        null.loc[flag, "flag_direction"] = np.where(null.loc[flag, "rejection_rate"].gt(LEVEL),
                                                    "above_nominal", "below_nominal")
    return null


def paired_differences(raw: pd.DataFrame) -> pd.DataFrame:
    valid = raw[raw.status.eq("ok") & raw.method.isin([m[0] for m in METHODS[:2]])]
    rows = []
    for learner, _ in LEARNERS:
        for amplitude in AMPLITUDES:
            group = valid[valid.learner.eq(learner) & valid.delta.eq(amplitude)]
            wide = group.pivot(index="rep", columns="method", values="reject").reindex(
                columns=[m[0] for m in METHODS[:2]]).dropna()
            difference = wide.iloc[:, 0] - wide.iloc[:, 1]
            count = len(difference)
            estimate = float(difference.mean()) if count else math.nan
            se = float(difference.std(ddof=1) / math.sqrt(count)) if count > 1 else math.nan
            half = norm.ppf(.975) * se
            rows.append({"learner": learner, "delta": amplitude, "n_pairs": count,
                         "difference_nord_minus_single_split": estimate,
                         "paired_monte_carlo_se": se,
                         "ci_lower": max(-1., estimate-half) if count > 1 else math.nan,
                         "ci_upper": min(1., estimate+half) if count > 1 else math.nan,
                         "nord_only_rejections": int(difference.eq(1).sum()),
                         "single_split_only_rejections": int(difference.eq(-1).sum())})
    return pd.DataFrame(rows)


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
        fig.savefig(output / f"simulation_cate_hierarchical.{extension}",
                    bbox_inches="tight", pad_inches=.04, dpi=220)
    plt.close(fig)


def _interval_text(row) -> str:
    if not row.n_valid:
        return "—"
    return f"{row.rejection_rate:.3f} [{row.ci_lower:.3f}, {row.ci_upper:.3f}]"


def write_report(summary: pd.DataFrame, null: pd.DataFrame, output: Path, *,
                 figure_output: Path | None) -> None:
    complete = bool(summary.complete.all())
    lines = ["# CATE simulation", "", "Status: **" + ("complete" if complete else "INCOMPLETE") + "**.", "",
             "One hierarchical-signal experiment uses n = 400, p = 20, 300 planned replications per amplitude, "
             "nominal level 0.05, and 200 centered Gaussian multiplier draws. Yan uses its normal calibration.", "",
             "The observed response is Y = b_N(X) + D a f_N(X) + (1 + 0.5 D) ε. "
             "The signal and its reference-sample normalization are unchanged from the existing hierarchical setting. "
             "The CATE is a f_N(X), and its population average is zero at every amplitude. At the null, "
             "conditional outcome variances differ between the two treatment arms.", "",
             f"Completed cells: {int(summary.complete.sum())}/{len(summary)}; "
             f"valid test results: {int(summary.n_valid.sum()):,}/{len(summary)*REPLICATIONS:,}; "
             f"failed results: {int(summary.n_failed.sum()):,}; missing results: {int(summary.n_missing.sum()):,}.", "",
             "The 156 distinct amplitude–test–learner cells comprise 12 amplitudes times 13 combinations. "
             "The oracle is computed once per dataset and repeated in the four panels; these repetitions "
             "are not additional replications or additional size tests.", "", "## Null rejection rates", "",
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
              "The figure retains numeric amplitude positions, matching the existing manuscript figures; "
              "the full grid is 0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.5, 4, 6, 20. "
              "The four learner panels show rejection-rate curves. Pointwise 95% exact binomial intervals "
              "are available in `rejection_summary.csv` and the null table above.", "",
              "`rejection_summary.csv` contains every planned cell, including missing cells. "
              "`paired_differences.csv` reports NORD minus single-split rejection rates using only matched successful "
              "replications, with pointwise normal Monte Carlo intervals based on the sample variance of paired "
              "rejection differences. These intervals describe Monte Carlo uncertainty and are not adjusted "
              "for comparisons over amplitudes or learners.", ""]
    if figure_output is not None:
        image = Path(os.path.relpath(figure_output / "simulation_cate_hierarchical.png", output))
        lines += [f"![CATE rejection curves]({image.as_posix()})", ""]
    else:
        lines += ["Figures were not generated (`--skip-plots`).", ""]
    lines += ["The experiment evaluates empirical calibration and power for this design. It does not verify "
              "the sufficient assumptions for every fitted learner or establish minimax optimality. "
              "Zero average treatment effect alone does not imply a zero propensity-weighted scalar moment.", ""]
    (output / "cate_report.md").write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure-output", type=Path,
                        help="optional figure directory (default: --output)")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    raw, config, provenance = load_inputs(args.inputs)
    summary = summarize(raw)
    null = screen_size(summary)
    summary = summary.merge(null[["learner", "method", "size_screen", "flag_direction"]],
                            on=["learner", "method"], how="left", validate="many_to_one")
    paired = paired_differences(raw)
    args.output.mkdir(parents=True, exist_ok=True)
    combined_path = args.output / "combined_raw_results.csv.gz"
    atomic_write_gzip_csv(combined_path, raw)
    atomic_write_csv(args.output / "rejection_summary.csv", summary)
    atomic_write_csv(args.output / "null_size.csv", null)
    atomic_write_csv(args.output / "paired_differences.csv", paired)
    failed = raw[raw.status.eq("failed")]
    atomic_write_csv(args.output / "failed_results.csv", failed)
    atomic_write_json(args.output / "report_manifest.json", {
        "complete": bool(summary.complete.all()), "null_family_complete": bool(null.family_complete.all()),
        "expected_cells": len(summary), "completed_cells": int(summary.complete.sum()),
        "expected_test_results": len(summary)*REPLICATIONS, "valid_test_results": int(summary.n_valid.sum()),
        "failed_test_results": int(summary.n_failed.sum()), "missing_test_results": int(summary.n_missing.sum()),
        "size_family": "cate_13_cells", "size_flags": int(null.size_screen.eq("flagged").sum()),
        "oracle_counted_once": True, "all_planned_curves_retained": True,
        "combined_raw_results": {"path": combined_path.name, "sha256": file_hash(combined_path),
                                 "bytes": combined_path.stat().st_size, "compression": "gzip; mtime=0"},
        "interval_method": "pointwise 95% Clopper-Pearson", "inputs": provenance,
        "configuration": config, "report_script_sha256": file_hash(Path(__file__))})
    if not args.skip_plots:
        figure_output = args.figure_output or args.output
        figure_output.mkdir(parents=True, exist_ok=True)
        plot_power(summary, null, figure_output)
    write_report(summary, null, args.output,
                 figure_output=(args.figure_output or args.output) if not args.skip_plots else None)
    print(f"{'Complete' if summary.complete.all() else 'INCOMPLETE'} CATE report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
