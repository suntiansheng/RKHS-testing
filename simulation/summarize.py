#!/usr/bin/env python3
"""Summarize rejection rates, null size, and nuisance diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, norm

from io_utils import atomic_write_csv, atomic_write_json, file_hash

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


METHOD_ORDER = [
    "orthogonal_nosplit",
    "orthogonal_single_split",
    "orthogonal_crossfit",
    "residual",
    "yan",
    "oracle_score",
]
METHOD_LABELS = {
    "orthogonal_nosplit": "Orthogonal",
    "orthogonal_single_split": "Single split",
    "orthogonal_crossfit": "Cross-fit",
    "residual": "Residual",
    "yan": "Yan",
    "oracle_score": "Oracle score",
}
METHOD_COLORS = {
    "orthogonal_nosplit": "#1B9E77",
    "orthogonal_single_split": "#E7298A",
    "orthogonal_crossfit": "#7570B3",
    "residual": "#D95F02",
    "yan": "#1F78B4",
    "oracle_score": "#4D4D4D",
}
DESIGN_LABELS = {
    "pooled_random_shift": "pooled i.i.d. random-count covariate shift",
}
MAIN_HOLM_FAMILY = "main_170_cells"
MAIN_METHODS = {"orthogonal_nosplit", "orthogonal_crossfit", "residual", "yan"}
SINGLE_SPLIT_METHOD = "orthogonal_single_split"
SINGLE_SPLIT_HOLM_FAMILY = "single_split_40_cells"
PRODUCTION_LEARNERS = {"gam", "rf", "lasso", "dnn"}
PRODUCTION_SCENARIOS = {
    "additive",
    "product",
    "composition",
    "friedman",
    "neural_composition",
}


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials == 0:
        return math.nan, math.nan
    z = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return center - half, center + half


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["design", "scenario", "p", "delta", "learner", "method", "alpha"]
    records: list[dict[str, Any]] = []
    for values, group in raw.groupby(keys, dropna=False, sort=True):
        record = dict(zip(keys, values))
        valid = group[group["status"] == "ok"]
        rejections = valid["reject"].dropna().astype(int)
        n_valid = int(len(rejections))
        rejection_count = int(rejections.sum()) if n_valid else 0
        estimate = rejection_count / n_valid if n_valid else math.nan
        lower, upper = wilson_interval(rejection_count, n_valid)
        record.update(
            {
                "rejection_rate": estimate,
                "monte_carlo_se": (
                    math.sqrt(estimate * (1.0 - estimate) / n_valid)
                    if n_valid
                    else math.nan
                ),
                "wilson_lower": lower,
                "wilson_upper": upper,
                "n_valid": n_valid,
                "n_failed": int((group["status"] != "ok").sum()),
                "mean_runtime_seconds": float(valid["runtime_seconds"].mean()),
                "mean_mu_mse": float(valid["mu_mse"].mean()),
                "mean_propensity_mse": float(valid["propensity_mse"].mean()),
                "mean_propensity_brier": float(valid["propensity_brier"].mean()),
                "mean_propensity_log_loss": float(valid["propensity_log_loss"].mean()),
                "mean_propensity_auc": float(valid["propensity_auc"].mean()),
                "mean_n1": float(valid["n1"].mean()),
                "sd_n1": float(valid["n1"].std(ddof=1)),
                "min_n1": int(valid["n1"].min()) if n_valid else 0,
                "max_n1": int(valid["n1"].max()) if n_valid else 0,
                "mean_n0": float(valid["n0"].mean()),
                "sd_n0": float(valid["n0"].std(ddof=1)),
                "mean_score_n": (
                    float(valid["score_n"].mean())
                    if "score_n" in valid.columns
                    else math.nan
                ),
                "mean_realized_group_fraction": float(
                    valid["realized_group_fraction"].mean()
                ),
                "mean_train_mse": float(valid["train_mse"].mean()),
                "mean_validation_mse": float(valid["validation_mse"].mean()),
                "mean_selected_epoch": float(valid["selected_epoch"].mean()),
                "mean_parameter_count": float(valid["parameter_count"].mean()),
                "fallback_count": int(
                    group["fallback_used"].fillna(False).astype(bool).sum()
                ),
            }
        )
        records.append(record)
    summary = pd.DataFrame(records)
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    summary["method_order"] = summary["method"].map(method_rank).fillna(999)
    summary = summary.sort_values(
        ["design", "scenario", "p", "learner", "method_order", "delta"]
    ).drop(columns="method_order")
    return summary


def size_audit(
    summary: pd.DataFrame,
    *,
    family: str = "current_summary_null_cells",
) -> pd.DataFrame:
    null = summary[np.isclose(summary["delta"], 0.0)].copy()
    added_columns = [
        "rejection_count",
        "binomial_95_lower_count",
        "binomial_95_upper_count",
        "within_binomial_95_band",
        "exact_binomial_p_value",
        "bh_q_value_global",
        "holm_p_value_global",
        "holm_flag_0_05",
        "holm_family",
    ]
    if null.empty:
        return pd.DataFrame(columns=list(summary.columns) + added_columns)
    records: list[dict[str, Any]] = []
    for row in null.to_dict("records"):
        n = int(row["n_valid"])
        rate = float(row["rejection_rate"])
        alpha = float(row["alpha"])
        count = int(round(rate * n)) if n else 0
        lower_count = int(binom.ppf(0.025, n, alpha)) if n else 0
        upper_count = int(binom.ppf(0.975, n, alpha)) if n else 0
        row.update(
            {
                "rejection_count": count,
                "binomial_95_lower_count": lower_count,
                "binomial_95_upper_count": upper_count,
                "within_binomial_95_band": (
                    bool(lower_count <= count <= upper_count) if n else False
                ),
                "exact_binomial_p_value": (
                    float(binomtest(count, n, alpha).pvalue) if n else math.nan
                ),
            }
        )
        records.append(row)
    audit = pd.DataFrame(records)
    p_values = audit["exact_binomial_p_value"].to_numpy(dtype=float)
    finite = np.isfinite(p_values)
    bh = np.full(len(audit), np.nan)
    holm = np.full(len(audit), np.nan)
    if finite.any():
        finite_p = p_values[finite]
        m = len(finite_p)
        order = np.argsort(finite_p)
        ordered = finite_p[order]
        bh_ordered = np.minimum.accumulate((ordered * m / np.arange(1, m + 1))[::-1])[
            ::-1
        ]
        holm_ordered = np.maximum.accumulate(ordered * (m - np.arange(m)))
        bh_values = np.empty(m)
        holm_values = np.empty(m)
        bh_values[order] = np.minimum(bh_ordered, 1.0)
        holm_values[order] = np.minimum(holm_ordered, 1.0)
        bh[finite] = bh_values
        holm[finite] = holm_values
    audit["bh_q_value_global"] = bh
    audit["holm_p_value_global"] = holm
    audit["holm_flag_0_05"] = audit["holm_p_value_global"] < 0.05
    audit["holm_family"] = family
    return audit


def size_family(summary: pd.DataFrame, *, single_split: bool) -> tuple[str, bool]:
    """Require the complete null grid before applying a production size gate."""
    null = summary[np.isclose(summary["delta"], 0.0)]
    pairs = (
        {(learner, SINGLE_SPLIT_METHOD) for learner in PRODUCTION_LEARNERS}
        if single_split
        else {
            (learner, method)
            for learner in PRODUCTION_LEARNERS
            for method in MAIN_METHODS
        }
        | {("oracle", "oracle_score")}
    )
    expected = {
        ("pooled_random_shift", scenario, p, learner, method)
        for scenario in PRODUCTION_SCENARIOS
        for p in (20, 50)
        for learner, method in pairs
    }
    keys = ["design", "scenario", "p", "learner", "method"]
    complete = (
        len(null) == len(expected)
        and set(null[keys].itertuples(index=False, name=None)) == expected
        and (null["n_valid"] == 300).all()
        and (null["n_failed"] == 0).all()
        and np.isclose(null["alpha"], 0.05).all()
    )
    if complete:
        return (SINGLE_SPLIT_HOLM_FAMILY if single_split else MAIN_HOLM_FAMILY), True
    label = "single_split" if single_split else "main"
    return f"nonproduction_{label}_{len(null)}_null_cells", False


def assess_size_families(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Keep the 170 main and 40 single-split null hypotheses in separate families."""
    audits = []
    complete = {}
    split_rows = summary["method"] == SINGLE_SPLIT_METHOD
    for single_split in (False, True):
        selected = summary[split_rows == single_split]
        if selected.empty:
            continue
        family, is_complete = size_family(selected, single_split=single_split)
        complete[family] = is_complete
        audits.append(size_audit(selected, family=family))
    return pd.concat(audits, ignore_index=True), complete


def attach_size_gate(summary: pd.DataFrame, sizes: pd.DataFrame) -> pd.DataFrame:
    """Attach the null-size disposition to every point on a power curve."""

    keys = ["design", "scenario", "p", "learner", "method"]
    gate = sizes[
        keys
        + [
            "within_binomial_95_band",
            "exact_binomial_p_value",
            "holm_p_value_global",
            "holm_flag_0_05",
            "holm_family",
        ]
    ].copy()
    gate = gate.rename(
        columns={
            "within_binomial_95_band": "size_within_raw_95_band",
            "exact_binomial_p_value": "size_exact_binomial_p_value",
            "holm_p_value_global": "size_holm_p_value_global",
            "holm_flag_0_05": "size_holm_flag_0_05",
            "holm_family": "size_holm_family",
        }
    )
    if gate.duplicated(keys).any():
        raise RuntimeError("Null size audit contains duplicate curve keys.")
    annotated = summary.merge(gate, on=keys, how="left", validate="many_to_one")
    annotated["size_gate_status"] = [
        "not_evaluated" if pd.isna(value) else ("fail_holm" if bool(value) else "pass")
        for value in annotated["size_holm_flag_0_05"]
    ]
    return annotated


def failure_summary(raw: pd.DataFrame) -> pd.DataFrame:
    failed = raw[raw["status"] != "ok"].copy()
    if failed.empty:
        return pd.DataFrame(columns=["learner", "method", "error", "failure_count"])
    return (
        failed.groupby(["learner", "method", "error"], dropna=False)
        .size()
        .rename("failure_count")
        .reset_index()
        .sort_values("failure_count", ascending=False)
    )


def propensity_diagnostic_tables(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract learner-specific propensity quality and calibration summaries."""

    methods = {
        "orthogonal_nosplit",
        "orthogonal_single_split",
        "orthogonal_crossfit",
        "oracle_score",
    }
    selected = raw[(raw["status"] == "ok") & raw["method"].isin(methods)].copy()
    identity = [
        "task_key",
        "design",
        "scenario",
        "p",
        "delta",
        "rep",
        "learner",
        "method",
        "n",
        "score_n",
        "n1",
        "n0",
        "realized_group_fraction",
    ]
    metrics = [
        "propensity_mse",
        "propensity_brier",
        "propensity_log_loss",
        "propensity_auc",
    ]
    quality = selected[identity + metrics].copy()
    quality["fit_regime"] = quality["method"].map(
        {
            "orthogonal_nosplit": "full_sample",
            "orthogonal_single_split": "single_split_testing",
            "orthogonal_crossfit": "out_of_fold",
            "oracle_score": "oracle_truth",
        }
    )

    bin_records: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        diagnostics = json.loads(row["diagnostics_json"])
        bins = diagnostics.get("propensity_calibration_bins")
        if not isinstance(bins, list) or not bins:
            raise RuntimeError(
                f"Missing propensity calibration bins for {row['task_key']} / {row['method']}"
            )
        base = {key: row[key] for key in identity}
        base["fit_regime"] = {
            "orthogonal_nosplit": "full_sample",
            "orthogonal_single_split": "single_split_testing",
            "orthogonal_crossfit": "out_of_fold",
            "oracle_score": "oracle_truth",
        }[row["method"]]
        for calibration_bin in bins:
            bin_payload = dict(calibration_bin)
            bin_payload["bin_n"] = int(bin_payload.pop("n"))
            bin_records.append({**base, **bin_payload})
    by_replication = pd.DataFrame(bin_records)
    if by_replication.empty:
        summary = pd.DataFrame()
    else:
        by_replication["prediction_total"] = (
            by_replication["bin_n"] * by_replication["mean_prediction"]
        )
        by_replication["observed_total"] = (
            by_replication["bin_n"] * by_replication["observed_rate"]
        )
        group_keys = [
            "design",
            "scenario",
            "p",
            "delta",
            "learner",
            "method",
            "fit_regime",
            "bin",
        ]
        summary = (
            by_replication.groupby(group_keys, dropna=False)
            .agg(
                replication_count=("rep", "nunique"),
                observation_count=("bin_n", "sum"),
                prediction_total=("prediction_total", "sum"),
                observed_total=("observed_total", "sum"),
                minimum_prediction=("minimum_prediction", "min"),
                maximum_prediction=("maximum_prediction", "max"),
            )
            .reset_index()
        )
        summary["mean_prediction"] = (
            summary["prediction_total"] / summary["observation_count"]
        )
        summary["observed_rate"] = (
            summary["observed_total"] / summary["observation_count"]
        )
        summary = summary.drop(columns=["prediction_total", "observed_total"])
        by_replication = by_replication.drop(
            columns=["prediction_total", "observed_total"]
        )
    return quality, by_replication, summary


def _make_plot_set(
    summary: pd.DataFrame,
    output_dir: Path,
    subdirectory: str,
    *,
    include_size_failures: bool,
    max_delta: Optional[float] = None,
) -> list[str]:
    figure_dir = output_dir / "figures" / subdirectory
    figure_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    normal = summary[summary["learner"] != "oracle"]
    if max_delta is not None:
        normal = normal[normal["delta"] <= max_delta + 1e-12]
    if not include_size_failures:
        normal = normal[normal["size_gate_status"] != "fail_holm"]
    group_keys = ["design", "scenario", "p"]
    for (design, scenario, p), setting in normal.groupby(group_keys, sort=True):
        learners = sorted(setting["learner"].unique())
        if not learners:
            continue
        columns = min(2, len(learners))
        rows = int(math.ceil(len(learners) / columns))
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6.0 * columns, 4.1 * rows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for axis, learner in zip(axes.flat, learners):
            learner_frame = setting[setting["learner"] == learner]
            for method in METHOD_ORDER:
                curve = learner_frame[learner_frame["method"] == method].sort_values(
                    "delta"
                )
                if curve.empty or method == "oracle_score":
                    continue
                invalid_size = bool((curve["size_gate_status"] == "fail_holm").any())
                label = METHOD_LABELS[method]
                if invalid_size:
                    label += " (fails size)"
                x_values = curve["delta"].to_numpy(dtype=float)
                y_values = curve["rejection_rate"].to_numpy(dtype=float)
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    color=METHOD_COLORS[method],
                    linestyle="--" if invalid_size else "-",
                    alpha=0.65 if invalid_size else 1.0,
                    label=label,
                )
                axis.fill_between(
                    x_values,
                    curve["wilson_lower"].to_numpy(dtype=float),
                    curve["wilson_upper"].to_numpy(dtype=float),
                    color=METHOD_COLORS[method],
                    alpha=0.06 if invalid_size else 0.10,
                    linewidth=0,
                )
            alpha = float(learner_frame["alpha"].iloc[0])
            axis.axhline(alpha, color="#777777", linestyle="--", linewidth=1)
            axis.set_title(learner.upper())
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.2)
            axis.set_xlabel(r"Signal $\Delta$")
            axis.set_ylabel("Rejection rate")
        for axis in axes.flat[len(learners) :]:
            axis.set_visible(False)
        handles_by_label: dict[str, Any] = {}
        for axis in axes.flat[: len(learners)]:
            panel_handles, panel_labels = axis.get_legend_handles_labels()
            for handle, label in zip(panel_handles, panel_labels):
                handles_by_label.setdefault(label, handle)
        labels = list(handles_by_label)
        handles = [handles_by_label[label] for label in labels]
        figure.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
        suffix = (
            " — diagnostic; dashed curves fail size" if include_size_failures else ""
        )
        design_label = DESIGN_LABELS.get(str(design), str(design))
        figure.suptitle(
            f"{str(scenario).replace('_', ' ').title()} structure; "
            f"{design_label}; p={int(p)}{suffix}"
        )
        figure.tight_layout(rect=(0, 0.08, 1, 0.95))
        stem = f"{design}_{scenario}_p{int(p)}"
        png_path = figure_dir / f"{stem}.png"
        pdf_path = figure_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=180)
        figure.savefig(pdf_path)
        plt.close(figure)
        files.extend(
            [
                str(png_path.relative_to(output_dir)),
                str(pdf_path.relative_to(output_dir)),
            ]
        )
    return files


def make_plots(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    files = []
    limits = [(None, "")]
    for maximum, suffix in ((1.6, "1_6"), (6.0, "6")):
        if float(summary["delta"].max()) > maximum + 1e-12:
            limits.append((maximum, f"_zoom_delta_le_{suffix}"))
    for maximum, suffix in limits:
        for diagnostic, prefix in ((False, "primary"), (True, "diagnostic")):
            directory = prefix + (
                suffix or ("_all_methods" if diagnostic else "_size_valid")
            )
            files.extend(
                _make_plot_set(
                    summary,
                    output_dir,
                    directory,
                    include_size_failures=diagnostic,
                    max_delta=maximum,
                )
            )
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    input_files: list[Path] = []
    frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for input_dir in arguments.inputs:
        raw_path = input_dir / "raw_results.csv"
        diagnostic_path = input_dir / "dgp_diagnostics.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        input_files.append(raw_path)
        frames.append(pd.read_csv(raw_path))
        if diagnostic_path.exists():
            diagnostic_frames.append(pd.read_csv(diagnostic_path))
    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        raise ValueError("No simulation results were supplied.")
    unknown_methods = set(raw["method"].dropna()) - set(METHOD_ORDER)
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    duplicate_columns = ["task_key", "method"]
    duplicates = raw.duplicated(duplicate_columns, keep=False)
    if duplicates.any():
        example = raw.loc[duplicates, duplicate_columns].head().to_dict("records")
        raise RuntimeError(f"Duplicate task/method rows across inputs: {example}")
    output_dir = arguments.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_results(raw)
    sizes, complete_families = assess_size_families(summary)
    summary = attach_size_gate(summary, sizes)
    complete = (
        summary["size_holm_family"].map(complete_families).fillna(False).astype(bool)
    )
    summary.loc[~complete, "size_gate_status"] = "not_evaluated"
    failures = failure_summary(raw)
    propensity_quality, propensity_bins, propensity_calibration = (
        propensity_diagnostic_tables(raw)
    )
    atomic_write_csv(output_dir / "combined_raw_results.csv", raw)
    atomic_write_csv(output_dir / "rejection_summary.csv", summary)
    atomic_write_csv(
        output_dir / "primary_rejection_summary.csv",
        summary[summary["size_gate_status"] != "fail_holm"],
    )
    curve_keys = ["design", "scenario", "p", "learner", "method"]
    curve_maximum = summary.groupby(curve_keys, dropna=False)["delta"].transform("max")
    maximum_signal = summary[np.isclose(summary["delta"], curve_maximum)].copy()
    atomic_write_csv(output_dir / "maximum_signal_summary.csv", maximum_signal)
    atomic_write_csv(
        output_dir / "primary_maximum_signal_summary.csv",
        maximum_signal[maximum_signal["size_gate_status"] != "fail_holm"],
    )
    atomic_write_csv(output_dir / "size_audit.csv", sizes)
    size_invalid = sizes[
        sizes["holm_flag_0_05"]
        & sizes["holm_family"].map(complete_families).fillna(False).astype(bool)
    ]
    atomic_write_csv(output_dir / "size_invalid_cells.csv", size_invalid)
    atomic_write_csv(output_dir / "failure_summary.csv", failures)
    atomic_write_csv(
        output_dir / "propensity_quality_by_replication.csv", propensity_quality
    )
    atomic_write_csv(
        output_dir / "propensity_calibration_by_replication.csv", propensity_bins
    )
    atomic_write_csv(
        output_dir / "propensity_calibration_summary.csv", propensity_calibration
    )
    nuisance = summary[
        (summary["method"] == "orthogonal_nosplit") & (summary["learner"] != "oracle")
    ].copy()
    nuisance["mu_mse_rank_within_cell"] = nuisance.groupby(
        ["design", "scenario", "p", "delta"], dropna=False
    )["mean_mu_mse"].rank(method="min", ascending=True)
    atomic_write_csv(output_dir / "nuisance_nosplit_summary.csv", nuisance)

    if diagnostic_frames:
        dgp = pd.concat(diagnostic_frames, ignore_index=True)
        shared_draw_keys = ["design", "p", "rep"]
        shared_draw_columns = [
            "n",
            "n1",
            "n0",
            "target_pi",
            "realized_group_fraction",
            "minimum_group_size",
            "e_min",
            "e_max",
            "e_mean",
            "pooled_mean_x1",
            "pooled_variance_x2",
            "mean_x1_difference",
            "variance_x2_difference",
            "covariance_x1_x2_difference",
        ]
        if not set(shared_draw_keys + shared_draw_columns).issubset(dgp.columns):
            raise RuntimeError("DGP diagnostics lack pooled-design columns.")
        inconsistent_draws = 0
        for _, group in dgp.groupby(shared_draw_keys, sort=False, dropna=False):
            inconsistent_draws += int(
                any(
                    group[column].nunique(dropna=False) != 1
                    for column in shared_draw_columns
                )
            )
        if inconsistent_draws:
            raise RuntimeError(
                "Pooled covariates/labels disagree across input runs for "
                f"{inconsistent_draws} shared replication(s)."
            )
        dgp_unique = dgp.drop_duplicates(
            ["design", "scenario", "p", "delta", "rep"]
        ).sort_values(["design", "scenario", "p", "delta", "rep"])
        atomic_write_csv(output_dir / "dgp_diagnostics_unique.csv", dgp_unique)
        diagnostic_summary = (
            dgp_unique.groupby(["design", "scenario", "p", "delta"])
            .mean(numeric_only=True)
            .reset_index()
        )
        atomic_write_csv(output_dir / "dgp_diagnostics_summary.csv", diagnostic_summary)
        random_count_draws = dgp.drop_duplicates(["design", "p", "rep"])
        random_count_summary = (
            random_count_draws.groupby(["design", "p"], dropna=False)
            .agg(
                monte_carlo_draws=("rep", "size"),
                total_n=("n", "first"),
                target_pi=("target_pi", "first"),
                mean_n1=("n1", "mean"),
                sd_n1=("n1", "std"),
                min_n1=("n1", "min"),
                max_n1=("n1", "max"),
                mean_n0=("n0", "mean"),
                mean_realized_group_fraction=("realized_group_fraction", "mean"),
            )
            .reset_index()
        )
        random_count_summary["binomial_expected_n1"] = (
            random_count_summary["total_n"] * random_count_summary["target_pi"]
        )
        random_count_summary["binomial_sd_n1"] = np.sqrt(
            random_count_summary["total_n"]
            * random_count_summary["target_pi"]
            * (1.0 - random_count_summary["target_pi"])
        )
        atomic_write_csv(output_dir / "random_count_summary.csv", random_count_summary)

    figures = make_plots(summary, output_dir)
    table_files = [
        "rejection_summary.csv",
        "primary_rejection_summary.csv",
        "maximum_signal_summary.csv",
        "primary_maximum_signal_summary.csv",
        "size_audit.csv",
        "size_invalid_cells.csv",
        "failure_summary.csv",
        "propensity_quality_by_replication.csv",
        "propensity_calibration_by_replication.csv",
        "propensity_calibration_summary.csv",
        "nuisance_nosplit_summary.csv",
    ]
    if diagnostic_frames:
        table_files.extend(
            [
                "dgp_diagnostics_unique.csv",
                "dgp_diagnostics_summary.csv",
                "random_count_summary.csv",
            ]
        )
    manifest = {
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_hash(path)}
            for path in input_files
        ],
        "combined_rows": int(len(raw)),
        "failed_rows": int((raw["status"] != "ok").sum()),
        "fallback_rows": int(raw["fallback_used"].fillna(False).astype(bool).sum()),
        "size_failed_cells_holm": int(len(size_invalid)),
        "diagnostic_holm_flags_in_available_null_cells": int(
            sizes["holm_flag_0_05"].sum()
        ),
        "complete_size_families": complete_families,
        "size_gate_null_cells": int(len(sizes)),
        "size_gate_methods": sorted(raw["method"].unique()),
        "primary_summary_excludes_size_failed_curves": True,
        "tables": table_files,
        "figures": figures,
        "learner_averaging_used": False,
    }
    atomic_write_json(output_dir / "summary_manifest.json", manifest)
    out_of_band = sizes[~sizes["within_binomial_95_band"]]
    holm_flags = sizes[sizes["holm_flag_0_05"]]
    print(
        f"Combined {len(raw)} rows; failures={manifest['failed_rows']}; "
        f"fallbacks={manifest['fallback_rows']}; null cells outside exact 95% band="
        f"{len(out_of_band)}; available-null Holm-significant cells={len(holm_flags)}; "
        f"complete size families={sum(complete_families.values())}/{len(complete_families)}"
    )
    if not out_of_band.empty:
        print(
            out_of_band[
                [
                    "design",
                    "scenario",
                    "p",
                    "learner",
                    "method",
                    "rejection_rate",
                    "exact_binomial_p_value",
                    "holm_p_value_global",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
