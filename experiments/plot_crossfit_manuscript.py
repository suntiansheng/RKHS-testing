"""Reproduce the two n=120,160 cross-fit figures used in the supplement.

Read the completed extension, retain every learner and amplitude, and use its
archived numerical validation. No simulation or nuisance fit is rerun. The
figure inputs are included in the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[1]
SIZES = (120, 160)
LEARNERS = (("gam", "Spline"), ("rf", "Random forest"),
            ("lasso", "Lasso"), ("dnn", "DNN"))
BLUE, PINK, GRAY = "#0072B2", "#CC79A7", "#333333"
STEMS = ("simulation_crossfit_additive_power_ratio",
         "simulation_crossfit_additive_null_size")
RATIO_TABLE = "crossfit_rejection_rate_ratios.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_results(run):
    config = json.loads((run / "config.json").read_text())["config"]
    validation = json.loads((run / "validation.json").read_text())
    require(config["sample_sizes"] == list(SIZES), "Unexpected sample sizes")
    require(config["null_replications"] == 2000 and config["power_replications"] == 1000,
            "Unexpected replication counts")
    require(config["bootstrap_draws"] == 999 and not config["five_fold_sensitivity"],
            "Unexpected calibration or folds")
    require(validation["complete"] and validation["null_family_cells"] == 34,
            "Incomplete extension or unexpected calibration family")
    summary = pd.read_csv(run / "summary.csv")
    differences = pd.read_csv(run / "paired_differences.csv")
    null = pd.read_csv(run / "null_size.csv")
    require(len(summary) == 306 and len(differences) == 144 and len(null) == 34,
            "Incomplete figure inputs")
    for frame in (summary, differences, null):
        require(set(frame.n) == set(SIZES), "Unexpected rows in figure inputs")
    require(differences.folds.eq(2).all(), "Non-two-fold comparison")
    keys = ["n", "learner", "propensity_mode", "amplitude"]
    require(not differences.duplicated(keys).any(), "Duplicate paired cells")
    rates = summary[summary.learner.ne("oracle")].pivot(
        index=keys, columns="method", values="rejection_rate")
    paired = differences.set_index(keys).loc[rates.index]
    np.testing.assert_allclose(paired.full_power, rates.full, atol=1e-14)
    np.testing.assert_allclose(paired.crossfit_power, rates.cf2, atol=1e-14)
    r = differences.replications.to_numpy()
    np.testing.assert_array_equal(r, np.where(differences.amplitude.eq(0), 2000, 1000))
    b = differences.full_only_rejections.to_numpy()
    c = differences.crossfit_only_rejections.to_numpy()
    mean = (b - c) / r
    # d_r is -1, 0, or 1; sum(d_r^2) equals the number of discordant pairs.
    se = np.sqrt((b + c - r * mean**2) / (r * (r - 1)))
    np.testing.assert_allclose(mean, differences.full_minus_crossfit, atol=1e-14)
    np.testing.assert_allclose(se, differences.paired_mc_se, atol=1e-14)
    np.testing.assert_allclose(mean - norm.ppf(.975) * se, differences.ci_lower, atol=1e-14)
    np.testing.assert_allclose(mean + norm.ppf(.975) * se, differences.ci_upper, atol=1e-14)
    # Validate the original screen; do not select panels according to power.
    pvalues = np.array([binomtest(int(row.rejections), int(row.replications), .05).pvalue
                       for row in null.itertuples()])
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(
        pvalues[order] * (len(pvalues) - np.arange(len(pvalues)))))
    np.testing.assert_allclose(adjusted, null.holm_adjusted_p_value, atol=1e-14)
    np.testing.assert_array_equal(adjusted <= .05, null.size_screen.eq("flagged"))
    z = norm.ppf(.975)
    r = null.replications.to_numpy()
    p = null.rejection_rate.to_numpy()
    center = (p + z**2 / (2 * r)) / (1 + z**2 / r)
    half = z * np.sqrt(p * (1 - p) / r + z**2 / (4 * r**2)) / (1 + z**2 / r)
    np.testing.assert_allclose(center - half, null.wilson_lower, atol=1e-14)
    np.testing.assert_allclose(center + half, null.wilson_upper, atol=1e-14)
    return differences, null


def paired_ratios(differences):
    """Compute rejection-rate ratios and paired log-scale Monte Carlo intervals."""
    r = differences.replications.to_numpy(dtype=float)
    require(np.isfinite(r).all() and (r > 1).all() and np.equal(r, np.rint(r)).all(),
            "Replication counts must be integers greater than one")

    def counts(values, name):
        values = np.asarray(values, dtype=float)
        require(np.isfinite(values).all()
                and np.allclose(values, np.rint(values), atol=1e-8, rtol=0)
                and (values >= 0).all() and (values <= r).all(),
                f"Invalid {name} counts")
        return np.rint(values).astype(np.int64)

    full = counts(differences.full_power.to_numpy() * r, "NORD rejection")
    crossfit = counts(differences.crossfit_power.to_numpy() * r, "CF-NORD rejection")
    b = counts(differences.full_only_rejections, "NORD-only rejection")
    c = counts(differences.crossfit_only_rejections, "CF-NORD-only rejection")
    both = full - b
    neither = r.astype(np.int64) - both - b - c
    require(np.array_equal(both, crossfit - c)
            and (both >= 0).all() and (neither >= 0).all(),
            "Inconsistent paired rejection counts")
    require((full > 0).all() and (crossfit > 0).all(),
            "Log-scale ratio intervals require positive rejection counts for both methods")

    ratio = full / crossfit
    # Paired delta-method variance, using the sample covariance with ddof=1.
    se = np.sqrt((r / (r - 1)) * (b + c) / (full.astype(float) * crossfit))
    z = norm.ppf(.975)
    result = differences[["n", "learner", "propensity_mode", "amplitude", "folds",
                          "replications", "full_power", "crossfit_power",
                          "full_only_rejections", "crossfit_only_rejections"]].copy()
    result["full_rejections"] = full
    result["crossfit_rejections"] = crossfit
    result["both_rejections"] = both
    result["neither_rejections"] = neither
    result["rejection_rate_ratio"] = ratio
    result["paired_log_mc_se"] = se
    result["ci_lower"] = ratio * np.exp(-z * se)
    result["ci_upper"] = ratio * np.exp(z * se)
    return result


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "lines.linewidth": 1.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=.04)
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def power_ratio(ratios, output):
    selected = ratios[ratios.propensity_mode.eq("estimated")]
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.9), sharex=True, sharey=True)
    for row, n in enumerate(SIZES):
        for col, (learner, label) in enumerate(LEARNERS):
            ax = axes[row, col]
            data = selected[selected.n.eq(n) & selected.learner.eq(learner)].sort_values("amplitude")
            y = data.rejection_rate_ratio
            ax.errorbar(data.amplitude, y,
                        yerr=np.vstack([y - data.ci_lower, data.ci_upper - y]),
                        color=BLUE, marker="o", markersize=3.5, capsize=2,
                        elinewidth=.9, linewidth=1.4)
            ax.set_title(f"{label}, $n={n}$", pad=5)
            ax.axhline(1, color=".5", linestyle=":", linewidth=.8)
            ax.grid(axis="y", color=".9", linewidth=.5)
            ax.set_axisbelow(True)
            ax.set_xlim(-.1, 6.1)
            ax.set_ylim(.5, 1.7)
            ax.set_xticks([0, 2, 4, 6])
            ax.set_yticks([.6, .8, 1, 1.2, 1.4, 1.6])
    fig.supylabel("Rejection-rate ratio", x=.015, fontsize=9)
    fig.supxlabel("Signal amplitude $a$", y=.135, fontsize=9)
    handle = Line2D([], [], color=BLUE, marker="o", markersize=4)
    fig.legend([handle], ["NORD / CF-NORD ($K=2$); pointwise 95% paired intervals"],
               loc="lower center", bbox_to_anchor=(.51, .049), frameon=False)
    fig.subplots_adjust(left=.082, right=.992, top=.92, bottom=.245, wspace=.18, hspace=.57)
    save(fig, output, STEMS[0])


def null_size(null, output):
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.0), sharey=True)
    specs = (("full", BLUE, "o", -.1), ("cf2", PINK, "D", .1))
    for row, n in enumerate(SIZES):
        for col, mode in enumerate(("estimated", "known")):
            ax = axes[row, col]
            selected = null[null.n.eq(n) & null.propensity_mode.eq(mode)]
            for j, (learner, _) in enumerate(LEARNERS):
                for method, color, marker, offset in specs:
                    result = selected[selected.learner.eq(learner) & selected.method.eq(method)].iloc[0]
                    y = result.rejection_rate
                    ax.errorbar(j + offset, y, yerr=[[y - result.wilson_lower],
                                                   [result.wilson_upper - y]],
                                color=color, marker=marker, markersize=4, capsize=3,
                                elinewidth=.9, linestyle="none")
            oracle = null[null.n.eq(n) & null.learner.eq("oracle")].iloc[0]
            y = oracle.rejection_rate
            ax.errorbar(4, y, yerr=[[y - oracle.wilson_lower], [oracle.wilson_upper - y]],
                        color=GRAY, marker="s", markersize=4, capsize=3,
                        elinewidth=.9, linestyle="none")
            ax.set_title(f"{mode.capitalize()} propensity, $n={n}$", pad=5)
            ax.axhline(.05, color=".5", linestyle=":", linewidth=.8)
            ax.set_ylim(0, .10)
            ax.set_yticks(np.arange(0, .101, .02))
            ax.set_xticks(np.arange(5), [label for _, label in LEARNERS] + ["Oracle"],
                          rotation=16, ha="right")
            ax.grid(axis="y", color=".9", linewidth=.5)
            ax.set_axisbelow(True)
            ax.set_xlim(-.32, 4.2)
    fig.supylabel("Null rejection rate", x=.015, fontsize=9)
    handles = [Line2D([], [], color=BLUE, marker="o", markersize=4, label="NORD"),
               Line2D([], [], color=PINK, marker="D", markersize=4, linestyle="--", label="CF-NORD ($K=2$)"),
               Line2D([], [], color=GRAY, marker="s", markersize=4, linestyle="--", label="Oracle")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.53, .042))
    fig.subplots_adjust(left=.115, right=.99, bottom=.21, top=.94, wspace=.16, hspace=.69)
    save(fig, output, STEMS[1])


def build(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    differences, null = load_results(run)
    ratios = paired_ratios(differences)
    output.mkdir(parents=True, exist_ok=True)
    style()
    power_ratio(ratios, output)
    null_size(null, output)
    ratios.to_csv(output / RATIO_TABLE, index=False)
    names = ["config.json", "validation.json", "summary.csv", "paired_differences.csv", "null_size.csv"]
    provenance = {
        "sample_sizes": list(SIZES), "folds": 2,
        "run": Path(os.path.relpath(run, ROOT)).as_posix(),
        "null_family_cells": 34,
        "source_sha256": {name: sha256(run / name) for name in names},
        "builder_sha256": sha256(__file__),
        "output_sha256": {f"{stem}.pdf": sha256(output / f"{stem}.pdf") for stem in STEMS},
        "ratio_table_sha256": sha256(output / RATIO_TABLE),
        "ratio": "NORD rejection probability / CF-NORD rejection probability",
        "interval": ("Pointwise paired log-scale normal interval: ratio * exp(+/- Phi^-1(.975) * s); "
                     "s^2 = R/(R-1) * (NORD-only + CF-NORD-only) / "
                     "(NORD rejections * CF-NORD rejections)"),
        "validation": "Paired rejection counts, ratio intervals and archived null counts verified.",
    }
    (output / "crossfit_manuscript_validation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/crossfit_comparison/n120_n160_k2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures")
    args = parser.parse_args()
    provenance = build(args.run, args.output)
    print(json.dumps({"output": str(args.output), "sample_sizes": provenance["sample_sizes"],
                      "validation": provenance["validation"]}, indent=2))


if __name__ == "__main__":
    main()
