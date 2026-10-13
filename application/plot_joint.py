#!/usr/bin/env python3
"""Four matched regression-test figures in the manuscript's existing style."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

if __package__:
    from . import plot as existing
else:
    import plot as existing


FIGURES = {
    ("baseline", "mean_satisfaction"): "casino_baseline_cate",
    ("baseline", "both_extremely_satisfied"): "casino_baseline_max_satisfaction",
    ("baseline_text", "mean_satisfaction"): "casino_results_v9",
    ("baseline_text", "both_extremely_satisfied"): "casino_results_max_satisfaction",
}
METHODS = [
    ("no_split", "NORD", "#0072B2", "o", -0.11, "-"),
    ("crossfit_5fold", "CF-NORD ($K=5$)", "#CC79A7", "D", 0.11, "--"),
]
LEARNERS = ["ridge", "lasso", "random_forest"]
FLOOR = 1 / 5001
TICKS = [FLOOR, 0.001, 0.01, 0.05, 1]
TICK_LABELS = ["1/5001", ".001", ".01", ".05", "1"]


def pvalue_axis(ax, *, horizontal=False):
    if horizontal:
        ax.set_xscale("log")
        ax.set_xlim(FLOOR / 1.5, 1.1)
        ax.set_xticks(TICKS, TICK_LABELS)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.get_xticklabels()[0].set_horizontalalignment("right")
        ax.axvline(0.05, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Multiplier p-value")
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    else:
        ax.set_yscale("log")
        ax.set_ylim(FLOOR / 1.5, 1.1)
        ax.set_yticks(TICKS, TICK_LABELS)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.axhline(0.05, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Multiplier p-value")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)


def make_figure(tests):
    primary = tests[tests.selection.eq("main")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8),
        gridspec_kw={"width_ratios": [1.05, 0.85, 1.55]})
    ax = axes[0]
    main = primary[primary.bandwidth.eq(1)]
    for regime, label, color, marker, offset, style in METHODS:
        rows = main[main.regime.eq(regime)].set_index("learner").loc[LEARNERS]
        ax.scatter(rows.p_value, np.arange(3) + offset, edgecolors=color,
            facecolors=color if regime == "no_split" else "none", marker=marker,
            s=26, linewidths=0.8, zorder=3)
    ax.set_yticks(range(3), ["Ridge", "Lasso", "Forest"])
    ax.set_ylim(2.5, -0.5)
    ax.set_title("(a) Nuisance learners", loc="left", fontsize=9)
    pvalue_axis(ax, horizontal=True)

    ax = axes[1]
    for regime, label, color, marker, offset, style in METHODS:
        rows = primary[primary.learner.eq("ridge") & primary.regime.eq(regime)].sort_values("bandwidth")
        np.testing.assert_array_equal(rows.bandwidth, [0.5, 1., 2.])
        ax.plot(rows.bandwidth, rows.p_value, color=color, marker=marker, markersize=4,
            markerfacecolor=color if regime == "no_split" else "none",
            linestyle=style, linewidth=1.1, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.43, 2.3)
    ax.set_xticks([0.5, 1, 2], ["0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xlabel("Bandwidth $h$")
    ax.set_title("(b) Kernel bandwidth", loc="left", fontsize=9)
    pvalue_axis(ax)

    ax = axes[2]
    names = ["main"] + [f"alternative_{i:02d}" for i in range(1, 11)]
    selection = tests[tests.learner.eq("ridge") & tests.bandwidth.eq(1) & tests.selection.isin(names)]
    for regime, label, color, marker, offset, style in METHODS:
        rows = selection[selection.regime.eq(regime)].set_index("selection").loc[names]
        x = np.arange(len(names)) + offset
        ax.plot(x, rows.p_value, color=color, marker=marker, markersize=3.8,
            markerfacecolor=color if regime == "no_split" else "none",
            linestyle=style, linewidth=0.85, zorder=3)
    ax.set_xticks(range(11), ["Main"] + [str(i) for i in range(1, 11)])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-0.5, 10.5)
    ax.set_xlabel("Selection ordering")
    ax.set_title("(c) Participant selection", loc="left", fontsize=9)
    pvalue_axis(ax)
    handles = [Line2D([], [], color=color, marker=marker, linestyle=style,
        markerfacecolor=color if regime == "no_split" else "none",
        linewidth=1.1, markersize=4, label=label) for regime, label, color, marker, _, style in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.52, -0.015))
    fig.subplots_adjust(left=0.072, right=0.994, bottom=0.28, top=0.88, wspace=0.65)
    return fig


def create_figures(results_dir, output_dir):
    summary = json.loads((results_dir / "analysis_summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("The matched analysis must be complete before plotting")
    tests = pd.read_csv(results_dir / "regression_tests.csv")
    if not np.isfinite(tests.p_value).all() or not tests.p_value.between(FLOOR - 1e-12, 1).all():
        raise ValueError("Invalid multiplier p-values")
    if not tests.bootstrap_draws.eq(5000).all():
        raise ValueError("The four figures require the same multiplier budget")
    existing.configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for (covariates, outcome), stem in FIGURES.items():
        selected = tests[tests.covariates.eq(covariates) & tests.outcome.eq(outcome)]
        fig = make_figure(selected)
        path = output_dir / f"{stem}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04, dpi=220)
        plt.close(fig)
        outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs/casino-joint")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in create_figures(args.results_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
