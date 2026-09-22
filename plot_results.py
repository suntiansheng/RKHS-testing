"""Render simulation figures and tables from the supplied summary data."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


RESULTS = Path(__file__).resolve().parent / "results"
SCENARIOS = ("additive", "product", "composition", "neural_composition")
LEARNERS = (
    ("gam", "Spline"),
    ("rf", "Random forest"),
    ("lasso", "Lasso"),
    ("dnn", "DNN"),
)
METHODS = (
    ("orthogonal_nosplit", "Full sample", "#0072B2", "o", "-"),
    ("orthogonal_single_split", "Single split", "#D55E00", "s", "-."),
    ("yan", "Yan", "#009E73", "^", ":"),
    ("oracle_score", "Oracle", "#333333", None, (0, (5, 2))),
)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def power_figure(data, scenario, output):
    full_grid = scenario == "neural_composition"
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.9), sharex=True, sharey=True)
    for row, p in enumerate((20, 50)):
        for col, (learner, label) in enumerate(LEARNERS):
            ax = axes[row, col]
            for method, _, color, marker, style in METHODS:
                points = data[
                    (data.scenario == scenario)
                    & (data.p == p)
                    & (data.method == method)
                ]
                if method != "oracle_score":
                    points = points[points.learner == learner]
                if not full_grid:
                    points = points[points.delta <= 6]
                points = points.sort_values("delta")
                ax.plot(
                    points.delta,
                    points.rejection_rate,
                    color=color,
                    marker=marker,
                    linestyle=style,
                    markersize=3,
                    alpha=0.94,
                )
            ax.axhline(0.05, color=".55", lw=0.6, ls=":")
            ax.set_title(label + rf", $p={p}$", pad=5)
            ax.set_ylim(-0.025, 1.035)
            ax.set_yticks((0, 0.5, 1))
            ax.set_xlim((-0.3, 20.3) if full_grid else (-0.1, 6.1))
            ax.set_xticks((0, 6, 12, 20) if full_grid else (0, 2, 4, 6))
            ax.grid(axis="y", color=".92", lw=0.5)
    fig.supylabel("Empirical rejection rate", x=0.012, fontsize=9)
    fig.supxlabel(r"Signal amplitude $a$", y=0.115, fontsize=9)
    handles = [
        Line2D([0], [0], color=c, marker=m, ls=s, markersize=4, label=label)
        for _, label, c, m, s in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.015),
        ncol=4,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.8,
    )
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.25, top=0.92, wspace=0.16, hspace=0.39
    )
    name = "hierarchical" if full_grid else f"{scenario}_compact"
    save_figure(fig, output, f"simulation_{name}")


def interpolation_figure(data, output):
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for method, label, color, marker, style in (
        ("distance_full", "Full sample", "#0072B2", "o", "-"),
        ("distance_crossfit", "Five-fold cross-fit", "#CC79A7", "D", "--"),
        ("oracle", "Oracle", "#333333", "s", ":"),
    ):
        points = data[data.method == method].sort_values("amplitude")
        ax.errorbar(
            points.amplitude,
            points.rejection_rate,
            yerr=[
                points.rejection_rate - points.wilson_lower,
                points.wilson_upper - points.rejection_rate,
            ],
            color=color,
            marker=marker,
            ls=style,
            markersize=4,
            capsize=3,
            elinewidth=0.8,
            markerfacecolor="none" if method == "oracle" else color,
            label=label,
        )
    ax.set_title("Distance-weighted interpolating kNN", pad=8)
    ax.set_xlabel(r"Signal amplitude $a$")
    ax.set_ylabel("Empirical rejection rate")
    ax.set_xticks((0, 0.25, 0.5, 1), ("0", "0.25", "0.5", "1"))
    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(-0.025, 1.025)
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1))
    ax.axhline(0.05, color=".5", lw=0.7, ls=(0, (2, 2)), zorder=-10)
    ax.grid(axis="y", color=".92", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.005),
        ncol=3,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.25, top=0.88)
    save_figure(fig, output, "simulation_crossfit_interpolation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "figures",
        help="output directory (default: results/figures)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
        }
    )
    data = pd.concat(
        [
            pd.read_csv(RESULTS / name)
            for name in ("frozen_summary.csv", "single_split_summary.csv")
        ],
        ignore_index=True,
    )
    data = data[
        data.scenario.isin(SCENARIOS) & data.method.isin([m[0] for m in METHODS])
    ]
    passed = data[data.size_gate_status == "pass"]
    for scenario in SCENARIOS:
        power_figure(passed, scenario, args.output)
    interpolation_figure(pd.read_csv(RESULTS / "interpolation.csv"), args.output)
    data[data.delta == 0].groupby("method").agg(
        cells=("p", "size"),
        minimum=("rejection_rate", "min"),
        maximum=("rejection_rate", "max"),
        failures=("size_gate_status", lambda values: (values != "pass").sum()),
    ).to_csv(args.output / "size_summary.csv")
    passed[passed.delta == 20].groupby(["scenario", "p", "method"]).rejection_rate.agg(
        ["min", "max", "count"]
    ).to_csv(args.output / "endpoint_summary.csv")
    matched = (
        passed[passed.method.isin([m[0] for m in METHODS[:2]])]
        .pivot(
            index=["scenario", "p", "learner", "delta"],
            columns="method",
            values="rejection_rate",
        )
        .dropna()
    )
    matched["gain"] = matched.orthogonal_nosplit - matched.orthogonal_single_split
    matched.to_csv(args.output / "matched_splitting_comparisons.csv")
    print(f"Saved five figures and three tables to {args.output}")


if __name__ == "__main__":
    main()
