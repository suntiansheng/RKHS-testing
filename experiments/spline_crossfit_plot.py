"""Plot the spline cross-fitting comparisons reported in the supplement."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
STEMS = (
    "simulation_crossfit_spline_null_revisit",
    "simulation_crossfit_spline_power_models",
)
MODES = (("estimated", "Estimated propensity"), ("known", "Known propensity"))
MODELS = (("original", "Model (i)"), ("fixed_pooled_mean", "Model (ii)"))


def load_results(run):
    run = Path(run)
    validation = json.loads((run / "validation.json").read_text())
    if not validation["complete"] or validation["failures"]:
        raise ValueError("The spline comparison requires a complete run without failures")
    config = json.loads((run / "config.json").read_text())
    summary = pd.read_csv(run / "summary.csv")
    keys = ["n", "dgp", "amplitude", "method", "propensity_mode"]
    if (len(summary) != 100 or summary.duplicated(keys).any()
            or not summary.complete.all()
            or not summary.replications.eq(np.where(summary.amplitude.eq(0), 5000, 3000)).all()
            or not summary.replications.eq(summary.target_replications).all()):
        raise ValueError("Unexpected or incomplete spline comparison cells")
    if (config["sample_sizes"] != [120, 160]
            or config["amplitudes"] != [0., 1.2, 2., 2.5, 3.2]
            or config["primary_amplitudes"] != [2., 2.5, 3.2]
            or config["bootstrap_draws"] != 999 or config["reference_n"] != 1000):
        raise ValueError("The spline configuration does not match the reported experiment")
    np.testing.assert_allclose(
        summary.rejection_rate, summary.rejections / summary.replications,
        rtol=0, atol=1e-14,
    )
    successes, counts = summary.rejections.to_numpy(), summary.replications.to_numpy()
    lower = np.where(successes == 0, 0., beta.ppf(.025, successes, counts-successes+1))
    upper = np.where(successes == counts, 1., beta.ppf(.975, successes+1, counts-successes))
    np.testing.assert_allclose(summary.ci_lower, lower, rtol=0, atol=1e-14)
    np.testing.assert_allclose(summary.ci_upper, upper, rtol=0, atol=1e-14)
    return summary, config


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=.04,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def null_figure(summary, output):
    null = summary[summary.amplitude.eq(0)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    methods = (("full", "NORD", "#0072B2", "o"),
               ("cf2", "CF-NORD", "#D55E00", "s"),
               ("oracle", "Oracle", "#333333", "^"))
    for ax, (mode, title) in zip(axes, MODES):
        for offset, (method, label, color, marker) in zip((-.13, 0., .13), methods):
            points = null[null.method.eq(method)
                          & (null.propensity_mode.eq(mode) | null.method.eq("oracle"))]
            points = points.sort_values("n")
            if points.n.tolist() != [120, 160]:
                raise ValueError("Missing spline null cells")
            ax.errorbar(np.arange(2)+offset, points.rejection_rate,
                        yerr=[points.rejection_rate-points.ci_lower,
                              points.ci_upper-points.rejection_rate],
                        color=color, marker=marker, ls="none", capsize=3,
                        markersize=4, elinewidth=.8, label=label)
        ax.axhline(.05, color=".5", lw=.7, ls=":")
        ax.set_title(title)
        ax.set_xticks([0, 1], [120, 160])
        ax.set_xlabel(r"Sample size $n$")
        ax.set_ylim(0, .105)
        ax.set_yticks(np.arange(0, .101, .02))
        ax.grid(axis="y", color=".92", lw=.5)
    axes[0].set_ylabel("Null rejection rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=.1, right=.99, top=.88, bottom=.27, wspace=.12)
    save(fig, output, STEMS[0])


def power_figure(summary, config, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.4), sharex=True, sharey=True)
    specs = ((120, "full", "NORD", "#0072B2", "o", "-"),
             (120, "cf2", "CF-NORD", "#D55E00", "s", "--"),
             (160, "full", "NORD", "#009E73", "^", "-"),
             (160, "cf2", "CF-NORD", "#CC79A7", "D", "--"))
    for row, (model, label) in enumerate(MODELS):
        for col, (mode, title) in enumerate(MODES):
            ax = axes[row, col]
            for n, method, _, color, marker, style in specs:
                points = summary[summary.n.eq(n) & summary.method.eq(method)
                                 & summary.propensity_mode.eq(mode)
                                 & summary.dgp.isin([model, "shared_null"])].sort_values("amplitude")
                if points.amplitude.tolist() != config["amplitudes"]:
                    raise ValueError("Missing spline power cells")
                ax.plot(points.amplitude, points.rejection_rate,
                        color=color, marker=marker, ls=style, markersize=3)
            ax.axhline(.05, color=".55", lw=.6, ls=":")
            ax.set_title(f"{title}\n{label}" if row == 0 else label)
            ax.set_xticks(config["amplitudes"], ["0", "1.2", "2", "2.5", "3.2"])
            ax.set_ylim(-.025, .65)
            ax.set_yticks((0, .2, .4, .6))
            ax.grid(axis="y", color=".92", lw=.5)
    handles = [Line2D([0], [0], color=color, marker=marker, ls=style,
                      markersize=4, label=f"{label}, $n={n}$")
               for n, _, label, color, marker, style in specs]
    fig.supylabel("Rejection rate", x=.015)
    fig.supxlabel(r"Signal amplitude $a$", y=.1)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               columnspacing=1.2, handlelength=2)
    fig.subplots_adjust(left=.09, right=.99, top=.88, bottom=.23, hspace=.4, wspace=.15)
    save(fig, output, STEMS[1])


def build(run, output):
    summary, config = load_results(run)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.linewidth": .6, "lines.linewidth": 1.2,
                         "legend.fontsize": 8})
    null_figure(summary, output)
    power_figure(summary, config, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/spline_crossfit/confirmation")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures")
    args = parser.parse_args()
    build(args.run, args.output)
    print(f"Saved the two spline comparison figures to {args.output}")


if __name__ == "__main__":
    main()
