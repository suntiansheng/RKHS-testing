"""Reproduce every final manuscript figure from the supplied result tables."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import plot_results
from experiments import cate_report, late_report, plot_crossfit_manuscript
from application import plot as application_plot
from application import plot_joint


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANUSCRIPT_NAMES = {
    "simulation_hierarchical": "simulation_hierarchical_v9",
    "simulation_additive_compact": "simulation_additive_compact_v9",
    "simulation_product_compact": "simulation_product_compact_v9",
    "simulation_composition_compact": "simulation_composition_compact_v9",
    "simulation_crossfit_interpolation": "simulation_crossfit_interpolation_v9",
}
CAUSAL_NAMES = ("simulation_cate_hierarchical", "simulation_late_hierarchical",
                "simulation_cate_hierarchical_crossfit", "simulation_late_hierarchical_crossfit")
WORD_NAME = "casino_word_contrasts_v5"
FIGURE_NAMES = (tuple(MANUSCRIPT_NAMES.values()) + tuple(plot_crossfit_manuscript.STEMS)
                + CAUSAL_NAMES + tuple(plot_joint.FIGURES.values()) + (WORD_NAME,))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archives():
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = RESULTS / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Archived result is missing or changed: {name}")
    return manifest["sha256"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures/jrssb")
    args = parser.parse_args(argv)
    sources = verify_archives()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    plot_results.main(["--output", str(output)])
    for old, new in MANUSCRIPT_NAMES.items():
        (output / f"{old}.pdf").replace(output / f"{new}.pdf")
    plot_crossfit_manuscript.build(
        RESULTS / "crossfit_comparison/n120_n160_k2", output
    )
    for name, report_module in (("cate", cate_report), ("late", late_report)):
        directory = RESULTS / f"{name}_hierarchical"
        summary = pd.read_csv(directory / "rejection_summary.csv")
        if (len(summary) != 204 or not summary.n_valid.eq(300).all()
                or not summary.n_failed.eq(0).all() or not summary.n_missing.eq(0).all()):
            raise ValueError(f"Incomplete {name.upper()} result archive")
        null = pd.read_csv(directory / "null_size.csv")
        report_module.plot_power(summary, null, output)
        report_module.plot_power(summary, null, output, include_crossfit=True)
    plot_joint.create_figures(RESULTS / "casino", output)
    application_plot.configure_style()
    words = pd.read_csv(RESULTS / "casino/word_log_odds.csv")
    application_plot.save_figure(application_plot.make_manuscript_word_figure(words), output, WORD_NAME)
    report = {
        "manuscript": "JRSSB submission: main.tex and supplement.tex",
        "input_sha256": sources,
        "output_sha256": {f"{name}.pdf": sha256(output / f"{name}.pdf") for name in FIGURE_NAMES},
        "interpolation_bootstrap_draws": 199,
    }
    (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved all {len(FIGURE_NAMES)} manuscript figure PDFs to {output}")


if __name__ == "__main__":
    main()
