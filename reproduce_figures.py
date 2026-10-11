"""Reproduce the seven simulation figures in v10 from archived summaries."""

import argparse
import hashlib
import json
from pathlib import Path

import plot_results
from experiments import plot_crossfit_manuscript


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANUSCRIPT_NAMES = {
    "simulation_hierarchical": "simulation_hierarchical_v9",
    "simulation_additive_compact": "simulation_additive_compact_v9",
    "simulation_product_compact": "simulation_product_compact_v9",
    "simulation_composition_compact": "simulation_composition_compact_v9",
    "simulation_crossfit_interpolation": "simulation_crossfit_interpolation_v9",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archives():
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = RESULTS / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Archived result is missing or changed: {name}")
    return manifest["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures/v10")
    args = parser.parse_args()
    sources = verify_archives()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    plot_results.main(["--output", str(output)])
    for old, new in MANUSCRIPT_NAMES.items():
        (output / f"{old}.pdf").replace(output / f"{new}.pdf")
    crossfit = plot_crossfit_manuscript.build(
        RESULTS / "crossfit_comparison/n120_n160_k2", output
    )
    names = list(MANUSCRIPT_NAMES.values()) + list(plot_crossfit_manuscript.STEMS)
    report = {
        "manuscript": "v10.tex",
        "input_sha256": sources,
        "output_sha256": {f"{name}.pdf": sha256(output / f"{name}.pdf") for name in names},
        "crossfit_null_family_cells": crossfit["null_family_cells"],
        "interpolation_bootstrap_draws": 199,
        "note": "The interpolation archive used 199 draws; v10.tex states 200. See results/README.md.",
    }
    (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved all seven simulation figures to {output}")


if __name__ == "__main__":
    main()
