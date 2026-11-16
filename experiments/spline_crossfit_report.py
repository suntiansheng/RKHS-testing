"""Check the pilot selection and summarize the spline comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spline_crossfit_plot import load_results


ROOT = Path(__file__).resolve().parents[1]


def select_amplitudes(pilot):
    config = json.loads((pilot / "config.json").read_text())
    validation = json.loads((pilot / "validation.json").read_text())
    if not validation["complete"] or validation["failures"]:
        raise ValueError("Pilot selection requires a complete run")
    summary = pd.read_csv(pilot / "summary.csv")
    selected = summary[summary.dgp.eq("original") & summary.propensity_mode.eq("estimated")
                       & summary.method.eq("full") & summary.amplitude.gt(0)]
    power = selected.pivot(index="amplitude", columns="n", values="rejection_rate")
    if list(power.columns) != config["sample_sizes"] or power.isna().any().any():
        raise ValueError("Missing pilot NORD power cells")
    eligible = power.ge(.10).all(axis=1) & power.le(.85).all(axis=1)
    return power.index[eligible].tolist()


def nuisance_comparisons(summary, amplitudes):
    selected = summary[summary.amplitude.isin(amplitudes)
                       & summary.propensity_mode.eq("estimated")
                       & summary.method.isin(["full", "cf2"])]
    result = selected.pivot(index=["n", "dgp", "amplitude"], columns="method",
                            values="independent_mu_mse").reset_index()
    result["NORD_to_CF2_MSE_ratio"] = result.full / result.cf2
    return result


def build(pilot, run, output):
    pilot, run, output = Path(pilot), Path(run), Path(output)
    amplitudes = select_amplitudes(pilot)
    summary, config = load_results(run)
    if amplitudes != config["primary_amplitudes"]:
        raise ValueError("Confirmation amplitudes differ from the independent pilot selection")
    pilot_config = json.loads((pilot / "config.json").read_text())
    if pilot_config["seed_base"] == config["seed_base"]:
        raise ValueError("Pilot and confirmation must use separate seeds")
    pairs = pd.read_csv(run / "paired_differences.csv")
    primary = pairs[pairs.family.eq("primary_estimated_cf2")]
    if len(primary) != 12 or not primary.replications.eq(3000).all():
        raise ValueError("Missing primary paired comparisons")
    means = nuisance_comparisons(summary, amplitudes)
    output.mkdir(parents=True, exist_ok=True)
    primary.to_csv(output / "primary_comparisons.csv", index=False)
    means.to_csv(output / "nuisance_prediction_comparison.csv", index=False)
    report = {
        "primary_amplitudes": amplitudes,
        "primary_comparisons": len(primary),
        "positive_observed_power_differences": int(primary.difference.gt(0).sum()),
        "minimum_power_difference": float(primary.difference.min()),
        "maximum_power_difference": float(primary.difference.max()),
        "minimum_MSE_reduction": float((1-means.NORD_to_CF2_MSE_ratio).min()),
        "maximum_MSE_reduction": float((1-means.NORD_to_CF2_MSE_ratio).max()),
        "null_rates": summary[summary.amplitude.eq(0)][
            ["n", "method", "propensity_mode", "rejection_rate"]].to_dict("records"),
    }
    if not np.isfinite(means[["full", "cf2", "NORD_to_CF2_MSE_ratio"]]).all().all():
        raise ValueError("Nonfinite prediction-error comparisons")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "results/spline_crossfit/pilot")
    parser.add_argument("--run", type=Path, default=ROOT / "results/spline_crossfit/confirmation")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/spline_crossfit")
    args = parser.parse_args()
    print(json.dumps(build(args.pilot, args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
