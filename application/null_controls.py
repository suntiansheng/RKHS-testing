"""Random-label null diagnostics conditional on the saved CaSiNo sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from core import outcome_model, propensity_model, rbf_kernel_matrix, stable_seed


def wilson(successes: int, trials: int) -> tuple[float, float]:
    z = norm.ppf(0.975)
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = (
        z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials**2)) / denominator
    )
    return float(center - half), float(center + half)


def run(
    data_dir: Path, output_dir: Path, replications: int, bootstrap_draws: int
) -> None:
    if replications < 1 or bootstrap_draws < 1:
        raise ValueError("Replications and bootstrap draws must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with np.load(data_dir / "integrative_potential_design.npz") as design:
        sample = pd.read_csv(data_dir / "integrative_potential_sample.csv")
        np.testing.assert_array_equal(design["dialogue_id"], sample.dialogue_id)
        np.testing.assert_array_equal(design["D"], sample.D)
        covariates, groups = design["X"], design["D"]
    rows, fits = [], []
    for group, title in [(0, "Aligned"), (1, "Maximal potential")]:
        mask = groups == group
        x = covariates[mask]
        y = sample.mean_satisfaction.to_numpy()[mask]
        n = len(y)
        outcome = outcome_model("ridge", stable_seed("v8-null-outcome", group), 5)
        outcome.fit(x, y)
        residual = y - np.clip(outcome.predict(x), 1, 5)
        fits.append(
            {
                "group": title,
                "n": n,
                "ridge_alpha": outcome.best_params_["model__alpha"],
            }
        )
        kernel = rbf_kernel_matrix(x, 1.0)
        for rep in range(replications):
            label_seed = stable_seed("v8-null-label", group, rep)
            labels = np.random.default_rng(label_seed).binomial(1, 0.5, n)
            if min(np.bincount(labels, minlength=2)) < 5:
                raise ValueError(
                    "A random-label group has fewer than five observations for tuning."
                )
            propensity = propensity_model(
                "ridge", stable_seed("v8-null-prop", group, rep), 5
            )
            propensity.set_params(estimator__model__random_state=None)
            propensity.fit(x, labels)
            fitted_e = propensity.predict_proba(x)[:, 1]
            multiplier_seed = stable_seed("v8-null-multiplier", group, rep)
            xi = np.random.default_rng(multiplier_seed).standard_normal(
                (n, bootstrap_draws)
            )
            xi -= xi.mean(axis=0, keepdims=True)
            for name, e in [
                ("Estimated propensity", fitted_e),
                ("Known propensity", 0.5),
            ]:
                scores = residual * (labels - e)
                observed = float(scores @ kernel @ scores / n)
                weighted = scores[:, None] * xi
                bootstrap = np.sum(weighted * (kernel @ weighted), axis=0) / n
                if not np.isfinite(bootstrap).all() or observed < -1e-12:
                    raise FloatingPointError("Invalid multiplier statistic.")
                p_value = (1 + np.count_nonzero(bootstrap >= observed)) / (
                    bootstrap_draws + 1
                )
                rows.append(
                    {
                        "group": title,
                        "n": n,
                        "repetition": rep,
                        "propensity": name,
                        "label_seed": label_seed,
                        "multiplier_seed": multiplier_seed,
                        "randomized_group_one": int(labels.sum()),
                        "statistic": float(np.sqrt(max(observed, 0))),
                        "p_value": p_value,
                        "reject": int(p_value <= 0.05),
                        "selected_C": float(propensity.best_params_["model__C"]),
                    }
                )
            if (rep + 1) % 50 == 0:
                pd.DataFrame(rows).to_csv(
                    output_dir / "null_control_raw.csv", index=False
                )
                print(f"{title}: {rep + 1}/{replications}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "null_control_raw.csv", index=False)
    summaries = []
    for (group, propensity), frame in raw.groupby(["group", "propensity"], sort=False):
        rejections = int(frame.reject.sum())
        lower, upper = wilson(rejections, len(frame))
        summaries.append(
            {
                "group": group,
                "n": int(frame.n.iloc[0]),
                "propensity": propensity,
                "repetitions": len(frame),
                "rejections": rejections,
                "rejection_rate": rejections / len(frame),
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "null_control_summary.csv", index=False)
    metadata = {
        "repetitions": replications,
        "bootstrap_draws": bootstrap_draws,
        "level": 0.05,
        "randomization": "Independent Bernoulli(1/2) labels within each saved regime; all assignments retained.",
        "estimand": "Random-label rejection probability conditional on the observed covariates and outcomes.",
        "fit_details": fits,
        "results": summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "inputs": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in (
                "integrative_potential_design.npz",
                "integrative_potential_sample.csv",
            )
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "null_control_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=here / "results_integrative_potential"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=here / "results_null_controls"
    )
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--bootstrap-draws", type=int, default=999)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir, args.output_dir, args.replications, args.bootstrap_draws)


if __name__ == "__main__":
    main()
