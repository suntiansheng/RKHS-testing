"""Compare full-sample and cross-fitted scores under nearest-neighbour interpolation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.neighbors import KNeighborsRegressor
from threadpoolctl import threadpool_limits

SEED = 2026092108
N = 400
FOLDS = 5
AMPLITUDES = (0, 0.25, 0.5, 1)
LEVEL = 0.05


def rng_for(rep, purpose):
    return np.random.default_rng(np.random.SeedSequence([SEED, 1, N, rep, purpose]))


def test_scores(scores, kernel, multipliers):
    observed = max(float(scores @ kernel @ scores / len(scores)), 0.0)
    weighted = scores[:, None] * multipliers
    bootstrap = np.maximum(
        np.sum(weighted * (kernel @ weighted), axis=0) / len(scores), 0
    )
    pvalue = (1 + np.count_nonzero(bootstrap >= observed)) / (len(bootstrap) + 1)
    return {
        "reject": int(pvalue <= LEVEL),
        "pvalue": float(pvalue),
        "statistic": float(np.sqrt(observed)),
        "score_rms": float(np.sqrt(np.mean(scores**2))),
    }


def run_replication(rep, bootstrap_draws=199):
    rng = rng_for(rep, 1)
    x = rng.normal(size=(N, 2)) @ np.linalg.cholesky([[1, 0.5], [0.5, 1]]).T
    v = 2 * ndtr(x) - 1
    propensity = 0.5 + 0.1 * v[:, 0] + 0.15 * (v[:, 1] ** 2 - 1 / 3)
    labels = rng.binomial(1, propensity)
    noise = rng.normal(size=N)
    baseline = np.sin(x[:, 0]) + 0.5 * x[:, 1]
    signal = 1 + 0.5 * np.sin(x[:, 0])
    folds = np.empty(N, dtype=int)
    folds[rng_for(rep, 2).permutation(N)] = np.arange(N) % FOLDS
    kernel = np.exp(-cdist(x, x, "sqeuclidean") / 2)
    multipliers = rng_for(rep, 3).standard_normal((N, bootstrap_draws))
    multipliers -= multipliers.mean(axis=0, keepdims=True)
    rows = []
    for amplitude in AMPLITUDES:
        y = baseline + amplitude * labels * signal + noise
        truth = baseline + amplitude * propensity * signal
        learner = KNeighborsRegressor(
            n_neighbors=20, weights="distance", algorithm="brute", n_jobs=1
        )
        learner.fit(x, y)
        full = learner.predict(x)
        heldout = np.empty(N)
        for fold in range(FOLDS):
            train, test = folds != fold, folds == fold
            learner.fit(x[train], y[train])
            heldout[test] = learner.predict(x[test])
        if not np.array_equal(full, y):
            raise RuntimeError(
                "Distance-weighted kNN did not interpolate the training data."
            )
        for method, prediction in (
            ("oracle", truth),
            ("distance_full", full),
            ("distance_crossfit", heldout),
        ):
            scores = (y - prediction) * (labels - propensity)
            rows.append(
                {
                    "design": "knn",
                    "n": N,
                    "amplitude": amplitude,
                    "rep": rep,
                    "method": method,
                    "evaluation_mu_mse": float(np.mean((prediction - truth) ** 2)),
                    **test_scores(scores, kernel, multipliers),
                }
            )
    return rows


def summarize(raw):
    summary = raw.groupby(["design", "n", "amplitude", "method"], as_index=False).agg(
        replications=("rep", "count"),
        rejections=("reject", "sum"),
        rejection_rate=("reject", "mean"),
        evaluation_mu_mse=("evaluation_mu_mse", "mean"),
    )
    z = norm.ppf(0.975)
    n, rate = summary.replications, summary.rejection_rate
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    summary["mc_se"] = np.sqrt(rate * (1 - rate) / n)
    summary["wilson_lower"] = np.minimum(rate, np.maximum(0, center - half))
    summary["wilson_upper"] = np.maximum(rate, np.minimum(1, center + half))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--bootstrap-draws", type=int, default=199)
    parser.add_argument("--output", type=Path, default=Path("runs/interpolation"))
    args = parser.parse_args()
    if args.replications < 1 or args.bootstrap_draws < 1:
        parser.error("replications and bootstrap-draws must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with threadpool_limits(limits=1):
        for rep in range(args.replications):
            rows.extend(run_replication(rep, args.bootstrap_draws))
            if (rep + 1) % 25 == 0 or rep + 1 == args.replications:
                print(
                    f"Completed {rep + 1}/{args.replications} replications", flush=True
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output / "raw.csv", index=False)
    summarize(raw).to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "seed": SEED,
        "n": N,
        "folds": FOLDS,
        "amplitudes": AMPLITUDES,
        "replications": args.replications,
        "bootstrap_draws": args.bootstrap_draws,
        "alpha": LEVEL,
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
