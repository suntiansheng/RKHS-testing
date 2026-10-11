"""Measure cancellation in the frozen simulation signals, without fitting learners.

Reuse the calibration covariates and their negatives. Standard errors refer to
the independent antithetic pairs, not to 2*n independent observations. This is
a descriptive population-moment calculation, not another power experiment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
import simcore as core
from io_utils import atomic_write_csv, atomic_write_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/gcm_comparison")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    calibrations = json.loads((run / "calibrations.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise RuntimeError(f"Frozen simulation source changed: {name}")

    records = []
    for p in protocol["dimensions"]:
        settings = calibrations[str(p)]
        reference = settings["additive"]
        n = reference["calibration_n"]
        seed = reference["calibration_seed"]
        x = core.draw_gaussian_covariates(
            n, p, reference["rho"], np.random.default_rng(seed)
        )
        e = core.pooled_random_propensity(x)
        e_negative = core.pooled_random_propensity(-x)
        weight = e * (1 - e)
        weight_negative = e_negative * (1 - e_negative)
        raw = core.raw_regression_components(x)
        raw_negative = core.raw_regression_components(-x)

        for scenario in protocol["scenarios"]:
            calibration = settings[scenario]
            assert calibration["calibration_seed"] == seed
            signal = raw[scenario][1] * calibration["signal_scale"]
            negative_signal = raw_negative[scenario][1] * calibration["signal_scale"]
            np.testing.assert_allclose(
                np.sqrt(np.mean(signal**2)), calibration["signal_target_l2"],
                rtol=1e-12, atol=1e-12,
            )
            signed_pairs = (weight * signal + weight_negative * negative_signal) / 2
            absolute_pairs = (
                weight * np.abs(signal) + weight_negative * np.abs(negative_signal)
            ) / 2
            signed_mean = float(signed_pairs.mean())
            absolute_mean = float(absolute_pairs.mean())
            ratio = abs(signed_mean) / absolute_mean
            assert np.isfinite(ratio) and 0 <= ratio <= 1
            odd_error = float(np.max(np.abs(signal + negative_signal)))
            if scenario == "neural_composition":
                assert odd_error < 1e-12
            records.append(dict(
                p=p, scenario=scenario, independent_pairs=n,
                calibration_seed=seed, signed_weighted_mean=signed_mean,
                signed_weighted_mean_mcse=float(signed_pairs.std(ddof=1) / np.sqrt(n)),
                absolute_weighted_mean=absolute_mean, cancellation_ratio=ratio,
                max_odd_symmetry_error=odd_error,
            ))

    result = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "gcm_signal_diagnostic.csv"
    atomic_write_csv(output, result)
    inputs = [Path(__file__), run / "protocol.json", run / "calibrations.json",
              ROOT / "simulation/simcore.py"]
    atomic_write_json(args.output / "gcm_signal_diagnostic_metadata.json", dict(
        description="Absolute signed weighted signal mean / weighted absolute signal mean",
        calculation="Frozen reference covariates and their negatives; true e_0 and f; no fitting",
        source_sha256={Path(os.path.relpath(path, ROOT)).as_posix(): file_hash(path) for path in inputs},
        output_sha256=file_hash(output),
        rows=len(result),
    ))
    print(result[["p", "scenario", "signed_weighted_mean", "cancellation_ratio"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
