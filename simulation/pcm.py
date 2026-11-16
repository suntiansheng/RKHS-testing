"""Run the authors' pinned R PCM code with its original Ranger defaults."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PCM_LEARNER = "pcm_ranger"
PCM_METHODS = ("pcm_single_split", "pcm_derandomized")
HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "pcm_code"
BRIDGE = HERE / "pcm_bridge.R"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pcm_provenance() -> dict[str, Any]:
    """Fail if a vendored upstream file differs from its pinned original."""
    manifest_path = VENDOR / "provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if _sha256(VENDOR / name) != expected:
            raise RuntimeError(f"Vendored PCM source has changed: {name}")
    return {
        **manifest,
        "adapter_sha256": _sha256(Path(__file__)),
        "bridge_sha256": _sha256(BRIDGE),
        "manifest_sha256": _sha256(manifest_path),
    }


def _rscript(options: Mapping[str, Any]) -> str:
    requested = str(options.get("rscript", os.environ.get("PCM_RSCRIPT", "Rscript")))
    executable = shutil.which(requested)
    if executable is None:
        raise RuntimeError("Original PCM requires Rscript; install R or set PCM_RSCRIPT.")
    return str(Path(executable).resolve())


def _options(options: Mapping[str, Any]) -> tuple[str, float]:
    unknown = set(options) - {"rscript", "timeout_seconds"}
    if unknown:
        raise ValueError(f"PCM statistical settings are fixed; unknown option(s): {sorted(unknown)}")
    timeout = float(options.get("timeout_seconds", 600))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("PCM timeout_seconds must be finite and positive.")
    return _rscript(options), timeout


def _execute(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["TMPDIR"] = tempfile.gettempdir()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Original PCM exceeded its {timeout:g}s execution limit.") from error
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise RuntimeError(f"Original PCM R process failed: {detail}")
    return completed


def pcm_runtime_metadata(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = dict(options or {})
    executable, timeout = _options(settings)
    provenance = pcm_provenance()
    completed = _execute([executable, "--vanilla", str(BRIDGE), "--metadata"], timeout)
    return {
        "source": provenance,
        "rscript": executable,
        "runtime": json.loads(completed.stdout),
        "learner": "original ranger_reg_method; 500 trees, mtry=all predictors, other defaults",
        "calibration": "upper standard-normal tail; DR uses the mean of six split statistics",
    }


def run_original_pcm(
    y: np.ndarray,
    d: np.ndarray,
    X: np.ndarray,
    *,
    seed: int,
    methods: Sequence[str] = PCM_METHODS,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run PCM with response y, tested predictor d, and conditioning covariates X.

    One R seed precedes all six sequential calls. Failed fits are reported
    without retrying or substituting another split.
    """
    selected = tuple(methods)
    if not selected or set(selected) - set(PCM_METHODS):
        raise ValueError("Choose one or both original PCM methods.")
    response = np.asarray(y, dtype=float)
    group = np.asarray(d, dtype=float)
    covariates = np.asarray(X, dtype=float)
    if (response.ndim != 1 or group.shape != response.shape or covariates.ndim != 2
            or covariates.shape[0] != len(response) or covariates.shape[1] < 1
            or len(response) < 8 or len(response) % 2):
        raise ValueError("PCM needs matching y, d, X arrays and an even sample size >= 8.")
    data = np.column_stack((response, group, covariates))
    if not np.isfinite(data).all() or not np.isin(group, (0, 1)).all():
        raise ValueError("PCM data must be finite and d must be binary.")
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= int(seed) < 2**31 - 1:
        raise ValueError("PCM seed must be an integer in [0, 2**31 - 1).")
    settings = dict(options or {})
    executable, timeout = _options(settings)
    provenance = pcm_provenance()
    splits = 6 if "pcm_derandomized" in selected else 1
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rkhs-pcm-") as temporary:
        directory = Path(temporary)
        input_path, output_path = directory / "data.csv", directory / "result.json"
        header = ",".join(["Y", "D", *(f"X{j}" for j in range(covariates.shape[1]))])
        np.savetxt(input_path, data, delimiter=",", header=header, comments="", fmt="%.17g")
        completed = _execute(
            [executable, "--vanilla", str(BRIDGE), str(VENDOR / "test_functions.R"),
             str(input_path), str(output_path), str(int(seed)), str(splits)], timeout,
        )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - started
    split_results = payload["splits"]
    overhead = max(0.0, elapsed - sum(part["elapsed_seconds"] for part in split_results))
    for part in split_results:
        if part["status"] == "ok":
            direction = np.asarray(part.pop("direction_indices"), dtype=np.int64)
            testing = np.asarray(part.pop("main_indices"), dtype=np.int64)
            if (len(direction) != len(response) // 2 or len(testing) != len(response) // 2
                    or set(direction) & set(testing)
                    or set(direction) | set(testing) != set(range(1, len(response) + 1))):
                raise RuntimeError("Original PCM returned an invalid partition.")
            part["partition_sha256"] = hashlib.sha256(
                json.dumps([direction.tolist(), testing.tolist()], separators=(",", ":")).encode()
            ).hexdigest()
    results = {method: payload["methods"][method] for method in selected}
    for method, result in results.items():
        if result["status"] == "ok":
            if (not math.isfinite(result["statistic"]) or not math.isfinite(result["p_value"])
                    or not 0 <= result["p_value"] <= 1):
                raise RuntimeError("Original PCM returned an invalid result.")
        result["runtime_seconds"] = (
            overhead + split_results[0]["elapsed_seconds"]
            if method == "pcm_single_split" else elapsed
        )
    return {
        "methods": results,
        "diagnostics": {
            "source": provenance,
            "runtime": payload["runtime"],
            "r_seed": int(seed),
            "n_splits": splits,
            "split_results": split_results,
            "total_runtime_seconds": elapsed,
            "r_stderr": completed.stderr.strip(),
            "tested_predictor": "D",
            "conditioning_covariates": "all X coordinates",
            "regression_settings": "unmodified original ranger_reg_method defaults",
            "calibration": "one-sided standard normal; DR averages six statistics",
        },
    }
