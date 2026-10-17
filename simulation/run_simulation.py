#!/usr/bin/env python3
"""Run and resume RKHS two-sample simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import torch

from io_utils import atomic_write_csv, atomic_write_json, file_hash
from simcore import (
    SCHEMA_VERSION,
    SUPPORTED_DESIGNS,
    SUPPORTED_LEARNERS,
    SUPPORTED_METHODS,
    SUPPORTED_SCENARIOS,
    ScenarioCalibration,
    calibrate_scenarios,
    package_versions,
    run_replication,
    stable_seed,
    validate_late_options,
    validate_treatment_noise_scales,
)


def parse_csv(value: Optional[str], cast: Any = str) -> Optional[list[Any]]:
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def task_manifest_hash(tasks: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(task["task_key"]) for task in tasks).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "name",
        "seed_base",
        "designs",
        "scenarios",
        "dimensions",
        "deltas",
        "replications",
        "n",
        "target_pi",
        "rho",
        "sigma",
        "alpha",
        "bootstrap_draws",
        "learners",
        "methods",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    unknown_designs = set(config["designs"]) - set(SUPPORTED_DESIGNS)
    unknown_scenarios = set(config["scenarios"]) - set(SUPPORTED_SCENARIOS)
    unknown_learners = set(config["learners"]) - set(SUPPORTED_LEARNERS)
    unknown_methods = set(config["methods"]) - set(SUPPORTED_METHODS)
    if unknown_designs:
        raise ValueError(f"Unknown designs: {sorted(unknown_designs)}")
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {sorted(unknown_scenarios)}")
    if unknown_learners:
        raise ValueError(f"Unknown learners: {sorted(unknown_learners)}")
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    if any(int(p) < 8 for p in config["dimensions"]):
        raise ValueError("Every dimension must be at least 8.")
    if int(config["n"]) < 8:
        raise ValueError("The pooled sample size must be at least 8.")
    if "orthogonal_crossfit" in config["methods"]:
        folds = config.get("crossfit_folds", 2)
        if (isinstance(folds, bool) or not isinstance(folds, int)
                or not 2 <= folds <= int(config["n"])):
            raise ValueError("Cross-fitting requires an integer number of folds between 2 and n.")
    if not math.isclose(float(config["target_pi"]), 0.5, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("The pooled random-shift design requires target_pi=0.5.")
    if int(config["replications"]) < 1 or int(config["bootstrap_draws"]) < 1:
        raise ValueError("replications and bootstrap_draws must be positive.")
    if not 0.0 < float(config["alpha"]) < 1.0:
        raise ValueError("alpha must lie in (0,1).")
    propensity_mode = config.get("propensity_mode", "estimated")
    if propensity_mode not in ("known", "oracle", "estimated"):
        raise ValueError("propensity_mode must be known, oracle, or estimated.")
    if propensity_mode != "estimated":
        raise ValueError(
            "Feasible methods must estimate propensity; "
            "the oracle learner receives the truth separately."
        )
    if "oracle" in config["learners"] and "oracle_score" not in config["methods"]:
        raise ValueError("The oracle learner requires oracle_score in methods.")
    if "treatment_noise_scales" in config:
        validate_treatment_noise_scales(config["treatment_noise_scales"])
    if "late_iv" in config["designs"]:
        if "late_options" not in config:
            raise ValueError("late_iv requires an explicit late_options configuration.")
        validate_late_options(config["late_options"])
        if (float(config["sigma"]) != 1.0
                or validate_treatment_noise_scales(config.get("treatment_noise_scales", []))
                != (1.0, 1.5)):
            raise ValueError("late_iv requires sigma=1 and treatment_noise_scales=[1,1.5].")


def task_key(task: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(task["design"]),
            str(task["scenario"]),
            f"p{int(task['p'])}",
            f"delta{float(task['delta']):.12g}",
            f"rep{int(task['rep'])}",
            str(task["learner"]),
        ]
    )


def build_tasks(
    config: Mapping[str, Any],
    *,
    designs: Optional[Sequence[str]],
    scenarios: Optional[Sequence[str]],
    dimensions: Optional[Sequence[int]],
    deltas: Optional[Sequence[float]],
    learners: Optional[Sequence[str]],
    rep_start: int,
    rep_stop: Optional[int],
) -> list[dict[str, Any]]:
    selected_designs = list(designs or config["designs"])
    selected_scenarios = list(scenarios or config["scenarios"])
    selected_dimensions = list(dimensions or config["dimensions"])
    selected_deltas = list(deltas if deltas is not None else config["deltas"])
    selected_learners = list(learners or config["learners"])
    stop = int(rep_stop or config["replications"])
    if rep_start < 1 or stop < rep_start or stop > int(config["replications"]):
        raise ValueError("Requested replication range is invalid.")
    for values, supported, label in (
        (selected_designs, SUPPORTED_DESIGNS, "design"),
        (selected_scenarios, SUPPORTED_SCENARIOS, "scenario"),
        (selected_learners, SUPPORTED_LEARNERS, "learner"),
    ):
        unknown = set(values) - set(supported)
        if unknown:
            raise ValueError(f"Unknown selected {label}(s): {sorted(unknown)}")

    tasks: list[dict[str, Any]] = []
    for design in selected_designs:
        for p in selected_dimensions:
            for scenario in selected_scenarios:
                for delta in selected_deltas:
                    for rep in range(rep_start, stop + 1):
                        # Covariates, random labels, and noise are paired across
                        # scenarios and deltas. Fit randomness changes with the
                        # response and learner; bootstrap randomness is shared.
                        dgp_seed = stable_seed(
                            config["seed_base"], design, int(p), rep, "dgp"
                        )
                        bootstrap_seed = stable_seed(
                            config["seed_base"],
                            design,
                            int(p),
                            scenario,
                            float(delta),
                            rep,
                            "bootstrap",
                        )
                        for learner in selected_learners:
                            task = {
                                "design": design,
                                "scenario": scenario,
                                "p": int(p),
                                "n": int(config["n"]),
                                "rho": float(config["rho"]),
                                "sigma": float(config["sigma"]),
                                "delta": float(delta),
                                "rep": rep,
                                "learner": learner,
                                "alpha": float(config["alpha"]),
                                "bootstrap_draws": int(config["bootstrap_draws"]),
                                "dgp_seed": dgp_seed,
                                "fit_seed": stable_seed(
                                    config["seed_base"],
                                    design,
                                    int(p),
                                    scenario,
                                    float(delta),
                                    rep,
                                    learner,
                                    "fit",
                                ),
                                "bootstrap_seed": bootstrap_seed,
                            }
                            if "treatment_noise_scales" in config:
                                task["treatment_noise_scales"] = list(
                                    validate_treatment_noise_scales(
                                        config["treatment_noise_scales"]
                                    )
                                )
                            if design == "late_iv":
                                task["late_options"] = validate_late_options(config["late_options"])
                            task["task_key"] = task_key(task)
                            tasks.append(task)
    return tasks


def make_calibrations(
    config: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, int, str], ScenarioCalibration]:
    requested = sorted(
        {(str(task["design"]), int(task["p"]), str(task["scenario"])) for task in tasks}
    )
    calibration_n = int(config.get("calibration_n", 100_000))
    output: dict[tuple[str, int, str], ScenarioCalibration] = {}
    for design, p, scenario in requested:
        # Keep the calibration seed shared across scenarios.
        seed = stable_seed(config["seed_base"], "equal", p, "calibration")
        calibrated = calibrate_scenarios(
            p=p,
            rho=float(config["rho"]),
            design=design,
            scenarios=[scenario],
            calibration_n=calibration_n,
            seed=seed,
        )
        output[(design, p, scenario)] = calibrated[scenario]
    return output


def execute_task(
    task: Mapping[str, Any],
    calibration: ScenarioCalibration,
    config: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    rows, dgp = run_replication(task, calibration, config)
    key = str(task["task_key"])
    for row in rows:
        row["task_key"] = key
        row["config_name"] = config["name"]
    diagnostic = {
        "task_key": key,
        "design": task["design"],
        "scenario": task["scenario"],
        "p": task["p"],
        "delta": task["delta"],
        "rep": task["rep"],
        "learner": task["learner"],
        **dgp,
    }
    return key, rows, diagnostic


def checkpoint(
    output_dir: Path,
    result_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    progress: Mapping[str, Any],
) -> None:
    result_frame = pd.DataFrame(result_rows)
    diagnostic_frame = pd.DataFrame(diagnostic_rows)
    atomic_write_csv(output_dir / "raw_results.csv", result_frame)
    atomic_write_csv(output_dir / "dgp_diagnostics.csv", diagnostic_frame)
    atomic_write_json(output_dir / "progress.json", progress)


def load_existing(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_path = output_dir / "raw_results.csv"
    diagnostic_path = output_dir / "dgp_diagnostics.csv"
    result_frame = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    diagnostic_frame = (
        pd.read_csv(diagnostic_path) if diagnostic_path.exists() else pd.DataFrame()
    )
    if result_frame.empty and diagnostic_frame.empty:
        return [], []
    if (
        "task_key" not in result_frame.columns
        or "task_key" not in diagnostic_frame.columns
    ):
        raise RuntimeError(
            "Existing checkpoint files are missing task_key; choose a new output path."
        )
    result_keys = set(result_frame["task_key"].astype(str))
    diagnostic_keys = set(diagnostic_frame["task_key"].astype(str))
    if result_keys != diagnostic_keys:
        common = result_keys & diagnostic_keys
        report = {
            "reason": "interrupted multi-file checkpoint",
            "action": "rollback in memory to common task keys and rerun the remainder",
            "result_tasks_before": len(result_keys),
            "diagnostic_tasks_before": len(diagnostic_keys),
            "common_tasks_retained": len(common),
            "result_only_count": len(result_keys - diagnostic_keys),
            "diagnostic_only_count": len(diagnostic_keys - result_keys),
            "result_only_examples": sorted(result_keys - diagnostic_keys)[:10],
            "diagnostic_only_examples": sorted(diagnostic_keys - result_keys)[:10],
            "recorded_unix": time.time(),
        }
        atomic_write_json(output_dir / "resume_reconciliation.json", report)
        print(
            "Checkpoint reconciliation: retaining "
            f"{len(common)} common task(s); rerunning "
            f"{len(result_keys | diagnostic_keys) - len(common)} uncommitted task(s).",
            flush=True,
        )
        result_frame = result_frame[
            result_frame["task_key"].astype(str).isin(common)
        ].copy()
        diagnostic_frame = diagnostic_frame[
            diagnostic_frame["task_key"].astype(str).isin(common)
        ].copy()
    return result_frame.to_dict("records"), diagnostic_frame.to_dict("records")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--designs")
    parser.add_argument("--scenarios")
    parser.add_argument("--dimensions")
    parser.add_argument("--deltas")
    parser.add_argument("--learners")
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--rep-stop", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    with arguments.config.open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = json.load(stream)
    validate_config(config)
    selected_learners = parse_csv(arguments.learners)
    if arguments.device:
        config.setdefault("learner_options", {}).setdefault("dnn", {})["device"] = (
            arguments.device
        )
    if selected_learners and "dnn" in selected_learners and arguments.workers != 1:
        raise ValueError("DNN runs require --workers 1.")
    if not selected_learners and "dnn" in config["learners"] and arguments.workers != 1:
        raise ValueError(
            "A mixed run containing DNN must use --workers 1; run CPU learners separately."
        )
    if arguments.workers < 1 or arguments.checkpoint_every < 1:
        raise ValueError("workers and checkpoint-every must be positive.")

    tasks = build_tasks(
        config,
        designs=parse_csv(arguments.designs),
        scenarios=parse_csv(arguments.scenarios),
        dimensions=parse_csv(arguments.dimensions, int),
        deltas=parse_csv(arguments.deltas, float),
        learners=selected_learners,
        rep_start=arguments.rep_start,
        rep_stop=arguments.rep_stop,
    )
    if arguments.max_tasks is not None:
        if arguments.max_tasks < 1:
            raise ValueError("max-tasks must be positive.")
        tasks = tasks[: arguments.max_tasks]
    if arguments.dry_run:
        print(f"Validated {len(tasks)} task(s).")
        for task in tasks[:10]:
            print(task["task_key"])
        return 0

    output_dir = arguments.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = config_hash(config)
    requested_task_hash = task_manifest_hash(tasks)
    current_core_hash = file_hash(Path(__file__).with_name("simcore.py"))
    current_runner_hash = file_hash(Path(__file__))
    metadata_path = output_dir / "metadata.json"
    previous_metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as stream:
            previous_metadata = json.load(stream)
        if previous_metadata.get("config_hash") != digest:
            raise RuntimeError(
                "Output directory contains a different configuration; choose a new output path."
            )
        if previous_metadata.get("scientific_core_sha256") != current_core_hash:
            raise RuntimeError(
                "Output directory was created by a different scientific core; "
                "choose a new output path."
            )
        if previous_metadata.get("runner_sha256") != current_runner_hash:
            raise RuntimeError(
                "Output directory was created by a different simulation runner; "
                "choose a new output path."
            )
        previous_task_hash = previous_metadata.get("task_manifest_hash")
        if previous_task_hash is not None and previous_task_hash != requested_task_hash:
            raise RuntimeError(
                "Output directory contains a different requested task grid; "
                "choose a new output path."
            )
    invoked_unix = time.time()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": digest,
        "scientific_core_sha256": current_core_hash,
        "runner_sha256": current_runner_hash,
        "task_manifest_hash": requested_task_hash,
        "tasks_requested": len(tasks),
        "config": config,
        "config_path": str(arguments.config.resolve()),
        "command": sys.argv,
        "invocations": previous_metadata.get("invocations", []) + [sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "started_unix": previous_metadata.get("started_unix", invoked_unix),
        "last_invoked_unix": invoked_unix,
    }
    for field in ("finished_unix", "elapsed_seconds", "result_rows"):
        if field in previous_metadata:
            metadata[field] = previous_metadata[field]
    atomic_write_json(metadata_path, metadata)

    calibrations = make_calibrations(config, tasks)
    atomic_write_json(
        output_dir / "calibrations.json",
        {"calibrations": [calibrations[key].to_dict() for key in sorted(calibrations)]},
    )

    result_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    if not arguments.no_resume:
        result_rows, diagnostic_rows = load_existing(output_dir)
    completed = {str(row["task_key"]) for row in result_rows if "task_key" in row}
    requested_keys = {str(task["task_key"]) for task in tasks}
    unexpected_completed = completed - requested_keys
    if unexpected_completed:
        example = sorted(unexpected_completed)[:5]
        raise RuntimeError(
            "Output directory contains task keys outside the requested grid; "
            f"examples={example}. Choose a new output path."
        )
    if arguments.retry_failures and result_rows:
        status_by_task: dict[str, list[str]] = {}
        for row in result_rows:
            status_by_task.setdefault(str(row["task_key"]), []).append(
                str(row["status"])
            )
        retry = {
            key for key, statuses in status_by_task.items() if "failed" in statuses
        }
        completed -= retry
        result_rows = [row for row in result_rows if str(row["task_key"]) not in retry]
        diagnostic_rows = [
            row for row in diagnostic_rows if str(row["task_key"]) not in retry
        ]
    pending = [task for task in tasks if task["task_key"] not in completed]
    print(
        f"[{config['name']}] total={len(tasks)} completed={len(tasks) - len(pending)} "
        f"pending={len(pending)} workers={arguments.workers}",
        flush=True,
    )
    if not pending:
        print("Nothing to do.")
        return 0

    started = time.perf_counter()
    finished_new = 0

    def accept(result: tuple[str, list[dict[str, Any]], dict[str, Any]]) -> None:
        nonlocal finished_new
        _, rows, diagnostic = result
        result_rows.extend(rows)
        diagnostic_rows.append(diagnostic)
        finished_new += 1
        if finished_new % arguments.checkpoint_every == 0:
            progress = {
                "tasks_requested": len(tasks),
                "tasks_completed": len(tasks) - len(pending) + finished_new,
                "tasks_completed_this_process": finished_new,
                "elapsed_seconds": time.perf_counter() - started,
                "complete": False,
            }
            checkpoint(output_dir, result_rows, diagnostic_rows, progress)
            print(
                f"checkpoint {finished_new}/{len(pending)} new tasks "
                f"({progress['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    if arguments.workers == 1:
        torch.set_num_threads(1)
        for task in pending:
            calibration = calibrations[
                (str(task["design"]), int(task["p"]), str(task["scenario"]))
            ]
            accept(execute_task(task, calibration, config))
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            futures = []
            for task in pending:
                calibration = calibrations[
                    (str(task["design"]), int(task["p"]), str(task["scenario"]))
                ]
                futures.append(executor.submit(execute_task, task, calibration, config))
            for future in as_completed(futures):
                accept(future.result())

    elapsed = time.perf_counter() - started
    progress = {
        "tasks_requested": len(tasks),
        "tasks_completed": len(tasks),
        "tasks_completed_this_process": finished_new,
        "elapsed_seconds": elapsed,
        "complete": True,
    }
    checkpoint(output_dir, result_rows, diagnostic_rows, progress)
    metadata["finished_unix"] = time.time()
    metadata["elapsed_seconds"] = elapsed
    metadata["result_rows"] = len(result_rows)
    atomic_write_json(metadata_path, metadata)
    print(f"Completed {finished_new} task(s) in {elapsed:.1f}s: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
