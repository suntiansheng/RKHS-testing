"""Replay the published grid, adding GCM to exactly matched full-sample scores.

No source simulation function, learner specification, or archived result is
changed. NORD is recalculated as a reproduction check. Its nuisance predictions
are shared with GCM. Checkpoints contain sufficient score moments for GCM.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
import simcore as core
from gcm import gcm_test
from io_utils import atomic_write_csv, atomic_write_json, file_hash
from summarize import size_audit, attach_size_gate, wilson_interval

SCENARIOS = ("additive", "product", "composition", "neural_composition")
LEARNERS = ("gam", "rf", "lasso", "dnn")
STATE = {}


def fit_seed(config, p, scenario, amplitude, rep, learner):
    return core.stable_seed(config["seed_base"], "pooled_random_shift", p,
                            scenario, float(amplitude), rep, learner, "fit")


def initialize(config, calibrations, output, fingerprint):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    STATE.update(config=config, calibrations=calibrations, output=Path(output),
                 fingerprint=fingerprint, thread_limit=threadpool_limits(limits=1))


def key_for(p, rep, phase):
    return f"p{p}_r{rep:04d}_{phase}"


def run_job(job):
    p, rep, phase = job
    config, output = STATE["config"], STATE["output"]
    key = key_for(*job)
    started = time.monotonic()
    seed = core.stable_seed(config["seed_base"], "pooled_random_shift", p, rep, "dgp")
    rng = np.random.default_rng(seed)
    x = core.draw_gaussian_covariates(config["n"], p, config["rho"], rng)
    e = core.pooled_random_propensity(x)
    d = rng.binomial(1, e, size=config["n"]).astype(int)
    noise = rng.normal(0, config["sigma"], config["n"])
    h = core.median_bandwidth(x)
    kernel = core.rbf_kernel_matrix(x, h)
    # LogisticRegressionCV uses deterministic, unshuffled folds and lbfgs.
    # Its random_state has no effect; equality across seeds is unit-tested.
    eseed = core.stable_seed(fit_seed(config, p, "additive", 0., rep, "gam"), "pooled_e")
    propensity = core.fit_propensity(x, d, x, design="pooled_random_shift",
        seed=eseed, clip=config["propensity_clip"], options=config["propensity_options"])
    ehat = propensity.prediction
    amplitudes = [a for a in config["deltas"] if (a == 0) == (phase == "null")]
    rows = []
    for scenario in SCENARIOS:
        calibration = core.ScenarioCalibration(**STATE["calibrations"][str(p)][scenario])
        base, signal = core.evaluate_regression_structure(x, scenario, calibration)
        for amplitude in amplitudes:
            y = base + d * amplitude * signal + noise
            truth = base + e * amplitude * signal
            bootstrap_seed = core.stable_seed(config["seed_base"], "pooled_random_shift",
                p, scenario, float(amplitude), rep, "bootstrap")
            for learner in LEARNERS:
                fseed = fit_seed(config, p, scenario, amplitude, rep, learner)
                fitted = core.fit_regression(learner, x, y, x,
                    seed=core.stable_seed(fseed, "pooled_mu"),
                    learner_options=config["learner_options"][learner])
                scores = (y - fitted.prediction) * (d - ehat)
                gcm = gcm_test(scores, alpha=config["alpha"])
                nord = core.multiplier_test(scores, kernel,
                    bootstrap_draws=config["bootstrap_draws"], alpha=config["alpha"],
                    seed=bootstrap_seed)
                rows.append(dict(design="pooled_random_shift", scenario=scenario,
                    p=p, n=config["n"], delta=amplitude, rep=rep, learner=learner,
                    method="gcm", alpha=config["alpha"], status="ok", n1=int(d.sum()),
                    dgp_seed=seed, fit_seed=fseed, bootstrap_seed=bootstrap_seed,
                    score_n=len(scores), score_mean=float(scores.mean()),
                    score_sd=float(scores.std(ddof=1)), **gcm,
                    nord_p_value=nord.p_value, nord_reject=nord.reject,
                    nord_statistic=nord.statistic,
                    mu_mse=float(np.mean((fitted.prediction - truth) ** 2)),
                    propensity_mse=float(np.mean((ehat - e) ** 2)),
                    outcome_fit_seconds=fitted.diagnostics["fit_seconds"],
                    selected_alpha=fitted.diagnostics.get("selected_alpha", np.nan),
                    selected_epoch=fitted.diagnostics.get("selected_epoch", np.nan)))
    path = output / "shards" / f"{key}.csv.gz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    os.replace(temporary, path)
    meta = dict(key=key, rows=len(rows), seconds=time.monotonic()-started,
                fingerprint=STATE["fingerprint"], sha256=file_hash(path))
    atomic_write_json(output / "checkpoints" / f"{key}.json", meta)
    return meta


def summarize(output, config):
    paths = sorted((output / "shards").glob("*.csv.gz"))
    if not paths:
        return
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["design", "scenario", "p", "delta", "learner", "method", "alpha"]
    assert not raw.duplicated(keys + ["rep"]).any()
    records = []
    for values, group in raw.groupby(keys, sort=True):
        count, hits = len(group), int(group.reject.sum())
        lower, upper = wilson_interval(hits, count)
        records.append(dict(zip(keys, values)) | dict(n_valid=count, n_failed=0,
            rejection_rate=hits/count, monte_carlo_se=np.sqrt(hits/count*(1-hits/count)/count),
            wilson_lower=lower, wilson_upper=upper,
            mean_mu_mse=group.mu_mse.mean(), mean_propensity_mse=group.propensity_mse.mean(),
            mean_fit_seconds=group.outcome_fit_seconds.mean(),
            paired_nord_rejection_rate=group.nord_reject.mean()))
    summary = pd.DataFrame(records)
    null = summary[summary.delta.eq(0)]
    complete_null = len(null) == 32 and null.n_valid.eq(300).all()
    sizes = size_audit(summary, family="gcm_32_cells" if complete_null else "incomplete_gcm")
    if not complete_null:
        sizes["holm_flag_0_05"] = np.nan
    summary = attach_size_gate(summary, sizes)
    old = pd.read_csv(ROOT / "results/frozen_summary.csv")
    old = old[old.method.eq("orthogonal_nosplit") & old.scenario.isin(SCENARIOS)]
    compare = summary.merge(old[["scenario", "p", "delta", "learner", "rejection_rate",
        "mean_mu_mse", "mean_propensity_mse"]], on=["scenario", "p", "delta", "learner"],
        suffixes=("", "_archived"), validate="one_to_one")
    compare["nord_rejection_rate_difference"] = (compare.paired_nord_rejection_rate
                                                - compare.rejection_rate_archived)
    compare["mu_mse_difference"] = compare.mean_mu_mse - compare.mean_mu_mse_archived
    compare["propensity_mse_difference"] = compare.mean_propensity_mse - compare.mean_propensity_mse_archived
    expected = len(SCENARIOS)*2*len(LEARNERS)*len(config["deltas"])*300
    complete = len(raw) == expected and summary.n_valid.eq(300).all() and complete_null
    raw.sort_values(keys+["rep"]).to_csv(output / "raw.csv.gz", index=False, compression="gzip")
    atomic_write_csv(output / "summary.csv", summary)
    atomic_write_csv(output / "null_size.csv", sizes)
    atomic_write_csv(output / "nord_reproduction.csv", compare)
    atomic_write_json(output / "validation.json", dict(complete=bool(complete),
        complete_null=bool(complete_null), rows=len(raw), expected_rows=expected,
        cells=len(summary), null_cells=len(null),
        max_nord_rejection_rate_difference=float(compare.nord_rejection_rate_difference.abs().max()),
        max_mu_mse_difference=float(compare.mu_mse_difference.abs().max()),
        max_propensity_mse_difference=float(compare.propensity_mse_difference.abs().max()),
        source_files={path.name:file_hash(path) for path in
            [output/"raw.csv.gz", output/"summary.csv", output/"null_size.csv"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/gcm_comparison")
    parser.add_argument("--phase", choices=["null", "power", "all"], default="all")
    parser.add_argument("--rep-stop", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.rep_stop <= 300 or args.workers < 1:
        parser.error("Require 1 <= rep-stop <= 300 and workers >= 1")
    config = json.loads((ROOT / "simulation/configs/main.json").read_text())
    config["scenarios"], config["learners"] = list(SCENARIOS), list(LEARNERS)
    config["methods"] = ["gcm", "orthogonal_nosplit"]
    config["learner_options"]["dnn"]["device"] = "cpu"
    config["gcm_calibration"] = "two-sided standard normal; equation (3), Shah and Peters (2020)"
    config["null_screen"] = "two-sided exact binomial tests with Holm adjustment across 32 GCM cells"
    config["reporting"] = "Empirical rejection rates for all reported designs, procedures and learners"
    config["source_sha256"] = {str(path.relative_to(ROOT)):file_hash(path) for path in
        [Path(__file__), ROOT/"simulation/gcm.py", ROOT/"simulation/simcore.py",
         ROOT/"simulation/configs/main.json", ROOT/"results/frozen_summary.csv"]}
    output = args.output
    for directory in [output, output/"shards", output/"checkpoints"]:
        directory.mkdir(parents=True, exist_ok=True)
    if (output/"protocol.json").exists():
        if json.loads((output/"protocol.json").read_text()) != config:
            raise ValueError("Protocol or source changed; use a separate output directory")
    else:
        atomic_write_json(output/"protocol.json", config)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.summarize_only:
        summarize(output, config)
        return
    calibrations = {}
    for p in config["dimensions"]:
        calculated = core.calibrate_scenarios(p=p, rho=config["rho"], design="pooled_random_shift",
            scenarios=SCENARIOS, calibration_n=config["calibration_n"],
            seed=core.stable_seed(config["seed_base"], "equal", p, "calibration"))
        calibrations[str(p)] = {name:cal.to_dict() for name,cal in calculated.items()}
    atomic_write_json(output/"calibrations.json", calibrations)
    phases = ["null", "power"] if args.phase == "all" else [args.phase]
    jobs = [(p, rep, phase) for phase in phases for rep in range(1,args.rep_stop+1)
            for p in config["dimensions"]]
    pending = []
    cached = 0
    for job in jobs:
        key = key_for(*job)
        checkpoint, shard = output/"checkpoints"/f"{key}.json", output/"shards"/f"{key}.csv.gz"
        if checkpoint.exists() and shard.exists():
            meta = json.loads(checkpoint.read_text())
            if meta["fingerprint"] != fingerprint or file_hash(shard) != meta["sha256"]:
                raise ValueError(f"Checkpoint integrity failure: {key}")
            cached += 1
        else:
            pending.append(job)
    started, finished, new_rows = time.monotonic(), 0, 0
    print(f"Frozen GCM grid: {len(jobs)} requested jobs; {cached} cached; {len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=initialize, initargs=(config, calibrations, str(output), fingerprint)) as executor:
        futures = [executor.submit(run_job, job) for job in pending]
        for future in as_completed(futures):
            meta = future.result()
            finished += 1
            new_rows += meta["rows"]
            elapsed = time.monotonic() - started
            progress = dict(phase=args.phase, completed_jobs=cached+finished,
                requested_jobs=len(jobs), new_rows=new_rows, elapsed_seconds=elapsed,
                rows_per_second=new_rows/elapsed, last_job=meta["key"])
            atomic_write_json(output/"progress.json", progress)
            if finished <= 2 or finished % 10 == 0 or finished == len(pending):
                print(json.dumps(progress), flush=True)
    summarize(output, config)
    print((output/"validation.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
