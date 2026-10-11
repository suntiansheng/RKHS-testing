"""Resumable four-learner full-sample/cross-fit experiment.

Uses the archived data-generating functions and nuisance learners, independent
outer folds, a fixed kernel, and independent prediction-error diagnostics.
Run from the repository root. Process workers each use one numerical thread.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
import simcore as core
from io_utils import atomic_write_csv, atomic_write_json, file_hash

LEARNERS = ("gam", "rf", "lasso", "dnn")
DEFAULT_AMPLITUDES = (0., .8, 1.2, 1.6, 2., 2.5, 3.2, 4., 6.)
STATE = {}


def independent_folds(n, k, seed):
    if not 2 <= k <= n:
        raise ValueError("Require 2 <= number of folds <= sample size")
    assignment = np.empty(n, dtype=int)
    assignment[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k
    return assignment


def score_test(scores, kernel, multipliers, level):
    """Squared-norm implementation of the existing centered multiplier test."""
    n = len(scores)
    observed_squared = max(float(scores @ kernel @ scores / n), 0.)
    weighted = scores[:, None] * multipliers
    bootstrap_squared = np.maximum(np.sum(weighted * (kernel @ weighted), axis=0) / n, 0.)
    pvalue = (1 + np.count_nonzero(bootstrap_squared >= observed_squared)) / (multipliers.shape[1] + 1)
    return {
        "reject": int(pvalue <= level), "p_value": float(pvalue),
        "statistic": math.sqrt(observed_squared),
        "critical_value": float(np.quantile(np.sqrt(bootstrap_squared), 1-level, method="linear")),
    }


def score_errors(error, kernel):
    n = len(error)
    r_squared = max(float(error @ kernel @ error / n), 0.)
    energy = float(np.mean(error**2 * np.diag(kernel)))
    return {"score_error_R": math.sqrt(r_squared), "score_error_energy": energy,
            "score_error_Q": max(energy-r_squared/n, 0.)}


def initialize_worker(config, output, fingerprint):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    STATE["thread_limit"] = threadpool_limits(limits=1)
    xref = core.draw_gaussian_covariates(
        config["reference_n"], config["p"], config["rho"],
        np.random.default_rng(core.stable_seed(config["seed"], "diagnostic_reference")),
    )
    bref, fref = core.raw_regression_components(xref)["additive"]
    STATE.update(config=config, output=Path(output), fingerprint=fingerprint,
                 xref=xref, bref=bref, fref=fref, eref=core.pooled_random_propensity(xref))


def fold_counts(config, n):
    return [2, 5] if config["five_fold_sensitivity"] and n == 200 else [2]


def task_amplitudes(config, rep):
    return [a for a in config["amplitudes"]
            if rep < (config["null_replications"] if a == 0 else config["power_replications"])]


def task_key(n, rep):
    return f"n{n}_r{rep:05d}"


def expected_rows(config, n, rep):
    methods = 1 + 2*len(config["learners"])*(1+len(fold_counts(config, n)))
    return methods * len(task_amplitudes(config, rep))


def checkpoint_rows(output, key, rows, meta):
    path = output / "shards" / f"{key}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    os.replace(temporary, path)
    meta.update(rows=len(rows), shard_bytes=path.stat().st_size, shard_sha256=file_hash(path))
    atomic_write_json(output / "checkpoints" / f"{key}.json", meta)


def run_task(task):
    n, rep = task
    config, seed, output = STATE["config"], STATE["config"]["seed"], STATE["output"]
    started = time.perf_counter()
    p = config["p"]
    rng = np.random.default_rng(core.stable_seed(seed, n, rep, "data"))
    x = core.draw_gaussian_covariates(n, p, config["rho"], rng)
    e = core.pooled_random_propensity(x)
    d = rng.binomial(1, e)
    noise = rng.normal(0, config["sigma"], size=n)
    b, f = core.raw_regression_components(x)["additive"]
    kernel = core.rbf_kernel_matrix(x, math.sqrt(config["bandwidth_squared"]))
    multiplier_seed = core.stable_seed(seed, n, rep, "multipliers")
    multipliers = np.random.default_rng(multiplier_seed).standard_normal((n, config["bootstrap_draws"]))
    multipliers -= multipliers.mean(axis=0, keepdims=True)
    partitions = {"full": [(np.arange(n), np.arange(n))]}
    for k in fold_counts(config, n):
        fold = independent_folds(n, k, core.stable_seed(seed, n, rep, "folds", k))
        partitions[f"cf{k}"] = [(np.flatnonzero(fold != j), np.flatnonzero(fold == j)) for j in range(k)]

    # These propensity fits do not depend on outcome learner or amplitude.
    ehat, emse, eseconds = {}, {}, {}
    for role, splits in partitions.items():
        ehat[role], emse[role] = np.empty(n), 0.
        start = time.perf_counter()
        for j, (train, test) in enumerate(splits):
            fitted = core.fit_propensity(
                x[train], d[train], np.vstack((x[test], STATE["xref"])),
                design="pooled_random_shift", seed=core.stable_seed(seed, n, rep, role, j, "e"),
                clip=config["propensity_clip"], options=config["propensity_options"],
            )
            ehat[role][test] = fitted.prediction[:len(test)]
            emse[role] += len(test)/n * float(np.mean((fitted.prediction[len(test):]-STATE["eref"])**2))
        eseconds[role] = time.perf_counter()-start

    rows = []
    fitting_diagnostics = []
    for amplitude in task_amplitudes(config, rep):
        y = b + amplitude*d*f + noise
        truth = b + amplitude*e*f
        truth_ref = STATE["bref"] + amplitude*STATE["eref"]*STATE["fref"]
        oracle_score = (y-truth)*(d-e)
        common = {"n": n, "p": p, "scenario": "additive", "amplitude": amplitude,
                  "rep": rep, "alpha": config["alpha"], "n1": int(d.sum()),
                  "bootstrap_draws": config["bootstrap_draws"], "score_n": n}

        def record(learner, role, mode, score, mu_error, e_error, fit_seconds, selected):
            k = int(role[2:]) if role.startswith("cf") else 0
            row = dict(common, learner=learner, method=role, folds=k, propensity_mode=mode,
                       independent_mu_mse=mu_error, independent_e_mse=e_error,
                       sqrt_n_rmse_product=math.sqrt(n*mu_error*e_error),
                       fitting_seconds=fit_seconds, selected_parameter=selected,
                       **score_test(score, kernel, multipliers, config["alpha"]),
                       **score_errors(score-oracle_score, kernel))
            rows.append(row)

        record("oracle", "oracle", "known", oracle_score, 0., 0., 0., float("nan"))
        for learner in config["learners"]:
            for role, splits in partitions.items():
                muhat, mu_error, selected_values = np.empty(n), 0., []
                start = time.perf_counter()
                for j, (train, test) in enumerate(splits):
                    fitted = core.fit_regression(
                        learner, x[train], y[train], np.vstack((x[test], STATE["xref"])),
                        seed=core.stable_seed(seed, n, rep, amplitude, learner, role, j, "mu"),
                        learner_options=config["learner_options"][learner],
                    )
                    muhat[test] = fitted.prediction[:len(test)]
                    mu_error += len(test)/n * float(np.mean((fitted.prediction[len(test):]-truth_ref)**2))
                    value = fitted.diagnostics.get("selected_alpha", fitted.diagnostics.get("selected_epoch"))
                    if value is not None:
                        selected_values.append(float(value))
                    fitting_diagnostics.append({"amplitude": amplitude, "learner": learner,
                                                "method": role, "fold": j, "n_train": len(train),
                                                "n_test": len(test), **fitted.diagnostics})
                mu_seconds = time.perf_counter()-start
                selected = float(np.mean(selected_values)) if selected_values else float("nan")
                for mode in ("estimated", "known"):
                    estimated = mode == "estimated"
                    score = (y-muhat)*(d-(ehat[role] if estimated else e))
                    record(learner, role, mode, score, mu_error,
                           emse[role] if estimated else 0.,
                           mu_seconds+(eseconds[role] if estimated else 0.), selected)
    key = task_key(n, rep)
    if len(rows) != expected_rows(config, n, rep):
        raise RuntimeError(f"Unexpected result count for {key}")
    elapsed = time.perf_counter()-started
    meta = {"n": n, "rep": rep, "fingerprint": STATE["fingerprint"],
            "elapsed_seconds": elapsed, "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fitting_diagnostics": fitting_diagnostics}
    checkpoint_rows(output, key, rows, meta)
    return {"n": n, "rep": rep, "rows": len(rows), "elapsed_seconds": elapsed}


def read_completed(output, config, fingerprint):
    completed = {}
    for path in sorted((output / "checkpoints").glob("*.json")):
        meta = json.loads(path.read_text())
        if meta["fingerprint"] != fingerprint:
            raise ValueError(f"Incompatible checkpoint: {path}")
        key = task_key(meta["n"], meta["rep"])
        shard = output / "shards" / f"{key}.csv.gz"
        if (shard.exists() and shard.stat().st_size == meta["shard_bytes"]
                and file_hash(shard) == meta["shard_sha256"]
                and meta["rows"] == expected_rows(config, meta["n"], meta["rep"])):
            completed[(meta["n"], meta["rep"])] = meta
    return completed


def wilson(successes, count):
    z = norm.ppf(.975)
    rate = successes/count
    denominator = 1+z*z/count
    center = (rate+z*z/(2*count))/denominator
    half = z*np.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/denominator
    return float(center-half), float(center+half)


def summarize(output, write_raw=False):
    metadata = json.loads((output / "config.json").read_text())
    config = metadata["config"]
    completed = read_completed(output, config, metadata["fingerprint"])
    if not completed:
        raise ValueError("No complete simulation tasks to summarize")
    frames = [pd.read_csv(output / "shards" / f"{task_key(*task)}.csv.gz") for task in sorted(completed)]
    raw = pd.concat(frames, ignore_index=True)
    keys = ["n", "p", "scenario", "learner", "method", "folds", "propensity_mode", "amplitude", "alpha"]
    if raw.duplicated(keys+["rep"]).any():
        raise ValueError("Duplicate replication rows")
    numeric_required = ["p_value", "statistic", "critical_value", "independent_mu_mse", "independent_e_mse",
                        "score_error_R", "score_error_Q", "score_error_energy"]
    if not np.isfinite(raw[numeric_required]).all().all():
        raise ValueError("Nonfinite simulation result")
    if not (raw.reject == (raw.p_value <= raw.alpha).astype(int)).all():
        raise ValueError("Inconsistent rejection decisions")
    if not (raw.score_n == raw.n).all():
        raise ValueError("The pooled cross-fit test must use all n observations")
    records = []
    for values, group in raw.groupby(keys, sort=True):
        count, successes = len(group), int(group.reject.sum())
        lower, upper = wilson(successes, count)
        row = dict(zip(keys, values))
        target = config["null_replications"] if row["amplitude"] == 0 else config["power_replications"]
        row.update(replications=count, target_replications=target, complete=count==target,
                   rejections=successes, rejection_rate=successes/count,
                   mc_se=math.sqrt(successes/count*(1-successes/count)/count),
                   wilson_lower=lower, wilson_upper=upper)
        for column in ["independent_mu_mse", "independent_e_mse", "sqrt_n_rmse_product", "score_error_R",
                       "score_error_Q", "score_error_energy", "fitting_seconds", "selected_parameter"]:
            row[column] = float(group[column].mean())
        records.append(row)
    summary = pd.DataFrame(records)
    atomic_write_csv(output / "summary.csv", summary)

    pairs = []
    feasible = raw[raw.learner != "oracle"]
    pair_keys = ["n", "learner", "propensity_mode", "amplitude"]
    for values, group in feasible.groupby(pair_keys, sort=True):
        wide = group.pivot(index="rep", columns="method", values="reject")
        for method in [name for name in wide if name.startswith("cf")]:
            selected = wide[["full", method]].dropna()
            differences = selected["full"]-selected[method]
            count = len(differences)
            se = float(differences.std(ddof=1)/np.sqrt(count)) if count>1 else float("nan")
            lower, upper = differences.mean()-norm.ppf(.975)*se, differences.mean()+norm.ppf(.975)*se
            pairs.append(dict(zip(pair_keys, values)) | {
                "folds": int(method[2:]), "replications": count, "full_power": float(selected.full.mean()),
                "crossfit_power": float(selected[method].mean()), "full_minus_crossfit": float(differences.mean()),
                "paired_mc_se": se, "ci_lower": lower, "ci_upper": upper,
                "full_only_rejections": int((differences == 1).sum()),
                "crossfit_only_rejections": int((differences == -1).sum()),
            })
    atomic_write_csv(output / "paired_differences.csv", pd.DataFrame(pairs))
    null = summary[summary.amplitude == 0].copy()
    null["binomial_p_value"] = [binomtest(int(s), int(r), config["alpha"]).pvalue
                                for s, r in zip(null.rejections, null.replications)]
    expected_null_cells = sum(1+2*len(config["learners"])*(1+len(fold_counts(config,n)))
                              for n in config["sample_sizes"])
    family_complete = len(null)==expected_null_cells and bool(null.complete.all())
    null["holm_adjusted_p_value"] = np.nan
    null["size_screen"] = "not_evaluated"
    if family_complete:
        order = np.argsort(null.binomial_p_value.to_numpy())
        adjusted = np.minimum(1., np.maximum.accumulate(
            null.binomial_p_value.to_numpy()[order] * (len(null)-np.arange(len(null)))))
        null.iloc[order, null.columns.get_loc("holm_adjusted_p_value")] = adjusted
        null["size_screen"] = np.where(null.holm_adjusted_p_value <= .05, "flagged", "not_flagged")
    atomic_write_csv(output / "null_size.csv", null)
    expected_tasks = len(config["sample_sizes"])*max(config["null_replications"],config["power_replications"])
    audit = {"completed_tasks": len(completed), "expected_tasks": expected_tasks,
             "rows": len(raw), "complete": len(completed)==expected_tasks and bool(summary.complete.all()),
             "null_family_complete": family_complete, "null_family_cells": expected_null_cells,
             "null_cells_flagged": int((null.size_screen=="flagged").sum()),
             "worker_cpu_seconds": float(sum(m["elapsed_seconds"] for m in completed.values()))}
    atomic_write_json(output / "validation.json", audit)
    if write_raw:
        raw.to_csv(output / "raw.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    return audit


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-sizes", default="200,400")
    parser.add_argument("--learners", default=",".join(LEARNERS))
    parser.add_argument("--amplitudes", default=",".join(map(str, DEFAULT_AMPLITUDES)))
    parser.add_argument("--null-replications", type=int, default=2000)
    parser.add_argument("--power-replications", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=999)
    parser.add_argument("--reference-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026100117)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--five-fold-sensitivity", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--write-raw", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = args.output.resolve()
    if args.summarize_only:
        print(json.dumps(summarize(output, args.write_raw), indent=2), flush=True)
        return
    old = json.loads((ROOT / "simulation/configs/main.json").read_text())
    old["learner_options"]["dnn"]["device"] = "cpu"
    config = {"schema":1, "sample_sizes":sorted(set(map(int,args.sample_sizes.split(",")))),
              "learners":args.learners.split(","), "amplitudes":sorted(set(map(float,args.amplitudes.split(",")))),
              "null_replications":args.null_replications,"power_replications":args.power_replications,
              "bootstrap_draws":args.bootstrap_draws, "reference_n":args.reference_n, "seed":args.seed,
              "five_fold_sensitivity":args.five_fold_sensitivity, "p":20,"rho":.5,"sigma":1.,
              "bandwidth_squared":40.,"alpha":.05,"propensity_clip":.01,
              "learner_options":old["learner_options"],"propensity_options":old["propensity_options"],
              "fold_scheme":"independent balanced random permutation",
              "propensity_modes":["estimated","known"]}
    if (min(config["sample_sizes"]) < 40 or min(args.null_replications,args.power_replications,args.bootstrap_draws,args.workers) < 1
            or args.reference_n < 100 or set(config["learners"])-set(LEARNERS)
            or len(config["learners"]) != len(set(config["learners"])) or 0. not in config["amplitudes"]):
        raise ValueError("Invalid experiment configuration")
    sources = {"driver":file_hash(Path(__file__)), "simcore":file_hash(ROOT/"simulation/simcore.py")}
    fingerprint = hashlib.sha256(json.dumps({"config":config,"sources":sources},sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    metadata = {"config":config,"source_hashes":sources,"fingerprint":fingerprint,
                "package_versions":core.package_versions(), "created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    if (output/"config.json").exists():
        existing=json.loads((output/"config.json").read_text())
        if existing["fingerprint"] != fingerprint:
            raise ValueError("Configuration or source changed; use a separate output directory")
    else:
        atomic_write_json(output/"config.json",metadata)
    completed=read_completed(output,config,fingerprint)
    # Replication-first ordering gives complete curves for all sample sizes early.
    tasks=[(n,rep) for rep in range(max(args.null_replications,args.power_replications))
           for n in config["sample_sizes"] if (n,rep) not in completed]
    if args.max_tasks is not None:
        tasks=tasks[:args.max_tasks]
    print(f"Completed checkpoints: {len(completed)}; pending this launch: {len(tasks)}; workers: {args.workers}",flush=True)
    started=time.perf_counter()
    finished, failed, last_report=0,0,0.
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker,initargs=(config,str(output),fingerprint)) as pool:
        futures={pool.submit(run_task,task):task for task in tasks}
        for future in as_completed(futures):
            task=futures[future]
            try:
                future.result()
                finished+=1
            except Exception:
                failed+=1
                detail=traceback.format_exc()
                atomic_write_json(output/"failures"/f"{task_key(*task)}.json",{"task":task,"traceback":detail})
                print(f"FAILED {task_key(*task)}: {detail}",flush=True)
            elapsed=time.perf_counter()-started
            if elapsed-last_report>=30 or finished+failed==len(tasks):
                status={"finished_this_launch":finished,"failed_this_launch":failed,
                        "resumed_tasks":len(completed),"pending_this_launch":len(tasks)-finished-failed,
                        "elapsed_seconds":elapsed,"workers":args.workers,
                        "updated_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
                atomic_write_json(output/"progress.json",status)
                print(json.dumps(status),flush=True)
                last_report=elapsed
    print(json.dumps(summarize(output,args.write_raw),indent=2),flush=True)
    if failed:
        raise SystemExit(f"{failed} tasks failed; inspect failures and rerun to retry")


if __name__ == "__main__":
    main()
