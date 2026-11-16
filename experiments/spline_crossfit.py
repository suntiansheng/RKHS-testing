"""Compare NORD and cross-fitted NORD with additive spline regression."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, norm
from threadpoolctl import threadpool_limits
import torch
import simcore as core

STATE = {}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(config, output, fingerprint):
    STATE["thread_limit"] = threadpool_limits(limits=1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    xref = core.draw_gaussian_covariates(config["reference_n"], config["p"], config["rho"],
                  np.random.default_rng(core.stable_seed(config["seed_base"], "evaluation_reference")))
    bref, fref = core.raw_regression_components(xref)["additive"]
    STATE.update(config=config, output=Path(output), fingerprint=fingerprint,
                 xref=xref, bref=bref, fref=fref, eref=core.pooled_random_propensity(xref))


def folds(n, k, seed):
    out = np.empty(n, dtype=int)
    out[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k
    return [(np.flatnonzero(out != j), np.flatnonzero(out == j)) for j in range(k)]


def response(variant, b, f, e, d, noise, amplitude):
    if variant in ("original", "shared_null"):
        return b + amplitude*d*f + noise, b + amplitude*e*f
    if variant == "fixed_pooled_mean":
        return b + amplitude*(d-e)*f + noise, b.copy()
    raise ValueError(variant)


def score_test(score, kernel, multipliers, alpha):
    n = len(score)
    squared = max(float(score @ kernel @ score / n), 0.)
    weighted = score[:, None] * multipliers
    bootstrap = np.sqrt(np.maximum(np.sum(weighted * (kernel @ weighted), axis=0)/n, 0.))
    statistic = math.sqrt(squared)
    pvalue = (1 + np.count_nonzero(bootstrap >= statistic))/(multipliers.shape[1]+1)
    return dict(statistic=statistic, p_value=float(pvalue), reject=int(pvalue <= alpha),
                critical_value=float(np.quantile(bootstrap, 1-alpha, method="linear")))


def active_amplitudes(config, rep):
    return [a for a in config["amplitudes"]
            if rep < (config["null_replications"] if a == 0 else config["power_replications"])]


def use_five(config, n, amplitude):
    return n in config["five_fold_n"] and amplitude in config["five_fold_amplitudes"]


def expected_rows(config, n, rep):
    return sum((1 if a == 0 else len(config["dgps"])) *
               (7 if use_five(config, n, a) else 5) for a in active_amplitudes(config, rep))


def run_task(task):
    n, rep = task
    config = STATE["config"]
    path = STATE["output"] / "checkpoints" / f"n{n}_r{rep:05d}.json"
    started = time.perf_counter()
    try:
        seed = config["seed_base"]
        data_seed = core.stable_seed(seed, n, rep, "data")
        rng = np.random.default_rng(data_seed)
        x = core.draw_gaussian_covariates(n, config["p"], config["rho"], rng)
        e = core.pooled_random_propensity(x)
        d = rng.binomial(1, e)
        noise = rng.normal(0, config["sigma"], n)
        b, f = core.raw_regression_components(x)["additive"]
        kernel = core.rbf_kernel_matrix(x, math.sqrt(config["bandwidth_squared"]))
        multiplier_seed = core.stable_seed(seed, n, rep, "multipliers")
        multipliers = np.random.default_rng(multiplier_seed).standard_normal((n, config["bootstrap_draws"]))
        multipliers -= multipliers.mean(axis=0, keepdims=True)
        amplitudes = active_amplitudes(config, rep)
        partitions = {"full": [(np.arange(n), np.arange(n))],
                      "cf2": folds(n, 2, core.stable_seed(seed, n, rep, "folds", 2))}
        if any(use_five(config, n, a) for a in amplitudes):
            partitions["cf5"] = folds(n, 5, core.stable_seed(seed, n, rep, "folds", 5))
        ehat, emse, metadata = {}, {}, []
        for role, split in partitions.items():
            predicted = np.empty(n)
            error = 0.
            for j, (train, test) in enumerate(split):
                if role != "full":
                    assert not np.intersect1d(train, test).size
                fitted = core.fit_propensity(x[train], d[train], np.vstack((x[test], STATE["xref"])),
                    design="pooled_random_shift", seed=core.stable_seed(seed, n, rep, role, j, "e"),
                    clip=config["propensity_clip"], options=config["propensity_options"])
                predicted[test] = fitted.prediction[:len(test)]
                error += len(test)/n * float(np.mean((fitted.prediction[len(test):]-STATE["eref"])**2))
                metadata.append(dict(kind="propensity", role=role, fold=j, n_train=len(train),
                                     n_test=len(test), **fitted.diagnostics))
            ehat[role], emse[role] = predicted, error
        rows = []
        for a in amplitudes:
            variants = ["shared_null"] if a == 0 else config["dgps"]
            for variant in variants:
                y, truth = response(variant, b, f, e, d, noise, a)
                truth_ref = STATE["bref"] + (a*STATE["eref"]*STATE["fref"] if variant == "original" else 0.)
                oracle_score = (y-truth)*(d-e)
                common = dict(n=n, rep=rep, dgp=variant, amplitude=a, score_n=n, data_seed=data_seed,
                              multiplier_seed=multiplier_seed, alpha=config["alpha"],
                              bootstrap_draws=config["bootstrap_draws"])
                def record(role, mode, score, mu_error, e_error, selected):
                    difference = score-oracle_score
                    error_R = math.sqrt(max(float(difference @ kernel @ difference/n), 0.))
                    row = dict(common, method=role, folds=0 if role in ("full", "oracle") else int(role[2:]),
                               propensity_mode=mode, independent_mu_mse=mu_error, independent_e_mse=e_error,
                               sqrt_n_rmse_product=math.sqrt(n*mu_error*e_error), selected_alpha=selected,
                               score_error_R=error_R, **score_test(score, kernel, multipliers, config["alpha"]))
                    rows.append(row)
                record("oracle", "known", oracle_score, 0., 0., 0.)
                for role, split in partitions.items():
                    if role == "cf5" and not use_five(config, n, a):
                        continue
                    muhat = np.empty(n)
                    mu_error = 0.
                    selected = []
                    for j, (train, test) in enumerate(split):
                        fitted = core.fit_regression("gam", x[train], y[train],
                            np.vstack((x[test], STATE["xref"])),
                            seed=core.stable_seed(seed, n, rep, variant, a, role, j, "mu"),
                            learner_options=config["learner_options"])
                        muhat[test] = fitted.prediction[:len(test)]
                        mu_error += len(test)/n * float(np.mean((fitted.prediction[len(test):]-truth_ref)**2))
                        selected.append(float(fitted.diagnostics["selected_alpha"]))
                    for mode in ("estimated", "known"):
                        score = (y-muhat)*(d-(ehat[role] if mode == "estimated" else e))
                        record(role, mode, score, mu_error, emse[role] if mode == "estimated" else 0.,
                               float(np.mean(selected)))
        assert len(rows) == expected_rows(config, n, rep)
        assert all(np.isfinite(r["statistic"]) and np.isfinite(r["p_value"]) for r in rows)
        result = dict(status="ok", n=n, rep=rep, fingerprint=STATE["fingerprint"], rows=rows,
                      propensity_diagnostics=metadata, elapsed_seconds=time.perf_counter()-started)
    except Exception:
        result = dict(status="failed", n=n, rep=rep, fingerprint=STATE["fingerprint"], rows=[],
                      error=traceback.format_exc(), elapsed_seconds=time.perf_counter()-started)
    write_json(path, result)
    return {key: value for key, value in result.items()
            if key not in ("rows", "propensity_diagnostics")}


def cp(count, total, tail):
    return (0. if count == 0 else float(beta.ppf(tail, count, total-count+1)),
            1. if count == total else float(beta.ppf(1-tail, count+1, total-count)))


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values)
    out = np.empty(len(values))
    out[order] = np.minimum(1., np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return out


def summarize(output):
    output = Path(output)
    config = json.loads((output/"config.json").read_text())
    fingerprint = json.loads((output/"metadata.json").read_text())["fingerprint"]
    payloads = [json.loads(p.read_text()) for p in sorted((output/"checkpoints").glob("*.json"))]
    seen = set()
    for payload in payloads:
        task = (payload["n"], payload["rep"])
        if task in seen or payload["fingerprint"] != fingerprint:
            raise ValueError("Duplicate or incompatible spline checkpoint")
        seen.add(task)
        if (task[0] not in config["sample_sizes"] or not 0 <= task[1] <
                max(config["null_replications"], config["power_replications"])):
            raise ValueError("Checkpoint does not belong to this task grid")
        if payload["status"] == "ok" and len(payload["rows"]) != expected_rows(config, *task):
            raise ValueError("Incomplete spline checkpoint")
    failed = [p for p in payloads if p["status"] != "ok"]
    raw = pd.DataFrame([r for p in payloads for r in p["rows"]])
    if raw.empty:
        raise RuntimeError("No valid simulation rows")
    keys = ["n", "dgp", "amplitude", "method", "propensity_mode"]
    assert not raw.duplicated(keys+["rep"]).any()
    assert raw.score_n.eq(raw.n).all()
    assert raw.reject.eq(raw.p_value.le(raw.alpha).astype(int)).all()
    records = []
    for key, group in raw.groupby(keys, sort=True):
        count, rejected = len(group), int(group.reject.sum())
        lo, hi = cp(rejected, count, .025)
        row = dict(zip(keys, key))
        target = config["null_replications"] if row["amplitude"] == 0 else config["power_replications"]
        row.update(replications=count, target_replications=target, complete=count==target,
                   rejections=rejected, rejection_rate=rejected/count, ci_lower=lo, ci_upper=hi)
        for name in ("independent_mu_mse", "independent_e_mse", "sqrt_n_rmse_product", "score_error_R", "selected_alpha"):
            row[name] = float(group[name].mean())
        records.append(row)
    summary = pd.DataFrame(records)
    summary.to_csv(output/"summary.csv", index=False)
    pairs = []
    for key, group in raw[raw.method.ne("oracle")].groupby(["n", "dgp", "amplitude", "propensity_mode"]):
        wide = group.pivot(index="rep", columns="method", values="reject")
        for method in ("cf2", "cf5"):
            if method not in wide:
                continue
            aligned = wide[["full", method]].dropna()
            difference = aligned.full-aligned[method]
            count = len(difference)
            b, c = int(difference.eq(1).sum()), int(difference.eq(-1).sum())
            se = float(difference.std(ddof=1)/math.sqrt(count))
            n, dgp, a, mode = key
            family = "null_difference" if a == 0 else (
                "five_fold_sensitivity" if method == "cf5" else
                ("primary_estimated_cf2" if mode == "estimated" else "known_propensity_cf2")
                if a in config["primary_amplitudes"] else "extra_cf2_sensitivity")
            pairs.append(dict(n=n, dgp=dgp, amplitude=a, propensity_mode=mode, comparator=method,
                family=family, replications=count, nord_power=float(aligned.full.mean()),
                cf_power=float(aligned[method].mean()), difference=float(difference.mean()), paired_se=se,
                pointwise_lower=float(difference.mean()-norm.ppf(.975)*se),
                pointwise_upper=float(difference.mean()+norm.ppf(.975)*se), nord_only=b, cf_only=c,
                exact_mcnemar_p=binomtest(b,b+c,.5).pvalue if b+c else 1.))
    paired = pd.DataFrame(pairs)
    for family, group in paired.groupby("family"):
        m = len(group)
        adjusted = holm(group.exact_mcnemar_p)
        for index, adj in zip(group.index, adjusted):
            row = paired.loc[index]
            lb, ub = cp(int(row.nord_only), int(row.replications), .05/(4*m))
            lc, uc = cp(int(row.cf_only), int(row.replications), .05/(4*m))
            paired.loc[index, "simultaneous_lower"] = lb-uc
            paired.loc[index, "simultaneous_upper"] = ub-lc
            paired.loc[index, "holm_mcnemar_p"] = adj
            paired.loc[index, "family_cells"] = m
    paired.to_csv(output/"paired_differences.csv", index=False)
    null = summary[summary.amplitude.eq(0)].copy()
    null["exact_binomial_p"] = [binomtest(int(r.rejections),int(r.replications),config["alpha"]).pvalue for r in null.itertuples()]
    null["holm_p"] = holm(null.exact_binomial_p)
    null["direction"] = np.where(null.rejection_rate.gt(config["alpha"]), "anti-conservative", "conservative")
    null.to_csv(output/"null_size.csv", index=False)
    raw.to_csv(output/"raw_results.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    write_json(output/"failures.json", failed)
    expected = len(config["sample_sizes"])*max(config["null_replications"],config["power_replications"])
    validation = dict(complete=len(payloads)==expected and not failed and bool(summary.complete.all()),
                      completed_tasks=len(payloads), expected_tasks=expected, failures=len(failed), rows=len(raw),
                      all_n_scores_used=True, duplicate_rows=0, null_cells=len(null),
                      flagged_null_cells=int(null.holm_p.le(.05).sum()),
                      primary_cells=int(paired.family.eq("primary_estimated_cf2").sum()),
                      primary_positive_lower_bounds=int((paired.family.eq("primary_estimated_cf2") & paired.simultaneous_lower.gt(0)).sum()))
    write_json(output/"validation.json", validation)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "simulation/configs/spline_crossfit_confirmation.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output.resolve()
    if args.summarize_only:
        print(json.dumps(summarize(output)), flush=True)
        return
    config = json.loads(args.config.read_text())
    if (not config["sample_sizes"] or min(config["sample_sizes"]) < 10
            or min(config["null_replications"], config["power_replications"],
                   config["bootstrap_draws"], config["reference_n"]) < 1
            or config["learner"] != "gam" or not 0 < config["alpha"] < 1
            or not 0 < config["propensity_clip"] < .5
            or config["bandwidth_squared"] <= 0
            or set(config["dgps"]) != {"original", "fixed_pooled_mean"}
            or len(set(config["amplitudes"])) != len(config["amplitudes"])
            or not np.isfinite(config["amplitudes"]).all()
            or min(config["amplitudes"]) < 0):
        parser.error("Invalid spline experiment configuration")
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(config=config, source={str(p.relative_to(ROOT)):sha(p) for p in
                     (Path(__file__), ROOT/"simulation/simcore.py", ROOT/"simulation/configs/main.json")},
                    packages=core.package_versions())
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    metadata = output/"metadata.json"
    if metadata.exists():
        if json.loads(metadata.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Source, configuration or software changed; use a new output directory")
    else:
        write_json(metadata,dict(identity=identity,fingerprint=fingerprint,started_unix=time.time()))
    write_json(output/"config.json",config)
    tasks = [(n,rep) for rep in range(max(config["null_replications"],config["power_replications"]))
             for n in config["sample_sizes"]]
    results = {}
    for task in tasks:
        path = output/"checkpoints"/f"n{task[0]}_r{task[1]:05d}.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value["fingerprint"] != fingerprint:
                raise ValueError(f"Incompatible checkpoint: {path}")
            if value["status"] == "ok" and len(value["rows"]) != expected_rows(config,*task):
                raise ValueError(f"Incomplete checkpoint: {path}")
            results[task] = {key: item for key, item in value.items()
                             if key not in ("rows", "propensity_diagnostics")}
    pending = [task for task in tasks if task not in results]
    started = time.perf_counter()
    initial = len(results)

    def status(complete=False):
        elapsed = time.perf_counter()-started
        new = len(results)-initial
        value = dict(
            complete=complete, requested=len(tasks), completed=len(results),
            failed=sum(result["status"] != "ok" for result in results.values()),
            elapsed_seconds=elapsed,
            eta_seconds=elapsed/new*(len(tasks)-len(results)) if new else None,
            updated_unix=time.time(),
        )
        write_json(output/"progress.json", value)
        print(json.dumps(value), flush=True)

    status()
    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize,
            initargs=(config, str(output), fingerprint), mp_context=mp.get_context("spawn"),
        ) as pool:
            iterator = iter(pending)
            first_tasks = [next(iterator) for _ in range(min(len(pending), args.workers*2))]
            futures = {pool.submit(run_task, task): task for task in first_tasks}
            last = time.perf_counter()
            while futures:
                done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    results[task] = future.result()
                    try:
                        task = next(iterator)
                    except StopIteration:
                        continue
                    futures[pool.submit(run_task, task)] = task
                if time.perf_counter()-last >= 10:
                    status()
                    last = time.perf_counter()
    status(complete=True)
    print(json.dumps(summarize(output)), flush=True)


if __name__ == "__main__":
    main()
