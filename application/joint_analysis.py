#!/usr/bin/env python3
"""Matched NORD/CF-NORD comparisons with baseline and baseline-plus-text covariates.

Samples and features follow the fixed manuscript specifications. Baseline
predictions can be reused from a completed companion run or fitted afresh.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
import sklearn

import analyze
import baseline_analysis as baseline
import core


VERSION = "casino-baseline-plus-text-v1"
SETS = ("baseline", "baseline_text")
OUTCOMES = tuple(analyze.OUTCOMES)
ROOT = Path(__file__).resolve().parents[1]


def combined_design(sample, speakers, *, include_age=True):
    pre, names, cleaning = baseline.baseline_design(sample, speakers, include_age=include_age)
    text, _ = analyze.build_text_design(sample)
    joint = np.column_stack((pre, text))
    return {"baseline": pre, "baseline_text": joint}, names, cleaning


def selection_overlap(samples):
    rows = []
    for i, (name_a, sample_a) in enumerate(samples.items()):
        ids_a = set(sample_a.dialogue_id)
        people_a = set(sample_a.speaker_a) | set(sample_a.speaker_b)
        assert len(people_a) == 2 * len(sample_a)
        for name_b, sample_b in list(samples.items())[i + 1:]:
            ids_b = set(sample_b.dialogue_id)
            people_b = set(sample_b.speaker_a) | set(sample_b.speaker_b)
            rows.append(dict(selection_a=name_a, selection_b=name_b,
                shared_dialogues=len(ids_a & ids_b), shared_participants=len(people_a & people_b),
                dialogue_jaccard=len(ids_a & ids_b) / len(ids_a | ids_b)))
    return pd.DataFrame(rows)


def run_job(payload):
    name, covariates, sample, X, draws, output, baseline_results = payload
    started = time.monotonic()
    directory = Path(output) / "jobs" / f"{name}__{covariates}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        return str(directory), True
    d = sample.D.to_numpy(dtype=int)
    folds = baseline.independent_folds(len(sample))
    predictions = sample[["dialogue_id", "D", *OUTCOMES]].copy()
    for j, (_, test) in enumerate(folds):
        predictions.loc[test, "fold"] = j
    reuse = baseline_results is not None and name == "main" and covariates == "baseline"
    if reuse:
        baseline_results = Path(baseline_results)
        cached = pd.read_csv(baseline_results / "frozen_main_predictions.csv")
        np.testing.assert_array_equal(cached.dialogue_id, sample.dialogue_id)
        np.testing.assert_array_equal(cached.D, d)
        np.testing.assert_array_equal(cached["fold"], predictions["fold"])
        design = np.load(baseline_results / "frozen_main_design.npz")
        np.testing.assert_array_equal(design["X"], X)
        old_tests = pd.read_csv(baseline_results / "baseline_tests.csv")
        old_tests = old_tests[old_tests["sample"].eq("frozen_main")]
    rows = []
    learners = core.LEARNERS if name == "main" else ("ridge",)
    kernels = {h: core.rbf_kernel_matrix(X, h) for h in (0.5, 1.0, 2.0)}
    for learner in learners:
        for regime in core.REGIMES:
            # Keep random seeds fixed across selections and covariate sets.
            e_seed = core.stable_seed(baseline.VERSION, "frozen_main", learner, regime, "e")
            if reuse:
                e = cached[f"e_{learner}_{regime}"].to_numpy()
                etune = old_tests.loc[old_tests.learner.eq(learner) & old_tests.regime.eq(regime), "propensity_tuning"].iloc[0]
            else:
                e, etune = core.fit_predictions(X, d, learner=learner, regime=regime,
                    kind="propensity", folds=folds, seed=e_seed)
            predictions[f"e_{learner}_{regime}"] = e
            for outcome in OUTCOMES:
                y = sample[outcome].to_numpy(dtype=float)
                m_seed = core.stable_seed(baseline.VERSION, "frozen_main", learner, regime, outcome)
                multiplier_seed = core.stable_seed(baseline.VERSION, "frozen_main", outcome, regime, "multipliers")
                if reuse:
                    m = cached[f"m_{outcome}_{learner}_{regime}"].to_numpy()
                    mtune = old_tests.loc[old_tests.learner.eq(learner) & old_tests.regime.eq(regime)
                        & old_tests.outcome.eq(outcome), "outcome_tuning"].iloc[0]
                else:
                    m, mtune = core.fit_predictions(X, y, learner=learner, regime=regime,
                        kind="outcome", folds=folds, seed=m_seed, bounds=analyze.OUTCOMES[outcome])
                predictions[f"m_{outcome}_{learner}_{regime}"] = m
                bandwidths = (0.5, 1.0, 2.0) if name == "main" and learner == "ridge" else (1.0,)
                for h in bandwidths:
                    result = core.multiplier_test((y - m) * (d - e), kernels[h],
                        bootstrap_draws=draws, seed=multiplier_seed)
                    if reuse:
                        old = old_tests.loc[old_tests.learner.eq(learner) & old_tests.regime.eq(regime)
                            & old_tests.outcome.eq(outcome) & old_tests.bandwidth.eq(h)]
                        if len(old):
                            for column in ("statistic", "critical_value_0_05", "p_value"):
                                np.testing.assert_allclose(result[column], old.iloc[0][column], rtol=1e-10, atol=1e-12)
                    rows.append(dict(selection=name, covariates=covariates, n=len(sample),
                        n0=int((d == 0).sum()), n1=int(d.sum()), dimensions=X.shape[1],
                        outcome=outcome, learner=learner, regime=regime, bandwidth=h,
                        bootstrap_draws=draws, bootstrap_seed=multiplier_seed,
                        propensity_seed=e_seed, outcome_seed=m_seed,
                        propensity_tuning=etune, outcome_tuning=mtune,
                        propensity_min=float(e.min()), propensity_max=float(e.max()),
                        bootstrap_numerator=int(round(result["p_value"] * (draws + 1))), **result))
    pd.DataFrame(rows).to_csv(directory / "tests.csv", index=False)
    predictions.to_csv(directory / "predictions.csv", index=False)
    np.savez_compressed(directory / "design.npz", X=X, dialogue_id=sample.dialogue_id.to_numpy())
    (directory / "complete.json").write_text(json.dumps(dict(selection=name, covariates=covariates,
        tests=len(rows), reused_main_baseline_predictions=reuse,
        elapsed_seconds=time.monotonic() - started), indent=2) + "\n")
    return str(directory), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=analyze.APPLICATION_DIR / "data/raw/casino-corpus")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "runs/casino-joint")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--baseline-results", type=Path,
                        help="optional completed baseline_analysis.py output to reuse")
    args = parser.parse_args()
    if args.bootstrap_draws != 5000:
        raise ValueError("The matched analysis fixes the multiplier budget at 5000")
    if args.baseline_results is not None:
        args.baseline_results = args.baseline_results.resolve()
        for filename in ("frozen_main_predictions.csv", "frozen_main_design.npz", "baseline_tests.csv"):
            if not (args.baseline_results / filename).is_file():
                raise ValueError(f"Baseline results are incomplete: missing {filename}")
    started = time.monotonic()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    archive = args.corpus_dir.parent / "casino-corpus.zip"
    assert core.file_sha256(archive) == core.ARCHIVE_SHA256
    corpus_hashes = core.verify_extracted_corpus(archive, args.corpus_dir)
    records, counts = analyze.load_integrative_potential_records(args.corpus_dir)
    eligible = [r for r in records if r["eligible_extreme"]]
    primary, main_hash = analyze.frozen_main_sample(eligible)
    speakers = json.loads((args.corpus_dir / "speakers.json").read_text())
    samples = {"main": primary}
    salts = {"main": analyze.MAIN_SELECTION_SALT}
    for i, salt in enumerate(analyze.SELECTION_SENSITIVITY_SALTS, 1):
        name = f"alternative_{i:02d}"
        samples[name], _ = analyze.outcome_blind_disjoint_sample(eligible, salt=salt)
        salts[name] = salt
    overlap = selection_overlap(samples)
    alternative, n_before_restriction = baseline.selection_before_restriction(records)
    samples["select_before_restrict"] = alternative
    samples["omit_age"] = primary
    all_columns = {}
    jobs = []
    for name, sample in samples.items():
        features, columns, cleaning = combined_design(sample, speakers, include_age=name != "omit_age")
        all_columns[name] = dict(baseline_columns=columns, **cleaning)
        for key, X in features.items():
            jobs.append((name, key, sample, X, args.bootstrap_draws, str(args.results_dir),
                         str(args.baseline_results) if args.baseline_results is not None else None))
    protocol = dict(version=VERSION, primary_sample_sha256=main_hash, corpus_member_sha256=corpus_hashes,
        outcomes={"mean_satisfaction": "(S1+S2)/2", "both_extremely_satisfied": "1{S1=5,S2=5}"},
        covariates={"baseline": "36 fixed baseline coordinates", "baseline_text": "concatenate baseline with 128 unit-norm signed text-hash coordinates"},
        primary_bandwidth=1, bandwidth_sensitivity=[0.5, 1, 2], bootstrap_draws=args.bootstrap_draws,
        main_learners=list(core.LEARNERS), sensitivity_learners=["ridge"], regimes=list(core.REGIMES),
        outer_folds="5 shuffled folds independent of labels; shared across outcomes and covariate sets",
        selection_salts=salts, include_age_and_order_sensitivities=True,
        random_seeds="Use the original baseline main-analysis seeds for all samples and covariate sets",
        source_sha256={p.name: core.file_sha256(p) for p in [Path(__file__), Path(core.__file__), Path(analyze.__file__), Path(baseline.__file__)]},
        archived_baseline_prediction_sha256=(core.file_sha256(args.baseline_results / "frozen_main_predictions.csv")
            if args.baseline_results is not None else None))
    plan_path = args.results_dir / "analysis_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != protocol:
        raise ValueError("Existing results use a different protocol; choose a new results directory")
    plan_path.write_text(json.dumps(protocol, indent=2) + "\n")
    overlap.to_csv(args.results_dir / "selection_overlap.csv", index=False)
    sample_rows = []
    for name, sample in samples.items():
        sample.drop(columns="early_text").to_csv(args.results_dir / f"{name}_sample.csv", index=False)
        sample_rows.append(dict(selection=name, n=len(sample), n0=int((sample.D == 0).sum()),
            n1=int(sample.D.sum()), unique_participants=len(set(sample.speaker_a) | set(sample.speaker_b))))
    pd.DataFrame(sample_rows).to_csv(args.results_dir / "sample_summary.csv", index=False)
    print(f"Frozen protocol: {len(jobs)} jobs; {len(primary)} main dialogues; two covariate sets and two outcomes", flush=True)
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, payload) for payload in jobs]
        for future in as_completed(futures):
            directory, cached = future.result()
            completed.append(directory)
            print(f"{len(completed)}/{len(jobs)} {Path(directory).name} {'cached' if cached else 'finished'}; {time.monotonic()-started:.1f}s", flush=True)
    tests = pd.concat([pd.read_csv(Path(path) / "tests.csv") for path in completed], ignore_index=True)
    tests = tests.sort_values(["selection", "covariates", "outcome", "learner", "regime", "bandwidth"])
    assert len(tests) == 136 and not tests.duplicated(["selection", "covariates", "outcome", "learner", "regime", "bandwidth"]).any()
    tests.to_csv(args.results_dir / "regression_tests.csv", index=False)
    summary = dict(version=VERSION, complete=True, tests=len(tests), corpus_counts=counts,
        sample_sha256=main_hash, design=all_columns, selection_before_restriction_total=n_before_restriction,
        elapsed_seconds=time.monotonic()-started, python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    (args.results_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary | {"design": "saved in analysis_summary.json"}), flush=True)


if __name__ == "__main__":
    main()
