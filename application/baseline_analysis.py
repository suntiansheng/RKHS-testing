#!/usr/bin/env python3
"""Exploratory CaSiNo comparisons using only pretreatment survey covariates.

This companion preserves the original sample and separates regression tests
from descriptive standardized mean contrasts. It does not assert that corpus
filtering preserves randomization or that rejecting zero effects proves
heterogeneity. All specifications are retained, regardless of their results.
"""

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
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold

import analyze
import core


VERSION = "casino-baseline-v1"
TRAITS = (
    "extraversion", "agreeableness", "conscientiousness",
    "emotional-stability", "openness-to-experiences",
)
CATEGORIES = {
    "gender": ("female", "male", "other"),
    "ethnicity": (
        "white american", "asian american", "black or african american",
        "alaska native or native american", "hispanic or latino",
        "native hawaiian or other specific islander", "multi-racial",
        "other (please specify)",
    ),
    "education": (
        "some high school, no diploma", "high school graduate / ged",
        "trade school", "some 2 year college, no degree",
        "some 2 year college, associate's degree", "some 4 year college, no degree",
        "some 4 year college, bachelor's degree", "master's degree", "doctorate degree",
    ),
    "svo": ("prosocial", "proself", "unclassified"),
}


def baseline_design(sample, speakers, *, include_age=True):
    """A fixed, bounded, swap-invariant map with no fitted preprocessing.

    Numeric traits contribute a pair mean and absolute difference. Categories
    contribute the fraction of the pair in each category. Age outside [18,100]
    is encoded as zero on the rescaled age coordinate, with a missing indicator.
    Learner standardization is subsequently fitted within training folds.
    """
    rows = []
    invalid_ages = set()
    for row in sample.itertuples():
        ids = [row.speaker_a, row.speaker_b]
        people = [speakers[speaker]["meta"] for speaker in ids]
        values = {}
        if include_age:
            ages = np.array([float(p["demographics"]["age"]) for p in people])
            valid = np.isfinite(ages) & (ages >= 18) & (ages <= 100)
            invalid_ages.update(s for s, ok in zip(ids, valid) if not ok)
            age = np.where(valid, (ages - 18.0) / 82.0, 0.0)
            values.update(age_mean=float(age.mean()), age_gap=float(abs(age[0]-age[1])),
                          age_missing_fraction=float((~valid).mean()))
        for trait in TRAITS:
            raw = np.array([float(p["personality"]["big-five"][trait]) for p in people])
            if not (np.isfinite(raw).all() and ((raw >= 1) & (raw <= 7)).all()):
                raise ValueError(f"Invalid Big Five values: {trait}")
            scaled = (raw - 1.0) / 6.0
            values[trait + "_mean"] = float(scaled.mean())
            values[trait + "_gap"] = float(abs(scaled[0] - scaled[1]))
        for field, levels in CATEGORIES.items():
            source = "personality" if field == "svo" else "demographics"
            observed = [p[source][field] for p in people]
            if any(value not in levels for value in observed):
                raise ValueError(f"Unexpected {field} category: {observed}")
            for level in levels:
                values[field + ":" + level] = observed.count(level) / 2.0
        rows.append(values)
    features = pd.DataFrame(rows)
    X = features.to_numpy(dtype=float)
    if not (np.isfinite(X).all() and ((X >= 0) & (X <= 1)).all()):
        raise AssertionError("Baseline design must be finite and bounded")
    return X, list(features.columns), {"invalid_age_participants": len(invalid_ages)}


def selection_before_restriction(records):
    """Use the original identifier order across all three preference regimes."""
    def key(row):
        value = f"{analyze.MAIN_SELECTION_SALT}|{row['dialogue_id']}".encode()
        return hashlib.sha256(value).hexdigest(), int(row["dialogue_id"])
    selected, used = [], set()
    for record in sorted(records, key=key):
        pair = {record["speaker_a"], record["speaker_b"]}
        if used.isdisjoint(pair):
            selected.append(record)
            used.update(pair)
    extreme = pd.DataFrame([r for r in selected if r["eligible_extreme"]])
    extreme = extreme.sort_values("dialogue_id").reset_index(drop=True)
    extreme["D"] = extreme["D"].astype(int)
    return extreme, len(selected)


def independent_folds(n):
    return list(KFold(5, shuffle=True, random_state=core.stable_seed(VERSION, "folds")).split(np.arange(n)))


def aipw_scores(y, d, e, mu0, mu1):
    """Cross-fitted standardized contrast; no propensity clipping."""
    if np.any((e <= 0) | (e >= 1)):
        raise ValueError("Standardization requires interior propensity predictions")
    return mu1 - mu0 + d * (y - mu1) / e - (1-d) * (y - mu0) / (1-e)


def arm_predictions(X, y, d, folds, learner, outcome):
    predictions = np.full((len(y), 2), np.nan)
    for fold, (train, test) in enumerate(folds):
        for arm in (0, 1):
            arm_train = train[d[train] == arm]
            model = core.outcome_model(learner, core.stable_seed(VERSION, "arm", learner, outcome, fold, arm), 4)
            model.fit(X[arm_train], y[arm_train])
            predictions[test, arm] = np.clip(model.predict(X[test]), *analyze.OUTCOMES[outcome])
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete arm predictions")
    return predictions


def analyze_sample(sample, speakers, *, name, draws, full, include_age=True):
    X, columns, cleaning = baseline_design(sample, speakers, include_age=include_age)
    d = sample["D"].to_numpy(dtype=int)
    folds = independent_folds(len(sample))
    predictions = sample[["dialogue_id", "D", "mean_satisfaction", "both_extremely_satisfied"]].copy()
    for index, (_, test) in enumerate(folds):
        predictions.loc[test, "fold"] = index
    tests, diagnostics, effects = [], [], []
    kernel = core.rbf_kernel_matrix(X, 1.0)
    learners = core.LEARNERS if full else ("ridge",)
    regimes = core.REGIMES if full else ("no_split",)
    for learner in learners:
        propensity_oof = None
        for regime in regimes:
            e, etune = core.fit_predictions(X, d, learner=learner, regime=regime,
                kind="propensity", folds=folds, seed=core.stable_seed(VERSION, name, learner, regime, "e"))
            predictions[f"e_{learner}_{regime}"] = e
            if regime == "crossfit_5fold":
                propensity_oof = e
            diagnostics.append(dict(sample=name, learner=learner, regime=regime,
                propensity_min=float(e.min()), propensity_max=float(e.max()),
                propensity_auc=float(roc_auc_score(d, e)), propensity_brier=float(brier_score_loss(d, e)),
                tuning=etune))
            outcomes = analyze.OUTCOMES if full else ("mean_satisfaction",)
            for outcome in outcomes:
                y = sample[outcome].to_numpy(dtype=float)
                m, mtune = core.fit_predictions(X, y, learner=learner, regime=regime,
                    kind="outcome", folds=folds, bounds=analyze.OUTCOMES[outcome],
                    seed=core.stable_seed(VERSION, name, learner, regime, outcome))
                predictions[f"m_{outcome}_{learner}_{regime}"] = m
                score = (y-m)*(d-e)
                bandwidths = (0.5, 1.0, 2.0) if learner == "ridge" and regime == "no_split" else (1.0,)
                for h in bandwidths:
                    result = core.multiplier_test(score, kernel if h == 1 else core.rbf_kernel_matrix(X, h),
                        bootstrap_draws=draws, seed=core.stable_seed(VERSION, name, outcome, regime, "multipliers"))
                    tests.append(dict(sample=name, n=len(sample), n0=int((d==0).sum()), n1=int(d.sum()),
                        dimensions=X.shape[1], learner=learner, regime=regime, outcome=outcome,
                        bandwidth=h, bootstrap_draws=draws, outcome_tuning=mtune, propensity_tuning=etune,
                        **result))
                print(f"{name}: {learner} {regime} {outcome} fitted", flush=True)
        if full:
            for outcome in analyze.OUTCOMES:
                y = sample[outcome].to_numpy(dtype=float)
                arm = arm_predictions(X, y, d, folds, learner, outcome)
                influence = aipw_scores(y, d, propensity_oof, arm[:, 0], arm[:, 1])
                estimate = float(influence.mean())
                se = float(influence.std(ddof=1)/np.sqrt(len(y)))
                effects.append(dict(sample=name, outcome=outcome, method=learner,
                    estimate=estimate, standard_error=se, ci_lower=estimate-1.96*se, ci_upper=estimate+1.96*se,
                    interval="cross-fitted influence-function normal interval, working iid approximation"))
                predictions[f"mu0_{outcome}_{learner}"] = arm[:, 0]
                predictions[f"mu1_{outcome}_{learner}"] = arm[:, 1]
                predictions[f"aipw_{outcome}_{learner}"] = influence
            print(f"{name}: {learner} standardized contrasts fitted", flush=True)
    return pd.DataFrame(tests), pd.DataFrame(diagnostics), pd.DataFrame(effects), predictions, X, columns, cleaning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=analyze.APPLICATION_DIR/"data/raw/casino-corpus")
    parser.add_argument("--results-dir", type=Path, default=analyze.APPLICATION_DIR.parent/"runs/casino-baseline")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_draws < 999:
        raise ValueError("Use at least 999 multipliers")
    started = time.monotonic()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    archive = args.corpus_dir.parent/"casino-corpus.zip"
    if core.file_sha256(archive) != core.ARCHIVE_SHA256:
        raise ValueError("Archive checksum mismatch")
    hashes = core.verify_extracted_corpus(archive, args.corpus_dir)
    records, counts = analyze.load_integrative_potential_records(args.corpus_dir)
    sample, sample_hash = analyze.frozen_main_sample([r for r in records if r["eligible_extreme"]])
    speakers = json.loads((args.corpus_dir/"speakers.json").read_text())
    all_tests, all_diagnostics, all_effects = [], [], []
    alternative, n_all_selected = selection_before_restriction(records)
    for name, frame, full, age in [
        ("frozen_main", sample, True, True),
        ("omit_age", sample, False, False),
        ("select_before_restrict", alternative, False, True),
    ]:
        result = analyze_sample(frame, speakers, name=name, draws=args.bootstrap_draws, full=full, include_age=age)
        tests, diagnostics, effects, predictions, X, columns, cleaning = result
        all_tests.append(tests); all_diagnostics.append(diagnostics)
        if not effects.empty:
            all_effects.append(effects)
        predictions.to_csv(args.results_dir/f"{name}_predictions.csv", index=False)
        frame.drop(columns="early_text").to_csv(args.results_dir/f"{name}_sample.csv", index=False)
        np.savez_compressed(args.results_dir/f"{name}_design.npz", X=X, D=frame.D.to_numpy(),
            dialogue_id=frame.dialogue_id.to_numpy(), feature_names=np.array(columns))
        if name == "frozen_main":
            design_metadata = dict(features=columns, dimensions=len(columns), **cleaning)
            balance=[]
            for j, column in enumerate(columns):
                a,b=X[frame.D==0,j],X[frame.D==1,j]
                pooled=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
                balance.append(dict(feature=column, aligned_mean=float(a.mean()), maximal_mean=float(b.mean()),
                    standardized_difference=float((b.mean()-a.mean())/pooled) if pooled > 0 else 0.0))
            pd.DataFrame(balance).to_csv(args.results_dir/"baseline_balance.csv",index=False)
        pd.concat(all_tests,ignore_index=True).to_csv(args.results_dir/"baseline_tests.csv",index=False)
    pd.concat(all_diagnostics,ignore_index=True).to_csv(args.results_dir/"baseline_diagnostics.csv",index=False)
    contrasts=analyze.descriptive_contrasts(sample,bootstrap_draws=10000)
    contrasts.to_csv(args.results_dir/"descriptive_contrasts.csv",index=False)
    for row in contrasts.itertuples():
        all_effects.append(pd.DataFrame([dict(sample="frozen_main",outcome=row.outcome,method="unadjusted",
            estimate=row.mean_difference_maximal_minus_aligned,standard_error=np.nan,
            ci_lower=row.bootstrap_ci_lower,ci_upper=row.bootstrap_ci_upper,
            interval="10000 stratified dialogue-bootstrap percentile interval, working iid approximation")]))
    pd.concat(all_effects,ignore_index=True).to_csv(args.results_dir/"standardized_contrasts.csv",index=False)
    metadata=dict(version=VERSION,corpus_counts=counts,corpus_member_sha256=hashes,sample_sha256=sample_hash,
        design=design_metadata,bootstrap_draws=args.bootstrap_draws,folds="independent shuffled KFold(5); not label-stratified",
        alternative_all_regime_disjoint_n=n_all_selected,alternative_extreme_n=len(alternative),
        interpretation="Exploratory conditional comparisons. CATE requires identification in the selected population. No heterogeneity claim.",
        elapsed_seconds=time.monotonic()-started,python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    (args.results_dir/"analysis_summary.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"elapsed_seconds":metadata["elapsed_seconds"],"results_dir":str(args.results_dir)}),flush=True)


if __name__ == "__main__":
    main()
