#!/usr/bin/env python3
"""Conditional satisfaction comparisons in the CaSiNo negotiation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


APPLICATION_DIR = Path(__file__).resolve().parent
if str(APPLICATION_DIR) not in sys.path:
    sys.path.insert(0, str(APPLICATION_DIR))
import core


APPLICATION_VERSION = "casino-integrative-potential-v1"
MAIN_SELECTION_SALT = "CaSiNo-preference-satisfaction-v1-disjoint"
SELECTION_SENSITIVITY_SALTS = tuple(
    f"CaSiNo-preference-satisfaction-v1-sensitivity-{index:02d}" for index in range(10)
)
MAIN_SAMPLE_SHA256 = "b5ab9409d51a33c461ed05a4ad4de53c7c888456cfa16bbe4176ab87eb0729a0"
FIXED_BANDWIDTH = 1.0
HASH_FEATURES = 128
N_EARLY_TURNS = 6
MINIMUM_WORD_DOCUMENT_FREQUENCY = 10
SCENARIO_NOUNS = set(core.SCENARIO_NOUNS) | {
    "foods",
    "waters",
    "wood",
    "woods",
    "pack",
    "packs",
    "ration",
    "rations",
}
WORD_STOPWORDS = set(core.WORD_STOPWORDS) | SCENARIO_NOUNS
LEARNERS = core.LEARNERS
REGIMES = core.REGIMES
OUTCOMES: dict[str, tuple[float, float]] = {
    "mean_satisfaction": (1.0, 5.0),
    "both_extremely_satisfied": (0.0, 1.0),
}
POPULATION_BY_D = {0: "aligned", 1: "maximal"}


def load_integrative_potential_records(
    corpus_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read all dialogues and derive the preference-extreme populations."""

    with (corpus_dir / "speakers.json").open(encoding="utf-8") as handle:
        speakers = json.load(handle)
    with (corpus_dir / "conversations.json").open(encoding="utf-8") as handle:
        conversations = json.load(handle)

    utterances_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    utterance_count = 0
    with (corpus_dir / "utterances.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            utterance = json.loads(line)
            utterances_by_conversation[utterance["conversation_id"]].append(utterance)
            utterance_count += 1

    if len(speakers) != 846 or len(conversations) != 1030 or utterance_count != 14297:
        raise ValueError(
            "Unexpected CaSiNo corpus dimensions: "
            f"{len(speakers)} speakers, {len(conversations)} dialogues, "
            f"{utterance_count} utterances."
        )

    records: list[dict[str, Any]] = []
    natural_utterances = 0
    control_utterances = 0
    maximum_counts: Counter[int] = Counter()
    for conversation_id, conversation in sorted(
        conversations.items(), key=lambda item: int(item[1]["meta"]["dialogue_id"])
    ):
        utterances = sorted(
            utterances_by_conversation[conversation_id],
            key=lambda utterance: core.utterance_number(utterance["id"]),
        )
        role_to_speaker: dict[str, str] = {}
        for utterance in utterances:
            role = utterance["meta"]["speaker_internal_id"]
            speaker = utterance["speaker"]
            if role in role_to_speaker and role_to_speaker[role] != speaker:
                raise ValueError(f"Nonunique role mapping in {conversation_id}")
            role_to_speaker[role] = speaker
            if core.is_control_utterance(utterance):
                control_utterances += 1
            else:
                natural_utterances += 1
        if len(role_to_speaker) != 2:
            raise ValueError(f"Expected two speakers in {conversation_id}")

        early_utterances: list[dict[str, Any]] = []
        for utterance in utterances:
            if core.is_control_utterance(utterance):
                break
            early_utterances.append(utterance)
            if len(early_utterances) == N_EARLY_TURNS:
                break
        if len(early_utterances) != N_EARLY_TURNS:
            raise ValueError(f"Fewer than six pre-action turns in {conversation_id}")
        early_speaker_counts = Counter(
            utterance["speaker"] for utterance in early_utterances
        )
        if sorted(early_speaker_counts.values()) != [3, 3]:
            raise ValueError(
                f"First six turns are not balanced across speakers in {conversation_id}"
            )

        participant_info = conversation["meta"]["participant_info"]
        if set(participant_info) != set(role_to_speaker):
            raise ValueError(f"Participant-role mismatch in {conversation_id}")
        priorities_by_role = {
            role: core.priority_by_issue(participant_info[role])
            for role in participant_info
        }
        maximum_joint_points = sum(
            3 * max(priorities_by_role[role][issue] for role in priorities_by_role)
            for issue in core.ISSUES
        )
        if maximum_joint_points not in {36, 39, 42}:
            raise ValueError(
                f"Unexpected maximum joint points in {conversation_id}: "
                f"{maximum_joint_points}"
            )
        maximum_counts[maximum_joint_points] += 1
        integrative_gain = maximum_joint_points - 36
        if maximum_joint_points == 36:
            population = "aligned"
            d_value = 0
        elif maximum_joint_points == 42:
            population = "maximal"
            d_value = 1
        else:
            population = "intermediate"
            d_value = None

        satisfaction_by_speaker = {
            role_to_speaker[role]: core.SATISFACTION_SCORE[
                participant_info[role]["outcomes"]["satisfaction"]
            ]
            for role in participant_info
        }
        speaker_pair = tuple(sorted(role_to_speaker.values()))
        satisfaction_a = int(satisfaction_by_speaker[speaker_pair[0]])
        satisfaction_b = int(satisfaction_by_speaker[speaker_pair[1]])
        mean_satisfaction = 0.5 * (satisfaction_a + satisfaction_b)
        both_extremely_satisfied = int(satisfaction_a == 5 and satisfaction_b == 5)

        action_values = {
            utterance.get("meta", {}).get("data")
            for utterance in utterances
            if core.is_control_utterance(utterance)
        }
        if "walk_away" in action_values:
            deal_status = "walk_away"
        elif "accept_deal" in action_values:
            deal_status = "accepted"
        elif "reject_deal" in action_values:
            deal_status = "rejected_without_acceptance"
        else:
            deal_status = "other"

        early_text = " ".join(
            core.normalize_text(utterance["text"]) for utterance in early_utterances
        )
        records.append(
            {
                "dialogue_id": int(conversation["meta"]["dialogue_id"]),
                "conversation_id": conversation_id,
                "speaker_a": speaker_pair[0],
                "speaker_b": speaker_pair[1],
                "population": population,
                "D": d_value,
                "eligible_extreme": int(d_value is not None),
                "maximum_joint_points": maximum_joint_points,
                "integrative_gain": integrative_gain,
                "early_turns": N_EARLY_TURNS,
                "early_speaker_a_turns": early_speaker_counts[speaker_pair[0]],
                "early_speaker_b_turns": early_speaker_counts[speaker_pair[1]],
                "early_token_count": len(core.WORD_RE.findall(early_text.lower())),
                "early_text": early_text,
                "satisfaction_a": satisfaction_a,
                "satisfaction_b": satisfaction_b,
                "mean_satisfaction": mean_satisfaction,
                "both_extremely_satisfied": both_extremely_satisfied,
                "deal_status": deal_status,
            }
        )

    if maximum_counts != Counter({36: 257, 39: 378, 42: 395}):
        raise ValueError(f"Unexpected integrative-potential counts: {maximum_counts}")
    corpus_counts = {
        "speakers": len(speakers),
        "dialogues": len(conversations),
        "utterances": utterance_count,
        "natural_utterances": natural_utterances,
        "control_utterances": control_utterances,
        "aligned_m36": maximum_counts[36],
        "intermediate_m39": maximum_counts[39],
        "maximal_m42": maximum_counts[42],
        "eligible_extreme_dialogues": maximum_counts[36] + maximum_counts[42],
    }
    return records, corpus_counts


def outcome_blind_disjoint_sample(
    eligible_records: list[dict[str, Any]], *, salt: str
) -> tuple[pd.DataFrame, str]:
    """Apply one global salted greedy pass across both populations."""

    def ordering_key(record: dict[str, Any]) -> tuple[str, int]:
        digest = hashlib.sha256(
            f"{salt}|{record['dialogue_id']}".encode("utf-8")
        ).hexdigest()
        return digest, int(record["dialogue_id"])

    selected: list[dict[str, Any]] = []
    used_speakers: set[str] = set()
    for record in sorted(eligible_records, key=ordering_key):
        pair = {record["speaker_a"], record["speaker_b"]}
        if used_speakers.isdisjoint(pair):
            selected.append(record)
            used_speakers.update(pair)
    frame = pd.DataFrame(selected).sort_values("dialogue_id").reset_index(drop=True)
    if len(used_speakers) != 2 * len(frame):
        raise AssertionError("Participant-disjoint selection failed")
    frame["D"] = frame["D"].astype(int)
    id_payload = ",".join(str(value) for value in frame["dialogue_id"])
    sample_sha256 = hashlib.sha256(id_payload.encode("utf-8")).hexdigest()
    return frame, sample_sha256


def frozen_main_sample(
    eligible_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, str]:
    sample, sample_sha256 = outcome_blind_disjoint_sample(
        eligible_records, salt=MAIN_SELECTION_SALT
    )
    counts = sample["population"].value_counts().to_dict()
    if counts != {"maximal": 159, "aligned": 73}:
        raise AssertionError(f"Unexpected frozen group counts: {counts}")
    if sample_sha256 != MAIN_SAMPLE_SHA256:
        raise AssertionError(
            f"Sample identity mismatch: expected {MAIN_SAMPLE_SHA256}, "
            f"got {sample_sha256}"
        )
    return sample, sample_sha256


def build_text_design(sample: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the frozen text-only design; priority features are forbidden."""

    vectorizer = HashingVectorizer(
        n_features=HASH_FEATURES,
        alternate_sign=True,
        lowercase=True,
        norm="l2",
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
        dtype=np.float64,
    )
    design = vectorizer.transform(sample["early_text"]).toarray()
    if design.shape != (len(sample), HASH_FEATURES):
        raise AssertionError(f"Unexpected design shape: {design.shape}")
    if not np.isfinite(design).all():
        raise ValueError("Nonfinite text design")
    if not np.allclose(np.linalg.norm(design, axis=1), 1.0):
        raise ValueError("Every hashed text row must have unit norm")
    metadata = {
        "dimensions": HASH_FEATURES,
        "text_map": "fixed 128-coordinate signed hashing vectorizer",
        "ngrams": [1, 2],
        "text_row_norm": "l2",
        "source_turns": "first six natural utterances before the first action",
        "priority_features": "excluded because they determine D exactly",
    }
    return design, metadata


def fit_all_nuisance_and_tests(
    sample: pd.DataFrame,
    design: np.ndarray,
    *,
    bootstrap_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    d = sample["D"].to_numpy(dtype=int)
    crossfit_seed = core.stable_seed(APPLICATION_VERSION, "crossfit-folds")
    fold_generator = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=crossfit_seed
    )
    folds = list(fold_generator.split(design, d))
    fold_assignment = np.zeros(len(sample), dtype=int)
    for fold_index, (_, test) in enumerate(folds, start=1):
        fold_assignment[test] = fold_index

    predictions = sample[
        [
            "dialogue_id",
            "D",
            "population",
            "mean_satisfaction",
            "both_extremely_satisfied",
        ]
    ].copy()
    predictions["crossfit_fold"] = fold_assignment
    propensity_predictions: dict[tuple[str, str], np.ndarray] = {}
    propensity_tuning: dict[tuple[str, str], str] = {}
    propensity_seeds: dict[str, int] = {}
    for regime in REGIMES:
        for learner in LEARNERS:
            seed = core.stable_seed(APPLICATION_VERSION, "propensity", learner, regime)
            fitted, tuning = core.fit_predictions(
                design,
                d,
                learner=learner,
                regime=regime,
                kind="propensity",
                folds=folds,
                seed=seed,
            )
            propensity_predictions[(learner, regime)] = fitted
            propensity_tuning[(learner, regime)] = tuning
            propensity_seeds[f"{learner}/{regime}"] = seed
            short_regime = "crossfit" if regime == "crossfit_5fold" else regime
            predictions[f"propensity_{learner}_{short_regime}"] = fitted

    median_bandwidth = core.median_bandwidth(design)
    kernels = {
        "fixed_h_1": (FIXED_BANDWIDTH, core.rbf_kernel_matrix(design, FIXED_BANDWIDTH)),
        "pooled_median_sensitivity": (
            median_bandwidth,
            core.rbf_kernel_matrix(design, median_bandwidth),
        ),
    }
    outcome_seeds: dict[str, int] = {}
    multiplier_seeds: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for outcome, bounds in OUTCOMES.items():
        y = sample[outcome].to_numpy(dtype=float)
        for regime in REGIMES:
            for learner in LEARNERS:
                outcome_seed = core.stable_seed(
                    APPLICATION_VERSION, "outcome", outcome, learner, regime
                )
                outcome_seeds[f"{outcome}/{learner}/{regime}"] = outcome_seed
                fitted_outcome, outcome_tuning = core.fit_predictions(
                    design,
                    y,
                    learner=learner,
                    regime=regime,
                    kind="outcome",
                    folds=folds,
                    seed=outcome_seed,
                    bounds=bounds,
                )
                short_regime = "crossfit" if regime == "crossfit_5fold" else regime
                predictions[f"outcome_{outcome}_{learner}_{short_regime}"] = (
                    fitted_outcome
                )
                propensity = propensity_predictions[(learner, regime)]
                score = (y - fitted_outcome) * (d - propensity)
                bootstrap_seed = core.stable_seed(
                    APPLICATION_VERSION, "multiplier", outcome, regime
                )
                multiplier_seeds[f"{outcome}/{regime}"] = bootstrap_seed
                for bandwidth_spec, (bandwidth, kernel) in kernels.items():
                    test = core.multiplier_test(
                        score,
                        kernel,
                        bootstrap_draws=bootstrap_draws,
                        seed=bootstrap_seed,
                    )
                    rows.append(
                        {
                            "outcome": outcome,
                            "regime": regime,
                            "learner": learner,
                            "bandwidth_spec": bandwidth_spec,
                            "bandwidth": bandwidth,
                            "is_primary": int(
                                outcome == "mean_satisfaction"
                                and regime == "no_split"
                                and learner == "ridge"
                                and bandwidth_spec == "fixed_h_1"
                            ),
                            "is_ordinal_sensitivity": int(
                                outcome == "both_extremely_satisfied"
                                and regime == "no_split"
                                and learner == "ridge"
                                and bandwidth_spec == "fixed_h_1"
                            ),
                            "n": len(sample),
                            "n_aligned": int(np.sum(d == 0)),
                            "n_maximal": int(np.sum(d == 1)),
                            "aligned_mean": float(np.mean(y[d == 0])),
                            "maximal_mean": float(np.mean(y[d == 1])),
                            "mean_difference_maximal_minus_aligned": float(
                                np.mean(y[d == 1]) - np.mean(y[d == 0])
                            ),
                            "bootstrap_draws": bootstrap_draws,
                            "bootstrap_seed": bootstrap_seed,
                            "outcome_seed": outcome_seed,
                            "propensity_seed": propensity_seeds[f"{learner}/{regime}"],
                            "prediction_rmse": float(
                                np.sqrt(np.mean((y - fitted_outcome) ** 2))
                            ),
                            "prediction_evaluation": (
                                "training" if regime == "no_split" else "out_of_fold"
                            ),
                            "propensity_auc": float(roc_auc_score(d, propensity)),
                            "propensity_brier": float(brier_score_loss(d, propensity)),
                            "propensity_min": float(np.min(propensity)),
                            "propensity_max": float(np.max(propensity)),
                            "propensity_fraction_0_1_to_0_9": float(
                                np.mean((propensity >= 0.1) & (propensity <= 0.9))
                            ),
                            "score_mean": float(np.mean(score)),
                            "score_sd": float(np.std(score, ddof=1)),
                            "outcome_tuning": outcome_tuning,
                            "propensity_tuning": propensity_tuning[(learner, regime)],
                            "outcome_tuning_at_boundary": int(
                                "grid_boundary=1" in outcome_tuning
                            ),
                            "propensity_tuning_at_boundary": int(
                                "grid_boundary=1"
                                in propensity_tuning[(learner, regime)]
                            ),
                            "bootstrap_exceedances": int(
                                round(test["p_value"] * (bootstrap_draws + 1) - 1)
                            ),
                            "bootstrap_plus_one_numerator": int(
                                round(test["p_value"] * (bootstrap_draws + 1))
                            ),
                            **test,
                        }
                    )

    seed_metadata = {
        "crossfit_folds": crossfit_seed,
        "propensity_root_seeds": propensity_seeds,
        "outcome_root_seeds": outcome_seeds,
        "multiplier_seeds": multiplier_seeds,
    }
    return predictions, pd.DataFrame(rows), seed_metadata


def descriptive_contrasts(
    sample: pd.DataFrame, *, bootstrap_draws: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        seed = core.stable_seed(APPLICATION_VERSION, "descriptive", outcome)
        rng = np.random.default_rng(seed)
        group0 = sample.loc[sample["D"] == 0, outcome].to_numpy(dtype=float)
        group1 = sample.loc[sample["D"] == 1, outcome].to_numpy(dtype=float)
        draws0 = rng.integers(0, len(group0), size=(bootstrap_draws, len(group0)))
        draws1 = rng.integers(0, len(group1), size=(bootstrap_draws, len(group1)))
        differences = group1[draws1].mean(axis=1) - group0[draws0].mean(axis=1)
        rows.append(
            {
                "outcome": outcome,
                "aligned_n": len(group0),
                "aligned_mean": float(np.mean(group0)),
                "aligned_sd": float(np.std(group0, ddof=1)),
                "aligned_median": float(np.median(group0)),
                "maximal_n": len(group1),
                "maximal_mean": float(np.mean(group1)),
                "maximal_sd": float(np.std(group1, ddof=1)),
                "maximal_median": float(np.median(group1)),
                "mean_difference_maximal_minus_aligned": float(
                    np.mean(group1) - np.mean(group0)
                ),
                "bootstrap_ci_lower": float(np.quantile(differences, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(differences, 0.975)),
                "bootstrap_draws": bootstrap_draws,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(rows)


def differential_word_statistics(
    sample: pd.DataFrame,
    *,
    minimum_document_frequency: int,
    bootstrap_draws: int,
) -> pd.DataFrame:
    def tokenize(text: str) -> set[str]:
        tokens = core.WORD_RE.findall(core.normalize_text(text).lower())
        return {
            token for token in tokens if len(token) >= 2 and token not in WORD_STOPWORDS
        }

    term_sets = [tokenize(text) for text in sample["early_text"]]
    document_frequency = Counter(term for terms in term_sets for term in terms)
    vocabulary = sorted(
        term
        for term, count in document_frequency.items()
        if count >= minimum_document_frequency
    )
    incidence = np.array(
        [[term in terms for term in vocabulary] for terms in term_sets], dtype=float
    )
    d = sample["D"].to_numpy(dtype=int)
    aligned = incidence[d == 0]
    maximal = incidence[d == 1]
    count0 = aligned.sum(axis=0)
    count1 = maximal.sum(axis=0)
    n0, n1 = len(aligned), len(maximal)

    def log_odds(c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
        return np.log((c1 + 0.5) / (n1 - c1 + 0.5)) - np.log(
            (c0 + 0.5) / (n0 - c0 + 0.5)
        )

    delta = log_odds(count0, count1)
    variance = (
        1.0 / (count1 + 0.5)
        + 1.0 / (n1 - count1 + 0.5)
        + 1.0 / (count0 + 0.5)
        + 1.0 / (n0 - count0 + 0.5)
    )
    standard_error = np.sqrt(variance)
    z_score = delta / standard_error
    seed = core.stable_seed(APPLICATION_VERSION, "word-bootstrap")
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap_draws, len(vocabulary)), dtype=float)
    for draw in range(bootstrap_draws):
        resample0 = aligned[rng.integers(0, n0, size=n0)].sum(axis=0)
        resample1 = maximal[rng.integers(0, n1, size=n1)].sum(axis=0)
        boot[draw] = log_odds(resample0, resample1)
    ci_lower, ci_upper = np.quantile(boot, [0.025, 0.975], axis=0)
    sign_stability = np.where(
        delta >= 0,
        np.mean(boot > 0, axis=0),
        np.mean(boot < 0, axis=0),
    )
    frame = pd.DataFrame(
        {
            "term": vocabulary,
            "aligned_documents": count0.astype(int),
            "maximal_documents": count1.astype(int),
            "aligned_document_rate": count0 / n0,
            "maximal_document_rate": count1 / n1,
            "log_odds_maximal_minus_aligned": delta,
            "standard_error": standard_error,
            "z_score": z_score,
            "abs_z_score": np.abs(z_score),
            "bootstrap_ci_lower": ci_lower,
            "bootstrap_ci_upper": ci_upper,
            "bootstrap_sign_stability": sign_stability,
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": seed,
            "favored_population": np.where(delta >= 0, "maximal", "aligned"),
        }
    )
    return frame.sort_values("abs_z_score", ascending=False).reset_index(drop=True)


def selection_sensitivity_analysis(
    eligible_records: list[dict[str, Any]], *, bootstrap_draws: int
) -> pd.DataFrame:
    specifications = [
        ("main", MAIN_SELECTION_SALT, 1),
        *[
            (f"alternative_{index + 1:02d}", salt, 0)
            for index, salt in enumerate(SELECTION_SENSITIVITY_SALTS)
        ],
    ]
    propensity_seed = core.stable_seed(
        APPLICATION_VERSION, "propensity", "ridge", "no_split"
    )
    outcome_seed = core.stable_seed(
        APPLICATION_VERSION,
        "outcome",
        "mean_satisfaction",
        "ridge",
        "no_split",
    )
    bootstrap_seed = core.stable_seed(
        APPLICATION_VERSION, "multiplier", "mean_satisfaction", "no_split"
    )
    rows: list[dict[str, Any]] = []
    for label, salt, is_primary in specifications:
        sample, sample_sha256 = outcome_blind_disjoint_sample(
            eligible_records, salt=salt
        )
        design, _ = build_text_design(sample)
        d = sample["D"].to_numpy(dtype=int)
        y = sample["mean_satisfaction"].to_numpy(dtype=float)
        propensity, propensity_tuning = core.fit_predictions(
            design,
            d,
            learner="ridge",
            regime="no_split",
            kind="propensity",
            folds=[],
            seed=propensity_seed,
        )
        fitted_outcome, outcome_tuning = core.fit_predictions(
            design,
            y,
            learner="ridge",
            regime="no_split",
            kind="outcome",
            folds=[],
            seed=outcome_seed,
            bounds=OUTCOMES["mean_satisfaction"],
        )
        score = (y - fitted_outcome) * (d - propensity)
        kernel = core.rbf_kernel_matrix(design, FIXED_BANDWIDTH)
        test = core.multiplier_test(
            score,
            kernel,
            bootstrap_draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        rows.append(
            {
                "selection": label,
                "selection_salt": salt,
                "is_primary": is_primary,
                "sample_sha256": sample_sha256,
                "n": len(sample),
                "n_aligned": int(np.sum(d == 0)),
                "n_maximal": int(np.sum(d == 1)),
                "aligned_mean_satisfaction": float(np.mean(y[d == 0])),
                "maximal_mean_satisfaction": float(np.mean(y[d == 1])),
                "mean_difference_maximal_minus_aligned": float(
                    np.mean(y[d == 1]) - np.mean(y[d == 0])
                ),
                "learner": "ridge",
                "regime": "no_split",
                "bandwidth_spec": "fixed_h_1",
                "bandwidth": FIXED_BANDWIDTH,
                "bootstrap_draws": bootstrap_draws,
                "bootstrap_seed": bootstrap_seed,
                "outcome_seed": outcome_seed,
                "propensity_seed": propensity_seed,
                "propensity_auc": float(roc_auc_score(d, propensity)),
                "propensity_brier": float(brier_score_loss(d, propensity)),
                "propensity_min": float(np.min(propensity)),
                "propensity_max": float(np.max(propensity)),
                "outcome_tuning": outcome_tuning,
                "propensity_tuning": propensity_tuning,
                "bootstrap_exceedances": int(
                    round(test["p_value"] * (bootstrap_draws + 1) - 1)
                ),
                "bootstrap_plus_one_numerator": int(
                    round(test["p_value"] * (bootstrap_draws + 1))
                ),
                **test,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=APPLICATION_DIR / "data" / "raw" / "casino-corpus",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=APPLICATION_DIR / "results_integrative_potential",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--descriptive-bootstrap-draws", type=int, default=10000)
    parser.add_argument("--word-bootstrap-draws", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        min(
            args.bootstrap_draws,
            args.descriptive_bootstrap_draws,
            args.word_bootstrap_draws,
        )
        < 99
    ):
        raise ValueError("Use at least 99 draws for every bootstrap")
    corpus_dir = args.corpus_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    archive_path = corpus_dir.parent / "casino-corpus.zip"
    if (
        not archive_path.is_file()
        or core.file_sha256(archive_path) != core.ARCHIVE_SHA256
    ):
        raise ValueError("The raw CaSiNo archive is missing or has the wrong checksum")
    extracted_member_hashes = core.verify_extracted_corpus(archive_path, corpus_dir)

    all_records, corpus_counts = load_integrative_potential_records(corpus_dir)
    eligible_records = [record for record in all_records if record["eligible_extreme"]]
    sample, sample_sha256 = frozen_main_sample(eligible_records)
    design, design_metadata = build_text_design(sample)
    d = sample["D"].to_numpy(dtype=int)
    dialogue_ids = sample["dialogue_id"].to_numpy(dtype=int)

    sample_path = results_dir / "integrative_potential_sample.csv"
    sample.to_csv(sample_path, index=False)
    np.savez_compressed(
        results_dir / "integrative_potential_design.npz",
        X=design,
        D=d,
        dialogue_id=dialogue_ids,
    )

    full_corpus = (
        pd.DataFrame(all_records).sort_values("dialogue_id").reset_index(drop=True)
    )
    full_corpus["selected_main"] = (
        full_corpus["dialogue_id"].isin(dialogue_ids).astype(int)
    )
    full_corpus["D"] = full_corpus["D"].astype("Int64")
    full_corpus.to_csv(
        results_dir / "integrative_potential_full_corpus.csv", index=False
    )

    nuisance_predictions, test_results, test_seed_metadata = fit_all_nuisance_and_tests(
        sample, design, bootstrap_draws=args.bootstrap_draws
    )
    nuisance_predictions.to_csv(
        results_dir / "integrative_potential_nuisance_predictions.csv", index=False
    )
    test_results.to_csv(
        results_dir / "integrative_potential_rkhs_results.csv", index=False
    )

    selection_sensitivity = selection_sensitivity_analysis(
        eligible_records, bootstrap_draws=args.bootstrap_draws
    )
    selection_sensitivity.to_csv(
        results_dir / "integrative_potential_selection_sensitivity.csv", index=False
    )
    contrasts = descriptive_contrasts(
        sample, bootstrap_draws=args.descriptive_bootstrap_draws
    )
    contrasts.to_csv(
        results_dir / "integrative_potential_descriptive_contrasts.csv", index=False
    )
    word_stats = differential_word_statistics(
        sample,
        minimum_document_frequency=MINIMUM_WORD_DOCUMENT_FREQUENCY,
        bootstrap_draws=args.word_bootstrap_draws,
    )
    word_stats.to_csv(
        results_dir / "integrative_potential_word_log_odds.csv", index=False
    )
    group_summary = (
        sample.groupby("population")
        .agg(
            dialogues=("dialogue_id", "size"),
            early_tokens_mean=("early_token_count", "mean"),
            early_tokens_sd=("early_token_count", "std"),
            mean_satisfaction_mean=("mean_satisfaction", "mean"),
            mean_satisfaction_sd=("mean_satisfaction", "std"),
            both_extremely_satisfied_mean=("both_extremely_satisfied", "mean"),
        )
        .reset_index()
    )
    group_summary.to_csv(
        results_dir / "integrative_potential_group_summary.csv", index=False
    )

    table_paths = [
        "integrative_potential_sample.csv",
        "integrative_potential_design.npz",
        "integrative_potential_full_corpus.csv",
        "integrative_potential_nuisance_predictions.csv",
        "integrative_potential_rkhs_results.csv",
        "integrative_potential_selection_sensitivity.csv",
        "integrative_potential_descriptive_contrasts.csv",
        "integrative_potential_word_log_odds.csv",
        "integrative_potential_group_summary.csv",
    ]
    artifact_hashes = {
        relative_path: core.file_sha256(results_dir / relative_path)
        for relative_path in table_paths
    }
    fold_payload = ",".join(
        f"{int(row.dialogue_id)}:{int(row.crossfit_fold)}"
        for row in nuisance_predictions[["dialogue_id", "crossfit_fold"]].itertuples(
            index=False
        )
    )
    fold_assignment_sha256 = hashlib.sha256(fold_payload.encode("utf-8")).hexdigest()
    analysis_summary = {
        "application_version": APPLICATION_VERSION,
        "source": {
            "dataset": "CaSiNo ConvoKit corpus",
            "download_url": (
                "https://zissou.infosci.cornell.edu/convokit/datasets/"
                "casino-corpus/casino-corpus.zip"
            ),
            "archive_sha256": core.ARCHIVE_SHA256,
            "extracted_member_sha256": extracted_member_hashes,
            "license": "CC BY 4.0",
            "citation": {
                "key": "chawla-etal-2021-casino",
                "doi": "10.18653/v1/2021.naacl-main.254",
                "url": "https://aclanthology.org/2021.naacl-main.254/",
            },
        },
        "corpus_counts": corpus_counts,
        "sample": {
            "selection_salt": MAIN_SELECTION_SALT,
            "selection_rule": (
                "Sort eligible dialogues by SHA-256 of '<salt>|<dialogue_id>'; "
                "greedily retain only participant-disjoint dialogues."
            ),
            "dialogues": len(sample),
            "aligned": int(np.sum(d == 0)),
            "maximal": int(np.sum(d == 1)),
            "unique_participants": 2 * len(sample),
            "sorted_dialogue_id_sha256": sample_sha256,
        },
        "estimand": {
            "D0": "identical priority orderings; maximum attainable joint points 36",
            "D1": "maximal integrative potential; maximum attainable joint points 42",
            "excluded": "intermediate maximum attainable joint points 39",
            "primary_outcome": "mean of the two coded 1-5 satisfaction ratings",
            "ordinal_sensitivity": (
                "indicator that both participants report Extremely satisfied (5)"
            ),
            "covariates": "fixed hash of first six natural pre-action turns only",
            "null": "E(Y|X,D=1)=E(Y|X,D=0), P_X-almost everywhere",
        },
        "design_matrix": design_metadata,
        "kernel": {
            "primary": "Gaussian RBF with bandwidth fixed at h=1",
            "median_sensitivity_bandwidth": core.median_bandwidth(design),
        },
        "run_configuration": {
            "bootstrap_draws": args.bootstrap_draws,
            "descriptive_bootstrap_draws": args.descriptive_bootstrap_draws,
            "word_bootstrap_draws": args.word_bootstrap_draws,
            "word_minimum_pooled_document_frequency": MINIMUM_WORD_DOCUMENT_FREQUENCY,
            "word_filtered_scenario_nouns": sorted(SCENARIO_NOUNS),
            "selection_sensitivity_salts": list(SELECTION_SENSITIVITY_SALTS),
            "crossfit_folds": 5,
            "crossfit_fold_assignment_sha256": fold_assignment_sha256,
            "learners": list(LEARNERS),
            "regimes": list(REGIMES),
            "model_configuration": "core.py",
        },
        "seeds": {
            "derivation": "first four bytes of SHA-256 over pipe-separated labels",
            **test_seed_metadata,
            "descriptive_bootstrap": {
                outcome: core.stable_seed(APPLICATION_VERSION, "descriptive", outcome)
                for outcome in OUTCOMES
            },
            "word_bootstrap": core.stable_seed(APPLICATION_VERSION, "word-bootstrap"),
        },
        "primary_fixed_h_tests": test_results[test_results["is_primary"] == 1][
            [
                "learner",
                "statistic",
                "critical_value_0_05",
                "p_value",
                "reject_0_05",
            ]
        ].to_dict(orient="records"),
        "ordinal_fixed_h_tests": test_results[
            (test_results["outcome"] == "both_extremely_satisfied")
            & (test_results["regime"] == "no_split")
            & (test_results["bandwidth_spec"] == "fixed_h_1")
        ][
            [
                "learner",
                "statistic",
                "critical_value_0_05",
                "p_value",
                "reject_0_05",
            ]
        ].to_dict(orient="records"),
        "selection_sensitivity": {
            "selections": len(selection_sensitivity),
            "rejections_0_05": int(selection_sensitivity["reject_0_05"].sum()),
            "n_range": [
                int(selection_sensitivity["n"].min()),
                int(selection_sensitivity["n"].max()),
            ],
            "p_value_range": [
                float(selection_sensitivity["p_value"].min()),
                float(selection_sensitivity["p_value"].max()),
            ],
        },
        "outputs": {"tables": table_paths, "sha256": artifact_hashes},
        "caveats": [
            "This population/outcome design was selected after exploratory screening; p-values are descriptive rather than preregistered confirmatory evidence.",
            "Priority features are excluded because they determine D exactly and would eliminate common support.",
            "Opening language is post-assignment behavior and may mediate the association between integrative potential and satisfaction.",
            "The participant-disjoint greedy extraction yields a selected dependent subset with unequal realized group counts and is not the manuscript's pooled-iid random-count experiment.",
            "The 1-5 satisfaction score is ordinal; the both-extremely-satisfied binary outcome is reported as a scale sensitivity.",
            "The word contrasts describes filtered marginal unigram presence P(W|D), not the tested conditional-mean contrast.",
            "Software nuisance learners have not been certified against the manuscript's learner-specific finite-sample conditions.",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "script_sha256": core.file_sha256(Path(__file__).resolve()),
            "core_script_sha256": core.file_sha256(APPLICATION_DIR / "core.py"),
        },
    }
    summary_path = results_dir / "integrative_potential_analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(core.to_builtin(analysis_summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Analyzed {len(sample)} participant-disjoint preference-extreme dialogues "
        f"({int(np.sum(d == 0))} aligned, {int(np.sum(d == 1))} maximal)."
    )
    print(f"Results: {results_dir}")
    for row in test_results[test_results["is_primary"] == 1].itertuples(index=False):
        print(f"  {row.learner}: T={row.statistic:.4f}, p={row.p_value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
