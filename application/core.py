"""Data validation, nuisance estimators, and centered multiplier calibration."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARCHIVE_SHA256 = "71448281f192ffcecfa05877e0aecb0b2f7c68db763b8cf8dc5f9d1010ece387"


CORPUS_MEMBERS = (
    "corpus.json",
    "index.json",
    "speakers.json",
    "conversations.json",
    "utterances.jsonl",
)


CONTROL_TEXT = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}


PRIORITY_SCORE = {"Low": 3, "Medium": 4, "High": 5}


SATISFACTION_SCORE = {
    "Extremely dissatisfied": 1,
    "Slightly dissatisfied": 2,
    "Undecided": 3,
    "Slightly satisfied": 4,
    "Extremely satisfied": 5,
}


ISSUES = ("Food", "Water", "Firewood")


LEARNERS = ("ridge", "lasso", "random_forest")


REGIMES = ("no_split", "crossfit_5fold")


WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


KEEP_STOPWORDS = {
    "i",
    "me",
    "my",
    "mine",
    "we",
    "us",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "both",
    "together",
}


WORD_STOPWORDS = set(ENGLISH_STOP_WORDS) - KEEP_STOPWORDS


SCENARIO_NOUNS = {
    "camp",
    "camping",
    "firewood",
    "food",
    "item",
    "items",
    "package",
    "packages",
    "supplies",
    "supply",
    "trip",
    "water",
}


WORD_STOPWORDS.update(SCENARIO_NOUNS)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_extracted_corpus(archive_path: Path, corpus_dir: Path) -> dict[str, str]:
    """Require every analyzed member to be byte-identical to the verified ZIP."""
    member_hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for filename in CORPUS_MEMBERS:
            extracted_path = corpus_dir / filename
            if not extracted_path.is_file():
                raise FileNotFoundError(
                    f"Missing extracted corpus member: {extracted_path}"
                )
            archived_bytes = archive.read(f"casino-corpus/{filename}")
            extracted_bytes = extracted_path.read_bytes()
            if extracted_bytes != archived_bytes:
                raise ValueError(
                    f"Extracted member differs from the verified archive: {extracted_path}"
                )
            member_hashes[filename] = hashlib.sha256(extracted_bytes).hexdigest()
    return member_hashes


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("’", "'")
    return " ".join(normalized.split())


def utterance_number(utterance_id: str) -> int:
    try:
        return int(utterance_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Unexpected utterance id: {utterance_id}") from error


def is_control_utterance(utterance: dict[str, Any]) -> bool:
    meta = utterance.get("meta", {})
    return (
        utterance.get("text") in CONTROL_TEXT
        or "data" in meta
        or "issue2youget" in meta
        or "issue2theyget" in meta
    )


def priority_by_issue(participant: dict[str, Any]) -> dict[str, int]:
    inverted = {
        issue: PRIORITY_SCORE[level]
        for level, issue in participant["value2issue"].items()
    }
    if set(inverted) != set(ISSUES):
        raise ValueError(f"Incomplete issue priorities: {inverted}")
    return inverted


def outcome_model(learner: str, seed: int, inner_folds: int) -> Any:
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    if learner == "ridge":
        return GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": np.logspace(-5, 6, 20)},
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            refit=True,
        )
    if learner == "lasso":
        return GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        Lasso(max_iter=300000, selection="cyclic"),
                    ),
                ]
            ),
            {"model__alpha": np.logspace(-5, 1, 20)},
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            refit=True,
        )
    if learner == "random_forest":
        return RandomForestRegressor(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=5,
            max_samples=0.8,
            bootstrap=True,
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown learner: {learner}")


def propensity_model(learner: str, seed: int, inner_folds: int) -> Any:
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    if learner == "ridge":
        return GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=20000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"model__C": np.logspace(-5, 3, 17)},
            cv=inner_cv,
            scoring="neg_log_loss",
            n_jobs=1,
            refit=True,
        )
    if learner == "lasso":
        return GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l1",
                            solver="liblinear",
                            max_iter=20000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"model__C": np.logspace(-5, 3, 17)},
            cv=inner_cv,
            scoring="neg_log_loss",
            n_jobs=1,
            refit=True,
        )
    if learner == "random_forest":
        return RandomForestClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=5,
            max_samples=0.8,
            bootstrap=True,
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown learner: {learner}")


def model_tuning_summary(model: Any) -> str:
    if isinstance(model, GridSearchCV):
        parameter = "alpha" if "model__alpha" in model.best_params_ else "C"
        key = "model__" + parameter
        grid = np.asarray(model.param_grid[key], dtype=float)
        best = float(model.best_params_[key])
        boundary = int(np.isclose(best, grid.min()) or np.isclose(best, grid.max()))
        return f"{parameter}={best:.8g};grid_boundary={boundary}"
    return (
        f"trees={model.n_estimators};min_leaf={model.min_samples_leaf};"
        f"max_features={model.max_features};max_samples={model.max_samples}"
    )


def fit_predictions(
    X: np.ndarray,
    target: np.ndarray,
    *,
    learner: str,
    regime: str,
    kind: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, str]:
    if kind not in {"outcome", "propensity"}:
        raise ValueError(f"Unknown fit kind: {kind}")
    factory: Callable[[str, int, int], Any] = (
        outcome_model if kind == "outcome" else propensity_model
    )

    def predict(model: Any, values: np.ndarray) -> np.ndarray:
        if kind == "propensity":
            result = model.predict_proba(values)[:, 1]
        else:
            result = model.predict(values)
            if bounds is not None:
                result = np.clip(result, bounds[0], bounds[1])
        return np.asarray(result, dtype=float)

    if regime == "no_split":
        model = factory(learner, seed, 5)
        with np.errstate(all="ignore"):
            model.fit(X, target)
            predictions = predict(model, X)
        tuning = model_tuning_summary(model)
    elif regime == "crossfit_5fold":
        predictions = np.full(len(target), np.nan, dtype=float)
        tuning_values: list[str] = []
        for fold_index, (train, test) in enumerate(folds):
            model = factory(learner, stable_seed(seed, fold_index), 4)
            with np.errstate(all="ignore"):
                model.fit(X[train], target[train])
                predictions[test] = predict(model, X[test])
            tuning_values.append(model_tuning_summary(model))
        tuning = " | ".join(tuning_values)
    else:
        raise ValueError(f"Unknown fitting regime: {regime}")

    if not np.isfinite(predictions).all():
        raise ValueError(f"Nonfinite {kind} predictions for {learner}/{regime}")
    if kind == "propensity" and np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("Propensity predictions outside [0,1].")
    return predictions, tuning


def median_bandwidth(X: np.ndarray) -> float:
    norms = np.sum(X**2, axis=1)
    with np.errstate(all="ignore"):
        squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * X @ X.T, 0.0)
    if not np.isfinite(squared).all():
        raise FloatingPointError("Nonfinite pairwise squared distance matrix.")
    upper = squared[np.triu_indices(X.shape[0], k=1)]
    positive = upper[upper > 1e-24]
    if positive.size == 0:
        return 1.0
    bandwidth = float(np.sqrt(np.median(positive)))
    return bandwidth if np.isfinite(bandwidth) and bandwidth > 1e-12 else 1.0


def rbf_kernel_matrix(X: np.ndarray, bandwidth: float) -> np.ndarray:
    norms = np.sum(X**2, axis=1)
    with np.errstate(all="ignore"):
        squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * X @ X.T, 0.0)
        kernel = np.exp(-squared / (2.0 * bandwidth**2))
    if not np.isfinite(kernel).all():
        raise FloatingPointError("Nonfinite Gaussian kernel matrix.")
    return kernel


def multiplier_test(
    scores: np.ndarray,
    kernel: np.ndarray,
    *,
    bootstrap_draws: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if kernel.shape != (n, n):
        raise ValueError("Kernel dimension does not match the score vector.")
    with np.errstate(all="ignore"):
        observed_squared = float(scores @ kernel @ scores / n)
    observed = math.sqrt(max(observed_squared, 0.0))
    rng = np.random.default_rng(seed)
    multipliers = rng.standard_normal((n, bootstrap_draws))
    multipliers -= multipliers.mean(axis=0, keepdims=True)
    weighted = scores[:, None] * multipliers
    with np.errstate(all="ignore"):
        bootstrap_squared = np.sum(weighted * (kernel @ weighted), axis=0) / n
    if not np.isfinite(observed_squared) or not np.isfinite(bootstrap_squared).all():
        raise FloatingPointError("Nonfinite RKHS statistic or bootstrap draw.")
    bootstrap = np.sqrt(np.maximum(bootstrap_squared, 0.0))
    p_value = float(
        (1 + np.count_nonzero(bootstrap >= observed)) / (bootstrap_draws + 1)
    )
    return {
        "statistic": observed,
        "statistic_squared": observed_squared,
        "critical_value_0_05": float(
            np.quantile(bootstrap, 1.0 - alpha, method="linear")
        ),
        "p_value": p_value,
        "reject_0_05": int(p_value <= alpha),
        "bootstrap_mean": float(np.mean(bootstrap)),
        "bootstrap_sd": float(np.std(bootstrap, ddof=1)),
    }


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
