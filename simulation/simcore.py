"""Data generation, nuisance estimation, and RKHS two-sample tests."""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.stats import norm
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


SCHEMA_VERSION = "1"
SUPPORTED_SCENARIOS = (
    "additive",
    "product",
    "composition",
    "friedman",
    "neural_composition",
)
SUPPORTED_DESIGNS = ("pooled_random_shift", "late_iv")
LATE_OPTIONS = {
    "always_taker_probability": 0.2,
    "complier_probability": 0.6,
    "latent_outcome_coefficient": 2.0,
    "noncomplier_treatment_effect": 1.0,
}
SUPPORTED_LEARNERS = ("gam", "rf", "lasso", "dnn", "oracle")
SUPPORTED_METHODS = (
    "orthogonal_nosplit",
    "orthogonal_single_split",
    "orthogonal_crossfit",
    "residual",
    "yan",
    "oracle_score",
)


def stable_seed(*parts: object) -> int:
    """Return a process-independent 32-bit seed from structured components."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ScenarioCalibration:
    scenario: str
    p: int
    design: str
    rho: float
    calibration_n: int
    calibration_seed: int
    raw_base_mean: float
    raw_base_sd: float
    base_target_mean: float
    base_target_sd: float
    base_scale: float
    raw_signal_l2: float
    signal_target_l2: float
    signal_scale: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimData:
    X1: np.ndarray
    y1: np.ndarray
    X0: np.ndarray
    y0: np.ndarray
    X: np.ndarray
    y: np.ndarray
    d: np.ndarray
    e_true: np.ndarray
    mu_true: np.ndarray
    m1_x1: np.ndarray
    m0_x1: np.ndarray
    m1_x0: np.ndarray
    m0_x0: np.ndarray
    signal_x1: np.ndarray
    signal_x0: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class PredictionFit:
    prediction: np.ndarray
    train_prediction: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class TestResult:
    reject: int
    p_value: float
    statistic: float
    critical_value: float
    diagnostics: dict[str, Any]


def toeplitz_covariance(p: int, rho: float) -> np.ndarray:
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])


def draw_gaussian_covariates(
    n: int, p: int, rho: float, rng: np.random.Generator
) -> np.ndarray:
    lower = np.linalg.cholesky(toeplitz_covariance(p, rho))
    return rng.standard_normal((n, p)) @ lower.T


def draw_reference_covariates(
    n: int,
    p: int,
    rho: float,
    design: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if design not in SUPPORTED_DESIGNS:
        raise ValueError(f"Unknown design: {design}")
    return draw_gaussian_covariates(n, p, rho, rng)


def pooled_random_propensity(X: np.ndarray) -> np.ndarray:
    """Propensity with population mean 1/2 and range [0.35, 0.70]."""

    if X.shape[1] < 2:
        raise ValueError("The pooled random-shift design requires p >= 2.")
    uniform_scores = 2.0 * norm.cdf(X[:, :2]) - 1.0
    propensity = (
        0.5
        + 0.10 * uniform_scores[:, 0]
        + 0.15 * (uniform_scores[:, 1] ** 2 - 1.0 / 3.0)
    )
    if np.any(propensity < 0.35 - 1e-12) or np.any(propensity > 0.70 + 1e-12):
        raise RuntimeError("The propensity left its theoretical bounds.")
    return propensity


def dimension_signal_constant(p: int) -> float:
    if p == 20:
        return 1.0
    if p == 50:
        return 2.0
    return math.sqrt(p / 20.0)


def _g_component(x: np.ndarray, cycle: int) -> np.ndarray:
    if cycle == 0:
        return np.sin(np.pi * x)
    if cycle == 1:
        return 0.8 * np.cos(2.0 * x)
    if cycle == 2:
        return 0.6 * (x**2 - 1.0)
    if cycle == 3:
        return 0.7 * np.sin(2.5 * x + 0.5)
    if cycle == 4:
        return 0.5 * np.cos(1.2 * x - 0.3)
    raise AssertionError("unreachable cycle")


def _h_component(x: np.ndarray, cycle: int) -> np.ndarray:
    if cycle == 0:
        return 0.8 * np.sin(1.3 * x + 0.2)
    if cycle == 1:
        return 0.6 * np.cos(1.7 * x - 0.4)
    if cycle == 2:
        return 0.7 * (x**2 - 0.5) * np.sin(x)
    if cycle == 3:
        return 0.9 * np.sin(0.8 * x) * np.cos(0.5 * x)
    if cycle == 4:
        return x**3 / 6.0
    raise AssertionError("unreachable cycle")


def raw_regression_components(
    X: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Uncalibrated baseline and perturbation for each regression scenario."""

    n, p = X.shape
    if p < 8:
        raise ValueError("The regression scenarios require p >= 8.")
    g_sum = np.zeros(n)
    h_sum = np.zeros(n)
    g_product_sum = np.zeros(n)
    h_product_sum = np.zeros(n)
    previous_g: Optional[np.ndarray] = None
    previous_h: Optional[np.ndarray] = None
    pair_count = 0
    for j in range(p):
        g_j = _g_component(X[:, j], j % 5)
        h_j = _h_component(X[:, j], j % 5)
        g_sum += g_j
        h_sum += h_j
        if j % 2 == 0:
            previous_g = g_j
            previous_h = h_j
        else:
            assert previous_g is not None and previous_h is not None
            g_product_sum += previous_g * g_j
            h_product_sum += previous_h * h_j
            pair_count += 1

    additive_signal = dimension_signal_constant(p) * h_sum / math.sqrt(p)
    product_normalizer = math.sqrt(max(pair_count, 1))
    product_base = g_product_sum / product_normalizer
    product_signal = h_product_sum / product_normalizer
    composition_base = np.sin(g_sum / math.sqrt(p))
    composition_signal = np.sin(h_sum / math.sqrt(p))

    uniform = norm.cdf(X[:, :5])
    centered_uniform = 2.0 * uniform - 1.0
    friedman_base = (
        10.0 * np.sin(np.pi * uniform[:, 0] * uniform[:, 1])
        + 20.0 * (uniform[:, 2] - 0.5) ** 2
        + 10.0 * uniform[:, 3]
        + 5.0 * uniform[:, 4]
    )
    friedman_signal = np.sin(
        np.pi * centered_uniform[:, 0] * centered_uniform[:, 1] * centered_uniform[:, 2]
    ) + 0.75 * np.tanh(
        2.0 * centered_uniform[:, 1] * centered_uniform[:, 3] * centered_uniform[:, 4]
    )

    u = np.tanh(X[:, :8] / math.sqrt(2.0))
    a1 = np.tanh(1.1 * u[:, 0] - 0.9 * u[:, 1] + 0.6 * u[:, 2])
    a2 = np.tanh(-0.8 * u[:, 1] + 1.0 * u[:, 3] + 0.7 * u[:, 4])
    a3 = np.tanh(0.9 * u[:, 2] - 0.7 * u[:, 5] + 1.1 * u[:, 6])
    a4 = np.tanh(0.8 * u[:, 0] + 0.6 * u[:, 6] - 1.0 * u[:, 7])
    b1 = np.tanh(1.2 * a1 - 0.8 * a2 + 0.7 * a3)
    b2 = np.tanh(-0.6 * a1 + 1.1 * a3 + 0.9 * a4)
    neural_base = 2.0 * b1 + 1.5 * b2
    neural_signal = 1.5 * np.tanh(1.1 * a1 + 0.8 * a2 - 0.9 * a4) + np.tanh(
        -0.7 * a1 + 1.2 * a3 + 0.8 * a4
    )
    return {
        "additive": (g_sum, additive_signal),
        "product": (product_base, product_signal),
        "composition": (composition_base, composition_signal),
        "friedman": (friedman_base, friedman_signal),
        "neural_composition": (neural_base, neural_signal),
    }


def calibrate_scenarios(
    p: int,
    rho: float,
    design: str,
    scenarios: Sequence[str],
    calibration_n: int,
    seed: int,
) -> dict[str, ScenarioCalibration]:
    """Match baseline mean/SD and signal L2 norm to the additive scenario."""

    unknown = set(scenarios) - set(SUPPORTED_SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    if calibration_n < 1_000:
        raise ValueError("calibration_n must be at least 1000")
    rng = np.random.default_rng(seed)
    X = draw_reference_covariates(calibration_n, p, rho, design, rng)
    raw = raw_regression_components(X)
    additive_base, additive_signal = raw["additive"]
    target_base_mean = float(np.mean(additive_base))
    target_base_sd = float(np.std(additive_base, ddof=0))
    target_signal_l2 = float(np.sqrt(np.mean(additive_signal**2)))
    if target_base_sd <= 0 or target_signal_l2 <= 0:
        raise RuntimeError("Degenerate additive calibration target")

    output: dict[str, ScenarioCalibration] = {}
    for scenario in scenarios:
        raw_base, raw_signal = raw[scenario]
        raw_base_mean = float(np.mean(raw_base))
        raw_base_sd = float(np.std(raw_base, ddof=0))
        raw_signal_l2 = float(np.sqrt(np.mean(raw_signal**2)))
        if raw_base_sd <= 0 or raw_signal_l2 <= 0:
            raise RuntimeError(f"Degenerate calibration for {scenario}")
        base_scale = 1.0 if scenario == "additive" else target_base_sd / raw_base_sd
        signal_scale = (
            1.0 if scenario == "additive" else target_signal_l2 / raw_signal_l2
        )
        output[scenario] = ScenarioCalibration(
            scenario=scenario,
            p=p,
            design=design,
            rho=rho,
            calibration_n=calibration_n,
            calibration_seed=seed,
            raw_base_mean=raw_base_mean,
            raw_base_sd=raw_base_sd,
            base_target_mean=target_base_mean,
            base_target_sd=target_base_sd,
            base_scale=float(base_scale),
            raw_signal_l2=raw_signal_l2,
            signal_target_l2=target_signal_l2,
            signal_scale=float(signal_scale),
        )
    return output


def evaluate_regression_structure(
    X: np.ndarray,
    scenario: str,
    calibration: ScenarioCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    raw_base, raw_signal = raw_regression_components(X)[scenario]
    if scenario == "additive":
        return raw_base, raw_signal
    base = calibration.base_target_mean + calibration.base_scale * (
        raw_base - calibration.raw_base_mean
    )
    signal = calibration.signal_scale * raw_signal
    return base, signal


def validate_treatment_noise_scales(value: Sequence[float]) -> tuple[float, float]:
    """Return the positive noise multipliers for control and treatment."""

    try:
        if isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError
        scales = tuple(float(scale) for scale in value)
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "treatment_noise_scales must contain two finite positive numbers."
        ) from error
    return scales


def validate_late_options(value: Optional[Mapping[str, Any]]) -> dict[str, float]:
    """Validate the fixed compliance and outcome parameters of the IV design."""

    if value is None:
        return dict(LATE_OPTIONS)
    if not isinstance(value, Mapping) or set(value) != set(LATE_OPTIONS):
        raise ValueError("late_options must specify all four fixed IV design parameters.")
    try:
        options = {key: float(value[key]) for key in LATE_OPTIONS}
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("late_options must contain finite fixed IV design values.") from error
    if any(not math.isclose(options[key], expected, rel_tol=0.0, abs_tol=1e-14)
           for key, expected in LATE_OPTIONS.items()):
        raise ValueError(f"late_options must match the fixed IV design: {LATE_OPTIONS}")
    return options


def generate_data(
    *,
    n: int,
    p: int,
    rho: float,
    sigma: float,
    delta: float,
    design: str,
    scenario: str,
    calibration: ScenarioCalibration,
    seed: int,
    treatment_noise_scales: Optional[Sequence[float]] = None,
    late_options: Optional[Mapping[str, Any]] = None,
) -> SimData:
    if design not in SUPPORTED_DESIGNS:
        raise ValueError(f"Unknown design: {design}")
    if n < 1:
        raise ValueError("The pooled sample size must be positive.")
    noise_scales = (
        ((1.0, 1.5) if design == "late_iv" else (1.0, 1.0))
        if treatment_noise_scales is None
        else validate_treatment_noise_scales(treatment_noise_scales)
    )
    rng = np.random.default_rng(seed)
    X = draw_reference_covariates(n, p, rho, design, rng)
    e_true = pooled_random_propensity(X)
    d = rng.binomial(1, e_true, size=n).astype(int)
    base_pool, signal_pool = evaluate_regression_structure(X, scenario, calibration)
    iv_diagnostics: dict[str, Any] = {}
    if design == "late_iv":
        options = validate_late_options(late_options)
        if sigma != 1.0 or noise_scales != (1.0, 1.5):
            raise ValueError("late_iv requires sigma=1 and treatment_noise_scales=[1,1.5].")
        always_probability = options["always_taker_probability"]
        complier_probability = options["complier_probability"]
        latent = rng.uniform(size=n)
        always_taker = latent <= always_probability
        complier = (latent > always_probability) & (
            latent <= always_probability + complier_probability
        )
        # Here d is the instrument A: all tests compare the two IV groups.
        treatment = (always_taker | ((d == 1) & complier)).astype(int)
        treatment_effect = (
            complier * delta * signal_pool
            + (~complier) * options["noncomplier_treatment_effect"]
        )
        # One independent draw has the observed-data law of epsilon_D when
        # epsilon_0 and epsilon_1 are independent standard normals.
        eps = rng.normal(size=n) * np.asarray(noise_scales)[treatment]
        y = (base_pool + options["latent_outcome_coefficient"] * (latent - 0.5)
             + treatment * treatment_effect + eps)
        regression_baseline = (
            base_pool + always_probability * options["noncomplier_treatment_effect"]
        )
        regression_shift = complier_probability * delta * signal_pool
        mu_true = regression_baseline + e_true * regression_shift
        treated_rates = [float(np.mean(treatment[d == arm]))
                         if np.any(d == arm) else math.nan for arm in (0, 1)]
        iv_diagnostics = {
            "group_variable": "instrument",
            "n_always_takers": int(np.sum(always_taker)),
            "n_compliers": int(np.sum(complier)),
            "n_never_takers": int(np.sum(~(always_taker | complier))),
            "n_treated": int(np.sum(treatment)),
            "n_treated_instrument0": int(np.sum(treatment[d == 0])),
            "n_treated_instrument1": int(np.sum(treatment[d == 1])),
            "treatment_rate_instrument0": treated_rates[0],
            "treatment_rate_instrument1": treated_rates[1],
            "first_stage_population": complier_probability,
            "first_stage_empirical": treated_rates[1] - treated_rates[0],
            "complier_fraction_empirical": float(np.mean(complier)),
            "late_empirical_l2": float(abs(delta) * np.sqrt(np.mean(signal_pool**2))),
            "reduced_form_empirical_l2": float(np.sqrt(np.mean(regression_shift**2))),
            **options,
        }
    else:
        eps = rng.normal(0.0, sigma, n)
        if noise_scales != (1.0, 1.0):
            eps *= np.asarray(noise_scales)[d]
        y = base_pool + d * delta * signal_pool + eps
        mu_true = base_pool + e_true * delta * signal_pool
        regression_baseline = base_pool
        regression_shift = delta * signal_pool

    group1 = d == 1
    group0 = ~group1
    X1, y1 = X[group1], y[group1]
    X0, y0 = X[group0], y[group0]
    base1, signal1 = base_pool[group1], signal_pool[group1]
    base0, signal0 = base_pool[group0], signal_pool[group0]
    m1_x1 = regression_baseline[group1] + regression_shift[group1]
    m0_x1 = regression_baseline[group1]
    m1_x0 = regression_baseline[group0] + regression_shift[group0]
    m0_x0 = regression_baseline[group0]
    n1 = int(np.count_nonzero(group1))
    n0 = int(np.count_nonzero(group0))

    def sample_variance(values: np.ndarray) -> float:
        return float(np.var(values, ddof=1)) if len(values) >= 2 else math.nan

    def sample_covariance(first: np.ndarray, second: np.ndarray) -> float:
        if len(first) < 2:
            return math.nan
        return float(np.cov(first, second, ddof=1)[0, 1])

    diagnostics = {
        "n": n,
        "n1": n1,
        "n0": n0,
        "target_pi": 0.5,
        "realized_group_fraction": n1 / n,
        "minimum_group_size": min(n1, n0),
        "e_min": float(np.min(e_true)),
        "e_max": float(np.max(e_true)),
        "e_mean": float(np.mean(e_true)),
        "signal_empirical_l2": float(np.sqrt(np.mean(signal_pool**2))),
        "base_empirical_sd": float(np.std(base_pool, ddof=0)),
        "pooled_mean_x1": float(np.mean(X[:, 0])),
        "pooled_variance_x2": sample_variance(X[:, 1]),
        "mean_x1_difference": (
            float(np.mean(X1[:, 0]) - np.mean(X0[:, 0])) if n1 and n0 else math.nan
        ),
        "variance_x2_difference": (
            sample_variance(X1[:, 1]) - sample_variance(X0[:, 1])
        ),
        "covariance_x1_x2_difference": (
            sample_covariance(X1[:, 0], X1[:, 1])
            - sample_covariance(X0[:, 0], X0[:, 1])
        ),
    }
    diagnostics.update(iv_diagnostics)
    if treatment_noise_scales is not None or design == "late_iv":
        diagnostics.update(
            noise_scale_control=noise_scales[0],
            noise_scale_treated=noise_scales[1],
            noise_sd_control=float(sigma * noise_scales[0]),
            noise_sd_treated=float(sigma * noise_scales[1]),
        )
    return SimData(
        X1=X1,
        y1=y1,
        X0=X0,
        y0=y0,
        X=X,
        y=y,
        d=d,
        e_true=e_true,
        mu_true=mu_true,
        m1_x1=m1_x1,
        m0_x1=m0_x1,
        m1_x0=m1_x0,
        m0_x0=m0_x0,
        signal_x1=signal1,
        signal_x0=signal0,
        diagnostics=diagnostics,
    )


def median_bandwidth(X: np.ndarray) -> float:
    norms = np.sum(X**2, axis=1)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * X @ X.T, 0.0)
    upper = squared[np.triu_indices(X.shape[0], k=1)]
    positive = upper[upper > 1e-24]
    if positive.size == 0:
        return 1.0
    bandwidth = float(np.sqrt(np.median(positive)))
    return bandwidth if np.isfinite(bandwidth) and bandwidth > 1e-12 else 1.0


def rbf_kernel_matrix(X: np.ndarray, bandwidth: float) -> np.ndarray:
    norms = np.sum(X**2, axis=1)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * X @ X.T, 0.0)
    return np.exp(-squared / (2.0 * bandwidth**2))


def rbf_paired_values(
    first: np.ndarray, second: np.ndarray, bandwidth: float
) -> np.ndarray:
    squared = np.sum((first - second) ** 2, axis=1)
    return np.exp(-squared / (2.0 * bandwidth**2))


def multiplier_test(
    scores: np.ndarray,
    kernel: np.ndarray,
    *,
    bootstrap_draws: int,
    alpha: float,
    seed: int,
) -> TestResult:
    """Empirically centered Gaussian multiplier test with plus-one p-value."""

    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if kernel.shape != (n, n):
        raise ValueError("Kernel dimension does not match the score vector.")
    observed_squared = float(scores @ kernel @ scores / n)
    observed = math.sqrt(max(observed_squared, 0.0))
    rng = np.random.default_rng(seed)
    multipliers = rng.standard_normal((n, bootstrap_draws))
    multipliers -= multipliers.mean(axis=0, keepdims=True)
    weighted = scores[:, None] * multipliers
    kernel_weighted = kernel @ weighted
    bootstrap_squared = np.sum(weighted * kernel_weighted, axis=0) / n
    bootstrap = np.sqrt(np.maximum(bootstrap_squared, 0.0))
    p_value = float(
        (1 + np.count_nonzero(bootstrap >= observed)) / (bootstrap_draws + 1)
    )
    critical = float(np.quantile(bootstrap, 1.0 - alpha, method="linear"))
    return TestResult(
        reject=int(p_value <= alpha),
        p_value=p_value,
        statistic=observed,
        critical_value=critical,
        diagnostics={
            "observed_squared": observed_squared,
            "bootstrap_mean": float(np.mean(bootstrap)),
            "bootstrap_sd": float(np.std(bootstrap, ddof=1)),
        },
    )


def yan_test(
    eta1: np.ndarray,
    X1: np.ndarray,
    eta0: np.ndarray,
    X0: np.ndarray,
    *,
    alpha: float,
) -> TestResult:
    """Paired Yan statistic with separate bandwidths and second-moment variance."""

    N1 = 2 * (len(eta1) // 2)
    N0 = 2 * (len(eta0) // 2)
    if N1 < 4 or N0 < 4:
        raise ValueError(
            "Yan's paired statistic requires at least four observations per group."
        )
    eta1 = np.asarray(eta1[:N1], dtype=float)
    eta0 = np.asarray(eta0[:N0], dtype=float)
    X1 = np.asarray(X1[:N1], dtype=float)
    X0 = np.asarray(X0[:N0], dtype=float)
    n1 = N1 // 2
    n0 = N0 // 2
    h1 = median_bandwidth(X1)
    h0 = median_bandwidth(X0)
    pair_kernel1 = rbf_paired_values(X1[:n1], X1[n1:], h1)
    pair_kernel0 = rbf_paired_values(X0[:n0], X0[n0:], h0)
    terms1 = eta1[:n1] * eta1[n1:] * pair_kernel1
    terms0 = eta0[:n0] * eta0[n0:] * pair_kernel0
    raw_statistic = float(np.mean(terms1) + np.mean(terms0))
    sigma1_squared = float(np.mean(terms1**2))
    sigma0_squared = float(np.mean(terms0**2))
    se_squared = sigma1_squared / n1 + sigma0_squared / n0
    if not np.isfinite(se_squared) or se_squared <= np.finfo(float).eps:
        raise RuntimeError("Yan variance estimate is degenerate.")
    z_value = raw_statistic / math.sqrt(se_squared)
    p_value = float(norm.sf(z_value))
    critical = float(norm.ppf(1.0 - alpha))
    return TestResult(
        reject=int(p_value <= alpha),
        p_value=p_value,
        statistic=float(z_value),
        critical_value=critical,
        diagnostics={
            "raw_statistic": raw_statistic,
            "sigma1_squared": sigma1_squared,
            "sigma0_squared": sigma0_squared,
            "standard_error": math.sqrt(se_squared),
            "bandwidth1": h1,
            "bandwidth0": h0,
            "pair_count1": n1,
            "pair_count0": n0,
        },
    )


def _target_transformer(regressor: Any) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
        check_inverse=False,
    )


def _prediction_fit_from_sklearn(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    extra: Optional[Mapping[str, Any]] = None,
) -> PredictionFit:
    prediction = np.asarray(model.predict(X_predict), dtype=float).reshape(-1)
    train_prediction = np.asarray(model.predict(X_train), dtype=float).reshape(-1)
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(train_prediction)):
        raise RuntimeError("Learner returned non-finite predictions.")
    diagnostics = {
        "train_mse": float(mean_squared_error(y_train, train_prediction)),
    }
    if extra:
        diagnostics.update(extra)
    return PredictionFit(prediction, train_prediction, diagnostics)


def fit_gam(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    seed: int,
    options: Mapping[str, Any],
) -> PredictionFit:
    del seed  # deterministic estimator
    n_knots = int(options.get("n_knots", 5))
    ridge_alphas = np.asarray(
        options.get("ridge_alphas", [0.01, 0.1, 1.0, 10.0, 100.0]), dtype=float
    )
    additive = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "splines",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=3,
                    knots="quantile",
                    extrapolation="linear",
                    include_bias=False,
                ),
            ),
            ("ridge", RidgeCV(alphas=ridge_alphas)),
        ]
    )
    model = _target_transformer(additive)
    model.fit(X_train, y_train)
    ridge = model.regressor_.named_steps["ridge"]
    spline = model.regressor_.named_steps["splines"]
    return _prediction_fit_from_sklearn(
        model,
        X_train,
        y_train,
        X_predict,
        {
            "selected_alpha": float(ridge.alpha_),
            "spline_features": int(spline.n_features_out_),
            "n_knots": n_knots,
        },
    )


def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    seed: int,
    options: Mapping[str, Any],
) -> PredictionFit:
    forest = RandomForestRegressor(
        n_estimators=int(options.get("n_estimators", 400)),
        max_features=options.get("max_features", "sqrt"),
        min_samples_leaf=int(options.get("min_samples_leaf", 5)),
        max_samples=float(options.get("max_samples", 0.8)),
        bootstrap=True,
        oob_score=True,
        n_jobs=int(options.get("n_jobs", 1)),
        random_state=seed,
    )
    model = _target_transformer(forest)
    model.fit(X_train, y_train)
    fitted_forest = model.regressor_
    return _prediction_fit_from_sklearn(
        model,
        X_train,
        y_train,
        X_predict,
        {
            "n_estimators": fitted_forest.n_estimators,
            "oob_r2": float(fitted_forest.oob_score_),
        },
    )


def fit_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    seed: int,
    options: Mapping[str, Any],
) -> PredictionFit:
    folds = max(2, min(int(options.get("cv_folds", 5)), X_train.shape[0]))
    alphas = np.asarray(
        options.get("alphas", np.logspace(-4, 0.5, 70).tolist()), dtype=float
    )
    lasso = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "lasso",
                LassoCV(
                    alphas=alphas,
                    cv=folds,
                    max_iter=int(options.get("max_iter", 20_000)),
                    tol=float(options.get("tol", 1e-5)),
                    n_jobs=1,
                    random_state=seed,
                    selection="cyclic",
                ),
            ),
        ]
    )
    model = _target_transformer(lasso)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)
    fitted_lasso = model.regressor_.named_steps["lasso"]
    convergence_warnings = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    return _prediction_fit_from_sklearn(
        model,
        X_train,
        y_train,
        X_predict,
        {
            "selected_alpha": float(fitted_lasso.alpha_),
            "nonzero_coefficients": int(np.count_nonzero(fitted_lasso.coef_)),
            "convergence_warnings": int(convergence_warnings),
        },
    )


class _MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Sequence[int], dropout: float):
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            layers.extend(
                [
                    torch.nn.Linear(previous, int(width)),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                ]
            )
            previous = int(width)
        layers.append(torch.nn.Linear(previous, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def _resolve_torch_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    seed: int,
    options: Mapping[str, Any],
) -> PredictionFit:
    """Select epochs on a holdout, then refit on all training observations."""

    device = _resolve_torch_device(str(options.get("device", "auto")))
    hidden_layers = tuple(int(x) for x in options.get("hidden_layers", [32, 16]))
    dropout = float(options.get("dropout", 0.10))
    learning_rate = float(options.get("learning_rate", 0.01))
    weight_decay = float(options.get("weight_decay", 1e-3))
    max_epochs = int(options.get("max_epochs", 400))
    patience = int(options.get("patience", 30))
    min_epochs = int(options.get("min_epochs", 20))
    validation_fraction = float(options.get("validation_fraction", 0.20))
    improvement_tolerance = float(options.get("improvement_tolerance", 1e-5))

    n = X_train.shape[0]
    validation_size = max(10, int(round(validation_fraction * n)))
    validation_size = min(validation_size, n - 10)
    if validation_size < 1:
        raise ValueError("DNN requires at least 11 training observations.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    validation_index = permutation[:validation_size]
    subtrain_index = permutation[validation_size:]

    x_mean = X_train[subtrain_index].mean(axis=0)
    x_sd = X_train[subtrain_index].std(axis=0, ddof=0)
    x_sd[~np.isfinite(x_sd) | (x_sd < 1e-8)] = 1.0
    y_mean = float(np.mean(y_train[subtrain_index]))
    y_sd = float(np.std(y_train[subtrain_index], ddof=0))
    if not np.isfinite(y_sd) or y_sd < 1e-8:
        y_sd = 1.0

    def tx(array: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> torch.Tensor:
        normalized = ((array - mean) / scale).astype(np.float32, copy=False)
        return torch.as_tensor(normalized, device=device)

    x_sub = tx(X_train[subtrain_index], x_mean, x_sd)
    x_val = tx(X_train[validation_index], x_mean, x_sd)
    y_sub = torch.as_tensor(
        ((y_train[subtrain_index] - y_mean) / y_sd).astype(np.float32),
        device=device,
    )
    y_val = torch.as_tensor(
        ((y_train[validation_index] - y_mean) / y_sd).astype(np.float32),
        device=device,
    )

    _seed_torch(seed)
    selection_model = _MLP(X_train.shape[1], hidden_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(
        selection_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        selection_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(selection_model(x_sub), y_sub)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selection_model.parameters(), max_norm=5.0)
        optimizer.step()
        selection_model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.nn.functional.mse_loss(selection_model(x_val), y_val).item()
            )
        if validation_loss < best_loss - improvement_tolerance:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break
    if best_epoch < 1 or not np.isfinite(best_loss):
        raise RuntimeError("DNN validation failed to select a finite epoch.")

    # Refit preprocessing and weights using the selected training duration.
    full_x_mean = X_train.mean(axis=0)
    full_x_sd = X_train.std(axis=0, ddof=0)
    full_x_sd[~np.isfinite(full_x_sd) | (full_x_sd < 1e-8)] = 1.0
    full_y_mean = float(np.mean(y_train))
    full_y_sd = float(np.std(y_train, ddof=0))
    if not np.isfinite(full_y_sd) or full_y_sd < 1e-8:
        full_y_sd = 1.0
    x_full = tx(X_train, full_x_mean, full_x_sd)
    x_predict_tensor = tx(X_predict, full_x_mean, full_x_sd)
    y_full = torch.as_tensor(
        ((y_train - full_y_mean) / full_y_sd).astype(np.float32), device=device
    )

    _seed_torch(stable_seed(seed, "refit"))
    model = _MLP(X_train.shape[1], hidden_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(best_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x_full), y_full)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_prediction = (
            model(x_full).detach().cpu().numpy().astype(float) * full_y_sd + full_y_mean
        )
        prediction = (
            model(x_predict_tensor).detach().cpu().numpy().astype(float) * full_y_sd
            + full_y_mean
        )
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(train_prediction)):
        raise RuntimeError("DNN returned non-finite predictions.")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return PredictionFit(
        prediction=np.asarray(prediction).reshape(-1),
        train_prediction=np.asarray(train_prediction).reshape(-1),
        diagnostics={
            "train_mse": float(mean_squared_error(y_train, train_prediction)),
            "validation_mse": float(best_loss * y_sd**2),
            "selected_epoch": int(best_epoch),
            "selection_epochs_run": int(epoch),
            "parameter_count": int(parameter_count),
            "validation_size": int(validation_size),
            "hidden_layers": list(hidden_layers),
            "dropout": dropout,
            "weight_decay": weight_decay,
            "device": str(device),
            "response_scale": full_y_sd,
        },
    )


def fit_regression(
    learner: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    *,
    seed: int,
    learner_options: Mapping[str, Any],
) -> PredictionFit:
    """Fit the requested nuisance learner."""

    started = time.perf_counter()
    if learner == "gam":
        fit = fit_gam(X_train, y_train, X_predict, seed, learner_options)
    elif learner == "rf":
        fit = fit_random_forest(X_train, y_train, X_predict, seed, learner_options)
    elif learner == "lasso":
        fit = fit_lasso(X_train, y_train, X_predict, seed, learner_options)
    elif learner == "dnn":
        fit = fit_dnn(X_train, y_train, X_predict, seed, learner_options)
    else:
        raise ValueError(f"Unknown fitted regression learner: {learner}")
    fit.diagnostics["fit_seconds"] = float(time.perf_counter() - started)
    fit.diagnostics["fallback_used"] = False
    return fit


def _propensity_features(X: np.ndarray, design: str) -> np.ndarray:
    if design in ("pooled_random_shift", "late_iv"):
        # The propensity depends on the first two covariates.
        return X[:, :2]
    return X


def fit_propensity(
    X_train: np.ndarray,
    d_train: np.ndarray,
    X_predict: np.ndarray,
    *,
    design: str,
    seed: int,
    clip: float,
    options: Mapping[str, Any],
) -> PredictionFit:
    X_train_features = _propensity_features(X_train, design)
    X_predict_features = _propensity_features(X_predict, design)
    if design in ("pooled_random_shift", "late_iv"):
        preprocessing: Any = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "splines",
                    SplineTransformer(
                        n_knots=int(options.get("n_knots", 5)),
                        degree=3,
                        knots="quantile",
                        extrapolation="linear",
                        include_bias=False,
                    ),
                ),
            ]
        )
    else:
        preprocessing = StandardScaler()
    folds = max(2, min(int(options.get("cv_folds", 5)), len(d_train) // 2))
    classifier = LogisticRegressionCV(
        Cs=np.asarray(options.get("Cs", np.logspace(-3, 2, 8).tolist())),
        cv=folds,
        scoring="neg_log_loss",
        solver="lbfgs",
        max_iter=int(options.get("max_iter", 3_000)),
        random_state=seed,
        n_jobs=1,
        refit=True,
    )
    model = Pipeline([("features", preprocessing), ("logistic", classifier)])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train_features, d_train)
    train_probability = np.asarray(model.predict_proba(X_train_features)[:, 1])
    probability = np.asarray(model.predict_proba(X_predict_features)[:, 1])
    probability = np.clip(probability, clip, 1.0 - clip)
    train_probability = np.clip(train_probability, clip, 1.0 - clip)
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("Propensity learner returned non-finite probabilities.")
    diagnostics = {
        "train_log_loss": float(log_loss(d_train, train_probability)),
        "train_brier": float(np.mean((d_train - train_probability) ** 2)),
        "selected_C": float(classifier.C_[0]),
        "convergence_warnings": int(
            sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        ),
        "fit_seconds": float(time.perf_counter() - started),
        "fallback_used": False,
    }
    if len(np.unique(d_train)) == 2:
        diagnostics["train_auc"] = float(roc_auc_score(d_train, train_probability))
    return PredictionFit(probability, train_probability, diagnostics)


def propensity_performance(
    d: np.ndarray, probability: np.ndarray, bins: int = 10
) -> dict[str, Any]:
    """Return prediction metrics and equal-frequency calibration bins."""

    d = np.asarray(d, dtype=int).reshape(-1)
    probability = np.asarray(probability, dtype=float).reshape(-1)
    if len(d) != len(probability) or len(d) == 0:
        raise ValueError(
            "Labels and propensity predictions must have equal positive length."
        )
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("Propensity predictions contain non-finite values.")
    if np.any(probability <= 0.0) or np.any(probability >= 1.0):
        raise RuntimeError("Propensity predictions must lie strictly inside (0,1).")
    if bins < 1:
        raise ValueError("The number of calibration bins must be positive.")

    order = np.argsort(probability, kind="mergesort")
    bin_id = np.empty(len(d), dtype=int)
    bin_id[order] = np.minimum(
        np.arange(len(d), dtype=int) * min(bins, len(d)) // len(d),
        min(bins, len(d)) - 1,
    )
    calibration_bins: list[dict[str, Any]] = []
    for index in range(int(np.max(bin_id)) + 1):
        selected = bin_id == index
        calibration_bins.append(
            {
                "bin": index + 1,
                "n": int(np.count_nonzero(selected)),
                "mean_prediction": float(np.mean(probability[selected])),
                "observed_rate": float(np.mean(d[selected])),
                "minimum_prediction": float(np.min(probability[selected])),
                "maximum_prediction": float(np.max(probability[selected])),
            }
        )
    return {
        "propensity_brier": float(np.mean((d - probability) ** 2)),
        "propensity_log_loss": float(log_loss(d, probability, labels=[0, 1])),
        "propensity_auc": (
            float(roc_auc_score(d, probability)) if len(np.unique(d)) == 2 else math.nan
        ),
        "propensity_min": float(np.min(probability)),
        "propensity_max": float(np.max(probability)),
        "propensity_calibration_bins": calibration_bins,
    }


def stratified_folds(d: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if folds < 2:
        raise ValueError("Cross-fitting requires at least two folds.")
    result = np.empty(len(d), dtype=int)
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        indices = np.flatnonzero(d == label)
        rng.shuffle(indices)
        result[indices] = np.arange(len(indices)) % folds
    return result


def stratified_single_split(d: np.ndarray, seed: int) -> np.ndarray:
    """Return a fixed 50/50 within-label split (0=estimation, 1=testing)."""

    labels = np.asarray(d, dtype=int)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError(
            "Single splitting requires a one-dimensional binary label vector."
        )
    counts = np.bincount(labels, minlength=2)
    if np.any(counts < 2):
        raise ValueError(
            "Single splitting requires at least two observations in each group."
        )
    assignment = stratified_folds(labels, 2, seed)
    for label in (0, 1):
        label_assignment = assignment[labels == label]
        estimation_n = int(np.count_nonzero(label_assignment == 0))
        testing_n = int(np.count_nonzero(label_assignment == 1))
        if estimation_n + testing_n != int(counts[label]):
            raise AssertionError("Single-split assignment does not cover its group.")
        if estimation_n < testing_n or estimation_n - testing_n > 1:
            raise AssertionError("Single-split group halves are not balanced.")
    return assignment


def _base_result_row(
    task: Mapping[str, Any],
    learner: str,
    method: str,
    dat: SimData,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "design": task["design"],
        "scenario": task["scenario"],
        "p": int(task["p"]),
        "n": int(task["n"]),
        "n1": int(dat.diagnostics["n1"]),
        "n0": int(dat.diagnostics["n0"]),
        "target_pi": float(dat.diagnostics["target_pi"]),
        "realized_group_fraction": float(dat.diagnostics["realized_group_fraction"]),
        "rho": float(task["rho"]),
        "sigma": float(task["sigma"]),
        "delta": float(task["delta"]),
        "rep": int(task["rep"]),
        "learner": learner,
        "method": method,
        "alpha": float(task["alpha"]),
        "bootstrap_draws": int(task["bootstrap_draws"]),
        "dgp_seed": int(task["dgp_seed"]),
        "fit_seed": int(task["fit_seed"]),
        "bootstrap_seed": int(task["bootstrap_seed"]),
        "status": "ok",
        "error": "",
        "fallback_used": False,
        "reject": np.nan,
        "p_value": np.nan,
        "statistic": np.nan,
        "critical_value": np.nan,
        "runtime_seconds": np.nan,
        "bandwidth": np.nan,
        "mu_mse": np.nan,
        "propensity_mse": np.nan,
        "propensity_brier": np.nan,
        "propensity_log_loss": np.nan,
        "propensity_auc": np.nan,
        "train_mse": np.nan,
        "validation_mse": np.nan,
        "selected_epoch": np.nan,
        "parameter_count": np.nan,
        "score_n": np.nan,
        "score_mean": np.nan,
        "score_sd": np.nan,
        "signal_empirical_l2": dat.diagnostics["signal_empirical_l2"],
        "e_true_min": dat.diagnostics["e_min"],
        "e_true_max": dat.diagnostics["e_max"],
        "diagnostics_json": "{}",
        **{
            key: dat.diagnostics[key]
            for key in (
                "noise_scale_control",
                "noise_scale_treated",
                "noise_sd_control",
                "noise_sd_treated",
            )
            if key in dat.diagnostics
        },
    }


def _successful_row(
    task: Mapping[str, Any],
    learner: str,
    method: str,
    dat: SimData,
    result: TestResult,
    *,
    runtime_seconds: float,
    bandwidth: Optional[float],
    score: Optional[np.ndarray],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    row = _base_result_row(task, learner, method, dat)
    row.update(
        {
            "reject": result.reject,
            "p_value": result.p_value,
            "statistic": result.statistic,
            "critical_value": result.critical_value,
            "runtime_seconds": runtime_seconds,
            "bandwidth": np.nan if bandwidth is None else bandwidth,
        }
    )
    if score is not None:
        row["score_n"] = int(len(score))
        row["score_mean"] = float(np.mean(score))
        row["score_sd"] = float(np.std(score, ddof=1))
    merged = dict(diagnostics)
    merged["test"] = result.diagnostics
    row["diagnostics_json"] = compact_json(merged)
    for key in (
        "mu_mse",
        "propensity_mse",
        "propensity_brier",
        "propensity_log_loss",
        "propensity_auc",
        "train_mse",
        "validation_mse",
        "selected_epoch",
        "parameter_count",
    ):
        if key in merged and merged[key] is not None:
            row[key] = merged[key]
    return row


def _failure_row(
    task: Mapping[str, Any],
    learner: str,
    method: str,
    dat: SimData,
    error: Exception,
    runtime_seconds: float,
) -> dict[str, Any]:
    row = _base_result_row(task, learner, method, dat)
    row.update(
        {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "runtime_seconds": runtime_seconds,
        }
    )
    return row


def _known_propensity(dat: SimData, mode: str) -> Optional[np.ndarray]:
    if mode == "oracle":
        return dat.e_true.copy()
    if mode == "known":
        raise ValueError(
            "The pooled random-shift experiment has a nonconstant propensity; "
            "use estimated or oracle mode."
        )
    if mode == "estimated":
        return None
    raise ValueError(f"Unknown propensity mode: {mode}")


def run_replication(
    task: Mapping[str, Any],
    calibration: ScenarioCalibration,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all requested methods for one learner/data replication task."""

    learner = str(task["learner"])
    methods = tuple(config["methods"])
    dat = generate_data(
        n=int(task["n"]),
        p=int(task["p"]),
        rho=float(task["rho"]),
        sigma=float(task["sigma"]),
        delta=float(task["delta"]),
        design=str(task["design"]),
        scenario=str(task["scenario"]),
        calibration=calibration,
        seed=int(task["dgp_seed"]),
        treatment_noise_scales=task.get(
            "treatment_noise_scales", config.get("treatment_noise_scales")
        ),
        late_options=task.get("late_options", config.get("late_options")),
    )
    pooled_kernel_methods = {
        "orthogonal_nosplit",
        "orthogonal_crossfit",
        "oracle_score",
    }
    if pooled_kernel_methods.intersection(methods):
        bandwidth_pool: Optional[float] = median_bandwidth(dat.X)
        kernel_pool: Optional[np.ndarray] = rbf_kernel_matrix(dat.X, bandwidth_pool)
    else:
        bandwidth_pool = None
        kernel_pool = None
    if "residual" in methods:
        bandwidth_1: Optional[float] = median_bandwidth(dat.X1)
        kernel_1: Optional[np.ndarray] = rbf_kernel_matrix(dat.X1, bandwidth_1)
    else:
        bandwidth_1 = None
        kernel_1 = None
    alpha = float(task["alpha"])
    bootstrap_draws = int(task["bootstrap_draws"])
    propensity_mode = str(config.get("propensity_mode", "estimated"))
    propensity_clip = float(config.get("propensity_clip", 0.01))
    learner_options = dict(config.get("learner_options", {}).get(learner, {}))
    propensity_options = dict(config.get("propensity_options", {}))
    rows: list[dict[str, Any]] = []

    if learner == "oracle":
        if "oracle_score" not in methods:
            return rows, dat.diagnostics
        assert bandwidth_pool is not None and kernel_pool is not None
        started = time.perf_counter()
        score = (dat.y - dat.mu_true) * (dat.d - dat.e_true)
        result = multiplier_test(
            score,
            kernel_pool,
            bootstrap_draws=bootstrap_draws,
            alpha=alpha,
            seed=int(task["bootstrap_seed"]),
        )
        rows.append(
            _successful_row(
                task,
                learner,
                "oracle_score",
                dat,
                result,
                runtime_seconds=time.perf_counter() - started,
                bandwidth=bandwidth_pool,
                score=score,
                diagnostics={
                    "mu_mse": 0.0,
                    "propensity_mse": 0.0,
                    **propensity_performance(dat.d, dat.e_true),
                },
            )
        )
        return rows, dat.diagnostics

    if learner not in SUPPORTED_LEARNERS:
        raise ValueError(f"Unknown learner: {learner}")

    if "orthogonal_nosplit" in methods:
        assert bandwidth_pool is not None and kernel_pool is not None
        started = time.perf_counter()
        try:
            mu_fit = fit_regression(
                learner,
                dat.X,
                dat.y,
                dat.X,
                seed=stable_seed(task["fit_seed"], "pooled_mu"),
                learner_options=learner_options,
            )
            e_hat = _known_propensity(dat, propensity_mode)
            propensity_diagnostics: dict[str, Any] = {"mode": propensity_mode}
            if e_hat is None:
                propensity_fit = fit_propensity(
                    dat.X,
                    dat.d,
                    dat.X,
                    design=str(task["design"]),
                    seed=stable_seed(task["fit_seed"], "pooled_e"),
                    clip=propensity_clip,
                    options=propensity_options,
                )
                e_hat = propensity_fit.prediction
                propensity_diagnostics.update(propensity_fit.diagnostics)
            score = (dat.y - mu_fit.prediction) * (dat.d - e_hat)
            result = multiplier_test(
                score,
                kernel_pool,
                bootstrap_draws=bootstrap_draws,
                alpha=alpha,
                seed=int(task["bootstrap_seed"]),
            )
            diagnostics = dict(mu_fit.diagnostics)
            diagnostics.update(
                {
                    "mu_mse": float(np.mean((mu_fit.prediction - dat.mu_true) ** 2)),
                    "propensity_mse": float(np.mean((e_hat - dat.e_true) ** 2)),
                    "propensity": propensity_diagnostics,
                    **propensity_performance(dat.d, e_hat),
                }
            )
            rows.append(
                _successful_row(
                    task,
                    learner,
                    "orthogonal_nosplit",
                    dat,
                    result,
                    runtime_seconds=time.perf_counter() - started,
                    bandwidth=bandwidth_pool,
                    score=score,
                    diagnostics=diagnostics,
                )
            )
        except Exception as error:
            rows.append(
                _failure_row(
                    task,
                    learner,
                    "orthogonal_nosplit",
                    dat,
                    error,
                    time.perf_counter() - started,
                )
            )

    if "orthogonal_single_split" in methods:
        started = time.perf_counter()
        try:
            split_seed = stable_seed(
                task["dgp_seed"], "orthogonal_single_split", "partition"
            )
            split_assignment = stratified_single_split(dat.d, split_seed)
            estimation_index = np.flatnonzero(split_assignment == 0)
            testing_index = np.flatnonzero(split_assignment == 1)
            mu_seed = stable_seed(task["fit_seed"], "orthogonal_single_split", "mu")
            propensity_seed = stable_seed(
                task["fit_seed"], "orthogonal_single_split", "e"
            )
            multiplier_seed = stable_seed(
                task["bootstrap_seed"], "orthogonal_single_split"
            )

            mu_fit = fit_regression(
                learner,
                dat.X[estimation_index],
                dat.y[estimation_index],
                dat.X[testing_index],
                seed=mu_seed,
                learner_options=learner_options,
            )
            known_propensity = _known_propensity(dat, propensity_mode)
            propensity_diagnostics: dict[str, Any] = {"mode": propensity_mode}
            if known_propensity is None:
                propensity_fit = fit_propensity(
                    dat.X[estimation_index],
                    dat.d[estimation_index],
                    dat.X[testing_index],
                    design=str(task["design"]),
                    seed=propensity_seed,
                    clip=propensity_clip,
                    options=propensity_options,
                )
                e_hat = propensity_fit.prediction
                propensity_diagnostics.update(propensity_fit.diagnostics)
            else:
                e_hat = known_propensity[testing_index]

            testing_X = dat.X[testing_index]
            testing_y = dat.y[testing_index]
            testing_d = dat.d[testing_index]
            bandwidth_testing = median_bandwidth(testing_X)
            kernel_testing = rbf_kernel_matrix(testing_X, bandwidth_testing)
            score = (testing_y - mu_fit.prediction) * (testing_d - e_hat)
            result = multiplier_test(
                score,
                kernel_testing,
                bootstrap_draws=bootstrap_draws,
                alpha=alpha,
                seed=multiplier_seed,
            )
            split_counts = {
                "n_estimation": int(len(estimation_index)),
                "n_testing": int(len(testing_index)),
                "n0_estimation": int(np.count_nonzero(dat.d[estimation_index] == 0)),
                "n0_testing": int(np.count_nonzero(testing_d == 0)),
                "n1_estimation": int(np.count_nonzero(dat.d[estimation_index] == 1)),
                "n1_testing": int(np.count_nonzero(testing_d == 1)),
            }
            diagnostics = dict(mu_fit.diagnostics)
            diagnostics.update(
                {
                    "mu_mse": float(
                        np.mean((mu_fit.prediction - dat.mu_true[testing_index]) ** 2)
                    ),
                    "propensity_mse": float(
                        np.mean((e_hat - dat.e_true[testing_index]) ** 2)
                    ),
                    "propensity": propensity_diagnostics,
                    "evaluation_n": int(len(testing_index)),
                    "bandwidth_source": "testing_pool",
                    "split": {
                        "scheme": ("fixed 50/50 split within each instrument A group"
                                   if task["design"] == "late_iv" else
                                   "fixed 50/50 split within each observed D group"),
                        "estimation_fold": 0,
                        "testing_fold": 1,
                        "split_seed": split_seed,
                        "mu_seed": mu_seed,
                        "propensity_seed": propensity_seed,
                        "multiplier_seed": multiplier_seed,
                        "assignment_sha256": hashlib.sha256(
                            split_assignment.astype(np.uint8).tobytes()
                        ).hexdigest(),
                        **split_counts,
                    },
                    **propensity_performance(testing_d, e_hat),
                }
            )
            rows.append(
                _successful_row(
                    task,
                    learner,
                    "orthogonal_single_split",
                    dat,
                    result,
                    runtime_seconds=time.perf_counter() - started,
                    bandwidth=bandwidth_testing,
                    score=score,
                    diagnostics=diagnostics,
                )
            )
        except Exception as error:
            rows.append(
                _failure_row(
                    task,
                    learner,
                    "orthogonal_single_split",
                    dat,
                    error,
                    time.perf_counter() - started,
                )
            )

    if "orthogonal_crossfit" in methods:
        assert bandwidth_pool is not None and kernel_pool is not None
        started = time.perf_counter()
        try:
            folds = int(config.get("crossfit_folds", 2))
            fold_id = stratified_folds(
                dat.d, folds, stable_seed(task["fit_seed"], "folds")
            )
            mu_hat = np.full(len(dat.y), np.nan)
            e_hat = np.full(len(dat.y), np.nan)
            fold_diagnostics: list[dict[str, Any]] = []
            for fold in range(folds):
                test_index = np.flatnonzero(fold_id == fold)
                train_index = np.flatnonzero(fold_id != fold)
                mu_fit = fit_regression(
                    learner,
                    dat.X[train_index],
                    dat.y[train_index],
                    dat.X[test_index],
                    seed=stable_seed(task["fit_seed"], "cf_mu", fold),
                    learner_options=learner_options,
                )
                mu_hat[test_index] = mu_fit.prediction
                known = _known_propensity(dat, propensity_mode)
                propensity_diag: dict[str, Any] = {"mode": propensity_mode}
                if known is None:
                    e_fit = fit_propensity(
                        dat.X[train_index],
                        dat.d[train_index],
                        dat.X[test_index],
                        design=str(task["design"]),
                        seed=stable_seed(task["fit_seed"], "cf_e", fold),
                        clip=propensity_clip,
                        options=propensity_options,
                    )
                    e_hat[test_index] = e_fit.prediction
                    propensity_diag.update(e_fit.diagnostics)
                else:
                    e_hat[test_index] = known[test_index]
                fold_diagnostics.append(
                    {
                        "fold": fold,
                        "n_train": len(train_index),
                        "n_test": len(test_index),
                        "mu": mu_fit.diagnostics,
                        "propensity": propensity_diag,
                    }
                )
            if not np.all(np.isfinite(mu_hat)) or not np.all(np.isfinite(e_hat)):
                raise RuntimeError(
                    "Cross-fitting left non-finite nuisance predictions."
                )
            score = (dat.y - mu_hat) * (dat.d - e_hat)
            result = multiplier_test(
                score,
                kernel_pool,
                bootstrap_draws=bootstrap_draws,
                alpha=alpha,
                seed=int(task["bootstrap_seed"]),
            )
            validation_values = [
                item["mu"].get("validation_mse")
                for item in fold_diagnostics
                if item["mu"].get("validation_mse") is not None
            ]
            epochs = [
                item["mu"].get("selected_epoch")
                for item in fold_diagnostics
                if item["mu"].get("selected_epoch") is not None
            ]
            parameters = [
                item["mu"].get("parameter_count")
                for item in fold_diagnostics
                if item["mu"].get("parameter_count") is not None
            ]
            diagnostics = {
                "mu_mse": float(np.mean((mu_hat - dat.mu_true) ** 2)),
                "propensity_mse": float(np.mean((e_hat - dat.e_true) ** 2)),
                "validation_mse": (
                    float(np.mean(validation_values)) if validation_values else None
                ),
                "selected_epoch": float(np.mean(epochs)) if epochs else None,
                "parameter_count": float(np.mean(parameters)) if parameters else None,
                "folds": fold_diagnostics,
                **propensity_performance(dat.d, e_hat),
            }
            rows.append(
                _successful_row(
                    task,
                    learner,
                    "orthogonal_crossfit",
                    dat,
                    result,
                    runtime_seconds=time.perf_counter() - started,
                    bandwidth=bandwidth_pool,
                    score=score,
                    diagnostics=diagnostics,
                )
            )
        except Exception as error:
            rows.append(
                _failure_row(
                    task,
                    learner,
                    "orthogonal_crossfit",
                    dat,
                    error,
                    time.perf_counter() - started,
                )
            )

    need_group0 = "residual" in methods or "yan" in methods
    group0_fit: Optional[PredictionFit] = None
    group0_error: Optional[Exception] = None
    group0_started = time.perf_counter()
    if need_group0:
        try:
            group0_fit = fit_regression(
                learner,
                dat.X0,
                dat.y0,
                dat.X1,
                seed=stable_seed(task["fit_seed"], "group0"),
                learner_options=learner_options,
            )
        except Exception as error:
            group0_error = error
    group0_seconds = time.perf_counter() - group0_started

    if "residual" in methods:
        assert bandwidth_1 is not None and kernel_1 is not None
        started = time.perf_counter()
        if group0_fit is None:
            assert group0_error is not None
            rows.append(
                _failure_row(
                    task,
                    learner,
                    "residual",
                    dat,
                    group0_error,
                    group0_seconds,
                )
            )
        else:
            try:
                score = dat.y1 - group0_fit.prediction
                result = multiplier_test(
                    score,
                    kernel_1,
                    bootstrap_draws=bootstrap_draws,
                    alpha=alpha,
                    seed=stable_seed(task["bootstrap_seed"], "residual"),
                )
                diagnostics = dict(group0_fit.diagnostics)
                diagnostics.update(
                    {
                        "mu_mse": float(
                            np.mean((group0_fit.prediction - dat.m0_x1) ** 2)
                        ),
                        "fit_role": "m0 trained in group 0, evaluated in group 1",
                    }
                )
                rows.append(
                    _successful_row(
                        task,
                        learner,
                        "residual",
                        dat,
                        result,
                        runtime_seconds=group0_seconds + time.perf_counter() - started,
                        bandwidth=bandwidth_1,
                        score=score,
                        diagnostics=diagnostics,
                    )
                )
            except Exception as error:
                rows.append(
                    _failure_row(
                        task,
                        learner,
                        "residual",
                        dat,
                        error,
                        group0_seconds + time.perf_counter() - started,
                    )
                )

    if "yan" in methods:
        group1_fit: Optional[PredictionFit] = None
        group1_error: Optional[Exception] = None
        group1_started = time.perf_counter()
        try:
            group1_fit = fit_regression(
                learner,
                dat.X1,
                dat.y1,
                dat.X0,
                seed=stable_seed(task["fit_seed"], "group1"),
                learner_options=learner_options,
            )
        except Exception as error:
            group1_error = error
        group1_seconds = time.perf_counter() - group1_started
        if group0_fit is None or group1_fit is None:
            failure = group0_error if group0_fit is None else group1_error
            assert failure is not None
            rows.append(
                _failure_row(
                    task,
                    learner,
                    "yan",
                    dat,
                    failure,
                    group0_seconds + group1_seconds,
                )
            )
        else:
            started = time.perf_counter()
            try:
                eta1 = dat.y1 - group0_fit.prediction
                eta0 = dat.y0 - group1_fit.prediction
                result = yan_test(eta1, dat.X1, eta0, dat.X0, alpha=alpha)
                diagnostics = {
                    "mu_mse": float(
                        0.5
                        * (
                            np.mean((group0_fit.prediction - dat.m0_x1) ** 2)
                            + np.mean((group1_fit.prediction - dat.m1_x0) ** 2)
                        )
                    ),
                    "group0_fit": group0_fit.diagnostics,
                    "group1_fit": group1_fit.diagnostics,
                }
                rows.append(
                    _successful_row(
                        task,
                        learner,
                        "yan",
                        dat,
                        result,
                        runtime_seconds=(
                            group0_seconds
                            + group1_seconds
                            + time.perf_counter()
                            - started
                        ),
                        bandwidth=None,
                        score=None,
                        diagnostics=diagnostics,
                    )
                )
            except Exception as error:
                rows.append(
                    _failure_row(
                        task,
                        learner,
                        "yan",
                        dat,
                        error,
                        group0_seconds + group1_seconds + time.perf_counter() - started,
                    )
                )

    return rows, dat.diagnostics


def package_versions() -> dict[str, str]:
    import pandas
    import scipy
    import sklearn

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        ),
    }
