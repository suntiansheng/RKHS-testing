"""Scalar Generalised Covariance Measure of Shah and Peters (2020)."""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm


def gcm_from_moments(n, mean, sample_sd, *, alpha=0.05):
    """Equation (3), using an archived standard deviation with ddof=1.

    The paper divides the centered sum of squares by n, not n-1.
    Calibration is two-sided standard normal, not a Student t test.
    """
    if int(n) != n or n < 2 or not 0 < alpha < 1:
        raise ValueError("GCM requires n >= 2 and 0 < alpha < 1")
    if not np.isfinite([mean, sample_sd]).all() or sample_sd <= 0:
        raise ValueError("GCM requires finite moments and positive score variance")
    denominator = float(sample_sd) * math.sqrt((n - 1) / n)
    statistic = math.sqrt(n) * float(mean) / denominator
    p_value = float(2 * norm.sf(abs(statistic)))
    return dict(statistic=statistic, p_value=p_value,
                reject=int(p_value <= alpha),
                critical_value=float(norm.ppf(1 - alpha / 2)))


def gcm_test(scores, *, alpha=0.05):
    """Test the mean of fitted outcome-times-propensity residual products."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector with at least two entries")
    return gcm_from_moments(len(scores), scores.mean(), scores.std(ddof=1), alpha=alpha)
