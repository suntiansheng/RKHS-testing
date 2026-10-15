"""Check paired rejection-rate ratios and the supplementary figure panels."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import plot_crossfit_manuscript as report


def paired_frame(indicators):
    full, crossfit = np.asarray(indicators, dtype=int).T
    return pd.DataFrame([{
        "n": 120, "learner": "gam", "propensity_mode": "estimated",
        "amplitude": 2., "folds": 2, "replications": len(full),
        "full_power": full.mean(), "crossfit_power": crossfit.mean(),
        "full_only_rejections": np.count_nonzero((full == 1) & (crossfit == 0)),
        "crossfit_only_rejections": np.count_nonzero((full == 0) & (crossfit == 1)),
    }])


class CrossfitManuscriptTests(unittest.TestCase):
    def setUp(self):
        self.indicators = np.array([[1, 1]] * 21 + [[1, 0]] * 7
                                   + [[0, 1]] * 3 + [[0, 0]] * 69)

    def test_log_interval_matches_explicit_paired_sample_covariance(self):
        result = report.paired_ratios(paired_frame(self.indicators)).iloc[0]
        p, q = self.indicators.mean(axis=0)
        gradient = np.array([1 / p, -1 / q])
        covariance = np.cov(self.indicators, rowvar=False, ddof=1)
        se = np.sqrt(gradient @ covariance @ gradient / len(self.indicators))
        expected = (p / q) * np.exp(np.array([-1, 1]) * norm.ppf(.975) * se)
        self.assertAlmostEqual(result.rejection_rate_ratio, p / q)
        self.assertAlmostEqual(result.paired_log_mc_se, se, places=14)
        np.testing.assert_allclose([result.ci_lower, result.ci_upper], expected,
                                   rtol=1e-13, atol=0)
        self.assertEqual(result.both_rejections, 21)
        self.assertEqual(result.neither_rejections, 69)

    def test_reversing_methods_gives_reciprocal_interval(self):
        forward = report.paired_ratios(paired_frame(self.indicators)).iloc[0]
        reverse = report.paired_ratios(paired_frame(self.indicators[:, ::-1])).iloc[0]
        self.assertAlmostEqual(reverse.rejection_rate_ratio, 1 / forward.rejection_rate_ratio)
        self.assertAlmostEqual(reverse.ci_lower, 1 / forward.ci_upper)
        self.assertAlmostEqual(reverse.ci_upper, 1 / forward.ci_lower)

    def test_identical_paired_decisions_have_ratio_one_and_zero_mc_variance(self):
        indicators = np.array([[1, 1]] * 7 + [[0, 0]] * 93)
        result = report.paired_ratios(paired_frame(indicators)).iloc[0]
        self.assertEqual(result.rejection_rate_ratio, 1)
        self.assertEqual(result.paired_log_mc_se, 0)
        self.assertEqual(result.ci_lower, 1)
        self.assertEqual(result.ci_upper, 1)

    def test_zero_counts_are_rejected_without_pseudocounts(self):
        for pair in [(0, 1), (1, 0), (0, 0)]:
            with self.subTest(pair=pair):
                indicators = np.array([pair] * 5 + [[0, 0]] * 95)
                with self.assertRaisesRegex(ValueError, "positive rejection counts"):
                    report.paired_ratios(paired_frame(indicators))

    def test_invalid_paired_tables_are_rejected(self):
        frame = paired_frame(self.indicators)
        frame.loc[0, "full_only_rejections"] += 1
        with self.assertRaisesRegex(ValueError, "Inconsistent paired rejection counts"):
            report.paired_ratios(frame)
        frame = paired_frame(self.indicators)
        frame.loc[0, "full_power"] += .001
        with self.assertRaisesRegex(ValueError, "Invalid NORD rejection counts"):
            report.paired_ratios(frame)

    def test_all_panels_include_every_archived_amplitude(self):
        differences, _ = report.load_results(
            ROOT / "results/crossfit_comparison/n120_n160_k2")
        ratios = report.paired_ratios(differences)
        self.assertEqual(len(ratios), 144)
        selected = ratios[ratios.propensity_mode.eq("estimated")]
        expected_amplitudes = np.array([0, .8, 1.2, 1.6, 2, 2.5, 3.2, 4, 6])
        captured = []

        def inspect(fig, output, stem):
            captured.append(stem)
            try:
                self.assertEqual(len(fig.axes), 8)
                for ax, (n, learner) in zip(fig.axes, ((n, learner) for n in report.SIZES
                                                     for learner, _ in report.LEARNERS)):
                    data = selected[selected.n.eq(n) & selected.learner.eq(learner)]
                    data = data.sort_values("amplitude")
                    curve = ax.containers[0].lines[0]
                    np.testing.assert_array_equal(curve.get_xdata(), expected_amplitudes)
                    np.testing.assert_allclose(np.asarray(curve.get_ydata(), dtype=float),
                                               data.rejection_rate_ratio.to_numpy())
                    np.testing.assert_allclose(ax.get_ylim(), [.5, 1.7])
                    self.assertTrue((data.ci_lower > .5).all())
                    self.assertTrue((data.ci_upper < 1.7).all())
                    self.assertTrue(any(np.all(np.asarray(line.get_ydata()) == 1)
                                        for line in ax.lines))
            finally:
                plt.close(fig)

        with patch.object(report, "save", side_effect=inspect):
            report.power_ratio(ratios, Path("unused"))
        self.assertEqual(captured, ["simulation_crossfit_additive_power_ratio"])


if __name__ == "__main__":
    unittest.main()
