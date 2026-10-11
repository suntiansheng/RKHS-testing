"""Scientific checks for the standalone cross-fit experiment."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import crossfit_comparison as experiment


class CrossfitComparisonTests(unittest.TestCase):
    def test_centered_multiplier_agrees_with_explicit_centered_features(self):
        rng = np.random.default_rng(915)
        features = rng.normal(size=(23, 7))
        scores = rng.normal(size=23)
        kernel = features @ features.T
        xi = rng.normal(size=(23, 199))
        centered_xi = xi-xi.mean(axis=0, keepdims=True)
        result = experiment.score_test(scores, kernel, centered_xi, .05)
        score_features = scores[:, None]*features
        centered_scores = score_features-score_features.mean(axis=0)
        explicit_bootstrap = np.sum((xi.T @ centered_scores / np.sqrt(23))**2, axis=1)
        observed = np.sum(score_features.mean(axis=0)**2)*23
        expected = (1+np.count_nonzero(explicit_bootstrap >= observed))/200
        self.assertEqual(result["p_value"], expected)
        self.assertAlmostEqual(result["statistic"]**2, observed, places=11)

    def test_oracle_zero_scores_use_weak_upper_tail(self):
        result = experiment.score_test(np.zeros(12), np.eye(12), np.zeros((12, 19)), .05)
        self.assertEqual(result["p_value"], 1.)
        self.assertEqual(result["reject"], 0)

    def test_centered_score_error_matches_direct_feature_variance(self):
        rng = np.random.default_rng(112)
        features = rng.normal(size=(31, 9))
        error = rng.normal(size=31)
        score_features = error[:, None]*features
        result = experiment.score_errors(error, features@features.T)
        centered = score_features-score_features.mean(axis=0)
        self.assertAlmostEqual(result["score_error_Q"], np.mean(np.sum(centered**2, axis=1)), places=12)
        self.assertAlmostEqual(result["score_error_R"]**2, 31*np.sum(score_features.mean(axis=0)**2), places=12)

    def test_folds_balanced_and_reproducible_without_labels(self):
        for n, k in [(200, 2), (201, 2), (200, 5)]:
            fold = experiment.independent_folds(n, k, 123)
            np.testing.assert_array_equal(fold, experiment.independent_folds(n, k, 123))
            self.assertLessEqual(np.ptp(np.bincount(fold)), 1)
            covered = np.zeros(n, dtype=int)
            for j in range(k):
                train, test = np.flatnonzero(fold!=j), np.flatnonzero(fold==j)
                self.assertEqual(len(np.intersect1d(train, test)), 0)
                covered[test] += 1
            np.testing.assert_array_equal(covered, np.ones(n))

    def test_existing_statistic_has_same_p_value(self):
        rng = np.random.default_rng(555)
        x = rng.normal(size=(45, 3))
        scores = rng.normal(size=45)
        kernel = experiment.core.rbf_kernel_matrix(x, 2.)
        xi = np.random.default_rng(19).standard_normal((45, 199))
        xi -= xi.mean(axis=0, keepdims=True)
        result = experiment.score_test(scores, kernel, xi, .05)
        previous = experiment.core.multiplier_test(scores, kernel, bootstrap_draws=199, alpha=.05, seed=19)
        self.assertEqual(result["p_value"], previous.p_value)
        self.assertAlmostEqual(result["critical_value"], previous.critical_value, places=12)

    def test_shard_validation_detects_incomplete_or_corrupt_result(self):
        config = {"learners":["gam"], "amplitudes":[0.], "null_replications":1,
                  "power_replications":1,"five_fold_sensitivity":False}
        rows = [{"n":200,"rep":0,"reject":0} for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            experiment.checkpoint_rows(output,"n200_r00000",rows,
                                       {"n":200,"rep":0,"fingerprint":"test","elapsed_seconds":1.})
            self.assertEqual(len(experiment.read_completed(output,config,"test")),1)
            path=output/"shards/n200_r00000.csv.gz"
            path.write_bytes(path.read_bytes()[:-3])
            self.assertEqual(len(experiment.read_completed(output,config,"test")),0)


if __name__ == "__main__":
    unittest.main()
