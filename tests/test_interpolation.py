import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.interpolation import run_replication, test_scores as evaluate_scores


class InterpolationTests(unittest.TestCase):
    def test_multiplier_matches_centered_feature_vectors(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(17, 4))
        scores = rng.normal(size=17)
        gaussian = rng.normal(size=(17, 99))
        centered = gaussian - gaussian.mean(axis=0)
        result = evaluate_scores(scores, features @ features.T, centered)

        weighted_features = scores[:, None] * features
        statistic = np.linalg.norm(weighted_features.sum(axis=0)) / np.sqrt(17)
        centered_features = weighted_features - weighted_features.mean(axis=0)
        bootstrap = np.linalg.norm(gaussian.T @ centered_features, axis=1) / np.sqrt(17)
        pvalue = (1 + np.count_nonzero(bootstrap >= statistic)) / 100
        self.assertAlmostEqual(result["statistic"], statistic)
        self.assertEqual(result["pvalue"], pvalue)

    def test_zero_scores_have_pvalue_one(self):
        result = evaluate_scores(np.zeros(4), np.eye(4), np.ones((4, 19)))
        self.assertEqual(result["statistic"], 0)
        self.assertEqual(result["pvalue"], 1)
        self.assertEqual(result["reject"], 0)

    def test_interpolation_and_heldout_evaluation(self):
        with threadpool_limits(limits=1):
            rows = run_replication(0, bootstrap_draws=19)
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertTrue(np.isfinite(row["statistic"]))
            if row["method"] == "distance_full":
                self.assertEqual(row["score_rms"], 0)
                self.assertEqual(row["pvalue"], 1)
            else:
                self.assertGreater(row["score_rms"], 0)


if __name__ == "__main__":
    unittest.main()
