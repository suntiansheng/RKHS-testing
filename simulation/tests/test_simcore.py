from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from simcore import (  # noqa: E402
    PredictionFit,
    calibrate_scenarios,
    draw_reference_covariates,
    fit_dnn,
    fit_propensity,
    fit_regression,
    generate_data,
    median_bandwidth,
    multiplier_test,
    pooled_random_propensity,
    propensity_performance,
    raw_regression_components,
    rbf_kernel_matrix,
    rbf_paired_values,
    run_replication,
    stable_seed,
    stratified_single_split,
    yan_test,
)


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibrations = calibrate_scenarios(
            p=20,
            rho=0.5,
            design="pooled_random_shift",
            scenarios=[
                "additive",
                "product",
                "composition",
                "friedman",
                "neural_composition",
            ],
            calibration_n=20_000,
            seed=1001,
        )

    def test_additive_calibration_is_identity(self) -> None:
        calibration = self.calibrations["additive"]
        self.assertEqual(calibration.base_scale, 1.0)
        self.assertEqual(calibration.signal_scale, 1.0)

    def test_null_functions_are_exactly_equal(self) -> None:
        data = generate_data(
            n=60,
            p=20,
            rho=0.5,
            sigma=1.0,
            delta=0.0,
            design="pooled_random_shift",
            scenario="product",
            calibration=self.calibrations["product"],
            seed=44,
        )
        np.testing.assert_allclose(data.m1_x1, data.m0_x1, rtol=0, atol=0)
        np.testing.assert_allclose(data.m1_x0, data.m0_x0, rtol=0, atol=0)
        self.assertTrue(np.all(data.e_true >= 0.35))
        self.assertTrue(np.all(data.e_true <= 0.70))
        self.assertEqual(len(data.y1) + len(data.y0), 60)

    def test_common_random_numbers_across_delta(self) -> None:
        common = dict(
            n=60,
            p=20,
            rho=0.5,
            sigma=1.0,
            design="pooled_random_shift",
            scenario="composition",
            calibration=self.calibrations["composition"],
            seed=55,
        )
        null = generate_data(delta=0.0, **common)
        alternative = generate_data(delta=0.75, **common)
        np.testing.assert_array_equal(null.X, alternative.X)
        np.testing.assert_array_equal(null.d, alternative.d)
        np.testing.assert_array_equal(null.e_true, alternative.e_true)
        np.testing.assert_allclose(
            alternative.y1 - null.y1,
            0.75 * alternative.signal_x1,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(null.y0, alternative.y0)

    def test_product_and_composition_have_comparable_reference_scale(self) -> None:
        rng = np.random.default_rng(909)
        X = draw_reference_covariates(30_000, 20, 0.5, "pooled_random_shift", rng)
        raw = raw_regression_components(X)
        additive_base, additive_signal = raw["additive"]
        target_base_sd = np.std(additive_base)
        target_signal_l2 = np.sqrt(np.mean(additive_signal**2))
        for scenario in (
            "product",
            "composition",
            "friedman",
            "neural_composition",
        ):
            calibration = self.calibrations[scenario]
            base_raw, signal_raw = raw[scenario]
            base = calibration.base_target_mean + calibration.base_scale * (
                base_raw - calibration.raw_base_mean
            )
            signal = calibration.signal_scale * signal_raw
            self.assertLess(abs(np.std(base) / target_base_sd - 1.0), 0.08)
            self.assertLess(
                abs(np.sqrt(np.mean(signal**2)) / target_signal_l2 - 1.0), 0.08
            )

    def test_composition_formulas_are_fixed_and_centered_by_symmetry(self) -> None:
        zero = np.zeros((2, 20))
        raw_zero = raw_regression_components(zero)
        expected_friedman = 10.0 * np.sin(np.pi / 4.0) + 7.5
        np.testing.assert_allclose(
            raw_zero["friedman"][0], expected_friedman, rtol=0, atol=1e-14
        )
        np.testing.assert_array_equal(raw_zero["neural_composition"][0], np.zeros(2))
        np.testing.assert_array_equal(raw_zero["neural_composition"][1], np.zeros(2))

        rng = np.random.default_rng(20260802)
        X = rng.normal(size=(40, 20))
        raw_positive = raw_regression_components(X)
        raw_negative = raw_regression_components(-X)
        for scenario in ("friedman", "neural_composition"):
            np.testing.assert_allclose(
                raw_positive[scenario][1],
                -raw_negative[scenario][1],
                rtol=1e-12,
                atol=1e-12,
            )

    def test_pooled_random_shift_and_estimated_propensity_path(self) -> None:
        calibration = calibrate_scenarios(
            p=20,
            rho=0.5,
            design="pooled_random_shift",
            scenarios=["additive"],
            calibration_n=5000,
            seed=880,
        )["additive"]
        data = generate_data(
            n=120,
            p=20,
            rho=0.5,
            sigma=1.0,
            delta=0.0,
            design="pooled_random_shift",
            scenario="additive",
            calibration=calibration,
            seed=881,
        )
        self.assertGreaterEqual(np.min(data.e_true), 0.35 - 1e-12)
        self.assertLessEqual(np.max(data.e_true), 0.70 + 1e-12)
        self.assertEqual(data.diagnostics["n1"] + data.diagnostics["n0"], 120)
        np.testing.assert_allclose(
            data.e_true, pooled_random_propensity(data.X), rtol=0, atol=0
        )
        fit = fit_propensity(
            data.X,
            data.d,
            data.X,
            design="pooled_random_shift",
            seed=882,
            clip=0.01,
            options={"cv_folds": 3, "n_knots": 4, "max_iter": 2000},
        )
        self.assertTrue(np.all(np.isfinite(fit.prediction)))
        self.assertGreaterEqual(np.min(fit.prediction), 0.01)
        self.assertLessEqual(np.max(fit.prediction), 0.99)
        self.assertFalse(fit.diagnostics["fallback_used"])

    def test_population_group_laws_have_different_covariances(self) -> None:
        rng = np.random.default_rng(20260804)
        X = draw_reference_covariates(200_000, 20, 0.5, "pooled_random_shift", rng)
        e = pooled_random_propensity(X)
        pi = float(np.mean(e))
        self.assertLess(abs(pi - 0.5), 0.002)

        mean1 = np.average(X[:, :2], axis=0, weights=e)
        mean0 = np.average(X[:, :2], axis=0, weights=1.0 - e)
        centered1 = X[:, :2] - mean1
        centered0 = X[:, :2] - mean0
        covariance1 = (centered1 * e[:, None]).T @ centered1 / np.sum(e)
        covariance0 = (centered0 * (1.0 - e)[:, None]).T @ centered0 / np.sum(1.0 - e)
        self.assertGreater(mean1[0] - mean0[0], 0.18)
        self.assertGreater(covariance1[1, 1] - covariance0[1, 1], 0.15)
        self.assertGreater(covariance1[0, 1] - covariance0[0, 1], 0.07)

    def test_replication_records_realized_counts_and_estimated_propensity(self) -> None:
        task = {
            "design": "pooled_random_shift",
            "scenario": "additive",
            "p": 20,
            "n": 80,
            "rho": 0.5,
            "sigma": 1.0,
            "delta": 0.0,
            "rep": 1,
            "learner": "gam",
            "alpha": 0.05,
            "bootstrap_draws": 9,
            "dgp_seed": 991,
            "fit_seed": 992,
            "bootstrap_seed": 993,
        }
        config = {
            "methods": ["orthogonal_nosplit", "orthogonal_crossfit"],
            "propensity_mode": "estimated",
            "propensity_clip": 0.01,
            "crossfit_folds": 2,
            "learner_options": {"gam": {"n_knots": 4}},
            "propensity_options": {
                "cv_folds": 3,
                "n_knots": 4,
                "max_iter": 2000,
            },
        }
        rows, diagnostics = run_replication(task, self.calibrations["additive"], config)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertTrue(all(row["n1"] + row["n0"] == row["n"] for row in rows))
        self.assertTrue(all(np.isfinite(row["propensity_mse"]) for row in rows))
        self.assertTrue(all(np.isfinite(row["propensity_brier"]) for row in rows))
        self.assertTrue(all(np.isfinite(row["propensity_log_loss"]) for row in rows))
        self.assertTrue(all(np.isfinite(row["propensity_auc"]) for row in rows))
        for row in rows:
            payload = json.loads(row["diagnostics_json"])
            self.assertEqual(
                sum(item["n"] for item in payload["propensity_calibration_bins"]),
                row["n"],
            )
        self.assertEqual(diagnostics["n1"] + diagnostics["n0"], 80)

    def test_single_split_replication_uses_only_the_testing_half(self) -> None:
        task = {
            "design": "pooled_random_shift",
            "scenario": "additive",
            "p": 20,
            "n": 80,
            "rho": 0.5,
            "sigma": 1.0,
            "delta": 0.5,
            "rep": 1,
            "learner": "gam",
            "alpha": 0.05,
            "bootstrap_draws": 9,
            "dgp_seed": 1191,
            "fit_seed": 1192,
            "bootstrap_seed": 1193,
        }
        config = {
            "methods": ["orthogonal_single_split"],
            "propensity_mode": "estimated",
            "propensity_clip": 0.01,
            "learner_options": {"gam": {"n_knots": 4}},
            "propensity_options": {
                "cv_folds": 3,
                "n_knots": 4,
                "max_iter": 2000,
            },
        }

        rows, _ = run_replication(task, self.calibrations["additive"], config)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["method"], "orthogonal_single_split")
        self.assertEqual(row["status"], "ok")
        payload = json.loads(row["diagnostics_json"])
        split = payload["split"]

        expected_seed = stable_seed(
            task["dgp_seed"], "orthogonal_single_split", "partition"
        )
        self.assertEqual(split["split_seed"], expected_seed)
        self.assertEqual(split["estimation_fold"], 0)
        self.assertEqual(split["testing_fold"], 1)
        self.assertEqual(split["n_estimation"] + split["n_testing"], task["n"])
        self.assertEqual(split["n0_estimation"] + split["n0_testing"], row["n0"])
        self.assertEqual(split["n1_estimation"] + split["n1_testing"], row["n1"])
        for label in (0, 1):
            estimation_n = split[f"n{label}_estimation"]
            testing_n = split[f"n{label}_testing"]
            self.assertIn(estimation_n - testing_n, (0, 1))

        self.assertEqual(row["score_n"], split["n_testing"])
        self.assertEqual(payload["evaluation_n"], split["n_testing"])
        self.assertLess(row["score_n"], row["n"])
        self.assertEqual(payload["bandwidth_source"], "testing_pool")
        self.assertEqual(
            sum(item["n"] for item in payload["propensity_calibration_bins"]),
            row["score_n"],
        )

        data = generate_data(
            n=task["n"],
            p=task["p"],
            rho=task["rho"],
            sigma=task["sigma"],
            delta=task["delta"],
            design=task["design"],
            scenario=task["scenario"],
            calibration=self.calibrations["additive"],
            seed=task["dgp_seed"],
        )
        assignment = stratified_single_split(data.d, expected_seed)
        testing_index = np.flatnonzero(assignment == 1)
        self.assertEqual(len(testing_index), row["score_n"])
        self.assertAlmostEqual(
            row["bandwidth"], median_bandwidth(data.X[testing_index]), places=13
        )
        self.assertTrue(np.isfinite(row["mu_mse"]))
        self.assertTrue(np.isfinite(row["propensity_mse"]))

    def test_single_split_nuisances_train_only_on_estimation_indices(self) -> None:
        task = {
            "design": "pooled_random_shift",
            "scenario": "additive",
            "p": 20,
            "n": 80,
            "rho": 0.5,
            "sigma": 1.0,
            "delta": 0.5,
            "rep": 1,
            "learner": "gam",
            "alpha": 0.05,
            "bootstrap_draws": 3,
            "dgp_seed": 1291,
            "fit_seed": 1292,
            "bootstrap_seed": 1293,
        }
        config = {
            "methods": ["orthogonal_single_split"],
            "propensity_mode": "estimated",
            "propensity_clip": 0.01,
            "learner_options": {"gam": {}},
            "propensity_options": {},
        }
        data = generate_data(
            n=task["n"],
            p=task["p"],
            rho=task["rho"],
            sigma=task["sigma"],
            delta=task["delta"],
            design=task["design"],
            scenario=task["scenario"],
            calibration=self.calibrations["additive"],
            seed=task["dgp_seed"],
        )
        split_seed = stable_seed(
            task["dgp_seed"], "orthogonal_single_split", "partition"
        )
        assignment = stratified_single_split(data.d, split_seed)
        estimation_index = np.flatnonzero(assignment == 0)
        testing_index = np.flatnonzero(assignment == 1)

        def fake_regression(
            learner, X_train, y_train, X_predict, *, seed, learner_options
        ):
            self.assertEqual(learner, "gam")
            np.testing.assert_array_equal(X_train, data.X[estimation_index])
            np.testing.assert_array_equal(y_train, data.y[estimation_index])
            np.testing.assert_array_equal(X_predict, data.X[testing_index])
            self.assertEqual(
                seed,
                stable_seed(task["fit_seed"], "orthogonal_single_split", "mu"),
            )
            return PredictionFit(
                prediction=np.zeros(len(testing_index)),
                train_prediction=np.zeros(len(estimation_index)),
                diagnostics={"fallback_used": False},
            )

        def fake_propensity(
            X_train, d_train, X_predict, *, design, seed, clip, options
        ):
            np.testing.assert_array_equal(X_train, data.X[estimation_index])
            np.testing.assert_array_equal(d_train, data.d[estimation_index])
            np.testing.assert_array_equal(X_predict, data.X[testing_index])
            self.assertEqual(design, task["design"])
            self.assertEqual(
                seed,
                stable_seed(task["fit_seed"], "orthogonal_single_split", "e"),
            )
            return PredictionFit(
                prediction=np.full(len(testing_index), 0.5),
                train_prediction=np.full(len(estimation_index), 0.5),
                diagnostics={"fallback_used": False},
            )

        with patch("simcore.fit_regression", side_effect=fake_regression) as mu_fit:
            with patch(
                "simcore.fit_propensity", side_effect=fake_propensity
            ) as propensity_fit:
                rows, _ = run_replication(task, self.calibrations["additive"], config)

        self.assertEqual(mu_fit.call_count, 1)
        self.assertEqual(propensity_fit.call_count, 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["score_n"], len(testing_index))

    def test_propensity_metrics_and_bins_cover_every_observation(self) -> None:
        d = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
        probability = np.linspace(0.2, 0.8, len(d))
        diagnostics = propensity_performance(d, probability, bins=5)
        self.assertTrue(0.0 <= diagnostics["propensity_brier"] <= 1.0)
        self.assertGreaterEqual(diagnostics["propensity_log_loss"], 0.0)
        self.assertTrue(0.0 <= diagnostics["propensity_auc"] <= 1.0)
        self.assertEqual(
            sum(item["n"] for item in diagnostics["propensity_calibration_bins"]),
            len(d),
        )


class TestConstructionTests(unittest.TestCase):
    def test_single_split_is_deterministic_disjoint_and_balanced_by_label(self) -> None:
        labels = np.array([0] * 5 + [1] * 8)
        first = stratified_single_split(labels, seed=20260901)
        second = stratified_single_split(labels, seed=20260901)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isin(first, (0, 1)).all())
        for label, expected_estimation, expected_testing in ((0, 3, 2), (1, 4, 4)):
            selected = first[labels == label]
            self.assertEqual(np.count_nonzero(selected == 0), expected_estimation)
            self.assertEqual(np.count_nonzero(selected == 1), expected_testing)

        estimation_index = set(np.flatnonzero(first == 0))
        testing_index = set(np.flatnonzero(first == 1))
        self.assertFalse(estimation_index & testing_index)
        self.assertEqual(estimation_index | testing_index, set(range(len(labels))))

    def test_single_split_rejects_nonbinary_or_too_small_groups(self) -> None:
        with self.assertRaises(ValueError):
            stratified_single_split(np.array([0, 0, 1, 2]), seed=1)
        with self.assertRaises(ValueError):
            stratified_single_split(np.array([0, 0, 1]), seed=1)

    def test_multiplier_matches_explicit_centered_draws(self) -> None:
        rng = np.random.default_rng(81)
        X = rng.normal(size=(16, 3))
        scores = rng.normal(loc=0.4, size=16)
        bandwidth = median_bandwidth(X)
        kernel = rbf_kernel_matrix(X, bandwidth)
        seed = 991
        B = 39
        result = multiplier_test(
            scores, kernel, bootstrap_draws=B, alpha=0.05, seed=seed
        )
        expected_observed = math.sqrt(max(scores @ kernel @ scores / len(scores), 0))
        draw_rng = np.random.default_rng(seed)
        xi = draw_rng.standard_normal((len(scores), B))
        xi -= xi.mean(axis=0, keepdims=True)
        weighted = scores[:, None] * xi
        bootstrap = np.sqrt(
            np.maximum(np.sum(weighted * (kernel @ weighted), axis=0) / len(scores), 0)
        )
        expected_p = (1 + np.count_nonzero(bootstrap >= expected_observed)) / (B + 1)
        self.assertAlmostEqual(result.statistic, expected_observed, places=13)
        self.assertAlmostEqual(result.p_value, expected_p, places=13)
        self.assertEqual(result.reject, int(expected_p <= 0.05))
        self.assertAlmostEqual(
            result.critical_value,
            np.quantile(bootstrap, 0.95, method="linear"),
            places=13,
        )

    def test_yan_uses_uncentered_second_moment_variance(self) -> None:
        rng = np.random.default_rng(102)
        X1 = rng.normal(size=(20, 4))
        X0 = rng.normal(size=(24, 4))
        eta1 = rng.normal(loc=0.2, size=20)
        eta0 = rng.normal(loc=-0.1, size=24)
        result = yan_test(eta1, X1, eta0, X0, alpha=0.05)
        n1, n0 = 10, 12
        h1, h0 = median_bandwidth(X1), median_bandwidth(X0)
        t1 = eta1[:n1] * eta1[n1:] * rbf_paired_values(X1[:n1], X1[n1:], h1)
        t0 = eta0[:n0] * eta0[n0:] * rbf_paired_values(X0[:n0], X0[n0:], h0)
        raw = np.mean(t1) + np.mean(t0)
        se2 = np.mean(t1**2) / n1 + np.mean(t0**2) / n0
        expected_z = raw / np.sqrt(se2)
        self.assertAlmostEqual(result.statistic, expected_z, places=13)
        self.assertAlmostEqual(result.p_value, norm.sf(expected_z), places=13)
        self.assertAlmostEqual(
            result.diagnostics["sigma1_squared"], np.mean(t1**2), places=13
        )


class LearnerTests(unittest.TestCase):
    def test_dnn_uses_holdout_epoch_and_full_refit(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(70, 5))
        y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(scale=0.1, size=70)
        fit = fit_dnn(
            X[:60],
            y[:60],
            X[60:],
            seed=9911,
            options={
                "device": "cpu",
                "hidden_layers": [12, 6],
                "dropout": 0.1,
                "max_epochs": 40,
                "min_epochs": 8,
                "patience": 6,
            },
        )
        self.assertEqual(fit.prediction.shape, (10,))
        self.assertEqual(fit.train_prediction.shape, (60,))
        self.assertGreaterEqual(fit.diagnostics["selected_epoch"], 1)
        self.assertEqual(fit.diagnostics["validation_size"], 12)
        self.assertFalse(fit.diagnostics.get("fallback_used", False))
        self.assertTrue(np.all(np.isfinite(fit.prediction)))

    def test_unknown_learner_does_not_fallback(self) -> None:
        X = np.arange(30, dtype=float).reshape(10, 3)
        y = np.arange(10, dtype=float)
        with self.assertRaises(ValueError):
            fit_regression(
                "not_a_learner",
                X,
                y,
                X,
                seed=1,
                learner_options={},
            )


if __name__ == "__main__":
    unittest.main()
