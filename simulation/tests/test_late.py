import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_simulation import build_tasks, make_calibrations, validate_config
from simcore import (
    calibrate_scenarios,
    draw_reference_covariates,
    evaluate_regression_structure,
    fit_propensity,
    generate_data,
    pooled_random_propensity,
    run_replication,
    validate_late_options,
)


class LateDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/late.json").read_text())
        cls.calibration = calibrate_scenarios(
            p=20, rho=0.5, design="late_iv", scenarios=["neural_composition"],
            calibration_n=2000, seed=123,
        )["neural_composition"]
        cls.common = dict(
            n=400, p=20, rho=0.5, sigma=1.0, delta=0.8, design="late_iv",
            scenario="neural_composition", calibration=cls.calibration, seed=234,
            treatment_noise_scales=[1.0, 1.5], late_options=cls.config["late_options"],
        )

    def reconstruct_latent_data(self):
        rng = np.random.default_rng(self.common["seed"])
        X = draw_reference_covariates(400, 20, 0.5, "late_iv", rng)
        pi = pooled_random_propensity(X)
        instrument = rng.binomial(1, pi, size=400).astype(int)
        latent = rng.uniform(size=400)
        d0 = (latent <= 0.2).astype(int)
        d1 = (latent <= 0.8).astype(int)
        complier = d1 - d0
        treatment = np.where(instrument, d1, d0)
        base, signal = evaluate_regression_structure(X, "neural_composition", self.calibration)
        noise = rng.normal(size=400)
        return X, pi, instrument, latent, d0, d1, complier, treatment, base, signal, noise

    def test_observed_law_and_instrument_grouping(self):
        dat = generate_data(**self.common)
        X, pi, A, U, D0, D1, C, D, base, signal, eps = self.reconstruct_latent_data()
        np.testing.assert_array_equal(dat.X, X)
        np.testing.assert_array_equal(dat.d, A)
        self.assertTrue(np.any(A != D))
        self.assertTrue(np.all(D1 >= D0))
        expected = base + 2 * (U - 0.5) + D * (C * 0.8 * signal + (1-C)) + (1+0.5*D)*eps
        np.testing.assert_array_equal(dat.y, expected)
        np.testing.assert_array_equal(dat.X1, X[A == 1])
        np.testing.assert_array_equal(dat.y0, expected[A == 0])
        self.assertEqual(dat.diagnostics["n_compliers"], int(C.sum()))
        self.assertEqual(dat.diagnostics["n_treated"], int(D.sum()))
        self.assertAlmostEqual(dat.diagnostics["first_stage_empirical"],
                               D[A == 1].mean() - D[A == 0].mean())

    def test_oracle_regressions_are_iv_means(self):
        dat = generate_data(**self.common)
        base, signal = evaluate_regression_structure(dat.X, "neural_composition", self.calibration)
        m0 = base + 0.2
        m1 = m0 + 0.6 * 0.8 * signal
        np.testing.assert_allclose(dat.mu_true, m0 + dat.e_true * (m1-m0), atol=1e-14)
        for arm in (0, 1):
            group = dat.d == arm
            np.testing.assert_array_equal(getattr(dat, f"m0_x{arm}"), m0[group])
            np.testing.assert_array_equal(getattr(dat, f"m1_x{arm}"), m1[group])
        # The orthogonal IV score targets pi(1-pi) times the reduced form.
        score_mean = dat.e_true * (1-dat.e_true) * (m1-m0)
        np.testing.assert_allclose(score_mean,
            0.6 * dat.e_true * (1-dat.e_true) * 0.8 * signal, atol=1e-14)

    def test_null_and_alternative_share_latent_draws(self):
        null = generate_data(**{**self.common, "delta": 0.0})
        alt = generate_data(**self.common)
        _, _, A, _, _, _, C, D, _, signal, _ = self.reconstruct_latent_data()
        np.testing.assert_array_equal(null.X, alt.X)
        np.testing.assert_array_equal(null.d, alt.d)
        np.testing.assert_allclose(alt.y-null.y, A*C*0.8*signal, atol=2e-14)
        np.testing.assert_array_equal(null.y0, alt.y0)

    def test_null_late_allows_nonzero_ate_and_endogenous_treatment(self):
        # Midpoint integration over U verifies the principal-stratum identities
        # without Monte Carlo tolerances or dependence on a random seed.
        u = (np.arange(10000) + 0.5) / 10000
        d0, d1 = (u <= 0.2), (u <= 0.8)
        complier = d1 & ~d0
        tau = (~complier).astype(float)
        y0 = 2*(u-0.5)
        y1 = y0 + tau
        self.assertAlmostEqual(np.mean((y1-y0)[complier]), 0.0)
        self.assertAlmostEqual(np.mean(y1-y0), 0.4)
        self.assertAlmostEqual(np.mean(d1.astype(float)-d0), 0.6)
        iv_means = []
        iv_variances = []
        for treatment in (d0, d1):
            location = y0 + treatment*tau
            iv_means.append(np.mean(location))
            iv_variances.append(np.var(location) + np.mean((1+0.5*treatment)**2))
        np.testing.assert_allclose(iv_means, [0.2, 0.2], atol=1e-14)
        np.testing.assert_allclose(iv_variances, [1.423333333333, 2.173333333333], atol=1e-8)
        # At any X, pi mixes D0 and D1; both have the same covariance with Y(0).
        for pi in (0.35, 0.5, 0.70):
            expected_treatment_given_u = (1-pi)*d0 + pi*d1
            covariance = np.mean(y0*expected_treatment_given_u) - np.mean(y0)*np.mean(expected_treatment_given_u)
            self.assertAlmostEqual(covariance, -0.16)

    def test_calibration_and_propensity_fit_match_existing_design(self):
        config = {**self.config, "calibration_n": 2000}
        tasks = build_tasks(config, designs=None, scenarios=None, dimensions=None,
                            deltas=[0.0], learners=["oracle"], rep_start=1, rep_stop=1)
        late = make_calibrations(config, tasks)[("late_iv", 20, "neural_composition")]
        original_tasks = [{**tasks[0], "design": "pooled_random_shift"}]
        original = make_calibrations(config, original_tasks)[("pooled_random_shift", 20, "neural_composition")]
        for key, value in original.to_dict().items():
            if key != "design":
                self.assertEqual(late.to_dict()[key], value)
        dat = generate_data(**self.common)
        kwargs = dict(seed=45, clip=0.01, options={"cv_folds": 2, "Cs": [0.1, 1.0]})
        fitted = [fit_propensity(dat.X, dat.d, dat.X,
                                 design=design, **kwargs).prediction
                  for design in ("pooled_random_shift", "late_iv")]
        np.testing.assert_array_equal(*fitted)

    def test_config_and_oracle_runner_preserve_the_protocol(self):
        validate_config(self.config)
        cate = json.loads((ROOT / "configs/cate.json").read_text())
        for key in ("seed_base", "n", "rho", "deltas", "dimensions", "replications",
                    "bootstrap_draws", "learners", "methods", "learner_options",
                    "propensity_options", "treatment_noise_scales"):
            self.assertEqual(self.config[key], cate[key])
        tasks = build_tasks(self.config, designs=None, scenarios=None, dimensions=None,
                            deltas=[0.0, 0.8], learners=["oracle"], rep_start=1, rep_stop=1)
        self.assertEqual(tasks[0]["dgp_seed"], tasks[1]["dgp_seed"])
        self.assertEqual(tasks[0]["late_options"], self.config["late_options"])
        rows, diagnostics = run_replication(tasks[0], self.calibration, self.config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(diagnostics["group_variable"], "instrument")
        self.assertEqual(diagnostics["first_stage_population"], 0.6)

    def test_changed_design_constants_are_rejected(self):
        for key in self.config["late_options"]:
            options = {**self.config["late_options"], key: 3.0}
            with self.assertRaises(ValueError):
                validate_late_options(options)
        for value in ({}, "invalid", {**self.config["late_options"], "extra": 1}):
            with self.assertRaises(ValueError):
                validate_late_options(value)
        for change in ({"sigma": 2}, {"treatment_noise_scales": [1, 1]}, {"late_options": {}}):
            with self.assertRaises(ValueError):
                validate_config({**self.config, **change})


if __name__ == "__main__":
    unittest.main()
