import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_simulation import build_tasks, validate_config
from simcore import (
    calibrate_scenarios,
    draw_reference_covariates,
    evaluate_regression_structure,
    generate_data,
    pooled_random_propensity,
    run_replication,
    validate_treatment_noise_scales,
)


class TreatmentNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = calibrate_scenarios(
            p=20,
            rho=0.5,
            design="pooled_random_shift",
            scenarios=["neural_composition"],
            calibration_n=2000,
            seed=123,
        )["neural_composition"]
        cls.common = {
            "n": 80,
            "p": 20,
            "rho": 0.5,
            "sigma": 1.2,
            "delta": 0.8,
            "design": "pooled_random_shift",
            "scenario": "neural_composition",
            "calibration": cls.calibration,
            "seed": 234,
        }
        cls.config = json.loads((ROOT / "configs/smoke.json").read_text())

    def test_default_preserves_original_draws_exactly(self):
        data = generate_data(**self.common)
        rng = np.random.default_rng(self.common["seed"])
        X = draw_reference_covariates(80, 20, 0.5, "pooled_random_shift", rng)
        propensity = pooled_random_propensity(X)
        treatment = rng.binomial(1, propensity, size=80).astype(int)
        base, signal = evaluate_regression_structure(
            X, "neural_composition", self.calibration
        )
        expected_y = base + treatment * 0.8 * signal + rng.normal(0.0, 1.2, 80)
        np.testing.assert_array_equal(data.X, X)
        np.testing.assert_array_equal(data.d, treatment)
        np.testing.assert_array_equal(data.y, expected_y)
        self.assertNotIn("noise_scale_control", data.diagnostics)
        unit_scales = generate_data(treatment_noise_scales=[1.0, 1.0], **self.common)
        np.testing.assert_array_equal(data.y, unit_scales.y)

    def test_treatment_residual_scaling_preserves_all_conditional_means(self):
        default = generate_data(**self.common)
        cate = generate_data(treatment_noise_scales=[1.0, 1.5], **self.common)
        base, signal = evaluate_regression_structure(
            default.X, "neural_composition", self.calibration
        )
        conditional_mean = base + default.d * self.common["delta"] * signal
        np.testing.assert_allclose(
            cate.y - conditional_mean,
            (default.y - conditional_mean) * (1.0 + 0.5 * default.d),
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_array_equal(cate.y0, default.y0)
        for name in (
            "X", "d", "e_true", "mu_true", "m1_x1", "m0_x1", "m1_x0", "m0_x0",
            "signal_x1", "signal_x0",
        ):
            np.testing.assert_array_equal(getattr(cate, name), getattr(default, name))
        self.assertEqual(cate.diagnostics["noise_sd_control"], 1.2)
        self.assertAlmostEqual(cate.diagnostics["noise_sd_treated"], 1.8)

    def test_noise_is_paired_across_null_and_alternative(self):
        arguments = {**self.common, "treatment_noise_scales": [1.0, 1.5]}
        null = generate_data(**{**arguments, "delta": 0.0})
        alternative = generate_data(**arguments)
        np.testing.assert_array_equal(null.X, alternative.X)
        np.testing.assert_array_equal(null.d, alternative.d)
        np.testing.assert_array_equal(null.y0, alternative.y0)
        np.testing.assert_allclose(
            alternative.y1 - null.y1,
            self.common["delta"] * alternative.signal_x1,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_invalid_noise_scales_are_rejected(self):
        invalid_values = (
            [], [1.0], [1.0, 1.5, 2.0], [0.0, 1.5], [1.0, -1.0],
            [1.0, np.inf], [np.nan, 1.0], [1.0, "invalid"], "12", 1.5, None,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "two finite positive"):
                    validate_treatment_noise_scales(value)
                config = {**self.config, "treatment_noise_scales": value}
                with self.assertRaisesRegex(ValueError, "two finite positive"):
                    validate_config(config)
                if value is not None:
                    with self.assertRaisesRegex(ValueError, "two finite positive"):
                        generate_data(treatment_noise_scales=value, **self.common)

    def test_configuration_reaches_generator_and_result_metadata(self):
        config = copy.deepcopy(self.config)
        config.update(
            scenarios=["neural_composition"],
            treatment_noise_scales=[1.0, 1.5],
            methods=["oracle_score"],
            learners=["oracle"],
        )
        validate_config(config)
        tasks = build_tasks(
            config, designs=None, scenarios=None, dimensions=None, deltas=[0.8],
            learners=None, rep_start=1, rep_stop=1,
        )
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["treatment_noise_scales"], [1.0, 1.5])
        rows, diagnostics = run_replication(task, self.calibration, config)
        self.assertEqual(rows[0]["status"], "ok")
        for key, expected in {
            "noise_scale_control": 1.0, "noise_scale_treated": 1.5,
            "noise_sd_control": 1.0, "noise_sd_treated": 1.5,
        }.items():
            self.assertEqual(rows[0][key], expected)
            self.assertEqual(diagnostics[key], expected)
        config_only_task = {k: v for k, v in task.items() if k != "treatment_noise_scales"}
        config_only_rows, _ = run_replication(config_only_task, self.calibration, config)
        self.assertEqual(rows[0]["statistic"], config_only_rows[0]["statistic"])
        self.assertEqual(rows[0]["p_value"], config_only_rows[0]["p_value"])


if __name__ == "__main__":
    unittest.main()
