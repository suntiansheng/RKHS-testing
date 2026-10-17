import copy
import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_simulation import build_tasks, make_calibrations, validate_config
from simcore import (
    PredictionFit,
    generate_data,
    independent_folds,
    median_bandwidth,
    rbf_kernel_matrix,
    run_replication,
    stable_seed,
)


class CausalCrossfitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.configs = {}
        for name in ("cate", "late"):
            config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
            config.update(n=80, bootstrap_draws=39, crossfit_folds=2, calibration_n=2000)
            config["learner_options"]["rf"]["n_estimators"] = 30
            config["learner_options"]["dnn"].update(
                max_epochs=3, min_epochs=1, patience=2, device="cpu"
            )
            config["propensity_options"].update(cv_folds=2, Cs=[0.1, 1.0])
            cls.configs[name] = config

    def task_and_calibration(self, config, learner="gam"):
        task = build_tasks(
            config, designs=None, scenarios=None, dimensions=None,
            deltas=[0.8], learners=[learner], rep_start=1, rep_stop=1,
        )[0]
        calibration = make_calibrations(config, [task])[
            (task["design"], task["p"], task["scenario"])
        ]
        return task, calibration

    def test_independent_folds_are_balanced_reproducible_and_nonempty(self):
        for n, folds in ((81, 2), (80, 5), (8, 8)):
            assignment = independent_folds(n, folds, 1021)
            counts = np.bincount(assignment, minlength=folds)
            self.assertEqual(len(assignment), n)
            self.assertGreater(counts.min(), 0)
            self.assertLessEqual(counts.max() - counts.min(), 1)
            np.testing.assert_array_equal(assignment, independent_folds(n, folds, 1021))
        for folds in (0, 1, 81, 2.5, "2", True):
            with self.subTest(folds=folds), self.assertRaisesRegex(ValueError, "integer number"):
                independent_folds(80, folds, 1021)

    def test_runner_rejects_invalid_fold_counts(self):
        for folds in (0, 1, 81, 2.5, "2", True):
            config = copy.deepcopy(self.configs["cate"])
            config["methods"] = ["orthogonal_crossfit", "oracle_score"]
            config["crossfit_folds"] = folds
            with self.subTest(folds=folds), self.assertRaisesRegex(ValueError, "integer number"):
                validate_config(config)

    def test_scores_use_only_outer_training_data_and_pooled_centering(self):
        for name, original in self.configs.items():
            with self.subTest(design=name):
                config = {**original, "methods": ["orthogonal_crossfit"]}
                task, calibration = self.task_and_calibration(config)
                dat = generate_data(
                    n=task["n"], p=task["p"], rho=task["rho"], sigma=task["sigma"],
                    delta=task["delta"], design=task["design"], scenario=task["scenario"],
                    calibration=calibration, seed=task["dgp_seed"],
                    treatment_noise_scales=config["treatment_noise_scales"],
                    late_options=config.get("late_options"),
                )
                seed = stable_seed(task["dgp_seed"], "crossfit_folds", 2)
                assignment = independent_folds(task["n"], 2, seed)
                expected_mu = np.empty(task["n"])
                expected_e = np.empty(task["n"])
                calls = {"mu": 0, "e": 0}

                def fitted(kind, X_train, target, X_predict):
                    fold = calls[kind]
                    calls[kind] += 1
                    train = assignment != fold
                    test = assignment == fold
                    np.testing.assert_array_equal(X_train, dat.X[train])
                    np.testing.assert_array_equal(X_predict, dat.X[test])
                    np.testing.assert_array_equal(target, (dat.y if kind == "mu" else dat.d)[train])
                    prediction = np.full(test.sum(), np.mean(target))
                    (expected_mu if kind == "mu" else expected_e)[test] = prediction
                    return PredictionFit(prediction, np.full(train.sum(), np.mean(target)), {})

                def fit_mu(learner, X_train, target, X_predict, **kwargs):
                    return fitted("mu", X_train, target, X_predict)

                def fit_e(X_train, target, X_predict, **kwargs):
                    return fitted("e", X_train, target, X_predict)

                with patch("simcore.fit_regression", side_effect=fit_mu), patch(
                    "simcore.fit_propensity", side_effect=fit_e
                ):
                    rows, _ = run_replication(task, calibration, config)
                self.assertEqual(calls, {"mu": 2, "e": 2})
                row = rows[0]
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["score_n"], task["n"])
                diagnostics = json.loads(row["diagnostics_json"])
                self.assertEqual(diagnostics["fold_assignment"], "independent")
                self.assertEqual(diagnostics["fold_seed"], seed)
                scores = (dat.y - expected_mu) * (dat.d - expected_e)
                kernel = rbf_kernel_matrix(dat.X, median_bandwidth(dat.X))
                eigenvalues, eigenvectors = np.linalg.eigh(kernel)
                features = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
                embedded = scores[:, None] * features
                mean = embedded.mean(axis=0)
                observed = math.sqrt(task["n"]) * np.linalg.norm(mean)
                multipliers = np.random.default_rng(task["bootstrap_seed"]).standard_normal(
                    (task["n"], task["bootstrap_draws"])
                )
                bootstrap = np.linalg.norm(multipliers.T @ (embedded - mean), axis=1) / math.sqrt(task["n"])
                expected_p = (1 + np.count_nonzero(bootstrap >= observed)) / (task["bootstrap_draws"] + 1)
                self.assertAlmostEqual(row["statistic"], observed, places=12)
                self.assertAlmostEqual(row["critical_value"], np.quantile(bootstrap, 0.95), places=12)
                self.assertEqual(row["p_value"], expected_p)

    def test_adding_crossfit_preserves_existing_method_results(self):
        def without_times(value):
            if isinstance(value, dict):
                return {key: without_times(item) for key, item in value.items()
                        if not key.endswith("seconds")}
            if isinstance(value, list):
                return [without_times(item) for item in value]
            return value

        def scientific_row(row):
            row = {**row, "diagnostics_json": json.loads(row["diagnostics_json"])}
            return json.dumps(without_times(row), sort_keys=True)

        methods = ["orthogonal_nosplit", "orthogonal_single_split", "yan", "oracle_score"]
        for name, original in self.configs.items():
            for learner in ("gam", "rf", "lasso", "dnn", "oracle"):
                with self.subTest(design=name, learner=learner):
                    config = {**original, "methods": methods}
                    task, calibration = self.task_and_calibration(config, learner)
                    before, _ = run_replication(task, calibration, config)
                    after, _ = run_replication(
                        task, calibration,
                        {**config, "methods": methods + ["orthogonal_crossfit"]},
                    )
                    self.assertTrue(all(row["status"] == "ok" for row in before + after))
                    old_rows = [scientific_row(row) for row in before]
                    preserved = [scientific_row(row) for row in after if row["method"] in methods]
                    self.assertEqual(old_rows, preserved)


if __name__ == "__main__":
    unittest.main()
