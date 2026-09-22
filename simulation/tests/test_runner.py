import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_simulation import (
    build_tasks,
    load_existing,
    make_calibrations,
    validate_config,
)
from simcore import SUPPORTED_METHODS, stable_seed


class ResumeTests(unittest.TestCase):
    def test_interrupted_multifile_checkpoint_rolls_back_to_common_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            pd.DataFrame(
                [
                    {"task_key": "committed", "method": "m1"},
                    {"task_key": "raw_only", "method": "m1"},
                ]
            ).to_csv(output / "raw_results.csv", index=False)
            pd.DataFrame([{"task_key": "committed", "value": 1.0}]).to_csv(
                output / "dgp_diagnostics.csv", index=False
            )

            results, diagnostics = load_existing(output)

            self.assertEqual({row["task_key"] for row in results}, {"committed"})
            self.assertEqual({row["task_key"] for row in diagnostics}, {"committed"})
            report = json.loads(
                (output / "resume_reconciliation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["result_only_count"], 1)
            self.assertEqual(report["diagnostic_only_count"], 0)
            self.assertEqual(report["common_tasks_retained"], 1)


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "configs" / "smoke.json").read_text(encoding="utf-8")
        )

    def test_smoke_configuration_uses_total_n_and_estimated_propensity(self) -> None:
        validate_config(self.config)
        self.assertEqual(set(self.config["methods"]), set(SUPPORTED_METHODS))
        self.assertIn("orthogonal_single_split", SUPPORTED_METHODS)
        tasks = build_tasks(
            self.config,
            designs=None,
            scenarios=None,
            dimensions=None,
            deltas=None,
            learners=None,
            rep_start=1,
            rep_stop=1,
        )
        self.assertTrue(tasks)
        self.assertTrue(all(task["n"] == 80 for task in tasks))
        self.assertTrue(all("n1" not in task and "n0" not in task for task in tasks))
        self.assertIn("oracle", {task["learner"] for task in tasks})

    def test_unknown_single_split_alias_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["methods"] = ["naive_split"]
        with self.assertRaisesRegex(ValueError, "Unknown methods"):
            validate_config(invalid)

    def test_known_propensity_is_rejected_for_feasible_design(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["propensity_mode"] = "known"
        with self.assertRaises(ValueError):
            validate_config(invalid)

    def test_calibration_seed_is_stable(self) -> None:
        tasks = build_tasks(
            self.config,
            designs=None,
            scenarios=["additive"],
            dimensions=[20],
            deltas=[0.0],
            learners=["oracle"],
            rep_start=1,
            rep_stop=1,
        )
        calibration = make_calibrations(self.config, tasks)[
            ("pooled_random_shift", 20, "additive")
        ]
        self.assertEqual(
            calibration.calibration_seed,
            stable_seed(self.config["seed_base"], "equal", 20, "calibration"),
        )


if __name__ == "__main__":
    unittest.main()
