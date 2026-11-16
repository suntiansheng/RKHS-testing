from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pcm
from run_simulation import build_tasks, make_calibrations, validate_config
from simcore import run_replication
from summarize import (
    PCM_HOLM_FAMILY, PRODUCTION_SCENARIOS, METHOD_LABELS,
    _make_plot_set, assess_size_families,
)


def tasks_for(config):
    return build_tasks(
        config, designs=None, scenarios=None, dimensions=[20], deltas=[0.0],
        learners=None, rep_start=1, rep_stop=1,
    )


class OriginalPCMConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs" / "pcm_smoke.json").read_text())

    def test_original_source_is_pinned_and_verified(self):
        provenance = pcm.pcm_provenance()
        self.assertEqual(provenance["commit"], "3822659390510d65c36857f8801c893e94850774")
        with tempfile.TemporaryDirectory() as temporary:
            vendor = Path(temporary)
            for source in pcm.VENDOR.iterdir():
                if source.is_file():
                    shutil.copyfile(source, vendor / source.name)
            with (vendor / "test_functions.R").open("a") as stream:
                stream.write("\n# altered\n")
            with patch.object(pcm, "VENDOR", vendor):
                with self.assertRaisesRegex(RuntimeError, "source has changed"):
                    pcm.pcm_provenance()

    def test_production_grid_matches_baseline_data_generation(self):
        production = json.loads((ROOT / "configs" / "pcm.json").read_text())
        baseline = json.loads((ROOT / "configs" / "main.json").read_text())
        validate_config(production)
        for field in ("seed_base", "n", "rho", "sigma", "target_pi", "scenarios",
                      "dimensions", "deltas", "replications", "calibration_n"):
            self.assertEqual(production[field], baseline[field])
        baseline["learners"] = ["rf"]
        self.assertEqual(tasks_for(production)[0]["dgp_seed"], tasks_for(baseline)[0]["dgp_seed"])
        self.assertEqual(production["learners"], [pcm.PCM_LEARNER])

    def test_tuning_overrides_and_wrong_learner_are_rejected(self):
        for changes in (
            {"pcm_options": {"mtry": 1}},
            {"learner_options": {pcm.PCM_LEARNER: {"n_estimators": 10}}},
            {"learners": ["rf"]},
            {"n": 81},
        ):
            invalid = copy.deepcopy(self.config)
            invalid.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(invalid)

    def test_native_process_failure_is_retained_for_both_methods(self):
        task = tasks_for(self.config)[0]
        calibration = make_calibrations(self.config, [task])[("pooled_random_shift", 20, "additive")]
        with patch.object(pcm, "run_original_pcm", side_effect=RuntimeError("R failed")):
            rows, _ = run_replication(task, calibration, self.config)
        self.assertEqual({row["method"] for row in rows}, set(pcm.PCM_METHODS))
        self.assertTrue(all(row["status"] == "failed" for row in rows))
        self.assertTrue(all(np.isnan(row["reject"]) for row in rows))
        self.assertTrue(all(row["bootstrap_draws"] == 0 for row in rows))


class OriginalPCMReportingTests(unittest.TestCase):
    @staticmethod
    def complete_nulls():
        return pd.DataFrame([
            dict(design="pooled_random_shift", scenario=scenario, p=p, delta=0.0,
                 learner=pcm.PCM_LEARNER, method=method, n_valid=300, n_failed=0,
                 alpha=0.05, rejection_rate=0.05)
            for scenario in PRODUCTION_SCENARIOS for p in (20, 50) for method in pcm.PCM_METHODS
        ])

    def test_pcm_has_twenty_distinct_null_cells_and_requires_complete_grid(self):
        summary = self.complete_nulls()
        sizes, complete = assess_size_families(summary)
        self.assertEqual(len(sizes), 20)
        self.assertEqual(complete, {PCM_HOLM_FAMILY: True})
        self.assertIs(type(complete[PCM_HOLM_FAMILY]), bool)
        self.assertEqual(json.loads(json.dumps(complete)), {PCM_HOLM_FAMILY: True})
        _, incomplete = assess_size_families(summary.iloc[1:])
        self.assertFalse(any(incomplete.values()))

    def test_pcm_does_not_redefine_historical_size_families(self):
        historical = pd.DataFrame([
            dict(design="pooled_random_shift", scenario="additive", p=20, delta=0.0,
                 learner="rf", method=method, n_valid=300, n_failed=0,
                 alpha=0.05, rejection_rate=0.12)
            for method in ("orthogonal_nosplit", "orthogonal_single_split")
        ])
        before, complete_before = assess_size_families(historical)
        after, complete_after = assess_size_families(pd.concat([historical, self.complete_nulls()]))
        pd.testing.assert_frame_equal(
            before.reset_index(drop=True),
            after[~after.method.isin(pcm.PCM_METHODS)].reset_index(drop=True),
        )
        self.assertTrue(all(complete_after[name] == value for name, value in complete_before.items()))

    def test_shared_pcm_curves_are_drawn_in_each_learner_panel(self):
        rows = []
        for learner, methods in (("gam", ["orthogonal_nosplit"]),
                                 ("rf", ["orthogonal_nosplit"]),
                                 (pcm.PCM_LEARNER, pcm.PCM_METHODS)):
            for method in methods:
                for delta, rate in ((0.0, 0.05), (0.5, 0.5)):
                    rows.append(dict(
                        design="pooled_random_shift", scenario="additive", p=20,
                        delta=delta, learner=learner, method=method, alpha=0.05,
                        rejection_rate=rate, wilson_lower=rate - 0.03,
                        wilson_upper=rate + 0.03, size_gate_status="not_evaluated",
                    ))
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, squeeze=False)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("summarize.plt.subplots", return_value=(figure, axes)):
                _make_plot_set(pd.DataFrame(rows), Path(temporary), "diagnostic")
        for axis in axes.flat:
            labels = {line.get_label() for line in axis.lines}
            self.assertTrue(all(METHOD_LABELS[method] in labels for method in pcm.PCM_METHODS))
        self.assertEqual(len(figure.axes), 2)


@unittest.skipUnless(shutil.which("Rscript"), "Rscript is not installed")
class NativeRReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.metadata = pcm.pcm_runtime_metadata()
        except RuntimeError as error:
            if "Install the R package" in str(error):
                raise unittest.SkipTest(str(error)) from error
            raise

    def test_exact_agreement_with_original_six_calls_and_rng_state(self):
        completed = subprocess.run(
            [self.metadata["rscript"], "--vanilla", str(ROOT / "tests" / "test_pcm_reference.R"), str(ROOT)],
            capture_output=True, text=True, timeout=120, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(len(report["reference_p_values"]), 6)
        self.assertEqual(report["maximum_absolute_difference"], 0)
        self.assertTrue(report["rng_states_identical"])

    def test_python_bridge_uses_native_pcm_and_returns_finite_results(self):
        rng = np.random.default_rng(762)
        X = rng.normal(size=(40, 3))
        d = rng.binomial(1, 0.5, size=40)
        y = X[:, 0] + d * X[:, 1] + rng.normal(size=40)
        payload = pcm.run_original_pcm(y, d, X, seed=613)
        self.assertEqual(set(payload["methods"]), set(pcm.PCM_METHODS))
        self.assertEqual(payload["diagnostics"]["runtime"], self.metadata["runtime"])
        self.assertEqual(len(payload["diagnostics"]["split_results"]), 6)
        for method in pcm.PCM_METHODS:
            result = payload["methods"][method]
            self.assertEqual(result["status"], "ok")
            self.assertTrue(np.isfinite(result["statistic"]))
            self.assertTrue(0 <= result["p_value"] <= 1)


if __name__ == "__main__":
    unittest.main()
