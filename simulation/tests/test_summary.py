import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summarize import (
    SINGLE_SPLIT_HOLM_FAMILY,
    MAIN_HOLM_FAMILY,
    MAIN_METHODS,
    assess_size_families,
    METHOD_LABELS,
    METHOD_ORDER,
    aggregate_results,
    attach_size_gate,
    size_family,
    make_plots,
    propensity_diagnostic_tables,
    size_audit,
)


class SizeGateTests(unittest.TestCase):
    def test_main_and_single_split_holm_families_remain_separate(self):
        rows = []
        pairs = [
            (learner, method)
            for learner in ("gam", "rf", "lasso", "dnn")
            for method in sorted(MAIN_METHODS | {"orthogonal_single_split"})
        ] + [("oracle", "oracle_score")]
        for scenario in (
            "additive",
            "product",
            "composition",
            "friedman",
            "neural_composition",
        ):
            for p in (20, 50):
                for learner, method in pairs:
                    rows.append(
                        {
                            "design": "pooled_random_shift",
                            "scenario": scenario,
                            "p": p,
                            "delta": 0.0,
                            "learner": learner,
                            "method": method,
                            "n_valid": 300,
                            "n_failed": 0,
                            "alpha": 0.05,
                            "rejection_rate": 0.10 if method == "residual" else 0.05,
                        }
                    )
        summary = pd.DataFrame(rows)
        sizes, complete = assess_size_families(summary)
        self.assertEqual(
            complete, {MAIN_HOLM_FAMILY: True, SINGLE_SPLIT_HOLM_FAMILY: True}
        )
        self.assertEqual(
            sizes.groupby("holm_family").size().to_dict(),
            {
                MAIN_HOLM_FAMILY: 170,
                SINGLE_SPLIT_HOLM_FAMILY: 40,
            },
        )
        main = summary[summary["method"] != "orthogonal_single_split"]
        expected = size_audit(main, family=MAIN_HOLM_FAMILY)
        actual = sizes[sizes["holm_family"] == MAIN_HOLM_FAMILY]
        pd.testing.assert_series_equal(
            actual["holm_p_value_global"].reset_index(drop=True),
            expected["holm_p_value_global"].reset_index(drop=True),
        )
        self.assertFalse(size_family(main.iloc[1:], single_split=False)[1])
        failed = main.copy()
        failed.loc[failed.index[0], "n_valid"] = 299
        failed.loc[failed.index[0], "n_failed"] = 1
        self.assertFalse(size_family(failed, single_split=False)[1])

    def test_production_family_name_requires_all_40_cells_and_300_reps(self):
        rows = []
        for scenario in (
            "additive",
            "product",
            "composition",
            "friedman",
            "neural_composition",
        ):
            for p in (20, 50):
                for learner in ("gam", "rf", "lasso", "dnn"):
                    rows.append(
                        {
                            "design": "pooled_random_shift",
                            "scenario": scenario,
                            "p": p,
                            "delta": 0.0,
                            "learner": learner,
                            "method": "orthogonal_single_split",
                            "n_valid": 300,
                            "n_failed": 0,
                            "alpha": 0.05,
                        }
                    )
        family, complete = size_family(pd.DataFrame(rows), single_split=True)
        self.assertTrue(complete)
        self.assertEqual(family, SINGLE_SPLIT_HOLM_FAMILY)

        partial_family, partial_complete = size_family(
            pd.DataFrame(rows[:4]), single_split=True
        )
        self.assertFalse(partial_complete)
        self.assertEqual(partial_family, "nonproduction_single_split_4_null_cells")

    def test_propensity_tables_preserve_learner_and_cover_bins(self):
        import json

        raw = pd.DataFrame(
            [
                {
                    "task_key": "task-1",
                    "design": "pooled_random_shift",
                    "scenario": "additive",
                    "p": 20,
                    "delta": 0.0,
                    "rep": 1,
                    "learner": "gam",
                    "method": "orthogonal_crossfit",
                    "status": "ok",
                    "n": 4,
                    "score_n": 4,
                    "n1": 2,
                    "n0": 2,
                    "realized_group_fraction": 0.5,
                    "propensity_mse": 0.01,
                    "propensity_brier": 0.22,
                    "propensity_log_loss": 0.63,
                    "propensity_auc": 0.75,
                    "diagnostics_json": json.dumps(
                        {
                            "propensity_calibration_bins": [
                                {
                                    "bin": 1,
                                    "n": 2,
                                    "mean_prediction": 0.4,
                                    "observed_rate": 0.0,
                                    "minimum_prediction": 0.3,
                                    "maximum_prediction": 0.5,
                                },
                                {
                                    "bin": 2,
                                    "n": 2,
                                    "mean_prediction": 0.6,
                                    "observed_rate": 1.0,
                                    "minimum_prediction": 0.5,
                                    "maximum_prediction": 0.7,
                                },
                            ]
                        }
                    ),
                }
            ]
        )
        quality, bins, summary = propensity_diagnostic_tables(raw)
        self.assertEqual(quality.loc[0, "learner"], "gam")
        self.assertEqual(int(bins["bin_n"].sum()), 4)
        self.assertEqual(int(summary["observation_count"].sum()), 4)

    def test_single_split_propensity_tables_use_held_out_score_n(self):
        import json

        raw = pd.DataFrame(
            [
                {
                    "task_key": "single-split-task",
                    "design": "pooled_random_shift",
                    "scenario": "additive",
                    "p": 20,
                    "delta": 0.0,
                    "rep": 1,
                    "learner": "gam",
                    "method": "orthogonal_single_split",
                    "status": "ok",
                    "n": 8,
                    "score_n": 4,
                    "n1": 4,
                    "n0": 4,
                    "realized_group_fraction": 0.5,
                    "propensity_mse": 0.01,
                    "propensity_brier": 0.22,
                    "propensity_log_loss": 0.63,
                    "propensity_auc": 0.75,
                    "diagnostics_json": json.dumps(
                        {
                            "propensity_calibration_bins": [
                                {
                                    "bin": 1,
                                    "n": 2,
                                    "mean_prediction": 0.4,
                                    "observed_rate": 0.0,
                                    "minimum_prediction": 0.3,
                                    "maximum_prediction": 0.5,
                                },
                                {
                                    "bin": 2,
                                    "n": 2,
                                    "mean_prediction": 0.6,
                                    "observed_rate": 1.0,
                                    "minimum_prediction": 0.5,
                                    "maximum_prediction": 0.7,
                                },
                            ]
                        }
                    ),
                }
            ]
        )

        quality, bins, summary = propensity_diagnostic_tables(raw)

        self.assertEqual(quality.loc[0, "fit_regime"], "single_split_testing")
        self.assertEqual(int(quality.loc[0, "score_n"]), 4)
        self.assertEqual(int(bins["bin_n"].sum()), 4)
        self.assertEqual(int(summary["observation_count"].sum()), 4)
        self.assertEqual(METHOD_LABELS["orthogonal_single_split"], "Single split")
        self.assertLess(
            METHOD_ORDER.index("orthogonal_single_split"),
            METHOD_ORDER.index("orthogonal_crossfit"),
        )

    def test_aggregate_results_reports_mean_score_n(self):
        rows = []
        for rep, score_n, reject in ((1, 4, 0), (2, 3, 1)):
            rows.append(
                {
                    "design": "pooled_random_shift",
                    "scenario": "additive",
                    "p": 20,
                    "delta": 0.5,
                    "learner": "gam",
                    "method": "orthogonal_single_split",
                    "alpha": 0.05,
                    "rep": rep,
                    "status": "ok",
                    "reject": reject,
                    "runtime_seconds": 0.1,
                    "mu_mse": 0.2,
                    "propensity_mse": 0.01,
                    "propensity_brier": 0.22,
                    "propensity_log_loss": 0.63,
                    "propensity_auc": 0.75,
                    "n1": 4,
                    "n0": 4,
                    "score_n": score_n,
                    "realized_group_fraction": 0.5,
                    "train_mse": 0.1,
                    "validation_mse": 0.2,
                    "selected_epoch": 1,
                    "parameter_count": 10,
                    "fallback_used": False,
                }
            )

        summary = aggregate_results(pd.DataFrame(rows))

        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary.loc[0, "mean_score_n"], 3.5)

    def test_holm_failure_propagates_to_entire_power_curve(self):
        rows = []
        for method, null_rate in (("orthogonal_nosplit", 0.05), ("residual", 0.30)):
            for delta, rate in ((0.0, null_rate), (0.5, 0.80)):
                rows.append(
                    {
                        "design": "pooled_random_shift",
                        "scenario": "additive",
                        "p": 20,
                        "delta": delta,
                        "learner": "gam",
                        "method": method,
                        "alpha": 0.05,
                        "rejection_rate": rate,
                        "n_valid": 300,
                    }
                )
        summary = pd.DataFrame(rows)
        sizes = size_audit(summary)
        annotated = attach_size_gate(summary, sizes)

        orthogonal = annotated[annotated["method"] == "orthogonal_nosplit"]
        residual = annotated[annotated["method"] == "residual"]
        self.assertEqual(set(orthogonal["size_gate_status"]), {"pass"})
        self.assertEqual(set(residual["size_gate_status"]), {"fail_holm"})

    def test_power_only_summary_is_marked_not_evaluated(self):
        summary = pd.DataFrame(
            [
                {
                    "design": "pooled_random_shift",
                    "scenario": "additive",
                    "p": 20,
                    "delta": 0.5,
                    "learner": "gam",
                    "method": "orthogonal_nosplit",
                    "alpha": 0.05,
                    "rejection_rate": 0.5,
                    "n_valid": 300,
                }
            ]
        )
        annotated = attach_size_gate(summary, size_audit(summary))
        self.assertEqual(annotated.loc[0, "size_gate_status"], "not_evaluated")

    def test_primary_and_diagnostic_plot_sets_are_created(self):
        rows = []
        for method, status in (
            ("orthogonal_nosplit", "pass"),
            ("residual", "fail_holm"),
        ):
            for delta, rate in ((0.0, 0.05), (0.5, 0.50)):
                rows.append(
                    {
                        "design": "pooled_random_shift",
                        "scenario": "additive",
                        "p": 20,
                        "delta": delta,
                        "learner": "gam",
                        "method": method,
                        "alpha": 0.05,
                        "rejection_rate": rate,
                        "wilson_lower": max(0.0, rate - 0.04),
                        "wilson_upper": min(1.0, rate + 0.04),
                        "size_gate_status": status,
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            files = make_plots(pd.DataFrame(rows), Path(temporary))
            self.assertEqual(len(files), 4)
            self.assertTrue(any("primary_size_valid" in path for path in files))
            self.assertTrue(any("diagnostic_all_methods" in path for path in files))
            self.assertTrue(all((Path(temporary) / path).exists() for path in files))

    def test_extended_grid_creates_full_and_zoomed_plot_sets(self):
        rows = []
        for delta, rate in (
            (0.0, 0.05),
            (1.2, 0.55),
            (6.0, 0.85),
            (20.0, 1.0),
        ):
            rows.append(
                {
                    "design": "pooled_random_shift",
                    "scenario": "neural_composition",
                    "p": 20,
                    "delta": delta,
                    "learner": "dnn",
                    "method": "orthogonal_nosplit",
                    "alpha": 0.05,
                    "rejection_rate": rate,
                    "wilson_lower": max(0.0, rate - 0.04),
                    "wilson_upper": min(1.0, rate + 0.04),
                    "size_gate_status": "pass",
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            files = make_plots(pd.DataFrame(rows), Path(temporary))
            self.assertEqual(len(files), 12)
            self.assertTrue(any("zoom_delta_le_1_6" in path for path in files))
            self.assertTrue(any("zoom_delta_le_6" in path for path in files))
            self.assertTrue(all((Path(temporary) / path).exists() for path in files))


if __name__ == "__main__":
    unittest.main()
