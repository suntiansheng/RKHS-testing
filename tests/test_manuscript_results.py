"""Check the numerical claims in the manuscript against the distributed tables."""

import csv
import math
from pathlib import Path
import unittest


RESULTS = Path(__file__).resolve().parents[1] / "results"
SCENARIOS = {"additive", "product", "composition", "neural_composition"}
LEARNERS = ("gam", "rf", "lasso", "dnn")


def rows(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main_null(name, method):
    return [row for row in rows(name)
            if row["scenario"] in SCENARIOS
            and row["method"] == method and float(row["delta"]) == 0]


class MainSimulationResults(unittest.TestCase):
    def test_each_displayed_procedure_has_32_complete_null_cells(self):
        for name, method in (
            ("gcm_paired_nord_summary.csv", "orthogonal_nosplit"),
            ("single_split_summary.csv", "orthogonal_single_split"),
            ("frozen_summary.csv", "yan"),
            ("gcm_summary.csv", "gcm"),
        ):
            with self.subTest(method=method):
                null = main_null(name, method)
                expected = {(scenario, p, learner) for scenario in SCENARIOS
                            for p in (20, 50) for learner in LEARNERS}
                self.assertEqual(len(null), 32)
                self.assertEqual(
                    {(row["scenario"], int(row["p"]), row["learner"]) for row in null},
                    expected,
                )
                self.assertTrue(all(int(row["n_valid"]) == 300
                                    and int(row["n_failed"]) == 0 for row in null))

    def test_reported_elevated_hierarchical_null_rates(self):
        for name, method, p, rejections in (
            ("frozen_summary.csv", "yan", 20, 33),
            ("single_split_summary.csv", "orthogonal_single_split", 50, 29),
        ):
            with self.subTest(method=method):
                selected = [row for row in main_null(name, method)
                            if row["scenario"] == "neural_composition"
                            and int(row["p"]) == p and row["learner"] == "rf"]
                self.assertEqual(len(selected), 1)
                self.assertAlmostEqual(float(selected[0]["rejection_rate"]), rejections / 300)

    def test_pcm_exceeds_nominal_size_in_15_of_16_displayed_cells(self):
        null = [row for row in rows("pcm_summary.csv")
                if row["scenario"] in SCENARIOS and float(row["delta"]) == 0]
        self.assertEqual(len(null), 16)
        expected = {(scenario, p, method) for scenario in SCENARIOS
                    for p in (20, 50)
                    for method in ("pcm_single_split", "pcm_derandomized")}
        self.assertEqual({(row["scenario"], int(row["p"]), row["method"])
                          for row in null}, expected)
        self.assertTrue(all(row["learner"] == "pcm_ranger"
                            and int(row["n_valid"]) == 300
                            and int(row["n_failed"]) == 0 for row in null))
        self.assertEqual(sum(float(row["rejection_rate"]) > .05 for row in null), 15)


class CausalSimulationResults(unittest.TestCase):
    def check_null_counts(self, name, expected, oracle):
        null = [row for row in rows(f"{name}_hierarchical/null_size.csv")
                if row["method"] != "orthogonal_crossfit"]
        actual = {(row["learner"], row["method"]): int(row["rejections"])
                  for row in null}
        counts = {(learner, method): count for method, values in expected.items()
                  for learner, count in zip(LEARNERS, values)}
        counts[("oracle", "oracle_score")] = oracle
        self.assertEqual(len(null), 13)
        self.assertEqual(actual, counts)
        for row in null:
            self.assertEqual(int(row["n_valid"]), 300)
            self.assertEqual(int(row["n_failed"]), 0)
            self.assertEqual(int(row["n_missing"]), 0)
            self.assertEqual(int(row["bootstrap_draws"]), 200)
            self.assertAlmostEqual(float(row["rejection_rate"]),
                                   int(row["rejections"]) / 300)

    def test_cate_null_table(self):
        self.check_null_counts("cate", {
            "orthogonal_nosplit": (14, 16, 13, 15),
            "orthogonal_single_split": (15, 18, 11, 13),
            "yan": (24, 32, 22, 26),
        }, oracle=15)

    def test_late_null_table(self):
        self.check_null_counts("late", {
            "orthogonal_nosplit": (14, 23, 15, 16),
            "orthogonal_single_split": (13, 18, 14, 10),
            "yan": (14, 32, 14, 18),
        }, oracle=13)


class CasinoResults(unittest.TestCase):
    def test_sample_dimensions_and_multiplier_grid(self):
        tests = rows("casino/regression_tests.csv")
        self.assertEqual(len(tests), 136)
        keys = [(row["selection"], row["covariates"], row["outcome"],
                 row["learner"], row["regime"], row["bandwidth"]) for row in tests]
        self.assertEqual(len(keys), len(set(keys)))
        main = [row for row in tests if row["selection"] == "main"]
        self.assertEqual(len(main), 40)
        for row in main:
            self.assertEqual((int(row["n"]), int(row["n0"]), int(row["n1"])),
                             (232, 73, 159))
            self.assertEqual(int(row["dimensions"]),
                             {"baseline": 36, "baseline_text": 164}[row["covariates"]])
        for row in tests:
            p = float(row["p_value"])
            numerator = int(row["bootstrap_numerator"])
            self.assertTrue(math.isfinite(p))
            self.assertGreaterEqual(p, 1 / 5001 - 1e-15)
            self.assertLessEqual(p, 1)
            self.assertTrue(1 <= numerator <= 5001)
            self.assertEqual(int(row["bootstrap_draws"]), 5000)
            self.assertAlmostEqual(p * 5001, numerator)
            self.assertEqual(int(row["reject_0_05"]), int(p <= .05))

    def test_reference_pvalues_and_bandwidth_checks(self):
        tests = [row for row in rows("casino/regression_tests.csv")
                 if row["selection"] == "main"]
        learner_order = ("ridge", "lasso", "random_forest")
        expected = {
            ("baseline", "no_split"): (5, 1, 6),
            ("baseline", "crossfit_5fold"): (5, 2, 3),
            ("baseline_text", "no_split"): (6, 7, 12),
            ("baseline_text", "crossfit_5fold"): (10, 7, 4),
        }
        for (covariates, regime), counts in expected.items():
            for learner, count in zip(learner_order, counts):
                selected = [row for row in tests if row["covariates"] == covariates
                            and row["regime"] == regime and row["learner"] == learner
                            and float(row["bandwidth"]) == 1]
                self.assertEqual(len(selected), 2)
                for row in selected:
                    expected_count = count if row["outcome"] == "mean_satisfaction" else 1
                    self.assertEqual(int(row["bootstrap_numerator"]), expected_count)
        for row in tests:
            if float(row["bandwidth"]) == .5:
                self.assertGreater(float(row["p_value"]), .05)
            elif float(row["bandwidth"]) == 2:
                self.assertEqual(int(row["bootstrap_numerator"]), 1)

    def test_alternative_selection_ranges(self):
        tests = [row for row in rows("casino/regression_tests.csv")
                 if row["selection"].startswith("alternative_")]
        self.assertEqual(len(tests), 80)
        mean = [row for row in tests if row["outcome"] == "mean_satisfaction"]
        self.assertLessEqual(max(int(row["bootstrap_numerator"]) for row in mean), 33)
        expected_max = {
            ("baseline", "no_split"): .123,
            ("baseline", "crossfit_5fold"): .095,
            ("baseline_text", "no_split"): .092,
            ("baseline_text", "crossfit_5fold"): .044,
        }
        for (covariates, regime), expected in expected_max.items():
            selected = [float(row["p_value"]) for row in tests
                        if row["outcome"] == "both_extremely_satisfied"
                        and row["covariates"] == covariates and row["regime"] == regime]
            self.assertEqual(len(selected), 10)
            self.assertEqual(round(max(selected), 3), expected)
        overlap = [row for row in rows("casino/selection_overlap.csv")
                   if row["selection_a"] == "main"]
        self.assertEqual(len(overlap), 10)
        for column, expected in (("shared_dialogues", (135, 147)),
                                 ("shared_participants", (395, 413))):
            values = [int(row[column]) for row in overlap]
            self.assertEqual((min(values), max(values)), expected)


class InterpolationResults(unittest.TestCase):
    def test_full_sample_interpolation_never_rejects(self):
        summary = rows("interpolation.csv")
        self.assertEqual(len(summary), 12)
        full = [row for row in summary if row["method"] == "distance_full"]
        self.assertEqual({float(row["amplitude"]) for row in full}, {0, .25, .5, 1})
        self.assertTrue(all(int(row["replications"]) == 500 for row in summary))
        self.assertTrue(all(int(row["rejections"]) == 0 for row in full))


if __name__ == "__main__":
    unittest.main()
