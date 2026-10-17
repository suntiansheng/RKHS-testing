"""Check CATE reporting, paired counts, and the distinct 17-cell size family."""

import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import cate_report as report


def config():
    result = json.loads((ROOT / "simulation/configs/cate.json").read_text())
    result.update(name="cate_test", scenarios=["neural_composition"], dimensions=[20],
                  treatment_noise_scales=[1., 1.5], methods=list(report.METHOD_LABELS),
                  crossfit_folds=2)
    return result


def data(*, reps=3, amplitudes=(0.,), cells=report.CELLS):
    rows = []
    for learner, method in cells:
        for amplitude in amplitudes:
            for rep in range(1, reps + 1):
                hit = int(rep % 20 == 0)
                rows.append(dict(task_key=f"cate|{amplitude}|{rep}|{learner}",
                    design="pooled_random_shift", scenario="neural_composition", n=400, p=20,
                    rho=.5, sigma=1., delta=amplitude, rep=rep, learner=learner, method=method,
                    alpha=.05, bootstrap_draws=200, dgp_seed=1000 + rep,
                    fit_seed=2000 + rep, bootstrap_seed=3000 + rep + int(amplitude * 100),
                    n0=200, n1=200, status="ok", reject=hit, p_value=.01 if hit else .7,
                    error=""))
    return pd.DataFrame(rows)


def write_run(directory, raw, *, settings=None, source="core"):
    directory.mkdir(parents=True, exist_ok=True)
    settings = settings or config()
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    (directory / "metadata.json").write_text(json.dumps({"config": settings,
        "config_hash": hashlib.sha256(encoded.encode()).hexdigest(),
        "scientific_core_sha256": source, "runner_sha256": "runner"}))
    (directory / "calibrations.json").write_text(json.dumps({"calibrations": ["frozen"]}))
    raw.to_csv(directory / "raw_results.csv", index=False)


class CateReportTests(unittest.TestCase):
    def test_compressed_csv_is_deterministic_and_preserves_contents(self):
        raw = data()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / "first.csv.gz", directory / "second.csv.gz"
            report.atomic_write_gzip_csv(first, raw)
            report.atomic_write_gzip_csv(second, raw)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes()[4:8], b"\0\0\0\0")
            self.assertEqual(gzip.decompress(first.read_bytes()), raw.to_csv(index=False).encode())
            self.assertEqual(len(list(directory.iterdir())), 2)

    def test_summary_keeps_missing_cells_and_counts_oracle_once(self):
        raw = data()
        report.validate_results(raw, config())
        summary = report.summarize(raw)
        self.assertEqual(len(summary), 204)
        null = report.screen_size(summary)
        self.assertEqual(len(null), 17)
        self.assertEqual(int(null.learner.eq("oracle").sum()), 1)
        self.assertEqual(int(summary.n_valid.sum()), 51)
        self.assertEqual(int(summary.n_missing.sum()), 61200 - 51)
        self.assertTrue(null.size_screen.eq("not_evaluated").all())
        self.assertTrue(null.holm_adjusted_p_value.isna().all())

    def test_failures_are_not_nonrejections_or_successful_runs(self):
        raw = data()
        raw.loc[0, ["status", "reject", "p_value", "error"]] = ["failed", np.nan, np.nan, "fit failed"]
        report.validate_results(raw, config())
        first = report.summarize(raw).iloc[0]
        self.assertEqual((first.n_observed, first.n_valid, first.n_failed, first.n_missing),
                         (3, 2, 1, 297))
        self.assertAlmostEqual(first.ci_upper, 1 - np.sqrt(.025))
        raw.loc[0, "reject"] = 0
        with self.assertRaisesRegex(ValueError, "Failed results"):
            report.validate_results(raw, config())

    def test_holm_uses_17_complete_distinct_cells(self):
        raw = data(reps=300)
        selected = raw.learner.eq("gam") & raw.method.eq("orthogonal_nosplit")
        raw.loc[selected, "reject"] = (raw.loc[selected, "rep"] <= 60).astype(int)
        summary = report.summarize(raw)
        null = report.screen_size(summary)
        self.assertTrue(null.family_complete.all())
        self.assertEqual(int(null.size_screen.eq("flagged").sum()), 1)
        flagged = null[null.size_screen.eq("flagged")].iloc[0]
        self.assertAlmostEqual(flagged.holm_adjusted_p_value,
                               17 * binomtest(60, 300, .05).pvalue)
        self.assertEqual(flagged.flag_direction, "above_nominal")
        self.assertFalse(summary.complete.all())
        summary.loc[summary.index[0], "n_valid"] = 299
        self.assertTrue(report.screen_size(summary).size_screen.eq("not_evaluated").all())

    def test_duplicate_natural_keys_and_unmatched_draws_are_rejected(self):
        raw = data()
        doubled = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report.validate_results(doubled, config())
        raw.loc[0, "dgp_seed"] += 9
        with self.assertRaisesRegex(ValueError, "matched groups"):
            report.validate_results(raw, config())

    def test_rejection_decisions_and_config_are_checked(self):
        raw = data()
        raw.loc[0, "p_value"] = .01
        with self.assertRaisesRegex(ValueError, "rejection decisions"):
            report.validate_results(raw, config())
        settings = config()
        settings["treatment_noise_scales"] = [1., 1.]
        with self.assertRaisesRegex(ValueError, "treatment_noise_scales"):
            report.validate_config(settings)
        for field, value in (("crossfit_folds", 5), ("seed_base", 10), ("propensity_clip", .02)):
            settings = config()
            settings[field] = value
            with self.assertRaisesRegex(ValueError, field):
                report.validate_config(settings)
        settings = config()
        settings["learner_options"]["rf"]["n_estimators"] = 50
        with self.assertRaisesRegex(ValueError, "learner_options"):
            report.validate_config(settings)

    def test_paired_difference_uses_only_matched_successful_replicates(self):
        raw = data(reps=4)
        nord = raw.learner.eq("gam") & raw.method.eq("orthogonal_nosplit")
        split = raw.learner.eq("gam") & raw.method.eq("orthogonal_single_split")
        raw.loc[nord, "reject"] = [1, 1, 0, 1]
        raw.loc[split, "reject"] = [0, 1, 1, 0]
        raw.loc[split & raw.rep.eq(4), "status"] = "failed"
        first = report.paired_differences(raw).iloc[0]
        self.assertEqual(first.n_pairs, 3)
        self.assertEqual(first.difference_nord_minus_single_split, 0)
        self.assertAlmostEqual(first.paired_monte_carlo_se, 1 / np.sqrt(3))
        self.assertEqual((first.nord_only_rejections, first.single_split_only_rejections), (1, 1))

    def test_crossfit_pairing_preserves_comparison_sign_and_legacy_columns(self):
        raw = data(reps=4)
        nord = raw.learner.eq("gam") & raw.method.eq("orthogonal_nosplit")
        crossfit = raw.learner.eq("gam") & raw.method.eq("orthogonal_crossfit")
        split = raw.learner.eq("gam") & raw.method.eq("orthogonal_single_split")
        raw.loc[nord, "reject"] = [0, 1, 0, 1]
        raw.loc[crossfit, "reject"] = [1, 1, 1, 0]
        raw.loc[split, "reject"] = [0, 0, 0, 0]
        raw.loc[crossfit & raw.rep.eq(4), "status"] = "failed"
        first = report.paired_crossfit_differences(raw).iloc[0]
        self.assertEqual(first.n_pairs, 3)
        self.assertAlmostEqual(first.difference_nord_minus_crossfit, -2 / 3)
        self.assertAlmostEqual(first.paired_monte_carlo_se, 1 / 3)
        self.assertEqual((first.nord_only_rejections, first.crossfit_only_rejections), (0, 2))
        self.assertEqual(report.paired_differences(raw).iloc[0].difference_nord_minus_single_split, .5)
        self.assertEqual(len(report.paired_crossfit_differences(raw)), 48)

    def test_main_and_supplement_figures_keep_distinct_method_sets(self):
        import matplotlib.figure
        raw = data()
        summary = report.summarize(raw)
        for include_crossfit in (False, True):
            with self.subTest(include_crossfit=include_crossfit):
                captured = []
                with patch.object(matplotlib.figure.Figure, "savefig", autospec=True,
                                  side_effect=lambda figure, path, **kwargs: captured.append((figure, path))):
                    if include_crossfit:
                        report.plot_power(summary, report.screen_size(summary), Path("unused"),
                                          include_crossfit=True)
                    else:
                        report.plot_power(summary, report.screen_size(summary), Path("unused"))
                expected_labels = ["NORD", "Single split", "Yan", "Oracle"]
                expected_colors = ["#0072B2", "#D55E00", "#009E73", "#333333"]
                if include_crossfit:
                    expected_labels.insert(1, "CF-NORD")
                    expected_colors.insert(1, "#CC79A7")
                suffix = "_crossfit" if include_crossfit else ""
                self.assertEqual([path.name for _, path in captured],
                                 [f"simulation_cate_hierarchical{suffix}.{extension}"
                                  for extension in ("pdf", "png")])
                figure = captured[0][0]
                self.assertEqual(len(figure.axes), 4)
                np.testing.assert_array_equal(figure.get_size_inches(), [7.6, 2.5])
                self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()], expected_labels)
                for axis in figure.axes:
                    self.assertEqual(len(axis.lines), len(expected_labels) + 1)
                    self.assertEqual([line.get_color() for line in axis.lines[:-1]], expected_colors)
                    self.assertEqual(len(axis.collections), 0)
                    self.assertEqual(axis.get_xlim(), (-.3, 20.3))
                    self.assertEqual(axis.get_ylim(), (-.025, 1.035))
                    np.testing.assert_array_equal(axis.get_xticks(), [0, 6, 12, 20])
                    if include_crossfit:
                        self.assertEqual(axis.lines[1].get_linestyle(), "--")
                self.assertFalse(any("Shading" in text.get_text() or "Amplitude grid" in text.get_text()
                                     for text in figure.texts))

    def test_incompatible_shards_rejected_and_incomplete_report_written(self):
        raw = data()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left, right = directory / "left", directory / "right"
            write_run(left, raw[raw.rep.le(1)])
            write_run(right, raw[raw.rep.gt(1)])
            combined, _, _ = report.load_inputs([left, right])
            self.assertEqual(len(combined), len(raw))
            output = directory / "report"
            report.main(["--inputs", str(left), str(right), "--output", str(output), "--skip-plots"])
            text = (output / "cate_report.md").read_text()
            self.assertIn("INCOMPLETE", text)
            self.assertIn("null results are incomplete", text)
            self.assertIn("CF-NORD", text)
            self.assertTrue((output / "paired_crossfit_differences.csv").exists())
            self.assertFalse(list(output.glob("*.pdf")))
            combined_path = output / "combined_raw_results.csv.gz"
            self.assertFalse((output / "combined_raw_results.csv").exists())
            self.assertEqual(len(pd.read_csv(combined_path)), len(raw))
            manifest = json.loads((output / "report_manifest.json").read_text())
            self.assertEqual(manifest["combined_raw_results"]["path"], combined_path.name)
            self.assertEqual(manifest["combined_raw_results"]["sha256"],
                             hashlib.sha256(combined_path.read_bytes()).hexdigest())
            write_run(right, raw[raw.rep.gt(1)], source="different core")
            with self.assertRaisesRegex(ValueError, "inconsistent configurations"):
                report.load_inputs([left, right])


if __name__ == "__main__":
    unittest.main()
