"""Check IV-specific provenance, target validation, and the LATE size family."""

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
import late_report as report
from test_cate_report import data as cate_data


def config():
    return json.loads((ROOT / "simulation/configs/late.json").read_text())


def data(*, reps=3):
    raw = cate_data(reps=reps)
    raw["design"] = "late_iv"
    raw["task_key"] = raw.task_key.str.replace("cate|", "late|", regex=False)
    diagnostic = raw.drop_duplicates("task_key")[[
        "task_key", "design", "scenario", "p", "delta", "rep", "learner", "n0", "n1"]].copy()
    diagnostic = diagnostic.assign(group_variable="instrument", n_always_takers=80,
        n_compliers=240, n_never_takers=80, n_treated=200,
        n_treated_instrument0=40, n_treated_instrument1=160,
        first_stage_population=.6, first_stage_empirical=.6,
        late_empirical_l2=0., reduced_form_empirical_l2=0., **report.LATE_OPTIONS)
    return raw, diagnostic


def write_run(directory, raw, diagnostic, *, source="core"):
    directory.mkdir(parents=True, exist_ok=True)
    settings = config()
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    (directory / "metadata.json").write_text(json.dumps({"config": settings,
        "config_hash": hashlib.sha256(encoded.encode()).hexdigest(),
        "scientific_core_sha256": source, "runner_sha256": "runner"}))
    (directory / "calibrations.json").write_text(json.dumps({"calibrations": ["frozen"]}))
    raw.to_csv(directory / "raw_results.csv", index=False)
    diagnostic.to_csv(directory / "dgp_diagnostics.csv", index=False)


class LateReportTests(unittest.TestCase):
    def test_config_requires_iv_construction_and_existing_tuning(self):
        report.validate_config(config())
        for field, value in (("designs", ["pooled_random_shift"]),
                             ("treatment_noise_scales", [1., 1.]),
                             ("bootstrap_draws", 199)):
            settings = config()
            settings[field] = value
            with self.assertRaisesRegex(ValueError, field):
                report.validate_config(settings)
        settings = config()
        settings["late_options"]["complier_probability"] = .2
        with self.assertRaisesRegex(ValueError, "late_options"):
            report.validate_config(settings)
        settings = config()
        settings["learner_options"]["rf"]["n_estimators"] = 50
        with self.assertRaisesRegex(ValueError, "learner_options"):
            report.validate_config(settings)

    def test_instrument_and_reduced_form_diagnostics_are_checked(self):
        raw, diagnostic = data()
        report.validate_results(raw, config())
        report.validate_diagnostics(diagnostic, raw)
        for column, value, message in (
            ("group_variable", "treatment", "instrument"),
            ("n_compliers", 239, "counts"),
            ("first_stage_empirical", .7, "first stages"),
            ("reduced_form_empirical_l2", .2, "Reduced-form"),
            ("rep", 40, "disagrees"),
        ):
            changed = diagnostic.copy()
            changed.loc[changed.index[0], column] = value
            with self.assertRaisesRegex(ValueError, message):
                report.validate_diagnostics(changed, raw)
        with self.assertRaisesRegex(ValueError, "do not match"):
            report.validate_diagnostics(diagnostic.iloc[1:], raw)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report.validate_diagnostics(pd.concat([diagnostic, diagnostic.iloc[:1]]), raw)

    def test_screen_keeps_13_late_cells_and_withholds_incomplete_family(self):
        raw, _ = data(reps=300)
        selected = raw.learner.eq("gam") & raw.method.eq("orthogonal_nosplit")
        raw.loc[selected, "reject"] = (raw.loc[selected, "rep"] <= 60).astype(int)
        raw.loc[selected, "p_value"] = np.where(raw.loc[selected, "reject"], .01, .7)
        summary = report.summarize(raw)
        null = report.screen_size(summary)
        self.assertEqual(len(summary), 156)
        self.assertEqual(len(null), 13)
        self.assertTrue(null.family.eq("late_13_cells").all())
        self.assertEqual(int(null.learner.eq("oracle").sum()), 1)
        flagged = null[null.size_screen.eq("flagged")]
        self.assertEqual(len(flagged), 1)
        self.assertAlmostEqual(flagged.iloc[0].holm_adjusted_p_value,
                               13 * binomtest(60, 300, .05).pvalue)
        raw.loc[0, ["status", "reject", "p_value"]] = ["failed", np.nan, np.nan]
        summary = report.summarize(raw)
        self.assertEqual((summary.iloc[0].n_valid, summary.iloc[0].n_failed), (299, 1))
        self.assertTrue(report.screen_size(summary).size_screen.eq("not_evaluated").all())

    def test_archives_preserve_iv_diagnostics_and_hashes_and_reject_mixed_sources(self):
        raw, diagnostic = data()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left, right = directory / "left", directory / "right"
            write_run(left, raw[raw.rep.eq(1)], diagnostic[diagnostic.rep.eq(1)])
            write_run(right, raw[raw.rep.gt(1)], diagnostic[diagnostic.rep.gt(1)])
            combined, iv, _, _ = report.load_inputs([left, right])
            self.assertEqual(len(combined), len(raw))
            self.assertEqual(len(iv), len(diagnostic))
            output = directory / "report"
            report.main(["--inputs", str(left), str(right), "--output", str(output), "--skip-plots"])
            text = (output / "late_report.md").read_text()
            self.assertIn("INCOMPLETE", text)
            self.assertIn("null results are incomplete", text)
            self.assertIn("ATE is 0.4", text)
            self.assertFalse(list(output.glob("*.pdf")))
            manifest = json.loads((output / "report_manifest.json").read_text())
            for name, frame in (("combined_raw_results", raw), ("combined_dgp_diagnostics", diagnostic)):
                archive = output / f"{name}.csv.gz"
                self.assertEqual(len(pd.read_csv(archive)), len(frame))
                self.assertEqual(manifest[name]["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertEqual(manifest["size_family"], "late_13_cells")
            self.assertIn("shared_report_script_sha256", manifest)
            write_run(right, raw[raw.rep.gt(1)], diagnostic[diagnostic.rep.gt(1)], source="changed core")
            with self.assertRaisesRegex(ValueError, "inconsistent configurations"):
                report.load_inputs([left, right])

    def test_figure_keeps_four_panels_and_all_curves_without_shading(self):
        import matplotlib.figure
        raw, _ = data()
        summary = report.summarize(raw)
        captured = []
        with patch.object(matplotlib.figure.Figure, "savefig",
                          autospec=True, side_effect=lambda figure, *args, **kwargs: captured.append(figure)):
            report.plot_power(summary, report.screen_size(summary), Path("unused"))
        self.assertEqual(len(captured), 2)
        figure = captured[0]
        self.assertEqual(len(figure.axes), 4)
        self.assertTrue(all(len(axis.lines) == 5 for axis in figure.axes))
        self.assertTrue(all(len(axis.collections) == 0 for axis in figure.axes))
        self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()],
                         [row[1] for row in report.METHODS])
        self.assertFalse(any("Shading" in text.get_text() or "Amplitude grid" in text.get_text()
                             for text in figure.texts))


if __name__ == "__main__":
    unittest.main()
