from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


APPLICATION_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = APPLICATION_DIR / "analyze.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_integrative_potential", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
integrative = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = integrative
SPEC.loader.exec_module(integrative)


class MultiplierTests(unittest.TestCase):
    def test_centered_feature_covariance_and_plus_one_value(self) -> None:
        rng = np.random.default_rng(2108)
        features = rng.normal(size=(13, 4))
        scores = rng.normal(size=13)
        kernel = features @ features.T
        draws, seed = 97, 812
        result = integrative.core.multiplier_test(
            scores,
            kernel,
            bootstrap_draws=draws,
            seed=seed,
        )
        score_features = scores[:, None] * features
        centered = score_features - score_features.mean(axis=0)
        multipliers = np.random.default_rng(seed).standard_normal((13, draws))
        bootstrap = np.linalg.norm(centered.T @ multipliers, axis=0) / np.sqrt(13)
        observed = np.linalg.norm(score_features.sum(axis=0)) / np.sqrt(13)
        self.assertAlmostEqual(result["statistic"], observed, places=12)
        self.assertAlmostEqual(
            result["critical_value_0_05"], np.quantile(bootstrap, 0.95), places=12
        )
        self.assertEqual(
            result["p_value"],
            (1 + np.count_nonzero(bootstrap >= observed)) / (draws + 1),
        )

    def test_zero_scores_use_weak_upper_tail(self) -> None:
        result = integrative.core.multiplier_test(
            np.zeros(5),
            np.eye(5),
            bootstrap_draws=19,
            seed=3,
        )
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["reject_0_05"], 0)


class IntegrativePotentialAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus_dir = Path(
            os.environ.get(
                "CASINO_CORPUS_DIR", APPLICATION_DIR / "data" / "raw" / "casino-corpus"
            )
        )
        if not cls.corpus_dir.is_dir():
            raise unittest.SkipTest(
                "Download the CaSiNo corpus or set CASINO_CORPUS_DIR."
            )
        cls.archive_path = cls.corpus_dir.parent / "casino-corpus.zip"
        cls.results_dir = Path(
            os.environ.get(
                "CASINO_RESULTS_DIR", APPLICATION_DIR / "results_integrative_potential"
            )
        )
        cls.records, cls.corpus_counts = integrative.load_integrative_potential_records(
            cls.corpus_dir
        )
        cls.eligible = [record for record in cls.records if record["eligible_extreme"]]
        cls.sample, cls.sample_hash = integrative.frozen_main_sample(cls.eligible)

    def test_population_sample_and_outcomes_are_frozen(self) -> None:
        self.assertEqual(
            self.corpus_counts,
            {
                "speakers": 846,
                "dialogues": 1030,
                "utterances": 14297,
                "natural_utterances": 11919,
                "control_utterances": 2378,
                "aligned_m36": 257,
                "intermediate_m39": 378,
                "maximal_m42": 395,
                "eligible_extreme_dialogues": 652,
            },
        )
        self.assertEqual(
            self.sample["population"].value_counts().to_dict(),
            {"maximal": 159, "aligned": 73},
        )
        speakers = pd.concat([self.sample["speaker_a"], self.sample["speaker_b"]])
        self.assertEqual(speakers.nunique(), 2 * len(self.sample))
        self.assertEqual(self.sample_hash, integrative.MAIN_SAMPLE_SHA256)
        self.assertTrue((self.sample["early_turns"] == 6).all())
        self.assertTrue((self.sample["early_speaker_a_turns"] == 3).all())
        self.assertTrue((self.sample["early_speaker_b_turns"] == 3).all())
        self.assertEqual(
            self.sample.groupby("D")["both_extremely_satisfied"].sum().to_dict(),
            {0: 10, 1: 64},
        )
        self.assertAlmostEqual(
            self.sample.loc[self.sample["D"] == 0, "mean_satisfaction"].mean(),
            4.0479452054794525,
        )
        self.assertAlmostEqual(
            self.sample.loc[self.sample["D"] == 1, "mean_satisfaction"].mean(),
            4.493710691823899,
        )

    def test_archive_and_text_only_design_are_valid(self) -> None:
        member_hashes = integrative.core.verify_extracted_corpus(
            self.archive_path, self.corpus_dir
        )
        self.assertEqual(set(member_hashes), set(integrative.core.CORPUS_MEMBERS))
        design, metadata = integrative.build_text_design(self.sample)
        self.assertEqual(design.shape, (232, 128))
        self.assertEqual(metadata["dimensions"], 128)
        self.assertIn("excluded", metadata["priority_features"])
        self.assertTrue(np.isfinite(design).all())
        self.assertTrue(np.allclose(np.linalg.norm(design, axis=1), 1.0))
        self.assertAlmostEqual(
            integrative.core.median_bandwidth(design), 1.1056512617463308
        )

    def test_result_schemas_and_primary_markers(self) -> None:
        if not self.results_dir.is_dir():
            self.skipTest("Run the analysis or set CASINO_RESULTS_DIR.")
        paths = {
            "sample": self.results_dir / "integrative_potential_sample.csv",
            "design": self.results_dir / "integrative_potential_design.npz",
            "predictions": self.results_dir
            / "integrative_potential_nuisance_predictions.csv",
            "tests": self.results_dir / "integrative_potential_rkhs_results.csv",
            "selections": self.results_dir
            / "integrative_potential_selection_sensitivity.csv",
            "words": self.results_dir / "integrative_potential_word_log_odds.csv",
        }
        self.assertTrue(
            all(path.exists() for path in paths.values()),
            "Analysis result artifacts are missing.",
        )
        sample = pd.read_csv(paths["sample"])
        predictions = pd.read_csv(paths["predictions"])
        results = pd.read_csv(paths["tests"])
        selections = pd.read_csv(paths["selections"])
        words = pd.read_csv(paths["words"])
        self.assertTrue(
            {"dialogue_id", "D", "population", "mean_satisfaction"} <= set(sample)
        )
        self.assertTrue(
            {"dialogue_id", "D", "propensity_ridge_crossfit"} <= set(predictions)
        )
        self.assertTrue(
            {
                "learner",
                "regime",
                "outcome",
                "statistic",
                "critical_value_0_05",
                "p_value",
                "bootstrap_draws",
                "bootstrap_exceedances",
                "bootstrap_plus_one_numerator",
            }
            <= set(results)
        )
        self.assertEqual(len(results), 24)
        self.assertEqual(results["is_primary"].sum(), 1)
        primary = results.loc[results["is_primary"] == 1].iloc[0]
        self.assertEqual(
            (primary["outcome"], primary["learner"], primary["regime"]),
            ("mean_satisfaction", "ridge", "no_split"),
        )
        self.assertEqual(primary["bandwidth_spec"], "fixed_h_1")
        self.assertEqual(results["is_ordinal_sensitivity"].sum(), 1)
        lattice = results["p_value"] * (results["bootstrap_draws"] + 1)
        self.assertTrue(np.allclose(lattice, np.round(lattice)))
        np.testing.assert_array_equal(
            results["bootstrap_plus_one_numerator"], np.round(lattice).astype(int)
        )
        np.testing.assert_array_equal(
            results["bootstrap_exceedances"], np.round(lattice).astype(int) - 1
        )
        self.assertEqual(len(selections), 11)
        self.assertEqual(selections["is_primary"].sum(), 1)
        self.assertTrue(
            {
                "term",
                "log_odds_maximal_minus_aligned",
                "z_score",
                "abs_z_score",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "favored_population",
            }
            <= set(words)
        )
        self.assertTrue(set(words["term"]).isdisjoint(integrative.SCENARIO_NOUNS))

    def test_all_saved_test_rows_replay_from_saved_inputs_and_seeds(self) -> None:
        if not self.results_dir.is_dir():
            self.skipTest("Run the analysis or set CASINO_RESULTS_DIR.")
        sample = pd.read_csv(self.results_dir / "integrative_potential_sample.csv")
        predictions = pd.read_csv(
            self.results_dir / "integrative_potential_nuisance_predictions.csv"
        )
        results = pd.read_csv(
            self.results_dir / "integrative_potential_rkhs_results.csv"
        )
        with np.load(
            self.results_dir / "integrative_potential_design.npz"
        ) as design_file:
            design = design_file["X"]
            d = design_file["D"].astype(int)
            dialogue_ids = design_file["dialogue_id"].astype(int)
        np.testing.assert_array_equal(sample["dialogue_id"], dialogue_ids)
        np.testing.assert_array_equal(predictions["dialogue_id"], dialogue_ids)
        np.testing.assert_array_equal(predictions["D"], d)
        self.assertEqual(sorted(predictions["crossfit_fold"].unique()), [1, 2, 3, 4, 5])
        kernels = {
            "fixed_h_1": integrative.core.rbf_kernel_matrix(
                design, integrative.FIXED_BANDWIDTH
            ),
            "pooled_median_sensitivity": integrative.core.rbf_kernel_matrix(
                design, integrative.core.median_bandwidth(design)
            ),
        }
        replay_fields = (
            "statistic",
            "statistic_squared",
            "critical_value_0_05",
            "p_value",
            "bootstrap_mean",
            "bootstrap_sd",
        )
        for row in results.itertuples(index=False):
            short_regime = "crossfit" if row.regime == "crossfit_5fold" else row.regime
            y = sample[row.outcome].to_numpy(dtype=float)
            fitted_outcome = predictions[
                f"outcome_{row.outcome}_{row.learner}_{short_regime}"
            ].to_numpy(dtype=float)
            propensity = predictions[
                f"propensity_{row.learner}_{short_regime}"
            ].to_numpy(dtype=float)
            score = (y - fitted_outcome) * (d - propensity)
            replay = integrative.core.multiplier_test(
                score,
                kernels[row.bandwidth_spec],
                bootstrap_draws=int(row.bootstrap_draws),
                seed=int(row.bootstrap_seed),
            )
            for field in replay_fields:
                self.assertAlmostEqual(
                    float(getattr(row, field)), float(replay[field]), places=12
                )
            self.assertEqual(int(row.reject_0_05), int(replay["reject_0_05"]))

    def test_selection_sensitivity_is_disjoint_and_reproducible(self) -> None:
        if not self.results_dir.is_dir():
            self.skipTest("Run the analysis or set CASINO_RESULTS_DIR.")
        saved = pd.read_csv(
            self.results_dir / "integrative_potential_selection_sensitivity.csv"
        )
        for row in saved.itertuples(index=False):
            sample, sample_hash = integrative.outcome_blind_disjoint_sample(
                self.eligible, salt=row.selection_salt
            )
            speakers = pd.concat([sample["speaker_a"], sample["speaker_b"]])
            self.assertEqual(speakers.nunique(), 2 * len(sample))
            self.assertEqual(sample_hash, row.sample_sha256)
            self.assertEqual(len(sample), row.n)
            self.assertEqual(int((sample["D"] == 0).sum()), row.n_aligned)
            self.assertEqual(int((sample["D"] == 1).sum()), row.n_maximal)
        tests = pd.read_csv(self.results_dir / "integrative_potential_rkhs_results.csv")
        primary = tests.loc[tests["is_primary"] == 1].iloc[0]
        main = saved.loc[saved["is_primary"] == 1].iloc[0]
        self.assertAlmostEqual(main["statistic"], primary["statistic"], places=12)
        self.assertAlmostEqual(main["p_value"], primary["p_value"], places=12)

    def test_summary_paths_are_relative_and_hashes_match(self) -> None:
        if not self.results_dir.is_dir():
            self.skipTest("Run the analysis or set CASINO_RESULTS_DIR.")
        summary_path = self.results_dir / "integrative_potential_analysis_summary.json"
        self.assertTrue(summary_path.exists(), "Analysis summary is missing.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            summary["sample"]["sorted_dialogue_id_sha256"], self.sample_hash
        )
        self.assertEqual(summary["sample"]["unique_participants"], 464)
        self.assertEqual(
            summary["run_configuration"]["word_filtered_scenario_nouns"],
            sorted(integrative.SCENARIO_NOUNS),
        )
        for relative_path, expected_hash in summary["outputs"]["sha256"].items():
            self.assertFalse(Path(relative_path).is_absolute())
            self.assertEqual(
                integrative.core.file_sha256(self.results_dir / relative_path),
                expected_hash,
            )


if __name__ == "__main__":
    unittest.main()
