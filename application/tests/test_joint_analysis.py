"""Validate the nested covariate sets and the meaning of selection overlap."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline_analysis as baseline
import joint_analysis as joint


class JointDesignTests(unittest.TestCase):
    def test_joint_features_retain_baseline_and_exclude_labels_and_outcomes(self):
        def person(age, gender):
            return {"meta": {"demographics": {"age": age, "gender": gender,
                "ethnicity": "white american", "education": "high school graduate / ged"},
                "personality": {"svo": "prosocial", "big-five": dict.fromkeys(baseline.TRAITS, 4)}}}
        speakers = {"a": person(30, "female"), "b": person(50, "male")}
        sample = pd.DataFrame([dict(dialogue_id=1, speaker_a="a", speaker_b="b",
            D=0, early_text="Let us find an agreement together", mean_satisfaction=3.,
            both_extremely_satisfied=0, maximum_joint_points=36)])
        designs, _, _ = joint.combined_design(sample, speakers)
        self.assertEqual(designs["baseline"].shape, (1, 36))
        self.assertEqual(designs["baseline_text"].shape, (1, 164))
        np.testing.assert_array_equal(designs["baseline"], designs["baseline_text"][:, :36])
        np.testing.assert_allclose(np.linalg.norm(designs["baseline_text"][:, 36:], axis=1), 1)
        altered = sample.assign(D=1, mean_satisfaction=5., both_extremely_satisfied=1, maximum_joint_points=42)
        for name, X in joint.combined_design(altered, speakers)[0].items():
            np.testing.assert_array_equal(X, designs[name])
        changed_text = joint.combined_design(sample.assign(early_text="A different opening conversation"), speakers)[0]
        np.testing.assert_array_equal(changed_text["baseline"], designs["baseline"])
        self.assertFalse(np.array_equal(changed_text["baseline_text"], designs["baseline_text"]))

    def test_overlap_distinguishes_reused_dialogues_from_reused_participants(self):
        main = pd.DataFrame(dict(dialogue_id=[1, 2], speaker_a=["a", "c"], speaker_b=["b", "d"]))
        alternative = pd.DataFrame(dict(dialogue_id=[1, 3], speaker_a=["a", "c"], speaker_b=["b", "e"]))
        result = joint.selection_overlap({"main": main, "alternative_01": alternative}).iloc[0]
        self.assertEqual(result.shared_dialogues, 1)
        self.assertEqual(result.shared_participants, 3)
        self.assertAlmostEqual(result.dialogue_jaccard, 1/3)
        invalid = main.assign(speaker_b=["b", "a"])
        with self.assertRaises(AssertionError):
            joint.selection_overlap({"invalid": invalid})


if __name__ == "__main__":
    unittest.main()
