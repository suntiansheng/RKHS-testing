import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation"))
sys.path.insert(0, str(ROOT / "experiments"))
from gcm import gcm_test, gcm_from_moments
import gcm_comparison as experiment
import simcore as core
from run_simulation import build_tasks


class GCMTests(unittest.TestCase):
    def test_paper_normalization_and_archived_moments(self):
        score = np.array([-2., -.4, .1, .9, 1.7, 2.4])
        result = gcm_test(score)
        z = score.sum() / np.sqrt(np.sum((score-score.mean())**2))
        self.assertAlmostEqual(result["statistic"], z)
        self.assertAlmostEqual(result["p_value"], 2*norm.sf(abs(z)))
        self.assertEqual(result, gcm_from_moments(len(score), score.mean(), score.std(ddof=1)))
        self.assertAlmostEqual(gcm_test(-3*score)["p_value"], result["p_value"])

    def test_degenerate_or_invalid_scores_are_not_reported_as_valid(self):
        for score in [[], [1.], [0.,0.], [2.,2.], [0.,np.nan], [[0.,1.]]]:
            with self.assertRaises(ValueError):
                gcm_test(score)

    def test_cached_runner_reproduces_original_scores_for_every_learner(self):
        config = json.loads((ROOT / "simulation/configs/main.json").read_text())
        config.update(n=80, bootstrap_draws=19, scenarios=list(experiment.SCENARIOS),
                      learners=list(experiment.LEARNERS), deltas=[0.], replications=1,
                      methods=["orthogonal_nosplit"])
        config["learner_options"]["rf"]["n_estimators"] = 12
        config["learner_options"]["dnn"].update(device="cpu", max_epochs=3, min_epochs=1, patience=2)
        config["propensity_options"].update(cv_folds=2, Cs=[.1,1.])
        cal = core.calibrate_scenarios(p=20, rho=config["rho"], design="pooled_random_shift",
            scenarios=experiment.SCENARIOS, calibration_n=1000, seed=91)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output/"shards").mkdir()
            (output/"checkpoints").mkdir()
            experiment.STATE.update(config=config, output=output, fingerprint="test",
                calibrations={"20":{name:value.to_dict() for name,value in cal.items()}})
            experiment.run_job((20,1,"null"))
            actual = pd.read_csv(output/"shards/p20_r0001_null.csv.gz").set_index(["scenario","learner"])
            tasks = build_tasks(config, designs=None, scenarios=None, dimensions=[20],
                deltas=None, learners=None, rep_start=1, rep_stop=1)
            for task in tasks:
                rows,_ = core.run_replication(task, cal[task["scenario"]], config)
                expected = rows[0]
                self.assertEqual(expected["status"],"ok")
                row = actual.loc[(task["scenario"],task["learner"])]
                for field in ["score_mean","score_sd","mu_mse","propensity_mse"]:
                    self.assertAlmostEqual(row[field],expected[field],places=12)
                self.assertAlmostEqual(row.nord_p_value,expected["p_value"],places=14)
                self.assertEqual(row.nord_reject,expected["reject"])


if __name__ == "__main__":
    unittest.main()
