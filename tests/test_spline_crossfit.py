import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import spline_crossfit as study


class SplineCrossfitTests(unittest.TestCase):
    def test_response_contrast_and_fixed_mean(self):
        rng = np.random.default_rng(51)
        x = study.core.draw_gaussian_covariates(60,20,.5,rng)
        b,f = study.core.raw_regression_components(x)["additive"]
        e = study.core.pooled_random_propensity(x)
        for variant in ("original","fixed_pooled_mean"):
            y0,mu0=study.response(variant,b,f,e,np.zeros(60),np.zeros(60),1.2)
            y1,mu1=study.response(variant,b,f,e,np.ones(60),np.zeros(60),1.2)
            np.testing.assert_allclose(y1-y0,1.2*f,atol=1e-14)
            np.testing.assert_allclose((1-e)*y0+e*y1,mu0,atol=1e-14)
            np.testing.assert_array_equal(mu0,mu1)
            if variant == "fixed_pooled_mean":
                np.testing.assert_array_equal(mu0, b)

    def test_multiplier_calibration_matches_original(self):
        rng=np.random.default_rng(93)
        x = rng.normal(size=(120, 20))
        score = rng.normal(size=120)
        kernel=study.core.rbf_kernel_matrix(x,np.sqrt(40.))
        multipliers=np.random.default_rng(204).standard_normal((120,999))
        multipliers-=multipliers.mean(axis=0,keepdims=True)
        with threadpool_limits(limits=1):
            result=study.score_test(score,kernel,multipliers,.05)
            reference=study.core.multiplier_test(score,kernel,bootstrap_draws=999,alpha=.05,seed=204)
        for key in ("statistic","p_value","reject","critical_value"):
            self.assertAlmostEqual(result[key],getattr(reference,key),places=13)

    def test_outer_folds_and_seed_separation(self):
        for n in (120,160):
            for k in (2,5):
                split=study.folds(n,k,31)
                np.testing.assert_array_equal(np.sort(np.concatenate([t for _,t in split])),np.arange(n))
                for train,test in split:
                    self.assertEqual(len(np.intersect1d(train,test)),0)
                    self.assertEqual(len(train)+len(test),n)
                    self.assertEqual(len(test),n//k)
        pilot={study.core.stable_seed(2026110601,n,r,"data") for n in (120,160) for r in range(150)}
        confirm={study.core.stable_seed(2026110602,n,r,"data") for n in (120,160) for r in range(5000)}
        self.assertFalse(pilot & confirm)
        self.assertEqual(len(pilot),300)
        self.assertEqual(len(confirm),10000)

    def test_one_dataset_against_independent_manual_scores(self):
        config=json.loads((ROOT/"simulation/configs/spline_crossfit_pilot.json").read_text())
        config.update(amplitudes=[0.,.5],primary_amplitudes=[.5],bootstrap_draws=99)
        with tempfile.TemporaryDirectory(prefix="spline-crossfit-test-") as tmp:
            study.initialize(config,tmp,"test")
            result=study.run_task((120,0))
            self.assertEqual(result["status"],"ok")
            payload=json.loads((Path(tmp)/"checkpoints/n120_r00000.json").read_text())
            self.assertEqual(len(payload["rows"]),15)
            self.assertTrue(all(r["score_n"]==120 for r in payload["rows"]))
            seed=study.core.stable_seed(config["seed_base"],120,0,"data")
            rng=np.random.default_rng(seed)
            x=study.core.draw_gaussian_covariates(120,20,.5,rng)
            e = study.core.pooled_random_propensity(x)
            d = rng.binomial(1, e)
            noise = rng.normal(size=120)
            b, f = study.core.raw_regression_components(x)["additive"]
            kernel=study.core.rbf_kernel_matrix(x,np.sqrt(40.))
            bseed=study.core.stable_seed(config["seed_base"],120,0,"multipliers")
            y,_=study.response("fixed_pooled_mean",b,f,e,d,noise,.5)
            assignment=study.folds(120,2,study.core.stable_seed(config["seed_base"],120,0,"folds",2))
            prediction=np.empty(120)
            for j,(train,test) in enumerate(assignment):
                fitted=study.core.fit_regression("gam",x[train],y[train],x[test],seed=0,
                                                  learner_options=config["learner_options"])
                prediction[test]=fitted.prediction
            reference=study.core.multiplier_test((y-prediction)*(d-e),kernel,
                           bootstrap_draws=99,alpha=.05,seed=bseed)
            actual=next(r for r in payload["rows"] if r["dgp"]=="fixed_pooled_mean" and
                        r["amplitude"]==.5 and r["method"]=="cf2" and r["propensity_mode"]=="known")
            self.assertAlmostEqual(actual["p_value"],reference.p_value,places=14)
            self.assertAlmostEqual(actual["statistic"],reference.statistic,places=12)

    def test_report_matches_the_independent_pilot_and_manuscript(self):
        import spline_crossfit_report as report
        pilot = ROOT / "results/spline_crossfit/pilot"
        self.assertEqual(report.select_amplitudes(pilot), [2., 2.5, 3.2])
        summary, config = report.load_results(ROOT / "results/spline_crossfit/confirmation")
        errors = report.nuisance_comparisons(summary, config["primary_amplitudes"])
        self.assertEqual(len(errors), 12)
        self.assertAlmostEqual(errors.NORD_to_CF2_MSE_ratio.min(), .6947527270738791)
        self.assertAlmostEqual(errors.NORD_to_CF2_MSE_ratio.max(), .7727769495442063)
        expected = {(120, "full", "estimated"): .0556,
                    (160, "full", "estimated"): .0590,
                    (120, "cf2", "estimated"): .0688,
                    (160, "cf2", "estimated"): .0616}
        for (n, method, mode), rate in expected.items():
            selected = summary[summary.n.eq(n) & summary.method.eq(method)
                               & summary.propensity_mode.eq(mode) & summary.amplitude.eq(0)]
            self.assertEqual(len(selected), 1)
            self.assertAlmostEqual(float(selected.iloc[0].rejection_rate), rate)


if __name__ == "__main__":
    unittest.main()
