"""Checks for pretreatment isolation, pair symmetry, and held-out estimation."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline_analysis as baseline


class BaselineDesignTests(unittest.TestCase):
    def setUp(self):
        self.sample = pd.DataFrame([dict(dialogue_id=1, speaker_a="a", speaker_b="b",
            D=0, early_text="initial words", mean_satisfaction=3.0)])
        def person(age, gender, trait, svo):
            return {"meta": {"demographics": {"age":age, "gender":gender,
                "ethnicity":"white american", "education":"high school graduate / ged"},
                "personality":{"svo":svo,"big-five":dict.fromkeys(baseline.TRAITS,trait)}}}
        self.speakers={"a":person(30,"female",2,"prosocial"),"b":person(50,"male",6,"proself")}

    def test_covariates_do_not_use_treatment_outcomes_or_dialogue(self):
        X, names, _=baseline.baseline_design(self.sample,self.speakers)
        altered=self.sample.copy()
        altered["D"]=1; altered["early_text"]="changed completely"; altered["mean_satisfaction"]=5
        altered["maximum_joint_points"]=42
        np.testing.assert_array_equal(X,baseline.baseline_design(altered,self.speakers)[0])
        self.assertEqual(len(names),36)

    def test_pair_order_is_irrelevant_but_personality_is_retained(self):
        X,names,_=baseline.baseline_design(self.sample,self.speakers)
        swapped=self.sample.assign(speaker_a="b",speaker_b="a")
        np.testing.assert_array_equal(X,baseline.baseline_design(swapped,self.speakers)[0])
        self.assertAlmostEqual(X[0,names.index("agreeableness_mean")],0.5)
        self.assertAlmostEqual(X[0,names.index("agreeableness_gap")],2/3)
        self.assertEqual(X[0,names.index("svo:prosocial")],0.5)

    def test_invalid_age_is_explicit_and_omit_age_uses_no_age_information(self):
        people=copy.deepcopy(self.speakers)
        people["a"]["meta"]["demographics"]["age"]=3
        X,names,metadata=baseline.baseline_design(self.sample,people)
        self.assertEqual(metadata["invalid_age_participants"],1)
        self.assertEqual(X[0,names.index("age_missing_fraction")],0.5)
        self.assertTrue(np.isfinite(X).all())
        a=baseline.baseline_design(self.sample,self.speakers,include_age=False)[0]
        b=baseline.baseline_design(self.sample,people,include_age=False)[0]
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape[1],33)


class StandardizationTests(unittest.TestCase):
    def test_oracle_deterministic_outcomes_recover_the_contrast(self):
        d=np.array([0,1,0,1]); m0=np.array([2.,3.,3.,2.]); m1=m0+0.5
        y=np.where(d==1,m1,m0); e=np.array([0.2,0.7,0.3,0.8])
        np.testing.assert_allclose(baseline.aipw_scores(y,d,e,m0,m1),0.5)
        with self.assertRaises(ValueError):
            baseline.aipw_scores(y,d,np.array([0.,0.7,0.3,0.8]),m0,m1)

    def test_arm_outcomes_are_fitted_without_held_out_rows(self):
        X=np.arange(40.)[:,None]; y=np.repeat(3.,40); d=np.arange(40)%2
        folds=baseline.independent_folds(len(y)); calls=[]
        class RecordingModel:
            def fit(self,x,y):
                self.training=set(x[:,0].astype(int)); return self
            def predict(self,x):
                test=set(x[:,0].astype(int))
                calls.append((self.training,test))
                return np.repeat(3.,len(x))
        with patch.object(baseline.core,"outcome_model",side_effect=lambda *args: RecordingModel()):
            result=baseline.arm_predictions(X,y,d,folds,"ridge","mean_satisfaction")
        self.assertEqual(len(calls),10)
        self.assertTrue(all(train.isdisjoint(test) for train,test in calls))
        np.testing.assert_array_equal(result,np.full((40,2),3.))
        for fold,(train,test) in enumerate(folds):
            for arm in (0,1):
                self.assertEqual(calls[2*fold+arm][0],set(train[d[train]==arm]))


if __name__ == "__main__":
    unittest.main()
