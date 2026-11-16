# Baseline CaSiNo companion

This companion uses the reference sample of 232 participant-disjoint dialogues
(73 aligned and 159 maximal potential), checked against
`analyze.MAIN_SAMPLE_SHA256`. The final manuscript uses the matched baseline
and baseline-plus-text comparisons from `joint_analysis.py`; see the
[application instructions](README.md).

Run from the repository root:

```sh
python application/download_data.py
python application/baseline_analysis.py --results-dir runs/casino-baseline
python application/plot_baseline.py --results-dir runs/casino-baseline --output runs/casino-baseline/figures/standardized_contrasts.pdf
```

The 36 covariates use only pre-survey fields. Age and each Big Five trait
contribute a pair mean and absolute difference. Gender, ethnicity, education
and social value orientation contribute category proportions, with an additional
missing-age fraction. Personality scores are rescaled from `[1,7]` and age from
`[18,100]` to `[0,1]`. An out-of-range age is coded as zero and flagged missing;
one participant reports age 3. The representation is invariant to participant
order and uses no fitted feature preprocessing.

Both satisfaction outcomes are evaluated with ridge, Lasso and random-forest
nuisances, NORD and five-fold CF-NORD, and 5,000 centered Gaussian multipliers.
Ridge NORD also uses bandwidths 0.5 and 2 alongside the reference bandwidth 1.
CF-NORD uses independent shuffled folds; tuning and standardization use only
the training observations. Additional checks omit the three age features and
select participant-disjoint dialogues before restricting the regime groups.
The latter gives 180 eligible dialogues from 287 selected across all regimes.

`baseline_tests.csv` saves the regression tests. `standardized_contrasts.csv`
contains unadjusted differences and cross-fitted augmented inverse-propensity
contrasts. The adjusted contrasts use arm-specific outcome fits and held-out
propensities without clipping. Outcome predictions are bounded by the response
range. Adjusted intervals use the empirical standard deviation of augmented
scores divided by `sqrt(n)` and a normal multiplier of 1.96. Unadjusted intervals
use 10,000 within-group dialogue bootstrap resamples.

These are pointwise working intervals. Their causal interpretation requires
consistency, ignorability and overlap in the selected population; corpus
filtering and participant removal are not accounted for in the intervals.
A nonzero contrast may be constant across covariates. Opening-language
comparisons condition on information recorded after ranking assignment.

Outputs include designs, nuisance predictions, augmented scores, propensity
diagnostics, balance summaries and metadata with versions, checksums and
feature names. Seeds are determined by the version string and specification.
These run files are excluded from version control.
