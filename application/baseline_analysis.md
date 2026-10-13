# Baseline CaSiNo companion

This analysis compares mean satisfaction between maximal-integrative-potential
and aligned-ranking negotiations using pretreatment survey characteristics.
It retains the original participant-disjoint sample (232 dialogues, 73 aligned
and 159 maximal), verified by `analyze.MAIN_SAMPLE_SHA256`.

## Run

```sh
python application/download_data.py
python application/baseline_analysis.py --results-dir runs/casino-baseline
python application/plot_baseline.py --results-dir runs/casino-baseline --output runs/casino-baseline/figures/standardized_contrasts.pdf
python -m unittest discover -s application/tests
```

The recorded run used Python 3.12.2, NumPy 1.26.4, pandas 2.3.1, SciPy 1.13.0,
scikit-learn 1.6.1, and Matplotlib 3.9.2. It took 38.8 seconds with BLAS and
OpenMP limited to one thread. Runtime depends on the environment. The existing
text application was replayed separately; its ridge and Lasso results reproduce,
while forest statistics can vary slightly across numerical environments.

## Design fixed before the companion fits

- Covariates: 36 symmetric pair summaries, using only the pre-survey fields.
  Pair means and absolute differences of age and the five personality traits
  contribute 12 features. Category fractions for gender, ethnicity, education,
  and social value orientation contribute 23. Missing-age fraction contributes
  one. Categories are declared explicitly, without learning them from outcomes.
- Personality scores map from [1, 7] to [0, 1]. Ages map from [18, 100] to [0, 1].
  Ages outside this range use zero plus the missing indicator; one selected
  participant reports age 3. The age-omission sensitivity removes all three
  age-related features. No participants are removed for that response.
- Primary outcome: mean dyad satisfaction (1 to 5). Binary sensitivity: both
  participants report the highest satisfaction category.
- Primary test: full-sample ridge, Gaussian bandwidth 1, 5,000 multipliers.
  Additional learners: Lasso and random forest, with the existing application
  tuning rules. Sensitivity: five outer folds independently shuffled without
  treatment-label stratification; all learner tuning is confined to training
  folds. There is no outer-fold preprocessing fitted on held-out outcomes.
- Report all ridge bandwidths 0.5, 1, and 2. The narrow bandwidth does not reject;
  this result must remain visible when describing the findings.
- Selection sensitivity: apply the original identifier ordering across all
  1,030 dialogues, select 287 participant-disjoint dialogues, then restrict to
  180 extreme-regime dialogues (56 aligned, 124 maximal). This does not repair
  corpus filtering or establish independent sampling.

## Estimates and uncertainty

`baseline_tests.csv` contains every kernel/learner/sample/outcome comparison.
`standardized_contrasts.csv` reports unadjusted differences and cross-fitted
augmented inverse-propensity contrasts. The latter fit a separate outcome
regression in each arm, in each outer training fold, and use the same held-out
propensity predictions as the baseline cross-fitted test. Outcome predictions
are bounded by the response range. Propensities are not clipped.

Unadjusted intervals use 10,000 within-regime dialogue bootstrap resamples.
Adjusted intervals use the empirical standard deviation of the cross-fitted
augmented scores divided by sqrt(n), with a normal 1.96 multiplier. These are
pointwise working intervals; they do not account for corpus filtering,
dependence induced by participant-disjoint selection, or earlier exploration
of outcomes and regime definitions. They are not confirmatory causal intervals.

The standardized target is E[mu_1(X_pre) - mu_0(X_pre)] in the analysed
population. The regression difference is a CATE only under consistency,
ignorability, and overlap for well-defined regime interventions in that
population. Define each regime by the distribution of ranking configurations
within it. Random assignment in the original experiment does not automatically
preserve ignorability after filtering. Rejection of zero effects does not
establish heterogeneity; a constant nonzero difference also violates the null.

The original opening text is postassignment. Its test remains a regression
comparison and cannot be used to identify a total treatment effect. Selected
word-frequency contrasts do not identify RKHS feature contributions or causal
effects of words.

## Outputs and provenance

The result directory saves survey designs, dialogue identifiers, all held-out
and full-sample nuisance predictions, augmented scores, propensity diagnostics,
baseline balance summaries, contrast tables, and JSON metadata with versions,
corpus checksums, design columns, and elapsed time. Seeds are deterministic
functions of the version string and specification names. Raw data and run
artifacts remain outside version control, following the existing repository
policy; the public input archive is checksum-verified on every run.

The final manuscript uses the matched comparisons from `joint_analysis.py`;
see [application/README.md](README.md).
