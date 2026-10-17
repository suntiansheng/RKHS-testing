# Conditional LATE simulation

Status: **complete**.

One hierarchical-signal experiment uses n = 400, p = 20, 300 planned replications per amplitude, nominal level 0.05, and 200 centered Gaussian multiplier draws. Yan uses its normal calibration. CF-NORD uses two balanced folds drawn independently of the data, fits nuisance functions on the other fold, and pools the held-out scores for the test.

The instrument is A | X ~ Bernoulli(pi(X)). Independently of (X,A), U is uniform on [0,1], D(0) = 1{U <= 0.2}, D(1) = 1{U <= 0.8}, and C = 1{0.2 < U <= 0.8}. Observed treatment is D = D(A), and Y(d) = b_N(X) + 2(U - 0.5) + d[C a f_N(X) + (1-C)] + (1 + 0.5d) epsilon_d. The potential-outcome errors are independent standard normals, independent of (X,U,A). The signal and its reference-sample normalization are unchanged from the hierarchical regression setting.

The conditional LATE is a f_N(X); the instrument first stage is 0.6, and the conditional reduced-form contrast is 0.6 a f_N(X). Its zero restriction is equivalent to the global zero conditional-LATE null in this design. The population LATE is zero at every amplitude. In contrast, the full-population CATE is 0.4 + 0.6 a f_N(X), and the ATE is 0.4. Treatment is endogenous through U. All tests compare instrument groups using (Y,X,A). NORD, CF-NORD, and single split use score (Y - mu_hat(X))(A - pi_hat(X)); Yan fits the two instrument-specific outcome regressions. No procedure groups on observed D or estimates a Wald ratio. The oracle uses mu(X) = b_N(X) + 0.2 + 0.6 pi(X) a f_N(X) and the true instrument propensity.

Completed cells: 204/204; valid test results: 61,200/61,200; failed results: 0; missing results: 0.

The 204 distinct amplitude–test–learner cells comprise 12 amplitudes times 17 combinations. The oracle is computed once per dataset and repeated in all four learner panels; these repetitions are not additional replications or additional size tests.

The 300 distinct observed replication draws have mean empirical instrument first stage 0.601 (range 0.474 to 0.740) and mean complier fraction 0.600. These diagnostics count each replication once across amplitudes and learners.

## Null rejection rates

Intervals are pointwise 95% Clopper–Pearson binomial intervals, conditional on successful runs. Failed and missing runs are reported separately and never counted as nonrejections.

| Learner | Test | Rejection rate [95% interval] | Valid | Failed | Missing |
|---|---|---:|---:|---:|---:|
| Spline | NORD | 0.047 [0.026, 0.077] | 300 | 0 | 0 |
| Spline | CF-NORD | 0.080 [0.052, 0.117] | 300 | 0 | 0 |
| Spline | Single split | 0.043 [0.023, 0.073] | 300 | 0 | 0 |
| Spline | Yan | 0.047 [0.026, 0.077] | 300 | 0 | 0 |
| Random forest | NORD | 0.077 [0.049, 0.113] | 300 | 0 | 0 |
| Random forest | CF-NORD | 0.080 [0.052, 0.117] | 300 | 0 | 0 |
| Random forest | Single split | 0.060 [0.036, 0.093] | 300 | 0 | 0 |
| Random forest | Yan | 0.107 [0.074, 0.147] | 300 | 0 | 0 |
| Lasso | NORD | 0.050 [0.028, 0.081] | 300 | 0 | 0 |
| Lasso | CF-NORD | 0.040 [0.021, 0.069] | 300 | 0 | 0 |
| Lasso | Single split | 0.047 [0.026, 0.077] | 300 | 0 | 0 |
| Lasso | Yan | 0.047 [0.026, 0.077] | 300 | 0 | 0 |
| DNN | NORD | 0.053 [0.031, 0.085] | 300 | 0 | 0 |
| DNN | CF-NORD | 0.047 [0.026, 0.077] | 300 | 0 | 0 |
| DNN | Single split | 0.033 [0.016, 0.060] | 300 | 0 | 0 |
| DNN | Yan | 0.060 [0.036, 0.093] | 300 | 0 | 0 |
| Shared oracle | Oracle | 0.043 [0.023, 0.073] | 300 | 0 | 0 |

## Power and paired comparisons

The numeric amplitude grid is 0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.5, 4, 6, 20. Pointwise 95% exact binomial intervals are available in `rejection_summary.csv` and the null table above.

`rejection_summary.csv` contains every planned cell, including missing cells. `paired_differences.csv` reports NORD minus single-split rejection rates, and `paired_crossfit_differences.csv` reports NORD minus CF-NORD rejection rates. Both use only matched successful replications. Their pointwise normal Monte Carlo intervals use the sample variance of paired rejection differences and are not adjusted over amplitudes or learners. `combined_raw_results.csv.gz` and `combined_dgp_diagnostics.csv.gz` retain test-level results and principal-stratum, treatment, and first-stage diagnostics; their checksums are in `report_manifest.json`.

![Conditional LATE rejection curves](simulation_late_hierarchical_crossfit.png)

This fixed-first-stage experiment evaluates empirical calibration and power. It does not address weak instruments, verify sufficient nuisance assumptions for every learner, or establish minimax optimality.
