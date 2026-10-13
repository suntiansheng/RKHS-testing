# CATE simulation

Status: **complete**.

One hierarchical-signal experiment uses n = 400, p = 20, 300 planned replications per amplitude, nominal level 0.05, and 200 centered Gaussian multiplier draws. Yan uses its normal calibration.

The observed response is Y = b_N(X) + D a f_N(X) + (1 + 0.5 D) ε. The signal and its reference-sample normalization are unchanged from the existing hierarchical setting. The CATE is a f_N(X), and its population average is zero at every amplitude. At the null, conditional outcome variances differ between the two treatment arms.

Completed cells: 156/156; valid test results: 46,800/46,800; failed results: 0; missing results: 0.

The 156 distinct amplitude–test–learner cells comprise 12 amplitudes times 13 combinations. The oracle is computed once per dataset and repeated in the four panels; these repetitions are not additional replications or additional size tests.

## Null rejection rates

Intervals are pointwise 95% Clopper–Pearson binomial intervals, conditional on successful runs. Failed and missing runs are reported separately and never counted as nonrejections.

| Learner | Test | Rejection rate [95% interval] | Valid | Failed | Missing |
|---|---|---:|---:|---:|---:|
| Spline | NORD | 0.047 [0.026, 0.077] | 300 | 0 | 0 |
| Spline | Single split | 0.050 [0.028, 0.081] | 300 | 0 | 0 |
| Spline | Yan | 0.080 [0.052, 0.117] | 300 | 0 | 0 |
| Random forest | NORD | 0.053 [0.031, 0.085] | 300 | 0 | 0 |
| Random forest | Single split | 0.060 [0.036, 0.093] | 300 | 0 | 0 |
| Random forest | Yan | 0.107 [0.074, 0.147] | 300 | 0 | 0 |
| Lasso | NORD | 0.043 [0.023, 0.073] | 300 | 0 | 0 |
| Lasso | Single split | 0.037 [0.018, 0.065] | 300 | 0 | 0 |
| Lasso | Yan | 0.073 [0.047, 0.109] | 300 | 0 | 0 |
| DNN | NORD | 0.050 [0.028, 0.081] | 300 | 0 | 0 |
| DNN | Single split | 0.043 [0.023, 0.073] | 300 | 0 | 0 |
| DNN | Yan | 0.087 [0.057, 0.124] | 300 | 0 | 0 |
| Shared oracle | Oracle | 0.050 [0.028, 0.081] | 300 | 0 | 0 |

## Power and paired comparisons

The figure retains numeric amplitude positions, matching the existing manuscript figures; the full grid is 0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.5, 4, 6, 20. The four learner panels show rejection-rate curves. Pointwise 95% exact binomial intervals are available in `rejection_summary.csv` and the null table above.

`rejection_summary.csv` contains every planned cell, including missing cells. `paired_differences.csv` reports NORD minus single-split rejection rates using only matched successful replications, with pointwise normal Monte Carlo intervals based on the sample variance of paired rejection differences. These intervals describe Monte Carlo uncertainty and are not adjusted for comparisons over amplitudes or learners.

Figures were not generated (`--skip-plots`).

The experiment evaluates empirical calibration and power for this design. It does not verify the sufficient assumptions for every fitted learner or establish minimax optimality. Zero average treatment effect alone does not imply a zero propensity-weighted scalar moment.
