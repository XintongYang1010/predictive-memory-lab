# V5.2 exploratory bad-case diagnosis

Posthoc analysis of already-open V5.1 FIT/confirmation/bridge caches. No refit, winner selection, new gate, GPU, or Test access. FIT predictions are in-sample and are not validation evidence.

## Horizontal displacement failure

| Axis | Confirm direction error | Bridge direction error | Bridge amplitude ratio (mean absolute) | Bridge prediction/target SD |
|---|---:|---:|---:|---:|
| A_dx | 10.94% | 44.14% | 1.646 | 1.878 |
| A_dy | 3.12% | 32.62% | 0.758 | 0.898 |
| B_dx | 16.80% | 42.38% | 1.849 | 2.089 |
| B_dy | 5.27% | 25.20% | 0.822 | 1.055 |

Bridge dx errors combine weak direction agreement with excessive prediction spread; the global signed bias alone is small. `axis_metrics.csv` verifies the exact bias/variance/covariance MSE decomposition.

## Shared nuisance association

- A_dx: correlation of pair-common residual with distractor dx = 0.8961 (exploratory95% CI 0.8735 to 0.9156); pair-common error contributes 81.36% of MSE.
- A_dy: correlation of pair-common residual with distractor dx = 0.1547 (exploratory95% CI 0.0367 to 0.2648); pair-common error contributes 49.83% of MSE.
- B_dx: correlation of pair-common residual with distractor dx = 0.9025 (exploratory95% CI 0.8822 to 0.9203); pair-common error contributes 84.40% of MSE.
- B_dy: correlation of pair-common residual with distractor dx = 0.2427 (exploratory95% CI 0.1271 to 0.3527); pair-common error contributes 62.52% of MSE.

The bridge's yellow marker moves horizontally and identically in both variants; quartet FIT/confirmation hold it fixed. This association is consistent with a nuisance-motion shortcut. It is not a causal test: geometry, support, reflections, and nuisance motion also change across these datasets. No subtraction or correction has been applied to predictions.

## FIT coverage and distances

- fit: 0/256 whole groups have at least one endpoint coordinate outside the observed FIT min/max box.
- confirm: 4/256 whole groups have at least one endpoint coordinate outside the observed FIT min/max box.
- bridge: 248/256 whole groups have at least one endpoint coordinate outside the observed FIT min/max box.

`primary_group_strata.csv` gives per-axis skill, direction errors and magnitude statistics by complete-group position coverage and nearest-FIT feature/position distances. Feature distances use the frozen centered normalized linear space and the selected kernel's RKHS. FIT neighbors exclude the entire same quartet. Thresholds are FIT group-max median/p90/max; no bridge-selected cutpoint or model is used. High-dimensional nearest-neighbor distance is a descriptive support measure, not a calibrated OOD probability.

All association intervals are unadjusted exploratory pair bootstraps; they condition on this frozen model and dataset. Positions and nuisance values are analysis-only labels, never actor inputs. Sealed original results remain unchanged.
