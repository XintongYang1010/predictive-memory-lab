# Additional exploratory temporal controls

This followup was designed after the core diagnostic improved concatenated-feature Dev motion R². It is not preregistered or independent confirmation. It reuses existing Train/Dev caches, changes no original gate, and opens no Test data.

Frame6 and frame7 controls use the same full-Train paired folds, alpha grid, centering, and motion-error objective as the core concat decoder. The choice between the two single-frame controls is made solely by held-out Train CV; both Dev results are shown.

| Features | Control | Alpha | Position R² | Displacement R² | Exploratory displacement 95% CI |
|---|---|---:|---:|---:|---|
| raw | frame6_train_cv | 1 | 0.88686 | 0.19946 | [0.12855, 0.26670] |
| raw | frame7_train_cv | 1 | 0.88707 | 0.23574 | [0.17010, 0.29718] |
| raw | concat_train_cv | 1 | 0.89260 | 0.23348 | [0.16608, 0.29647] |
| raw | concat_query_reversed | 1 | 0.90285 | -1.02239 | [-1.17047, -0.89655] |
| pooled | frame6_train_cv | 1 | 0.91413 | 0.20200 | [0.12932, 0.26800] |
| pooled | frame7_train_cv | 1 | 0.91499 | 0.25806 | [0.19140, 0.32242] |
| pooled | concat_train_cv | 1 | 0.92688 | 0.24390 | [0.17109, 0.30961] |
| pooled | concat_query_reversed | 1 | 0.93525 | -1.08123 | [-1.22257, -0.95423] |

## Matched temporal advantage

| Features | Train-selected single frame | Concat minus selected-frame displacement R² | Paired 95% CI |
|---|---|---:|---|
| raw | frame6 | 0.03402 | [0.01812, 0.05084] |
| pooled | frame7 | -0.01416 | [-0.03436, 0.00493] |

## Query-order intervention

The original fitted concat decoder receives [z7,z6] instead of [z6,z7]. Correct labels are [p6,−displacement]; reversed-query R² uses the correspondingly reversed Train target mean in its denominator. No reverse-order fitting or Dev tuning is performed. This intervention changes the generator's position/direction prior, so it is a distribution-shift diagnostic, not proof of a causal mechanism.

| Features | RMSE(predicted motion forward + reversed) | RMSE / true-motion RMS | Fraction of predicted signs reversing |
|---|---:|---:|---:|
| raw | 0.014385 | 1.1064 | 0.0742 |
| pooled | 0.014921 | 1.1477 | 0.0645 |

Oddness is a signed-displacement consistency check: a pure displacement mapping would sum to zero under reversal. Nonzero values can include the fitted intercept and the asymmetric generator prior; they do not alone isolate feature pooling, regularization, or a model mechanism.

Bootstrap resamples complete Dev pairs with both variants. Concat-versus-single-frame differences use identical resamples and the same original Train-mean denominator. Intervals describe this already inspected sample only. No automatic Dev winner or downstream training approval is emitted.

[Post-result followup plan](followup_plan.json) · [Train-only frame/alpha selection](train_selection.json) · [Predictions](dev_predictions.csv) · [Complete metrics and provenance](diagnostics.json)
