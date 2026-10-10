# Exploratory CPU probe diagnostics

This reuses the already inspected Dev set. Alpha selection and sample subsets use Train only. No original gate, prior result, or Test data is changed. There is no automatic Dev winner or permission to resume scientific training.

| Features | Method | Train pairs | Selection | Alpha | Position R² | Displacement R² | Exploratory displacement 95% CI |
|---|---|---:|---|---:|---:|---:|---|
| raw | concat | 16 | train_cv | 1 | 0.61651 | 0.16843 | not computed |
| raw | difference_hybrid | 16 | train_cv | 10 | 0.79306 | 0.00533 | not computed |
| raw | shared_position | 16 | train_cv | 10 | 0.33490 | -0.69028 | not computed |
| raw | concat | 32 | train_cv | 1 | 0.79861 | 0.22255 | not computed |
| raw | difference_hybrid | 32 | train_cv | 10 | 0.91407 | 0.00342 | not computed |
| raw | shared_position | 32 | train_cv | 10 | 0.51214 | -1.26263 | not computed |
| raw | concat | 64 | train_cv | 1 | 0.89260 | 0.23348 | [0.1661, 0.2965] |
| raw | difference_hybrid | 64 | train_cv | 1 | 0.95713 | 0.02034 | [-0.0124, 0.0504] |
| raw | shared_position | 64 | train_cv | 10 | 0.68092 | -1.75202 | [-2.2526, -1.3261] |
| raw | concat | 64 | fixed | 0.01 | 0.95713 | 0.08055 | [-0.0392, 0.1887] |
| raw | difference_hybrid | 64 | fixed | 0.01 | 0.95713 | -0.06023 | [-0.1403, 0.0147] |
| pooled | concat | 16 | train_cv | 1 | 0.70598 | 0.14662 | not computed |
| pooled | difference_hybrid | 16 | train_cv | 10 | 0.87374 | 0.00320 | not computed |
| pooled | shared_position | 16 | train_cv | 10 | 0.43421 | -2.11237 | not computed |
| pooled | concat | 32 | train_cv | 1 | 0.86273 | 0.23536 | not computed |
| pooled | difference_hybrid | 32 | train_cv | 10 | 0.95223 | 0.00092 | not computed |
| pooled | shared_position | 32 | train_cv | 10 | 0.61179 | -3.86105 | not computed |
| pooled | concat | 64 | train_cv | 1 | 0.92688 | 0.24390 | [0.1711, 0.3096] |
| pooled | difference_hybrid | 64 | train_cv | 10 | 0.97469 | 0.01047 | [-0.0017, 0.0216] |
| pooled | shared_position | 64 | train_cv | 0.01 | 0.98091 | -1.86199 | [-2.4937, -1.3203] |
| pooled | concat | 64 | fixed | 0.01 | 0.97469 | -0.02093 | [-0.1561, 0.1046] |
| pooled | difference_hybrid | 64 | fixed | 0.01 | 0.97469 | -0.28275 | [-0.4364, -0.1447] |

`difference_hybrid` retains fixed-alpha 0.01 concat position predictions and fits motion to frame differences. `shared_position` uses a single position decoder with diagnostic labels p6=p7−displacement; it derives motion by subtracting its two decoded positions. Neither structured fit supplies labels to the actor.

R² uses the Dev error of the corresponding Train target mean as its denominator. CV minimizes the mean four-displacement-dimension MSE ratio across held-out Train folds. Both variants and both frames remain in the same fold. Bootstrap resamples Dev pairs and retains both variants; intervals describe this exploratory sample, not independent confirmation or repeated seeds.

[Frozen plan](frozen_plan.json) · [Train-only selection and folds](train_selection.json) · [Gram energy/scaling](gram_energy.json) · [Per-dimension metrics](per_dimension.csv) · [Dev predictions](dev_predictions.csv) · [Timing, memory, and complete results](diagnostics.json)
