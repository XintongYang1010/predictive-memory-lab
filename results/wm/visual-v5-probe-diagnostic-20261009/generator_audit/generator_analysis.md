# Existing-Dev generator and motion-direction audit

This is exploratory analysis of the original 64 Dev pairs (128 records), using archived predictions and the frozen archived generator. No Train or Test episode is generated, no model is trained, and no original result or threshold is changed.

All 2,048 CSV targets exactly match regenerated `probe_state` (maximum difference 0). Both objects are visible at t=6 and t=7.

## Direction errors and absolute position

| Dimension | Raw wrong sign | Pooled wrong sign | corr(true delta, position7) | corr(pooled prediction, position7) |
|---|---:|---:|---:|---:|
| A_dx_t6_to_t7 | 44/128 | 44/128 | 0.444 | 0.569 |
| A_dy_t6_to_t7 | 20/128 | 26/128 | 0.794 | 0.725 |
| B_dx_t6_to_t7 | 50/128 | 51/128 | 0.438 | 0.677 |
| B_dy_t6_to_t7 | 26/128 | 34/128 | 0.597 | 0.685 |

Across pooled predictions, x has 95/256 wrong signs, y has 60/256. These are descriptive paired observations, not independent significance tests.

The generator samples each initial y coordinate within width 0.09, versus x within width 0.60; the reflecting lanes have width 0.25 versus x width 0.80. Absolute terminal position can therefore carry a motion prior, especially on y. Higher y predictability alone does not prove better temporal-motion extraction.

Within an unreflected pair, p7 = p0 ± 7v: the variants share p0 but have opposite initial velocity. Thus paired generation itself creates a correlation between terminal position and direction; it does not balance terminal position across directions. The paired mean pooled x prediction correlates with initial x by 0.583 for A and 0.752 for B in these 64 pairs. This is compatible with a position prior and requires an input/order ablation before attributing the error to that prior.

## Finite rendering and boundaries

| Archived chosen case | True displacement (pixels) | Rendered center displacement (pixels) | Reflection during cue |
|---|---:|---:|---|
| dev-000001-0 B_dx_t6_to_t7 | +3.9503 | +4.0 | False |
| dev-000058-0 A_dx_t6_to_t7 | +3.5224 | +4.0 | False |
| dev-000033-1 A_dx_t6_to_t7 | +2.2556 | +2.5 | False |

A diagnostic color-mask bounding-box center yields 0/512 directions opposite to true displacement, and 4/512 quantized zero displacements. Maximum center error is 0.9395 pixels; displacement RMSE is 0.3406 pixels.
Of 155 pooled sign errors, 154 occur when pixel motion is nonzero. Privileged color segmentation is only a label/rendering diagnostic, not a proposed scientific model input.

| Group | Pooled wrong sign | Rate |
|---|---:|---:|
| Reflection during t6–t7: False | 148/490 | 30.2% |
| Reflection during t6–t7: True | 7/22 | 31.8% |
| Within 0.02 of reflecting boundary: False | 130/423 | 30.7% |
| Within 0.02 of reflecting boundary: True | 25/89 | 28.1% |
| 4×4 cell-center crossing: False | 151/493 | 30.6% |
| 4×4 cell-center crossing: True | 4/19 | 21.1% |

These subgroup associations are observational and confounded by axis and position. Boundary proximity is not a diagnosis of cause. In the frozen generator, both cue frames precede all update and extra-observation events, so differences grouped by family/gap/write count cannot by themselves show interference or update failure.

Pooled variant-0 and variant-1 sign errors are 79/256 and 76/256. Absolute-position quartiles give:

| Dimension | Lowest position quartile | Second | Third | Highest |
|---|---:|---:|---:|---:|
| A_dx_t6_to_t7 | 5/32 | 13/32 | 16/32 | 10/32 |
| A_dy_t6_to_t7 | 6/32 | 9/32 | 5/32 | 6/32 |
| B_dx_t6_to_t7 | 7/32 | 18/32 | 16/32 | 10/32 |
| B_dy_t6_to_t7 | 14/32 | 17/32 | 3/32 | 0/32 |

Quartile boundaries and full family/gap/write-count summaries are in JSON. These were inspected after the failed gate and are exploratory.

## Paired variants

| Dimension | Opposite true signs | Same predicted sign within those pairs | Pair-common fraction of squared error |
|---|---:|---:|---:|
| A_dx_t6_to_t7 | 64/64 | 34 | 42.1% |
| A_dy_t6_to_t7 | 56/64 | 18 | 62.3% |
| B_dx_t6_to_t7 | 63/64 | 38 | 43.8% |
| B_dy_t6_to_t7 | 36/64 | 21 | 73.6% |

For paired errors e0/e1, the common part is (e0+e1)/2 and the antisymmetric part is (e0−e1)/2; their energies sum exactly to mean squared error. A large common component is compatible with a pair-specific offset or position/appearance shortcut, but does not identify its source. Reflection means velocity reversal need not reverse the observed displacement sign.

## Strong falsifiers before claiming a fix

1. **Rendering/target mismatch hypothesis:** already challenged by exact target regeneration and finite-pixel directions. The worst sign errors remain directly visible; a global direction-label inversion or absent subpixel motion cannot explain all failures.
2. **The probe reads temporal direction:** with existing cached Train/Dev features, fit matched-budget probes using frame difference and pairwise frame order; evaluate a Dev cue-order swap and its negated displacement target. A temporal-direction readout should reverse displacement predictions under the swap. Keep the archived gate unchanged and report these as exploratory ablations.
3. **Absolute-position shortcut hypothesis:** compare a current-frame-only control against a temporal-difference readout, and test position-balanced positive/negative directions in a future freshly generated, separate confirmation split. If temporal-order-sensitive advantage persists after balancing positions and excluding reflections, that weakens the shortcut explanation. Do not use reserved Test for iterative debugging.
4. **Pooling specifically causes the failure:** apply identical centering/scaling, ridge budget and order-sensitive features to raw versus pooled caches. Recovery by both after a probe change would weaken a pooling-specific explanation; a stable raw–pooled gap on fresh confirmation would support further pooling investigation.

Archived generator SHA-256: `ce5082f3be9006dcceb46fbaffadc27349f977a77abcf8992eda3e80f8775368`. Prediction CSV SHA-256: `b114eb1770e3625863a651b2b7b678e22924feb894717143a33bb96b5e21ab8d`.

From the repository root, reproduce with `python3 results/wm/visual-v5-probe-diagnostic-20261009/generator_audit/generator_analysis.py`. The script locates the original frozen archive automatically; `--archive PATH` and `--out PATH` override locations. Full counts and case evidence: `generator_analysis.json`.
