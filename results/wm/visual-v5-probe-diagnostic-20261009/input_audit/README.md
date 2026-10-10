# V5 probe input audit

The original frozen gate is reproduced from the existing Train/Dev cache. No Test data or new GPU inference is used; all controls keep alpha=0.01.

| Representation | Input | Position R2 | Displacement R2 |
|---|---|---:|---:|
| raw | concat_t6_t7 | 0.957126 | 0.080553 |
| raw | t6_only | 0.950074 | -0.067145 |
| raw | t7_only | 0.953947 | -0.050833 |
| raw | repeat_t6 | 0.950074 | -0.067145 |
| raw | repeat_t7 | 0.953947 | -0.050833 |
| original_pooled | concat_t6_t7 | 0.974691 | -0.020927 |
| original_pooled | t6_only | 0.965890 | -0.166888 |
| original_pooled | t7_only | 0.969549 | -0.222929 |
| original_pooled | repeat_t6 | 0.965890 | -0.166888 |
| original_pooled | repeat_t7 | 0.969549 | -0.222929 |
| raw_repooled_bf16 | concat_t6_t7 | 0.974654 | -0.020736 |
| raw_repooled_bf16 | t6_only | 0.965813 | -0.166573 |
| raw_repooled_bf16 | t7_only | 0.969501 | -0.223073 |
| raw_repooled_bf16 | repeat_t6 | 0.965813 | -0.166573 |
| raw_repooled_bf16 | repeat_t7 | 0.969501 | -0.223073 |
| raw_repooled_fp32 | concat_t6_t7 | 0.974677 | -0.020055 |
| raw_repooled_fp32 | t6_only | 0.965852 | -0.166051 |
| raw_repooled_fp32 | t7_only | 0.969514 | -0.223614 |
| raw_repooled_fp32 | repeat_t6 | 0.965852 | -0.166051 |
| raw_repooled_fp32 | repeat_t7 | 0.969514 | -0.223614 |

`raw_repooled_bf16` spatially pools the cached BF16 raw tokens then rounds to BF16. `raw_repooled_fp32` omits that last rounding. The original pooled tokens were pooled before their storage cast and came from a separate encoder batch; these diagnostics cannot uniquely attribute differences to batch size.

Repeating a single frame and using it once give the same trace-normalized ridge kernel; the numerical equivalence is recorded. The two-frame input contains both frames, but its score must be compared with single-frame controls before claiming temporal dependence.

## Renderer and non-temporal controls

| Control | Position R2 | Displacement R2 |
|---|---:|---:|
| rgb_two_frame_centroid_oracle | 0.999443 | 0.977881 |
| privileged_true_t7_position_degree_1 | 1.000000 | 0.314991 |
| privileged_true_t7_position_degree_3 | 0.999804 | 0.314277 |
| rgb_t7_centroid_degree_1 | 0.999840 | 0.314306 |

Non-temporal controls measure generator priors. True positions are privileged diagnostic inputs. RGB centroids use the known renderer color rule. None enter the actor or replace the original feature gate.

These are exploratory controls, not replacement acceptance criteria. All original stage status and gate files remain unchanged.

[Numerical results and provenance](input_audit.json) · [Controls CSV](input_controls.csv) · [Per-episode predictions](dev_predictions.csv) · [Generator controls](generator_controls.json)
