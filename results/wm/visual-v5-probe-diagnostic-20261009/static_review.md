# Frozen V5 feature/probe review

Reviewed frozen scientific source `d0b4fef25e98eefb58bfba737997cb0145f27daa`. No frozen source, original result, SSH session, or GitHub state was modified. All referenced line numbers below are in that source. New diagnostic scripts/results are separate exploratory evidence; the original failed gate remains failed.

## Findings that materially change interpretation

### 1. A positive displacement probe score is not, by itself, evidence of temporal dependence

`data.py:50–58,60–66` samples an initial position from restricted intervals, evolves it under signed velocities, then reflects it at known lane boundaries. Consequently, current position can predict part of displacement through the generator's prior, without looking at a previous frame. This is a falsifiable confound, not just a generic caution.

I regenerated the original Train 128 / Dev 128 samples and fitted the same centered, trace-normalized Train-only linear ridge with alpha 0.01, using **only true positions at t=7** as input. Its Dev mean displacement R2 is approximately **0.314991**, compared with the recorded raw-feature two-frame score **0.080553**. A separately reported degree-3 position control gives approximately **0.314277**; neither was selected on Dev. These are privileged diagnostic inputs, not proposed actor inputs. The script also adds a t=7 RGB-centroid-only control, which does not access a previous frame.

Therefore, do not say that raw features have established temporal understanding merely because raw displacement R2 is positive. Required discriminating test: the same raw/pooled ridge using z6 only, z7 only, duplicated z6, duplicated z7, and the original [z6,z7]. Improvement over the single-frame controls is the relevant temporal-dependence observation. The original gate may remain a useful cue-screening prerequisite, but it is not this causal test.

The newly completed input audit provides the actual discriminating controls:

| Representation | Two frames | t6 only | t7 only |
|---|---:|---:|---:|
| Raw | 0.080553 | -0.067145 | -0.050833 |
| Original pooled | -0.020927 | -0.166888 | -0.222929 |

Both representations improve when supplied both frames under this fixed probe. Therefore **pooled negative R2 is not evidence of no temporal information**. The single-frame RGB centroid control gives 0.314306, reinforcing the separate point that a positive absolute score can also arise from generator priors. These findings are compatible: the original learned probes use additional two-frame evidence yet remain inefficient relative to the image-space cue.

Evidence and reproducibility: `validation/static_probe_checks.py`, `validation/static_probe_checks.json`; committed-ready `scripts/audit_visual_probe_inputs_v5.py` and `results/wm/visual-v5-probe-diagnostic-20261009/input_audit/` include the reusable controls.

### 2. The currently compared raw and pooled inputs are not produced from exactly the same numerical encoder output

`features.py:97–104` obtains stored pooled t6/t7 features in episode batches of up to 32 images, but re-encodes t6/t7 separately with batch size 2 to obtain raw cues. `features.py:42–48` pools the encoder's FP32-cast output before casting pooled features to BF16, whereas raw cues are stored as BF16 separately.

Thus the actual comparison is approximately:

- original pooled: `BF16(Pool(FP32(z_batch32)))`;
- cached raw repooled: `BF16(Pool(FP32(BF16(z_batch2))))`.

They differ by pooling, encoder batch shape, and quantization order. In eval mode there is no intentional cross-image conditioning, but numerical equality is unproven. Falsifiable CPU-only test: reconstruct 4x4 pooled features from existing raw cues, report tensor differences against stored pooled t6/t7, and re-run the unchanged alpha-0.01 probe. Compare both BF16-stored and FP32-kept repooled outputs. A changed score would show that the original comparison was not a pure spatial-pooling ablation; it would not uniquely identify batch size as the cause. Separating batch shape from pre-pooling quantization would require retaining/recomputing the same FP32 z in a newly identified diagnostic run.

**Executed falsification:** BF16 raw-repooled displacement R2 is -0.020736; FP32-kept raw-repooled is -0.020055; original pooled is -0.020927. Original-versus-BF16-repooled relative RMSE is 0.15097% on Train and 0.15139% on Dev. All remain below the unchanged gate. This narrows the diagnosis: the numerical batch/quantization confound exists but does not explain the observed failure by itself; the raw-to-pooled gap persists when pooling the same cached raw input.

### 3. Concatenation itself is not incapable of expressing a displacement; fixed ridge geometry can suppress the useful directions

`features.py:163–171` applies a **linear** kernel to `[z6,z7]` after Train centering, with one scalar feature-energy normalization and fixed ridge alpha 0.01. Its kernel is `K6+K7`. With `s=(z6+z7)/sqrt(2)` and `d=(z7-z6)/sqrt(2)`, concatenating `[s,d]` is an orthogonal change of coordinates and gives exactly the same kernel, trace normalization, ridge predictions, and score (up to arithmetic rounding). Merely renaming concatenation as sum/difference cannot fix this gate.

The change that could matter is a **different prior**, such as `[s,gamma*d]` with gamma not equal to 1, or a difference-only kernel normalized by its own smaller energy. Those changes reweight static versus temporal feature directions; they do not create new information. The falsifiable control is gamma=1 numerical equivalence, followed by a predeclared Train-pair cross-validation of alpha/gamma or a shared position decoder. Do not choose a new parameter because it makes the already-inspected Dev gate pass.

Crucially, do not claim that the four position labels dominate a joint multi-output loss: `torch.linalg.solve(..., y-ym)` fits output columns independently. A high-variance position target does **not** directly steal capacity from a displacement output. The possible domination is in the **feature geometry and regularization**, not competition among target columns.

### 4. The recorded raw-versus-pooled difference is more established than the sign of either absolute score

From the already-published per-Dev predictions, resampling all 64 independent pairs together for 5,000 fixed-fit bootstrap replicates (seed 171) gives approximate 95% intervals:

| Quantity | 95% fixed-fit pair-bootstrap interval |
|---|---:|
| Raw mean displacement R2 | [-0.037717, 0.184695] |
| Pooled mean displacement R2 | [-0.163236, 0.100472] |
| Raw minus pooled | [0.033256, 0.174858] |

These intervals condition on the already-fitted models and omit training-set uncertainty. The original point-estimate gate is unchanged: pooled failed. The intervals argue against a broad conclusion that raw reliably clears zero while pooled fundamentally cannot contain motion, while supporting a difference between these two fitted probes on this Dev sample.

`features.py:173` uses squared error against the **Train mean** as denominator, not the usual Dev-mean R2 denominator. The existing diagnostic README accurately states this. Recomputing with the Dev mean leaves the same raw-positive/pooled-negative point-estimate pattern; it is not the cause of failure.

## Checks that rule out several concrete implementation hypotheses

- **Wrong frame indices:** `data.py:45–49` always puts observed times 0…7 first; interference observations start at 7.3 and update observations later. I checked all 256 original Train/Dev episodes: indices 6 and 7 are exactly t=6 and t=7. `features.py:104,163` selects the intended frames.
- **Unobservable pre-reflection velocity label:** `data.py:109–110` labels actual displacement `position(7)-position(6)`, using the reflected positions. It does not ask the probe to infer the original latent sign before reflection. The legacy field name `velocity_r2` should continue to be explained as observed displacement.
- **No motion in the raster:** Every original Train/Dev episode has different t6 and t7 RGB. A fixed-color centroid decoder applied directly to those two visible images obtains mean Dev displacement R2 approximately **0.977881**. This is a renderer-specific diagnostic oracle, not a fair learned model or an actor input; it demonstrates that the scored cue is observable in the rendered pixels.
- **Paired recent-input leak:** All regenerated pairs have identical recent four RGB observations. Public shutter schedules and times depend on family/gap/write count, not variant velocity (`data.py:40–49,94–101`). Pair IDs, masks, labels, and variants are cached as metadata/evaluation fields; they are not supplied by `predictor.batch` to ActorBatch.
- **Train/Dev label fitting leak:** `features.py:165–171` uses Train feature means, Train targets, and Train-only solve; Dev supplies evaluation features and labels only. The same pair number across splits does not mean the same world because `seed_for` hashes the split (`data.py:16–17`). Pairwise bootstrap/grouped cross-validation must still treat the two variants as one independent unit: there are 64 pairs per split, not 128 independent worlds.
- **Full-stream velocity target versus update timing:** The current feature gate tests t6→7 before later updates, irrespective of family. It therefore screens an early visible motion cue, not whether an update observation or memory writer can recover a new velocity. A pass would not itself validate update handling; a failure does not compare the future writer architectures.

## Immediate diagnostic order

1. Completed: original gate reproduced from cache with maximum per-dimension R2 difference below 1.6e-14, recording cache/source hashes.
2. Completed: raw-derived pooling gives almost the same failed score; retain the quantization-order caveat but do not present it as the current root cause.
3. Completed: both raw and pooled benefit from two frames; duplicated and single-frame predictions agree exactly. All dimensions are reported, since the original raw gain is especially concentrated in A-dy while B-dx is negative.
4. In a separately labeled exploration, use Train-pair-only tuning for temporal reweighting/regularization, and a shared position decoder followed by displacement. Keep Dev as a reported exploratory holdout, never retrospectively replace the original acceptance record.
5. Only after diagnosing the representation/probe issue should a fresh, versioned scientific run be proposed. No automatic continuation into P0 or memory comparisons is justified by this review.
