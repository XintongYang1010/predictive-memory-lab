# V5.3 execution: matched nuisance-balanced reader repair

Registered 2026-10-10 UTC. The machine-readable authority is
[`scripts/v5_3_plan.json`](../scripts/v5_3_plan.json). Freeze and publish the full
implementation, this document and that plan before generating any scientific
V5.3 FIT, anchor or bridge sample. This protocol records a design, not a result.

V5.2 found that freezing the yellow distractor reduced known-bridge horizontal
MSE by 85.63%; both static anchors helped and swapping the nuisance reversed
many common residuals. That supports a rendering contribution in a fixed
pipeline. V5.3 asks whether matched balanced supervision repairs transfer to
**new original-distribution scenes** without erasing useful target motion.
It remains a representation/readout diagnostic. Original V5 and V5.1 gates stay
`failed_unchanged`; `test_access=false` and `p0_authorized=false` in every receipt.

## Matched FIT worlds and exact view layout

Use new schema `v5.3-nuisance-repair-1`, roles `fit`, `anchor` and
`software_test`, and blocks 0–255 per scientific role. Seed with the first eight
bytes, little-endian, of SHA256(`schema/role/block`), then NumPy `default_rng`.
Software tests may generate only their separate role. Do not render, inspect,
encode or score future scientific samples during testing or preparation.

Each world draws the same center, nonzero displacement, colors and geometry as
the V5.1 quartet construction, then independent yellow locations `a,b` uniform
on `[0.1,0.9)`. Center bounds are `[.20,.20,.20,.65]` to
`[.80,.35,.80,.80]`; displacement magnitudes are `.008–.018`, independent signs;
round center/displacement to multiples of `2^-20`. These target cue trajectories
have no reflection. Reuse the original grid, object glyphs, colors, lane/shutter
rendering, and rectangle rasterization. Keep a and b when they round to the same
pixel; report their frequency without rejection sampling or new seeds.

Encode exactly these six logical image slots once, in pose-major order:

| Index | Target positions | Yellow position |
|---:|---|---|
| 0 | C−d | a |
| 1 | C−d | b |
| 2 | C | a |
| 3 | C | b |
| 4 | C+d | a |
| 5 | C+d | b |

Four ordered target views have pose indices `(0,1),(2,1),(1,2),(1,0)`: shared-end
`+d,−d`, then shared-start `+d,−d`. Cross each with nuisance paths `aa,bb,ab,ba`.
Flatten target-major / path-minor: `flat_index=4*target_view+path`; an image index
is `2*pose+nuisance`. Every world has 16 ordered views. The target is the end
position and observed end-minus-start displacement; it is identical across
nuisance paths. Metadata, path IDs, coordinates and masks never enter a reader.

This six-image construction preserves exact shared start/end image anchors
within a path while allowing opposite target motions to share the same forward
nuisance motion. A moving three-image quartet would force nuisance reversal
with target reversal and would not implement this comparison. Full RGB reversal
maps both the target view **and** `ab↔ba`; do not assert a same-path reversal
invariant for moving paths.

S uses only the eight `aa,bb` views/world. B uses all 16 views/world. Both arms
use the same 256 worlds, six encoded images/world and target labels. Each world
has empirical loss weight `1/256`; uniform within-arm view weights are `1/8` or
`1/16`. Thus B gets more nuisance assignments, not doubled world weight.

## One shared kernel geometry and fixed readers

Keep the official I-JEPA encoder and 4×4 pooled representation unchanged. Exact
source/checkpoint hashes are in the plan. Retain BF16 CUDA inference, promotion
to FP32 for pooling, pooled BF16 quantization, then FP32 cache storage. This is
not a full-FP32 extraction experiment. No new encoder, pooling, raw-token branch
or image preprocessing candidate is permitted.

Fit both S and B for each input `concat`, `t6`, `t7`: six models total. Every
model uses the **quadratic** kernel and mean-loss alpha **7.8125e-7**; no CV,
grid, tuning, early stopping or winner selection. This applies the previously
chosen primary kernel/penalty to the controls too; it does not reuse the old
single-frame linear model or its different alpha.

For each input, use the full 4,096 B FIT views as the common *unlabeled*
normalization reference. Center and trace-normalize the linear kernel there,
apply `(1+k)^2`, then center and trace-normalize the quadratic RKHS kernel on
the same reference. Freeze both levels' moments once. S takes its 2,048-row
submatrix of that fixed kernel; B takes all 4,096 rows. **Do not re-center or
re-normalize either arm.** Query centering means must also use every B-reference
column before selecting S columns. Computing the query mean over S columns
would silently change the kernel.

Fit an **unpenalized intercept in both arms**, with `n_S=2048`, `n_B=4096`.
In FP64 solve the augmented system
`[[K_arm+n_arm*alpha*I,1],[1^T,0]] [coef;intercept] = [y_arm;0]`, or its
algebraically equivalent Schur complement. Predict
`K_query,arm @ coef + intercept`, saving the fitted intercept for every model.
Keep the same shared FIT target mean as the score baseline: the S/B normalized
label distributions agree, but their prediction intercepts need not.

This retains the same ridge family in both arms. The B-centered full kernel
has zero row means, for which its fitted intercept equals the FIT target mean
and the solve is equivalent to ordinary centered KRR. The S submatrix need not
have zero row means, so forcing its intercept to that mean would introduce a
second modeling restriction along with the supervision change. Do not instead
re-center the shared S kernel; fit the free intercept through the system above.
This choice is frozen before any scientific FIT or evaluation generation.

Accordingly S is **static-supervision with shared unlabeled FIT normalization**,
not an arm that never encounters the moving-view feature distribution. This
choice controls kernel geometry while isolating the supervised nuisance-path
coverage. All shared moments are FIT-only; no anchor or bridge features/labels
may affect them.

For t6 and t7 separately, the weighted input-label empirical distribution is
identical in S and B: B only duplicates each single-frame assignment. Their
predictions must agree to maximum absolute error `1e-7` on FIT and each
evaluation set. Treat a violation as a weighting/normalization/numerical error,
save the attempt and stop interpretation. This equivalence is a software and
design check, not additional independent evidence of a successful repair.
Validate the fixed tolerance with software fixtures before scientific release;
never loosen it after seeing scientific FIT or evaluation values.

## Release sequence and data independence

1. Complete the path/hash freshness audit and live resource checks. Freeze and
   publish source, plan and protocol, then independently read back the publication.
2. Extract FIT: 256 worlds × six slots = 1,536 images. Fit all six predetermined
   readers on CPU and verify shared moments, weights and single-frame equivalence.
3. Freeze all six coefficient arrays and intercepts, shared moments/target mean, the complete
   FIT feature basis, S/B row mappings, source/plan/environment and input audit.
   Write a final freeze receipt with hashes. Publish it and the complete models
   to GitHub; independently verify readback before any evaluation access.
4. Release fresh `anchor` blocks 0–255, exactly once: another 1,536 images.
   Evaluate all arms/readers and prescribed controls. Do not change the design
   after seeing anchor results.
5. Release original-generator Dev pairs **384–639**, both variants: 256 paired
   worlds / 512 episodes, the t6/t7 cues = 1,024 images. Keep original geometry,
   moving nuisance, shutters and reflections; use observed displacement labels.
   Evaluate the same frozen models without fitting or transforming outputs.
6. Report both evaluations and every fixed model. A poor anchor score does not
   authorize changing or skipping the locked bridge comparison. Stop only for
   identity, freshness, software/numerical, resource, budget or execution failure.

Negative controls reuse the fresh anchor center images `[2,3]` and `[3,2]`:
target positions are fixed at C, target displacement is exactly zero, and only
yellow moves. They add no encoding and never enter FIT supervision. Static
center repeats may be evaluated from the same features as labeled numerical
checks, not substituted for the moving-nuisance negative control.

The local read-only audit supporting the new range found:

- The original Dev probe manifest has 128 records / pairs 0–63; the archived
  freshness audit records that range and reserves pairs 0–127 for old P0 Dev.
  That audit's SHA256 is
  `0ac68b3c94e2da88df129c462f2de9cf74027ce53031e0dcf81436df0c4864dc`.
- Frozen V5.1 extraction/reader enforce exactly Dev pairs 128–383. Its bridge
  prediction file has SHA256
  `c9d510b314ef13276e9cb224b047749441ec69969f8f74c8655bf470497e6b80`.
  The file has no pair-ID array; its ordering is bound by the verified cache
  receipt and extractor rather than inferred from its prediction values.
- The sealed V5.2 prediction NPZ explicitly contains 256 pair IDs, min 128,
  max 383, with SHA256
  `0c1ddd7e96a16ea4cdf8b92c70143e212287d84be499cc1ddf532b993ddb5e09`.
  Its extractor and scorer also enforce that exact range. Reviewed existing
  tests use software namespaces/fixtures or original low-numbered pairs.
- A **seed-hash-only** check, without RNG draws or images, found 256 unique
  32-bit seeds for new Dev 384–639 and no collision with Dev 0–383 or planned
  Train 0–1023. New 64-bit `fit`, `anchor`, `software_test` seeds are unique per
  role, disjoint across roles, and disjoint from old V5.1 role seeds 0–255.

These checks cover the known local artifacts and reviewed implementation. They
do not establish that nobody accessed these scenes in an unknown external run.
The executor must also inspect the known remote directories/logs and record
paths/hashes plus a no-access attestation before release. A namespace change
is not proof of freshness by itself. Any contrary evidence stops this
registration; do not silently substitute new pair IDs or extend sample counts.

## Primary contrast and safeguards against output collapse

The **only primary endpoint** uses concat on the fresh original bridge. For
each pair, average squared displacement error over both variants and the two
horizontal coordinates A_dx/B_dx. Compute `Delta = mean_pair(L_S − L_B)`.
Positive favors B; units are normalized image-coordinate squared. Use NumPy
`default_rng(7811)`, 10,000 bootstrap resamples of 256 complete pairs, with both
variants and both arms retained together. Report the two-sided percentile 95%
interval. Its lower endpoint above zero establishes relative horizontal error
improvement under this fixed design, conditional on these fitted models.

Do not call relative improvement alone a motion-preserving repair: shrinking
all outputs can improve MSE when S is badly biased. Fixed supporting evidence:

- Fresh bridge: each coordinate's MSE and skill against the shared FIT mean,
  four-motion NMSE, position skill, signed bias, direction errors with explicit
  zero/tie counts, mean absolute amplitude ratio and prediction/target SD ratio.
  Report both arms and all three inputs. A zero predictor's motion skill is the
  structural zero baseline; positive B skill matters beyond an S−B improvement.
- Fresh anchors: for every nuisance path, compute end contrast `(view0−view1)/2`
  and start contrast `(view2−view3)/2` against the known nonzero d. Report all
  eight coordinate contrast skills, absolute motion skills, and fixed static
  (`aa,bb`) versus moving (`ab,ba`) aggregates. Compute each path's prediction
  contrast and loss first, then average paths; do not cancel opposite nuisance
  responses before squaring. A zero-motion predictor has
  zero contrast skill. Preserve the appropriate shared-anchor single-frame nulls.
- Moving-nuisance / static-target negatives: report per-coordinate motion RMS,
  its ratio to corresponding nonzero FIT displacement RMS, both directions,
  both arms, and their paired difference. Low negative-control RMS alone could
  mean collapse; interpret it with nonzero-target anchors and fresh bridge.
  Square each direction's predicted motion first and then average before the
  square root; averaging signed forward/reverse predictions first is invalid.
- Use pair/block bootstrap with the same fixed seed and 10,000 resamples for
  bridge S−B dx/Motion-NMSE differences, anchor motion/contrast skills, and
  negative-control RMS comparisons. Secondary intervals are descriptive and
  unadjusted. Direction, amplitude, bias and covariance remain point diagnostics
  without separate significance claims; correlated metrics and single-frame
  equivalence are not replications.

If the primary interval is positive but B's fresh-bridge mean motion skill is
nonpositive, state "relative horizontal error improvement with unresolved
motion transfer", not "repair". A positive mean still does not establish every
coordinate or target: report each coordinate and the per-path nonzero anchor
contrasts. Report a mixed outcome when the primary improves but motion evidence
weakens or stays absent. No hidden composite score, threshold search, posthoc scaling,
direction correction, new candidate, or automatically relaxed P0 gate is allowed.
The protocol does not set an omnibus "repair passed" flag; retain the actual
primary estimate and supporting profile. Even a favorable result concerns
pooled representation/readout transfer, not a trained world model or memory.

## Strict cumulative budget and evidence retention

Inherit the reconciled cumulative charge **2.2082544136047373 GPUh**, with its
ledger hash and earlier leases. The total experiment cap remains 60 GPUh and
at least **54 GPUh stays reserved for a future pilot**. V5.3 has at most
**3.75 additional attributable GPUh**, including every failed attempt and
management overhead. The effective bound is the minimum of this version cap
and `60 − 54 − current_cumulative`; recheck before each step, not just at launch.

Use the existing eight authorized H800 devices after revalidating the live
Slurm account/partition/QOS, ownership, UUID binding, free VRAM and existing
workloads. There are exactly three GPU stages, FIT/anchor/bridge. Each has a
six-minute outer Slurm time limit and inner `.09` hour lease. Reserve the
controller envelope `(6/60 + .05) × 8 = 1.2 GPUh` per stage, so all three
envelopes total **3.6 GPUh**. The remaining `.15 GPUh` in the version cap covers
other attributable management time; it is not another scientific attempt.
Fit/evaluation run in CPU-only steps. Never charge the entire shared parent
allocation to this diagnostic, and never kill IndexMem or other existing work.

The 4,096 total image slots are roughly twice V5.2's 2,048; its measured cost
suggests about 1 GPUh for this extraction volume. This is planning evidence,
not a runtime promise: three model-load/startup costs and CPU fitting differ.
If a phase cannot fit its reserved envelope, stop and retain the partial scene.
Do not extend the time, restart to find favorable scores, reset the ledger,
borrow from the 54 GPUh reserve, or silently enlarge the cap.

Archive all code/plan/protocol identities, original data/encoder hashes,
freshness evidence, commands and environment, live GPU receipts, actual Slurm
job/step IDs and accounting, worker logs, FIT features and model coefficients,
evaluation input/cache hashes, all model predictions/targets, metrics/plots,
both publication/readback receipts, numerical audits and the final aggregate
ledger. Re-read and verify the uploaded archive before considering cleanup.
Execution/archive failures and unresolved scientific repair retain the new
remote scene; old failed scenes and sealed archives remain intact. Completion
of this bounded diagnostic never automatically launches P0 or removes evidence.
