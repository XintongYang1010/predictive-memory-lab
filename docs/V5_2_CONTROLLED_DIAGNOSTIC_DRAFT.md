# V5.2: frozen-reader nuisance counterfactual protocol

Protocol dated 2026-10-10 UTC. Freeze this document and the extraction, scoring,
and launch source before the diagnostic GPU step. The filename retains `DRAFT`
for continuity; the four-condition experiment below is the execution protocol
once its Git commit and file hashes are recorded. It contains no new result.

This is an exploratory controlled intervention on the **already inspected**
V5.1 bridge. It does not create a fresh confirmation set, repair either earlier
gate, authorize P0, or test memory. The original V5 gate and V5.1 bridge remain
failed. Preserve the sealed archives and record this work in a new directory.

## Question and fixed evidence

Does changing only the yellow distractor's trajectory reduce the frozen
pipeline's horizontal-displacement error on the existing bridge scenes?

The motivating posthoc analysis found correlations of approximately 0.8961 and
0.9025 between the paired-world common residual and the yellow marker's actual
pixel displacement for A_dx and B_dx. Common residual energy accounts for
81.36% and 84.40% of the corresponding MSE. These associations motivate an
intervention; they do not establish a causal contribution on their own.

The authoritative previous artifacts are
[`visual-v5_1-20261010`](../results/wm/visual-v5_1-20261010/README.md), its frozen
reader receipt, original bridge input receipts/cache, and bridge predictions.
The frozen reader's model SHA256 is
`2136993ed79ace121d45086698ad2d6b6800c2a3017d44474c757ed48b6bcfdc`;
its selection JSON SHA256 is
`642dc9d9d3fa72cf5aa34268e0a81c9f7a8ca753c47debec62adfb74dcf16f3b`.
Verify the complete receipts, not just these two strings.

The primary reader remains the selected two-frame `concat` reader, with its
frozen quadratic kernel, coefficients, feature moments and target mean. Keep
the already frozen `t6` and `t7` readers as secondary controls. Do not refit,
retune alpha, change normalization, select a reader by these results, or
subtract the observed nuisance correlation from predictions.

Use the same official I-JEPA ViT-H/14 target encoder, source commit
`52c1ae95d05f743e000e8f10a1f3a79b10cff048`, and checkpoint SHA256
`0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`.
Retain the original 4×4 spatial pooling and precision path: CUDA BF16 autocast,
model outputs promoted to FP32 for pooling, pooled values cast to BF16, then
stored as FP32 by the extractor. FP32 cache storage is not full-precision
inference. Do not change this path in an attempt to improve the diagnostic.

## Existing scenes and four interventions

Use exactly Dev pairs **128–383**, both original variants, for 256 complete
paired worlds / 512 episodes. No selection of bad cases, fresh seeds, top-ups,
or Test access. Include every existing family, geometry and reflection case.
The original targets remain `[A_x,A_y,B_x,B_y,A_dx,A_dy,B_dx,B_dy]`, where each
displacement is the observed position difference from t6 to t7, including any
reflection. Do not replace it with the initial latent velocity.

For each episode let `P6,P7` denote the unchanged target renderings, and let
`n6,n7` be the original yellow marker's integer-pixel centers. The original
generator uses `int((0.5 + 0.4*sin(2.3*t + pair))*224)`; both variants have the
same `n6 -> n7` nuisance trajectory. Target trajectories have their original
variant-specific directions. Construct four image slots:

| Slot | Target rendering | Yellow center |
|---|---|---|
| 0: `u00` | original t6 | `n6` |
| 1: `u01` | original t6 | `n7` |
| 2: `u10` | original t7 | `n6` |
| 3: `u11` | original t7 | `n7` |

The condition order and feature-index mapping are fixed:

| Condition | Ordered slots | Meaning |
|---|---|---|
| `original` | `[0,3]` | original moving nuisance |
| `static_t6` | `[0,2]` | both frames retain the original t6 nuisance position |
| `static_t7` | `[1,3]` | both frames retain the original t7 nuisance position |
| `swap_nuisance` | `[1,2]` | reverse only the nuisance trajectory |

Target time order, positions, colors, glyphs, shutters, background and labels
are identical across these four conditions. `static_t6` shares its exact first
image with `original`; `static_t7` shares its exact second image. The swap does
not reverse target motion or labels. Report all four conditions even if neither
static intervention helps; do not pick the better static anchor as the result.

Restore the underlying original background before relocating yellow, then draw
the same rectangle with the same PIL rasterization and color. For each edited
image, pixels **outside the union of the old and new rectangle raster masks
must be exactly unchanged**. Verify exact target pixels/masks as well. The
yellow lane is disjoint from both target lanes in this generator, but enforce
this fact with the actual masks rather than assuming a continuous-coordinate
bounding box matches PIL. Hash the masks, original/edited RGB and labels in
per-pair receipts. Reconstructed original RGB and labels must match the archived
bridge input hashes exactly before extraction can be considered valid.

Encode the four slots once per variant and reuse the exact cached tensors to
assemble every condition: eight logical image slots per pair / 2,048 total.
Record any coincident image hashes explicitly; do not assume every slot has
different pixels. Preserve the original batch layout for parity where feasible:
one batch `[v0:u00,v0:u11,v1:u00,v1:u11]`, matching the original bridge call,
then `[v0:u01,v0:u10,v1:u01,v1:u10]`. Do not independently re-encode copies of
an anchor for different conditions. Cache the explicit mapping and condition
names so the scorer can validate them before prediction.

## Original-condition parity: required numerical validity

Verify the encoder, frozen reader, original source, input RGB, labels, pair IDs,
variants and cue times against archived identities. Exact identity failures
stop scientific interpretation. Re-evaluate the original condition and compare
its predictions to the archived original bridge predictions for every frozen
reader. Record differences for all eight target coordinates; the following
required tolerances apply separately to each of the four motion coordinates.

Let `s_k = sqrt(mean(y_k**2))` on the fixed 512 original bridge episodes and
`delta_k = prediction_new_original_k - prediction_archived_original_k`.
Require finite values and `s_k > 0`, then require all of:

- `sqrt(mean(delta_k**2)) / s_k <= 0.01`;
- `max(abs(delta_k)) / s_k <= 0.10`;
- absolute change in the coordinate's original bridge skill `<= 0.02`, using
  the same frozen FIT target-mean denominator for both calculations.

Report all coordinate differences and comparisons, including position values;
position parity has no additional acceptance threshold. These checks apply to
`concat`, `t6` and `t7`, not only the most favorable reader. Any required parity
failure produces `diagnostic_inconclusive`, saves all available outputs and
receipts, and stops the controller with an error. Preserve the scene. Do not
relax tolerances, change the encoder batch scheme, or rerun until a favorable
result under the same frozen protocol. A necessary repair requires an explicit
new source/protocol identity with the failed attempt retained.

Use the **new matched original** for every intervention contrast. The archived
original is the parity reference, not a mismatched treatment comparator.
Report numerical drift alongside effect sizes; effects comparable to that drift
do not warrant a strong directional interpretation even when parity passes.

## One primary endpoint and paired uncertainty

For pair `p` and condition `c`, define horizontal displacement loss in normalized
image-coordinate squared units:

`L[p,c] = mean_{variant in {0,1}, k in {A_dx,B_dx}} (prediction[p,variant,c,k] - y[p,variant,k])**2`.

The **single primary endpoint** is raw horizontal MSE reduction:

`Delta_orig = mean_p (L[p,original] - (L[p,static_t6] + L[p,static_t7])/2)`.

Positive values mean the average of the two fixed static interventions reduces
horizontal error. Average the two **losses**, not the predictions: averaging
predictions first would introduce an unregistered ensemble. Report each anchor's
separate horizontal MSE reduction too, so an average does not hide an adverse
anchor. Never choose the anchor after seeing its score.

Use NumPy `default_rng(7810)`, **10,000** bootstrap replicates, sampling 256
paired-world indices with replacement per replicate. Each sampled pair carries
both variants, all coordinates, all four conditions and all readers together.
Report the percentile two-sided 95% interval `[q0.025,q0.975]` for the primary
mean paired effect. The interval is conditional on this one frozen reader and
the inspected bridge population; it does not account for the posthoc choice of
the nuisance hypothesis. It is not a new generalization gate.

Predeclare the following secondary analyses without choosing among them:

1. The balanced horizontal contrast
   `Delta_balanced = mean_p ((L[p,original]+L[p,swap_nuisance])/2 - (L[p,static_t6]+L[p,static_t7])/2)`.
   Each time slot has the same marginal mixture of yellow positions in the two
   sides of this contrast. This helps distinguish cross-time assignment from a
   particularly favorable static location. Report its paired bootstrap interval.
2. All four conditions' per-coordinate raw MSE and frozen-FIT-mean skill,
   average position skill and four-coordinate motion NMSE; both anchors'
   separate reductions; their mean static NMSE reduction; swap versus original.
   For coordinate `k`, use `D_k = mean((y_k - frozen_FIT_mean_k)**2)` shared by
   every condition, and `NMSE_c = mean_{four motion k}(MSE_c,k / D_k)`.
   Recompute that common denominator within each paired bootstrap replicate.
3. For each coordinate, signed bias, prediction/target amplitude, direction
   errors, pair-common residual energy, and its correlation with the exact
   integer-pixel nuisance displacement `eta_p=(n7-n6)/224`.
4. Nuisance-swap response with targets and chronology held fixed. Report the
   paired common prediction contrast
   `R[p,k]=mean_variant(prediction_original - prediction_swap)/2`, its signed
   alignment with `eta_p`, through-origin descriptive slope
   `sum(eta_p*R[p,k])/sum(eta_p**2)`, and original-versus-swap residual sign
   reversals. Report zero-displacement and zero-response/tie counts explicitly;
   no such exclusions affect the primary endpoint. This is a descriptive
   response statistic, not a newly trained correction or reader.
5. The already frozen t6/t7 controls for the same contrasts. In particular,
   t6 predictions must agree for original/static_t6 and static_t7/swap inputs,
   and t7 predictions for original/static_t7 and static_t6/swap, up to reported
   numerical arithmetic error (maximum absolute difference at most `1e-10`;
   otherwise the diagnostic is inconclusive). These are reused-input implementation checks,
   not evidence of temporal reasoning.

Secondary intervals are descriptive and unadjusted. Do not count multiple
correlated favorable intervals as independent replications, choose a new
primary endpoint, or use their signs to launch training. Save case-level
predictions/targets, including unchanged or worse cases, for every condition.

## Interpretation, execution and provenance

A verified change in this matched comparison identifies a causal contribution
of **yellow-rendering assignment to this frozen encoder-plus-reader pipeline's
error on these known scenes**. The balanced contrast has the stronger nuisance
marginal control. Neither contrast localizes the cause to the encoder alone,
proves that target motion information is absent, proves the reader uses target
temporal information, or demonstrates memory utility. Static recovery would
not rehabilitate the original moving-nuisance task or its failed bridge. A null
effect would not eliminate interactions with geometry or other distribution
changes. No result here authorizes P0, adaptation or replay.

Run one new, bounded extraction step on the eight previously authorized H800
devices after checking the live allocation and selected UUIDs. Preserve all
other workloads, including existing IndexMem. Maximum step wall time is 12
minutes: reserve 1.6 GPUh for extraction plus at most 0.4 GPUh for the controller
and startup/shutdown envelope, **2.0 new GPUh total**. Include failed attempts
and any idle selected-GPU time. CPU scoring uses no GPU step.

The original **60 cumulative attributable GPUh** cap remains in force. Import
the previous reconciled ledger and hash; never reset it because a directory or
version changed. The sealed V5.1 total was 1.7294136471218544 GPUh. Recheck any
later charges before launch and reserve at least 54 GPUh for the possible
future pilot: this diagnostic's full 2 GPUh envelope fits that constraint at
the sealed charge. Budget exhaustion yields a partial diagnostic, not a
scientific conclusion or permission to enlarge the cap.

Archive source/protocol hashes, prior-scene and frozen-reader hashes, exact
commands/configuration, environment and precision, live GPU binding receipts,
actual Slurm step IDs/accounting, worker logs, input-edit checks, per-pair/cache
hashes, parity results, all four conditions' predictions and scores, confidence
interval settings, figures, and the cumulative ledger. State `test_access=false`,
`p0_authorized=false`, and both old gates `failed_unchanged`. Hash the final
README and plots after writing them, then rebuild and independently verify the
inventory. Re-read the uploaded archive before any authorized cleanup. On an
execution, parity or archive failure, retain the new remote scene and report
the specific failure. Never change the older sealed archives.

## Future suggestion only: position × nuisance factorial

The following is **not released for execution by this protocol** and is not
needed before the four-condition diagnosis. A future frozen-reader experiment
could pair interior versus horizontal-edge target geometry with static versus
moving nuisance while preserving target displacement, colors and background.
It needs its own frozen definition and fresh namespace if used for confirmation.

The existing quartet uses three images at target positions `C-d,C,C+d` and
reuses a center anchor. With a moving nuisance, assigning just one nuisance
position to each of those three images necessarily reverses nuisance motion
when a view reverses. That differs from the original bridge's same-direction
nuisance across opposite-target-motion variants. It cannot diagnose the two
factors cleanly while pretending the old three-image construction is unchanged.

Instead use six image slots `(C-d,C,C+d) × (n6,n7)` per target geometry. For
the forward nuisance schedule, the four target views are
`(C-d,n6)->(C,n7)`, `(C+d,n6)->(C,n7)`,
`(C,n6)->(C+d,n7)`, `(C,n6)->(C-d,n7)`.
They preserve shared end/start anchors but do not have literal RGB reversal
across target-reversed views. Therefore the old exact full-image reversal
invariant must not be imposed on this altered design.

Balance both static schedules `(n6,n6),(n7,n7)` and both moving schedules
`(n6,n7),(n7,n6)` without selecting a direction. With two geometry conditions,
this uses 12 unique image slots per base block and yields 32 ordered views.
One possible fixed design is 256 matched base blocks, legal `C±d` positions
without reflection, interior x coordinates inside the FIT range, and balanced
left/right x-edge bands outside that range; keep y positions within the same
FIT lanes so the factor is specifically horizontal support. Preserve each
block's displacement and appearance across geometries and keep nuisance
direction independent of target direction. Define the precise edge bands and
sampling maps before generation; this suggestion does not claim to reproduce
the full original reflected bridge distribution.

Bootstrap whole matched base blocks. Predeclare nuisance main effect, geometry
main effect and their interaction with explicit multiplicity treatment if used
as confirmatory endpoints. This design may separate nuisance sensitivity and
support shift, but it remains a frozen-readout representation diagnosis until
a separately registered original-task prerequisite and memory experiment pass.
