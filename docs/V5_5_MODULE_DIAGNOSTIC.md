# V5.5: object interventions and module-boundary diagnostics

Status: design before execution, 2026-10-10. This is an exploratory diagnostic
of **already evaluated** V5.4 Dev640–895, not a fresh confirmation or a new
acceptance gate. Freeze and publish/read back the implementation and this
protocol before rendering or encoding its new counterfactual images. Keep all
sealed V5–V5.4 archives, source, models, caches and failed gates unchanged.
No Test access, encoder training, hyperparameter selection, P0 or memory
training. Keep the original W reader fixed; the one new lane-restricted reader
intervention below is fitted only on already archived FIT data.

## Question and claims permitted

V5.2 established a causal effect of yellow-motion interventions on the frozen
encoder→pool→reader chain in its inspected scenes. V5.3 showed that balanced
target/nuisance supervision improves fresh transfer with the same representation.
V5.4 still has a large own-FIT versus held-out gap and did not confirm a benefit
of widening FIT support for its horizontal primary endpoint. None of these
findings uniquely assigns remaining error to encoder, pooling or reader.

V5.5 asks which object's change drives the fixed W reader's output, whether
object changes interact, and which limited facts can be established at the
raw-token→pool and pooled-feature→reader boundaries. A correctly implemented
module may still be a modeling bottleneck. Do not replace this distinction with
an unsupported single-module failure claim.

## Immutable inputs and intervention population

- Use the complete published V5.4 W concat reader, including joint N/W basis,
  coefficients, free intercept and both frozen kernel-normalization stages.
  Do not refit or substitute this baseline. V5.4 source is
  `3279bbe49f8db5ec1bd4b2f5c343db1ce9e5684b`.
- Bind the reader model, freeze/publication receipts, all original bridge
  predictions and input-shard receipts, V5.4 sealed-scene manifest and GitHub
  readback receipt by actual SHA256 in the new registration. The sealed V5.4
  manifest SHA256 is
  `ec7bc9c17da8417e9ecd11ba0f21415a2df0ee7af620af9d12f911c3ab26e75c`.
- Use all 256 existing paired worlds, Dev IDs 640–895, both variants: 512
  episodes. Preserve pairing and count every case, including reflections,
  raster coincidences, zero/tiny observed displacement and unchanged yellow
  raster positions. Do not resample, reject or replace examples.
- The three already published bad episodes (844/1, 761/1, 891/1) are fixed
  illustrations, not a new sample or a basis for choosing analysis settings.

Reconstruct only the known t6/t7 cues using the pinned original renderer and
seed/draw/reflect rules. Do not invoke unrelated future-frame generation or
open other IDs. Verify all 1,024 original cue RGB arrays against the archived
shape/dtype/SHA256 receipts and all original targets bitwise before interpreting
new images. Latent positions and object identities are diagnostic annotations,
never inputs to the frozen W reader.

## Eight endpoint images, fixed first frame

For each episode, the first frame is always the original t6 image. Make eight
second-frame images by independently selecting A's pose, B's pose and yellow's
position from t6 or t7. Index masks as integers 0–7 with bits **A,B,Y**, A the
most significant bit: `000,001,010,011,100,101,110,111`. `000` is exactly t6;
`111` is exactly t7. Each bit changes the whole object's two-dimensional pose,
not a separately selected x/y coordinate. Keep colors, glyphs, radius, grid,
draw order, camera and open shutters unchanged.

For an edge changing one object, pixels outside the union of that object's
old/new rendered support must be exactly unchanged. Preserve renderer layering;
do not paste objects over uncleared old pixels or interpolate images. Verify
the edge invariant for all 12 cube edges and exact original endpoint identities.
The maximum worklist is 512×8 = **4,096 image slots**. Equal raster images stay
as registered slots; any encoding reuse must retain the complete slot mapping.

The target for mask `s` is its selected endpoint positions and their observed
displacements from original t6. Thus A/B displacement is respectively zero or
its original observed displacement, including reflection; yellow never changes
the target labels. This is an image intervention on a two-frame observation,
not a physically integrated intermediate trajectory or a future prediction.

Per paired world, encode its four original endpoint images first in exactly
the old batch order `[variant0:000,111; variant1:000,111]`. Encode the remaining
12 endpoints in a separate fixed batch. This preserves the original batch
shape for parity and avoids treating changed batching as an object effect.
Both variants' first-frame features are reused across their eight transitions.

## One encoder call supplies the precision controls

Keep the exact official target encoder, input normalization, evaluation mode,
BF16 CUDA autocast and checkpoint identity used in V5.4. In a new wrapper,
capture the existing `z = model(normalized_rgb).float()` result; do not modify
the old encoder implementation or run a second encoder for a different control.
FP32 here describes the captured tensor/storage, **not FP32 encoder inference**.

From that same tensor compute:

1. Pre-quantization pooled FP32: reshape spatial raster tokens to 16×16 and
   average each non-overlapping 4×4 block into the 4×4 pooled grid.
2. Canonical pooled: exactly the existing pooling result rounded to BF16 and
   converted to FP32 for storage. These are the only new factorial features
   fed to the unchanged W and separately frozen lane-restricted readers.
3. Raw FP32 tokens, retained only for `000` and `111`: 1,024 images with shape
   `[256,1280]` each, 1,342,177,280 uncompressed bytes (1.25 GiB).

Save canonical and pre-quantization pooled tensors for all 4,096 slots, about
320 MiB each before compression. Save all 8-endpoint predictions. Preserve
per-worker receipts, image/tensor shapes and hashes, batch order, source and
precision metadata. Do not silently cast retained raw tokens to BF16.

For the original endpoints require canonical pooled arrays to be bitwise equal
to the archived V5.4 bridge features. Compare `f(111)` against all eight columns
of the original W predictions with fixed absolute tolerance `1e-9`, `rtol=0`.
Record maximum errors and exact-match flags. Any original RGB/target/feature
identity failure or prediction-parity failure marks the diagnostic inconclusive;
save the failed checks and outputs, stop interpretation, and do not relax the
tolerance after observing results. Pure-software fixtures must establish these
checks before scientific execution. Baseline parity applies to the original
global W reader; a newly fitted lane-restricted reader is expected to differ.

## Fixed lane-restricted reader intervention on existing features

This is the sole new reader comparison. The baseline is the unchanged V5.4
global W concat reader. The intervention changes access to the **same** archived
4×4 pooled features: for every frame, reader A sees row 0 and row 1 (raster
tokens 0–7), and reader B sees row 2 and row 3 (tokens 8–15). All channels remain;
each reader receives 8×1280 = 10,240 coordinates per frame. The assignment is
fixed by the known top/bottom scene layout, independent of target position,
labels, errors or observed saliency. No ground-truth masks or per-example crop.

For each lane independently, recompute the unchanged linear-center/mean-diagonal
then quadratic-center/mean-diagonal normalization formula over all **8,192
original N+W FIT views**, now restricted to that lane's coordinates. Those are
unlabeled reference moments only. Fit on the same W 4,096 supervised rows and
the same 256 worlds, equally weighted, with fixed quadratic kernel, mean-loss
alpha `7.8125e-7`, diagonal penalty `4096*alpha=.0032`, FP64 solve and an
unpenalized intercept. Reader A fits columns `[0,1,4,5]` (A position/motion);
reader B fits `[2,3,6,7]`. Reassemble predictions in the original eight-column
order. Use the original frozen joint-FIT target mean and motion RMS for all
scoring; they are not a fixed predictive intercept.

No alpha search, FIT resampling, feature-dimension selection, alternative lane
boundary, per-case choice or output calibration. Check raster/channel indexing,
both reference/query centering passes, W row selection and each free-intercept
fit against software-sized explicit masked concat/augmented-system calculations.
Check finite arrays, exact coefficient and model identities, and include the
normal-equation own-FIT identity as a numerical check. Publish the source and
fixed configuration; freeze/publish/read back the entire lane models and their
reference moments before releasing new factorial image extraction. Any CPU
fitting performed before publication must use that exact soon-to-be-published
source and be disclosed; no inspected score may change the registered design.

Evaluate both readers on all 512 existing bridge episodes and every factorial
transition. Report the paired difference `global_W dx MSE − lane_W dx MSE`,
its whole-pair bootstrap interval using the fixed procedure below, both absolute
four-coordinate motion scores, direction errors, amplitudes, position scores,
and each reader's repeated-image/yellow/other-target responses. Show every
condition and retain negative or mixed results. This is a posthoc intervention
on known Dev, not fresh confirmation or an automatic downstream gate.

An improvement establishes that changing this downstream feature-access and
kernel geometry can reduce error while holding encoder outputs fixed. It does
not uniquely identify a defective encoder, prove all encoded motion is intact,
or isolate cross-target mixing from dimensionality/regularization-geometry
effects. The yellow rectangle at y≈.48–.52 crosses the row-half boundary, so
both selected halves may contain yellow pixels. I-JEPA spatial tokens already
have global context. Call this **lane-restricted reading**, not yellow removal,
an independently encoded object crop, or an intervention on the encoder itself.

## Factorial response, attribution and error reporting

Let `f(S)` be the eight-coordinate prediction of one specified reader for fixed t6 and endpoint set
`S⊆{A,B,Y}`. Preserve all eight predictions and targets. Report `f(empty)`
separately; it is a repeated-image control, not an assumed zero to subtract
silently from reported task predictions.
Compute the decomposition separately for global W and lane W; do not choose
between them per episode or mix their coefficients or contributions.

Use the exact three-player Shapley response

`phi_i = sum_{S subset others} |S|! (2-|S|)! / 3! * [f(S+i)-f(S)]`.

Check `sum_i phi_i = f(ABC)-f(empty)` numerically. Also retain the exact
Möbius decomposition: main effects `f(i)-f(empty)`, pair interactions
`f(ij)-f(i)-f(j)+f(empty)`, and the corresponding third-order interaction.
These describe a chosen finite renderer intervention; Shapley shares
interactions according to a convention. They are **not layer responsibility
percentages**, and the signed contributions may cancel.

For target A motion, desired contributions are A's true displacement from A
and zero from B/Y; reverse that assignment for B motion. Report per-coordinate
self-response error, other-target/yellow response RMS, repeated-image residual,
interaction magnitudes, original and counterfactual MSE, direction errors,
signed bias and output amplitude. Always keep absolute coordinate units.
Normalize supporting response RMS only by the existing frozen W FIT motion
RMS; never divide individual contributions by tiny per-episode true motion.
For signed averages, also report alignment with the corresponding true-motion
sign so opposite variants cannot hide large responses by cancellation. Retain
zero/tie counts. Position and motion columns remain separate outputs.

If sampling intervals are reported, use 10,000 whole-pair bootstrap resamples,
`default_rng(7812)`, preserving both variants, all eight masks and all compared
outputs; report two-sided percentile 95% intervals. They describe this already
inspected population conditional on one frozen model and this analysis, not
independent confirmation or FIT/reader-selection uncertainty. Report all fixed
conditions regardless of sign; no new pass threshold or automatic downstream
authorization is introduced.

## What the module-boundary checks establish

**Raw→pool arithmetic and compression.** For each original t6/t7 pair, reproduce
the pre-quantization pool from its retained raw tensor with an independent
spatial-block mean implementation. Compare canonical BF16 rounding as a
separate step. Report numerical differences, not an encoder-precision claim.
For `DeltaZ=Z7-Z6`, let `Q` lift each 4×4-grid average back over its original
4×4 raw-token block. Report `||Q DeltaZ||² / ||DeltaZ||²`, plus retained/lost
norms; mark zero denominators null. In exact arithmetic Q is an orthogonal
spatial-mean projection, so loss of feature-change energy is measured directly.
Report pre-quantization and BF16-rounding effects separately. This energy is
not semantic motion information or a bound on achievable prediction accuracy.
Only original endpoint raw tensors are available: do not claim raw-level
factorial object attribution from this cache.

**Pooled→reader structure.** CPU-only analysis may audit the frozen coefficients,
time-reversal relations and the expansion of the quadratic concat kernel into
single-frame and cross-frame bilinear terms. Reconstructed predictions must
match archived predictions before interpreting the terms. Oddness under frame
swap does not imply a difference of two single-frame functions: an
antisymmetric cross-frame bilinear term can remain. This structural audit
neither changes the reader nor localizes error to the encoder.

**RGB observability.** A fixed renderer-color centroid diagnostic may measure
two-frame target displacement on the same images, with target-color identity
explicitly labeled privileged diagnostic knowledge. Record raster error and
all failures without thresholds chosen from new results. Good RGB recovery
shows the image contains a recoverable cue under that rule; it does not prove
the encoder preserves it or the frozen reader can use it.

**Further correspondence test, not an executed promise of this round.** To
distinguish loss in spatial pooling from failure to read retained local signal,
a later fixed local-correspondence control can compare raw and pooled fields
from these same calls. It needs a predeclared common physical displacement
search grid/interpolation and source-object selection rule; supplying latent
masks makes it privileged. Raw 14-pixel and pooled 56-pixel token-center hard
nearest-neighbor matching are not equally resolved tests of 2–4-pixel motion.
Do not fit/select such a decoder on these inspected labels or declare encoder
failure when one arbitrary matcher fails. Any supervised raw-vs-pool reader
comparison needs matched FIT data, fixed training/selection and a separately
registered evaluation; the present frozen pooled W reader cannot directly
consume raw-dimensional inputs.

## Bounded execution, archive and stopping rules

Inherit the exact V5.4 final ledger: `4.522264341248407` charged GPUh. The new
outer envelope is at most **1.2 GPUh**, one actual eight-H800 Slurm step capped
at nine minutes including its startup/exit; maximum cumulative charge is
`5.722264341248407`, below the existing six-GPUh diagnostic boundary. The
original round remains 60 GPUh; no budget-cap increase or borrowing is needed.
CPU source checks, the fixed lane-reader fit, existing-model arithmetic and
reporting allocate no GPU.
Preflight from existing allocation evidence; do not spend an extra GPU step
on management queries or a new profile. The 4,096-slot runtime estimate is
planning evidence, not a measured result. Include raw/pool write time in the
single bound. No implicit retry or replacement sample set.

Use the existing authorized eight GPUs concurrently with unrelated jobs and
the existing resource checks. Create only a new independently identified V5.5
directory and step; on time/resource/identity failure terminate only that owned
new step, record actual elapsed charge, and preserve partial evidence. Poor
scientific scores do not cancel particular conditions or justify tuning.

Publish source/plan provenance before launch, then archive complete renderer
and precision receipts, all predictions and canonical/prequant pooled arrays,
retained raw endpoints, original-model references, complete new lane-reader
parameters and reference moments, logs, exact executed
commands, ledger and accounting. Archive derived attribution/control tables
and figures with the analysis code and parameter identities. Verify full GitHub
readback before any allowed cleanup decision. Because prior scientific failures
remain unresolved, retain investigation scenes and all existing content; this
diagnostic does not independently authorize deleting them.

The final report must distinguish: (a) measured object-intervention response of
the full chain, (b) verified numerical facts about pooling and reader structure,
and (c) encoder/pooling/reader explanations still requiring matched tests. A
mixed or negative result is an admissible complete diagnostic outcome.
