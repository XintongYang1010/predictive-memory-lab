# Independent V5.2 audit — posthoc, CPU only

The matched intervention identifies a large contribution of yellow-rendering
assignment to this **frozen encoder-plus-reader pipeline's error on the known
bridge scenes**. It does not establish where within that pipeline the dependence
arises, repair the moving-nuisance task, or demonstrate memory utility.

`recompute.py` imports no production scorer. It independently reconstructs losses
from saved predictions and computes each of 10,000 bootstrap replicates in an
explicit loop, carrying both variants and all conditions of each sampled world.
Seed 7810 and percentile endpoints 0.025/0.975 match the registered protocol.
The shared FIT-mean denominator is recomputed within each replicate for NMSE.
All 234 checked numeric fields/arrays, across three readers, agree with the
production report to a maximum absolute difference of **6.94e-18**. All 2,048
case-table rows match the prediction arrays exactly. Original predictions agree
with the archived bridge predictions exactly for all three readers. The four
reused single-frame comparisons are zero. Inputs were checked against the sealed
scene inventory and rehashed after the audit; no scientific source or input was
modified. These checks exercise stored outputs, not a new encoder run or fit.

## Intervention and swap response

For the frozen concat reader, original horizontal MSE was 0.00072967541. The
average of the two static **losses** was 0.00010485168, an **85.6304% reduction**.

| Registered contrast | MSE reduction | Paired 95% interval |
|---|---:|---:|
| Primary: original minus average static | 0.00062482373 | [0.00057371496, 0.00067697316] |
| Secondary: average(original, swap) minus average static | 0.00061554958 | [0.00056790083, 0.00066371923] |

The balanced contrast keeps the yellow-position marginal mixture the same in
each time slot. It supports a cross-time nuisance-assignment effect rather than
an explanation based only on one favorable fixed yellow location. The intervals
are conditional on the frozen reader and these inspected worlds; they do not
account for the posthoc selection of the nuisance hypothesis.

Of 256 worlds, 245 improve under average static loss and 11 worsen; both static
anchors improve in 243 and both worsen in 9. Both anchors remain in the result.
The nuisance-swap common prediction response aligns with original yellow dx in
236/256 A_dx and 248/256 B_dx worlds. The original-versus-swap common residual
changes sign in 229/256 and 230/256 respectively, with no zero-displacement or
zero-response cases. Through-origin response slopes are 0.0408359 and 0.0443932
in normalized target coordinates per normalized yellow displacement. Swap does
not improve error reliably: its original-minus-swap dx-MSE interval includes zero.
Changing a large erroneous response's sign can leave its squared error large.

## Remaining errors and cases

Mean motion skill is -1.53852 originally, 0.38982 at static_t6, 0.38104 at
static_t7, and -1.49281 after swap. The corresponding single-frame skills change
little (t6 approximately -0.121, t7 approximately 0.101). No reader is selected
from these results.

Static conditions remove the large dx residual correlation with original yellow
motion (A/B correlations range from -0.0086 to 0.0696). However, horizontal
direction errors remain 17.58%–19.92%; average absolute predicted dx is only
0.570–0.629 of the true magnitude. Residual scatter and all four coordinate
tables document this underestimation. The remaining signal is not fully decoded.

Whole worlds are ranked by the average of both variants, both static-anchor
losses, and all four motion coordinates; ties use ascending pair ID. The largest
remaining errors are **300, 209, 142, 355, 246**. All five have some position
outside the FIT bounding box and no cue reflection. This is descriptive: 248/256
worlds are outside that box, so those five do not identify a causal support effect.
Static mean-motion skill is 0.414 in 87 reflected worlds and 0.371 in 169
non-reflected worlds. Eight fully in-box worlds reach 0.655 versus 0.379 in 248
outside worlds, but the very small in-box stratum and correlated differences
prevent attributing this gap to position coverage.

The previously illustrated worst worlds 380/334/356 improve horizontal MSE by
89.34%/98.62%/89.59%, yet now rank 13/146/73 by remaining static motion error.
The new top-ten table and case predictions retain all conditions, without
replacing the old bad-case evidence.

## Is nuisance-balanced readout training meaningful next?

**Yes, as a new, bounded prospective test.** The intervention now provides a
specific mechanism to address: the frozen readout was fitted to static-nuisance
quartets and responds strongly to nuisance motion in the original task. A new
FIT-only readout could test whether supervision over balanced nuisance schedules
can suppress that response while retaining target displacement. Success is not
guaranteed: residual amplitude/support problems remain, and this intervention
does not localize the issue to the readout alone.

A candidate minimal construction uses new FIT target worlds shared by both
training arms, preserving the declared target-geometry and appearance distribution,
with six images per block: three target poses times two nuisance
positions. Four target views times four nuisance schedules (both static, both
directions) yield 16 views per block and 4,096 views for 256 base blocks. Reuse
the exact same image tensors; all nuisance variants share target labels. Group
all views of a base block together in any inner-FIT validation. Normalize loss
within each base world and then equally across worlds. The frozen encoder, pooling,
precision and label semantics remain fixed. Freeze the shared kernel/alpha, loss normalization,
number of blocks, nuisance sampling and computational limits before extraction.

To specifically attribute any training improvement to nuisance balancing, include
a matched static-only training arm using the same FIT image pool and both static
positions, with the same encoder, kernel, alpha and base-world-normalized loss.
If hyperparameter choice is necessary, prescribe a shared FIT-only selection rule
before data generation; independently choosing different alphas would mix the
nuisance-training effect with a regularization change. Repeating
its static examples can equalize nominal view counts but does not create new
independent samples. Compare prespecified arms without picking a fresh-Dev winner.
Keep concat and both single-frame readers in both arms. The old frozen reader is
a reference, not a substitute for that matched training control.

Fresh evaluation must use a separately frozen, untouched namespace or original
Dev pair IDs beyond all previously opened IDs (0–383 are unavailable for fresh
claims). Fit, select and serialize every coefficient/moment before release. Keep
the moving nuisance in the original-task evaluation. Existing bridge/confirm
data may be used for clearly exploratory checks only and must not enter FIT.
If a matched-quartet temporal check is retained, use fresh blocks: within a fixed
forward-nuisance schedule the old literal full-image reversal pairs are absent;
exact reversals exist across the opposite nuisance schedule and require an
explicit revised index mapping. A new result cannot rewrite either old failed
gate or authorize P0 without the separately registered prerequisites.

The sealed cumulative account is **2.2082544136 GPUh**. Of the 60 GPUh cap,
57.7917455864 remain; retaining 54 GPUh for a possible pilot leaves only
**3.7917455864 GPUh** for further preparation and diagnostics. Recheck the live
ledger before any authorized launch, set a conservative outer envelope, and
charge overhead and failures. The proposed 256-block FIT uses 1,536 unique
images; a fresh matched confirmation plus 256-world original bridge would bring
the corresponding total to 4,096 images, before any extra controls. This is a
count, not a measured runtime guarantee. A 4,096-by-4,096 float64 Gram matrix is
128 MiB and repeated CV solves should be profiled on CPU before a full fit. Its
1,536-by-20,480 float32 feature basis is 120 MiB before packaging, requiring
release-asset archival or a verified sharded format rather than one ordinary
GitHub file. No such data, fit, or experiment was created by this audit.

## Reproduction

Runtime: `/Library/Developer/CommandLineTools/usr/bin/python3` (Python 3.9.6).
From this repository root, reproduce the independent statistics in a new output
directory with the following command. The scene is the verified extraction of the
V5.2 release; the descriptor and reflection tables are the preserved local V5.2
bad-case and V5.1 posthoc outputs. The exact inputs and their SHA256 values are
recorded in `audit_summary.json`.

```sh
mkdir -p execution/visual-v5_2-20261010/independent_audit_reproduction
/Library/Developer/CommandLineTools/usr/bin/python3 \
  results/wm/visual-v5_2-20261010/independent_audit/recompute.py \
  --scene execution/visual-v5_2-20261010/diagnostic-20261010-v52 \
  --out execution/visual-v5_2-20261010/independent_audit_reproduction \
  --descriptors results/wm/visual-v5_2-20261010/exploration/whole_group_descriptors.csv \
  --reflections results/wm/visual-v5_1-20261010/diagnostics/bridge/reflection_flags.csv
```

`recompute.py` receives `--scene`, `--out`, `--descriptors`, and `--reflections`;
the latter two point to the existing bad-case world descriptors and V5.1
reflection flags. See `audit.log`, `audit_summary.json`, and
`metric_comparison.csv`. `static_residuals.py` uses these saved audit outputs and
the sibling scene, producing descriptive tables and `static_remaining_predictions.png`.
Output creation is exclusive to preserve earlier audit attempts. Test remained
unopened; no GPU, model fitting, winner selection, or P0 launch occurred.
