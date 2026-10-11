# V5.8: separate spatial averaging from coarse sampling and reconstruction

This is a registered mechanistic diagnostic on the already inspected Dev640–895:
256 worlds, two variants and two objects, totaling 1,024 object queries. It is
exploratory, not an independent confirmation. Original Test remains unopened.
No encoder, learned reader or memory writer is trained. No gate, threshold,
sample inventory or budget is changed. Publish source and the exact design JSON
to GitHub and independently read them back before scientific scoring.

## Fixed inputs and execution

Use the authenticated V5.5 raw I-JEPA cache and the V5.6 matching baseline.
The scene manifest SHA256 is
`b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663`;
the baseline cost NPZ SHA256 is
`265766a0d1de852f5eb863c068d08de11c696ed961e76768468797a6e9879eb0`.
Check every opened scientific input against its saved hash. Preserve originals.
Reconstruct and authenticate the same source RGB images and privileged nine-point
source ROIs. Keep all 625 displacement candidates (both axes -6 to 6 pixels in
0.5-pixel increments), channel-centered cosine distance, point aggregation and
the original tie rule fixed. Labels only score or rank predictions; they do not
select descriptors, ROIs, offsets or predictions.

Outputs go to `execution/visual-v5_8-20261010/`. CPU processes use at most four
BLAS threads. Reusing the frozen cache needs no Slurm step, GPU time or new remote
directory. The inherited 5.56779064169697 diagnostic GPUh remains unchanged.
Previously verified batch/quantization parity is not an unresolved confound.

## The two-by-two intervention and phase-matched dense control

The raw 16-by-16 token grid has physical centers `6.5 + 14*i` on each axis.
Do not stretch an intermediate grid to fill the image. Use its actual origin
and spacing when calculating bilinear interpolation weights.

| Arm | Explicit four-by-four averaging B | Coarse sampling/reconstruction D | Grid centers |
| --- | --- | --- | --- |
| R | No | No | 16: `6.5 + 14*i` |
| BR | Valid moving mean, stride one | No | 13: `27.5 + 14*j` |
| DR | No | Yes | 4: `27.5 + 56*k` |
| DBR | Valid moving mean, stride one | Yes | 4: `27.5 + 56*k` |
| H | Two-by-two half-grid interpolation average | No coarse sampling | 13: `27.5 + 14*j` |

B averages each four-by-four window with FP64 accumulation and FP32 storage.
D samples the corresponding dense bilinear field at the four old-pool centers,
stores FP32 values, then uses the same bilinear reconstruction for all queries.
For BR these are exact indices 0, 4, 8 and 12; DBR must reproduce the old pool4
bank, complete costs and predictions. R must also replay the raw baseline.

For R, D samples halfway between raw token centers in both axes. DR therefore
includes a two-by-two interpolation average. It omits the explicit four-by-four
box filter but is **not a completely unfiltered point-decimation intervention**.
D also combines sampling density and reconstruction; it cannot by itself
separate those two operations. B's even-sized kernel changes grid phase as well
as averaging. These are precise interventions on cached descriptor fields, not
new encoder forward passes or proofs about encoder translation equivariance.

A pre-scoring external design review identified a useful fifth arm, H: sample R
at the same 13-by-13 centers as BR with FP64 interpolation and FP32 storage, then
query that native dense grid directly. H is the two-by-two half-grid average;
its rows/columns 0, 4, 8 and 12 must exactly equal DR. BR-H therefore holds grid
phase, density and reconstruction fixed while increasing averaging support from
two-by-two to four-by-four. DR-H holds that dense field fixed while reducing its
sampling density and reconstructing. H/BR/DR/DBR form a phase-matched factorial;
R remains the original baseline. H-R still includes resampling and implicit
averaging, so it is not a pure grid-phase effect. This addition is frozen before
reading any new scientific score, not selected after observing results.

All five fields use border replication outside their physical support. Save
source and candidate-target clamp flags. Report results on all 1,024 queries
and on the fixed common interior subset: for every arm (including H), all source ROI points
and all ROI points for all 625 candidates must be unclamped. Select that subset
from geometry alone, before inspecting errors. Report its world/episode/object
counts; it changes the population and cannot stand for an unbiased full-set
result. Plot and report the original named worlds 761/1, 844/1 and 891/1, both
objects, without selecting new examples by the new outcome.

## Analysis and checks

Save complete cost surfaces and predictions, geometry/clamp diagnostics,
ground-truth-nearest candidate ranks globally and within the correct-y row,
and finite/norm summaries. Distinguish wrong nonzero horizontal signs from zero
predictions using the frozen |true dx| >= 1 pixel criterion. Report x and y error,
four-coordinate error/skill and per-query outcomes; no training checkpoint is
created because nothing is fitted.

For each population report the following paired squared-error differences:
BR-R (B at dense resolution), DR-R (D without explicit B), DBR-BR (D after B),
DBR-DR (B at coarse resolution), and DBR-BR-DR+R (original interaction).
Also report H-R, BR-H, DR-H and DBR-BR-DR+H for the phase-matched control.
Only BR-H and DBR-DR hold grid phase/density fixed while changing averaging
support. Prefer those matched comparisons when interpreting averaging range.
Use 10,000 whole-world paired bootstrap replicates with seed 7819, preserving
both variants and both objects. For the subset use per-world loss sums and
valid-query counts, so unequal valid counts do not change the query-weighted
estimand. Report intervals and every arm, not only favorable contrasts. An
interaction is not an additive percentage of responsibility of a neural module.
Intervals are conditional exploratory uncertainty estimates, not multiplicity-
adjusted hypothesis tests.

Before production, independently test physical coordinates, constant and affine
fields in the common interior, DR's half-grid average, DBR's exact pooling
identity, H-to-DR exact subsampling, cosine distances and tie handling. After production independently
recompute predictions, key statistics and paired intervals, and verify both
baseline replays. Numerical warnings and failures remain in the logs; do not
silently suppress them or relax scientific thresholds.

The experiment can show which concrete field operation is sufficient to
degrade this matcher under fixed inputs. It cannot identify an encoder defect,
irreversible loss of all useful information, success of a learned reader, or
effectiveness of the original memory. A promising fix still requires FIT-only
development and a new independent confirmation set.

## Publication and recovery

Save source/config/input hashes, commands, environment, tests, all costs,
predictions, tables, figures and limitations. Finish independent scientific and
prose review before freezing the complete archive. Publish readable results on
the experiment branch and the complete new output to a GitHub Release. Download
it independently and verify the inventory and every member hash. Retain local
originals. No remote cleanup applies because this round creates no remote root.
If execution or archival fails, preserve the scene and logs and report failure.
