# V5.4: one matched spatial-coverage confirmation

Design recorded on 2026-10-10. The machine-readable authority is
[`scripts/v5_4_plan.json`](../scripts/v5_4_plan.json). This records a bounded
experiment, not an executed result. Complete software checks and publish/read
back the full source, plan and this protocol before generating new scientific
W FIT data. Freeze and publish/read back both complete fitted models before
generating or inspecting any new evaluation data.

V5.3's balanced reader improved fresh-original transfer while leaving substantial
motion error. Its posthoc analysis found greater error outside the narrow
training coordinate envelope. That is an exploratory lead, not a causal finding
about spatial coverage. V5.4 tests one locked intervention: widen supervised FIT
target-center coverage while holding the latent world draws, reader family,
sample weight and shared kernel geometry fixed. The primary endpoint is a new
original-generator bridge. Original V5 and V5.1 failed gates remain unchanged;
no Test, P0, memory training or automatic continuation is authorized by this plan.

## Exactly matched N/W FIT worlds

Use schema `v5.4-spatial-support-1`. N is the existing complete V5.3 narrow FIT
cache, with 256 blocks and six unique image slots per block. Verify its exact
cache SHA256 `107fa44e56c3152ab08ec02413d4a1b0e051179cebf4ea8b851efbf67f513833`
and original receipt SHA256
`18f52d9e6ecca5be3872a916d87dfcc67b789ff2062e84a9c9cab84aa51e17e6`,
plus the source, encoder, configuration, array identities, block IDs, view order
and actual-input inventory. Keep the original receipt as V5.3 provenance and
record its reuse explicitly; do not rewrite an old cache as a new V5.4 product.
N features are read-only and require no new encoding.
The sealed V5.3 root `archive_manifest.json` is bound by SHA256
`bfa3006ad1af90869deab7fa24d69b309bfaf76ca9b885532373145acb95ea13`
(1,799 files, 465,241,564 bytes). Its creation is not publication verification:
the executor must finish full V5.3 archive upload and independent readback
before releasing any V5.4 science.

W deliberately replays each original `v5.3-nuisance-repair-1/fit/block` seed,
blocks 0–255, using the unchanged first-eight-SHA256-bytes/little-endian
`default_rng` algorithm and draw order. Replace only the bounds of the first
center uniform draw. Use the same **pre-rounding uniform quantiles**; never
recover quantiles by inverting already rounded N centers. The displacement
magnitudes/signs, object colors and two yellow positions must remain exactly
matched, as must the rounded displacement and all motion target labels.
Validate that correspondence before fitting. Tests may generate only paired
`v5.3-nuisance-repair-1/software_test/block` fixtures, never scientific FIT.

| Coordinate | Narrow center | Wide center |
|---|---|---|
| A_x, B_x | [.20,.80) | [.118,.882) |
| A_y | [.20,.35) | [.168,.382) |
| B_y | [.65,.80) | [.618,.832) |

Keep displacement magnitudes uniform [.008,.018), independent signs, and round
center and displacement to multiples of `2^-20`. Keep the original rendering,
colors, grid, glyphs, yellow a/b distribution and pixel rounding. A/b may map to
the same pixel; retain such blocks and report the count, without rejection or
reseeding. The wide center bounds ensure C−d,C,C+d stay within legal target
lanes without reflections. They nominally extend the coordinate envelope to
the original legal lane boundaries, but rounding leaves small gaps: x spans
`[.10000038146972656,.8999996185302734]`, A_y spans
`[.15000057220458984,.39999961853027344]`, and B_y spans
`[.6000003814697266,.8499994277954102]`. Do not call this exact pointwise or
complete joint-distribution coverage; the original bridge still includes
reflections, correlated trajectories, and different nuisance trajectories.
At fixed 256 worlds, widening the range also lowers spatial sample density and
changes target-to-yellow relative geometry and raster context. These are part
of the registered coverage intervention, not independently held factors. A
negative result therefore cannot exclude every possible coverage remedy.

Use the unchanged six-image pose-major order
`p0a,p0b,p1a,p1b,p2a,p2b`, with poses `C-d,C,C+d`. Four target views are pose
pairs `(0,1),(2,1),(1,2),(1,0)`, crossed with yellow paths `aa,bb,ab,ba` in
target-major/path-minor order. Both N and W use **all 16 views/world**, 4,096
rows each. Each world has total empirical loss weight `1/256`; views are
uniform within a world. Position targets move with the center; motion targets
are exactly paired. The eight response columns are separate ridge fits:
position-target scale does not couple into the fitted motion columns.

Both arms keep the frozen official I-JEPA checkpoint and 4×4 pooled features.
Encoder/checkpoint identities are in the plan. The existing BF16 CUDA path,
FP32 pooling, pooled BF16 quantization and FP32 storage remain unchanged. This
does not test a higher-precision encoder, different pooling, resolution or model.
Encode 1,536 new W images; reuse the 1,536 N images' exact cached features.

## Refit two readers in one shared kernel geometry

Fit **concat only** for N and W, both quadratic kernel and fixed mean-loss alpha
`7.8125e-7`, with no CV, candidates, regularization selection or output scaling.
Refit N as well as W; an old V5.3 B model cannot serve as the control after
changing the common normalization reference. Old and new N predictions need
not agree because that reference changes by design.

The common unlabeled reference is the 8,192 FIT views: all N rows in block/view
order, then all W rows in the same order, equally weighted. Center and
trace-normalize the linear kernel on this joint reference, apply `(1+k)^2`,
then center and trace-normalize the quadratic RKHS kernel on the same reference.
Freeze these moments. Each arm uses its 4,096-row submatrix; do not re-center
or re-normalize either arm. Both query-centering passes use all 8,192 reference
columns before selecting an arm's basis columns. No evaluation input or label
may enter these moments.

Fit an unpenalized intercept for each output in each arm. In FP64 solve
`[[K_arm+4096*alpha*I,1],[1^T,0]] [coef;intercept] = [y_arm;0]`, or an
algebraically equivalent Schur system. Save coefficients and fitted intercepts.
The diagonal penalty is **4096 × alpha = .0032 in each arm**, not 8192 × alpha;
the 8,192-row reference controls normalization, not either arm's loss count.
The common scoring baseline is the mean of all joint N+W FIT target rows.
Position means may differ between N and W; do not force their fitted intercepts
or target means to match. Motion labels and their RMS are identical, which
gives a common nonzero FIT normalizer for negative controls.

This tests **supervised spatial coverage under shared unlabeled N+W kernel
normalization**. N also encounters the wide feature distribution through those
unlabeled moments, so it is not an untouched narrow-only pipeline. It is a
controlled comparison within this family, not the total effect of giving a
system any wide data. The old S/B single-frame equivalence proof does not apply
to N/W, whose feature/label distributions differ; do not import its `1e-7`
cross-arm equality gate. This registration fits no t6-only or t7-only models.

Use software-only fixtures to check common-reference/query centering, both
arm index maps, equal world weighting, paired motion labels and the free
intercept solve against an explicit polynomial feature or augmented-system
reference. Check fixed source/shape/finite contracts before scientific release.
Numerical tolerances belong to these software tests, not to observed science
scores, and may not be relaxed after FIT or evaluation access.

## Evaluation release and freshness

Seal V5.3 first. New W FIT, new V5.4 anchors and Dev640–895 must not have been
generated during preparation. Reading the existing N FIT provenance and old
results is permitted and is disclosed: this design is an adaptation motivated
by old observed evidence. N/W FIT seeds are intentionally paired old seeds,
not a claim of new independent FIT worlds.

After frozen implementation/source publication and verified readback, extract
W FIT once. Fit both readers on CPU and freeze the entire joint feature basis,
row mappings, both normalization levels, coefficients/intercepts, common score
baseline, nonzero FIT motion RMS, matching audit and complete source/cache
identities. Publish the complete model/basis, not only a summary hash, and
independently verify its GitHub readback before releasing either evaluation.

There are two held-out evaluations, both locked before generation:

1. **Fresh wide anchor:** 256 blocks, schema
   `v5.4-spatial-support-1/anchor/block`, 64-bit SHA seed algorithm unchanged.
   Same wide geometry and all 16 target/nuisance views; 1,536 unique images.
   This is a support/anti-collapse test on the intervention distribution, so
   an anchor advantage alone does not establish original-distribution transfer.
2. **Fresh original bridge:** original schema `v5-rgb-pairs-1`, split `dev`,
   pairs **640–895**, both original variants, t6/t7 cues: 256 pairs / 512
   episodes / 1,024 images. Keep original target/nuisance motion, reflections
   and observed-displacement labels. Never substitute the wide generator for
   these scenes or filter hard cases after seeing scores.

Extract both evaluation roles in a **single eight-GPU step**, 2,560 images
total, using a fixed complete worklist. Each GPU handles an even share of
both roles (32 anchor blocks × 6 plus 32 bridge pairs × 4 = 320 images).
One worker loads the official encoder once and reuses it across roles. The
stage has one shared wall-clock lease, not a reset budget for each role.
Then evaluate both roles on CPU with frozen models. Do not inspect anchor
scores to decide whether to finish the bridge. Poor metrics do not stop or
change the registered comparison; identity/numerical/resource failures do.

Negative controls reuse wide-anchor center images `[2,3]` and `[3,2]`, with
stationary targets and moving yellow. They require no extra encoding and never
enter supervised FIT. Square each direction's prediction before averaging;
signed cancellation would understate response to the distractor.

The read-only local freshness checks found exact V5.3 bridge IDs 384–639 in
the archived predictions, SHA256
`fc2ce9a5cab09c03b25ccfb3355bdd8bea512aecb1bd0a2b9be832adfec3929b`.
Earlier known original Dev probes/bridges occupy 0–383. A **seed-hash-only**
audit found 256 unique new 32-bit Dev640–895 seeds, no collision with Dev0–639
or planned Train0–1023, and 256 unique new V5.4 anchor seeds with no collision
with old V5.3 or V5.1 fit/anchor/confirm/software role seeds 0–255. This audit
made no scientific RNG draws and rendered no new data. Reusing the V5.3
software namespace is deliberate for matched tests, not scientific access.

These facts cover known local artifacts and reviewed sources; a new namespace
alone does not prove nobody accessed it elsewhere. Before release the executor
must inspect known remote logs/directories and archive path/hash evidence plus
a no-access attestation. Discovery of earlier access stops this registration;
do not silently pick another range, seed or sample count.

## One primary endpoint and fixed supporting evidence

The only primary endpoint is fresh-bridge concat error. For each complete pair,
average squared motion error over both variants and the two horizontal target
coordinates A_dx/B_dx. Report `Delta = mean_pair(L_N − L_W)`, in normalized
coordinate squared units; positive favors wide coverage. Bootstrap 256 whole
pairs, preserving both arms, variants and coordinates, 10,000 times using
`default_rng(7811)`, with two-sided percentile 95% endpoints. Reusing this
fixed bootstrap seed is intentional and does not reuse evaluation samples.
The interval is conditional on these fitted models, not variability across
repeated FIT draws. One interval lower bound above zero supports relative
horizontal-error improvement within the registered design, not an omnibus
repair, memory result or P0 permission.

Always report both arms' eight coordinate MSE/skills against the common FIT
mean, four-motion mean skill/NMSE, position skill, direction errors with
zero/tie counts, signed bias/covariance, amplitude and SD/RMS ratios. A positive
relative primary can arise by shrinking bad predictions. Check absolute W
coordinate skills together with each wide-anchor path's nonzero target
contrasts and negative-control RMS; retain mixed outcomes rather than choosing
a favorable endpoint. Do not invent a new combined pass score after evaluation.

For every anchor nuisance path independently, compute end contrast
`(view0−view1)/2` and start contrast `(view2−view3)/2` against known nonzero d.
Report each coordinate, absolute motion skills and static/moving aggregates.
Compute contrasts/losses separately per path before averaging. Report negative
RMS per direction, per motion coordinate and overall, both in coordinate units
and divided by the matched nonzero FIT motion RMS. Low negative RMS alone
cannot establish useful motion preservation.

The spatial-support diagnostic is fixed in advance from the **narrow** schema's
rounded coordinate envelope, not estimated from new scores. Lower bounds in
A_x,A_y,B_x,B_y order are
`[.18200016021728516,.18200016021728516,.18200016021728516,.6319999694824219]`;
upper bounds are
`[.8179998397827148,.3680000305175781,.8179998397827148,.8179998397827148]`.
Convert archived FP32 labels to FP64; reconstruct `p7=position`, `p6=p7-motion`.
Use strict comparison with these bounds and tolerance zero. This is explicitly
a label-quantization approximation of original continuous cue positions.
A pair is outside if any coordinate at either time in either variant is
outside. Also report coordinate-level groups by whether that coordinate of
that whole world is outside at either time in either variant; keep both
variants together and report both world and example counts. Report N/W dx/motion MSE
and per-axis/mean motion skills for pair groups, and per-coordinate MSE/skill
for coordinate groups.

Apply these same bounds and reconstruction to the **new wide anchor**, without
reading any old anchor: a block is inside only if all coordinates in every one
of its 16 views and both cue slots are inside. For each individual coordinate,
the coordinate-level block is inside if that coordinate is inside for every
view/slot. Report inside/outside counts and the same fixed N/W MSE/skill
statistics, averaging views within worlds and preserving equal world weights.
Each skill uses its conditional group's squared target-minus-frozen-joint-FIT-
mean denominator; empty groups or zero denominators are `null`, not a reason
to alter a threshold. These are descriptive point statistics with no subgroup
confidence intervals, new gates or causal claims. Inside means only
coordinatewise-envelope membership, not full joint support or actual empirical
FIT coverage. Wide-anchor overall success alone does not prove narrow-region
ability was preserved; report this predeclared narrow-envelope subset and say
when its sample count is too small to support a firm conclusion.

Supporting aggregate motion, anchor contrast and negative-RMS paired/block
intervals may use the same fixed bootstrap seed/count and are descriptive,
unadjusted secondary intervals. Direction/amplitude/bias/covariance and spatial
subgroups are point diagnostics. Correlated endpoints are not replications.
Report all registered outputs regardless of signs; do not refit, select, scale,
direction-correct, extend the samples or alter old gates based on the result.

## Two GPU stages, strict inherited budget, and retention

Inherit the exact reconciled charge **3.573925619125368 GPUh** and original
ledger hash. The round cap remains 60 GPUh, with **54 GPUh reserved**, hence
the diagnostic ceiling is 6 GPUh. V5.4 has at most **2.4 additional GPUh**:
two eight-H800 GPU stages, each with a six-minute Slurm limit, `.09` hour inner
lease and `1.2 GPUh` controller envelope `(6/60+.05)*8`. Stage one encodes W
FIT (1,536 images); stage two encodes both evaluations (2,560 images).
The worst reserved cumulative charge is **5.973925619125368 GPUh**.

There is no extra GPU profile, management/query step or automatic retry. The
remaining `.026074380874632 GPUh` below six is not another scientific stage.
CPU-only resource queries, fitting, scoring, plotting and archival request
zero GPUs. Existing profile/history informs feasibility but does not guarantee
that 320 images/device plus model startup fit the time limit. One persistent
encoder per worker is necessary; restarting eight encoders between evaluation
roles would consume a different execution envelope. If the stage cannot fit,
stop and retain its partial outputs without extending time or resetting leases.

Before each stage require no unresolved active envelope, reconciled accounting,
and enough room for the full next envelope under both the version and cumulative
limits. Include startup/shutdown, failed attempts and Slurm/accounting variance
in attributable charge; do not bill the whole shared parent allocation to this
run. Revalidate ownership and binding of the authorized eight H800 devices and
existing workload constraints, never terminate existing IndexMem work. Use
CPU-only reads of existing resource/accounting state to avoid an extra GPU job.

Archive old N provenance unchanged, W matching evidence, source/plan/protocol,
all cache/input checksums, both complete newly fitted models/reference basis,
both publication/readback receipts, actual job IDs/logs/accounting, every
registered prediction/target/metric/plot and final aggregate ledger. Re-read
and verify GitHub archive contents. Preserve all old sealed/failed scenes;
preserve this scene on execution/archive failure or unresolved science. This
diagnostic does not itself authorize cleanup or P0.
