# V5.7: locate correspondence errors before changing the learned memory

This is a registered extension on already inspected Dev640–895: 256 worlds,
two variants, two objects per episode. Both experiments below are posthoc
mechanistic diagnostics, not held-out confirmation. Original Test remains
unopened. No encoder, reader or writer is trained; no P0 gate changes. All arms,
including failed or worse arms, are reported. Source and exact design JSONs must
be published to GitHub and read back before scientific scoring.

## Inputs and execution

Reuse the fully archived V5.5 raw I-JEPA descriptors and V5.6 complete matching
costs. The V5.5 scene manifest SHA256 is
`b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663`.
The V5.6 `costs_and_predictions.npz` SHA256 is
`265766a0d1de852f5eb863c068d08de11c696ed961e76768468797a6e9879eb0`.
Check every opened source artifact against its authenticated inventory. Preserve
the originals. All new output goes under `execution/visual-v5_7-20261010/`.
CPU execution uses at most four BLAS threads per process. No remote directory or
Slurm step is needed; this round adds zero GPUh to the inherited 5.56779064169697.
The six-GPUh diagnostic reserve and original 60-GPUh total remain unchanged.

Prior V5.5 batch/quantization controls passed exact parity; this experiment does
not reopen those resolved implementation questions. Pretrained weights and old
checkpoints are unchanged and remain in their prior verified releases.

## A. Intervene on the candidate competition

Use all six frozen V5.6 banks and every saved 25-by-25 physical displacement cost
surface. Compare five fixed readouts: original minimum over both axes; minimum
over x on the row nearest ground-truth y; minimum over x on the zero-y row;
minimum over y on the column nearest ground-truth x; and minimum over x on a
mismatched ground-truth-y row. The last control takes y from world index
`(index + 37) % 256` with identical variant and object, shared by all six methods.
This fixed derangement preserves the y marginal and 25-candidate count. Compare
it against the true oracle row to separate correct y information from mere
search restriction. No permutation is selected by its result.
Different donor worlds can have the same quantized y row. Report that frequency
and the oracle-versus-donor comparison on the different-row subset; do not
describe every donor query as an incorrect row.
Oracle row/column choices
are privileged diagnostic interventions, not prediction algorithms. No costs
are refitted or rescaled. Replay original predictions with the original tie rule.

Report full predictions; true-nearest-grid global and correct-row cost ranks,
ties and margin; geometric
grid-quantization error; out-of-support cases; x/y error; and sign rescue/harm at
the previously fixed |true displacement| >= 1 pixel threshold. Wrong nonzero
sign and zero predictions are separate. A constrained row can itself create
errors, so report both rescue and harm. It cannot identify an additive fraction
of errors caused by any neural module. The zero-y row is a counterfactual
restriction, not a ground-truth-free replacement for modeling real y motion.

## B. Factor descriptor origin, metric and spatial resolution

Keep the source color ROI (a privileged 3-by-3 query grid), candidate grid,
interpolation, raw inputs and point aggregation fixed. Native raw16-by-16 and
derived pool4-by-4 are separate factors. At each resolution compare:

1. No spatial-mean subtraction.
2. Subtract the source-frame global channel vector from both frames.
3. Subtract each frame's own global channel vector.

Cross these with centered cosine distance and mean squared descriptor distance,
for twelve arms. Per-descriptor channel centering is common to all arms. The
mean squared distance does not unit-normalize descriptors: average squared
channel differences over 1280 channels and nine points, so it is not merely a
rescaling of cosine distance. Bootstrap seed for this experiment is 7818.
The global channel vectors come from all 256 raw tokens in FP64; the same vectors
are used for both resolutions. Native pooling retains V5.6's FP64 accumulation
and FP32 storage convention before the origin intervention. No thresholds or
coefficients are fitted on Dev.

The shared-source subtraction leaves temporal descriptor differences unchanged
but changes the angular origin used by cosine. Per-frame subtraction additionally
removes the global temporal change. Neither vector is assumed to be pure
background. Mean squared distance must be invariant to shared-source subtraction;
this is a mandatory numerical negative control. Original raw/pool cosine costs
and predictions must reproduce the archived baseline. Record all candidate costs,
norm/clamping diagnostics, finite checks and any tie-sensitive discrepancy.
Full-raw access is used here; this is not an equal-memory-budget comparison or
an implemented learned selector.

## Common analysis and interpretation

Use 10,000 whole-world paired bootstrap samples with fixed seed 7817 for A and a
separately fixed seed in B's design. Both variants and both objects stay together.
Conditional exploratory intervals are not multiplicity-adjusted hypothesis
tests. Display all arms rather than selecting a winner for confirmation.
Revisit the previously named cases 761/1, 844/1 and 891/1; keep both objects.
No new bad-case selection is made from these results.

The combination of A and B may narrow the problem to candidate competition,
descriptor geometry or remaining spatial/matching limitations. It cannot prove
that I-JEPA contains no motion information, that an encoder module is defective,
or that original learned memory is effective. Any promising intervention needs
FIT-only training/calibration and a new independent confirmation before a
generalization or deployment claim.

## Publication and recovery

Record exact source/config hashes, commands, environment, software checks,
complete predictions/costs, per-case figures and an honest interpretation.
Publish readable results to a new branch and larger artifacts to a Release.
Download the complete new archive independently and verify every member before
claiming archival completion. Local originals stay available. No remote cleanup
is performed in this CPU-only round because no new remote root is created.
Execution or archival failure preserves partial outputs and diagnostic logs.
