# V5.1 execution: temporal confirmation before P0

Registered on 2026-10-10 UTC, after the failed V5 feature gate and the independent
CPU diagnostics. The user has authorized continuation on the existing eight H800
GPUs, including the previously authorized sharing arrangement. This document
specifies how to execute the new prerequisite and, only if it passes, continue
the original downstream pilot. It does not record a successful confirmation.

## Authority and immutable evidence

- Original scientific source: `d0b4fef25e98eefb58bfba737997cb0145f27daa`.
- Original failed run: `d4b6fc622d0fe08e28605aaf6e48100201ab2008`; feature step
  `415453.15`, after profile step `415453.14`.
- Diagnostic evidence: `ca365e1aa1026c51be0d23008b63650354c5bf31` and subsequent
  explicitly identified verification commits.
- Designer review: `b13c9877339779b8c770d175a483ea3489d2420c`, including
  [the confirmation plan](../scripts/v5_temporal_confirmation_plan.json),
  [CPU primitives](../scripts/v5_temporal_confirmation.py), and
  [scientific reasoning](V5_PROBE_REVIEW_20261010.md).
- Original P0 and memory protocol: [VISUAL_MEMORY_V5.md](VISUAL_MEMORY_V5.md).

The original pooled displacement score remains negative and its gate remains
failed. Preserve that source, run directory, archive and hashes. Use a new run
directory, a new source commit and a distinct V5.1 prerequisite receipt. Do not
write a successful `feature_gate.json` into either the old directory or a copy
that is presented as the original V5 experiment.

Freeze the complete execution implementation before generating scientific FIT
data. Its identity must include the new extractor, reader, controller, protocol
JSON and this document, as well as imported generator/encoder/training code.
The old `source_identity()` glob alone does not cover every new script name.
Record exact paths and file SHA256 values, the Git commit and environment.

## Resource and cumulative-budget contract

The historical allocation was parent job `415453`, node `dgx-097`, account
`hansirui`, partition/QOS `llm`, eight H800 80 GB devices. These are historical
facts, not permission to invent a replacement Slurm environment. Re-query the
live allocation, owner, partition/QOS, GPU UUIDs, processes and free VRAM before
launching each new step. Keep the existing IndexMem processes and all unrelated
directories intact. Each worker uses exactly one validated GPU. Eight workers
may run feature extraction; eight independent candidate jobs may run in parallel.

The original cap remains **60 cumulative attributable GPUh**, including both
versions and failed attempts. Charge selected GPU count times wall time while
the step owns those devices, including encoder loading, idle time and shutdown.
The pre-existing parent's reservation and other experiments are reported
separately. A new output directory does not reset this cap. CPU fitting between
extraction phases should release its GPU step.

The archived original ledger records `0.39161167637833216` GPUh: profile
`0.007657659230495106`, features `0.38395401714783617`. This leaves
`59.60838832362167` GPUh before new work, subject to accounting reconciliation
of any additional or still-open task leases. Keep a single transactional round
ledger, or import the immutable earlier charge into a new aggregate ledger with
the old ledger hash and explicit provenance. Never charge an old lease twice.
An interrupted lease stays fully charged until actual Slurm accounting permits
reconciliation; do not infer zero usage from a missing completion receipt.

Original stage ceilings are not expected runtime:

| Stage | GPUh ceiling | Parallelism |
|---|---:|---|
| Original profile | 0.5 | one selected GPU |
| Original probe/full feature cache | 8 | eight workers permitted |
| Small fit, LR trials, P0 and history gate | 12 | one selected GPU |
| A0/A1/B0/B1/C0/C1 | 4 each, 24 total | independent arms |
| full/recent/fifo references | 2 each, 6 total | independent arms |
| Fixed-writer reader replay and reporting | 4 | one selected GPU |

Those original ceilings total 54.5 GPUh. The new prerequisite consumes part of
the same 60 GPUh, so reserve it explicitly before planning downstream leases.
Preserving all 54 GPUh of the original post-profile ceilings would leave only
5.60838832362167 GPUh for additional prerequisite work after the observed old
charge. Actual refunds can free headroom; they cannot be assumed in advance of
closed receipts. Preserve original phase ceilings and resume charges rather
than silently granting another full phase allowance. A budget stop is a partial
result, not a failed scientific hypothesis or permission to enlarge the cap.

## Release order and fresh-data audit

The control flow is sequential even when each extraction uses eight GPUs:

1. Verify source, encoder and resource identity; archive the freshness audit.
2. Generate/encode FIT only, role `fit`, blocks 0–255.
3. Perform FIT-only grouped CV for the registered readers, fit the selected
   reader, and freeze its coefficients, moments, scores and hash receipts.
4. After that freeze, release role `confirm`, blocks 0–255, exactly once.
5. Apply all confirmation checks. A failed prerequisite stops downstream work.
6. If confirmation passes, release and evaluate the registered original-task
   bridge, `dev` pair IDs 128–383, both variants, without refitting the reader.
7. Only after confirmation and bridge both pass, extract complete original-task
   Train/Dev features for P0 and proceed through the original downstream gates.

There is no Test access in any of these stages. Generation, image previews,
encoding and metric evaluation all count as access; do not pre-generate fresh
confirmation images while FIT selection is still in progress. Do not inspect
failed confirmation data to choose a replacement candidate under the same plan.

The pre-execution audit of local archived evidence found:

- The original eight worker logs and probe manifests cover Dev pairs 0–63 only,
  exactly 128 records. The verified local Dev cache filenames are
  `000000-0.pt`/`000000-1.pt` through `000063-0.pt`/`000063-1.pt`.
- Original feature diagnostic predictions, new input-audit predictions, core
  probe predictions and matched temporal-control predictions all cover the same
  Dev pair range 0–63. No recorded bridge IDs occur in those outputs.
- The independent archive-audit script uses `specs(split, 128)`. Software tests
  use the separate `software_test`/smoke namespaces or low-numbered Dev records;
  no reviewed test accesses the registered bridge range.
- The original full Dev plan is 256 **episodes**, i.e. pairs 0–127, and that full
  cache was not started. Thus the bridge is also disjoint from future P0 Dev.
- A seed-hash-only check, without generating images, found 256 distinct bridge
  seeds and no collision with planned Train pairs 0–1023, original Dev pairs
  0–127 or profile pairs 0–2, under the frozen 32-bit seed scheme.

This is an evidence-based attestation over the archived run, diagnostic outputs,
verified local caches and reviewed scripts/tests; it is not a claim to have
searched unknown external experiments. The executor must record the inspected
paths/hashes and check any newly discovered local or remote access records before
bridge release. If prior access is discovered, stop and register the resolution
before evaluating any new bridge outcome.

## Confirmation implementation requirements

Use the registered pooled I-JEPA representation and checkpoint. For each quartet,
encode its three unique images once, then reuse the exact same tensors through
`ORDERS = [[0,1],[2,1],[1,2],[1,0]]`. Repeated-center input is `[1,1]`. Assert
bitwise equality of each anchor tensor, not just RGB equality. Check complete,
unique role/block IDs, tensor shape/dtype/finiteness, all four view labels and
the exact file hash inventory. Block IDs, positions, displacements, role and
view labels are analysis metadata; they cannot enter the predictor.

Both primary kernels, all six mean-loss penalties, grouped four-fold splits,
selection objective and tie break remain exactly as registered. Each fold fits
its own centering/scaling. Retain every candidate score and both single-frame
controls; do not choose by confirmation. `K + n_fit * alpha * I` is the new
mean-loss convention; do not substitute the old fixed diagonal lambda.

The primary reader must satisfy all nine registered lower-bound checks,
position skill >0.1, numerical reversal/repeated-frame checks and provenance
requirements. The two sets of anchor contrasts are mathematically redundant for
an exactly antisymmetric fitted reader: if view3 = −view0 and view2 = −view1,
then the end/start contrasts coincide. They are not eight independent pieces of
evidence. Keeping the registered nine-endpoint Bonferroni calculation is
conservative; do not change it after observing data. Reversal consistency alone
does not show accurate use of motion evidence.

`score_quartets()` explicitly returns `status="metric_checks_only"` and
`p0_authorized=false`. Its `metric_conditions_met` may be true for the eight
software-test blocks. A production prerequisite must additionally enforce all
256 unique unseen confirmation blocks, frozen FIT/model identities, input audit
and bridge success. Never treat the helper's metric boolean as the total gate.

The bridge retains the frozen original generator, moving distractor and
reflections. Extract actual t6/t7 cues and evaluate the same frozen reader. Use
the FIT position/motion baseline specified by the plan, the complete two-variant
world as bootstrap unit, all four displacement point scores >0 and the mean
motion lower bound >0. Report reflection strata without changing the criterion.
The quartet's static nuisance and lack of reflection make this transfer check
necessary. Neither bridge scores nor fresh confirmation targets may refit the
reader or change its selection.

## Safe handoff to original P0

Create a new versioned prerequisite receipt containing schema, registered plan
hash, source/encoder/checkpoint identities, frozen reader hash, FIT/confirmation/
bridge manifests and scores, freshness audit, all required checks and the final
pass/fail decision. Every downstream entry must validate that receipt and its
referenced artifacts. Invoke unchanged training primitives through a new V5.1
entry point; do not satisfy the original CLI by fabricating its old feature gate.
Keep feature-cache completion status distinct from scientific prerequisite
success. Version the final report so it checks V5.1 prerequisites rather than
requiring a misleading original `features` success record.

P0 uses complete independently encoded original-task episodes, not quartet
images and not the fitted probe's weights. The intended full cache remains Train
2048 episodes (pairs 0–1023), Dev 256 episodes (pairs 0–127), seed 42; Test 512
episodes remains unopened. Pair variants remain adjacent for interventions.
Original probe-cache reuse is allowed only with exact encoder, generator,
schema, precision and per-file identity checks; a simpler fresh full extraction
must still fit the cumulative budget. The cache must contain observed features,
future latent labels and evaluation masks in their original separate fields.

The unchanged P0 procedure in `pmr/visual_memory/train.py` is:

1. Train-only normalization; `Predictor(feature_dim=1280,tokens=16,dim=128,
   heads=4,local_layers=3)`. Only RGB features, public times/shutters, valid masks
   and query time enter `ActorBatch`.
2. The first 32 Train episodes, 300 steps at LR `1e-3`, batch size 32. The best
   small-sample MSE must fall by at least 20% from initialization.
3. LR trials `[1e-4,3e-4,1e-3]`, 150 steps each, same initialization and sample
   order; select by original Dev MSE. Restart the final fit from the common
   initialization for 5000 steps, batch size 32, seed 42. Full/subset-history
   masks alternate; the subset randomly keeps roughly 75% of history frames.
4. Retain per-step checkpoints/optimizer/RNG state and learning curves. Select
   the best Dev checkpoint under the unchanged implementation; save `p0.pt`.
5. Apply `history_gate`: the paired-world 95% bootstrap lower bound of
   recent-only minus full-history MSE must be positive; swapped-history minus
   full-history mean must be positive; hold/interference/update family means
   must all be positive. Hash the exact P0 checkpoint into that gate.

P0 is a single-GPU training algorithm with fixed global batch 32. Eight-card
authorization does not justify silently changing optimization into DDP or
altering learning rates. Parallelism applies to independent extraction shards
and later independent arms. The P0/history gate is a development screen on the
same Dev used for checkpoint selection, not an untouched confirmation estimate.
Passing the new temporal confirmation does not establish future prediction or
long-memory benefits. A saved P0 checkpoint with a failed history gate is a valid
failure artifact and must not authorize candidate training.

## Candidates, final evidence and stopping

Only a passed history gate whose checkpoint hash matches `p0.pt` opens the six
writer/reader candidates and full/recent/fifo references. Reuse the original
training primitives, common initialization, three 150-step LR trials followed
by 2000 fitting steps, and the original per-arm ceilings. Freeze all P0 weights.
Each arm is an independent one-GPU job; parallel scheduling must preserve arm
identity, paired evaluation and a single atomic aggregate budget ledger.

Fixed-writer replay begins only after C0 succeeds. Keep the original 500-step,
LR `1e-4` reader-only adaptation and verify every writer tensor remains unchanged.
Report all ten result variants, every stage status, LR trials, per-episode and
family/pressure metrics, memory-intervention diagnostics, checkpoint hashes,
state-size accounting and limitations. Neither an individually successful arm
nor a queued/running parent allocation makes the experiment complete.

An implementation failure may be diagnosed and resumed under an unchanged
scientific identity and reconciled remaining budget. A failed registered
scientific prerequisite is a conclusion: stop its downstream chain, preserve
all evidence, and do not retry seeds or candidates to obtain a pass. The new
registration would be required for a changed hypothesis. Budget/time exhaustion
is reported separately as incomplete.

Archive the new source, receipts, complete score tables, plots, checkpoints and
reconstruction/hash inventories to GitHub, then re-download the immutable
commit/release and verify every required item. Exclude reproducible official
weights, full feature tensors and private environment files while preserving
their source/hash/reconstruction metadata. The original failed remote directory
remains preserved. Remove only the separately identified new directory after
the entire intended new chain succeeds and its archive passes readback checks;
any scientific, execution or archive failure preserves the new scene as well.
