# V5 visual-memory pilot — GPU validation pending

DGX execution update: platform adaptation is implemented. The user approved
reusing the existing H800 allocation; see [DGX execution record](DGX_V5_EXECUTION.md).
The original A100 protocol below remains the scientific baseline. This note
does not claim completed GPU experiments.

This implements the first runnable slice of the 2026-10-09 work order. It is
not an A100 result, an established best architecture, or an implementation of
the later parameter-sharing and learned-selector stages.

## Current DGX handoff

The current execution target is `dgx-127-slurm`, using the user's existing Mac
SSH configuration and `/aifs4su/guhao/KVCache/WM` as the remote workspace.
Follow [CODEX_DGX_HANDOFF.md](CODEX_DGX_HANDOFF.md) before deployment.
The launcher and scientific CLI still enforce the original NYU account: a local
executor must add an explicit DGX/Slurm backend using the actual cluster
configuration. Do not run the unchanged NYU launcher on DGX or spoof Slurm
variables. The scientific protocol below is unchanged.

## Current execution status (2026-10-09)

- CPU software checks pass on Python 3.12 / PyTorch 2.9.1+cpu.
- No V5 GPU job has been submitted by the assistant. No V5 job ID exists.
- The V5 source, tests and DGX handoff are included in this repository update;
  no experiment results or pretrained weights are included.
- DGX connectivity, Slurm account/partition selection and deployment remain to
  be checked from the user's local SSH environment.
- Official I-JEPA weight loading, A100 throughput, live partition names, Slurm
  submission, remaining personal quota, and scientific gates remain unverified.

## Original NYU backend: course resource requirements

The Computer Vision announcement dated 2026-10-02 specifies:

- Account: `csci_ga_2271-2026fa`.
- Initial personal allocation: 300 GPUh, or 18,000 GPU minutes.
- Use Cloud Bursting only; no Torch compute fallback.
- A100 40GB and L4 are offered; this runner requires one A100.
- Off-campus portal access needs NYU VPN.
- Low GPU utilization for 20 minutes can terminate a node. Checkpoint and stop
  safely; never run artificial keepalive kernels.

Portal: https://ood.burst.hpc.nyu.edu/
Course instructions: https://brightspace.nyu.edu/d2l/le/lessons/627988/topics/14038330
Public filesystem/Slurm guidance:
https://services.rt.nyu.edu/docs/cloud/hpc_bursting_to_cloud/intro/

The login-node and Burst compute filesystems differ. Prepare the code, Python
environment, checkpoint and output paths in a **Burst CPU compute session**
before submitting A100 jobs. Do not assume a path visible on a login node is
visible on an A100 node.

## Prepare (no GPU allocation needed)

Use the course's working CUDA PyTorch Python. The dependency file deliberately
does not install or replace PyTorch. If the course environment has not been
created, follow its Brightspace Conda/Singularity instructions first; a container
image or module name has not been guessed here.

```bash
cd /absolute/path/to/predictive-memory-lab
bash scripts/prepare_visual_v5.sh /absolute/burst/work/v5
```

Preparation pins the official I-JEPA repository at
`52c1ae95d05f743e000e8f10a1f3a79b10cff048`, downloads the official ViT-H/14
224-resolution ImageNet-1K checkpoint, runs CPU checks and generates paired-world
contact sheets. Run downloads/setup on CPU, before the idle-GPU timer starts.
Checkpoint loading uses `weights_only=True` and exact state-dict matching;
there is no unsafe-pickle or random-weight fallback. A failed official checkpoint
load needs investigation, not a silent substitution.

Inspect the live environment:

```bash
hostname
sinfo -N -h -o '%P|%G|%m|%c'
sacct -X -S 2026-09-01 -A csci_ga_2271-2026fa -u xy3257 \
  --format=JobID,State,Elapsed,AllocTRES
```

Choose the A100 partition actually returned by `sinfo`. Its name is deliberately
not hard-coded from another course or year. Inspect the contact sheets before
allocating the full chain.

## Inspect and submit

First run without `--submit`. Replace the three path/partition placeholders with
real values from the Burst compute session. This is an example with explicitly
unresolved environment values, not a claimed tested HPC command.

```bash
python scripts/launch_visual_v5.py \
  --out /absolute/burst/work/v5/run1 \
  --assets /absolute/burst/work/v5/assets \
  --partition ACTUAL_A100_PARTITION --all
```

Add `--submit` to queue the chain. Without `--all`, only the 30-minute profile
job is prepared/submitted. Jobs are sequential and require `afterok`; a failed
feature or history gate prevents downstream GPU allocations. The script checks
the actual GPU name in the job, accounts for current usage, refuses conflicting
course GPU jobs, and records each returned job ID immediately.

| Stage | Maximum one-GPU wall hours |
|---|---:|
| Real encoder profile | 0.5 |
| Probe and feature cache | 8 |
| Small-sample fit, LR trials, P0, history gate | 12 |
| Six writer/reader candidates, 4 each | 24 |
| Three references, 2 each | 6 |
| Fixed-writer reader adaptation and reports | 4 |
| **Whole initial chain** | **54.5** |

The persistent round cap is 60 GPUh. These are scheduler ceilings, not measured
runtime. Subsequent runs resume within the original phase/round ceiling; they do
not consume the other 240h automatically. Stage limits are conservative leases:
normal exits refund unused reserved time; a hard kill keeps the entire lease
charged until a human reconciles it against Slurm accounting.

Monitor returned IDs with `squeue -j ID` and `sacct -j ID`. The status file
`submission.json` is only a submission receipt. Scientific completion requires
stage status files and passed gates. A queued/pending job is not training yet.

For an inactive interrupted chain, add `--resume --all --submit` with the same
paths. Successful stages are skipped; remaining phase walltime is reduced by
already charged time. No automatic requeue loops are enabled. Unchanged config,
data identity and source revision are required.

The downloadable package also runs without `.git`; a hash of its actual source
files is recorded. A local ZIP edit cannot silently resume an older checkpoint.

## Implemented pilot protocol

- Two textured moving targets occupy separate image regions. One can receive a
  visible velocity update while the other remains occluded. Distractors occupy
  a third region. This is a controlled RGB mechanism task, not realistic video.
- Eight early observations, independently scheduled interference observations
  (0/8/32), four update/control observations and four recent occluded observations.
  There are at most 48 observed frames, embedded in continuous simulated time;
  this is the explicit observation-event interpretation of the work order's
  64-frame starting point. Gap varies over 8/16/32 at fixed write count.
- Each pair has identical recent RGB, observation times and shutter actions.
  Earlier velocities and each world's correct future labels differ. Random
  updates are evaluated only after observable evidence of the new motion.
- The actor sees only independently encoded RGB features, timestamps and public
  shutter actions. IDs, velocities, evaluation ROI masks and roles remain outside
  the actor. Future targets are revealed images at horizons 1/4/8.
- Frozen official I-JEPA target encoder, 4x4 pooled spatial tokens. A Train-only
  ridge probe compares raw versus pooled representations on a held-out Dev set.
  Targets are current positions and observed displacement, not unobservable
  pre-reflection velocity. Feature gate: mean position R2>0.1 and displacement
  R2>0. This heuristic gate is recorded and is not a universal threshold.
- Train/Dev episode counts 2048/256, seed 42. All paired variants stay together.
  A 512-episode Test namespace is reserved but neither extracted nor evaluated.
- P0: d=128, H=4, three local Transformer layers, recent-only local read and one
  explicit history attention. No indefinite contextual history in stored tokens.
  History QKV are learned here; image pretraining and temporal pretraining differ.
- P0 first fits 32 examples, tries 3 learning rates for 150 steps each, then trains
  up to 5000 steps from the common initialization. Full/subset history alternate.
  Correct history must beat recent-only with a positive paired bootstrap lower
  confidence bound, beat swapped history, and improve every family mean on Dev.
- A0/A1 additive; B0/B1 gated additive; C0/C1 gated delta. Q/K are L2 normalized
  only in the compressed-memory operation. Every candidate uses the same address
  feature map, matrix size, output RMS normalization and shared fusion interface.
  Alpha is in [0.95,1]; beta in [0,1]. Gated arms share initialization. Events are
  written in chronological order, patches within an event in raster order.
- All P0 parameters freeze during this round. Query correction is a zero-delta
  rank-8 adapter. Added trainable parameters and memory fusion scale are logged.
  All six arms and three references receive the same LR candidates/step budgets:
  3x150 short trials plus at most 2000 final fitting steps, fresh common start.
- R=4 independent FP32 frame encodings, E=8 BF16 KV frames, M=4x32x32 FP32.
  Exact-only E=10 matches the tensor byte total. Four int64 controls are included.
  Python object overhead, model weights and temporary/autograd workspace are
  reported separately from the persistent tensor budget; this is not total VRAM.
- A bounded `MemoryBank` reference provides `append/read/reset/bytes`. Its states
  match the differentiable legal-prefix training replay. Frame entries move once
  from recent to exact to compressed storage. Padding causes no updates.
- Each Dev evaluation includes normalized latent MSE, future-object spatial ROI
  latent MSE, paired response error, zero-memory and same-model paired-memory
  swap. ROI metrics are pooled feature errors, not physical tracking accuracy.
- A C0-to-C1 reader-only run freezes both the writer and all state-producing
  projections; deterministic prefix replay therefore holds M fixed. Reports
  select candidate failures but do not invent a cause or claim they are fixed.

Still pending after this pilot: gap=64 generalization, more than two target
objects, appearance/address-similarity sweeps, recovery-latency curves, failure
minimization with fresh confirmation samples, sharing/LoRA comparisons, and
learned selectors. Do not present the pilot as the complete work order.

## Validation and outputs

```bash
python -m unittest discover -s tests/visual_memory -v
```

Tests exercise paired recent-input equality, split separation, independent
gap/write axes, nontrivial gated-delta math, orthogonal-address revision without
collateral damage, padding/empty state, chronological replay, bounded tensor
bytes, single writes through FIFO, identity query initialization, no label/role
input, real adapter/gate gradients, frozen P0/writer parameters, budget caps and
checkpoint recovery. The tiny RGB-derived CPU encoder is test-only; it cannot
be selected by the scientific CLI.

Outputs include `run_manifest.json`, encoder hashes, `profile.json`, feature and
history gates, P0 checkpoint/hash, LR trials and curves, candidate checkpoints,
per-episode interventions, grouped CSVs, tensor-state costs, pressure plots,
failure candidates and persistent budget/submission receipts. Partial jobs are
marked partial. Falling training loss is never marked as evidence that the
scientific hypothesis holds.

Sources for implementation:
https://github.com/facebookresearch/ijepa
https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml
https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
