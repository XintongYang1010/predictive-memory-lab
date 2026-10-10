# V5 DGX execution handoff

## Task and current status

Run the V5 visual world model + bounded memory pilot from this repository using
the user's local Mac SSH environment. The cloud authoring session has not
connected to DGX, created the remote directory, or submitted a V5 GPU job.
CPU tests passed; official I-JEPA loading, GPU throughput and scientific gates
still need validation. This file describes remaining work, not completed work.

| Item | Target |
|---|---|
| Existing SSH alias | `dgx-127-slurm` |
| Remote work root | `/aifs4su/guhao/KVCache/WM` |
| Repository | `https://github.com/XintongYang1010/predictive-memory-lab` |
| Code | `pmr/visual_memory/`, `scripts/*visual_v5*` |
| Protocol | `docs/VISUAL_MEMORY_V5.md` |
| GPU | One allocated A100 |
| First-round cumulative cap | 60 GPUh; initial stage ceilings total 54.5 GPUh |
| Cleanup | Only this task's newly created remote directory, after verified archival |

The repository contains the V5 code; no separate chat ZIP is required. Read
applicable `AGENTS.md` files and preserve local changes when fetching the latest
`main`. Record the actual code commit and subsequent changes.

## 1. Connect and inspect the cluster

Use `ssh dgx-127-slurm` from the existing Mac SSH configuration, including its
jump hosts and authentication. Do not guess new hostnames, rebuild working SSH
settings or expose credentials in logs/GitHub. Follow normal executor approvals.

Inspect hostname, Slurm configuration, partitions/GRES, account/QOS associations,
the user's existing jobs, storage and Python/CUDA options. Useful read-only
commands include `sinfo -N -h -o '%P|%G|%m|%c'`, `scontrol show config`,
`sacctmgr show assoc where user="$USER"`, and `squeue -u "$USER"` where supported.
Choose an authorized A100 resource from actual output; report a concrete blocker
if access or A100 availability cannot be established. Follow login-node rules
and run training inside Slurm allocations. Preserve unrelated jobs and projects.

## 2. Isolate storage and adapt the backend

Inspect `/aifs4su/guhao/KVCache/WM` before writing. If absent, create it and record
its creation and resolved path. If it exists, preserve its contents and create a
unique run-id subdirectory; only that new subdirectory is eligible for cleanup.
Keep this run's code, independent environment, assets, logs and results inside
the tracked root. Preserve shared environments and installed PyTorch versions.

The launcher and scientific CLI still implement the original NYU backend.
Explicitly adapt these files for DGX before scientific execution:

- `scripts/launch_visual_v5.py`: real account, partition, optional QOS, resource
  parsing, backend-specific checks, accounting and durable submission receipts.
- `pmr/visual_memory/__main__.py`: explicit backend/account validation retaining
  genuine Slurm-allocation, single-A100, restart and budget checks.
- `scripts/prepare_visual_v5.sh`: appropriate CPU preparation location and an
  isolated Python/CUDA environment.

Never set fake `SLURM_JOB_ACCOUNT` values to impersonate the NYU allocation.
NYU's 300-hour initial quota is not a DGX quota. Retain the independent first-round
60-GPUh cap. Test dry-run behavior and submission/accounting guards. Preserve the
scientific protocol; record any necessary changes and use a fresh run identity
when the previous configuration or source identity is no longer compatible.

## 3. Prepare, profile and run the pilot

Read the full V5 protocol. Prepare pinned official I-JEPA source and weights,
run existing CPU tests and inspect paired RGB contact sheets. First allocate one
A100 for actual weight loading and encoder profiling. Confirm GPU model, job ID,
memory usage and measured throughput. Then run the prerequisite-gated chain:

1. Profile.
2. Raw/pooled feature probe and feature cache.
3. Small-sample fit, P0 training and history-necessity gate.
4. Six writer/reader configurations: A0/A1/B0/B1/C0/C1.
5. Full-history, recent-only and equal-byte FIFO references.
6. Fixed-writer reader-only replay and reporting.

Preserve causal paired inputs, fair adaptation budgets, held-out Dev, unopened
Test, memory-swap/zero interventions, checkpointing and provenance. Investigate
retention, interference and update failures using per-case evidence. Inspect a
failed gate and distinguish implementation failure from scientific findings;
do not lower thresholds or repeatedly reseed just to obtain a pass.

Use durable Slurm jobs/dependencies and checkpoints so SSH disconnection does
not kill training. Check actual state/logs after submission: PENDING is not
RUNNING, and submission is not scientific completion. Persist IDs, paths and
recovery commands. Bound retries and reconcile Slurm usage before resuming
within the original cumulative cap. Mark incomplete stages partial.

## 4. Archive readable results to GitHub

Use existing GitHub authentication, preferably from the Mac, to archive to this
repository. Preserve its visibility. Commit results and needed reproducibility
changes on a new results branch, without force-pushing or overwriting unrelated
work. Use a clear directory such as `results/wm/<run-id>/`.

Include actual source version/patches, configuration/seeds, environment,
data/encoder hashes, submission receipts, stage statuses, logs, metrics CSV/JSON,
training curves, per-case interventions and main checkpoints. Publish a readable
README with tables, figures, failure cases, limitations and next experiments;
show the findings directly on GitHub as well as in downloadable archives.

Use suitable Release assets or the existing large-file mechanism for large
checkpoints. Official pretrained weights and reproducible caches may instead
have exact sources, versions, hashes and reconstruction steps. Record exclusions,
an artifact inventory and SHA-256 checksums.

## 5. Verify archival and clean up only this run

Read results back from GitHub and download archive assets to verify hashes and
required contents. A local commit or an upload command alone is insufficient.

Only clean up after the planned experiment finishes, required results are
recoverable from GitHub and no active job/process is writing to this work root.
Immediately before cleanup, verify the creation record and resolved path,
including symlink/mount boundaries. Delete only the newly created WM directory,
or this run's new subdirectory if WM preexisted. Preserve its parent and all
unrelated projects/environments.

If training is incomplete, uploading fails or verification fails, preserve the
remote directory and report the blocker. Leave a recoverable state record if
the local session stops; do not claim follow-up actions ran unless verified.

## Final report

Report actual GPU, job IDs, completed/partial stages, GPU-hour accounting,
comparisons and bad cases, limits on conclusions, exact GitHub commit/result/
asset links, and the verified cleanup outcome or reason for retaining files.
