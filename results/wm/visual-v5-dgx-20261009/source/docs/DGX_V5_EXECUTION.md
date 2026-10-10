# V5 DGX execution record and recovery

The source baseline is `df9e780fbe8372269ab4066c11231ea04b19e4e3`.
The execution branch adds an explicit DGX backend while retaining the original
NYU backend and scientific gates. Actual results belong under `results/wm/`;
this page is the deployment procedure, not a completed experiment claim.

## Observed allocation and approved hardware change

On 2026-10-09 (America/New_York), live inspection through `dgx-127-slurm`
found Slurm cluster `slurm`, user `guhao`, account `hansirui`, partition `llm`,
QOS `llm`, and running parent job `415453` on `dgx-097` with eight NVIDIA H800
80GB devices. GRES is untyped (`gpu:8`), so it does not establish an A100 model.
The user explicitly requested using this already held IndexMem allocation after
the H800/A100 difference was explained. This run therefore measures H800, not A100.

The existing IndexMem step `415453.5` remains untouched. Initial inspection found
six busy GPUs and idle physical indices 3 and 7. Profile/P0/replay select index3;
the eight-GPU feature and adaptation stages coexist with the preserved IndexMem
processes. Explicit `--allow-gpu-sharing` records existing PIDs and checks at least
40GiB free VRAM on each selected GPU. Concurrent workloads can affect throughput;
these measurements are not exclusive-device benchmarks.
The cluster did not narrow CUDA visibility with `--gpu-bind=map_gpu:3`; the wrapper
therefore selects the verified NVIDIA UUID before importing CUDA. It never changes
`SLURM_JOB_ACCOUNT`, job IDs, or other Slurm identity fields.

The user subsequently requested eight-GPU parallel execution. Feature extraction
uses eight paired-data shards after a one-GPU encoder profile. P0 retains its
original single-GPU training schedule; after its history gate, up to eight
independent candidates/references run concurrently, followed by reader replay.
No learning-rate, seed, update-count or effective batch-size change is implied.

This is a shared preexisting eight-GPU reservation. Receipts distinguish parent
and step allocation from selected GPU count. The 60-GPUh V5 cap applies to the
sum of attributable device elapsed time across all stages; Slurm's
parent/overlapping-step reservation totals are reported separately and must not
be described as a one-GPU allocation or added together as independent reservations.
No A100-equivalent timing claim is justified.

## Storage and preparation

`/aifs4su/guhao/KVCache/WM` did not exist before this task. Its creation record
captures its resolved path, owner, inode, device and unique marker. All task code,
private environment, caches, assets, logs and results remain under this root.
The shared IndexMem environment is not modified. The DGX preparation script creates
an independent venv with PyTorch 2.8.0/CUDA 12.8, pinned official I-JEPA source and
the official checkpoint. Its exact package versions and checkpoint SHA256 are saved.

```bash
cd /aifs4su/guhao/KVCache/WM/code
PMR_BASE_PYTHON=/aifs4su/guhao/KVCache/IndexMem/.venv/bin/python \
  bash scripts/prepare_visual_v5.sh /aifs4su/guhao/KVCache/WM dgx
```

The paired hold/interference/update contact sheets were inspected on the Mac:
recent occluded observations match within each pair, while earlier trajectories
and revealed future targets differ. This inspection does not validate I-JEPA cues;
the held-out feature probe remains required.

## Profile and dependency-gated chain

Use `bash -lc` for remote commands so that the cluster's Slurm binaries are on PATH.
First invoke the launcher without `--submit` to inspect real account/resources and
record the dry-run plan. Then run profile alone under a detached controller. Only
a successful profile with the same encoder identity allows the feature stage.

```bash
/aifs4su/guhao/KVCache/WM/.venv/bin/python scripts/launch_visual_v5.py \
  --backend dgx --account hansirui --partition llm --qos llm \
  --gpu-model H800 --allocation-job 415453 \
  --shared-allocation-gpus 8 --gpu-index 3 --parallel-gpus 8 --allow-gpu-sharing \
  --out /aifs4su/guhao/KVCache/WM/run1 \
  --assets /aifs4su/guhao/KVCache/WM/assets
```

Add `--submit` to run profile. After inspecting its outcome, use
`--all --resume --submit` for the remaining gated stages, with parallel feature extraction and adaptation.
A failed prerequisite prevents downstream launches; already-running independent
sibling stages retain their checkpoint/exit handling. Concurrent budget leases
share one atomic ledger and remain within the original aggregate 60 GPUh cap. Do not alter gates, sample counts, seeds or
source identity to obtain a pass. Interrupted leases require accounting review;
source/configuration changes require a fresh run identity and preserved old state.

Inspect `submission.json`, `steps/*.json`, stage logs/statuses, `budget.json` and
`sacct -j 415453 --format=JobID,State,ElapsedRaw,AllocTRES`. A parent hold being
RUNNING is not evidence that a V5 stage is running or complete.

## Archival and cleanup

Archive exact source/configuration/environment, creation and Slurm receipts,
gates, logs, metrics, figures, interventions and training checkpoints. Official
pretrained weights and feature caches can be excluded only with hashes and exact
reconstruction instructions. Preserve the private repository's visibility.
Read GitHub results back and download checkpoint assets to verify inventory and
SHA256 checksums. Only a complete experiment with verified recoverable artifacts
and no remaining writers permits deletion of the exact newly created root.
Any failed/incomplete training, upload or verification retains the remote root.
