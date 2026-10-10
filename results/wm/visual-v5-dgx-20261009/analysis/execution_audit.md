# Independent execution-evidence audit

Scope: read-only inspection of the locally packaged `results/wm/visual-v5-dgx-20261009` archive. No SSH, source edits, job actions, or GitHub read-back were performed for this audit.

The archived evidence supports **an eight-GPU feature-probe run that stopped at its failed prerequisite**, not a completed training/comparison campaign.

| Check | Finding |
|---|---|
| Source identity | Run manifest records `d0b4fef25e98eefb58bfba737997cb0145f27daa`; all 16 recorded scientific-source file hashes match the archived source snapshot. |
| Real Slurm execution | Existing parent `415453`, account `hansirui`, partition/QOS `llm`, node `dgx-097`. Final accounting records profile step `415453.14` **COMPLETED**, exit `0:0`, 32 seconds; feature step `415453.15` **FAILED**, exit `1:0`, 180 seconds. |
| GPU selection | Profile selected physical H800 GPU 3. Features validated all eight distinct H800 UUIDs on indices 0–7; eight worker receipts each report that step and a distinct matching UUID. Both Slurm steps reported eight allocated GPUs. |
| Actual worker work | All eight workers completed 16 Train and 16 Dev records each. Parsing their JSONL logs yields exactly 128 unique records per split, with no duplicates; those names exactly match each cache manifest. Each split contains 64 pair IDs with two variants. |
| Completion boundary | `profile_status.json` is complete. `features_status.json`, its step receipt, and the controller identify the pooled-cue gate failure. Submission receipts contain only profile and features. P0, adaptations, references, and replay did not run. |

All eight worker process lifetimes overlap for approximately 27.44 seconds. Their successful task receipts and per-record logs establish that all eight workers performed encoder work; this archive has no per-kernel timeline proving continuous simultaneous GPU utilization. Encoder loading was staggered. This limitation does not turn the worker count into merely a planned allocation.

The failed criterion is supported directly by `run/feature_gate.json`: pooled position R² is `0.9746909151`, but pooled observed-displacement R² is `-0.02092683065`, below the unchanged `> 0` requirement. The recorded raw-feature displacement R² is `0.08055265852`. `feature_diagnostics/verification.json` reports exact reconstruction of the saved gate. This is a prerequisite representation/probe finding; it is not evidence about the relative performance of memory architectures. The completion assessment correctly remains **partial**, and Test evaluation is recorded as unopened.

## GPU-hour interpretation

The chosen-device budget ledger is internally consistent:

| Stage | Selected GPUs | Ledger GPUh |
|---|---:|---:|
| Profile | 1 | 0.007657659230495079 |
| Features | 8 | 0.3839540171478358 |
| Total | — | **0.39161167637833216** |

The feature charge covers eight selected devices across the stage's approximately 172.779-second budget window, including staggered initialization and waiting. This is selected-device walltime accounting, not measured kernel-busy time or incremental Slurm billing. The separate Slurm elapsed values include startup/teardown outside that budget window: `32×1/3600 + 180×8/3600 = 0.408888889 GPUh`. Multiplying both recorded step reservations by eight gives `0.471111111 GPUh`. These overlapping steps reused an existing eight-GPU parent; their reservation figures must not be added to that parent's elapsed allocation as independent new billing. The parent snapshot itself was still RUNNING with 42:31:28 elapsed (340.195556 allocation-GPUh since its earlier start), mostly unrelated to V5. All V5 interpretations remain far below the 60-GPUh round ceiling.

## Existing work and archive boundaries

The six preflight IndexMem PID/UUID pairs—PIDs `4084652`, `4084796`, `4084680`, `4084873`, `4084697`, `4084713`—are all present in the post-run GPU and process snapshots. Their executable/command paths still identify IndexMem, and final Slurm accounting leaves its step `415453.5` RUNNING. No V5 worker PID appears in the post-run GPU-process listing. This supports preservation of the existing workers at the archived observation time; it does not establish that sharing caused no performance slowdown.

All **153 inventory-listed archive files** matched their recorded sizes and SHA-256 values during this audit. The archive contains no `.pt` cache payloads or trained checkpoint asset. Cache record names/hashes and reconstruction instructions are present; the gate prevented production of downstream trained checkpoints. Accordingly, this audit verifies the packaged evidence and its consistency, not fresh recomputation from omitted cache tensors, GitHub recoverability, or remote cleanup. Those require separate verification. The failed run must remain preserved remotely under the handoff's cleanup rule.

Primary evidence: `run/workers/*.{json,log}`, `run/gpu_selection/*.json`, `run/steps/*.json`, `run/submission.json`, `run/budget.json`, both `run/probe_features/*/manifest.json`, `logs/slurm-accounting-final.txt`, `logs/eight-gpu-preflight.txt`, `logs/post-run-gpu-processes.txt`, and `archive_manifest.json`.
