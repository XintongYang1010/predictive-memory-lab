# V5.6 — controlled ablations and optimization trials

**2026-10-10 progress snapshot: three CPU experiments completed; new GPU extraction / J-X confirmation not yet executed. This is not the final archive.** Source and protocol were frozen at [031788e](https://github.com/XintongYang1010/predictive-memory-lab/commit/031788e0499c242a3671f57d026f3ed71da4872e) before these runs and all114 source blobs were read back byte-for-byte. Original Test/P0 and all older results remain unchanged.

## What the executed CPU trials currently show

The target-aligned readout advantage survives the original global kernel scale and FIT-only effective-df matching. Matching df alone does not repair the full reader. Fixed raw local correspondence substantially outperforms matching after averaging the same raw tokens, and an equal-byte source-RGB-ROI bank preserves much of that advantage. A naive change-magnitude retention policy fails. The frozen-reader local-change write gate also fails to beat latest/FIFO on the bounded endpoint replay. These are exploratory results on already opened256 paired Dev worlds/512 episodes.

### Frozen reader ablation

All arms use identical W4096 supervision. Df targets come only from FIT; no Dev alpha selection. Horizontal errors refer to the original000→111 query; direction column counts episodes with at least one horizontal sign error. Full all8-mask data are in [reader_ablation.json](cpu/reader_ablation.json).

| Reader | Horizontal RMSE (px) | Any horizontal sign error |
|---|---:|---:|
| global__original | 2.438 | 196/512 |
| global__df_matched | 2.441 | 197/512 |
| additive__original | 2.360 | 181/512 |
| additive__df_matched | 2.355 | 180/512 |
| target_common__original | 2.099 | 144/512 |
| target_common__df_matched | 2.072 | 140/512 |
| correct_lane__original | 2.043 | 139/512 |
| correct_lane__df_matched | 2.043 | 139/512 |
| opposite_lane__original | 3.580 | 330/512 |
| opposite_lane__df_matched | 3.583 | 333/512 |
| checkerboard__original | 2.722 | 250/512 |
| checkerboard__df_matched | 2.721 | 249/512 |

Original-query parity max errors: global1.43e-12, lane1.87e-12. Common-scale target kernels retain about26–28% MSE improvement, while opposite lanes and checkerboards fail. This supports target-aligned feature access within the fixed kernel family; it does not establish learned object identity.

### Same-source correspondence and equal-byte storage

| Fixed matcher input | Horizontal RMSE (px) | Mean four-axis skill |
|---|---:|---:|
| Raw16x16 |1.534|0.758|
| Mean pool4x4 from same raw |3.878|−0.775|
| Uniform bank, equal bytes |3.753|−0.745|
| Source-RGB ROI bank, equal bytes |1.738|0.675|
| Random detail bank |4.772|−1.642|
| Two-frame change-priority bank |6.663|−4.227|

All625 displacement costs and every prediction are retained locally for the final full archive. See [correspondence.json](cpu/correspondence.json). The source RGB supplies privileged object ROIs. The sparse banks use a fixed IDW reconstruction; change-priority reads both frames. A failing matcher does not prove that pooled features contain no useful motion information. These are two-frame storage diagnostics, not long-term memory results.

### Fixed-reader update replay: negative result

| Policy | Position RMSE (px) | Missed writes on2048 true object changes |
|---|---:|---:|
| Latest |2.691|0|
| FIFO |2.685|0|
| Feature-surprise threshold |3.124|1375|
| Local-feature threshold |3.001|842|

All policies use the same frozen lane position reader and32B state payload plus8B common metadata. Thresholds were fixed from FIT nuisance-only changes. Local gating reduces misses relative to full-feature surprise but remains worse than latest/FIFO; stale-state failures are retained. This is a binary-endpoint stress test with observation-free hold gaps, not a trained writer or a physically integrated motion sequence. See [replay.json](cpu/replay.json), which also contains all64-transition and yellow feature-path summaries.

## Pending work

Eight-card FIT/axis extraction, J/X fitting and model publication, fresh64 confirmation, full result/checkpoint archive and byte verification. Remote directories are retained while this work is incomplete. Full source configuration: [V5_6_EXECUTION_PROTOCOL.md](../../../docs/V5_6_EXECUTION_PROTOCOL.md).

## Detailed CPU evidence

- [Twelve-arm reader report and figures](reader_ablation/README.md). Complete model/CSV/binary predictions are reserved for the final Release.
- [Fixed bad cases, uncertainty intervals and selected-token coverage](correspondence_interpretation/README.md).
- [Independent update replay reconstruction](cpu/replay_verification.json).
- [Web ChatGPT method review and replies](web_review.md). No scientific thresholds or models were changed in response.
