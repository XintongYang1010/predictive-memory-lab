# V5 feature-probe diagnostic plan

This is a follow-up to the stopped H800 run, source `d0b4fef` and archived result
`d4b6fc6`. It does not change that run, its gate, or its stopped status. The user
requested consultation with the original ChatGPT design conversation and further
diagnosis, with code and conclusions synchronized through GitHub.

## Fixed scope before running the new probe suite

- Reuse only the existing 128 Train / 128 Dev episode caches, each containing
  64 paired worlds. Verify all tensor SHA256 hashes against the original archive.
- No new encoder run, no GPU allocation, and no Test access. The already inspected
  Dev set is exploratory. No revised model is authorized for promotion by its
  score on these same Dev records.
- Preserve original pooled position R² > 0.1 and displacement R² > 0 gate as a
  historical result. Any future continuation needs a separately specified
  confirmation protocol; a diagnostic success cannot relabel this gate as passed.
- Use four pair-grouped inner-Train folds. Both variants and both frames of a
  pair stay in the same fold. Centering and normalization are fitted within each
  training fold. Select ridge alpha only by inner-Train normalized displacement
  MSE; do not select alpha, sample size or method using Dev outcomes.
- Fixed alpha grid: 0.0001, 0.001, 0.01, 0.1, 1, 10. Fixed nested Train pair counts:
  16, 32, 64. Report every planned result, not only favorable ones.

## Comparisons

Run each diagnostic on raw spatial tokens and the original 4×4 pooled tokens:

1. Original two-frame concatenation at fixed alpha 0.01. Verify reconstruction
   against the archived gate, then report the Train-CV alpha variant separately.
2. Temporal feature difference for displacement. Keep a concatenation position
   readout; report fixed alpha 0.01 and the Train-CV alpha variant. Record kernel
   energy/normalization so that a gain is not attributed to pooling alone.
3. A shared single-frame position decoder trained on both frames. Derive position
   at t=6 from position at t=7 minus the recorded displacement; obtain displacement
   by subtracting the decoder's two position predictions. This is diagnostic
   supervision only: state labels must not enter the actor.

Use chunked float64 Gram matrices and bounded CPU threads. Report all eight target
dimensions, position/displacement means, selected alphas, inner-Train CV scores,
pair-grouped 95% bootstrap intervals on Dev, runtime, and cache/source identities.
The intervals describe this exploratory sample, not confirmation of a new gate.

## Independent checks and consultation

Audit time indices, reflection-aware targets, paired split identity, encoder
preprocessing and pooling, and the relationship between visible pixel motion and
recorded targets. Analyze existing bad-case errors by axis, pair variant and
boundary context without making causal claims from observational subgroups.

The original design conversation, `6ac89457-ab14-83e9-85fb-745d12e0003d`, has been
sent the immutable GitHub evidence and asked to publish a separate review branch.
Its analysis will be treated as hypotheses until checked against code and data.

Results will go into `results/wm/visual-v5-probe-diagnostic-20261009/`. Preserve
IndexMem, the remote WM scene, the frozen first-run source, and all prior archives.
