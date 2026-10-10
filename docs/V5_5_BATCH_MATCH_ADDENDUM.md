# V5.5 batch-shape validation addendum

This is an implementation-level validation of the same 512 already-inspected
Dev episodes and the same fixed global/lane readers. It does not add training,
new scenes, selection, Test access or a P0 gate. The original V5.5 source and
results remain immutable. The known original-cue lane comparison reduced dx
MSE by 29.78%; this comparison uses identical old cached inputs and does not
depend on the factorial batches.

The first extraction encoded original endpoints in batches of four and edited
endpoints in batches of twelve. A transformer without BatchNorm is batch
independent mathematically, but floating-point execution can still depend on
batch shape. We therefore validate every image, not a selected subset, before
interpreting small factorial responses as object effects.

Repeat all4096 image slots using four calls per world, each batch size four.
The paired endpoint-mask groups are (0,7), (1,2), (3,4), (5,6); each call orders
variant0's two images before variant1's two images. Reuse the same official
encoder/checkpoint, normalization, precision,4x4 pooling and final BF16
rounding. Preserve canonical features and all input hashes. Original endpoints
must remain bitwise equal to the first extraction. Compare every edited image
and every frozen-reader output against the first extraction. Report exact
equality counts, maximum/RMS differences, all8 conditions, and full matched-
batch predictions and attribution. No tolerance is chosen to discard cases.
The first raw-token/pooling arithmetic audit remains a separate result.

This supplement may run only after the first eight-GPU step is confirmed
terminal, its actual charge is closed, and the first cache is complete. The
entire V5.5 round, including this validation, retains the original1.2 GPUh
maximum. The new step reserves0.45 GPUh and is not launched unless at least
that much remains. It uses the same eight assigned H800 GPUs with per-worker
free-memory guards, a3-minute Slurm limit,165-second internal deadline, and
198-second controller deadline. All eight model loads may initialize in
parallel within384GiB host memory. Failure is preserved without automatic
retry; only this invocation's exact owned numeric step may be cancelled.

Publish and read back this addendum, new source and source manifest before any
supplementary image generation. Its registration binds the first cache
manifest, closed ledger, exact remaining budget and source commit. Write only
to the new batch_match_run child. The first cache and first controller/ledger
are never replaced. CPU scoring is separately recorded with zero GPUs. If the
repeat cannot be completed within budget, retain the first results and state
that the batch-shape ambiguity was not resolved.
