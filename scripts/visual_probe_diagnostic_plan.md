# Fixed CPU probe diagnostic plan

This plan is recorded before examining the follow-up caches. The original Dev set is already inspected; all further Dev results are exploratory. The original gate and prior result archive remain unchanged, and Test is never opened.

Compare raw and pooled features using the original concatenation ridge, a temporal-difference motion decoder with the original position decoder, and a shared frame-to-position decoder whose two predictions are differenced. The last method derives t6 positions from the existing position/displacement labels strictly for this diagnostic fit; these labels never enter an actor.

For each method, select alpha only by four-fold, pair-grouped inner-Train cross-validation. Minimize the mean displacement-dimension normalized MSE, using each fold's Train mean as the baseline. This deliberately focuses selection on the failed motion prerequisite. Report all eight target dimensions. Examine nested 16/32/64-Train-pair subsets separately, without selecting a sample size from Dev. At all 64 Train pairs, retain fixed-alpha 0.01 concatenation and difference controls.

Use chunked float64 Gram matrices and bounded CPU threads. Record centered trace/energy, selection folds and scores, wall time and peak process memory. Full-Train Dev reports include a 2,000-replicate paired bootstrap preserving both variants. Confidence intervals are exploratory and do not replace the gate. No Dev winner or downstream training permission is produced. Exact parameters and seeds are in `visual_probe_diagnostic_plan.json`.
