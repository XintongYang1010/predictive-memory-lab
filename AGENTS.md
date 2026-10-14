# Agent rules for this repository (V6 onward)

These rules apply to any automated agent (Codex, Claude, others) working here.
They exist because V5.1–V5.8 spent eight rounds on perception diagnostics and
never ran the memory experiment the project is about.

## Research object

1. The object of study is the **world-model memory transition operator**: how a
   bounded memory keeps, evolves, corrects and protects beliefs about objects it
   cannot currently see.  Perception (encoders, pooling, readout of motion from
   pixels/features) is out of scope for V6.
2. If a failure might come from anything upstream of memory, switch the input
   to ground-truth object tokens (L0/L1/R) and continue the memory experiment.
   Do not open a perception-diagnostic round.
3. Every experimental round must contain at least one **budget-matched
   memory-vs-memory comparison** (same state scalars, same inputs, same loss).
   A round without one is not a round.
4. Only bad cases on memory metrics (ER, NHE, recovery, interference, history
   separation, extrapolation, surprise AUROC/MRA) may drive the next round, and
   one bad case gets at most two diagnostic rounds.
5. Establish a conclusion on L0 before moving it to L1, and on L1 before R.

## Protocol discipline

6. Thresholds live in `docs/PREREGISTRATION_V6.md` and
   `pmr/predictive_memory_v6/prereg.py`.  Never lower a threshold or re-seed to
   obtain a pass.  A failed gate is first debugged as a possible bug, then
   reported as a finding.  Protocol changes need the user's approval and a new
   preregistration version.
7. The test split is opened once, in stage FINAL, after the preregistration is
   frozen.  Dev numbers are never reused as confirmation.
8. Respect the GPU-hour ledger (`budget_ledger.json`): stage caps from
   `prereg.BUDGET_GPUH`, hard cap 160 GPUh.  Never raise a cap on your own.
9. Report negative results plainly.  "Missing" is not "pass".

## Engineering

10. Fix code bugs minimally, with a regression test, in a separate commit whose
    message says what was wrong.  Do not change science in a bug-fix commit.
11. Keep verification proportionate: archive results and verify them once;
    do not let receipts and re-verification commits outnumber result commits.
12. Never touch other users' jobs, directories or shared environments; never
    guess cluster accounts or spoof scheduler variables (see
    `docs/CODEX_DGX_HANDOFF.md`).
