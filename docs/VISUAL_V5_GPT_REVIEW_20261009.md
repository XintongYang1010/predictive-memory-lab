# V5 design-conversation consultation

The user authorized consultation with the original ChatGPT conversation
**分支 · Saining Lec6&实验** (`6ac89457-ab14-83e9-85fb-745d12e0003d`).
Code and conclusions continue to be exchanged through this GitHub repository.

## Evidence supplied

Three messages supplied the actual stopped GPU run and then the CPU follow-up:

1. Original source `d0b4fef25e98eefb58bfba737997cb0145f27daa`, archived results
   `d4b6fc622d0fe08e28605aaf6e48100201ab2008`, real Slurm steps `415453.14` and
   `415453.15`, failed pooled probe, and preservation of the IndexMem allocation.
2. CPU diagnostic snapshot `8b2ee24d1beabf69efa4f9b2329fdf43963f3883`:
   fixed versus Train-selected regularization, direct difference/shared-position
   probes, input/rendering audit, and the single-frame position prior.
3. Matched temporal-control snapshot `ca365e1aa1026c51be0d23008b63650354c5bf31`:
   no pooled two-frame advantage over the Train-selected single-frame control,
   query-order reversal results, independent numerical verification and 62 tests.

The review request asks for overlooked implementation errors, a diagnosis that
distinguishes probe readability from temporal dependence, and a fixed independent
confirmation protocol. In particular, it asks how to prevent single-frame
position priors from substituting for motion information. It prohibits changing
the original failed gate, opening Test, launching GPU jobs, or overwriting the
existing main/result/diagnostic branches as part of that review.

## Received response and independent local verification

The actual new answer was received on 2026-10-10 UTC (2026-10-09 New York),
message `888ba774-01fa-57b7-96d8-68805b03d858`. It cites independent review branch
`review/v5-designer-probe-audit-20261010`, commit
[`b13c9877339779b8c770d175a483ea3489d2420c`](https://github.com/XintongYang1010/predictive-memory-lab/commit/b13c9877339779b8c770d175a483ea3489d2420c).
The older `df9e780` upload message was not mistaken for this review.

We fetched that exact commit and inspected its eight added files. Its parent is
the earlier frozen scientific source; its branch snapshot therefore omits later
archives. We integrated only its additive commit with `cherry-pick -x` as
`577aa09`, preserving all existing results and diagnostics.

The reviewer agrees that the current blocker is the validity of the temporal
prerequisite. It found no newly demonstrated causal implementation error in
frame indices, labels, pooling, splits or ridge. Regularization changes the
score, but pooled concat does not beat the matched single-frame control. This
does not prove the encoder lacks motion information or that all improved score
comes from position priors. The old failure remains unchanged.

Received deliverables:

- [Full design review](V5_PROBE_REVIEW_20261010.md)
- [Fixed quartet confirmation plan](../scripts/v5_temporal_confirmation_plan.json)
- [CPU quartet, kernel and scoring helpers](../scripts/v5_temporal_confirmation.py)
- [Reviewer evidence](../results/wm/visual-v5-probe-review-20261010/README.md) and
  [independent local validation](../results/wm/visual-v5-probe-review-20261010/local_validation.json)
- [Independent local diagnostic measurements](../results/wm/visual-v5-probe-diagnostic-20261009/README.md)

Local validation passed **73 applicable CPU tests**, including the reviewer's
11 new software checks. Its archived-prediction audit was rerun locally: all
non-environment JSON fields match exactly (maximum numeric difference 0), despite
different NumPy/Pillow versions. Independent source review found the matched
anchor proof sound and no blocking numerical/statistical defect in the helpers.

Two qualifications matter for execution:

1. The helper deliberately returns `p0_authorized=false`. Its metric-only flag
   can be true on eight software-test blocks; a new runner must enforce exactly
   256 unique fresh confirmation blocks, identical anchor tensors, FIT/model/
   protocol hashes, the original-distribution bridge and Test isolation. No such
   GPU extraction/provenance runner is delivered yet.
2. For the registered reversal-balanced ridge and swap-invariant kernels, the
   two anchor contrasts are algebraically equal, not independent evidence.
   Independent software checks found differences below 1e−16. The nine-endpoint
   Bonferroni procedure is still conservative; independence is not required.

The accepted next experiment is a **new confirmation**, not a repaired historical
success: freeze the new reader on 256 FIT quartets, then evaluate 256 unseen
quartets and the predeclared original-distribution bridge. The same frozen
pooled reader must satisfy all registered criteria before a separately versioned
P0. FIT/confirmation/bridge scientific data have not been generated in this
consultation. No new GPU job or remote cleanup occurred, and Test remains sealed.

## Local reasoning while awaiting review

A future criterion should not require that both exact endpoint positions carry
zero conditional-mean information about nontrivial Euclidean displacement.
Let d = p7 − p6 with finite second moments. If E[d|p6] = E[d|p7] = 0, then
E[||d||²] = E[d·p7] − E[d·p6] = 0, so d is zero almost surely. More generally,
if both conditional means are the same constant mean displacement, displacement
has zero variance. This statement concerns exact endpoint coordinates; it is
not a theorem about every lossy image representation or a fitted finite-sample
probe score.

Consequently, the useful confirmation question is whether two frames provide
reproducible **incremental** predictive value over matched single-frame controls.
Holding the terminal rendered image exactly fixed while changing earlier
history can additionally test whether the prediction responds to that history.
A periodic-space direction label can remove different marginal shortcuts, but
changes the task away from the original unwrapped Euclidean displacement and
must be declared as a separate diagnostic. These are candidate design constraints,
not a new registered protocol or a claim of experimental confirmation.

A concrete candidate is a matched-terminal family: choose an interior anchor
p7 = z and nonzero vector u, and pair p6 = z − u with p6 = z + u, giving targets
+u and −u. Both variants must have byte-identical terminal RGB and identical
nuisance variables, while earlier RGB differs. A deterministic terminal-only
predictor has exactly zero within-pair prediction contrast, versus the required
2u. A symmetric matched-initial family holds p6 fixed and changes p7 to test
dependence on the later frame. Report these families separately; combining
them does not eliminate every single-frame statistical prior.

Before any new extraction, fix the generator version, independent Train and
confirmation namespaces, group all related variants in the same split/fold,
specify the Train-only model/alpha selection and interval procedure, and state
the confirmation decision. Verify identical-image hashes, nonzero visible
pixel motion and no reflection/wrap ambiguity. The original Test split remains
sealed. This would be a new confirmation experiment, not a relabeling of the
old failed run or a change to its archived metrics.
