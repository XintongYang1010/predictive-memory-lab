# BR-0.2: connect existing memory bad cases, frozen-parent accounting

Frozen before computation on 2026-10-22 (America/New_York). This follows the
user's request to connect previously analyzed predictive-memory cases with the
WR/PaFu/CV oracle cases. It is not a new generic bad-case search, a V6 protocol
amendment, a new retention method, or a perception diagnostic.

## Fixed inputs and comparison

Use exactly the 864 saved BR-0.1 development parents at K=4/16/64, R=4, anchors
400/464/528/592, eight development source indices, and the original six-query
objective. Reconstruct each candidate from its saved valid-observation ordinal.
Verify all original candidate losses before interpretation. Compare the original
policy's chosen K-item subset with the exact same-parent oracle's K-item subset:
same input candidates, reader, recent context, horizon, loss and memory budget.
This is retrospective input-deletion accounting, not an online policy rollout.

Use the existing R=.25 and R=.5 noise-mixture archives only after their hashes,
source identities, sample order, and actual O/W/key arrays agree. Recover the
same pre-injection residual C and injected component E algebraically, without
restoring data or generating new noise:

`C=(R25-sqrt(.5)*R50)/(sqrt(.75)-.5); E=sqrt(2)*R50-C; U=sqrt(.5)*C; N=sqrt(.5)*E`.

These are standardized slow-model residual components, not physical noiseless
ground truth. Missing Z/actions/pose/RGB remain missing. For each frozen selected
subset apply its original masked-mean weights to U and N. The exact realized
loss identity is `(pU-U)^2 + (pN-N)^2 + 2*(pU-U)*(pN-N)`. No conditional expected
noise formula is used because selection can depend on observed noise.

## Fixed diagnostics

1. For all parents, enumerate original losses and each signal/noise/cross part;
   report chosen-minus-oracle decomposition, zero-memory reference, coverage,
   and whether original oracle decisions remain optimal for the signal part.
   No signal-only prediction is treated as a deployable policy.
2. Reconstruct actual future conditional history attention, using the fixed
   BR-0.1 max-over-future-query rule. Compare its deletion with the same-parent
   task oracle, reporting nonempty coverage and tie-aware optimal sets.
3. Describe the oracle-deleted item's age and pre/post-splice identity. A label
   "pre-splice" is evaluation-only; it is not proof that age caused error.
4. Check exact collisions of the allowed bounded bank state (candidates,
   recent, timestamps and current decision time) and, separately, the entire
   available causal residual/mask/key prefix. Contrast future loss vectors only
   inside identical-state groups; absence of collisions does not establish
   causal learnability. No student or utility scorer is fitted.
5. Retain the previously selected maximum-regret K16/PaFu/CHANGE/source31/t464
   case; also report every parent and aggregate, not only that case. Link to the
   old A2 object-cost and signal/noise findings without conflating representations.

No threshold tuning, reseeding, new checkpoint, FINAL/test access, GPU allocation,
old result overwrite, or existing V6 gate edits. This is one diagnostic round
for the already observed attention-versus-utility residual-bank case. Aggregate
parent rows repeat policy/K choices on eight source clusters; do not treat 864
parents as independent trials. Per-source bootstrap intervals are descriptive,
with adjacent CHANGE-source dependence explicitly retained as a limitation.
