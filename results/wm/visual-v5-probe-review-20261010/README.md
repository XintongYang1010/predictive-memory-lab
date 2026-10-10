# V5 design review — no new scientific run

The final review incorporates diagnostic commit `ca365e1aa1026c51be0d23008b63650354c5bf31`, including matched single-frame controls and independent numerical checks. It is based on frozen scientific source `d0b4fef25e98eefb58bfba737997cb0145f27daa`.

**Decision:** the original gate remains failed. Ridge selection affects predictive scores, but pooled positive displacement skill has not established use of temporal evidence. No new implementation error explaining the failure was identified in the audited path. No P0, memory comparison, GPU job or Test access is authorized by these review artifacts.

- [Final review and falsifiable fresh-confirmation protocol](../../../docs/V5_PROBE_REVIEW_20261010.md)
- [Machine-readable protocol](../../../scripts/v5_temporal_confirmation_plan.json)
- [CPU generator/kernel/scoring primitives](../../../scripts/v5_temporal_confirmation.py)
- [Software validation receipt](review_validation.json)
- [Independent archive-only audit](archive_audit.json)
- [Archive audit reproduction code](../../../scripts/audit_v5_archived_predictions.py)

The archive-only audit was completed before the follow-up diagnostic updates arrived. Its inputs were the archived prediction CSV and regenerated original Train/Dev RGB/targets; it did not fit cached I-JEPA features. Its pair-bootstrap intervals use a different explicitly recorded seed from the local Codex audit, so small interval differences are expected. The follow-up cache calculations were not repeated here.

Fresh FIT/confirmation/bridge images and features have **not** been generated. The new sidecar's software tests use a distinct `software_test` namespace only. The primary control uses four ordered views sharing both terminal and initial anchors; a last-frame-only function cannot pass the terminal-anchor contrast, and a first-frame-only function cannot pass the initial-anchor contrast. Both must pass for the same frozen reader.

No existing scientific source or archived result is edited by this branch. Main, the original results branch and the diagnostic branch are left untouched.
