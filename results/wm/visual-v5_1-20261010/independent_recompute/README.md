# Independent frozen-reader replay

Confirmation passes; original-distribution bridge fails. P0 remains unauthorized, original V5 gate remains failed.

Verified the original 1739 files (274246065 bytes) before and after replay; the scene remained unchanged.
All three role caches contain exactly256 canonical groups; their arrays exactly match every checked shard. The frozen basis equals FIT cache tensors and IDs. Eight distinct complete GPU-worker receipts were verified per role.

| Role | Primary decision | Max prediction absolute difference | Max report-number absolute difference |
|---|---|---:|---:|
| confirm | passes | 2.05e-15 | 8.88e-16 |
| bridge | fails | 2.93e-15 | 3.55e-15 |

Prediction/score recomputation used the saved coefficients and moments only. Fit, selection and data-generation functions were disabled; only the original bootstrap seed7709 was permitted. No Test or GPU access occurred.

- This is numerical replay of the same frozen artifacts, not a new independent scientific sample.
- Input RGB hashes and freshness attestations are audited as archived provenance; no inputs were regenerated.
- Cross-platform BLAS can differ in last floating-point bits; prediction tolerance1e-9 and report tolerance1e-8 were fixed before replay.
- Bootstrap intervals reuse the registered seed and remain conditional on the frozen reader and registered sample.
