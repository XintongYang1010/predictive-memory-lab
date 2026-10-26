# BR-0.1: independent frozen-residual bank development reference

The user approved implementing the separately proposed bounded bank baseline
route on 2026-10-22. This is a new development protocol, not an amendment or a
rerun of V6.1/BF/A2 gates. No existing preregistration, threshold, checkpoint,
data file, or FINAL/test split is modified. No GPU or raw-data restoration is
needed. Negative and missing evidence are reported explicitly.

## Available data and claim boundary

The available `a2_frozen_inputs.npz` has 32 streams each of EXACT, NEWPATH and
CHANGE, 799 transitions, residual R[34], observed-coordinate O[34], write mask W,
oracle cell×octant key, and frozen A2 gates. It does **not** contain Z, original
observations, actions, continuous pose, RGB, or Y. The accompanying manifest
has hashes for Z/EPS but these arrays were not saved. This experiment is an
**L0-derived frozen-residual diagnostic**, not a complete L0 world-model or
visual memory evaluation. The loader checks available fields, not manifest
claims alone. Explicit replacement NPZ/manifest paths are supported; supplying
additional fields does not automatically upgrade this diagnostic's claims.

## Frozen scope, before results

- Seed 17; recent observations R=4; historical K in {4,16,64}; one final model
  per K, 100 epochs, hidden width 32, learning rate .001, shared PaFu helper.
- Ordered original A indices 0..22 train; 24..31 development; index 23 excluded
  in all three conditions. CHANGE uses A_i→A_(i+1): removing the boundary makes
  the union of A/B source identities disjoint. This gives 69 train, 24 dev,
  and 3 excluded streams. This is reused exploratory dev, not confirmation.
- Training candidate sets come from FIFO after observations at original t=
  400,464,528,592, with actual future query horizon 6. No dev labels enter fit.
  Empty future-attention examples are skipped and coverage reported. No sweeps,
  validation-selected epoch, reseeding, or mechanisms added after results.
- All online arms read and score **before** observing the current residual.
  W=0 splice transitions are neither stored nor scored. Policy timestamps count
  valid observations; metric windows retain original transition indices.
- Arm names: FIFO, WR-inspired, PaFu-inspired. A1/A2/A6 are separately reported
  matrix-state references; first reproduce their archived POST/DET sums.

## Common reader and mechanisms

An item is float32 `[R[34], O[34], key/647]`, plus ID/time metadata. The reader
matches the current already-known oracle key, averages previously observed
residual coordinates using their masks, and predicts zero where none exist.
Its attention is the actual per-coordinate masked mean, not learned attention.
Recent and historical entries are disjoint. No retained item can be resurrected.
The scalar key/647 is an ordinal descriptor, not a learned geometric embedding;
its adequacy is an adaptation limitation. A2's frozen gate originally used full
Z, now absent: replaying it is not a deployable input-parity bank comparison.

WR uses the shared causal Gaussian/standardized-L1, past-only median/MAD, delayed
peak, and causal NMS implementation. It uses only the resident previous four
features plus a bounded scalar score ring; this is not the paper's visual system.

PaFu's student sees only candidate/current/recent item features and age. Offline
teacher labels use the fixed reader's actual attention on **future queries**
q+1..q+6, with the same frozen candidate and recent parent. We condition those
weights on long-memory candidates and omit queries with zero long-memory mass.
The per-key timestamp grouping helper is used, but all these legal queries are
future relative to old keys: past/same groups are unavailable. Thus this is a
future-only conditional-attention proxy, not a reproduction of PaFu's three-group
DiT teacher. No future residual/error or deletion utility enters teacher labels.
Probability SmoothL1 and normalization are declared adaptations. Report label
nonuniformity and coverage; a degenerate teacher is a limitation, not success.

## Budgets and oracle

Within each K every arm has the same declared reserved envelope: `(K+4)*288`
bytes for float32 items/4-byte IDs/8-byte timestamps, plus 16,384 bytes for
controller state. Unused reservation is not actual allocation. Actual logical
controller bytes, item bytes, weights, and shared lookup bytes are also reported;
Python allocator/RSS and temporary workspaces are explicitly excluded. The
PaFu model's weights are reported separately and not disguised as equal compute.
Matrix references have 22,680 float32 dynamic scalars (90,720 bytes), different
representation and compression, and are not called equal-byte bank comparisons.

At the four fixed anchor times, enumerate every one-item deletion from the SAME
causal parent's K+1 history candidates; recent R is identical. Score the frozen
bank on the next six true queries/residuals with no intervening writes. This
is an exact **local frozen-parent deletion oracle**, not a globally optimal
online policy. Report chosen loss, minimum loss, regret, positive-regret fraction,
and candidate diagnostics. The all-observations causal mean is a separately
labelled reference without the K-item constraint (implemented using 352,512
bytes of float64 per-key sums/counts, not an observation archive). OC/OL are not
reused as selection oracles.

POST=[400,600), DET=[400,420); error is observed-coordinate SSE/count. Report
each condition and paired source-cluster uncertainty on held-out development
episodes; do not reinterpret development reuse as independent generalization.
If both banks and the oracle fail, the reader/data interface may be limiting.
If the oracle gap is absent, stop this diagnostic without inventing a mechanism.
