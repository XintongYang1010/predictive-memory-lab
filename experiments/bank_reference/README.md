# Independent bank references (BR-0.1)

This path is isolated from the original V6/BF gate protocol. Read
[PROTOCOL.md](PROTOCOL.md) for the authorized scope and frozen development split.
The executable uses the shared causal WR/PaFu adapters from the sibling
`CV_project` repository, with an explicit `--cv-root` override when needed.

```bash
python3 experiments/bank_reference/run.py --inventory-only
python3 -m unittest discover -s experiments/bank_reference -p 'test_*.py' -v
python3 experiments/bank_reference/run.py
```

The default output refuses to overwrite any prior run. CPU/NumPy/PyTorch only;
no package installation, network download, scheduler operation, original-gate
execution, or FINAL/test access is performed. The output freezes source hashes,
input identities, split and settings before fitting, and verifies source files
are unchanged afterward. The original A1/A2/A6 sums must reproduce first.

Available inputs are residual/mask/key/gate arrays, not complete L0 tokens.
`--input`, `--manifest`, and `--references` accept explicit paths to the same
frozen schema. A future complete-L0 adapter needs aligned `Z` (causal previous
observations, action and known pose), `Y`, observation masks, timestamps/episode
identities, keys, frozen predictor parameters and normalization provenance.
RGB is additionally needed for visual bad-case panels. Manifest-only hashes
cannot substitute for actual arrays. This runner deliberately continues to call
its scope a residual diagnostic even if additional unused arrays are supplied.

Outputs include held-out development MPE and paired exploratory intervals,
same-parent local deletion oracle regret, top key/time bad-case traces,
teacher-attention traces, final students, actual/reserved logical-byte accounts,
and a human-readable `REPORT.md`. A2 is replayed from frozen full-Z gates; its
input advantages are stated rather than called a matched bank comparison.
