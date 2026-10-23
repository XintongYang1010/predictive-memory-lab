# Reproducing the saved BR-0.1 development result

This document was added after the run to describe artifact availability. It does
not amend the frozen experimental protocol or source snapshots in the result.

The code depends on NumPy and CPU PyTorch plus two modules from the separate
`JasonLIN0226/CV_project` repository. Pass that checkout explicitly with
`--cv-root /absolute/path/to/CV_project`. The exact module bytes used are:

| CV path | SHA-256 |
| --- | --- |
| `experiments/phase2/literature_baselines.py` | `e83a4a46eb661601f37d2aca02e2ccad7c6d7179bb0f5675fb3914852f8ccc0d` |
| `experiments/phase2/memory.py` | `9f5dfe5aaa3052a2af4a9c79a2f8c5558be258feff1459ea82f2d86e66da073c` |

Both are also copied under the saved result's `source/` directory for audit.
Those renamed copies are evidence, not an automatically importable replacement
for the CV package. The runner records the actual imported file paths and
freezes their hashes before fitting, then checks they did not change.

The three existing local development inputs were not tracked in this checkout.
A fresh Git clone alone therefore cannot execute the default input paths:

| Required input | Bytes | SHA-256 |
| --- | ---: | --- |
| `a2_frozen_inputs.npz` | 6,204,297 | `c256d8ee6b80e422c5ddc6d6e120b3eb3bfb8272a9083f3b64ca5683db9700b3` |
| `input_manifest.json` | 17,122 | `108563c48bdc528ea2e6e8985494c06d220b58b5e37e024f49a6fd0c63bc2b50` |
| `reference_rows.jsonl.gz` | 2,098,776 | `b014a53d0a6b0c23be0eebc288c6561adac98d1dd63a70311ca6eaf83918d807` |

All three inputs, source snapshots, output JSON, plots, test log and the three
final student checkpoints are included in the locally verified archive:

- `results/br-0.1-dev-seed17-evidence.tgz`
- 8,679,258 bytes; 30 members.
- SHA-256: `719643d1fba4353810f3582a336fe9d288fe4a8448ba8a29ce0e59ea56181d9a`.

The archive is a local evidence artifact; this document does not claim it was
uploaded or published. Student `.pt` files are ignored by the repository's
existing rules. The input hashes and archive member hashes are recorded in
`results/br-0.1-dev-seed17/artifact-manifest.json`; the archive verification
receipt is alongside it. The archive does **not** contain full L0 Z/Y/action/RGB
arrays: it cannot support a complete L0 or visual-world-model reproduction.

After obtaining and extracting the verified archive into a separate directory,
use a new empty output directory. For example, with shell variables set to the
actual extraction and CV-checkout locations:

```bash
python3 experiments/bank_reference/run.py \
  --input "$BR_EVIDENCE/inputs/a2_frozen_inputs.npz" \
  --manifest "$BR_EVIDENCE/inputs/input_manifest.json" \
  --references "$BR_EVIDENCE/inputs/reference_rows.jsonl.gz" \
  --cv-root "$BR_CV_ROOT" \
  --output "$BR_OUTPUT"
```

No original V6 gate or FINAL split is opened. The run first reproduces the
archived A1/A2/A6 sums, then fits seed 17 on the frozen purged training split
and evaluates the separate development sources. See `PROTOCOL.md` for the
reader, teacher, oracle, budget and claim limitations. Numeric reproducibility
across different NumPy/PyTorch versions or hardware is not guaranteed bitwise.
