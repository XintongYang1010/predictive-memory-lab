# Reproduce the CPU follow-up

Run from the repository root on the published diagnostic commit. The actual
environment and test result are in [validation/environment.json](validation/environment.json).
These commands use CPU only. Torch, NumPy, Pillow and Matplotlib are required.
The original project declares Python >=3.11; the recorded diagnostic run used
the available Python 3.9.6 runtime and passed the applicable 62 tests there.

Set `V5_CACHE` to the preserved original `run1/probe_features` directory, or a
local copy. It must contain `train/manifest.json`, `dev/manifest.json` and all
256 referenced `.pt` tensors. Their total size is 655,082,496 bytes. These tensors
are not duplicated in Git: they are retained in the original remote workspace.
The published original cache manifests record all file hashes; the follow-up
verified every hash and size before reading the tensors. Do not substitute new
features, regenerated data, or a different checkpoint for this reproduction.

```sh
export V5_CACHE=/path/to/original/probe_features
export V5_REPRO=/path/to/new-empty-reproduction-directory
export V5_ARCHIVE=results/wm/visual-v5-dgx-20261009
mkdir -p "$V5_REPRO"
python3 scripts/diagnose_visual_probe_v5.py \
  --cache "$V5_CACHE" --out "$V5_REPRO/core_probe" --threads 4
python3 scripts/audit_visual_probe_inputs_v5.py \
  --cache "$V5_CACHE" --gate "$V5_ARCHIVE/run/feature_gate.json" \
  --run-manifest "$V5_ARCHIVE/run/run_manifest.json" \
  --out "$V5_REPRO/input_audit" --threads 4
python3 scripts/diagnose_visual_probe_temporal_v5.py \
  --cache "$V5_CACHE" --core-results "$V5_REPRO/core_probe" \
  --out "$V5_REPRO/temporal_controls" --threads 4
python3 -m unittest discover -s tests/visual_memory -v
```

The core plan is fixed in `scripts/visual_probe_diagnostic_plan.json` and copied
into the output. Later temporal controls are expressly exploratory, designed
after the core results. All hyperparameter/frame selection uses grouped Train
CV. No script needs Test, new GPU inference, or a training checkpoint.

The generator audit supports `--out` to a new directory and reads the frozen
source in the original archive:

```sh
python3 results/wm/visual-v5-probe-diagnostic-20261009/generator_audit/generator_analysis.py \
  --out "$V5_REPRO/generator_audit"
```

The independent Torch reference, static-control script and figure renderer are
also published alongside their outputs. They write beside themselves by
default; copy the diagnostic result directory before rerunning them if you
want to preserve its recorded bytes. Numerical comparisons should tolerate
floating-point library differences; elapsed times and process RSS will vary.
Archive integrity checks are separate from a rerun of the scientific analysis.
