# Replay the archived V5.4 results

The full Release includes the exact source, union N/W model, all three new caches and per-input receipts, reports, logs, ledger, posthoc figures and analysis tools. The separately verified V5.3 Release contains the original narrow FIT cache; the complete N feature basis and labels also live in the V5.4 frozen model. The 10 GB official I-JEPA checkpoint is referenced by immutable SHA/source in dependencies.json rather than duplicated. No Test was accessed.

Use a checkout of the V5.4 results commit (the full-scene Release target), preserving any unrelated local modifications. Python3.11/NumPy2.4.6/Pillow12.3 were used remotely. Local replay used Python3.9.6/NumPy2.0.2/Pillow11.3; source software tests passed and all6 bad-case raw RGB hashes matched exactly. Replaying summaries and existing FIT needs NumPy/Matplotlib/Pillow but no CUDA or encoder weights. Original requirements are in the source tree. Numerical validation of saved training predictions is in training_generalization/training-generalization-numeric-verification.json.

From the repository root, use a new empty directory under execution/ so the historical helpers resolve the correct source root:

```sh
set -eu
mkdir -p execution
REPLAY_DIR=$(mktemp -d "$PWD/execution/v54-replay.XXXXXX")
RESULTS_DIR="$PWD/results/wm/visual-v5_4-20261010"
cp "$RESULTS_DIR/archive_tools/"*.py "$REPLAY_DIR/"
RESULT_COMMIT=$(gh api repos/XintongYang1010/predictive-memory-lab/releases/tags/visual-v5_4-20261010 --jq .target_commitish)
python3 "$REPLAY_DIR/verify_github.py" \
  --commit "$RESULT_COMMIT" \
  --source-commit 3279bbe49f8db5ec1bd4b2f5c343db1ce9e5684b \
  --out "$REPLAY_DIR/github-readback" \
  --conclusion 'Independent replay of the archived mixed result; no new experiment.'
# Extraction occurs only after every downloaded archive member has been verified.
tar -xzf "$REPLAY_DIR/github-readback/visual-v5_4-20261010-remote-scene.tar.gz" -C "$REPLAY_DIR"
SCENE_DIR="$REPLAY_DIR/diagnostic-20261010-v54"
python3 "$REPLAY_DIR/build_results.py" \
  --scene "$SCENE_DIR" --out "$REPLAY_DIR/summary-replay" \
  --plan scripts/v5_4_plan.json
python3 "$REPLAY_DIR/posthoc_bad_cases.py" \
  --scene "$SCENE_DIR" \
  --verification "$RESULTS_DIR/report-snapshot-verification.json" \
  --generator pmr/visual_memory/data.py --out "$REPLAY_DIR/bad-case-replay"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
VECLIB_MAXIMUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
python3 "$REPLAY_DIR/training_generalization.py" --execute \
  --model "$SCENE_DIR/reader" --publication "$SCENE_DIR/freeze_publication.json" \
  --evaluation-controller "$SCENE_DIR/evaluation_controller.json" \
  --anchor-report "$SCENE_DIR/anchor_report/report.json" \
  --bridge-report "$SCENE_DIR/bridge_report/report.json" \
  --out "$REPLAY_DIR/fit-replay"
```

The snapshot-verification receipt is additionally published as a Git artifact; its inventory binds the unchanged pre-analysis reports and input shards inside the complete scene. Figure files are rendered outputs; compare recomputed scientific metrics/identities, not new plot timestamps or environment-dependent PNG metadata. Existing outputs are never overwritten.

Historical deployment, resource query, actual commands, official weight/source hashes, registration and publication ordering are retained in the full scene. A new GPU experiment needs a new directory, allocation identity, budget and freshness registration; do not rerun historical launch helpers against 415453 or replace any preserved scene. These replay commands perform no scientific fitting, model selection, new evaluation sampling or Slurm work.
