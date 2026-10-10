# Reproduce the frozen V5.2 diagnostic

Use source commit `9792b93f4e0203b52a2e26ab7700eb1975150231`, or the matching `code/` tree in the Release asset. Verify the asset digest in `release_asset.json`, then verify every entry of `archive_manifest.json`. Do not overwrite a pre-existing run directory.

## CPU score replay

The current Release is sufficient for scoring; it includes the immutable reader checkpoint, counterfactual cache, receipt and original bridge predictions. From the extracted scene's `code/` directory, set `SCENE` to its absolute path and `REPLAY_OUT` to a new output directory outside that scene:

```sh
python3 scripts/score_v5_2_counterfactual.py \
  --cache "$SCENE/counterfactual_cache/cache.npz" \
  --receipt "$SCENE/counterfactual_cache/receipt.json" \
  --model "$SCENE/reader" \
  --prior-predictions "$SCENE/prior_bridge/predictions.npz" \
  --out "$REPLAY_OUT"
```

Keep the replay output outside the sealed scene if preserving its exact inventory. The output directory must not exist. The code validates source hashes, feature shapes, image-slot mapping, encoder identity and frozen reader before scoring. Compare metrics with `run/report/report.json`; platform-specific BLAS differences may affect final bits, while the preregistered numerical guards remain unchanged.

The actual remote runtime is fully listed in `runtime_packages.json`: Python3.11.15, Torch2.8.0+cu128, NumPy2.4.6, Pillow12.3.0. The generic repository `requirements.txt` still pins NumPy1.26.4 and does **not** describe this isolated DGX environment. Use the recorded runtime for exact image reconstruction. No shared training environment was modified. The local independent CPU statistics audit used NumPy2.0.2 and agreed within6.94e-18.

## Independent statistics and figures

The archived `independent_audit/recompute.py` recomputes bootstrap statistics without importing the scientific scorer. Run `--help` for its scene/out arguments. `report_tools/render_report.py --scene <scene> --out <new-directory>` independently checks saved predictions against the report before producing the figures. These posthoc scripts do not train, select, change gates or open Test. Their outputs have separate checksums.

## Full GPU extraction

Download and verify the [V5.1 Release](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_1-20261010) as well: full extraction validates the original bridge cache, original rendered-input hashes, prior ledger, and frozen reader against that sealed scene. The new archive's `dependencies.json` specifies the old manifest SHA. The exact8-worker `srun` commands are in `run/diagnostic_controller.json`; they reuse an existing authorized allocation and must be adapted to a currently valid allocation, not blindly replayed against job415453.

Official I-JEPA source commit: `52c1ae95d05f743e000e8f10a1f3a79b10cff048`. Official checkpoint: `IN1K-vit.h.14-300e.pth.tar`, SHA256 `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`. These external assets are referenced in the previous deployment documentation; the10GB weight is not duplicated. The same BF16 autocast / FP32 pooling / BF16 quantization path is required; FP32 storage alone does not mean FP32 inference.

Original and edited image batches follow the frozen protocol exactly. Keep all four image slots, two variants and four conditions together for each world. A new extraction needs a new directory, a copied/inherited cumulative budget and its own publication/identity receipt. It has no authorization to overwrite old failed gates or touch Test.
