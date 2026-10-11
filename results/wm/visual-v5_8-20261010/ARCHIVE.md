# V5.8 archive recovery and completed verification

The scientific archive and identity companion were independently downloaded from [the Release](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_8-20261010). All 110 archive members and 39 report files in Git were verified, including every regular member SHA256.

- Archive: `visual-v5_8-20261010-complete.tar.gz`
- Size: 28,642,922 bytes
- SHA256: `b018b6f4ceff3e23a9fd80e1f016e2f799f4282d11594352be648e61e7918df7`
- Inventory SHA256: `9f6f7745ebd1c206182762aaebfad714298810220c00f301b2edbe5c1d0616b0`
- Result commit verified: `56dbcf46702a7078aaf4449566211fe520b9bf79`
- Frozen scientific source: `980a194adf503bbf6b64ce0ce4b9f66bc066ab86`

There are 109 regular content files plus the inventory. Prefixes are `outputs/spatial/`, `source/`, `execution/` and `report/`. This includes complete costs, predictions, clamp flags, source ROIs, bootstrap samples, identities, all figures, CSVs, source/design, reproduction commands, software/interpretation/result audits and logs. No new training checkpoint exists because no training was performed.

## Reproduction

Prior raw/model inputs are referenced by exact hashes instead of duplicated. Download V5.5 `visual-v5_5-20261010-remote-scene.tar.gz.part000` through `part005` with its `archive_parts.json`; download V5.6's complete archive `part000` and `part001` with its own inventory into a separate folder. Verify every part, reassemble in numeric order and verify the complete archives before restoring.

Restore the V5.5 scene under `execution/visual-v5_5-20261010/diagnostic-20261010-v55/`. Its manifest SHA256 is `b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663`. Restore V5.6 `local/correspondence/results/` at `execution/visual-v5_6-20261010/correspondence/results/`; its cost NPZ SHA256 is `265766a0d1de852f5eb863c068d08de11c696ed961e76768468797a6e9879eb0`.

Checkout frozen source `980a194adf503bbf6b64ce0ce4b9f66bc066ab86`. Restore this archive's `execution/registration_publication.json` and `execution/reproduce.sh` under `execution/visual-v5_8-20261010/`. Run `bash execution/visual-v5_8-20261010/reproduce.sh NEW_OUTPUT_DIRECTORY` from the repository root. The output directory must not already exist. The script verifies frozen source, protocol, design and all scientific input identities before scoring. The environment is recorded in `environment.json`.

## Status and later records

Both the scientific run and independent result verification passed. The first archive download was attempted before asynchronous upload of the large asset had finished; only the metadata companion was available. That attempt and log remain preserved. After upload completion, a fresh full download passed every hash check. `archive_retry.json` and `verification-before-upload-complete.log` record the recovered ordering error; no scientific result or source changed.

The final download receipt, this recovery note, the recovered-download log and [the later web discussion](web_final_review.md) are Git records written after archive creation, rather than recursively included in the archive they verify. The web discussion proposes an unexecuted transport control and makes no changes to the scientific conclusions. Raw generated SVG/CSV bytes are preserved exactly, including their original serialization whitespace.

No new Slurm job/step, GPU charge or remote directory was created. Remote cleanup is not applicable. All local originals and prior remote content are retained.
