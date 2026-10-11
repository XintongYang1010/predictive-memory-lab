# V5.7 archive recovery and verification

The Release has one complete scientific archive and two metadata companions:

- `visual-v5_7-20261010-complete.tar.gz` — 300,422,555 bytes; SHA256 `4333c8fd061eac96d8092a4b73246aede0a2c87ab90215f6603692cbf8ad6cac`.
- `asset_identity.json` — archive and manifest identity.
- `report_revision.json` — the corrected report text/generator and subsequent web review. A prose review clarified that raw framewise L2 has a small favorable point estimate with a CI spanning zero. No scientific values, arrays, source design or plots changed. The corrected report in Git is authoritative; the companion binds both report versions by SHA256.

The tar contains 382 regular files plus `archive_inventory.json`: all new predictions, costs, 256 FP64 norm shards, software/source/configuration, plots, CSVs, run logs and independent checks. Archive prefixes are `outputs/cost/`, `outputs/descriptor/`, `source/`, `execution/`, and `report/`. Verify every entry against the manifest before extracting into a fresh directory.

Prior inputs are deliberately referenced rather than duplicated. V5.5 scene manifest SHA256 is `b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663`; its Release provides `visual-v5_5-20261010-remote-scene.tar.gz.part000` through `part005` and an `archive_parts.json`. V5.6's Release provides its complete archive as `part000` and `part001` plus its own parts inventory. Reassemble each in numeric order and verify the part and whole-archive hashes in the corresponding inventory. Keep the two releases in separate directories.

Restore the V5.5 scene at `execution/visual-v5_5-20261010/diagnostic-20261010-v55/`. Restore V5.6 `local/correspondence/results/` at `execution/visual-v5_6-20261010/correspondence/results/`; the required cost NPZ is SHA256 `265766a0d1de852f5eb863c068d08de11c696ed961e76768468797a6e9879eb0`. Restore this run's `execution/registration_publication.json` and `execution/reproduce.sh` under `execution/visual-v5_7-20261010/`. Checkout frozen source `4375e57c8e5f736a43ffae5d4ce74393555d41b9`, then run `bash execution/visual-v5_7-20261010/reproduce.sh NEW_OUTPUT_DIRECTORY`. The scientific scripts verify their frozen source, protocol, input and publication identities.

No new trained checkpoint exists. Original encoder/model checkpoints remain in the prior releases/reference provenance. There was no new Slurm step, GPU charge, remote directory or remote cleanup in this CPU-only round. Local data and prior remote content are preserved.

See `github_verification.json` for the independent full download, all member hashes, metadata companions and Git report readback. Final verification and this explanatory note are later Git artifacts rather than recursively included in the archive they verify.
