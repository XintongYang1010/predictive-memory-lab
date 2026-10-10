#!/usr/bin/env python3
"""Re-render only the three worst already-open bridge worlds, posthoc labels."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo))
from pmr.visual_memory.data import episode

scene = repo/"execution/visual-v5_1-20261010/continuation-20261010"
out = Path(__file__).resolve().parent
assert (scene/"bridge_report/report.json").is_file()
freeze = json.loads((scene/"reader/freeze_receipt.json").read_text())
source = repo/"pmr/visual_memory/data.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == freeze["cache_identity"]["source_identity"]["pmr/visual_memory/data.py"]
with np.load(scene/"bridge_report/predictions.npz", allow_pickle=False) as z:
    target, prediction = z["targets"], z["concat"]
ranking = list(csv.DictReader((out/"worst_bridge_worlds.csv").open()))[:3]
assert [int(row["group_id"]) for row in ranking] == [380, 334, 356]
fig, axes = plt.subplots(6, 2, figsize=(10, 23))
checks = []
for world, selected in enumerate(ranking):
    pair = int(selected["group_id"])
    shard = json.loads((scene/"bridge_cache/shards"/f"{pair:06d}.json").read_text())
    for variant in (0, 1):
        ep = episode("dev", pair, variant)
        np.testing.assert_array_equal(ep.probe_state, target[pair-128, variant])
        check = dict(pair=pair, variant=variant, targets_exact=True, cue_hashes_exact=[])
        yellow_centers = []
        row = world*2+variant
        for col, t in enumerate((6, 7)):
            index = np.flatnonzero(ep.times == t)
            assert len(index) == 1
            rgb = ep.rgb[index[0]]
            image_identity = dict(sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                                  dtype=str(rgb.dtype), shape=list(rgb.shape))
            assert image_identity == shard["inputs"]["cue_rgb"][variant*2+col]
            check["cue_hashes_exact"].append(image_identity["sha256"])
            yy, xx = np.nonzero(np.all(rgb == [240, 220, 50], axis=2))
            assert len(xx) > 0
            yellow_centers.append(float(xx.mean()))
            axes[row, col].imshow(rgb)
            axes[row, col].set_title(f"Known bridge pair {pair}, variant {variant}, t={t}", fontsize=10)
            axes[row, col].axis("off")
        truth = np.array2string(target[pair-128, variant, 4:], precision=4)
        pred = np.array2string(prediction[pair-128, variant, 4:], precision=4)
        yellow_pixel_dx = yellow_centers[1]-yellow_centers[0]
        np.testing.assert_allclose(yellow_pixel_dx/224, float(selected["yellow_dx"]), atol=1e-15)
        check["yellow_pixel_dx"] = yellow_pixel_dx; checks.append(check)
        axes[row, 0].text(1.05, -.035, f"[A dx, A dy, B dx, B dy]  target={truth}\nfrozen prediction={pred}; yellow dx={yellow_pixel_dx:+.0f}px ({float(selected['yellow_dx']):+.4f})",
                          transform=axes[row, 0].transAxes, ha="center", va="top", fontsize=9)
fig.suptitle("Worst three paired worlds by original bridge motion MSE\nPosthoc labels; all 12 cue-image hashes match the sealed bridge shards", fontsize=12)
fig.subplots_adjust(top=.94, bottom=.045, left=.08, right=.92, hspace=.46, wspace=.10)
fig.savefig(out/"worst_three_bridge_worlds.png", dpi=150)
plt.close(fig)
(out/"bad_case_render_verification.json").write_text(json.dumps(dict(posthoc_only=True,
    selection="top three descending whole-pair concat motion MSE, ascending pair-ID tie break", checks=checks,
    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+"\n")
