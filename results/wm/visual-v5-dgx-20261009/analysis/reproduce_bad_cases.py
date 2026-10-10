#!/usr/bin/env python3
"""Recreate selected Dev cue illustrations from archived source and predictions."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
DATA = ARCHIVE / "source/pmr/visual_memory/data.py"
PREDICTIONS = ARCHIVE / "run/feature_diagnostics/dev_predictions.csv"


def main():
    spec = importlib.util.spec_from_file_location("archived_v5_data", DATA)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with PREDICTIONS.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["mode"] == "pooled" and int(row["dimension_index"]) >= 4]
    by_pair = {}
    for row in rows:
        if not row["episode_id"].startswith("dev-") or not row["pair_id"].startswith("dev-"):
            raise ValueError("Only Dev cases may be illustrated")
        by_pair.setdefault(row["pair_id"], []).append(row)
    scores = {pair: max(float(row["squared_error"]) for row in values) for pair, values in by_pair.items()}
    selected = sorted(by_pair, key=lambda pair: (-scores[pair], pair))[:3]
    figure, axes = plt.subplots(3, 2, figsize=(13, 13))
    figure.subplots_adjust(top=.93, bottom=.16, left=.045, right=.985, hspace=.80, wspace=.12)
    figure.suptitle("Largest pooled displacement errors: three distinct Dev pairs", fontsize=16, y=.985)
    figure.text(.5, .955, "Each panel: t=6 on the left, t=7 on the right. Values are recorded probe targets and predictions.",
                ha="center", fontsize=10)
    exported, summaries = [], []
    maximum_target_difference = 0.
    for rank, pair_id in enumerate(selected, 1):
        pair = int(pair_id.split("-")[1])
        worst = sorted(by_pair[pair_id], key=lambda row: (-float(row["squared_error"]),
                       int(row["variant"]), int(row["dimension_index"])))[0]
        summaries.append((rank, pair_id, scores[pair_id], worst))
        for variant in (0, 1):
            episode = module.episode("dev", pair, variant)
            indices = [int(np.flatnonzero(episode.times == t)[0]) for t in (6., 7.)]
            if indices != [6, 7] or not episode.shutters[indices].all():
                raise ValueError("Expected fully visible original cue frames at t=6 and t=7")
            values = sorted([row for row in by_pair[pair_id] if int(row["variant"]) == variant],
                            key=lambda row: int(row["dimension_index"]))
            if [int(row["dimension_index"]) for row in values] != [4, 5, 6, 7]:
                raise ValueError("Missing or duplicated displacement dimensions")
            annotation = ["         target      prediction", ""]
            for row in values:
                dim = int(row["dimension_index"])
                difference = abs(float(episode.probe_state[dim]) - float(row["target"]))
                maximum_target_difference = max(maximum_target_difference, difference)
                if difference != 0:
                    raise ValueError(f"Redrawn episode differs from archived target: {pair_id}/{variant}/{dim}")
                target, prediction = float(row["target"]), float(row["prediction"])
                if not np.isclose((prediction - target)**2, float(row["squared_error"]), rtol=1e-12, atol=1e-15):
                    raise ValueError("Prediction/target squared error differs from archived table")
                exported.append(dict(pair_rank=rank, pair_maximum_squared_error=scores[pair_id],
                    is_pair_maximum_error=float(row["squared_error"]) == scores[pair_id], **row))
                label = ("A dx", "A dy", "B dx", "B dy")[dim-4]
                marker = "*" if float(row["squared_error"]) == scores[pair_id] else " "
                annotation.append(f"{marker}{label:4s}  {target:+.6f}    {prediction:+.6f}")
            axis = axes[rank-1, variant]
            axis.imshow(np.concatenate([episode.rgb[index] for index in indices], axis=1))
            axis.axvline(223.5, color="white", linewidth=2)
            axis.text(.25, .98, "t = 6", ha="center", va="top", transform=axis.transAxes,
                      color="white", bbox=dict(facecolor="black", alpha=.7, edgecolor="none"), fontsize=9)
            axis.text(.75, .98, "t = 7", ha="center", va="top", transform=axis.transAxes,
                      color="white", bbox=dict(facecolor="black", alpha=.7, edgecolor="none"), fontsize=9)
            axis.set_title(f"#{rank}  {pair_id} · variant {variant} · {episode.metadata['family']}", fontsize=11)
            axis.axis("off")
            axis.text(.04, -.10, "\n".join(annotation), transform=axis.transAxes, va="top",
                      fontfamily="monospace", fontsize=10, linespacing=1.18)
    figure.text(.5, .017, "* = pair's largest scalar squared error. Observed displacement is position(7) − position(6).\n"
                "These are selected failure examples, not causal diagnoses or fresh evaluation samples.",
                ha="center", fontsize=10)
    figure.savefig(HERE / "cue_frames.png", dpi=170)
    plt.close(figure)
    with (HERE / "bad_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0])); writer.writeheader(); writer.writerows(exported)
    notes = ["# Selected pooled-feature motion failures", "",
             "Selection is deterministic: retain only pooled displacement dimensions 4–7; score each Dev pair "
             "by its **maximum scalar squared error over both variants and all four displacement dimensions**; "
             "sort descending by that score, break ties by pair ID, and select the first three distinct pairs. "
             "Both variants and all four displacement targets are retained for every selected pair.", "",
             "| Rank | Dev pair | Pair maximum squared error | Worst episode | Worst dimension | Target | Prediction |",
             "|---:|---|---:|---|---|---:|---:|"]
    for rank, pair, score, worst in summaries:
        notes.append(f"| {rank} | {pair} | {score:.9g} | {worst['episode_id']} | {worst['dimension']} | "
                     f"{float(worst['target']):+.7f} | {float(worst['prediction']):+.7f} |")
    notes.extend(["", "![Reconstructed Dev cue frames and actual probe values](cue_frames.png)", "",
                  "The images reconstruct the archived generator's Dev episodes at t=6 and t=7. Object A is "
                  "the upper red target; object B is the lower blue target. Both are visible in both cue frames. "
                  "The columns labeled target/prediction use the actual archived ridge-probe values, not values "
                  "estimated from the illustration. Targets are normalized position differences after reflection. "
                  "The asterisk marks each pair's largest scalar error.", "",
                  f"Validation: all {len(exported)} displayed/exported displacement targets exactly equal "
                  "the regenerated `probe_state[4:8]` values; maximum absolute difference is "
                  f"**{maximum_target_difference:g}**. Recomputed squared errors agree with the CSV. "
                  "Cue times are exactly 6 and 7, with both shutters open. Only the selected Dev pairs are regenerated; "
                  "Test is not read or generated.", "",
                  "These deliberately selected examples illustrate observed errors and their paired counterparts. "
                  "They cannot establish a pooling mechanism, reflection failure, or a comparison of memory architectures. "
                  "No scientific metric or gate has been modified.", "",
                  "[All 24 selected displacement records](bad_cases.csv) · [Reproduction script](reproduce_bad_cases.py)", "",
                  "Reproduce from this directory with `python reproduce_bad_cases.py`; it locates `../source/` "
                  "and `../run/feature_diagnostics/dev_predictions.csv` automatically.", "",
                  f"Archived generator SHA256: `{hashlib.sha256(DATA.read_bytes()).hexdigest()}`.", "",
                  f"Prediction CSV SHA256: `{hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest()}`.", ""])
    (HERE / "bad_cases.md").write_text("\n".join(notes))
    print(json.dumps(dict(selected_pairs=selected, exported_displacement_rows=len(exported),
                         maximum_target_difference=maximum_target_difference, test_opened=False), indent=2))


if __name__ == "__main__":
    main()
