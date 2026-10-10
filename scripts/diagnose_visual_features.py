#!/usr/bin/env python3
"""Offline Dev diagnostics for the unchanged V5 Train-only ridge feature gate."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmr.visual_memory.common import atomic_json, digest
from pmr.visual_memory.features import load_records


DIMENSIONS = ("A_x_t7", "A_y_t7", "B_x_t7", "B_y_t7",
              "A_dx_t6_to_t7", "A_dy_t6_to_t7", "B_dx_t6_to_t7", "B_dy_t6_to_t7")


def reconstruct(train, dev):
    """Exactly match features.probe, retaining its predictions for inspection."""
    train = [row for row in train if row["raw_cue"] is not None]
    dev = [row for row in dev if row["raw_cue"] is not None]
    for split, rows in (("train", train), ("dev", dev)):
        if not rows:
            raise ValueError(f"No complete raw/pooled probe records for {split}")
        for row in rows:
            if (row["metadata"]["split"] != split or row["probe_state"].shape != (8,)
                    or row["times"][6:8].tolist() != [6., 7.]):
                raise ValueError("Probe identity, target dimensions, or cue timestamps differ from V5")
    result, per_case, per_dimension = {}, [], []
    for mode in ("raw", "pooled"):
        def features(rows):
            return torch.stack([(row["raw_cue"] if mode == "raw" else row["features"][6:8])
                                .float().flatten() for row in rows]).double()
        a, b = features(train), features(dev)
        mean = a.mean(0); a = a - mean; b = b - mean
        scale = a.square().sum(1).mean().clamp_min(1e-12)
        y = torch.stack([row["probe_state"] for row in train]).double()
        target = torch.stack([row["probe_state"] for row in dev]).double()
        ym = y.mean(0); k = a @ a.T / scale
        coef = torch.linalg.solve(k + .01 * torch.eye(len(k)), y - ym)
        prediction = (b @ a.T / scale) @ coef + ym
        error = (prediction - target).square().mean(0)
        baseline = (target - ym).square().mean(0).clamp_min(1e-12)
        r2 = 1 - error / baseline
        result[mode] = dict(r2_per_dimension=r2.tolist(), mse=float(error.mean()),
                            position_r2=float(r2[:4].mean()), velocity_r2=float(r2[4:].mean()))
        for dimension, name in enumerate(DIMENSIONS):
            per_dimension.append(dict(mode=mode, dimension=name, dimension_index=dimension,
                target_kind="position" if dimension < 4 else "observed_displacement",
                r2=float(r2[dimension]), mse=float(error[dimension]),
                baseline_mse=float(baseline[dimension]), train_target_mean=float(ym[dimension]),
                train_episodes=len(train), dev_episodes=len(dev)))
            for index, row in enumerate(dev):
                metadata = row["metadata"]
                residual = float(prediction[index, dimension] - target[index, dimension])
                per_case.append(dict(mode=mode, episode_id=metadata["episode_id"],
                    pair_id=metadata["pair_id"], variant=metadata["variant"], family=metadata["family"],
                    gap=metadata["gap"], interference_writes=metadata["interference_writes"],
                    dimension=name, dimension_index=dimension, target=float(target[index, dimension]),
                    prediction=float(prediction[index, dimension]), residual=residual,
                    squared_error=residual ** 2))
    result["passed"] = result["pooled"]["position_r2"] > .1 and result["pooled"]["velocity_r2"] > 0.
    result["criterion"] = "pilot gate: held-out position R2>0.1 and velocity R2>0; inspect per-dimension values"
    return result, per_case, per_dimension


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def diagnose(out):
    out = Path(out).resolve()
    gate_path = out / "feature_gate.json"
    gate = json.loads(gate_path.read_text())
    cache = out / "probe_features"
    train, dev = load_records(cache, "train"), load_records(cache, "dev")
    reconstructed, cases, dimensions = reconstruct(train, dev)
    comparisons = []
    for mode in ("raw", "pooled"):
        for metric in ("r2_per_dimension", "mse", "position_r2", "velocity_r2"):
            actual, expected = np.asarray(reconstructed[mode][metric]), np.asarray(gate[mode][metric])
            comparisons.append(dict(mode=mode, metric=metric,
                matches=bool(np.allclose(actual, expected, rtol=1e-8, atol=1e-10)),
                maximum_absolute_difference=float(np.max(np.abs(actual - expected)))))
    matches = all(row["matches"] for row in comparisons) and reconstructed["passed"] == gate["passed"]
    destination = out / "feature_diagnostics"; destination.mkdir(exist_ok=True)
    write_csv(destination / "dev_predictions.csv", cases)
    write_csv(destination / "r2_by_dimension.csv", dimensions)
    atomic_json(destination / "verification.json", dict(status="verified" if matches else "mismatch",
        matches_recorded_feature_gate=matches, recorded_gate=gate, reconstructed_gate=reconstructed,
        comparisons=comparisons, evaluation_split="dev", test_opened=False,
        feature_gate_sha256=digest(gate_path),
        cache_manifest_sha256={split: digest(cache / split / "manifest.json") for split in ("train", "dev")},
        target_definition="Positions A(x,y), B(x,y) at t=7, then their observed displacements position(7)-position(6); not hidden velocities.",
        baseline_definition="Per-dimension Dev squared error against the Train target mean; identical to features.probe.",
        ridge_coefficient=.01, torch_threads=torch.get_num_threads()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(12, 5))
    x = np.arange(8)
    for offset, mode in ((-.2, "raw"), (.2, "pooled")):
        axis.bar(x + offset, reconstructed[mode]["r2_per_dimension"], .38, label=mode)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(x, DIMENSIONS, rotation=28, ha="right")
    axis.set_ylabel("Dev R² (baseline: Train target mean)")
    axis.set_title("Frozen encoder feature probe: per-dimension predictability")
    axis.legend(); figure.tight_layout(); figure.savefig(destination / "r2_by_dimension.png", dpi=170)
    plt.close(figure)
    for mode in ("raw", "pooled"):
        figure, axes = plt.subplots(2, 4, figsize=(13, 7))
        for dimension, axis in enumerate(axes.flat):
            selected = [row for row in cases if row["mode"] == mode and row["dimension_index"] == dimension]
            target = [row["target"] for row in selected]; prediction = [row["prediction"] for row in selected]
            low, high = min(target + prediction), max(target + prediction)
            axis.scatter(target, prediction, s=13, alpha=.65)
            axis.plot([low, high], [low, high], color="black", linewidth=.8, linestyle="--")
            axis.set_title(f"{DIMENSIONS[dimension]} · R²={reconstructed[mode]['r2_per_dimension'][dimension]:.3f}", fontsize=9)
            axis.set_xlabel("Dev target"); axis.set_ylabel("Prediction")
        figure.suptitle(f"{mode.capitalize()} features: unchanged Train-only ridge probe")
        figure.tight_layout(); figure.savefig(destination / f"{mode}_predictions.png", dpi=170)
        plt.close(figure)
    rows = ["# Feature probe diagnostics", "",
            f"Reproduction of recorded gate: **{'verified' if matches else 'MISMATCH'}**. "
            f"Recorded pooled-feature gate: **{'passed' if gate['passed'] else 'failed'}**.", "",
            "This is the original Train-only ridge fit evaluated on the same Dev records. "
            "No gate, target, seed, or regularization value is changed; Test remains unopened.", "",
            "| Features | Position R² | Observed displacement R² | Mean MSE |",
            "|---|---:|---:|---:|"]
    for mode in ("raw", "pooled"):
        value = reconstructed[mode]
        rows.append(f"| {mode} | {value['position_r2']:.6g} | {value['velocity_r2']:.6g} | {value['mse']:.6g} |")
    rows.extend(["", "The first four target dimensions are A/B x/y positions at t=7. The last four "
                 "are their visible t=6→7 displacements, including reflections. The legacy JSON key "
                 "`velocity_r2` refers to these observed displacements. R² uses the Train target mean baseline.", "",
                 "![R² by target dimension](r2_by_dimension.png)", "",
                 "![Raw feature predictions](raw_predictions.png)", "",
                 "![Pooled feature predictions](pooled_predictions.png)", "",
                 "[Per-Dev-episode target, prediction and error](dev_predictions.csv) · "
                 "[Dimension metrics](r2_by_dimension.csv) · [Numerical verification](verification.json)", "",
                 "These probe diagnostics measure recoverable visual cues. A failed prerequisite "
                 "does not establish a comparison between memory architectures or a causal failure mechanism.", ""])
    (destination / "README.md").write_text("\n".join(rows))
    if not matches:
        raise ValueError("Diagnostic reconstruction differs from recorded feature gate; inspect verification.json")
    return dict(out=str(destination), matches_recorded_feature_gate=True,
                dev_prediction_rows=len(cases), dimensions=len(dimensions), gate_passed=gate["passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Existing V5 run directory")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    print(json.dumps(diagnose(args.out), indent=2))
