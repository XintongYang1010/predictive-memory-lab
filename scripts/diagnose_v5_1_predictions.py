#!/usr/bin/env python3
"""Posthoc CPU diagnostics of completed V5.1 reports; no fitting or new gate.

Only already-reported confirmation blocks may be re-rendered. Bridge reflection
strata use scalar t6/t7 state reconstruction, never new images or Test samples.
This analysis script stays outside the frozen remote scientific deployment.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "4")

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_v5_archived_predictions import physical_state
from v5_temporal_confirmation import make_quartet

DIMENSIONS = ("A_x7", "A_y7", "B_x7", "B_y7", "A_dx6to7", "A_dy6to7", "B_dx6to7", "B_dy6to7")
METHODS = ("concat", "t6", "t7")
VIEWS = ("end_plus", "end_minus", "start_plus", "start_minus")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    with open(path, "x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rescore_confirm(prediction, target, repeat, fit_mean):
    """Independent arithmetic from persisted predictions and the fixed plan."""
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    d = y[:, 0, 4:]
    if not (np.array_equal(d, -y[:, 1, 4:]) and np.array_equal(d, y[:, 2, 4:]) and np.array_equal(d, -y[:, 3, 4:])):
        raise ValueError("Confirmation labels violate quartet order")
    contrasts = ((p[:, 0, 4:]-p[:, 1, 4:])/2, (p[:, 2, 4:]-p[:, 3, 4:])/2)
    energy = d**2
    errors = [(contrast-d)**2 for contrast in contrasts]
    absolute_error = ((p[:, :, 4:]-y[:, :, 4:])**2).mean(1)
    if np.any(energy.mean(0) <= 0):
        raise ValueError("Degenerate confirmation targets")
    contrast_skill = [1-error.mean(0)/energy.mean(0) for error in errors]
    absolute = float((1-absolute_error.mean(0)/energy.mean(0)).mean())
    position = float((1-((p[:, :, :4]-y[:, :, :4])**2).mean((0, 1))/
                     np.maximum(((y[:, :, :4]-fit_mean[:4])**2).mean((0, 1)), 1e-12)).mean())
    rng = np.random.default_rng(7709)
    draws = []
    for _ in range(40):
        indices = rng.integers(0, len(p), (250, len(p)))
        den = energy[indices].mean(1)
        draws.append(np.concatenate([1-errors[0][indices].mean(1)/den,
                                     1-errors[1][indices].mean(1)/den,
                                     (1-absolute_error[indices].mean(1)/den).mean(1, keepdims=True)], axis=1))
    lower = np.quantile(np.concatenate(draws), .05/9, axis=0)
    reversal = float(max(np.abs(p[:, 0, 4:]+p[:, 3, 4:]).max(), np.abs(p[:, 1, 4:]+p[:, 2, 4:]).max()))
    repeated = float(np.abs(repeat).max())
    checks = dict(nine_lower_bounds_positive=bool(np.all(lower > 0)), position_skill_over_point_one=position > .1,
                  reversal_max_abs_at_most_1e_minus_6=reversal <= 1e-6,
                  repeated_frame_max_abs_at_most_1e_minus_6=repeated <= 1e-6)
    return dict(contrast_skills={key: value.tolist() for key, value in zip(("end_anchor", "start_anchor"), contrast_skill)},
                absolute_displacement_skill=absolute, position_skill=position,
                primary_lower_bounds=lower.tolist(), reversal_max_abs=reversal,
                repeated_center_max_abs=repeated, checks=checks, metric_conditions_met=all(checks.values()))


def rescore_bridge(prediction, target, fit_mean):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    error, baseline = ((p-y)**2).mean(1), ((y-fit_mean)**2).mean(1)
    if np.any(baseline.mean(0) <= 0):
        raise ValueError("Degenerate bridge targets")
    skill = 1-error.mean(0)/baseline.mean(0)
    rng = np.random.default_rng(7709)
    draws = []
    for _ in range(40):
        indices = rng.integers(0, len(p), (250, len(p)))
        draws.append((1-error[indices, 4:].mean(1)/baseline[indices, 4:].mean(1)).mean(1))
    lower = float(np.quantile(np.concatenate(draws), .05))
    checks = dict(position_skill_over_point_one=float(skill[:4].mean()) > .1,
                  all_four_motion_coordinate_skills_positive=bool(np.all(skill[4:] > 0)),
                  motion_mean_lower_bound_positive=lower > 0)
    return dict(per_dimension_skill=skill.tolist(), position_skill=float(skill[:4].mean()),
                motion_skill=float(skill[4:].mean()), motion_mean_lower_bound=lower,
                checks=checks, metric_conditions_met=all(checks.values()))


def compare_score(recomputed, reported):
    differences = []
    def visit(actual, expected):
        if isinstance(actual, dict):
            for k, value in actual.items():
                if k not in expected:
                    raise ValueError("Missing recorded score field: "+k)
                visit(value, expected[k])
        elif isinstance(actual, bool):
            if actual is not expected:
                raise ValueError("Recorded scientific decision differs from independent rescore")
        else:
            a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Recorded metric shape/finiteness differs")
            differences.append(float(np.max(np.abs(a-b))))
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    visit(recomputed, reported)
    maximum = max(differences, default=0.)
    return dict(all_decisions_exact=True, all_numeric_values_bitwise_equal=maximum == 0,
                maximum_absolute_metric_difference=maximum, tolerance=dict(rtol=1e-12, atol=1e-12))


def read_frozen(run_root):
    path = run_root/"reader/freeze_receipt.json"
    freeze = read_json(path)
    if freeze.get("status") != "frozen_before_confirmation" or freeze.get("fit_blocks") != 256:
        raise ValueError("Completed 256-block FIT freeze required")
    for name, expected in freeze["files"].items():
        if sha(run_root/"reader"/name) != expected:
            raise ValueError("Frozen artifact hash mismatch: "+name)
    for name, expected in freeze["sources"].items():
        if sha(ROOT/"scripts"/name) != expected:
            raise ValueError("Frozen reader/helper/plan differs from the locally available source")
    with np.load(run_root/"reader/frozen_model.npz", allow_pickle=False) as z:
        means = {name: z[name+"__target_mean"] for name in METHODS}
    return freeze, means, sha(path)


def read_report(run_root, role, freeze, frozen_sha):
    folder = run_root/(role+"_report")
    report = read_json(folder/"report.json")  # Required before any data is opened.
    counts = dict(blocks=256, views=1024) if role == "confirm" else dict(paired_worlds=256, episodes=512)
    if (report.get("role") != role or report.get("counts") != counts or report.get("test_access") is not False or
            report.get("frozen_reader_sha256") != frozen_sha or report.get("cache_identity") != freeze["cache_identity"] or
            report.get("sources") != freeze["sources"]):
        raise ValueError("Completed report role/count/identity mismatch")
    if sha(folder/"predictions.npz") != report["predictions_sha256"] or sha(folder/"cache_receipt.json") != report["cache_receipt_sha256"]:
        raise ValueError("Prediction or input receipt hash mismatch")
    cache = read_json(folder/"cache_receipt.json")
    with np.load(folder/"predictions.npz", allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    target = arrays["targets"]
    expected = (256, 4, 8) if role == "confirm" else (256, 2, 8)
    if target.shape != expected or not np.isfinite(target).all():
        raise ValueError("Unexpected target dimensions")
    entry = dict(sha256=hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest(), dtype=str(target.dtype), shape=list(target.shape))
    if cache["entry_hashes"]["targets"] != entry:
        raise ValueError("Prediction targets differ from extraction receipt")
    for method in METHODS:
        if arrays[method].shape != expected or not np.isfinite(arrays[method]).all():
            raise ValueError("Unexpected prediction dimensions")
        if role == "confirm" and (arrays[method+"_repeat"].shape != (256, 4) or not np.isfinite(arrays[method+"_repeat"]).all()):
            raise ValueError("Invalid repeated-center predictions")
    return report, arrays


def tables_and_plots(role, arrays, means, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = arrays["targets"].astype(float)
    group_ids = np.arange(256) + (128 if role == "bridge" else 0)
    residuals, group_rows, dimension_rows = [], [], []
    primary_errors = ((arrays["concat"][:, :, 4:]-target[:, :, 4:])**2).mean((1, 2))
    order = np.lexsort((group_ids, -primary_errors))
    rank = {int(index): place+1 for place, index in enumerate(order)}
    for method in METHODS:
        prediction = arrays[method]
        error = prediction-target
        mse = (error**2).mean((0, 1))
        baseline = ((target-means[method])**2).mean((0, 1))
        for dim, label in enumerate(DIMENSIONS):
            dimension_rows.append(dict(role=role, method=method, dimension=label, mse=float(mse[dim]),
                rmse=float(np.sqrt(mse[dim])), mean_residual=float(error[:, :, dim].mean()),
                fit_mean_baseline_skill=float(1-mse[dim]/max(baseline[dim], 1e-12))))
        for group, group_id in enumerate(group_ids):
            group_rows.append(dict(role=role, method=method, group_id=int(group_id),
                primary_whole_group_rank=rank[group], views=target.shape[1],
                motion_mse=float((error[group, :, 4:]**2).mean()),
                position_mse=float((error[group, :, :4]**2).mean()),
                maximum_coordinate_motion_squared_error=float((error[group, :, 4:]**2).max())))
            for view in range(target.shape[1]):
                for dim, label in enumerate(DIMENSIONS):
                    residuals.append(dict(role=role, method=method, group_id=int(group_id),
                        view=VIEWS[view] if role == "confirm" else str(view), dimension=label,
                        target=float(target[group, view, dim]), prediction=float(prediction[group, view, dim]),
                        residual=float(error[group, view, dim]), squared_error=float(error[group, view, dim]**2)))
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for dim, ax in enumerate(axes.flat):
            actual, predicted = target[:, :, dim].ravel(), prediction[:, :, dim].ravel()
            ax.scatter(actual, predicted, s=7, alpha=.35, rasterized=True)
            lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            ax.plot([lo, hi], [lo, hi], color="black", lw=1)
            ax.set(xlabel="Target", ylabel="Frozen prediction", title=DIMENSIONS[dim])
        fig.suptitle(f"{role} · {method} · posthoc diagnostic, no refit")
        fig.tight_layout(); fig.savefig(out/(method+"_predictions.png"), dpi=150); plt.close(fig)
    write_csv(out/"residuals.csv", residuals)
    write_csv(out/"per_dimension.csv", dimension_rows)
    group_rows.sort(key=lambda row: (row["primary_whole_group_rank"], METHODS.index(row["method"])))
    write_csv(out/"whole_group_errors.csv", group_rows)
    write_csv(out/"top_groups.csv", [row for row in group_rows if row["primary_whole_group_rank"] <= 10])
    return order[:2]


def render_bad_confirm(arrays, selected, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    checks = []
    for row, block in enumerate(selected):
        quartet = make_quartet("confirm", int(block))
        np.testing.assert_array_equal(quartet.targets, arrays["targets"][block])
        checks.append(dict(block_id=int(block), targets_equal=True))
        for column, label in enumerate(("C−d", "C", "C+d")):
            ax = axes[row, column]
            ax.imshow(quartet.unique_rgb[column]); ax.axis("off")
            ax.set_title(f"Block {block:03d} · {label}")
        delta = arrays["targets"][block, 0, 4:]
        plus, minus = arrays["concat"][block, :2, 4:]
        text = "[A dx, A dy, B dx, B dy]  target d="+np.array2string(delta, precision=4)
        text += "\nend+ prediction="+np.array2string(plus, precision=4)+"; end− prediction="+np.array2string(minus, precision=4)
        axes[row, 1].text(.5, -.08, text, transform=axes[row, 1].transAxes, ha="center", va="top", fontsize=9)
    fig.suptitle("Worst two whole blocks by primary motion MSE · posthoc labels, never actor inputs")
    fig.subplots_adjust(left=.03, right=.97, top=.93, bottom=.10, wspace=.04, hspace=.44)
    fig.savefig(out/"worst_two_quartets.png", dpi=160); plt.close(fig)
    write_json(out/"render_verification.json", dict(selection_rule="descending mean squared motion error over all four views and four coordinates; ties ascending block ID",
        checks=checks, report_required_before_render=True, posthoc_only=True))


def bridge_strata(arrays, means, source_manifest, out):
    helper = ROOT/"scripts/audit_v5_archived_predictions.py"
    expected = source_manifest["files"].get("scripts/audit_v5_archived_predictions.py")
    if expected is None or sha(helper) != expected:
        raise ValueError("Archived scalar audit helper identity unavailable or changed")
    reflected = np.zeros((256, 2, 4), dtype=bool)
    target = arrays["targets"]
    for index, pair in enumerate(range(128, 384)):
        for variant in (0, 1):
            state, flags = physical_state("dev", pair, variant)
            np.testing.assert_array_equal(state, target[index, variant])
            reflected[index, variant] = flags
    world_reflected = reflected.any((1, 2))
    rows, flags_rows = [], []
    for index, pair in enumerate(range(128, 384)):
        flags_rows.append(dict(pair_id=pair, any_cue_reflection=bool(world_reflected[index]),
            variant0_coordinates=",".join(DIMENSIONS[d+4] for d in np.flatnonzero(reflected[index, 0])),
            variant1_coordinates=",".join(DIMENSIONS[d+4] for d in np.flatnonzero(reflected[index, 1]))))
    for name, mask in (("any_coordinate_either_variant_reflects", world_reflected), ("no_cue_reflection", ~world_reflected)):
        if not mask.any():
            continue
        for method in METHODS:
            y, p = target[mask].astype(float), arrays[method][mask]
            error = ((p-y)**2).mean((0, 1))
            baseline = ((y-means[method])**2).mean((0, 1))
            skill = 1-error/np.maximum(baseline, 1e-12)
            rows.append(dict(stratum=name, method=method, paired_worlds=int(mask.sum()), episodes=int(mask.sum()*2),
                motion_mse=float(error[4:].mean()), position_mse=float(error[:4].mean()),
                mean_motion_fit_baseline_skill=float(skill[4:].mean()),
                **{DIMENSIONS[d]+"_skill": float(skill[d]) for d in range(4, 8)}))
    write_csv(out/"reflection_flags.csv", flags_rows)
    write_csv(out/"reflection_strata.csv", rows)
    write_json(out/"reflection_verification.json", dict(all_512_scalar_targets_exact=True,
        helper_sha256=sha(helper), descriptive_only=True,
        stratification="a complete pair is reflected if either variant has any boundary crossing between t6 and t7",
        no_stratum_gate_or_hyperparameter_selection=True))


def run(run_root, out):
    run_root, out = Path(run_root).resolve(), Path(out).resolve()
    roles = [role for role in ("confirm", "bridge") if (run_root/(role+"_report")/"report.json").is_file()]
    if not roles:
        raise ValueError("No completed confirmation/bridge report; no scientific data may be opened")
    freeze, means, frozen_sha = read_frozen(run_root)
    loaded = {role: read_report(run_root, role, freeze, frozen_sha) for role in roles}
    out.mkdir(parents=True, exist_ok=False)
    verification = dict(posthoc_only=True, fitting=False, new_gate=False, test_access=False,
                        frozen_reader_sha256=frozen_sha, diagnostic_script_sha256=sha(__file__), roles={})
    for role, (report, arrays) in loaded.items():
        folder = out/role; folder.mkdir()
        scores, comparison = {}, {}
        for method in METHODS:
            scores[method] = (rescore_confirm(arrays[method], arrays["targets"], arrays[method+"_repeat"], means[method])
                              if role == "confirm" else rescore_bridge(arrays[method], arrays["targets"], means[method]))
            comparison[method] = compare_score(scores[method], report["scores"][method])
        verification["roles"][role] = dict(report_sha256=sha(run_root/(role+"_report")/"report.json"), comparison=comparison)
        write_json(folder/"independent_scores.json", scores)
        selected = tables_and_plots(role, arrays, means, folder)
        if role == "confirm":
            render_bad_confirm(arrays, selected, folder)
        else:
            bridge_strata(arrays, means, read_json(run_root/"source_manifest.json"), folder)
    write_json(out/"verification.json", verification)
    (out/"README.md").write_text("# V5.1 posthoc prediction diagnostics\n\n"
        "These diagnostics reopen completed frozen-reader reports and preserve their primary decisions. "
        "No fitting, candidate switching, new gate, or Test access occurs. Numeric reproduction uses a 1e-12 tolerance; "
        "verification.json separately records bitwise equality and exact decision agreement.\n\n"
        "Each role contains all coordinate residuals, per-coordinate metrics, three prediction plots, and whole-group error rankings. "
        "Ranking uses primary concat motion MSE averaged over every view/variant and all four motion dimensions, with ascending ID ties. "
        "The two worst confirmation blocks are re-rendered only after their completed report is verified. Displayed labels are posthoc analysis metadata.\n\n"
        "If a bridge report exists, reflection strata preserve both variants of every world. They describe boundary crossings from t6 to t7; "
        "strata differ in geometry and difficulty, so differences are not causal effects. They do not change the registered acceptance criterion.\n")
    return verification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.run_root, args.out), indent=2))


if __name__ == "__main__":
    main()
