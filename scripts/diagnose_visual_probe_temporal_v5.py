#!/usr/bin/env python3
"""Additional exploratory temporal controls, designed after the core probe result."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
if __name__ == "__main__":
    startup = argparse.ArgumentParser(add_help=False)
    startup.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=4)
    opts, _ = startup.parse_known_args()
    for name in THREAD_VARIABLES:
        os.environ[name] = str(opts.threads)

import numpy as np
import torch

CORE_PATH = Path(__file__).with_name("diagnose_visual_probe_v5.py")
SPEC = importlib.util.spec_from_file_location("v5_temporal_core", CORE_PATH)
core = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(core)


def temporal_views(train_gram, cross_gram):
    return {
        "frame6": {"concat": (train_gram[0::2, 0::2], cross_gram[0::2, 0::2])},
        "frame7": {"concat": (train_gram[1::2, 1::2], cross_gram[1::2, 1::2])},
    }


def swapped_concat_cross(cross_gram):
    # Query is [z7,z6], while the fitted Train concat remains [z6,z7].
    return cross_gram[1::2, 0::2] + cross_gram[0::2, 1::2]


def reversed_targets(targets):
    return np.concatenate([targets[:, :4] - targets[:, 4:], -targets[:, 4:]], axis=1)


def choose_single_frame(selections):
    """Choose frame solely by its best held-out Train CV objective."""
    candidates = []
    for frame in ("frame6", "frame7"):
        selection = selections[frame]
        chosen = next(trial for trial in selection["trials"] if trial["alpha"] == selection["selected_alpha"])
        candidates.append((chosen["mean_score"], frame, selection["selected_alpha"]))
    score, frame, alpha = min(candidates)  # Exact ties favor frame6, then alpha.
    return dict(selected_frame=frame, selected_alpha=alpha, train_cv_motion_error=score,
                selection_basis="minimum inner-Train grouped-CV mean normalized motion error; no Dev input")


def run(cache, core_results, out, threads=4):
    started = time.monotonic()
    cache, core_results, out = Path(cache).resolve(), Path(core_results).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a fresh temporal-control directory; previous diagnostics remain unchanged")
    if not 1 <= threads <= 4:
        raise ValueError("Use one to four CPU threads")
    torch.set_num_threads(threads)
    prior = json.loads((core_results/"diagnostics.json").read_text())
    plan = json.loads((core_results/"frozen_plan.json").read_text())
    if prior["script_sha256"] != core.sha256(CORE_PATH) or prior["plan_sha256"] != core.sha256(core.PLAN_PATH):
        raise ValueError("The core implementation or plan changed since its recorded diagnostic run")
    manifest_hashes = {split: core.sha256(cache/split/"manifest.json") for split in ("train", "dev")}
    if manifest_hashes != prior["cache_manifest_sha256"]:
        raise ValueError("Temporal controls must use the exact core diagnostic caches")
    out.mkdir(parents=True)
    followup_plan = dict(status="additional_exploratory_followup_designed_after_core_result",
        trigger="Core Train-CV concat improved Dev motion R2, requiring matched single-frame and order-swap controls before temporal claims.",
        original_core_results_sha256=core.sha256(core_results/"diagnostics.json"),
        original_core_script_sha256=prior["script_sha256"],
        data="Existing 128 Train and 128 previously inspected Dev records; Test remains unopened",
        representations=plan["representations"],
        single_frame_controls=["frame6", "frame7"], train_cv=plan["cv"], alpha_grid=plan["alpha_grid"],
        frame_selection="Choose frame6/frame7 by minimum best-alpha Train CV score; exact ties choose frame6. Report both Dev controls.",
        reference="Reuse the core full-Train concat alpha selected by Train CV, with no new Dev selection",
        comparison="Paired Dev bootstrap of concat R2 minus the Train-selected single-frame R2; identical original Train-mean denominator",
        query_order_intervention="Fit original Train concat, reverse only Dev cue-frame order; targets become [p6,-displacement]",
        reversed_denominator="Reversed Train target mean: [mean(p7)-mean(displacement), -mean(displacement)]",
        oddness="RMSE of forward plus reversed predicted displacement; zero is ideal signed-displacement antisymmetry",
        bootstrap=plan["bootstrap"], limitations=[
            "Followup was triggered by observed core Dev results, not preregistered.",
            "All Dev findings and intervals are exploratory; no automatic model winner or gate change.",
            "Reversing query order changes the generator's directional/position prior and is a distribution-shift intervention.",
            "No GPU, Test access, downstream training, or modification of archived results."])
    core.write_json(out/"followup_plan.json", followup_plan)
    train, dev = core.load_records(cache, "train"), core.load_records(cache, "dev")
    if len(train) != 128 or len(dev) != 128:
        raise ValueError("Expected the existing complete 64 paired worlds per split")
    for split, records in (("train", train), ("dev", dev)):
        if any(row["metadata"]["split"] != split or row["times"][6:8].tolist() != [6., 7.]
               or row["probe_state"].shape != (8,) for row in records):
            raise ValueError("Unexpected input split, cue ordering, or labels")
    y = torch.stack([row["probe_state"] for row in train]).double().numpy()
    target = torch.stack([row["probe_state"] for row in dev]).double().numpy()
    pairs = [row["metadata"]["pair_id"] for row in train]
    dev_pairs = [row["metadata"]["pair_id"] for row in dev]
    if set(pairs) & set(dev_pairs) or any(pairs.count(pair) != 2 for pair in set(pairs)):
        raise ValueError("Train/Dev overlap or incomplete paired Train group")
    indices, query_indices = np.arange(len(train)), np.arange(len(dev))
    kernels, selections = {}, {}
    # Complete and persist all new Train-only choices before any Dev prediction.
    for mode in plan["representations"]:
        gram, cross, width = core.frame_grams(train, dev, mode, plan["numerics"]["feature_chunk"])
        views = temporal_views(gram, cross)
        frame_selections = {frame: core.select_alpha("concat", view, indices, y, pairs, plan)
                            for frame, view in views.items()}
        selections[mode] = dict(controls=frame_selections, chosen=choose_single_frame(frame_selections))
        kernels[mode] = (gram, cross, views)
    core.write_json(out/"train_selection.json", selections)
    original_mean, reverse_mean = y.mean(0), reversed_targets(y).mean(0)
    target_reverse = reversed_targets(target)
    summary, dimensions, case_rows, comparisons = [], [], [], {}

    def record(mode, method, alpha, prediction, truth, baseline_mean, boot=None):
        metrics = core.metric_values(prediction, truth, baseline_mean)
        summary.append(dict(representation=mode, method=method, alpha=alpha,
            position_r2=metrics["position_r2"], displacement_r2=metrics["displacement_r2"],
            displacement_ci_low=boot["displacement_r2_ci"][0] if boot else "",
            displacement_ci_high=boot["displacement_r2_ci"][1] if boot else ""))
        for dim, original_name in enumerate(core.DIMENSIONS):
            name = original_name
            if method == "concat_query_reversed":
                name = original_name.replace("_t7", "_t6") if dim < 4 else original_name.replace("_t6_to_t7", "_t7_to_t6")
            semantics = "p6,-displacement" if method == "concat_query_reversed" else "p7,displacement"
            dimensions.append(dict(representation=mode, method=method, dimension=name,
                target_semantics=semantics,
                r2=metrics["r2_per_dimension"][dim], mse=metrics["mse_per_dimension"][dim],
                baseline_mse=metrics["baseline_mse_per_dimension"][dim]))
            for index, row in enumerate(dev):
                residual = float(prediction[index, dim] - truth[index, dim])
                case_rows.append(dict(representation=mode, method=method, episode_id=row["metadata"]["episode_id"],
                    pair_id=dev_pairs[index], dimension=name, target_semantics=semantics, target=float(truth[index, dim]),
                    prediction=float(prediction[index, dim]), residual=residual, squared_error=residual**2))
        return metrics

    for mode in plan["representations"]:
        gram, cross, controls = kernels[mode]
        original_views = core.kernel_views(gram, cross)
        core_fit = prior["results"][f"{mode}/concat/pairs64/train_cv"]
        alpha = core_fit["alpha"]
        forward = core.method_predict("concat", original_views, indices, query_indices, y, alpha)
        forward_metrics = core.metric_values(forward, target, original_mean)
        difference = float(np.max(np.abs(np.asarray(forward_metrics["r2_per_dimension"])-np.asarray(core_fit["metrics"]["r2_per_dimension"]))))
        if difference > 1e-9:
            raise ValueError("Recomputed original concat differs from the frozen core result")
        control_predictions, control_results = {}, {}
        for frame, views in controls.items():
            chosen_alpha = selections[mode]["controls"][frame]["selected_alpha"]
            prediction = core.method_predict("concat", views, indices, query_indices, y, chosen_alpha)
            boot = core.paired_bootstrap(prediction, target, original_mean, dev_pairs, plan["bootstrap"])
            metrics = record(mode, frame+"_train_cv", chosen_alpha, prediction, target, original_mean, boot)
            control_predictions[frame] = prediction
            control_results[frame] = dict(alpha=chosen_alpha, metrics=metrics, bootstrap=boot)
        selected = selections[mode]["chosen"]["selected_frame"]
        selected_prediction = control_predictions[selected]
        selected_metrics = control_results[selected]["metrics"]
        boot = core.paired_bootstrap(forward, target, original_mean, dev_pairs, plan["bootstrap"], reference=selected_prediction)
        record(mode, "concat_train_cv", alpha, forward, target, original_mean, boot)
        reverse = core.dual_predict(original_views["concat"][0], swapped_concat_cross(cross), y, alpha)
        reverse_boot = core.paired_bootstrap(reverse, target_reverse, reverse_mean, dev_pairs, plan["bootstrap"])
        reverse_metrics = record(mode, "concat_query_reversed", alpha, reverse, target_reverse, reverse_mean, reverse_boot)
        summed = forward[:, 4:] + reverse[:, 4:]
        true_rms = float(np.sqrt(np.mean(target[:, 4:]**2)))
        oddness_rmse = float(np.sqrt(np.mean(summed**2)))
        comparisons[mode] = dict(original_concat_alpha=alpha, original_core_reproduction_max_r2_difference=difference,
            original_metrics=forward_metrics, all_single_frame_controls=control_results,
            train_selected_single_frame=selections[mode]["chosen"],
            concat_minus_train_selected_single_frame_displacement_r2=forward_metrics["displacement_r2"]-selected_metrics["displacement_r2"],
            concat_minus_train_selected_single_frame_ci=boot["difference_from_original_ci"],
            reversed_query_metrics=reverse_metrics, reversed_query_bootstrap=reverse_boot,
            original_train_target_mean=original_mean.tolist(), reversed_train_target_mean=reverse_mean.tolist(),
            oddness=dict(forward_plus_reverse_motion_rmse=oddness_rmse,
                per_dimension_rmse=np.sqrt(np.mean(summed**2,axis=0)).tolist(),
                true_displacement_rms=true_rms, rmse_over_true_displacement_rms=oddness_rmse/true_rms,
                prediction_sign_reversal_fraction=float(np.mean(forward[:,4:]*reverse[:,4:]<0))))
    for name, rows in (("summary.csv", summary), ("per_dimension.csv", dimensions), ("dev_predictions.csv", case_rows)):
        core.write_csv(out/name, rows)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    diagnostics = dict(status="additional_exploratory_temporal_controls_complete", test_opened=False, gpu_used=False,
        gate_changed=False, automatic_dev_winner=None, downstream_training_authorized=False,
        design_timing="after core diagnostic results; exploratory followup, not preregistered",
        script_sha256=core.sha256(__file__), core_script_sha256=core.sha256(CORE_PATH),
        core_diagnostics_sha256=core.sha256(core_results/"diagnostics.json"), cache_manifest_sha256=manifest_hashes,
        followup_plan_sha256=core.sha256(out/"followup_plan.json"), results=comparisons,
        wall_seconds=time.monotonic()-started, cpu_threads=threads,
        peak_process_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform=="darwin" else 1024)))
    core.write_json(out/"diagnostics.json", diagnostics)
    lines=["# Additional exploratory temporal controls", "",
        "This followup was designed after the core diagnostic improved concatenated-feature Dev motion R². "
        "It is not preregistered or independent confirmation. It reuses existing Train/Dev caches, changes no original gate, and opens no Test data.", "",
        "Frame6 and frame7 controls use the same full-Train paired folds, alpha grid, centering, and motion-error objective as the core concat decoder. "
        "The choice between the two single-frame controls is made solely by held-out Train CV; both Dev results are shown.", "",
        "| Features | Control | Alpha | Position R² | Displacement R² | Exploratory displacement 95% CI |",
        "|---|---|---:|---:|---:|---|"]
    for row in summary:
        lines.append(f"| {row['representation']} | {row['method']} | {row['alpha']:g} | {row['position_r2']:.5f} | "
                     f"{row['displacement_r2']:.5f} | [{row['displacement_ci_low']:.5f}, {row['displacement_ci_high']:.5f}] |")
    lines += ["", "## Matched temporal advantage", "",
        "| Features | Train-selected single frame | Concat minus selected-frame displacement R² | Paired 95% CI |",
        "|---|---|---:|---|"]
    for mode, item in comparisons.items():
        lo, hi = item["concat_minus_train_selected_single_frame_ci"]
        lines.append(f"| {mode} | {item['train_selected_single_frame']['selected_frame']} | "
                     f"{item['concat_minus_train_selected_single_frame_displacement_r2']:.5f} | [{lo:.5f}, {hi:.5f}] |")
    lines += ["", "## Query-order intervention", "",
        "The original fitted concat decoder receives [z7,z6] instead of [z6,z7]. Correct labels are [p6,−displacement]; "
        "reversed-query R² uses the correspondingly reversed Train target mean in its denominator. No reverse-order fitting or Dev tuning is performed. "
        "This intervention changes the generator's position/direction prior, so it is a distribution-shift diagnostic, not proof of a causal mechanism.", "",
        "| Features | RMSE(predicted motion forward + reversed) | RMSE / true-motion RMS | Fraction of predicted signs reversing |",
        "|---|---:|---:|---:|"]
    for mode, item in comparisons.items():
        odd = item["oddness"]
        lines.append(f"| {mode} | {odd['forward_plus_reverse_motion_rmse']:.6f} | {odd['rmse_over_true_displacement_rms']:.4f} | "
                     f"{odd['prediction_sign_reversal_fraction']:.4f} |")
    lines += ["", "Oddness is a signed-displacement consistency check: a pure displacement mapping would sum to zero under reversal. "
        "Nonzero values can include the fitted intercept and the asymmetric generator prior; they do not alone isolate feature pooling, regularization, or a model mechanism.", "",
        "Bootstrap resamples complete Dev pairs with both variants. Concat-versus-single-frame differences use identical resamples and the same original Train-mean denominator. "
        "Intervals describe this already inspected sample only. No automatic Dev winner or downstream training approval is emitted.", "",
        "[Post-result followup plan](followup_plan.json) · [Train-only frame/alpha selection](train_selection.json) · "
        "[Predictions](dev_predictions.csv) · [Complete metrics and provenance](diagnostics.json)", ""]
    (out/"README.md").write_text("\n".join(lines))
    return dict(out=str(out),status=diagnostics["status"],results=comparisons,
                wall_seconds=diagnostics["wall_seconds"],peak_process_rss_bytes=diagnostics["peak_process_rss_bytes"])


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True,type=Path)
    p.add_argument("--core-results",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--threads",type=int,choices=(1,2,3,4),default=4)
    a=p.parse_args()
    print(json.dumps(run(a.cache,a.core_results,a.out,a.threads),indent=2))
