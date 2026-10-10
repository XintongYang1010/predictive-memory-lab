#!/usr/bin/env python3
"""CPU-only, Train-selected exploratory diagnostics; never replaces the V5 gate."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

# Set BLAS limits before importing NumPy/PyTorch; no extra dependency is needed.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
if __name__ == "__main__":
    startup_parser = argparse.ArgumentParser(add_help=False)
    startup_parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=4)
    startup_args, _ = startup_parser.parse_known_args()
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(startup_args.threads)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmr.visual_memory.features import load_records

PLAN_PATH = Path(__file__).with_name("visual_probe_diagnostic_plan.json")
DIMENSIONS = ("A_x_t7", "A_y_t7", "B_x_t7", "B_y_t7",
              "A_dx_t6_to_t7", "A_dy_t6_to_t7", "B_dx_t6_to_t7", "B_dy_t6_to_t7")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_grams(train, dev, mode, chunk=4096):
    """Only N×N kernels and small N×chunk blocks are materialized in float64."""
    if mode not in ("raw", "pooled") or chunk < 1:
        raise ValueError("Expected raw/pooled features and a positive chunk size")
    def frames(rows):
        result = []
        for row in rows:
            source = row["raw_cue"] if mode == "raw" else row["features"][6:8]
            if source is None or len(source) != 2 or source.device.type != "cpu":
                raise ValueError("Exactly two CPU cue frames per episode are required")
            result.extend(frame.reshape(-1) for frame in source)
        return result
    a, b = frames(train), frames(dev)
    width = a[0].numel()
    if any(x.numel() != width for x in a+b):
        raise ValueError("Feature widths differ")
    train_gram = torch.zeros((len(a), len(a)), dtype=torch.float64)
    cross_gram = torch.zeros((len(b), len(a)), dtype=torch.float64)
    for start in range(0, width, chunk):
        x = torch.stack([row[start:start+chunk].double() for row in a])
        z = torch.stack([row[start:start+chunk].double() for row in b])
        if not torch.isfinite(x).all() or not torch.isfinite(z).all():
            raise ValueError("Nonfinite cached feature")
        train_gram.addmm_(x, x.T)
        cross_gram.addmm_(z, x.T)
    return train_gram.numpy(), cross_gram.numpy(), width


def kernel_views(train_gram, cross_gram):
    def concat(g):
        return g[0::2, 0::2] + g[1::2, 1::2]
    def difference(g):
        return g[1::2, 1::2] - g[1::2, 0::2] - g[0::2, 1::2] + g[0::2, 0::2]
    return {"concat": (concat(train_gram), concat(cross_gram)),
            "difference_hybrid": (difference(train_gram), difference(cross_gram)),
            "shared_position": (train_gram, cross_gram)}


def centered_energy(gram):
    centered = gram - gram.mean(0)[None, :] - gram.mean(1)[:, None] + gram.mean()
    return dict(observations=len(gram), uncentered_gram_trace=float(np.trace(gram)),
                centered_gram_trace=float(np.trace(centered)),
                mean_centered_feature_energy=float(np.diag(centered).mean()),
                normalization_scale=max(float(np.diag(centered).mean()), 1e-12))


def dual_predict(train_gram, cross_gram, targets, alpha):
    """Center every fit on its own fitting rows, including inner-CV folds."""
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    columns, total = train_gram.mean(0), train_gram.mean()
    centered = train_gram - columns[None, :] - train_gram.mean(1)[:, None] + total
    cross = cross_gram - columns[None, :] - cross_gram.mean(1)[:, None] + total
    scale = max(float(np.diag(centered).mean()), 1e-12)
    mean = targets.mean(0)
    # features.probe forms alpha*torch.eye in default FP32 before adding it to
    # its FP64 kernel. Preserve that scalar rounding for the original control.
    ridge = float(np.float32(alpha))
    coefficients = np.linalg.solve(centered / scale + ridge * np.eye(len(centered)), targets - mean)
    return (cross / scale) @ coefficients + mean


def frame_indices(episodes):
    episodes = np.asarray(episodes, dtype=int)
    return np.stack([2*episodes, 2*episodes+1], 1).reshape(-1)


def position_targets(targets):
    """Analysis-only supervision; never passed into the scientific actor."""
    return np.stack([targets[:, :4] - targets[:, 4:], targets[:, :4]], 1).reshape(-1, 4)


def method_predict(method, views, fit_indices, query_indices, targets, alpha):
    fit_indices, query_indices = np.asarray(fit_indices), np.asarray(query_indices)
    gram, cross = views[method]
    if method == "shared_position":
        fit_frames, query_frames = frame_indices(fit_indices), frame_indices(query_indices)
        predicted = dual_predict(gram[np.ix_(fit_frames, fit_frames)],
                                 cross[np.ix_(query_frames, fit_frames)],
                                 position_targets(targets[fit_indices]), alpha).reshape(-1, 2, 4)
        return np.concatenate([predicted[:, 1], predicted[:, 1] - predicted[:, 0]], 1)
    selected = targets[fit_indices] if method == "concat" else targets[fit_indices, 4:]
    predicted = dual_predict(gram[np.ix_(fit_indices, fit_indices)],
                             cross[np.ix_(query_indices, fit_indices)], selected, alpha)
    if method == "difference_hybrid":
        # Position scores deliberately retain the original fixed-alpha decoder.
        original = method_predict("concat", views, fit_indices, query_indices, targets, .01)
        predicted = np.concatenate([original[:, :4], predicted], 1)
    return predicted


def grouped_folds(pair_ids, indices, folds=4, seed=8317):
    indices = np.asarray(indices, dtype=int)
    pairs = np.asarray(sorted({pair_ids[i] for i in indices}))
    if len(pairs) < folds:
        raise ValueError("Too few paired worlds for the fixed CV folds")
    shuffled = pairs[np.random.default_rng(seed).permutation(len(pairs))]
    result = []
    for validation_pairs in np.array_split(shuffled, folds):
        held = set(validation_pairs)
        validation = np.asarray([i for i in indices if pair_ids[i] in held], dtype=int)
        fitting = np.asarray([i for i in indices if pair_ids[i] not in held], dtype=int)
        result.append((fitting, validation))
    return result


def normalized_motion_error(prediction, target, train_mean):
    error = ((prediction[:, 4:] - target[:, 4:]) ** 2).mean(0)
    baseline = ((target[:, 4:] - train_mean[4:]) ** 2).mean(0).clip(1e-12)
    return float((error / baseline).mean())


def select_alpha(method, views, indices, targets, pair_ids, plan):
    # Cross-validation uses Train×Train kernels only. Dev is not an argument.
    train_views = {name: (gram, gram) for name, (gram, _) in views.items()}
    folds = grouped_folds(pair_ids, indices, plan["cv"]["folds"], plan["cv"]["seed"])
    trials = []
    for alpha in plan["alpha_grid"]:
        scores = []
        for fitting, validation in folds:
            prediction = method_predict(method, train_views, fitting, validation, targets, alpha)
            scores.append(normalized_motion_error(prediction, targets[validation], targets[fitting].mean(0)))
        trials.append(dict(alpha=alpha, fold_scores=scores, mean_score=float(np.mean(scores))))
    chosen = min(trials, key=lambda item: (item["mean_score"], item["alpha"]))
    return dict(selected_alpha=chosen["alpha"], trials=trials,
                folds=[dict(fit_pair_ids=sorted({pair_ids[i] for i in fitting}),
                            validation_pair_ids=sorted({pair_ids[i] for i in validation}))
                       for fitting, validation in folds])


def metric_values(prediction, target, train_mean):
    mse = ((prediction - target)**2).mean(0)
    baseline = ((target - train_mean)**2).mean(0).clip(1e-12)
    r2 = 1 - mse / baseline
    return dict(mse=float(mse.mean()), position_r2=float(r2[:4].mean()),
                displacement_r2=float(r2[4:].mean()), r2_per_dimension=r2.tolist(),
                mse_per_dimension=mse.tolist(), baseline_mse_per_dimension=baseline.tolist())


def paired_bootstrap(prediction, target, train_mean, pair_ids, plan, reference=None):
    pairs = sorted(set(pair_ids))
    groups = [[i for i, pair in enumerate(pair_ids) if pair == name] for name in pairs]
    if any(len(group) != 2 for group in groups):
        raise ValueError("Bootstrap requires two complete variants per pair")
    rng = np.random.default_rng(plan["seed"])
    scores, deltas = [], []
    for _ in range(plan["replicates"]):
        indices = np.asarray(groups)[rng.integers(0, len(groups), len(groups))].reshape(-1)
        score = metric_values(prediction[indices], target[indices], train_mean)["displacement_r2"]
        scores.append(score)
        if reference is not None:
            baseline = metric_values(reference[indices], target[indices], train_mean)["displacement_r2"]
            deltas.append(score - baseline)
    tail = (1-plan["confidence"])/2
    return dict(displacement_r2_ci=np.quantile(scores, [tail, 1-tail]).tolist(),
                difference_from_original_ci=np.quantile(deltas, [tail, 1-tail]).tolist() if deltas else None,
                replicates=plan["replicates"], pair_count=len(pairs), interpretation="exploratory")


def run(cache, out, threads=4):
    started = time.monotonic()
    cache, out = Path(cache).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a fresh diagnostic output directory; prior results are preserved")
    if threads < 1 or threads > 4:
        raise ValueError("CPU threads must be between 1 and 4")
    torch.set_num_threads(threads)
    plan = json.loads(PLAN_PATH.read_text())
    out.mkdir(parents=True)
    write_json(out / "frozen_plan.json", plan)
    train, dev = load_records(cache, "train"), load_records(cache, "dev")
    if (len(train), len(dev)) != (plan["data"]["train_episodes"], plan["data"]["dev_episodes"]):
        raise ValueError("This fixed plan requires exactly the existing 128 Train and 128 Dev records")
    for split, rows in (("train", train), ("dev", dev)):
        for row in rows:
            if (row["metadata"]["split"] != split or row["probe_state"].shape != (8,)
                    or row["times"][6:8].tolist() != [6., 7.]):
                raise ValueError("Unexpected split, target dimensions, or cue times")
    y = torch.stack([row["probe_state"] for row in train]).double().numpy()
    target = torch.stack([row["probe_state"] for row in dev]).double().numpy()
    pairs = [row["metadata"]["pair_id"] for row in train]
    dev_pairs = [row["metadata"]["pair_id"] for row in dev]
    if set(pairs) & set(dev_pairs) or any(pairs.count(pair) != 2 for pair in set(pairs)):
        raise ValueError("Split overlap or incomplete Train pairs")
    shuffled = np.asarray(sorted(set(pairs)))
    shuffled = shuffled[np.random.default_rng(plan["sample_size_sensitivity"]["subset_seed"]).permutation(len(shuffled))]
    kernels, selections, energies = {}, {}, {}
    jobs = []
    # Freeze all alpha/fold/subset choices before producing any Dev prediction.
    for mode in plan["representations"]:
        kernel_start = time.monotonic()
        gram, cross, width = frame_grams(train, dev, mode, plan["numerics"]["feature_chunk"])
        views = kernel_views(gram, cross); kernels[mode] = views
        energies[mode] = {name: centered_energy(value[0]) for name, value in views.items()}
        energies[mode].update(frame_feature_dimensions=width, kernel_wall_seconds=time.monotonic()-kernel_start,
            difference_to_concat_centered_energy_ratio=energies[mode]["difference_hybrid"]["mean_centered_feature_energy"] /
                                                       energies[mode]["concat"]["normalization_scale"])
        for count in plan["sample_size_sensitivity"]["train_pairs"]:
            selected_pairs = set(shuffled[:count])
            indices = np.asarray([i for i, pair in enumerate(pairs) if pair in selected_pairs])
            for method in plan["methods"]:
                name = f"{mode}/{method}/pairs{count}/train_cv"
                selection = select_alpha(method, views, indices, y, pairs, plan)
                selections[name] = dict(train_pair_ids=sorted(selected_pairs), **selection)
                jobs.append((name, mode, method, count, indices, selection["selected_alpha"], "train_cv"))
        for method in plan["fixed_alpha_controls"]["methods"]:
            jobs.append((f"{mode}/{method}/pairs64/fixed", mode, method, 64,
                         np.arange(len(train)), plan["fixed_alpha_controls"]["alpha"], "fixed"))
    write_json(out / "train_selection.json", selections)
    write_json(out / "gram_energy.json", energies)
    summary, dimensions, case_rows, details = [], [], [], {}
    references = {mode: method_predict("concat", views, np.arange(len(train)),
                  np.arange(len(dev)), y, .01) for mode, views in kernels.items()}
    for name, mode, method, count, indices, alpha, selection in jobs:
        prediction = method_predict(method, kernels[mode], indices, np.arange(len(dev)), y, alpha)
        mean = y[indices].mean(0)
        metrics = metric_values(prediction, target, mean)
        boot = paired_bootstrap(prediction, target, mean, dev_pairs, plan["bootstrap"], references[mode]) if count == 64 else None
        base = dict(fit=name, representation=mode, method=method, train_pairs=count, alpha=alpha, selection=selection)
        summary.append(dict(**base, position_r2=metrics["position_r2"], displacement_r2=metrics["displacement_r2"],
            mse=metrics["mse"], displacement_ci_low=boot["displacement_r2_ci"][0] if boot else "",
            displacement_ci_high=boot["displacement_r2_ci"][1] if boot else "",
            delta_from_original_ci_low=boot["difference_from_original_ci"][0] if boot else "",
            delta_from_original_ci_high=boot["difference_from_original_ci"][1] if boot else ""))
        details[name] = dict(**base, metrics=metrics, bootstrap=boot)
        for dim, label in enumerate(DIMENSIONS):
            dimensions.append(dict(**base, dimension=label, r2=metrics["r2_per_dimension"][dim],
                mse=metrics["mse_per_dimension"][dim], baseline_mse=metrics["baseline_mse_per_dimension"][dim]))
            for index, row in enumerate(dev):
                error = float(prediction[index, dim] - target[index, dim])
                case_rows.append(dict(**base, episode_id=row["metadata"]["episode_id"], pair_id=dev_pairs[index],
                    dimension=label, target=float(target[index, dim]), prediction=float(prediction[index, dim]),
                    residual=error, squared_error=error**2))
    for filename, rows in (("summary.csv", summary), ("per_dimension.csv", dimensions), ("dev_predictions.csv", case_rows)):
        write_csv(out / filename, rows)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_bytes = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    diagnostics = dict(status="exploratory_diagnostic_complete", test_opened=False,
        gate_changed=False, downstream_training_authorized=False, automatic_dev_winner=None,
        wall_seconds=time.monotonic()-started, peak_process_rss_bytes=peak_bytes, cpu_threads=threads,
        cpu_thread_environment={name: os.environ.get(name) for name in THREAD_VARIABLES},
        plan_sha256=sha256(PLAN_PATH), script_sha256=sha256(__file__),
        cache_manifest_sha256={split: sha256(cache/split/"manifest.json") for split in ("train", "dev")},
        results=details)
    write_json(out / "diagnostics.json", diagnostics)
    lines = ["# Exploratory CPU probe diagnostics", "",
             "This reuses the already inspected Dev set. Alpha selection and sample subsets use Train only. "
             "No original gate, prior result, or Test data is changed. There is no automatic Dev winner or permission to resume scientific training.", "",
             "| Features | Method | Train pairs | Selection | Alpha | Position R² | Displacement R² | Exploratory displacement 95% CI |",
             "|---|---|---:|---|---:|---:|---:|---|"]
    for row in summary:
        interval = f"[{row['displacement_ci_low']:.4f}, {row['displacement_ci_high']:.4f}]" if row["displacement_ci_low"] != "" else "not computed"
        lines.append(f"| {row['representation']} | {row['method']} | {row['train_pairs']} | {row['selection']} | "
                     f"{row['alpha']:g} | {row['position_r2']:.5f} | {row['displacement_r2']:.5f} | {interval} |")
    lines.extend(["", "`difference_hybrid` retains fixed-alpha 0.01 concat position predictions and fits motion to frame differences. "
                  "`shared_position` uses a single position decoder with diagnostic labels p6=p7−displacement; it derives motion "
                  "by subtracting its two decoded positions. Neither structured fit supplies labels to the actor.", "",
                  "R² uses the Dev error of the corresponding Train target mean as its denominator. CV minimizes "
                  "the mean four-displacement-dimension MSE ratio across held-out Train folds. Both variants and both "
                  "frames remain in the same fold. Bootstrap resamples Dev pairs and retains both variants; intervals "
                  "describe this exploratory sample, not independent confirmation or repeated seeds.", "",
                  "[Frozen plan](frozen_plan.json) · [Train-only selection and folds](train_selection.json) · "
                  "[Gram energy/scaling](gram_energy.json) · [Per-dimension metrics](per_dimension.csv) · "
                  "[Dev predictions](dev_predictions.csv) · [Timing, memory, and complete results](diagnostics.json)", ""])
    (out/"README.md").write_text("\n".join(lines))
    return dict(out=str(out), status=diagnostics["status"], fits=len(jobs),
                wall_seconds=diagnostics["wall_seconds"], peak_process_rss_bytes=peak_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path, help="Existing probe_features root containing Train/Dev caches")
    parser.add_argument("--out", required=True, type=Path, help="Fresh output directory")
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.cache, args.out, args.threads), indent=2))
