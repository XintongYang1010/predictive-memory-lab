#!/usr/bin/env python3
"""Exploratory CPU diagnostics of the already-open V5.1 caches and bad cases.

No fitting, model/alpha selection, new gate, RGB generation, GPU, or Test access.
Frozen FIT predictions are in-sample descriptions, not validation evidence.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "4"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import v5_confirmation_reader as reader

AXES = ("A_dx", "A_dy", "B_dx", "B_dy")
METHODS = ("concat", "t6", "t7")


def dump(path, data):
    with open(path, "x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def csvwrite(path, rows):
    with open(path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def position_vectors(target):
    y = np.asarray(target, dtype=np.float64).reshape(-1, 8)
    return np.concatenate((y[:, :4]-y[:, 4:], y[:, :4]), axis=1)


def squared_norms(x):
    result = np.zeros(len(x), dtype=np.float64)
    for lo in range(0, x.shape[1], 2048):
        a = x[:, lo:lo+2048].astype(np.float64)
        result += np.einsum("ij,ij->i", a, a)
    return result


def pair_delta_energy(unique, index):
    energy = np.zeros(len(index), dtype=np.float64)
    for lo in range(0, unique.shape[1], 2048):
        delta = unique[index[:, 1], lo:lo+2048].astype(float)-unique[index[:, 0], lo:lo+2048].astype(float)
        energy += np.einsum("ij,ij->i", delta, delta)
    return energy


def kernel_geometry(cross, query_raw_diag, state, quadratic):
    """Cross-kernel and query norms in the frozen, Train-centered RKHS."""
    mean = cross.mean(1)
    qdiag = (query_raw_diag-2*mean+state["linear_grand"])/state["linear_scale"]
    k = (cross-mean[:, None]-state["linear_columns"]+state["linear_grand"])/state["linear_scale"]
    if quadratic:
        raw = (1+k)**2
        qdiag = ((1+qdiag)**2-2*raw.mean(1)+state["quadratic_grand"])/state["quadratic_scale"]
        k = (raw-raw.mean(1, keepdims=True)-state["quadratic_columns"]+state["quadratic_grand"])/state["quadratic_scale"]
    return k, qdiag


def nearest_distance(cross, query_diag, train_diag, own_group=None):
    d2 = query_diag[:, None]+train_diag[None]-2*cross
    if own_group is not None:
        d2[np.asarray(own_group)[:, None] == np.repeat(np.arange(256), 4)[None]] = np.inf
    return np.sqrt(np.maximum(d2.min(1), 0))


def axis_metrics(p, y, fit_mean):
    p, y = np.asarray(p).reshape(-1, 8)[:, 4:], np.asarray(y, dtype=float).reshape(-1, 8)[:, 4:]
    result = []
    for d, name in enumerate(AXES):
        actual, predicted = y[:, d], p[:, d]
        residual = predicted-actual
        mean_actual, mean_prediction = actual.mean(), predicted.mean()
        cov = np.mean((actual-mean_actual)*(predicted-mean_prediction))
        denominator = np.mean((actual-fit_mean[d+4])**2)
        mse = np.mean(residual**2)
        decomposed = residual.mean()**2 + actual.var()+predicted.var()-2*cov
        np.testing.assert_allclose(mse, decomposed, rtol=1e-12, atol=1e-15)
        eligible = np.abs(actual) > 1e-8
        result.append(dict(axis=name, observations=len(actual), target_mean=float(mean_actual),
            prediction_mean=float(mean_prediction), residual_bias=float(residual.mean()),
            target_std=float(actual.std()), prediction_std=float(predicted.std()),
            prediction_to_target_std_ratio=float(predicted.std()/actual.std()),
            target_mean_abs=float(np.abs(actual).mean()), prediction_mean_abs=float(np.abs(predicted).mean()),
            prediction_to_target_mean_abs_ratio=float(np.abs(predicted).mean()/np.abs(actual).mean()),
            target_prediction_pearson=corr(actual, predicted),
            direction_error_rate=float(np.mean(actual[eligible]*predicted[eligible] < 0)),
            direction_eligible_count=int(eligible.sum()), zero_prediction_count=int((predicted == 0).sum()),
            mse=float(mse), rmse=float(np.sqrt(mse)), fit_baseline_skill=float(1-mse/denominator),
            bias_squared=float(residual.mean()**2), target_variance=float(actual.var()),
            prediction_variance=float(predicted.var()), twice_target_prediction_covariance=float(2*cov)))
    return result


def run(scene, out):
    started = time.monotonic()
    scene, out = Path(scene).resolve(), Path(out).resolve()
    # Must already have completed scientific reports; never release fresh data.
    for role in ("confirm", "bridge"):
        if not (scene/(role+"_report")/"report.json").is_file():
            raise ValueError("Both completed reports must exist before exploratory cache access")
    arrays, selections, freeze = reader.load_frozen(scene/"reader")
    fit = reader.load_cache(scene/"fit_cache/cache.npz", scene/"fit_cache/receipt.json", "fit")
    out.mkdir(parents=True, exist_ok=False)
    fit_unique = arrays["fit_features"].reshape(768, -1)
    fit_index = reader.quartet_indices(256)
    fit_raw_unique = reader.dot_gram(fit_unique, fit_unique)
    fit_raw = reader.view_gram(fit_raw_unique, fit_index, fit_index, "concat")
    state = reader.state_for(arrays, "concat")
    quadratic = selections["concat"]["selected"]["kernel"] == "quadratic"
    _, fit_linear_diag = kernel_geometry(fit_raw, np.diag(fit_raw), state, False)
    _, fit_kernel_diag = kernel_geometry(fit_raw, np.diag(fit_raw), state, quadratic)
    fit_pos = position_vectors(fit["targets"])
    pos_mean, pos_scale = fit_pos.mean(0), fit_pos.std(0)
    pos_lo, pos_hi = fit_pos.min(0), fit_pos.max(0)
    fit_pos_norm = (fit_pos-pos_mean)/pos_scale
    all_axes, observations, groups, roles = [], [], [], {}
    for role in ("fit", "confirm", "bridge"):
        print("diagnosing "+role, flush=True)
        data = fit if role == "fit" else reader.load_cache(scene/(role+"_cache/cache.npz"), scene/(role+"_cache/receipt.json"), role, freeze)
        features, target = data["features"], data["targets"]
        views = 2 if role == "bridge" else 4
        unique = features.reshape(-1, features.shape[-1])
        index = np.arange(1024).reshape(512, 2) if role == "bridge" else fit_index
        raw_unique = fit_raw_unique if role == "fit" else reader.dot_gram(unique, fit_unique)
        raw = reader.view_gram(raw_unique, index, fit_index, "concat")
        norms = squared_norms(unique)
        query_diag = norms[index].sum(1)
        linear, linear_diag = kernel_geometry(raw, query_diag, state, False)
        selected, selected_diag = kernel_geometry(raw, query_diag, state, quadratic)
        own = np.repeat(np.arange(256), views) if role == "fit" else None
        linear_distance = nearest_distance(linear, linear_diag, fit_linear_diag, own)
        selected_distance = nearest_distance(selected, selected_diag, fit_kernel_diag, own)
        predicted = {}
        if role == "fit":
            for method in METHODS:
                k = reader.view_gram(raw_unique, index, fit_index, method)
                predicted[method] = reader.predict_kernel(k, reader.state_for(arrays, method), selections[method]["selected"]["kernel"]).reshape(256, views, 8)
        else:
            report = json.loads((scene/(role+"_report/report.json")).read_text())
            prediction_path = scene/(role+"_report/predictions.npz")
            if reader.sha256(prediction_path) != report["predictions_sha256"] or report["frozen_reader_sha256"] != freeze["receipt_sha256"]:
                raise ValueError("Frozen prediction provenance mismatch")
            with np.load(prediction_path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["targets"], target)
                predicted = {method: z[method] for method in METHODS}
        pos = position_vectors(target)
        pn = (pos-pos_mean)/pos_scale
        distances = ((pn[:, None]-fit_pos_norm[None])**2).mean(2)
        if own is not None:
            distances[own[:, None] == np.repeat(np.arange(256), 4)[None]] = np.inf
        position_distance = np.sqrt(distances.min(1))
        outside = (pos < pos_lo) | (pos > pos_hi)
        delta_energy = pair_delta_energy(unique, index)
        ids = np.arange(256)+(128 if role == "bridge" else 0)
        # Static quartet distractor versus exact original renderer integer-pixel motion.
        yellow_dx = (np.floor((.5+.4*np.sin(7*2.3+ids))*224)-np.floor((.5+.4*np.sin(6*2.3+ids))*224))/224 if role == "bridge" else np.zeros(256)
        descriptors = dict(linear_nn_distance=linear_distance.reshape(256, views),
            selected_kernel_nn_distance=selected_distance.reshape(256, views),
            position_nn_distance=position_distance.reshape(256, views),
            temporal_feature_energy=delta_energy.reshape(256, views),
            position_outside_fit_box=outside.any(1).reshape(256, views),
            x_outside_fit_box=outside[:, [0, 2, 4, 6]].any(1).reshape(256, views),
            y_outside_fit_box=outside[:, [1, 3, 5, 7]].any(1).reshape(256, views))
        role_groups = []
        for i, group_id in enumerate(ids):
            row = dict(role=role, group_id=int(group_id), views=views, yellow_dx=float(yellow_dx[i]),
                primary_motion_mse=float(((predicted["concat"][i, :, 4:]-target[i, :, 4:])**2).mean()))
            for name, values in descriptors.items():
                row[name+"_max"] = bool(values[i].any()) if values.dtype == bool else float(values[i].max())
            role_groups.append(row); groups.append(row)
            for view in range(views):
                row = dict(role=role, group_id=int(group_id), view=view, yellow_dx=float(yellow_dx[i]))
                row.update({name: bool(value[i, view]) if value.dtype == bool else float(value[i, view]) for name, value in descriptors.items()})
                for d, name in enumerate(AXES):
                    row[name+"_target"] = float(target[i, view, d+4])
                    row[name+"_prediction"] = float(predicted["concat"][i, view, d+4])
                observations.append(row)
        for method in METHODS:
            all_axes += [dict(role=role, method=method, **m) for m in axis_metrics(predicted[method], target, arrays[method+"__target_mean"])]
        roles[role] = dict(target=target, predicted=predicted, descriptors=descriptors, groups=role_groups, yellow_dx=yellow_dx)
    csvwrite(out/"axis_metrics.csv", all_axes)
    csvwrite(out/"observation_descriptors.csv", observations)
    csvwrite(out/"whole_group_descriptors.csv", groups)
    csvwrite(out/"worst_bridge_worlds.csv", sorted(roles["bridge"]["groups"], key=lambda row: (-row["primary_motion_mse"], row["group_id"]))[:20])
    strata, reference = [], {}
    for descriptor in ("linear_nn_distance", "selected_kernel_nn_distance", "position_nn_distance", "temporal_feature_energy"):
        values = roles["fit"]["descriptors"][descriptor].max(1)
        reference[descriptor] = np.quantile(values, [.5, .9, 1.]).tolist()
    for role, value in roles.items():
        partitions = {}
        for descriptor in ("position_outside_fit_box", "x_outside_fit_box", "y_outside_fit_box"):
            flag = value["descriptors"][descriptor].any(1)
            partitions[descriptor+"=false"] = ~flag
            partitions[descriptor+"=true"] = flag
        for descriptor, cutpoints in reference.items():
            values = value["descriptors"][descriptor].max(1)
            bins = np.searchsorted(cutpoints, values, side="left")
            for i, label in enumerate(("at_or_below_FIT_median", "FIT_median_to_p90", "FIT_p90_to_max", "above_FIT_max")):
                partitions[descriptor+":"+label] = bins == i
        for label, mask in partitions.items():
            if mask.any():
                metrics = axis_metrics(value["predicted"]["concat"][mask], value["target"][mask], arrays["concat__target_mean"])
                strata += [dict(role=role, stratum=label, whole_groups=int(mask.sum()), **row) for row in metrics]
    csvwrite(out/"primary_group_strata.csv", strata)
    dump(out/"FIT_distance_reference.json", dict(interpretation="Exploratory strata; thresholds from FIT only. FIT nearest neighbors exclude all four views of the same block. Whole-group values use the maximum across views.",
         selected_kernel=selections["concat"]["selected"], position_order=["A_x6", "A_y6", "B_x6", "B_y6", "A_x7", "A_y7", "B_x7", "B_y7"],
         position_min=pos_lo.tolist(), position_max=pos_hi.tolist(), quantiles=reference))
    nuisance = nuisance_diagnostics(roles["bridge"], out)
    make_plots(all_axes, nuisance, out)
    dump(out/"provenance.json", dict(posthoc_only=True, no_fitting_or_selection=True, test_access=False,
         source_scene=str(scene), frozen_reader_sha256=freeze["receipt_sha256"], script_sha256=reader.sha256(__file__),
         cache_sha256={role: reader.sha256(scene/(role+"_cache/cache.npz")) for role in roles},
         elapsed_seconds=time.monotonic()-started))
    write_report(all_axes, strata, nuisance, roles, out)
    return dict(output=str(out), elapsed_seconds=time.monotonic()-started, nuisance_summary=nuisance["summary"])


def nuisance_diagnostics(bridge, out):
    error = bridge["predicted"]["concat"][:, :, 4:]-bridge["target"][:, :, 4:]
    common, contrast = error.mean(1), (error[:, 0]-error[:, 1])/2
    np.testing.assert_allclose((error**2).mean(1), common**2+contrast**2, rtol=1e-12, atol=1e-15)
    dx = bridge["yellow_dx"]
    rng = np.random.default_rng(20261010)
    indices = rng.integers(0, 256, (2000, 256))
    rows = []
    for d, axis in enumerate(AXES):
        a, b = common[indices, d], dx[indices]
        ac, bc = a-a.mean(1, keepdims=True), b-b.mean(1, keepdims=True)
        correlation = np.sum(ac*bc, axis=1)/np.sqrt(np.sum(ac**2, axis=1)*np.sum(bc**2, axis=1))
        fraction = np.mean(a**2, axis=1)/np.mean((error[indices, :, d]**2).mean(2), axis=1)
        rows.append(dict(axis=axis, common_residual_distractor_dx_pearson=corr(common[:, d], dx),
            correlation_ci95_low=float(np.quantile(correlation, .025)), correlation_ci95_high=float(np.quantile(correlation, .975)),
            common_residual_fraction_of_mse=float((common[:, d]**2).mean()/(error[:, :, d]**2).mean()),
            fraction_ci95_low=float(np.quantile(fraction, .025)), fraction_ci95_high=float(np.quantile(fraction, .975)),
            common_residual_rmse=float(np.sqrt((common[:, d]**2).mean())),
            pair_contrast_residual_rmse=float(np.sqrt((contrast[:, d]**2).mean()))))
    csvwrite(out/"distractor_common_error.csv", rows)
    dump(out/"distractor_analysis.json", dict(posthoc_hypothesis_not_causal=True,
        yellow_dx_definition="(int((0.5+0.4*sin(7*2.3+pair))*224)-int((0.5+0.4*sin(6*2.3+pair))*224))/224; positive coordinates make floor and int identical",
        error_decomposition="mean_variants(error^2) = ((e0+e1)/2)^2 + ((e0-e1)/2)^2",
        bootstrap=dict(replicates=2000, seed=20261010, unit="whole bridge paired world", interval="unadjusted exploratory percentile95, conditional on frozen predictions"),
        summary=rows))
    return dict(summary=rows, dx=dx, common=common)


def make_plots(metrics, nuisance, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, d in zip(axes, (0, 2)):
        ax.scatter(nuisance["dx"], nuisance["common"][:, d], s=12, alpha=.65)
        ax.axhline(0, c="gray", lw=.7); ax.axvline(0, c="gray", lw=.7)
        ax.set(xlabel="Yellow distractor horizontal displacement (image fraction)", ylabel="Pair-common target-motion residual",
               title=f"{AXES[d]}: r={nuisance['summary'][d]['common_residual_distractor_dx_pearson']:.3f}")
    fig.suptitle("Exploratory association: shared distractor motion and dx prediction error")
    fig.tight_layout(); fig.savefig(out/"bridge_dx_distractor_association.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(4)
    for i, role in enumerate(("confirm", "bridge")):
        values = [row for row in metrics if row["role"] == role and row["method"] == "concat"]
        axes[0].bar(x+(i-.5)*.34, [row["direction_error_rate"] for row in values], .34, label=role)
        axes[1].bar(x+(i-.5)*.34, [row["prediction_to_target_std_ratio"] for row in values], .34, label=role)
    for ax, title in zip(axes, ("Direction error rate", "Prediction / target standard deviation")):
        ax.set_xticks(x, AXES); ax.set_title(title); ax.legend()
    axes[1].axhline(1, color="black", lw=.7)
    fig.suptitle("Frozen primary reader: confirmation versus original-task bridge")
    fig.tight_layout(); fig.savefig(out/"axis_failure_summary.png", dpi=160); plt.close(fig)


def write_report(metrics, strata, nuisance, roles, out):
    lines = ["# V5.2 exploratory bad-case diagnosis", "", "Posthoc analysis of already-open V5.1 FIT/confirmation/bridge caches. No refit, winner selection, new gate, GPU, or Test access. FIT predictions are in-sample and are not validation evidence.", "",
        "## Horizontal displacement failure", "", "| Axis | Confirm direction error | Bridge direction error | Bridge amplitude ratio (mean absolute) | Bridge prediction/target SD |", "|---|---:|---:|---:|---:|"]
    for axis in AXES:
        c = next(r for r in metrics if (r["role"], r["method"], r["axis"]) == ("confirm", "concat", axis))
        b = next(r for r in metrics if (r["role"], r["method"], r["axis"]) == ("bridge", "concat", axis))
        lines.append(f"| {axis} | {c['direction_error_rate']:.2%} | {b['direction_error_rate']:.2%} | {b['prediction_to_target_mean_abs_ratio']:.3f} | {b['prediction_to_target_std_ratio']:.3f} |")
    lines += ["", "Bridge dx errors combine weak direction agreement with excessive prediction spread; the global signed bias alone is small. `axis_metrics.csv` verifies the exact bias/variance/covariance MSE decomposition.", "", "## Shared nuisance association", ""]
    for row in nuisance["summary"]:
        lines.append(f"- {row['axis']}: correlation of pair-common residual with distractor dx = {row['common_residual_distractor_dx_pearson']:.4f} (exploratory95% CI {row['correlation_ci95_low']:.4f} to {row['correlation_ci95_high']:.4f}); pair-common error contributes {row['common_residual_fraction_of_mse']:.2%} of MSE.")
    lines += ["", "The bridge's yellow marker moves horizontally and identically in both variants; quartet FIT/confirmation hold it fixed. This association is consistent with a nuisance-motion shortcut. It is not a causal test: geometry, support, reflections, and nuisance motion also change across these datasets. No subtraction or correction has been applied to predictions.", "", "## FIT coverage and distances", ""]
    for role in ("fit", "confirm", "bridge"):
        value = roles[role]
        coverage = value["descriptors"]["position_outside_fit_box"].any(1)
        lines.append(f"- {role}: {int(coverage.sum())}/256 whole groups have at least one endpoint coordinate outside the observed FIT min/max box.")
    lines += ["", "`primary_group_strata.csv` gives per-axis skill, direction errors and magnitude statistics by complete-group position coverage and nearest-FIT feature/position distances. Feature distances use the frozen centered normalized linear space and the selected kernel's RKHS. FIT neighbors exclude the entire same quartet. Thresholds are FIT group-max median/p90/max; no bridge-selected cutpoint or model is used. High-dimensional nearest-neighbor distance is a descriptive support measure, not a calibrated OOD probability.", "", "All association intervals are unadjusted exploratory pair bootstraps; they condition on this frozen model and dataset. Positions and nuisance values are analysis-only labels, never actor inputs. Sealed original results remain unchanged."]
    (out/"README.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_root, args.out), indent=2))
