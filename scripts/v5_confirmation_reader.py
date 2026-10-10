#!/usr/bin/env python3
"""CPU-only frozen readers for the registered V5 temporal confirmation.

Consumes independently extracted, pickle-free caches. Never generates data,
loads an encoder, selects on confirmation, or changes the original V5 gate.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "4")

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from v5_temporal_confirmation import (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, KERNELS,
    ORDERS, SCHEMA, choose_reader, contrast_arrays, score_quartets)

INPUTS = ("concat", "t6", "t7")
IDENTITY_KEYS = ("plan_sha256", "source_commit", "encoder_identity", "source_identity",
                 "pooling", "feature_dtype")
PLAN = Path(__file__).with_name("v5_temporal_confirmation_plan.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with open(path, "x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def identity(receipt):
    if any(k not in receipt for k in IDENTITY_KEYS):
        raise ValueError("Cache receipt lacks required identity fields")
    return {k: receipt[k] for k in IDENTITY_KEYS}


def validate_entry_hashes(data, entries):
    if set(data) != set(entries):
        raise ValueError("NPZ entry/hash key sets differ")
    for key, array in data.items():
        expected = dict(sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
                        dtype=str(array.dtype), shape=list(array.shape))
        if entries[key] != expected:
            raise ValueError("NPZ entry hash/dtype/shape mismatch: "+key)


def load_cache(path, receipt_path, role, frozen=None):
    """Production boundary: exact registered counts, IDs, schema, hashes, roles."""
    if role not in ("fit", "confirm", "bridge"):
        raise ValueError("Only registered fit/confirm/bridge caches are accepted")
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("role") != role:
        raise ValueError("Cache receipt schema/role mismatch")
    if receipt.get("cache_sha256") != sha256(path):
        raise ValueError("Cache SHA256 mismatch")
    cache_identity = identity(receipt)
    if cache_identity["plan_sha256"] != sha256(PLAN):
        raise ValueError("Cache uses a different registered plan")
    if receipt["pooling"] != "4x4_flattened" or receipt["feature_dtype"] != "float32":
        raise ValueError("Primary cache must contain float32 flattened 4x4 pooled features")
    if not isinstance(receipt.get("entry_hashes"), dict) or not receipt["entry_hashes"]:
        raise ValueError("Cache receipt requires entry hashes")
    if frozen is not None:
        if cache_identity != frozen["cache_identity"]:
            raise ValueError("FIT and evaluation feature identities differ")
        if receipt.get("frozen_reader_sha256") != frozen["receipt_sha256"]:
            raise ValueError("Evaluation was not bound to this frozen reader receipt")
        if receipt.get("frozen_before_generation") is not True or receipt.get("unseen_before_release") is not True:
            raise ValueError("Evaluation requires freeze-before-generation and unseen-data attestations")
    elif role != "fit":
        raise ValueError("Evaluation requires a verified frozen reader")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if data.get("role", np.array("")).shape != () or str(data["role"].item()) != role:
        raise ValueError("NPZ role mismatch")
    x, y = data.get("features"), data.get("targets")
    expected_x = (256, 2, 2) if role == "bridge" else (256, 3)
    expected_y = (256, 2, 8) if role == "bridge" else (256, 4, 8)
    if x is None or x.dtype != np.float32 or x.ndim != len(expected_x)+1 or x.shape[:-1] != expected_x:
        raise ValueError("Feature shape/count/dtype does not match registered production cache")
    if x.shape[-1] < 16 or x.shape[-1] % 16 or y is None or y.shape != expected_y:
        raise ValueError("Targets or flattened 4x4 pooled dimension mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite cache")
    key = "pair_ids" if role == "bridge" else "block_ids"
    ids = data.get(key)
    expected = np.arange(128, 384) if role == "bridge" else np.arange(256)
    if ids is None or not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(ids, expected):
        raise ValueError("Cache must have exactly 256 unique registered IDs in canonical order")
    validate_entry_hashes(data, receipt["entry_hashes"])
    if role != "bridge":
        if receipt.get("unique_image_rule") != "encode_once_reuse":
            raise ValueError("Quartet images must be encoded once and reused")
        contrast_arrays(y[:, :, 4:], y[:, :, 4:])
        # Position labels must agree with the registered quartet view order.
        if not (np.array_equal(y[:, 0, :4], y[:, 1, :4]) and
                np.allclose(y[:, 2, :4]-y[:, 0, :4], y[:, 0, 4:], rtol=0, atol=1e-7) and
                np.allclose(y[:, 0, :4]-y[:, 3, :4], y[:, 0, 4:], rtol=0, atol=1e-7)):
            raise ValueError("Quartet position labels disagree with registered view order")
    data["receipt"] = receipt
    data["receipt_sha256"] = sha256(receipt_path)
    return data


def quartet_indices(blocks):
    return (np.arange(blocks)[:, None, None]*3 + ORDERS[None]).reshape(-1, 2)


def dot_gram(query_unique, train_unique, chunk=2048):
    """Float64 Gram without materializing expanded high-dimensional views."""
    a, b = np.asarray(query_unique), np.asarray(train_unique)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Incompatible feature dimensions")
    out = np.zeros((len(a), len(b)), dtype=np.float64)
    for lo in range(0, a.shape[1], chunk):
        out += a[:, lo:lo+chunk].astype(np.float64) @ b[:, lo:lo+chunk].astype(np.float64).T
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite Gram matrix")
    return out


def view_gram(unique_gram, query_index, fit_index, input_name):
    if input_name not in INPUTS:
        raise ValueError("Unregistered reader input")
    columns = (0, 1) if input_name == "concat" else ((0,) if input_name == "t6" else (1,))
    return sum(unique_gram[np.ix_(query_index[:, c], fit_index[:, c])] for c in columns)


def fit_kernel(gram, targets, kernel, alpha):
    """Persist every moment needed to predict without refitting/target access."""
    if kernel not in KERNELS or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Unregistered kernel or invalid regularization")
    g = np.asarray(gram, dtype=np.float64)
    state = {}
    for name in (("linear", "quadratic") if kernel == "quadratic" else ("linear",)):
        columns, grand = g.mean(0), float(g.mean())
        centered = g-g.mean(1, keepdims=True)-columns+grand
        scale = max(float(np.diag(centered).mean()), 1e-12)
        state[name+"_columns"] = columns
        state[name+"_grand"] = np.array(grand)
        state[name+"_scale"] = np.array(scale)
        g = centered/scale
        if name == "linear" and kernel == "quadratic":
            g = (1+g)**2
    y = np.asarray(targets, dtype=np.float64)
    state["target_mean"] = y.mean(0)
    state["coefficients"] = np.linalg.solve(g+len(y)*alpha*np.eye(len(y)), y-state["target_mean"])
    return state


def predict_kernel(cross, state, kernel):
    q = np.asarray(cross, dtype=np.float64)
    for name in (("linear", "quadratic") if kernel == "quadratic" else ("linear",)):
        q = (q-q.mean(1, keepdims=True)-state[name+"_columns"]+state[name+"_grand"])/state[name+"_scale"]
        if name == "linear" and kernel == "quadratic":
            q = (1+q)**2
    prediction = q @ state["coefficients"] + state["target_mean"]
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite frozen prediction")
    return prediction


def train_models(features, targets, block_ids):
    """Mathematical primitive; CLI count/role enforcement occurs in load_cache."""
    blocks = len(features)
    unique = np.asarray(features).reshape(blocks*3, -1)
    index = quartet_indices(blocks)
    raw = dot_gram(unique, unique)
    groups = np.repeat(np.asarray(block_ids), 4)
    y = np.asarray(targets).reshape(-1, 8)
    arrays = {"fit_features": np.asarray(features), "fit_block_ids": np.asarray(block_ids)}
    selections = {}
    for name in INPUTS:
        gram = view_gram(raw, index, index, name)
        selection = choose_reader(gram, y, groups)
        selected = selection["selected"]
        state = fit_kernel(gram, y, selected["kernel"], selected["mean_loss_alpha"])
        arrays.update({name+"__"+k: v for k, v in state.items()})
        selections[name] = selection
    return arrays, selections


def state_for(arrays, name):
    return {k.split("__", 1)[1]: v for k, v in arrays.items() if k.startswith(name+"__")}


def predict_readers(arrays, selections, query_unique, query_index):
    fit = arrays["fit_features"]
    fit_index = quartet_indices(len(fit))
    raw = dot_gram(query_unique, fit.reshape(len(fit)*3, -1))
    return {name: predict_kernel(view_gram(raw, query_index, fit_index, name),
             state_for(arrays, name), selections[name]["selected"]["kernel"])
            for name in INPUTS}


def current_sources():
    return {p.name: sha256(p) for p in (Path(__file__), PLAN, Path(__file__).with_name("v5_temporal_confirmation.py"))}


def fit(cache_path, receipt_path, out):
    started = time.time()
    data = load_cache(cache_path, receipt_path, "fit")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous fit/freeze.
    arrays, selections = train_models(data["features"], data["targets"], data["block_ids"])
    np.savez(out/"frozen_model.npz", **arrays)
    write_json(out/"train_selection.json", selections)
    receipt = dict(schema=SCHEMA, status="frozen_before_confirmation", fit_blocks=256,
        fit_views=1024, fit_cache_sha256=sha256(cache_path), fit_cache_receipt_sha256=sha256(receipt_path),
        cache_identity=identity(data["receipt"]), sources=current_sources(),
        environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
        files={p.name: sha256(p) for p in (out/"frozen_model.npz", out/"train_selection.json")},
        predictor_input_audit=dict(features_only=True, inputs=list(INPUTS), labels_or_group_ids_used_at_prediction=False),
        elapsed_seconds=time.time()-started, original_v5_gate="failed_unchanged", p0_authorized=False)
    # Written last: its existence means all model artifacts have been serialized.
    write_json(out/"freeze_receipt.json", receipt)
    return receipt


def load_frozen(folder):
    folder = Path(folder)
    path = folder/"freeze_receipt.json"
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "frozen_before_confirmation":
        raise ValueError("No completed registered freeze receipt")
    if receipt.get("fit_blocks") != 256 or receipt.get("fit_views") != 1024:
        raise ValueError("Frozen production fit count mismatch")
    if receipt.get("sources") != current_sources():
        raise ValueError("Reader/helper/plan changed after freeze")
    for filename in ("frozen_model.npz", "train_selection.json"):
        if receipt.get("files", {}).get(filename) != sha256(folder/filename):
            raise ValueError("Frozen model/selection hash mismatch")
    with np.load(folder/"frozen_model.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if arrays["fit_features"].shape[:2] != (256, 3) or not np.array_equal(arrays["fit_block_ids"], np.arange(256)):
        raise ValueError("Frozen training basis count/order mismatch")
    selections = json.loads((folder/"train_selection.json").read_text())
    receipt["receipt_sha256"] = sha256(path)
    return arrays, selections, receipt


def score_bridge(prediction, target, fit_mean):
    """Paired-world bootstrap, conditional on this one frozen FIT reader."""
    p, y, mean = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64), np.asarray(fit_mean, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (2, 8) or mean.shape != (8,):
        raise ValueError("Expected paired-world [pairs,2,8] predictions and FIT mean")
    if not all(np.isfinite(a).all() for a in (p, y, mean)):
        raise ValueError("Nonfinite bridge inputs")
    error = ((p-y)**2).mean(1)
    baseline = ((y-mean)**2).mean(1)
    if np.any(baseline.mean(0) <= 0):
        raise ValueError("Degenerate bridge baseline")
    skill = 1-error.mean(0)/baseline.mean(0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = []
    for lo in range(0, BOOTSTRAP_REPLICATES, 250):
        ids = rng.integers(0, len(p), (min(250, BOOTSTRAP_REPLICATES-lo), len(p)))
        den = baseline[ids, 4:].mean(1)
        if np.any(den <= 0):
            raise ValueError("Degenerate resampled bridge baseline")
        boot.append((1-error[ids, 4:].mean(1)/den).mean(1))
    lower = float(np.quantile(np.concatenate(boot), .05))
    checks = dict(position_skill_over_point_one=float(skill[:4].mean()) > .1,
                  all_four_motion_coordinate_skills_positive=bool(np.all(skill[4:] > 0)),
                  motion_mean_lower_bound_positive=lower > 0)
    return dict(per_dimension_skill=skill.tolist(), position_skill=float(skill[:4].mean()),
        motion_skill=float(skill[4:].mean()), motion_mean_lower_bound=lower, checks=checks,
        metric_conditions_met=all(checks.values()), p0_authorized=False,
        bootstrap=dict(replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, unit="paired_world",
                       one_sided_quantile=.05, baseline="frozen_FIT_target_mean", conditional_on_frozen_reader=True))


def evaluate(model_folder, cache_path, receipt_path, role, out):
    arrays, selections, frozen = load_frozen(model_folder)
    data = load_cache(cache_path, receipt_path, role, frozen)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    x, y = data["features"], data["targets"]
    if x.shape[-1] != arrays["fit_features"].shape[-1]:
        raise ValueError("FIT/evaluation feature dimensions differ")
    if role == "confirm":
        index = quartet_indices(256)
        repeat_index = np.repeat((np.arange(256)*3+1)[:, None], 2, axis=1)
        combined = predict_readers(arrays, selections, x.reshape(768, -1), np.concatenate((index, repeat_index)))
        predictions = {k: p[:1024].reshape(256, 4, 8) for k, p in combined.items()}
        repeats = {k: p[1024:, 4:] for k, p in combined.items()}
        scores = {k: score_quartets(predictions[k], y, repeats[k], arrays[k+"__target_mean"][:4]) for k in INPUTS}
        saved = {**{k: v for k, v in predictions.items()}, **{k+"_repeat": v for k, v in repeats.items()}, "targets": y}
    elif role == "bridge":
        index = np.arange(1024).reshape(512, 2)
        predictions = {k: p.reshape(256, 2, 8) for k, p in predict_readers(arrays, selections, x.reshape(1024, -1), index).items()}
        scores = {k: score_bridge(predictions[k], y, arrays[k+"__target_mean"]) for k in INPUTS}
        saved = {**predictions, "targets": y}
    else:
        raise ValueError("Evaluation role must be confirm or bridge")
    np.savez(out/"predictions.npz", **saved)
    report = dict(schema=SCHEMA, role=role, cache_sha256=sha256(cache_path), cache_receipt_sha256=sha256(receipt_path),
        frozen_reader_sha256=frozen["receipt_sha256"], sources=current_sources(), scores=scores,
        primary_reader="concat", controls="t6 and t7 reported without winner selection",
        primary_metric_conditions_met=scores["concat"]["metric_conditions_met"], p0_authorized=False,
        original_v5_gate="failed_unchanged", predictions_sha256=sha256(out/"predictions.npz"))
    report.update(cache_identity=identity(data["receipt"]), test_access=False,
                  counts=dict(blocks=256, views=1024) if role == "confirm" else dict(paired_worlds=256, episodes=512))
    # Copy exact bytes so the report's receipt SHA refers to the archived copy.
    with open(out/"cache_receipt.json", "xb") as f:
        f.write(Path(receipt_path).read_bytes())
    write_json(out/"report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "confirm", "bridge"))
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", type=Path)
    args = p.parse_args()
    if args.phase == "fit":
        if args.model is not None:
            p.error("FIT cannot read an existing model")
        result = fit(args.cache, args.receipt, args.out)
    else:
        if args.model is None:
            p.error("Confirmation/bridge require --model frozen directory")
        result = evaluate(args.model, args.cache, args.receipt, args.phase, args.out)
    print(json.dumps({k: result[k] for k in ("schema", "p0_authorized")}, sort_keys=True))


if __name__ == "__main__":
    main()
