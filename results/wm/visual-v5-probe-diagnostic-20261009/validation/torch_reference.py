#!/usr/bin/env python3
"""Independent feature-matrix/Torch check of the pooled concat ridge results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def load_pooled(cache, split):
    """Stream one record at a time; retain only two pooled frames and labels."""
    if split not in ("train", "dev"):
        raise ValueError("Only Train and Dev are allowed")
    folder = cache/split
    manifest = json.loads((folder/"manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["configuration"]["count"] != 128:
        raise ValueError("Expected the complete original 128-episode cache")
    x = None
    y = torch.empty((128, 8), dtype=torch.float64)
    pairs = []
    for index in range(128):
        pair, variant = divmod(index, 2)
        path = folder/f"{pair:06d}-{variant}.pt"
        expected_hash = manifest["files"][path.name]
        if path.stat().st_size != expected_hash["bytes"] or file_hash(path) != expected_hash["sha256"]:
            raise ValueError(f"Cache hash mismatch: {path}")
        record = torch.load(path, map_location="cpu", weights_only=True)
        if (record["metadata"]["episode_id"] != f"{split}-{pair:06d}-{variant}"
                or record["times"][6:8].tolist() != [6., 7.]):
            raise ValueError("Wrong episode or cue times")
        feature = record["features"][6:8].float().flatten().double()
        if x is None:
            x = torch.empty((128, len(feature)), dtype=torch.float64)
        x[index] = feature
        y[index] = record["probe_state"].double()
        pairs.append(record["metadata"]["pair_id"])
        del record, feature
    return x, y, pairs


def explicit_fit(x, query, y, alpha):
    """Match the original probe directly in feature space, without Gram centering."""
    mean = x.mean(0)
    a = x-mean
    b = query-mean
    scale = a.square().sum(1).mean().clamp_min(1e-12)
    target_mean = y.mean(0)
    kernel = a@a.T/scale
    coefficients = torch.linalg.solve(kernel + alpha*torch.eye(len(a)), y-target_mean)
    return (b@a.T/scale)@coefficients+target_mean


def validate(cache):
    started = time.monotonic()
    torch.set_num_threads(4)
    core = RESULTS/"core_probe"
    plan = json.loads((core/"frozen_plan.json").read_text())
    selection = json.loads((core/"train_selection.json").read_text())["pooled/concat/pairs64/train_cv"]
    recorded = json.loads((core/"diagnostics.json").read_text())["results"]["pooled/concat/pairs64/train_cv"]
    x, y, pairs = load_pooled(cache, "train")
    unique = np.asarray(sorted(set(pairs)))
    shuffled = unique[np.random.default_rng(plan["cv"]["seed"]).permutation(len(unique))]
    folds = []
    for held_pairs in np.array_split(shuffled, plan["cv"]["folds"]):
        held = set(held_pairs)
        fitting = [index for index, pair in enumerate(pairs) if pair not in held]
        validation = [index for index, pair in enumerate(pairs) if pair in held]
        folds.append((fitting, validation))
    fold_identical = [dict(fit_pair_ids=sorted({pairs[i] for i in fitting}),
                           validation_pair_ids=sorted({pairs[i] for i in held}))
                      for fitting, held in folds] == selection["folds"]
    if not fold_identical:
        raise AssertionError("Independent Train pair folds differ")
    trials = []
    maximum_cv_difference = 0.
    for alpha in plan["alpha_grid"]:
        scores = []
        for fitting, held in folds:
            fit_y = y[fitting]
            predicted = explicit_fit(x[fitting], x[held], fit_y, alpha)
            error = (predicted[:, 4:]-y[held, 4:]).square().mean(0)
            baseline = (y[held, 4:]-fit_y.mean(0)[4:]).square().mean(0).clamp_min(1e-12)
            scores.append(float((error/baseline).mean()))
        old = next(trial for trial in selection["trials"] if trial["alpha"] == alpha)
        actual = torch.tensor(scores, dtype=torch.float64)
        expected = torch.tensor(old["fold_scores"], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-10)
        delta = float((actual-expected).abs().max())
        maximum_cv_difference = max(maximum_cv_difference, delta)
        trials.append(dict(alpha=alpha, fold_scores=scores, mean_score=float(actual.mean()),
                           maximum_absolute_difference=delta, matches=True))
    chosen = min(trials, key=lambda value: (value["mean_score"], value["alpha"]))["alpha"]
    if chosen != 1. or chosen != selection["selected_alpha"]:
        raise AssertionError("Independent pooled concat alpha selection differs")
    # Dev is loaded only after independently verifying every Train-CV score.
    z, target, _ = load_pooled(cache, "dev")
    predicted = explicit_fit(x, z, y, chosen)
    error = (predicted-target).square().mean(0)
    baseline = (target-y.mean(0)).square().mean(0).clamp_min(1e-12)
    r2 = 1-error/baseline
    expected = torch.tensor(recorded["metrics"]["r2_per_dimension"], dtype=torch.float64)
    torch.testing.assert_close(r2, expected, rtol=1e-7, atol=1e-10)
    elapsed = time.monotonic()-started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(status="verified", implementation="Explicit centered pooled feature matrices + torch.linalg.solve; no NumPy solve or Gram-space centering",
        reference="core_probe/train_selection.json and core_probe/diagnostics.json",
        cpu_threads=4, test_opened=False, original_gate_changed=False, rtol=1e-7, atol=1e-10,
        grouped_folds_identical=fold_identical, checked_fold_scores=len(trials)*len(folds),
        trials=trials, maximum_cv_score_difference=maximum_cv_difference, selected_alpha=chosen,
        dev_r2_per_dimension=r2.tolist(), dev_position_r2=float(r2[:4].mean()),
        dev_displacement_r2=float(r2[4:].mean()),
        maximum_dev_r2_difference=float((r2-expected).abs().max()),
        all_outputs_finite=bool(torch.isfinite(r2).all()),
        wall_seconds=elapsed, peak_process_rss_bytes=int(usage.ru_maxrss*(1 if sys.platform=="darwin" else 1024)),
        script_sha256=file_hash(Path(__file__)),
        cache_manifest_sha256={split: file_hash(cache/split/"manifest.json") for split in ("train", "dev")},
        interpretation="Independent reproduction of the exploratory diagnostic; no new gate or independent confirmation dataset.")
    (HERE/"torch_reference.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.cache.resolve()), indent=2))
