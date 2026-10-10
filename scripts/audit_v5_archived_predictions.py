#!/usr/bin/env python3
"""CPU audit of archived predictions; no encoder, cache fitting, GPU, or Test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmr.visual_memory.data import episode, seed_for, specs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_state(split, pair, variant):
    """Independent scalar fold, not data.reflect; only t6/t7, before any update."""
    rng = np.random.default_rng(seed_for(split, pair))
    p = rng.uniform([.2, .23], [.8, .32], (2, 2))
    p[1, 1] += .42
    v = rng.uniform(.008, .018, (2, 2)) * rng.choice([-1, 1], (2, 2))
    v *= 1 if variant == 0 else -1
    bounds = [(.1, .9), (.15, .40), (.1, .9), (.60, .85)]
    positions, tiles = [], []
    for t in (6, 7):
        folded, cell = [], []
        for q, (lo, hi) in zip((p + t*v).ravel(), bounds):
            span = hi - lo
            k = int(np.floor((q - lo) / span))
            r = (q - lo) - k * span
            folded.append(lo + (r if k % 2 == 0 else span-r))
            cell.append(k)
        positions.append(folded); tiles.append(cell)
    p6, p7 = np.array(positions)
    return np.r_[p7, p7-p6].astype(np.float32), np.array(tiles[0]) != tiles[1]


def pixel_positions(rgb):
    """Generator-specific color oracle; label information, never actor input."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    masks = [(r > 200) & (g < 120) & (b < 100),
             (b > 200) & (g < 120) & (r < 100)]
    positions = []
    for mask in masks:
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError("Visible object missing from RGB color oracle")
        positions += [xx.mean()/rgb.shape[1], yy.mean()/rgb.shape[0]]
    return np.array(positions)


def score(target, prediction, train_mean):
    mse = ((prediction-target)**2).mean(0)
    baseline = ((target-train_mean)**2).mean(0)
    skill = 1-mse/np.maximum(baseline, 1e-12)
    conventional = 1-mse/np.maximum(((target-target.mean(0))**2).mean(0), 1e-12)
    return dict(skill_train_mean=skill.tolist(), r2_dev_mean=conventional.tolist(),
                position_skill=float(skill[:4].mean()), displacement_skill=float(skill[4:].mean()),
                mse=mse.tolist(), rmse_pixels=(np.sqrt(mse)*224).tolist(),
                baseline_rmse_pixels=(np.sqrt(baseline)*224).tolist())


def run(archive):
    csv_path = archive / "run/feature_diagnostics/dev_predictions.csv"
    dim_path = archive / "run/feature_diagnostics/r2_by_dimension.csv"
    rows = list(csv.DictReader(csv_path.open(newline="")))
    dims = list(csv.DictReader(dim_path.open(newline="")))
    means = np.array([float(x["train_target_mean"]) for x in dims if x["mode"] == "pooled"])
    target, predictions = np.full((128, 8), np.nan), {}
    for mode in ("raw", "pooled"):
        prediction = np.full((128, 8), np.nan)
        seen = set()
        for row in rows:
            if row["mode"] != mode:
                continue
            pair = int(row["pair_id"].rsplit("-", 1)[1]); variant = int(row["variant"])
            index, dim = pair*2+variant, int(row["dimension_index"])
            assert row["episode_id"] == f"dev-{pair:06d}-{variant}"
            assert (index, dim) not in seen; seen.add((index, dim))
            value, pred = float(row["target"]), float(row["prediction"])
            if np.isfinite(target[index, dim]):
                assert target[index, dim] == value
            target[index, dim] = value; prediction[index, dim] = pred
            assert np.isclose((pred-value)**2, float(row["squared_error"]), rtol=1e-12, atol=1e-15)
        assert len(seen) == 128*8 and np.isfinite(prediction).all()
        predictions[mode] = prediction

    generated, bounces, pixels, frame_checks, recent_checks = {}, {}, {}, 0, 0
    seed_sets = {}
    for split in ("train", "dev"):
        states, reflected, oracles = [], [], []
        previous = None
        seed_sets[split] = {seed_for(split, pair) for pair in range(64)}
        for sp, pair, variant in specs(split, 128):
            ep = episode(sp, pair, variant)
            expected, bounce = physical_state(sp, pair, variant)
            np.testing.assert_array_equal(ep.probe_state, expected)
            np.testing.assert_array_equal(ep.times[6:8], [6., 7.])
            np.testing.assert_array_equal(ep.shutters[6:8], np.ones((2, 2)))
            p6, p7 = (pixel_positions(ep.rgb[t]) for t in (6, 7))
            states.append(ep.probe_state); reflected.append(bounce); oracles.append(np.r_[p7, p7-p6])
            frame_checks += 1
            if variant == 1:
                np.testing.assert_array_equal(previous.rgb[-4:], ep.rgb[-4:])
                np.testing.assert_array_equal(previous.times, ep.times)
                np.testing.assert_array_equal(previous.shutters, ep.shutters)
                recent_checks += 1
            previous = ep
        generated[split], bounces[split], pixels[split] = map(np.array, (states, reflected, oracles))
    np.testing.assert_array_equal(target, generated["dev"].astype(float))
    np.testing.assert_allclose(means, generated["train"].astype(float).mean(0), rtol=0, atol=1e-15)
    assert len(seed_sets["train"]) == len(seed_sets["dev"]) == 64
    assert not seed_sets["train"] & seed_sets["dev"]

    # Fixed-model uncertainty, resampling paired worlds jointly for every metric.
    # This does not include training-set/model-selection uncertainty.
    rng = np.random.default_rng(20261010)
    draws = rng.integers(0, 64, size=(5000, 64))
    boot = {}
    baseline_pair = ((target[:, 4:]-means[4:])**2).reshape(64, 2, 4).mean(1)
    metrics = {}
    for mode, prediction in predictions.items():
        metrics[mode] = score(target, prediction, means)
        errors = ((prediction[:, 4:]-target[:, 4:])**2).reshape(64, 2, 4).mean(1)
        boot[mode] = (1-errors[draws].mean(1)/baseline_pair[draws].mean(1)).mean(1)
        metrics[mode]["displacement_skill_pair_bootstrap_95pct"] = np.quantile(boot[mode], [.025, .975]).tolist()
        metrics[mode]["displacement_sign_accuracy"] = (np.sign(prediction[:, 4:]) == np.sign(target[:, 4:])).mean(0).tolist()
        metrics[mode]["displacement_sign_majority_train_baseline"] = (np.sign(means[4:]) == np.sign(target[:, 4:])).mean(0).tolist()
        metrics[mode]["bounce_stratified_mse"] = []
        for dim in range(4):
            stats = {}
            for value, name in ((False, "no_t6_t7_bounce"), (True, "t6_t7_bounce")):
                take = bounces["dev"][:, dim] == value
                stats[name] = dict(n=int(take.sum()), mse=float(((prediction[take, dim+4]-target[take, dim+4])**2).mean()) if take.any() else None)
            metrics[mode]["bounce_stratified_mse"].append(stats)
    zero = np.broadcast_to(means, target.shape).copy(); zero[:, 4:] = 0
    return dict(schema="visual-v5-archived-prediction-review-1", source_commit="d0b4fef25e98eefb58bfba737997cb0145f27daa",
                result_commit="d4b6fc622d0fe08e28605aaf6e48100201ab2008",
                evidence_sha256={"dev_predictions.csv": digest(csv_path), "r2_by_dimension.csv": digest(dim_path),
                                 "data.py": digest(Path(__file__).resolve().parents[1]/"pmr/visual_memory/data.py")},
                scope="Archived predictions and deterministic Train/Dev RGB regeneration only; feature tensors unavailable here",
                gpu_jobs_started=0, test_opened=False, original_gate_passed=False,
                checks=dict(cue_episodes=frame_checks, recent_pair_equality=recent_checks,
                            archived_targets_exactly_match_generator=True, independent_fold_exact=True,
                            train_dev_seed_collision=False, per_split_independent_worlds=64),
                bootstrap=dict(seed=20261010, replicates=5000, unit="paired_world", uncertainty="conditional on fixed fitted models"),
                metrics=metrics,
                pooled_minus_raw_displacement_skill_ci=np.quantile(boot["pooled"]-boot["raw"], [.025, .975]).tolist(),
                zero_displacement_control=score(target, zero, means),
                rgb_color_oracle=score(target, pixels["dev"], means),
                bounce_counts={k:v.sum(0).tolist() for k,v in bounces.items()},
                non_antisymmetric_pair_displacements={k:(np.abs(v.reshape(64, 2, 8)[:, :, 4:].sum(1)) > 1e-7).sum(0).tolist() for k,v in generated.items()},
                versions=dict(numpy=np.__version__, pillow=__import__("PIL").__version__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an existing review output")
    if args.archive.resolve() in args.out.resolve().parents:
        parser.error("Review output must be outside the archived result directory")
    result = run(args.archive.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"out":str(args.out), "original_gate_passed":False, "metrics":result["metrics"]}, indent=2))
