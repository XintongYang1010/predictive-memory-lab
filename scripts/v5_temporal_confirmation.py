#!/usr/bin/env python3
"""V5 review sidecar: matched temporal quartets and CPU scoring primitives.

No encoder, GPU, Slurm, Test, fitting, or production-gate integration. The only
samples generated during review tests use the separate `software_test` namespace.
Real fit/confirm generation and encoding have NOT been executed by this review.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "v5-temporal-confirm-1"
ROLES = ("software_test", "fit", "confirm")
ORDERS = np.array(((0, 1), (2, 1), (1, 2), (1, 0)), dtype=int)
# Unique images represent C-d, C, C+d. Rows are end+, end-, start+, start-.
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 7709
FAMILYWISE_ALPHA = .05
PRIMARY_ENDPOINTS = 9  # eight anchored coordinate skills plus absolute mean skill
KERNELS = ("linear", "quadratic")
MEAN_LOSS_ALPHAS = tuple(x/128 for x in (1e-4, 1e-3, .01, .1, 1., 10.))


@dataclass
class Quartet:
    unique_rgb: np.ndarray
    positions: np.ndarray
    targets: np.ndarray
    metadata: dict

    def actor_inputs(self):
        """Public two-cue observations only. Analysis identity/labels stay outside."""
        return dict(rgb=self.unique_rgb[ORDERS],
                    times=np.tile(np.array([6., 7.], dtype=np.float32), (4, 1)),
                    shutters=np.ones((4, 2, 2), dtype=np.float32))


def seed_for(role, block):
    if role not in ROLES or not isinstance(block, int) or not 0 <= block < 256:
        raise ValueError("Allowed roles: software_test/fit/confirm; block IDs 0..255. No Test role.")
    return int.from_bytes(hashlib.sha256(f"{SCHEMA}/{role}/{block}".encode()).digest()[:8], "little")


def render(positions, colors, distractor_x, size=224):
    """Same grid, object glyphs and lane layout as V5; static nuisance per block."""
    im = Image.new("RGB", (size, size), (30, 35, 42))
    draw = ImageDraw.Draw(im)
    for xx in range(0, size, max(1, size//16)):
        draw.line((xx, 0, xx, size), fill=(42, 47, 54))
    radius = size*.06
    for obj, (px, py) in enumerate(positions.reshape(2, 2)):
        x, y = px*size, py*size
        box = (int(x-radius), int(y-radius), int(x+radius), int(y+radius))
        draw.ellipse(box, fill=colors[obj], outline=(250, 250, 240), width=max(1, size//100))
        draw.line((x-radius/2, y, x+radius/2, y), fill=(255, 255, 255), width=max(1, size//60))
        if obj == 1:
            draw.line((x, y-radius/2, x, y+radius/2), fill=(255, 255, 255), width=max(1, size//60))
    x = int(distractor_x*size)
    draw.rectangle((x-5, size*.48, x+5, size*.52), fill=(240, 220, 50))
    return np.asarray(im).copy()


def make_quartet(role, block):
    """Three images, four ordered views; primary confirmation has no cue bounce."""
    rng = np.random.default_rng(seed_for(role, block))
    center = rng.uniform([.20, .20, .20, .65], [.80, .35, .80, .80])
    displacement = rng.uniform(.008, .018, 4)*rng.choice([-1, 1], 4)
    # Dyadic coordinates remove roundoff ambiguity in exact reversal/equality
    # checks; resolution is far below one image pixel. No rejection sampling.
    center = np.round(center*2**20)/2**20
    displacement = np.round(displacement*2**20)/2**20
    positions = np.stack((center-displacement, center, center+displacement))
    colors = [tuple(int(x) for x in c) for c in rng.integers(60, 240, (2, 3))]
    colors[0] = (235, colors[0][1]//2, 60)
    colors[1] = (60, colors[1][1]//2, 235)
    nuisance_x = rng.uniform(.1, .9)
    images = np.stack([render(p, colors, nuisance_x) for p in positions])
    start, end = positions[ORDERS[:, 0]], positions[ORDERS[:, 1]]
    targets = np.concatenate((end, end-start), axis=1).astype(np.float32)
    return Quartet(images, positions, targets,
                   dict(schema=SCHEMA, role=role, block=block,
                        group_id=f"{SCHEMA}/{role}/{block:06d}", cue_reflections=False))


def centered_normalized_kernel(train_gram, cross_gram):
    g, cross = np.asarray(train_gram, dtype=np.float64), np.asarray(cross_gram, dtype=np.float64)
    columns, grand = g.mean(0), g.mean()
    centered = g-g.mean(1, keepdims=True)-columns+grand
    query = cross-cross.mean(1, keepdims=True)-columns+grand
    scale = max(float(np.diag(centered).mean()), 1e-12)
    return centered/scale, query/scale


def kernel_predict(train_gram, cross_gram, train_targets, kernel, mean_loss_alpha):
    """CPU FP64 fit; only training moments determine either kernel geometry.

    Gram inputs are uncentered dot products of concatenated cues (or a registered
    single-frame control). Quadratic means (1 + normalized linear kernel)^2,
    followed by Train-only RKHS centering and trace normalization.
    """
    if kernel not in KERNELS or mean_loss_alpha <= 0:
        raise ValueError("Unregistered kernel or nonpositive regularization")
    g, cross = centered_normalized_kernel(train_gram, cross_gram)
    if kernel == "quadratic":
        g, cross = centered_normalized_kernel((1+g)**2, (1+cross)**2)
    y = np.asarray(train_targets, dtype=np.float64)
    mean = y.mean(0)
    coef = np.linalg.solve(g+len(y)*mean_loss_alpha*np.eye(len(y)), y-mean)
    return cross@coef+mean


def choose_reader(train_gram, train_targets, groups):
    """Twelve fixed candidates, four grouped FIT folds; no confirmation input."""
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < 4 or np.any(counts != 4):
        raise ValueError("Every fitting block must have all four views")
    rng = np.random.default_rng(8317)
    unique = unique[rng.permutation(len(unique))]
    folds = [np.flatnonzero(np.isin(groups, held)) for held in np.array_split(unique, 4)]
    gram, y = np.asarray(train_gram, dtype=np.float64), np.asarray(train_targets, dtype=np.float64)
    if gram.shape != (len(groups), len(groups)) or y.shape != (len(groups), 8):
        raise ValueError("Mismatched FIT Gram, target, or group dimensions")
    trials = []
    for kernel in KERNELS:
        for alpha in MEAN_LOSS_ALPHAS:
            scores = []
            for held in folds:
                fitting = np.setdiff1d(np.arange(len(groups)), held)
                prediction = kernel_predict(gram[np.ix_(fitting, fitting)], gram[np.ix_(held, fitting)], y[fitting], kernel, alpha)
                baseline = np.maximum(((y[held, 4:]-y[fitting, 4:].mean(0))**2).mean(0), 1e-12)
                scores.append(float((((prediction[:, 4:]-y[held, 4:])**2).mean(0)/baseline).mean()))
            trials.append(dict(kernel=kernel, mean_loss_alpha=alpha, fold_losses=scores, mean_loss=float(np.mean(scores))))
    selected = min(trials, key=lambda r:(r["mean_loss"], KERNELS.index(r["kernel"]), -r["mean_loss_alpha"]))
    return dict(selected=selected, trials=trials, cv_seed=8317,
                held_groups=[sorted(set(groups[held].tolist())) for held in folds])


def contrast_arrays(prediction, target):
    """Inputs [blocks, four views, four displacement coordinates]."""
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (4, 4):
        raise ValueError("Expected [blocks, 4 views, 4 motion coordinates]")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite prediction/label")
    if not (np.array_equal(y[:, 0], -y[:, 1]) and np.array_equal(y[:, 0], y[:, 2])
            and np.array_equal(y[:, 0], -y[:, 3])):
        raise ValueError("Quartet labels/order do not satisfy the registered reversals")
    end = (p[:, 0]-p[:, 1])/2
    start = (p[:, 2]-p[:, 3])/2
    delta = y[:, 0]
    return delta, end, start


def contrast_skills(prediction, target):
    delta, end, start = contrast_arrays(prediction, target)
    energy = np.mean(delta**2, axis=0)
    if np.any(energy <= 0):
        raise ValueError("Degenerate displacement target")
    return {name:1-np.mean((pred-delta)**2, axis=0)/energy
            for name,pred in (("end_anchor", end), ("start_anchor", start))}


def score_quartets(prediction, target, repeated_center_prediction, train_position_mean):
    """Metric checks only; never grants P0 or edits an existing gate.

    prediction/target: [B,4,8], repeat: [B,4] motion predictions on [C,C].
    Fit/confirm provenance and original-distribution bridge are separate gates.
    """
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (4, 8):
        raise ValueError("Expected [blocks, 4 views, 8 targets]")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite prediction/label")
    repeat = np.asarray(repeated_center_prediction, dtype=np.float64)
    if repeat.shape != (len(p), 4) or not np.isfinite(repeat).all():
        raise ValueError("Expected one finite repeated-center motion prediction per block")
    delta, end, start = contrast_arrays(p[:, :, 4:], y[:, :, 4:])
    energy = delta**2
    if np.any(energy.mean(0) <= 0):
        raise ValueError("Degenerate target")
    end_error, start_error = (end-delta)**2, (start-delta)**2
    absolute_error = ((p[:, :, 4:]-y[:, :, 4:])**2).mean(1)
    skills = contrast_skills(p[:, :, 4:], y[:, :, 4:])
    absolute = float((1-absolute_error.mean(0)/energy.mean(0)).mean())
    position_mean = np.asarray(train_position_mean, dtype=np.float64)
    if position_mean.shape != (4,) or not np.isfinite(position_mean).all():
        raise ValueError("Use the frozen FIT position mean, not confirmation labels")
    baseline = ((y[:, :, :4]-position_mean)**2).mean((0, 1))
    position = float((1-((p[:, :, :4]-y[:, :, :4])**2).mean((0, 1))/np.maximum(baseline, 1e-12)).mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bounds = []
    # Bound memory independently of bootstrap size.
    for begin in range(0, BOOTSTRAP_REPLICATES, 250):
        count = min(250, BOOTSTRAP_REPLICATES-begin)
        index = rng.integers(0, len(p), (count, len(p)))
        den = energy[index].mean(1)
        ce = 1-end_error[index].mean(1)/den
        cs = 1-start_error[index].mean(1)/den
        ab = (1-absolute_error[index].mean(1)/den).mean(1, keepdims=True)
        bounds.append(np.concatenate((ce, cs, ab), axis=1))
    lower = np.quantile(np.concatenate(bounds), FAMILYWISE_ALPHA/PRIMARY_ENDPOINTS, axis=0)
    reversal = float(max(np.abs(p[:, 0, 4:]+p[:, 3, 4:]).max(), np.abs(p[:, 1, 4:]+p[:, 2, 4:]).max()))
    repeat_error = float(np.abs(repeat).max())
    checks = dict(nine_lower_bounds_positive=bool(np.all(lower > 0)), position_skill_over_point_one=position > .1,
                  reversal_max_abs_at_most_1e_minus_6=reversal <= 1e-6,
                  repeated_frame_max_abs_at_most_1e_minus_6=repeat_error <= 1e-6)
    return dict(schema=SCHEMA, status="metric_checks_only", p0_authorized=False,
                blocks=len(p), expected_confirmation_blocks=256,
                contrast_skills={k:v.tolist() for k,v in skills.items()}, absolute_displacement_skill=absolute,
                position_skill=position, primary_lower_bounds=lower.tolist(),
                bound_order=[f"{anchor}_{dim}" for anchor in ("end", "start") for dim in ("A_dx", "A_dy", "B_dx", "B_dy")]+["absolute_mean"],
                bootstrap=dict(replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, unit="complete_four_view_block",
                               one_sided_quantile=FAMILYWISE_ALPHA/PRIMARY_ENDPOINTS,
                               interpretation="Bonferroni-adjusted percentile bootstrap; finite-sample coverage approximate, fixed fitted reader"),
                reversal_max_abs=reversal, repeated_center_max_abs=repeat_error, checks=checks,
                metric_conditions_met=all(checks.values()),
                remaining_requirements=["256 unique unseen confirmation blocks", "independently verified input/model receipts",
                                        "fresh original-distribution bridge", "new versioned P0 protocol; original V5 gate stays failed"])
