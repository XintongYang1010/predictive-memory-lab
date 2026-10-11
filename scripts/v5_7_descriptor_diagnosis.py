#!/usr/bin/env python3
"""Registered CPU-only descriptor-origin/metric/resolution diagnosis on known Dev.

Production scoring requires GitHub read-back of this source, its dependencies,
the protocol, and an exact design. Self-test uses synthetic arrays only.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '4'

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import v5_6_correspondence as old

SCHEMA = 'v5.7-descriptor-diagnosis-1'
REPRESENTATIONS = ('raw16', 'pool4')
ORIGINS = ('none', 'shared_source', 'framewise')
METRICS = ('cosine', 'l2')
GROUPS = tuple((r, o) for r in REPRESENTATIONS for o in ORIGINS)
ARMS = tuple((r, o, m) for r, o in GROUPS for m in METRICS)
ARM_NAMES = tuple('_'.join(arm) for arm in ARMS)
FIXED_CASES = ((761, 1), (844, 1), (891, 1))
PROTOCOL = 'docs/V5_7_EXECUTION_PROTOCOL.md'
SOURCE_FILES = tuple(sorted(set(old.SOURCE_FILES) | {
    'scripts/v5_7_descriptor_diagnosis.py', PROTOCOL,
    'scripts/extract_v5_confirmation.py', 'scripts/v5_temporal_confirmation.py',
    'scripts/v5_3_data.py', 'scripts/v5_4_data.py',
} | {p.relative_to(ROOT).as_posix() for p in (ROOT / 'pmr/visual_memory').glob('*.py')}))
DESIGN = dict(
    schema=SCHEMA, posthoc=True, no_training=True, test_opened=False,
    population=dict(split='dev', pair_ids=[640, 895], worlds=256, variants=2,
                    episodes=512, objects_per_episode=2),
    immutable_scene_inputs=old.DESIGN['immutable_inputs'],
    baseline=dict(
        report_sha256='47f10e6589fc26650f319712b6f588d3bcde94e4636e4d32b480738e42fe8d33',
        costs_sha256='265766a0d1de852f5eb863c068d08de11c696ed961e76768468797a6e9879eb0',
        schema=old.DESIGN['schema'], methods=['raw_native', 'pool_native'],
        cost_atol=1e-10, cost_rtol=0.0, predictions='exact array equality'),
    arms=[dict(name=name, representation=r, origin=o, metric=m)
          for name, (r, o, m) in zip(ARM_NAMES, ARMS)],
    roi=old.DESIGN['roi'], grid=old.DESIGN['grid'],
    interpolation=old.DESIGN['interpolation'], tie=old.DESIGN['tie'],
    native_pooling='Same raw FP32 input; 4x4 spatial means accumulated FP64 and stored FP32, before origin subtraction; no final BF16 conversion.',
    global_mean='Per-frame, per-channel mean of all 256 raw tokens, accumulated FP64; identical vectors used for raw16 and pool4.',
    origin=dict(none='No spatial-origin subtraction.',
                shared_source='Subtract source global vector mu0 from both frame banks.',
                framewise='Subtract each frame global vector mut from its own bank.'),
    descriptor='Subtract origin in FP64, then center each descriptor over channels. Centering commutes with row-stochastic interpolation.',
    cosine='Mean of nine centered cosine distances. Squared norm <=1e-24 produces similarity zero and an explicit undefined diagnostic.',
    l2='Mean squared channel difference over 1280 channels, then mean over nine ROI points. NO unit norm normalization. L2 is not a cosine rescaling.',
    numerical=dict(norm_negative_tolerance=1e-8, l2_negative_tolerance=1e-10,
                   l2_shared_origin_atol=1e-10, l2_shared_origin_rtol=1e-10,
                   l2_shared_origin_predictions='exact equality under original tie rule',
                   negative_roundoff='Record signed minimum before clipping tiny negative squared norms/distances to zero.'),
    storage=dict(costs='All 512 x 12 x 2 x 625 FP64 costs.',
                 norms='Per-world compressed FP64 source9 and target625x9 squared norms, shared across metrics; all raw global means also saved.',
                 clamps='Every source and candidate/point clamp flag at each native resolution.',
                 equal_memory_comparison=False,
                 extra_access='Full raw256 tokens from both frames; framewise origin uses target-wide context. No equal-byte-bank claim.'),
    metrics=dict(baseline='Original archived frozen FIT target mean; no Dev-mean normalization.',
                 reported=['per-axis MSE/skill', 'dx MSE', 'motion MSE/skill',
                           'signed bias', 'absolute amplitude ratio',
                           'wrong nonzero sign and predicted zero separately for abs(true displacement)>=1 pixel',
                           'ties, margins, boundary optima, finite/norm/clamp checks']),
    bootstrap=dict(seed=7818, replicates=10000, unit='whole world; both variants and objects remain together',
                   metrics=['dx_mse', 'motion_mse', 'motion_nmse'],
                   contrasts=['each subtracted origin minus none at fixed resolution/metric',
                              'pool4 minus raw16 at fixed origin/metric',
                              'l2 minus cosine at fixed resolution/origin'],
                   interval='percentile 2.5/97.5; paired, conditional exploratory, not multiplicity adjusted'),
    fixed_cases=[list(x) for x in FIXED_CASES],
    figures='All 12 arms and both objects at every fixed case; no result-dependent selection.',
    limits='Global means contain object, position and contextual signals; improvements cannot establish encoder background removal or causal module failure. No learned selector, unseen confirmation, memory training, or P0 authorization.',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def source_identity():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def record(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha(path))


def checked_scene_path(scene, relative, manifest, opened):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or rel.as_posix() != relative:
        raise ValueError('Unsafe scene path: ' + relative)
    path = scene / rel
    if any(p.is_symlink() for p in [path, *path.parents] if p != scene.parent):
        raise ValueError('Symlink in immutable scene path: ' + relative)
    current = record(path)
    if manifest['files'].get(relative) != current:
        raise ValueError('Sealed input changed: ' + relative)
    opened[relative] = current
    return path


def validate_publication(publication, plan):
    p = json.loads(Path(publication).read_text())
    design = json.loads(Path(plan).read_text())
    if p.get('verified_readback') is not True or design.get('design') != DESIGN:
        raise ValueError('Published exact design is required before scoring')
    if design.get('sources') != source_identity():
        raise ValueError('Source or protocol changed after design freeze')
    if not np.isfinite(p.get('verified_unix', 0)) or p.get('verified_unix', 0) < design['created_unix']:
        raise ValueError('Read-back must follow design freeze')
    for path in [*(ROOT / name for name in SOURCE_FILES), Path(plan).resolve()]:
        rel = path.relative_to(ROOT).as_posix()
        item = p.get('files', {}).get(rel)
        digest = item.get('sha256') if isinstance(item, dict) else item
        if digest != sha(path):
            raise ValueError('Read-back omits or mismatches source/design: ' + rel)
    return p, design


def origin_bank(raw, pooled, means, representation, origin):
    bank = np.array(raw if representation == 'raw16' else pooled, dtype=np.float64, copy=True)
    if representation not in REPRESENTATIONS or origin not in ORIGINS:
        raise ValueError('Unregistered descriptor intervention')
    if origin == 'shared_source':
        bank -= means[0][None, None, :]
    elif origin == 'framewise':
        bank -= means[:, None, :]
    return bank


def candidate_costs(bank, source_weights, target_weights):
    """Both metrics use identical centered descriptors and small Gram matrices."""
    s, t = np.asarray(source_weights), np.asarray(target_weights)
    active = np.flatnonzero((np.abs(s).sum(0) + np.abs(t).sum((0, 1))) > 0)
    s, t = s[:, active], t[:, :, active]
    x = bank[0, active].astype(np.float64)
    y = bank[1, active].astype(np.float64)
    x -= x.mean(1, keepdims=True)
    y -= y.mean(1, keepdims=True)
    source = s @ x
    source_norm = np.sum(source * source, axis=1)
    numerator = np.einsum('pn,kpn->kp', source @ y.T, t)
    target_norm = np.einsum('kpn,nm,kpm->kp', t, y @ y.T, t)
    if target_norm.min() < -DESIGN['numerical']['norm_negative_tolerance']:
        raise ValueError('Negative squared descriptor norm beyond frozen tolerance')
    nonnegative = np.maximum(target_norm, 0)
    denominator = np.sqrt(source_norm[None, :] * nonnegative)
    valid = (source_norm[None, :] > 1e-24) & (target_norm > 1e-24)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid)
    cosine_cost = np.mean(1 - np.clip(cosine, -1, 1), axis=1)
    # No unit normalization here: shared translations must cancel in L2.
    point_l2 = (source_norm[None, :] + target_norm - 2 * numerator) / bank.shape[-1]
    if point_l2.min() < -DESIGN['numerical']['l2_negative_tolerance']:
        raise ValueError('Negative squared distance beyond frozen tolerance')
    l2_cost = np.mean(np.maximum(point_l2, 0), axis=1)
    arrays = (cosine_cost, l2_cost, source_norm, target_norm)
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite descriptor cost/norm')
    return (cosine_cost, l2_cost), source_norm, target_norm, dict(
        undefined_cosine_comparisons=int((~valid).sum()), total_comparisons=int(valid.size),
        source_zero_norm_count=int((source_norm <= 1e-24).sum()),
        target_zero_norm_count=int((target_norm <= 1e-24).sum()),
        minimum_unclipped_target_norm2=float(target_norm.min()),
        minimum_unclipped_point_l2=float(point_l2.min()),
        clipped_negative_target_norm_count=int((target_norm < 0).sum()),
        clipped_negative_point_l2_count=int((point_l2 < 0).sum()))


def choose(cost):
    tied = np.flatnonzero(cost <= cost.min() + DESIGN['tie']['absolute_cost_tolerance'])
    order = np.lexsort((old.OFFSETS[tied, 1], old.OFFSETS[tied, 0],
                        np.sum(old.OFFSETS[tied] ** 2, axis=1)))
    best = int(tied[order[0]])
    sorted_cost = np.sort(cost)
    return best, dict(best_index=best, best_cost=float(cost[best]),
                      second_minus_best=float(sorted_cost[1] - sorted_cost[0]),
                      tie_count=int(len(tied)), near_best_count=int(np.sum(cost <= cost.min() + 1e-6)),
                      boundary_optimum=bool(np.any(np.abs(old.OFFSETS[best]) == 6)))


def fixed_contrasts():
    pairs = []
    for r in REPRESENTATIONS:
        for m in METRICS:
            for o in ORIGINS[1:]:
                pairs.append((ARMS.index((r, o, m)), ARMS.index((r, 'none', m))))
    for o in ORIGINS:
        for m in METRICS:
            pairs.append((ARMS.index(('pool4', o, m)), ARMS.index(('raw16', o, m))))
    for r in REPRESENTATIONS:
        for o in ORIGINS:
            pairs.append((ARMS.index((r, o, 'l2')), ARMS.index((r, o, 'cosine'))))
    return pairs


def bootstrap(losses, replicates=10000, seed=7818):
    """Resample whole worlds; loss axes are world, arm, metric."""
    losses = np.asarray(losses, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, *losses.shape[1:]), dtype=np.float64)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.integers(0, len(losses), size=(count, len(losses)))
        samples[start:start + count] = losses[indices].mean(axis=1)
    return samples


def summarize(predictions, targets, fit_mean):
    den = np.mean((targets - fit_mean[4:]) ** 2, axis=0)
    if np.any(den <= 0) or not np.isfinite(den).all():
        raise ValueError('Nonpositive frozen-FIT skill denominator')
    err2 = (predictions - targets[:, None, :]) ** 2
    losses = np.stack((err2[:, :, [0, 2]].mean(axis=2), err2.mean(axis=2),
                       (err2 / den[None, None, :]).mean(axis=2)), axis=2)
    world_losses = losses.reshape(256, 2, len(ARMS), 3).mean(axis=1)
    samples = bootstrap(world_losses, DESIGN['bootstrap']['replicates'], DESIGN['bootstrap']['seed'])
    scores = {}
    eligible = np.abs(targets * 224) >= 1
    for arm, name in enumerate(ARM_NAMES):
        p = predictions[:, arm]
        mse = err2[:, arm].mean(axis=0)
        wrong = (p * targets < 0) & eligible
        zero = (p == 0) & eligible
        scores[name] = dict(
            per_axis_mse=mse.tolist(), per_axis_skill=(1 - mse / den).tolist(),
            motion_skill=float(np.mean(1 - mse / den)), dx_mse=float(mse[[0, 2]].mean()),
            motion_mse=float(mse.mean()), signed_bias=(p - targets).mean(axis=0).tolist(),
            mean_abs_amplitude_ratio=(np.abs(p).mean(axis=0) / np.abs(targets).mean(axis=0)).tolist(),
            sign_eligible_count=eligible.sum(axis=0).tolist(), wrong_nonzero_sign_count=wrong.sum(axis=0).tolist(),
            predicted_zero_eligible_count=zero.sum(axis=0).tolist(), predicted_zero_all_count=(p == 0).sum(axis=0).tolist(),
            conditional_bootstrap={metric: dict(point=float(world_losses[:, arm, j].mean()),
                ci95=np.percentile(samples[:, arm, j], [2.5, 97.5]).tolist())
                for j, metric in enumerate(DESIGN['bootstrap']['metrics'])})
    contrasts = []
    for a, b in fixed_contrasts():
        contrasts.append(dict(first=ARM_NAMES[a], second=ARM_NAMES[b], direction='first loss minus second loss; negative favors first',
            metrics={metric: dict(point=float((world_losses[:, a, j] - world_losses[:, b, j]).mean()),
                ci95=np.percentile(samples[:, a, j] - samples[:, b, j], [2.5, 97.5]).tolist())
                for j, metric in enumerate(DESIGN['bootstrap']['metrics'])}))
    return scores, contrasts, samples, world_losses


def make_figures(out, predictions, truth, costs, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = out / 'figures'
    figures.mkdir()
    for pair, variant in FIXED_CASES:
        idx = 2 * (pair - 640) + variant
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
        for obj in range(2):
            for axis in range(2):
                ax = axes[obj, axis]
                col = obj * 2 + axis
                ax.scatter(predictions[idx, :, col] * 224, np.arange(12), s=35, color='#2366a8')
                ax.axvline(truth[idx, col] * 224, color='#be381f', label='Ground truth', linewidth=1.6)
                ax.axvline(0, color='gray', linewidth=.5)
                ax.set_yticks(np.arange(12), ARM_NAMES)
                ax.set_xlim(-6.5, 6.5)
                ax.set_title(f"Object {'AB'[obj]} / {'xy'[axis]} displacement")
                ax.set_xlabel('Pixels')
                ax.grid(axis='x', alpha=.2)
        axes[0, 0].invert_yaxis()
        axes[0, 0].legend(loc='best')
        fig.suptitle(f'Known Dev {pair}/{variant}: all 12 fixed descriptor arms (source RGB ROI)')
        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(figures / f'case_{pair}_{variant}_predictions.{ext}', dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(6, 4, figsize=(15, 21), sharex=True, sharey=True)
        for group, (r, o) in enumerate(GROUPS):
            for obj in range(2):
                for metric in range(2):
                    arm = group * 2 + metric
                    ax = axes[group, obj * 2 + metric]
                    ax.imshow(costs[idx, arm, obj].reshape(25, 25), origin='lower',
                              extent=(-6.25, 6.25, -6.25, 6.25), aspect='equal')
                    ax.scatter(*predictions[idx, arm, obj * 2:obj * 2 + 2] * 224,
                               marker='x', color='white', s=40, linewidth=1.5)
                    ax.scatter(*truth[idx, obj * 2:obj * 2 + 2] * 224,
                               marker='*', color='#ff6152', s=55, edgecolors='black', linewidth=.3)
                    ax.set_title(f'{r}/{o}\n{METRICS[metric]}, object {"AB"[obj]}', fontsize=9)
                    ax.set_xlabel('dx (pixels)')
                    ax.set_ylabel('dy (pixels)')
        fig.suptitle(f'Known Dev {pair}/{variant}: x = prediction; red star = truth; each panel has its own cost scale', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .98))
        for ext in ('png', 'svg'):
            fig.savefig(figures / f'case_{pair}_{variant}_costs.{ext}', dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6))
    values = [scores[name]['dx_mse'] * 224 ** 2 for name in ARM_NAMES]
    ax.barh(np.arange(12), values, color=['#3178ad'] * 6 + ['#b8792a'] * 6)
    ax.set_yticks(np.arange(12), ARM_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel('dx MSE (pixel squared), all 256 worlds / 512 variants')
    ax.set_title('All registered arms; conditional posthoc comparison')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(figures / f'all_arms_dx_mse.{ext}', dpi=150)
    plt.close(fig)


def self_test():
    """No archived observations, real generator roles, models, or labels read."""
    rng = np.random.default_rng(57071)
    raw = rng.normal(size=(2, 256, 11)).astype(np.float32)
    pooled = old.pool(raw, 4)
    means = raw.mean(axis=1, dtype=np.float64)
    regions = [np.array([[x, y] for y in (81., 94., 103.) for x in (78., 91., 108.)]),
               np.array([[x, y] for y in (-2., 1., 222.) for x in (-3., 0., 226.)])]
    max_shared = 0.0
    for r in REPRESENTATIONS:
        side = 16 if r == 'raw16' else 4
        for roi in regions:
            sw, sc = old.bilinear_weights(roi, side)
            tw, tc = old.bilinear_weights(roi[None] + old.OFFSETS[:, None], side)
            np.testing.assert_allclose(sw.sum(axis=-1), 1, atol=1e-15)
            np.testing.assert_allclose(tw.sum(axis=-1), 1, atol=1e-15)
            if roi is regions[1]:
                assert sc.any() and tc.any()
            outputs = {}
            for origin in ORIGINS:
                bank = origin_bank(raw, pooled, means, r, origin)
                outputs[origin], sn, tn, _ = candidate_costs(bank, sw, tw)
                source = sw @ bank[0]
                target = tw @ bank[1]
                source -= source.mean(axis=-1, keepdims=True)
                target -= target.mean(axis=-1, keepdims=True)
                explicit_l2 = np.mean((source[None] - target) ** 2, axis=(1, 2))
                explicit_cosine = 1 - np.einsum('pc,kpc->kp', source, target) / (
                    np.linalg.norm(source, axis=-1)[None] * np.linalg.norm(target, axis=-1))
                np.testing.assert_allclose(outputs[origin][0], explicit_cosine.mean(axis=1), atol=2e-14, rtol=0)
                np.testing.assert_allclose(outputs[origin][1], explicit_l2, atol=2e-14, rtol=0)
                np.testing.assert_allclose(sn, (source ** 2).sum(axis=-1), atol=1e-13, rtol=0)
                np.testing.assert_allclose(tn, (target ** 2).sum(axis=-1), atol=1e-13, rtol=0)
            old_cost, _ = old.costs(raw if r == 'raw16' else pooled, sw, tw)
            np.testing.assert_array_equal(outputs['none'][0], old_cost)
            np.testing.assert_allclose(outputs['none'][1], outputs['shared_source'][1], atol=1e-10, rtol=1e-10)
            max_shared = max(max_shared, float(np.max(np.abs(outputs['none'][1] - outputs['shared_source'][1]))))
            assert choose(outputs['none'][1])[0] == choose(outputs['shared_source'][1])[0]
    # Constant centered fields retain their undefined cosine and all-candidate tie.
    constant = np.ones_like(raw)
    sw, _ = old.bilinear_weights(regions[0], 16)
    tw, _ = old.bilinear_weights(regions[0][None] + old.OFFSETS[:, None], 16)
    (cos, l2), _, _, diag = candidate_costs(constant, sw, tw)
    np.testing.assert_array_equal(cos, np.ones(625))
    np.testing.assert_array_equal(l2, np.zeros(625))
    for cost in (cos, l2):
        best, info = choose(cost)
        np.testing.assert_array_equal(old.OFFSETS[best], [0, 0])
        assert info['tie_count'] == 625
    assert diag['undefined_cosine_comparisons'] == 625 * 9
    # Framewise centering removes a deliberately spatially constant temporal drift.
    source = rng.normal(size=(256, 11))
    drift = np.arange(11, dtype=np.float64)
    toy = np.stack((source, source + drift))
    corrected = origin_bank(toy, old.pool(toy, 4), toy.mean(axis=1), 'raw16', 'framewise')
    np.testing.assert_allclose(corrected[0], corrected[1], atol=1e-14, rtol=0)
    # Point aggregation is MSE, not unit-normalized Euclidean/cosine.
    base = np.arange(11, dtype=np.float64)
    scaled = np.broadcast_to(np.stack((base, 2 * base))[:, None, :], (2, 256, 11)).copy()
    (cos, l2), _, _, _ = candidate_costs(scaled, sw, tw)
    np.testing.assert_allclose(cos, 0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(l2, np.mean((base - base.mean()) ** 2), atol=1e-13, rtol=0)
    losses = np.arange(8 * 12 * 3, dtype=float).reshape(8, 12, 3)
    boot = bootstrap(losses, replicates=17, seed=7818)
    rr = np.random.default_rng(7818)
    expected = losses[rr.integers(0, 8, size=(17, 8))].mean(axis=1)
    np.testing.assert_array_equal(boot, expected)
    assert len(fixed_contrasts()) == 20 and len(set(fixed_contrasts())) == 20
    return dict(status='passed', software_only=True, scientific_inputs_opened=False,
                explicit_descriptor_math=True, exact_legacy_cosine=True,
                border_replication=True, constant_ambiguity=True, non_unit_L2=True,
                whole_world_bootstrap=True, maximum_synthetic_shared_L2_difference=max_shared)


def run(scene, costs_path, baseline_report, out, publication, plan):
    started = time.time()
    scene, out = Path(scene).resolve(), Path(out).resolve()
    costs_path, baseline_report = Path(costs_path).resolve(), Path(baseline_report).resolve()
    validate_publication(publication, plan)
    initial = dict(sources=source_identity(), plan=sha(plan), publication=sha(publication))
    if out == scene or scene in out.parents or out == costs_path.parent or costs_path.parent in out.parents:
        raise ValueError('Fresh output must be outside both immutable input directories')
    if out.exists():
        raise FileExistsError('New output directory required; never replace prior results')
    for name, digest in DESIGN['immutable_scene_inputs'].items():
        if sha(scene / name) != digest:
            raise ValueError('Pinned V5.5 input changed: ' + name)
    if sha(costs_path) != DESIGN['baseline']['costs_sha256'] or sha(baseline_report) != DESIGN['baseline']['report_sha256']:
        raise ValueError('Pinned V5.6 baseline changed')
    baseline = json.loads(baseline_report.read_text())
    if baseline['status'] != 'complete' or baseline['schema'] != DESIGN['baseline']['schema']:
        raise ValueError('A completed matching V5.6 baseline is required')
    for name, digest in baseline['provenance']['source_identity'].items():
        if initial['sources'].get(name) != digest:
            raise ValueError('Baseline generating source changed: ' + name)
    manifest = json.loads((scene / 'archive_manifest.json').read_text())
    opened = {}
    def checked(name):
        return checked_scene_path(scene, name, manifest, opened)
    receipt = json.loads(checked('counterfactual_cache/receipt.json').read_text())
    if receipt['groups'] != 256 or receipt['test_opened'] is not False:
        raise ValueError('Original population changed')
    with np.load(checked('report/predictions.npz'), allow_pickle=False) as z:
        truth8 = z['targets'][..., 7, :].reshape(512, 8).astype(np.float64)
    with np.load(checked('lane_reader/frozen_model.npz'), allow_pickle=False) as z:
        fit_mean = z['shared_target_mean'].copy()
    with np.load(costs_path, allow_pickle=False) as z:
        if tuple(z['methods'].tolist()) != old.METHODS:
            raise ValueError('Baseline methods or order changed')
        for key, expected in (('targets', truth8[:, 4:]), ('offsets_pixels', old.OFFSETS),
                              ('pair_ids', np.repeat(np.arange(640, 896), 2)),
                              ('variants', np.tile([0, 1], 256)), ('frozen_fit_mean', fit_mean)):
            np.testing.assert_array_equal(z[key], expected)
        legacy_costs, legacy_predictions = z['costs'][:, :2].copy(), z['predictions'][:, :2].copy()
    if legacy_costs.shape != (512, 2, 2, 625) or legacy_predictions.shape != (512, 2, 4):
        raise ValueError('Baseline cost/prediction population mismatch')
    out.mkdir(parents=True)
    norm_dir = out / 'norm_shards'
    norm_dir.mkdir()
    all_costs = np.empty((512, 12, 2, 625), dtype=np.float64)
    predictions = np.empty((512, 12, 4), dtype=np.float64)
    source_clamps = np.empty((512, 2, 2, 9), dtype=bool)
    target_clamps = np.empty((512, 2, 2, 625, 9), dtype=bool)
    rois = np.empty((512, 2, 9, 2), dtype=np.float64)
    rows, norm_records, rgb_records = [], {}, []
    max_cos_diff, max_l2_diff = 0.0, 0.0
    for pair in range(640, 896):
        raw_record = receipt['raw_records'][str(pair)]
        rawpath = checked('counterfactual_cache/' + raw_record['path'])
        if sha(rawpath) != raw_record['sha256']:
            raise ValueError('Raw receipt hash differs')
        with np.load(rawpath, allow_pickle=False) as z:
            if set(z.files) != {'raw'}:
                raise ValueError('Unexpected raw arrays')
            raw = z['raw']
        if raw.shape != (2, 2, 256, 1280) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError('Invalid original raw descriptors')
        if {'raw': old.entry_identity(raw)} != raw_record['entry_hashes']:
            raise ValueError('Raw entry identities differ')
        shard = json.loads(checked(f'counterfactual_cache/shards/{pair:06d}.json').read_text())
        pair_source_norm = np.empty((2, 6, 2, 9), dtype=np.float64)
        pair_target_norm = np.empty((2, 6, 2, 625, 9), dtype=np.float64)
        pair_means = raw.mean(axis=2, dtype=np.float64)
        for variant in (0, 1):
            idx = 2 * (pair - 640) + variant
            pos, colors, yellow = old.cue_geometry('dev', pair, variant)
            rgb, _ = old.render_endpoint(pos[0], colors, yellow[0])
            rgb_identity = old.entry_identity(rgb)
            if rgb_identity != shard['inputs']['variants'][variant]['unique_rgb'][0]:
                raise ValueError('Source RGB does not reproduce authenticated original')
            labels = np.concatenate((pos[1].ravel(), (pos[1] - pos[0]).ravel())).astype(np.float32)
            np.testing.assert_array_equal(labels.astype(np.float64), truth8[idx])
            regions, _, _ = old.source_roi(rgb)
            rois[idx] = regions
            rgb_records.append(dict(pair_id=pair, variant=variant, source_rgb=rgb_identity,
                                    roi=old.entry_identity(regions)))
            pooled = old.pool(raw[variant], 4)
            for resolution, r in enumerate(REPRESENTATIONS):
                side = 16 if r == 'raw16' else 4
                for obj in range(2):
                    sw, sc = old.bilinear_weights(regions[obj], side)
                    tw, tc = old.bilinear_weights(regions[obj][None] + old.OFFSETS[:, None], side)
                    source_clamps[idx, resolution, obj] = sc
                    target_clamps[idx, resolution, obj] = tc
                    for origin, o in enumerate(ORIGINS):
                        group = resolution * 3 + origin
                        bank = origin_bank(raw[variant], pooled, pair_means[variant], r, o)
                        pair_costs, sn, tn, common = candidate_costs(bank, sw, tw)
                        pair_source_norm[variant, group, obj] = sn
                        pair_target_norm[variant, group, obj] = tn
                        for metric, m in enumerate(METRICS):
                            arm = group * 2 + metric
                            cost = pair_costs[metric]
                            best, info = choose(cost)
                            prediction = old.OFFSETS[best]
                            all_costs[idx, arm, obj] = cost
                            predictions[idx, arm, obj * 2:obj * 2 + 2] = prediction / 224
                            rows.append(dict(pair_id=pair, variant=variant, object='AB'[obj], arm=ARM_NAMES[arm],
                                representation=r, origin=o, metric=m,
                                prediction_dx_pixels=float(prediction[0]), prediction_dy_pixels=float(prediction[1]),
                                target_dx_pixels=float(truth8[idx, 4 + obj * 2] * 224),
                                target_dy_pixels=float(truth8[idx, 5 + obj * 2] * 224),
                                source_clamped_points=int(sc.sum()), best_target_clamped_points=int(tc[best].sum()),
                                all_target_clamped_points=int(tc.sum()), source_norm2_min=float(sn.min()),
                                source_norm2_mean=float(sn.mean()), source_norm2_max=float(sn.max()),
                                best_target_norm2_min=float(tn[best].min()), best_target_norm2_mean=float(tn[best].mean()),
                                best_target_norm2_max=float(tn[best].max()), **common, **info))
                    none_cos, none_l2, shared_l2 = resolution * 6, resolution * 6 + 1, resolution * 6 + 3
                    delta = float(np.max(np.abs(all_costs[idx, none_cos, obj] - legacy_costs[idx, resolution, obj])))
                    max_cos_diff = max(max_cos_diff, delta)
                    np.testing.assert_allclose(all_costs[idx, none_cos, obj], legacy_costs[idx, resolution, obj],
                        atol=DESIGN['baseline']['cost_atol'], rtol=DESIGN['baseline']['cost_rtol'])
                    np.testing.assert_array_equal(predictions[idx, none_cos, obj * 2:obj * 2 + 2],
                                                  legacy_predictions[idx, resolution, obj * 2:obj * 2 + 2])
                    delta = float(np.max(np.abs(all_costs[idx, none_l2, obj] - all_costs[idx, shared_l2, obj])))
                    max_l2_diff = max(max_l2_diff, delta)
                    np.testing.assert_allclose(all_costs[idx, none_l2, obj], all_costs[idx, shared_l2, obj],
                        atol=DESIGN['numerical']['l2_shared_origin_atol'], rtol=DESIGN['numerical']['l2_shared_origin_rtol'])
                    np.testing.assert_array_equal(predictions[idx, none_l2, obj * 2:obj * 2 + 2],
                                                  predictions[idx, shared_l2, obj * 2:obj * 2 + 2])
        norm_path = norm_dir / f'{pair:06d}.npz'
        np.savez_compressed(norm_path, source_norm2=pair_source_norm, target_norm2=pair_target_norm,
                            raw_global_means=pair_means, groups=np.array(['_'.join(x) for x in GROUPS]),
                            variants=np.array([0, 1]), pair_id=np.array(pair))
        norm_records[norm_path.relative_to(out).as_posix()] = record(norm_path)
        if (pair - 639) % 32 == 0:
            print(json.dumps(dict(progress_worlds=pair - 639, total_worlds=256)), flush=True)
    y = truth8[:, 4:]
    scores, contrasts, bootstrap_samples, world_losses = summarize(predictions, y, fit_mean)
    np.savez_compressed(out / 'costs_and_predictions.npz', costs=all_costs, predictions=predictions, targets=y,
        methods=np.array(ARM_NAMES), pair_ids=np.repeat(np.arange(640, 896), 2), variants=np.tile([0, 1], 256),
        offsets_pixels=old.OFFSETS, frozen_fit_mean=fit_mean, roi_points_pixels=rois,
        source_clamped=source_clamps, target_clamped=target_clamps, clamp_representations=np.array(REPRESENTATIONS))
    np.savez_compressed(out / 'bootstrap.npz', mean_losses=bootstrap_samples, world_losses=world_losses,
        methods=np.array(ARM_NAMES), metric_names=np.array(DESIGN['bootstrap']['metrics']),
        seed=np.array(DESIGN['bootstrap']['seed']), world_ids=np.arange(640, 896))
    with (out / 'per_episode_object.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(out / 'norm_inventory.json', dict(schema=SCHEMA, files=norm_records,
        squared_norm_precision='float64', source_shape=[2, 6, 2, 9], target_shape=[2, 6, 2, 625, 9],
        axes='variant,representation-origin,object[,candidate],ROI point',
        mean_shape=[2, 2, 1280], mean_axes='variant,frame,channel',
        groups=['_'.join(x) for x in GROUPS], point_order='source ROI Cartesian raster; offsets dy-major dx-minor'))
    save(out / 'source_rgb_records.json', rgb_records)
    save(out / 'fixed_cases.json', [dict(pair_id=p, variant=v,
        targets_pixels=(y[2 * (p - 640) + v] * 224).tolist(),
        predictions_pixels={name: (predictions[2 * (p - 640) + v, i] * 224).tolist() for i, name in enumerate(ARM_NAMES)})
        for p, v in FIXED_CASES])
    make_figures(out, predictions, y, all_costs, scores)
    for name, expected in opened.items():
        if record(scene / name) != expected:
            raise ValueError('Archived input changed while scoring: ' + name)
    if (source_identity() != initial['sources'] or sha(plan) != initial['plan'] or
            sha(publication) != initial['publication'] or
            sha(costs_path) != DESIGN['baseline']['costs_sha256'] or
            sha(baseline_report) != DESIGN['baseline']['report_sha256'] or
            sha(scene / 'archive_manifest.json') != DESIGN['immutable_scene_inputs']['archive_manifest.json']):
        raise ValueError('Source, design, publication, or pinned baseline changed during scoring')
    save(out / 'report.json', dict(schema=SCHEMA, status='complete', design=DESIGN,
        scores=scores, paired_bootstrap_contrasts=contrasts,
        checks=dict(baseline_cost_max_abs=max_cos_diff, baseline_predictions_exact=True,
                    shared_source_L2_cost_max_abs=max_l2_diff, shared_source_L2_predictions_exact=True,
                    all_costs_predictions_norms_finite=True, source_rgb_authentication_count=len(rgb_records)),
        ambiguity={name: dict(tie_gt1=sum(row['tie_count'] > 1 for row in rows if row['arm'] == name),
            boundary_optima=sum(row['boundary_optimum'] for row in rows if row['arm'] == name)) for name in ARM_NAMES},
        provenance=dict(source_identity=initial['sources'], plan_sha256=initial['plan'], publication_sha256=initial['publication'],
            old_scene_manifest_sha256=sha(scene / 'archive_manifest.json'), scene_inputs=opened,
            baseline_costs=record(costs_path), baseline_report=record(baseline_report), command=sys.argv),
        runtime=dict(python=sys.version, numpy=np.__version__, platform=platform.platform(), blas_threads=4,
                     elapsed_seconds=time.time() - started), test_opened=False, no_training=True,
        no_equal_memory_claim=True, no_encoder_background_removal_claim=True))
    save(out / 'archive_manifest.json', dict(schema=SCHEMA, files={
        p.relative_to(out).as_posix(): record(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(status='complete', out=str(out), worlds=256, arms=len(ARMS),
                baseline_cost_max_abs=max_cos_diff, shared_source_L2_cost_max_abs=max_l2_diff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    sub.add_parser('self-test')
    p = sub.add_parser('freeze-design')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--costs', type=Path, required=True)
    p.add_argument('--baseline-report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    a = parser.parse_args()
    if a.phase == 'self-test':
        result = self_test()
    elif a.phase == 'freeze-design':
        result = dict(design=DESIGN, sources=source_identity(), software_verification=self_test(), created_unix=time.time())
        save(a.out, result)
    else:
        allowed_failure = (not a.out.exists() and a.out.resolve() != a.scene.resolve() and
                           a.scene.resolve() not in a.out.resolve().parents and
                           a.out.resolve() != a.costs.resolve().parent and
                           a.costs.resolve().parent not in a.out.resolve().parents)
        try:
            result = run(a.scene, a.costs, a.baseline_report, a.out, a.publication, a.plan)
        except Exception as exc:
            if allowed_failure:
                a.out.mkdir(parents=True, exist_ok=True)
                if not (a.out / 'failure.json').exists():
                    save(a.out / 'failure.json', dict(status='failed', error=str(exc), traceback=traceback.format_exc(), command=sys.argv))
            raise
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
