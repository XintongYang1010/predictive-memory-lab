#!/usr/bin/env python3
"""Fixed CPU-only smoothing/sampling correspondence diagnosis on known Dev.

No production observations are read before exact source/design publication.
Self-test uses synthetic arrays only. Original scenes and sources stay immutable.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import v5_6_correspondence as old

SCHEMA = 'v5.8-spatial-diagnosis-1'
METHODS = ('R', 'BR', 'DR', 'DBR', 'H')
GRID = {'R': (16, 6.5, 14.), 'BR': (13, 27.5, 14.),
        'DR': (4, 27.5, 56.), 'DBR': (4, 27.5, 56.), 'H': (13, 27.5, 14.)}
CASES = ((761, 1), (844, 1), (891, 1))
PROTOCOL = 'docs/V5_8_EXECUTION_PROTOCOL.md'
CONTRASTS = (
    ('B_before_coarse_sampling', (-1., 1., 0., 0., 0.), 'BR minus R'),
    ('B_with_coarse_sampling', (0., 0., -1., 1., 0.), 'DBR minus DR'),
    ('D_without_box_filter', (-1., 0., 1., 0., 0.), 'DR minus R'),
    ('D_with_box_filter', (0., -1., 0., 1., 0.), 'DBR minus BR'),
    ('difference_in_differences', (1., -1., -1., 1., 0.), 'DBR minus DR minus BR plus R'),
    ('half_grid_sampling', (-1., 0., 0., 0., 1.), 'H minus R'),
    ('matched_dense_phase_filter', (0., 1., 0., 0., -1.), 'BR minus H'),
    ('coarse_sampling_of_H', (0., 0., 1., 0., -1.), 'DR minus H'),
    ('phase_matched_interaction', (0., -1., -1., 1., 1.), 'DBR minus BR minus DR plus H'),
)
SOURCE_FILES = tuple(sorted(set(old.SOURCE_FILES) | {
    'scripts/v5_8_spatial_diagnosis.py', PROTOCOL,
    'tests/visual_memory/test_v5_8_spatial_diagnosis.py',
    'scripts/extract_v5_confirmation.py', 'scripts/v5_temporal_confirmation.py',
    'scripts/v5_3_data.py', 'scripts/v5_4_data.py',
} | {p.relative_to(ROOT).as_posix() for p in (ROOT / 'pmr/visual_memory').glob('*.py')}))
DESIGN = dict(
    schema=SCHEMA, posthoc=True, no_training=True, test_opened=False,
    population=dict(split='dev', pair_ids=[640, 895], worlds=256, episodes=512, object_queries=1024),
    immutable_scene_inputs=old.DESIGN['immutable_inputs'],
    baseline=dict(report_sha256='47f10e6589fc26650f319712b6f588d3bcde94e4636e4d32b480738e42fe8d33',
        costs_sha256='265766a0d1de852f5eb863c068d08de11c696ed961e76768468797a6e9879eb0',
        R='raw_native', DBR='pool_native', cost_atol=1e-10, cost_rtol=0.,
        bank='R unchanged and DBR bitwise equal to old.pool(raw,4)', predictions='exact equality'),
    arms=[dict(name='R', bank='Original raw16x16 descriptors, FP32; grid origin6.5 spacing14.'),
          dict(name='BR', bank='Valid4x4 box mean, stride1; FP64 accumulation then FP32; 13x13 grid origin27.5 spacing14.'),
          dict(name='DR', bank='Sample raw bilinear field at (27.5+56k,27.5+56j), FP64 interpolation then FP32; reconstruct4x4 bilinearly.'),
          dict(name='DBR', bank='Select BR rows/columns0,4,8,12 exactly; reconstruct4x4 bilinearly; must reproduce oldpool4.'),
          dict(name='H', bank='Sample R at all13x13 BR centers27.5+14j: raw2x2 averages starting atj+1, FP64 accumulation then FP32; directly query13grid. H[::4,::4] must equal DR bitwise.')],
    grid_specs={name: list(value) for name, value in GRID.items()},
    DR_limitation='Half-grid raw sampling is itself a2x2 linear average. This is not a no-low-pass operator, and the factorial contrasts do not isolate perfectly orthogonal physical mechanisms.',
    H_limitation='BR and H match phase/density but compare4x4 versus2x2 averaging, not filtering versus unfiltered input. H minus R changes both phase and interpolation averaging, so it is not a pure-phase effect.',
    B_limitation='Even4 valid averaging shifts the dense native grid phase by half a raw spacing modulo14pixels; R versus BR changes grid phase as well as averaging.',
    roi=old.DESIGN['roi'], candidate_grid=old.DESIGN['grid'], tie=old.DESIGN['tie'],
    interpolation='Bilinear on each arm native physical centers; border replication; identical9 source ROI points and625 candidate translations. Record every source/target clamp.',
    descriptor=old.DESIGN['descriptor'], metric='old.costs centered cosine; no origin change or fitted normalization.',
    correct_y_rank='Analysis only: nearest ground-truth y row, nearest ground-truth x column. Nearest-grid ties prefer smallest absolute offset then smaller signed offset. Rank=1+number of costs strictly below target-cell cost minus1e-12, both globally625 and within the correct-y row25. Report ties, gaps to respective minima and out-of-support status.',
    subsets=dict(all='All1024 object queries.',
                 common_unclamped='An object query enters iff all9 source points AND all625x9 target points are unclamped in ALL5 arms; fixed intersection, no cost/label selection. Independently assert every ROI coordinate is within closed interval[33.5,189.5].'),
    analysis=dict(metrics=['dx_mse', 'motion_mse'], units='normalized image coordinates squared; also report pixel RMSE',
                  sign='abs(true axis displacement)>=1pixel; wrong nonzero sign and predicted zero separate',
                  descriptive='dy MSE/RMSE and four-coordinate MSE/skill with immutable original frozen FIT mean; no additional bootstrap contrasts.',
                  bootstrap=dict(seed=7819, replicates=10000, unit='256 whole worlds; keep both variants/objects together',
                    subset_estimator='Within each resample divide summed included-object losses by summed included-object counts. Do not equally weight worlds with different included counts.',
                    interval='paired percentile2.5/97.5, conditional exploratory, no multiplicity adjustment'),
                  contrasts=[dict(name=n, weights=list(w), definition=d) for n, w, d in CONTRASTS]),
    fixed_cases=[list(x) for x in CASES],
    outputs='Every arm cost/prediction/source-clamp/target-clamp, common subset flags, ROI points, correct-y ranks, norm summaries, world loss sums/counts, bootstrap draws, all fixed-case figures.',
    limits='Known-Dev source-RGB-privileged correspondence only. Different native boundary supports are controlled with the common unclamped subset. Coarse sampling and reconstruction remain coupled; phase and DR implicit smoothing matter. No module-responsibility percentage, neural defect, memory-training, fresh-confirmation or equal-byte-storage claim.',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def record(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha(path))


def source_identity():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def validate_publication(publication, plan):
    p, d = json.loads(Path(publication).read_text()), json.loads(Path(plan).read_text())
    if p.get('verified_readback') is not True or d.get('design') != DESIGN or d.get('sources') != source_identity():
        raise ValueError('Exact published source, protocol and design required before scientific reads')
    if p.get('verified_unix', 0) < d['created_unix']:
        raise ValueError('Read-back must follow design freeze')
    for path in [*(ROOT / name for name in SOURCE_FILES), Path(plan).resolve()]:
        rel = path.relative_to(ROOT).as_posix()
        item = p.get('files', {}).get(rel)
        if (item.get('sha256') if isinstance(item, dict) else item) != sha(path):
            raise ValueError('Unverified publication file: ' + rel)
    return p, d


def weights(points, side, origin, step):
    """Bilinear interpolation on explicitly declared native physical centers."""
    points = np.asarray(points, dtype=np.float64)
    u = (points - origin) / step
    clamp = np.any((u < 0) | (u > side - 1), axis=-1)
    u = np.clip(u, 0, side - 1)
    left = np.floor(u).astype(int)
    right = np.minimum(left + 1, side - 1)
    frac = u - left
    matrix = np.zeros((points.reshape(-1, 2).shape[0], side * side), dtype=np.float64)
    row = np.arange(len(matrix))
    l, r, f = left.reshape(-1, 2), right.reshape(-1, 2), frac.reshape(-1, 2)
    for xx, yy, w in ((l[:, 0], l[:, 1], (1-f[:, 0])*(1-f[:, 1])),
                      (r[:, 0], l[:, 1], f[:, 0]*(1-f[:, 1])),
                      (l[:, 0], r[:, 1], (1-f[:, 0])*f[:, 1]),
                      (r[:, 0], r[:, 1], f[:, 0]*f[:, 1])):
        np.add.at(matrix, (row, yy * side + xx), w)
    return matrix.reshape(*points.shape[:-1], side * side), clamp


def make_banks(raw):
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[:2] != (2, 256) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Two finite FP32 raw16x16 token frames required')
    channels = raw.shape[-1]
    grid = raw.reshape(2, 16, 16, channels)
    box = np.empty((2, 13, 13, channels), dtype=np.float32)
    half = np.empty_like(box)
    for y in range(13):
        for x in range(13):
            box[:, y, x] = grid[:, y:y+4, x:x+4].mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
            half[:, y, x] = grid[:, y+1:y+3, x+1:x+3].mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
    a = 27.5 + 56 * np.arange(4)
    xx, yy = np.meshgrid(a, a)
    locations = np.stack((xx.ravel(), yy.ravel()), axis=1)
    w, clamped = weights(locations, *GRID['R'])
    if clamped.any():
        raise AssertionError('Registered decimation locations are internal to raw grid')
    sampled = np.stack([w @ frame.astype(np.float64) for frame in raw]).astype(np.float32)
    banks = dict(R=raw, BR=box.reshape(2, 169, channels), DR=sampled,
                 DBR=box[:, ::4, ::4].reshape(2, 16, channels).copy(), H=half.reshape(2, 169, channels))
    np.testing.assert_array_equal(banks['DBR'], old.pool(raw, 4))
    np.testing.assert_array_equal(half[:, ::4, ::4].reshape(2, 16, channels), banks['DR'])
    return banks


def choose(cost, allowed=None):
    ids = np.arange(len(cost)) if allowed is None else np.asarray(allowed)
    tied = ids[cost[ids] <= cost[ids].min() + 1e-12]
    order = np.lexsort((old.OFFSETS[tied, 1], old.OFFSETS[tied, 0], (old.OFFSETS[tied] ** 2).sum(axis=1)))
    best = int(tied[order[0]])
    ordered = np.sort(cost[ids])
    return best, dict(best_index=best, best_cost=float(cost[best]), tie_count=int(len(tied)),
        second_minus_best=float(ordered[1]-ordered[0]), near_best_count=int(np.sum(cost[ids] <= cost[ids].min()+1e-6)),
        boundary_optimum=bool(np.any(np.abs(old.OFFSETS[best]) == 6)))


def correct_row_rank(cost, truth_pixels):
    axis = np.arange(-6, 6.01, .5)
    nearest = [min(range(25), key=lambda i: (abs(axis[i]-v), abs(axis[i]), axis[i])) for v in truth_pixels]
    ix, iy = nearest
    row = iy * 25 + np.arange(25)
    cell = iy * 25 + ix
    target_cost = cost[cell]
    best, _ = choose(cost, row)
    return dict(correct_y_row_index=int(iy), nearest_truth_x_index=int(ix),
        true_nearest_grid_dx=float(axis[ix]), true_nearest_grid_dy=float(axis[iy]),
        true_cell_row_rank=int(1 + np.sum(cost[row] < target_cost-1e-12)),
        true_cell_row_ties=int(np.sum(np.abs(cost[row]-target_cost) <= 1e-12)),
        true_cell_row_cost_gap=float(target_cost-cost[row].min()), correct_row_best_dx=float(old.OFFSETS[best, 0]),
        true_cell_global_rank=int(1+np.sum(cost < target_cost-1e-12)),
        true_cell_global_ties=int(np.sum(np.abs(cost-target_cost) <= 1e-12)),
        true_cell_global_cost_gap=float(target_cost-cost.min()),
        true_displacement_out_of_support=bool(np.any(np.abs(truth_pixels) > 6)))


def match(bank, roi, arm):
    sw, sc = weights(roi, *GRID[arm])
    tw, tc = weights(roi[None] + old.OFFSETS[:, None], *GRID[arm])
    cost, diagnostic = old.costs(bank, sw, tw)
    best, selected = choose(cost)
    # Norm summaries only, using the exact same centered descriptors/interpolation.
    active = np.flatnonzero((abs(sw).sum(0)+abs(tw).sum((0, 1))) > 0)
    x, y = bank[0, active].astype(np.float64), bank[1, active].astype(np.float64)
    x -= x.mean(1, keepdims=True)
    y -= y.mean(1, keepdims=True)
    sn = ((sw[:, active] @ x) ** 2).sum(1)
    t = tw[:, :, active]
    tn = np.einsum('kpn,nm,kpm->kp', t, y @ y.T, t)
    if not np.isfinite(sn).all() or not np.isfinite(tn).all() or tn.min() < -1e-8:
        raise ValueError('Nonfinite or invalid descriptor squared norms')
    diagnostic.update(selected)
    diagnostic.update(source_clamped_points=int(sc.sum()), best_target_clamped_points=int(tc[best].sum()),
        all_target_clamped_points=int(tc.sum()), source_norm2_min=float(sn.min()),
        source_norm2_mean=float(sn.mean()), source_norm2_max=float(sn.max()),
        target_norm2_min=float(tn.min()), target_norm2_mean=float(tn.mean()), target_norm2_max=float(tn.max()),
        best_target_norm2_min=float(tn[best].min()), best_target_norm2_mean=float(tn[best].mean()),
        best_target_norm2_max=float(tn[best].max()))
    return old.OFFSETS[best].copy(), cost, sc, tc, diagnostic


def bootstrap_means(sums, counts, replicates=10000, seed=7819):
    """sums[subset,world,arm,metric]; counts[subset,world]."""
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, sums.shape[0], sums.shape[2], sums.shape[3]), dtype=np.float64)
    for start in range(0, replicates, 256):
        n = min(256, replicates-start)
        ids = rng.integers(0, sums.shape[1], size=(n, sums.shape[1]))
        denominator = counts[:, ids].sum(axis=2).T
        if np.any(denominator <= 0):
            raise ValueError('Empty common subset in a bootstrap replicate; retain failure, do not redraw')
        totals = sums[:, ids].sum(axis=2).transpose(1, 0, 2, 3)
        result[start:start+n] = totals / denominator[:, :, None, None]
    return result


def summarize(predictions, truth, common, fit_mean):
    # Worlds x variants x objects x arms x metrics; no object or variant resampling.
    errors = (predictions-truth[:, None]).reshape(256, 2, len(METHODS), 2, 2)
    loss = np.stack((errors[..., 0]**2, (errors**2).mean(axis=-1)), axis=-1).transpose(0, 1, 3, 2, 4)
    masks = np.stack((np.ones((256, 2, 2), dtype=bool), common.reshape(256, 2, 2)))
    sums = np.stack([(loss*m[:, :, :, None, None]).sum(axis=(1, 2)) for m in masks])
    counts = masks.sum(axis=(2, 3))
    if np.any(counts.sum(axis=1) == 0):
        raise ValueError('Registered common unclamped subset is empty')
    samples = bootstrap_means(sums, counts)
    result = {}
    for si, name in enumerate(('all', 'common_unclamped')):
        mask = masks[si].reshape(512, 2)
        y = truth.reshape(512, 2, 2)[mask]
        result[name] = dict(objects=int(mask.sum()), objects_excluded=int(1024-mask.sum()),
            episodes_with_objects=int(mask.any(axis=1).sum()), episodes_excluded=int((~mask.any(axis=1)).sum()),
            worlds_with_objects=int(np.sum(counts[si] > 0)), worlds_excluded=int(np.sum(counts[si] == 0)),
            per_world_counts=counts[si].tolist(), arms={}, contrasts=[])
        for a, arm in enumerate(METHODS):
            p = predictions[:, a].reshape(512, 2, 2)[mask]
            eligible = abs(y*224) >= 1
            point = sums[si, :, a].sum(axis=0)/counts[si].sum()
            dy_mse = float(np.mean((p[:, 1]-y[:, 1])**2))
            coordinate_mse, coordinate_skill, coordinate_count = [], [], []
            for column in range(4):
                valid = mask[:, column//2]
                if not valid.any(): raise ValueError('Empty object-specific common subset')
                mse = float(np.mean((predictions[valid, a, column]-truth[valid, column])**2))
                den = float(np.mean((truth[valid, column]-fit_mean[4+column])**2))
                if den <= 0: raise ValueError('Degenerate frozen-FIT baseline denominator')
                coordinate_mse.append(mse); coordinate_skill.append(1-mse/den); coordinate_count.append(int(valid.sum()))
            result[name]['arms'][arm] = dict(dx_rmse_pixels=float(np.sqrt(point[0])*224),
                motion_rmse_pixels=float(np.sqrt(point[1])*224),
                dy_mse=dy_mse, dy_rmse_pixels=float(np.sqrt(dy_mse)*224),
                coordinate_order=['Ax', 'Ay', 'Bx', 'By'], coordinate_counts=coordinate_count,
                per_coordinate_mse=coordinate_mse, per_coordinate_skill=coordinate_skill,
                motion_skill=float(np.mean(coordinate_skill)),
                sign_eligible_xy=eligible.sum(axis=0).tolist(),
                wrong_nonzero_sign_xy=((p*y < 0)&eligible).sum(axis=0).tolist(),
                predicted_zero_eligible_xy=((p == 0)&eligible).sum(axis=0).tolist(),
                metrics={metric: dict(point=float(point[j]), ci95=np.percentile(samples[:, si, a, j], [2.5, 97.5]).tolist())
                         for j, metric in enumerate(('dx_mse', 'motion_mse'))})
        for cname, weights_, definition in CONTRASTS:
            w = np.array(weights_)
            difference = (samples[:, si]*w[None, :, None]).sum(axis=1)
            point = ((sums[si].sum(axis=0)/counts[si].sum())*w[:, None]).sum(axis=0)
            result[name]['contrasts'].append(dict(name=cname, definition=definition,
                metrics={metric: dict(point=float(point[j]), ci95=np.percentile(difference[:, j], [2.5, 97.5]).tolist())
                         for j, metric in enumerate(('dx_mse', 'motion_mse'))}))
    return result, samples, sums, counts


def figures(out, predictions, truth, costs, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = out/'figures'
    d.mkdir()
    for pair, variant in CASES:
        e = 2*(pair-640)+variant
        fig, axes = plt.subplots(2, len(METHODS), figsize=(17, 8), sharex=True, sharey=True)
        for o in range(2):
            for a, arm in enumerate(METHODS):
                ax = axes[o, a]
                ax.imshow(costs[e, a, o].reshape(25, 25), origin='lower', extent=(-6.25, 6.25, -6.25, 6.25))
                ax.scatter(*(predictions[e, a, o*2:o*2+2]*224), marker='x', color='white', s=50)
                ax.scatter(*(truth[e, o*2:o*2+2]*224), marker='*', color='#ff6040', edgecolors='black', linewidth=.3, s=65)
                ax.set_title(f'{arm}, object {"AB"[o]}')
                ax.set_xlabel('dx pixels'); ax.set_ylabel('dy pixels')
        fig.suptitle(f'Known Dev {pair}/{variant}: white x prediction; red star truth; each panel has its own cost scale')
        fig.tight_layout()
        for ext in ('png', 'svg'): fig.savefig(d/f'case_{pair}_{variant}_costs.{ext}', dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
        for o in range(2):
            for coordinate in range(2):
                ax = axes[o, coordinate]; column = 2*o+coordinate
                ax.scatter(predictions[e, :, column]*224, np.arange(len(METHODS)), color='#286f9b')
                ax.axvline(truth[e, column]*224, color='#bd351d', label='Ground truth')
                ax.axvline(0, color='gray', linewidth=.5)
                ax.set_yticks(np.arange(len(METHODS)), METHODS); ax.set_xlim(-6.5, 6.5)
                ax.set_title(f'Object {"AB"[o]}, {"xy"[coordinate]} displacement'); ax.set_xlabel('Pixels')
        axes[0, 0].invert_yaxis(); axes[0, 0].legend()
        fig.suptitle(f'Known Dev {pair}/{variant}: all five fixed spatial operators')
        fig.tight_layout()
        for ext in ('png', 'svg'): fig.savefig(d/f'case_{pair}_{variant}_predictions.{ext}', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, subset in zip(axes, ('all', 'common_unclamped')):
        s = scores[subset]
        ax.bar(METHODS, [s['arms'][a]['dx_rmse_pixels'] for a in METHODS])
        ax.set_title(f'{subset}: {s["objects"]} object queries')
        ax.set_ylabel('dx RMSE (pixels)')
    fig.suptitle('Known Dev spatial-operator diagnosis; source RGB privileged ROI')
    fig.tight_layout()
    for ext in ('png', 'svg'): fig.savefig(d/f'all_arms_dx_rmse.{ext}', dpi=150)
    plt.close(fig)


def self_test():
    rng = np.random.default_rng(58071)
    raw = rng.normal(size=(2, 256, 11)).astype(np.float32)
    banks = make_banks(raw)
    np.testing.assert_array_equal(banks['DBR'], old.pool(raw, 4))
    # Half-grid D is exactly the 2x2 average at indices1/2,5/6,9/10,13/14.
    grid = raw.reshape(2, 16, 16, 11)
    explicit = np.stack([grid[:, y:y+2, x:x+2].mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
                         for y in (1, 5, 9, 13) for x in (1, 5, 9, 13)], axis=1)
    np.testing.assert_array_equal(banks['DR'], explicit)
    for y in range(13):
        for x in range(13):
            np.testing.assert_array_equal(banks['BR'][:, y*13+x],
                grid[:, y:y+4, x:x+4].mean(axis=(1, 2), dtype=np.float64).astype(np.float32))
            np.testing.assert_array_equal(banks['H'][:, y*13+x],
                grid[:, y+1:y+3, x+1:x+3].mean(axis=(1, 2), dtype=np.float64).astype(np.float32))
    np.testing.assert_array_equal(banks['H'].reshape(2, 13, 13, 11)[:, ::4, ::4].reshape(2, 16, 11), banks['DR'])
    interior = np.array([[x, y] for y in (81., 95., 106.) for x in (77., 93., 110.)])
    border = np.array([[x, y] for y in (-2., 5., 222.) for x in (-1., 2., 226.)])
    for arm in METHODS:
        for roi in (interior, border):
            pred, cost, sc, tc, _ = match(banks[arm], roi, arm)
            sw, _ = weights(roi, *GRID[arm]); tw, _ = weights(roi[None]+old.OFFSETS[:, None], *GRID[arm])
            np.testing.assert_allclose(sw.sum(-1), 1, atol=1e-15)
            np.testing.assert_allclose(tw.sum(-1), 1, atol=1e-15)
            source = sw@banks[arm][0].astype(np.float64); target = tw@banks[arm][1].astype(np.float64)
            source -= source.mean(-1, keepdims=True); target -= target.mean(-1, keepdims=True)
            explicit_cost = np.mean(1-np.einsum('pc,kpc->kp', source, target)/(
                np.linalg.norm(source, axis=-1)[None]*np.linalg.norm(target, axis=-1)), axis=1)
            np.testing.assert_allclose(cost, explicit_cost, atol=1e-13, rtol=0)
            if roi is border: assert sc.any() and tc.any()
            else: assert not sc.any() and not tc.any()
            if arm in ('R', 'DBR'):
                side = 16 if arm == 'R' else 4
                old_pred, old_cost, _ = old.match(banks[arm], roi, side=side)
                np.testing.assert_allclose(cost, old_cost, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(pred, old_pred)
    constant = make_banks(np.ones_like(raw))
    for arm in METHODS:
        pred, cost, _, _, info = match(constant[arm], interior, arm)
        np.testing.assert_array_equal(pred, [0, 0]); np.testing.assert_array_equal(cost, np.ones(625))
        assert info['tie_count'] == 625 and info['source_zero_norm_count'] == 9
    # The closed common-support interval accounts for every +/-6 candidate.
    for x, expected in ((33.5, True), (189.5, True), (33.5-1e-6, False), (189.5+1e-6, False)):
        points = np.tile([x, 111.], (9, 1)); unclamped = True
        for arm in METHODS:
            _, sc = weights(points, *GRID[arm]); _, tc = weights(points[None]+old.OFFSETS[:, None], *GRID[arm])
            unclamped = unclamped and not sc.any() and not tc.any()
        assert unclamped == expected
    # Bilinear fields preserve affine spatial descriptors on the interior.
    loc = old.centers(16); linear = np.stack((loc[:, 0], loc[:, 1], loc[:, 0]+2*loc[:, 1]), 1)
    affine = make_banks(np.stack((linear, linear)).astype(np.float32))
    for arm in METHODS:
        sw, _ = weights(interior, *GRID[arm]); actual = sw@affine[arm][0]
        np.testing.assert_allclose(actual, np.stack((interior[:, 0], interior[:, 1], interior[:, 0]+2*interior[:, 1]), 1), atol=1e-12, rtol=0)
    toy_cost = ((old.OFFSETS-np.array([1.5, -2.]))**2).sum(1)
    row = correct_row_rank(toy_cost, np.array([1.6, -2.1])); assert row['true_cell_row_rank'] == 1 and row['true_cell_global_rank'] == 1 and row['correct_row_best_dx'] == 1.5
    sums = rng.uniform(size=(2, 8, len(METHODS), 2)); counts = np.array([[4]*8, [1, 2, 3, 4, 1, 2, 3, 4]])
    samples = bootstrap_means(sums, counts, replicates=19)
    ids = np.random.default_rng(7819).integers(0, 8, size=(19, 8))
    expected = np.array([[sums[s, ix].sum(0)/counts[s, ix].sum() for s in range(2)] for ix in ids])
    np.testing.assert_allclose(samples, expected, atol=1e-15, rtol=0)
    return dict(status='passed', software_only=True, scientific_inputs_opened=False,
        old_pool_bank_exact=True, DR_2x2_mean_exact=True, dense_box_filter_all169_positions=True,
        H_dense_half_grid_all169_positions=True, H_coarse_equals_DR_bitwise=True,
        baseline_cost_prediction_match=True, explicit_cosine=True, border_replication=True,
        affine_field_preserved=True, constant_ambiguity=True, correct_y_and_global_rank=True,
        common_support_closed_interval=True,
        whole_world_subset_count_weighted_bootstrap=True)


def run(scene, costs_path, baseline_report, out, publication, plan):
    started = time.time()
    scene, costs_path, baseline_report, out = map(lambda p: Path(p).resolve(), (scene, costs_path, baseline_report, out))
    validate_publication(publication, plan)
    initial = dict(sources=source_identity(), plan=sha(plan), publication=sha(publication))
    if out.exists() or out == scene or scene in out.parents or out == costs_path.parent or costs_path.parent in out.parents:
        raise ValueError('Fresh output outside both immutable input directories required')
    for name, digest in DESIGN['immutable_scene_inputs'].items():
        if sha(scene/name) != digest: raise ValueError('Pinned scene input changed: '+name)
    if sha(costs_path) != DESIGN['baseline']['costs_sha256'] or sha(baseline_report) != DESIGN['baseline']['report_sha256']:
        raise ValueError('Pinned V5.6 baseline changed')
    previous = json.loads(baseline_report.read_text())
    if previous.get('status') != 'complete': raise ValueError('Completed V5.6 baseline required')
    for name, digest in previous['provenance']['source_identity'].items():
        if initial['sources'].get(name) != digest: raise ValueError('Baseline source changed: '+name)
    manifest = json.loads((scene/'archive_manifest.json').read_text()); opened = {}
    def checked(name):
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or rel.as_posix() != name: raise ValueError('Unsafe scene member')
        p = scene/rel
        if any(x.is_symlink() for x in [p, *p.parents]): raise ValueError('Scene symlink forbidden')
        item = record(p)
        if manifest['files'].get(name) != item: raise ValueError('Scene member changed: '+name)
        opened[name] = item
        return p
    receipt = json.loads(checked('counterfactual_cache/receipt.json').read_text())
    if receipt['groups'] != 256 or receipt['test_opened'] is not False: raise ValueError('Original population changed')
    with np.load(checked('report/predictions.npz'), allow_pickle=False) as z:
        truth8 = z['targets'][..., 7, :].reshape(512, 8).astype(np.float64)
    with np.load(checked('lane_reader/frozen_model.npz'), allow_pickle=False) as z:
        fit_mean = z['shared_target_mean'].copy()
    with np.load(costs_path, allow_pickle=False) as z:
        if tuple(z['methods'].tolist()) != old.METHODS: raise ValueError('Baseline method order changed')
        for key, expected in (('targets', truth8[:, 4:]), ('offsets_pixels', old.OFFSETS),
                              ('pair_ids', np.repeat(np.arange(640, 896), 2)), ('variants', np.tile([0, 1], 256)),
                              ('frozen_fit_mean', fit_mean)):
            np.testing.assert_array_equal(z[key], expected)
        baseline_cost, baseline_prediction = z['costs'][:, :2].copy(), z['predictions'][:, :2].copy()
    if baseline_cost.shape != (512, 2, 2, 625) or baseline_prediction.shape != (512, 2, 4): raise ValueError('Baseline shape changed')
    out.mkdir(parents=True)
    predictions = np.empty((512, len(METHODS), 4), dtype=np.float64)
    all_costs = np.empty((512, len(METHODS), 2, 625), dtype=np.float64)
    source_clamp = np.empty((512, len(METHODS), 2, 9), dtype=bool)
    target_clamp = np.empty((512, len(METHODS), 2, 625, 9), dtype=bool)
    rois = np.empty((512, 2, 9, 2), dtype=np.float64)
    rows, rgb_records, bank_records = [], [], []
    baseline_max = 0.
    for pair in range(640, 896):
        rr = receipt['raw_records'][str(pair)]
        rp = checked('counterfactual_cache/'+rr['path'])
        if sha(rp) != rr['sha256']: raise ValueError('Raw receipt mismatch')
        with np.load(rp, allow_pickle=False) as z:
            if set(z.files) != {'raw'}: raise ValueError('Unexpected raw entries')
            raw = z['raw']
        if raw.shape != (2, 2, 256, 1280) or raw.dtype != np.float32 or {'raw': old.entry_identity(raw)} != rr['entry_hashes']:
            raise ValueError('Raw array identity mismatch')
        shard = json.loads(checked(f'counterfactual_cache/shards/{pair:06d}.json').read_text())
        for variant in (0, 1):
            e = 2*(pair-640)+variant
            pos, colors, yellow = old.cue_geometry('dev', pair, variant)
            rgb, _ = old.render_endpoint(pos[0], colors, yellow[0])
            rgb_id = old.entry_identity(rgb)
            if rgb_id != shard['inputs']['variants'][variant]['unique_rgb'][0]: raise ValueError('Source RGB differs')
            labels = np.concatenate((pos[1].ravel(), (pos[1]-pos[0]).ravel())).astype(np.float32)
            np.testing.assert_array_equal(labels.astype(np.float64), truth8[e])
            regions, _, _ = old.source_roi(rgb); rois[e] = regions
            rgb_records.append(dict(pair_id=pair, variant=variant, source_rgb=rgb_id, roi=old.entry_identity(regions)))
            banks = make_banks(raw[variant])
            bank_records.append(dict(pair_id=pair, variant=variant, banks={name: old.entry_identity(banks[name]) for name in METHODS}))
            for a, arm in enumerate(METHODS):
                for obj in (0, 1):
                    pred, cost, sc, tc, diag = match(banks[arm], regions[obj], arm)
                    predictions[e, a, obj*2:obj*2+2] = pred/224
                    all_costs[e, a, obj] = cost
                    source_clamp[e, a, obj], target_clamp[e, a, obj] = sc, tc
                    y = truth8[e, 4+obj*2:6+obj*2]*224
                    rows.append(dict(pair_id=pair, variant=variant, object='AB'[obj], arm=arm,
                        prediction_dx_pixels=float(pred[0]), prediction_dy_pixels=float(pred[1]),
                        target_dx_pixels=float(y[0]), target_dy_pixels=float(y[1]),
                        **diag, **correct_row_rank(cost, y)))
                    if arm in ('R', 'DBR'):
                        j = 0 if arm == 'R' else 1
                        delta = float(abs(cost-baseline_cost[e, j, obj]).max()); baseline_max = max(baseline_max, delta)
                        np.testing.assert_allclose(cost, baseline_cost[e, j, obj], atol=1e-10, rtol=0)
                        np.testing.assert_array_equal(pred/224, baseline_prediction[e, j, obj*2:obj*2+2])
        if (pair-639) % 32 == 0: print(json.dumps(dict(progress_worlds=pair-639, total_worlds=256)), flush=True)
    common = ~(source_clamp.any(axis=(1, 3)) | target_clamp.any(axis=(1, 3, 4)))
    geometry_common = ((rois >= 33.5) & (rois <= 189.5)).all(axis=(2, 3))
    np.testing.assert_array_equal(common, geometry_common)
    scores, samples, sums, counts = summarize(predictions, truth8[:, 4:], common, fit_mean)
    for row in rows:
        e = 2*(row['pair_id']-640)+row['variant']; obj = 'AB'.index(row['object'])
        row['common_unclamped'] = bool(common[e, obj])
    np.savez_compressed(out/'costs_and_predictions.npz', costs=all_costs, predictions=predictions, targets=truth8[:, 4:],
        methods=np.array(METHODS), pair_ids=np.repeat(np.arange(640, 896), 2), variants=np.tile([0, 1], 256),
        offsets_pixels=old.OFFSETS, source_clamped=source_clamp, target_clamped=target_clamp,
        common_unclamped=common, roi_points_pixels=rois, frozen_fit_mean=fit_mean)
    np.savez_compressed(out/'bootstrap.npz', mean_losses=samples, world_loss_sums=sums, world_counts=counts,
        methods=np.array(METHODS), subsets=np.array(['all', 'common_unclamped']),
        metrics=np.array(['dx_mse', 'motion_mse']), world_ids=np.arange(640, 896), seed=np.array(7819))
    with (out/'per_episode_object.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    save(out/'source_rgb_records.json', rgb_records); save(out/'bank_identities.json', bank_records)
    save(out/'fixed_cases.json', [dict(pair_id=p, variant=v,
        targets_pixels=(truth8[2*(p-640)+v, 4:]*224).tolist(), common_unclamped=common[2*(p-640)+v].tolist(),
        predictions_pixels={arm: (predictions[2*(p-640)+v, a]*224).tolist() for a, arm in enumerate(METHODS)}) for p, v in CASES])
    figures(out, predictions, truth8[:, 4:], all_costs, scores)
    for name, expected in opened.items():
        if record(scene/name) != expected: raise ValueError('Input changed during scoring: '+name)
    if (source_identity() != initial['sources'] or sha(plan) != initial['plan'] or sha(publication) != initial['publication'] or
        sha(costs_path) != DESIGN['baseline']['costs_sha256'] or sha(baseline_report) != DESIGN['baseline']['report_sha256'] or
        sha(scene/'archive_manifest.json') != DESIGN['immutable_scene_inputs']['archive_manifest.json']):
        raise ValueError('Source/design/baseline changed during scoring')
    save(out/'report.json', dict(schema=SCHEMA, status='complete', design=DESIGN, scores=scores,
        checks=dict(baseline_cost_max_abs=baseline_max, baseline_predictions_exact=True, DBR_bank_bitwise_old_pool=True,
                    H_coarse_bitwise_DR=True, common_mask_matches_closed_physical_interval=True,
                    finite_costs_predictions_norms=True, source_rgb_hashes_verified=len(rgb_records)),
        ambiguity={arm: dict(ties_gt1=sum(row['tie_count'] > 1 for row in rows if row['arm'] == arm),
            boundary_optima=sum(row['boundary_optimum'] for row in rows if row['arm'] == arm)) for arm in METHODS},
        provenance=dict(source_identity=initial['sources'], plan_sha256=initial['plan'], publication_sha256=initial['publication'],
            old_scene_manifest_sha256=sha(scene/'archive_manifest.json'), scene_inputs=opened,
            baseline_costs=record(costs_path), baseline_report=record(baseline_report), command=sys.argv),
        runtime=dict(python=sys.version, numpy=np.__version__, blas_threads=4, elapsed_seconds=time.time()-started),
        test_opened=False, no_training=True, no_module_responsibility_fraction=True))
    save(out/'archive_manifest.json', dict(schema=SCHEMA, files={p.relative_to(out).as_posix(): record(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(status='complete', out=str(out), baseline_cost_max_abs=baseline_max,
                all_object_queries=1024, common_unclamped_objects=int(common.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    sub.add_parser('self-test')
    p = sub.add_parser('freeze-design'); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run')
    for name in ('scene', 'costs', 'baseline-report', 'out', 'publication', 'plan'):
        p.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    if a.phase == 'self-test': result = self_test()
    elif a.phase == 'freeze-design':
        result = dict(design=DESIGN, sources=source_identity(), software_verification=self_test(), created_unix=time.time())
        save(a.out, result)
    else:
        failure_ok = (not a.out.exists() and a.out.resolve() != a.scene.resolve() and a.scene.resolve() not in a.out.resolve().parents and
                      a.out.resolve() != a.costs.resolve().parent and a.costs.resolve().parent not in a.out.resolve().parents)
        try: result = run(a.scene, a.costs, a.baseline_report, a.out, a.publication, a.plan)
        except Exception as exc:
            if failure_ok:
                a.out.mkdir(parents=True, exist_ok=True)
                if not (a.out/'failure.json').exists(): save(a.out/'failure.json', dict(status='failed', error=str(exc), traceback=traceback.format_exc(), command=sys.argv))
            raise
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
