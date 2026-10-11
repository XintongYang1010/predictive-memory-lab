#!/usr/bin/env python3
"""Published, fixed posthoc diagnostics of existing V5.6 correspondence costs.

This script never encodes images, fits a model, or opens Test. Oracle rows and
columns use already inspected Dev labels and are privileged diagnostics.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('raw_native', 'pool_native', 'uniform_budget', 'roi_adaptive_budget',
           'random_budget', 'change_priority_budget')
READOUTS = ('min2d', 'oracle_gt_y', 'zero_y', 'oracle_gt_x', 'mismatched_gt_y')
AXES = ('A_dx', 'A_dy', 'B_dx', 'B_dy')
SOURCE_FILES = ('scripts/v5_7_cost_diagnosis.py', 'scripts/v5_6_correspondence.py',
                'docs/V5_7_EXECUTION_PROTOCOL.md')
INPUT_SHA256 = '265766a0d1de852f5eb863c068d08de11c696ed961e76768468797a6e9879eb0'
TOL = 1e-12
DESIGN = {
    'schema': 'v5.7-saved-cost-diagnosis-design-1',
    'input_sha256': INPUT_SHA256,
    'population': {'split': 'known_dev', 'worlds': 256, 'episodes': 512,
                   'pair_ids': [640, 895], 'variants': [0, 1], 'objects': ['A', 'B']},
    'methods': list(METHODS), 'readouts': list(READOUTS),
    'grid': {'minimum_pixels': -6., 'maximum_pixels': 6., 'step_pixels': .5,
             'count': 625, 'order': 'dy major; dx minor; each offset is [dx,dy]'},
    'cost_tie': {'absolute_tolerance': TOL,
                 'selection': 'cost <= minimum+1e-12; smallest norm squared, then dx, then dy'},
    'nearest_axis_tie': 'minimum exact absolute distance; smallest absolute ordinate, then smaller ordinate',
    'readout_rules': {
        'min2d': 'Original 625-candidate minimum; must replay every archived prediction exactly.',
        'oracle_gt_y': 'Use nearest grid row to original GT dy; minimize its 25 dx candidates.',
        'zero_y': 'Use dy=0 row; minimize its 25 dx candidates.',
        'oracle_gt_x': 'Use nearest grid column to original GT dx; minimize its 25 dy candidates.',
        'mismatched_gt_y': 'Donor world=(world_index+37)%256, preserving variant/object; nearest donor GT dy row, minimize dx. All methods use the same donors.'},
    'rank': '1 + count(cost < nearest-GT-grid cost - 1e-12); tie count uses absolute difference <=1e-12. Report globally and within original GT-y row.',
    'margins': 'GT-grid cost minus actual minimum, and GT-grid cost minus best other candidate; global and GT-y row. Negative best-other margin means GT grid is strictly better.',
    'geometric_lower_bound': 'Nearest grid point to continuous GT, independent of costs; lower bound only for predictors restricted to this finite grid.',
    'metrics': {'units': 'Original normalized displacement; pixels=normalized*224.',
                'skill': 'Mean of four per-axis 1-MSE/mean((GT-frozen FIT mean)^2); never Dev-centered variance.',
                'oracle_primary': 'oracle_gt_y and mismatched_gt_y: dx; oracle_gt_x: dy. Full4D scores contain the supplied privileged coordinate.',
                'direction': 'abs(GT dx pixels)>=1 inclusive; correct=product>0, wrong=product<0, zero=prediction==0. Non-correct=wrong+zero. GT zero and 0<abs(GT)<1 reported separately.'},
    'bootstrap': {'worlds': 256, 'samples': 10000, 'seed': 7817,
                  'interval': 'two-sided percentile95%; resample whole worlds preserving both variants and objects; same draws for all comparisons; conditional on these costs/readers, no multiplicity correction'},
    'effects': 'min2d minus each alternative MSE (dx/dy/4D), and mismatched_gt_y minus oracle_gt_y dx MSE; direction rescue/harm/net rates with whole-world ratio bootstrap. Report donor/true quantized-row equality; additionally report different-row dx effect with whole-world ratio bootstrap preserving its denominator. Donor-world mismatch does not imply a different y row.',
    'fixed_cases': [[761, 1], [844, 1], [891, 1]],
    'limits': 'Privileged posthoc known-Dev localization, not a deployable reader, new confirmation, or module responsibility/explained-error fraction. Nearest-grid cost rank describes this fixed surface and cannot establish descriptor information absence.',
    'no_training': True, 'test_opened': False, 'p0_authorized': False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def source_identity():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def offsets():
    axis = np.arange(-6., 6.01, .5)
    dx, dy = np.meshgrid(axis, axis)
    return np.stack((dx.ravel(), dy.ravel()), axis=1)


OFFSETS = offsets()
GRID_AXIS = np.arange(-6., 6.01, .5)


def choose_minimum(cost, candidates=None):
    """Reuse the exact V5.6 cost and displacement tie convention on a subset."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.shape != (625,) or not np.isfinite(cost).all():
        raise ValueError('A finite 625-candidate surface is required')
    candidates = np.arange(625) if candidates is None else np.asarray(candidates)
    if candidates.ndim != 1 or not len(candidates) or np.any((candidates < 0) | (candidates >= 625)):
        raise ValueError('Invalid candidate subset')
    tied = candidates[cost[candidates] <= cost[candidates].min() + TOL]
    xy = OFFSETS[tied]
    order = np.lexsort((xy[:, 1], xy[:, 0], np.sum(xy ** 2, axis=1)))
    return int(tied[order[0]])


def nearest_axis(value):
    distance = np.abs(GRID_AXIS - float(value))
    tied = np.flatnonzero(distance == distance.min())
    return int(tied[np.lexsort((GRID_AXIS[tied], np.abs(GRID_AXIS[tied])))[0]])


def donor_episodes(episodes):
    if episodes % 2 or episodes < 4:
        raise ValueError('At least two complete paired worlds are required')
    worlds = episodes // 2
    return (np.arange(episodes) // 2 + 37) % worlds * 2 + np.arange(episodes) % 2


def rank_record(cost, index, candidates):
    candidates = np.asarray(candidates)
    selected = float(cost[index])
    other = candidates[candidates != index]
    return (int(1 + np.sum(cost[candidates] < selected - TOL)),
            int(np.sum(np.abs(cost[candidates] - selected) <= TOL)),
            selected - float(cost[candidates].min()),
            selected - float(cost[other].min()))


def diagnose(costs, targets):
    """Pure array diagnostic; small populations are allowed only for software tests."""
    costs = np.asarray(costs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    episodes = len(targets)
    if costs.shape != (episodes, len(METHODS), 2, 625) or targets.shape != (episodes, 4):
        raise ValueError('Cost or target shape differs')
    if not np.isfinite(costs).all() or not np.isfinite(targets).all():
        raise ValueError('Nonfinite saved input')
    truth_pixels = targets.reshape(episodes, 2, 2) * 224
    nearest = np.array([nearest_axis(x) for x in truth_pixels.ravel()]).reshape(episodes, 2, 2)
    gt_indices = nearest[..., 1] * 25 + nearest[..., 0]
    donors = donor_episodes(episodes)
    picks = np.empty((episodes, len(METHODS), len(READOUTS), 2), dtype=np.int64)
    rank_global = np.empty((episodes, len(METHODS), 2, 4), dtype=np.float64)
    rank_row = np.empty_like(rank_global)
    for episode in range(episodes):
        for obj in (0, 1):
            gt_row = np.arange(25) + 25 * nearest[episode, obj, 1]
            column = np.arange(25) * 25 + nearest[episode, obj, 0]
            donor_row = np.arange(25) + 25 * nearest[donors[episode], obj, 1]
            subsets = (None, gt_row, np.arange(25) + 25 * 12, column, donor_row)
            for method in range(len(METHODS)):
                surface = costs[episode, method, obj]
                picks[episode, method, :, obj] = [choose_minimum(surface, subset) for subset in subsets]
                rank_global[episode, method, obj] = rank_record(surface, gt_indices[episode, obj], np.arange(625))
                rank_row[episode, method, obj] = rank_record(surface, gt_indices[episode, obj], gt_row)
    return {'predictions': (OFFSETS[picks] / 224).reshape(episodes, len(METHODS), len(READOUTS), 4),
            'selected_indices': picks, 'nearest_gt_grid_indices': gt_indices,
            'nearest_gt_grid_predictions': (OFFSETS[gt_indices] / 224).reshape(episodes, 4),
            'global_gt_rank_tie_gap_margin': rank_global,
            'gt_y_row_rank_tie_gap_margin': rank_row,
            'donor_episode_indices': donors,
            'donor_row_matches_gt_row': nearest[donors, :, 1] == nearest[:, :, 1],
            'out_of_support_axes': (np.abs(truth_pixels) > 6).reshape(episodes, 4)}


def bootstrap_indices(worlds, samples=10000, seed=7817):
    return np.random.default_rng(seed).integers(0, worlds, size=(samples, worlds))


def mean_interval(values, draws):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != draws.shape[1]:
        raise ValueError('Bootstrap unit must be one scalar per world')
    samples = values[draws].mean(axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    return {'mean': float(values.mean()), 'ci95': [float(lo), float(hi)]}


def ratio_interval(numerator_by_world, denominator_by_world, draws):
    numerator = np.asarray(numerator_by_world, dtype=np.float64)
    denominator = np.asarray(denominator_by_world, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.shape != (draws.shape[1],):
        raise ValueError('Ratio bootstrap must preserve world numerators/denominators')
    if denominator.sum() == 0:
        return {'rate': None, 'ci95': None, 'undefined_draws': len(draws)}
    denom = denominator[draws].sum(axis=1)
    valid = denom > 0
    samples = numerator[draws].sum(axis=1)[valid] / denom[valid]
    lo, hi = np.quantile(samples, [.025, .975]) if len(samples) else (None, None)
    return {'rate': float(numerator.sum() / denominator.sum()),
            'ci95': None if lo is None else [float(lo), float(hi)],
            'undefined_draws': int((~valid).sum())}


def score(prediction, truth, mean):
    prediction = np.asarray(prediction)
    error = (prediction - truth) ** 2
    mse = error.mean(axis=0)
    denominator = np.mean((truth - mean) ** 2, axis=0)
    skill = [float(1 - m / d) if d > 0 else None for m, d in zip(mse, denominator)]
    result = {'mse_by_axis': dict(zip(AXES, map(float, mse))),
              'skill_by_axis': dict(zip(AXES, skill)),
              'mean_motion_skill': None if None in skill else float(np.mean(skill)),
              'dx_mse': float(mse[[0, 2]].mean()), 'dy_mse': float(mse[[1, 3]].mean()),
              'motion_mse': float(mse.mean()),
              'dx_rmse_pixels': float(np.sqrt(mse[[0, 2]].mean()) * 224),
              'dy_rmse_pixels': float(np.sqrt(mse[[1, 3]].mean()) * 224),
              'motion_rmse_pixels': float(np.sqrt(mse.mean()) * 224)}
    result['direction_by_axis'] = {}
    for index, name in enumerate(AXES):
        gt, pred = truth[:, index] * 224, prediction[:, index] * 224
        active = np.abs(gt) >= 1
        wrong, zero, correct = pred * gt < 0, pred == 0, pred * gt > 0
        result['direction_by_axis'][name] = {
            'active_abs_gt_ge_1px': int(active.sum()), 'gt_zero': int(np.sum(gt == 0)),
            'gt_tiny_0_to_1px_exclusive': int(np.sum((np.abs(gt) > 0) & (np.abs(gt) < 1))),
            'correct_active': int(np.sum(active & correct)), 'wrong_active': int(np.sum(active & wrong)),
            'zero_active': int(np.sum(active & zero)), 'non_correct_active': int(np.sum(active & (wrong | zero))),
            'wrong_all': int(wrong.sum()), 'zero_all': int(zero.sum()),
            'non_correct_all_legacy_sign': int(np.sum(np.sign(pred) != np.sign(gt)))}
    return result


def mse_effect(first, second, truth, draws):
    delta = (first - truth) ** 2 - (second - truth) ** 2
    result = {}
    for label, columns in (('dx', [0, 2]), ('dy', [1, 3]), ('motion4d', [0, 1, 2, 3])):
        per_world = delta[:, columns].mean(axis=1).reshape(-1, 2).mean(axis=1)
        entry = mean_interval(per_world, draws)
        entry['mean_pixels_squared'] = entry['mean'] * 224 ** 2
        entry['ci95_pixels_squared'] = [value * 224 ** 2 for value in entry['ci95']]
        result[label] = entry
    return result


def direction_effect(base, alternative, truth, draws):
    gt = truth[:, [0, 2]] * 224
    old, new = base[:, [0, 2]], alternative[:, [0, 2]]
    active = np.abs(gt) >= 1
    old_correct, new_correct = old * gt > 0, new * gt > 0
    groups = {
        'non_correct_to_correct': active & ~old_correct & new_correct,
        'correct_to_non_correct': active & old_correct & ~new_correct,
        'wrong_to_correct': active & (old * gt < 0) & new_correct,
        'zero_to_correct': active & (old == 0) & new_correct,
        'correct_to_wrong': active & old_correct & (new * gt < 0),
        'correct_to_zero': active & old_correct & (new == 0),
        'still_non_correct': active & ~old_correct & ~new_correct,
        'still_correct': active & old_correct & new_correct}
    denominator = active.reshape(-1, 2, 2).sum(axis=(1, 2))
    result = {'active_object_episode_coordinates': int(active.sum()), 'groups': {}}
    for label, mask in groups.items():
        entry = ratio_interval(mask.reshape(-1, 2, 2).sum(axis=(1, 2)), denominator, draws)
        entry['count'] = int(mask.sum())
        selected = mask
        entry['base_dx_mse'] = float(((old - truth[:, [0, 2]]) ** 2)[selected].mean()) if selected.any() else None
        entry['alternative_dx_mse'] = float(((new - truth[:, [0, 2]]) ** 2)[selected].mean()) if selected.any() else None
        result['groups'][label] = entry
    net = groups['non_correct_to_correct'].astype(int) - groups['correct_to_non_correct'].astype(int)
    result['net_correctness_gain'] = ratio_interval(net.reshape(-1, 2, 2).sum(axis=(1, 2)), denominator, draws)
    return result


def masked_dx_effect(first, second, truth, mask, draws):
    """Conditional dx MSE difference, bootstrapping sums and counts by world."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(truth), 2):
        raise ValueError('Subset must select whole object-episode dx observations')
    delta = ((first - truth) ** 2 - (second - truth) ** 2)[:, [0, 2]]
    num = (delta * mask).reshape(-1, 2, 2).sum(axis=(1, 2))
    den = mask.reshape(-1, 2, 2).sum(axis=(1, 2))
    result = ratio_interval(num, den, draws)
    result['mean'] = result.pop('rate')
    result['mean_pixels_squared'] = None if result['mean'] is None else result['mean'] * 224 ** 2
    result['ci95_pixels_squared'] = None if result['ci95'] is None else [v * 224 ** 2 for v in result['ci95']]
    result.update(object_episode_coordinates=int(mask.sum()), episodes=int(mask.any(axis=1).sum()), worlds=int((den > 0).sum()))
    return result


def describe_ranks(array):
    flat = array.reshape(-1, 4)
    return {'queries': len(flat), 'rank_median': float(np.median(flat[:, 0])),
            'rank_p90': float(np.quantile(flat[:, 0], .9)),
            'rank1_count': int(np.sum(flat[:, 0] == 1)),
            'tie_count_mean': float(flat[:, 1].mean()),
            'gt_cost_minus_min_mean': float(flat[:, 2].mean()),
            'gt_cost_minus_best_other_mean': float(flat[:, 3].mean())}


def make_report(arrays, diagnosis, draws):
    truth, mean = arrays['targets'], arrays['frozen_fit_mean'][4:]
    prediction = diagnosis['predictions']
    denominator = np.mean((truth - mean) ** 2, axis=0)
    support = diagnosis['out_of_support_axes']
    same_row = diagnosis['donor_row_matches_gt_row']
    report = {'schema': 'v5.7-saved-cost-diagnosis-report-1', 'status': 'complete',
              'posthoc': True, 'privileged': True, 'test_opened': False,
              'p0_authorized': False, 'population': DESIGN['population'],
              'bootstrap': DESIGN['bootstrap'], 'definitions': DESIGN['metrics'],
              'limits': DESIGN['limits'], 'frozen_fit_mean_motion': mean.tolist(),
              'skill_denominator_by_axis': dict(zip(AXES, map(float, denominator))),
              'donor_y_row_match': {'same_count': int(same_row.sum()), 'total_object_episode_queries': int(same_row.size),
                                    'same_fraction': float(same_row.mean()), 'different_count': int((~same_row).sum()),
                                    'note': 'World identity is mismatched; the nearest y row can coincide. Same-row predictions must be exact.'},
              'out_of_support': {'axes': dict(zip(AXES, map(int, support.sum(axis=0)))),
                                 'object_episode_queries': int(support.reshape(-1, 2).any(axis=1).sum()),
                                 'episodes': int(support.any(axis=1).sum()),
                                 'worlds': int(support.reshape(-1, 2, 4).any(axis=(1, 2)).sum())},
              'geometric_grid_lower_bound': score(diagnosis['nearest_gt_grid_predictions'], truth, mean),
              'arms': {}}
    for index, method in enumerate(METHODS):
        p = prediction[:, index]
        arm = {'scores': {name: score(p[:, j], truth, mean) for j, name in enumerate(READOUTS)},
               'min2d_minus_alternative_mse': {}, 'horizontal_direction_transitions': {},
               'global_nearest_gt_grid_cost': describe_ranks(diagnosis['global_gt_rank_tie_gap_margin'][:, index]),
               'gt_y_row_nearest_gt_grid_cost': describe_ranks(diagnosis['gt_y_row_rank_tie_gap_margin'][:, index])}
        for j, name in enumerate(READOUTS[1:], 1):
            arm['min2d_minus_alternative_mse'][name] = mse_effect(p[:, 0], p[:, j], truth, draws)
            arm['horizontal_direction_transitions'][name] = direction_effect(p[:, 0], p[:, j], truth, draws)
        arm['mismatched_y_minus_correct_oracle_y_mse'] = mse_effect(p[:, 4], p[:, 1], truth, draws)
        arm['mismatched_y_minus_correct_oracle_y_different_row_dx_mse'] = masked_dx_effect(p[:, 4], p[:, 1], truth, ~same_row, draws)
        np.testing.assert_array_equal(p[:, 4].reshape(-1, 2, 2)[same_row], p[:, 1].reshape(-1, 2, 2)[same_row])
        report['arms'][method] = arm
    return report


def validate_arrays(arrays):
    expected = {'costs', 'predictions', 'targets', 'offsets_pixels', 'methods', 'pair_ids', 'variants', 'frozen_fit_mean'}
    if set(arrays) != expected:
        raise ValueError('Unexpected NPZ entries')
    shapes = {'costs': (512, 6, 2, 625), 'predictions': (512, 6, 4), 'targets': (512, 4),
              'offsets_pixels': (625, 2), 'methods': (6,), 'pair_ids': (512,),
              'variants': (512,), 'frozen_fit_mean': (8,)}
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError('Input shape differs: ' + key)
        if key != 'methods' and not np.isfinite(arrays[key]).all():
            raise ValueError('Input not finite: ' + key)
    for key in ('costs', 'predictions', 'targets', 'offsets_pixels', 'frozen_fit_mean'):
        if arrays[key].dtype != np.float64:
            raise ValueError('Input precision differs: ' + key)
    if tuple(arrays['methods']) != METHODS:
        raise ValueError('Method order differs')
    np.testing.assert_array_equal(arrays['offsets_pixels'], OFFSETS)
    np.testing.assert_array_equal(arrays['pair_ids'], np.repeat(np.arange(640, 896), 2))
    np.testing.assert_array_equal(arrays['variants'], np.tile([0, 1], 256))


def validate_publication(publication, plan):
    """This check runs before np.load, including on an attempted resumed run."""
    pub, frozen = json.loads(Path(publication).read_text()), json.loads(Path(plan).read_text())
    if frozen.get('design') != DESIGN or frozen.get('sources') != source_identity():
        raise ValueError('Frozen design/source mismatch')
    if frozen.get('software_verification', {}).get('status') != 'passed':
        raise ValueError('Software verification is absent')
    if pub.get('verified_readback') is not True or pub.get('verified_unix', 0) < frozen.get('created_unix', float('inf')):
        raise ValueError('Publication/readback must follow this design freeze')
    for path in [ROOT / name for name in SOURCE_FILES] + [Path(plan).resolve()]:
        relative = path.relative_to(ROOT).as_posix()
        record = pub.get('files', {}).get(relative)
        digest = record.get('sha256') if isinstance(record, dict) else record
        if digest != sha(path):
            raise ValueError('Published source/design missing or changed: ' + relative)
    return pub, frozen


def write_case_csv(path, arrays, result):
    rows = []
    for episode, (pair, variant) in enumerate(zip(arrays['pair_ids'], arrays['variants'])):
        for m, method in enumerate(METHODS):
            for obj in (0, 1):
                truth = arrays['targets'][episode, obj * 2:obj * 2 + 2] * 224
                near = result['nearest_gt_grid_predictions'][episode, obj * 2:obj * 2 + 2] * 224
                donor = result['donor_episode_indices'][episode]
                row = {'pair_id': int(pair), 'variant': int(variant), 'object': 'AB'[obj], 'method': method,
                       'gt_dx_pixels': float(truth[0]), 'gt_dy_pixels': float(truth[1]),
                       'nearest_grid_dx_pixels': float(near[0]), 'nearest_grid_dy_pixels': float(near[1]),
                       'dx_grid_quantization_error_pixels': float(near[0] - truth[0]),
                       'dy_grid_quantization_error_pixels': float(near[1] - truth[1]),
                       'gt_dx_out_of_support': bool(abs(truth[0]) > 6),
                       'gt_dy_out_of_support': bool(abs(truth[1]) > 6),
                       'dx_active_abs_gt_ge_1px': bool(abs(truth[0]) >= 1),
                       'donor_pair_id': int(arrays['pair_ids'][donor]),
                       'donor_variant': int(arrays['variants'][donor]),
                       'donor_y_grid_row_equals_gt_row': bool(result['donor_row_matches_gt_row'][episode, obj]),
                       'donor_gt_dy_pixels': float(arrays['targets'][donor, obj * 2 + 1] * 224)}
                for prefix, key in (('global', 'global_gt_rank_tie_gap_margin'), ('gt_y_row', 'gt_y_row_rank_tie_gap_margin')):
                    for label, value in zip(('rank', 'tie_count', 'gt_cost_minus_min', 'gt_cost_minus_best_other'), result[key][episode, m, obj]):
                        row[prefix + '_' + label] = int(value) if label in ('rank', 'tie_count') else float(value)
                for j, name in enumerate(READOUTS):
                    pred = result['predictions'][episode, m, j, obj * 2:obj * 2 + 2] * 224
                    row[name + '_dx_pixels'], row[name + '_dy_pixels'] = map(float, pred)
                    row[name + '_dx_squared_error_pixels'] = float((pred[0] - truth[0]) ** 2)
                    row[name + '_dy_squared_error_pixels'] = float((pred[1] - truth[1]) ** 2)
                    row[name + '_index'] = int(result['selected_indices'][episode, m, j, obj])
                    row[name + '_dx_direction'] = 'zero' if pred[0] == 0 else ('gt_zero' if truth[0] == 0 else ('correct' if pred[0] * truth[0] > 0 else 'wrong'))
                old, new = row['min2d_dx_direction'], row['oracle_gt_y_dx_direction']
                row['oracle_y_direction_group'] = ('inactive_abs_gt_lt_1px' if abs(truth[0]) < 1 else
                    ('repaired_non_correct' if old != 'correct' and new == 'correct' else
                     'still_non_correct' if old != 'correct' and new != 'correct' else
                     'harmed_correct' if old == 'correct' and new != 'correct' else 'still_correct'))
                rows.append(row)
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def make_figures(out, arrays, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    for pair, variant in DESIGN['fixed_cases']:
        found = np.flatnonzero((arrays['pair_ids'] == pair) & (arrays['variants'] == variant))
        if len(found) != 1:
            raise ValueError('Fixed case missing')
        episode = int(found[0])
        fig, panels = plt.subplots(6, 2, figsize=(12, 23), constrained_layout=True)
        for m, method in enumerate(METHODS):
            for obj in (0, 1):
                ax = panels[m, obj]
                cost = arrays['costs'][episode, m, obj].reshape(25, 25)
                im = ax.imshow(cost, origin='lower', extent=(-6.25, 6.25, -6.25, 6.25), aspect='equal', cmap='viridis')
                truth = arrays['targets'][episode, 2 * obj:2 * obj + 2] * 224
                p = result['predictions'][episode, m, :, 2 * obj:2 * obj + 2] * 224
                ax.axhline(p[1, 1], color='#ef68c1', linewidth=1, alpha=.65)
                ax.plot(*p[0], marker='x', color='white', markersize=9, mew=2, linestyle='none')
                ax.plot(*p[1], marker='o', markerfacecolor='none', markeredgecolor='#ff70d3', markersize=10, mew=2, linestyle='none')
                ax.plot(*truth, marker='*', color='#ffde45', markeredgecolor='black', markersize=13, linestyle='none')
                global_rank = int(result['global_gt_rank_tie_gap_margin'][episode, m, obj, 0])
                row_rank = int(result['gt_y_row_rank_tie_gap_margin'][episode, m, obj, 0])
                ax.set_title(f'{method} | object {"AB"[obj]}\nGT-grid rank: {global_rank}/625; within GT-y row: {row_rank}/25', fontsize=10)
                ax.set_xlabel('Candidate dx (pixels)'); ax.set_ylabel('Candidate dy (pixels)')
                ax.set_xticks([-6, -3, 0, 3, 6]); ax.set_yticks([-6, -3, 0, 3, 6])
                fig.colorbar(im, ax=ax, label='Saved centered-cosine cost', shrink=.8)
        handles = [Line2D([], [], marker='x', color='black', linestyle='none', label='Original 2D minimum'),
                   Line2D([], [], marker='o', markerfacecolor='none', color='#da259e', linestyle='none', label='Privileged GT-y row minimum'),
                   Line2D([], [], marker='*', color='#dfb600', linestyle='none', label='Continuous ground truth')]
        fig.suptitle(f'Fixed prior case {pair}/{variant}: existing cost surfaces\nKnown Dev; privileged posthoc localization, not a learned-model benchmark', fontsize=15)
        fig.legend(handles=handles, loc='outside lower center', ncol=3)
        for extension in ('png', 'svg'):
            fig.savefig(out / f'case_{pair}_{variant}.{extension}', dpi=160)
        plt.close(fig)


def write_summary(path, report):
    lines = ['# V5.7：已冻结代价面的事后诊断', '',
             '输入为既有 256 个 world、512 个 Dev episode 的完整 625 点代价面。没有重新编码或拟合。', '',
             'GT-y/GT-x 使用真实标签，是特权定位控制；错配 GT-y 固定循环平移 37 个 world。'
             '以下 GT-y 比较只解释未提供的 dx；提供 oracle 轴后的 4D 指标不能当可部署性能。', '',
             '| Arm | 原 2D dx RMSE (px) | GT-y dx RMSE | y=0 dx RMSE | 错配 GT-y dx RMSE | 原−GT-y dx MSE, 95% CI (px²) |',
             '|---|---:|---:|---:|---:|---|']
    for method in METHODS:
        arm = report['arms'][method]; scores = arm['scores']
        effect = arm['min2d_minus_alternative_mse']['oracle_gt_y']['dx']
        lo, hi = effect['ci95_pixels_squared']
        values = [scores[name]['dx_rmse_pixels'] for name in ('min2d', 'oracle_gt_y', 'zero_y', 'mismatched_gt_y')]
        lines.append('| ' + method + ' | ' + ' | '.join(f'{value:.4f}' for value in values) +
                     f' | {effect["mean_pixels_squared"]:.4f} [{lo:.4f}, {hi:.4f}] |')
    lines += ['', '方向计数按 |真实 dx|≥1 px，wrong、zero 和合计 non-correct 分开；'
              '完整 per-axis 分母、GT 网格化误差、out-of-support、全局/行内 rank、方向修复及伤害见 report.json 和逐例 CSV。', '',
              f'错配 donor 与真实 y 落在同一网格行：{report["donor_y_row_match"]["same_count"]}/1024；这些预测应精确一致。'
              '报告另列不同网格行子集的 dx MSE 差，bootstrap 同时重采样每个 world 的误差和子集数量。', '',
              'MSE 使用原 normalized displacement；px²=MSE×224²。四轴 skill 是逐轴 '
              '1−MSE/mean((GT−冻结 FIT 均值)²) 后求均值，不使用 Dev 方差。', '',
              '95% CI 使用固定 seed 7817 的 10,000 次 whole-world bootstrap，同时保留两个 variant 和两个对象。'
              '区间条件于当前代价面和固定规则，多臂描述性比较未做多重校正。', '',
              '几何 oracle 下界仅适用于当前有限网格。GT-nearest-grid 的 cost/rank 仅描述这张固定代价面，'
              '不能证明 descriptor 无信息；oracle 收益也不是某模块造成的错误比例。没有新确认、Test 或 P0 授权。', '',
              '三张图固定选择既往 761/1、844/1、891/1，未按本轮效果选例。']
    Path(path).write_text('\n'.join(lines) + '\n')


def run(input_path, out, plan, publication):
    start = time.time()
    validate_publication(publication, plan)
    input_path, out = Path(input_path).resolve(), Path(out).resolve()
    if out == input_path.parent or input_path.parent in out.parents:
        raise ValueError('Output must be outside the immutable input result directory')
    out.mkdir(parents=True, exist_ok=False)
    initial_sources = source_identity()
    initial_plan, initial_publication = sha(plan), sha(publication)
    try:
        if sha(input_path) != INPUT_SHA256:
            raise ValueError('Pinned NPZ hash mismatch; input was not opened')
        with np.load(input_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        validate_arrays(arrays)
        result = diagnose(arrays['costs'], arrays['targets'])
        np.testing.assert_array_equal(result['predictions'][:, :, 0], arrays['predictions'],
                                      err_msg='Original min2d predictions did not replay exactly')
        draws = bootstrap_indices(256)
        report = make_report(arrays, result, draws)
        report['verification'] = {'input_sha256': INPUT_SHA256, 'min2d_all_predictions_exact': True,
                                  'replayed_scalar_predictions': int(arrays['predictions'].size),
                                  'costs_finite': True, 'no_cases_excluded': True}
        report['provenance'] = {'sources': initial_sources, 'plan_sha256': initial_plan,
                                'publication_sha256': initial_publication, 'input_path': str(input_path),
                                'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                                'command': sys.argv, 'blas_threads': 4}
        np.savez_compressed(out / 'predictions_and_diagnostics.npz', **result,
                            targets=arrays['targets'], methods=arrays['methods'], readouts=np.array(READOUTS),
                            pair_ids=arrays['pair_ids'], variants=arrays['variants'], frozen_fit_mean=arrays['frozen_fit_mean'],
                            skill_denominator_by_axis=np.mean((arrays['targets'] - arrays['frozen_fit_mean'][4:]) ** 2, axis=0),
                            rank_field_names=np.array(['rank', 'tie_count', 'gt_cost_minus_min', 'gt_cost_minus_best_other']))
        write_case_csv(out / 'per_case.csv', arrays, result)
        make_figures(out, arrays, result)
        if (source_identity() != initial_sources or sha(plan) != initial_plan or
                sha(publication) != initial_publication or sha(input_path) != INPUT_SHA256):
            raise ValueError('Input/source/design/publication changed during run')
        report['verification']['sources_and_inputs_unchanged'] = True
        report['wall_seconds'] = time.time() - start
        save_json(out / 'report.json', report)
        write_summary(out / 'README.md', report)
        files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
        save_json(out / 'output_manifest.json', {'schema': 'v5.7-derived-output-manifest-1', 'files': files,
                                                'input_sha256': INPUT_SHA256, 'sources': initial_sources})
        return {'status': 'complete', 'out': str(out), 'wall_seconds': report['wall_seconds'],
                'min2d_all_predictions_exact': True, 'source_identity_unchanged': True}
    except Exception:
        save_json(out / 'failure.json', {'status': 'failed_preserved', 'traceback': traceback.format_exc(),
                                        'sources_before': initial_sources, 'wall_seconds': time.time() - start})
        raise


def self_test():
    # Only synthetic cost surfaces; no scientific cache or renderer is accessed.
    flat = np.zeros(625)
    assert choose_minimum(flat) == 312
    assert tuple(OFFSETS[choose_minimum(flat, np.arange(25) + 25 * 16)]) == (0., 2.)
    tied = np.ones(625)
    a = np.flatnonzero(np.all(OFFSETS == [-1., 0.], axis=1))[0]
    b = np.flatnonzero(np.all(OFFSETS == [1., 0.], axis=1))[0]
    tied[[a, b]] = 0
    assert choose_minimum(tied) == a
    assert GRID_AXIS[nearest_axis(.25)] == 0 and GRID_AXIS[nearest_axis(-.25)] == 0
    assert GRID_AXIS[nearest_axis(10)] == 6 and GRID_AXIS[nearest_axis(-10)] == -6
    truth_pixels = np.array([[2., 1., -2., -1.], [1., 2., -1., -2.],
                             [3., 0., -3., 0.], [2., -2., -2., 2.],
                             [7., 1., -7., -1.], [1., 3., -1., -3.],
                             [4., 4., -4., -4.], [2., 5., -2., -5.]])
    target = truth_pixels / 224
    surfaces = np.empty((8, 6, 2, 625))
    for i in range(8):
        for obj in (0, 1):
            desired = truth_pixels[i, obj * 2:obj * 2 + 2]
            surfaces[i, :, obj] = np.sum((OFFSETS - desired) ** 2, axis=1)
    # A competing wrong-direction minimum outside the true-y row is deliberate.
    bad = np.flatnonzero(np.all(OFFSETS == [-2., -4.], axis=1))[0]
    surfaces[0, :, 0, bad] = -1
    result = diagnose(surfaces, target)
    np.testing.assert_array_equal(result['predictions'][0, :, 0, :2] * 224, np.tile([-2., -4.], (6, 1)))
    np.testing.assert_array_equal(result['predictions'][0, :, 1, :2] * 224, np.tile([2., 1.], (6, 1)))
    assert result['global_gt_rank_tie_gap_margin'][0, 0, 0, 0] == 2
    assert result['gt_y_row_rank_tie_gap_margin'][0, 0, 0, 0] == 1
    assert result['out_of_support_axes'].sum() == 2
    np.testing.assert_array_equal(result['donor_episode_indices'], [2, 3, 4, 5, 6, 7, 0, 1])
    draws = np.array([[0, 0, 1], [1, 2, 2], [0, 1, 2]])
    effect = np.array([1., 4., 10.])
    ci = mean_interval(effect, draws)
    np.testing.assert_array_equal(ci['ci95'], np.quantile([2., 8., 5.], [.025, .975]))
    ratio = ratio_interval(np.array([1., 1., 0.]), np.array([1., 2., 0.]), draws)
    assert ratio['rate'] == 2 / 3
    np.testing.assert_allclose(ratio['ci95'], np.quantile([3 / 4, 1 / 2, 2 / 3], [.025, .975]), atol=0, rtol=0)
    return {'status': 'passed', 'software_only': True, 'tests': [
        'V56 cost tie and row subset', 'nearest-axis boundary and midpoint',
        'wrong-minimum rescue and rank', 'fixed donor preserves variant/object',
        'grid lower bound and out-of-support', 'whole-world mean and ratio bootstrap']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    commands.add_parser('self-test')
    freeze = commands.add_parser('freeze-design'); freeze.add_argument('--out', type=Path, required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--input', type=Path, required=True)
    execute.add_argument('--out', type=Path, required=True)
    execute.add_argument('--plan', type=Path, required=True)
    execute.add_argument('--publication', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'self-test':
        result = self_test()
    elif args.phase == 'freeze-design':
        result = {'design': DESIGN, 'sources': source_identity(),
                  'software_verification': self_test(), 'created_unix': time.time()}
        save_json(args.out, result)
    else:
        result = run(args.input, args.out, args.plan, args.publication)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
