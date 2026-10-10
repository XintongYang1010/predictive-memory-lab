#!/usr/bin/env python3
"""Posthoc CPU audit of saved V5.2 arrays; no scorer imports or model fitting."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

CONDITIONS = ('original', 'static_t6', 'static_t7', 'swap_nuisance')
AXES = ('A_x', 'A_y', 'B_x', 'B_y', 'A_dx', 'A_dy', 'B_dx', 'B_dy')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, v):
    with Path(p).open('x') as f:
        json.dump(v, f, indent=2, allow_nan=False)
        f.write('\n')


def rows(p, data):
    with Path(p).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def compute(pred, target, mean):
    """Explicit sample-loop percentile bootstrap, independent of scoring code."""
    target = target.astype(np.float64)
    err = np.empty((256, 4, 8))
    den = np.empty((256, 8))
    for i in range(256):
        den[i] = ((target[i, 0]-mean)**2 + (target[i, 1]-mean)**2)/2
        for c in range(4):
            err[i, c] = ((pred[i, 0, c]-target[i, 0])**2 +
                         (pred[i, 1, c]-target[i, 1])**2)/2
    dx = (err[:, :, 4]+err[:, :, 6])/2
    primary = dx[:, 0]-(dx[:, 1]+dx[:, 2])/2
    balanced = (dx[:, 0]+dx[:, 3]-dx[:, 1]-dx[:, 2])/2
    motion_delta = err[:, 0, 4:]-(err[:, 1, 4:]+err[:, 2, 4:])/2
    rng = np.random.default_rng(7810)
    boot = np.empty((10000, 9))
    for b in range(10000):
        index = rng.integers(0, 256, 256)
        denom = den[index, 4:].mean(0)
        boot[b, :3] = (primary[index].mean(), balanced[index].mean(),
                       (motion_delta[index].mean(0)/denom).mean())
        for c in range(1, 4):
            boot[b, 2+c] = (dx[index, 0]-dx[index, c]).mean()
            boot[b, 5+c] = ((err[index, 0, 4:]-err[index, c, 4:]).mean(0)/denom).mean()
    ci = np.quantile(boot, [.025, .975], axis=0)
    mse = err.mean(0)
    skill = 1-mse/den.mean(0)
    score = dict(primary_dx_mse_reduction=float(primary.mean()), primary_dx_paired_95ci=ci[:, 0].tolist(),
                 secondary_balanced_dx_mse_reduction=float(balanced.mean()),
                 secondary_balanced_dx_paired_95ci=ci[:, 1].tolist(),
                 secondary_static_average_motion_nmse_reduction=float((motion_delta.mean(0)/den[:, 4:].mean(0)).mean()),
                 secondary_motion_paired_95ci=ci[:, 2].tolist(), conditions={}, individual_comparisons={})
    for c, name in enumerate(CONDITIONS):
        score['conditions'][name] = dict(per_dimension_mse=mse[c].tolist(), per_dimension_skill=skill[c].tolist(),
                                        position_skill=float(skill[c, :4].mean()), motion_skill=float(skill[c, 4:].mean()),
                                        motion_nmse=float(1-skill[c, 4:].mean()), dx_mse=float(dx[:, c].mean()))
        if c:
            score['individual_comparisons'][name] = dict(
                nmse_reduction=float(((err[:, 0, 4:]-err[:, c, 4:]).mean(0)/den[:, 4:].mean(0)).mean()),
                dx_mse_reduction=float((dx[:, 0]-dx[:, c]).mean()),
                dx_mse_reduction_descriptive_95ci=ci[:, 2+c].tolist(),
                descriptive_paired_95ci=ci[:, 5+c].tolist())
    return score, err, dx


def compare(expected, actual, prefix=''):
    checks = []
    for key, value in expected.items():
        if isinstance(value, dict):
            checks += compare(value, actual[key], prefix+key+'/')
        else:
            a, b = np.asarray(value), np.asarray(actual[key])
            if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
                raise AssertionError((prefix+key, value, actual[key]))
            checks.append(dict(metric=prefix+key, max_absolute_difference=float(np.max(np.abs(a-b)))))
    return checks


def main(a):
    start = time.monotonic()
    scene = a.scene.resolve()
    manifest = json.loads((scene/'archive_manifest.json').read_text())
    registered = {}
    def reg(name):
        p = scene/name
        expected = manifest['files'][name]
        assert expected == dict(bytes=p.stat().st_size, sha256=sha(p)), name
        registered[name] = expected
        return p
    report = json.loads(reg('report/report.json').read_text())
    assert report['status'] == 'diagnostic_complete' and not report['software_only']
    assert report['test_access'] is False and report['p0_authorized'] is False
    assert tuple(report['conditions']) == CONDITIONS
    with np.load(reg('report/predictions.npz'), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    with np.load(reg('reader/frozen_model.npz'), allow_pickle=False) as z:
        means = {name: z[name+'__target_mean'].copy() for name in ('concat', 't6', 't7')}
    with np.load(reg('prior_bridge/predictions.npz'), allow_pickle=False) as z:
        prior = {k: z[k] for k in z.files}
    frozen = json.loads(reg('reader/freeze_receipt.json').read_text())
    receipt = json.loads(reg('counterfactual_cache/receipt.json').read_text())
    reg('counterfactual_cache/cache.npz')
    accounting = json.loads(reg('final_accounting.json').read_text())
    assert report['predictions_sha256'] == registered['report/predictions.npz']['sha256']
    assert report['frozen_reader_sha256'] == registered['reader/freeze_receipt.json']['sha256']
    assert report['cache_sha256'] == registered['counterfactual_cache/cache.npz']['sha256'] == receipt['cache_sha256']
    assert receipt['frozen_reader_sha256'] == report['frozen_reader_sha256']
    assert np.array_equal(data['targets'], prior['targets'])
    assert np.array_equal(data['pair_ids'], np.arange(128, 384))
    y = data['targets'].astype(np.float64)
    assert y.shape == (256, 2, 8) and np.isfinite(y).all()
    eta = np.array([(int((.5+.4*np.sin(7*2.3+int(i)))*224)-int((.5+.4*np.sin(6*2.3+int(i)))*224))/224
                    for i in data['pair_ids']])
    checks, scores, parity, responses = [], {}, {}, {}
    for reader in ('concat', 't6', 't7'):
        p = data[reader]
        assert p.shape == (256, 2, 4, 8) and np.isfinite(p).all()
        scores[reader], err, dx = compute(p, y, means[reader])
        checks += compare(scores[reader], report['scores'][reader], reader+'/')
        parity[reader] = float(np.max(np.abs(p[:, :, 0]-prior[reader])))
        assert parity[reader] == 0
        response = (p[:, 0, 0, 4:]+p[:, 1, 0, 4:]-p[:, 0, 3, 4:]-p[:, 1, 3, 4:])/4
        original = (p[:, :, 0, 4:]-y[:, :, 4:]).mean(1)
        swapped = (p[:, :, 3, 4:]-y[:, :, 4:]).mean(1)
        response_rows = []
        responses[reader] = {}
        for d, name in enumerate(AXES[4:]):
            r = response[:, d]
            eligible = (eta != 0) & (r != 0)
            result = dict(through_origin_slope=float(np.sum(eta*r)/np.sum(eta*eta)),
                signed_alignment_fraction=float((eta[eligible]*r[eligible]>0).mean()) if eligible.any() else None,
                aligned_count=int((eta*r>0).sum()), opposed_count=int((eta*r<0).sum()),
                zero_nuisance_count=int((eta==0).sum()), zero_response_count=int((r==0).sum()),
                residual_sign_reversal_count=int((original[:, d]*swapped[:, d]<0).sum()),
                residual_sign_tie_count=int((original[:, d]*swapped[:, d]==0).sum()), total_pairs=256)
            checks += compare(result, report['nuisance_swap_response'][reader][name], reader+'/response/'+name+'/')
            responses[reader][name] = result
        if reader == 'concat':
            primary_err, primary_dx = err, dx
            for i, pair in enumerate(data['pair_ids']):
                row = dict(pair_id=int(pair), original_yellow_dx=float(eta[i]))
                for d, name in enumerate(AXES[4:]):
                    row[name+'_half_original_minus_swap_prediction'] = float(response[i, d])
                    row[name+'_original_common_residual'] = float(original[i, d])
                    row[name+'_swap_common_residual'] = float(swapped[i, d])
                response_rows.append(row)
            rows(a.out/'per_world_swap_response.csv', response_rows)
    reuse = [np.max(np.abs(data['t6'][:, :, 0]-data['t6'][:, :, 1])),
             np.max(np.abs(data['t6'][:, :, 2]-data['t6'][:, :, 3])),
             np.max(np.abs(data['t7'][:, :, 0]-data['t7'][:, :, 2])),
             np.max(np.abs(data['t7'][:, :, 1]-data['t7'][:, :, 3]))]
    assert max(reuse) <= 1e-10
    csv_rows = list(csv.DictReader(reg('report/per_case.csv').open()))
    assert len(csv_rows) == 2048
    observed = set()
    for row in csv_rows:
        i, v, c = int(row['pair'])-128, int(row['variant']), CONDITIONS.index(row['condition'])
        assert (i, v, c) not in observed
        observed.add((i, v, c))
        assert float(row['nuisance_original_dx']) == eta[i]
        for d, name in enumerate(AXES[4:], 4):
            assert float(row[name+'_prediction']) == data['concat'][i, v, c, d]
            assert float(row[name+'_target']) == y[i, v, d]
    descriptors = {int(r['group_id']): r for r in csv.DictReader(a.descriptors.open()) if r['role']=='bridge'}
    reflections = {int(r['pair_id']): r['any_cue_reflection']=='True' for r in csv.DictReader(a.reflections.open())}
    detail, axes = [], []
    p = data['concat']
    for i, pair in enumerate(data['pair_ids']):
        orig_motion = float(primary_err[i, 0, 4:].mean())
        static_motion = float(primary_err[i, 1:3, 4:].mean())
        row = dict(pair_id=int(pair), yellow_dx=float(eta[i]), original_motion_mse=orig_motion,
                   static_average_motion_mse=static_motion, original_dx_mse=float(primary_dx[i, 0]),
                   static_average_dx_mse=float(primary_dx[i, 1:3].mean()),
                   static_t6_dx_mse=float(primary_dx[i, 1]), static_t7_dx_mse=float(primary_dx[i, 2]),
                   swap_dx_mse=float(primary_dx[i, 3]),
                   any_cue_reflection=reflections[int(pair)])
        for key in ('position_outside_fit_box_max', 'x_outside_fit_box_max', 'y_outside_fit_box_max',
                    'linear_nn_distance_max', 'selected_kernel_nn_distance_max'):
            row[key] = descriptors[int(pair)][key]
        detail.append(row)
    detail.sort(key=lambda r: (-r['static_average_motion_mse'], r['pair_id']))
    rows(a.out/'worlds_ranked_by_remaining_static_motion_error.csv', detail)
    for c, cond in enumerate(CONDITIONS):
        r = p[:, :, c]-y
        for d in range(4, 8):
            mse = float((r[:, :, d]**2).mean())
            common = r[:, :, d].mean(1)
            contrast = (r[:, 0, d]-r[:, 1, d])/2
            assert np.isclose(mse, np.mean(common**2+contrast**2), rtol=1e-13)
            axes.append(dict(condition=cond, axis=AXES[d], mse=mse,
                skill=scores['concat']['conditions'][cond]['per_dimension_skill'][d],
                direction_error_rate=float((np.sign(p[:, :, c, d])!=np.sign(y[:, :, d])).mean()),
                amplitude_ratio=float(np.abs(p[:, :, c, d]).mean()/np.abs(y[:, :, d]).mean()),
                common_mse_fraction=float(np.mean(common**2)/mse),
                residual_correlation_original_yellow_dx=float(np.corrcoef(common, eta)[0, 1])))
    rows(a.out/'remaining_error_per_axis.csv', axes)
    improvement = primary_dx[:, 0]-primary_dx[:, 1:3].mean(1)
    summary = dict(schema='v5.2.independent-posthoc-audit.1', test_access=False, refit=False, source_edits=False,
        original_parity_max_abs=parity, singleframe_reuse_max_abs=list(map(float, reuse)),
        checked_metric_leaves=len(checks), max_metric_absolute_difference=max(r['max_absolute_difference'] for r in checks),
        csv_rows_verified=len(csv_rows), bootstrap=dict(seed=7810, replicates=10000, method='explicit one-replicate loop'),
        scores=scores, nuisance_response=responses, remaining_errors=axes,
        world_counts=dict(static_average_dx_improved=int((improvement>0).sum()),
                          static_average_dx_worsened=int((improvement<0).sum()),
                          both_static_anchors_improved=int(np.all(primary_dx[:, 1:3]<primary_dx[:, :1], axis=1).sum()),
                          both_static_anchors_worsened=int(np.all(primary_dx[:, 1:3]>primary_dx[:, :1], axis=1).sum())),
        worst_remaining_static_worlds=detail[:10], accounting=accounting,
        inputs=registered, auxiliary_inputs={str(a.descriptors):sha(a.descriptors), str(a.reflections):sha(a.reflections)},
        wall_seconds=time.monotonic()-start,
        interpretation='Posthoc known scenes, conditional on one frozen pipeline; no new gate or generalization claim')
    for name, entry in registered.items():
        assert sha(scene/name) == entry['sha256'], 'Input changed: '+name
    rows(a.out/'metric_comparison.csv', checks)
    dump(a.out/'audit_summary.json', summary)
    dump(a.out/'recomputed_scores.json', scores)
    print(json.dumps(dict(primary=scores['concat'], counts=summary['world_counts'],
                         max_difference=summary['max_metric_absolute_difference'],
                         worst_worlds=[r['pair_id'] for r in detail[:10]], wall_seconds=summary['wall_seconds']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'out', 'descriptors', 'reflections'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
