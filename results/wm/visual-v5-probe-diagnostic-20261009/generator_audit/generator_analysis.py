#!/usr/bin/env python3
"""Exploratory audit of only the existing 64 Dev pairs; no Train/Test generation."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-12 and b.std() > 1e-12 else None


def summarize(rows):
    target = np.array([r['target'] for r in rows])
    pred = np.array([r['prediction'] for r in rows])
    if not len(rows):
        return {'n': 0}
    return dict(n=len(rows), sign_errors=int(np.sum(target * pred < 0)),
                sign_error_rate=float(np.mean(target * pred < 0)), mse=float(np.mean((pred-target)**2)),
                target_mean=float(target.mean()), prediction_mean=float(pred.mean()),
                target_position_t7_correlation=corr(target, [r['position_t7'] for r in rows]),
                prediction_position_t7_correlation=corr(pred, [r['position_t7'] for r in rows]))


def color_center(rgb, obj):
    red, blue = (235, 60) if obj == 0 else (60, 235)
    mask = (rgb[:, :, 0] == red) & (rgb[:, :, 2] == blue)
    yy, xx = np.where(mask)
    if len(xx) < 50:
        raise AssertionError('Target fill mask missing or too small')
    # Ellipse bounding-box center: invariant to the interior white markings.
    return np.array([(xx.min()+xx.max())/2, (yy.min()+yy.max())/2]), int(rgb[yy[0], xx[0], 1])


def find_archive():
    for parent in Path(__file__).resolve().parents:
        candidate=parent/'results/wm/visual-v5-dgx-20261009'
        if (candidate/'source/pmr/visual_memory/data.py').is_file():
            return candidate
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=find_archive(), help='Original frozen V5 archive; auto-located within this repository')
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    if a.archive is None:p.error('Supply --archive pointing to the original frozen V5 archive')
    a.out.mkdir(parents=True, exist_ok=True)
    source = a.archive/'source/pmr/visual_memory/data.py'
    spec = importlib.util.spec_from_file_location('v5_archived_dev_generator', source)
    generator = importlib.util.module_from_spec(spec); sys.modules[spec.name] = generator; spec.loader.exec_module(generator)
    prediction_path = a.archive/'run/feature_diagnostics/dev_predictions.csv'
    original = list(csv.DictReader(prediction_path.open()))
    expected_ids = {f'dev-{pair:06d}-{variant}' for pair in range(64) for variant in (0, 1)}
    assert {r['episode_id'] for r in original} == expected_ids
    pred = {(r['mode'], r['episode_id'], int(r['dimension_index'])): r for r in original}
    evidence = []; target_max_error = 0.; position_pixel_errors = []; labels = {}
    for pair in range(64):
        rng = np.random.default_rng(generator.seed_for('dev', pair))
        initial = rng.uniform([.2, .23], [.8, .32], (2, 2)); initial[1, 1] += .42
        initial_velocity = rng.uniform(.008, .018, (2, 2)) * rng.choice([-1, 1], (2, 2))
        for variant in (0, 1):
            # Exactly the original Dev episodes, with no changed generation args.
            ep = generator.episode('dev', pair, variant)
            assert np.array_equal(ep.times[6:8], [6, 7]) and np.all(ep.shutters[6:8] == 1)
            for mode in ('raw', 'pooled'):
                for dim in range(8):
                    target_max_error = max(target_max_error, abs(float(pred[(mode, ep.metadata['episode_id'], dim)]['target']) - float(ep.probe_state[dim])))
            for obj in range(2):
                c6, color = color_center(ep.rgb[6], obj); c7, color7 = color_center(ep.rgb[7], obj)
                assert color == color7
                for axis in range(2):
                    dim = 4 + obj * 2 + axis; posdim = obj * 2 + axis
                    dimension = pred[('pooled', ep.metadata['episode_id'], dim)]['dimension']; labels[dim] = dimension
                    target = float(ep.probe_state[dim]); p7 = float(ep.probe_state[posdim]); p6 = p7-target
                    low, high = (.1, .9) if axis == 0 else ((.15, .4) if obj == 0 else (.6, .85))
                    velocity = float(initial_velocity[obj, axis]) * (1 if variant == 0 else -1)
                    raw6 = float(initial[obj, axis]) + 6*velocity; raw7 = float(initial[obj, axis]) + 7*velocity
                    crossing = int(np.floor((raw6-low)/(high-low))) != int(np.floor((raw7-low)/(high-low)))
                    pixel_delta = float(c7[axis]-c6[axis]); pixel_errors = [abs(c6[axis]-p6*224), abs(c7[axis]-p7*224)]
                    position_pixel_errors.extend(pixel_errors)
                    boundary_distance = min(p6-low, high-p6, p7-low, high-p7)
                    for mode in ('raw', 'pooled'):
                        row = pred[(mode, ep.metadata['episode_id'], dim)]
                        evidence.append(dict(mode=mode, episode_id=ep.metadata['episode_id'], pair=pair, variant=variant,
                            dimension=dimension, dimension_index=dim, object='AB'[obj], axis='xy'[axis],
                            family=ep.metadata['family'], gap=ep.metadata['gap'], writes=ep.metadata['interference_writes'],
                            target=target, prediction=float(row['prediction']), initial_position=float(initial[obj, axis]),
                            initial_velocity=velocity, position_t6=p6, position_t7=p7, color_green=color,
                            pixel_displacement=pixel_delta, target_displacement_pixels=224*target,
                            pixel_delta_error=pixel_delta-224*target, reflected_during_cue=crossing,
                            reflected_before_or_during=(raw6 < low or raw6 > high or raw7 < low or raw7 > high),
                            boundary_distance=boundary_distance, near_reflection_boundary=boundary_distance <= .02,
                            pooling_cell_center_crossing=int(p6*4) != int(p7*4)))
    assert target_max_error == 0
    quantitative = {}
    for mode in ('raw', 'pooled'):
        rows = [r for r in evidence if r['mode'] == mode]
        sections = {'all': summarize(rows), 'dimension': {}, 'axis': {}, 'variant': {}, 'reflection': {},
                    'near_boundary': {}, 'pool_center_crossing': {}, 'family': {}, 'gap': {}, 'writes': {}, 'position_quartiles': {}}
        for grouping, key, values in [('dimension','dimension_index',range(4,8)),('axis','axis','xy'),
                ('variant','variant',(0,1)),('reflection','reflected_during_cue',(False,True)),
                ('near_boundary','near_reflection_boundary',(False,True)),('pool_center_crossing','pooling_cell_center_crossing',(False,True)),
                ('family','family',('hold','interference','update')),('gap','gap',(8,16,32)),('writes','writes',(0,8,32))]:
            for value in values:
                sections[grouping][str(value)] = summarize([r for r in rows if r[key] == value])
        for dim in range(4,8):
            selected = [r for r in rows if r['dimension_index']==dim]
            positions = np.array([r['position_t7'] for r in selected]); edges = np.quantile(positions,[0,.25,.5,.75,1])
            bins = np.minimum(np.searchsorted(edges[1:-1],positions,side='right'),3)
            sections['position_quartiles'][str(dim)] = [dict(position_min=float(edges[i]),position_max=float(edges[i+1]),
                **summarize([r for r,b in zip(selected,bins) if b==i])) for i in range(4)]
        quantitative[mode] = sections
    paired = {}
    for mode in ('raw','pooled'):
        paired[mode]={}
        for dim in range(4,8):
            groups = [[next(r for r in evidence if r['mode']==mode and r['dimension_index']==dim and r['pair']==pair and r['variant']==v) for v in (0,1)] for pair in range(64)]
            target=np.array([[r['target'] for r in g] for g in groups]); prediction=np.array([[r['prediction'] for r in g] for g in groups]); errors=prediction-target
            even=errors.mean(axis=1); odd=(errors[:,0]-errors[:,1])/2
            opposite=target[:,0]*target[:,1] < 0; same_pred=prediction[:,0]*prediction[:,1] > 0
            target_even=target.mean(axis=1); pred_even=prediction.mean(axis=1)
            total=float(np.mean(even**2+odd**2)); initial=np.array([g[0]['initial_position'] for g in groups]); color=np.array([g[0]['color_green'] for g in groups])
            paired[mode][str(dim)]=dict(pairs=64, target_opposite_sign_pairs=int(opposite.sum()),
                same_prediction_sign_among_opposite_target_pairs=int((opposite&same_pred).sum()),
                opposite_target_pair_response_sign_accuracy=float(np.mean((prediction[:,0]-prediction[:,1])[opposite]*(target[:,0]-target[:,1])[opposite]>0)),
                target_even_rms=float(np.sqrt(np.mean(target_even**2))),prediction_even_rms=float(np.sqrt(np.mean(pred_even**2))),
                error_even_energy_fraction=float(np.mean(even**2)/total),error_odd_energy_fraction=float(np.mean(odd**2)/total),
                prediction_even_initial_position_correlation=corr(pred_even,initial),
                prediction_even_color_green_correlation=corr(pred_even,color))
    unique=[r for r in evidence if r['mode']=='pooled']; pixel=np.array([r['pixel_displacement'] for r in unique]); true=np.array([r['target_displacement_pixels'] for r in unique])
    pixel_summary=dict(components=len(unique), position_center_max_absolute_error_pixels=max(position_pixel_errors),
        displacement_max_absolute_error_pixels=float(np.max(np.abs(pixel-true))), displacement_rmse_pixels=float(np.sqrt(np.mean((pixel-true)**2))),
        pixel_direction_opposite_to_true=int(np.sum(pixel*true<0)), pixel_direction_zero=int(np.sum(pixel==0)),
        prediction_sign_errors=int(sum(r['target']*r['prediction']<0 for r in unique)),
        prediction_sign_errors_with_nonzero_pixel_motion=int(sum(r['target']*r['prediction']<0 and r['pixel_displacement']!=0 for r in unique)),
        near_zero_pixel_motion_cases=[r for r in unique if r['pixel_displacement']==0])
    worst=sorted(unique,key=lambda r:(-(r['prediction']-r['target'])**2,r['episode_id'],r['dimension_index']))[:12]
    bad_case_path=a.archive/'analysis/bad_cases.csv'
    bad_case_rows=[row for row in csv.DictReader(bad_case_path.open()) if row['is_pair_maximum_error']=='True']
    selected_bad_cases=[]
    for row in sorted(bad_case_rows,key=lambda row:int(row['pair_rank'])):
        item=next(r for r in unique if r['episode_id']==row['episode_id'] and r['dimension_index']==int(row['dimension_index']))
        selected_bad_cases.append(dict(rank=int(row['pair_rank']),**item))
    result=dict(status='exploratory_existing_dev_only', source_sha256=sha(source), prediction_csv_sha256=sha(prediction_path),
        split_scope=dict(split='dev',pair_ids=[0,63],pairs=64,paired_records=128,train_generated=False,test_generated=False),
        bad_case_csv_sha256=sha(bad_case_path),
        target_regeneration_max_absolute_difference=target_max_error, dimensions=labels, finite_pixel_check=pixel_summary,
        summary=quantitative, paired_decomposition=paired, worst_pooled_components=worst, archived_chosen_bad_cases=selected_bad_cases,
        generator_facts=dict(initial_x_range=[.2,.8],initial_A_y_range=[.23,.32],initial_B_y_range=[.65,.74],
            reflected_x_bounds=[.1,.9],reflected_A_y_bounds=[.15,.4],reflected_B_y_bounds=[.6,.85],
            observed_delta_is_reflected_position7_minus_position6=True,
            variants_reverse_initial_velocity_but_do_not_guarantee_opposite_observed_delta=True,
            cue_times_precede_every_update_and_distractor_write=True),
        limitations=['All findings use the already inspected Dev set and are exploratory.',
            'Associations do not establish a causal pooling failure or a general model deficit.',
            'Finite-pixel centers use privileged object colors for a diagnostic oracle; not a permitted scientific predictor input.',
            'No new gate, encoder run, training run, or Test evaluation was performed.'])
    (a.out/'generator_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Existing-Dev generator and motion-direction audit','',
        'This is exploratory analysis of the original 64 Dev pairs (128 records), using archived predictions and the frozen archived generator. No Train or Test episode is generated, no model is trained, and no original result or threshold is changed.','',
        f'All 2,048 CSV targets exactly match regenerated `probe_state` (maximum difference {target_max_error:g}). Both objects are visible at t=6 and t=7.','',
        '## Direction errors and absolute position','',
        '| Dimension | Raw wrong sign | Pooled wrong sign | corr(true delta, position7) | corr(pooled prediction, position7) |',
        '|---|---:|---:|---:|---:|']
    for dim in range(4,8):
        rr=quantitative['raw']['dimension'][str(dim)]; pp=quantitative['pooled']['dimension'][str(dim)]
        lines.append(f"| {labels[dim]} | {rr['sign_errors']}/{rr['n']} | {pp['sign_errors']}/{pp['n']} | {pp['target_position_t7_correlation']:.3f} | {pp['prediction_position_t7_correlation']:.3f} |")
    lines+=['',f"Across pooled predictions, x has {quantitative['pooled']['axis']['x']['sign_errors']}/256 wrong signs, y has {quantitative['pooled']['axis']['y']['sign_errors']}/256. These are descriptive paired observations, not independent significance tests.",'',
        'The generator samples each initial y coordinate within width 0.09, versus x within width 0.60; the reflecting lanes have width 0.25 versus x width 0.80. Absolute terminal position can therefore carry a motion prior, especially on y. Higher y predictability alone does not prove better temporal-motion extraction.','',
        f"Within an unreflected pair, p7 = p0 ± 7v: the variants share p0 but have opposite initial velocity. Thus paired generation itself creates a correlation between terminal position and direction; it does not balance terminal position across directions. The paired mean pooled x prediction correlates with initial x by {paired['pooled']['4']['prediction_even_initial_position_correlation']:.3f} for A and {paired['pooled']['6']['prediction_even_initial_position_correlation']:.3f} for B in these 64 pairs. This is compatible with a position prior and requires an input/order ablation before attributing the error to that prior.",'', 
        '## Finite rendering and boundaries','',
        '| Archived chosen case | True displacement (pixels) | Rendered center displacement (pixels) | Reflection during cue |',
        '|---|---:|---:|---|']
    for row in selected_bad_cases:
        lines.append(f"| {row['episode_id']} {row['dimension']} | {row['target_displacement_pixels']:+.4f} | {row['pixel_displacement']:+.1f} | {row['reflected_during_cue']} |")
    lines+=['', 
        f"A diagnostic color-mask bounding-box center yields {pixel_summary['pixel_direction_opposite_to_true']}/512 directions opposite to true displacement, and {pixel_summary['pixel_direction_zero']}/512 quantized zero displacements. Maximum center error is {pixel_summary['position_center_max_absolute_error_pixels']:.4f} pixels; displacement RMSE is {pixel_summary['displacement_rmse_pixels']:.4f} pixels.",
        f"Of {pixel_summary['prediction_sign_errors']} pooled sign errors, {pixel_summary['prediction_sign_errors_with_nonzero_pixel_motion']} occur when pixel motion is nonzero. Privileged color segmentation is only a label/rendering diagnostic, not a proposed scientific model input.",'',
        '| Group | Pooled wrong sign | Rate |','|---|---:|---:|']
    for title,key in [('Reflection during t6–t7','reflection'),('Within 0.02 of reflecting boundary','near_boundary'),('4×4 cell-center crossing','pool_center_crossing')]:
        for state in ['False','True']:
            item=quantitative['pooled'][key][state];lines.append(f"| {title}: {state} | {item['sign_errors']}/{item['n']} | {100*item['sign_error_rate']:.1f}% |")
    lines+=['','These subgroup associations are observational and confounded by axis and position. Boundary proximity is not a diagnosis of cause. In the frozen generator, both cue frames precede all update and extra-observation events, so differences grouped by family/gap/write count cannot by themselves show interference or update failure.','',
        f"Pooled variant-0 and variant-1 sign errors are {quantitative['pooled']['variant']['0']['sign_errors']}/256 and {quantitative['pooled']['variant']['1']['sign_errors']}/256. Absolute-position quartiles give:", '', 
        '| Dimension | Lowest position quartile | Second | Third | Highest |', '|---|---:|---:|---:|---:|']
    for dim in range(4,8):
        bins=quantitative['pooled']['position_quartiles'][str(dim)]
        counts=' | '.join(f"{item['sign_errors']}/{item['n']}" for item in bins)
        lines.append(f"| {labels[dim]} | {counts} |")
    lines+=['','Quartile boundaries and full family/gap/write-count summaries are in JSON. These were inspected after the failed gate and are exploratory.','', 
        '## Paired variants','',
        '| Dimension | Opposite true signs | Same predicted sign within those pairs | Pair-common fraction of squared error |','|---|---:|---:|---:|']
    for dim in range(4,8):
        item=paired['pooled'][str(dim)];lines.append(f"| {labels[dim]} | {item['target_opposite_sign_pairs']}/64 | {item['same_prediction_sign_among_opposite_target_pairs']} | {100*item['error_even_energy_fraction']:.1f}% |")
    lines+=['','For paired errors e0/e1, the common part is (e0+e1)/2 and the antisymmetric part is (e0−e1)/2; their energies sum exactly to mean squared error. A large common component is compatible with a pair-specific offset or position/appearance shortcut, but does not identify its source. Reflection means velocity reversal need not reverse the observed displacement sign.','',
        '## Strong falsifiers before claiming a fix','',
        '1. **Rendering/target mismatch hypothesis:** already challenged by exact target regeneration and finite-pixel directions. The worst sign errors remain directly visible; a global direction-label inversion or absent subpixel motion cannot explain all failures.',
        '2. **The probe reads temporal direction:** with existing cached Train/Dev features, fit matched-budget probes using frame difference and pairwise frame order; evaluate a Dev cue-order swap and its negated displacement target. A temporal-direction readout should reverse displacement predictions under the swap. Keep the archived gate unchanged and report these as exploratory ablations.',
        '3. **Absolute-position shortcut hypothesis:** compare a current-frame-only control against a temporal-difference readout, and test position-balanced positive/negative directions in a future freshly generated, separate confirmation split. If temporal-order-sensitive advantage persists after balancing positions and excluding reflections, that weakens the shortcut explanation. Do not use reserved Test for iterative debugging.',
        '4. **Pooling specifically causes the failure:** apply identical centering/scaling, ridge budget and order-sensitive features to raw versus pooled caches. Recovery by both after a probe change would weaken a pooling-specific explanation; a stable raw–pooled gap on fresh confirmation would support further pooling investigation.','',
        f"Archived generator SHA-256: `{sha(source)}`. Prediction CSV SHA-256: `{sha(prediction_path)}`.",'',
        'From the repository root, reproduce with `python3 results/wm/visual-v5-probe-diagnostic-20261009/generator_audit/generator_analysis.py`. The script locates the original frozen archive automatically; `--archive PATH` and `--out PATH` override locations. Full counts and case evidence: `generator_analysis.json`.' ]
    (a.out/'generator_analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(target_match=target_max_error,pixel=pixel_summary,axis=quantitative['pooled']['axis'],paired=paired['pooled']),indent=2))


if __name__=='__main__':
    main()
