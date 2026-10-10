#!/usr/bin/env python3
"""Posthoc CPU-only frozen FIT replay, allowed only after V5.4 evaluation completes.

No fitting, hyperparameter selection, encoder loading, RGB generation, or fresh
data access. This helper reads the existing model's union FIT basis and labels.
Optional official reports add descriptive gaps; they never change any pass gate.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
DIMENSIONS = ['A_x','A_y','B_x','B_y','A_dx','A_dy','B_dx','B_dy']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError('A regular existing input file is required: '+str(path))
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def completed_evaluation(path):
    controller = read(path)
    if (controller.get('status') != 'repair_diagnostic_complete' or
            [job.get('stage') for job in controller.get('jobs',[])] !=
            ['v54_eval','v54_anchor_reader','v54_bridge_reader'] or
            any(job.get('status')!='complete' or job.get('returncode')!=0 for job in controller['jobs'])):
        raise ValueError('Do not run this posthoc replay before all evaluation stages complete')
    return controller


def detailed_score(reader, prediction, target, mean):
    import numpy as np
    score = reader.score_outputs(prediction,target,mean)
    p = np.asarray(prediction,dtype=np.float64).reshape(-1,8)
    y = np.asarray(target,dtype=np.float64).reshape(-1,8)
    mse = np.mean((p-y)**2,axis=0)
    score.update(per_dimension_rmse=np.sqrt(mse).tolist(),
        per_dimension_baseline_mse=np.mean((y-mean)**2,axis=0).tolist(),
        dx_mse=float(mse[[4,6]].mean()),motion_mse=float(mse[4:].mean()),
        motion_prediction_rms=np.sqrt(np.mean(p[:,4:]**2,axis=0)).tolist(),
        motion_target_rms=np.sqrt(np.mean(y[:,4:]**2,axis=0)).tolist())
    return score


def optional_comparison(path, role, frozen, publication_sha, fit_scores):
    report = read(path)
    if (report.get('schema')!='v5.4-reader-evaluation-1' or
            report.get('role')!=role or report.get('status')!='evaluation_complete' or
            report.get('frozen_reader_sha256')!=frozen['receipt_sha256'] or
            report.get('freeze_publication_sha256')!=publication_sha or
            report.get('test_opened') is not False):
        raise ValueError('Official report does not belong to the same published frozen reader: '+str(path))
    result = {}
    for arm in ('N','W'):
        key = arm+'_concat'
        score = report['scores']['readers'][key]
        if role=='anchor':
            score = score['all_views']
        fitted = fit_scores[arm]['all_views']
        result[arm] = dict(official_score=score,
            evaluation_minus_own_fit_per_dimension_mse=[x-y for x,y in zip(score['per_dimension_mse'],fitted['per_dimension_mse'])],
            own_fit_minus_evaluation_motion_skill=fitted['motion_skill']-score['motion_skill'],
            comparison='Descriptive difference across distributions using the same frozen joint FIT mean; not a causal decomposition')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--model',type=Path,default=BASE/'frozen/reader')
    parser.add_argument('--publication',type=Path,default=BASE/'freeze_publication.json')
    parser.add_argument('--evaluation-controller',type=Path,default=BASE/'report-snapshot/evaluation_controller.json')
    parser.add_argument('--anchor-report',type=Path)
    parser.add_argument('--bridge-report',type=Path)
    parser.add_argument('--out',type=Path,default=BASE/'training-generalization')
    args = parser.parse_args()
    if not args.execute:
        print('Preview only: after evaluation completion, replay the fixed N and W models on their own existing '
              '4096 FIT views; score with the frozen joint FIT mean. No fit/encoder/new data or gate changes.')
        return
    started = time.time()
    controller = completed_evaluation(args.evaluation_controller)
    if args.out.exists():
        raise FileExistsError('Existing diagnostic output is preserved; use a fresh --out')
    if args.model.resolve() in args.out.resolve().parents:
        raise ValueError('Diagnostic output must remain outside the immutable frozen-model directory')
    # Import only verified numerical reader code, and never create __pycache__
    # in the archived science source tree.
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(REPO/'scripts'))
    import numpy as np
    import v5_4_reader as reader
    arrays, frozen = reader.load_frozen(args.model)
    publication_sha = reader.validate_publication(args.publication,frozen)
    if controller['source_commit']!=frozen['cache_identity']['source_commit']:
        raise ValueError('Evaluation controller and frozen model source differ')
    input_paths = dict(freeze_receipt=args.model/'freeze_receipt.json',
        frozen_model=args.model/'frozen_model.npz',training_config=args.model/'training_config.json',
        freeze_publication=args.publication,evaluation_controller=args.evaluation_controller,
        diagnostic_script=Path(__file__).resolve())
    for role,path in (('anchor',args.anchor_report),('bridge',args.bridge_report)):
        if path is not None:
            input_paths[role+'_report']=path
    inputs = {key:record(path) for key,path in input_paths.items()}
    # Query exactly the saved 512-world joint FIT basis. Both readers are
    # evaluated by the unchanged production predictor; only each reader's own
    # 256 worlds are retained in this diagnostic.
    features = arrays['fit_features']
    prediction = reader.predict_models(arrays,features.reshape(-1,features.shape[-1]),
                                        arrays['fit_view_indices'])
    mean = arrays['shared_target_mean']
    scores, saved = {}, dict(block_ids=arrays['fit_block_ids'],shared_target_mean=mean,
                            role=np.array('posthoc_own_fit_replay'))
    for arm,world_slice in (('N',slice(0,256)),('W',slice(256,512))):
        p = prediction[arm+'_concat'].reshape(512,16,8)[world_slice]
        y = arrays['fit_targets'][world_slice]
        saved[arm+'_prediction'] = p
        saved[arm+'_targets'] = y
        scores[arm] = dict(worlds=256,examples=4096,input_domain=arm+'_FIT',
            all_views=detailed_score(reader,p,y,mean),paths={},schedule_groups={})
        for index,name in enumerate(('aa','bb','ab','ba')):
            scores[arm]['paths'][name] = detailed_score(reader,p[:,index::4,:],y[:,index::4,:],mean)
        static = np.array([i for i in range(16) if i%4 in (0,1)])
        moving = np.array([i for i in range(16) if i%4 in (2,3)])
        for name,index in (('static',static),('moving',moving)):
            scores[arm]['schedule_groups'][name] = detailed_score(reader,p[:,index,:],y[:,index,:],mean)
    comparisons = {}
    for role,path in (('anchor',args.anchor_report),('bridge',args.bridge_report)):
        if path is not None:
            comparisons[role] = optional_comparison(path,role,frozen,publication_sha,scores)
    # Recheck source and every input after prediction before writing any result.
    reader.validate_identity(frozen['cache_identity'])
    if inputs != {key:record(path) for key,path in input_paths.items()}:
        raise ValueError('An immutable diagnostic input changed during replay')
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/'predictions.npz',**saved)
    report = dict(schema='v5.4.posthoc-frozen-fit-replay-1',status='complete',
        posthoc=True,support_only=True,new_fit=False,new_generation=False,
        alpha_unchanged=float(arrays['mean_loss_alpha']),test_opened=False,p0_authorized=False,
        original_v5_gate='failed_unchanged',source_commit=frozen['cache_identity']['source_commit'],
        input_files=inputs,cache_identity=frozen['cache_identity'],narrow_reference=frozen['narrow_reference'],
        model_identity=frozen['model_identity'],source_hashes=frozen['sources'],
        original_fit_receipt_hashes={arm:hashlib.sha256(text.encode()).hexdigest()
                                     for arm,text in frozen['fit_receipt_utf8'].items()},
        training_config=read(args.model/'training_config.json'),dimensions=DIMENSIONS,
        baseline=dict(name='same joint N/W FIT target mean used by official evaluation',
                      values=mean.tolist(),per_dataset_denominator='mean((target - shared_target_mean)^2) by dimension',
                      world_weighting='equal world weight; 16 equally weighted views within each FIT world'),
        own_fit=scores,official_evaluation_comparisons=comparisons,
        predictions_file=record(args.out/'predictions.npz'),
        predictions_entry_hashes={key:reader.entry_identity(value) for key,value in saved.items()},
        limitations=['In-sample training diagnostics do not establish generalization.',
            'A train/evaluation gap can reflect covariate or task shift; it does not isolate a cause.',
            'No thresholds, alpha selection, new bootstrap intervals, or pass gates are introduced.'],
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
            threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']),elapsed_seconds=time.time()-started)
    save(args.out/'report.json',report)
    rows = []
    for arm in ('N','W'):
        score = scores[arm]['all_views']
        rows.append('| '+arm+' | '+f"{score['motion_skill']:.6f} | {score['dx_mse']:.8g}"+' | '+
                    ', '.join(f'{v:.4f}' for v in score['motion_direction_error_rate'])+' | '+
                    ', '.join(f'{v:.4f}' for v in score['motion_absolute_amplitude_ratio'])+' |')
    note = ('# V5.4 posthoc frozen FIT replay\n\n'
        'Each unchanged reader was replayed on its own 256 saved FIT worlds (4096 views). '
        'Scoring uses the same frozen joint N/W target mean as the official evaluation. '
        'The script loaded no encoder, generated no RGB, and performed no refitting or alpha selection.\n\n'
        '| Reader | Own-FIT motion skill | dx MSE | Direction error (A_dx, A_dy, B_dx, B_dy) | Absolute amplitude ratio |\n'
        '| --- | --- | --- | --- | --- |\n'+'\n'.join(rows)+'\n\n'
        'High own-FIT accuracy with weaker held-out accuracy is evidence of an observed generalization gap. '
        'Poor own-FIT accuracy instead shows that this fixed reader does not fit even its training inputs well. '
        'These are descriptive distinctions, not causal conclusions or a replacement pass gate. '
        'Per-axis denominators, errors, direction ties, amplitude/RMS ratios, static/moving groups, '
        'condition metrics and all input hashes are preserved in report.json.\n')
    if comparisons:
        note += '\nOfficial anchor/bridge report comparisons were supplied and bound to the same model/publication hashes; '
        note += 'their differences are in official_evaluation_comparisons.\n'
    (args.out/'README.md').write_text(note)
    print(json.dumps(dict(status='complete',out=str(args.out),elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
