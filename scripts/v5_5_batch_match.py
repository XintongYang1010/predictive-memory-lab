#!/usr/bin/env python3
"""Published, bounded batch-four replay of every already-seen V5.5 endpoint.

Extraction writes canonical features only. Comparison is a separate CPU action,
uses frozen readers, and reports all masks/worlds without fitting or selection.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import signal
import sys
import time
import traceback

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(key, '1')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from v5_5_counterfactual import (PAIR_IDS, MASK_NAMES, composite_images, inventory,
    array_identity, json_sha, atomic_json, digest, Encoder, WorkerBudget, validate_runtime,
    ORIGINAL_DATA_SHA256)

SCHEMA = 'v5.5-batch-match-1'
ADDENDUM = 'docs/V5_5_BATCH_MATCH_ADDENDUM.md'
BATCH_MASKS = ((0, 7), (1, 2), (3, 4), (5, 6))
BASELINE = 4.522264341248407


def save_npz(path, **arrays):
    """Uncompressed atomic output avoids compression inside the GPU lease."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular receipt: '+str(path))
    return json.loads(path.read_text())


def file_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Expected a regular file')
    return dict(bytes=path.stat().st_size, sha256=digest(path))


def validate_registration(args):
    root, prior = args.run_root, args.prior_run
    if root.parent != prior or root.name != 'batch_match_run':
        raise ValueError('Use only the new batch_match_run child of the prior V5.5 run')
    st = root.stat(); creation = read_json(root/'creation.json')
    expected = dict(root=str(root), inode=st.st_ino, device=st.st_dev, uid=st.st_uid)
    if (any(creation.get(k) != v for k, v in expected.items()) or
            creation.get('preexisted') is not False or creation.get('parent_preserved') is not True or
            not isinstance(creation.get('token'), str) or not creation['token']):
        raise ValueError('Supplementary run creation identity mismatch')
    sources = read_json(root/'source_manifest.json')
    publication = read_json(root/'registration_publication.json')
    old_receipt = read_json(prior/'counterfactual_cache/receipt.json')
    required = set(old_receipt['source_identity']) | {
        'scripts/v5_5_batch_match.py', 'scripts/launch_v5_5_batch_match.py', ADDENDUM}
    if (re.fullmatch('[0-9a-f]{40}', args.source_commit) is None or
            sources.get('commit') != args.source_commit or not required.issubset(sources['files'])):
        raise ValueError('Complete published old/new source manifest required')
    for name, sha in sources['files'].items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or (ROOT/path).is_symlink() or
                digest(ROOT/path) != sha): raise ValueError('Source differs: '+name)
    if any(sources['files'][k] != v for k, v in old_receipt['source_identity'].items()):
        raise ValueError('First V5.5 scientific source must remain unchanged')
    if sources['files']['pmr/visual_memory/data.py'] != ORIGINAL_DATA_SHA256:
        raise ValueError('Original generator changed')
    state = read_json(prior/'extraction_controller.json')
    ledger = read_json(prior/'ledger/budget.json')
    if (state.get('status') != 'extraction_complete' or
            not ledger.get('leases') or ledger['leases'][-1].get('status') != 'closed' or
            not ledger.get('controller_envelopes') or ledger['controller_envelopes'][-1].get('status') != 'closed'):
        raise ValueError('First extraction must be terminal and its charge closed')
    charged = float(ledger['charged_gpuh'])
    remaining = 1.2-(charged-BASELINE)
    expected = dict(source_commit=args.source_commit, addendum_sha256=digest(ROOT/ADDENDUM),
        first_cache_manifest_sha256=digest(prior/'counterfactual_cache/manifest.json'),
        first_ledger_sha256=digest(prior/'ledger/budget.json'))
    if (publication.get('verified_readback') is not True or
            any(publication.get(k) != v for k, v in expected.items()) or
            not 0 < float(publication.get('verified_unix', 0)) <= time.time() or
            abs(float(publication.get('remaining_gpuh', -1))-remaining) > 1e-12 or
            charged < BASELINE or remaining < .45 or charged+.45 > 6):
        raise ValueError('Read-back publication or remaining closed budget differs')
    return dict(source_commit=args.source_commit, source_identity=sources['files'],
        source_manifest_sha256=digest(root/'source_manifest.json'),
        registration_publication_sha256=digest(root/'registration_publication.json'),
        creation_sha256=digest(root/'creation.json'), **{k: v for k, v in expected.items() if k != 'source_commit'},
        first_charged_gpuh=charged, remaining_registered_gpuh=remaining)


def load_prior(prior, expected_manifest_sha=None):
    """Authenticate the complete first cache seal, and load only needed arrays.

    Raw and prequant payloads are not consumed here; their first-run manifest
    entries stay pinned. This replay does not redo the separate raw-token audit.
    """
    folder = Path(prior)/'counterfactual_cache'
    manifest_path = folder/'manifest.json'; manifest = read_json(manifest_path)
    if expected_manifest_sha and digest(manifest_path) != expected_manifest_sha:
        raise ValueError('First cache manifest changed')
    if manifest.get('status') != 'complete': raise ValueError('First cache incomplete')
    for name in ('cache.npz', 'receipt.json'):
        if file_record(folder/name) != manifest['files'][name]: raise ValueError('First file hash differs: '+name)
    receipt = read_json(folder/'receipt.json')
    if (receipt.get('schema') != 'v5.5-module-localization-1' or receipt.get('groups') != 256 or
            receipt.get('episodes') != 512 or receipt.get('images_encoded') != 4096 or
            receipt.get('original_feature_parity', {}).get('bitwise_equal') is not True or
            receipt.get('test_opened') is not False or receipt.get('new_reader_fit') is not False or
            receipt.get('cache_sha256') != manifest['files']['cache.npz']['sha256']):
        raise ValueError('First cache contract or original parity failed')
    with np.load(folder/'cache.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in ('features', 'original_targets', 'endpoint_positions', 'pair_ids')}
    shapes = dict(features=(256,2,8,20480), original_targets=(256,2,8), endpoint_positions=(256,2,8,4), pair_ids=(256,))
    for key, value in arrays.items():
        if (value.shape != shapes[key] or value.dtype != (np.int64 if key == 'pair_ids' else np.float32) or
                not np.isfinite(value).all() or array_identity(value) != receipt['entry_hashes'][key]):
            raise ValueError('First cache array differs: '+key)
    if not np.array_equal(arrays['pair_ids'], PAIR_IDS): raise ValueError('First pair inventory changed')
    inputs = {}
    for pair in PAIR_IDS:
        name = f'shards/{pair:06d}.json'
        if file_record(folder/name) != manifest['files'][name]: raise ValueError('First input receipt differs')
        row = read_json(folder/name)
        if row['pair_id'] != pair or row['configuration_sha256'] != json_sha(manifest['configuration']):
            raise ValueError('First input provenance differs')
        inputs[str(pair)] = row['inputs']
    if json_sha(inputs) != receipt['input_records_sha256']: raise ValueError('First input aggregate differs')
    return arrays, inputs, receipt


def encode_pair(pair, encoder, expected_inputs, expected_target, split='dev'):
    rows = [composite_images(split, pair, v) for v in (0,1)]
    targets = np.stack([row[1] for row in rows])
    if ([row[3] for row in rows] != expected_inputs['variants'] or
            not np.array_equal(targets, expected_target) or
            array_identity(targets) != expected_inputs['targets']):
        raise ValueError('Any endpoint RGB, mask, geometry, or original label differs from first extraction')
    result = None
    for masks in BATCH_MASKS:
        batch = np.concatenate([row[0][list(masks)] for row in rows])
        canonical = encoder(batch).float().numpy()
        if canonical.dtype != np.float32 or len(canonical) != 4 or not np.isfinite(canonical).all():
            raise ValueError('Invalid canonical encoder output')
        canonical = canonical.reshape(2,2,-1)
        if result is None: result = np.empty((2,8,canonical.shape[-1]), dtype=np.float32)
        result[:,list(masks)] = canonical
    return result, dict(prior_inputs_sha256=json_sha(expected_inputs),
        unique_rgb=[[row[3]['unique_rgb'][m] for m in range(8)] for row in rows],
        all_rgb_masks_labels_equal_first_extraction=True, encoder_batch_sizes=[4,4,4,4],
        batch_masks=[list(m) for m in BATCH_MASKS])


def read_shard(out, pair, config_sha):
    path = out/'shards'/f'{pair:06d}.npz'; meta = path.with_suffix('.json')
    if not path.exists() and not meta.exists(): return None
    row = read_json(meta)
    if row['pair_id'] != pair or row['configuration_sha256'] != config_sha or file_record(path) != row['file']:
        raise ValueError('Supplementary shard identity differs')
    with np.load(path, allow_pickle=False) as z:
        if z.files != ['features']: raise ValueError('Unexpected shard entries')
        feature = z['features']
    if (feature.shape != (2,8,20480) or feature.dtype != np.float32 or not np.isfinite(feature).all() or
            array_identity(feature) != row['entry_hashes']['features']): raise ValueError('Shard features differ')
    return feature, row


def worker(index, args, config_sha, targets, inputs, events, cancel, begin, deadline, encoder_identity):
    torch.set_num_threads(1); budget = WorkerBudget(deadline, cancel)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1): signal.signal(sig, lambda *_: cancel.set())
    pairs = [p for p in PAIR_IDS if p%8 == index]
    key = f'worker_{index}_{os.getpid()}'; path = args.out/'workers'/(key+'.json')
    receipt = dict(worker=index, pid=os.getpid(), assigned_pairs=pairs, completed_pairs=[], status='starting',
        start_unix=time.time(), job_id=os.environ.get('SLURM_JOB_ID'), step_id=os.environ.get('SLURM_STEP_ID'))
    atomic_json(path, receipt)
    with (args.out/'workers'/(key+'.log')).open('a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            budget.check(); torch.cuda.set_device(index)
            free, total = torch.cuda.mem_get_info(index)
            if free < 40*1024**3 or 'H800' not in torch.cuda.get_device_name(index):
                raise ValueError('Each GPU must be H800 with at least40GiB free before load')
            receipt['before_encoder_memory'] = dict(free_bytes=free, total_bytes=total)
            encoder = Encoder(args.ijepa, args.checkpoint, device=f'cuda:{index}', grid=4)
            if (encoder.identity['code_commit'] != encoder_identity['source_commit'] or
                    encoder.identity['checkpoint_sha256'] != encoder_identity['checkpoint_sha256']):
                raise ValueError('Official encoder identity differs')
            receipt.update(status='ready', encoder=encoder.identity, ready_unix=time.time(),
                visible_uuid=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[index])
            atomic_json(path, receipt); events.put(dict(event='ready', worker=index))
            while not begin.wait(.2): budget.check()
            for pair in pairs:
                budget.check()
                feature, evidence = encode_pair(pair, encoder, inputs[str(pair)], targets[pair-640])
                budget.check(); target = args.out/'shards'/f'{pair:06d}.npz'
                if target.exists() or target.with_suffix('.json').exists(): raise FileExistsError('Preserve earlier attempt')
                save_npz(target, features=feature)
                atomic_json(target.with_suffix('.json'), dict(pair_id=pair, configuration_sha256=config_sha,
                    inputs=evidence, file=file_record(target), entry_hashes={'features': array_identity(feature)}))
                receipt['completed_pairs'].append(pair); receipt['status']='running'; atomic_json(path, receipt)
                print(json.dumps(dict(pair=pair, images_encoded=16, batch_sizes=[4,4,4,4])), flush=True)
            receipt['status']='complete'
        except BaseException as exc:
            receipt.update(status='partial', reason=str(exc)); traceback.print_exc()
            events.put(dict(event='error', worker=index, reason=str(exc)))
        finally:
            receipt['end_unix']=time.time(); atomic_json(path, receipt)
        if receipt['status']=='complete': events.put(dict(event='done', worker=index))


def extract_parallel(args, config, old, inputs, identity, budget, ctx, cancel, deadline):
    events, begin = ctx.Queue(), ctx.Event(); processes=[]; success=False
    def wait(kind):
        pending=set(range(8))
        while pending:
            budget.check()
            try: msg=events.get(timeout=.2)
            except queue.Empty:
                if any(processes[i].exitcode is not None for i in pending):
                    raise RuntimeError('Worker exited without completion receipt')
                continue
            if msg['event']=='error': raise RuntimeError('Worker failed: '+msg['reason'])
            if msg['event']!=kind or msg['worker'] not in pending: raise RuntimeError('Unexpected worker event')
            pending.remove(msg['worker'])
    try:
        # Unlike the first extraction, initialize all eight independent loaders
        # concurrently. No feature array is pickled into child processes.
        for index in range(8):
            selected={str(p): inputs[str(p)] for p in PAIR_IDS if p%8==index}
            process=ctx.Process(target=worker, args=(index,args,json_sha(config),old['original_targets'],selected,
                events,cancel,begin,deadline,identity), daemon=True)
            processes.append(process); process.start()
        wait('ready'); begin.set(); wait('done'); success=True
    finally:
        if not success: cancel.set(); begin.set()
        end=time.monotonic()+3
        for process in processes: process.join(timeout=max(0,end-time.monotonic()))
        for process in processes:
            if process.is_alive(): process.kill(); process.join(timeout=1)
        events.close()


def run_extract(args):
    if args.workers!=8 or not 0<args.deadline_unix-time.time()<=165:
        raise ValueError('Exactly8 workers and at most165seconds internal deadline required')
    for key in ('out','run_root','prior_run'):
        path=getattr(args,key)
        if path.is_symlink(): raise ValueError('No linked scene paths')
        setattr(args,key,path.resolve())
    if args.out.parent!=args.run_root or args.out.name!='cache': raise ValueError('Use the child cache directory')
    config=validate_registration(args)
    ctx=mp.get_context('spawn'); cancel=ctx.Event()
    deadline=time.monotonic()+args.deadline_unix-time.time(); budget=WorkerBudget(deadline,cancel)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1): signal.signal(sig,lambda *_:cancel.set())
    old, inputs, receipt=load_prior(args.prior_run,config['first_cache_manifest_sha256']); budget.check()
    config.update(schema=SCHEMA,pair_ids=list(PAIR_IDS),batch_masks=[list(m) for m in BATCH_MASKS],workers=8,
        first_cache_sha256=receipt['cache_sha256'],first_cache_receipt_sha256=digest(args.prior_run/'counterfactual_cache/receipt.json'),
        encoder_identity=receipt['encoder_identity'],first_input_records_sha256=receipt['input_records_sha256'])
    if args.out.exists(): raise FileExistsError('Fresh child cache required; failed attempts stay preserved')
    for name in ('shards','workers'): (args.out/name).mkdir(parents=True)
    manifest=args.out/'manifest.json'; atomic_json(manifest,dict(status='partial',configuration=config))
    try:
        slurm=validate_runtime('dgx',args.account,args.partition,'H800',torch.cuda.device_count(),torch.cuda.get_device_name(0),
            allocation_job=args.allocation_job,qos=args.qos,shared_allocation_gpus=8,expected_worker_gpus=8,
            allow_gpu_sharing=args.allow_gpu_sharing,min_free_gpu_gib=40.)
        extract_parallel(args,config,old,inputs,receipt['encoder_identity'],budget,ctx,cancel,deadline)
        feature=np.empty((256,2,8,20480),dtype=np.float32); records={}
        for i,pair in enumerate(PAIR_IDS):
            budget.check(); feature[i],row=read_shard(args.out,pair,json_sha(config)); records[str(pair)]=row['inputs']
        if {p.name for p in (args.out/'shards').iterdir()}!={f'{p:06d}{s}' for p in PAIR_IDS for s in ('.npz','.json')}:
            raise ValueError('Shard inventory differs')
        equal=np.array_equal(feature[:,:,[0,7]],old['features'][:,:,[0,7]])
        arrays=dict(features=feature,pair_ids=np.asarray(PAIR_IDS,dtype=np.int64))
        save_npz(args.out/'cache.npz',**arrays); budget.check()
        result=dict(role='batch_match',groups=256,episodes=512,images_encoded=4096,
            **config,cache_sha256=digest(args.out/'cache.npz'),cache_bytes=(args.out/'cache.npz').stat().st_size,
            entry_hashes={k:array_identity(v) for k,v in arrays.items()},configuration_sha256=json_sha(config),
            input_records_sha256=json_sha(records),original_features_bitwise_equal=bool(equal),
            all_rgb_masks_labels_equal_first_extraction=True,canonical_pooling='unchanged Encoder.__call__; BF16 then FP32',
            raw_retained=False,prequant_retained=False,slurm=slurm,deadline_unix=args.deadline_unix,
            budget_owner='separate child controller; no extractor lease',test_opened=False,new_reader_fit=False,
            new_unseen_dev=False,p0_authorized=False)
        atomic_json(args.out/'receipt.json',result)
        if not equal: raise ValueError('Original batch-four endpoints changed; preserve scene, do not interpret')
        for name,sha in config['source_identity'].items():
            if digest(ROOT/name)!=sha: raise ValueError('Source changed during extraction')
        for path,key in ((args.prior_run/'counterfactual_cache/manifest.json','first_cache_manifest_sha256'),
                         (args.prior_run/'ledger/budget.json','first_ledger_sha256')):
            if digest(path)!=config[key]: raise ValueError('Prior evidence changed during extraction')
        files=inventory(args.out); budget.check()
        atomic_json(manifest,dict(status='complete',configuration=config,files=files)); return result
    except BaseException as exc:
        atomic_json(manifest,dict(status='partial',configuration=config,reason=str(exc),files=inventory(args.out)))
        raise


def distribution(values):
    x=np.asarray(values,dtype=np.float64)
    return dict(count=x.size,exact_zero_count=int(np.count_nonzero(x==0)),mean=float(x.mean()),
        rms=float(np.sqrt(np.mean(x*x))),max_abs=float(np.abs(x).max()),
        absolute_quantiles=dict(zip(('0','0.5','0.9','0.99','0.999','1'),
            np.quantile(np.abs(x),[0,.5,.9,.99,.999,1]).tolist())))


def run_compare(args):
    from v5_4_reader import load_frozen,predict_models
    from v5_5_lane_reader import load_frozen as load_lane,predict as predict_lane
    from score_v5_5 import factorial_targets,stats,decompose,attribution,bootstrap_comparison
    started=time.time(); cache=args.cache.resolve(); manifest=read_json(cache/'manifest.json')
    if manifest['status']!='complete' or inventory(cache)!=manifest['files']: raise ValueError('New cache seal differs')
    receipt=read_json(cache/'receipt.json')
    if (receipt['schema']!=SCHEMA or receipt['original_features_bitwise_equal'] is not True or
            file_record(cache/'cache.npz')!=dict(bytes=receipt['cache_bytes'],sha256=receipt['cache_sha256'])):
        raise ValueError('New cache failed original parity or hash')
    for name,sha in receipt['source_identity'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or digest(ROOT/path)!=sha:
            raise ValueError('Registered comparison source changed: '+name)
    with np.load(cache/'cache.npz',allow_pickle=False) as z: data={k:z[k] for k in z.files}
    if (set(data)!={'features','pair_ids'} or data['features'].shape!=(256,2,8,20480) or
            data['features'].dtype!=np.float32 or not np.isfinite(data['features']).all() or
            not np.array_equal(data['pair_ids'],PAIR_IDS) or
            {k:array_identity(v) for k,v in data.items()}!=receipt['entry_hashes']): raise ValueError('New entries differ')
    old,_,old_receipt=load_prior(args.prior_run,receipt['first_cache_manifest_sha256'])
    if (old_receipt['cache_sha256']!=receipt['first_cache_sha256'] or
            not np.array_equal(data['features'][:,:,[0,7]],old['features'][:,:,[0,7]])):
        raise ValueError('Original batch-four canonical parity failed')
    model,frozen=load_frozen(args.frozen_model); lane,lr=load_lane(args.lane_model)
    if (frozen['receipt_sha256']!=old_receipt['reference']['frozen_reader_sha256'] or
            lr['original_frozen_reader_sha256']!=frozen['receipt_sha256']): raise ValueError('Reader freeze differs')
    publication=read_json(args.prior_run/'registration_publication.json')
    lane_files={p.relative_to(args.lane_model).as_posix():file_record(p) for p in args.lane_model.rglob('*') if p.is_file()}
    if publication['lane_reader']['files']!=lane_files: raise ValueError('Lane publication differs')
    args.out.mkdir(parents=True,exist_ok=False)
    index=np.stack((np.repeat(np.arange(512)*8,8),np.arange(4096)),axis=1)
    predictions={}
    for label,features in (('first',old['features']),('batch4',data['features'])):
        x=features.reshape(-1,20480)
        predictions[label]={k:v.reshape(256,2,8,8) for k,v in predict_models(model,x,index).items()}
        predictions[label]['lane_W']=predict_lane(lane,x,index).reshape(256,2,8,8)
    y=factorial_targets(old['original_targets'],old['endpoint_positions']); rms=model['fit_motion_rms']
    feature_summary={}; feature_rows=[]
    for m in range(8):
        delta=data['features'][:,:,m].astype(np.float64)-old['features'][:,:,m]
        feature_summary[MASK_NAMES[m]]=distribution(delta)
        for i,pair in enumerate(PAIR_IDS):
            for v in (0,1):
                d=delta[i,v]; feature_rows.append(dict(pair_id=pair,variant=v,mask=MASK_NAMES[m],
                    exact_equal=bool(np.all(d==0)),changed_values=int(np.count_nonzero(d)),
                    rms=float(np.sqrt(np.mean(d*d))),max_abs=float(np.max(np.abs(d)))))
    with (args.out/'feature_per_endpoint.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(feature_rows[0]));writer.writeheader();writer.writerows(feature_rows)
    reports={}; output=dict(targets=y,pair_ids=data['pair_ids'])
    for label,values in predictions.items():
        reports[label]={}
        for reader,p in values.items():
            phi,mob=decompose(p)
            reports[label][reader]=dict(masks={MASK_NAMES[m]:stats(p[:,:,m],y[:,:,m],rms) for m in range(8)},
                attribution=attribution(phi,mob,y[:,:,7],rms))
            output[label+'__'+reader]=p;output[label+'__'+reader+'__shapley']=phi
            output[label+'__'+reader+'__moebius']=mob
        reports[label]['global_vs_lane']=bootstrap_comparison(values['W_concat'],values['lane_W'],y)
    drift={}
    for reader in predictions['first']:
        delta=predictions['batch4'][reader]-predictions['first'][reader]
        output['delta__'+reader]=delta
        drift[reader]={MASK_NAMES[m]:dict(all_outputs=distribution(delta[:,:,m]),
            outputs=[distribution(delta[:,:,m,k]) for k in range(8)]) for m in range(8)}
        if np.max(np.abs(delta[:,:,[0,7]]))>1e-9: raise ValueError('Frozen original predictions changed')
    save_npz(args.out/'predictions.npz',**output)
    report=dict(schema=SCHEMA,status='complete_cpu_frozen_comparison',groups=256,episodes=512,masks=8,
        feature_differences=feature_summary,prediction_differences=drift,readers=reports,
        original_masks=[0,7],edited_masks=[1,2,3,4,5,6],no_case_selection=True,new_fit=False,test_opened=False,
        interpretation='Descriptive batch-shape sensitivity on all already-seen endpoints; no new performance gate or neural causal attribution',
        provenance=dict(first_cache_sha256=old_receipt['cache_sha256'],new_cache_sha256=receipt['cache_sha256'],
            new_receipt_sha256=digest(cache/'receipt.json'),first_manifest_sha256=receipt['first_cache_manifest_sha256'],
            frozen_reader_sha256=frozen['receipt_sha256'],lane_reader_sha256=digest(args.lane_model/'freeze_receipt.json'),
            source_sha256=digest(Path(__file__)),source_commit=receipt['source_commit']),elapsed_seconds=time.time()-started)
    atomic_json(args.out/'report.json',report)
    atomic_json(args.out/'manifest.json',dict(status='complete',files=inventory(args.out)))
    return dict(status=report['status'],elapsed_seconds=report['elapsed_seconds'])


def parser():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    extract=sub.add_parser('extract')
    for name in ('out','run-root','prior-run','ijepa','checkpoint'):extract.add_argument('--'+name,type=Path,required=True)
    for name in ('source-commit','allocation-job','account','partition','qos'):extract.add_argument('--'+name,required=True)
    extract.add_argument('--deadline-unix',type=float,required=True)
    extract.add_argument('--workers',type=int,choices=(8,),default=8)
    extract.add_argument('--allow-gpu-sharing',action='store_true')
    compare=sub.add_parser('compare')
    for name in ('cache','prior-run','frozen-model','lane-model','out'):compare.add_argument('--'+name,type=Path,required=True)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    print(json.dumps(run_extract(args) if args.action=='extract' else run_compare(args),indent=2,allow_nan=False))
