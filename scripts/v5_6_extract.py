#!/usr/bin/env python3
"""Eight-worker V5.6 extraction. External controller owns the complete GPU lease.

fit-axis encodes four new FIT images/world and four axis endpoints/episode.
confirm is released only after reader freeze and its publication are verified.
FIT uses six-image calls (four new plus two old parity anchors), matching the
old FIT batch size; axis/confirm calls have four. Labels never enter the encoder.
"""
import argparse
from contextlib import redirect_stderr,redirect_stdout
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

if __name__=='__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(key,'1')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from v5_6_data import (SCHEMA,make_block,make_axes,X_VIEW_INDICES,J_VIEW_INDICES,UNIQUE_ORDER,
    AXIS_NAMES,AXIS_NEW_INDICES,AXIS_OLD_INDICES,AXIS_OLD_MASKS,OBJECT_CONDITION_INDICES)
from v5_5_counterfactual import (PAIR_IDS,composite_images,array_identity,json_sha,atomic_json,digest,
    Encoder,WorkerBudget,validate_runtime,ORIGINAL_DATA_SHA256,inventory)

PLAN='scripts/v5_6_plan.json'
PROTOCOL='docs/V5_6_EXECUTION_PROTOCOL.md'
V54_MANIFEST='ec7bc9c17da8417e9ecd11ba0f21415a2df0ee7af620af9d12f911c3ab26e75c'
V55_MANIFEST='b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663'
BASELINE_GPUH=5.32908766817043
NEW_LIMIT_GPUH=.65
SOURCE_REQUIRED={'scripts/v5_6_data.py','scripts/v5_6_extract.py','scripts/v5_6_combination.py',
    'scripts/v5_6_reader_ablation.py','scripts/v5_4_reader.py','scripts/v5_3_reader.py',
    'scripts/v5_3_data.py','scripts/v5_5_lane_reader.py','scripts/score_v5_5.py',
    'scripts/v5_4_plan.json','scripts/v5_5_plan.json','scripts/launch_v5_5.py',
    'scripts/launch_v5_6.py',PLAN,PROTOCOL,'scripts/v5_5_counterfactual.py','scripts/v5_4_data.py',
    'scripts/v5_temporal_confirmation.py','scripts/extract_v5_confirmation.py'} | {
    p.relative_to(ROOT).as_posix() for p in (ROOT/'pmr/visual_memory').glob('*.py')}


def read_json(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Regular receipt required: '+str(path))
    return json.loads(path.read_text())


def file_record(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Regular file required: '+str(path))
    return dict(bytes=path.stat().st_size,sha256=digest(path))


def save_npz(path,**arrays):
    path=Path(path);temporary=path.with_suffix('.npz.tmp')
    if path.exists():raise FileExistsError('Existing evidence must remain unchanged')
    with temporary.open('xb') as stream:
        np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def validate_lease(ledger,stage,deadline_unix,now=None):
    """Read the controller's reservation; the extractor never mutates it."""
    now=time.time() if now is None else now
    charged=float(ledger['charged_gpuh'])
    active=[x for x in ledger.get('controller_envelopes',[]) if x.get('status')=='active']
    if (not BASELINE_GPUH<=charged<=min(6.,BASELINE_GPUH+NEW_LIMIT_GPUH)+1e-12 or
            len(active)!=1 or active[0].get('stage')!='v56_'+stage):
        raise ValueError('Exact active stage reservation inside V5.6/diagnostic ceiling required')
    envelope=active[0];gpuh=float(envelope['gpuh']);start=float(envelope['start_unix'])
    if (not 0<gpuh<=NEW_LIMIT_GPUH+1e-12 or not 0<start<=now or
            not now<deadline_unix<=start+gpuh*3600/8+1e-6):
        raise ValueError('Extractor deadline exceeds its authenticated eight-GPU reservation')
    return dict(stage=envelope['stage'],invocation_id=envelope['invocation_id'],reserved_gpuh=gpuh,
                reserved_start_unix=start,extractor_deadline_unix=deadline_unix,
                cumulative_reserved_gpuh=charged)


def source_and_release(args):
    root=args.run_root;st=root.stat();creation=read_json(root/'creation.json')
    expected=dict(root=str(root),inode=st.st_ino,device=st.st_dev,uid=st.st_uid)
    if (any(creation.get(k)!=v for k,v in expected.items()) or creation.get('preexisted') is not False or
            creation.get('parent_preserved') is not True or not creation.get('token')):
        raise ValueError('Run creation identity differs')
    manifest=read_json(root/'source_manifest.json');publication=read_json(root/'registration_publication.json')
    if (re.fullmatch('[0-9a-f]{40}',args.source_commit) is None or manifest.get('commit')!=args.source_commit or
            not SOURCE_REQUIRED.issubset(manifest['files'])):raise ValueError('Complete frozen source identity required')
    for name,sha in manifest['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or (ROOT/path).is_symlink() or digest(ROOT/path)!=sha:
            raise ValueError('Source changed: '+name)
    if manifest['files']['pmr/visual_memory/data.py']!=ORIGINAL_DATA_SHA256:raise ValueError('Original generator changed')
    expected=dict(source_commit=args.source_commit,plan_sha256=digest(ROOT/PLAN),protocol_sha256=digest(ROOT/PROTOCOL))
    if (publication.get('verified_readback') is not True or any(publication.get(k)!=v for k,v in expected.items()) or
            not 0<float(publication.get('verified_unix',0))<=time.time()):
        raise ValueError('Published source/protocol readback required before generation')
    freshness=read_json(root/'freshness_audit.json')
    if (freshness.get('schema')!='v5.6-freshness-audit-1' or freshness.get('verified') is not True or
            freshness.get('test_opened') is not False or freshness.get('unseen_namespaces')!=[SCHEMA+'/confirm'] or
            freshness.get('source_commit')!=args.source_commit or freshness.get('plan_sha256')!=expected['plan_sha256']):
        raise ValueError('Exact confirmation namespace freshness evidence required')
    ledger=read_json(root/'ledger/budget.json')
    lease=validate_lease(ledger,args.stage,args.deadline_unix)
    result=dict(**expected,source_identity=manifest['files'],source_manifest_sha256=digest(root/'source_manifest.json'),
        registration_publication_sha256=digest(root/'registration_publication.json'),
        freshness_audit_sha256=digest(root/'freshness_audit.json'),creation_sha256=digest(root/'creation.json'),
        extraction_reservation=lease)
    if args.stage=='confirm':
        frozen_path=root/'reader/freeze_receipt.json';frozen=read_json(frozen_path)
        published=read_json(root/'freeze_publication.json')
        if (frozen.get('schema')!='v5.6-reader-freeze-1' or frozen.get('status')!='frozen_before_confirmation' or
                frozen.get('source_commit')!=args.source_commit or frozen.get('plan_sha256')!=expected['plan_sha256'] or
                frozen.get('fit_cache_sha256')!=digest(root/'fit_cache/cache.npz')):
            raise ValueError('Frozen J/X reader identity differs')
        if not {'frozen_model.npz','training_config.json'}.issubset(frozen.get('files',{})):
            raise ValueError('Full frozen model and training configuration required')
        for name,sha in frozen['files'].items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or file_record(frozen_path.parent/path)['sha256']!=sha:
                raise ValueError('Frozen reader asset differs')
        if (published.get('verified_readback') is not True or
                published.get('frozen_reader_sha256')!=digest(frozen_path) or
                published.get('source_commit')!=args.source_commit or
                published.get('plan_sha256')!=expected['plan_sha256'] or
                not 0<float(frozen.get('created_unix',0))<=float(published.get('verified_unix',0))<=time.time()):
            raise ValueError('Read back the completed frozen reader before confirmation generation')
        result.update(frozen_reader_sha256=digest(frozen_path),freeze_publication_sha256=digest(root/'freeze_publication.json'))
    return result


def prior_reference(scene,expected_sha,cache_name):
    if digest(scene/'archive_manifest.json')!=expected_sha:raise ValueError('Prior sealed scene changed')
    seal=read_json(scene/'archive_manifest.json')['files'];folder=scene/cache_name
    manifest=read_json(folder/'manifest.json');receipt=read_json(folder/'receipt.json')
    if manifest.get('status')!='complete':raise ValueError('Prior cache incomplete')
    for name in ('manifest.json','receipt.json','cache.npz'):
        if file_record(folder/name)!=seal[cache_name+'/'+name]:raise ValueError('Prior archive membership mismatch')
    if receipt['cache_sha256']!=digest(folder/'cache.npz'):raise ValueError('Prior aggregate identity differs')
    return dict(scene_manifest_sha256=expected_sha,cache_name=cache_name,cache_sha256=receipt['cache_sha256'],
        cache_receipt_sha256=digest(folder/'receipt.json'),cache_manifest_sha256=digest(folder/'manifest.json'),
        encoder_identity=receipt['encoder_identity']),manifest,receipt


def old_unit(scene,cache_name,unit,manifest):
    folder=scene/cache_name;prefix=f'shards/{unit:06d}'
    for suffix in ('.json','.npz'):
        if file_record(folder/(prefix+suffix))!=manifest['files'][prefix+suffix]:
            raise ValueError('Prior shard hash differs')
    receipt=read_json(folder/(prefix+'.json'))
    if receipt['configuration_sha256']!=json_sha(manifest['configuration']):raise ValueError('Prior shard configuration differs')
    with np.load(folder/(prefix+'.npz'),allow_pickle=False) as z:
        keys=('features','targets') if cache_name=='fit_cache' else ('features','original_targets')
        arrays={k:z[k] for k in keys}
    entries=receipt['entry_hashes'] if cache_name=='fit_cache' else receipt['pooled']['entry_hashes']
    if any(array_identity(a)!=entries[k] for k,a in arrays.items()):raise ValueError('Prior shard entry differs')
    return arrays,receipt['inputs']


def encode_batch(encoder,images,batch_size,budget=None):
    if (batch_size not in (4,6) or images.dtype!=np.uint8 or images.ndim!=4 or
            images.shape[1:]!=(224,224,3) or not len(images) or len(images)%batch_size):
        raise ValueError('RGB input must consist of complete registered batches')
    output=[]
    for start in range(0,len(images),batch_size):
        if budget:budget.check()
        x=encoder(images[start:start+batch_size])
        if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[:2]!=(batch_size,16):raise ValueError('Expected 4x4 token fields')
        output.append(x.float().cpu().numpy().reshape(batch_size,-1))
    result=np.concatenate(output)
    if result.dtype!=np.float32 or not np.isfinite(result).all():raise ValueError('Invalid pooled features')
    return result


def encode4(encoder,images,budget=None):
    return encode_batch(encoder,images,4,budget)


def fit_unit(unit,encoder,previous,inputs,budget=None,software_block=None):
    block=make_block('fit',unit) if software_block is None else software_block
    if ([array_identity(x) for x in block.unique_rgb[:6]]!=inputs['unique_rgb'] or
            not np.array_equal(block.x_targets[:16],previous['targets'])):
        raise ValueError('Old W FIT RGB or labels changed')
    encoded=encode_batch(encoder,np.concatenate((block.unique_rgb[6:],block.unique_rgb[:2])),6,budget)
    if not np.array_equal(encoded[4:],previous['features'][:2]):
        raise ValueError('FIT batch6 old-anchor canonical parity failed; preserve without tolerance adjustment')
    features=np.concatenate((previous['features'],encoded[:4]))
    return dict(features=features,targets=block.x_targets,j_targets=block.j_targets),dict(
        unique_rgb=[array_identity(x) for x in block.unique_rgb],metadata=block.metadata,
        old_six_features_reused_exact=True,old_input_sha256=json_sha(inputs),images_encoded=6,
        parity=dict(old_slots=[0,1],batch_slots=[4,5],all_bitwise_equal=True,images_encoded=2),
        encoder_batch_size=6,new_images_encoded=4)


def axis_unit(pair,encoder,previous,inputs,budget=None,split='dev'):
    blocks=[make_axes(split,pair,v) for v in (0,1)]
    for v,block in enumerate(blocks):
        if block.metadata['old_v55_inputs']!=inputs['variants'][v]:raise ValueError('Old V5.5 RGB/mask inputs changed')
        if not np.array_equal(block.targets[3,4:6],previous['original_targets'][v,4:6]):raise ValueError('A displacement differs')
        if not np.array_equal(block.targets[6,6:8],previous['original_targets'][v,6:8]):raise ValueError('B displacement differs')
    new=encode4(encoder,np.concatenate([b.unique_rgb[AXIS_NEW_INDICES] for b in blocks]),budget).reshape(2,4,-1)
    features=np.empty((2,7,new.shape[-1]),dtype=np.float32)
    features[:,AXIS_NEW_INDICES]=new;features[:,AXIS_OLD_INDICES]=previous['features'][:,AXIS_OLD_MASKS]
    return dict(features=features,targets=np.stack([b.targets for b in blocks])),dict(
        variants=[dict(unique_rgb=[array_identity(x) for x in b.unique_rgb],metadata=b.metadata) for b in blocks],
        old_features_reused_exact=True,old_input_sha256=json_sha(inputs),images_encoded=8)


def confirm_units(units,encoder,budget=None,software=False):
    if len(units)!=2:raise ValueError('Two10-image blocks give five complete batches of4')
    blocks=[make_block('software_test' if software else 'confirm',u) for u in units]
    features=encode4(encoder,np.concatenate([b.unique_rgb for b in blocks]),budget).reshape(2,10,-1)
    return [(dict(features=features[i],targets=b.x_targets,j_targets=b.j_targets),
             dict(unique_rgb=[array_identity(x) for x in b.unique_rgb],metadata=b.metadata,images_encoded=10,
                  companion_unit=units[1-i],batch_group_units=list(units))) for i,b in enumerate(blocks)]


def save_unit(root,role,unit,config,arrays,inputs):
    path=root/(role+'_cache')/'shards'/f'{unit:06d}.npz'
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError('Existing shard preserved')
    save_npz(path,**arrays)
    atomic_json(path.with_suffix('.json'),dict(role=role,unit=unit,configuration_sha256=json_sha(config),
        file=file_record(path),entry_hashes={k:array_identity(a) for k,a in arrays.items()},inputs=inputs))


def worker(index,args,config,manifests,events,cancel,begin,deadline):
    torch.set_num_threads(1);budget=WorkerBudget(deadline,cancel)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,lambda *_:cancel.set())
    path=args.run_root/('workers_'+args.stage)/f'worker_{index}_{os.getpid()}.json'
    record=dict(worker=index,pid=os.getpid(),stage=args.stage,status='starting',completed={},images_encoded=0,
        start_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'))
    atomic_json(path,record)
    with path.with_suffix('.log').open('a',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        try:
            budget.check();torch.cuda.set_device(index);free,total=torch.cuda.mem_get_info(index)
            if free<40*1024**3 or 'H800' not in torch.cuda.get_device_name(index):raise ValueError('H800 with40GiB free required')
            record['before_encoder_memory']=dict(free_bytes=free,total_bytes=total)
            encoder=Encoder(args.ijepa,args.checkpoint,device=f'cuda:{index}',grid=4)
            identity=config['encoder_identity']
            if (encoder.identity['code_commit']!=identity['source_commit'] or
                    encoder.identity['checkpoint_sha256']!=identity['checkpoint_sha256']):raise ValueError('Encoder identity differs')
            record.update(status='ready',encoder=encoder.identity,visible_uuid=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[index])
            atomic_json(path,record);events.put(dict(event='ready',worker=index))
            while not begin.wait(.2):budget.check()
            if args.stage=='fit-axis':
                pair=640+index;old_axis,old_inputs=old_unit(args.prior_v55,'counterfactual_cache',pair,manifests['axis'])
                rows=[composite_images('dev',pair,v) for v in (0,1)]
                if [r[3] for r in rows]!=old_inputs['variants']:raise ValueError('Dev parity pixels changed')
                axis_parity=encode4(encoder,np.concatenate([r[0][[0,7]] for r in rows]),budget).reshape(2,2,-1)
                if not np.array_equal(axis_parity,old_axis['features'][:,[0,7]]):
                    raise ValueError('Batch4 old-endpoint canonical parity failed; preserve without tolerance adjustment')
                record.update(images_encoded=4,parity=dict(dev_pair=pair,dev_old_slots=[0,7],
                    all_bitwise_equal=True,images_encoded=4,fit_parity='two old anchors checked separately in every FIT shard'))
                for role,units in (('fit',range(index,256,8)),('axis',range(640+index,896,8))):
                    record['completed'][role]=[]
                    for unit in units:
                        budget.check()
                        scene,folder,key=(args.prior_v54,'fit_cache','fit') if role=='fit' else (args.prior_v55,'counterfactual_cache','axis')
                        old,inputs=old_unit(scene,folder,unit,manifests[key])
                        arrays,evidence=(fit_unit(unit,encoder,old,inputs,budget) if role=='fit' else axis_unit(unit,encoder,old,inputs,budget))
                        budget.check();save_unit(args.run_root,role,unit,config,arrays,evidence)
                        record['completed'][role].append(unit);record['images_encoded']+=evidence['images_encoded']
                        record['status']='running';atomic_json(path,record)
                        print(json.dumps(dict(role=role,unit=unit,status='complete')),flush=True)
            else:
                units=list(range(index,64,8));record['completed']['confirm']=[]
                for start in range(0,len(units),2):
                    chosen=units[start:start+2];budget.check();values=confirm_units(chosen,encoder,budget)
                    for unit,(arrays,evidence) in zip(chosen,values):
                        budget.check();save_unit(args.run_root,'confirm',unit,config,arrays,evidence)
                        record['completed']['confirm'].append(unit);record['images_encoded']+=10
                    record['status']='running';atomic_json(path,record)
                    print(json.dumps(dict(role='confirm',units=chosen,status='complete')),flush=True)
            record['status']='complete'
        except BaseException as exc:
            record.update(status='partial',reason=str(exc));traceback.print_exc();events.put(dict(event='error',worker=index,reason=str(exc)))
        finally:
            record['end_unix']=time.time();atomic_json(path,record)
        if record['status']=='complete':events.put(dict(event='done',worker=index))


def parallel(args,config,manifests,budget,ctx,cancel,deadline):
    queue_,begin=ctx.Queue(),ctx.Event();processes=[];success=False
    def wait(kind):
        pending=set(range(8))
        while pending:
            budget.check()
            try:msg=queue_.get(timeout=.2)
            except queue.Empty:
                if any(processes[i].exitcode is not None for i in pending):raise RuntimeError('Worker exited without success receipt')
                continue
            if msg['event']=='error':raise RuntimeError(msg['reason'])
            if msg['event']!=kind or msg['worker'] not in pending:raise RuntimeError('Unexpected worker event')
            pending.remove(msg['worker'])
    try:
        for index in range(8):
            p=ctx.Process(target=worker,args=(index,args,config,manifests,queue_,cancel,begin,deadline),daemon=True)
            processes.append(p);p.start()
        wait('ready');begin.set();wait('done');success=True
    finally:
        if not success:cancel.set();begin.set()
        end=time.monotonic()+3
        for p in processes:p.join(timeout=max(0,end-time.monotonic()))
        for p in processes:
            if p.is_alive():p.kill();p.join(timeout=1)
        queue_.close()


def merge(args,role,config,slurm,budget):
    folder=args.run_root/(role+'_cache');units=range(640,896) if role=='axis' else range(256 if role=='fit' else 64)
    shapes=dict(features=(2,7,20480),targets=(2,7,8)) if role=='axis' else dict(features=(10,20480),targets=(32,8),j_targets=(32,8))
    arrays={k:np.empty((len(units),*shape),dtype=np.float32) for k,shape in shapes.items()};records={}
    for i,unit in enumerate(units):
        budget.check();path=folder/'shards'/f'{unit:06d}.npz';receipt=read_json(path.with_suffix('.json'))
        if (receipt['role']!=role or receipt['unit']!=unit or receipt['configuration_sha256']!=json_sha(config) or
                receipt['file']!=file_record(path)):raise ValueError('New shard provenance mismatch')
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(shapes):raise ValueError('New shard keys differ')
            for key,shape in shapes.items():
                a=z[key]
                if (a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all() or
                        array_identity(a)!=receipt['entry_hashes'][key]):raise ValueError('New shard contents differ')
                arrays[key][i]=a
        records[str(unit)]=receipt['inputs']
    expected={f'{u:06d}{s}' for u in units for s in ('.npz','.json')}
    if {p.name for p in (folder/'shards').iterdir()}!=expected:raise ValueError('Exact shard inventory differs')
    arrays['role']=np.array(role)
    arrays['pair_ids' if role=='axis' else 'block_ids']=np.array(list(units),dtype=np.int64)
    if role=='axis':arrays['object_condition_indices']=OBJECT_CONDITION_INDICES
    else:arrays.update(x_view_indices=X_VIEW_INDICES,j_view_indices=J_VIEW_INDICES)
    save_npz(folder/'cache.npz',**arrays);budget.check()
    result=dict(**config,role=role,cache_sha256=digest(folder/'cache.npz'),cache_bytes=(folder/'cache.npz').stat().st_size,
        entry_hashes={k:array_identity(a) for k,a in arrays.items()},configuration_sha256=json_sha(config),
        input_records_sha256=json_sha(records),groups=len(units),images_encoded=dict(fit=1536,axis=2048,confirm=640)[role],
        new_images_encoded=dict(fit=1024,axis=2048,confirm=640)[role],
        parity_images_encoded=512 if role=='fit' else 0,
        parity_images_encoded_at_shared_stage=32 if args.stage=='fit-axis' else 0,
        old_features_reused=role!='confirm',slurm=slurm,test_opened=False,new_reader_fit=False,
        feature_dtype='float32',pooling='4x4_flattened',encoder_batch_size=6 if role=='fit' else 4,
        scientific_scope='fresh object-combination confirmation; no direct axis intervention' if role=='confirm' else role,
        frozen_before_generation=args.stage=='confirm',unseen_before_release=args.stage=='confirm')
    atomic_json(folder/'receipt.json',result)
    atomic_json(folder/'manifest.json',dict(status='complete',configuration=config,files=inventory(folder)))
    return result


def run(args):
    if args.workers!=8 or not 0<args.deadline_unix-time.time()<=NEW_LIMIT_GPUH*3600/8:
        raise ValueError('Eight workers and a deadline inside the registered .65GPUh allowance required')
    for key in ('run_root','prior_v54','prior_v55'):
        path=getattr(args,key)
        if path.is_symlink():raise ValueError('Symlink scene path')
        setattr(args,key,path.resolve())
    if any(args.run_root==p or p in args.run_root.parents or args.run_root in p.parents for p in (args.prior_v54,args.prior_v55)):
        raise ValueError('New scene must be disjoint from both immutable prior scenes')
    released=source_and_release(args);ctx=mp.get_context('spawn');cancel=ctx.Event()
    deadline=time.monotonic()+args.deadline_unix-time.time();budget=WorkerBudget(deadline,cancel)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,lambda *_:cancel.set())
    references={};manifests={}
    for role,scene,seal,folder in (('fit',args.prior_v54,V54_MANIFEST,'fit_cache'),('axis',args.prior_v55,V55_MANIFEST,'counterfactual_cache')):
        if args.stage=='fit-axis':references[role],manifests[role],_=prior_reference(scene,seal,folder)
        else:
            if digest(scene/'archive_manifest.json')!=seal:raise ValueError('Prior archive seal differs')
    plan=read_json(ROOT/PLAN)
    if plan.get('schema')!=SCHEMA:raise ValueError('Plan schema differs')
    identity={k:plan['encoder'][k] for k in ('source_commit','checkpoint_sha256','model')}
    if any(r['encoder_identity']!=identity for r in references.values()):raise ValueError('Old/new encoder identity differs')
    config=dict(schema=SCHEMA,stage=args.stage,**released,encoder_identity=identity,references=references,
        batch_sizes=dict(fit=6,axis=4,confirm=4),workers=8,unique_order=list(UNIQUE_ORDER),axis_conditions=list(AXIS_NAMES))
    roles=('fit','axis') if args.stage=='fit-axis' else ('confirm',)
    for role in roles:
        folder=args.run_root/(role+'_cache')
        if folder.exists():raise FileExistsError('Prior attempt preserved; no automatic replay')
        (folder/'shards').mkdir(parents=True)
        atomic_json(folder/'manifest.json',dict(status='partial',configuration=config))
    (args.run_root/('workers_'+args.stage)).mkdir(exist_ok=False)
    budget.check()
    try:
        slurm=validate_runtime('dgx',args.account,args.partition,'H800',torch.cuda.device_count(),torch.cuda.get_device_name(0),
            allocation_job=args.allocation_job,qos=args.qos,shared_allocation_gpus=8,expected_worker_gpus=8,
            allow_gpu_sharing=args.allow_gpu_sharing,min_free_gpu_gib=40.)
        parallel(args,config,manifests,budget,ctx,cancel,deadline)
        results={role:merge(args,role,config,slurm,budget) for role in roles};budget.check()
        if any(digest(ROOT/p)!=h for p,h in config['source_identity'].items()):raise ValueError('Source changed during extraction')
        if (digest(args.prior_v54/'archive_manifest.json')!=V54_MANIFEST or
                digest(args.prior_v55/'archive_manifest.json')!=V55_MANIFEST):raise ValueError('Prior seal changed during extraction')
        return dict(status='complete',stage=args.stage,roles={k:v['cache_sha256'] for k,v in results.items()},
            images_encoded=3616 if args.stage=='fit-axis' else 640,test_opened=False)
    except BaseException as exc:
        for role in roles:
            folder=args.run_root/(role+'_cache')
            atomic_json(folder/'manifest.json',dict(status='partial',configuration=config,reason=str(exc),files=inventory(folder)))
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('fit-axis','confirm'))
    for name in ('run-root','prior-v54','prior-v55','ijepa','checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-commit','allocation-job','account','partition','qos'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--workers',type=int,choices=(8,),default=8)
    p.add_argument('--allow-gpu-sharing',action='store_true');return p


if __name__=='__main__':print(json.dumps(run(parser().parse_args()),indent=2,allow_nan=False))
