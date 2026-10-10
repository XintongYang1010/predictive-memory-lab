#!/usr/bin/env python3
"""Posthoc eight-endpoint object interventions on already observed V5.4 Dev.

The controller owns the only GPU budget lease. This entrypoint consumes its
absolute deadline, verifies published source, and never fits or scores a reader.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
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
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from extract_v5_confirmation import array_identity, json_sha, atomic_npz, ORIGINAL_DATA_SHA256
from pmr.visual_memory.common import atomic_json, digest
from pmr.visual_memory.data import seed_for, reflect
from pmr.visual_memory.feature_workers import WorkerBudget
from pmr.visual_memory.features import Encoder
from pmr.visual_memory.slurm_backend import validate_runtime

SCHEMA='v5.5-module-localization-1'
PAIR_IDS=tuple(range(640,896))
BITS=np.asarray([(m>>2&1,m>>1&1,m&1) for m in range(8)],dtype=np.int64)
MASK_NAMES=tuple(format(m,'03b') for m in range(8))
PROTOCOL='docs/V5_5_MODULE_DIAGNOSTIC.md'
PLAN='scripts/v5_5_plan.json'


def cue_geometry(split,pair,variant,size=224):
    """Replay only the original cue RNG draws; no later RGB frames are rendered."""
    if split not in ('software_test','dev') or (split=='dev' and pair not in PAIR_IDS) or variant not in (0,1):
        raise ValueError('Only software fixtures or already observed Dev640..895 are allowed')
    rng=np.random.default_rng(seed_for(split,pair))
    p=rng.uniform([.2,.23],[.8,.32],(2,2));p[1,1]+=.42
    velocity=rng.uniform(.008,.018,(2,2))*rng.choice([-1,1],(2,2))
    velocity*=1 if variant==0 else -1
    # Original future-update RNG consumption precedes the two colors.
    rng.uniform(.008,.018,2);rng.choice([-1,1],2)
    colors=[tuple(int(x) for x in c) for c in rng.integers(60,240,(2,3))]
    colors[0]=(235,colors[0][1]//2,60);colors[1]=(60,colors[1][1]//2,235)
    positions=[]
    for t in (6.,7.):
        xy=p+velocity*t
        xy[:,0]=reflect(xy[:,0],.10,.90)
        xy[0,1]=reflect(xy[0,1],.15,.40);xy[1,1]=reflect(xy[1,1],.60,.85)
        positions.append(xy)
    positions=np.stack(positions)
    yellow=[int((.5+.4*np.sin(t*2.3+pair))*size) for t in (6.,7.)]
    return positions,colors,yellow


def render_endpoint(positions,colors,yellow,size=224):
    im=Image.new('RGB',(size,size),(30,35,42));draw=ImageDraw.Draw(im)
    for x in range(0,size,max(1,size//16)):draw.line((x,0,x,size),fill=(42,47,54))
    masks=[];radius=size*.06
    for obj in range(2):
        x,y=positions[obj]*size
        box=(int(x-radius),int(y-radius),int(x+radius),int(y+radius))
        draw.ellipse(box,fill=colors[obj],outline=(250,250,240),width=max(1,size//100))
        draw.line((x-radius/2,y,x+radius/2,y),fill=(255,255,255),width=max(1,size//60))
        if obj==1:draw.line((x,y-radius/2,x,y+radius/2),fill=(255,255,255),width=max(1,size//60))
        mask=Image.new('L',(size,size),0);ImageDraw.Draw(mask).ellipse(box,fill=1)
        masks.append(np.asarray(mask).astype(bool))
    draw.rectangle((yellow-5,size*.48,yellow+5,size*.52),fill=(240,220,50))
    mask=Image.new('L',(size,size),0)
    ImageDraw.Draw(mask).rectangle((yellow-5,size*.48,yellow+5,size*.52),fill=1)
    masks.append(np.asarray(mask).astype(bool))
    masks=np.stack(masks)
    if np.any(masks.sum(0)>1):raise ValueError('Target/nuisance masks overlap; intervention invalid')
    return np.asarray(im).copy(),masks


def composite_images(split,pair,variant):
    """Return all eight endpoints in binary A/B/Y order plus analysis-only state."""
    positions,colors,yellow=cue_geometry(split,pair,variant)
    images,masks,endpoints=[],[],[]
    for a,b,y in BITS:
        state=np.stack((positions[a,0],positions[b,1]))
        rgb,mask=render_endpoint(state,colors,yellow[y])
        images.append(rgb);masks.append(mask);endpoints.append(state.reshape(-1))
    images=np.stack(images);masks=np.stack(masks);endpoints=np.stack(endpoints)
    allowed_hashes=[]
    for slot,bits in enumerate(BITS):
        allowed=np.zeros((224,224),dtype=bool)
        for obj,bit in enumerate(bits):
            if bit:allowed|=masks[0,obj]|masks[slot,obj]
            elif not np.array_equal(images[slot][masks[0,obj]],images[0][masks[0,obj]]):
                raise ValueError('An unchanged object pixel changed')
        if not np.array_equal(images[slot][~allowed],images[0][~allowed]):
            raise ValueError('Changed pixel outside intervened object old/new mask union')
        allowed_hashes.append(array_identity(allowed))
    edges=[]
    for start in range(8):
        for obj,bit in enumerate((4,2,1)):
            if start&bit:continue
            end=start|bit
            allowed=masks[start,obj]|masks[end,obj]
            if not np.array_equal(images[start][~allowed],images[end][~allowed]):
                raise ValueError('Cube edge changed pixels outside its one-object old/new support union')
            for other in range(3):
                if other!=obj and not np.array_equal(images[start][masks[start,other]],images[end][masks[end,other]]):
                    raise ValueError('Cube edge changed a non-intervened object')
            edges.append(dict(start_mask=start,end_mask=end,object=('A','B','Y')[obj],
                allowed_mask=array_identity(allowed),outside_support_unchanged=True,
                changed_pixels=int(np.any(images[start]!=images[end],axis=-1).sum())))
    if len(edges)!=12:raise ValueError('Exactly12 one-object cube edges required')
    original_target=np.concatenate((positions[1].reshape(-1),(positions[1]-positions[0]).reshape(-1))).astype(np.float32)
    inputs=dict(split=split,pair=pair,variant=variant,cue_times=[6,7],colors=[list(c) for c in colors],
        nuisance_centers=yellow,unique_rgb=[array_identity(image) for image in images],
        object_masks=[array_identity(mask) for mask in masks],allowed_change_masks=allowed_hashes,
        cue_positions=array_identity(positions),original_target=array_identity(original_target),
        outside_intervened_object_unions_unchanged=True,object_masks_disjoint=True,
        cube_edges=edges,all_twelve_cube_edges_verified=True)
    return images,original_target,endpoints.astype(np.float32),inputs


class LocalizationEncoder(Encoder):
    """Exact original forward, with raw FP32 and pre-storage pooled FP32 retained."""
    @torch.inference_mode()
    def __call__(self,rgb):
        x=torch.from_numpy(rgb.copy()).permute(0,3,1,2).to(self.device).float()/255
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=str(self.device).startswith('cuda')):
            z=self.model((x-mean)/std).float()
        side=int(z.shape[1]**.5)
        if side*side!=z.shape[1]:raise ValueError('Expected spatial tokens without CLS')
        pool=F.adaptive_avg_pool2d(z.transpose(1,2).reshape(len(z),-1,side,side),self.grid)
        pool=pool.flatten(2).transpose(1,2).cpu()
        canonical=pool.to(torch.bfloat16).float()
        return canonical.numpy(),pool.numpy(),z.cpu().numpy()


def encode_pair(pair,encoder,expected_target,expected_inputs,split='dev'):
    rows=[composite_images(split,pair,v) for v in (0,1)]
    targets=np.stack([r[1] for r in rows])
    cue_hashes=[rows[v][3]['unique_rgb'][m] for v in (0,1) for m in (0,7)]
    if (cue_hashes!=expected_inputs['cue_rgb'] or array_identity(targets)!=expected_inputs['targets']
            or not np.array_equal(targets,expected_target)):
        raise ValueError('Original 000/111 RGB or labels differ from archived original bridge')
    original=np.concatenate([row[0][[0,7]] for row in rows])
    edited=np.concatenate([row[0][1:7] for row in rows])
    x0,p0,z0=encoder(original);x1,p1,z1=encoder(edited)
    if not all(a.dtype==np.float32 and np.isfinite(a).all() for a in (x0,p0,z0,x1,p1,z1)):
        raise ValueError('Nonfinite or non-FP32 retained encoder outputs')
    for canonical,pool in ((x0,p0),(x1,p1)):
        if not np.array_equal(canonical,torch.from_numpy(pool).to(torch.bfloat16).float().numpy()):
            raise ValueError('Canonical pool is not exact BF16 roundtrip of same-call raw pooling')
    d=x0.shape[-2]*x0.shape[-1]
    features=np.empty((2,8,d),dtype=np.float32);pre=np.empty_like(features)
    features[:,[0,7]]=x0.reshape(2,2,d);features[:,1:7]=x1.reshape(2,6,d)
    pre[:,[0,7]]=p0.reshape(2,2,d);pre[:,1:7]=p1.reshape(2,6,d)
    arrays=dict(features=features,pooled_prequant=pre,original_targets=targets,
        endpoint_positions=np.stack([r[2] for r in rows]))
    # Only the original two cues' raw tokens persist; edited raw tokens are
    # discarded after their same-call pooling, without another encoder pass.
    raw=z0.reshape(2,2,*z0.shape[1:])
    return arrays,raw,dict(variants=[r[3] for r in rows],cue_rgb=cue_hashes,
        targets=array_identity(targets),original_rgb_hashes_equal_archived_bridge=True,
        same_call_bf16_pool_parity=True,encoder_batch_sizes=[4,12])


def inventory(root):
    files={}
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink():raise ValueError('Symlinks are forbidden')
        if p.is_file() and p!=Path(root)/'manifest.json':
            files[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=digest(p))
    return files


def prior_inputs(args):
    import v5_4_reader as reader
    scene=args.prior_scene
    seal=scene/'archive_manifest.json'
    if digest(seal)!=args.prior_manifest_sha256:raise ValueError('Prior scene seal SHA mismatch')
    frozen_arrays,frozen=reader.load_frozen(scene/'reader');del frozen_arrays
    old,receipt=reader.load_cache(scene/'bridge_cache/cache.npz',scene/'bridge_cache/receipt.json',
        'bridge',frozen,scene/'freeze_publication.json')
    manifest=json.loads((scene/'bridge_cache/manifest.json').read_text())
    if manifest['status']!='complete' or manifest['files']!=inventory(scene/'bridge_cache'):
        raise ValueError('Archived bridge cache is incomplete or changed')
    sealed=json.loads(seal.read_text())['files']
    for name in ('bridge_cache/cache.npz','bridge_cache/receipt.json','reader/freeze_receipt.json'):
        p=scene/name
        if sealed[name]!=dict(bytes=p.stat().st_size,sha256=digest(p)):
            raise ValueError('Prior archive membership mismatch: '+name)
    inputs={}
    for pair in PAIR_IDS:
        row=json.loads((scene/'bridge_cache/shards'/f'{pair:06d}.json').read_text())
        if row['role']!='bridge' or row['unit']!=pair or row['configuration_sha256']!=json_sha(manifest['configuration']):
            raise ValueError('Prior shard provenance mismatch')
        inputs[str(pair)]=row['inputs']
    if json_sha(inputs)!=receipt['input_records_sha256']:raise ValueError('Prior original RGB input records differ')
    reference=dict(prior_scene=str(scene),prior_archive_manifest_sha256=args.prior_manifest_sha256,
        original_bridge_cache_sha256=digest(scene/'bridge_cache/cache.npz'),
        original_bridge_receipt_sha256=digest(scene/'bridge_cache/receipt.json'),
        frozen_reader_sha256=frozen['receipt_sha256'],encoder_identity=receipt['encoder_identity'])
    return old,inputs,reference


def read_shard(out,pair,config_sha):
    path=out/'shards'/f'{pair:06d}.npz';meta=path.with_suffix('.json');raw=out/'raw_shards'/path.name
    if not any(p.exists() for p in (path,meta,raw)):return None
    if not all(p.is_file() and not p.is_symlink() for p in (path,meta,raw)):
        raise ValueError('Partial shard preserved; inspect before resuming')
    r=json.loads(meta.read_text())
    if r['pair_id']!=pair or r['configuration_sha256']!=config_sha:
        raise ValueError('Shard identity mismatch')
    for p,key in ((path,'pooled'),(raw,'raw')):
        if r[key]['bytes']!=p.stat().st_size or r[key]['sha256']!=digest(p):raise ValueError('Shard hash mismatch')
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    shapes=dict(features=(2,8,20480),pooled_prequant=(2,8,20480),original_targets=(2,8),endpoint_positions=(2,8,4))
    if set(arrays)!=set(shapes) or any(a.shape!=shapes[k] or a.dtype!=np.float32 or not np.isfinite(a).all() for k,a in arrays.items()):
        raise ValueError('Malformed pooled shard')
    if r['pooled']['entry_hashes']!={k:array_identity(a) for k,a in arrays.items()}:raise ValueError('Pooled entries differ')
    with np.load(raw,allow_pickle=False) as z:
        if z.files!=['raw']:raise ValueError('Unexpected raw-token shard entries')
        raw_array=z['raw']
        if raw_array.shape!=(2,2,256,1280) or raw_array.dtype!=np.float32 or not np.isfinite(raw_array).all():
            raise ValueError('Malformed original raw-token shard')
        if r['raw']['entry_hashes']!={'raw':array_identity(raw_array)}:raise ValueError('Raw entries differ')
    return arrays,r


def save_shard(out,pair,config_sha,arrays,raw,inputs):
    path=out/'shards'/f'{pair:06d}.npz';raw_path=out/'raw_shards'/path.name
    if any(p.exists() for p in (path,path.with_suffix('.json'),raw_path)):raise FileExistsError('Do not overwrite a shard')
    atomic_npz(path,**arrays);atomic_npz(raw_path,raw=raw)
    atomic_json(path.with_suffix('.json'),dict(pair_id=pair,configuration_sha256=config_sha,inputs=inputs,
        pooled=dict(bytes=path.stat().st_size,sha256=digest(path),entry_hashes={k:array_identity(v) for k,v in arrays.items()}),
        raw=dict(path=raw_path.relative_to(out).as_posix(),bytes=raw_path.stat().st_size,sha256=digest(raw_path),entry_hashes={'raw':array_identity(raw)})))


def worker(index,pairs,args,config_sha,reference,targets,inputs,events,cancel,begin,deadline):
    torch.set_num_threads(1);budget=WorkerBudget(deadline,cancel)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,lambda *_:cancel.set())
    key=f'worker_{index}_{os.getpid()}';path=args.out/'workers'/(key+'.json')
    receipt=dict(worker=index,pid=os.getpid(),assigned_pairs=pairs,completed_pairs=[],status='starting',
        start_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'))
    atomic_json(path,receipt)
    with (args.out/'workers'/(key+'.log')).open('a',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        try:
            budget.check();torch.cuda.set_device(index)
            free,total=torch.cuda.mem_get_info(index)
            if free<40*1024**3 or 'H800' not in torch.cuda.get_device_name(index):
                raise ValueError('Each actual worker GPU must be H800 with at least40GiB free before model load')
            receipt['before_encoder_memory']=dict(free_bytes=free,total_bytes=total)
            encoder=LocalizationEncoder(args.ijepa,args.checkpoint,device=f'cuda:{index}',grid=4)
            expected=reference['encoder_identity']
            if encoder.identity['code_commit']!=expected['source_commit'] or encoder.identity['checkpoint_sha256']!=expected['checkpoint_sha256']:
                raise ValueError('Official encoder identity changed')
            receipt.update(status='ready',encoder=encoder.identity,gpu=torch.cuda.get_device_name(index),
                visible_uuid=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[index])
            atomic_json(path,receipt);events.put(dict(event='ready',worker=index))
            while not begin.wait(.2):budget.check()
            for pair in pairs:
                budget.check();value=encode_pair(pair,encoder,targets[pair-640],inputs[str(pair)])
                budget.check();save_shard(args.out,pair,config_sha,*value)
                receipt['completed_pairs'].append(pair);receipt['status']='running';atomic_json(path,receipt)
                print(json.dumps(dict(pair=pair,images_encoded=16,raw_images_retained=4,status='complete')),flush=True)
            receipt['status']='complete'
        except BaseException as exc:
            receipt.update(status='partial',reason=str(exc));traceback.print_exc();events.put(dict(event='error',worker=index,reason=str(exc)))
        finally:
            receipt['end_unix']=time.time();atomic_json(path,receipt)
        if receipt['status']=='complete':events.put(dict(event='done',worker=index))


def extract_parallel(args,pairs,config_sha,reference,old,inputs,budget,context,cancel,deadline):
    events,begin=context.Queue(),context.Event();processes=[];success=False
    def wait(event,pending):
        pending=set(pending)
        while pending:
            budget.check()
            try:message=events.get(timeout=.25)
            except queue.Empty:
                if any(processes[i].exitcode is not None for i in pending):raise RuntimeError('Worker exited without success receipt')
                continue
            if message['event']=='error':raise RuntimeError('Worker failed: '+message['reason'])
            if message['event']!=event or message['worker'] not in pending:raise RuntimeError('Unexpected worker receipt')
            pending.remove(message['worker'])
    try:
        for index in range(8):
            process=context.Process(target=worker,args=(index,[p for p in pairs if p%8==index],args,config_sha,
                reference,old['targets'],inputs,events,cancel,begin,deadline),daemon=True)
            processes.append(process);process.start();wait('ready',{index})
        begin.set();wait('done',range(8));success=True
    finally:
        if not success:cancel.set();begin.set()
        end=time.monotonic()+5
        for process in processes:process.join(timeout=max(0,end-time.monotonic()))
        for process in processes:
            if process.is_alive():process.kill();process.join(timeout=2)
        events.close()


def merge(args,config,reference,old,slurm,budget):
    rows=[];inputs={};raw_records={}
    for pair in PAIR_IDS:
        budget.check();value=read_shard(args.out,pair,json_sha(config))
        if value is None:raise ValueError('Missing pair shard')
        arrays,receipt=value;rows.append(arrays);inputs[str(pair)]=receipt['inputs'];raw_records[str(pair)]=receipt['raw']
    if {p.name for p in (args.out/'shards').iterdir()}!={f'{p:06d}{s}' for p in PAIR_IDS for s in ('.json','.npz')}:
        raise ValueError('Unexpected pooled shard inventory')
    if {p.name for p in (args.out/'raw_shards').iterdir()}!={f'{p:06d}.npz' for p in PAIR_IDS}:
        raise ValueError('Unexpected raw shard inventory')
    arrays={k:np.stack([row[k] for row in rows]) for k in rows[0]}
    arrays.update(pair_ids=np.array(PAIR_IDS,dtype=np.int64),mask_bits=BITS,mask_names=np.array(MASK_NAMES),role=np.array('counterfactual'))
    if not np.array_equal(arrays['original_targets'],old['targets']):raise ValueError('Merged original labels changed')
    original=arrays['features'][:,:,[0,7],:].astype(np.float64);prior=old['features'].astype(np.float64);delta=original-prior
    parity=dict(bitwise_equal=bool(np.array_equal(original,prior)),max_abs=float(np.abs(delta).max()),
        rmse=float(np.sqrt(np.mean(delta**2))),relative_l2=float(np.linalg.norm(delta.ravel())/np.linalg.norm(prior.ravel())))
    cache=args.out/'cache.npz'
    if cache.exists() or (args.out/'receipt.json').exists():raise FileExistsError('Existing aggregate preserved')
    atomic_npz(cache,**arrays);budget.check()
    receipt=dict(schema=SCHEMA,role='counterfactual',cache_sha256=digest(cache),cache_bytes=cache.stat().st_size,
        entry_hashes={k:array_identity(v) for k,v in arrays.items()},configuration_sha256=json_sha(config),
        source_commit=args.source_commit,source_identity=config['source_identity'],protocol_sha256=config['protocol_sha256'],
        plan_sha256=config['plan_sha256'],registration_publication_sha256=config['registration_publication_sha256'],
        reference=reference,encoder_identity=reference['encoder_identity'],slurm=slurm,
        groups=256,episodes=512,images_encoded=4096,original_raw_images_retained=1024,
        mask_names=list(MASK_NAMES),mask_bits=BITS.tolist(),bit_order=['A','B','Y'],first_frame_mask=0,
        feature_dtype='float32',canonical_pooling='FP32 spatial adaptive average4x4 then BF16 storage then FP32',
        pooled_prequant='same-call FP32 before BF16 storage',raw_dtype='float32',raw_masks=[0,7],raw_records=raw_records,
        batch_layout=[['v0:000','v0:111','v1:000','v1:111'],['v'+str(v)+':'+MASK_NAMES[m] for v in (0,1) for m in range(1,7)]],
        original_feature_parity=parity,original_rgb_hashes_equal_archived_bridge=True,
        input_records_sha256=json_sha(inputs),outside_intervened_object_unions_unchanged=True,
        cube_edges_verified_per_episode=12,total_cube_edges_verified=512*12,
        same_call_bf16_pool_parity=True,test_opened=False,new_reader_fit=False,new_unseen_dev=False,p0_authorized=False,
        budget_owner='external single controller lease; this extractor adds no lease',deadline_unix=args.deadline_unix,
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda))
    atomic_json(args.out/'receipt.json',receipt)
    return receipt


def validate_registration(run_root,source_commit):
    """Validate local/deployed source and the new directory before any image generation."""
    run_root=Path(run_root)
    if run_root.is_symlink() or not run_root.is_dir():raise ValueError('Run root must be an existing real directory')
    run_root=run_root.resolve();st=run_root.stat()
    creation=json.loads((run_root/'creation.json').read_text())
    if (any(creation.get(k)!=v for k,v in dict(root=str(run_root),inode=st.st_ino,device=st.st_dev,uid=st.st_uid).items())
            or creation.get('preexisted') is not False or creation.get('parent_preserved') is not True
            or not isinstance(creation.get('token'),str) or not creation['token']):
        raise ValueError('New run directory creation identity mismatch')
    if re.fullmatch('[0-9a-f]{40}',source_commit) is None:raise ValueError('Frozen commit required')
    manifest=json.loads((run_root/'source_manifest.json').read_text())
    publication=json.loads((run_root/'registration_publication.json').read_text())
    sources=manifest['files']
    required={'scripts/v5_5_counterfactual.py','scripts/launch_v5_5.py','scripts/score_v5_5.py',
              'scripts/extract_v5_confirmation.py','pmr/visual_memory/features.py','pmr/visual_memory/data.py',PROTOCOL,PLAN}
    if not required.issubset(sources) or manifest['commit']!=source_commit:raise ValueError('Incomplete source manifest')
    for name,h in sources.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or digest(ROOT/p)!=h:raise ValueError('Frozen source changed: '+name)
    if (sources['pmr/visual_memory/data.py']!=ORIGINAL_DATA_SHA256 or publication.get('verified_readback') is not True
            or publication.get('source_commit')!=source_commit or publication.get('protocol_sha256')!=digest(ROOT/PROTOCOL)
            or publication.get('plan_sha256')!=digest(ROOT/PLAN) or not 0<float(publication.get('verified_unix',0))<=time.time()):
        raise ValueError('Exact original generator and published protocol/source required before scientific generation')
    return dict(source_identity=sources,protocol_sha256=digest(ROOT/PROTOCOL),plan_sha256=digest(ROOT/PLAN),
        registration_publication_sha256=digest(run_root/'registration_publication.json'),
        source_manifest_sha256=digest(run_root/'source_manifest.json'),creation_sha256=digest(run_root/'creation.json'))


def run(args):
    if args.workers!=8 or not 0<args.deadline_unix-time.time()<=540:
        raise ValueError('Eight workers and an absolute controller deadline within9minutes are required')
    if args.out.is_symlink() or args.run_root.is_symlink() or args.prior_scene.is_symlink():
        raise ValueError('Scientific scene roots cannot be symlinks')
    args.out=args.out.resolve();args.run_root=args.run_root.resolve();args.prior_scene=args.prior_scene.resolve()
    if args.out==args.prior_scene or args.prior_scene in args.out.parents or args.run_root==args.prior_scene:
        raise ValueError('Previous scene must remain read-only')
    if args.run_root not in args.out.parents:raise ValueError('Cache must be inside the separate registered run root')
    registration=validate_registration(args.run_root,args.source_commit);sources=registration['source_identity']
    context=mp.get_context('spawn');cancel=context.Event()
    deadline=time.monotonic()+args.deadline_unix-time.time();budget=WorkerBudget(deadline,cancel)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,lambda *_:cancel.set())
    old,inputs,reference=prior_inputs(args);budget.check()
    if digest(args.checkpoint)!=reference['encoder_identity']['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    config=dict(schema=SCHEMA,source_commit=args.source_commit,reference=reference,**registration,
        pair_ids=list(PAIR_IDS),mask_bits=BITS.tolist(),batch_sizes=[4,12],workers=8)
    manifest_path=args.out/'manifest.json'
    if args.out.exists() and any(p.is_symlink() for p in args.out.rglob('*')):raise ValueError('Linked output path')
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['configuration']!=config:raise ValueError('Resume configuration differs')
        if previous['status']=='complete':
            if previous['files']!=inventory(args.out):raise ValueError('Completed cache changed')
            return json.loads((args.out/'receipt.json').read_text())
    elif args.out.exists() and any(args.out.iterdir()):raise ValueError('Use a fresh cache output')
    for name in ('shards','raw_shards','workers'):(args.out/name).mkdir(parents=True,exist_ok=True)
    atomic_json(manifest_path,dict(status='partial',configuration=config))
    try:
        slurm=validate_runtime('dgx',args.account,args.partition,'H800',torch.cuda.device_count(),torch.cuda.get_device_name(0),
            allocation_job=args.allocation_job,qos=args.qos,shared_allocation_gpus=8,expected_worker_gpus=8,
            allow_gpu_sharing=args.allow_gpu_sharing,min_free_gpu_gib=40.)
        pending=[p for p in PAIR_IDS if read_shard(args.out,p,json_sha(config)) is None]
        if pending:extract_parallel(args,pending,json_sha(config),reference,old,inputs,budget,context,cancel,deadline)
        budget.check();result=merge(args,config,reference,old,slurm,budget)
        if any(digest(ROOT/name)!=h for name,h in sources.items()):raise ValueError('Source changed during extraction')
        if digest(args.prior_scene/'archive_manifest.json')!=args.prior_manifest_sha256:raise ValueError('Prior seal changed')
        files=inventory(args.out);budget.check()
        atomic_json(manifest_path,dict(status='complete',configuration=config,files=files))
        return result
    except BaseException as exc:
        atomic_json(manifest_path,dict(status='partial',configuration=config,reason=str(exc),files=inventory(args.out)))
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','run-root','prior-scene','ijepa','checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-commit','prior-manifest-sha256','allocation-job','account','partition','qos'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--workers',type=int,choices=(8,),default=8)
    p.add_argument('--allow-gpu-sharing',action='store_true')
    return p


if __name__=='__main__':print(json.dumps(run(parser().parse_args()),indent=2,allow_nan=False))
