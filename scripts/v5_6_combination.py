#!/usr/bin/env python3
"""J/X object-sign coverage readers: shared X reference, equal mean-loss weight.

No generator is called in production. FIT only consumes completed fit_cache;
confirmation queries require the complete model publication/readback receipt.
"""
import os
_THREADS=int(os.environ.get('V56_COMBINATION_THREADS','16'))
if not 1<=_THREADS<=32:raise ValueError('V56_COMBINATION_THREADS must be in1..32')
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]=str(_THREADS)
import argparse
import csv
import json
from pathlib import Path
import platform
import re
import resource
import sys
import time
import numpy as np
import v5_4_reader as old
from v5_6_data import SCHEMA as DATA_SCHEMA, X_VIEW_INDICES, J_VIEW_INDICES, UNIQUE_ORDER, PATHS

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='v5.6-reader-freeze-1'
ALPHA=7.8125e-7
ATOL=1e-7
V54_SEAL='ec7bc9c17da8417e9ecd11ba0f21415a2df0ee7af620af9d12f911c3ab26e75c'
V55_SEAL='b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663'
SOURCE_NAMES=('scripts/v5_6_combination.py','scripts/v5_6_data.py','scripts/v5_6_extract.py',
    'scripts/v5_4_reader.py','scripts/v5_3_reader.py','scripts/v5_4_data.py','scripts/v5_3_data.py',
    'scripts/v5_6_plan.json','docs/V5_6_EXECUTION_PROTOCOL.md')


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def record(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():raise ValueError('Regular input required: '+str(p))
    return dict(bytes=p.stat().st_size,sha256=old.sha256(p))


def sources():return {name:old.sha256(ROOT/name) for name in SOURCE_NAMES}


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def config():
    return dict(schema=SCHEMA,input_schema=DATA_SCHEMA,kernel='quadratic',input='concat',
        mean_loss_alpha=ALPHA,reference='X8192 complete FIT views',arms=['J','X'],
        conceptual_rows=dict(J=8192,X=8192),solved_rows=dict(J=4096,X=8192),
        ridge_diagonal=dict(J=.0032,X=.0064),worlds=256,views_per_world=32,
        j_collapse='two exact copies of each old16 view; uniform mean loss equals4096 unique rows with4096*alpha',
        free_intercept=True,shared_normalization=True,shared_scoring_baseline='unchanged V5.4 frozen joint-FIT mean and motion RMS',
        primary='fresh confirmation all32 views paired J-minus-X dxMSE; no automatic pass gate',
        unique_order=list(UNIQUE_ORDER),x_view_indices=X_VIEW_INDICES.tolist(),j_view_indices=J_VIEW_INDICES.tolist(),
        cv=False,selection=False,regularization_search=False,test_opened=False,p0_authorized=False,
        scientific_scope='object-sign combinations; neither independent within-object axes nor memory claims',
        bootstrap=dict(seed=7816,replicates=10000,unit='whole_world'),numerical_atol=ATOL)


def registered(path,source_commit=None):
    p=Path(path);data=old.read_json(p)
    if data.get('verified_readback') is not True:raise ValueError('Source publication readback required')
    commit=data.get('source_commit',data.get('commit'))
    if re.fullmatch('[0-9a-f]{40}',str(commit)) is None or (source_commit and commit!=source_commit):
        raise ValueError('Source commit mismatch')
    for name,h in sources().items():
        v=data.get('files',{}).get(name)
        if (v.get('sha256') if isinstance(v,dict) else v)!=h:raise ValueError('Unpublished source: '+name)
    return data,dict(**record(p),source_commit=commit)


def sealed(scene,seal,names):
    scene=Path(scene)
    if scene.is_symlink() or old.sha256(scene/'archive_manifest.json')!=seal:raise ValueError('Prior scene identity differs')
    manifest=old.read_json(scene/'archive_manifest.json');checks={}
    for name in names:
        checks[name]=record(scene/name)
        if checks[name]!=manifest['files'][name]:raise ValueError('Prior file changed: '+name)
    return checks


def validate_arrays(data,role):
    if role not in ('fit','confirm','axis'):raise ValueError('Unknown new cache role')
    n=256 if role in ('fit','axis') else 64
    expected={'features','targets','role','pair_ids','object_condition_indices'} if role=='axis' else {'features','targets','j_targets','role','block_ids','x_view_indices','j_view_indices'}
    if set(data)!=expected or str(data['role'].item())!=role:raise ValueError('Cache schema/role mismatch')
    shapes={'features':(n,2,7,20480),'targets':(n,2,7,8)} if role=='axis' else {'features':(n,10,20480),'targets':(n,32,8),'j_targets':(n,32,8)}
    for key,shape in shapes.items():
        if data[key].shape!=shape or data[key].dtype!=np.float32 or not np.isfinite(data[key]).all():raise ValueError('Shape/dtype/finite: '+key)
    if not np.array_equal(data['pair_ids' if role=='axis' else 'block_ids'],np.arange(640,896) if role=='axis' else np.arange(n)):
        raise ValueError('Exact registered IDs required')
    if role=='axis':
        if not np.array_equal(data['object_condition_indices'],[[0,1,2,3],[0,4,5,6]]):raise ValueError('Axis order changed')
    else:
        if not np.array_equal(data['x_view_indices'],X_VIEW_INDICES) or not np.array_equal(data['j_view_indices'],J_VIEW_INDICES):raise ValueError('View mapping changed')
        validate_labels(data['targets'],data['j_targets'])


def validate_labels(x,j):
    if not np.array_equal(j[:,:16],j[:,16:]) or not np.array_equal(x[:,:16],j[:,:16]):raise ValueError('J duplicated labels changed')
    if not np.array_equal(x.reshape(len(x),8,4,8),np.repeat(x.reshape(len(x),8,4,8)[:,:,:1],4,axis=2)):
        raise ValueError('Yellow paths changed labels')
    paths=np.tile(PATHS,(8,1))
    for i in range(len(x)):
        for obj in (0,1):
            cols=[obj*2,obj*2+1,obj*2+4,obj*2+5]
            xv=np.concatenate((x[i,:,cols].T,paths),1);jv=np.concatenate((j[i,:,cols].T,paths),1)
            if sorted(map(tuple,xv.tolist()))!=sorted(map(tuple,jv.tolist())):raise ValueError('Per-object target/path margins differ')


def cache(folder,role,frozen=None,freeze_publication=None):
    folder=Path(folder);receipt=old.read_json(folder/'receipt.json');manifest=old.read_json(folder/'manifest.json')
    if manifest.get('status')!='complete' or receipt.get('schema')!=DATA_SCHEMA or receipt.get('role')!=role or receipt.get('test_opened') is not False:
        raise ValueError('Complete registered cache required')
    if receipt.get('plan_sha256')!=old.sha256(ROOT/'scripts/v5_6_plan.json') or receipt.get('protocol_sha256')!=old.sha256(ROOT/'docs/V5_6_EXECUTION_PROTOCOL.md'):
        raise ValueError('Cache plan/protocol changed')
    mapping=receipt.get('source_identity',{})
    if not set(SOURCE_NAMES).issubset(mapping):raise ValueError('Missing complete registered source')
    for name,h in mapping.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or old.sha256(ROOT/p)!=h:raise ValueError('Cache source differs')
    aggregate=folder/'cache.npz'
    if record(aggregate)!=dict(bytes=receipt['cache_bytes'],sha256=receipt['cache_sha256']):raise ValueError('Aggregate cache changed')
    actual={}
    for p in folder.rglob('*'):
        if p.is_symlink():raise ValueError('Cache symlink')
        if p.is_file() and p!=folder/'manifest.json':actual[p.relative_to(folder).as_posix()]=record(p)
    if actual!=manifest['files']:raise ValueError('Exact complete cache inventory differs')
    data=read_npz(aggregate);validate_arrays(data,role)
    if receipt['entry_hashes']!={k:old.entry_identity(v) for k,v in data.items()}:raise ValueError('Cache array identity differs')
    if receipt['feature_dtype']!='float32' or receipt['pooling']!='4x4_flattened' or receipt['encoder_batch_size']!=(6 if role=='fit' else 4):
        raise ValueError('Representation contract changed')
    if frozen is not None:
        if receipt['source_commit']!=frozen['source_commit'] or receipt['encoder_identity']!=frozen['encoder_identity']:
            raise ValueError('Evaluation identity differs from frozen reader')
    if role=='confirm':
        if frozen is None or freeze_publication is None:raise ValueError('Confirmation requires reader freeze publication')
        p=Path(freeze_publication);pub=old.read_json(p)
        if (pub.get('verified_readback') is not True or pub.get('frozen_reader_sha256')!=frozen['receipt_sha256'] or
            pub.get('source_commit')!=frozen['source_commit'] or pub.get('plan_sha256')!=frozen['plan_sha256'] or
            not frozen['created_unix']<=pub.get('verified_unix',0)):
            raise ValueError('Reader freeze was not published/read back')
        if (receipt.get('frozen_reader_sha256')!=frozen['receipt_sha256'] or receipt.get('freeze_publication_sha256')!=old.sha256(p) or
            receipt.get('frozen_before_generation') is not True or receipt.get('unseen_before_release') is not True):
            raise ValueError('Confirmation not released after this exact model freeze')
    return data,receipt


def indices(blocks,local=X_VIEW_INDICES):
    return (np.arange(blocks)[:,None,None]*10+np.asarray(local)[None]).reshape(-1,2)


def train_math(features,targets,jtargets,mean,motion_rms,verbose=True):
    validate_labels(targets,jtargets);n=len(features);idx=indices(n);jrows=(np.arange(n)[:,None]*32+np.arange(16)).ravel()
    flat=features.reshape(-1,features.shape[-1]);y=targets.reshape(-1,8).astype(np.float64)
    started=time.time();unique=old.dot_gram(flat,flat)
    shared,state=old.shared_kernel(old.view_gram(unique,idx,idx,'concat'));del unique
    arrays=dict(fit_features=features,fit_targets=targets,fit_j_targets=jtargets,fit_view_indices=idx,
        J_rows=jrows,X_rows=np.arange(n*32),shared_target_mean=np.asarray(mean),fit_motion_rms=np.asarray(motion_rms))
    arrays.update({'concat__'+key:value for key,value in state.items()});audits={}
    if verbose:print(json.dumps(dict(stage='shared_kernel',rows=len(idx),seconds=time.time()-started)),flush=True)
    for arm,rows in [('J',jrows),('X',np.arange(n*32))]:
        started=time.time();coef,bias=old.ridge_fit(shared,y,rows,ALPHA)
        residual=float(np.max(np.abs(shared[np.ix_(rows,rows)]@coef+bias+len(rows)*ALPHA*coef-y[rows])))
        constraint=float(np.max(np.abs(coef.sum(0))))
        if not np.isfinite([residual,constraint]).all() or max(residual,constraint)>ATOL:raise ValueError('Free-intercept normal equations failed')
        arrays[arm+'__coefficients']=coef;arrays[arm+'__intercept']=bias
        arrays[arm+'__target_mean']=y[rows].mean(0)
        audits[arm]=dict(conceptual_supervised_rows=n*32,solved_rows=len(rows),diagonal=len(rows)*ALPHA,
            normal_equation_max_abs=residual,coefficient_sum_max_abs=constraint,elapsed_seconds=time.time()-started)
        if verbose:print(json.dumps(dict(stage='fit_arm',arm=arm,**audits[arm])),flush=True)
    return arrays,audits


def predict(model,query,index,chunk=128):
    query=np.asarray(query);index=np.asarray(index)
    width=model['fit_features'].shape[-1]
    if query.ndim!=2 or query.shape[1]!=width or index.ndim!=2 or index.shape[1]!=2 or not np.isfinite(query).all():raise ValueError('Feature-only query schema')
    if index.size and (index.min()<0 or index.max()>=len(query)):raise ValueError('Query index outside features')
    flat=model['fit_features'].reshape(-1,width);unique=old.dot_gram(query,flat)
    state={k:model['concat__'+k] for k in ('linear_columns','linear_grand','linear_scale','quadratic_columns','quadratic_grand','quadratic_scale')}
    out={arm:np.empty((len(index),8)) for arm in ('J','X')}
    for lo in range(0,len(index),chunk):
        cross=old.view_gram(unique,index[lo:lo+chunk],model['fit_view_indices'],'concat')
        kernel=old.apply_shared_kernel(cross,state)
        for arm in out:out[arm][lo:lo+len(cross)]=kernel[:,model[arm+'_rows']]@model[arm+'__coefficients']+model[arm+'__intercept']
    if any(not np.isfinite(v).all() for v in out.values()):raise ValueError('Nonfinite predictions')
    return out


def self_test():
    from v5_6_data import make_block
    blocks=[make_block('software_test',i) for i in range(3)]
    rng=np.random.default_rng(56117);features=rng.normal(size=(3,10,16)).astype(np.float32)
    y=np.stack([b.x_targets for b in blocks]);j=np.stack([b.j_targets for b in blocks]);mean=y.reshape(-1,8).mean(0)
    model,audits=train_math(features,y,j,mean,np.sqrt(np.mean(y.reshape(-1,8)[:,4:]**2,0)),verbose=False)
    flat=features.reshape(-1,16);idx=indices(3);g,state=old.shared_kernel(old.view_gram(old.dot_gram(flat,flat),idx,idx,'concat'))
    duplicated=model['J_rows'].reshape(3,16);duplicated=np.tile(duplicated,(1,2)).ravel()
    c,b=old.ridge_fit(g,y.reshape(-1,8),duplicated,ALPHA)
    q=rng.normal(size=(8,16)).astype(np.float32);qi=np.arange(8).reshape(-1,2);p=predict(model,q,qi,chunk=2)
    k=old.apply_shared_kernel(old.view_gram(old.dot_gram(q,flat),qi,idx,'concat'),state)
    expected=k[:,duplicated]@c+b;collapse=float(np.max(np.abs(p['J']-expected)))
    if collapse>ATOL:raise ValueError('J duplicate mean-loss equivalence failed')
    # Independent explicit feature map verifies X reference/query centering.
    z=flat[idx].reshape(len(idx),-1).astype(np.float64);qz=q[qi].reshape(len(qi),-1).astype(np.float64)
    mu=z.mean(0);z-=mu;qz-=mu;scale=np.sqrt(np.mean(np.sum(z*z,1)));z/=scale;qz/=scale
    def phi(v):return np.concatenate((np.ones((len(v),1)),np.sqrt(2)*v,(v[:,:,None]*v[:,None,:]).reshape(len(v),-1)),1)
    zz=phi(z);qq=phi(qz);m=zz.mean(0);zz-=m;qq-=m;s=np.sqrt(np.mean(np.sum(zz*zz,1)));zz/=s;qq/=s
    np.testing.assert_allclose(k,qq@zz.T,atol=1e-12,rtol=0)
    for arm in ('J','X'):
        rows=model[arm+'_rows'];kg=zz[rows]@zz[rows].T;n=len(rows)
        augmented=np.block([[kg+n*ALPHA*np.eye(n),np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
        solution=np.linalg.solve(augmented,np.vstack((y.reshape(-1,8)[rows],np.zeros((1,8)))))
        np.testing.assert_allclose(p[arm],qq@zz[rows].T@solution[:-1]+solution[-1],atol=ATOL,rtol=0)
    zero=diagnostics(np.zeros((2,2,8)),np.zeros((2,2,8)),np.zeros(8))
    if zero['scores']['motion_skill'] is not None:raise ValueError('Zero denominator must be null')
    return dict(status='passed',software_only=True,seed=56117,blocks=3,j_duplicate_prediction_max_abs=collapse,
        explicit_polynomial_query_max_abs=float(np.max(np.abs(k-qq@zz.T))),numerical_checks=audits,
        source_identity={n:old.sha256(ROOT/n) for n in SOURCE_NAMES if (ROOT/n).exists()})


def fit(args):
    start=time.time();pub,pubrecord=registered(args.publication,args.source_commit);src=sources()
    for source in (args.prior_scene,args.cache):
        if args.out.resolve()==source.resolve() or source.resolve() in args.out.resolve().parents:
            raise ValueError('Input scene/cache must remain read-only')
    data,receipt=cache(args.cache,'fit')
    if receipt['source_commit']!=args.source_commit:raise ValueError('FIT source commit differs')
    if receipt['registration_publication_sha256']!=pubrecord['sha256']:raise ValueError('FIT source publication differs')
    checked=sealed(args.prior_scene,V54_SEAL,('reader/frozen_model.npz','reader/freeze_receipt.json','reader/training_config.json','freeze_publication.json'))
    baseline,prior=old.load_frozen(args.prior_scene/'reader');old.validate_publication(args.prior_scene/'freeze_publication.json',prior)
    if receipt['encoder_identity']!=prior['cache_identity']['encoder_identity']:raise ValueError('Encoder changed')
    # New ten-image FIT extends exactly the archived W six-image FIT.
    np.testing.assert_array_equal(data['features'][:,:6],baseline['fit_features'][256:])
    np.testing.assert_array_equal(data['targets'][:,:16],baseline['fit_targets'][256:])
    software=self_test();args.out.mkdir(parents=True,exist_ok=False)
    model,audit=train_math(data['features'],data['targets'],data['j_targets'],baseline['shared_target_mean'],baseline['fit_motion_rms'])
    np.savez_compressed(args.out/'frozen_model.npz',**model)
    save(args.out/'training_config.json',config());save(args.out/'software_verification.json',software);save(args.out/'numerical_checks.json',audit)
    if sources()!=src:raise ValueError('Source changed during FIT')
    receipt_out=dict(schema=SCHEMA,status='frozen_before_confirmation',source_commit=args.source_commit,
        plan_sha256=old.sha256(ROOT/'scripts/v5_6_plan.json'),protocol_sha256=old.sha256(ROOT/'docs/V5_6_EXECUTION_PROTOCOL.md'),
        source_identity=src,fit_cache_sha256=receipt['cache_sha256'],fit_receipt_sha256=old.sha256(args.cache/'receipt.json'),
        fit_manifest_sha256=old.sha256(args.cache/'manifest.json'),encoder_identity=receipt['encoder_identity'],
        fit_source_receipt_utf8=(args.cache/'receipt.json').read_text(),fit_publication=pubrecord,
        prior_manifest_sha256=V54_SEAL,prior_inputs=checked,prior_frozen_reader_sha256=prior['receipt_sha256'],
        configuration=config(),fit_worlds=256,conceptual_rows=dict(J=8192,X=8192),reference_rows=8192,
        files={p.name:old.sha256(p) for p in sorted(args.out.iterdir()) if p.is_file()},
        model_entry_hashes={k:old.entry_identity(v) for k,v in model.items()},created_unix=time.time(),
        no_confirmation_read=True,test_opened=False,p0_authorized=False,elapsed_seconds=time.time()-start,
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),blas_threads=_THREADS,
                         peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),command=sys.argv)
    save(args.out/'freeze_receipt.json',receipt_out);restored,_=load_frozen(args.out)
    for k in model:np.testing.assert_array_equal(restored[k],model[k])
    return dict(status=receipt_out['status'],receipt_sha256=old.sha256(args.out/'freeze_receipt.json'),numerical_checks=audit,elapsed_seconds=receipt_out['elapsed_seconds'])


def load_frozen(folder):
    folder=Path(folder);r=old.read_json(folder/'freeze_receipt.json')
    if (r.get('schema')!=SCHEMA or r.get('status')!='frozen_before_confirmation' or r.get('source_identity')!=sources() or
        r.get('configuration')!=config() or r.get('no_confirmation_read') is not True or r.get('test_opened') is not False):
        raise ValueError('Frozen reader contract differs')
    actual={p.name:old.sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='freeze_receipt.json'}
    if actual!=r['files']:raise ValueError('Frozen reader file inventory differs')
    arrays=read_npz(folder/'frozen_model.npz')
    if r['model_entry_hashes']!={k:old.entry_identity(v) for k,v in arrays.items()}:raise ValueError('Frozen model entry differs')
    if arrays['fit_features'].shape!=(256,10,20480) or arrays['fit_targets'].shape!=(256,32,8):raise ValueError('Frozen FIT shape')
    r['receipt_sha256']=old.sha256(folder/'freeze_receipt.json');return arrays,r


def paired_interval(delta):
    v=np.asarray(delta).reshape(len(delta),-1).mean(1);rng=np.random.default_rng(7816);boot=np.empty(10000)
    for lo in range(0,10000,250):boot[lo:lo+250]=v[rng.integers(0,len(v),(250,len(v)))].mean(1)
    return dict(J_minus_X_dx_mse=float(v.mean()),paired_95ci=np.quantile(boot,[.025,.975]).tolist(),worlds=len(v))


def diagnostics(p,y,mean):
    p=np.asarray(p);y=np.asarray(y);motion=p[...,4:];truth=y[...,4:];nonzero=truth!=0
    axes=tuple(range(motion.ndim-1));den=np.mean(np.abs(truth),axis=axes)
    pp=p.reshape(-1,8);yy=y.reshape(-1,8);mse=np.mean((pp-yy)**2,axis=0);baseline=np.mean((yy-mean)**2,axis=0)
    skill=[float(1-v/d) if d>0 else None for v,d in zip(mse,baseline)]
    aggregate=lambda values:float(np.mean(values)) if all(v is not None for v in values) else None
    motion_skill=aggregate(skill[4:])
    scores=dict(per_dimension_mse=mse.tolist(),per_dimension_skill=skill,
        position_skill=aggregate(skill[:4]),motion_skill=motion_skill,
        motion_nmse=1-motion_skill if motion_skill is not None else None,
        dx_mse=float(mse[[4,6]].mean()),dx_rmse_pixels=float(np.sqrt(mse[[4,6]].mean())*224),
        degenerate_score_axes=[i for i,d in enumerate(baseline) if d<=0])
    return dict(scores=scores,
        direction_errors=np.sum((np.sign(motion)!=np.sign(truth))&nonzero,axis=axes).tolist(),
        nonzero_targets=np.sum(nonzero,axis=axes).tolist(),
        amplitude_ratio=[float(v/d) if d else None for v,d in zip(np.mean(np.abs(motion),axis=axes),den)],
        motion_response_rms=np.sqrt(np.mean(motion*motion,axis=axes)).tolist())


def score(args):
    start=time.time();model,frozen=load_frozen(args.model);pub,pubrec=registered(args.publication,frozen['source_commit'])
    for source in (args.cache,args.model):
        if args.out.resolve()==source.resolve() or source.resolve() in args.out.resolve().parents:
            raise ValueError('Input cache/model must remain read-only')
    inputs={};negative=None;known=False
    if args.role=='known':
        scene=args.cache;inputs=sealed(scene,V55_SEAL,('counterfactual_cache/cache.npz','counterfactual_cache/receipt.json','report/predictions.npz'))
        data=read_npz(scene/'counterfactual_cache/cache.npz');old_receipt=old.read_json(scene/'counterfactual_cache/receipt.json')
        if old_receipt['entry_hashes']!={k:old.entry_identity(v) for k,v in data.items()}:raise ValueError('Known-cache entries changed')
        x=data['features'].reshape(-1,20480);idx=np.stack((np.repeat(np.arange(512)*8,8),np.arange(4096)),1)
        target=np.empty((256,2,8,8),dtype=np.float64);target[...,:4]=data['endpoint_positions']
        for m in range(8):
            target[...,m,4:6]=data['original_targets'][...,4:6]*bool(m&4);target[...,m,6:8]=data['original_targets'][...,6:8]*bool(m&2)
        archived=read_npz(scene/'report/predictions.npz');np.testing.assert_array_equal(target,archived['targets'])
        shape=(256,2,8,8);group_ids=data['pair_ids'];known=True
    else:
        data,receipt=cache(args.cache,args.role,frozen,args.freeze_publication)
        if receipt['registration_publication_sha256']!=pubrec['sha256']:raise ValueError('Evaluation source publication differs')
        inputs={name:record(args.cache/name) for name in ('cache.npz','receipt.json','manifest.json')}
        x=data['features'].reshape(-1,20480);target=data['targets'];group_ids=data['block_ids' if args.role=='confirm' else 'pair_ids']
        if args.role=='confirm':idx=indices(64);shape=(64,32,8)
        else:idx=np.stack((np.repeat(np.arange(512)*7,7),np.arange(3584)),1);shape=(256,2,7,8)
    args.out.mkdir(parents=True,exist_ok=False);prediction={k:v.reshape(shape) for k,v in predict(model,x,idx).items()}
    if args.role=='confirm':
        neg_idx=(np.arange(64)[:,None,None]*10+np.array([[2,3],[3,2]])[None]).reshape(-1,2)
        negative={k:v.reshape(64,2,8) for k,v in predict(model,x,neg_idx).items()}
        neg_y=np.zeros((64,2,8),dtype=np.float64);neg_y[...,:4]=target[:,None,0,:4]
    arrays=dict(targets=target,group_ids=group_ids,**prediction)
    if negative is not None:arrays.update({k+'_negative':v for k,v in negative.items()});arrays['negative_targets']=neg_y
    np.savez_compressed(args.out/'predictions.npz',**arrays)
    mean=model['shared_target_mean'];summary={arm:diagnostics(p,target,mean) for arm,p in prediction.items()}
    comparisons={};subsets={}
    if args.role=='confirm':subsets=dict(all_views=np.arange(32),old_joint_views=np.arange(16),mixed_object_views=np.arange(16,32),**{'path_'+str(i):np.arange(i,32,4) for i in range(4)})
    else:subsets={'condition_'+str(m):np.array([m]) for m in range(8 if known else 7)}
    for name,rows in subsets.items():
        yy=target[:,rows] if args.role=='confirm' else target[:,:,rows]
        pp={k:(p[:,rows] if args.role=='confirm' else p[:,:,rows]) for k,p in prediction.items()}
        comparisons[name]=dict(readers={k:diagnostics(p,yy,mean) for k,p in pp.items()},
            comparison=paired_interval(((pp['J']-yy)**2-(pp['X']-yy)**2)[...,[4,6]]))
    neg_summary=None
    if negative is not None:
        neg_summary={k:dict(**diagnostics(p,neg_y,mean),motion_rms_over_frozen_fit=(np.sqrt(np.mean(p[...,4:]**2,axis=(0,1)))/model['fit_motion_rms']).tolist()) for k,p in negative.items()}
    with (args.out/'per_example.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['world_id','flat_view','arm']+['true_'+str(i) for i in range(8)]+['pred_'+str(i) for i in range(8)])
        for i,gid in enumerate(group_ids):
            yy=target[i].reshape(-1,8)
            for arm,p in prediction.items():
                for j,(truth,pred) in enumerate(zip(yy,p[i].reshape(-1,8))):w.writerow([int(gid),j,arm]+truth.tolist()+pred.tolist())
    report=dict(schema='v5.6-combination-report-1',status='complete',role=args.role,
        groups=len(group_ids),views_per_group=int(np.prod(shape[1:-1])),readers=summary,subsets=comparisons,
        negative_controls=neg_summary,shared_score_mean=mean.tolist(),configuration=config(),
        source_identity=sources(),source_commit=frozen['source_commit'],plan_sha256=frozen['plan_sha256'],
        frozen_reader_sha256=frozen['receipt_sha256'],inputs=inputs,publication=pubrec,
        fresh_confirmation=args.role=='confirm',known_dev_posthoc=args.role!='confirm',
        limits='Object-sign combinations are varied; within-object x/y remain coupled in confirmation. No memory or automatic P0 claim.',
        test_opened=False,p0_authorized=False,elapsed_seconds=time.time()-start)
    save(args.out/'report.json',report);save(args.out/'manifest.json',{p.name:record(p) for p in args.out.iterdir() if p.is_file()})
    return dict(status='complete',role=args.role,groups=len(group_ids),elapsed_seconds=report['elapsed_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True);sub.add_parser('self-test')
    f=sub.add_parser('fit')
    for name in ('cache','prior-scene','out','publication'):f.add_argument('--'+name,type=Path,required=True)
    f.add_argument('--source-commit',required=True)
    s=sub.add_parser('score');s.add_argument('--role',choices=('confirm','axis','known'),required=True)
    for name in ('cache','model','out','publication'):s.add_argument('--'+name,type=Path,required=True)
    s.add_argument('--freeze-publication',type=Path)
    args=p.parse_args();result=self_test() if args.phase=='self-test' else fit(args) if args.phase=='fit' else score(args)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
