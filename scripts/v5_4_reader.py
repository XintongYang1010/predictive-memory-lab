#!/usr/bin/env python3
"""Joint-reference narrow/wide coverage KRR; FIT-only training, frozen evaluation."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key,'4')
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time
import numpy as np

from v5_4_data import (SCHEMA, VIEW_INDICES, STATIC_VIEW_INDICES, NEGATIVE_INDICES,
                       UNIQUE_ORDER, PATH_NAMES)
# Only pure numerical and serialization helpers are reused. V5.3 production
# identity validators and its S/B or single-frame model paths are never called.
from v5_3_reader import (sha256, read_json, write_json, entry_identity, check_targets,
    block_indices, dot_gram, view_gram, shared_kernel, apply_shared_kernel,
    ridge_fit, score_outputs, bootstrap_weights, bootstrap_mean, validate_publication)

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'scripts/v5_4_plan.json'
ARMS=('N','W')
ALPHA=7.8125e-7
BOOTSTRAPS=10000
BOOT_SEED=7811
FREEZE_SCHEMA='v5.4-reader-freeze-1'
IDENTITY_KEYS=('plan_sha256','source_commit','encoder_identity','source_identity','pooling','feature_dtype')
SOURCE_NAMES=('scripts/v5_4_reader.py','scripts/v5_4_data.py','scripts/v5_4_plan.json',
              'scripts/v5_3_reader.py','scripts/v5_3_data.py')


def sources():
    return {name:sha256(ROOT/name) for name in SOURCE_NAMES}


def validate_identity(receipt):
    if any(k not in receipt for k in IDENTITY_KEYS):
        raise ValueError('Missing V5.4 cache identity')
    identity={k:receipt[k] for k in IDENTITY_KEYS}
    if re.fullmatch('[0-9a-f]{40}',str(identity['source_commit'])) is None:
        raise ValueError('Frozen source commit required')
    if identity['plan_sha256']!=sha256(PLAN):
        raise ValueError('Plan SHA mismatch')
    plan=read_json(PLAN)
    if identity['encoder_identity']!={k:plan['encoder'][k] for k in ('source_commit','checkpoint_sha256','model')}:
        raise ValueError('Encoder identity mismatch')
    if identity['pooling']!='4x4_flattened' or identity['feature_dtype']!='float32':
        raise ValueError('Pooling/precision mismatch')
    required=set(SOURCE_NAMES)|{'scripts/extract_v5_4.py','scripts/launch_v5_4.py',
        'scripts/v5_temporal_confirmation.py','docs/V5_4_EXECUTION_PROTOCOL.md'}
    required|={p.relative_to(ROOT).as_posix() for p in (ROOT/'pmr/visual_memory').glob('*.py')}
    mapping=identity['source_identity']
    if not isinstance(mapping,dict) or not required.issubset(mapping):
        raise ValueError('Required source identity missing')
    for name,digest in mapping.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha256(ROOT/path)!=digest:
            raise ValueError('Source changed: '+name)
    return identity


def validate_narrow_reference(reference):
    expected=read_json(PLAN)['narrow_reference']
    if reference!=expected:
        raise ValueError('Narrow reference differs from registered immutable V5.3 cache')
    if set(reference)!={'cache_sha256','receipt_sha256','manifest_sha256','source_commit','encoder_identity'}:
        raise ValueError('Incomplete narrow reference')
    for key in ('cache_sha256','receipt_sha256','manifest_sha256'):
        if re.fullmatch('[0-9a-f]{64}',str(reference[key])) is None:
            raise ValueError('Unpinned narrow reference: '+key)
    if reference['encoder_identity']!={k:read_json(PLAN)['encoder'][k] for k in ('source_commit','checkpoint_sha256','model')}:
        raise ValueError('N/W encoder identity differs')
    return reference


def validate_arrays(data,role):
    if role not in ('fit','anchor','bridge'):
        raise ValueError('Only fit, anchor and bridge roles are allowed')
    keys={'features','targets','role','pair_ids'} if role=='bridge' else {'features','targets','negative_targets','block_ids','role'}
    if set(data)!=keys or data['role'].shape!=() or str(data['role'].item())!=role:
        raise ValueError('Cache entries or role mismatch')
    x,y=data['features'],data['targets']
    if (x.shape!=((256,2,2,20480) if role=='bridge' else (256,6,20480)) or
        y.shape!=((256,2,8) if role=='bridge' else (256,16,8)) or x.dtype!=np.float32 or y.dtype!=np.float32):
        raise ValueError('Exactly 256 worlds, registered feature width and float32 required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite cache')
    ids=data['pair_ids' if role=='bridge' else 'block_ids']
    if (not np.issubdtype(ids.dtype,np.integer) or
            not np.array_equal(ids,np.arange(640,896) if role=='bridge' else np.arange(256))):
        raise ValueError('Exact unique ordered role IDs required')
    if role!='bridge':check_targets(y,data['negative_targets'])


def validate_mapping(receipt):
    for key,value in (('view_indices',VIEW_INDICES.tolist()),('static_view_indices',STATIC_VIEW_INDICES.tolist()),
                      ('negative_indices',NEGATIVE_INDICES.tolist()),('unique_order',list(UNIQUE_ORDER))):
        if receipt.get(key)!=value:raise ValueError('View mapping mismatch: '+key)


def read_arrays(cache,receipt,role):
    if receipt.get('cache_sha256')!=sha256(cache):raise ValueError('Cache SHA mismatch')
    if receipt.get('cache_bytes')!=Path(cache).stat().st_size:raise ValueError('Cache byte count mismatch')
    with np.load(cache,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    validate_arrays(data,role)
    if receipt.get('entry_hashes')!={k:entry_identity(v) for k,v in data.items()}:
        raise ValueError('Cache entry hash/dtype/shape mismatch')
    return data


def load_narrow(cache,receipt_path,manifest_path,reference):
    """Historical provenance is pinned; never compare its old plan to V5.4."""
    validate_narrow_reference(reference)
    for path,key in ((cache,'cache_sha256'),(receipt_path,'receipt_sha256'),(manifest_path,'manifest_sha256')):
        if sha256(path)!=reference[key]:raise ValueError('Pinned narrow artifact changed: '+key)
    receipt,manifest=read_json(receipt_path),read_json(manifest_path)
    if (receipt.get('schema')!='v5.3-nuisance-repair-1' or receipt.get('role')!='fit' or
        receipt.get('test_opened') is not False or receipt.get('source_commit')!=reference['source_commit'] or
        receipt.get('encoder_identity')!=reference['encoder_identity'] or receipt.get('pooling')!='4x4_flattened' or
        receipt.get('feature_dtype')!='float32' or manifest.get('source_commit')!=reference['source_commit']):
        raise ValueError('Narrow role/source/encoder provenance mismatch')
    for path,name in ((cache,'fit_cache/cache.npz'),(receipt_path,'fit_cache/receipt.json')):
        if manifest.get('files',{}).get(name)!={'sha256':sha256(path),'bytes':Path(path).stat().st_size}:
            raise ValueError('Narrow manifest membership mismatch: '+name)
    validate_mapping(receipt)
    return read_arrays(cache,receipt,'fit'),receipt


def load_cache(cache,receipt_path,role,frozen=None,publication=None):
    receipt=read_json(receipt_path)
    if receipt.get('schema')!=SCHEMA or receipt.get('role')!=role or receipt.get('test_opened') is not False:
        raise ValueError('V5.4 role/schema/Test declaration mismatch')
    identity=validate_identity(receipt)
    validate_narrow_reference(receipt.get('narrow_reference'))
    validate_mapping(receipt)
    for key in ('freshness_audit_sha256','input_records_sha256'):
        if re.fullmatch('[0-9a-f]{64}',str(receipt.get(key,''))) is None:raise ValueError('Missing '+key)
    if role=='fit':
        if frozen is not None or publication is not None:raise ValueError('FIT cannot consume evaluation release metadata')
    elif role in ('anchor','bridge'):
        if frozen is None or publication is None:raise ValueError('Evaluation requires frozen model and verified publication')
        publication_sha=validate_publication(publication,frozen)
        if (identity!=frozen['cache_identity'] or receipt['narrow_reference']!=frozen['narrow_reference'] or
            receipt['freshness_audit_sha256']!=frozen['freshness_audit_sha256'] or
            receipt.get('frozen_reader_sha256')!=frozen['receipt_sha256'] or
            receipt.get('freeze_publication_sha256')!=publication_sha or
            receipt.get('frozen_before_generation') is not True or receipt.get('unseen_before_release') is not True):
            raise ValueError('Evaluation cache not generated after this published freeze')
    else:raise ValueError('No FIT/Test or unknown evaluation role')
    return read_arrays(cache,receipt,role),receipt


def train_models(narrow_features,narrow_targets,wide_features,wide_targets,block_ids):
    """Software-sized inputs allowed here; scientific cache loaders enforce 256."""
    n,w=np.asarray(narrow_features),np.asarray(wide_features)
    yn,yw=np.asarray(narrow_targets),np.asarray(wide_targets)
    ids=np.asarray(block_ids)
    if (n.ndim!=3 or n.shape[1]!=6 or n.shape!=w.shape or len(n)==0 or
        yn.shape!=(len(n),16,8) or yw.shape!=yn.shape or ids.shape!=(len(n),) or
        len(np.unique(ids))!=len(ids)):
        raise ValueError('Matched complete six-image/sixteen-view worlds required')
    if any(not np.isfinite(a).all() for a in (n,w,yn,yw)):raise ValueError('Finite FIT arrays required')
    check_targets(yn);check_targets(yw)
    if not np.array_equal(yn[:,:,4:],yw[:,:,4:]):raise ValueError('Paired N/W FIT motion labels must be bitwise equal')
    x=np.concatenate((n,w)).astype(np.float32)
    y=np.concatenate((yn,yw)).astype(np.float32)
    index=block_indices(len(x));count=len(n)*16
    flat=y.reshape(-1,8).astype(np.float64)
    rows={'N':np.arange(count),'W':np.arange(count,2*count)}
    arrays=dict(fit_features=x,fit_targets=y,fit_block_ids=ids.astype(np.int64),fit_view_indices=index,
        shared_target_mean=flat.mean(0),fit_motion_rms=np.sqrt(np.mean(flat[:,4:]**2,axis=0)),
        mean_loss_alpha=np.array(ALPHA),N_rows=rows['N'],W_rows=rows['W'])
    raw=dot_gram(x.reshape(-1,x.shape[-1]),x.reshape(-1,x.shape[-1]))
    gram,state=shared_kernel(view_gram(raw,index,index,'concat'))
    arrays.update({'concat__'+k:v for k,v in state.items()})
    for arm in ARMS:
        coef,intercept=ridge_fit(gram,flat,rows[arm],ALPHA)
        arrays.update({arm+'_concat__coefficients':coef,arm+'_concat__intercept':intercept,
                       arm+'_concat__target_mean':flat[rows[arm]].mean(0),
                       arm+'_concat__coefficient_sum':coef.sum(0)})
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite fitted model')
    return arrays


def predict_models(arrays,query_unique,query_index,chunk=256):
    """Feature-only query against all joint-reference rows before arm selection."""
    fit=arrays['fit_features']
    query_index=np.asarray(query_index)
    if query_index.ndim!=2 or query_index.shape[1]!=2 or not np.issubdtype(query_index.dtype,np.integer):
        raise ValueError('Two-frame integer query indices required')
    if np.any(query_index<0) or np.any(query_index>=len(query_unique)):raise ValueError('Query index outside unique images')
    raw=dot_gram(query_unique,fit.reshape(-1,fit.shape[-1]))
    state={k.split('__',1)[1]:v for k,v in arrays.items() if k.startswith('concat__')}
    output={arm+'_concat':np.empty((len(query_index),8)) for arm in ARMS}
    for lo in range(0,len(query_index),chunk):
        q=apply_shared_kernel(view_gram(raw,query_index[lo:lo+chunk],arrays['fit_view_indices'],'concat'),state)
        for arm in ARMS:
            key=arm+'_concat'
            output[key][lo:lo+chunk]=q[:,arrays[arm+'_rows']]@arrays[key+'__coefficients']+arrays[key+'__intercept']
    if any(not np.isfinite(v).all() for v in output.values()):raise ValueError('Nonfinite frozen prediction')
    return output


def training_config():
    return dict(kernel='quadratic',mean_loss_alpha=ALPHA,inputs=['concat'],arms=list(ARMS),
        normalization='shared_joint_N_W_FIT_linear_then_quadratic_center_and_mean_diagonal',
        reference_order='all_N_worlds_then_all_W_worlds; target-major/path-minor views',
        intercept='arm_specific_unpenalized_intercept_via_SPD_solve_and_Schur_complement',
        baseline='joint_N_W_FIT_target_mean_for_scoring_only',
        loss='mean_within_world_then_equal_mean_over_worlds',cv=False,selection=False,
        view_indices=VIEW_INDICES.tolist(),regularization={'N':4096*ALPHA,'W':4096*ALPHA},
        historical_V53_model_used=False)


def model_identity(cache_receipt,fit_cache_sha):
    return dict(cache_identity={k:cache_receipt[k] for k in IDENTITY_KEYS},
                narrow_reference=cache_receipt['narrow_reference'],wide_fit_cache_sha256=fit_cache_sha,
                training_config=training_config())



def matched_label_audit(arrays):
    blocks=len(arrays['fit_block_ids'])
    n,w=arrays['fit_targets'][:blocks],arrays['fit_targets'][blocks:]
    return dict(scope='reader checks of paired IDs and labels; no new images or latent reconstruction',
        fit_blocks=blocks,views_per_arm=int(blocks*16),
        block_ids=entry_identity(arrays['fit_block_ids']),
        motion_target_entries={'N':entry_identity(n[:,:,4:]),'W':entry_identity(w[:,:,4:])},
        motion_labels_bitwise_equal=bool(np.array_equal(n[:,:,4:],w[:,:,4:])),
        maximum_absolute_motion_difference=float(np.max(np.abs(n[:,:,4:].astype(np.float64)-w[:,:,4:]))),
        position_target_mean={'N':arrays['N_concat__target_mean'][:4].tolist(),
                              'W':arrays['W_concat__target_mean'][:4].tolist()})


def validate_embedded_receipts(frozen):
    """The three-file model release retains the exact original JSON bytes."""
    texts=frozen.get('fit_receipt_utf8',{})
    if set(texts)!={'N','W'} or any(not isinstance(t,str) for t in texts.values()):
        raise ValueError('Exact N/W FIT receipt texts required')
    expected={'N':frozen['narrow_reference']['receipt_sha256'],'W':frozen['fit_cache_receipt_sha256']}
    parsed={}
    def invalid(value):raise ValueError('Nonfinite embedded receipt: '+value)
    for arm,text in texts.items():
        if hashlib.sha256(text.encode('utf-8')).hexdigest()!=expected[arm]:
            raise ValueError('Embedded original FIT receipt bytes changed: '+arm)
        parsed[arm]=json.loads(text,parse_constant=invalid)
    n,w=parsed['N'],parsed['W'];ref=frozen['narrow_reference']
    if (n.get('schema')!='v5.3-nuisance-repair-1' or w.get('schema')!=SCHEMA or
        any(v.get('role')!='fit' or v.get('test_opened') is not False for v in parsed.values()) or
        n.get('cache_sha256')!=ref['cache_sha256'] or n.get('source_commit')!=ref['source_commit'] or
        n.get('encoder_identity')!=ref['encoder_identity'] or
        w.get('cache_sha256')!=frozen['fit_cache_sha256'] or w.get('narrow_reference')!=ref or
        {k:w.get(k) for k in IDENTITY_KEYS}!=frozen['cache_identity'] or
        w.get('freshness_audit_sha256')!=frozen['freshness_audit_sha256']):
        raise ValueError('Embedded FIT provenance mismatch')
    if n.get('entry_hashes',{}).get('block_ids')!=w.get('entry_hashes',{}).get('block_ids'):
        raise ValueError('N/W original FIT block IDs differ')
    return parsed


def fit(cache,receipt,out,narrow_cache,narrow_receipt,narrow_manifest):
    started=time.time()
    wide,cache_receipt=load_cache(cache,receipt,'fit')
    narrow,_=load_narrow(narrow_cache,narrow_receipt,narrow_manifest,cache_receipt['narrow_reference'])
    if not np.array_equal(narrow['block_ids'],wide['block_ids']):raise ValueError('Paired FIT IDs differ')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    arrays=train_models(narrow['features'],narrow['targets'],wide['features'],wide['targets'],wide['block_ids'])
    np.savez(out/'frozen_model.npz',**arrays)
    write_json(out/'training_config.json',training_config())
    frozen=dict(schema=FREEZE_SCHEMA,status='frozen_before_evaluation',created_unix=time.time(),
        fit_blocks=256,views_per_fit_block=16,fit_views=8192,arm_views={'N':4096,'W':4096},
        cache_identity={k:cache_receipt[k] for k in IDENTITY_KEYS},narrow_reference=cache_receipt['narrow_reference'],
        freshness_audit_sha256=cache_receipt['freshness_audit_sha256'],sources=sources(),
        fit_cache_sha256=sha256(cache),fit_cache_receipt_sha256=sha256(receipt),
        model_identity=model_identity(cache_receipt,sha256(cache)),
        fit_receipt_utf8={'N':Path(narrow_receipt).read_bytes().decode('utf-8'),
                          'W':Path(receipt).read_bytes().decode('utf-8')},
        matched_label_audit=matched_label_audit(arrays),
        files={name:sha256(out/name) for name in ('frozen_model.npz','training_config.json')},
        model_entry_hashes={k:entry_identity(v) for k,v in arrays.items()},
        coefficient_sum_max_abs={arm:float(np.max(np.abs(arrays[arm+'_concat__coefficient_sum']))) for arm in ARMS},
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        prediction_inputs='features_only',fit_elapsed_seconds=time.time()-started,
        motion_labels_bitwise_equal=True,test_opened=False,p0_authorized=False,original_v5_gate='failed_unchanged')
    validate_embedded_receipts(frozen)
    write_json(out/'freeze_receipt.json',frozen)
    return frozen


def validate_model_arrays(arrays,blocks=256,width=20480):
    if (arrays['fit_features'].shape!=(2*blocks,6,width) or arrays['fit_features'].dtype!=np.float32 or
        arrays['fit_targets'].shape!=(2*blocks,16,8) or arrays['fit_targets'].dtype!=np.float32 or
        not np.array_equal(arrays['fit_block_ids'],np.arange(blocks)) or
        not np.array_equal(arrays['fit_view_indices'],block_indices(2*blocks)) or
        float(arrays['mean_loss_alpha'])!=ALPHA):raise ValueError('Frozen dimensions/IDs/alpha mismatch')
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite frozen arrays')
    y=arrays['fit_targets'];check_targets(y)
    if not np.array_equal(y[:blocks,:,4:],y[blocks:,:,4:]):raise ValueError('Frozen paired motion mismatch')
    flat=y.reshape(-1,8).astype(np.float64)
    if (not np.array_equal(arrays['shared_target_mean'],flat.mean(0)) or
        not np.array_equal(arrays['fit_motion_rms'],np.sqrt(np.mean(flat[:,4:]**2,axis=0)))):
        raise ValueError('Joint scoring moments mismatch')
    expected={'fit_features','fit_targets','fit_block_ids','fit_view_indices','mean_loss_alpha',
              'shared_target_mean','fit_motion_rms','N_rows','W_rows'}
    for level in ('linear','quadratic'):
        for key in ('columns','grand','scale'):expected.add('concat__'+level+'_'+key)
        if (arrays['concat__'+level+'_columns'].shape!=(32*blocks,) or
            arrays['concat__'+level+'_grand'].shape!=() or arrays['concat__'+level+'_scale'].shape!=() or
            float(arrays['concat__'+level+'_scale'])<=0):raise ValueError('Joint reference kernel moments mismatch')
    for arm,start in (('N',0),('W',16*blocks)):
        rows=np.arange(start,start+16*blocks)
        if not np.array_equal(arrays[arm+'_rows'],rows):raise ValueError('Frozen arm rows mismatch')
        for key in ('coefficients','intercept','target_mean','coefficient_sum'):expected.add(arm+'_concat__'+key)
        coef=arrays[arm+'_concat__coefficients']
        if (coef.shape!=(16*blocks,8) or arrays[arm+'_concat__intercept'].shape!=(8,) or
            not np.array_equal(arrays[arm+'_concat__target_mean'],flat[rows].mean(0)) or
            not np.array_equal(arrays[arm+'_concat__coefficient_sum'],coef.sum(0))):
            raise ValueError('Frozen arm coefficients/moments mismatch')
    if set(arrays)!=expected:raise ValueError('Frozen model array schema mismatch')


def load_frozen(folder):
    folder=Path(folder);receipt=read_json(folder/'freeze_receipt.json')
    if (receipt.get('schema')!=FREEZE_SCHEMA or receipt.get('status')!='frozen_before_evaluation' or
        receipt.get('fit_blocks')!=256 or receipt.get('fit_views')!=8192 or receipt.get('views_per_fit_block')!=16 or
        receipt.get('arm_views')!={'N':4096,'W':4096} or receipt.get('sources')!=sources() or
        receipt.get('test_opened') is not False or receipt.get('motion_labels_bitwise_equal') is not True or
        not isinstance(receipt.get('created_unix'),(int,float))):raise ValueError('Incomplete or changed reader freeze')
    validate_identity(receipt['cache_identity']);validate_narrow_reference(receipt['narrow_reference'])
    validate_embedded_receipts(receipt)
    combined=dict(receipt['cache_identity'],narrow_reference=receipt['narrow_reference'])
    if receipt.get('model_identity')!=model_identity(combined,receipt['fit_cache_sha256']):raise ValueError('Frozen model identity mismatch')
    for name in ('frozen_model.npz','training_config.json'):
        if receipt.get('files',{}).get(name)!=sha256(folder/name):raise ValueError('Frozen artifact changed: '+name)
    if read_json(folder/'training_config.json')!=training_config():raise ValueError('Frozen training configuration mismatch')
    with np.load(folder/'frozen_model.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    if receipt.get('model_entry_hashes')!={k:entry_identity(v) for k,v in arrays.items()}:raise ValueError('Frozen entry identity mismatch')
    validate_model_arrays(arrays)
    if receipt.get('matched_label_audit')!=matched_label_audit(arrays):raise ValueError('Frozen matching audit mismatch')
    receipt['receipt_sha256']=sha256(folder/'freeze_receipt.json')
    return arrays,receipt


def score_bridge(predictions,target,mean,replicates=BOOTSTRAPS):
    y=np.asarray(target,dtype=np.float64)
    if y.ndim!=3 or y.shape[1:]!=(2,8) or set(predictions)!={'N_concat','W_concat'}:
        raise ValueError('Bridge requires both variants and exactly the two concat readers')
    scores={key:score_outputs(p,y,mean) for key,p in predictions.items()}
    errors={key:((p-y)**2).mean(1) for key,p in predictions.items()}
    diff=errors['N_concat']-errors['W_concat'];den=((y-mean)**2).mean(1)
    rng=np.random.default_rng(BOOT_SEED);boots=[]
    for begin in range(0,replicates,250):
        idx=rng.integers(0,len(y),(min(250,replicates-begin),len(y)))
        d=diff[idx].mean(1);base=den[idx,4:].mean(1)
        boots.extend(np.stack((d[:,[4,6]].mean(1),(d[:,4:]/base).mean(1)),axis=1))
    ci=np.quantile(boots,[.025,.975],axis=0)
    primary=dict(N_minus_W_dx_mse=float(diff[:,[4,6]].mean()),dx_mse_paired_95ci=ci[:,0].tolist(),
        N_minus_W_motion_nmse=float((diff[:,4:].mean(0)/den[:,4:].mean(0)).mean()),
        motion_nmse_descriptive_paired_95ci=ci[:,1].tolist())
    return dict(primary_reader='concat',primary=primary,readers=scores,
        primary_improvement_interval_above_zero=bool(ci[0,0]>0),
        bootstrap=dict(seed=BOOT_SEED,replicates=replicates,unit='whole_world_both_variants',primary_endpoints=1),
        p0_authorized=False,interpretation='Relative spatial-coverage evidence; not an original V5 gate or P0 authorization')


def score_spatial_support(predictions,target,mean,specification):
    """Descriptive coordinate-envelope proxy from quantized cached labels only."""
    y=np.asarray(target,dtype=np.float64)
    lower=np.asarray(specification['lower'],dtype=np.float64);upper=np.asarray(specification['upper'],dtype=np.float64)
    if (y.ndim!=3 or y.shape[-1]!=8 or lower.shape!=(4,) or upper.shape!=(4,) or
        specification['tolerance']!=0):raise ValueError('Registered narrow envelope and whole-world targets required')
    p7=y[:,:,:4];p6=p7-y[:,:,4:]
    outside=((p6<lower)|(p6>upper)|(p7<lower)|(p7>upper)).any(1)
    any_outside=outside.any(1)
    def statistics(p,mask,coordinate=None):
        if not mask.any():return None
        err=np.mean((p[mask,:,4:]-y[mask,:,4:])**2,axis=(0,1))
        den=np.mean((y[mask,:,4:]-mean[4:])**2,axis=(0,1))
        skill=[float(1-e/d) if d>0 else None for e,d in zip(err,den)]
        if coordinate is not None:return dict(motion_mse=float(err[coordinate]),motion_skill=skill[coordinate])
        return dict(per_coordinate_motion_mse=err.tolist(),per_coordinate_motion_skill=skill,
                    dx_mse=float(err[[0,2]].mean()),motion_mse=float(err.mean()),
                    mean_motion_skill=float(np.mean(skill)) if all(v is not None for v in skill) else None)
    result=dict(definition='cached float32 p7 and displacement cast to float64; p6=p7-d; all views per world; exact envelope, tolerance zero',
                lower=lower.tolist(),upper=upper.tolist(),world_groups={},coordinate_groups={},
                support_only=True,intervals=False,views_per_world=int(y.shape[1]))
    for group,mask in (('inside',~any_outside),('outside',any_outside)):
        result['world_groups'][group]=dict(worlds=int(mask.sum()),examples=int(mask.sum()*y.shape[1]),
            readers={key:statistics(p,mask) for key,p in predictions.items()})
    for j,dimension in enumerate(('A_dx','A_dy','B_dx','B_dy')):
        result['coordinate_groups'][dimension]={}
        for group,mask in (('inside',~outside[:,j]),('outside',outside[:,j])):
            result['coordinate_groups'][dimension][group]=dict(worlds=int(mask.sum()),examples=int(mask.sum()*y.shape[1]),
                readers={key:statistics(p,mask,j) for key,p in predictions.items()})
    return result


def score_anchor(predictions, target, negative_predictions, negative_target, mean, fit_motion_rms,
                 replicates=BOOTSTRAPS):
    y = np.asarray(target,dtype=np.float64).reshape(-1,4,4,8)
    fit_motion_rms = np.asarray(fit_motion_rms,dtype=np.float64)
    if fit_motion_rms.shape != (4,) or np.any(fit_motion_rms<=0):
        raise ValueError('Positive nonzero FIT motion RMS required')
    weights = bootstrap_weights(len(y),replicates)
    boot_denom = bootstrap_mean(weights,((y-mean)**2).mean(1))
    boot_delta_denom = bootstrap_mean(weights,y[:,0,:,4:]**2)
    fit_overall_rms = float(np.sqrt(np.mean(fit_motion_rms**2)))
    result,boot_contrasts,boot_motion,boot_negative = {},{},{},{}
    for key,prediction in predictions.items():
        p = prediction.reshape(-1,4,4,8)
        contrasts = np.stack(((p[:,0,:,4:]-p[:,1,:,4:])/2,
                              (p[:,2,:,4:]-p[:,3,:,4:])/2),axis=2)
        delta = y[:,0,:,4:]
        contrast_error = (contrasts-delta[:,:,None,:])**2
        boot_contrasts[key] = 1-bootstrap_mean(weights,contrast_error)/boot_delta_denom[:,:,None,:]
        boot_motion[key] = 1-(bootstrap_mean(weights,((p-y)**2).mean(1))[:,:,4:]/boot_denom[:,:,4:]).mean(2)
        paths = {}
        for path,name in enumerate(PATH_NAMES):
            mse = contrast_error[:,path].mean(0)
            denominator = (delta[:,path]**2).mean(0)
            paths[name] = dict(end_start_per_coordinate_skill=(1-mse/denominator).tolist(),
                end_start_per_coordinate_skill_descriptive_95ci=np.quantile(boot_contrasts[key][:,path],[.025,.975],axis=0).tolist(),
                end_start_motion_mse=mse.tolist(),
                end_start_prediction_rms=np.sqrt((contrasts[:,path]**2).mean(0)).tolist(),
                target_displacement_rms=np.sqrt(denominator).tolist(),
                absolute_outputs=score_outputs(p[:,:,path,:],y[:,:,path,:],mean),
                mean_motion_skill_descriptive_95ci=np.quantile(boot_motion[key][:,path],[.025,.975]).tolist())
        groups = {}
        for name,indices in (('static',(0,1)),('moving',(2,3))):
            groups[name] = score_outputs(p[:,:,indices,:],y[:,:,indices,:],mean)
            groups[name]['mean_motion_skill_descriptive_95ci']=np.quantile(boot_motion[key][:,indices].mean(1),[.025,.975]).tolist()
        negative = negative_predictions[key]
        # Square each direction before aggregation: ab/ba sign reversal must not
        # cancel a nuisance response in this zero-target diagnostic.
        sq = negative[:,:,4:]**2
        rms = np.sqrt(sq.mean((0,1)))
        boot_by_direction = np.sqrt(bootstrap_mean(weights,sq))
        boot_rms = np.sqrt(np.mean(boot_by_direction**2,axis=1))
        boot_overall = np.sqrt(np.mean(boot_rms**2,axis=1))
        boot_negative[key] = dict(axis=boot_rms,direction=boot_by_direction,overall=boot_overall)
        result[key] = dict(all_views=score_outputs(prediction,target,mean),paths=paths,schedule_groups=groups,
            negative_control_motion_rms=rms.tolist(),
            negative_control_motion_rms_by_direction=np.sqrt(sq.mean(0)).tolist(),
            negative_control_motion_rms_by_direction_over_FIT=(np.sqrt(sq.mean(0))/fit_motion_rms).tolist(),
            negative_control_motion_rms_over_FIT_motion_rms=(rms/fit_motion_rms).tolist(),
            negative_control_overall_motion_rms=float(np.sqrt(sq.mean())),
            negative_control_overall_motion_rms_over_FIT=float(np.sqrt(sq.mean())/fit_overall_rms),
            negative_control_motion_rms_descriptive_95ci=np.quantile(boot_rms,[.025,.975],axis=0).tolist(),
            negative_control_motion_rms_by_direction_descriptive_95ci=np.quantile(boot_by_direction,[.025,.975],axis=0).tolist(),
            negative_control_motion_rms_by_direction_over_FIT_descriptive_95ci=(np.quantile(boot_by_direction,[.025,.975],axis=0)/fit_motion_rms).tolist(),
            negative_control_motion_rms_over_FIT_descriptive_95ci=(np.quantile(boot_rms,[.025,.975],axis=0)/fit_motion_rms).tolist(),
            negative_control_overall_motion_rms_descriptive_95ci=np.quantile(boot_overall,[.025,.975]).tolist(),
            negative_control_overall_motion_rms_over_FIT_descriptive_95ci=(np.quantile(boot_overall,[.025,.975])/fit_overall_rms).tolist(),
            negative_control_motion_mean=negative[:,:,4:].mean((0,1)).tolist(),
            negative_control_position_mse=np.mean((negative[:,:,:4]-negative_target[:,:,:4])**2,axis=(0,1)).tolist())
    comparisons={}
    for name in ('concat',):
        s,b='N_'+name,'W_'+name
        axis_delta=boot_negative[s]['axis']-boot_negative[b]['axis']
        direction_delta=boot_negative[s]['direction']-boot_negative[b]['direction']
        comparisons[name]=dict(
            W_minus_N_per_path_mean_motion_skill=[result[b]['paths'][path]['absolute_outputs']['motion_skill']-
                result[s]['paths'][path]['absolute_outputs']['motion_skill'] for path in PATH_NAMES],
            W_minus_N_per_path_end_start_coordinate_skill=[(np.array(result[b]['paths'][path]['end_start_per_coordinate_skill'])-
                np.array(result[s]['paths'][path]['end_start_per_coordinate_skill'])).tolist() for path in PATH_NAMES],
            W_minus_N_per_path_mean_motion_skill_descriptive_95ci=np.quantile(boot_motion[b]-boot_motion[s],[.025,.975],axis=0).tolist(),
            W_minus_N_per_path_end_start_coordinate_skill_descriptive_95ci=np.quantile(boot_contrasts[b]-boot_contrasts[s],[.025,.975],axis=0).tolist(),
            N_minus_W_negative_motion_rms=(np.array(result[s]['negative_control_motion_rms'])-np.array(result[b]['negative_control_motion_rms'])).tolist(),
            N_minus_W_negative_motion_rms_descriptive_95ci=np.quantile(axis_delta,[.025,.975],axis=0).tolist(),
            N_minus_W_negative_motion_rms_over_FIT=((np.array(result[s]['negative_control_motion_rms'])-
                np.array(result[b]['negative_control_motion_rms']))/fit_motion_rms).tolist(),
            N_minus_W_negative_motion_rms_over_FIT_descriptive_95ci=(np.quantile(axis_delta,[.025,.975],axis=0)/fit_motion_rms).tolist(),
            N_minus_W_negative_motion_rms_by_direction_descriptive_95ci=np.quantile(direction_delta,[.025,.975],axis=0).tolist(),
            N_minus_W_negative_motion_rms_by_direction_over_FIT_descriptive_95ci=(np.quantile(direction_delta,[.025,.975],axis=0)/fit_motion_rms).tolist(),
            N_minus_W_negative_overall_motion_rms=result[s]['negative_control_overall_motion_rms']-result[b]['negative_control_overall_motion_rms'],
            N_minus_W_negative_overall_motion_rms_descriptive_95ci=np.quantile(boot_negative[s]['overall']-boot_negative[b]['overall'],[.025,.975]).tolist())
    return dict(readers=result,comparisons=comparisons,fit_motion_rms=fit_motion_rms.tolist(),
        bootstrap=dict(seed=BOOT_SEED,replicates=replicates,unit='whole_block_all_views_directions_arms',intervals='descriptive_unadjusted'),
        support_only=True,p0_authorized=False)



def evaluate(role,cache,receipt,model,publication,out):
    if role not in ('anchor','bridge'):raise ValueError('Evaluation cannot fit or open FIT/Test')
    arrays,frozen=load_frozen(model)
    data,cache_receipt=load_cache(cache,receipt,role,frozen,publication)
    x,y=data['features'],data['targets']
    index=np.arange(1024).reshape(512,2) if role=='bridge' else block_indices(256)
    prediction={k:v.reshape(y.shape) for k,v in predict_models(arrays,x.reshape(-1,x.shape[-1]),index).items()}
    negative={}
    if role=='anchor':
        negative={k:v.reshape(256,2,8) for k,v in predict_models(arrays,x.reshape(-1,x.shape[-1]),
            block_indices(256,NEGATIVE_INDICES)).items()}
        score=score_anchor(prediction,y,negative,data['negative_targets'],arrays['shared_target_mean'],arrays['fit_motion_rms'])
    else:
        score=score_bridge(prediction,y,arrays['shared_target_mean'])
    score['spatial_support']=score_spatial_support(prediction,y,arrays['shared_target_mean'],read_json(PLAN)['supporting_metrics']['spatial_groups'])
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    saved=dict(prediction,targets=y,role=np.array(role))
    idkey='pair_ids' if role=='bridge' else 'block_ids';saved[idkey]=data[idkey]
    if negative:
        saved.update({'negative_'+k:v for k,v in negative.items()});saved['negative_targets']=data['negative_targets']
    np.savez_compressed(out/'predictions.npz',**saved)
    report=dict(schema='v5.4-reader-evaluation-1',role=role,status='evaluation_complete',scores=score,
        groups=256,examples=512 if role=='bridge' else 4096,reader_arms=list(ARMS),inputs=['concat'],
        coincident_nuisance_pixel_blocks=cache_receipt.get('coincident_nuisance_pixel_blocks'),
        cache_sha256=sha256(cache),cache_receipt_sha256=sha256(receipt),
        frozen_reader_sha256=frozen['receipt_sha256'],freeze_publication_sha256=sha256(publication),
        narrow_reference=frozen['narrow_reference'],predictions_sha256=sha256(out/'predictions.npz'),
        test_opened=False,p0_authorized=False,original_v5_gate='failed_unchanged')
    write_json(out/'cache_receipt.json',cache_receipt);write_json(out/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fit','anchor','bridge'))
    for name in ('cache','receipt','out'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('model','publication','narrow-cache','narrow-receipt','narrow-manifest'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    narrow=(args.narrow_cache,args.narrow_receipt,args.narrow_manifest)
    if args.phase=='fit':
        if args.model is not None or args.publication is not None or any(p is None for p in narrow):
            parser.error('FIT requires all three --narrow-* paths and accepts no evaluation model/publication')
        result=fit(args.cache,args.receipt,args.out,*narrow)
    else:
        if args.model is None or args.publication is None or any(p is not None for p in narrow):
            parser.error('Evaluation requires --model/--publication and accepts no FIT cache paths')
        result=evaluate(args.phase,args.cache,args.receipt,args.model,args.publication,args.out)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
