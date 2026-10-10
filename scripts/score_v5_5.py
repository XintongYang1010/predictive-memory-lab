#!/usr/bin/env python3
"""Frozen factorial/module diagnostic on already opened V5.4 Dev only."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key,'4')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from v5_4_reader import load_frozen, load_cache, predict_models, sha256, entry_identity, score_outputs

ROOT=Path(__file__).resolve().parents[1]
BITS=(4,2,1)

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def ratio(n,d):
    return [float(a/b) if b else None for a,b in zip(n,d)]

def stats(p,y,fit_rms):
    p=np.asarray(p).reshape(-1,8);y=np.asarray(y,dtype=np.float64).reshape(-1,8)
    d=p-y;pm,ym=p[:,4:],y[:,4:];nonzero=ym!=0
    return dict(episodes=len(p),per_dimension_mse=np.mean(d*d,0).tolist(),
        dx_mse=float(np.mean(d[:,[4,6]]**2)),motion_mse=float(np.mean(d[:,4:]**2)),
        motion_direction_error_rate_nonzero=ratio(np.sum((np.sign(pm)!=np.sign(ym))&nonzero,0),nonzero.sum(0)),
        motion_target_zero_count=np.sum(~nonzero,0).tolist(),motion_prediction_exact_zero_count=np.sum(pm==0,0).tolist(),
        motion_sign_tie_count=np.sum(pm*ym==0,0).tolist(),motion_signed_bias=d[:,4:].mean(0).tolist(),
        motion_error_aligned_to_true_sign=np.mean(d[:,4:]*np.sign(ym),0).tolist(),
        motion_absolute_amplitude_ratio=ratio(np.mean(np.abs(pm),0),np.mean(np.abs(ym),0)),
        motion_rms_ratio=ratio(np.sqrt(np.mean(pm*pm,0)),np.sqrt(np.mean(ym*ym,0))),
        motion_response_rms=np.sqrt(np.mean(pm*pm,0)).tolist(),
        motion_response_rms_over_fit_rms=(np.sqrt(np.mean(pm*pm,0))/fit_rms).tolist())

def factorial_targets(original,positions):
    # Original continuous displacement labels are authoritative; do not subtract
    # two independently rounded FP32 coordinates to recreate the label.
    y=np.empty(positions.shape[:-1]+(8,),dtype=np.float64);y[...,:4]=positions
    for m in range(8):
        y[...,m,4:6]=original[...,4:6]*bool(m&4)
        y[...,m,6:8]=original[...,6:8]*bool(m&2)
    if not np.array_equal(y[...,7,:],original):raise ValueError('Original target parity failed')
    return y

def decompose(p):
    """Exact three-player Shapley and all nonempty Boolean Moebius terms."""
    p=np.asarray(p,dtype=np.float64);phi=[];mob=np.zeros_like(p)
    mob[...,0,:]=p[...,0,:]
    for m in range(1,8):
        total=np.zeros_like(p[...,0,:]);s=m
        while True:
            total+=(-1)**(bin(m).count('1')-bin(s).count('1'))*p[...,s,:]
            if s==0:break
            s=(s-1)&m
        mob[...,m,:]=total
    for b in BITS:
        total=np.zeros_like(p[...,0,:])
        for s in range(8):
            if s&b:continue
            k=bin(s).count('1');weight=1/3 if k in (0,2) else 1/6
            total+=weight*(p[...,s|b,:]-p[...,s,:])
        phi.append(total)
    phi=np.stack(phi,axis=-2)
    if not np.allclose(phi.sum(-2),p[...,7,:]-p[...,0,:],atol=1e-12,rtol=0):raise ValueError('Shapley identity failed')
    if not np.allclose(mob.sum(-2),p[...,7,:],atol=1e-12,rtol=0):raise ValueError('Moebius identity failed')
    return phi,mob

def attribution(phi,mob,y,fit_rms):
    p=phi.reshape(-1,3,8)[...,4:];truth=y.reshape(-1,8)[:,4:]
    own=np.stack([p[:,0,0],p[:,0,1],p[:,1,2],p[:,1,3]],1)
    other=np.stack([p[:,1,0],p[:,1,1],p[:,0,2],p[:,0,3]],1)
    yellow=p[:,2,:]
    def row(v):
        rms=np.sqrt(np.mean(v*v,0))
        return dict(rms=rms.tolist(),rms_over_fit_rms=(rms/fit_rms).tolist(),
                    signed_mean=v.mean(0).tolist(),aligned_to_true_sign=(v*np.sign(truth)).mean(0).tolist())
    return dict(own_response=row(own),own_response_error=row(own-truth),other_target_response=row(other),yellow_response=row(yellow),
        interactions={format(m,'03b'):row(mob[...,m,4:].reshape(-1,4)) for m in (3,5,6,7)},
        interpretation='Scene response attribution; not fractions of neural-module fault or explained variance')

def bootstrap_comparison(global_p,lane_p,y):
    # Average both variants before resampling worlds; all8 masks kept together.
    diff=((global_p-y)**2-(lane_p-y)**2).mean(1)
    samples=np.empty((10000,8));rng=np.random.default_rng(7812)
    dx=diff[...,[4,6]].mean(-1)
    for lo in range(0,10000,250):
        ix=rng.integers(0,len(y),(250,len(y)))
        samples[lo:lo+250]=dx[ix].mean(1)
    ci=np.quantile(samples,[.025,.975],axis=0)
    return {format(m,'03b'):dict(global_minus_lane_dx_mse=float(dx[:,m].mean()),paired_95ci=ci[:,m].tolist()) for m in range(8)}

def original_direction(p,y):
    p=p.reshape(-1,8);y=y.reshape(-1,8);bad=np.sign(p[:,[4,6]])!=np.sign(y[:,[4,6]])
    opposite=y[:,4]*y[:,6]<0;same=p[:,4]*p[:,6]>0
    return dict(target_errors=bad.sum(0).tolist(),any_wrong=int(bad.any(1).sum()),both_wrong=int(bad.all(1).sum()),
        episodes=len(y),true_opposite_count=int(opposite.sum()),pred_same_given_true_opposite=int((opposite&same).sum()))

def block_pool(raw):
    raw=np.asarray(raw,dtype=np.float64)
    return raw.reshape(*raw.shape[:-2],4,4,4,4,1280).mean(axis=(-4,-2)).reshape(*raw.shape[:-2],20480)

def pool_audit(cache,data,receipt):
    records=[];maxdiff=0.;sqdiff=0.;count=0;bf16_equal=True
    for i,pair in enumerate(data['pair_ids']):
        rec=receipt['raw_records'][str(pair)];path=cache/rec['path']
        if sha256(path)!=rec['sha256'] or path.stat().st_size!=rec['bytes']:raise ValueError('Raw shard mismatch')
        with np.load(path,allow_pickle=False) as z:raw=z['raw']
        if {'raw':entry_identity(raw)}!=rec['entry_hashes']:raise ValueError('Raw array identity mismatch')
        pool=block_pool(raw);prior=data['pooled_prequant'][i][:,[0,7]]
        delta=pool-prior;maxdiff=max(maxdiff,float(np.max(np.abs(delta))));sqdiff+=float(np.sum(delta*delta));count+=delta.size
        canonical=torch.from_numpy(prior).to(torch.bfloat16).float().numpy()
        bf16_equal &= np.array_equal(canonical,data['features'][i][:,[0,7]])
        d=raw[:,1].astype(np.float64)-raw[:,0].astype(np.float64)
        q=block_pool(d).reshape(2,4,4,1280).repeat(4,1).repeat(4,2).reshape(2,256,1280)
        for v in (0,1):
            total=float(np.sum(d[v]**2));retained=float(np.sum(q[v]**2));lost=float(np.sum((d[v]-q[v])**2))
            records.append(dict(pair_id=int(pair),variant=v,total_delta_squared_norm=total,
                retained_squared_norm=retained,lost_squared_norm=lost,
                retained_fraction=retained/total if total else None,
                orthogonality_residual=abs(total-retained-lost)))
    fractions=[r['retained_fraction'] for r in records if r['retained_fraction'] is not None]
    return dict(raw_to_prequant_max_abs=maxdiff,raw_to_prequant_rmse=float(np.sqrt(sqdiff/count)),
        same_call_bf16_exact=bool(bf16_equal),projection_summary=dict(
        retained_fraction_mean=float(np.mean(fractions)) if fractions else None,
        retained_fraction_quantiles=np.quantile(fractions,[0,.25,.5,.75,1]).tolist() if fractions else None,
        zero_delta_episodes=len(records)-len(fractions),
        interpretation='Fraction of feature-change energy retained; not fraction of motion information'),episodes=records)

def validate_cache_evidence(a,receipt,data,old,frozen,plan):
    """Authenticate aggregate, inventory, source and original-label membership."""
    expected_bits=np.array([[bool(m&b) for b in BITS] for m in range(8)],dtype=np.int64)
    expected_names=[format(m,'03b') for m in range(8)]
    if (receipt.get('schema')!='v5.5-module-localization-1' or receipt.get('role')!='counterfactual'
            or receipt.get('test_opened') is not False or receipt.get('p0_authorized') is not False
            or receipt.get('groups')!=256 or receipt.get('episodes')!=512
            or receipt.get('mask_names')!=expected_names or receipt.get('mask_bits')!=expected_bits.tolist()
            or receipt.get('first_frame_mask')!=0 or receipt.get('raw_masks')!=[0,7]):
        raise ValueError('Counterfactual receipt population or mask contract mismatch')
    keys={'features','pooled_prequant','original_targets','endpoint_positions','pair_ids','mask_bits','mask_names','role'}
    if set(data)!=keys or str(data['role'].item())!='counterfactual':raise ValueError('Counterfactual array schema mismatch')
    for key,shape in {'features':(256,2,8,20480),'pooled_prequant':(256,2,8,20480),
                      'original_targets':(256,2,8),'endpoint_positions':(256,2,8,4)}.items():
        if data[key].shape!=shape or data[key].dtype!=np.float32 or not np.isfinite(data[key]).all():
            raise ValueError('Counterfactual shape/dtype/finite mismatch: '+key)
    if (not np.array_equal(data['pair_ids'],np.arange(640,896)) or not np.array_equal(data['mask_bits'],expected_bits)
            or data['mask_names'].tolist()!=expected_names or not np.array_equal(data['original_targets'],old['targets'])):
        raise ValueError('Original labels, pair IDs or mask order changed')
    positions=data['endpoint_positions']
    for mask in range(8):
        for obj,bit in enumerate((4,2)):
            if not np.array_equal(positions[:,:,mask,obj*2:obj*2+2],positions[:,:,7 if mask&bit else 0,obj*2:obj*2+2]):
                raise ValueError('Endpoint positions do not follow object mask bits')
    seal=json.loads((a.prior_scene/'archive_manifest.json').read_text())
    for name in ('bridge_cache/cache.npz','bridge_cache/receipt.json','reader/freeze_receipt.json',
                 'bridge_report/predictions.npz'):
        path=a.prior_scene/name
        if seal['files'][name]!=dict(bytes=path.stat().st_size,sha256=sha256(path)):
            raise ValueError('Original sealed artifact changed: '+name)
    reference=dict(prior_archive_manifest_sha256=plan['prior_manifest_sha256'],
        original_bridge_cache_sha256=sha256(a.prior_scene/'bridge_cache/cache.npz'),
        original_bridge_receipt_sha256=sha256(a.prior_scene/'bridge_cache/receipt.json'),
        frozen_reader_sha256=frozen['receipt_sha256'],encoder_identity=frozen['cache_identity']['encoder_identity'])
    if any(receipt.get('reference',{}).get(k)!=v for k,v in reference.items()):raise ValueError('Counterfactual reference differs from loaded original scene')
    required={'scripts/v5_5_counterfactual.py','scripts/launch_v5_5.py','scripts/score_v5_5.py',
              'scripts/v5_5_lane_reader.py','scripts/v5_5_plan.json','docs/V5_5_MODULE_DIAGNOSTIC.md',
              'pmr/visual_memory/features.py','pmr/visual_memory/data.py'}
    sources=receipt.get('source_identity',{})
    if not required.issubset(sources):raise ValueError('Missing required published source identity')
    for name,h in sources.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or sha256(ROOT/p)!=h:raise ValueError('Published source changed: '+name)
    manifest=json.loads((a.cache/'manifest.json').read_text())
    if manifest.get('status')!='complete':raise ValueError('Counterfactual cache is incomplete')
    configuration=manifest['configuration']
    configuration_sha=hashlib.sha256(json.dumps(configuration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if configuration_sha!=receipt.get('configuration_sha256') or configuration.get('source_identity')!=sources:
        raise ValueError('Counterfactual configuration identity mismatch')
    actual=set()
    for path in a.cache.rglob('*'):
        if path.is_symlink():raise ValueError('Cache contains a symlink')
        if not path.is_file() or path==a.cache/'manifest.json':continue
        name=path.relative_to(a.cache).as_posix();actual.add(name);record=manifest['files'].get(name)
        if record is None or path.stat().st_size!=record['bytes']:raise ValueError('Cache inventory/size mismatch: '+name)
        if name.startswith('raw_shards/'):
            pair=str(int(path.stem));raw=receipt['raw_records'].get(pair,{})
            if raw.get('path')!=name or any(raw.get(k)!=record[k] for k in ('bytes','sha256')):
                raise ValueError('Raw inventory differs from receipt')
            # pool_audit reads and hashes every raw shard once below.
        elif sha256(path)!=record['sha256']:raise ValueError('Cache inventory hash mismatch: '+name)
    if actual!=set(manifest['files']):raise ValueError('Cache file inventory differs')
    publication_path=a.cache.parent/'registration_publication.json'
    if sha256(publication_path)!=receipt.get('registration_publication_sha256'):raise ValueError('Source publication identity changed')
    publication=json.loads(publication_path.read_text())
    if (publication.get('verified_readback') is not True or publication.get('source_commit')!=receipt['source_commit']
            or publication.get('plan_sha256')!=sha256(ROOT/'scripts/v5_5_plan.json')
            or publication.get('protocol_sha256')!=sha256(ROOT/'docs/V5_5_MODULE_DIAGNOSTIC.md')):
        raise ValueError('Source publication contract mismatch')
    lane=publication.get('lane_reader',{})
    if a.lane_model.is_symlink() or any(p.is_symlink() for p in a.lane_model.rglob('*')):
        raise ValueError('Published lane model contains a symlink')
    actual_lane={p.relative_to(a.lane_model).as_posix():dict(bytes=p.stat().st_size,sha256=sha256(p))
                 for p in a.lane_model.rglob('*') if p.is_file() and not p.is_symlink()}
    if lane.get('all_files_readback_verified') is not True or actual_lane!=lane.get('files'):
        raise ValueError('Lane model differs from complete published file inventory')

def run(a):
    started=time.time();a.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/'scripts/v5_5_plan.json').read_text())
    if sha256(a.prior_scene/'archive_manifest.json')!=plan['prior_manifest_sha256']:raise ValueError('Prior manifest changed')
    model,frozen=load_frozen(a.prior_scene/'reader')
    old,_=load_cache(a.prior_scene/'bridge_cache/cache.npz',a.prior_scene/'bridge_cache/receipt.json','bridge',frozen,a.prior_scene/'freeze_publication.json')
    receipt=json.loads((a.cache/'receipt.json').read_text());cache=a.cache/'cache.npz'
    if receipt['cache_sha256']!=sha256(cache) or receipt['cache_bytes']!=cache.stat().st_size:raise ValueError('New cache hash differs')
    with np.load(cache,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    if receipt['entry_hashes']!={k:entry_identity(v) for k,v in data.items()}:raise ValueError('Cache entries differ')
    validate_cache_evidence(a,receipt,data,old,frozen,plan)
    feature_parity=np.array_equal(data['features'][:,:,[0,7]],old['features'])
    y=factorial_targets(data['original_targets'],data['endpoint_positions'])
    x=data['features'].reshape(-1,20480);idx=np.stack((np.repeat(np.arange(512)*8,8),np.arange(4096)),1)
    predictions={k:val.reshape(256,2,8,8) for k,val in predict_models(model,x,idx).items()}
    from v5_5_lane_reader import load_frozen as load_lane,predict as predict_lane
    lane,lr=load_lane(a.lane_model);predictions['lane_W']=predict_lane(lane,x,idx).reshape(256,2,8,8)
    if (lr['original_frozen_reader_sha256']!=frozen['receipt_sha256']
            or lr['archive_manifest_sha256']!=plan['prior_manifest_sha256']):
        raise ValueError('Lane reader derives from a different original model/scene')
    with np.load(a.prior_scene/'bridge_report/predictions.npz',allow_pickle=False) as z:
        archived={k:z[k] for k in z.files}
    drift={arm:float(np.max(np.abs(predictions[arm][...,7,:]-archived[arm]))) for arm in ('N_concat','W_concat')}
    parity=dict(features_bitwise_equal=bool(feature_parity),predictions_max_abs=drift,
        passed=bool(feature_parity and all(v<=1e-9 for v in drift.values())))
    save(a.out/'parity.json',parity)
    if not parity['passed']:
        np.savez_compressed(a.out/'unmatched_predictions.npz',**predictions)
        raise ValueError('Original parity failed; outputs preserved, do not interpret matched interventions')
    pooled=pool_audit(a.cache,data,receipt)
    q=data['pooled_prequant'][:,:,[0,7]].reshape(-1,20480);qi=np.arange(1024).reshape(512,2)
    prequant=predict_models(model,q,qi)['W_concat'].reshape(256,2,8)
    arrays=dict(targets=y,prequant_original_W=prequant)
    reports={};rms=model['fit_motion_rms'];mean=model['shared_target_mean']
    for name,pred in predictions.items():
        phi,mob=decompose(pred);arrays[name]=pred;arrays[name+'_shapley']=phi;arrays[name+'_moebius']=mob
        reports[name]=dict(original=score_outputs(pred[...,7,:],y[...,7,:],mean),
            original_direction=original_direction(pred[...,7,:],y[...,7,:]),
            masks={format(m,'03b'):stats(pred[...,m,:],y[...,m,:],rms) for m in range(8)},
            attribution=attribution(phi,mob,y[...,7,:],rms),
            shapley_identity_max_abs=float(np.max(np.abs(phi.sum(-2)-pred[...,7,:]+pred[...,0,:]))))
    pd=prequant-predictions['W_concat'][...,7,:]
    report=dict(schema='v5.5-module-report-1',status='diagnostic_complete',
        original_parity=parity,population=dict(worlds=256,episodes=512,already_opened_dev=True),
        readers=reports,global_vs_lane=bootstrap_comparison(predictions['W_concat'],predictions['lane_W'],y),
        bootstrap=dict(replicates=10000,seed=7812,unit='whole_world_all_variants_and_masks',interpretation='conditional descriptive interval on opened Dev'),
        pooling={k:v for k,v in pooled.items() if k!='episodes'},
        final_bf16_rounding=dict(prequant_original_scores=score_outputs(prequant,y[...,7,:],mean),
            prediction_change_rms=np.sqrt(np.mean(pd**2,axis=(0,1))).tolist(),
            prediction_change_max_abs=np.max(np.abs(pd),axis=(0,1)).tolist()),
        provenance=dict(source_commit=receipt['source_commit'],cache_sha256=sha256(cache),
            cache_receipt_sha256=sha256(a.cache/'receipt.json'),lane_model_receipt=lr,
            prior_manifest_sha256=plan['prior_manifest_sha256'],script_sha256=sha256(__file__)),
        test_opened=False,p0_authorized=False,elapsed_seconds=time.time()-started)
    save(a.out/'report.json',report);save(a.out/'pool_per_episode.json',pooled['episodes'])
    np.savez_compressed(a.out/'predictions.npz',**arrays)
    with (a.out/'per_episode.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pair_id','variant','mask','reader','A_dx_true','A_dx_pred','B_dx_true','B_dx_pred','dx_mse'])
        for i,pair in enumerate(data['pair_ids']):
            for v in (0,1):
                for m in range(8):
                    for name,pred in predictions.items():
                        target=y[i,v,m];p=pred[i,v,m]
                        writer.writerow([pair,v,format(m,'03b'),name,target[4],p[4],target[6],p[6],float(np.mean((p[[4,6]]-target[[4,6]])**2))])
    save(a.out/'manifest.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(a.out.iterdir())})
    return dict(status=report['status'],original_global_vs_lane=report['global_vs_lane']['111'],elapsed_seconds=report['elapsed_seconds'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','prior-scene','out','lane-model'):parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(parser.parse_args()),indent=2))
