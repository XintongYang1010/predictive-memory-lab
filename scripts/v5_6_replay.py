#!/usr/bin/env python3
"""Frozen image transitions, feature-path interventions and bounded update replay.

No fitting uses known Dev. Update replay is an interface stress test, not a
world-model memory benchmark: object states are binary endpoints and gaps hold.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='4'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import v5_4_reader as old
import v5_5_lane_reader as lane
from score_v5_5 import factorial_targets

ROOT=Path(__file__).resolve().parents[1]
SEQUENCE=np.array([0,1,0,1,4,5,4,5,6,7,6,7,2,3,2,3,0,1,0])
V54_SEAL='ec7bc9c17da8417e9ecd11ba0f21415a2df0ee7af620af9d12f911c3ab26e75c'
V55_SEAL='b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663'
SOURCE_NAMES=('scripts/v5_6_replay.py','scripts/v5_6_plan.json','docs/V5_6_EXECUTION_PROTOCOL.md',
    'scripts/v5_4_reader.py','scripts/v5_3_reader.py','scripts/v5_4_data.py','scripts/v5_3_data.py',
    'scripts/v5_4_plan.json','scripts/v5_5_plan.json','scripts/v5_5_lane_reader.py','scripts/score_v5_5.py')

def identity(path):
    p=Path(path);return dict(bytes=p.stat().st_size,sha256=old.sha256(p))

def publication(path,extra=()):
    p=old.read_json(path)
    if not p.get('verified_readback'):raise ValueError('Source readback required')
    for name in SOURCE_NAMES+tuple(extra):
        record=p.get('files',{}).get(name);digest=record.get('sha256') if isinstance(record,dict) else record
        if digest!=old.sha256(ROOT/name):raise ValueError('Unpublished source '+name)
    return p

def authenticate(scene,seal,names):
    scene=Path(scene)
    if scene.is_symlink() or old.sha256(scene/'archive_manifest.json')!=seal:raise ValueError('Archived scene seal mismatch')
    manifest=old.read_json(scene/'archive_manifest.json');checked={}
    for name in names:
        path=scene/name
        if path.is_symlink() or identity(path)!=manifest['files'][name]:raise ValueError('Sealed artifact changed: '+name)
        checked[name]=identity(path)
    return checked

def original_cache(scene):
    names=('counterfactual_cache/cache.npz','counterfactual_cache/receipt.json','report/predictions.npz',
           'lane_reader/frozen_model.npz','lane_reader/freeze_receipt.json')
    evidence=authenticate(scene,V55_SEAL,names)
    data=load_npz(Path(scene)/names[0]);r=old.read_json(Path(scene)/names[1])
    if (r.get('schema')!='v5.5-module-localization-1' or r.get('role')!='counterfactual' or
        r.get('test_opened') is not False or r.get('groups')!=256 or r.get('episodes')!=512 or
        r.get('mask_names')!=[format(m,'03b') for m in range(8)] or
        r.get('cache_sha256')!=evidence[names[0]]['sha256'] or
        r.get('entry_hashes')!={k:old.entry_identity(v) for k,v in data.items()}):raise ValueError('Original cache contract differs')
    if (data['features'].shape!=(256,2,8,20480) or data['original_targets'].shape!=(256,2,8) or
        data['endpoint_positions'].shape!=(256,2,8,4) or
        not np.array_equal(data['pair_ids'],np.arange(640,896))):raise ValueError('Original population/order differs')
    return data,evidence

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def query_models(base,regional,unique,index):
    return dict(global_W=old.predict_models(base,unique,index)['W_concat'],
                lane_W=lane.predict(regional,unique,index))

def transition_labels(original,positions):
    endpoint=factorial_targets(original,positions).reshape(-1,8,8)
    out=np.zeros((len(endpoint),8,8,8))
    out[...,:4]=endpoint[:,None,:,:4]
    out[...,4:]=endpoint[:,None,:,4:]-endpoint[:,:,None,4:]
    return out

def summary(pred,truth,baseline):
    p=np.asarray(pred).reshape(-1,8);y=np.asarray(truth).reshape(-1,8)
    motion_error=(p[:,4:]-y[:,4:])*224
    active=np.abs(y[:,4:]*224)>=1
    registered=old.score_outputs(p,y,baseline)
    return dict(position_rmse_px=float(np.sqrt(np.mean(((p[:,:4]-y[:,:4])*224)**2))),
        motion_rmse_px=float(np.sqrt(np.mean(motion_error**2))),
        horizontal_rmse_px=float(np.sqrt(np.mean(motion_error[:,[0,2]]**2))),
        large_axis_wrong=int(((p[:,4:]*y[:,4:]<0)&active).sum()),
        large_axis_zero=int(((p[:,4:]==0)&active).sum()),large_axis_count=int(active.sum()),
        motion_skill=registered['motion_skill'],position_skill=registered['position_skill'],
        per_dimension_mse=registered['per_dimension_mse'],per_dimension_skill=registered['per_dimension_skill'],
        pooled_motion_skill=float(1-np.mean((p[:,4:]-y[:,4:])**2)/np.mean((y[:,4:]-baseline[4:])**2)))

def transition_checks(pred,truth):
    p=np.asarray(pred);y=np.asarray(truth);diag=np.arange(8)
    return dict(identity_motion_max_abs=float(np.max(np.abs(p[:,diag,diag,4:]))),
        reverse_motion_sum_max_abs=float(np.max(np.abs(p[...,4:]+p.swapaxes(1,2)[...,4:]))),
        target_identity_motion_max_abs=float(np.max(np.abs(y[:,diag,diag,4:]))),
        target_reverse_motion_sum_max_abs=float(np.max(np.abs(y[...,4:]+y.swapaxes(1,2)[...,4:]))),
        diagnostic_only=True)

def replay(pred,features,positions,fit):
    # FIT-only thresholds, target-static a/b nuisance pairs at all three poses.
    d=(fit[256:,1::2]-fit[256:,::2]).reshape(-1,16,1280).astype(np.float64)
    full_threshold=float(np.quantile(np.linalg.norm(d.reshape(len(d),-1),axis=1),.95))
    local_threshold=np.array([np.quantile(np.linalg.norm(d[:,t].reshape(len(d),-1),axis=1),.95)
                              for t in (np.arange(4),np.arange(12,16))])
    x=features.reshape(-1,8,16,1280);pos=positions.reshape(-1,8,4)
    answer={}; traces={};n=len(x)
    for policy in ('latest','fifo','surprise','trusted_local'):
        initial=pred[:,0,0,:4].reshape(n,2,2)
        bank=np.repeat(initial[:,:,None,:],2,axis=2).astype(np.float32)
        metadata=np.zeros((n,2),dtype=np.uint32)
        estimates=[];writes=[];true_changes=[];scores=[]
        previous=0
        for t,current in enumerate(SEQUENCE):
            change=x[:,current].astype(np.float64)-x[:,previous]
            full=np.linalg.norm(change.reshape(n,-1),axis=1)
            local=np.stack([np.linalg.norm(change[:,tokens].reshape(n,-1),axis=1)
                            for tokens in (np.arange(4),np.arange(12,16))],axis=1)
            if policy in ('latest','fifo'):write=np.ones((n,2),dtype=bool)
            elif policy=='surprise':write=np.repeat((full>full_threshold)[:,None],2,axis=1)
            else:write=local>local_threshold
            if t==0:write[:]=False
            candidate=pred[:,previous,current,:4].reshape(n,2,2).astype(np.float32)
            metadata=np.where(write,np.uint32(t),metadata)
            if policy=='latest':
                bank=np.where(write[:,:,None,None],np.repeat(candidate[:,:,None,:],2,axis=2),bank)
            else:
                shifted=np.stack((bank[:,:,1],candidate),axis=2)
                bank=np.where(write[:,:,None,None],shifted,bank)
            estimate=bank.mean(2).reshape(n,4)
            truth=pos[:,current]
            actual=(truth-pos[:,previous]).reshape(n,2,2)
            true=np.any(actual!=0,axis=2)
            estimates.append(estimate);writes.append(write);true_changes.append(true)
            scores.append(np.mean(((estimate-truth)*224)**2,axis=1))
            previous=current
        est=np.stack(estimates,axis=1);w=np.stack(writes,axis=1);tc=np.stack(true_changes,axis=1)
        mse=np.stack(scores,axis=1)
        # Three observation-free ticks after each fourth observed state hold the
        # last estimate and state; report separately rather than count them as N.
        gaps=np.array([3,7,11,15])
        error=((est-pos[:,SEQUENCE])*224).reshape(n,len(SEQUENCE),2,2)**2
        answer[policy]=dict(position_rmse_px=float(np.sqrt(mse.mean())),
            change_step_rmse_px=float(np.sqrt(error[tc].mean())),
            unchanged_object_rmse_px=float(np.sqrt(error[~tc].mean())),
            gap_rmse_px=float(np.sqrt(mse[:,gaps].mean())),
            writes=int(w.sum()),missed_change_writes=int((tc&~w).sum()),
            unchanged_object_writes=int((~tc&w).sum()),true_object_changes=int(tc.sum()),
            payload_bytes_per_episode=bank.nbytes//n,metadata_bytes_per_episode=metadata.nbytes//n,
            reader='frozen lane position output; every policy reads mean of two 2D states per object')
        traces[policy+'_state']=est;traces[policy+'_write']=w
    return answer,traces,dict(full_nuisance_q95=full_threshold,local_nuisance_q95=local_threshold.tolist(),
        threshold_data='only archived W FIT target-static nuisance pairs',sequence=SEQUENCE.tolist(),
        caveats=['known Dev endpoint replay','no true occlusion rendering; observation-free holds',
                 'no trained writer','no long-term memory/full-history gate claim'])

def run(a):
    start=time.time();pub=publication(a.publication);out=Path(a.out)
    scene=Path(a.scene);v55=Path(a.v55_scene)
    if any(out.resolve()==s.resolve() or s.resolve() in out.resolve().parents for s in (scene,v55)):raise ValueError('Output cannot modify archived scenes')
    out.mkdir(parents=True,exist_ok=False)
    fit_evidence=authenticate(scene,V54_SEAL,('reader/frozen_model.npz','reader/freeze_receipt.json','reader/training_config.json'))
    base,_=old.load_frozen(scene/'reader');regional,_=lane.load_frozen(v55/'lane_reader')
    cache=v55/'counterfactual_cache/cache.npz';data,cache_evidence=original_cache(v55)
    x=data['features'].reshape(-1,8,20480);n=len(x)
    idx=np.array([[8*i+s,8*i+t] for i in range(n) for s in range(8) for t in range(8)])
    predictions={k:v.reshape(n,8,8,8) for k,v in query_models(base,regional,x.reshape(-1,20480),idx).items()}
    truth=transition_labels(data['original_targets'],data['endpoint_positions'])
    original=load_npz(v55/'report/predictions.npz')
    print('transitions scored',flush=True)
    np.savez_compressed(out/'transitions.npz',truth=truth,indices=idx,**predictions)
    stats={k:dict(**summary(v,truth,base['shared_target_mean']),checks=transition_checks(v,truth)) for k,v in predictions.items()}
    # Check against the actual historical artifact without depending on its key
    # names: require an original V55 prediction array for both readers.
    parity={}
    for key,p in predictions.items():
        name='W_concat' if key=='global_W' else 'lane_W';expected=original[name]
        err=float(np.max(np.abs(expected.reshape(n,8,8)-p[:,0])))
        if err>1e-7:raise ValueError('Original-query parity failed')
        parity[key]=dict(artifact_key=name,max_abs=err)
    path_predictions={};path_report={}
    # Frozen source image is old mask000. For each fixed A/B configuration,
    # replace yellow-conditioned target top/bottom rows or its complement.
    for obj,tokens in enumerate((np.arange(4),np.arange(12,16))):
        bank=[]
        for i in range(n):
            bank.append(x[i,0])
            for mask in (0,2,4,6):
                lo=x[i,mask].reshape(16,1280);hi=x[i,mask+1].reshape(16,1280)
                for own,other in ((0,0),(1,0),(0,1),(1,1)):
                    mixed=(hi if other else lo).copy();mixed[tokens]=(hi if own else lo)[tokens]
                    bank.append(mixed.reshape(-1))
        bank=np.asarray(bank,dtype=np.float32)
        pi=np.array([[17*i,17*i+1+j] for i in range(n) for j in range(16)])
        values=query_models(base,regional,bank,pi)
        for model,value in values.items():
            v=value.reshape(n,4,4,8);key=f'{model}_object{obj}'
            path_predictions[key]=v
            direct=v[:,:,1]-v[:,:,0];remote=v[:,:,2]-v[:,:,0]
            interaction=v[:,:,3]-v[:,:,2]-v[:,:,1]+v[:,:,0]
            effect=v[:,:,3]-v[:,:,0];col=4+2*obj
            path_report[key]={name:dict(horizontal_rms_px=float(np.sqrt(np.mean(arr[...,col]**2))*224),
                                    horizontal_mean_px=float(arr[...,col].mean()*224))
                              for name,arr in (('target_rows',direct),('complement',remote),('interaction',interaction),('total',effect))}
        print('feature pathway object',obj,'complete',flush=True)
    np.savez_compressed(out/'feature_paths.npz',**path_predictions)
    update,trace,config=replay(predictions['lane_W'],data['features'],data['endpoint_positions'],base['fit_features'])
    np.savez_compressed(out/'update_replay.npz',**trace)
    report=dict(status='complete',known_dev_worlds=256,episodes=512,transitions_per_episode=64,
        independent_denominator='256 paired worlds, not32768 independent samples',
        transitions=stats,original_parity=parity,feature_paths=path_report,
        feature_path_caveat='off-manifold activations; outer rows avoid direct yellow pooling blocks; contextual leakage remains',
        update_replay=update,update_config=config,test_opened=False,p0_changed=False,
        publication_sha256=old.sha256(a.publication),source_commit=pub['source_commit'],
        cache=identity(cache),authenticated_inputs=dict(v54=fit_evidence,v55=cache_evidence),
        source_identity={name:old.sha256(ROOT/name) for name in SOURCE_NAMES},elapsed_seconds=time.time()-start)
    old.write_json(out/'report.json',report);print(json.dumps(report),flush=True)

def self_test():
    # Independent label checks: positions broadcast to target endpoint, motion
    # is antisymmetric, identity is exactly zero and paths telescope.
    y=np.array([1,1,1,1,1,1,1,1],dtype=float)[None,None];p=np.zeros((1,1,8,4))
    for m in range(8):p[0,0,m]=[m//4,m//4,(m//2)%2,(m//2)%2]
    t=transition_labels(y,p)
    np.testing.assert_array_equal(t[...,4:],-t.swapaxes(1,2)[...,4:])
    np.testing.assert_array_equal(t[:,0,7,4:],t[:,0,3,4:]+t[:,3,7,4:])
    checks=transition_checks(t,t)
    assert checks['identity_motion_max_abs']==checks['reverse_motion_sum_max_abs']==0
    corrupt=t.copy();corrupt[0,2,2,4]=.125
    assert transition_checks(corrupt,t)['identity_motion_max_abs']==.125
    rng=np.random.default_rng(5618);truth=rng.normal(size=(24,8))*np.arange(1,9);pred=truth+rng.normal(size=(24,8))
    result=summary(pred,truth,np.zeros(8));expected=1-((pred-truth)**2).mean(0)/(truth**2).mean(0)
    np.testing.assert_allclose(result['per_dimension_skill'],expected,atol=1e-14)
    np.testing.assert_allclose(result['motion_skill'],expected[4:].mean(),atol=1e-14)
    assert not np.isclose(result['motion_skill'],result['pooled_motion_skill'])
    return dict(status='passed',software_only=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['self-test','run'])
    for x in ('scene','v55-scene','out','publication'):p.add_argument('--'+x)
    a=p.parse_args()
    if a.action=='self-test':print(json.dumps(self_test()))
    else:run(a)
