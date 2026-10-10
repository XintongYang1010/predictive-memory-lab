#!/usr/bin/env python3
"""Evaluate prespecified pixel-axis 2x2 interventions with frozen old readers."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='4'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import v5_6_replay as replay
import v5_4_reader as old
import v5_5_lane_reader as lane
from v5_6_data import SCHEMA,AXIS_NAMES,OBJECT_CONDITION_INDICES

EXTRA_SOURCES=('scripts/v5_6_axis_score.py','scripts/v5_6_data.py','scripts/v5_6_extract.py',
               'scripts/v5_5_counterfactual.py','pmr/visual_memory/data.py')

def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_arrays(z,r):
    keys={'features','targets','role','pair_ids','object_condition_indices'}
    if set(z)!=keys:raise ValueError('Axis cache array inventory differs')
    for name,shape in (('features',(256,2,7,20480)),('targets',(256,2,7,8))):
        a=z[name]
        if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Axis shape/dtype/finite mismatch: '+name)
    if (z['role'].shape!=() or z['role'].dtype.kind!='U' or z['role'].item()!='axis' or
        z['pair_ids'].dtype!=np.int64 or not np.array_equal(z['pair_ids'],np.arange(640,896)) or
        z['object_condition_indices'].dtype!=np.int64 or not np.array_equal(z['object_condition_indices'],OBJECT_CONDITION_INDICES)):
        raise ValueError('Axis role/population/condition mapping differs')
    if r.get('entry_hashes')!={k:old.entry_identity(v) for k,v in z.items()}:raise ValueError('Axis array hashes differ')

def load_axis(folder,publication_path,pub,original):
    cache=folder/'cache.npz';r=old.read_json(folder/'receipt.json');manifest=old.read_json(folder/'manifest.json')
    if (r.get('schema')!=SCHEMA or r.get('role')!='axis' or r.get('stage')!='fit-axis' or
        r.get('test_opened') is not False or r.get('new_reader_fit') is not False or r.get('groups')!=256 or
        r.get('axis_conditions')!=list(AXIS_NAMES) or r.get('feature_dtype')!='float32' or
        r.get('pooling')!='4x4_flattened' or r.get('encoder_batch_size')!=4 or
        r.get('source_commit')!=pub.get('source_commit') or
        r.get('plan_sha256')!=old.sha256(replay.ROOT/'scripts/v5_6_plan.json') or
        r.get('protocol_sha256')!=old.sha256(replay.ROOT/'docs/V5_6_EXECUTION_PROTOCOL.md') or
        r.get('registration_publication_sha256')!=old.sha256(publication_path)):
        raise ValueError('Published axis receipt identity differs')
    if r.get('cache_sha256')!=old.sha256(cache) or r.get('cache_bytes')!=cache.stat().st_size:raise ValueError('Axis cache digest/bytes differ')
    sources=r.get('source_identity',{})
    if not set(replay.SOURCE_NAMES+EXTRA_SOURCES).issubset(sources):raise ValueError('Axis source identity incomplete')
    for name,digest in sources.items():
        rel=Path(name);entry=pub.get('files',{}).get(name);published=entry.get('sha256') if isinstance(entry,dict) else entry
        if rel.is_absolute() or '..' in rel.parts or published!=digest or old.sha256(replay.ROOT/rel)!=digest:raise ValueError('Axis extraction source differs: '+name)
    config=manifest.get('configuration',{})
    if (manifest.get('status')!='complete' or json_sha(config)!=r.get('configuration_sha256') or
        any(r.get(k)!=v for k,v in config.items()) or
        manifest.get('files',{}).get('cache.npz')!=replay.identity(cache) or
        manifest.get('files',{}).get('receipt.json')!=replay.identity(folder/'receipt.json')):
        raise ValueError('Axis complete cache manifest differs')
    plan=old.read_json(replay.ROOT/'scripts/v5_6_plan.json')
    if r.get('encoder_identity')!={k:plan['encoder'][k] for k in ('source_commit','checkpoint_sha256','model')}:raise ValueError('Axis encoder identity differs')
    z=replay.load_npz(cache);validate_arrays(z,r)
    # Authenticate reused images and every coordinate target against old cache.
    np.testing.assert_array_equal(z['features'][:,:, [0,3,6]],original['features'][:,:, [0,4,2]])
    bits=np.array(((0,0,0,0),(1,0,0,0),(0,1,0,0),(1,1,0,0),(0,0,1,0),(0,0,0,1),(0,0,1,1)))
    expected=np.empty_like(z['targets'])
    expected[...,:4]=np.where(bits,original['original_targets'][:,:,None,:4],original['endpoint_positions'][:,:,None,0,:])
    expected[...,4:]=original['original_targets'][:,:,None,4:]*bits
    np.testing.assert_array_equal(z['targets'],expected)
    return z,r

def direction_counts(px,pxy,target):
    active=np.abs(target*224)>=1
    x_product=px*target;xy_product=pxy*target
    return dict(active_x_episodes=int(active.sum()),x_only_wrong=int(((x_product<0)&active).sum()),
        xy_wrong=int(((xy_product<0)&active).sum()),x_only_zero=int(((x_product==0)&active).sum()),
        xy_zero=int(((xy_product==0)&active).sum()),
        x_only_correct_xy_wrong=int(((x_product>0)&(xy_product<0)&active).sum()),
        x_only_wrong_xy_correct=int(((x_product<0)&(xy_product>0)&active).sum()),
        direction_rule='strict positive product=correct;negative=wrong;exact zero separately;abs target>=1px')

def self_test():
    result=direction_counts(np.array([0.,1.,-1.,1.]),np.array([-1.,-1.,1.,1.]),np.array([1.,1.,1.,.001]))
    assert result['active_x_episodes']==3 and result['x_only_zero']==1
    assert result['x_only_correct_xy_wrong']==1 and result['x_only_wrong_xy_correct']==1
    assert result['x_only_wrong']==1 and result['xy_wrong']==2
    rng=np.random.default_rng(5620);p=rng.normal(size=(3,7,8))
    for ix in OBJECT_CONDITION_INDICES:
        v=p[:,ix];x=v[:,1]-v[:,0];y=v[:,2]-v[:,0];interaction=v[:,3]-v[:,2]-v[:,1]+v[:,0]
        np.testing.assert_allclose(x+y+interaction,v[:,3]-v[:,0],atol=1e-14,rtol=0)
    try:validate_arrays({'features':np.zeros((1,7,4))},{})
    except ValueError:pass
    else:raise AssertionError('Malformed population accepted')
    return dict(status='passed',software_only=True,zero_not_counted_correct=True,axis_mapping=True,malformed_population_rejected=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    for name in ('scene','v55-scene','cache','out','publication'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()));return
    if any(getattr(a,k) is None for k in ('scene','v55_scene','cache','out','publication')):p.error('All five paths required for scoring')
    pub=replay.publication(a.publication,EXTRA_SOURCES)
    if any(a.out.resolve()==s.resolve() or s.resolve() in a.out.resolve().parents for s in (a.scene,a.v55_scene,a.cache)):raise ValueError('Output cannot modify input scenes/cache')
    fit_evidence=replay.authenticate(a.scene,replay.V54_SEAL,('reader/frozen_model.npz','reader/freeze_receipt.json','reader/training_config.json'))
    original,old_evidence=replay.original_cache(a.v55_scene)
    z,r=load_axis(a.cache,a.publication,pub,original)
    base,_=old.load_frozen(a.scene/'reader');regional,_=lane.load_frozen(a.v55_scene/'lane_reader')
    x=z['features'].reshape(-1,7,20480);y=z['targets'].reshape(-1,7,8)
    index=np.array([[7*i,7*i+k] for i in range(len(x)) for k in range(7)])
    preds={k:v.reshape(len(x),7,8) for k,v in replay.query_models(base,regional,x.reshape(-1,20480),index).items()}
    a.out.mkdir(parents=True,exist_ok=False)
    summary={};decomposition={}
    for name,v in preds.items():
        summary[name]={}
        for obj,ix in enumerate(((0,1,2,3),(0,4,5,6))):
            p4=v[:,ix];truth=y[:,ix];col=4+2*obj
            x_effect=p4[:,1]-p4[:,0];y_effect=p4[:,2]-p4[:,0]
            interaction=p4[:,3]-p4[:,2]-p4[:,1]+p4[:,0]
            np.testing.assert_allclose(x_effect+y_effect+interaction,p4[:,3]-p4[:,0],atol=1e-12,rtol=0)
            active=np.abs(truth[:,3,col]*224)>=1
            summary[name][str(obj)]=dict(**direction_counts(p4[:,1,col],p4[:,3,col],truth[:,3,col]),
                x_only_rmse_px=float(np.sqrt(np.mean(((p4[:,1,col]-truth[:,1,col])*224)**2))),
                y_only_false_x_rms_px=float(np.sqrt(np.mean((y_effect[:,col]*224)**2))),
                interaction_x_rms_px=float(np.sqrt(np.mean((interaction[:,col]*224)**2))),
                pure_y_opposes_true_x=int(((y_effect[:,col]*truth[:,3,col]<0)&active).sum()))
            for label,val in (('x',x_effect),('y',y_effect),('interaction',interaction)):
                decomposition[f'{name}_{obj}_{label}']=val
    np.savez_compressed(a.out/'predictions.npz',targets=y,**preds,**decomposition)
    report=dict(status='complete',episodes=512,paired_worlds=256,summary=summary,
        intervention='same target x/y 2x2; other object and yellow fixed',
        test_opened=False,reader_refit=False,input_receipt_sha256=old.sha256(a.cache/'receipt.json'),
        cache_sha256=old.sha256(a.cache/'cache.npz'),publication_sha256=old.sha256(a.publication),source_commit=pub['source_commit'],
        source_identity={name:old.sha256(replay.ROOT/name) for name in replay.SOURCE_NAMES+EXTRA_SOURCES},
        authenticated_inputs=dict(v54=fit_evidence,v55=old_evidence),condition_names=list(AXIS_NAMES))
    old.write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
