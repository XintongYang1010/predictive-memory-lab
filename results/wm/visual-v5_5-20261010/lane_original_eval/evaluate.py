#!/usr/bin/env python3
"""Prespecified known-Dev original-cue evaluation; no refit or new images."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='4'
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
import v5_4_reader as old
import v5_5_lane_reader as lane


def run(scene,runroot,out):
    start=time.time();scene=Path(scene);runroot=Path(runroot);out=Path(out)
    if (out/'report.json').exists() or (out/'predictions.npz').exists():raise FileExistsError('Existing outputs preserved')
    pubpath=runroot/'registration_publication.json';pub=old.read_json(pubpath)
    if pub.get('verified_readback') is not True or pub.get('lane_reader',{}).get('all_files_readback_verified') is not True:raise ValueError('Published readback required')
    for name,rec in pub['files'].items():
        if old.sha256(ROOT/name)!=rec['sha256']:raise ValueError('Published source changed: '+name)
    modeldir=runroot/'lane_reader'
    actual={p.relative_to(modeldir).as_posix():dict(bytes=p.stat().st_size,sha256=old.sha256(p)) for p in modeldir.rglob('*') if p.is_file()}
    if actual!=pub['lane_reader']['files']:raise ValueError('Published lane files differ')
    lanemodel,lr=lane.load_frozen(modeldir)
    if pub['verified_unix']<lr['created_unix']:raise ValueError('Publication predates freeze')
    plan=old.read_json(ROOT/'scripts/v5_5_plan.json');manifest=old.read_json(scene/'archive_manifest.json')
    if old.sha256(scene/'archive_manifest.json')!=plan['prior_manifest_sha256']:raise ValueError('Old scene manifest differs')
    inputs={}
    for name in ('reader/frozen_model.npz','reader/freeze_receipt.json','reader/training_config.json','freeze_publication.json',
                 'bridge_cache/cache.npz','bridge_cache/receipt.json','bridge_report/predictions.npz','bridge_report/report.json'):
        p=scene/name;record=dict(bytes=p.stat().st_size,sha256=old.sha256(p))
        if record!=manifest['files'][name]:raise ValueError('Old scene artifact differs: '+name)
        inputs[name]=record
    original,frozen=old.load_frozen(scene/'reader')
    if frozen['receipt_sha256']!=lr['original_frozen_reader_sha256']:raise ValueError('Reader original identity differs')
    data,_=old.load_cache(scene/'bridge_cache/cache.npz',scene/'bridge_cache/receipt.json','bridge',frozen,scene/'freeze_publication.json')
    with np.load(scene/'bridge_report/predictions.npz',allow_pickle=False) as z:
        y=z['targets'];global_prediction=z['W_concat']
    np.testing.assert_array_equal(y,data['targets']);x=data['features'].reshape(1024,20480);index=np.arange(1024).reshape(512,2)
    prediction=lane.predict(lanemodel,x,index).reshape(256,2,8)
    reproduced=old.predict_models(original,x,index)['W_concat'].reshape(256,2,8)
    np.testing.assert_allclose(reproduced,global_prediction,rtol=0,atol=1e-9)
    scores={name:old.score_outputs(p,y,original['shared_target_mean']) for name,p in (('global_W',global_prediction),('lane_W',prediction))}
    delta=((global_prediction-y)**2-(prediction-y)**2).mean(1)[:,[4,6]].mean(1)
    rng=np.random.default_rng(7812);samples=np.empty(10000)
    for i in range(10000):samples[i]=delta[rng.integers(0,256,256)].mean()
    comparison=dict(global_minus_lane_dx_mse=float(delta.mean()),paired_95ci=np.quantile(samples,[.025,.975]).tolist(),
        global_dx_mse=float(np.mean((global_prediction[..., [4,6]]-y[..., [4,6]])**2)),
        lane_dx_mse=float(np.mean((prediction[..., [4,6]]-y[..., [4,6]])**2)))
    comparison['dx_mse_relative_reduction']=comparison['global_minus_lane_dx_mse']/comparison['global_dx_mse']
    direction={}
    for name,p in (('global_W',global_prediction),('lane_W',prediction)):
        bad=np.sign(p[...,[4,6]])!=np.sign(y[...,[4,6]])
        direction[name]=dict(per_object_errors=bad.sum((0,1)).tolist(),any_wrong=int(bad.any(-1).sum()),both_wrong=int(bad.all(-1).sum()),episodes=512)
    np.savez_compressed(out/'predictions.npz',targets=y,global_W=global_prediction,lane_W=prediction,pair_ids=data['pair_ids'],variants=np.array([0,1]),bootstrap_dx_difference=samples)
    report=dict(schema='v5.5-original-lane-evaluation-1',status='complete',posthoc=True,population=dict(worlds=256,episodes=512,pair_ids=[640,895],already_opened_dev=True),
        comparison=comparison,readers=scores,horizontal_direction=direction,bootstrap=dict(seed=7812,replicates=10000,unit='whole_world_both_variants',interpretation='conditional descriptive interval, not fresh confirmation'),
        original_prediction_reproduction_max_abs=float(np.max(np.abs(reproduced-global_prediction))),
        expected_future_match='New GPU factorial111 global and lane arrays must match this original-cue run within absolute1e-9; features must match bitwise',
        provenance=dict(script_sha256=old.sha256(__file__),source_commit=pub['source_commit'],publication_sha256=old.sha256(pubpath),
            lane_freeze_sha256=lr['receipt_sha256'],original_freeze_sha256=frozen['receipt_sha256'],source_identity=lr['source_identity'],old_inputs=inputs,
            predictions_sha256=old.sha256(out/'predictions.npz'),original_manifest_sha256=plan['prior_manifest_sha256']),
        test_opened=False,p0_authorized=False,new_fit=False,new_images=False,elapsed_seconds=time.time()-start)
    old.write_json(out/'report.json',report)
    with (out/'per_episode.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pair_id','variant','reader']+[prefix+'_'+axis for prefix in ('true','pred') for axis in ('Ax','Ay','Bx','By','Adx','Ady','Bdx','Bdy')]+['dx_mse'])
        for i,pair in enumerate(data['pair_ids']):
            for v in (0,1):
                for name,p in (('global_W',global_prediction),('lane_W',prediction)):
                    writer.writerow([int(pair),v,name]+y[i,v].tolist()+p[i,v].tolist()+[float(np.mean((p[i,v,[4,6]]-y[i,v,[4,6]])**2))])
    for name,record in inputs.items():
        if old.sha256(scene/name)!=record['sha256']:raise ValueError('Old input changed during evaluation')
    print(json.dumps({key:report[key] for key in ('comparison','readers','horizontal_direction','elapsed_seconds')},indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.scene,a.run_root,a.out)
