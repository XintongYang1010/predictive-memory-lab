#!/usr/bin/env python3
"""Posthoc analysis of archived V5.4 bridge predictions; no generation or fitting.

Reproduce from repo root, writing to a new output directory:
python3 results/wm/visual-v5_4-badcase-review-20261010/analyze.py \
  --scene execution/visual-v5_4-20261010/report-snapshot \
  --verification execution/visual-v5_4-20261010/report-snapshot-verification.json \
  --out /path/to/new-output
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
AXES=(4,6)
RULE='512 episodes ranked by W concat mean squared error over A_dx and B_dx descending; ties use ascending pair ID then variant; select3 without distinct-pair constraint.'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def entry(a):
    a=np.asarray(a)
    return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def transition(n_wrong,w_wrong):
    return dict(observations=int(n_wrong.size),fixed_wrong_to_correct=int((n_wrong&~w_wrong).sum()),
                broken_correct_to_wrong=int((~n_wrong&w_wrong).sum()),remained_correct=int((~n_wrong&~w_wrong).sum()),
                remained_wrong=int((n_wrong&w_wrong).sum()))


def analyze(scene,verification,source_root,out):
    scene=Path(scene).resolve();out=Path(out).resolve();source_root=Path(source_root).resolve()
    if out==scene or scene in out.parents:raise ValueError('Cannot write into immutable source scene')
    original_verification_sha=sha(verification)
    snapshot=json.loads(Path(verification).read_text())
    if snapshot.get('status')!='all_snapshot_files_verified':raise ValueError('Verified snapshot required')
    input_hashes={}
    def checked(name):
        p=scene/name;observed=dict(bytes=p.stat().st_size,sha256=sha(p))
        if snapshot['files'][name]!=observed:raise ValueError('Snapshot changed: '+name)
        input_hashes[name]=observed;return p
    report=json.loads(checked('bridge_report/report.json').read_text())
    receipt=json.loads(checked('bridge_cache/receipt.json').read_text())
    frozen=json.loads(checked('reader/freeze_receipt.json').read_text())
    publication=json.loads(checked('freeze_publication.json').read_text())
    path=checked('bridge_report/predictions.npz')
    if (report.get('schema')!='v5.4-reader-evaluation-1' or report.get('status')!='evaluation_complete' or
        report.get('role')!='bridge' or report.get('groups')!=256 or report.get('examples')!=512 or
        report.get('test_opened') is not False or report.get('p0_authorized') is not False):raise ValueError('Completed V5.4 bridge required')
    if (sha(path)!=report['predictions_sha256'] or sha(scene/'bridge_cache/receipt.json')!=report['cache_receipt_sha256'] or
        sha(scene/'reader/freeze_receipt.json')!=report['frozen_reader_sha256'] or
        sha(scene/'freeze_publication.json')!=report['freeze_publication_sha256'] or
        publication.get('verified_readback') is not True or publication.get('frozen_reader_sha256')!=report['frozen_reader_sha256']):
        raise ValueError('Report/model/cache/publication identity mismatch')
    source_hashes=dict(frozen['sources'])
    source_hashes['pmr/visual_memory/data.py']=receipt['source_identity']['pmr/visual_memory/data.py']
    for name,digest in source_hashes.items():
        if sha(source_root/name)!=digest:raise ValueError('Scientific source identity changed: '+name)
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if not np.array_equal(a['pair_ids'],np.arange(640,896)) or str(a['role'].item())!='bridge':raise ValueError('Unexpected IDs/role')
    for name in ('N_concat','W_concat','targets'):
        if a[name].shape!=(256,2,8) or not np.isfinite(a[name]).all():raise ValueError('Invalid shape/nonfinite '+name)
    for i,pair in enumerate(a['pair_ids']):
        shard=json.loads(checked(f'bridge_cache/shards/{pair:06d}.json').read_text())
        ident=entry(a['targets'][i])
        if (shard['inputs']['targets']!=ident or shard['entry_hashes']['targets']!=ident or
            shard['role']!='bridge' or shard['unit']!=int(pair) or [v['variant'] for v in shard['inputs']['metadata']]!=[0,1]):
            raise ValueError('Original target shard mismatch')
    means=frozen['matched_label_audit']['position_target_mean']
    mean=np.r_[(np.asarray(means['N'])+np.asarray(means['W']))/2,np.zeros(4)]
    if entry(mean)!=frozen['model_entry_hashes']['shared_target_mean']:raise ValueError('Frozen mean baseline hash mismatch')
    y=a['targets'].astype(float);truth=y[:,:,AXES]
    right=(truth>0).all(-1);left=(truth<0).all(-1);opposite=truth.prod(-1)<0
    result=dict(schema='v5.4.posthoc.badcase-context.1',posthoc=True,
        scope='Full archived Dev640..895:512episodes/256pairedworlds. Descriptive contextual counts, no new experiment or CI.',
        definitions=dict(sign_error='sign(predicted dx) differs from sign(actual t7-minus-t6 dx)',
            both_wrong='Both A and B horizontal signs wrong in the same episode',
            both_right_to_both_left='Both true dx>0 and both predicted dx<0; denominator is both-right target episodes',
            both_left_to_both_right='Both true dx<0 and both predicted dx>0; denominator is both-left target episodes',
            same_when_true_opposite='True A_dx*B_dx<0 and predicted A_dx*B_dx>0; zeros excluded from same/opposite',
            variant_score='Each variant contains256worlds; skills use the frozen common FIT target mean',
            dx_times_224='Normalized-coordinate dx multiplied by224 for pixel-coordinate scale; not measured integer raster displacement',
            fixed='N incorrect sign becomes W correct sign',broken='N correct sign becomes W incorrect sign',
            top3_error_mass_fraction='Sum of W mean A/B horizontal squared error for the selected3episodes divided by its sum over all512episodes'),
        target_patterns=dict(both_right=int(right.sum()),both_left=int(left.sum()),opposite=int(opposite.sum()),zero_horizontal_targets=int((truth==0).sum())),
        integrity=dict(source_commit=receipt['source_commit'],predictions_sha256=sha(path),report_sha256=sha(scene/'bridge_report/report.json'),
            frozen_receipt_sha256=report['frozen_reader_sha256'],snapshot_verification_sha256=original_verification_sha,
            sources=source_hashes,all_target_shards_verified=256),arms={})
    predictions={arm:a[arm+'_concat'].astype(float) for arm in ('N','W')}
    wrong={arm:np.sign(p[:,:,AXES])!=np.sign(truth) for arm,p in predictions.items()}
    mse={arm:(p-y)**2 for arm,p in predictions.items()}
    horizontal={arm:e[:,:,AXES].mean(-1) for arm,e in mse.items()}
    ranking={}
    for arm,p in predictions.items():
        dx=p[:,:,AXES];bad=wrong[arm];pright=(dx>0).all(-1);pleft=(dx<0).all(-1);same=dx.prod(-1)>0
        axis_mse=mse[arm].mean((0,1))
        if not np.allclose(axis_mse,report['scores']['readers'][arm+'_concat']['per_dimension_mse'],rtol=1e-9,atol=1e-12):
            raise ValueError('Predictions do not reproduce complete report MSE')
        record=dict(A_horizontal_wrong=int(bad[:,:,0].sum()),B_horizontal_wrong=int(bad[:,:,1].sum()),
            any_horizontal_sign_error=int(bad.any(-1).sum()),any_horizontal_sign_error_rate=float(bad.any(-1).mean()),
            both_wrong=int(bad.all(-1).sum()),exactly_one_wrong=int((bad.sum(-1)==1).sum()),both_correct=int((~bad).all(-1).sum()),
            both_right_to_both_left=int((right&pleft).sum()),both_left_to_both_right=int((left&pright).sum()),
            same_when_true_opposite=int((opposite&same).sum()),zero_horizontal_predictions=int((dx==0).sum()),variants={})
        for v in (0,1):
            axis_error=mse[arm][:,v].mean(0);den=((y[:,v]-mean)**2).mean(0);skill=1-axis_error/den
            record['variants'][str(v)]=dict(episodes=256,dx_mse=float(horizontal[arm][:,v].mean()),motion_skill=float(skill[4:].mean()),
                motion_axis_skills=skill[4:].tolist(),A_horizontal_wrong=int(bad[:,v,0].sum()),B_horizontal_wrong=int(bad[:,v,1].sum()),
                both_wrong=int(bad[:,v].all(-1).sum()),both_right_truth=int(right[:,v].sum()),both_left_truth=int(left[:,v].sum()),opposite_truth=int(opposite[:,v].sum()),
                both_right_to_both_left=int((right[:,v]&pleft[:,v]).sum()),both_left_to_both_right=int((left[:,v]&pright[:,v]).sum()),
                same_when_true_opposite=int((opposite[:,v]&same[:,v]).sum()))
        h=horizontal[arm]
        record['variant1_to_variant0_dx_mse_ratio']=float(h[:,1].mean()/h[:,0].mean())
        record['variant1_higher_dx_error_worlds']=int((h[:,1]>h[:,0]).sum())
        record['variant0_higher_dx_error_worlds']=int((h[:,0]>h[:,1]).sum())
        record['variant_error_ties']=int((h[:,0]==h[:,1]).sum())
        ranking[arm]=sorted(((float(h[i,v]),int(pair),v,i) for i,pair in enumerate(a['pair_ids']) for v in (0,1)),key=lambda r:(-r[0],r[1],r[2]))
        record['top_error_variant1_counts']={str(k):sum(r[2]==1 for r in ranking[arm][:k]) for k in (3,10,25,50,100)}
        result['arms'][arm]=record
    nw,ww=wrong['N'],wrong['W']
    ncount=nw.sum(-1);wcount=ww.sum(-1)
    result['N_to_W_sign_transitions']=dict(A=transition(nw[:,:,0],ww[:,:,0]),B=transition(nw[:,:,1],ww[:,:,1]),
        pooled_target_observations=transition(nw,ww),
        episode_full_correctness=transition(nw.any(-1),ww.any(-1)),
        episode_wrong_count=dict(improved=int((wcount<ncount).sum()),worsened=int((wcount>ncount).sum()),unchanged=int((wcount==ncount).sum()),
            unchanged_count_but_one_object_fixed_other_broken=int(((wcount==ncount)&(nw!=ww).any(-1)).sum())),
        by_variant={str(v):dict(A=transition(nw[:,v,0],ww[:,v,0]),B=transition(nw[:,v,1],ww[:,v,1]),
            pooled_target_observations=transition(nw[:,v],ww[:,v])) for v in (0,1)})
    selected=[]
    for rank,(loss,pair,v,i) in enumerate(ranking['W'][:3],1):
        row=dict(rank=rank,pair_id=pair,variant=v,episode_id=f'dev-{pair:06d}-{v}',W_dx_mse=loss,N_dx_mse=float(horizontal['N'][i,v]))
        for obj,j in (('A',0),('B',1)):
            row[obj+'_true_dx']=float(truth[i,v,j])
            for arm in ('N','W'):
                row[obj+'_'+arm+'_dx']=float(predictions[arm][i,v,AXES[j]])
                row[obj+'_'+arm+'_wrong']=bool(wrong[arm][i,v,j])
            row[obj+'_fixed']=bool(nw[i,v,j] and not ww[i,v,j]);row[obj+'_broken']=bool(not nw[i,v,j] and ww[i,v,j])
        for obj in ('A','B'):
            for name in ('true','N','W'):row[obj+'_'+name+'_dx_times_224']=row[obj+'_'+name+'_dx']*224
        selected.append(row)
    mass=sum(r['W_dx_mse'] for r in selected)/horizontal['W'].sum()
    result['top3']=dict(selection_rule=RULE,episodes=selected,episode_fraction=3/512,
        W_error_mass_fraction=float(mass),W_error_mass_percent=float(100*mass),
        caution='Selected worst cases are not representative and do not establish a causal mechanism.')
    result['variant_interpretation']='Top3 all variant1 is an outcome-selected observation. Compare the full256/256 means, paired-world worse counts and top50/top100 mix; this analysis adds no significance test or broad population claim.'
    result['scientific_data_generated']=False;result['test_opened']=False;result['p0_authorized']=False
    result['reproduction']=dict(script_sha256=sha(__file__),python=sys.version,numpy=np.__version__)
    out.mkdir(parents=True,exist_ok=True)
    planned=[out/name for name in ('context.json','episode_diagnostics.csv','top3.csv','input_inventory.json')]
    if any(p.exists() for p in planned):raise FileExistsError('Use a new output directory; existing analysis will not be overwritten')
    write_json(out/'context.json',result)
    rows=[]
    for i,pair in enumerate(a['pair_ids']):
        for v in (0,1):
            row=dict(pair_id=int(pair),variant=v,true_pattern='both_right' if right[i,v] else ('both_left' if left[i,v] else ('opposite' if opposite[i,v] else 'zero')),
                     N_dx_mse=float(horizontal['N'][i,v]),W_dx_mse=float(horizontal['W'][i,v]))
            for obj,j in (('A',0),('B',1)):
                row[obj+'_true_dx']=float(truth[i,v,j])
                for arm in ('N','W'):
                    row[obj+'_'+arm+'_dx']=float(predictions[arm][i,v,AXES[j]]);row[obj+'_'+arm+'_wrong']=bool(wrong[arm][i,v,j])
                row[obj+'_transition']='fixed' if nw[i,v,j] and not ww[i,v,j] else ('broken' if not nw[i,v,j] and ww[i,v,j] else ('remained_wrong' if ww[i,v,j] else 'remained_correct'))
            rows.append(row)
    for name,data in (('episode_diagnostics.csv',rows),('top3.csv',selected)):
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    for name,record in input_hashes.items():
        p=scene/name
        if record!=dict(bytes=p.stat().st_size,sha256=sha(p)):raise ValueError('Input changed during read-only analysis')
    if sha(verification)!=original_verification_sha:raise ValueError('Snapshot verification changed')
    for name,digest in source_hashes.items():
        if sha(source_root/name)!=digest:raise ValueError('Source changed during analysis')
    write_json(out/'input_inventory.json',dict(inputs=input_hashes,source_hashes=source_hashes,input_scene_unchanged=True,
        script_sha256=sha(__file__),outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in planned[:-1]}))
    return dict(output=str(out),top3_error_mass_fraction=float(mass),N_to_W_sign_transitions=result['N_to_W_sign_transitions'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--verification',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=ROOT);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.scene,args.verification,args.source_root,args.out),indent=2))


if __name__=='__main__':main()
