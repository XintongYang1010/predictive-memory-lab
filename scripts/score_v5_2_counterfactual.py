#!/usr/bin/env python3
"""Evaluate the unchanged V5.1 reader on four predeclared interventions.

Posthoc diagnosis on known Dev pairs, never training, selection, or gate repair.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from v5_confirmation_reader import load_frozen, predict_readers, sha256, validate_entry_hashes

CONDITIONS = ('original','static_t6','static_t7','swap_nuisance')
INDICES = np.array(((0,3),(0,2),(1,3),(1,2)))
UNIQUE_ORDER = ('t6@x6','t6@x7','t7@x6','t7@x7')
BOOTSTRAPS = 10000
BOOT_SEED = 7810


def paired_effect(predictions, target, fit_mean, replicates=BOOTSTRAPS):
    """Average losses over variants; static average is a loss, not an ensemble."""
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.shape != (len(y),2,4,8) or y.shape[1:] != (2,8):
        raise ValueError('Expected pair,variant,condition,target predictions')
    error = ((p-y[:,:,None,:])**2).mean(1)
    baseline = ((y-fit_mean)**2).mean(1)
    if np.any(baseline.mean(0)<=0):
        raise ValueError('Degenerate baseline')
    skills = 1-error.mean(0)/baseline.mean(0)
    deltas = error[:,0:1,4:]-error[:,1:,4:]
    static_delta = (deltas[:,0]+deltas[:,1])/2
    primary = float((static_delta.mean(0)/baseline[:,4:].mean(0)).mean())
    dx_error=error[:,:,[4,6]].mean(2)
    primary_dx=dx_error[:,0]-(dx_error[:,1]+dx_error[:,2])/2
    balanced_dx=(dx_error[:,0]+dx_error[:,3]-dx_error[:,1]-dx_error[:,2])/2
    rng = np.random.default_rng(BOOT_SEED)
    boot_primary=[];boot_delta=[];boot_dx=[];boot_balanced=[];boot_anchor_dx=[]
    for begin in range(0,replicates,250):
        ids=rng.integers(0,len(y),(min(250,replicates-begin),len(y)))
        den=baseline[ids,4:].mean(1)
        boot_primary.extend((static_delta[ids].mean(1)/den).mean(1))
        boot_delta.extend((deltas[ids].mean(1)/den[:,None,:]).mean(2))
        boot_dx.extend(primary_dx[ids].mean(1))
        boot_balanced.extend(balanced_dx[ids].mean(1))
        boot_anchor_dx.extend((dx_error[ids,0,None]-dx_error[ids,1:]).mean(1))
    intervals=np.quantile(np.asarray(boot_delta),[.025,.975],axis=0)
    dx_intervals=np.quantile(np.asarray(boot_anchor_dx),[.025,.975],axis=0)
    return dict(primary_dx_mse_reduction=float(primary_dx.mean()),
        primary_dx_paired_95ci=np.quantile(boot_dx,[.025,.975]).tolist(),
        secondary_balanced_dx_mse_reduction=float(balanced_dx.mean()),
        secondary_balanced_dx_paired_95ci=np.quantile(boot_balanced,[.025,.975]).tolist(),
        secondary_static_average_motion_nmse_reduction=primary,
        secondary_motion_paired_95ci=np.quantile(boot_primary,[.025,.975]).tolist(),
        conditions={name:dict(per_dimension_skill=skills[i].tolist(),
            per_dimension_mse=error[:,i].mean(0).tolist(),
            position_skill=float(skills[i,:4].mean()),motion_skill=float(skills[i,4:].mean()),
            motion_nmse=float(1-skills[i,4:].mean()),dx_mse=float(dx_error[:,i].mean())) for i,name in enumerate(CONDITIONS)},
        individual_comparisons={name:dict(nmse_reduction=float((deltas[:,i].mean(0)/baseline[:,4:].mean(0)).mean()),
            dx_mse_reduction=float((dx_error[:,0]-dx_error[:,i+1]).mean()),
            dx_mse_reduction_descriptive_95ci=dx_intervals[:,i].tolist(),
            descriptive_paired_95ci=intervals[:,i].tolist()) for i,name in enumerate(CONDITIONS[1:])},
        bootstrap=dict(replicates=replicates,seed=BOOT_SEED,unit='paired_world_with_all_variants_and_conditions',
            primary_endpoints=1,secondary_intervals='descriptive_unadjusted',
            interpretation='conditional on frozen reader; posthoc inspected Dev pairs, not fresh confirmation'))


def residual_diagnostics(predictions,target,pairs):
    # Original generator's exact integer-pixel yellow center displacement.
    distractor=np.array([(int((.5+.4*np.sin(7*2.3+int(i)))*224)-int((.5+.4*np.sin(6*2.3+int(i)))*224))/224 for i in pairs])
    rows=[]; summary={}
    for ci,condition in enumerate(CONDITIONS):
        residual=predictions[:,:,ci,:]-target
        out={}
        for d,name in enumerate(('A_dx','A_dy','B_dx','B_dy'),4):
            pred=predictions[:,:,ci,d];truth=target[:,:,d];r=residual[:,:,d]
            common=r.mean(1)
            common_mse=float(np.mean(common**2));total=float(np.mean(r**2))
            correlation=(float(np.corrcoef(common,distractor)[0,1])
                         if np.std(common)>0 and np.std(distractor)>0 else None)
            out[name]=dict(mse=total,mean_bias=float(r.mean()),pair_common_mse_fraction=common_mse/total if total>0 else None,
                pair_common_residual_corr_original_nuisance_dx=correlation,
                direction_error_rate=float(np.mean(np.sign(pred)!=np.sign(truth))),
                absolute_amplitude_ratio=float(np.abs(pred).mean()/np.abs(truth).mean()))
        summary[condition]=out
        for i,pair in enumerate(pairs):
            for variant in (0,1):
                row=dict(pair=int(pair),variant=variant,condition=condition,nuisance_original_dx=float(distractor[i]))
                for d,name in enumerate(('A_dx','A_dy','B_dx','B_dy'),4):
                    row[name+'_target']=float(target[i,variant,d]);row[name+'_prediction']=float(predictions[i,variant,ci,d])
                rows.append(row)
    return summary,rows


def original_parity(current,prior,target,fit_mean):
    difference=np.asarray(current,dtype=np.float64)-np.asarray(prior,dtype=np.float64)
    y=np.asarray(target,dtype=np.float64)
    rms=np.sqrt(np.mean(y[:,:,4:]**2,axis=(0,1)))
    if np.any(rms<=0):
        raise ValueError('Degenerate target RMS for numerical parity')
    rms_drift=np.sqrt(np.mean(difference[:,:,4:]**2,axis=(0,1)))/rms
    max_drift=np.max(np.abs(difference[:,:,4:]),axis=(0,1))/rms
    baseline=np.mean((y-fit_mean)**2,axis=(0,1))
    drift=np.abs(np.mean((current-y)**2-(prior-y)**2,axis=(0,1))/baseline)
    checks=dict(motion_rms_drift_fraction_at_most_001=bool(np.all(rms_drift<=.01)),
        motion_max_drift_fraction_at_most_01=bool(np.all(max_drift<=.1)),
        each_motion_skill_drift_at_most_002=bool(np.all(drift[4:]<=.02)))
    return dict(max_abs_prediction_difference=float(np.max(np.abs(difference))),
        motion_rms_drift_fraction=rms_drift.tolist(),motion_max_drift_fraction=max_drift.tolist(),
        all_dimension_skill_abs_drift=drift.tolist(),checks=checks,passed=all(checks.values()))


def nuisance_response(prediction,target,pairs):
    eta=np.array([(int((.5+.4*np.sin(7*2.3+int(i)))*224)-int((.5+.4*np.sin(6*2.3+int(i)))*224))/224 for i in pairs])
    response=(prediction[:,:,0,4:]-prediction[:,:,3,4:]).mean(1)/2
    original=(prediction[:,:,0,4:]-target[:,:,4:]).mean(1)
    swapped=(prediction[:,:,3,4:]-target[:,:,4:]).mean(1)
    result={}
    for d,name in enumerate(('A_dx','A_dy','B_dx','B_dy')):
        r=response[:,d]
        eligible=(eta!=0)&(r!=0)
        result[name]=dict(through_origin_slope=float(eta@r/(eta@eta)),
            signed_alignment_fraction=float(np.mean(eta[eligible]*r[eligible]>0)) if eligible.any() else None,
            aligned_count=int(np.sum(eta*r>0)),opposed_count=int(np.sum(eta*r<0)),
            zero_nuisance_count=int(np.sum(eta==0)),zero_response_count=int(np.sum(r==0)),
            residual_sign_reversal_count=int(np.sum(original[:,d]*swapped[:,d]<0)),
            residual_sign_tie_count=int(np.sum(original[:,d]*swapped[:,d]==0)),
            total_pairs=len(eta))
    return result


def evaluate(a):
    receipt=json.loads(a.receipt.read_text())
    if receipt.get('schema')!='v5_2-yellow-counterfactual-1' or receipt.get('role')!='counterfactual':
        raise ValueError('Counterfactual schema and role required')
    if receipt.get('cache_sha256')!=sha256(a.cache):
        raise ValueError('Cache digest mismatch')
    with np.load(a.cache,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    if 'entry_hashes' not in receipt:
        raise ValueError('Entry identity required')
    validate_entry_hashes(data,receipt['entry_hashes'])
    x,y,pairs=data['features'],data['targets'],data['pair_ids']
    if not np.array_equal(data.get('condition_indices'),INDICES) or tuple(data.get('condition_names',()))!=CONDITIONS:
        raise ValueError('Declared condition names or unique-image order mapping differs')
    if tuple(data.get('unique_image_order',()))!=UNIQUE_ORDER:
        raise ValueError('Unique-image order differs from the frozen protocol')
    if x.shape!=(256,2,4,20480) or y.shape!=(256,2,8) or not np.array_equal(pairs,np.arange(128,384)):
        raise ValueError('Exact fixed counts, feature width and original pair IDs required')
    if x.dtype!=np.float32 or y.dtype!=np.float32 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid cache dtype or values')
    arrays,selections,frozen=load_frozen(a.model)
    if receipt.get('frozen_reader_sha256')!=frozen['receipt_sha256']:
        raise ValueError('Frozen reader receipt mismatch')
    if (receipt.get('encoder_identity')!=frozen['cache_identity']['encoder_identity'] or
            receipt.get('original_cache_identity')!=frozen['cache_identity'] or
            receipt.get('pooling')!='4x4_flattened' or receipt.get('feature_dtype')!='float32'):
        raise ValueError('Encoder, original identity, pooling or feature storage differs')
    root=Path(__file__).resolve().parents[1]
    if not receipt.get('source_identity'):
        raise ValueError('Extraction source identity is required')
    for name,digest in receipt['source_identity'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(root/name)!=digest:
            raise ValueError('Execution source changed: '+name)
    with np.load(a.prior_predictions,allow_pickle=False) as z:
        prior={k:z[k] for k in z.files}
    if not np.array_equal(y,prior['targets']):
        raise ValueError('Original bridge target labels changed')
    index=(np.arange(512)[:,None,None]*4+INDICES[None,:,:]).reshape(-1,2)
    prediction={k:v.reshape(256,2,4,8) for k,v in predict_readers(arrays,selections,x.reshape(2048,-1),index).items()}
    scores={k:paired_effect(v,y,arrays[k+'__target_mean']) for k,v in prediction.items()}
    parity={k:original_parity(v[:,:,0,:],prior[k],y,arrays[k+'__target_mean']) for k,v in prediction.items()}
    parity_passed=all(v['passed'] for v in parity.values())
    diagnostics,rows=residual_diagnostics(prediction['concat'],y,pairs)
    reuse=dict(t6_original_static6=float(np.max(np.abs(prediction['t6'][:,:,0]-prediction['t6'][:,:,1]))),
        t6_static7_swap=float(np.max(np.abs(prediction['t6'][:,:,2]-prediction['t6'][:,:,3]))),
        t7_original_static7=float(np.max(np.abs(prediction['t7'][:,:,0]-prediction['t7'][:,:,2]))),
        t7_static6_swap=float(np.max(np.abs(prediction['t7'][:,:,1]-prediction['t7'][:,:,3]))))
    reuse_passed=all(v<=1e-10 for v in reuse.values())
    a.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.out/'predictions.npz',**prediction,targets=y,pair_ids=pairs,conditions=np.array(CONDITIONS))
    report=dict(schema='v5.2.nuisance-counterfactual.1',
        analysis='software_only_duplicate_features' if receipt.get('software_only') else 'posthoc_diagnostic_known_bridge',
        software_only=bool(receipt.get('software_only',False)),
        primary_reader='concat',conditions=list(CONDITIONS),scores=scores,original_prediction_parity=parity,
        status='diagnostic_complete' if parity_passed and reuse_passed else 'diagnostic_inconclusive_numerical_checks_failed',
        reused_single_frame_input_max_differences=reuse,reused_single_frame_inputs_passed=reuse_passed,
        nuisance_swap_response={k:nuisance_response(v,y,pairs) for k,v in prediction.items()},
        primary_residual_diagnostics=diagnostics,cache_sha256=sha256(a.cache),
        frozen_reader_sha256=frozen['receipt_sha256'],prior_predictions_sha256=sha256(a.prior_predictions),
        predictions_sha256=sha256(a.out/'predictions.npz'),test_access=False,p0_authorized=False,
        original_v5_gate='failed_unchanged',v5_1_bridge='failed_unchanged',
        interpretation='Intervention changes only yellow nuisance locations in the same known scenes; both static anchors reported; no model fit or selection')
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (a.out/'per_case.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'primary':scores['concat'],'original_parity':parity,'p0_authorized':False},indent=2))
    if not parity_passed or not reuse_passed:
        raise RuntimeError('Numerical checks failed; preserve reports and do not interpret the intervention')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','receipt','model','prior-predictions','out'):
        p.add_argument('--'+name,required=True,type=Path)
    evaluate(p.parse_args())


if __name__=='__main__':
    main()
