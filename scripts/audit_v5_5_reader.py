#!/usr/bin/env python3
"""CPU-only structural audit of the archived frozen V5.4 W quadratic reader.

No refit, encoder execution, new RGB, new labels, or untouched split access.
Swapped/identical/cyclic feature tuples are algebraic probes of known cached features.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='4'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import v5_4_reader as reader


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def summary(x):
    x=np.asarray(x,dtype=np.float64)
    return dict(max_abs=float(np.abs(x).max()),rms=float(np.sqrt(np.mean(x*x))),
                per_axis_max_abs=np.max(np.abs(x),axis=0).tolist(),per_axis_rms=np.sqrt(np.mean(x*x,axis=0)).tolist())


def save_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def run(scene,out):
    start=time.monotonic();scene=Path(scene).resolve();out=Path(out).resolve()
    if out==scene or scene in out.parents:raise ValueError('Read-only input scene required')
    manifest=json.loads((scene/'archive_manifest.json').read_text());inputs={}
    def checked(name):
        p=scene/name;record=dict(bytes=p.stat().st_size,sha256=sha(p))
        if manifest['files'][name]!=record:raise ValueError('Sealed input hash mismatch: '+name)
        inputs[name]=record;return p
    for name in ('reader/frozen_model.npz','reader/freeze_receipt.json','reader/training_config.json',
                 'bridge_cache/cache.npz','bridge_cache/receipt.json','bridge_report/predictions.npz',
                 'bridge_report/report.json','freeze_publication.json'):checked(name)
    model,frozen=reader.load_frozen(scene/'reader')
    cache,receipt=reader.load_cache(scene/'bridge_cache/cache.npz',scene/'bridge_cache/receipt.json',
                                    'bridge',frozen,scene/'freeze_publication.json')
    report=json.loads((scene/'bridge_report/report.json').read_text())
    if report['predictions_sha256']!=inputs['bridge_report/predictions.npz']['sha256']:raise ValueError('Prediction report mismatch')
    with np.load(scene/'bridge_report/predictions.npz',allow_pickle=False) as z:archived=z['W_concat'].reshape(-1,8);targets=z['targets'].reshape(-1,8)
    if not np.array_equal(targets,cache['targets'].reshape(-1,8)):raise ValueError('Archived target mismatch')
    basis=model['fit_features'].reshape(-1,model['fit_features'].shape[-1]);reference=model['fit_view_indices']
    q=cache['features'].reshape(-1,basis.shape[-1]);index=np.arange(len(q)).reshape(-1,2)
    raw=reader.dot_gram(q,basis)
    state={key.split('__',1)[1]:value for key,value in model.items() if key.startswith('concat__')}
    rows=model['W_rows'];coef=model['W_concat__coefficients'][:,4:];intercept=model['W_concat__intercept'][4:]
    reverse_lookup={tuple(v):i for i,v in enumerate(reference)}
    reverse=np.array([reverse_lookup[(int(v),int(u))] for u,v in reference])
    if not np.array_equal(reverse[reverse],np.arange(len(reference))):raise ValueError('Reverse view mapping not involutive')
    arm_reverse=reverse[rows]-rows[0]
    if not np.array_equal(model['fit_targets'].reshape(-1,8)[reverse[rows],4:],-model['fit_targets'].reshape(-1,8)[rows,4:]):
        raise ValueError('FIT motion labels not exact reverse pairs')
    weights=np.bincount(reference[:,0],minlength=len(basis))/len(reference)
    weights2=np.bincount(reference[:,1],minlength=len(basis))/len(reference)
    if not np.array_equal(weights,weights2):raise ValueError('FIT slot marginals differ')
    # A common linear feature origin mu is fixed by the joint FIT marginal.
    mu=np.empty(basis.shape[1],dtype=np.float64)
    for lo in range(0,basis.shape[1],2048):mu[lo:lo+2048]=weights@basis[:,lo:lo+2048].astype(np.float64)
    fit_mu=reader.dot_gram(basis,mu[None]).ravel();query_mu=raw@weights
    centered=(raw-query_mu[:,None]-fit_mu[None,:]+np.dot(mu,mu))/float(state['linear_scale'])
    scale=float(state['quadratic_scale']);sumc=coef.sum(0)
    constant=intercept+(sumc-state['quadratic_columns'][rows]@coef+float(state['quadratic_grand'])*sumc)/scale
    def parts(pairs):
        answer={key:np.empty((len(pairs),4)) for key in ('full','single_x','single_y','cross','center_correction','linear_reconstruction_error')}
        for begin in range(0,len(pairs),128):
            take=pairs[begin:begin+128];a=centered[take[:,0]][:,reference[:,0]];b=centered[take[:,1]][:,reference[:,1]]
            rawcross=reader.view_gram(raw,take,reference,'concat')
            normalized=(rawcross-rawcross.mean(1,keepdims=True)-state['linear_columns']+state['linear_grand'])/state['linear_scale']
            answer['linear_reconstruction_error'][begin:begin+len(take)]=np.max(np.abs(a+b-normalized),axis=1)[:,None]
            answer['full'][begin:begin+len(take)]=reader.apply_shared_kernel(rawcross,state)[:,rows]@coef+intercept
            ar,br=a[:,rows],b[:,rows]
            answer['single_x'][begin:begin+len(take)]=(2*ar+ar*ar)@coef/scale
            answer['single_y'][begin:begin+len(take)]=(2*br+br*br)@coef/scale
            answer['cross'][begin:begin+len(take)]=2*(ar*br)@coef/scale
            answer['center_correction'][begin:begin+len(take)]=-np.mean((1+a+b)**2,axis=1)[:,None]*sumc/scale
        if any(not np.isfinite(v).all() for v in answer.values()):raise ValueError('Nonfinite audit output')
        return answer
    forward=parts(index);swapped=parts(index[:,::-1]);identical0=parts(np.repeat(index[:,:1],2,axis=1));identical1=parts(np.repeat(index[:,1:],2,axis=1))
    np.testing.assert_allclose(forward['full'],archived[:,4:],rtol=1e-7,atol=1e-9)
    recomposed=forward['single_x']+forward['single_y']+forward['cross']+constant+forward['center_correction']
    np.testing.assert_allclose(recomposed,forward['full'],rtol=1e-6,atol=1e-8)
    # An additive potential difference h(y)-h(x) has zero sum around any triangle.
    # z is a frame already present in the same authenticated cache, no new RGB.
    z=np.roll(index[:,0],1)
    yz=parts(np.column_stack((index[:,1],z)))['full'];zx=parts(np.column_stack((z,index[:,0])))['full']
    cycle=forward['full']+yz+zx
    single=forward['single_x']+forward['single_y'];cross=forward['cross']
    h_difference=forward['single_y']-identical0['single_y']
    target=targets[:,4:].astype(np.float64);den=np.mean((target-model['shared_target_mean'][4:])**2,axis=0)
    score={}
    for name,pred in (('full',forward['full']),('single_frame_terms_only',single),('cross_frame_term_only',cross)):
        error=np.mean((pred-target)**2,axis=0)
        score[name]=dict(per_axis_mse=error.tolist(),per_axis_skill=(1-error/den).tolist(),mean_motion_skill=float(np.mean(1-error/den)))
    statistics=dict(archived_prediction_reproduction=summary(forward['full']-archived[:,4:]),
        recomposition_error=summary(recomposed-forward['full']),coefficient_reverse_sum=summary(coef+coef[arm_reverse]),
        coefficient_sum=sumc.tolist(),intercept=intercept.tolist(),effective_constant=constant.tolist(),
        forward_plus_swapped=summary(forward['full']+swapped['full']),
        identical_frame0=summary(identical0['full']),identical_frame1=summary(identical1['full']),
        full=summary(forward['full']),single_frame_terms=summary(single),cross_frame_term=summary(cross),
        cross_plus_swapped=summary(cross+swapped['cross']),single_potential_difference_error=summary(single-h_difference),
        identical_cross=summary(identical0['cross']),triangle_cycle=summary(cycle),
        center_correction=summary(forward['center_correction']),linear_moment_reconstruction=summary(forward['linear_reconstruction_error']),
        cross_rms_over_full_rms=np.sqrt(np.mean(cross**2,axis=0)/np.mean(forward['full']**2,axis=0)).tolist(),
        component_correlation=[float(np.corrcoef(single[:,j],cross[:,j])[0,1]) for j in range(4)],
        cross_to_target_correlation=[float(np.corrcoef(cross[:,j],target[:,j])[0,1]) for j in range(4)],
        component_scores_descriptive=score)
    equation=dict(definition='a_i=<x-mu,u_i-mu>/s; b_i=<y-mu,v_i-mu>/s; s=joint linear trace; t=joint quadratic trace',
        expansion='f(x,y)=[sum c_i(2a_i+a_i^2)]+[sum c_i(2b_i+b_i^2)]+2 sum c_i a_i b_i, each divided by t, plus effective constant and finite-sum centering correction',
        centering='Constant=b+(sumc-columns_quadratic dot c+grand_quadratic*sumc)/t; correction=-mean_all8192[(1+a+b)^2]*sumc/t',
        theory='Reversal-balanced FIT and unique positive-regularized solution force motion antisymmetry. They do not force additive separability: antisymmetric bilinear x^T M y with M=-M^T remains possible.',
        additive_test='h(y)-h(x) requires zero mixed bilinear operator and zero triangle cycle; nonzero observed cross/cycle disproves exact global collapse.',
        interpretation='Single-frame terms are an arbitrary fitted linear-plus-quadratic feature potential, not an identified position decoder. Cross terms are feature interactions, not established correspondence, object binding or causal temporal reasoning.')
    result=dict(schema='v5.5.frozen-reader-structural-audit.1',posthoc=True,
        population='Original already evaluated Dev640..895;512episodes; four motion outputs only',arms=['frozen V5.4 W'],
        tolerance_for_descriptive_numerical_identity=1e-7,
        antisymmetry_holds_on_all_observed_cues=bool(statistics['forward_plus_swapped']['max_abs']<=1e-7),
        identical_inputs_zero_on_all_observed_cues=bool(max(statistics['identical_frame0']['max_abs'],statistics['identical_frame1']['max_abs'])<=1e-7),
        additive_single_frame_difference_disproved=bool(statistics['triangle_cycle']['max_abs']>1e-7 and statistics['cross_frame_term']['max_abs']>1e-7),
        statistics=statistics,algebra=equation,inputs=inputs,archive_manifest_sha256=sha(scene/'archive_manifest.json'),
        source_identity=frozen['sources'],source_commit=frozen['cache_identity']['source_commit'],frozen_reader_sha256=frozen['receipt_sha256'],
        script_sha256=sha(__file__),runtime=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),threads=4,
            elapsed_seconds=time.monotonic()-start,peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        no_refit=True,no_new_rgb=True,no_new_labels=True,test_opened=False,p0_authorized=False,
        limits='Algebraic probes of known features and descriptive opened-Dev component scores; no new confirmation or encoder localization. No reorientation/calibration/weight selection is applied.')
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'components.npz',pair_ids=np.repeat(cache['pair_ids'],2),variants=np.tile([0,1],256),
        targets=target,forward=forward['full'],swapped=swapped['full'],identical0=identical0['full'],identical1=identical1['full'],
        single_frame=single,cross_frame=cross,effective_constant=constant,center_correction=forward['center_correction'],cycle=cycle)
    save_json(out/'audit.json',result)
    with (out/'per_axis.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['axis','full_rms','single_rms','cross_rms','cross_full_rms_ratio','single_cross_correlation','full_skill','single_only_skill','cross_only_skill'])
        for j,name in enumerate(('A_dx','A_dy','B_dx','B_dy')):w.writerow([name,statistics['full']['per_axis_rms'][j],statistics['single_frame_terms']['per_axis_rms'][j],
            statistics['cross_frame_term']['per_axis_rms'][j],statistics['cross_rms_over_full_rms'][j],statistics['component_correlation'][j],
            score['full']['per_axis_skill'][j],score['single_frame_terms_only']['per_axis_skill'][j],score['cross_frame_term_only']['per_axis_skill'][j]])
    for name,record in inputs.items():
        if sha(scene/name)!=record['sha256']:raise ValueError('Archived input changed during audit')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=run(a.scene,a.out)
    print(json.dumps({k:r[k] for k in ('antisymmetry_holds_on_all_observed_cues','identical_inputs_zero_on_all_observed_cues',
        'additive_single_frame_difference_disproved','statistics','runtime')},indent=2))


if __name__=='__main__':main()
