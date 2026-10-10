#!/usr/bin/env python3
"""Fixed CPU-only lane-restricted W reader on authenticated historical FIT.

fit --scene OLD_V54_SCENE --out MODEL_DIR
load_frozen(MODEL_DIR) -> (arrays, receipt)
predict(arrays, query_unique[N,20480], query_index[M,2]) -> float64[M,8]
Prediction consumes only features and integer indices; it never refits.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='4'
import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
import v5_4_reader as old

ROOT=Path(__file__).resolve().parents[1]
ALPHA=7.8125e-7
SCHEMA='v5.5-lane-reader-freeze-1'
LANES={'A':dict(tokens=np.arange(8),outputs=np.array([0,1,4,5])),
       'B':dict(tokens=np.arange(8,16),outputs=np.array([2,3,6,7]))}
SOURCE_NAMES=('scripts/v5_5_lane_reader.py','scripts/v5_5_plan.json','docs/V5_5_MODULE_DIAGNOSTIC.md',
              'scripts/v5_4_reader.py','scripts/v5_4_data.py','scripts/v5_4_plan.json',
              'scripts/v5_3_reader.py','scripts/v5_3_data.py','pmr/visual_memory/features.py')
NUMERICAL_ATOL=1e-7


def sources():return {name:old.sha256(ROOT/name) for name in SOURCE_NAMES}


def config():
    return dict(kernel='quadratic',reference='all archived N+W8192 views, independently masked per lane',
        supervised='same archived W4096 views;256 equally weighted worlds',mean_loss_alpha=ALPHA,
        ridge_diagonal=.0032,unpenalized_intercept=True,lanes={k:{p:v.tolist() for p,v in spec.items()} for k,spec in LANES.items()},
        feature_layout='16 raster-row-major spatial tokens,1280 channels per token',channels=1280,
        normalization='linear center/mean diagonal then quadratic center/mean diagonal, both all8192 reference rows',
        score_baseline='unchanged original frozen joint N/W FIT target mean and motion RMS',
        selection=False,cv=False,calibration=False,posthoc=True,test_opened=False,p0_authorized=False)


def masked(unique,lane,channels):
    x=np.asarray(unique)
    if x.ndim!=2 or x.shape[1]!=16*channels or lane not in LANES:raise ValueError('Raster token/channel shape mismatch')
    return x.reshape(len(x),16,channels)[:,LANES[lane]['tokens'],:].reshape(len(x),8*channels)


def train_models(base,channels=1280):
    """Pure math permits software-sized FIT; production fit validates the old freeze."""
    x=np.asarray(base['fit_features']);target=np.asarray(base['fit_targets']);index=base['fit_view_indices'];rows=base['W_rows']
    if (x.ndim!=3 or x.shape[1:]!=(6,16*channels) or len(x)%2 or
        target.shape!=(len(x),16,8) or not np.array_equal(index,old.block_indices(len(x))) or
        not np.array_equal(rows,np.arange(len(x)*8,len(x)*16))):raise ValueError('Joint reference or W rows differ')
    if not np.array_equal(target[:len(x)//2,:,4:],target[len(x)//2:,:,4:]):raise ValueError('Matched FIT motion labels differ')
    if not np.isfinite(x).all() or not np.isfinite(target).all():raise ValueError('Nonfinite FIT')
    outputs=dict(fit_features=x.astype(np.float32),fit_targets=target.astype(np.float32),
        fit_view_indices=np.asarray(index,dtype=np.int64),W_rows=np.asarray(rows,dtype=np.int64),
        fit_block_ids=np.asarray(base['fit_block_ids'],dtype=np.int64),mean_loss_alpha=np.array(ALPHA),
        shared_target_mean=np.asarray(base['shared_target_mean']),fit_motion_rms=np.asarray(base['fit_motion_rms']),
        channels=np.array(channels,dtype=np.int64))
    y=target.reshape(-1,8).astype(np.float64);flat=x.reshape(-1,x.shape[-1]);audit={}
    for lane,spec in LANES.items():
        start=time.monotonic();features=masked(flat,lane,channels)
        gram,state=old.shared_kernel(old.view_gram(old.dot_gram(features,features),index,index,'concat'))
        coef,bias=old.ridge_fit(gram,y[:,spec['outputs']],rows,ALPHA)
        own=gram[np.ix_(rows,rows)]@coef+bias
        identity_error=float(np.max(np.abs(own-(y[rows][:,spec['outputs']]-len(rows)*ALPHA*coef))))
        coefsum=float(np.max(np.abs(coef.sum(0))))
        if max(identity_error,coefsum)>NUMERICAL_ATOL:raise ValueError('Own-FIT normal equations failed')
        outputs.update({lane+'__'+k:v for k,v in state.items()})
        outputs.update({lane+'__coefficients':coef,lane+'__intercept':bias,
                        lane+'__tokens':spec['tokens'],lane+'__outputs':spec['outputs']})
        audit[lane]=dict(normal_equation_max_abs=identity_error,coefficient_sum_max_abs=coefsum,
            supervised_rows=len(rows),reference_rows=len(index),diagonal=len(rows)*ALPHA,
            linear_scale=float(state['linear_scale']),quadratic_scale=float(state['quadratic_scale']),
            elapsed_seconds=time.monotonic()-start)
        del gram,features
    if any(not np.isfinite(v).all() for v in outputs.values()):raise ValueError('Nonfinite model')
    return outputs,audit


def predict(arrays,query_unique,query_index,chunk=256):
    """Return [M,8] original column order using only feature queries, with no fit."""
    q=np.asarray(query_unique);idx=np.asarray(query_index);channels=int(arrays['channels'])
    if (q.ndim!=2 or q.shape[1]!=16*channels or not np.isfinite(q).all() or
        idx.ndim!=2 or idx.shape[1]!=2 or not np.issubdtype(idx.dtype,np.integer) or
        (idx.size and (idx.min()<0 or idx.max()>=len(q))) or chunk<1):raise ValueError('Invalid feature-only query')
    answer=np.empty((len(idx),8),dtype=np.float64);fit=arrays['fit_features'].reshape(-1,q.shape[1])
    for lane,spec in LANES.items():
        raw=old.dot_gram(masked(q,lane,channels),masked(fit,lane,channels))
        state={key:arrays[lane+'__'+key] for key in ('linear_columns','linear_grand','linear_scale','quadratic_columns','quadratic_grand','quadratic_scale')}
        for lo in range(0,len(idx),chunk):
            cross=old.view_gram(raw,idx[lo:lo+chunk],arrays['fit_view_indices'],'concat')
            kernel=old.apply_shared_kernel(cross,state)[:,arrays['W_rows']]
            answer[lo:lo+chunk,spec['outputs']]=kernel@arrays[lane+'__coefficients']+arrays[lane+'__intercept']
    if not np.isfinite(answer).all():raise ValueError('Nonfinite prediction')
    return answer


def self_test():
    """Independent explicit masked polynomial and augmented intercept software test."""
    rng=np.random.default_rng(55107);blocks=3;channels=2
    x=rng.normal(size=(2*blocks,6,16*channels)).astype(np.float32)
    # Arbitrary targets exercise all eight outputs, including nonzero offsets.
    y=rng.normal(size=(2*blocks,16,8)).astype(np.float32)+np.arange(8,dtype=np.float32)
    y[blocks:,:,4:]=y[:blocks,:,4:]
    flat=y.reshape(-1,8).astype(np.float64);index=old.block_indices(2*blocks)
    base=dict(fit_features=x,fit_targets=y,fit_view_indices=index,W_rows=np.arange(16*blocks,32*blocks),
        fit_block_ids=np.arange(blocks),shared_target_mean=flat.mean(0),fit_motion_rms=np.sqrt(np.mean(flat[:,4:]**2,0)))
    model,audit=train_models(base,channels=channels)
    query=rng.normal(size=(10,16*channels));idx=np.arange(10).reshape(-1,2);observed=predict(model,query,idx,chunk=2)
    errors={}
    for lane,spec in LANES.items():
        # Explicit concatenate and polynomial feature map, without kernel helpers.
        dims=np.concatenate([np.arange(int(t)*channels,(int(t)+1)*channels) for t in spec['tokens']])
        ref=x.reshape(-1,16*channels)[:,dims][index].reshape(len(index),-1).astype(np.float64)
        q=query[:,dims][idx].reshape(len(idx),-1)
        mu=ref.mean(0);ref-=mu;q-=mu;s=np.mean(np.sum(ref*ref,axis=1))
        def phi(z):
            z=z/np.sqrt(s)
            return np.concatenate((np.ones((len(z),1)),np.sqrt(2)*z,(z[:,:,None]*z[:,None,:]).reshape(len(z),-1)),axis=1)
        f,g=phi(ref),phi(q);mu=f.mean(0);f-=mu;g-=mu;t=np.mean(np.sum(f*f,axis=1));f/=np.sqrt(t);g/=np.sqrt(t)
        rows=base['W_rows'];n=len(rows);k=f[rows]@f[rows].T
        system=np.block([[k+n*ALPHA*np.eye(n),np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
        sol=np.linalg.solve(system,np.vstack((flat[rows][:,spec['outputs']],np.zeros((1,4)))))
        expected=g@f[rows].T@sol[:-1]+sol[-1]
        errors[lane]=float(np.max(np.abs(observed[:,spec['outputs']]-expected)))
        np.testing.assert_allclose(observed[:,spec['outputs']],expected,rtol=1e-7,atol=NUMERICAL_ATOL)
        # Changing the other lane's query dimensions must leave this output unchanged.
        altered=query.copy();other=np.setdiff1d(np.arange(query.shape[1]),dims);altered[:,other]+=111
        np.testing.assert_array_equal(predict(model,altered,idx)[:,spec['outputs']],observed[:,spec['outputs']])
    # Exact raster mapping is independent of random data and output ordering.
    sentinel=np.arange(16*channels).reshape(1,-1)
    np.testing.assert_array_equal(masked(sentinel,'A',channels),sentinel[:,:8*channels])
    np.testing.assert_array_equal(masked(sentinel,'B',channels),sentinel[:,8*channels:])
    return dict(status='passed',software_only=True,explicit_polynomial_augmented_system_max_abs=errors,normal_equations=audit)


def validate_arrays(arrays):
    base={'fit_features','fit_targets','fit_view_indices','W_rows','fit_block_ids','mean_loss_alpha',
          'shared_target_mean','fit_motion_rms','channels'}
    for lane in LANES:
        base|={lane+'__'+key for key in ('linear_columns','linear_grand','linear_scale','quadratic_columns','quadratic_grand',
                    'quadratic_scale','coefficients','intercept','tokens','outputs')}
    if set(arrays)!=base:raise ValueError('Model entry schema mismatch')
    if (arrays['fit_features'].shape!=(512,6,20480) or arrays['fit_features'].dtype!=np.float32 or
        arrays['fit_targets'].shape!=(512,16,8) or arrays['fit_targets'].dtype!=np.float32 or
        not np.array_equal(arrays['fit_block_ids'],np.arange(256)) or
        not np.array_equal(arrays['fit_view_indices'],old.block_indices(512)) or
        not np.array_equal(arrays['W_rows'],np.arange(4096,8192)) or
        int(arrays['channels'])!=1280 or float(arrays['mean_loss_alpha'])!=ALPHA):raise ValueError('Model dimensions or supervised rows mismatch')
    old.check_targets(arrays['fit_targets'])
    y=arrays['fit_targets'].reshape(-1,8).astype(np.float64)
    if (not np.array_equal(arrays['fit_targets'][:256,:,4:],arrays['fit_targets'][256:,:,4:]) or
        not np.array_equal(arrays['shared_target_mean'],y.mean(0)) or
        not np.array_equal(arrays['fit_motion_rms'],np.sqrt(np.mean(y[:,4:]**2,axis=0)))):raise ValueError('Scoring baseline or motion label mismatch')
    for lane,spec in LANES.items():
        if (arrays[lane+'__coefficients'].shape!=(4096,4) or arrays[lane+'__intercept'].shape!=(4,) or
            not np.array_equal(arrays[lane+'__tokens'],spec['tokens']) or not np.array_equal(arrays[lane+'__outputs'],spec['outputs'])):raise ValueError('Lane/output mapping mismatch')
        for level in ('linear','quadratic'):
            if (arrays[lane+'__'+level+'_columns'].shape!=(8192,) or arrays[lane+'__'+level+'_grand'].shape!=() or
                arrays[lane+'__'+level+'_scale'].shape!=() or float(arrays[lane+'__'+level+'_scale'])<=0):raise ValueError('Reference moments mismatch')
    if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite frozen model')


def fit(scene,out):
    start=time.time();scene=Path(scene).resolve();out=Path(out).resolve()
    if out==scene or scene in out.parents:raise ValueError('Read-only historical scene required')
    source=sources();plan=old.read_json(ROOT/'scripts/v5_5_plan.json');manifest=old.read_json(scene/'archive_manifest.json')
    if old.sha256(scene/'archive_manifest.json')!=plan['prior_manifest_sha256']:raise ValueError('Prior scene manifest mismatch')
    checked={}
    for name in ('reader/frozen_model.npz','reader/freeze_receipt.json','reader/training_config.json','freeze_publication.json'):
        p=scene/name;rec=dict(bytes=p.stat().st_size,sha256=old.sha256(p))
        if rec!=manifest['files'][name]:raise ValueError('Prior scene artifact mismatch: '+name)
        checked[name]=rec
    base,prior=old.load_frozen(scene/'reader')
    old.validate_publication(scene/'freeze_publication.json',prior)
    if prior['cache_identity']['source_commit']!=plan['prior_source_commit']:raise ValueError('Prior source mismatch')
    # Frozen full basis and targets must hash to the original per-arm FIT receipts.
    embedded=old.validate_embedded_receipts(prior)
    for arm,take in (('N',slice(0,256)),('W',slice(256,512))):
        for key,arr in (('features',base['fit_features'][take]),('targets',base['fit_targets'][take]),('block_ids',base['fit_block_ids'])):
            if old.entry_identity(arr)!=embedded[arm]['entry_hashes'][key]:raise ValueError('Original FIT entry differs: '+arm+'/'+key)
    software=self_test()
    out.mkdir(parents=True,exist_ok=False)
    model,numerical=train_models(base);validate_arrays(model)
    # Before any evaluated data are read, serialize the entire frozen reader.
    np.savez_compressed(out/'frozen_model.npz',**model)
    old.write_json(out/'training_config.json',config());old.write_json(out/'software_verification.json',software)
    if source!=sources():raise ValueError('Source changed during CPU fit; preserve partial output')
    for name,record in checked.items():
        if old.sha256(scene/name)!=record['sha256']:raise ValueError('Historical input changed during fit')
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    receipt=dict(schema=SCHEMA,status='frozen_before_factorial_evaluation',created_unix=time.time(),
        fit_before_publication=True,fit_source_commit_at_execution=git,source_identity=source,
        plan_sha256=source['scripts/v5_5_plan.json'],protocol_sha256=source['docs/V5_5_MODULE_DIAGNOSTIC.md'],
        fit_blocks=256,fit_views=4096,reference_views=8192,lanes=['A','B'],configuration=config(),
        original_frozen_reader_sha256=prior['receipt_sha256'],original_source_commit=prior['cache_identity']['source_commit'],
        original_model_sha256=checked['reader/frozen_model.npz']['sha256'],archive_manifest_sha256=plan['prior_manifest_sha256'],
        original_cache_identity=prior['cache_identity'],original_fit_receipt_utf8=prior['fit_receipt_utf8'],
        original_freeze_receipt_utf8=(scene/'reader/freeze_receipt.json').read_text(),
        original_publication_receipt_utf8=(scene/'freeze_publication.json').read_text(),
        inputs=checked,files={name:old.sha256(out/name) for name in ('frozen_model.npz','training_config.json','software_verification.json')},
        model_entry_hashes={key:old.entry_identity(value) for key,value in model.items()},numerical_checks=numerical,
        software_checks=software,environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),threads=4,
            peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),elapsed_seconds=time.time()-start,
        prediction_inputs='features_only',no_evaluation_read=True,test_opened=False,p0_authorized=False,
        command=sys.argv,limits='Fixed lane access and kernel geometry intervention; globally contextual tokens, no yellow removal or encoder correctness claim')
    old.write_json(out/'freeze_receipt.json',receipt)
    restored,_=load_frozen(out)
    for key in model:np.testing.assert_array_equal(restored[key],model[key])
    return receipt


def load_frozen(folder):
    folder=Path(folder);receipt=old.read_json(folder/'freeze_receipt.json')
    if (receipt.get('schema')!=SCHEMA or receipt.get('status')!='frozen_before_factorial_evaluation' or
        receipt.get('source_identity')!=sources() or receipt.get('configuration')!=config() or
        receipt.get('fit_blocks')!=256 or receipt.get('fit_views')!=4096 or receipt.get('reference_views')!=8192 or
        receipt.get('test_opened') is not False or receipt.get('p0_authorized') is not False or
        receipt.get('no_evaluation_read') is not True):raise ValueError('Frozen lane receipt mismatch')
    for name in ('frozen_model.npz','training_config.json','software_verification.json'):
        if receipt.get('files',{}).get(name)!=old.sha256(folder/name):raise ValueError('Frozen lane file changed: '+name)
    if old.read_json(folder/'training_config.json')!=config():raise ValueError('Training configuration changed')
    with np.load(folder/'frozen_model.npz',allow_pickle=False) as archive:model={key:archive[key] for key in archive.files}
    if receipt.get('model_entry_hashes')!={key:old.entry_identity(value) for key,value in model.items()}:raise ValueError('Frozen lane array identity changed')
    validate_arrays(model);receipt['receipt_sha256']=old.sha256(folder/'freeze_receipt.json')
    return model,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='phase',required=True)
    f=s.add_parser('fit');f.add_argument('--scene',type=Path,required=True);f.add_argument('--out',type=Path,required=True)
    s.add_parser('self-test');args=p.parse_args()
    result=self_test() if args.phase=='self-test' else fit(args.scene,args.out)
    print(json.dumps(result if args.phase=='self-test' else {key:result[key] for key in ('schema','status','numerical_checks','elapsed_seconds','files')},indent=2,allow_nan=False))


if __name__=='__main__':main()
