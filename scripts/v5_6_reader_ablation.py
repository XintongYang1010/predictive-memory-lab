#!/usr/bin/env python3
"""CPU-only, prespecified V5.6 reader interventions on archived FIT/known Dev.

self-test: synthetic arrays only.
fit: authenticate V5.4 FIT; freeze all kernels/alphas before opening V5.5 queries.
score: authenticate V5.5 eight-condition cache; report every fitted control.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import v5_4_reader as old

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'v5.6-reader-ablation-1'
ALPHA = 7.8125e-7
DF_ATOL = 1e-6
PARITY_ATOL = 1e-7
NUMERICAL_ATOL = 1e-7
PSD_REL_TOL = 1e-10
V54_SEAL = 'ec7bc9c17da8417e9ecd11ba0f21415a2df0ee7af620af9d12f911c3ab26e75c'
V55_SEAL = 'b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663'
SOURCE_NAMES = ('scripts/v5_6_reader_ablation.py', 'scripts/v5_4_reader.py',
                'scripts/v5_3_reader.py', 'scripts/v5_4_data.py', 'scripts/v5_3_data.py',
                'scripts/v5_5_lane_reader.py', 'scripts/score_v5_5.py',
                'scripts/v5_4_plan.json', 'scripts/v5_5_plan.json')
OUTPUTS = {'A': [0, 1, 4, 5], 'B': [2, 3, 6, 7]}
TOKENS = {'upper': np.arange(8), 'lower': np.arange(8, 16),
          'even': np.array([r*4+c for r in range(4) for c in range(4) if (r+c)%2 == 0]),
          'odd': np.array([r*4+c for r in range(4) for c in range(4) if (r+c)%2 == 1])}
KINDS = ('upper_native', 'lower_native', 'full', 'additive', 'upper_common',
         'lower_common', 'even_native', 'odd_native')
ARM_KERNELS = {'global': ('full', 'full'), 'additive': ('additive', 'additive'),
               'target_common': ('upper_common', 'lower_common'),
               'correct_lane': ('upper_native', 'lower_native'),
               'opposite_lane': ('lower_native', 'upper_native'),
               'checkerboard': ('even_native', 'odd_native')}


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def record(path):
    p = Path(path)
    return dict(bytes=p.stat().st_size, sha256=old.sha256(p))


def sources():
    return {name: old.sha256(ROOT/name) for name in SOURCE_NAMES}


def configuration():
    return dict(schema=SCHEMA, alpha=ALPHA, supervised='archived W 4096 views / 256 worlds',
        reference='same archived N+W 8192 views; no evaluation reference',
        output_groups=OUTPUTS, tokens={k:v.tolist() for k,v in TOKENS.items()},
        kernels=list(KINDS), arms={k:list(v) for k,v in ARM_KERNELS.items()},
        regularizations=['original', 'df_matched'], free_intercept=True,
        df='1+sum(eig(H_W K_WW H_W)/(eig+n*alpha)); target original corresponding native lane',
        df_atol=DF_ATOL, df_matching='FIT only, log-alpha bisection [-40,40]; unreachable retained as unavailable',
        negative_eigenvalues='reject below -1e-10*max(1,maxabs eigenvalue); clip only negative roundoff to zero',
        common_kernel='a=H G_up H/s_linear; b=H G_down H/s_linear; C components centered separately over entire FIT; original global quadratic scale',
        native_mask_kernel='same two-stage center/trace recipe independently fit to each fixed 8-token mask',
        parity_atol=PARITY_ATOL, numerical_atol=NUMERICAL_ATOL, blas_threads=4,
        selection=False, cv=False, alpha_search=False, random_projection=False,
        population='already opened Dev640..895; all 256 worlds x2 variants x8 masks',
        bootstrap=dict(seed=7816, replicates=10000, unit='whole_world', conditional_posthoc=True),
        test_opened=False, p0_authorized=False, gpu_requested=False)


def publication(path):
    p = Path(path); data = old.read_json(p)
    if data.get('verified_readback') is not True:
        raise ValueError('Published source readback required')
    checked=dict(sources())
    checked['docs/V5_6_EXECUTION_PROTOCOL.md']=old.sha256(ROOT/'docs/V5_6_EXECUTION_PROTOCOL.md')
    for name, digest in checked.items():
        value = data.get('files', {}).get(name)
        if (value.get('sha256') if isinstance(value, dict) else value) != digest:
            raise ValueError('Publication does not bind current source: '+name)
    return dict(path=str(p), **record(p), source_commit=data.get('source_commit', data.get('commit')),
                protocol_sha256=checked['docs/V5_6_EXECUTION_PROTOCOL.md'])


def authenticate(scene, seal, names):
    scene = Path(scene)
    if scene.is_symlink() or old.sha256(scene/'archive_manifest.json') != seal:
        raise ValueError('Archived scene manifest mismatch')
    manifest = old.read_json(scene/'archive_manifest.json'); checks = {}
    for name in names:
        p = scene/name
        if p.is_symlink() or record(p) != manifest['files'][name]:
            raise ValueError('Archived artifact changed: '+name)
        checks[name] = record(p)
    return checks


def mask(x, part):
    return np.asarray(x).reshape(len(x),16,-1)[:,TOKENS[part],:].reshape(len(x),-1)


def cross_views(unique_gram, q, ref):
    return unique_gram[np.ix_(q[:,0],ref[:,0])] + unique_gram[np.ix_(q[:,1],ref[:,1])]


def linear_moments(gram, index, chunk=256):
    n = len(index); columns = np.zeros(n)
    for lo in range(0,n,chunk):
        columns += cross_views(gram,index[lo:lo+chunk],index).sum(0)/n
    grand = float(columns.mean())
    diagonal = np.diag(gram)[index].sum(1)
    return dict(columns=columns, grand=np.array(grand), scale=np.array(max(float(diagonal.mean()-grand),1e-12)))


def linear_cross(gram, q, ref, moments, scale):
    raw = cross_views(gram,q,ref)
    return (raw-raw.mean(1,keepdims=True)-moments['columns']+moments['grand'])/scale


def part_names(kind):
    return ('upper','lower') if kind in ('full','additive') else (kind.split('_')[0],)


def component_cross(kind, grams, q, ref, linear, full_linear_scale):
    values = []
    for part in part_names(kind):
        scale = linear[part]['scale'] if kind.endswith('_native') else full_linear_scale
        values.append(linear_cross(grams[part],q,ref,linear[part],scale))
    if kind == 'full': return (1+values[0]+values[1])**2
    if kind == 'additive': return 2*values[0]+values[0]**2+2*values[1]+values[1]**2
    if kind.endswith('_native'): return (1+values[0])**2
    return 2*values[0]+values[0]**2


def construct_kernel(kind, grams, index, rows, linear, global_scales, chunk=256):
    n = len(index); columns = np.zeros(n); diagonal_sum = 0.; ww = np.empty((len(rows),len(rows)))
    row_lookup = {int(v):i for i,v in enumerate(rows)}
    for lo in range(0,n,chunk):
        stop = min(lo+chunk,n)
        c = component_cross(kind,grams,index[lo:stop],index,linear,global_scales[0])
        columns += c.sum(0)/n
        diagonal_sum += c[np.arange(stop-lo),np.arange(lo,stop)].sum()
        selected = [v for v in range(lo,stop) if v in row_lookup]
        if selected:
            ww[[row_lookup[v] for v in selected]] = c[np.ix_(np.array(selected)-lo,rows)]
    grand = float(columns.mean())
    scale = max(float(diagonal_sum/n-grand),1e-12) if kind.endswith('_native') else float(global_scales[1])
    ww = (ww-columns[rows,None]-columns[None,rows]+grand)/scale
    ww = (ww+ww.T)*.5
    state = dict(quadratic_columns=columns, quadratic_grand=np.array(grand), quadratic_scale=np.array(scale))
    return ww,state


def eigensystem(k):
    columns = k.mean(0); grand = float(columns.mean())
    centered = k-columns[None]-columns[:,None]+grand
    signed, vectors = np.linalg.eigh((centered+centered.T)*.5)
    tol = PSD_REL_TOL*max(1.,float(np.max(np.abs(signed))))
    if float(signed.min()) < -tol: raise ValueError('Materially indefinite kernel')
    values = np.maximum(signed,0.)
    return values,vectors,columns,grand,dict(min_signed_eigenvalue=float(signed.min()),
        max_eigenvalue=float(values.max()),negative_roundoff_count=int(np.sum(signed<0)),
        psd_tolerance=tol,positive_eigenvalue_count=int(np.sum(values>0)),
        eigenvalue_quantiles=np.quantile(signed,[0,.01,.1,.5,.9,.99,1]).tolist())


def effective_df(values, alpha):
    return float(1+np.sum(values/(values+len(values)*alpha)))


def match_df(values, target):
    low, high = -40.,40.
    attainable = (effective_df(values,np.exp(high)),effective_df(values,np.exp(low)))
    if not attainable[0] < target < attainable[1]:
        return dict(status='unreachable',target_df=float(target),attainable_df=list(attainable))
    for _ in range(160):
        middle = (low+high)/2; alpha = float(np.exp(middle)); observed = effective_df(values,alpha)
        if abs(observed-target)<=DF_ATOL: break
        if observed>target:low=middle
        else:high=middle
    if abs(observed-target)>DF_ATOL:raise ValueError('DF root did not converge')
    return dict(status='matched',target_df=float(target),actual_df=observed,alpha=alpha,
                residual=observed-target,attainable_df=list(attainable))


def solve(eigen, y, alpha, k):
    values,vectors,columns,grand,_ = eigen
    mean = y.mean(0); cy = y-mean
    coefficients = vectors@((vectors.T@cy)/(values[:,None]+len(values)*alpha))
    # Free-intercept constraint; the centering zero mode carries roundoff only.
    coefficients -= coefficients.mean(0)
    intercept = mean-columns@coefficients
    error = float(np.max(np.abs(k@coefficients+intercept+len(values)*alpha*coefficients-y)))
    constraint = float(np.max(np.abs(coefficients.sum(0))))
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all() or not np.isfinite([error,constraint]).all():
        raise ValueError('Nonfinite ridge solve')
    if max(error,constraint)>NUMERICAL_ATOL:
        raise ValueError('Ridge normal equations failed: '+str((error,constraint)))
    return coefficients,intercept,dict(normal_equation_max_abs=error,coefficient_sum_max_abs=constraint)


def fit_math(base, output):
    """Only supplied FIT arrays are accessed; kernels are processed sequentially."""
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    x=base['fit_features'].reshape(-1,base['fit_features'].shape[-1])
    index=base['fit_view_indices'];rows=base['W_rows'];y=base['fit_targets'].reshape(-1,8).astype(np.float64)[rows]
    grams={};linear={};scales=(float(base['concat__linear_scale']),float(base['concat__quadratic_scale']))
    for part in TOKENS:
        started=time.time();z=mask(x,part);grams[part]=old.dot_gram(z,z)
        linear[part]=linear_moments(grams[part],index)
        print(json.dumps(dict(stage='fit_linear',part=part,elapsed_seconds=time.time()-started)),flush=True)
    common=dict(fit_features=base['fit_features'],fit_view_indices=index,W_rows=rows,
        shared_target_mean=base['shared_target_mean'],fit_motion_rms=base['fit_motion_rms'],
        fit_targets=base['fit_targets'],fit_block_ids=base['fit_block_ids'],global_scales=np.array(scales))
    for part,moments in linear.items():
        for key,value in moments.items():common[part+'__'+key]=value
    np.savez_compressed(output/'basis.npz',**common)
    targets={};native={};summaries={}
    for part,axis in [('upper','A'),('lower','B')]:
        kind=part+'_native';k,state=construct_kernel(kind,grams,index,rows,linear,scales)
        eigen=eigensystem(k);targets[axis]=effective_df(eigen[0],ALPHA);native[kind]=(k,state,eigen)
        print(json.dumps(dict(stage='fit_target_df',group=axis,df=targets[axis])),flush=True)
    for kind in KINDS:
        started=time.time()
        if kind in native:k,state,eigen=native.pop(kind)
        else:
            k,state=construct_kernel(kind,grams,index,rows,linear,scales);eigen=eigensystem(k)
        arrays=dict(state,eigenvalues=eigen[0]);summary=dict(eigen[4],original_alpha=ALPHA,
            original_df=effective_df(eigen[0],ALPHA),matches={},solves={})
        choices={'original':dict(status='matched',alpha=ALPHA)}
        for group,target in targets.items():choices['df_'+group]=match_df(eigen[0],target)
        for choice,matched in choices.items():
            summary['matches'][choice]=matched
            if matched['status']!='matched':continue
            coef,bias,audit=solve(eigen,y,matched['alpha'],k)
            arrays[choice+'__coefficients']=coef;arrays[choice+'__intercept']=bias
            arrays[choice+'__alpha']=np.array(matched['alpha']);summary['solves'][choice]=audit
        np.savez_compressed(output/(kind+'.npz'),**arrays)
        summary['elapsed_seconds']=time.time()-started;summaries[kind]=summary
        save(output/(kind+'.json'),summary)
        print(json.dumps(dict(stage='fit_kernel_complete',kind=kind,seconds=summary['elapsed_seconds'],df=summary['original_df'])),flush=True)
        del k,state,eigen,arrays;gc.collect()
    save(output/'spectral_summary.json',dict(target_lane_df=targets,kernels=summaries))
    return summaries


def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:return {k:archive[k] for k in archive.files}


def predict_directory(folder, q, qindex, chunk=128):
    folder=Path(folder);basis=load_npz(folder/'basis.npz')
    flat=basis['fit_features'].reshape(-1,basis['fit_features'].shape[-1]);index=basis['fit_view_indices'];rows=basis['W_rows']
    linear={part:{k:basis[part+'__'+k] for k in ('columns','grand','scale')} for part in TOKENS}
    grams={part:old.dot_gram(mask(q,part),mask(flat,part)) for part in TOKENS}
    output={};unavailable={};summary=old.read_json(folder/'spectral_summary.json')
    for arm,parts in ARM_KERNELS.items():
        for setting in ('original','df_matched'):
            name=arm+'__'+setting;pred=np.empty((len(qindex),8));failed=[]
            for group,kind in zip(('A','B'),parts):
                choice='original' if setting=='original' else 'df_'+group
                if summary['kernels'][kind]['matches'][choice]['status']!='matched':failed.append(group);continue
                model=load_npz(folder/(kind+'.npz'))
                for lo in range(0,len(qindex),chunk):
                    c=component_cross(kind,grams,qindex[lo:lo+chunk],index,linear,float(basis['global_scales'][0]))
                    kernel=(c-c.mean(1,keepdims=True)-model['quadratic_columns']+model['quadratic_grand'])/model['quadratic_scale']
                    value=kernel[:,rows]@model[choice+'__coefficients']+model[choice+'__intercept']
                    pred[lo:lo+len(c),OUTPUTS[group]]=value[:,OUTPUTS[group]]
            if failed:unavailable[name]=dict(groups=failed,reason='FIT target df unreachable')
            else:
                if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
                output[name]=pred
            print(json.dumps(dict(stage='predict_arm',arm=name,status='unavailable' if failed else 'complete')),flush=True)
    return output,unavailable


def self_test():
    """Independent explicit feature kernels and augmented free-intercept solves."""
    rng=np.random.default_rng(56106);blocks=3;channels=1
    x=rng.normal(size=(blocks*2,6,16*channels)).astype(np.float32)
    index=old.block_indices(blocks*2);rows=np.arange(blocks*16,blocks*32)
    flat=x.reshape(-1,16);view=flat[index].reshape(len(index),-1).astype(np.float64)
    global_gram,global_state=old.shared_kernel(view@view.T)
    scales=(float(global_state['linear_scale']),float(global_state['quadratic_scale']))
    grams={part:old.dot_gram(mask(flat,part),mask(flat,part)) for part in TOKENS}
    linear={part:linear_moments(g,index,chunk=13) for part,g in grams.items()}
    centered=view-view.mean(0);up=np.concatenate([np.arange(8),np.arange(16,24)]);down=np.setdiff1d(np.arange(32),up)
    a=centered[:,up]@centered[:,up].T/scales[0];b=centered[:,down]@centered[:,down].T/scales[0]
    h=np.eye(len(index))-np.ones((len(index),len(index)))/len(index)
    explicit={'full':(1+a+b)**2,'additive':2*a+a*a+2*b+b*b,'upper_common':2*a+a*a,'lower_common':2*b+b*b}
    query=rng.normal(size=(10,16)).astype(np.float32);qi=np.arange(10).reshape(-1,2)
    qview=query[qi].reshape(len(qi),-1).astype(np.float64);qcenter=qview-view.mean(0)
    qa=qcenter[:,up]@centered[:,up].T/scales[0];qb=qcenter[:,down]@centered[:,down].T/scales[0]
    explicit_query={'full':(1+qa+qb)**2,'additive':2*qa+qa*qa+2*qb+qb*qb,
                    'upper_common':2*qa+qa*qa,'lower_common':2*qb+qb*qb}
    qgrams={part:old.dot_gram(mask(query,part),mask(flat,part)) for part in TOKENS}
    y=rng.normal(size=(len(rows),8))+np.arange(8);errors={};kernel_ww={}
    for kind in KINDS:
        k,state=construct_kernel(kind,grams,index,rows,linear,scales,chunk=13)
        if kind in explicit:
            truth=(h@explicit[kind]@h/scales[1])[np.ix_(rows,rows)]
            cquery=explicit_query[kind];ctrain=explicit[kind]
            qtruth=(cquery-cquery.mean(1,keepdims=True)-ctrain.mean(0)+ctrain.mean())/scales[1]
        else:
            part=kind.split('_')[0];features=mask(flat,part)[index].reshape(len(index),-1).astype(np.float64)
            zz=features-features.mean(0);zz/=np.sqrt(np.mean(np.sum(zz*zz,1)))
            # Explicit polynomial map has inner product (1+z.z')^2.
            def polynomial(z):return np.concatenate([np.ones((len(z),1)),np.sqrt(2)*z,(z[:,:,None]*z[:,None,:]).reshape(len(z),-1)],1)
            phi=polynomial(zz);phi_mean=phi.mean(0);phi-=phi_mean
            quadratic_norm=np.sqrt(np.mean(np.sum(phi*phi,1)));phi/=quadratic_norm
            truth=(phi@phi.T)[np.ix_(rows,rows)]
            qz=mask(query,part)[qi].reshape(len(qi),-1).astype(np.float64)-features.mean(0)
            qz/=np.sqrt(np.mean(np.sum((features-features.mean(0))**2,1)))
            qtruth=(polynomial(qz)-phi_mean)/quadratic_norm@phi.T
        query_raw=component_cross(kind,qgrams,qi,index,linear,scales[0])
        query_kernel=(query_raw-query_raw.mean(1,keepdims=True)-state['quadratic_columns']+state['quadratic_grand'])/state['quadratic_scale']
        np.testing.assert_allclose(query_kernel,qtruth,atol=1e-12,rtol=0)
        np.testing.assert_allclose(k,truth,atol=1e-12,rtol=0);kernel_ww[kind]=k
        eigen=eigensystem(k);coef,bias,audit=solve(eigen,y,ALPHA,k)
        augmented=np.block([[k+len(rows)*ALPHA*np.eye(len(rows)),np.ones((len(rows),1))],[np.ones((1,len(rows))),np.zeros((1,1))]])
        answer=np.linalg.solve(augmented,np.vstack([y,np.zeros((1,8))]))
        pred_error=float(np.max(np.abs(k@coef+bias-(k@answer[:-1]+answer[-1]))))
        if pred_error>NUMERICAL_ATOL:raise ValueError('Explicit augmented solve differs')
        query_pred_error=float(np.max(np.abs(query_kernel[:,rows]@coef+bias-(qtruth[:,rows]@answer[:-1]+answer[-1]))))
        if query_pred_error>NUMERICAL_ATOL:raise ValueError('Out-of-sample augmented solve differs')
        target=effective_df(eigen[0],ALPHA*3);matched=match_df(eigen[0],target)
        if matched['status']!='matched' or abs(matched['actual_df']-target)>DF_ATOL:raise ValueError('DF self-test')
        if match_df(eigen[0],len(rows)+2)['status']!='unreachable':raise ValueError('Unreachable df accepted')
        errors[kind]=dict(kernel_max_abs=float(np.max(np.abs(k-truth))),query_kernel_max_abs=float(np.max(np.abs(query_kernel-qtruth))),
                         augmented_prediction_max_abs=pred_error,query_prediction_max_abs=query_pred_error,**audit)
    np.testing.assert_allclose(kernel_ww['full'],global_gram[np.ix_(rows,rows)],atol=1e-12,rtol=0)
    np.testing.assert_allclose(kernel_ww['additive'],kernel_ww['upper_common']+kernel_ww['lower_common'],atol=1e-12,rtol=0)
    cross=(h@(2*a*b)@h/scales[1])[np.ix_(rows,rows)]
    np.testing.assert_allclose(kernel_ww['full'],kernel_ww['additive']+cross,atol=1e-12,rtol=0)
    return dict(status='passed',software_only=True,seed=56106,blocks=blocks,channels=channels,
        source_identity=sources(),checks=errors,kernel_decomposition=True,unreachable_df_rejected=True)


def fit(a):
    start=time.time();pub=publication(a.publication);src=sources()
    names=('reader/frozen_model.npz','reader/freeze_receipt.json','reader/training_config.json','freeze_publication.json')
    inputs=authenticate(a.scene,V54_SEAL,names);base,frozen=old.load_frozen(a.scene/'reader')
    old.validate_publication(a.scene/'freeze_publication.json',frozen)
    embedded=old.validate_embedded_receipts(frozen)
    for arm,take in (('N',slice(0,256)),('W',slice(256,512))):
        for key,value in [('features',base['fit_features'][take]),('targets',base['fit_targets'][take]),('block_ids',base['fit_block_ids'])]:
            if old.entry_identity(value)!=embedded[arm]['entry_hashes'][key]:raise ValueError('FIT basis identity differs')
    if a.out.resolve()==a.scene.resolve() or a.scene.resolve() in a.out.resolve().parents:raise ValueError('Historical scene is read-only')
    software=self_test();fit_math(base,a.out)
    save(a.out/'configuration.json',configuration());save(a.out/'software_verification.json',software)
    if sources()!=src:raise ValueError('Source changed during fitting')
    files={p.name:record(p) for p in sorted(a.out.iterdir()) if p.is_file()}
    freeze=dict(schema=SCHEMA,status='frozen_before_v56_evaluation',source_identity=src,publication=pub,
        configuration=configuration(),v54_manifest_sha256=V54_SEAL,inputs=inputs,
        original_frozen_reader_sha256=frozen['receipt_sha256'],files=files,fit_worlds=256,
        supervised_rows=4096,reference_rows=8192,no_evaluation_read=True,test_opened=False,p0_authorized=False,
        created_unix=time.time(),elapsed_seconds=time.time()-start,command=sys.argv,
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),blas_threads=4,
                         peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    save(a.out/'freeze_receipt.json',freeze)
    return dict(status=freeze['status'],elapsed_seconds=freeze['elapsed_seconds'],files=len(files))


def load_model(folder):
    folder=Path(folder);freeze=old.read_json(folder/'freeze_receipt.json')
    if freeze['schema']!=SCHEMA or freeze['status']!='frozen_before_v56_evaluation' or freeze['source_identity']!=sources() or freeze['configuration']!=configuration():raise ValueError('Frozen model source/configuration differs')
    actual={p.name:record(p) for p in folder.iterdir() if p.is_file() and p.name!='freeze_receipt.json'}
    if actual!=freeze['files']:raise ValueError('Frozen model inventory changed')
    return freeze


def score(a):
    from score_v5_5 import factorial_targets, stats, original_direction
    start=time.time();pub=publication(a.publication);freeze=load_model(a.model)
    if pub['protocol_sha256']!=freeze['publication']['protocol_sha256']:
        raise ValueError('Execution protocol changed after FIT freeze')
    original=authenticate(a.scene,V54_SEAL,tuple(freeze['inputs']))
    if original!=freeze['inputs']:raise ValueError('Scoring uses a different original FIT scene')
    names=('counterfactual_cache/cache.npz','counterfactual_cache/receipt.json','report/predictions.npz','report/report.json')
    inputs=authenticate(a.v55_scene,V55_SEAL,names)
    data=load_npz(a.v55_scene/'counterfactual_cache/cache.npz');archived=load_npz(a.v55_scene/'report/predictions.npz')
    receipt=old.read_json(a.v55_scene/'counterfactual_cache/receipt.json')
    if receipt['entry_hashes']!={k:old.entry_identity(v) for k,v in data.items()}:raise ValueError('V55 cache entries changed')
    if data['features'].shape!=(256,2,8,20480) or not np.array_equal(data['pair_ids'],np.arange(640,896)):raise ValueError('Population changed')
    y=factorial_targets(data['original_targets'],data['endpoint_positions'])
    np.testing.assert_array_equal(y,archived['targets'])
    q=data['features'].reshape(-1,20480);idx=np.stack((np.repeat(np.arange(512)*8,8),np.arange(4096)),1)
    a.out.mkdir(parents=True,exist_ok=False)
    predictions,unavailable=predict_directory(a.model,q,idx)
    predictions={k:v.reshape(256,2,8,8) for k,v in predictions.items()}
    parity={name:float(np.max(np.abs(predictions[name]-archived[oldname]))) for name,oldname in [('global__original','W_concat'),('correct_lane__original','lane_W')]}
    np.savez_compressed(a.out/'predictions.npz',targets=y,**predictions)
    save(a.out/'parity.json',dict(atol=PARITY_ATOL,rtol=0,max_abs=parity,passed=all(v<=PARITY_ATOL for v in parity.values())))
    if any(v>PARITY_ATOL for v in parity.values()):raise ValueError('Original baseline parity failed; preserved outputs')
    basis=load_npz(a.model/'basis.npz');mean=basis['shared_target_mean'];rms=basis['fit_motion_rms']
    rng=np.random.default_rng(7816);samples=rng.integers(0,256,(10000,256));summaries={};baseline=predictions['global__original']
    for name,p in predictions.items():
        delta=((baseline-y)**2-(p-y)**2).mean(1)[...,[4,6]].mean(-1)
        ci=np.quantile(delta[samples].mean(1),[.025,.975],axis=0)
        summaries[name]=dict(original=old.score_outputs(p[...,7,:],y[...,7,:],mean),
            original_direction=original_direction(p[...,7,:],y[...,7,:]),
            masks={format(m,'03b'):dict(**stats(p[...,m,:],y[...,m,:],rms),
                global_minus_arm_dx_mse=float(delta[:,m].mean()),paired_95ci=ci[:,m].tolist()) for m in range(8)})
    with (a.out/'per_episode.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pair_id','variant','mask','arm']+[k+'_'+str(i) for k in ('true','pred') for i in range(8)]+['dx_mse'])
        for i,pair in enumerate(data['pair_ids']):
            for v in (0,1):
                for m in range(8):
                    for name,p in predictions.items():writer.writerow([int(pair),v,format(m,'03b'),name]+y[i,v,m].tolist()+p[i,v,m].tolist()+[float(np.mean((p[i,v,m,[4,6]]-y[i,v,m,[4,6]])**2))])
    report=dict(schema=SCHEMA,status='complete',configuration=configuration(),parity=parity,readers=summaries,
        unavailable=unavailable,population=dict(worlds=256,episodes=512,masks=8,known_dev=True),
        source_identity=sources(),publication=pub,inputs=inputs,v55_manifest_sha256=V55_SEAL,
        frozen_model_receipt=record(a.model/'freeze_receipt.json'),elapsed_seconds=time.time()-start,
        conclusion_scope='Exploratory known-Dev interventions; matching df does not match anisotropic shrinkage, and one matched alpha does not rule out all scalar regularization.',
        test_opened=False,p0_authorized=False)
    save(a.out/'report.json',report)
    save(a.out/'manifest.json',{p.name:record(p) for p in sorted(a.out.iterdir()) if p.is_file()})
    return dict(status='complete',arms=list(predictions),unavailable=unavailable,parity=parity,elapsed_seconds=time.time()-start)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='phase',required=True)
    sub.add_parser('self-test')
    f=sub.add_parser('fit');f.add_argument('--scene',type=Path,required=True);f.add_argument('--out',type=Path,required=True);f.add_argument('--publication',type=Path,required=True)
    s=sub.add_parser('score');s.add_argument('--scene',type=Path,required=True);s.add_argument('--v55-scene',type=Path,required=True);s.add_argument('--model',type=Path,required=True);s.add_argument('--out',type=Path,required=True);s.add_argument('--publication',type=Path,required=True)
    args=parser.parse_args();result=self_test() if args.phase=='self-test' else fit(args) if args.phase=='fit' else score(args)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
