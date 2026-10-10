"""Software-only mathematical and protocol-boundary tests; no scientific roles."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import v5_3_reader as reader


def toy(blocks=4, width=5):
    rng = np.random.default_rng(144)
    x = rng.normal(size=(blocks,6,width)).astype(np.float32)
    center = np.round(rng.uniform(.3,.7,(blocks,4))*1024)/1024
    delta = np.round(rng.uniform(.01,.04,(blocks,4))*1024)/1024
    delta *= rng.choice([-1,1],delta.shape)
    targets = []
    for c,d in zip(center,delta):
        rows = np.stack((np.r_[c,d], np.r_[c,-d], np.r_[c+d,d], np.r_[c-d,-d]))
        targets.append(np.repeat(rows,4,axis=0))
    return x, np.asarray(targets,dtype=np.float32), np.arange(blocks)


def explicit_polynomial(reference, query):
    """Explicit finite features independently reproduce the registered kernel."""
    mean = reference.mean(0)
    centered = reference-mean
    scale = np.mean(np.sum(centered**2,axis=1))
    normalized, q = centered/np.sqrt(scale),(query-mean)/np.sqrt(scale)
    def phi(x):
        return np.concatenate((np.ones((len(x),1)),np.sqrt(2)*x,
                               (x[:,:,None]*x[:,None,:]).reshape(len(x),-1)),axis=1)
    train, query = phi(normalized),phi(q)
    mu = train.mean(0)
    train, query = train-mu,query-mu
    scale = np.mean(np.sum(train**2,axis=1))
    return train/np.sqrt(scale),query/np.sqrt(scale)


def test_shared_kernel_and_unpenalized_intercept_match_explicit_augmented_and_primal():
    x,y,_ = toy()
    index = reader.block_indices(len(x))
    flat = x.reshape(-1,x.shape[-1]).astype(np.float64)
    reference = flat[index].reshape(len(index),-1)
    rng = np.random.default_rng(919)
    queries = rng.normal(size=(7,reference.shape[1]))
    gram,state = reader.shared_kernel(reference@reference.T)
    cross = reader.apply_shared_kernel(queries@reference.T,state)
    phi,qphi = explicit_polynomial(reference,queries)
    np.testing.assert_allclose(gram,phi@phi.T,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(cross,qphi@phi.T,rtol=1e-12,atol=1e-12)
    static = (np.arange(len(x))[:,None]*16+reader.STATIC_VIEW_INDICES).ravel()
    assert np.max(np.abs(gram[np.ix_(static,static)].mean(0)))>1e-4
    target = y.reshape(-1,8).astype(np.float64)
    mean = target.mean(0)
    coef,intercept = reader.ridge_fit(gram,target,static,alpha=.02)
    n=len(static)
    augmented=np.block([[gram[np.ix_(static,static)]+n*.02*np.eye(n),np.ones((n,1))],
                        [np.ones((1,n)),np.zeros((1,1))]])
    explicit=np.linalg.solve(augmented,np.vstack((target[static],np.zeros((1,8)))))
    np.testing.assert_allclose(coef,explicit[:-1],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(intercept,explicit[-1],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(coef.sum(0),np.zeros(8),atol=1e-12)
    # Noncentered S changes the unpenalized intercept, especially position axes.
    assert np.max(np.abs(intercept[:4]-mean[:4]))>1e-4
    mu=phi[static].mean(0);centered=phi[static]-mu
    w = np.linalg.solve(centered.T@centered+n*.02*np.eye(phi.shape[1]),
                        centered.T@(target[static]-target[static].mean(0)))
    bias=target[static].mean(0)-mu@w
    np.testing.assert_allclose(cross[:,static]@coef+intercept,qphi@w+bias,rtol=1e-11,atol=1e-12)
    with pytest.raises(ValueError,match='all B-reference'):
        reader.apply_shared_kernel((queries@reference.T)[:,static],state)


def test_duplicate_static_views_preserve_world_normalized_fit():
    x,y,_ = toy()
    index = reader.block_indices(len(x))
    raw = reader.dot_gram(x.reshape(-1,5),x.reshape(-1,5))
    gram,_ = reader.shared_kernel(reader.view_gram(raw,index,index,'concat'))
    static = (np.arange(len(x))[:,None]*16+reader.STATIC_VIEW_INDICES).ravel()
    duplicate = np.repeat(static,2)
    flat = y.reshape(-1,8).astype(float);mean=flat.mean(0)
    original,b1 = reader.ridge_fit(gram,flat,static,alpha=.001)
    repeated,b2 = reader.ridge_fit(gram,flat,duplicate,alpha=.001)
    np.testing.assert_allclose(gram[:,static]@original+b1,gram[:,duplicate]@repeated+b2,rtol=1e-11,atol=1e-12)


def test_all_six_models_roundtrip_and_singleframe_distribution_identity(tmp_path):
    x,y,ids = toy()
    arrays = reader.train_models(x,y,ids)
    before = {k:v.copy() for k,v in arrays.items()}
    path = tmp_path/'model.npz'
    np.savez(path,**arrays)
    with np.load(path,allow_pickle=False) as z:
        restored = {k:z[k] for k in z.files}
    query = np.random.default_rng(401).normal(size=(9,5))
    indices = np.array([[0,1],[2,3],[0,4],[7,8]])
    pred = reader.predict_models(restored,query,indices,chunk=2)
    assert set(pred) == {arm+'_'+name for arm in reader.ARMS for name in reader.INPUTS}
    for name in ('t6','t7'):
        np.testing.assert_allclose(pred['S_'+name],pred['B_'+name],rtol=1e-8,atol=1e-9)
    assert np.max(np.abs(pred['S_concat']-pred['B_concat']))>1e-4
    for k,v in arrays.items():
        np.testing.assert_array_equal(v,before[k])
    np.testing.assert_allclose(reader.predict_models(restored,query,indices,chunk=99)['B_concat'],pred['B_concat'])


def test_bridge_bootstrap_retains_worlds_and_recomputes_nmse_denominator():
    rng = np.random.default_rng(707)
    y = rng.normal(size=(7,2,8))
    y[:,:,4:] += np.arange(7)[:,None,None]/3
    mean = rng.normal(size=8)
    predictions = {}
    for name in reader.INPUTS:
        predictions['S_'+name] = y+rng.normal(size=y.shape)*np.arange(1,8)[:,None,None]/7
        predictions['B_'+name] = y+rng.normal(size=y.shape)*.2
    score = reader.score_bridge(predictions,y,mean,replicates=301)
    expected = []
    rng = np.random.default_rng(reader.BOOT_SEED)
    for _ in range(301):
        idx = rng.integers(0,7,7)
        s = predictions['S_concat'][idx];b=predictions['B_concat'][idx];t=y[idx]
        mse = ((s-t)**2-(b-t)**2).mean((0,1))
        den = ((t-mean)**2).mean((0,1))
        expected.append([mse[[4,6]].mean(),(mse[4:]/den[4:]).mean()])
    ci = np.quantile(expected,[.025,.975],axis=0)
    np.testing.assert_allclose(score['primary']['dx_mse_paired_95ci'],ci[:,0],atol=1e-14)
    np.testing.assert_allclose(score['primary']['motion_nmse_descriptive_paired_95ci'],ci[:,1],atol=1e-14)
    assert not score['p0_authorized'] and score['bootstrap']['primary_endpoints']==1


def test_anchor_view_mapping_and_zero_motion_negative_controls():
    _,y,_ = toy()
    mean = y.reshape(-1,8).mean(0)
    negative_target=np.concatenate((y[:,0,:4],np.zeros((len(y),4))),axis=1)
    negative_target=np.repeat(negative_target[:,None,:],2,axis=1)
    prediction={arm+'_'+name:y.copy() for arm in reader.ARMS for name in reader.INPUTS}
    negatives={name:negative_target.copy() for name in prediction}
    fit_rms=np.sqrt(np.mean(y[:,:,4:]**2,axis=(0,1)))
    score=reader.score_anchor(prediction,y,negatives,negative_target,mean,fit_rms,replicates=101)
    for output in score['readers'].values():
        for path in output['paths'].values():
            np.testing.assert_array_equal(path['end_start_per_coordinate_skill'],np.ones((2,4)))
        np.testing.assert_array_equal(output['negative_control_motion_rms'],np.zeros(4))
    corrupted=y.copy();corrupted[:,2,4] *= -1
    with pytest.raises(ValueError,match='Nuisance schedules'):
        reader.check_targets(corrupted)


def test_negative_control_squares_both_directions_before_pooling_and_bootstrap():
    _,y,_=toy()
    mean=y.reshape(-1,8).mean(0)
    fit_rms=np.sqrt(np.mean(y[:,:,4:]**2,axis=(0,1)))
    targets=np.repeat(np.concatenate((y[:,0,:4],np.zeros((len(y),4))),axis=1)[:,None,:],2,axis=1)
    pred={arm+'_'+name:y.copy() for arm in reader.ARMS for name in reader.INPUTS}
    neg={key:targets.copy() for key in pred}
    amplitude=np.arange(1,len(y)+1)*.01
    for key in neg:
        neg[key][:,0,4:]=amplitude[:,None]
        neg[key][:,1,4:]=-amplitude[:,None]
        if key.startswith('B_'):neg[key][:,:,4:]*=.5
    score=reader.score_anchor(pred,y,neg,targets,mean,fit_rms,replicates=101)
    expected=float(np.sqrt(np.mean(amplitude**2)))
    np.testing.assert_allclose(score['readers']['S_concat']['negative_control_motion_rms'],[expected]*4)
    np.testing.assert_array_equal(score['readers']['S_concat']['negative_control_motion_mean'],np.zeros(4))
    rng=np.random.default_rng(reader.BOOT_SEED)
    independent=[.5*np.sqrt(np.mean(amplitude[rng.integers(0,len(y),len(y))]**2)) for _ in range(101)]
    ci=np.quantile(independent,[.025,.975])
    np.testing.assert_allclose(score['comparisons']['concat']['S_minus_B_negative_overall_motion_rms_descriptive_95ci'],ci)


def test_role_count_boundary_and_no_fitting_during_prediction(monkeypatch):
    x,y,ids=toy()
    data=dict(features=x,targets=y,block_ids=ids,role=np.array('fit'),
              negative_targets=np.zeros((len(x),2,8),np.float32))
    with pytest.raises(ValueError,match='Exactly 256'):
        reader.validate_arrays(data,'fit')
    with pytest.raises(ValueError,match='Only fit'):
        reader.validate_arrays(data,'test')
    arrays=reader.train_models(x,y,ids)
    def forbidden(*a,**k):
        raise AssertionError('Fitting or role loading occurred during frozen prediction')
    monkeypatch.setattr(reader,'train_models',forbidden)
    monkeypatch.setattr(reader,'ridge_fit',forbidden)
    monkeypatch.setattr(reader,'shared_kernel',forbidden)
    reader.predict_models(arrays,x.reshape(-1,x.shape[-1]),reader.block_indices(len(x)))
    monkeypatch.setattr(reader,'load_frozen',forbidden)
    with pytest.raises(ValueError,match='Evaluation cannot fit'):
        reader.evaluate('fit',None,None,None,None,None)


def test_publication_requires_exact_freeze_and_after_freeze_readback(tmp_path):
    frozen=dict(receipt_sha256='a'*64,created_unix=40,
                cache_identity=dict(source_commit='b'*40,plan_sha256='c'*64))
    valid=dict(verified_readback=True,frozen_reader_sha256='a'*64,verified_unix=41,
               source_commit='b'*40,plan_sha256='c'*64)
    path=tmp_path/'publication.json';path.write_text(json.dumps(valid))
    assert reader.validate_publication(path,frozen)==reader.sha256(path)
    for key,value in (('verified_readback',False),('verified_unix',39),('frozen_reader_sha256','d'*64),
                       ('plan_sha256','e'*64),('source_commit','f'*40)):
        wrong=dict(valid);wrong[key]=value;path.write_text(json.dumps(wrong))
        with pytest.raises(ValueError,match='verified publication'):
            reader.validate_publication(path,frozen)


def test_identity_enforces_plan_encoder_and_current_sources(tmp_path,monkeypatch):
    monkeypatch.setattr(reader,'ROOT',tmp_path)
    plan=tmp_path/'scripts/v5_3_plan.json';plan.parent.mkdir()
    encoder=dict(source_commit='1'*40,checkpoint_sha256='2'*64,model='vit_huge')
    plan.write_text(json.dumps(dict(encoder=encoder)))
    monkeypatch.setattr(reader,'PLAN',plan)
    required=set(reader.SOURCE_NAMES)|{'scripts/extract_v5_3.py','scripts/launch_v5_3.py',
        'scripts/v5_temporal_confirmation.py','docs/V5_3_EXECUTION_PROTOCOL.md'}
    for name in required:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path!=plan:path.write_text('software_only')
    identity=dict(plan_sha256=reader.sha256(plan),source_commit='3'*40,encoder_identity=encoder,
                  source_identity={name:reader.sha256(tmp_path/name) for name in required},
                  pooling='4x4_flattened',feature_dtype='float32')
    assert reader.validate_identity(identity)==identity
    (tmp_path/'scripts/v5_3_data.py').write_text('modified')
    with pytest.raises(ValueError,match='Source changed'):
        reader.validate_identity(identity)
