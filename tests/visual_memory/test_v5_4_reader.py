"""Software-only V5.4 mathematical, frozen-prediction and provenance checks."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import v5_4_reader as reader
from v5_3_data import make_block as narrow_block
from v5_4_data import make_block as wide_block


def toy(blocks=4,width=4):
    rng=np.random.default_rng(1723)
    n=rng.normal(size=(blocks,6,width)).astype(np.float32)
    w=(rng.normal(size=(blocks,6,width))*1.3+.7).astype(np.float32)
    yn=np.stack([narrow_block('software_test',i).targets for i in range(blocks)])
    yw=np.stack([wide_block('software_test',i).targets for i in range(blocks)])
    return n,yn,w,yw,np.arange(blocks)


def explicit_polynomial(reference,query):
    mu=reference.mean(0);centered=reference-mu
    scale=np.mean(np.sum(centered**2,axis=1))
    def phi(x):
        x=x/np.sqrt(scale)
        return np.concatenate((np.ones((len(x),1)),np.sqrt(2)*x,
                               (x[:,:,None]*x[:,None,:]).reshape(len(x),-1)),axis=1)
    train,q=phi(centered),phi(query-mu)
    center=train.mean(0);train=train-center;q=q-center
    scale=np.mean(np.sum(train**2,axis=1))
    return train/np.sqrt(scale),q/np.sqrt(scale)


def test_both_refitted_arms_match_explicit_joint_features_and_augmented_intercept():
    n,yn,w,yw,ids=toy()
    arrays=reader.train_models(n,yn,w,yw,ids)
    reader.validate_model_arrays(arrays,blocks=len(n),width=n.shape[-1])
    unique=np.concatenate((n,w)).reshape(-1,n.shape[-1]).astype(np.float64)
    ref=unique[reader.block_indices(len(n)*2)].reshape(len(n)*32,-1)
    rng=np.random.default_rng(4154)
    query=rng.normal(size=(14,n.shape[-1]));indices=np.arange(14).reshape(7,2)
    phi,qphi=explicit_polynomial(ref,query[indices].reshape(7,-1))
    targets=np.concatenate((yn,yw)).reshape(-1,8).astype(np.float64)
    predictions=reader.predict_models(arrays,query,indices,chunk=2)
    np.testing.assert_array_equal(arrays['shared_target_mean'],targets.mean(0))
    assert np.max(np.abs(arrays['N_concat__target_mean'][:4]-arrays['W_concat__target_mean'][:4]))>1e-3
    for arm in reader.ARMS:
        rows=arrays[arm+'_rows'];k=phi[rows]@phi[rows].T
        # An arm subset is not centered in joint-reference RKHS coordinates.
        assert np.max(np.abs(k.mean(0)))>1e-3
        count=len(rows)
        augmented=np.block([[k+count*reader.ALPHA*np.eye(count),np.ones((count,1))],
                            [np.ones((1,count)),np.zeros((1,1))]])
        solution=np.linalg.solve(augmented,np.vstack((targets[rows],np.zeros((1,8)))))
        expected=qphi@phi[rows].T@solution[:-1]+solution[-1]
        np.testing.assert_allclose(predictions[arm+'_concat'],expected,rtol=2e-8,atol=1e-8)
        np.testing.assert_allclose(arrays[arm+'_concat__coefficients'].sum(0),0,atol=1e-8)
    # Equal arm size uses 4096*alpha in production, not 8192*alpha.
    assert reader.training_config()['regularization']=={'N':.0032,'W':.0032}
    assert set(predictions)=={'N_concat','W_concat'}


def test_arm_label_translation_changes_only_intercept_and_shared_scoring_baseline():
    n,yn,w,yw,ids=toy()
    first=reader.train_models(n,yn,w,yw,ids)
    shifted=yw.copy();shifted[:,:,:4]+=np.float32(.125)
    second=reader.train_models(n,yn,w,shifted,ids)
    q=np.random.default_rng(201).normal(size=(8,n.shape[-1]));indices=np.arange(8).reshape(4,2)
    before=reader.predict_models(first,q,indices);after=reader.predict_models(second,q,indices)
    np.testing.assert_allclose(before['N_concat'],after['N_concat'],atol=1e-11)
    expected=np.r_[np.full(4,.125),np.zeros(4)]
    np.testing.assert_allclose(after['W_concat']-before['W_concat'],np.repeat(expected[None],4,axis=0),atol=2e-9)
    np.testing.assert_array_equal(second['shared_target_mean']-first['shared_target_mean'],expected/2)


def test_duplicate_supervised_rows_preserves_mean_loss_under_fixed_joint_kernel():
    n,yn,w,yw,ids=toy()
    unique=np.concatenate((n,w)).reshape(-1,n.shape[-1]).astype(np.float64)
    idx=reader.block_indices(2*len(n));ref=unique[idx].reshape(32*len(n),-1)
    gram,_=reader.shared_kernel(ref@ref.T)
    y=np.concatenate((yn,yw)).reshape(-1,8).astype(np.float64)
    rows=np.arange(16*len(n));twice=np.repeat(rows,2)
    c,b=reader.ridge_fit(gram,y,rows,.01);c2,b2=reader.ridge_fit(gram,y,twice,.01)
    np.testing.assert_allclose(gram[:,rows]@c+b,gram[:,twice]@c2+b2,rtol=1e-11,atol=1e-12)


def test_nw_motion_must_match_and_frozen_prediction_never_refits(tmp_path,monkeypatch):
    args=toy();n,yn,w,yw,ids=args
    corrupt=yw.copy();corrupt[:,:,4:]*=np.float32(2)
    # Even labels that are internally consistent but not paired must fail.
    center=yw[:,0,:4];d=yw[:,0,4:]*2
    rows=np.stack((np.concatenate((center,d),1),np.concatenate((center,-d),1),
                   np.concatenate((center+d,d),1),np.concatenate((center-d,-d),1)),1)
    corrupt=np.repeat(rows,4,axis=1)
    with pytest.raises(ValueError,match='bitwise equal'):reader.train_models(n,yn,w,corrupt,ids)
    arrays=reader.train_models(*args);path=tmp_path/'frozen_model.npz';np.savez(path,**arrays)
    with np.load(path,allow_pickle=False) as z:restored={k:z[k] for k in z.files}
    def forbidden(*a,**kw):raise AssertionError('Fitting called during prediction')
    for name in ('train_models','ridge_fit','shared_kernel'):monkeypatch.setattr(reader,name,forbidden)
    reader.predict_models(restored,n.reshape(-1,n.shape[-1]),reader.block_indices(len(n)))
    with pytest.raises(ValueError,match='cannot fit'):reader.evaluate('fit',None,None,None,None,None)
    with pytest.raises(ValueError,match='Exactly 256'):
        reader.validate_arrays(dict(features=n,targets=yn,negative_targets=np.zeros((len(n),2,8),np.float32),
                                    block_ids=ids,role=np.array('fit')),'fit')
    with pytest.raises(ValueError,match='Only fit'):reader.validate_arrays({},'test')


def test_world_bootstrap_matches_independent_whole_world_recomputation():
    rng=np.random.default_rng(701)
    target=rng.normal(size=(7,2,8));mean=rng.normal(size=8)
    predictions={'N_concat':target+rng.normal(size=target.shape)*np.arange(1,8)[:,None,None],
                 'W_concat':target+rng.normal(size=target.shape)*.3}
    score=reader.score_bridge(predictions,target,mean,replicates=301)
    expected=[];rng=np.random.default_rng(reader.BOOT_SEED)
    for _ in range(301):
        ix=rng.integers(0,7,7);y=target[ix]
        difference=((predictions['N_concat'][ix]-y)**2-(predictions['W_concat'][ix]-y)**2).mean((0,1))
        denominator=((y-mean)**2).mean((0,1))
        expected.append([difference[[4,6]].mean(),(difference[4:]/denominator[4:]).mean()])
    ci=np.quantile(expected,[.025,.975],axis=0)
    np.testing.assert_allclose(score['primary']['dx_mse_paired_95ci'],ci[:,0],atol=1e-12)
    np.testing.assert_allclose(score['primary']['motion_nmse_descriptive_paired_95ci'],ci[:,1],atol=1e-12)
    assert score['bootstrap']['primary_endpoints']==1 and not score['p0_authorized']


def test_anchor_names_target_contrasts_and_no_negative_direction_cancellation():
    _,_,_,y,_=toy()
    negatives=np.stack([wide_block('software_test',i).negative_targets for i in range(len(y))])
    prediction={a+'_concat':y.copy() for a in reader.ARMS};neg={k:negatives.copy() for k in prediction}
    amplitude=np.arange(1,len(y)+1)*.01
    for key,p in neg.items():
        p[:,0,4:]=amplitude[:,None];p[:,1,4:]=-amplitude[:,None]
        if key.startswith('W'):p[:,:,4:]*=.5
    fit_rms=np.sqrt(np.mean(y[:,:,4:].astype(float)**2,axis=(0,1)))
    score=reader.score_anchor(prediction,y,neg,negatives,y.reshape(-1,8).mean(0),fit_rms,replicates=101)
    for output in score['readers'].values():
        for path in output['paths'].values():np.testing.assert_array_equal(path['end_start_per_coordinate_skill'],np.ones((2,4)))
    expected=np.sqrt(np.mean(amplitude**2))
    np.testing.assert_allclose(score['readers']['N_concat']['negative_control_motion_rms'],[expected]*4)
    np.testing.assert_array_equal(score['readers']['N_concat']['negative_control_motion_mean'],np.zeros(4))
    assert score['comparisons']['concat']['N_minus_W_negative_overall_motion_rms']==pytest.approx(expected/2)
    assert 'S_concat' not in json.dumps(score) and 'B_concat' not in json.dumps(score)


def test_narrow_envelope_counts_all_views_and_coordinates_without_new_gate():
    y=np.ones((3,16,8));y[:,:,:4]=.5;y[:,:,4:]=.1
    # p6 crosses the envelope only on one view; p7 itself stays within it.
    y[1,15,4]=.31
    # Another coordinate of another world crosses upper bound.
    y[2,0,1]=.81
    predictions={'N_concat':y+.2,'W_concat':y+.1};mean=np.zeros(8)
    spec=dict(lower=[.2]*4,upper=[.8]*4,tolerance=0)
    result=reader.score_spatial_support(predictions,y,mean,spec)
    assert result['world_groups']['inside']['worlds']==1
    assert result['world_groups']['outside']['worlds']==2
    assert result['world_groups']['inside']['examples']==16
    assert result['coordinate_groups']['A_dx']['outside']['worlds']==1
    assert result['coordinate_groups']['A_dy']['outside']['worlds']==1
    assert result['coordinate_groups']['B_dx']['outside']['readers']['W_concat'] is None
    assert result['world_groups']['inside']['readers']['W_concat']['dx_mse']==pytest.approx(.01)
    assert result['support_only'] and result['intervals'] is False


def test_publication_binds_exact_model_source_plan_and_readback_time(tmp_path):
    frozen=dict(receipt_sha256='a'*64,created_unix=40,cache_identity=dict(source_commit='b'*40,plan_sha256='c'*64))
    correct=dict(verified_readback=True,frozen_reader_sha256='a'*64,source_commit='b'*40,plan_sha256='c'*64,verified_unix=41)
    path=tmp_path/'publication.json';path.write_text(json.dumps(correct))
    assert reader.validate_publication(path,frozen)==reader.sha256(path)
    for key,value in (('verified_readback',False),('verified_unix',39),('frozen_reader_sha256','d'*64),('source_commit','e'*40)):
        bad=dict(correct);bad[key]=value;path.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='verified publication'):reader.validate_publication(path,frozen)


def test_source_identity_and_narrow_artifact_pins_are_not_interchangeable(tmp_path,monkeypatch):
    monkeypatch.setattr(reader,'ROOT',tmp_path)
    plan=tmp_path/'scripts/v5_4_plan.json';plan.parent.mkdir();monkeypatch.setattr(reader,'PLAN',plan)
    encoder=dict(source_commit='1'*40,checkpoint_sha256='2'*64,model='software_fake_encoder')
    narrow=dict(cache_sha256='a'*64,receipt_sha256='b'*64,manifest_sha256='c'*64,
                source_commit='3'*40,encoder_identity=encoder)
    plan.write_text(json.dumps(dict(encoder=encoder,narrow_reference=narrow)))
    required=set(reader.SOURCE_NAMES)|{'scripts/extract_v5_4.py','scripts/launch_v5_4.py',
        'scripts/v5_temporal_confirmation.py','docs/V5_4_EXECUTION_PROTOCOL.md'}
    for name in required:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path!=plan:path.write_text('software_only')
    identity=dict(plan_sha256=reader.sha256(plan),source_commit='4'*40,encoder_identity=encoder,
                  source_identity={name:reader.sha256(tmp_path/name) for name in required},
                  pooling='4x4_flattened',feature_dtype='float32')
    assert reader.validate_identity(identity)==identity
    assert reader.validate_narrow_reference(narrow)==narrow
    corrupted=dict(narrow);corrupted['source_commit']=identity['source_commit']
    with pytest.raises(ValueError,match='registered immutable'):reader.validate_narrow_reference(corrupted)
    artifact=tmp_path/'old.npz';artifact.write_bytes(b'software fixture, not scientific data')
    with pytest.raises(ValueError,match='Pinned narrow artifact changed'):
        reader.load_narrow(artifact,artifact,artifact,narrow)
    (tmp_path/'scripts/v5_3_reader.py').write_text('changed reused mathematical helper')
    with pytest.raises(ValueError,match='Source changed'):reader.validate_identity(identity)


def test_exact_original_receipts_and_matching_audit_survive_model_release():
    import hashlib
    args=toy();arrays=reader.train_models(*args)
    audit=reader.matched_label_audit(arrays)
    assert audit['motion_labels_bitwise_equal'] and audit['maximum_absolute_motion_difference']==0
    assert audit['motion_target_entries']['N']==audit['motion_target_entries']['W']
    assert audit['position_target_mean']['N']!=audit['position_target_mean']['W']
    ids=reader.entry_identity(args[-1]);encoder=dict(model='software_encoder')
    n=dict(schema='v5.3-nuisance-repair-1',role='fit',test_opened=False,source_commit='old',
           cache_sha256='ncache',encoder_identity=encoder,entry_hashes=dict(block_ids=ids))
    ntext=json.dumps(n,indent=3)+'\r\n'  # exact original whitespace and line endings matter
    ref=dict(cache_sha256='ncache',receipt_sha256=hashlib.sha256(ntext.encode()).hexdigest(),
             source_commit='old',encoder_identity=encoder)
    identity=dict(plan_sha256='plan',source_commit='new',encoder_identity=encoder,
                  source_identity={},pooling='4x4_flattened',feature_dtype='float32')
    w=dict(identity,schema=reader.SCHEMA,role='fit',test_opened=False,cache_sha256='wcache',
           narrow_reference=ref,freshness_audit_sha256='fresh',entry_hashes=dict(block_ids=ids))
    wtext=json.dumps(w,sort_keys=True)+'\n'
    frozen=dict(narrow_reference=ref,cache_identity=identity,fit_cache_sha256='wcache',
                freshness_audit_sha256='fresh',fit_cache_receipt_sha256=hashlib.sha256(wtext.encode()).hexdigest(),
                fit_receipt_utf8=dict(N=ntext,W=wtext))
    restored=json.loads(json.dumps(frozen))
    assert reader.validate_embedded_receipts(restored)=={'N':n,'W':w}
    restored['fit_receipt_utf8']['N']=ntext.replace('\r\n','\n')
    with pytest.raises(ValueError,match='receipt bytes changed'):reader.validate_embedded_receipts(restored)
