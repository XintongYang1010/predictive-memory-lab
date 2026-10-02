import torch
import pytest
from pmr.adapter import evidence,rotation,observed_only,RealMotionAdapter,FEATURES
from pmr.common import clone
from pmr.experiment import correctness,rollout,replay,baseline_adapter_check
from pmr.metrics import scores,frame_scores
from pmr.data import synthetic_raw,create_synthetic,make_manifest
from pmr.upstream import activate


def test_noop_isolation_future_blindness_scene_reset(adapter,sequence):
    assert all(correctness(adapter,sequence).values())


def test_world_residual_preserves_absolute_error(adapter,sequence):
    _,state=adapter.step(sequence[0])
    frame=sequence[1]
    actual=frame['x_positions'][0,0,-10:]@rotation(frame['theta'][0]).T+frame['origin'][0]
    state['memory']['glo_y_hat'][0,:,:10]=actual-state['memory']['origin'][0]
    torch.testing.assert_close(evidence(frame,state)['mse'],torch.zeros(6),atol=1e-9,rtol=0)
    # A common 4m translation of old predictions is a genuine conflict.
    state['memory']['glo_y_hat'][...,0]+=4
    torch.testing.assert_close(evidence(frame,state)['mse'],torch.full((6,),16.),atol=1e-4,rtol=0)


def test_rigid_transform_invariance(adapter,sequence):
    _,state=adapter.step(sequence[0])
    frame=sequence[1]
    before=evidence(frame,state)['mse']
    phi=torch.tensor(.7)
    rot=rotation(phi)
    shift=torch.tensor([200.,-300.])
    frame2,state2=clone(frame),clone(state)
    frame2['origin']=frame2['origin']@rot.T+shift
    frame2['theta']+=phi
    mem=state2['memory']
    mem['origin']=mem['origin']@rot.T+shift
    mem['theta']+=phi
    mem['glo_y_hat']=mem['glo_y_hat']@rot.T
    torch.testing.assert_close(evidence(frame2,state2)['mse'],before,atol=2e-3,rtol=2e-5)


def test_evidence_only_new_observed_overlap_and_no_compounding(adapter,sequence):
    _,state=adapter.step(sequence[0])
    first=evidence(sequence[1],state)
    second=evidence(sequence[1],state)
    assert (first['from_step'],first['until_step'],first['valid_count'])==(30,40,10)
    torch.testing.assert_close(first['mse'],second['mse'])
    altered=clone(state)
    altered['memory']['glo_y_hat'][:,:,10:]+=10000
    torch.testing.assert_close(evidence(sequence[1],altered)['mse'],first['mse'])
    no_new=evidence(sequence[0],state)
    assert no_new['valid_count']==0


def test_missing_overlap_is_not_zero_error_evidence(adapter,sequence):
    _,state=adapter.step(sequence[0])
    frame=clone(sequence[1]);frame['x_valid_mask'][0,0,-10:]=False
    e=evidence(frame,state)
    assert e['valid_count']==0 and e['features'][FEATURES.index('has_evidence')]==0


def test_keep_matches_hook_free_pinned_model(adapter,sequence):
    assert baseline_adapter_check(adapter,sequence)['passed']


def test_stage_and_rebuild_features_are_causal(adapter,sequence):
    _,states,_=rollout(adapter,sequence)
    for i,time in ((1,4.),(2,5.)):
        f=evidence(sequence[i],states[i-1])['features']
        assert f[FEATURES.index('update_time_seconds')]==time
        assert f[FEATURES.index('rebuild_available')]==(i==2)
        altered=clone(sequence[i]);altered['target']+=1e6
        torch.testing.assert_close(evidence(altered,states[i-1])['features'],f,atol=0,rtol=0)


def test_replay_logs_full_horizon_target_independently_of_common_loss(adapter,sequence):
    _,states,_=rollout(adapter,sequence)
    branch=replay(adapter,sequence,1,states[0],'reset')
    first,state=adapter.step(sequence[1],states[0],'reset')
    final,_=adapter.step(sequence[2],state,'keep')
    assert branch['final']==frame_scores(final,sequence[2])
    assert branch['final']['valid_steps']==60
    assert all(x['valid_steps']==50 for x in branch['curve'])


def test_random_deletion_matches_selective_count(adapter,sequence):
    _,state=adapter.step(sequence[0])
    masks=[]
    handle=adapter.model.traj_interact.register_forward_pre_hook(lambda m,a,k: masks.append(k.get('mask').clone()) if k.get('mask') is not None else None,with_kwargs=True)
    try:
        adapter.step(sequence[1],state,'drop_worst',drop_count=2)
        adapter.step(sequence[1],state,'random_drop',drop_count=2,seed=9)
        adapter.step(sequence[1],state,'random_drop',drop_count=2,seed=9)
    finally:handle.remove()
    assert all(int(torch.isneginf(m[0]).sum())==2 for m in masks)
    assert torch.equal(masks[1],masks[2])


def test_reset_skips_only_trajectory_correction(adapter,sequence):
    _,state=adapter.step(sequence[0])
    decoded=[]
    handle=adapter.model.decoder.register_forward_hook(lambda m,a,o: decoded.append(o[0].detach().clone()))
    try: result,newstate=adapter.step(sequence[1],state,'reset')
    finally:handle.remove()
    torch.testing.assert_close(result['y_hat'],decoded[-1],atol=0,rtol=0)
    assert 'x_encoder' in newstate['memory']


def test_rebuild_first_event_noop_and_second_event_extra_forward(adapter,sequence):
    _,states,_=rollout(adapter,sequence)
    keep,_=adapter.step(sequence[1],states[0],'keep')
    rebuild,_=adapter.step(sequence[1],states[0],'rebuild')
    torch.testing.assert_close(keep['y_hat'],rebuild['y_hat'],atol=0,rtol=0)
    calls=adapter.forward_calls
    adapter.step(sequence[2],states[1],'rebuild')
    assert adapter.forward_calls-calls==2


def test_future_mutation_before_preprocessing_cannot_change_current_input():
    _,module=activate()
    raw=synthetic_raw('future',11)
    other=clone(raw)
    other['x_positions'][:,40:]+=9999
    other['x_valid_mask'][:,40:]=False
    ds=module.Av2Dataset('.',split='train',num_historical_steps=30,split_points=[30,40,50])
    fa=module.collate_fn([[ds.process_single_agent(raw,40)]])[0]
    fb=module.collate_fn([[ds.process_single_agent(other,40)]])[0]
    for key,val in observed_only(fa).items():
        if isinstance(val,torch.Tensor): torch.testing.assert_close(val,fb[key],atol=0,rtol=0)


def test_common_target_window_matches_world_coordinates(sequence):
    targets=[]
    for frame in sequence[1:]:
        step=round(float(frame['timestamp'][0])*10)
        local=frame['target'][0,0,50-step:100-step]
        targets.append(local@rotation(frame['theta'][0]).T+frame['origin'][0])
    torch.testing.assert_close(targets[0],targets[1],atol=2e-5,rtol=1e-6)


def test_energy_score_known_mixture():
    out=scores(torch.tensor([[[0.,0.]],[[2.,0.]]]),torch.tensor([.5,.5]),torch.tensor([[1.,0.]]),torch.tensor([True]))
    assert out['energy_score']==.5


def test_all_invalid_targets_raise():
    with pytest.raises(ValueError):scores(torch.zeros(6,2,2),torch.ones(6)/6,torch.zeros(2,2),torch.zeros(2,dtype=torch.bool))


def test_scene_partitions_disjoint_and_manifest_rejects_duplicates(tmp_path):
    create_synthetic(tmp_path/'data')
    groups=make_manifest(tmp_path/'data',tmp_path/'manifest.json')
    ids=[r['scene_id'] for group in groups.values() for r in group]
    assert len(ids)==len(set(ids))
    import shutil
    file=next((tmp_path/'data/train').glob('*.pt'))
    shutil.copy(file,tmp_path/'data/val'/file.name)
    with pytest.raises(ValueError,match='Duplicate'):make_manifest(tmp_path/'data',tmp_path/'manifest2.json')


def test_checkpoint_load_is_strict(adapter,tmp_path):
    state=adapter.model.state_dict()
    state.pop(next(iter(state)))
    file=tmp_path/'bad.pt';torch.save(state,file)
    with pytest.raises(RuntimeError):RealMotionAdapter('cpu',file)


def test_no_random_weights_in_real_runs():
    with pytest.raises(ValueError,match='require'):RealMotionAdapter('cpu')


@pytest.mark.parametrize('prefix',['net.','model.','module.model.'])
def test_checkpoint_wrapper_prefixes(adapter,tmp_path,prefix):
    weights={prefix+k:v for k,v in adapter.model.state_dict().items()}
    file=tmp_path/'wrapped.pt';torch.save({'state_dict':weights},file)
    loaded=RealMotionAdapter('cpu',file)
    for k,v in loaded.model.state_dict().items():
        torch.testing.assert_close(v,adapter.model.state_dict()[k],atol=0,rtol=0)
