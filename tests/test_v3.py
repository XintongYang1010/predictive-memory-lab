"""Causal feature/state checks and independent candidate-replay accounting."""
import json
import pytest
import torch
from pmr.adapter import FEATURES,BASE_FEATURES,TEMPORAL_FEATURES,evidence,influence_features,rotation
from pmr.common import clone
from pmr.controller import Controller,select_controller,load_controller
from pmr.experiment import rollout
from pmr.report import audit_selections
from test_controller import feature,fixed_model,candidate


def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree_equal(x,y)
    else:assert a==b


def test_temporal_evidence_separates_growing_and_shrinking_equal_rmse(adapter,sequence):
    _,state=adapter.step(sequence[0])
    frame=sequence[1]
    actual=frame['x_positions'][0,0,-10:]@rotation(frame['theta'][0]).T+frame['origin'][0]
    features=[]
    for offsets in (torch.arange(1.,11.),torch.arange(10.,0.,-1)):
        trial=clone(state)
        trial['memory']['glo_y_hat'][0,:,:10]=actual-trial['memory']['origin'][0]
        trial['memory']['glo_y_hat'][0,:,:10,0]+=offsets
        features.append(evidence(frame,trial)['features'])
    torch.testing.assert_close(features[0][:5],features[1][:5],atol=1e-5,rtol=1e-5)
    assert features[0][FEATURES.index('rmse_growth')]>0
    assert features[1][FEATURES.index('rmse_growth')]<0


def test_temporal_history_is_scene_local_and_follows_realized_prefix(adapter,sequence):
    _,states,_=rollout(adapter,sequence)
    first=evidence(sequence[1],states[0])['features']
    second=evidence(sequence[2],states[1])['features']
    assert first[FEATURES.index('has_previous_evidence')]==0
    assert second[FEATURES.index('has_previous_evidence')]==1
    torch.testing.assert_close(second[FEATURES.index('previous_rmse')],first[FEATURES.index('weighted_rmse')])
    assert states[0]['previous_evidence'] is None
    snapshot=clone(states[0])
    adapter.step(sequence[1],states[0],'reset')
    assert_tree_equal(states[0],snapshot)


def test_probe_never_reads_targets_and_never_commits_state(adapter,sequence):
    _,state=adapter.step(sequence[0]);before=clone(state)
    calls,probes=adapter.forward_calls,adapter.probe_forward_calls
    first=adapter.decision_evidence(sequence[1],state)
    assert adapter.forward_calls-calls==adapter.probe_forward_calls-probes==2
    assert_tree_equal(state,before)
    altered=clone(sequence[1]);altered['target']+=1e7;altered['target_mask'].fill_(False)
    second=adapter.decision_evidence(altered,state)
    torch.testing.assert_close(first['features'],second['features'],atol=0,rtol=0)
    no_targets={k:v for k,v in sequence[1].items() if k not in ('target','target_mask')}
    torch.testing.assert_close(first['features'],adapter.decision_evidence(no_targets,state)['features'],atol=0,rtol=0)
    assert first['features'][FEATURES.index('has_influence_probe')]==1


def test_probe_distance_permutation_and_rigid_transform_invariant():
    torch.manual_seed(41)
    a={'y_hat':torch.randn(1,6,60,2),'pi':torch.randn(1,6)}
    b={'y_hat':torch.randn(1,6,60,2),'pi':torch.randn(1,6)}
    original=influence_features(a,b,torch.tensor(.2))
    rot=rotation(torch.tensor(.5));shift=torch.tensor([2.,-3.])
    aa,bb=clone(a),clone(b)
    for obj,order in [(aa,torch.tensor([1,0,3,2,5,4])),(bb,torch.tensor([5,1,0,2,4,3]))]:
        obj['y_hat']=obj['y_hat'][:,order]@rot.T+shift
        obj['pi']=obj['pi'][:,order]
    torch.testing.assert_close(influence_features(aa,bb,torch.tensor(.2)),original,atol=2e-6,rtol=1e-5)
    assert float(influence_features(a,a,torch.tensor(.2))[2])==pytest.approx(0.,abs=1e-6)


@pytest.mark.parametrize('feature_set,active',[('base',len(BASE_FEATURES)),('temporal',len(BASE_FEATURES)+len(TEMPORAL_FEATURES))])
def test_ablations_cannot_see_masked_features(feature_set,active):
    model=Controller(feature_set)
    with torch.no_grad():model.net[-1].weight.fill_(.1)
    x=feature();y=x.clone();y[active:]=100
    torch.testing.assert_close(model(x),model(y),atol=0,rtol=0)
    assert int(model.feature_mask.sum())==active


def test_native_and_probe_keep_have_identical_predictions_and_explicit_cost(adapter,sequence):
    calls=adapter.forward_calls
    plain,_,_=rollout(adapter,sequence,'keep')
    assert adapter.forward_calls-calls==3
    calls,probes=adapter.forward_calls,adapter.probe_forward_calls
    matched,_,_=rollout(adapter,sequence,'keep_probe')
    assert adapter.forward_calls-calls==7 and adapter.probe_forward_calls-probes==4
    for a,b in zip(plain,matched):torch.testing.assert_close(a['y_hat'],b['y_hat'],atol=0,rtol=0)
    fallback=Controller('base');fallback.keep_only.fill_(True)
    calls=adapter.forward_calls
    outs,_,_=rollout(adapter,sequence,fallback)
    assert adapter.forward_calls-calls==7
    for a,b in zip(plain,outs):torch.testing.assert_close(a['y_hat'],b['y_hat'],atol=0,rtol=0)
    resetting=fixed_model([-.1,-.1,.2])
    direct,_,edits=rollout(adapter,sequence,'reset')
    controlled,_,chosen=rollout(adapter,sequence,resetting)
    assert edits==chosen==['keep','reset','reset']
    for a,b in zip(direct,controlled):torch.testing.assert_close(a['y_hat'],b['y_hat'],atol=0,rtol=0)


def test_selection_can_be_recomputed_and_tampering_is_detected(tmp_path):
    model=fixed_model([-.1,-.1,.2])
    def evaluate(policy):
        return [{'scene_id':str(i),'group':'ordinary','energy_score':1. if isinstance(policy,str) else .9,
                 'edits':['keep','reset','reset']} for i in range(20)]
    path=tmp_path/'round0.controller.selection.json'
    result=select_controller([candidate(model)],model,evaluate,{str(i) for i in range(20)},path)
    assert audit_selections(tmp_path)['passed']
    assert load_controller(tmp_path/result['candidates'][0]['checkpoint_file']).choose(feature())=='reset'
    record=tmp_path/result['candidates'][0]['rollout_file']
    data=[json.loads(l) for l in record.read_text().splitlines()]
    data[0]['energy_score']+=.5
    record.write_text(''.join(json.dumps(r)+'\n' for r in data))
    with pytest.raises(ValueError,match='mismatch'):audit_selections(tmp_path)
