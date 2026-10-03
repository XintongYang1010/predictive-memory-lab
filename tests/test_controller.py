import copy
import json
import pytest
import torch
from pmr.adapter import ACTIONS, FEATURES, FEATURE_SCHEMA
from pmr.controller import Controller, training_tensors, select_controller, train_controller, load_controller


def feature(time=5., evidence=True, rebuild=True):
    x=torch.zeros(len(FEATURES))
    x[FEATURES.index('update_time_seconds')]=time
    x[FEATURES.index('has_evidence')]=evidence
    x[FEATURES.index('rebuild_available')]=rebuild
    return x


def fixed_model(gains):
    model=Controller()
    with torch.no_grad():model.net[-1].bias.copy_(torch.tensor(gains))
    return model


def row(sid):
    return {'scene_id':sid,'event_step':50,'feature_schema':FEATURE_SCHEMA,'feature_names':list(FEATURES),
            'features':feature().tolist(),'label':0,'weight':999,
            'branches':{a:{'loss':999.,'final':{'energy_score':loss}}
                        for a,loss in zip(ACTIONS,[2.,1.8,2.5,1.])}}


def candidate(model,step=1,margin=.01):
    return {'state_dict':copy.deepcopy(model.state_dict()),'step':step,'margin':margin,'offline_dev_gain':100.}


def test_gain_targets_use_all_final_outcomes_not_class_or_common_window():
    r=row('train');_,g=training_tensors([r],'cpu')
    torch.testing.assert_close(g,torch.tensor([[0.,.2,-.5,1.]]))
    r['label']=3;r['weight']=0;r['branches']['keep']['loss']=-999
    torch.testing.assert_close(training_tensors([r],'cpu')[1],g)


def test_stage_mask_gain_margin_and_missing_evidence():
    m=fixed_model([.02,2.,.1])
    assert m.choose(feature(4.,rebuild=False))=='reset'
    assert m.choose(feature())=='rebuild'
    assert m.choose(feature(evidence=False))=='keep'
    m.gain_margin.fill_(3.)
    assert m.choose(feature())=='keep'
    assert m(feature())[0]==0


@pytest.mark.parametrize('gains,expected',[([.3,-.1,.1],'reweight'),([-.1,.3,.1],'rebuild'),([-.1,.1,.3],'reset'),([-.1,-.2,-.3],'keep')])
def test_each_action_can_be_selected(gains,expected):
    assert fixed_model(gains).choose(feature())==expected


def test_v1_rows_and_weights_fail_clearly(tmp_path):
    old=row('x');old.pop('feature_schema')
    with pytest.raises(ValueError,match='V3 needs fresh'):training_tensors([old],'cpu')
    torch.save({'features':list(FEATURES)[:12]},tmp_path/'v1.pt')
    with pytest.raises(ValueError,match='V1/V2 weights'):load_controller(tmp_path/'v1.pt')


def test_positive_offline_gain_cannot_override_harmful_closed_loop(tmp_path):
    m=fixed_model([-.1,-.1,.2])
    def evaluate(policy):
        return [{'scene_id':str(i),'group':'ordinary','energy_score':1. if isinstance(policy,str) else 1.2,
                 'edits':['keep']*3} for i in range(20)]
    selection=select_controller([candidate(m)],m,evaluate,{str(i) for i in range(20)},tmp_path/'s.json')
    assert selection['mode']=='keep_fallback'
    assert not selection['candidates'][0]['eligible']
    assert m.choose(feature())=='keep'


def test_ordinary_harm_vetoes_positive_aggregate_gain(tmp_path):
    m=fixed_model([-.1,-.1,.2])
    def evaluate(policy):
        return [{'scene_id':str(i),'group':'ordinary' if i==0 else 'high_conflict',
                 'energy_score':1. if isinstance(policy,str) else (1.1 if i==0 else .5),
                 'edits':['keep']*3} for i in range(20)]
    selection=select_controller([candidate(m)],m,evaluate,{str(i) for i in range(20)},tmp_path/'s.json')
    assert selection['candidates'][0]['paired_ES_improvement']['ci95'][0]>0
    assert selection['mode']=='keep_fallback'


def test_best_rollout_restored_instead_of_last_candidate(tmp_path):
    best=fixed_model([.4,-.1,-.1]);last=fixed_model([-.1,-.1,.2])
    def evaluate(policy):
        error=1. if isinstance(policy,str) else (.7 if policy.choose(feature())=='reweight' else .9)
        return [{'scene_id':str(i),'group':'ordinary','energy_score':error,'edits':['keep']*3} for i in range(20)]
    selection=select_controller([candidate(best,50,.1),candidate(last,100,.05)],last,evaluate,{str(i) for i in range(20)},tmp_path/'s.json')
    assert selection['selected']['step']==50 and last.choose(feature())=='reweight'
    assert float(last.gain_margin)==pytest.approx(.1)


def test_development_rejects_foreign_or_duplicate_scenes(tmp_path):
    m=Controller()
    def evaluate(policy):return [{'scene_id':'eval-scene','group':'ordinary','energy_score':1.,'edits':['keep']*3}]
    with pytest.raises(ValueError,match='foreign scenes'):
        select_controller([],m,evaluate,{'dev-scene'},tmp_path/'s.json')


def test_train_overlap_rejected_before_selection(tmp_path):
    with pytest.raises(ValueError,match='Scene leakage'):
        train_controller([row('same')],[row('same')],tmp_path/'c.pt',dev_evaluator=lambda p:[])


def test_regression_checkpoint_roundtrip_and_safe_fallback(tmp_path):
    seen=[]
    def evaluate(policy):
        seen.append(policy)
        return [{'scene_id':'dev','group':'ordinary','energy_score':1.,'edits':['keep']*3}]
    target=tmp_path/'c.pt'
    model=train_controller([row('train')],[row('dev')],target,steps=2,dev_evaluator=evaluate)
    loaded=load_controller(target)
    assert seen[0]=='keep' and model.choose(feature())==loaded.choose(feature())=='keep'
    selection=json.loads(target.with_suffix('.selection.json').read_text())
    assert selection['mode']=='keep_fallback'
    torch.testing.assert_close(model(feature()),loaded(feature()))
