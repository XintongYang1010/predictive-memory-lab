import json
import pytest
import torch
from pmr.adapter import FEATURES
from pmr.controller import Controller, train_controller
from pmr.diagnostics import diagnose_rows, tiny_overfit, regression_metrics, parameter_updates
from test_controller import row, feature


def test_zero_head_has_exact_expected_regression_initial_loss_and_can_learn():
    torch.manual_seed(12)
    x=torch.randn(10,len(FEATURES))
    g=torch.cat((torch.zeros(10,1),torch.randn(10,3)*.2),1)
    model=Controller()
    assert torch.equal(model(x),torch.zeros_like(g))
    before=torch.random.get_rng_state().clone()
    diagnostic=tiny_overfit(x,g,steps=300,seed=3)
    assert torch.equal(before,torch.random.get_rng_state())
    assert diagnostic['initial_MSE']==pytest.approx(float(g[:,1:].square().mean()))
    assert diagnostic['status']=='passed'
    assert diagnostic['best_over_initial_MSE']<.05


def test_tiny_fit_exposes_indistinguishable_features_and_no_signal():
    x=torch.zeros(2,len(FEATURES))
    g=torch.tensor([[0.,1.,-1.,2.],[0.,-1.,1.,-2.]])
    r=tiny_overfit(x,g,steps=2)
    assert r['status']=='conflicting_targets'
    assert r['duplicate_feature_MSE_floor']==pytest.approx(r['initial_MSE'])
    r=tiny_overfit(x,torch.zeros_like(g),steps=2)
    assert r['status']=='no_signal' and r['best_over_initial_MSE'] is None


def test_offline_diagnostic_masks_ineligible_actions_and_reports_harm():
    x=torch.stack((feature(4.,rebuild=False),feature(evidence=False),feature()))
    gains=torch.tensor([[0.,.2,99.,-.1],[0.,2.,99.,3.],[0.,-.2,-.1,-.3]])
    pred=torch.tensor([[0.,.3,100.,-.1],[0.,2.,99.,3.],[0.,.2,-.1,-.3]])
    r=regression_metrics(pred,gains,x)
    assert r['action_counts']=={'reweight':2,'keep':1}
    assert r['offline_mean_gain']==pytest.approx(0.)
    assert r['offline_oracle_regret']==pytest.approx(.2/3)
    assert r['harm_fraction_among_interventions']==pytest.approx(.5)


def test_baselines_fit_only_train_and_require_disjoint_scenes():
    train=[row('train')];dev=[row('dev')]
    r=diagnose_rows(train,dev,tiny_steps=2)
    for name in ('train_mean_gain','stage_mean_gain','residual_ridge','all_feature_ridge'):
        assert r['dev_baselines'][name]['MSE']<1e-10
    dev[0]['branches']['reset']['final']['energy_score']=3.
    changed=diagnose_rows(train,dev,tiny_steps=2)
    assert changed['dev_baselines']['train_mean_gain']['MSE']>1.
    assert changed['dev_baselines']['train_mean_gain']['action_counts']=={'reset':1}
    with pytest.raises(ValueError,match='leakage'):
        diagnose_rows(train,train,tiny_steps=2)


def test_actual_adamw_parameter_update_uses_before_weights_and_zero_is_undefined():
    model=Controller()
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    loss=(model(feature())[1:]-1).square().mean()
    loss.backward();opt.step()
    stats=parameter_updates(model,before)
    assert stats['net.4.weight']['update_weight_ratio'] is None
    assert stats['net.4.weight']['update_norm']>0
    assert stats['net.0.weight']['update_weight_ratio']==pytest.approx(.001,rel=1e-3)


def test_training_progress_survives_rollout_failure(tmp_path):
    def fail(_):raise RuntimeError('rollout unavailable')
    target=tmp_path/'c.pt'
    with pytest.raises(RuntimeError,match='rollout unavailable'):
        train_controller([row('train')],[row('dev')],target,steps=2,dev_evaluator=fail)
    history=json.loads(target.with_suffix('.training.json').read_text())
    assert history['status']=='selecting' and len(history['history'])==2
    assert history['history'][0]['parameter_updates']['net.4.weight']['update_weight_ratio'] is None
    assert target.with_suffix('.diagnostics.json').exists()


def test_calibration_uses_final_horizon_even_if_common_window_disagrees(monkeypatch,tmp_path):
    import pmr.experiment as exp
    class Adapter:
        forward_calls=0
    monkeypatch.setattr(exp,'load_sequence',lambda *a: [0,1,2])
    monkeypatch.setattr(exp,'evidence',lambda *a: {'features':[0.]*len(FEATURES)})
    def fake_rollout(adapter,seq,kind='keep',sigma=2.,tau=1.,**kwargs):
        return [{'sigma':sigma,'tau':tau,'kind':kind}]*3,[None]*3,[]
    monkeypatch.setattr(exp,'rollout',fake_rollout)
    def fake_scores(out,frame,common=False):
        value=out['sigma'] if out['kind']=='reweight' else out['tau']
        return {'energy_score':-value if common else value}
    monkeypatch.setattr(exp,'frame_scores',fake_scores)
    result=exp.calibrate(Adapter(),[{'scene_id':'calib'}],tmp_path/'cal.json','cpu')
    assert result['sigma']==.5 and result['tau']==.25
    assert result['objective']=='final_60_frame_energy_score'
