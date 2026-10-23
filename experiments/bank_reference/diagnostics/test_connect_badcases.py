import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('connect_badcases',Path(__file__).with_name('connect_badcases.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def test_two_archived_mixtures_reconstruct_signal_and_realized_loss():
    c=np.array([1.,-2.,3.]);e=np.array([-.5,1.,4.])
    r25=np.sqrt(.75)*c+.5*e;r50=np.sqrt(.5)*(c+e)
    u,n=d.components(r25,r50)
    np.testing.assert_allclose(u,np.sqrt(.5)*c,atol=1e-14)
    np.testing.assert_allclose(n,np.sqrt(.5)*e,atol=1e-14)
    parts=d.loss_parts(u*.2,n*.3,u,n,np.ones(3))
    assert abs(parts['total']-parts['signal']-parts['noise']-parts['cross'])<1e-14


def test_reader_never_uses_other_keys_or_unobserved_coordinates():
    observed=np.array([[1,0],[1,1],[1,1]])
    weights=d.reader_weights(observed,np.array([1,1,2]),1)
    np.testing.assert_array_equal(weights,[[.5,0],[.5,1],[0,0]])
    np.testing.assert_array_equal(d.reader_weights(observed,np.array([1,1,2]),9),np.zeros((3,2)))


def test_frozen_equal_capacity_deletion_and_past_hashes_exclude_future():
    r=np.zeros((1,15,34),np.float32);r[0,0]=10
    o=np.ones_like(r);w=np.ones((1,15),np.float32);keys=np.ones((1,15),np.int64)
    u=np.zeros_like(r,dtype=np.float64);u[0,0]=6;n=r-u
    arrays={'EXACT__R':r,'EXACT__O':o,'EXACT__W':w,'EXACT__key':keys}
    parent={'condition':'EXACT','source_index':0,'method':'K4/FIFO','original_t':8,'decision_time':8,
            'candidate_times':list(range(5)),'recent_times':list(range(5,9)),
            'candidate_keys':[1]*5,'future_original_times':list(range(9,15)),'future_keys':[1]*6,
            'future_den':204.,'candidate_losses':[0.,1.5625,1.5625,1.5625,1.5625],
            'chosen_drop':4,'oracle_drop':0,'regret':1.5625}
    result=d.analyze_parent(parent,arrays,{'EXACT':(u,n)})
    assert result['chosen_minus_oracle']=={'signal':.5625,'noise':.25,'cross':.75,'total':1.5625}
    assert result['task_optimal_drop_set']==[0]
    hashes=[result[k] for k in ('bounded_past_sha256','full_available_past_sha256')]
    r[0,9:]=2;u[0,9:]=2
    parent['candidate_losses']=[4.,.5625,.5625,.5625,.5625];parent['oracle_drop']=1;parent['regret']=0
    result2=d.analyze_parent(parent,arrays,{'EXACT':(u,n)})
    assert [result2[k] for k in ('bounded_past_sha256','full_available_past_sha256')]==hashes
    assert not set(result['task_optimal_drop_set']) & set(result2['task_optimal_drop_set'])
    assert d.collision_audit([result,result2])['bounded_past_sha256']['conflicting_optimal_set_groups']==1
