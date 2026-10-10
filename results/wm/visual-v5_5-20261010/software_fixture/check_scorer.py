#!/usr/bin/env python3
"""Independent arithmetic fixtures only; never load scientific caches or images."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
import numpy as np
import score_v5_5 as s


def main():
    begin=time.time();checks={}
    vector=np.arange(1,9,dtype=np.float64)/8
    for label,terms in [('additive',{4:vector,2:vector*-2,1:vector*3}),
                        ('pure_pair',{6:vector*4}),('pure_triple',{7:vector*-3})]:
        p=np.stack([vector+sum((v for mask,v in terms.items() if m&mask==mask),np.zeros(8)) for m in range(8)])
        phi,mob=s.decompose(p)
        em=np.zeros_like(p);em[0]=vector
        ep=np.zeros_like(phi)
        for mask,value in terms.items():
            em[mask]=value
            for j,bit in enumerate((4,2,1)):
                if mask&bit:ep[j]+=value/bin(mask).count('1')
        assert np.allclose(phi,ep,rtol=0,atol=1e-15) and np.array_equal(mob,em)
        checks[label]=dict(shapley_max_abs=float(np.max(np.abs(phi-ep))),mobius_max_abs=float(np.max(np.abs(mob-em))))
    original=np.tile(np.array([.5,.25,.6,.75,.012,-.014,-.009,.017],dtype=np.float32),(3,2,1))
    positions=np.empty((3,2,8,4),dtype=np.float32)
    for mask in range(8):
        positions[...,mask,:2]=original[...,:2] if mask&4 else original[...,:2]-original[...,4:6]
        positions[...,mask,2:]=original[...,2:4] if mask&2 else original[...,2:4]-original[...,6:8]
    targets=s.factorial_targets(original,positions)
    for mask in range(8):
        assert np.array_equal(targets[...,mask,4:6],original[...,4:6]*bool(mask&4))
        assert np.array_equal(targets[...,mask,6:8],original[...,6:8]*bool(mask&2))
        assert np.array_equal(targets[...,mask,:],targets[...,mask^1,:])
    checks['factorial_targets']=dict(masks_checked=8,original_111_exact=bool(np.array_equal(targets[...,7,:],original)),yellow_never_changes_labels=True)
    h,w,c=np.indices((16,16,1280));raster=(1024*h+16*w+c/2048).astype(np.float64)
    pooled=s.block_pool(raster.reshape(256,1280)).reshape(4,4,1280)
    reference=np.stack([np.stack([raster[i*4:(i+1)*4,j*4:(j+1)*4].mean((0,1)) for j in range(4)]) for i in range(4)])
    assert np.array_equal(pooled,reference)
    projected=pooled.repeat(4,0).repeat(4,1)
    assert all(np.array_equal(projected[i,j],reference[i//4,j//4]) for i in range(16) for j in range(16))
    random=np.random.default_rng(1234).normal(size=(2,256,1280))
    projection=s.block_pool(random).reshape(2,4,4,1280).repeat(4,1).repeat(4,2).reshape(2,256,1280)
    total=np.sum(random**2);retained=np.sum(projection**2);lost=np.sum((random-projection)**2)
    residual=abs(total-retained-lost)/total
    assert residual<1e-14
    checks['pool_axes']=dict(raster_max_abs=float(np.abs(pooled-reference).max()),repeat_projection_exact=True,relative_energy_identity_error=float(residual))
    # Variants have different values, so averaging them before the world
    # bootstrap is distinguishable from treating1024 or16 rows as independent.
    world=np.arange(5)[:,None,None];variant=np.array([1.,3.])[None,:,None];mask=np.arange(8)[None,None,:]+1
    y=np.zeros((5,2,8,8));global_p=np.zeros_like(y);lane_p=np.zeros_like(y)
    global_p[...,4]=np.sqrt((world+1)*variant*mask);global_p[...,6]=2*global_p[...,4]
    lane_p[...,4]=np.sqrt(variant*mask/2);lane_p[...,6]=-lane_p[...,4]
    actual=s.bootstrap_comparison(global_p,lane_p,y)
    per_world=((((global_p-y)**2-(lane_p-y)**2)[...,[4,6]]).mean(-1)).mean(1)
    rng=np.random.default_rng(7812)
    indices=rng.integers(0,5,size=(10000,5))
    expected=np.quantile(per_world[indices].mean(1),[.025,.975],axis=0)
    maxdiff=0.
    for m in range(8):
        row=actual[format(m,'03b')]
        maxdiff=max(maxdiff,float(np.max(np.abs(np.asarray(row['paired_95ci'])-expected[:,m]))))
        assert abs(row['global_minus_lane_dx_mse']-per_world[:,m].mean())<1e-13
        assert np.allclose(row['paired_95ci'],expected[:,m],rtol=0,atol=1e-13)
    checks['bootstrap']=dict(all_masks_checked=8,replicates=10000,worlds=5,variants_per_world=2,paired_interval_max_abs=maxdiff)
    result=dict(status='software_scorer_checks_passed',software_only=True,scientific_data_opened=False,
        source_sha256=hashlib.sha256((ROOT/'scripts/score_v5_5.py').read_bytes()).hexdigest(),checks=checks,elapsed_seconds=time.time()-begin)
    with Path(__file__).with_name('scorer-result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
