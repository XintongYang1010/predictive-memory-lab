"""CPU diagnostics of data observability and published predictions; no feature-gate edits."""
import csv,json
from pathlib import Path
import sys
repo=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(repo/"results/wm/visual-v5-dgx-20261009/source"))
import numpy as np
import torch
torch.set_num_threads(1)
from pmr.visual_memory.data import episode,specs

root=Path(__file__).resolve().parent
ys={};oracles={};split_stats={}
for split in ('train','dev'):
    targets=[];pixel_predictions=[];differences=[];times_ok=True;recent_ok=True;raw_sign_reflected=0
    for _,pair,variant in specs(split,128):
        ep=episode(split,pair,variant)
        targets.append(ep.probe_state)
        times_ok &= ep.times[6:8].tolist()==[6.,7.]
        centers=[]
        for frame in ep.rgb[6:8]:
            centers_now=[]
            for red,blue in ((235,60),(60,235)):
                mask=(frame[:,:,0]==red)&(frame[:,:,2]==blue)
                yy,xx=np.nonzero(mask)
                centers_now.extend([xx.mean()/224,yy.mean()/224])
            centers.append(np.array(centers_now))
        pixel_predictions.append(np.r_[centers[1],centers[1]-centers[0]])
        differences.append(np.any(ep.rgb[6]!=ep.rgb[7]))
        if variant==0:
            paired=episode(split,pair,1)
            recent_ok &= np.array_equal(ep.rgb[-4:],paired.rgb[-4:])
    ys[split]=np.stack(targets).astype('float64');oracles[split]=np.stack(pixel_predictions)
    split_stats[split]=dict(episodes=128,pairs=64,timestamps_6_7_correct=bool(times_ok),paired_recent_equal=bool(recent_ok),visible_rgb_frame_changes=int(sum(differences)))

def score(pred):
    baseline=((ys['dev']-ys['train'].mean(0))**2).mean(0)
    r2=1-((pred-ys['dev'])**2).mean(0)/baseline
    return dict(per_dimension=r2.tolist(),position_r2=float(r2[:4].mean()),displacement_r2=float(r2[4:].mean()))
results={'schema':'offline-static-data-checks-1','test_opened':False,'split_checks':split_stats,'rgb_color_centroid_oracle':score(oracles['dev'])}
# Non-temporal control: these controls contain only t7 position, no temporal input.
# Predetermined polynomial degrees show the possible effect of bounded-generator priors.
controls={}
for degree in (1,3):
    xa=np.concatenate([ys['train'][:,:4]**k for k in range(1,degree+1)],1)
    xb=np.concatenate([ys['dev'][:,:4]**k for k in range(1,degree+1)],1)
    mean=xa.mean(0);xa=xa-mean;xb=xb-mean
    ta,tb=torch.from_numpy(xa),torch.from_numpy(xb)
    scale=ta.square().sum(1).mean();gram=ta@ta.T/scale
    ty=torch.from_numpy(ys['train']);ym=ty.mean(0)
    coef=torch.linalg.solve(gram+.01*torch.eye(len(gram),dtype=torch.float64),ty-ym)
    control_pred=((tb@ta.T/scale)@coef+ym).numpy()
    assert np.isfinite(control_pred).all()
    controls[f'position_only_polynomial_degree_{degree}']=score(control_pred)
results['no_temporal_input_controls']=controls

csvfile=repo/'results/wm/visual-v5-dgx-20261009/run/feature_diagnostics/dev_predictions.csv'
rows=list(csv.DictReader(csvfile.open()))
pred={mode:np.empty((128,8)) for mode in ('raw','pooled')}
for row in rows:
    _,pair,variant=row['episode_id'].split('-');index=int(pair)*2+int(variant)
    dimension=int(row['dimension_index']);pred[row['mode']][index,dimension]=float(row['prediction'])
    assert np.isclose(float(row['target']),ys['dev'][index,dimension])
results['recomputed_published_predictions']={mode:score(p) for mode,p in pred.items()}
results['r2_using_dev_mean']={mode:(1-((p-ys['dev'])**2).mean(0)/((ys['dev']-ys['dev'].mean(0))**2).mean(0)).tolist() for mode,p in pred.items()}
# Pair-bootstrap uncertainty conditional on the already-fitted models; no refitting.
rng=np.random.default_rng(171)
means=ys['train'].mean(0)
boot={mode:[] for mode in pred};diff=[]
for _ in range(5000):
    pairs=rng.integers(0,64,64);indices=(pairs[:,None]*2+np.array([0,1])).ravel()
    denom=((ys['dev'][indices]-means)**2).mean(0)
    values={mode:float((1-((p[indices]-ys['dev'][indices])**2).mean(0)/denom)[4:].mean()) for mode,p in pred.items()}
    for mode,value in values.items():boot[mode].append(value)
    diff.append(values['raw']-values['pooled'])
results['fixed_fit_pair_bootstrap_95ci']={mode:np.quantile(values,[.025,.975]).tolist() for mode,values in boot.items()}
results['fixed_fit_pair_bootstrap_95ci']['raw_minus_pooled']=np.quantile(diff,[.025,.975]).tolist()
results['limitations']=['Exploratory controls; not a new gate or threshold change.','Oracle uses known target colors to locate visible objects; not an actor input.','Bootstrap conditions on fixed fitted models and excludes training uncertainty.','Degree1 and degree3 controls reported together; no Dev-based choice or retraining of the official gate.']
(root/'static_probe_checks.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
