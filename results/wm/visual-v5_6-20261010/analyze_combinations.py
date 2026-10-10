#!/usr/bin/env python3
"""Read-only posthoc analysis of explicit completed V5.6 result directories.

Does not search for results, generate data, fit models or call a GPU. Actual
results must be supplied explicitly after the parent authorizes their reading.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEED, REPLICATES = 7816, 10000
MOTION = ('A_dx', 'A_dy', 'B_dx', 'B_dy')
AXIS_NAMES = ('none', 'Ax', 'Ay', 'Axy', 'Bx', 'By', 'Bxy')
AXIS_GROUPS = ((0, 1, 2, 3), (0, 4, 5, 6))


def identity(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def read_json(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def readback(folder):
    path=folder.parent/(folder.name+'_readback.json')
    if not path.exists():return None
    receipt=read_json(path)
    assert receipt['status']=='verified'
    assert set(receipt['files'])=={p.name for p in folder.iterdir() if p.is_file()}
    for name,expected in receipt['files'].items():assert identity(folder/name)==expected
    return dict(receipt=identity(path),step_id=receipt['step_id'],all_downloaded_bytes_verified=True)


def draws(worlds):
    return np.random.default_rng(SEED).integers(0, worlds, size=(REPLICATES, worlds))


def paired_loss(first, second, truth, indices=None):
    assert first.shape == second.shape == truth.shape and truth.shape[-1] == 8
    n = len(truth)
    delta = ((first - truth)**2 - (second - truth)**2)[..., [4, 6]].reshape(n, -1).mean(1)
    index = draws(n) if indices is None else indices
    boot = delta[index].mean(1)
    return dict(first_minus_second_dx_mse=float(delta.mean()),
                paired_world_95ci=np.quantile(boot, [.025, .975]).tolist(), worlds=n,
                positive_means_second_lower_error=True)


def response_rms(effect, indices=None):
    values = np.asarray(effect, np.float64)
    n = len(values); per_world = (values**2).reshape(n, -1).mean(1)
    index = draws(n) if indices is None else indices
    return dict(rms_pixels=float(np.sqrt(per_world.mean()) * 224),
                paired_world_95ci_pixels=(np.quantile(np.sqrt(per_world[index].mean(1)), [.025, .975]) * 224).tolist())


def paired_rms(first, second):
    assert first.shape==second.shape
    n=len(first);a=(first**2).reshape(n,-1).mean(1);b=(second**2).reshape(n,-1).mean(1)
    index=draws(n);sample=(np.sqrt(a[index].mean(1))-np.sqrt(b[index].mean(1)))*224
    return dict(first_minus_second_rms_px=float((np.sqrt(a.mean())-np.sqrt(b.mean()))*224),
                paired_world_95ci_pixels=np.quantile(sample,[.025,.975]).tolist(),worlds=n)


def diagnose(prediction, truth, score_mean):
    prediction = np.asarray(prediction, np.float64); truth = np.asarray(truth, np.float64)
    assert prediction.shape == truth.shape and prediction.shape[-1] == 8
    assert np.isfinite(prediction).all() and np.isfinite(truth).all()
    p, y = prediction.reshape(-1, 8), truth.reshape(-1, 8)
    mse = ((p-y)**2).mean(0); denominator = ((y-score_mean)**2).mean(0)
    skills = [float(1-m/d) if d > 0 else None for m,d in zip(mse, denominator)]
    axis = {}
    for j, name in enumerate(MOTION, start=4):
        target, estimate = y[:, j], p[:, j]
        nonzero, static, large = target != 0, target == 0, np.abs(target*224) >= 1
        product = target*estimate
        def direction(mask):
            count = int(mask.sum()); wrong = int(((product < 0)&mask).sum()); zero = int(((product == 0)&mask).sum())
            return dict(denominator=count, wrong=wrong, exact_zero_prediction=zero,
                        correct=int(((product > 0)&mask).sum()),
                        wrong_fraction=wrong/count if count else None,
                        wrong_or_zero_fraction=(wrong+zero)/count if count else None)
        axis[name] = dict(nonzero_target_direction=direction(nonzero),
                         at_least_one_pixel_direction=direction(large),
                         static_target_count=int(static.sum()),
                         static_false_response_rms_px=float(np.sqrt(np.mean(estimate[static]**2))*224) if static.any() else None,
                         static_nonzero_prediction_count=int(np.count_nonzero(estimate[static])),
                         static_prediction_max_abs_px=float(np.max(np.abs(estimate[static]))*224) if static.any() else None,
                         amplitude_ratio=float(np.abs(estimate).mean()/np.abs(target).mean()) if nonzero.any() else None,
                         mse_normalized=float(mse[j]), rmse_pixels=float(np.sqrt(mse[j])*224),
                         prediction_rms_pixels=float(np.sqrt(np.mean(estimate**2))*224))
    average = lambda entries: float(np.mean(entries)) if all(x is not None for x in entries) else None
    return dict(examples=len(p), per_dimension_mse=mse.tolist(), per_dimension_skill=skills,
                mean_motion_skill=average(skills[4:]), position_skill=average(skills[:4]),
                dx_mse=float(mse[[4, 6]].mean()), dx_rmse_pixels=float(np.sqrt(mse[[4, 6]].mean())*224),
                baseline_denominator=denominator.tolist(), axis=axis)


def load_combination(folder, role):
    folder=Path(folder); report=read_json(folder/'report.json'); manifest=read_json(folder/'manifest.json')
    assert report['schema']=='v5.6-combination-report-1' and report['status']=='complete' and report['role']==role
    assert not report['test_opened'] and not report['p0_authorized']
    assert set(manifest)=={p.name for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'}
    for name,expected in manifest.items():
        assert Path(name).name==name and identity(folder/name)==expected, name
    data=load_npz(folder/'predictions.npz')
    shape={'known':(256,2,8,8),'axis':(256,2,7,8),'confirm':(64,32,8)}[role]
    for name in ('targets','J','X'):
        assert data[name].shape==shape and np.isfinite(data[name]).all()
    assert np.array_equal(data['group_ids'], np.arange(64) if role=='confirm' else np.arange(640,896))
    expected={'targets','group_ids','J','X'} | ({'J_negative','X_negative','negative_targets'} if role=='confirm' else set())
    assert set(data)==expected
    if role=='confirm':
        assert report['fresh_confirmation'] and not report['known_dev_posthoc']
        for name in ('J_negative','X_negative','negative_targets'):
            assert data[name].shape==(64,2,8) and np.isfinite(data[name]).all()
        assert np.count_nonzero(data['negative_targets'][...,4:])==0
    return report,data,dict(folder=str(folder),files={name:identity(folder/name) for name in ('report.json','predictions.npz','manifest.json')},all_manifest_members_verified=True,remote_readback=readback(folder))


def subset_arrays(data, role):
    if role=='confirm':
        names={'all_views':np.arange(32),'old_joint_views':np.arange(16),'mixed_object_views':np.arange(16,32),
               **{f'path_{i}':np.arange(i,32,4) for i in range(4)}}
        return {name:{k:data[k][:,rows] for k in ('targets','J','X')} for name,rows in names.items()}
    return {f'condition_{i}':{k:data[k][:,:,i:i+1] for k in ('targets','J','X')} for i in range(8 if role=='known' else 7)}


def analyze_combination(folder, role):
    report,data,evidence=load_combination(folder,role); baseline=np.array(report['shared_score_mean'])
    n=len(data['targets']); index=draws(n); result={}; max_comparison_drift=0.
    for name,arrays in subset_arrays(data,role).items():
        stats={arm:diagnose(arrays[arm],arrays['targets'],baseline) for arm in ('J','X')}
        contrast=paired_loss(arrays['J'],arrays['X'],arrays['targets'],index)
        stored=report['subsets'][name]['comparison']
        delta=max(abs(contrast['first_minus_second_dx_mse']-stored['J_minus_X_dx_mse']),
                  float(np.max(np.abs(np.array(contrast['paired_world_95ci'])-stored['paired_95ci']))))
        assert delta<1e-12,(name,delta);max_comparison_drift=max(max_comparison_drift,delta)
        for arm in ('J','X'):
            np.testing.assert_allclose(stats[arm]['per_dimension_mse'],report['subsets'][name]['readers'][arm]['scores']['per_dimension_mse'],rtol=0,atol=1e-12)
        result[name]=dict(readers=stats,J_minus_X=contrast)
    negative=None
    if role=='confirm':
        negative={arm:dict(diagnostics=diagnose(data[arm+'_negative'],data['negative_targets'],baseline),
                           all_motion=response_rms(data[arm+'_negative'][...,4:],index),
                           axes={name:response_rms(data[arm+'_negative'][...,i+4],index) for i,name in enumerate(MOTION)}) for arm in ('J','X')}
    return dict(role=role,source_commit=report['source_commit'],frozen_reader_sha256=report['frozen_reader_sha256'],
                evidence=evidence,score_mean=baseline.tolist(),subsets=result,negative_controls=negative,
                original_000_to_111_subset='condition_7' if role=='known' else None,
                primary_subset='all_views' if role=='confirm' else None,
                frozen_comparison_max_abs_drift=max_comparison_drift),data,report


def axis_effects(prediction, truth, baseline):
    n=len(prediction); assert prediction.shape==truth.shape==(n,2,7,8)
    index=draws(n); answer={}
    for obj, rows in enumerate(AXIS_GROUPS):
        p=prediction[:,:,rows];y=truth[:,:,rows];col=4+2*obj
        x=p[:,:,1]-p[:,:,0];vertical=p[:,:,2]-p[:,:,0]
        interaction=p[:,:,3]-p[:,:,2]-p[:,:,1]+p[:,:,0]
        np.testing.assert_allclose(x+vertical+interaction,p[:,:,3]-p[:,:,0],rtol=0,atol=1e-12)
        target=y[:,:,3,col];active=np.abs(target*224)>=1
        prod_x=p[:,:,1,col]*target;prod_xy=p[:,:,3,col]*target
        counts=dict(active_x_episodes=int(active.sum()),x_only_wrong=int(((prod_x<0)&active).sum()),
                    xy_wrong=int(((prod_xy<0)&active).sum()),x_only_zero=int(((prod_x==0)&active).sum()),
                    xy_zero=int(((prod_xy==0)&active).sum()),
                    x_only_correct_xy_wrong=int(((prod_x>0)&(prod_xy<0)&active).sum()),
                    x_only_wrong_xy_correct=int(((prod_x<0)&(prod_xy>0)&active).sum()),
                    pure_y_opposes_true_x=int(((vertical[...,col]*target<0)&active).sum()))
        answer['A' if obj==0 else 'B']=dict(direction=counts,
            conditions={name:diagnose(p[:,:,i],y[:,:,i],baseline) for i,name in enumerate(('none','x_only','y_only','xy'))},
            effects={name:response_rms(value[...,col],index) for name,value in (('x_only_change',x),('y_only_false_x',vertical),('interaction_x',interaction))})
    return answer


def load_old_axis(folder):
    folder=Path(folder); report=read_json(folder/'report.json');data=load_npz(folder/'predictions.npz')
    assert report['status']=='complete' and report['paired_worlds']==256 and report['episodes']==512
    assert not report['test_opened'] and not report['reader_refit'] and report['condition_names']==list(AXIS_NAMES)
    for key in ('targets','global_W','lane_W'):
        assert data[key].shape==(512,7,8) and np.isfinite(data[key]).all()
    return report,{k:data[k].reshape(256,2,7,8) for k in ('targets','global_W','lane_W')},dict(folder=str(folder),files={name:identity(folder/name) for name in ('report.json','predictions.npz')},remote_readback=readback(folder))


def plot(final, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    color={'J':'#327497','X':'#d48a42','global_W':'#607781','lane_W':'#378472'}
    reports=final['combination'];supp=final['supplemental']
    ax=axs[0,0]
    if 'known' in reports:
        row=reports['known']['subsets']['condition_7'];names=['J','X'];v=[row['readers'][k]['dx_mse']*1e4 for k in names]
        if 'known_original_vs_old_lane' in supp:
            names+=['old lane'];v+=[supp['known_original_vs_old_lane']['old_lane']['dx_mse']*1e4]
        ax.bar(names,v,color=[color.get(k,'#378472') for k in names]);ax.set_ylabel('Horizontal MSE (x 10^-4)')
        ax.set_title('Known Dev: original 000 -> 111\nOld lane is a cross-pipeline reference')
    else:ax.text(.5,.5,'Known results not supplied',ha='center',va='center',transform=ax.transAxes)
    ax=axs[0,1]
    if 'confirm' in reports:
        labels=['all32','old16','mixed16'];keys=['all_views','old_joint_views','mixed_object_views']
        for i,arm in enumerate(('J','X')):
            vals=[reports['confirm']['subsets'][k]['readers'][arm]['dx_mse']*1e4 for k in keys]
            ax.bar(np.arange(3)+(i-.5)*.32,vals,width=.30,color=color[arm],label=arm)
        ax.set_xticks(range(3),labels);ax.set_ylabel('Horizontal MSE (x 10^-4)');ax.legend()
        ax.set_title('Fresh64 worlds: object-sign composition')
    else:
        ax.text(.5,.5,'Fresh confirmation not supplied / not accessed',ha='center',va='center',transform=ax.transAxes)
        ax.set_axis_off()
    axis=supp.get('axis_same_inputs',{}).get('readers',{})
    for j in (0,1):
        ax=axs[1,j]
        if axis:
            names=list(axis);width=.36
            for offset,obj,hatch in ((-.18,'A',''),(.18,'B','//')):
                if j==0:
                    records=[axis[k][obj]['effects']['y_only_false_x'] for k in names]
                    vals=np.array([x['rms_pixels'] for x in records]);ci=np.array([x['paired_world_95ci_pixels'] for x in records])
                else:
                    vals=np.array([axis[k][obj]['direction']['xy_wrong']-axis[k][obj]['direction']['x_only_wrong'] for k in names])
                ax.bar(np.arange(len(names))+offset,vals,width=width,label=obj,hatch=hatch,color=[color[k] for k in names],edgecolor='white')
                if j==0:ax.errorbar(np.arange(len(names))+offset,vals,yerr=np.maximum(0,np.vstack((vals-ci[:,0],ci[:,1]-vals))),fmt='none',ecolor='black',capsize=3)
            ax.set_xticks(range(len(names)),names)
            if j==0:ax.legend(title='Output object',loc='lower left')
            if j==0:ax.set_title('Known axis intervention: Y-only false X');ax.set_ylabel('Horizontal response RMS (px)')
            else:
                ax.set_title('X-only -> XY: net wrong-sign change\nA solid / B hatched; |true X| >= 1px')
                ax.set_ylabel('XY wrong - X-only wrong (episodes)');ax.axhline(0,color='#555555',lw=.8)
        else:ax.text(.5,.5,'Axis results not supplied',ha='center',va='center',transform=ax.transAxes)
    for ax in axs.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('V5.6 combination and axis diagnostics: fixed readers, separate populations',fontsize=14)
    fig.text(.5,.012,'Known: 256 worlds x2 variants. Fresh: 64 worlds x32 views. Error bars: conditional paired-world 95% bootstrap; no new gate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95))
    for suffix in ('png','svg'):fig.savefig(out/('overview.'+suffix),dpi=160)
    plt.close(fig)


def self_test():
    # Fake software values only: prove zeros have their own denominators and the
    # first axis is the complete paired-world resampling unit.
    y=np.zeros((3,2,4,8));p=np.zeros_like(y)
    y[0,0,0,4]=.01;p[0,0,0,4]=-.01
    y[0,1,0,4]=.01;p[0,1,0,4]=0
    y[1,0,0,4]=.001;p[1,0,0,4]=.001
    p[2,1,2,4]=.2
    m=diagnose(p,y,np.zeros(8))
    assert m['axis']['A_dx']['nonzero_target_direction']['denominator']==3
    assert m['axis']['A_dx']['at_least_one_pixel_direction']['denominator']==2
    assert m['axis']['A_dx']['static_target_count']==21
    assert m['axis']['A_dx']['static_nonzero_prediction_count']==1
    assert m['axis']['A_dy']['nonzero_target_direction']['wrong_fraction'] is None
    assert m['mean_motion_skill'] is None
    rng=np.random.default_rng(56);a=rng.normal(size=(3,2,4,8));b=rng.normal(size=a.shape)
    indices=draws(3);got=paired_loss(a,b,y,indices)
    world=np.array([np.mean(((a[i]-y[i])**2-(b[i]-y[i])**2)[...,[4,6]]) for i in range(3)])
    reference=np.array([np.mean([world[j] for j in row]) for row in indices])
    np.testing.assert_allclose(got['paired_world_95ci'],np.quantile(reference,[.025,.975]),rtol=0,atol=1e-14)
    data={k:np.arange(2*32*8).reshape(2,32,8) for k in ('J','X','targets')}
    split=subset_arrays(data,'confirm')
    assert np.array_equal(np.concatenate((split['old_joint_views']['targets'],split['mixed_object_views']['targets']),axis=1),data['targets'])
    assert set(np.concatenate([np.arange(i,32,4) for i in range(4)]))==set(range(32))
    return dict(status='passed',software_only=True,world_bootstrap_independent_reference=True,
                zero_static_and_large_axis_denominators=True,fresh_old16_mixed16_and_paths=True,
                scientific_results_accessed=False)


def run(args):
    if not any((args.known,args.confirm,args.axis,args.old_axis)):
        raise ValueError('Supply completed result directories explicitly; no automatic result discovery')
    inputs=[p.resolve() for p in (args.known,args.confirm,args.axis,args.old_axis,args.old_known) if p]
    target=args.out.resolve()
    if any(target==p or p in target.parents for p in inputs):raise ValueError('Output must be outside every input directory')
    assert not args.out.exists(),'Use a new output directory; preserve previous analyses'
    reports={};datasets={};originals={}
    for role in ('known','confirm','axis'):
        path=getattr(args,role)
        if path:
            reports[role],datasets[role],originals[role]=analyze_combination(path,role)
    if len(reports)>1:
        assert len({r['frozen_reader_sha256'] for r in reports.values()})==1
        assert len({r['source_commit'] for r in reports.values()})==1
    baseline=np.array(next(iter(reports.values()))['score_mean']) if reports else None
    supplemental={}
    if args.old_known:
        assert args.known,'Old known predictions require J/X known targets for exact comparison'
        oldpath=args.old_known/'predictions.npz';old=load_npz(oldpath);current=datasets['known']
        assert identity(oldpath)==originals['known']['inputs']['report/predictions.npz']
        assert np.array_equal(old['targets'],current['targets'])
        baseline=np.array(reports['known']['score_mean']);truth=current['targets'][:,:,7]
        assert old['lane_W'].shape==current['targets'].shape
        supplemental['known_original_vs_old_lane']=dict(evidence=identity(oldpath),
            old_lane=diagnose(old['lane_W'][:,:,7],truth,baseline),
            comparisons={arm:paired_loss(old['lane_W'][:,:,7],current[arm][:,:,7],truth) for arm in ('J','X')},
            scope='Same original000-to111 inputs and target. Descriptive cross-pipeline reference, not a single-variable data-coverage effect.')
    if args.old_axis:
        assert args.axis,'Old axis comparison requires J/X axis for shared targets/cache binding'
        original,old,evidence=load_old_axis(args.old_axis);current=datasets['axis']
        assert original['cache_sha256']==originals['axis']['inputs']['cache.npz']['sha256']
        assert np.array_equal(old['targets'],current['targets'])
        arms={**{k:old[k] for k in ('global_W','lane_W')},**{k:current[k] for k in ('J','X')}}
        stats={name:axis_effects(p,current['targets'],baseline) for name,p in arms.items()}
        for model in ('global_W','lane_W'):
            for obj,name in enumerate(('A','B')):
                for key,value in stats[model][name]['direction'].items():
                    assert value==original['summary'][model][str(obj)][key],(model,name,key)
        supplemental['axis_same_inputs']=dict(evidence=evidence,readers=stats,
            frozen_old_direction_counts_exactly_reproduced=True,
            J_minus_X_each_condition={AXIS_NAMES[i]:paired_loss(current['J'][:,:,i],current['X'][:,:,i],current['targets'][:,:,i]) for i in range(7)})
    elif args.axis:
        supplemental['axis_same_inputs']=dict(readers={arm:axis_effects(datasets['axis'][arm],datasets['axis']['targets'],baseline) for arm in ('J','X')})
    if args.axis:
        current=datasets['axis'];contrasts={}
        for obj,rows in enumerate(AXIS_GROUPS):
            values={arm:current[arm][:,:,rows] for arm in ('J','X')};col=4+2*obj
            eff={arm:dict(y_only_false_x=p[:,:,2,col]-p[:,:,0,col],
                          interaction_x=p[:,:,3,col]-p[:,:,2,col]-p[:,:,1,col]+p[:,:,0,col]) for arm,p in values.items()}
            contrasts['A' if obj==0 else 'B']={name:paired_rms(eff['J'][name],eff['X'][name]) for name in ('y_only_false_x','interaction_x')}
        supplemental['axis_same_inputs']['J_minus_X_effect_rms']=contrasts
    caveats=[
        'J vs X uses the same256 worlds, same X8192 reference, free-intercept kernel and alpha; J duplicated supervision collapses exactly to4096 rows. This contrast changes target-object sign-combination coverage.',
        'Fresh64 confirms object-sign composition only. Within-object x/y remain coupled; fresh32 views are not32 independent worlds.',
        'Known000-to111 has256 worlds with2 variants. Fresh all32 has64 new worlds. They use different populations and must never be pooled or treated as one headline denominator.',
        'Old lane uses restricted spatial access and the old N+W reference. J/X use full spatial access and the new X reference. J/X-versus-lane is not an isolated coverage effect.',
        'All reports retain both raw MSE and directional counts. Static targets are excluded from sign denominators and report false-motion RMS separately; zero predictions are neither correct nor wrong sign and are counted separately.',
        'At-least1px direction counts are separate from all nonzero target counts. Mean-motion skill is null if any fixed-baseline denominator is zero.',
        'New intervals added here are conditional posthoc paired-world intervals, not multiplicity-corrected new gates. Only the already registered freshall32 J-X comparison is the prespecified primary.',
        'Axis interventions and knownDev remain exploratory. No memory-module benefit, original gate recovery or automatic P0 authorization follows.'
    ]
    final=dict(schema='v5.6-result-crosscheck-1',status='complete',script=identity(Path(__file__)),
               software_fixture=self_test(),combination=reports,supplemental=supplemental,
               bootstrap=dict(seed=SEED,replicates=REPLICATES,unit='whole world'),caveats=caveats)
    args.out.mkdir(parents=True)
    write(args.out/'report.json',final)
    plot(final,args.out)
    lines=['# V5.6 J/X 与 axis 独立复核','',
           '仅分析明确传入的已完成结果。方向统计的运动轴、≥1px 轴、静止轴和零预测分别保留；没有重训或重新选择参数。','']
    if 'known' in reports:
        lines+=['## 已看 Dev：固定原始 000→111','']
        row=reports['known']['subsets']['condition_7']
        lines += [f"- {arm}: dx MSE {row['readers'][arm]['dx_mse']:.9g}；motion skill {row['readers'][arm]['mean_motion_skill']}。" for arm in ('J','X')]
        lines += [f"- J−X dx MSE = {row['J_minus_X']['first_minus_second_dx_mse']:.9g}；配对 CI {row['J_minus_X']['paired_world_95ci']}。",'']
    if 'confirm' in reports:
        lines+=['## Fresh64：固定 all32 / old16 / mixed16','',
                '| 子集 | J dx MSE | X dx MSE | J−X [配对 95% CI] |','|---|---:|---:|---|']
        for name in ('all_views','old_joint_views','mixed_object_views'):
            row=reports['confirm']['subsets'][name];comparison=row['J_minus_X']
            lines.append(f"| {name} | {row['readers']['J']['dx_mse']:.9g} | {row['readers']['X']['dx_mse']:.9g} | {comparison['first_minus_second_dx_mse']:.9g} {comparison['paired_world_95ci']} |")
        lines+=['','4 个黄色路径、逐轴方向、静止轴和 center 负控全部保留在 JSON；不以子组挑选赢家。','']
    if 'axis_same_inputs' in supplemental:
        lines+=['## 同输入 axis：水平输出被垂直变化影响','',
                '| reader / 目标 | y-only false-x RMS px [95% CI] | x-only 错 / xy 错 | 正确→错 / 错→正确 | ≥1px 分母 |',
                '|---|---|---:|---:|---:|']
        for arm,objects in supplemental['axis_same_inputs']['readers'].items():
            for obj,v in objects.items():
                d=v['direction'];effect=v['effects']['y_only_false_x']
                lines.append(f"| {arm}/{obj} | {effect['rms_pixels']:.6f} {effect['paired_world_95ci_pixels']} | {d['x_only_wrong']} / {d['xy_wrong']} | {d['x_only_correct_xy_wrong']} / {d['x_only_wrong_xy_correct']} | {d['active_x_episodes']} |")
        lines+=['','y-only 项和二阶 interaction 均保留在 JSON；xy 错误变化同时包含伤害与救回，不能全部归因到 y-only 项。','']
    lines+=['![概览](overview.png)','','## 公平比较与范围','']+['- '+value for value in caveats]+['']
    (args.out/'README.md').write_text('\n'.join(lines))
    write(args.out/'manifest.json',{name:identity(args.out/name) for name in ('report.json','README.md','overview.png','overview.svg')})
    print(json.dumps(dict(status='complete',roles=list(reports),output=str(args.out),manifest=identity(args.out/'manifest.json'))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    for name in ('known','confirm','axis','old-axis','old-known','out'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    else:
        if args.out is None:parser.error('--out is required')
        run(args)


if __name__=='__main__':main()
