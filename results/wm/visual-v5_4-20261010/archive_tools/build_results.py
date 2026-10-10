#!/usr/bin/env python3
"""Read-only V5.4 reports, independent CPU rescoring and figures; no fitting/data generation."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key,'4')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np

ARMS=('N','W')
KEYS=('N_concat','W_concat')
PATHS=('aa','bb','ab','ba')
DIMS=('A_dx','A_dy','B_dx','B_dy')
ALL_DIMS=('A_x','A_y','B_x','B_y')+DIMS
COLORS=('#687c99','#cb662f')
ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def array_identity(a):
    a=np.asarray(a)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def read_json(path):
    def invalid(value):raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def equal(actual,expected,name):
    if not np.allclose(actual,expected,rtol=1e-9,atol=1e-12):raise ValueError('Report/prediction mismatch: '+name)


def compare_tree(actual,expected,name):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Schema mismatch: '+name)
        for key in expected:compare_tree(actual[key],expected[key],name+'/'+key)
    elif expected is None or isinstance(expected,(str,bool)):
        if actual!=expected:raise ValueError('Value mismatch: '+name)
    else:equal(actual,expected,name)


class Evidence:
    def __init__(self,root):self.root=Path(root).resolve();self.inventory={}
    def record(self,name):
        path=self.root/name
        self.inventory[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
        return path
    def read(self,name,optional=False):
        if optional and not (self.root/name).exists():return None
        return read_json(self.record(name))
    def verify_unchanged(self):
        for name,value in self.inventory.items():
            path=self.root/name
            if path.stat().st_size!=value['bytes'] or sha(path)!=value['sha256']:
                raise ValueError('Immutable input changed during analysis: '+name)


def load_role(evidence,role):
    report=evidence.read(role+'_report/report.json')
    if (report.get('schema')!='v5.4-reader-evaluation-1' or report.get('status')!='evaluation_complete' or
        report.get('role')!=role or report.get('groups')!=256 or
        report.get('examples')!=(512 if role=='bridge' else 4096) or
        report.get('reader_arms')!=list(ARMS) or report.get('inputs')!=['concat'] or
        report.get('test_opened') is not False or report.get('p0_authorized') is not False or
        report.get('original_v5_gate')!='failed_unchanged'):
        raise ValueError('Role/schema/count/gate declaration mismatch: '+role)
    path=evidence.record(role+'_report/predictions.npz')
    if sha(path)!=report['predictions_sha256']:raise ValueError('Prediction hash mismatch: '+role)
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    idkey='pair_ids' if role=='bridge' else 'block_ids'
    keys={*KEYS,'targets','role',idkey}
    if role=='anchor':keys|={'negative_'+k for k in KEYS}|{'negative_targets'}
    if set(a)!=keys or a['role'].shape!=() or str(a['role'].item())!=role:
        raise ValueError('Prediction schema mismatch: '+role)
    expected=np.arange(640,896) if role=='bridge' else np.arange(256)
    if not np.array_equal(a[idkey],expected):raise ValueError('Prediction IDs mismatch: '+role)
    shape=(256,2,8) if role=='bridge' else (256,16,8)
    for key in (*KEYS,'targets'):
        if a[key].shape!=shape or not np.isfinite(a[key]).all():raise ValueError('Invalid '+role+'/'+key)
    if role=='anchor':
        for key in ('negative_targets',*('negative_'+k for k in KEYS)):
            if a[key].shape!=(256,2,8) or not np.isfinite(a[key]).all():raise ValueError('Invalid negative control')
        if np.any(a['negative_targets'][:,:,4:]!=0):raise ValueError('Negative target motion not zero')
    copied=evidence.read(role+'_report/cache_receipt.json')
    original=evidence.read(role+'_cache/receipt.json')
    if original!=copied or sha(evidence.root/(role+'_cache/receipt.json'))!=report['cache_receipt_sha256']:
        raise ValueError('Original cache receipt identity mismatch')
    for key in ('cache_sha256','frozen_reader_sha256','freeze_publication_sha256','narrow_reference'):
        if copied.get(key)!=report.get(key):raise ValueError('Report/cache reference mismatch: '+key)
    if copied.get('schema')!='v5.4-spatial-support-1' or copied.get('role')!=role or copied.get('test_opened') is not False:
        raise ValueError('Cache schema/role/Test mismatch')
    if copied.get('frozen_before_generation') is not True or copied.get('unseen_before_release') is not True:
        raise ValueError('Cache missing frozen/fresh release attestation')
    if report['scores']['bootstrap'].get('seed')!=7811 or report['scores']['bootstrap'].get('replicates')!=10000:
        raise ValueError('Bootstrap contract changed')
    return dict(report=report,arrays=a,receipt=copied)


def authenticate_moments(evidence,anchor,bridge,plan_path):
    frozen=evidence.read('reader/freeze_receipt.json')
    publication=evidence.read('freeze_publication.json')
    digest=sha(evidence.root/'reader/freeze_receipt.json')
    if (frozen.get('schema')!='v5.4-reader-freeze-1' or frozen.get('status')!='frozen_before_evaluation' or
        frozen.get('fit_views')!=8192 or frozen.get('arm_views')!={'N':4096,'W':4096} or
        frozen.get('test_opened') is not False or frozen.get('p0_authorized') is not False):
        raise ValueError('Frozen model declarations differ')
    identity=frozen['cache_identity'];plan=read_json(plan_path)
    if sha(plan_path)!=identity['plan_sha256']:raise ValueError('Plan differs from frozen model')
    if (publication.get('verified_readback') is not True or publication.get('frozen_reader_sha256')!=digest or
        publication.get('plan_sha256')!=identity['plan_sha256'] or publication.get('source_commit')!=identity['source_commit'] or
        publication.get('verified_unix',0)<frozen['created_unix']):raise ValueError('Model publication invalid')
    for role in (anchor,bridge):
        r,c=role['report'],role['receipt']
        if r['frozen_reader_sha256']!=digest or r['freeze_publication_sha256']!=sha(evidence.root/'freeze_publication.json'):
            raise ValueError('Evaluation linked to different freeze/publication')
        if any(c.get(k)!=v for k,v in identity.items()):raise ValueError('Evaluation/model cache identity mismatch')
        if c.get('narrow_reference')!=frozen['narrow_reference']:raise ValueError('Narrow reference mismatch')
    audit=frozen['matched_label_audit']
    if audit.get('motion_labels_bitwise_equal') is not True or audit.get('maximum_absolute_motion_difference')!=0:
        raise ValueError('Matched FIT labels not verified')
    # Each registered target has its exact reverse. Recover the joint scoring
    # mean from both arm means, then authenticate its bytes against frozen NPZ.
    positions=(np.asarray(audit['position_target_mean']['N'],dtype=np.float64)+
               np.asarray(audit['position_target_mean']['W'],dtype=np.float64))/2
    mean=np.concatenate((positions,np.zeros(4,dtype=np.float64)))
    fit_rms=np.asarray(anchor['report']['scores']['fit_motion_rms'],dtype=np.float64)
    for name,value in (('shared_target_mean',mean),('fit_motion_rms',fit_rms)):
        if array_identity(value)!=frozen['model_entry_hashes'][name]:raise ValueError('Frozen scoring moment hash mismatch: '+name)
    return frozen,plan,mean,fit_rms


def metrics(prediction,target,mean):
    p=np.asarray(prediction,dtype=np.float64).reshape(-1,8)
    y=np.asarray(target,dtype=np.float64).reshape(-1,8)
    mse=np.sum((p-y)**2,axis=0)/len(y);den=np.sum((y-mean)**2,axis=0)/len(y)
    if np.any(den<=0):raise ValueError('Degenerate full-population scoring denominator')
    skill=1-mse/den;pp,yy=p[:,4:],y[:,4:]
    return dict(per_dimension_mse=mse.tolist(),per_dimension_skill=skill.tolist(),
        position_skill=float(np.mean(skill[:4])),motion_skill=float(np.mean(skill[4:])),motion_nmse=float(np.mean(mse[4:]/den[4:])),
        motion_direction_error_rate=np.mean(np.sign(pp)!=np.sign(yy),axis=0).tolist(),
        motion_target_zero_count=np.count_nonzero(yy==0,axis=0).tolist(),
        motion_prediction_zero_count=np.count_nonzero(pp==0,axis=0).tolist(),motion_sign_tie_count=np.count_nonzero(pp*yy==0,axis=0).tolist(),
        motion_signed_bias=np.mean(pp-yy,axis=0).tolist(),
        motion_prediction_target_covariance=np.mean((pp-pp.mean(0))*(yy-yy.mean(0)),axis=0).tolist(),
        motion_absolute_amplitude_ratio=(np.mean(np.abs(pp),axis=0)/np.mean(np.abs(yy),axis=0)).tolist(),
        motion_rms_ratio=np.sqrt(np.mean(pp**2,axis=0)/np.mean(yy**2,axis=0)).tolist(),
        motion_std_ratio=(np.std(pp,axis=0)/np.std(yy,axis=0)).tolist())


def spatial_metrics(a,mean,spec):
    y=a['targets'].astype(np.float64);lower=np.asarray(spec['lower']);upper=np.asarray(spec['upper'])
    if spec['tolerance']!=0:raise ValueError('Spatial tolerance changed')
    p7=y[:,:,:4];p6=p7-y[:,:,4:]
    coordinate_out=np.zeros((len(y),4),dtype=bool)
    # Explicit complete-world reduction, independently of the scorer helper.
    for world in range(len(y)):
        for axis in range(4):
            cue=np.r_[p6[world,:,axis],p7[world,:,axis]]
            coordinate_out[world,axis]=bool(np.any(cue<lower[axis]) or np.any(cue>upper[axis]))
    result=dict(world_groups={},coordinate_groups={})
    def group(mask,axis=None):
        rows={}
        for key in KEYS:
            if not mask.any():rows[key]=None;continue
            err=np.mean((a[key][mask,:,4:]-y[mask,:,4:])**2,axis=(0,1))
            denominator=np.mean((y[mask,:,4:]-mean[4:])**2,axis=(0,1))
            skill=[float(1-e/d) if d>0 else None for e,d in zip(err,denominator)]
            rows[key]=(dict(motion_mse=float(err[axis]),motion_skill=skill[axis]) if axis is not None else
                dict(per_coordinate_motion_mse=err.tolist(),per_coordinate_motion_skill=skill,dx_mse=float(np.mean(err[[0,2]])),
                     motion_mse=float(np.mean(err)),mean_motion_skill=float(np.mean(skill)) if all(v is not None for v in skill) else None))
        return dict(worlds=int(mask.sum()),examples=int(mask.sum()*y.shape[1]),readers=rows)
    outside=np.any(coordinate_out,axis=1)
    for label,mask in (('inside',~outside),('outside',outside)):result['world_groups'][label]=group(mask)
    for axis,name in enumerate(DIMS):
        result['coordinate_groups'][name]={label:group(mask,axis) for label,mask in
            (('inside',~coordinate_out[:,axis]),('outside',coordinate_out[:,axis]))}
    return result


def verify_points(anchor,bridge,mean,fit_rms,spec):
    recomputed={}
    for role,data in (('anchor',anchor),('bridge',bridge)):
        a,r=data['arrays'],data['report']['scores'];y=a['targets'].astype(np.float64)
        recomputed[role]={}
        for key in KEYS:
            measured=metrics(a[key],y,mean);row=r['readers'][key]
            compare_tree(measured,row['all_views'] if role=='anchor' else row,role+'/'+key)
            recomputed[role][key]=measured
            if role!='anchor':continue
            yy=y.reshape(256,4,4,8);pp=a[key].reshape(256,4,4,8)
            for path_id,path in enumerate(PATHS):
                recorded=row['paths'][path]
                p=pp[:,:,path_id];t=yy[:,:,path_id]
                compare_tree(metrics(p,t,mean),recorded['absolute_outputs'],key+'/'+path+'/absolute')
                contrasts=np.stack(((p[:,0,4:]-p[:,1,4:])/2,(p[:,2,4:]-p[:,3,4:])/2),axis=1)
                d=t[:,0,4:];mse=np.mean((contrasts-d[:,None,:])**2,axis=0)
                skill=1-mse/np.mean(d**2,axis=0)
                equal(skill,recorded['end_start_per_coordinate_skill'],key+'/'+path+'/contrast')
                equal(mse,recorded['end_start_motion_mse'],key+'/'+path+'/contrast MSE')
                equal(np.sqrt(np.mean(contrasts**2,axis=0)),recorded['end_start_prediction_rms'],key+'/'+path+'/contrast amplitude')
            for group,indices in (('static',[0,1]),('moving',[2,3])):
                point=metrics(pp[:,:,indices],yy[:,:,indices],mean)
                for field,value in point.items():equal(value,row['schedule_groups'][group][field],key+'/'+group+'/'+field)
            negative=a['negative_'+key];sq=negative[:,:,4:]**2
            rms=np.sqrt(np.mean(sq,axis=(0,1)));bydirection=np.sqrt(np.mean(sq,axis=0))
            for field,value in (
                ('negative_control_motion_rms',rms),('negative_control_motion_rms_by_direction',bydirection),
                ('negative_control_motion_rms_over_FIT_motion_rms',rms/fit_rms),
                ('negative_control_motion_rms_by_direction_over_FIT',bydirection/fit_rms),
                ('negative_control_overall_motion_rms',np.sqrt(np.mean(sq))),
                ('negative_control_overall_motion_rms_over_FIT',np.sqrt(np.mean(sq))/np.sqrt(np.mean(fit_rms**2))),
                ('negative_control_motion_mean',np.mean(negative[:,:,4:],axis=(0,1))),
                ('negative_control_position_mse',np.mean((negative[:,:,:4]-a['negative_targets'][:,:,:4])**2,axis=(0,1)))):
                equal(value,row[field],key+'/'+field)
        spatial=spatial_metrics(a,mean,spec)
        for field,value in spatial.items():compare_tree(value,r['spatial_support'][field],role+'/spatial/'+field)
        recomputed[role]['spatial_support']=spatial
    return recomputed


def verify_bootstrap(bridge,mean):
    """Independent direct world-index loop; never call the frozen scorer."""
    a=bridge['arrays'];y=a['targets'].astype(np.float64)
    errors={arm:(a[arm+'_concat']-y)**2 for arm in ARMS}
    point=(errors['N']-errors['W']).mean((0,1));den=((y-mean)**2).mean((0,1))
    primary=bridge['report']['scores']['primary']
    equal(point[[4,6]].mean(),primary['N_minus_W_dx_mse'],'primary point')
    equal(np.mean(point[4:]/den[4:]),primary['N_minus_W_motion_nmse'],'secondary NMSE point')
    draws=np.empty((10000,2));rng=np.random.default_rng(7811)
    for replicate in range(10000):
        indices=rng.integers(0,256,256)
        difference=(errors['N'][indices]-errors['W'][indices]).mean((0,1))
        denominator=((y[indices]-mean)**2).mean((0,1))
        draws[replicate]=[difference[[4,6]].mean(),np.mean(difference[4:]/denominator[4:])]
    ci=np.quantile(draws,[.025,.975],axis=0)
    equal(ci[:,0],primary['dx_mse_paired_95ci'],'independent primary bootstrap')
    equal(ci[:,1],primary['motion_nmse_descriptive_paired_95ci'],'independent secondary bootstrap')
    if bridge['report']['scores']['primary_improvement_interval_above_zero']!=bool(ci[0,0]>0):
        raise ValueError('Primary interval flag mismatch')
    mse={arm:float(errors[arm][:,:,[4,6]].mean()) for arm in ARMS}
    return dict(method='independent direct world-index loop; 10000 resamples, seed7811',
        conditional_on='one fixed fitted N/W model pair; no training-resampling uncertainty',
        N_dx_mse=mse['N'],W_dx_mse=mse['W'],N_minus_W_dx_mse=float(point[[4,6]].mean()),
        relative_dx_mse_reduction_percent=float(100*(mse['N']-mse['W'])/mse['N']) if mse['N']>0 else None,
        dx_mse_paired_95ci=ci[:,0].tolist(),motion_nmse_descriptive_paired_95ci=ci[:,1].tolist(),
        primary_ci_max_absolute_difference=float(np.max(np.abs(ci[:,0]-primary['dx_mse_paired_95ci']))),
        secondary_ci_max_absolute_difference=float(np.max(np.abs(ci[:,1]-primary['motion_nmse_descriptive_paired_95ci']))))


def plots(out,anchor,bridge,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    ar,br=anchor['report']['scores'],bridge['report']['scores']
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for i,arm in enumerate(ARMS):
        row=br['readers'][arm+'_concat'];x=np.arange(4)+(i-.5)*.34
        for ax,values in zip(axes,(row['per_dimension_skill'][4:],row['motion_direction_error_rate'],row['motion_absolute_amplitude_ratio'])):
            ax.bar(x,values,width=.34,color=COLORS[i],label=arm)
    for ax in axes:ax.set_xticks(np.arange(4),DIMS);ax.legend()
    axes[0].axhline(0,color='black',lw=.8);axes[2].axhline(1,color='black',lw=.8,ls='--')
    axes[0].set(title='Fresh bridge motion skill',ylabel='1 - MSE / joint FIT-mean baseline')
    axes[1].set(title='Direction errors',ylabel='Sign mismatch rate',ylim=(0,1))
    axes[2].set(title='Predicted motion amplitude',ylabel='Mean |prediction| / mean |target|')
    fig.suptitle('N: narrow supervision; W: wide supervision; both refitted on shared joint FIT kernel',fontsize=11)
    fig.tight_layout();fig.savefig(out/'bridge_motion_diagnostics.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(ARMS,[audit['N_dx_mse'],audit['W_dx_mse']],color=COLORS)
    axes[0].set(title='Primary loss: A_dx and B_dx',ylabel='Mean raw squared displacement error')
    effect=audit['N_minus_W_dx_mse'];interval=np.asarray(audit['dx_mse_paired_95ci'])
    axes[1].plot([0,0],interval,color='black',lw=2)
    axes[1].scatter([0],[effect],s=55,color=COLORS[1]);axes[1].axhline(0,color='grey',ls='--')
    axes[1].set(xticks=[0],xticklabels=['N minus W'],ylabel='Raw MSE reduction',title='10,000 whole-world bootstrap 95% CI')
    fig.suptitle('Conditional on this one frozen N/W model pair; 256 paired worlds',fontsize=10)
    fig.tight_layout();fig.savefig(out/'primary_bridge_effect.png');plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(11,8.5),gridspec_kw={'height_ratios':[1,2]})
    for i,arm in enumerate(ARMS):
        values=[ar['readers'][arm+'_concat']['paths'][p]['absolute_outputs']['motion_skill'] for p in PATHS]
        axes[0].bar(np.arange(4)+(i-.5)*.34,values,width=.34,color=COLORS[i],label=arm)
    axes[0].set(xticks=np.arange(4),xticklabels=['aa static','bb static','ab moving','ba moving'],
                title='Fresh wide anchor: absolute motion by nuisance path',ylabel='Mean motion skill')
    axes[0].axhline(0,color='black',lw=.8);axes[0].legend()
    heat=np.asarray([ar['readers'][key]['paths'][p]['end_start_per_coordinate_skill'] for key in KEYS for p in PATHS]).reshape(8,8)
    bound=max(1,float(np.abs(heat).max()))
    im=axes[1].imshow(heat,cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
    axes[1].set(yticks=np.arange(8),yticklabels=[a+'/'+p for a in ARMS for p in PATHS],xticks=np.arange(8),
        xticklabels=['end '+d for d in DIMS]+['start '+d for d in DIMS],title='Nonzero target contrasts: each path and coordinate retained')
    axes[1].tick_params(axis='x',rotation=25)
    for r in range(8):
        for c in range(8):axes[1].text(c,r,f'{heat[r,c]:.2f}',ha='center',va='center',fontsize=8,
                                       color='white' if abs(heat[r,c])>bound*.6 else 'black')
    fig.colorbar(im,ax=axes[1],shrink=.85,label='Contrast skill')
    fig.tight_layout();fig.savefig(out/'anchor_path_diagnostics.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    fit=np.asarray(ar['fit_motion_rms']);overall=np.sqrt(np.mean(fit**2))
    for i,arm in enumerate(ARMS):
        row=ar['readers'][arm+'_concat']
        axes[0].bar(np.arange(4)+(i-.5)*.34,row['negative_control_motion_rms_over_FIT_motion_rms'],width=.34,color=COLORS[i],label=arm)
        direction=np.sqrt(np.mean(np.asarray(row['negative_control_motion_rms_by_direction'])**2,axis=1))/overall
        axes[1].bar(np.arange(2)+(i-.5)*.34,direction,width=.34,color=COLORS[i],label=arm)
    for ax in axes:ax.legend();ax.set_ylabel('Predicted RMS / nonzero FIT motion RMS')
    axes[0].set(xticks=np.arange(4),xticklabels=DIMS,title='Static target, moving nuisance')
    axes[1].set(xticks=[0,1],xticklabels=['a to b','b to a'],title='Directions squared before aggregation')
    fig.suptitle('Lower negative response is useful only alongside preserved nonzero target motion',fontsize=10)
    fig.tight_layout();fig.savefig(out/'negative_control_rms.png');plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for r,(role,data) in enumerate((('Wide anchor',anchor),('Fresh bridge',bridge))):
        groups=data['report']['scores']['spatial_support']['world_groups']
        labels=[f'{g}\nn={groups[g]["worlds"]} worlds' for g in ('inside','outside')]
        for i,key in enumerate(KEYS):
            for c,field in enumerate(('dx_mse','mean_motion_skill')):
                values=[groups[g]['readers'][key][field] if groups[g]['readers'][key] is not None and groups[g]['readers'][key][field] is not None else np.nan for g in ('inside','outside')]
                axes[r,c].bar(np.arange(2)+(i-.5)*.34,values,width=.34,label=ARMS[i],color=COLORS[i])
                axes[r,c].set(xticks=np.arange(2),xticklabels=labels,title=role+': narrow-envelope proxy',
                              ylabel='Raw horizontal MSE' if c==0 else 'Mean motion skill')
        for ax in axes[r]:ax.legend()
        axes[r,1].axhline(0,color='black',lw=.8)
    fig.suptitle('Descriptive fixed groups; all views/cues/coordinates retained per world; no subgroup CI or preservation gate',fontsize=10)
    fig.tight_layout();fig.savefig(out/'spatial_coverage_groups.png');plt.close(fig)


def fmt(value):return '无定义/空组' if value is None else f'{float(value):.7g}'
def ci(value):return '['+', '.join(fmt(v) for v in value)+']'


def write_tables(out,anchor,bridge):
    br=bridge['report']['scores'];ar=anchor['report']['scores']
    with (out/'bridge_axis_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['reader','axis','mse','skill','sign_error_rate','abs_amplitude_ratio','rms_ratio','std_ratio','signed_bias'])
        for key in KEYS:
            row=br['readers'][key]
            for d,name in enumerate(DIMS):w.writerow([key,name,row['per_dimension_mse'][4+d],row['per_dimension_skill'][4+d],
                row['motion_direction_error_rate'][d],row['motion_absolute_amplitude_ratio'][d],row['motion_rms_ratio'][d],
                row['motion_std_ratio'][d],row['motion_signed_bias'][d]])
    with (out/'anchor_contrast_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['reader','path','anchor','axis','skill','lower95','upper95','interval_source'])
        for key in KEYS:
            for path in PATHS:
                row=ar['readers'][key]['paths'][path]
                for side,label in enumerate(('end','start')):
                    for d,name in enumerate(DIMS):w.writerow([key,path,label,name,row['end_start_per_coordinate_skill'][side][d],
                        row['end_start_per_coordinate_skill_descriptive_95ci'][0][side][d],
                        row['end_start_per_coordinate_skill_descriptive_95ci'][1][side][d],'frozen scorer; descriptive unadjusted'])
    with (out/'spatial_group_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['role','unit','axis','group','worlds','examples','reader','dx_mse','mean_motion_mse','mean_motion_skill'])
        for role,data in (('anchor',anchor),('bridge',bridge)):
            spatial=data['report']['scores']['spatial_support']
            for group,row in spatial['world_groups'].items():
                for key,value in row['readers'].items():w.writerow([role,'whole_world','all',group,row['worlds'],row['examples'],key,
                    value['dx_mse'] if value else None,value['motion_mse'] if value else None,value['mean_motion_skill'] if value else None])
            for axis,groups in spatial['coordinate_groups'].items():
                for group,row in groups.items():
                    for key,value in row['readers'].items():w.writerow([role,'whole_world',axis,group,row['worlds'],row['examples'],key,
                        None,value['motion_mse'] if value else None,value['motion_skill'] if value else None])


def write_readme(out,anchor,bridge,audit,metadata,mean,fit_rms):
    ar,br=anchor['report']['scores'],bridge['report']['scores'];n,w=(br['readers'][k] for k in KEYS)
    low,high=audit['dx_mse_paired_95ci']
    if low>0:conclusion='主区间整体为正：本轮支持 W 相对 N 降低 fresh bridge 水平位移误差。'
    elif high<0:conclusion='主区间整体为负：本轮 W 的 fresh bridge 水平位移误差高于 N，不支持此次宽域覆盖干预带来改善。'
    else:conclusion='主区间包含零：本轮无法明确确认 W 相对 N 降低 fresh bridge 水平位移误差。'
    if w['motion_skill']<=0:conclusion+='W 的总体 motion skill 非正，不能称绝对运动读出已经修复。'
    elif min(w['per_dimension_skill'][4:])<=0:conclusion+='W 的总体 motion skill 为正，但至少一个运动坐标未超过固定基线；绝对表现仍有缺口。'
    else:conclusion+='W 的四个运动坐标点估计均超过固定基线；这仍不等于准确追踪或完整协议通过。'
    lines=['# V5.4：匹配窄域/宽域覆盖确认','',conclusion,'',
        'N 复用 V5.3 已有 FIT 特征和标签，W 使用同一组 latent seeds 与量化前中心分位数，只扩大中心采样范围。'
        '两组各 256 个世界、16 个平衡 nuisance 视图；联合 8192 个 FIT 视图定义同一 quadratic 核及两级中心化/尺度。'
        '两组各用 4096 个监督行重新拟合，不惩罚截距，固定 mean-loss alpha=7.8125e-7（对角项 0.0032）。'
        'N 不是直接复用 V5.3 旧模型，也使用联合 N+W 的无标签归一化；本轮只有固定 concat 输入。','',
        f'- 唯一主端点：新 original-generator Dev 640–895；每世界两个 variants 和 A_dx/B_dx 的原始平方误差。',
        f'- N dx MSE：**{fmt(audit["N_dx_mse"])}**；W：**{fmt(audit["W_dx_mse"])}**；相对下降 **{fmt(audit["relative_dx_mse_reduction_percent"])}%**（负值代表增加）。',
        f'- **N−W = {fmt(audit["N_minus_W_dx_mse"])}**；配对 95% CI **{ci(audit["dx_mse_paired_95ci"])}**。',
        '- 10,000 次完整世界重采样，seed 7811；同时保留两个 variants、两组预测与坐标。'
        'CI 条件于这一对已经拟合的冻结模型，不包括重新抽取 FIT、重复训练或模型选择不确定性。',
        '- 该工具独立重算主效应及 bootstrap，未调用科学评分器、拟合模型、重新选择 alpha 或生成任何评估样本。','',
        '## Fresh bridge 的绝对表现','',
        '| Arm | mean motion skill | A_dx | A_dy | B_dx | B_dy | position skill |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in KEYS:
        row=br['readers'][key];lines.append('| '+key+' | '+' | '.join(fmt(v) for v in [row['motion_skill'],*row['per_dimension_skill'][4:],row['position_skill']])+' |')
    lines+=['','| Arm / 坐标 | sign error | mean abs 幅度比 | RMS 比 | SD 比 | signed bias |','|---|---:|---:|---:|---:|---:|']
    for key in KEYS:
        row=br['readers'][key]
        for d,name in enumerate(DIMS):lines.append('| '+key+'/'+name+' | '+' | '.join(fmt(row[field][d]) for field in
            ('motion_direction_error_rate','motion_absolute_amplitude_ratio','motion_rms_ratio','motion_std_ratio','motion_signed_bias'))+' |')
    lines+=['','幅度比以真实非零运动为分母；只看 MSE 降低可能掩盖输出缩幅。符号错误把预测/标签符号不同计为错误，零和 tie 计数保留于原报告。',
        '','![Bridge motion](bridge_motion_diagnostics.png)','','![Primary effect](primary_bridge_effect.png)','',
        '## Fresh wide anchor：每条路径保留非零目标对照','',
        '| Arm/path | absolute motion skill | 8 个 contrast 中最低点估计 | mean-motion 描述性 95% CI |',
        '|---|---:|---:|---:|---:|']
    minimum={}
    for key in KEYS:
        minimum[key]=min(float(np.min(ar['readers'][key]['paths'][path]['end_start_per_coordinate_skill'])) for path in PATHS)
        for path in PATHS:
            row=ar['readers'][key]['paths'][path]
            lines.append(f'| {key}/{path} | {fmt(row["absolute_outputs"]["motion_skill"])} | {fmt(np.min(row["end_start_per_coordinate_skill"]))} | {ci(row["mean_motion_skill_descriptive_95ci"])} |')
    lines+=['',f'N / W 的逐路径、逐坐标最低 contrast skill 分别为 **{fmt(minimum["N_concat"])} / {fmt(minimum["W_concat"])}**。',
        'aa/bb 是静止 nuisance，ab/ba 是两种移动路径；每条路径先作 end/start 差分、平方误差，再汇总。'
        '辅助 CI 使用已冻结评分器输出，属于未调整的描述性区间；相关坐标和反向视图不能当作独立重复。',
        '','![Anchor diagnostics](anchor_path_diagnostics.png)','',
        '## 静止目标负控','',
        '| Arm | A_dx RMS/FIT | A_dy RMS/FIT | B_dx RMS/FIT | B_dy RMS/FIT | overall RMS/FIT |',
        '|---|---:|---:|---:|---:|---:|']
    for key in KEYS:
        row=ar['readers'][key]
        lines.append('| '+key+' | '+' | '.join(fmt(v) for v in [*row['negative_control_motion_rms_over_FIT_motion_rms'],row['negative_control_overall_motion_rms_over_FIT']])+' |')
    lines+=['','负控标签为零目标位移。a→b 和 b→a 的响应分别平方后再求 RMS，未先平均有符号响应。'
        '分母是冻结 FIT 非零目标 motion RMS；负控更低只有结合非零目标表现才可解释为减少虚假运动响应。',
        f'Anchor 中 nuisance 两位置落在同一像素的 block 数为 {anchor["report"].get("coincident_nuisance_pixel_blocks","未提供")}，按计划保留。',
        '','![Negative controls](negative_control_rms.png)','',
        '## 原窄域外包络内外的描述性表现','',
        '| Population/group | worlds | N dx MSE | W dx MSE | N mean motion skill | W mean motion skill |',
        '|---|---:|---:|---:|---:|---:|']
    for role,data in (('wide anchor',anchor),('fresh bridge',bridge)):
        for group,row in data['report']['scores']['spatial_support']['world_groups'].items():
            readers=row['readers']
            values=[readers[k][field] if readers[k] is not None else None for field in ('dx_mse','mean_motion_skill') for k in KEYS]
            lines.append('| '+role+'/'+group+' | '+str(row['worlds'])+' | '+' | '.join(fmt(v) for v in values)+' |')
    inside=ar['spatial_support']['world_groups']['inside']
    lines+=['',f'Wide-anchor 的全坐标窄域内子集实际只有 **{inside["worlds"]} 个世界**；原样报告，不补抽、不合并成新的通过阈值。',
        'p7 使用缓存 FP32 标签转 FP64，p6=p7−d；按预注册窄域量化外包络、零容差判定。'
        '整体分组中，只要一个世界任一视图/variant、任一 cue、任一坐标越界，该世界即 outside。'
        '逐坐标分组同样保留整个世界，完整 counts/MSE/skill 见 CSV。空组或零分母为 null。'
        '这些是条件组点估计，无 subgroup CI；外包络不是完整联合支持集或实际 FIT 最近邻覆盖。'
        '宽 anchor 总分改善不能证明窄域能力完整保留，尤其不能忽略很小的 inside 子集。',
        '','![Spatial groups](spatial_coverage_groups.png)','',
        '## 结论边界与复核','',
        '扩大位置范围同时改变空间密度与目标相对 nuisance 的几何关系。即使相对误差改善，也不能仅据本轮区分所有机制；'
        '负结果也不能排除其他覆盖设计。所有解释限于这一冻结 I-JEPA pooled-feature 表示、固定 quadratic 读出器和注册分布。'
        'BF16 推理、FP32 pooling、BF16 量化后 FP32 缓存不等于完整 FP32 特征提取。','',
        '**原 V5 feature gate 与 V5.1 bridge 的失败档案不变；本轮没有替换旧门槛、打开 Test 或授权 P0。**'
        '尚不能据此声称 world-model、历史必要性、记忆机制或完整 V5 训练协议有效。','',
        f'- 主 bootstrap 独立复算与冻结报告的最大绝对 CI 差：{fmt(audit["primary_ci_max_absolute_difference"])}。',
        '- 全坐标 MSE/skill、方向/幅度/偏差、逐路径 anchor contrasts、零目标 RMS、窄域内外 counts 与点分数均从实际预测重新计算。',
        '- 评分均值及 FIT motion RMS 的数组字节 SHA 与冻结模型收据一致；预测、原始 cache receipt、模型发布收据均核对。',
        '- `independent_recomputation.json` 保存复算结果；`input_inventory.json` 记录所有实际输入及哈希；输入目录保持只读。','',
        '## 执行与复现','']
    for name in ('fit_controller.json','evaluation_controller.json'):
        controller=metadata.get(name)
        if not controller:continue
        lines.append(f'- `{name}`：`{controller.get("status")}`，source `{controller.get("source_commit")}`。')
        for job in controller.get('jobs',[]):
            ids=sorted({m.group(1) for p in job.get('logs',[]) for m in [re.search(r'_(\d+\.\d+)\.log$',p)] if m})
            lines.append(f'- Stage `{job.get("stage")}`：`{job.get("status")}`；Slurm step `{", ".join(ids) or "未提供"}`；selected GPUs `{job.get("selected_gpus","未提供")}`。')
    ledger=metadata.get('ledger/budget.json')
    if ledger:lines.append(f'- 已下载 ledger charged GPUh：`{ledger.get("charged_gpuh")}`；最终对账以完整归档为准。')
    lines+=['- 这个派生报告工具不运行远端命令，也不执行或推定清理；清理状态由最终归档记录说明。',
        '- 主图、CSV、README、独立复算 JSON 的字节哈希记录在 `SHA256SUMS`；脚本本身随输出复制。','',
        '```sh','/Library/Developer/CommandLineTools/usr/bin/python3 build_results.py \\',
        '  --scene /path/to/verified/report-snapshot \\',
        '  --plan /path/to/frozen/scripts/v5_4_plan.json \\',
        '  --out /path/to/new-analysis-directory','```','']
    (out/'README.md').write_text('\n'.join(lines))


def build(scene,out,plan_path):
    evidence=Evidence(scene);anchor=load_role(evidence,'anchor');bridge=load_role(evidence,'bridge')
    frozen,plan,mean,fit_rms=authenticate_moments(evidence,anchor,bridge,plan_path)
    recomputed=verify_points(anchor,bridge,mean,fit_rms,plan['supporting_metrics']['spatial_groups'])
    bootstrap=verify_bootstrap(bridge,mean)
    metadata={name:evidence.read(name,optional=True) for name in ('fit_controller.json','evaluation_controller.json',
        'registration_publication.json','ledger/budget.json','final_accounting.json')}
    out=Path(out).resolve()
    if out==evidence.root or evidence.root in out.parents:raise ValueError('Outputs must be outside immutable input scene')
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'independent_recomputation.json',dict(bootstrap=bootstrap,points=recomputed,
        frozen_target_mean=mean.tolist(),frozen_motion_rms=fit_rms.tolist(),test_opened=False,p0_authorized=False))
    write_tables(out,anchor,bridge);plots(out,anchor,bridge,bootstrap)
    write_readme(out,anchor,bridge,bootstrap,metadata,mean,fit_rms)
    evidence.verify_unchanged()
    shutil.copyfile(__file__,out/'build_results.py')
    dump(out/'input_inventory.json',dict(schema='v5.4.posthoc-report-inputs.1',inputs=evidence.inventory,
        plan=dict(path=str(Path(plan_path).resolve()),sha256=sha(plan_path)),renderer_sha256=sha(__file__),
        frozen_reader_sha256=bridge['report']['frozen_reader_sha256'],input_files_unchanged=True,
        confidence_intervals=dict(primary='independently recomputed direct wholeworld bootstrap',
            secondary_bridge_motion='independently recomputed with resampled denominator',
            anchor_and_negative_support='copied from frozen scorer; descriptive unadjusted'),
        independent_of_scientific_scorer=True,no_fit_or_data_generation=True))
    paths=sorted(p for p in out.rglob('*') if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in paths))
    for line in (out/'SHA256SUMS').read_text().splitlines():
        digest,relative=line.split('  ',1)
        if sha(out/relative)!=digest:raise ValueError('Output hash check failed')
    return dict(output=str(out),files=len(paths)+1,inputs=len(evidence.inventory),primary=bootstrap,
                test_opened=False,p0_authorized=False,scientific_source_modified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'scripts/v5_4_plan.json')
    args=parser.parse_args()
    print(json.dumps(build(args.scene,args.out,args.plan),indent=2,allow_nan=False))


if __name__=='__main__':main()
