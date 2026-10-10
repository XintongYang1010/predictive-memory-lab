#!/usr/bin/env python3
"""Plot an already completed V5.5 report. No fitting, scoring, or sample selection.

Thumbnails are optional and require published-source and per-image hash checks;
without --thumbnails this script reads only the three completed report artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
READERS=('W_concat','lane_W')
NAMES={'W_concat':'Global W','lane_W':'Lane W'}
COLORS={'W_concat':'#35658a','lane_W':'#ce7c35'}
MASKS=tuple(format(m,'03b') for m in range(8))
MASK_LABELS=('000\nnone','001\nY','010\nB','011\nB+Y','100\nA','101\nA+Y','110\nA+B','111\nA+B+Y')
DIMENSIONS=('A dx','A dy','B dx','B dy')
BAD_CASES=((844,1),(761,1),(891,1))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def evidence(path):
    return dict(path=str(Path(path).resolve()),bytes=Path(path).stat().st_size,sha256=sha(path))


def verified_inputs(folder):
    required=('report.json','predictions.npz','pool_per_episode.json')
    manifest=read(folder/'manifest.json')
    for name in required:
        path=folder/name
        if path.is_symlink() or manifest.get(name)!={'bytes':path.stat().st_size,'sha256':sha(path)}:
            raise ValueError('Report manifest mismatch: '+name)
    report=read(folder/'report.json')
    if report.get('status')!='diagnostic_complete' or report.get('schema')!='v5.5-module-report-1':
        raise ValueError('Only a completed V5.5 diagnostic is supported')
    if report.get('original_parity',{}).get('passed') is not True:
        raise ValueError('Original parity must pass before plotting matched interventions')
    return report,{name:evidence(folder/name) for name in required+('manifest.json',)}


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#e6e8eb',linewidth=.7,zorder=0)
    ax.set_axisbelow(True)


def savefig(fig,out,stem,plt):
    fig.savefig(out/(stem+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(out/(stem+'.svg'),bbox_inches='tight',facecolor='white',metadata={'Date':None})
    plt.close(fig)


def all_masks(report,out,plt,np):
    fig,(ax,delta)=plt.subplots(2,1,figsize=(11,7),height_ratios=(2.1,1.2),layout='constrained')
    x=np.arange(8);width=.34
    for j,reader in enumerate(READERS):
        value=[report['readers'][reader]['masks'][m]['dx_mse']*1e4 for m in MASKS]
        ax.bar(x+(j-.5)*width,value,width,label=NAMES[reader],color=COLORS[reader],zorder=3)
    ax.set_xticks(x,MASK_LABELS);ax.set_ylabel('A/B mean dx MSE (×10⁻⁴)')
    ax.set_title('All endpoint combinations; first frame fixed at original t6',loc='left',weight='bold')
    ax.legend(frameon=False);style(ax)
    means=np.array([report['global_vs_lane'][m]['global_minus_lane_dx_mse'] for m in MASKS])*1e4
    intervals=np.array([report['global_vs_lane'][m]['paired_95ci'] for m in MASKS])*1e4
    # Use lines directly: a percentile interval need not contain its point estimate.
    delta.vlines(x,intervals[:,0],intervals[:,1],color='#33475b',linewidth=2)
    delta.scatter(x,means,color='#33475b',s=28,zorder=3)
    delta.axhline(0,color='#66717a',linewidth=.8)
    delta.set_xticks(x,MASKS);delta.set_ylabel('Global − lane dx MSE\n(×10⁻⁴; positive favors lane)')
    delta.set_xlabel('Bits: A, B, yellow. Conditional descriptive 95% paired intervals; world resampling.')
    style(delta);savefig(fig,out,'01_all_masks_dx_mse',plt)


def shapley_rms(report,out,plt,np):
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    components=(('own_response','Own target','#397b91'),('other_target_response','Other target','#a4658c'),('yellow_response','Yellow','#b59d30'))
    for row,reader in enumerate(READERS):
        for obj in range(2):
            ax=axes[row,obj];index=np.array([obj*2,obj*2+1]);x=np.arange(2);width=.24
            for j,(key,label,color) in enumerate(components):
                vals=np.asarray(report['readers'][reader]['attribution'][key]['rms_over_fit_rms'])[index]
                ax.bar(x+(j-1)*width,vals,width,label=label,color=color,zorder=3)
            ax.set_xticks(x,['dx','dy']);ax.set_ylabel('Shapley response RMS / FIT motion RMS')
            ax.set_title(NAMES[reader]+' · target '+('A' if obj==0 else 'B'),loc='left')
            ax.axhline(1,color='#77828d',linestyle=':',linewidth=.9);style(ax)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
    fig.supxlabel('Scene-component response attribution. RMS components do not sum; not fractions of neural-module fault.',fontsize=9)
    savefig(fig,out,'02_shapley_component_rms',plt)


def direction_amplitude(report,out,plt,np):
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    x=np.arange(4);width=.35
    for j,reader in enumerate(READERS):
        original=report['readers'][reader]['original']
        values=(np.asarray(original['motion_direction_error_rate'])*100,
                np.asarray(original['motion_absolute_amplitude_ratio']))
        for ax,value in zip(axes,values):
            ax.bar(x+(j-.5)*width,value,width,label=NAMES[reader],color=COLORS[reader],zorder=3)
    for ax in axes:ax.set_xticks(x,DIMENSIONS);style(ax)
    axes[0].set_ylabel('Direction error (%)');axes[0].set_title('Original transition: sign',loc='left')
    axes[1].set_ylabel('Mean |prediction| / mean |target|');axes[1].set_title('Original transition: amplitude',loc='left')
    axes[1].axhline(1,color='#556575',linestyle='--',linewidth=.9)
    axes[0].legend(frameon=False)
    fig.supxlabel('Original mask111 only; 512 already observed episodes. Ratios describe output scale, not causality.',fontsize=9)
    savefig(fig,out,'03_original_direction_amplitude',plt)


def pool_energy(report,rows,out,plt,np):
    values=np.array([r['retained_fraction'] for r in rows if r['retained_fraction'] is not None])
    if len(rows)!=512 or len({(r['pair_id'],r['variant']) for r in rows})!=512:
        raise ValueError('Expected all512 original cue-difference energy records')
    if not len(values) or np.any(~np.isfinite(values)) or np.min(values)<-1e-10 or np.max(values)>1+1e-10:
        raise ValueError('Invalid projection energy fractions')
    if not np.isclose(values.mean(),report['pooling']['projection_summary']['retained_fraction_mean'],atol=1e-12,rtol=0):
        raise ValueError('Pooling rows and report summary differ')
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    axes[0].hist(values,bins=np.linspace(0,1,26),color='#53758e',edgecolor='white')
    axes[0].axvline(np.median(values),color='#bc783b',linewidth=1.5,label=f'Median {np.median(values):.3f}')
    axes[0].set_ylabel('Episodes');axes[0].set_xlabel('Retained squared-norm fraction');axes[0].legend(frameon=False)
    axes[1].plot(np.sort(values),np.arange(1,len(values)+1)/len(values),color='#53758e',linewidth=2)
    axes[1].set_ylabel('Empirical cumulative fraction');axes[1].set_xlabel('Retained squared-norm fraction')
    for ax in axes:ax.set_xlim(0,1);style(ax)
    fig.suptitle('Original raw-token change after 16×16 → 4×4 mean pooling',weight='bold')
    fig.supxlabel('Projection energy retained is not the fraction of motion information retained.',fontsize=9)
    savefig(fig,out,'04_pool_projection_energy',plt)


def case_predictions(data,out,plt,np):
    for pair,variant in BAD_CASES:
        index=pair-640
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
        fig.subplots_adjust(left=.085,right=.985,bottom=.10,top=.87,wspace=.24,hspace=.24)
        for j,ax in enumerate(axes.flat):
            dim=4+j;truth=data['targets'][index,variant,:,dim]
            ax.plot(range(8),truth*224,color='#27343f',marker='o',markersize=4,linewidth=1.6,label='Target')
            for reader in READERS:
                ax.plot(range(8),data[reader][index,variant,:,dim]*224,color=COLORS[reader],
                        marker='s' if reader=='W_concat' else '^',markersize=4,linewidth=1.3,label=NAMES[reader])
            ax.axhline(0,color='#98a1a8',linewidth=.7)
            ax.set_title(DIMENSIONS[j],loc='left');ax.set_ylabel('Observed displacement (pixels)')
            ax.set_xticks(range(8),MASKS);style(ax)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.952),ncol=3,frameon=False)
        fig.suptitle(f'Fixed known case {pair}/{variant}: all eight endpoint combinations',weight='bold',y=.995)
        fig.supxlabel('Bits A/B/Y; target displacement uses the original physical labels. These cases were fixed before this run.',fontsize=9)
        savefig(fig,out,f'case_{pair}_{variant}_predictions',plt)


def thumbnails(args,report,inputs,out,plt,np):
    if args.cache_dir is None or args.registration_publication is None:
        raise ValueError('Thumbnails require --cache-dir and --registration-publication')
    path=args.cache_dir/'receipt.json';receipt=read(path);pub=read(args.registration_publication)
    if (sha(path)!=report['provenance']['cache_receipt_sha256'] or pub.get('verified_readback') is not True
            or pub.get('source_commit')!=report['provenance']['source_commit']
            or pub.get('plan_sha256')!=receipt['plan_sha256'] or pub.get('protocol_sha256')!=receipt['protocol_sha256']):
        raise ValueError('Published scientific source and cache receipt must match before thumbnails')
    module='scripts/v5_5_counterfactual.py'
    if sha(REPO/module)!=receipt['source_identity'][module]:raise ValueError('Composite renderer differs from published source')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(REPO/'scripts'))
    from v5_5_counterfactual import composite_images,array_identity
    inputs['cache_receipt']=evidence(path);inputs['registration_publication']=evidence(args.registration_publication)
    for pair,variant in BAD_CASES:
        shard=args.cache_dir/'shards'/f'{pair:06d}.json';record=read(shard)
        images,_,_,_=composite_images('dev',pair,variant)
        expected=record['inputs']['variants'][variant]['unique_rgb']
        if [array_identity(rgb) for rgb in images]!=expected:raise ValueError('Published composite RGB hashes differ')
        inputs[f'case_{pair}_input_receipt']=evidence(shard)
        fig,axes=plt.subplots(3,3,figsize=(9,9),layout='constrained')
        axes[0,0].imshow(images[0]);axes[0,0].set_title('Fixed first frame: original t6',fontsize=10)
        for mask,ax in enumerate(list(axes.flat)[1:]):
            ax.imshow(images[mask]);ax.set_title('Endpoint '+MASKS[mask],fontsize=10)
        for ax in axes.flat:ax.set_axis_off()
        fig.suptitle(f'Fixed known case {pair}/{variant} · bitwise-verified endpoint RGB',weight='bold')
        savefig(fig,out,f'case_{pair}_{variant}_thumbnails',plt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=BASE/'figures')
    parser.add_argument('--thumbnails',action='store_true')
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--registration-publication',type=Path)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve existing figures; choose a fresh --out')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.labelsize':10,'svg.fonttype':'none','savefig.transparent':False})
    report,inputs=verified_inputs(args.report_dir)
    with np.load(args.report_dir/'predictions.npz',allow_pickle=False) as z:
        data={key:z[key] for key in ('targets','W_concat','lane_W')}
    if any(v.shape!=(256,2,8,8) or not np.isfinite(v).all() for v in data.values()):
        raise ValueError('Expected finite predictions/labels for the complete fixed population')
    for reader in READERS:
        for mask,name in enumerate(MASKS):
            mse=float(np.mean((data[reader][...,mask,[4,6]]-data['targets'][...,mask,[4,6]])**2))
            if not np.isclose(mse,report['readers'][reader]['masks'][name]['dx_mse'],rtol=1e-12,atol=1e-15):
                raise ValueError('Saved predictions disagree with reported dx MSE')
    rows=read(args.report_dir/'pool_per_episode.json')
    args.out.mkdir(parents=True,exist_ok=False)
    all_masks(report,args.out,plt,np);shapley_rms(report,args.out,plt,np)
    direction_amplitude(report,args.out,plt,np);pool_energy(report,rows,args.out,plt,np)
    case_predictions(data,args.out,plt,np)
    if args.thumbnails:thumbnails(args,report,inputs,args.out,plt,np)
    manifest=dict(schema='v5.5.descriptive-figures-1',created_unix=time.time(),
        renderer=evidence(__file__),inputs=inputs,source_commit=report['provenance']['source_commit'],
        fixed_cases=[dict(pair_id=p,variant=v) for p,v in BAD_CASES],no_fit=True,no_model_selection=True,
        thumbnails_generated=args.thumbnails,scientific_result_modified=False,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(args.out.iterdir())})
    (args.out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out),figures=len(manifest['files']),status='written_pending_visual_inspection')),flush=True)


if __name__=='__main__':main()
