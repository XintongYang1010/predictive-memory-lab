#!/usr/bin/env python3
"""Posthoc read-only interpretation of the completed V5.6 correspondence run."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='4'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import __version__ as pillow_version

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts'))
from v5_5_counterfactual import cue_geometry,render_endpoint
from v5_4_reader import entry_identity

CASES=((761,1),(844,1),(891,1))
METHODS=('raw_native','pool_native','uniform_budget','roi_adaptive_budget','random_budget','change_priority_budget')
LABELS=('Raw native','Pool native','Uniform budget','ROI budget','Random budget','Change priority')
COLORS=('#2171b5','#e6550d','#756bb1','#238b45','#9e9e9e','#cb181d')
SEAL='b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def csv_write(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def check_sums(folder):
    answer={}
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if sha(folder/name)!=digest:raise ValueError('Locked correspondence file changed: '+name)
        answer[name]=digest
    return answer

def intervals(error):
    rng=np.random.default_rng(7816);world_indices=rng.integers(0,256,(10000,256));answer={}
    for name,baseline,method in (('pool_minus_raw',1,0),('uniform_minus_roi',2,3)):
        # Both horizontal objects, then both variants; one number per world.
        delta=(error[:,baseline,:,0]-error[:,method,:,0]).mean(1)
        worlds=delta.reshape(256,2).mean(1);boot=worlds[world_indices].mean(1)
        ci=np.quantile(boot,[.025,.975])
        answer[name]=dict(mse_reduction=float(worlds.mean()),paired_95ci=ci.tolist(),
            pixel_squared_reduction=float(worlds.mean()*224**2),pixel_squared_95ci=(ci*224**2).tolist(),
            worlds=256,episodes=512,bootstrap_replicates=10000,seed=7816,
            inference='posthoc conditional on this fixed diagnostic/data population; no model or matcher reselection')
    return answer

def clamp_groups(error,rows):
    flag=np.zeros((512,2),dtype=bool)
    for row in rows:
        if row['method']=='pool_native':
            i=(int(row['pair_id'])-640)*2+int(row['variant']);o=('A','B').index(row['object'])
            flag[i,o]=int(row['source_clamped_points'])>0
    output={}
    for yes in (False,True):
        keep=flag==yes
        group=dict(object_queries=int(keep.sum()),episodes=int(keep.any(1).sum()),
                   worlds=int(keep.reshape(256,2,2).any((1,2)).sum()),methods={})
        for m,name in enumerate(METHODS):
            e=error[:,m][keep]
            group['methods'][name]=dict(dx_mse=float(e[:,0].mean()),motion_mse=float(e.mean()),
                dx_rmse_pixels=float(np.sqrt(e[:,0].mean())*224))
        for key,base,m in (('pool_minus_raw',1,0),('uniform_minus_roi',2,3)):
            delta=error[:,base,:,0]-error[:,m,:,0]
            group[key]=dict(dx_mse_reduction=float(delta[keep].mean()),
                share_of_total_signed_dx_mse_reduction=float(delta[keep].sum()/delta.sum()))
        output['source_pool_clamped' if yes else 'source_pool_unclamped']=group
    return output

def plot_case(out,pair,variant,images,pred,truth,cost,case_rows):
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(18.5,10.7),layout='constrained')
    grid=fig.add_gridspec(3,6,height_ratios=[1.45,1,1])
    for t,cols in enumerate(((0,2),(2,4))):
        ax=fig.add_subplot(grid[0,cols[0]:cols[1]]);ax.imshow(images[t]);ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'Authenticated original RGB: t={6+t}',fontweight='bold')
    for obj in (0,1):
        ax=fig.add_subplot(grid[0,4+obj]);ax.axhline(0,color='.85',lw=.7);ax.axvline(0,color='.85',lw=.7)
        d=truth[obj*2:obj*2+2]*224
        ax.annotate('',xy=d,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color='black',lw=2.7))
        for m,color in enumerate(COLORS):
            v=pred[m,obj*2:obj*2+2]*224
            ax.annotate('',xy=v,xytext=(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=1.4,alpha=.8))
            ax.scatter(*v,s=16,color=color,zorder=3)
        ax.scatter(*d,marker='*',s=95,color='black',zorder=4)
        ax.set(xlim=(-7,7),ylim=(7,-7),xlabel='dx (pixels)',ylabel='dy (pixels, down)',aspect='equal')
        ax.set_title(f'Object {"AB"[obj]} displacement\nGT ({d[0]:+.2f}, {d[1]:+.2f})')
    for obj in (0,1):
        for m,name in enumerate(METHODS):
            ax=fig.add_subplot(grid[1+obj,m]);c=cost[m,obj].reshape(25,25)
            im=ax.imshow(c,extent=(-6.25,6.25,-6.25,6.25),origin='lower',cmap='viridis',aspect='equal')
            ax.invert_yaxis();d=truth[obj*2:obj*2+2]*224;v=pred[m,obj*2:obj*2+2]*224
            ax.scatter(*d,marker='*',s=58,color='white',edgecolors='black',linewidths=.5)
            ax.scatter(*v,marker='x',s=48,color='#f781bf',linewidths=1.8)
            row=case_rows[(name,'AB'[obj])]
            ax.set_title(f'{LABELS[m]} — {"AB"[obj]}\npred ({v[0]:+.1f}, {v[1]:+.1f}), gap {float(row["second_minus_best"]):.1e}',color=COLORS[m],fontsize=10)
            ax.set_xticks([-6,0,6]);ax.set_yticks([-6,0,6]);ax.set_xlabel('dx (pixels)')
            if m==0:ax.set_ylabel('dy (pixels, down)')
            fig.colorbar(im,ax=ax,shrink=.72,pad=.01,format='%.3g')
    handles=[Line2D([0],[0],color='black',lw=2,label='Continuous GT')]+[Line2D([0],[0],color=c,lw=2,label=s) for c,s in zip(COLORS,LABELS)]
    fig.legend(handles=handles,loc='outside lower center',ncol=7,frameon=False,fontsize=10)
    fig.suptitle(f'Previously selected bad case {pair}/{variant} — fixed correspondence, known Dev\nSource-RGB ROI is privileged; cost colors scale independently per panel; ★ GT, × selected displacement',fontsize=15)
    for suffix in ('png','svg'):fig.savefig(out/f'case_{pair}_{variant}.{suffix}',dpi=180)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--correspondence',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'analysis.json').exists():raise FileExistsError('Preserve completed interpretation')
    inputs=check_sums(a.correspondence)
    if sha(a.scene/'archive_manifest.json')!=SEAL:raise ValueError('Original scene seal differs')
    manifest=json.loads((a.scene/'archive_manifest.json').read_text())
    with np.load(a.correspondence/'results/costs_and_predictions.npz',allow_pickle=False) as z:
        pred=z['predictions'];truth=z['targets'];cost=z['costs'];methods=z['methods'].tolist()
        np.testing.assert_array_equal(z['pair_ids'],np.repeat(np.arange(640,896),2));np.testing.assert_array_equal(z['variants'],np.tile([0,1],256))
    assert methods==list(METHODS)
    error=((pred-truth[:,None])**2).reshape(512,6,2,2)
    rows=list(csv.DictReader((a.correspondence/'results/per_episode_object.csv').open()))
    selections=json.loads((a.correspondence/'results/selected_banks.json').read_text())
    banks={(x['pair_id'],x['variant'],x['method']):x for x in selections}
    coverage=[];rgb_records=[];case_predictions=[]
    for pair in range(640,896):
        rel=f'counterfactual_cache/shards/{pair:06d}.json';path=a.scene/rel
        if dict(bytes=path.stat().st_size,sha256=sha(path))!=manifest['files'][rel]:raise ValueError('Prior shard hash differs')
        shard=json.loads(path.read_text());inputs[str(path)]=sha(path)
        for variant in (0,1):
            idx=(pair-640)*2+variant;pos,colors,yellow=cue_geometry('dev',pair,variant)
            original=shard['inputs']['variants'][variant];images=[];masks=[]
            for t,maskid in ((0,0),(1,7)):
                rgb,mask=render_endpoint(pos[t],colors,yellow[t])
                if entry_identity(rgb)!=original['unique_rgb'][maskid] or entry_identity(mask)!=original['object_masks'][maskid]:raise ValueError('Authenticated original RGB/mask mismatch')
                images.append(rgb);masks.append(mask)
                rgb_records.append(dict(pair_id=pair,variant=variant,cue_time=6+t,rgb=entry_identity(rgb),mask=entry_identity(mask)))
            labels=np.concatenate((pos[1].ravel(),(pos[1]-pos[0]).ravel())).astype(np.float32)
            if entry_identity(labels)!=original['original_target']:raise ValueError('Original target SHA differs')
            np.testing.assert_array_equal(labels[4:].astype(float),truth[idx])
            for name in METHODS[3:]:
                codes=np.asarray(banks[pair,variant,name]['codes']);selected=codes[codes<256]
                assert len(selected)==12 and len(np.unique(selected))==12
                selected_cells=np.zeros(256,dtype=bool);selected_cells[selected]=True
                selected_pixels=np.repeat(np.repeat(selected_cells.reshape(16,16),14,axis=0),14,axis=1)
                row=dict(pair_id=pair,variant=variant,method=name,raw_tokens=12,raw_token_pixel_area=int(selected_pixels.sum()))
                for scope,mm in (('source',masks[0]),('either_frame_union',masks[0]|masks[1])):
                    for object_id,label in enumerate(('A','B','yellow')):
                        pixels=mm[object_id];counts=pixels.reshape(16,14,16,14).sum((1,3)).ravel()
                        covered=int(pixels[selected_pixels].sum())
                        row[f'{scope}_{label}_touching_raw_tokens']=int((counts[selected]>0).sum())
                        row[f'{scope}_{label}_covered_pixels']=covered
                        row[f'{scope}_{label}_total_pixels']=int(pixels.sum())
                        row[f'{scope}_{label}_area_coverage']=covered/float(pixels.sum())
                coverage.append(row)
            if (pair,variant) in CASES:
                rr={(x['method'],x['object']):x for x in rows if int(x['pair_id'])==pair and int(x['variant'])==variant}
                plot_case(a.out,pair,variant,images,pred[idx],truth[idx],cost[idx],rr)
                for m,name in enumerate(METHODS):
                    for obj in (0,1):
                        case_predictions.append(dict(pair_id=pair,variant=variant,object='AB'[obj],method=name,
                            true_dx_px=float(truth[idx,2*obj]*224),true_dy_px=float(truth[idx,2*obj+1]*224),
                            predicted_dx_px=float(pred[idx,m,2*obj]*224),predicted_dy_px=float(pred[idx,m,2*obj+1]*224)))
    cover_summary={}
    for name in METHODS[3:]:
        selected=[r for r in coverage if r['method']==name]
        means={k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ('pair_id','variant','method')}
        means['neither_target_touched_source_episodes']=sum(r['source_A_touching_raw_tokens']==0 and r['source_B_touching_raw_tokens']==0 for r in selected)
        means['at_least_one_target_untouched_source_episodes']=sum(r['source_A_touching_raw_tokens']==0 or r['source_B_touching_raw_tokens']==0 for r in selected)
        means['yellow_touched_source_episodes']=sum(r['source_yellow_touching_raw_tokens']>0 for r in selected)
        cover_summary[name]=means
    csv_write(a.out/'token_coverage.csv',coverage);csv_write(a.out/'fixed_case_predictions.csv',case_predictions)
    write(a.out/'rgb_hash_readback.json',rgb_records)
    output=dict(status='complete',posthoc=True,known_dev=True,test_opened=False,no_training=True,
        bootstrap=intervals(error),source_pool_clamp_strata=clamp_groups(error,rows),token_coverage=cover_summary,
        coverage_definition='Only the twelve saved raw-detail cell footprints (14x14 pixels each), excluding four global2x2 mean tokens; source and union-of-both-frames raster ellipse/yellow supports. Fractions are covered mask pixels / total mask pixels. Purely descriptive, not feature receptive fields.',
        fixed_cases=[list(x) for x in CASES],case_selection='Three pre-existing variant1 examples in old RGB observability README; not selected using these correspondence outcomes',
        provenance=dict(command=sys.argv,script_sha256=sha(__file__),pillow=pillow_version,numpy=np.__version__,matplotlib=matplotlib.__version__,
            correspondence_inputs=inputs,old_scene_manifest_sha256=SEAL,source_identity={p:sha(REPO/p) for p in ('scripts/v5_5_counterfactual.py','pmr/visual_memory/data.py','scripts/v5_6_correspondence.py')}),
        limits=['CI is posthoc and conditional on one fixed matcher and already observed Dev worlds.','No new learning, Test, tuning, protocol gate, or long-term memory evaluation.',
                'RGB colors select a privileged source ROI. Equal-byte comparisons share fixed IDW and physical candidate grid.',
                'Change priority reads both frames; overlap statistics do not identify a neural module or prove causal responsibility.',
                'Mean pooling changes spatial resolution and local descriptors; this finite matcher result does not show that all pooled decoders lack motion information.'])
    if check_sums(a.correspondence)!=inputs_without_shards(inputs):raise ValueError('Correspondence directory changed during analysis')
    write(a.out/'analysis.json',output)
    print(json.dumps(dict(status='complete',bootstrap=output['bootstrap'],clamp=output['source_pool_clamp_strata'],coverage=cover_summary),indent=2))

def inputs_without_shards(inputs):return {k:v for k,v in inputs.items() if not k.startswith('execution/') and not k.startswith('/')}

if __name__=='__main__':main()
