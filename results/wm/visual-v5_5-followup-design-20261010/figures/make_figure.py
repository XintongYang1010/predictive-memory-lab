#!/usr/bin/env python3
"""Reproduce the known-Dev RGB observability figure from sealed predictions only."""
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'rgb_observability'
# Bind the previously sealed nine-file artifact; do not modify any input.
for line in (INPUT/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1)
    if hashlib.sha256((INPUT/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Sealed RGB observability artifact changed: '+name)
with np.load(INPUT/'predictions.npz',allow_pickle=False) as z:
    arrays={k:z[k] for k in z.files}
ids,variants=arrays['pair_ids'],arrays['variants']
if not np.array_equal(ids,np.repeat(np.arange(640,896),2)) or not np.array_equal(variants,np.tile([0,1],256)):
    raise ValueError('Known-Dev episode order changed')
TRUTH=arrays['targets'][:,[4,6]]*224
METHODS=[('W_concat','Global W','#CC624B'),('lane_W','Lane W','#426CB5'),
         ('rgb_centroid_primary','RGB centroid','#258B75')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,
    'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,
    'svg.fonttype':'none','savefig.facecolor':'white'})
fig=plt.figure(figsize=(15,9),facecolor='white')
gs=fig.add_gridspec(2,3,height_ratios=[1.05,.85],left=.065,right=.985,bottom=.125,top=.86,hspace=.48,wspace=.23)
fig.suptitle('RGB observability and remaining readout errors',x=.525,y=.971,fontsize=20,fontweight='bold')
fig.text(.525,.93,'Known Dev: 512 episodes × 2 horizontal object coordinates; fixed readers, no refitting',ha='center',fontsize=12,color='#45515E')
limit=np.ceil(max(np.abs(TRUTH).max(),*(np.abs(arrays[k][:,[4,6]]*224).max() for k,_,_ in METHODS)))
for col,(key,label,color) in enumerate(METHODS):
    ax=fig.add_subplot(gs[0,col]);prediction=arrays[key][:,[4,6]]*224
    ax.scatter(TRUTH.ravel(),prediction.ravel(),s=10,alpha=.40,color=color,edgecolors='none',rasterized=False,zorder=2)
    ax.plot([-limit,limit],[-limit,limit],color='#444C55',lw=1,ls='--',zorder=1)
    ax.axhline(0,color='#D9DFE5',lw=.7,zorder=0);ax.axvline(0,color='#D9DFE5',lw=.7,zorder=0)
    ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit);ax.set_aspect('equal',adjustable='box')
    ax.set_title(label+(' · privileged diagnostic' if key=='rgb_centroid_primary' else ''),color=color,pad=10)
    ax.set_xlabel('Ground-truth horizontal displacement (px)')
    if col==0:ax.set_ylabel('Predicted horizontal displacement (px)')
    rmse=np.sqrt(np.mean((prediction-TRUTH)**2))
    ax.text(.04,.95,f'dx RMSE = {rmse:.3f} px\nn = 1,024 coordinates',transform=ax.transAxes,va='top',fontsize=10,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.91,pad=3))
    ax.grid(alpha=.13,linewidth=.65,zorder=0)
true_color='#39434F';bar_width=.185
for col,pair in enumerate((761,844,891)):
    ax=fig.add_subplot(gs[1,col]);idx=np.flatnonzero((ids==pair)&(variants==1))
    if len(idx)!=1:raise ValueError('Fixed bad case not unique')
    idx=int(idx[0]);series=[('True',TRUTH[idx],true_color)]+[(label,arrays[key][idx,[4,6]]*224,color) for key,label,color in METHODS]
    for j,(label,values,color) in enumerate(series):
        xx=np.arange(2)+(j-1.5)*bar_width
        ax.bar(xx,values,bar_width*.94,color=color,label=label,zorder=3)
        for x,y in zip(xx,values):
            ax.text(x,y+(.14 if y>=0 else -.14),f'{y:+.2f}',ha='center',va='bottom' if y>=0 else 'top',fontsize=8.5,rotation=0)
    ax.axhline(0,color='#5F6874',lw=.9,zorder=2);ax.set_xticks([0,1],['Object A','Object B'])
    ax.set_title(f'Fixed bad case {pair} / variant 1',pad=10)
    ax.set_xlim(-.57,1.57);ax.set_ylim(-5.25,5.25);ax.set_yticks([-4,-2,0,2,4]);ax.grid(axis='y',alpha=.15,zorder=0)
    if col==0:ax.set_ylabel('Horizontal displacement (px)')
handles=[Patch(facecolor=true_color,label='True displacement')]+[Patch(facecolor=c,label=l) for _,l,c in METHODS]
fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.52,.050),frameon=False,columnspacing=2.5,handlelength=1.4,fontsize=11)
fig.text(.525,.026,'RGB centroid uses renderer color identity. Privileged diagnostic only — not a learned-model benchmark.',ha='center',fontsize=10,color='#45515E')
fig.text(.525,.006,'Pixel units = normalized coordinates × 224. Bottom cases were fixed before this RGB audit; the auxiliary bbox rule is not plotted.',ha='center',fontsize=9,color='#66717E')
fig.savefig(HERE/'rgb_observability.png',dpi=240)
fig.savefig(HERE/'rgb_observability.svg')
# Matplotlib emits trailing spaces in SVG path lines; preserve geometry while
# making the generated text clean for Git review.
svg_path=HERE/'rgb_observability.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
plt.close(fig)
print('Saved rgb_observability.png and rgb_observability.svg; sealed input hashes verified.')
