"""Render reported CPU diagnostics; no fitting or model selection."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
selections=json.loads((root/'core_probe/train_selection.json').read_text())
summary=list(csv.DictReader((root/'core_probe/summary.csv').open()))
colors={'raw':'#326FA8','pooled':'#D07827'}
fig,ax=plt.subplots(figsize=(8,4.8))
for mode in ['raw','pooled']:
    row=selections[f'{mode}/concat/pairs64/train_cv']
    trials=row['trials']
    ax.plot([t['alpha'] for t in trials],[t['mean_score'] for t in trials],
            marker='o',color=colors[mode],label=mode)
ax.axvline(.01,color='#888888',linestyle=':',label='original fixed alpha')
ax.axvline(1,color='#333333',linestyle='--',label='Train-CV selection (both)')
ax.set_xscale('log');ax.set_xlabel('Ridge alpha')
ax.set_ylabel('Inner-Train motion error / Train-mean baseline (lower is better)')
ax.set_title('Four paired folds: regularization selected without Dev scores')
ax.legend(fontsize=9);ax.grid(alpha=.15);fig.tight_layout()
fig.savefig(root/'train_cv_regularization.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
methods=[('concat','fixed','Original concat'),('concat','train_cv','Train-CV concat'),
         ('difference_hybrid','fixed','Fixed frame difference'),
         ('difference_hybrid','train_cv','Train-CV frame difference')]
for ax,mode in zip(axes,['raw','pooled']):
    for i,(method,selection,label) in enumerate(methods):
        row=next(r for r in summary if r['representation']==mode and r['method']==method
                 and r['selection']==selection and r['train_pairs']=='64')
        value=float(row['displacement_r2']);low=float(row['displacement_ci_low']);high=float(row['displacement_ci_high'])
        ax.errorbar(value,i,xerr=[[value-low],[high-value]],fmt='o',color=colors[mode],capsize=4)
        ax.text(high+.015,i,f'{value:.3f}',va='center',fontsize=9)
    ax.axvline(0,color='#888888',linewidth=.8);ax.set_yticks(range(4),[m[2] for m in methods])
    ax.set_xlim(-.48,.43);ax.set_title(mode)
    ax.set_xlabel('Exploratory Dev displacement R²');ax.grid(axis='x',alpha=.15)
axes[0].invert_yaxis()
fig.suptitle('Fixed data, different readout: 95% paired bootstrap intervals')
fig.tight_layout();fig.savefig(root/'probe_comparisons.png',dpi=170);plt.close(fig)
print('Rendered train_cv_regularization.png and probe_comparisons.png')

temporal=json.loads((root/'temporal_controls/diagnostics.json').read_text())['results']
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,mode in zip(axes,['raw','pooled']):
    data=temporal[mode]
    rows=[('Frame 6',data['all_single_frame_controls']['frame6']['metrics']['displacement_r2']),
          ('Frame 7',data['all_single_frame_controls']['frame7']['metrics']['displacement_r2']),
          ('Frames 6 + 7',data['original_metrics']['displacement_r2'])]
    bars=ax.bar(range(3),[r[1] for r in rows],color=colors[mode],alpha=.8)
    ax.set_xticks(range(3),[r[0] for r in rows]);ax.set_ylim(0,.34)
    for bar,(_,v) in zip(bars,rows):
        ax.text(bar.get_x()+bar.get_width()/2,v+.006,f'{v:.3f}',ha='center')
    selected=data['train_selected_single_frame']['selected_frame']
    gain=data['concat_minus_train_selected_single_frame_displacement_r2']
    low,high=data['concat_minus_train_selected_single_frame_ci']
    ax.set_title(f'{mode}: Train selects {selected.replace("frame", "frame ")}')
    ax.set_xlabel(f'Dual minus Train-selected single: {gain:+.3f}\n95% paired CI [{low:+.3f}, {high:+.3f}]',fontsize=10)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Exploratory Dev displacement R²')
fig.suptitle('Matched Train-CV controls: positive R² does not establish temporal use')
fig.tight_layout();fig.savefig(root/'matched_temporal_controls.png',dpi=170);plt.close(fig)
print('Rendered matched_temporal_controls.png')
