#!/usr/bin/env python3
"""Summarize the existing BR-0.2 accounting; no model execution or fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    rows=json.loads((a.input/'parents.json').read_text());summary=json.loads((a.input/'summary.json').read_text())
    output={}
    for method in sorted({r['method'] for r in rows}):
        for cond in ('EXACT','NEWPATH','CHANGE','ALL'):
            selected=[r for r in rows if r['method']==method and (cond=='ALL' or r['condition']==cond)]
            ids=sorted({r['source_index'] for r in selected})
            den=np.array([sum(r['future_den'] for r in selected if r['source_index']==i) for i in ids])
            draws=np.random.default_rng(17).integers(0,len(ids),(2000,len(ids)))
            group={}
            for label in ('chosen_minus_oracle','chosen_minus_zero'):
                parts={}
                for part in ('signal','noise','cross','total'):
                    num=np.array([sum(r[label][part]*r['future_den'] for r in selected if r['source_index']==i) for i in ids])
                    boot=num[draws].sum(1)/den[draws].sum(1)
                    parts[part]={'mean':float(num.sum()/den.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
                group[label]=parts
            output[f'{method}/{cond}']=group
    positive=[r for r in rows if r['positive_task_regret']]
    extra={'bootstrap':{'seed':17,'resamples':2000,'unit':'8 original A source indices; repeated anchors/policies are not independent',
                        'limitation':'Adjacent CHANGE B sources overlap within development. Intervals are descriptive, not confirmation.'},
           'groups':output,
           'parent_coverage':{'positive_regret':len(positive),
               'oracle_decreases_scored_coverage':sum(r['candidate_scored_coverage'][r['oracle_drop']] < r['candidate_scored_coverage'][r['chosen_drop']]-1e-8 for r in positive),
               'oracle_increases_scored_coverage':sum(r['candidate_scored_coverage'][r['oracle_drop']] > r['candidate_scored_coverage'][r['chosen_drop']]+1e-8 for r in positive)},
           'exact_state_audit_clarification':{'unique_full_available_prefixes':len({r['full_available_past_sha256'] for r in rows}),
               'parent_states':len({r['bounded_past_sha256'] for r in rows}),
               'interpretation':'96 causal prefixes are each evaluated under nine memory states. All 864 candidate/recent states are distinct, so there are no matched-input competing-future examples in this archive. This is missing identifiability evidence, not evidence of learnability.'}}
    (a.input/'connection_statistics.json').write_text(json.dumps(extra,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    colors={'signal':'#4c78a8','noise':'#e08a45','cross':'#8c6bb1'}
    top=json.loads((a.input/'previous_top_case.json').read_text())
    parts=top['chosen_minus_oracle']
    axs[0].bar(list(colors),[parts[x] for x in colors],color=list(colors.values()))
    for j,name in enumerate(colors):axs[0].text(j,parts[name]+.025,f'{parts[name]:.3f}',ha='center')
    axs[0].set_title('Existing K16 / PaFu / CHANGE case\nTotal local regret = 2.30755')
    axs[0].set_ylabel('Chosen loss minus same-parent oracle')
    axs[0].set_ylim(0,1.85)
    methods=['FIFO','WR','PaFu'];x=np.arange(3)
    for j,part in enumerate(colors):
        value=[summary['groups'][f'K64/{m}/ALL']['chosen_minus_oracle'][part] for m in methods]
        axs[1].bar(x+(j-1)*.24,value,width=.23,label=part,color=colors[part])
    axs[1].set_xticks(x,methods);axs[1].axhline(0,color='black',lw=.7)
    axs[1].set_title('K64: all existing frozen parents\nSignal / realized noise / cross components')
    axs[1].legend(frameon=False)
    fig.suptitle('Why the existing residual-bank oracle deletes an attended item')
    fig.text(.5,-.035,'U is a pre-injection slow-model residual, not physical ground truth.\nThese are retrospective same-capacity parent comparisons, not online policy gains.',ha='center',fontsize=9,color='#555555')
    fig.savefig(a.input/'connection_components.png',dpi=170,bbox_inches='tight')
    fig.savefig(a.input/'connection_components.pdf',bbox_inches='tight')
    plt.close(fig)
    (a.input/'presentation_manifest.json').write_text(json.dumps({'runner_sha256':sha(__file__),
        'input_parent_sha256':sha(a.input/'parents.json'),'input_summary_sha256':sha(a.input/'summary.json'),
        'files':{name:{'sha256':sha(a.input/name),'bytes':(a.input/name).stat().st_size} for name in
            ('connection_statistics.json','connection_components.png','connection_components.pdf')},
        'new_model_calls':0,'new_training_steps':0},indent=2)+'\n')


if __name__=='__main__':main()
