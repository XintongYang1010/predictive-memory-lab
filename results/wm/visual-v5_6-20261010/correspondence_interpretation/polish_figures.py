#!/usr/bin/env python3
"""Display-only title wrapping; preserve analysis.py and all numeric outputs."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib.axes
import analyze as base

def main():
    p=argparse.ArgumentParser();p.add_argument('--correspondence',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out
    if (out/'presentation_revision.json').exists():raise FileExistsError('Preserve completed display revision')
    analysis=json.loads((out/'analysis.json').read_text())
    if base.sha(out/'analyze.py')!=analysis['provenance']['script_sha256']:raise ValueError('Numeric analysis source changed')
    hashes={(r['pair_id'],r['variant'],r['cue_time']):r['rgb'] for r in json.loads((out/'rgb_hash_readback.json').read_text())}
    with np.load(a.correspondence/'results/costs_and_predictions.npz',allow_pickle=False) as z:
        pred=z['predictions'];truth=z['targets'];cost=z['costs']
    rows=list(csv.DictReader((a.correspondence/'results/per_episode_object.csv').open()))
    original_title=matplotlib.axes.Axes.set_title
    def wrapped(self,label,*args,**kwargs):
        if 'pred (' in label:label=label.replace(', gap ','\ngap ')
        return original_title(self,label,*args,**kwargs)
    matplotlib.axes.Axes.set_title=wrapped
    before={};after={}
    for pair,variant in base.CASES:
        pos,colors,yellow=base.cue_geometry('dev',pair,variant);images=[]
        for t in (0,1):
            rgb,_=base.render_endpoint(pos[t],colors,yellow[t])
            if base.entry_identity(rgb)!=hashes[pair,variant,6+t]:raise ValueError('RGB changed')
            images.append(rgb)
        for suffix in ('png','svg'):
            name=f'case_{pair}_{variant}.{suffix}';before[name]=base.sha(out/name)
        idx=(pair-640)*2+variant
        rr={(r['method'],r['object']):r for r in rows if int(r['pair_id'])==pair and int(r['variant'])==variant}
        base.plot_case(out,pair,variant,images,pred[idx],truth[idx],cost[idx],rr)
        for suffix in ('png','svg'):
            name=f'case_{pair}_{variant}.{suffix}';after[name]=base.sha(out/name)
    base.write(out/'presentation_revision.json',dict(display_only=True,change='Wrap cost gap onto third title line; all panels/data unchanged',
        script_sha256=base.sha(__file__),analysis_json_sha256=base.sha(out/'analysis.json'),before=before,after=after))

if __name__=='__main__':main()
