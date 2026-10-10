#!/usr/bin/env python3
"""Describe remaining static errors in inspected bridge worlds; never fit."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent/'diagnostic-20261010-v52'
summary = json.loads((HERE/'audit_summary.json').read_text())
worlds = list(csv.DictReader((HERE/'worlds_ranked_by_remaining_static_motion_error.csv').open()))
by_pair = {int(r['pair_id']): r for r in worlds}
with np.load(SCENE/'report/predictions.npz') as z:
    p, y = z['concat'].copy(), z['targets'].astype(np.float64)
with np.load(SCENE/'reader/frozen_model.npz') as z:
    mean = z['concat__target_mean'].copy()
error = (p-y[:, :, None, :])**2
strata = []
for name, ids in (
    ('all', np.arange(256)),
    ('any_reflection', np.array([i for i in range(256) if by_pair[i+128]['any_cue_reflection']=='True'])),
    ('no_reflection', np.array([i for i in range(256) if by_pair[i+128]['any_cue_reflection']=='False'])),
    ('outside_FIT_position_box', np.array([i for i in range(256) if by_pair[i+128]['position_outside_fit_box_max']=='True'])),
    ('inside_FIT_position_box', np.array([i for i in range(256) if by_pair[i+128]['position_outside_fit_box_max']=='False'])),
):
    den = ((y[ids]-mean)**2).mean((0, 1))
    for label, mse in (('original', error[ids, :, 0].mean((0, 1))),
                       ('mean_two_static_losses', error[ids, :, 1:3].mean((0, 1, 2)))):
        skill = 1-mse/den
        strata.append(dict(stratum=name, pairs=len(ids), condition=label,
                           motion_mse=float(mse[4:].mean()), motion_skill=float(skill[4:].mean()),
                           dx_mse=float(mse[[4, 6]].mean()), A_dx_skill=float(skill[4]), B_dx_skill=float(skill[6])))
with (HERE/'remaining_static_strata.csv').open('x', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(strata[0]));w.writeheader();w.writerows(strata)
top = [int(r['pair_id']) for r in worlds[:10]]
cases=[]
for pair in top:
    i=pair-128
    for variant in (0,1):
        for c, condition in enumerate(('original','static_t6','static_t7','swap_nuisance')):
            row=dict(pair=pair,variant=variant,condition=condition)
            for d,axis in enumerate(('A_x','A_y','B_x','B_y','A_dx','A_dy','B_dx','B_dy')):
                row[axis+'_target']=float(y[i,variant,d]);row[axis+'_prediction']=float(p[i,variant,c,d])
            cases.append(row)
with (HERE/'worst_ten_static_predictions.csv').open('x', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
fig,axs=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True)
for row,c in enumerate((1,2)):
    for col,axis in enumerate(('A_dx','A_dy','B_dx','B_dy'),4):
        ax=axs[row,col-4]
        ax.scatter(y[:,:,col].ravel(),p[:,:,c,col].ravel(),s=9,alpha=.45,color='#157c80')
        lo,hi=-.05,.05
        ax.plot([lo,hi],[lo,hi],color='#ad6633',lw=1)
        ax.axhline(0,lw=.4,color='gray');ax.axvline(0,lw=.4,color='gray')
        ax.set(title=f'{axis}; static at t{5+c}',xlabel='Target',ylabel='Prediction',xlim=(lo,hi),ylim=(lo,hi))
fig.suptitle('Posthoc: residual error under both fixed static interventions\n512 existing bridge episodes; no prediction averaging, fitting, or new gate')
fig.tight_layout();fig.savefig(HERE/'static_remaining_predictions.png',dpi=180);plt.close(fig)
old_worst=[]
for pair in (380,334,356):
    row=dict(by_pair[pair]);row['rank_remaining_static_motion']=next(i+1 for i,r in enumerate(worlds) if int(r['pair_id'])==pair)
    row['horizontal_error_reduction_fraction']=1-float(row['static_average_dx_mse'])/float(row['original_dx_mse'])
    old_worst.append(row)
with (HERE/'old_worst_case_followup.json').open('x') as f:
    json.dump(old_worst,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(strata=strata,previous_worst=old_worst),indent=2))
