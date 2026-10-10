#!/usr/bin/env python3
"""Two bounded eight-GPU steps; use authenticated V5.5 ownership/reaping logic."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import uuid
import launch_v5_5 as control

BASELINE=5.32908766817043
ENVELOPE=.65

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit-axis','confirm'])
    for name in ('root','prior-v54','prior-v55','python','assets'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-commit','allocation-job','account','partition','qos'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=control.checked_root(a.root);code=Path(__file__).resolve().parents[1]
    source=json.loads((root/'source_manifest.json').read_text());pub=json.loads((root/'registration_publication.json').read_text())
    if (source['commit']!=a.source_commit or pub.get('verified_readback') is not True or
        pub['source_commit']!=a.source_commit or any(control.sha(code/n)!=h for n,h in source['files'].items())):
        raise ValueError('Source identity/readback mismatch')
    seal=json.loads((a.prior_v55/'archive_manifest.json').read_text())
    if control.sha(a.prior_v55/'archive_manifest.json')!='b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663':
        raise ValueError('Wrong sealed V5.5 scene')
    prior_path=a.prior_v55/'batch_match_run/ledger/budget.json'
    if seal['files']['batch_match_run/ledger/budget.json']!={'bytes':prior_path.stat().st_size,'sha256':control.sha(prior_path)}:
        raise ValueError('Final V5.5 supplementary ledger membership')
    prior=json.loads(prior_path.read_text())
    if abs(prior['charged_gpuh']-BASELINE)>1e-10:raise ValueError('Prior budget mismatch')
    stage_root=root/'stages'/a.stage
    if stage_root.exists():raise FileExistsError('No implicit retry; preserve stage')
    if a.stage=='fit-axis':
        budget=copy.deepcopy(prior);before=BASELINE
        if (root/'ledger/budget.json').exists():raise FileExistsError('Existing current ledger')
    else:
        first=json.loads((root/'stages/fit-axis/extraction_controller.json').read_text())
        if first['status']!='extraction_complete':raise ValueError('Previous GPU extraction not successful')
        budget=json.loads((root/'ledger/budget.json').read_text());before=budget['charged_gpuh']
        if any(x.get('status') not in ('closed','complete','partial') for x in budget.get('leases',[])):
            raise ValueError('Unreconciled lease')
        if not (root/'freeze_publication.json').exists():raise ValueError('Published reader freeze required')
    remaining=BASELINE+ENVELOPE-before
    if remaining<=.1:raise RuntimeError('Insufficient bounded envelope for another eight-GPU load')
    for folder in (stage_root/'logs',stage_root/'ledger',root/'ledger'):folder.mkdir(parents=True,exist_ok=True)
    # Old monitor remains byte-identical. These explicit run parameters replace
    # its previous experiment's constants, never prior scientific artifacts.
    control.BASELINE=before;control.ENVELOPE=remaining
    control.CONTROLLER_SECONDS=max(1,math.floor(remaining*450)-20)
    start=time.time();inv=uuid.uuid4().hex;name='v56_'+a.stage.replace('-','_')+'_'+inv[:16]
    state=dict(status='running',stage=a.stage,pid=os.getpid(),source_commit=a.source_commit,start_unix=start,jobs=[])
    lease=dict(stage='v56_'+a.stage,lease_id=inv,status='reserved_or_interrupted',charged_gpuh=remaining,
        gpu_count=8,start_unix=start,accounting='eight selected GPUs times controller elapsed through confirmed exact-step exit')
    env=dict(stage='v56_'+a.stage,invocation_id=inv,status='active',gpuh=remaining,start_unix=start)
    budget.setdefault('leases',[]).append(lease);budget.setdefault('controller_envelopes',[]).append(env)
    budget.update(charged_gpuh=before+remaining,inherited_gpuh=BASELINE,diagnostic_cap_gpuh=6,round_cap_gpuh=60)
    control.save(root/'ledger/budget.json',budget)
    extract=[str(a.python),'scripts/v5_6_extract.py',a.stage,'--run-root',str(root),
        '--prior-v54',str(a.prior_v54),'--prior-v55',str(a.prior_v55),'--ijepa',str(a.assets/'ijepa'),
        '--checkpoint',str(a.assets/'IN1K-vit.h.14-300e.pth.tar'),'--source-commit',a.source_commit,
        '--deadline-unix',str(start+control.CONTROLLER_SECONDS-20),'--workers','8',
        '--allocation-job',a.allocation_job,'--account',a.account,'--partition',a.partition,'--qos',a.qos,'--allow-gpu-sharing']
    bind=[str(a.python),'-m','pmr.visual_memory.bind_gpu','--allocation-job',a.allocation_job,
        '--account',a.account,'--partition',a.partition,'--qos',a.qos,'--shared-allocation-gpus','8',
        '--worker-gpus','8','--allow-gpu-sharing','--min-free-gpu-gib','40','--out',str(root),'--stage',name,'--']
    command=['srun','--jobid='+a.allocation_job,'--overlap','--exact','--immediate=30','--nodes=1','--ntasks=1',
        '--gres=gpu:8','--cpus-per-task=32','--mem=384G','--time='+str(math.ceil(control.CONTROLLER_SECONDS/60)),
        '--signal=USR1@30','--kill-on-bad-exit=1','--job-name='+name,'--chdir='+str(code),
        '--output='+str(stage_root/'logs'/('v56_'+a.stage+'_%J.log')),'env','PYTHONDONTWRITEBYTECODE=1',
        'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',str(a.python),'-c',control.STEP_ENTRY,
        str(stage_root/'owned_step.json'),a.allocation_job,inv,*bind,*extract]
    entry=dict(stage='v56_'+a.stage,status='running',selected_gpus=8,start_unix=start,command=command,
        allocation_job=a.allocation_job,slurm_job_name=name);state['jobs'].append(entry)
    control.save(stage_root/'extraction_controller.json',state,exclusive=True)
    ok=control.monitor(command,stage_root,state,entry,budget,lease,env,a.allocation_job,inv,name)
    control.save(root/'ledger/budget.json',budget)
    if not ok:raise RuntimeError('Stage failed or exceeded envelope; owned step reconciled, scene preserved')
    print(json.dumps(dict(stage=a.stage,status='complete',cumulative_gpuh=budget['charged_gpuh'],steps=entry['step_ids'])),flush=True)

if __name__=='__main__':main()
