#!/usr/bin/env python3
"""Bounded batch-shape validation; preserve the first V5.5 extraction."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import time
import uuid
import launch_v5_5 as controller

ENVELOPE=.45

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','prior-run','python','assets'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    a=p.parse_args();root=controller.checked_root(a.run_root);prior=a.prior_run.resolve();code=Path(__file__).resolve().parents[1]
    old=json.loads((prior/'extraction_controller.json').read_text())
    if json.loads((prior/'counterfactual_cache/manifest.json').read_text()).get('status')!='complete':raise ValueError('First cache is incomplete')
    budget_path=prior/'ledger/budget.json';base=json.loads(budget_path.read_text())
    pub=json.loads((root/'registration_publication.json').read_text());manifest=json.loads((root/'source_manifest.json').read_text())
    if old['status']!='extraction_complete' or base['leases'][-1]['status']!='closed' or base['controller_envelopes'][-1]['status']!='closed':raise ValueError('First extraction is not closed')
    used=base['charged_gpuh']-4.522264341248407;remaining=1.2-used
    if remaining<ENVELOPE or base['charged_gpuh']+ENVELOPE>6:raise ValueError('Insufficient remaining registered budget; do not launch')
    expected=dict(source_commit=a.source_commit,first_cache_manifest_sha256=controller.sha(prior/'counterfactual_cache/manifest.json'),first_ledger_sha256=controller.sha(budget_path),addendum_sha256=controller.sha(code/'docs/V5_5_BATCH_MATCH_ADDENDUM.md'))
    if (pub.get('verified_readback') is not True or any(pub.get(k)!=v for k,v in expected.items()) or abs(pub['remaining_gpuh']-remaining)>1e-12):raise ValueError('Batch validation publication differs')
    if manifest['commit']!=a.source_commit or any(controller.sha(code/name)!=h for name,h in manifest['files'].items()):raise ValueError('New published source differs')
    controller.validate_lane_publication(prior,json.loads((prior/'registration_publication.json').read_text()))
    path=root/'extraction_controller.json';ledger=root/'ledger/budget.json'
    if any(p.exists() for p in (path,ledger,root/'owned_step.json')):raise FileExistsError('Previous validation attempt preserved')
    (root/'logs').mkdir(exist_ok=True);(root/'ledger').mkdir(exist_ok=True)
    nonce=uuid.uuid4().hex;stage='v55_batch4';job_name=stage+'_'+nonce[:16];start=time.time()
    controller.BASELINE=base['charged_gpuh'];controller.ENVELOPE=ENVELOPE;controller.CONTROLLER_SECONDS=198
    budget=copy.deepcopy(base);budget['inherited_gpuh']=base['charged_gpuh'];budget['charged_gpuh']=base['charged_gpuh']+ENVELOPE;budget['stages'][stage]=ENVELOPE
    lease=dict(stage=stage,lease_id=nonce,status='reserved_or_interrupted',charged_gpuh=ENVELOPE,gpu_count=8,start_unix=start)
    envelope=dict(stage=stage,invocation_id=nonce,status='active',gpuh=ENVELOPE,start_unix=start)
    budget['leases'].append(lease);budget['controller_envelopes'].append(envelope)
    state=dict(status='running',pid=os.getpid(),source_commit=a.source_commit,start_unix=start,jobs=[],invocation_id=nonce,test_opened=False,p0_authorized=False)
    extract=[str(a.python),'scripts/v5_5_batch_match.py','extract','--run-root',str(root),'--out',str(root/'cache'),'--prior-run',str(prior),'--ijepa',str(a.assets/'ijepa'),'--checkpoint',str(a.assets/'IN1K-vit.h.14-300e.pth.tar'),'--source-commit',a.source_commit,'--deadline-unix',str(start+165),'--workers','8','--allocation-job','415453','--account','hansirui','--partition','llm','--qos','llm','--allow-gpu-sharing']
    bind=[str(a.python),'-m','pmr.visual_memory.bind_gpu','--allocation-job','415453','--account','hansirui','--partition','llm','--qos','llm','--shared-allocation-gpus','8','--worker-gpus','8','--allow-gpu-sharing','--min-free-gpu-gib','40','--out',str(root),'--stage',stage,'--']
    command=['srun','--jobid=415453','--overlap','--exact','--immediate=30','--nodes=1','--ntasks=1','--gres=gpu:8','--cpus-per-task=32','--mem=384G','--time=3','--signal=USR1@15','--kill-on-bad-exit=1','--job-name='+job_name,'--chdir='+str(code),'--output='+str(root/'logs'/(stage+'_%J.log')),'env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',str(a.python),'-c',controller.STEP_ENTRY,str(root/'owned_step.json'),'415453',nonce,*bind,*extract]
    entry=dict(stage=stage,status='running',selected_gpus=8,start_unix=start,command=command,allocation_job='415453',slurm_job_name=job_name)
    state['jobs'].append(entry);controller.save(path,state,exclusive=True);controller.save(ledger,budget,exclusive=True)
    print(shlex.join(command),flush=True)
    if not controller.monitor(command,root,state,entry,budget,lease,envelope,'415453',nonce,job_name):raise RuntimeError('Validation failed; scene preserved, no retry')

if __name__=='__main__':main()
