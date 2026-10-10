#!/usr/bin/env python3
"""Two-phase, bounded V5.4 execution in an existing shared allocation."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from launch_v5_1 import sha, save

GPU_STAGES = ('v54_fit', 'v54_eval')
STAGE_ENVELOPE = 1.2  # Eight devices, six minutes plus three-minute startup margin.
DIAGNOSTIC_CAP = 6.
VERSION_BASELINE = 3.573925619125368
VERSION_CAP = 2.4


def check_budget(ledger, stage):
    if stage not in GPU_STAGES:
        raise ValueError('Unregistered GPU stage')
    if any(v.get('status') != 'closed' for v in ledger.get('controller_envelopes', [])):
        raise RuntimeError('Inspect outstanding controller envelope before a new stage')
    if any(v['status'] == 'reserved_or_interrupted' for v in ledger['leases']):
        raise RuntimeError('Unreconciled GPU lease; preserve and inspect')
    if ledger['stages'].get(stage, 0.) != 0:
        raise RuntimeError('Do not repeat a scientific stage under this registration')
    if ledger['charged_gpuh'] + STAGE_ENVELOPE > min(DIAGNOSTIC_CAP, VERSION_BASELINE+VERSION_CAP) + 1e-8:
        raise RuntimeError('Cannot preserve 54 GPUh within the cumulative 60 GPUh cap')


class Controller:
    def __init__(self, args):
        self.a = args
        self.root = args.root.resolve()
        self.code = Path(__file__).resolve().parents[1]
        self.path = self.root/(args.action+'_controller.json')
        if self.path.exists():
            raise FileExistsError('Preserve the earlier controller and inspect before resuming')
        c = json.loads((self.root/'creation.json').read_text()); st = self.root.stat()
        if c['root'] != str(self.root) or c['inode'] != st.st_ino or c['device'] != st.st_dev:
            raise ValueError('New directory identity changed')
        manifest = json.loads((self.root/'source_manifest.json').read_text())
        if manifest['commit'] != args.source_commit or any(sha(self.code/n) != h for n,h in manifest['files'].items()):
            raise ValueError('Deployed source differs from its frozen manifest')
        inherited=self.root/'ledger/inherited_budget.json'
        if sha(inherited)!=sha(args.prior_budget) or abs(json.loads(inherited.read_text())['charged_gpuh']-VERSION_BASELINE)>1e-10:
            raise ValueError('Prior V5.3 ledger and version baseline differ')
        publication = json.loads((self.root/'registration_publication.json').read_text())
        if (publication.get('verified_readback') is not True or
            publication.get('source_commit') != args.source_commit or
            publication.get('plan_sha256') != sha(self.code/'scripts/v5_4_plan.json') or
            publication.get('protocol_sha256') != sha(self.code/'docs/V5_4_EXECUTION_PROTOCOL.md')):
            raise ValueError('Verified source and protocol publication required')
        self.state = dict(action=args.action, status='running', pid=os.getpid(),
            source_commit=args.source_commit, start_unix=time.time(), jobs=[],
            test_access=False, p0_authorized=False)
        save(self.path,self.state)

    @contextmanager
    def ledger(self):
        with (self.root/'ledger/active.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads((self.root/'ledger/budget.json').read_text())
            yield data
            save(self.root/'ledger/budget.json', data)

    def execute(self, stage, command, gpu=False):
        a = self.a
        env = ['env','OMP_NUM_THREADS='+('1' if gpu else '4'),
               'OPENBLAS_NUM_THREADS='+('1' if gpu else '4'),
               'MKL_NUM_THREADS='+('1' if gpu else '4')]
        if gpu:
            with self.ledger() as ledger:
                check_budget(ledger,stage)
                prior_ids = [v['lease_id'] for v in ledger['leases']]
                ledger.setdefault('controller_envelopes',[]).append(dict(stage=stage,
                    status='active',prior_lease_ids=prior_ids,gpuh=STAGE_ENVELOPE))
            command = [str(a.python),'-m','pmr.visual_memory.bind_gpu',
                '--allocation-job',a.allocation_job,'--account',a.account,
                '--partition',a.partition,'--qos',a.qos,'--shared-allocation-gpus','8',
                '--worker-gpus','8','--allow-gpu-sharing','--min-free-gpu-gib','40',
                '--out',str(self.root),'--stage',stage,'--']+command
        else:
            env += ['CUDA_VISIBLE_DEVICES=']
        cmd = ['srun','--jobid='+a.allocation_job,'--overlap','--exact','--immediate=30',
               '--nodes=1','--ntasks=1','--gres='+('gpu:8' if gpu else 'none'),
               '--cpus-per-task='+('32' if gpu else '8'),'--mem='+('384G' if gpu else '32G'),
               '--time='+('6' if gpu else '20'),'--signal=USR1@60','--kill-on-bad-exit=1',
               '--job-name='+stage,'--chdir='+str(self.code),
               '--output='+str(self.root/'logs'/(stage+'_%J.log')),*env,*command]
        entry = dict(stage=stage,command=cmd,selected_gpus=8 if gpu else 0,
                     status='running',start_unix=time.time(),allocation_job=a.allocation_job)
        self.state['jobs'].append(entry);save(self.path,self.state)
        print(shlex.join(cmd),flush=True)
        timeout = False
        try:
            result = subprocess.run(cmd,timeout=510 if gpu else 1290)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            timeout=True;rc=124
        if gpu:
            with self.ledger() as ledger:
                leased=sum(v['charged_gpuh'] for v in ledger['leases'] if v['lease_id'] not in prior_ids and v['stage']==stage)
                measured=(time.time()-entry['start_unix'])*8/3600
                charged=max(measured,STAGE_ENVELOPE) if timeout else measured
                overhead=max(0.,charged-leased)
                ledger['charged_gpuh']+=overhead
                ledger['stages'][stage]=ledger['stages'].get(stage,0.)+overhead
                ledger.setdefault('overhead_receipts',[]).append(dict(stage=stage,gpu_count=8,
                    charged_gpuh=overhead,controller=a.action,
                    method='srun selected-device wall time beyond stage leases; timeout retains full outer envelope'))
                envelope=next(v for v in ledger['controller_envelopes'] if v['stage']==stage and v['status']=='active')
                envelope['status']='unreconciled_timeout' if timeout else 'closed'
                entry['conservative_overhead_gpuh']=overhead
        entry.update(returncode=rc,status='complete' if rc==0 else 'failed',
            end_unix=time.time(),logs=[str(p.relative_to(self.root)) for p in (self.root/'logs').glob(stage+'_*.log')])
        save(self.path,self.state)
        if rc:
            raise RuntimeError(stage+' failed; preserve logs, reconcile its exact step, no automatic retry')

    def extract(self,stage):
        a=self.a
        cmd=[str(a.python),'scripts/extract_v5_4.py','--stage',stage,
            '--out',str(self.root),'--ijepa',str(a.assets/'ijepa'),
            '--checkpoint',str(a.assets/'IN1K-vit.h.14-300e.pth.tar'),
            '--source-commit',a.source_commit,'--budget-root',str(self.root/'ledger'),
            '--prior-budget',str(a.prior_budget),'--allocation-job',a.allocation_job,
            '--account',a.account,'--partition',a.partition,'--qos',a.qos,
            '--workers','8','--lease-hours','.09','--allow-gpu-sharing',
            '--narrow-scene',str(a.narrow_scene),'--narrow-manifest-sha256',a.narrow_manifest_sha256,
            '--freshness-audit',str(self.root/'freshness_audit.json')]
        if stage!='fit':
            cmd+=['--freeze-receipt',str(self.root/'reader/freeze_receipt.json'),
                  '--freeze-publication',str(self.root/'freeze_publication.json')]
        self.execute('v54_'+stage,cmd,gpu=True)

    def reader(self,role):
        a=self.a
        cmd=[str(a.python),'scripts/v5_4_reader.py',role,
             '--cache',str(self.root/(role+'_cache/cache.npz')),
             '--receipt',str(self.root/(role+'_cache/receipt.json')),
             '--out',str(self.root/('reader' if role=='fit' else role+'_report'))]
        if role!='fit':cmd+=['--model',str(self.root/'reader'),
                            '--publication',str(self.root/'freeze_publication.json')]
        else:cmd+=['--narrow-cache',str(a.narrow_scene/'fit_cache/cache.npz'),
                   '--narrow-receipt',str(a.narrow_scene/'fit_cache/receipt.json'),
                   '--narrow-manifest',str(a.narrow_scene/'archive_manifest.json')]
        self.execute('v54_'+role+'_reader',cmd)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['fit','evaluation'])
    for n in ['root','python','assets','prior-budget','narrow-scene']:p.add_argument('--'+n,type=Path,required=True)
    for n in ['source-commit','allocation-job','account','partition','qos','narrow-manifest-sha256']:p.add_argument('--'+n,required=True)
    a=p.parse_args();c=Controller(a)
    try:
        if a.action=='fit':
            c.extract('fit');c.reader('fit')
            c.state['status']='reader_frozen_awaiting_publication'
        else:
            if not (a.root/'freeze_publication.json').is_file():
                raise ValueError('Publish and verify the frozen models before evaluation')
            c.extract('eval')
            for role in ('anchor','bridge'):
                c.reader(role)
            c.state['status']='repair_diagnostic_complete'
    except BaseException as exc:
        c.state.update(status='execution_failed',reason=str(exc),end_unix=time.time())
        save(c.path,c.state);raise
    c.state['end_unix']=time.time();save(c.path,c.state)


if __name__=='__main__':main()
