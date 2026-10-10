#!/usr/bin/env python3
"""Durable, gated continuation inside an existing eight-GPU Slurm allocation.

Run FIT, publish its freeze receipt, then run confirmation and the bridge.
Only independently validated success permits the unchanged downstream pilot.
Never cancels the parent allocation or any unrelated process.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


class Controller:
    def __init__(self, a):
        self.a, self.root = a, a.root.resolve()
        self.code = Path(__file__).resolve().parents[1]
        self.guard = threading.Lock()
        self.failure = threading.Event()
        self.status_path = self.root/(a.action+'_controller.json')
        if self.status_path.exists():
            raise FileExistsError('Preserve existing controller receipt; inspect before a new attempt')
        creation = json.loads((self.root/'creation.json').read_text())
        st = self.root.stat()
        if creation['root'] != str(self.root) or creation['inode'] != st.st_ino or creation['device'] != st.st_dev:
            raise ValueError('Continuation directory identity changed')
        source = json.loads((self.root/'source_manifest.json').read_text())
        if source['commit'] != a.source_commit or any(sha(self.code/p) != h for p,h in source['files'].items()):
            raise ValueError('Immutable deployed source does not match its manifest')
        if not (self.root/'ledger/budget.json').is_file():
            raise ValueError('Inherited cumulative budget required')
        self.state = dict(action=a.action, status='running', pid=os.getpid(), source_commit=a.source_commit,
                          start_unix=time.time(), jobs=[], test_access=False)
        save(self.status_path, self.state)

    def execute(self, stage, command, gpu_count=0, gpu_index=3, minutes=20):
        a = self.a
        env = ['env', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4']
        binding = []
        if gpu_count:
            env = ['env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
            selector = [str(a.python), '-m', 'pmr.visual_memory.bind_gpu',
                '--allocation-job', a.allocation_job, '--account', a.account,
                '--partition', a.partition, '--qos', a.qos, '--shared-allocation-gpus', '8',
                '--worker-gpus', str(gpu_count), '--allow-gpu-sharing', '--min-free-gpu-gib', '40',
                '--out', str(self.root), '--stage', stage]
            if gpu_count == 1:
                selector += ['--gpu-index', str(gpu_index)]
                binding = ['--gpu-bind=map_gpu:'+str(gpu_index)]
            command = selector+['--']+command
        else:
            env += ['CUDA_VISIBLE_DEVICES=']
        cmd = ['srun', '--jobid='+a.allocation_job, '--overlap', '--exact', *binding,
            '--nodes=1', '--ntasks=1', '--gres='+('gpu:8' if gpu_count else 'none'),
            '--cpus-per-task='+str(4*gpu_count if gpu_count else 8),
            '--mem='+str(48*gpu_count if gpu_count else 32)+'G', '--time='+str(minutes),
            '--signal=USR1@120', '--kill-on-bad-exit=1', '--job-name=v51_'+stage,
            '--chdir='+str(self.code), '--output='+str(self.root/'logs'/(stage+'_%J.log')),
            *env, *command]
        entry = dict(stage=stage, command=cmd, selected_gpus=gpu_count,
            status='running', start_unix=time.time(), allocation_job=a.allocation_job)
        prior_ids = {v['lease_id'] for v in json.loads((self.root/'ledger/budget.json').read_text())['leases']}
        if gpu_count:
            with open(self.root/'ledger/active.lock', 'a') as ledger_lock:
                fcntl.flock(ledger_lock, fcntl.LOCK_EX)
                ledger = json.loads((self.root/'ledger/budget.json').read_text())
                envelopes = ledger.setdefault('controller_envelopes', [])
                outstanding = 0.
                for item in envelopes:
                    if item['status'] == 'active':
                        used = sum(v['charged_gpuh'] for v in ledger['leases']
                            if v['lease_id'] not in item['prior_lease_ids'] and v['stage'] == item['stage'])
                        outstanding += max(0., item['gpuh']-used)
                envelope = dict(stage=stage, status='active', prior_lease_ids=sorted(prior_ids),
                    gpuh=(minutes/60+.05)*gpu_count)
                stage_limit = (2. if stage.startswith('temporal_') else
                    8. if stage == 'features' else 12. if stage == 'p0' else
                    2. if stage in ('adapt_full','adapt_recent','adapt_fifo') else 4.)
                if ledger['stages'].get(stage, 0.)+envelope['gpuh'] > stage_limit+1e-8:
                    raise RuntimeError('Outer Slurm step cannot fit remaining original stage budget')
                if ledger['charged_gpuh']+outstanding+envelope['gpuh'] > 60:
                    raise RuntimeError('Cumulative budget cannot cover the outer Slurm step and startup margin')
                envelopes.append(envelope)
                save(self.root/'ledger/budget.json', ledger)
        with self.guard:
            self.state['jobs'].append(entry); save(self.status_path, self.state)
        print(shlex.join(cmd), flush=True)
        result = subprocess.run(cmd)
        if gpu_count:
            # Budget leases start inside Python. Conservatively charge wrapper,
            # imports, shutdown and any srun wait as selected-device wall time.
            # Never reduce an interrupted reservation without Slurm evidence.
            with open(self.root/'ledger/active.lock', 'a') as ledger_lock:
                fcntl.flock(ledger_lock, fcntl.LOCK_EX)
                ledger = json.loads((self.root/'ledger/budget.json').read_text())
                leased = sum(v['charged_gpuh'] for v in ledger['leases']
                             if v['lease_id'] not in prior_ids and v['stage'] == stage)
                overhead = max(0., (time.time()-entry['start_unix'])*gpu_count/3600-leased)
                ledger['charged_gpuh'] += overhead
                ledger['stages'][stage] = ledger['stages'].get(stage, 0.)+overhead
                ledger.setdefault('overhead_receipts', []).append(dict(stage=stage,
                    controller=self.a.action, gpu_count=gpu_count, charged_gpuh=overhead,
                    method='max(0, srun wall time times selected GPUs minus new stage leases)'))
                next(v for v in ledger['controller_envelopes'] if v['stage'] == stage and v['status'] == 'active')['status'] = 'closed'
                save(self.root/'ledger/budget.json', ledger)
                entry['conservative_overhead_gpuh'] = overhead
        with self.guard:
            entry.update(returncode=result.returncode, status='complete' if result.returncode == 0 else 'failed',
                end_unix=time.time(), logs=[str(p.relative_to(self.root)) for p in (self.root/'logs').glob(stage+'_*.log')])
            save(self.status_path, self.state)
        if result.returncode:
            self.failure.set()
            raise RuntimeError(stage+' failed; inspect preserved logs and scientific reports')

    def extraction(self, role):
        a = self.a
        cmd = [str(a.python), 'scripts/extract_v5_confirmation.py', '--role', role,
            '--out', str(self.root/(role+'_cache')), '--ijepa', str(a.assets/'ijepa'),
            '--checkpoint', str(a.assets/'IN1K-vit.h.14-300e.pth.tar'),
            '--source-commit', a.source_commit, '--budget-root', str(self.root/'ledger'),
            '--allocation-job', a.allocation_job, '--account', a.account,
            '--partition', a.partition, '--qos', a.qos, '--gpu-model', 'H800',
            '--workers', '8', '--lease-hours', '.20', '--allow-gpu-sharing']
        if role != 'fit':
            cmd += ['--freeze-receipt', str(self.root/'reader/freeze_receipt.json'), '--attest-unseen']
        self.execute('temporal_'+role, cmd, gpu_count=8, minutes=12)

    def reader(self, role):
        out = self.root/('reader' if role == 'fit' else role+'_report')
        cmd = [str(self.a.python), 'scripts/v5_confirmation_reader.py', role,
            '--cache', str(self.root/(role+'_cache')/'cache.npz'),
            '--receipt', str(self.root/(role+'_cache')/'receipt.json'), '--out', str(out)]
        if role != 'fit':
            cmd += ['--model', str(self.root/'reader')]
        self.execute('reader_'+role, cmd)
        if role != 'fit':
            report = json.loads((out/'report.json').read_text())
            return report['primary_metric_conditions_met'] is True
        return True

    def downstream(self, phase, variant=None, gpu_index=3):
        a = self.a
        lease = {'features': .85, 'p0': 11.9, 'adapt': 1.9 if variant in ('full','recent','fifo') else 3.9,
                 'replay': 3.9, 'report': .1}[phase]
        cmd = [str(a.python), 'scripts/run_v5_1_downstream.py', phase,
            '--out', str(self.root/'p0_run'), '--budget-root', str(self.root/'ledger'),
            '--reader-model', str(self.root/'reader'), '--confirmation', str(self.root/'confirm_report'),
            '--bridge', str(self.root/'bridge_report'), '--prior-profile', str(a.prior_profile),
            '--ijepa-code', str(a.assets/'ijepa'), '--ijepa-checkpoint', str(a.assets/'IN1K-vit.h.14-300e.pth.tar'),
            '--backend', 'dgx', '--account', a.account, '--partition', a.partition, '--qos', a.qos,
            '--gpu-model', 'H800', '--allocation-job', a.allocation_job, '--shared-allocation-gpus', '8',
            '--gpu-index', str(gpu_index), '--feature-workers', '8', '--allow-gpu-sharing',
            '--lease-hours', str(lease)]
        if variant:
            cmd += ['--variant', variant]
        stage = phase+('_'+variant if variant else '')
        self.execute(stage, cmd, 0 if phase == 'report' else (8 if phase == 'features' else 1),
                     gpu_index=gpu_index, minutes=math.ceil(lease*60) if phase != 'report' else 20)

    def run(self):
        try:
            if self.a.action == 'fit':
                self.extraction('fit'); self.reader('fit')
                status = 'awaiting_freeze_publication'
            elif self.a.action == 'confirm':
                publication = json.loads((self.root/'freeze_publication.json').read_text())
                if (publication.get('verified_readback') is not True or
                    publication.get('freeze_sha256') != sha(self.root/'reader/freeze_receipt.json') or
                    publication.get('source_commit') != self.a.source_commit):
                    raise ValueError('Verified GitHub freeze publication required before confirmation release')
                for role in ('confirm','bridge'):
                    self.extraction(role)
                    if not self.reader(role):
                        self.state.update(status='scientific_prerequisite_failed', failed_role=role, end_unix=time.time())
                        save(self.status_path, self.state)
                        return
                status = 'temporal_and_bridge_passed'
            else:
                self.downstream('features'); self.downstream('p0')
                # Each lane owns a distinct physical GPU; finish active work on
                # a sibling failure, without touching unrelated processes.
                lanes = [('A0','fifo'), ('A1',), ('B0',), ('B1',), ('C0',), ('C1',), ('full',), ('recent',)]
                def lane(index, variants):
                    for variant in variants:
                        if self.failure.is_set():
                            return
                        try:
                            self.downstream('adapt', variant, index)
                        except BaseException:
                            self.failure.set()
                            raise
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(lane, i, variants) for i, variants in enumerate(lanes)]
                    for future in futures:
                        future.result()
                self.downstream('replay'); self.downstream('report')
                report = json.loads((self.root/'p0_run/report_status.json').read_text())
                manifest = json.loads((self.root/'p0_run/run_manifest.json').read_text())
                if (report.get('status') != 'pilot_complete' or manifest.get('status') != 'complete'
                    or report.get('missing_variants') or report.get('incomplete_stages')):
                    raise RuntimeError('Final report is incomplete; preserve all evidence')
                status = 'pilot_complete'
            self.state.update(status=status, end_unix=time.time())
            save(self.status_path, self.state)
        except BaseException as exc:
            self.state.update(status='execution_failed', reason=str(exc), end_unix=time.time())
            save(self.status_path, self.state)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('fit','confirm','downstream'))
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--python', required=True, type=Path)
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--prior-profile', required=True, type=Path)
    p.add_argument('--allocation-job', required=True)
    p.add_argument('--account', required=True)
    p.add_argument('--partition', required=True)
    p.add_argument('--qos', required=True)
    a = p.parse_args()
    with open(a.root/'controller.lock','a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        Controller(a).run()


if __name__ == '__main__':
    main()
