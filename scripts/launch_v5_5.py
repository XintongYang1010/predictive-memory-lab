#!/usr/bin/env python3
"""One bounded shared-allocation V5.5 extraction; preserve all older scenes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import time
import uuid

BASELINE = 4.522264341248407
ENVELOPE = 1.2
GPU_COUNT = 8
CONTROLLER_SECONDS = 540
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}

# Runs inside this new step before GPU discovery or the scientific process.
# A fresh receipt binds the exact numeric Slurm step to this controller nonce.
STEP_ENTRY = '''import json, os, sys, time
path, expected, nonce = sys.argv[1:4]
job, step = os.environ.get('SLURM_JOB_ID', ''), os.environ.get('SLURM_STEP_ID', '')
if job != expected or not job.isdigit() or not step.isdigit():
    raise RuntimeError('Expected a numeric step inside the requested allocation')
with open(path, 'x') as f:
    json.dump(dict(allocation_job=job, step_id=job+'.'+step, invocation_id=nonce,
                   uid=os.getuid(), created_unix=time.time()), f)
    f.write('\\n'); f.flush(); os.fsync(f.fileno())
os.execvpe(sys.argv[4], sys.argv[4:], os.environ)
'''


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def save(p, value, exclusive=False):
    p = Path(p)
    if p.is_symlink(): raise ValueError('Refuse a symlink output: '+str(p))
    encoded = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if exclusive:
        with p.open('x') as f: f.write(encoded)
        return
    with tempfile.NamedTemporaryFile(mode='w', dir=p.parent, prefix=p.name+'.',
                                     suffix='.tmp', delete=False) as f:
        f.write(encoded); f.flush(); os.fsync(f.fileno()); temporary = f.name
    os.replace(temporary, p)


def checked_root(path):
    # Check the supplied path before resolving, which would hide a symlink.
    path = Path(path).absolute()
    if path.is_symlink(): raise ValueError('New root must not be a symlink')
    root = path.resolve(strict=True)
    if not root.is_dir(): raise ValueError('New root must be an existing directory')
    creation = json.loads((root/'creation.json').read_text()); st = root.stat()
    if any(creation[k] != v for k,v in
           dict(root=str(root), inode=st.st_ino, device=st.st_dev).items()):
        raise ValueError('New root identity differs')
    for name in ('logs', 'ledger', 'gpu_selection'):
        if (root/name).is_symlink(): raise ValueError('Output directory is a symlink: '+name)
    return root


def exact_step(identity, allocation):
    if re.fullmatch(re.escape(allocation)+r'\.[0-9]+', str(identity)) is None:
        raise ValueError('Cancellation requires an exact owned numeric job.step, never a parent job')
    return identity


def validate_lane_publication(root, publication):
    lane = publication.get('lane_reader', {})
    folder = Path(root)/'lane_reader'
    if lane.get('all_files_readback_verified') is not True or folder.is_symlink() or not folder.is_dir():
        raise ValueError('Complete lane model publication/readback required before GPU launch')
    actual = {}
    for path in folder.rglob('*'):
        if path.is_symlink(): raise ValueError('Lane publication contains a symlink')
        if path.is_file(): actual[path.relative_to(folder).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path))
    if (not {'freeze_receipt.json','frozen_model.npz'}.issubset(actual)
            or actual != lane.get('files')):
        raise ValueError('Deployed lane model differs from exact published file inventory')


def receipt_step(path, allocation, invocation):
    if not path.exists(): return None
    if path.is_symlink(): raise ValueError('Owned-step receipt must not be a symlink')
    try: row = json.loads(path.read_text())
    except json.JSONDecodeError: return None  # Writer has not flushed yet.
    if (row.get('invocation_id') != invocation or row.get('allocation_job') != allocation
            or row.get('uid') != os.getuid()):
        raise ValueError('Owned-step receipt identity mismatch')
    return exact_step(row.get('step_id'), allocation)


def query_steps(allocation, job_name):
    """Read allocation state; only the nonce-specific name belongs to this run."""
    evidence = {}; live = {}; terminal = {}; known = set()
    for name,cmd in (
            ('scontrol', ['scontrol','show','step',allocation,'-o']),
            ('sacct', ['sacct','-n','-P','-j',allocation,
                       '--format=JobIDRaw,JobName%128,State,ExitCode'])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            evidence[name] = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            evidence[name] = dict(error=str(exc)); continue
        if r.returncode: continue
        for line in r.stdout.splitlines():
            if name == 'scontrol':
                fields = dict(x.split('=',1) for x in shlex.split(line) if '=' in x)
                if fields.get('Name') != job_name: continue
                if str(fields.get('UserId','')).split('(')[0] != str(os.getuid()):
                    raise ValueError('Nonce-matching step has another owner')
                sid = exact_step(fields.get('StepId'), allocation)
                known.add(sid); state = fields.get('State','').split()[0].split('+')[0]
                if state in TERMINAL: terminal[sid] = dict(state=state, evidence='scontrol')
                else: live[sid] = state
            else:
                cells = line.split('|')
                if len(cells) < 4 or cells[1] != job_name: continue
                sid = exact_step(cells[0], allocation); known.add(sid)
                state = cells[2].split()[0].split('+')[0]
                if state in TERMINAL:
                    terminal[sid] = dict(state=state, exit_code=cells[3], evidence='sacct')
    # Absence is meaningful only after a successful live-state read. Sacct can lag.
    return dict(known=known, live=live, terminal=terminal, evidence=evidence,
                live_query_ok=evidence.get('scontrol',{}).get('returncode') == 0)


def cancel_owned(step_id, allocation):
    sid = exact_step(step_id, allocation)
    # --signal alone can leave the step record alive; ordinary step cancellation
    # asks Slurm to terminate and reap precisely this step, with its normal grace.
    result = subprocess.run(['scancel', sid], capture_output=True, text=True, timeout=10)
    return dict(step_id=sid, returncode=result.returncode, stdout=result.stdout,
                stderr=result.stderr, requested_unix=time.time())


def charge_for(elapsed, confirmed):
    measured = max(0., elapsed)*GPU_COUNT/3600
    # Keep the reservation until exit is confirmed. Never cap an actual overrun.
    return measured if confirmed else max(ENVELOPE, measured)


def monitor(command, root, state, entry, budget, lease, envelope, allocation, invocation, job_name):
    path = root/'extraction_controller.json'; ledger = root/'ledger/budget.json'
    started = time.monotonic(); owned = set(); cancellation = {}; stop = []
    process = None; launched = False; terminal = {}; last_query = 0.; last_save = 0.
    previous = {}
    def request_stop(signum, frame):
        if not stop: stop.append('controller_signal_'+str(signum))
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.signal(signum, request_stop)
    try:
        try:
            process = subprocess.Popen(command, start_new_session=True); launched = True
        except OSError as exc:
            stop.append('launch_failed: '+str(exc)); entry['launch_error'] = str(exc)
        local_term_at = None
        while True:
            now = time.monotonic(); elapsed = now-started
            rc = process.poll() if process is not None else 127
            if elapsed >= CONTROLLER_SECONDS and not stop: stop.append('controller_timeout')
            try:
                sid = receipt_step(root/'owned_step.json', allocation, invocation)
                if sid: owned.add(sid)
            except ValueError as exc:
                if not stop: stop.append('ownership_receipt_failed: '+str(exc))
            if len(owned) > 1 and not stop: stop.append('multiple_steps_for_single_invocation')
            if rc is not None and rc != 0 and not stop: stop.append('srun_failed_'+str(rc))
            confirmed = not launched
            # A normal srun exit still requires the owned Slurm step's terminal
            # state; killing the local srun process alone is never exit evidence.
            if launched and (stop or rc is not None) and now-last_query >= 2.:
                last_query = now
                try:
                    q = query_steps(allocation, job_name)
                    owned.update(q['known']); terminal.update(q['terminal'])
                    entry['last_step_query'] = q['evidence']
                    confirmed = bool(owned) and owned.issubset(terminal) and not q['live'] and q['live_query_ok']
                    if not owned and rc is not None and q['live_query_ok']:
                        # No task launched: srun is gone, nonce-specific live
                        # steps are absent and no early step receipt exists.
                        confirmed = True; entry['no_step_created'] = True
                except (ValueError, OSError) as exc:
                    entry['last_step_query_error'] = str(exc)
                    if not stop: stop.append('step_identity_query_failed')
            if stop and launched and not confirmed:
                for sid in sorted(owned-terminal.keys()):
                    if now-cancellation.get(sid,{}).get('monotonic',float('-inf')) >= 15:
                        try: result = cancel_owned(sid, allocation)
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            result = dict(step_id=sid, error=str(exc), requested_unix=time.time())
                        cancellation[sid] = dict(monotonic=now, **result)
                if not owned and process is not None and rc is None:
                    # Cancel an unallocated/starting local launch. Only this
                    # Popen child is signaled; allocation state is then checked.
                    if local_term_at is None: process.terminate(); local_term_at = now
                    elif now-local_term_at >= 15: process.kill()
            # Queries/cancellation can themselves wait for Slurm. Account that
            # elapsed time too, including the final successful terminal read.
            elapsed = time.monotonic()-started
            charged = charge_for(elapsed, confirmed)
            budget['charged_gpuh'] = BASELINE+charged; budget['stages'][entry['stage']] = charged
            lease.update(charged_gpuh=charged, elapsed_wall_seconds=elapsed,
                         last_accounted_unix=time.time(), step_ids=sorted(owned))
            entry.update(charged_gpuh=charged, step_ids=sorted(owned), terminal_steps=terminal,
                         cancellation_requests=list(cancellation.values()), stop_reasons=stop.copy(),
                         exit_confirmed=confirmed)
            envelope.update(charged_gpuh=charged, step_ids=sorted(owned))
            if charged > ENVELOPE+1e-10:
                entry['envelope_exceeded'] = True
                if not stop: stop.append('envelope_exceeded')
            if confirmed:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
                rc = process.returncode if process is not None else 127
                successful = (rc == 0 and not stop and not entry.get('no_step_created') and
                              charged <= ENVELOPE+1e-10 and
                              all(x['state']=='COMPLETED' and x.get('exit_code','0:0')=='0:0' for x in terminal.values()))
                lease.update(status='closed', end_unix=time.time(), returncode=rc, exit_confirmed=True)
                envelope.update(status='closed', end_unix=time.time(), exit_confirmed=True)
                entry.update(status='complete' if successful else 'failed', returncode=rc,
                             end_unix=time.time(), stop_reasons=stop.copy(),
                             logs=[str(f.relative_to(root)) for f in (root/'logs').glob(entry['stage']+'_*.log')])
                state.update(status='extraction_complete' if successful else 'failed_preserved', end_unix=time.time())
                save(ledger,budget); save(path,state)
                return successful
            if now-last_save >= 2.:
                state['status'] = 'cancelling_owned_step' if stop else 'running'
                lease['status'] = 'unreconciled_timeout' if stop else 'reserved_or_interrupted'
                envelope['status'] = 'unreconciled_timeout' if stop else 'active'
                save(ledger,budget); save(path,state); last_save=now
                if stop: print(json.dumps(dict(status=state['status'], step_ids=sorted(owned),
                                               charged_gpuh=charged, reason=stop)), flush=True)
            time.sleep(.5)
    finally:
        for signum, handler in previous.items(): signal.signal(signum, handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','prior-scene','python','assets'): p.add_argument('--'+n,type=Path,required=True)
    for n in ('source-commit','allocation-job','account','partition','qos'): p.add_argument('--'+n,required=True)
    a=p.parse_args(); root=checked_root(a.root); code=Path(__file__).resolve().parents[1]
    if not a.allocation_job.isdigit(): raise ValueError('A numeric parent allocation is required')
    manifest=json.loads((root/'source_manifest.json').read_text())
    if manifest['commit']!=a.source_commit or any(sha(code/n)!=h for n,h in manifest['files'].items()): raise ValueError('Deployed source mismatch')
    pub=json.loads((root/'registration_publication.json').read_text())
    if not pub.get('verified_readback') or pub['source_commit']!=a.source_commit or pub['plan_sha256']!=sha(code/'scripts/v5_5_plan.json') or pub['protocol_sha256']!=sha(code/'docs/V5_5_MODULE_DIAGNOSTIC.md'): raise ValueError('Source publication not verified')
    validate_lane_publication(root, pub)
    plan=json.loads((code/'scripts/v5_5_plan.json').read_text())
    if sha(a.prior_scene/'archive_manifest.json')!=plan['prior_manifest_sha256']: raise ValueError('Prior sealed scene mismatch')
    prior_manifest=json.loads((a.prior_scene/'archive_manifest.json').read_text())
    prior_ledger=a.prior_scene/'ledger/budget.json'
    if prior_manifest['files']['ledger/budget.json'] != dict(bytes=prior_ledger.stat().st_size,sha256=sha(prior_ledger)):
        raise ValueError('Prior budget bytes differ from the sealed inventory')
    prior=json.loads((a.prior_scene/'ledger/budget.json').read_text())
    if (abs(prior['charged_gpuh']-BASELINE)>1e-10 or BASELINE+ENVELOPE>6 or
            any(x.get('status') not in ('closed','complete','partial') for x in prior.get('leases',[])) or
            any(x.get('status')!='closed' for x in prior.get('controller_envelopes',[]))):
        raise ValueError('Budget mismatch or prior unreconciled work')
    path=root/'extraction_controller.json'; ledger=root/'ledger/budget.json'
    if path.exists() or ledger.exists() or (root/'owned_step.json').exists(): raise FileExistsError('Earlier attempt preserved; no automatic retry')
    ledger.parent.mkdir(exist_ok=True); (root/'logs').mkdir(exist_ok=True)
    invocation=uuid.uuid4().hex; stage='v55_counterfactual'; job_name=stage+'_'+invocation[:16]
    start=time.time()
    state=dict(status='running',pid=os.getpid(),source_commit=a.source_commit,start_unix=start,jobs=[],
               invocation_id=invocation,test_opened=False,p0_authorized=False)
    save(path,state,exclusive=True)
    budget=copy.deepcopy(prior)
    budget.update(charged_gpuh=BASELINE+ENVELOPE,inherited_gpuh=BASELINE,
                  round_cap_gpuh=60,diagnostic_cap_gpuh=6)
    budget.setdefault('stages',{})[stage]=ENVELOPE
    lease=dict(stage=stage,lease_id=invocation,status='reserved_or_interrupted',charged_gpuh=ENVELOPE,
               gpu_count=GPU_COUNT,start_unix=start,
               accounting='Eight selected GPUs times controller wall time until owned-step exit confirmation; never truncate overruns')
    budget.setdefault('leases',[]).append(lease)
    envelope=dict(stage=stage,invocation_id=invocation,status='active',gpuh=ENVELOPE,start_unix=start)
    budget.setdefault('controller_envelopes',[]).append(envelope)
    save(ledger,budget,exclusive=True)
    extract=[str(a.python),'scripts/v5_5_counterfactual.py','--run-root',str(root),'--out',str(root/'counterfactual_cache'),'--prior-scene',str(a.prior_scene),'--prior-manifest-sha256',plan['prior_manifest_sha256'],'--ijepa',str(a.assets/'ijepa'),'--checkpoint',str(a.assets/'IN1K-vit.h.14-300e.pth.tar'),'--source-commit',a.source_commit,'--deadline-unix',str(start+510),'--workers','8','--allocation-job',a.allocation_job,'--account',a.account,'--partition',a.partition,'--qos',a.qos,'--allow-gpu-sharing']
    bind=[str(a.python),'-m','pmr.visual_memory.bind_gpu','--allocation-job',a.allocation_job,'--account',a.account,'--partition',a.partition,'--qos',a.qos,'--shared-allocation-gpus','8','--worker-gpus','8','--allow-gpu-sharing','--min-free-gpu-gib','40','--out',str(root),'--stage',stage,'--']
    command=['srun','--jobid='+a.allocation_job,'--overlap','--exact','--immediate=30','--nodes=1','--ntasks=1','--gres=gpu:8','--cpus-per-task=32','--mem=384G','--time=8','--signal=USR1@30','--kill-on-bad-exit=1','--job-name='+job_name,'--chdir='+str(code),'--output='+str(root/'logs'/(stage+'_%J.log')),'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',str(a.python),'-c',STEP_ENTRY,str(root/'owned_step.json'),a.allocation_job,invocation,*bind,*extract]
    entry=dict(stage=stage,status='running',selected_gpus=8,start_unix=start,command=command,
               allocation_job=a.allocation_job,slurm_job_name=job_name)
    state['jobs'].append(entry); save(path,state); print(shlex.join(command),flush=True)
    if not monitor(command,root,state,entry,budget,lease,envelope,a.allocation_job,invocation,job_name):
        raise RuntimeError('Scientific extraction failed or exceeded its envelope; owned-step exit reconciled, scene preserved, no automatic retry')


if __name__=='__main__': main()
