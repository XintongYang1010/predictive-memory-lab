#!/usr/bin/env python3
"""V5.4 launch/status helper. Preview by default; --execute performs SSH actions."""
import argparse
import json
import shlex
from helper_common import BASE, REMOTE, NARROW, PYTHON, ASSETS, STAGES, frozen_source, remote_python, save_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('fit', 'evaluation', 'status'))
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.action == 'status':
        if not a.execute:
            print('Read-only status of ' + REMOTE + '; add --execute to connect.')
            return
        code = '''import json,subprocess
from pathlib import Path
p=Path(REMOTE)
for n in ['fit_controller.json','evaluation_controller.json','fit_extraction.json','eval_extraction.json','ledger/budget.json']:
 f=p/n
 if f.exists():
  d=json.loads(f.read_text())
  print(n,json.dumps(d if n!='ledger/budget.json' else {k:d.get(k) for k in ['charged_gpuh','stages','controller_envelopes']}))
for name in ['fit_workers','eval_workers']:
 d=p/name
 print(name,sorted(x.name for x in d.glob('*')))
print(subprocess.check_output(['sacct','-j','415453','--format=JobIDRaw,JobName%28,State,ElapsedRaw,ExitCode','-n','-P'],text=True))
for f in sorted((p/'logs').glob('*.log')):print(str(f),f.read_text()[-1000:])
'''.replace('REMOTE', repr(REMOTE))
        remote_python(code)
        return
    identity = frozen_source()
    args = [PYTHON, REMOTE+'/code/scripts/launch_v5_4.py', a.action,
            '--root', REMOTE, '--python', PYTHON, '--assets', ASSETS,
            '--prior-budget', NARROW+'/ledger/budget.json', '--narrow-scene', NARROW,
            '--narrow-manifest-sha256', identity['narrow_manifest_sha256'],
            '--source-commit', identity['source_commit'], '--allocation-job', '415453',
            '--account', 'hansirui', '--partition', 'llm', '--qos', 'llm']
    print(shlex.join(args), flush=True)
    if not a.execute:
        print('Preview only. Expected steps: ' + ', '.join(STAGES[a.action]))
        return
    launch = BASE / (a.action+'_launch.json')
    if launch.exists():
        raise FileExistsError('Preserve previous launch receipt; inspect before any retry')
    code = '''import json,subprocess,time
from pathlib import Path
p=Path(REMOTE);identity=IDENTITY
registration=json.loads((p/'registration_publication.json').read_text())
assert registration.get('verified_readback') is True
assert all(registration.get(k)==identity[k] for k in ('source_commit','plan_sha256','protocol_sha256'))
assert not (p/(ACTION+'_controller.json')).exists()
assert json.loads((p/'source_manifest.json').read_text())['commit']==identity['source_commit']
if ACTION=='evaluation':
 pub=json.loads((p/'freeze_publication.json').read_text())
 assert pub.get('verified_readback') is True and pub.get('union_reference_fully_verified') is True
 assert pub.get('both_fit_receipts_verified') is True and pub.get('source_commit')==identity['source_commit']
with (p/'logs'/(ACTION+'_launcher.log')).open('xb') as f:
 process=subprocess.Popen(ARGS,cwd=p/'code',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(action=ACTION,pid=process.pid,launched_unix=time.time(),source_commit=identity['source_commit'])))
'''.replace('REMOTE', repr(REMOTE)).replace('IDENTITY', repr(identity)).replace('ACTION', repr(a.action)).replace('ARGS', repr(args))
    result = remote_python(code, capture_output=True, text=True)
    record = json.loads(result.stdout)
    save_new(launch, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
