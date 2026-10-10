import argparse,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','evaluation','status']);a=p.parse_args()
B=Path('execution/visual-v5_3-20261010');remote='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v53'
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','dgx-127-slurm']
if Path('/tmp/pmr-v5-dgx-20261010.sock').exists():ssh[1:1]=['-S','/tmp/pmr-v5-dgx-20261010.sock']
if a.action=='status':
 code='''import json,subprocess;from pathlib import Path;p=Path(REMOTE)
for n in ['fit_controller.json','evaluation_controller.json','ledger/budget.json']:
 f=p/n
 if f.exists():
  d=json.loads(f.read_text());print(n,json.dumps(d if n!='ledger/budget.json' else {k:d[k] for k in ['charged_gpuh','stages']}))
print(subprocess.check_output(['sacct','-j','415453','--format=JobIDRaw,JobName%28,State,ElapsedRaw,ExitCode','-n','-P'],text=True))
for f in sorted((p/'logs').glob('*.log')):
 print(str(f),f.read_text()[-1000:])
'''.replace('REMOTE',repr(remote))
 result=subprocess.run(ssh+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(code))],capture_output=True,text=True);print(result.stdout);print(result.stderr);raise SystemExit(result.returncode)
commit=json.loads((B/'deployment.json').read_text())['source_commit']
args=['/aifs4su/guhao/KVCache/WM/.venv/bin/python',remote+'/code/scripts/launch_v5_3.py',a.action,'--root',remote,'--python','/aifs4su/guhao/KVCache/WM/.venv/bin/python','--assets','/aifs4su/guhao/KVCache/WM/assets','--prior-budget','/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v52/ledger/budget.json','--source-commit',commit,'--allocation-job','415453','--account','hansirui','--partition','llm','--qos','llm']
code='''import json,subprocess,time;from pathlib import Path;p=Path(REMOTE)
assert not (p/(ACTION+'_controller.json')).exists()
with (p/'logs'/(ACTION+'_launcher.log')).open('xb') as f:
 process=subprocess.Popen(ARGS,cwd=p/'code',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(action=ACTION,pid=process.pid,launched_unix=time.time())))
'''.replace('REMOTE',repr(remote)).replace('ACTION',repr(a.action)).replace('ARGS',repr(args))
r=subprocess.run(ssh+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(code))],capture_output=True,text=True);print(r.stdout);assert r.returncode==0,r.stderr
(B/(a.action+'_launch.json')).write_text(r.stdout)
