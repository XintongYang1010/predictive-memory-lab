import argparse,json,time
import register_deploy as h
p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','monitor','score']);a=p.parse_args()
pre="""import json,os,subprocess
from pathlib import Path
prior=Path(PRIOR);root=prior/'batch_match_run';runtime='/aifs4su/guhao/KVCache/WM/.venv/bin/python'
""".replace('PRIOR',repr(h.REMOTE))
if a.action=='launch':
 code=pre+"""assert not (root/'controller_process.json').exists()
source=json.loads((root/'source_manifest.json').read_text())['commit']
args=[runtime,'scripts/launch_v5_5_batch_match.py','--run-root',str(root),'--prior-run',str(prior),'--python',runtime,'--assets','/aifs4su/guhao/KVCache/WM/assets','--source-commit',source]
with (root/'controller.log').open('x') as f:p=subprocess.Popen(args,cwd=prior/'code',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
r=dict(pid=p.pid,args=args);(root/'controller_process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
"""
elif a.action=='score':
 code=pre+"""assert json.loads((root/'extraction_controller.json').read_text())['status']=='extraction_complete'
assert not (root/'score_process.json').exists() and not (root/'comparison').exists()
args=['srun','--jobid=415453','--overlap','--exact','--immediate=30','--nodes=1','--ntasks=1','--gres=gpu:0','--cpus-per-task=8','--mem=32G','--time=10','--job-name=v55_batch_score','--chdir='+str(prior/'code'),'--output='+str(root/'logs/v55_batch_score_%J.log'),'env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',runtime,'scripts/v5_5_batch_match.py','compare','--cache',str(root/'cache'),'--prior-run',str(prior),'--frozen-model','/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v54/reader','--lane-model',str(prior/'lane_reader'),'--out',str(root/'comparison')]
with (root/'score_controller.log').open('x') as f:p=subprocess.Popen(args,cwd=prior/'code',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,args=args);(root/'score_process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
"""
else:
 code=pre+"""r={}
for name in ('owned_step.json','extraction_controller.json','ledger/budget.json','cache/manifest.json','comparison/report.json'):
 p=root/name
 if p.exists():
  v=json.loads(p.read_text())
  if name=='extraction_controller.json':v={k:v[k] for k in ('status','jobs')}
  if name=='ledger/budget.json':v=dict(charged_gpuh=v['charged_gpuh'],last_lease=v['leases'][-1])
  if name=='cache/manifest.json':v={k:v[k] for k in ('status','reason') if k in v}
  r[name]=v
r['workers']=[{k:v for k,v in json.loads(p.read_text()).items() if k in ('worker','status','completed_pairs','reason')} for p in (root/'cache/workers').glob('*.json')]
r['logs']={p.name:p.read_text()[-2000:] for p in (root/'logs').glob('*') if p.is_file()}
if (root/'controller.log').exists():r['controller_tail']=(root/'controller.log').read_text()[-2000:]
r['accounting']=subprocess.check_output(['sacct','-j','415453','--format=JobIDRaw,JobName%55,State,ExitCode,ElapsedRaw','-n','-P'],text=True).splitlines()[-5:]
print(json.dumps(r))
"""
r=h.remote(code);h.save(h.HERE/('batch-'+a.action+'-'+str(time.time_ns())+'.json'),r)
if a.action=='monitor':
 for worker in r['workers']:worker['completed_pairs']=len(worker.get('completed_pairs',[]))
 for job in r.get('extraction_controller.json',{}).get('jobs',[]):job.pop('command',None)
print(json.dumps(r,indent=2))
