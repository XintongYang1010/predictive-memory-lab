import argparse,json,shlex,subprocess,sys
from pathlib import Path
import register_deploy as h
ROOT=h.REMOTE;PY_EXECUTABLE='/aifs4su/guhao/KVCache/WM/.venv/bin/python'
p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','monitor','score']);a=p.parse_args()
if a.action=='launch':
 code='''import json,subprocess,os
from pathlib import Path
root=Path(ROOT);source=json.loads((root/'source_manifest.json').read_text())['commit']
assert not (root/'controller_process.json').exists() and not (root/'extraction_controller.json').exists()
args=[PY_EXECUTABLE,'scripts/launch_v5_5.py','--root',str(root),'--prior-scene',OLD,'--python',PY_EXECUTABLE,'--assets','/aifs4su/guhao/KVCache/WM/assets','--source-commit',source,'--allocation-job','415453','--account','hansirui','--partition','llm','--qos','llm']
with (root/'controller.log').open('x') as f:
 process=subprocess.Popen(args,cwd=root/'code',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
r=dict(pid=process.pid,args=args);(root/'controller_process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
elif a.action=='monitor':
 code='''import json,subprocess
from pathlib import Path
root=Path(ROOT);out={}
for name in ['extraction_controller.json','owned_step.json','ledger/budget.json','score_process.json','report/report.json']:
 p=root/name
 if p.exists():
  value=json.loads(p.read_text());out[name]=value if name!='report/report.json' else {k:value[k] for k in ('status','original_parity','global_vs_lane','pooling')}
for name in ['controller.log','score_controller.log']:
 p=root/name
 if p.exists():out[name]=p.read_text()[-4000:]
workers=root/'counterfactual_cache/workers'
if workers.exists():out['workers']=[{k:v for k,v in json.loads(p.read_text()).items() if k in ('worker','status','completed_pairs','reason')} for p in workers.glob('*.json')]
for p in (root/'logs').glob('*'):
 if p.is_file():out[p.name]=p.read_text()[-2500:]
r=subprocess.run(['sacct','-j','415453','--format=JobIDRaw,JobName%55,State,ExitCode,ElapsedRaw','-n','-P'],capture_output=True,text=True);out['accounting']=r.stdout.splitlines()[-8:]
print(json.dumps(out))
'''
else:
 code='''import json,subprocess,os,shlex
from pathlib import Path
root=Path(ROOT)
state=json.loads((root/'extraction_controller.json').read_text());assert state['status']=='extraction_complete'
assert not (root/'score_process.json').exists() and not (root/'report').exists()
args=['srun','--jobid=415453','--overlap','--exact','--immediate=30','--nodes=1','--ntasks=1','--gres=gpu:0','--cpus-per-task=8','--mem=32G','--time=10','--job-name=v55_score','--chdir='+str(root/'code'),'--output='+str(root/'logs/v55_score_%J.log'),'env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',PY_EXECUTABLE,'scripts/score_v5_5.py','--cache',str(root/'counterfactual_cache'),'--prior-scene',OLD,'--out',str(root/'report'),'--lane-model',str(root/'lane_reader')]
with (root/'score_controller.log').open('x') as f:p=subprocess.Popen(args,cwd=root/'code',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,args=args);(root/'score_process.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
code=code.replace('ROOT',repr(ROOT)).replace('PY_EXECUTABLE',repr(PY_EXECUTABLE)).replace('OLD',repr(h.OLD))
result=h.remote(code);h.save(h.HERE/(a.action+'-'+str(__import__('time').time_ns())+'.json'),result);
if a.action=='monitor':
 brief=dict(status=result.get('extraction_controller.json',{}).get('status'),owned=result.get('owned_step.json'),charged_gpuh=result.get('ledger/budget.json',{}).get('charged_gpuh'),workers=[dict(worker=w.get('worker'),status=w.get('status'),completed=len(w.get('completed_pairs',[])),reason=w.get('reason')) for w in result.get('workers',[])],accounting=result.get('accounting'),report=result.get('report/report.json'))
 brief['logs']={k:v for k,v in result.items() if k.endswith('.log')}
 print(json.dumps(brief,indent=2))
else: print(json.dumps(result,indent=2))
