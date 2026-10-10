import json,time
from helper_common import BASE,REMOTE,remote_python
code='''import json,subprocess
from pathlib import Path
p=Path(REMOTE);result={}
for action in ['fit','evaluation']:
 f=p/(action+'_controller.json')
 if f.exists():
  c=json.loads(f.read_text());result[action]={'status':c['status'],'reason':c.get('reason'),'jobs':[{'stage':j['stage'],'status':j['status'],'returncode':j.get('returncode'),'logs':j.get('logs',[])} for j in c['jobs']]}
for stage in ['fit','eval']:
 w=[json.loads(x.read_text()) for x in (p/(stage+'_workers')).glob('*.json')]
 result[stage+'_workers']=[{k:d.get(k) for k in ['worker','status','job_id','step_id']}|{'completed':len(d.get('completed_units',[])),'assigned':len(d['assigned_units'])} for d in w]
b=json.loads((p/'ledger/budget.json').read_text());result['ledger']={'charged_gpuh':b['charged_gpuh'],'new_stages':{k:v for k,v in b['stages'].items() if k.startswith('v54_')}}
rows=subprocess.check_output(['sacct','-j','415453','--format=JobIDRaw,JobName%30,State,ElapsedRaw,ExitCode','-n','-P'],text=True)
result['sacct']=[x for x in rows.splitlines() if '|v54_' in x or x.split('|')[0] in ['415453','415453.5']]
print(json.dumps(result))
'''.replace('REMOTE',repr(REMOTE))
r=remote_python(code,capture_output=True,text=True);data=json.loads(r.stdout)
path=BASE/('monitor-'+str(int(time.time()))+'.json');path.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
