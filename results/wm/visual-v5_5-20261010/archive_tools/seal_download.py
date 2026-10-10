import hashlib,io,json,re,shlex,subprocess,tarfile,time
from pathlib import Path
import register_deploy as h
# Validate all completed work before writing only this new root's seal.
owner=json.loads((h.HERE/'creation.json').read_text())
child_owner=json.loads((h.HERE/'batch_deployment.json').read_text())['creation']
code='''import hashlib,json,math,re,stat,subprocess,time
from pathlib import Path
root=Path(REMOTE);old=Path(OLD);child=root/'batch_match_run'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def inventory(folder):
 result={};device=folder.stat().st_dev
 for p in sorted(folder.rglob('*')):
  st=p.lstat();assert not p.is_symlink() and st.st_dev==device,str(p)
  assert stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode),str(p)
  if stat.S_ISREG(st.st_mode) and p!=folder/'archive_manifest.json':result[p.relative_to(folder).as_posix()]=dict(bytes=st.st_size,sha256=sha(p))
 return result
def owned(folder,expected):
 assert not folder.is_symlink() and not folder.parent.is_symlink()
 st=folder.stat();actual=read(folder/'creation.json')
 assert actual==expected and actual['root']==str(folder) and actual['preexisted'] is False
 assert (st.st_ino,st.st_dev,st.st_uid)==(actual['inode'],actual['device'],actual['uid'])
owned(root,OWNER);owned(child,CHILD_OWNER)
assert root!=old and not old.is_symlink()
assert not (root/'archive_manifest.json').exists() and not (root/'final_accounting.json').exists()
prior_sha='ec7bc9c17da8417e9ecd11ba0f21415a2df0ee7af620af9d12f911c3ab26e75c'
assert sha(old/'archive_manifest.json')==prior_sha
prior=read(old/'archive_manifest.json');assert inventory(old)==prior['files'],'Prior scene changed'
baseline=4.522264341248407;prior_budget=read(old/'ledger/budget.json')
assert abs(prior_budget['charged_gpuh']-baseline)<1e-12
states=[read(folder/'extraction_controller.json') for folder in (root,child)]
budgets=[read(folder/'ledger/budget.json') for folder in (root,child)]
steps=[];previous=prior_budget;charges=[]
for folder,state,budget,cap,score_prefix in zip((root,child),states,budgets,(1.2,.45),('v55_score_','v55_batch_score_')):
 assert state['status']=='extraction_complete' and len(state['jobs'])==1
 assert state['test_opened'] is False and state['p0_authorized'] is False
 job=state['jobs'][0];assert job['status']=='complete' and job['returncode']==0 and job['exit_confirmed'] is True
 assert job['selected_gpus']==8 and not job['stop_reasons'] and not job['cancellation_requests']
 assert len(job['step_ids'])==1
 assert all(re.fullmatch(r'415453\.\d+',step) for step in job['step_ids'])
 assert all(v.get('status') in ('closed','complete','partial') for v in budget['leases'])
 assert all(v.get('status')=='closed' for v in budget['controller_envelopes'])
 assert budget['leases'][:-1]==previous['leases'] and budget['controller_envelopes'][:-1]==previous['controller_envelopes']
 for key in ('overhead_receipts','prior_step_overhead_reconciliation'):
  assert budget.get(key)==previous.get(key),'Inherited accounting changed'
 for key,value in previous['stages'].items():assert budget['stages'][key]==value
 lease=budget['leases'][-1];env=budget['controller_envelopes'][-1]
 assert lease['lease_id']==env['invocation_id']==state['invocation_id']
 assert lease['stage']==env['stage']==job['stage']
 assert lease['status']=='closed' and lease['exit_confirmed'] is True and lease['returncode']==0
 assert env['exit_confirmed'] is True and env['gpuh']==cap
 charge=job['charged_gpuh'];assert math.isfinite(charge) and 0<=charge<=cap+1e-12
 assert abs(charge-lease['elapsed_wall_seconds']*8/3600)<1e-12
 assert all(abs(v-charge)<1e-12 for v in (lease['charged_gpuh'],env['charged_gpuh'],budget['stages'][job['stage']]))
 assert abs(budget['inherited_gpuh']-previous['charged_gpuh'])<1e-12
 assert abs(budget['charged_gpuh']-previous['charged_gpuh']-charge)<1e-12
 assert lease['step_ids']==env['step_ids']==job['step_ids']
 charges.append(charge);steps.extend(job['step_ids'])
 scores=list((folder/'logs').glob(score_prefix+'*.log'));assert len(scores)==1
 match=re.search(r'(415453\.\d+)\.log$',scores[0].name);assert match
 steps.append(match.group(1));previous=budget
assert len(steps)==4 and len(set(steps))==4
ledger=budgets[-1];total=ledger['charged_gpuh'];new=total-baseline
assert math.isfinite(total) and 0<=new<=1.2+1e-12 and total<=6 and total<=60
assert abs(new-sum(charges))<1e-12
assert ledger['diagnostic_cap_gpuh']==6 and ledger['round_cap_gpuh']==60
records={}
for step in steps:
 rows=subprocess.check_output(['sacct','-j',step,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES','-n','-P'],text=True).splitlines()
 r=[r.split('|') for r in rows if r.split('|')[0]==step];assert len(r)==1 and r[0][1:3]==['COMPLETED','0:0'],rows
 records[step]=dict(state=r[0][1],exit_code=r[0][2],elapsed_seconds=int(r[0][3]),allocated_tres=r[0][4])
for job,charge in zip((s['jobs'][0] for s in states),charges):
 assert charge+1e-12>=records[job['step_ids'][0]]['elapsed_seconds']*8/3600,'GPU charge below Slurm elapsed'
report=read(root/'report/report.json');comparison=read(child/'comparison/report.json')
assert report['status']=='diagnostic_complete' and report['original_parity']['passed'] is True
assert report['test_opened'] is False and report['p0_authorized'] is False
assert comparison['status']=='complete_cpu_frozen_comparison'
assert (comparison['groups'],comparison['episodes'],comparison['masks'])==(256,512,8)
assert comparison['test_opened'] is False and comparison['new_fit'] is False and comparison['no_case_selection'] is True
assert report['provenance']['source_commit']==states[0]['source_commit']
assert comparison['provenance']['source_commit']==states[1]['source_commit']
# Hash each new scene file once, then check complete exact child inventories.
files=inventory(root)
def record(name):return files[name]
def check_manifest(prefix,wrapped=True):
 manifest=read(root/prefix/'manifest.json')
 if wrapped:
  assert manifest['status']=='complete';expected=manifest['files']
 else:expected=manifest
 actual={name[len(prefix)+1:]:value for name,value in files.items() if name.startswith(prefix+'/') and name!=prefix+'/manifest.json'}
 assert actual==expected,'Incomplete or changed '+prefix
for prefix in ('counterfactual_cache','batch_match_run/cache','batch_match_run/comparison'):check_manifest(prefix)
check_manifest('report',False)
p=report['provenance'];c=comparison['provenance']
assert p['cache_sha256']==c['first_cache_sha256']==record('counterfactual_cache/cache.npz')['sha256']
assert p['cache_receipt_sha256']==record('counterfactual_cache/receipt.json')['sha256']
assert p['prior_manifest_sha256']==prior_sha
assert c['first_manifest_sha256']==record('counterfactual_cache/manifest.json')['sha256']
assert c['new_cache_sha256']==record('batch_match_run/cache/cache.npz')['sha256']
assert c['new_receipt_sha256']==record('batch_match_run/cache/receipt.json')['sha256']
assert c['lane_reader_sha256']==record('lane_reader/freeze_receipt.json')['sha256']
assert c['frozen_reader_sha256']==prior['files']['reader/freeze_receipt.json']['sha256']
for manifest_path in ('source_manifest.json','batch_match_run/source_manifest.json'):
 source=read(root/manifest_path)
 for name,digest in source['files'].items():assert record('code/'+name)['sha256']==digest,'Source changed '+name
accounting=dict(steps=records,all_new_steps_complete=True,active_new_leases=False,cumulative_charged_gpuh=total,inherited_gpuh=baseline,new_charged_gpuh=new,per_gpu_stage_charged_gpuh=charges,v55_cap_gpuh=1.2,diagnostic_cap_gpuh=6,round_cap_gpuh=60,all_caps_satisfied=True,supplement_source_commit=states[1]['source_commit'],comparison_status=comparison['status'],complete_submanifests_verified=True,prior_scene_rehashed_unchanged=True,prior_manifest_sha256=prior_sha,test_opened=False,p0_authorized=False)
with (root/'final_accounting.json').open('x') as f:json.dump(accounting,f,indent=2)
files['final_accounting.json']=dict(bytes=(root/'final_accounting.json').stat().st_size,sha256=sha(root/'final_accounting.json'))
seal=dict(source_commit=report['provenance']['source_commit'],created_unix=time.time(),files=files,dependencies=dict(prior_scene_manifest=prior_sha,prior_release='https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_4-20261010',official_encoder='unchanged shared asset; identity in counterfactual_cache/receipt.json'))
with (root/'archive_manifest.json').open('x') as f:json.dump(seal,f,indent=2)
print(json.dumps(dict(accounting=accounting,seal=seal,manifest_record=dict(bytes=(root/'archive_manifest.json').stat().st_size,sha256=sha(root/'archive_manifest.json')))))
'''.replace('REMOTE',repr(h.REMOTE)).replace('OLD',repr(h.OLD)).replace('CHILD_OWNER',repr(child_owner)).replace('OWNER',repr(owner))
sealed=h.remote(code);h.save(h.HERE/'remote_seal_receipt.json',sealed)
expected=dict(sealed['seal']['files']);expected['archive_manifest.json']=sealed['manifest_record']
archive=h.HERE/'visual-v5_5-20261010-remote-scene.tar.gz'
export_code="""import hashlib,json,stat,sys,tarfile,time
from pathlib import Path
root=Path(REMOTE);export=Path(str(root)+'.export.tar.gz');assert not export.exists() and not export.is_symlink()
assert not root.is_symlink();owner=OWNER;s=root.stat()
assert (s.st_ino,s.st_dev,s.st_uid)==(owner['inode'],owner['device'],owner['uid'])
assert json.loads((root/'creation.json').read_text())==owner
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'archive_manifest.json')==MANIFEST_SHA
files=json.loads((root/'archive_manifest.json').read_text())['files'];names=sorted(files)+['archive_manifest.json']
actual=[]
for p in root.rglob('*'):
 st=p.lstat();assert not p.is_symlink() and st.st_dev==s.st_dev
 assert stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)
 if p.is_file():actual.append(p.relative_to(root).as_posix())
assert set(actual)==set(names)
for name,record in files.items():
 p=root/name;assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],name
with export.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz',compresslevel=1) as tar:
 for name in names:tar.add(root/name,arcname=name,recursive=False)
h=hashlib.sha256()
with export.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
st=export.stat();print(json.dumps(dict(path=str(export),bytes=st.st_size,sha256=h.hexdigest(),inode=st.st_ino,device=st.st_dev,uid=st.st_uid,created_by_this_run=True,root=str(root),root_token=owner['token'],manifest_sha256=MANIFEST_SHA,created_unix=time.time())))
""".replace('REMOTE',repr(h.REMOTE)).replace('OWNER',repr(owner)).replace('MANIFEST_SHA',repr(sealed['manifest_record']['sha256']))
export=h.remote(export_code);h.save(h.HERE/'remote_export.json',export)
print(json.dumps(dict(stage='remote_export_ready',**export)),flush=True)
# Independent read ranges improve throughput over the high-latency SSH route.
import concurrent.futures
parts_dir=h.HERE/'download_parts';parts_dir.mkdir(exist_ok=False);chunk=128*1024**2
ranges=[(i,offset,min(chunk,export['bytes']-offset)) for i,offset in enumerate(range(0,export['bytes'],chunk))]
def download(row):
 i,offset,length=row;path=parts_dir/f'{i:04d}.part'
 code="""import sys
with open(EXPORT,'rb') as f:
 f.seek(OFFSET);remaining=LENGTH
 while remaining:
  b=f.read(min(1048576,remaining))
  if not b:raise RuntimeError('Unexpected export EOF')
  sys.stdout.buffer.write(b);remaining-=len(b)
""".replace('EXPORT',repr(export['path'])).replace('OFFSET',str(offset)).replace('LENGTH',str(length))
 with path.open('xb') as f:subprocess.run(h.helper.ssh()+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(code))],stdout=f,check=True)
 assert path.stat().st_size==length
 print(json.dumps(dict(stage='range_downloaded',part=i,bytes=length)),flush=True)
 return path
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(download,ranges))
with archive.open('xb') as out:
 for part in parts:
  with part.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''):out.write(block)
assert archive.stat().st_size==export['bytes'] and h.sha(archive)==export['sha256']
h.helper.extract_verified(archive,h.HERE/'diagnostic-20261010-v55',expected)
h.save(h.HERE/'local_archive_verification.json',dict(archive_sha256=h.sha(archive),archive_bytes=archive.stat().st_size,files=len(expected),all_files_verified=True,manifest_sha256=sealed['manifest_record']['sha256']))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(expected),all_files_verified=True)),flush=True)
