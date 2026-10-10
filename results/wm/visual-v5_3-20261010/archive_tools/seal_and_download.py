#!/usr/bin/env python3
"""Seal completed V5.3 evidence, download it, and verify every archived byte.

Requires --execute. Never deletes scenes, cancels jobs, starts Slurm steps, or
charges the ledger. Existing output is preserved on any failure.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def save_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


REMOTE = r'''
import hashlib,json,re,subprocess,time
from pathlib import Path,PurePosixPath
cfg=CONFIGURATION
p=Path(cfg['root']);old=Path(cfg['prior_root'])
def require(value,message):
 if not value:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(name):return json.loads((p/name).read_text())
def save(name,value):
 with (p/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def safe(name):
 q=PurePosixPath(name)
 require(name and not q.is_absolute() and '..' not in q.parts and str(q)==name,'Unsafe manifest path: '+name)
 return q
def inventory(root,exclude=()):
 result={}
 for path in sorted(root.rglob('*')):
  require(not path.is_symlink(),'Symlink in scene: '+str(path))
  if path.is_dir():continue
  require(path.is_file(),'Nonregular scene entry: '+str(path))
  name=path.relative_to(root).as_posix()
  if name in exclude:continue
  result[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
 return result
require(not p.is_symlink() and p.is_dir() and p!=old,'Invalid new scene')
c=read('creation.json');st=p.stat()
require(c['root']==str(p) and c['token']==cfg['token'] and c['inode']==st.st_ino and c['device']==st.st_dev and c['uid']==st.st_uid,'Creation identity mismatch')
require(c.get('preexisted') is False and c.get('parent_preserved') is True,'Scene was not newly created')
prior_manifest=old/'archive_manifest.json'
require(not old.is_symlink() and sha(prior_manifest)==cfg['prior_manifest_sha256'],'Prior V5.2 seal changed')
om=json.loads(prior_manifest.read_text())
for name in om['files']:safe(name)
require(inventory(old,('archive_manifest.json',))==om['files'],'Prior V5.2 inventory or content changed')
source=read('source_manifest.json')
require(source['commit']==cfg['source_commit'],'Unexpected source commit')
for name,digest in source['files'].items():
 safe(name);require(sha(p/'code'/name)==digest,'Changed frozen source: '+name)
publication=read('registration_publication.json')
require(publication.get('verified_readback') is True and publication['source_commit']==cfg['source_commit'],'Source publication missing')
stages={'fit':('reader_frozen_awaiting_publication',['v53_fit','v53_fit_reader']),
        'evaluation':('repair_diagnostic_complete',['v53_anchor','v53_anchor_reader','v53_bridge','v53_bridge_reader'])}
jobs=[];controllers={}
for action,(status,expected_stages) in stages.items():
 state=read(action+'_controller.json');controllers[action]=state
 require(state['status']==status and state['source_commit']==cfg['source_commit'],action+' controller incomplete')
 require([j['stage'] for j in state['jobs']]==expected_stages,'Controller stage inventory changed')
 for j in state['jobs']:
  require(j['status']=='complete' and j['returncode']==0,'Controller step failed: '+j['stage'])
  require(j['allocation_job']==cfg['parent_job'],'Unexpected parent allocation')
  ids=set()
  for name in j['logs']:
   safe(name);require((p/name).is_file(),'Missing step log')
   match=re.search(r'_(%s\.\d+)\.log$'%re.escape(cfg['parent_job']),name)
   if match:ids.add(match.group(1))
  require(len(ids)==1,'Cannot identify unique actual Slurm step: '+j['stage'])
  jobs.append(dict(stage=j['stage'],step_id=next(iter(ids)),selected_gpus=j['selected_gpus']))
require(len({j['step_id'] for j in jobs})==6,'Repeated scientific Slurm step')
if cfg['expected_steps']:require({j['step_id'] for j in jobs}==set(cfg['expected_steps']),'Actual steps differ from requested IDs')
acct=subprocess.check_output(['sacct','-j',cfg['parent_job'],'--format=JobIDRaw,JobName%60,State,ElapsedRaw,AllocTRES%100,ExitCode','-P','-n'],text=True)
rows={line.split('|')[0]:line.split('|') for line in acct.splitlines() if line.strip()}
for job in jobs:
 row=rows.get(job['step_id']);require(row is not None and row[2]=='COMPLETED' and row[5]=='0:0','Slurm step not successfully completed: '+job['step_id'])
 require(row[1]==job['stage'],'Slurm step name differs from controller')
 job['sacct']=row
budget=read('ledger/budget.json');inherited=read('ledger/inherited_budget.json')
require(abs(inherited['charged_gpuh']-2.2082544136047373)<1e-10,'Wrong inherited baseline')
require(all(v['status'] in ('complete','partial') for v in budget['leases']),'Active/interrupted budget lease')
require(all(v['status']=='closed' for v in budget.get('controller_envelopes',[])),'Unclosed controller envelope')
require(budget['charged_gpuh']<=min(6.,2.2082544136047373+3.75)+1e-8,'Diagnostic budget cap exceeded')
require(abs(sum(budget['stages'].values())-budget['charged_gpuh'])<1e-8,'Ledger stage total mismatch')
management=[r for r in budget['overhead_receipts'] if r.get('stage')=='management_v5_3']
require(len(management)==1 and management[0]['job_id']==cfg['management_step'],'Existing management charge missing/duplicated')
mr=rows.get(cfg['management_step']);require(mr is not None and mr[2]=='COMPLETED' and mr[5]=='0:0','Management step not complete')
require(abs(management[0]['charged_gpuh']-int(mr[3])*8/3600)<1e-10,'Management charge differs from Slurm')
for role in ('fit','anchor','bridge'):
 folder=p/(role+'_cache');m=json.loads((folder/'manifest.json').read_text())
 require(m['status']=='complete' and m['files']==inventory(folder,('manifest.json',)),role+' cache incomplete or changed')
 receipt=json.loads((folder/'receipt.json').read_text())
 require(receipt['role']==role and receipt['source_commit']==cfg['source_commit'] and receipt['test_opened'] is False,'Wrong cache role/source/Test state')
 require(sha(folder/'cache.npz')==receipt['cache_sha256'],role+' cache digest changed')
 require(len(list((folder/'shards').glob('*.json')))==256 and len(list((folder/'shards').glob('*.npz')))==256,'Missing pair/block shards')
 for name,digest in receipt['source_identity'].items():safe(name);require(sha(p/'code'/name)==digest,'Cache source identity changed')
freeze=read('reader/freeze_receipt.json');fp=read('freeze_publication.json')
require(freeze['status']=='frozen_before_evaluation' and freeze['cache_identity']['source_commit']==cfg['source_commit'],'Incomplete reader freeze')
for name,digest in freeze['files'].items():safe(name);require(sha(p/'reader'/name)==digest,'Frozen model/config changed')
require(fp['verified_readback'] is True and fp['frozen_reader_sha256']==sha(p/'reader/freeze_receipt.json') and fp['verified_unix']>=freeze['created_unix'],'Reader publication did not follow exact freeze')
require(fp['source_commit']==cfg['source_commit'] and fp['plan_sha256']==freeze['cache_identity']['plan_sha256'],'Reader publication source/plan mismatch')
required=['reader/frozen_model.npz','reader/training_config.json','freshness_audit.json',
          'anchor_report/report.json','anchor_report/predictions.npz','bridge_report/report.json','bridge_report/predictions.npz']+cfg['require_paths']
for name in required:safe(name);require((p/name).exists(),'Required archive evidence missing: '+name)
if (p/'archive_manifest.json').exists():
 manifest=read('archive_manifest.json')
 require(manifest['source_commit']==cfg['source_commit'] and manifest['creation']==c,'Existing seal identity differs')
 require(manifest['files']==inventory(p,('archive_manifest.json',)),'Existing sealed scene changed')
else:
 save('dependencies.json',dict(prior_scene=str(old),prior_release=cfg['prior_release'],prior_archive_manifest_sha256=sha(prior_manifest),
      prior_files_verified=len(om['files']),prior_scene_full_readback_unchanged=True,
      external_encoder_checkpoint='Official IN1K-vit.h.14-300e.pth.tar SHA256 0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22',
      external_encoder_source_commit='52c1ae95d05f743e000e8f10a1f3a79b10cff048'))
 with (p/'slurm_accounting.txt').open('x') as f:f.write(acct)
 save('final_accounting.json',dict(checked_unix=time.time(),cumulative_charged_gpuh=budget['charged_gpuh'],
      inherited_gpuh=inherited['charged_gpuh'],v53_gpuh=budget['charged_gpuh']-inherited['charged_gpuh'],
      management_already_in_ledger=True,management_charge_added_by_sealer=False,steps=jobs,
      parent_allocation=rows.get(cfg['parent_job']),indexmem_step=rows.get(cfg['indexmem_step']),
      indexmem_state_is_observation_not_required_running=True,all_new_steps_complete=True,
      active_new_leases=False,scientific_status=cfg['scientific_status'],test_opened=False,p0_started=False,
      cleanup='Retained investigation scene; no deletion or job cancellation. Original V5/V5.1 gates unchanged.'))
 manifest=dict(schema='v5.3.remote-scene.v1',root=str(p),sealed_unix=time.time(),source_commit=cfg['source_commit'],creation=c,files=inventory(p))
 save('archive_manifest.json',manifest)
require(inventory(p,('archive_manifest.json',))==manifest['files'],'Scene changed while sealing')
print(json.dumps(dict(root=str(p),files=len(manifest['files']),bytes=sum(x['bytes'] for x in manifest['files'].values()),
 manifest_sha256=sha(p/'archive_manifest.json'),source_commit=cfg['source_commit'],creation_token=c['token'],
 cumulative_gpuh=budget['charged_gpuh'],ledger_unchanged_by_sealer=True,steps=jobs,
 cleanup='retained; no deletion, cancellation, or new GPU step'),indent=2))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--root',default='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v53')
    parser.add_argument('--token',default='1ef44dc737d54795b62dcba432cdd007')
    parser.add_argument('--source-commit',default='875be45515276cc8559ffcdb21c7412c6048d5ad')
    parser.add_argument('--prior-root',default='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v52')
    parser.add_argument('--prior-manifest-sha256',default='5e6cc30bbe1952081732f8117a7791372f2409b02317d862e75898c911ac635a')
    parser.add_argument('--prior-release',default='https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_2-20261010')
    parser.add_argument('--parent-job',default='415453')
    parser.add_argument('--indexmem-step',default='415453.5')
    parser.add_argument('--management-step',default='415453.28')
    parser.add_argument('--expected-steps',nargs='*',default=[])
    parser.add_argument('--require-path',action='append',dest='require_paths',default=[])
    parser.add_argument('--scientific-status',required=True,help='Concrete final scientific finding; does not change any gate')
    parser.add_argument('--ssh-alias',default='dgx-127-slurm')
    parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--archive-name',default='visual-v5_3-20261010-remote-scene.tar.gz')
    args=parser.parse_args()
    if not args.execute:parser.error('Review the script, then pass --execute to seal and download')
    if PurePosixPath(args.archive_name).name!=args.archive_name or args.archive_name in ('.','..'):
        raise ValueError('Archive filename must be a basename')
    cfg={key:value for key,value in vars(args).items() if key not in ('execute','base','archive_name','ssh_alias')}
    args.base.mkdir(parents=True,exist_ok=True)
    receipt_path=args.base/'remote_seal_receipt.json';archive=args.base/args.archive_name
    scene_name=PurePosixPath(args.root).name;scene=args.base/scene_name
    if archive.exists() or scene.exists():raise FileExistsError('Existing archive/scene preserved; use a fresh --base')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',args.ssh_alias]
    if Path('/tmp/pmr-v5-dgx-20261010.sock').exists():ssh[1:1]=['-S','/tmp/pmr-v5-dgx-20261010.sock']
    code=REMOTE.replace('CONFIGURATION',repr(cfg),1)
    command='bash -lc '+shlex.quote('python3 -c '+shlex.quote(code))
    result=subprocess.run(ssh+[command],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    sealed=json.loads(result.stdout)
    if receipt_path.exists():
        if json.loads(receipt_path.read_text())!=sealed:raise ValueError('Existing local seal receipt differs')
    else:save_new(receipt_path,sealed)
    print(json.dumps(sealed),flush=True)
    command='tar -czf - -C '+shlex.quote(str(PurePosixPath(args.root).parent))+' '+shlex.quote(scene_name)
    with archive.open('xb') as stream:
        result=subprocess.run(ssh+['bash -lc '+shlex.quote(command)],stdout=stream,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stderr.decode())
    seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            name=PurePosixPath(member.name)
            if (str(name) in seen or name.is_absolute() or '..' in name.parts or not name.parts
                    or name.parts[0]!=scene_name or not(member.isfile() or member.isdir()) or member.issparse()):
                raise ValueError('Unsafe or duplicate archive member: '+member.name)
            seen.add(str(name));target=args.base.joinpath(*name.parts)
            if member.isdir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as output:
                    source=tar.extractfile(member)
                    for chunk in iter(lambda:source.read(1048576),b''):output.write(chunk)
    manifest=json.loads((scene/'archive_manifest.json').read_text())
    if sha(scene/'archive_manifest.json')!=sealed['manifest_sha256']:raise ValueError('Seal digest mismatch after download')
    actual={}
    for path in scene.rglob('*'):
        if path.is_symlink():raise ValueError('Local scene symlink')
        if path.is_file() and path!=scene/'archive_manifest.json':
            actual[path.relative_to(scene).as_posix()]=dict(bytes=path.stat().st_size,sha256=sha(path))
    if actual!=manifest['files']:raise ValueError('Downloaded exact file inventory/hash mismatch')
    receipt=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),scene_files=len(actual),
                 scene_bytes=sum(x['bytes'] for x in actual.values()),manifest_sha256=sealed['manifest_sha256'],
                 local_snapshot_hashes_verified=True,verified_unix=time.time(),cleanup='retained; no deletion')
    save_new(args.base/'local_snapshot_receipt.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
