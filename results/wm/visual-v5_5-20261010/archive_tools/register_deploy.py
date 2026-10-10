#!/usr/bin/env python3
"""Publish/readback current source and lane model; deploy a new owned root only."""
import base64,hashlib,importlib.util,io,json,os,shlex,subprocess,sys,tarfile,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
REPO='XintongYang1010/predictive-memory-lab';TAG='visual-v5_5-fit-20261010'
REMOTE='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v55';OLD='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v54'
spec=importlib.util.spec_from_file_location('prior_registration',HERE.parent/'visual-v5_4-20261010/register_and_deploy.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
spec=importlib.util.spec_from_file_location('prior_helpers',HERE.parent/'visual-v5_4-20261010/helper_common.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
sha=helper.sha;inv=helper.inventory;save=helper.save_new

def cmd(args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()
def api(path):return json.loads(cmd(['gh','api','repos/'+REPO+'/'+path]))
def remote(code,stream=None):
 r=subprocess.run(helper.ssh()+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(code))],stdin=stream,capture_output=True,check=True)
 return json.loads(r.stdout)

def main():
 source=cmd(['git','rev-parse','HEAD'])
 paths=cmd(['git','ls-tree','-r','--name-only',source,'--','pmr','scripts','docs','tests','START_HERE.md','pyproject.toml','requirements.txt']).splitlines()
 for name in paths:
  if (ROOT/name).read_bytes()!=subprocess.check_output(['git','show',source+':'+name],cwd=ROOT):raise ValueError('Local source drift '+name)
 print('Verifying source blobs from GitHub',flush=True)
 files=old.github_files(source,[ROOT/n for n in paths])
 source_manifest=dict(commit=source,files={n:r['sha256'] for n,r in files.items()})
 save(HERE/'source_manifest.json',source_manifest)
 model=HERE/'lane_reader';records=inv(model)
 archive=HERE/'visual-v5_5-20261010-lane-reader.tar.gz'
 with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz') as t:
  for name in records:t.add(model/name,arcname=name,recursive=False)
 notes=HERE/'freeze-release-notes.md';notes.write_text('Fixed lane-restricted W reader, same archived FIT and fixed ridge. Complete joint FIT feature basis, lane kernel moments, coefficients, labels and receipts. No new counterfactual evaluation was generated before this freeze.\n\nSource: '+source+'\nSHA256: '+sha(archive)+'\n')
 subprocess.run(['gh','release','create',TAG,str(archive),'--repo',REPO,'--target',source,'--title','V5.5 fixed lane reader before counterfactual extraction','--notes-file',str(notes)],cwd=ROOT,check=True)
 release=api('releases/tags/'+TAG);asset=next(v for v in release['assets'] if v['name']==archive.name)
 readback=HERE/'lane-reader-github-readback.tar.gz'
 with readback.open('xb') as f:subprocess.run(['gh','api','repos/'+REPO+'/releases/assets/'+str(asset['id']),'-H','Accept: application/octet-stream'],stdout=f,check=True)
 if sha(readback)!=sha(archive) or readback.stat().st_size!=archive.stat().st_size:raise ValueError('Full model release readback mismatch')
 helper.extract_verified(readback,HERE/'lane-reader-github-readback',records)
 publication=dict(verified_readback=True,source_commit=source,plan_sha256=sha(ROOT/'scripts/v5_5_plan.json'),protocol_sha256=sha(ROOT/'docs/V5_5_MODULE_DIAGNOSTIC.md'),verified_unix=time.time(),files=files,
  lane_reader=dict(release_url=release['html_url'],asset_id=asset['id'],asset_sha256=sha(archive),asset_bytes=archive.stat().st_size,files=records,all_files_readback_verified=True))
 save(HERE/'registration_publication.json',publication)
 print('Source and complete lane model verified; creating new remote root',flush=True)
 create='''import json,time,uuid,hashlib
from pathlib import Path
p=Path(REMOTE);old=Path(OLD)
assert not p.exists() and not p.is_symlink() and not p.parent.is_symlink()
assert hashlib.sha256((old/'archive_manifest.json').read_bytes()).hexdigest()==SEAL
b=json.loads((old/'ledger/budget.json').read_text());assert abs(b['charged_gpuh']-4.522264341248407)<1e-12
assert not any(v.get('status')=='reserved_or_interrupted' for v in b['leases'])
p.mkdir(mode=0o700);s=p.stat();r=dict(root=str(p),inode=s.st_ino,device=s.st_dev,uid=s.st_uid,token=uuid.uuid4().hex,preexisted=False,parent_preserved=True,created_unix=time.time())
(p/'creation.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))
'''.replace('REMOTE',repr(REMOTE)).replace('OLD',repr(OLD)).replace('SEAL',repr(json.loads((ROOT/'scripts/v5_5_plan.json').read_text())['prior_manifest_sha256']))
 creation=remote(create);save(HERE/'creation.json',creation)
 bundle=HERE/'deployment.tar.gz';bundle_records={}
 with bundle.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz') as t:
  def add(p,name):
   p=Path(p);d=p.read_bytes();info=tarfile.TarInfo(name);info.size=len(d);info.mode=0o644;t.addfile(info,io.BytesIO(d));bundle_records[name]=dict(bytes=len(d),sha256=hashlib.sha256(d).hexdigest())
  for name in paths:add(ROOT/name,'code/'+name)
  for name in ('source_manifest.json','registration_publication.json','resource-query.json'):add(HERE/name,name)
  for name in records:add(model/name,'lane_reader/'+name)
  for folder in ('cpu_audit','software_fixture'):
   for p in (HERE/folder).rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:add(p,folder+'/'+p.relative_to(HERE/folder).as_posix())
  add(__file__,'execution_tools/register_deploy.py')
 deploy='''import hashlib,json,sys,tarfile
from pathlib import Path,PurePosixPath
root=Path(REMOTE);s=root.stat();c=json.loads((root/'creation.json').read_text());assert c==CREATION and c['inode']==s.st_ino and c['device']==s.st_dev
seen=set()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as t:
 for m in t:
  p=PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and m.name not in seen
  seen.add(m.name);out=root/p;out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('xb') as f:
   src=t.extractfile(m)
   for b in iter(lambda:src.read(1048576),b''):f.write(b)
expected=EXPECTED;assert seen==set(expected)
for name,r in expected.items():
 p=root/name;assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],name
print(json.dumps(dict(verified_files=len(seen),source_commit=SOURCE,root=str(root))))
'''.replace('REMOTE',repr(REMOTE)).replace('CREATION',repr(creation)).replace('EXPECTED',repr(bundle_records)).replace('SOURCE',repr(source))
 with bundle.open('rb') as f:deployed=remote(deploy,f)
 save(HERE/'deployment.json',dict(source_commit=source,verified_remote=deployed,bundle_sha256=sha(bundle),files=bundle_records))
 print(json.dumps(deployed),flush=True)
if __name__=='__main__':main()
