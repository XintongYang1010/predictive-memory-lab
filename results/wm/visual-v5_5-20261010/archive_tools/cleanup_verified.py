"""Remove only this turn's owned, sealed, fully read-back remote scene/export."""
import json,subprocess
import register_deploy as h
def main():
 v=json.loads((h.HERE/'github_verification.json').read_text());r=json.loads((h.HERE/'verification_receipt_readback.json').read_text())
 assert v['full_github_download'] and v['all_archive_member_hashes_verified'] and v['exact_inventory'] and r['files']
 receipt_name='results/wm/visual-v5_5-20261010/github_verification.json'
 local_receipt=h.HERE/'github_verification.json';git_receipt=h.ROOT/receipt_name
 assert local_receipt.read_bytes()==git_receipt.read_bytes()
 assert receipt_name in r['files'] and r['files'][receipt_name]['sha256']==h.sha(local_receipt) and r['files'][receipt_name]['bytes']==local_receipt.stat().st_size
 seal=json.loads((h.HERE/'remote_seal_receipt.json').read_text());owner=json.loads((h.HERE/'creation.json').read_text());export=json.loads((h.HERE/'remote_export.json').read_text())
 assert v['manifest_sha256']==seal['manifest_record']['sha256'] and v['archive']['sha256']==export['sha256']
 assert v['archive']['bytes']==export['bytes'] and v['archive_member_count']==len(seal['seal']['files'])+1
 assert export['root']==owner['root'] and export['root_token']==owner['token'] and export['manifest_sha256']==v['manifest_sha256']
 code='''import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(REMOTE);old=Path(OLD);export=EXPORT;owner=OWNER
assert str(root)=='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v55'
assert not root.is_symlink() and root.parent.is_dir() and not root.parent.is_symlink()
assert owner['root']==str(root) and owner['preexisted'] is False
s=root.stat();assert (s.st_ino,s.st_dev,s.st_uid)==(owner['inode'],owner['device'],owner['uid'])
assert json.loads((root/'creation.json').read_text())==owner
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def inventory(folder):
 result={}
 for p in sorted(folder.rglob('*')):
  assert not p.is_symlink() and not p.is_mount() and p.stat().st_dev==folder.stat().st_dev,str(p)
  if p.is_file():result[p.relative_to(folder).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
  else:assert p.is_dir(),str(p)
 return result
assert sha(root/'archive_manifest.json')==MANIFEST_SHA
manifest=json.loads((root/'archive_manifest.json').read_text());expected=dict(manifest['files']);expected['archive_manifest.json']=MANIFEST_RECORD
assert inventory(root)==expected,'Post-seal change; retain scene'
account=json.loads((root/'final_accounting.json').read_text());assert account['all_new_steps_complete'] and not account['active_new_leases']
step_status={}
for step in account['steps']:
 rows=subprocess.check_output(['sacct','-j',step,'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True).splitlines()
 records=[r.split('|') for r in rows if r.split('|')[0]==step]
 assert len(records)==1 and records[0][1:3]==['COMPLETED','0:0'],rows
 step_status[step]=records[0][1:3]
assert sha(old/'archive_manifest.json')==account['prior_manifest_sha256']
prior=json.loads((old/'archive_manifest.json').read_text());prior_expected=dict(prior['files']);prior_expected['archive_manifest.json']=dict(bytes=(old/'archive_manifest.json').stat().st_size,sha256=sha(old/'archive_manifest.json'))
assert inventory(old)==prior_expected,'Preexisting prior scene changed'
ep=Path(export['path']);assert ep==Path(str(root)+'.export.tar.gz') and export['created_by_this_run'] is True
assert not ep.is_symlink();es=ep.stat();assert (es.st_ino,es.st_dev,es.st_uid,es.st_size)==(export['inode'],export['device'],export['uid'],export['bytes']) and sha(ep)==export['sha256']
parent_before={p.name:(p.lstat().st_ino,p.lstat().st_dev,p.lstat().st_mode) for p in root.parent.iterdir() if p not in (root,ep)}
assert root.stat().st_ino==owner['inode'] and ep.stat().st_ino==export['inode']
shutil.rmtree(root);ep.unlink()
assert not root.exists() and not ep.exists()
parent_after={p.name:(p.lstat().st_ino,p.lstat().st_dev,p.lstat().st_mode) for p in root.parent.iterdir()}
assert all(parent_after.get(k)==v for k,v in parent_before.items()),'Existing parent entry altered'
assert inventory(old)==prior_expected
print(json.dumps(dict(status='completed',removed=[str(root),str(ep)],deleted_only_owned_verified_artifacts=True,verified_archive_sha256=ARCHIVE_SHA,github_verification_commit=VERIFIED_COMMIT,steps=step_status,prior_scene_all_file_hashes_unchanged=True,existing_parent_entries_preserved=True,no_job_cancellation=True,completed_unix=time.time())))
'''.replace('REMOTE',repr(h.REMOTE)).replace('OLD',repr(h.OLD)).replace('EXPORT',repr(export)).replace('OWNER',repr(owner)).replace('MANIFEST_SHA',repr(v['manifest_sha256'])).replace('MANIFEST_RECORD',repr(seal['manifest_record'])).replace('ARCHIVE_SHA',repr(v['archive']['sha256'])).replace('VERIFIED_COMMIT',repr(r['commit']))
 result=h.remote(code);h.save(h.HERE/'cleanup_receipt.json',result)
 out=h.ROOT/'results/wm/visual-v5_5-20261010/cleanup_receipt.json';h.save(out,result)
 subprocess.run(['git','add','--',str(out.relative_to(h.ROOT))],cwd=h.ROOT,check=True)
 subprocess.run(['git','commit','-m','Record verified V5.5 cleanup preserving prior scenes and allocation'],cwd=h.ROOT,check=True)
 subprocess.run(['git','push','origin','HEAD'],cwd=h.ROOT,check=True)
 commit=h.cmd(['git','rev-parse','HEAD']);files=h.old.github_files(commit,[out]);h.save(h.HERE/'cleanup_receipt_readback.json',dict(commit=commit,files=files))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
