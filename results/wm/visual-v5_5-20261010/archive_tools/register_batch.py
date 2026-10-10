import io,json,shlex,subprocess,tarfile,time
from pathlib import Path
import register_deploy as h
NAMES=['scripts/v5_5_batch_match.py','scripts/launch_v5_5_batch_match.py','docs/V5_5_BATCH_MATCH_ADDENDUM.md']
source=h.cmd(['git','rev-parse','HEAD']);files=h.old.github_files(source,[h.ROOT/n for n in NAMES])
manifest=h.helper.read(h.HERE/'source_manifest.json');manifest['commit']=source;manifest['files'].update({n:r['sha256'] for n,r in files.items()})
info=h.remote('''import hashlib,json
from pathlib import Path
p=Path(REMOTE);s=json.loads((p/'extraction_controller.json').read_text());assert s['status']=='extraction_complete'
b=json.loads((p/'ledger/budget.json').read_text());assert b['leases'][-1]['status']=='closed' and b['controller_envelopes'][-1]['status']=='closed'
remaining=1.2-(b['charged_gpuh']-4.522264341248407);assert remaining>=.45 and b['charged_gpuh']+.45<=6
assert json.loads((p/'counterfactual_cache/manifest.json').read_text())['status']=='complete'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
print(json.dumps(dict(first_cache_manifest_sha256=sha(p/'counterfactual_cache/manifest.json'),first_ledger_sha256=sha(p/'ledger/budget.json'),remaining_gpuh=remaining,charged_gpuh=b['charged_gpuh'])))
'''.replace('REMOTE',repr(h.REMOTE)))
pub=dict(info,source_commit=source,verified_readback=True,verified_unix=time.time(),addendum_sha256=h.sha(h.ROOT/NAMES[-1]),files=files)
h.save(h.HERE/'batch_source_manifest.json',manifest);h.save(h.HERE/'batch_registration_publication.json',pub)
archive=h.HERE/'batch_deployment.tar.gz'
with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz') as t:
 for n in NAMES:t.add(h.ROOT/n,arcname='code/'+n,recursive=False)
 for n,name in [('batch_source_manifest.json','source_manifest.json'),('batch_registration_publication.json','registration_publication.json')]:t.add(h.HERE/n,arcname='batch_match_run/'+name,recursive=False)
code='''import hashlib,json,sys,tarfile,time,uuid
from pathlib import Path,PurePosixPath
p=Path(REMOTE);child=p/'batch_match_run';assert not child.exists() and not child.is_symlink()
child.mkdir();s=child.stat();c=dict(root=str(child),inode=s.st_ino,device=s.st_dev,uid=s.st_uid,token=uuid.uuid4().hex,preexisted=False,parent_preserved=True,created_unix=time.time())
(child/'creation.json').write_text(json.dumps(c,indent=2))
seen=set()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as t:
 for m in t:
  name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts and m.name not in seen
  seen.add(m.name);dst=p/name;dst.parent.mkdir(parents=True,exist_ok=True)
  with dst.open('xb') as f:f.write(t.extractfile(m).read())
manifest=json.loads((child/'source_manifest.json').read_text())
for name,sha in manifest['files'].items():assert hashlib.sha256((p/'code'/name).read_bytes()).hexdigest()==sha,name
assert hashlib.sha256((p/'ledger/budget.json').read_bytes()).hexdigest()==LEDGER
print(json.dumps(dict(source_commit=manifest['commit'],creation=c,verified_source_files=len(manifest['files']))))
'''.replace('REMOTE',repr(h.REMOTE)).replace('LEDGER',repr(info['first_ledger_sha256']))
with archive.open('rb') as f:deployed=h.remote(code,f)
h.save(h.HERE/'batch_deployment.json',deployed);print(json.dumps(deployed,indent=2))
