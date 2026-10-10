"""Publish and re-download all frozen readers before evaluation release."""
import base64,hashlib,json,shlex,subprocess,tarfile,time
from pathlib import Path,PurePosixPath
B=Path('execution/visual-v5_3-20261010');D=B/'frozen';D.mkdir(exist_ok=True)
R=Path('results/wm/visual-v5_3-20261010-freeze');R.mkdir(parents=True,exist_ok=True)
REMOTE='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v53';REPO='XintongYang1010/predictive-memory-lab';TAG='visual-v5_3-fit-20261010'
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','dgx-127-slurm']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def cmd(a):return subprocess.check_output(a,text=True).strip()
def api(p):return json.loads(cmd(['gh','api','repos/'+REPO+'/'+p]))
check='import json;from pathlib import Path;p=Path('+repr(REMOTE)+');s=json.loads((p/"fit_controller.json").read_text());assert s["status"]=="reader_frozen_awaiting_publication" and all(j["returncode"]==0 for j in s["jobs"]);assert not (p/"anchor_cache").exists() and not (p/"bridge_cache").exists();print(s["source_commit"])'
commit=cmd(ssh+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(check))])
archive=B/'visual-v5_3-20261010-frozen-readers.tar.gz'
with archive.open('xb') as f:subprocess.run(ssh+['bash -lc '+shlex.quote('tar -czf - -C '+REMOTE+' reader fit_cache/receipt.json source_manifest.json registration_publication.json freshness_audit.json fit_controller.json')],stdout=f,check=True)
with tarfile.open(archive,'r:gz') as t:
 for m in t:
  q=PurePosixPath(m.name);assert not q.is_absolute() and '..' not in q.parts and (m.isfile() or m.isdir())
  out=D/m.name
  if m.isdir():out.mkdir(parents=True,exist_ok=True)
  else:out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(t.extractfile(m).read())
freeze=json.loads((D/'reader/freeze_receipt.json').read_text());assert freeze['cache_identity']['source_commit']==commit
for n,h in freeze['files'].items():assert sha(D/'reader'/n)==h
for name in ['reader/freeze_receipt.json','reader/training_config.json','fit_cache/receipt.json','fit_controller.json']:
 out=R/name;out.parent.mkdir(exist_ok=True,parents=True);out.write_bytes((D/name).read_bytes())
meta=dict(file=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,tag=TAG,source_commit=commit)
save(R/'release_asset.json',meta)
(R/'README.md').write_text('# V5.3 complete reader freeze\n\nAll six S/B × concat/t6/t7 models were trained on FIT only and frozen before any anchor or bridge access. The complete model basis, coefficients, intercepts and normalization moments are in the Release asset. Evaluation is forbidden until the asset and receipts are downloaded and verified.\n')
subprocess.run(['git','add',str(R)],check=True);subprocess.run(['git','commit','-m','Publish complete V5.3 FIT reader freeze before evaluation'],check=True);subprocess.run(['git','push'],check=True)
freezecommit=cmd(['git','rev-parse','HEAD'])
notes=B/'freeze-release-notes.md';notes.write_text('Complete frozen V5.3 readers, all six arms/inputs, including FIT feature basis and provenance. Created before fresh anchor/bridge generation. This freeze is not an evaluation result.\n\nSource: '+commit+'\nAsset SHA256: '+meta['sha256']+'\n')
subprocess.run(['gh','release','create',TAG,str(archive),'--repo',REPO,'--target',freezecommit,'--title','V5.3 frozen FIT readers before evaluation','--notes-file',str(notes)],check=True)
release=api('releases/tags/'+TAG);asset=next(v for v in release['assets'] if v['name']==archive.name)
assert asset['size']==meta['bytes'] and asset['state']=='uploaded'
readback=B/'frozen-github-readback.tar.gz'
with readback.open('xb') as f:subprocess.run(['gh','api','repos/'+REPO+'/releases/assets/'+str(asset['id']),'-H','Accept: application/octet-stream'],stdout=f,check=True)
assert sha(readback)==meta['sha256']
records={};seen=set()
with tarfile.open(readback,'r:gz') as t:
 for m in t:
  assert m.name not in seen;seen.add(m.name)
  q=PurePosixPath(m.name);assert not q.is_absolute() and '..' not in q.parts and (m.isfile() or m.isdir())
  if m.isfile():
   data=t.extractfile(m).read();assert data==(D/m.name).read_bytes();records[m.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
assert set(records)=={str(p.relative_to(D)) for p in D.rglob('*') if p.is_file()}
for path in R.rglob('*'):
 if path.is_file():
  blob=api('contents/'+str(path)+'?ref='+freezecommit);assert base64.b64decode(blob['content'])==path.read_bytes()
publication=dict(verified_readback=True,frozen_reader_sha256=sha(D/'reader/freeze_receipt.json'),source_commit=commit,plan_sha256=freeze['cache_identity']['plan_sha256'],verified_unix=time.time(),freeze_commit=freezecommit,release_url=release['html_url'],asset_id=asset['id'],asset_sha256=meta['sha256'],asset_bytes=meta['bytes'],all_model_and_receipt_files_downloaded_and_verified=True,files=records)
save(B/'freeze_publication.json',publication);save(R/'freeze_publication.json',publication)
subprocess.run(['git','add',str(R/'freeze_publication.json')],check=True);subprocess.run(['git','commit','-m','Verify GitHub reader archive before releasing V5.3 evaluation'],check=True);subprocess.run(['git','push'],check=True)
pubcommit=cmd(['git','rev-parse','HEAD']);blob=api('contents/'+str(R/'freeze_publication.json')+'?ref='+pubcommit);assert base64.b64decode(blob['content'])==(B/'freeze_publication.json').read_bytes()
with (B/'freeze_publication.json').open('rb') as f:subprocess.run(ssh+['bash -lc '+shlex.quote('test ! -e '+REMOTE+'/freeze_publication.json && cat > '+REMOTE+'/freeze_publication.json')],stdin=f,check=True)
print(json.dumps(publication,indent=2),flush=True)
