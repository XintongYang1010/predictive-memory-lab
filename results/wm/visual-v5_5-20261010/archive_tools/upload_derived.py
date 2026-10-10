"""Upload final scientific analyses to fresh paths inside this run only."""
import io,json,tarfile,hashlib
import register_deploy as h
def main():
 mapping={h.HERE/'figures':'final_analysis/figures',h.HERE/'lane_original_eval':'final_analysis/lane_original_eval'}
 names=['FINAL_README.md','REPRODUCE.md','independent_interpretation.md','independent_interpretation.json','final_interpretation.md','final_interpretation.json','web_review.md','figure_visual_review.json','prior-tests-summary.json','batch-report-snapshot-verification.json','report-snapshot-verification.json','batch_registration_publication.json','batch_deployment.json','batch_source_manifest.json']
 mapping.update({h.HERE/n:'final_analysis/'+n for n in names})
 names=['control.py','control_batch.py','register_batch.py','download_report.py','download_batch_report.py','make_figures.py','recompute_interpretation.py','finalize_interpretation.py','test_controller_lifecycle.py','seal_download.py','release_and_verify.py','cleanup_verified.py','upload_derived.py','curate_final.py']
 mapping.update({h.HERE/n:'final_analysis/execution_tools/'+n for n in names})
 mapping[h.HERE.parent/'visual-v5_4-20261010/helper_common.py']='final_analysis/execution_tools/prior_helper_common.py'
 records={};paths={}
 for src,dest in mapping.items():
  assert src.exists(),str(src)
  for p in ([src] if src.is_file() else sorted(src.rglob('*'))):
   assert not p.is_symlink()
   if p.is_file() and '__pycache__' not in p.parts:
    name=dest if p==src else dest+'/'+p.relative_to(src).as_posix();assert name not in paths
    paths[name]=p;records[name]=dict(bytes=p.stat().st_size,sha256=h.sha(p))
 bundle=h.HERE/'derived-upload.tar.gz'
 with bundle.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz') as t:
  for name,p in paths.items():t.add(p,arcname=name,recursive=False)
 owner=json.loads((h.HERE/'creation.json').read_text())
 code='''import hashlib,json,sys,tarfile
from pathlib import Path,PurePosixPath
root=Path(REMOTE);owner=OWNER;expected=EXPECTED
assert json.loads((root/'creation.json').read_text())==owner
assert not root.is_symlink() and root.stat().st_ino==owner['inode'] and root.stat().st_dev==owner['device']
assert not (root/'archive_manifest.json').exists() and not (root/'final_analysis').exists()
seen=set()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as t:
 for m in t:
  p=PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and str(p)==m.name and m.name not in seen and m.name in expected
  seen.add(m.name);out=root/p;out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('xb') as f:
   src=t.extractfile(m)
   for b in iter(lambda:src.read(1048576),b''):f.write(b)
assert seen==set(expected)
for n,r in expected.items():
 p=root/n;assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],n
print(json.dumps(dict(verified_files=len(seen),root=str(root))))
'''.replace('REMOTE',repr(h.REMOTE)).replace('OWNER',repr(owner)).replace('EXPECTED',repr(records))
 with bundle.open('rb') as f:result=h.remote(code,f)
 h.save(h.HERE/'derived_upload_receipt.json',dict(files=records,remote_verified=result))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
