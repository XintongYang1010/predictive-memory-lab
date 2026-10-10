"""Copy only newly created posthoc analysis into the new investigation scene."""
import hashlib,json,shlex,subprocess,tarfile
from pathlib import Path
B=Path('execution/visual-v5_4-20261010');REMOTE='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v54'
mapping={B/'posthoc_report':'analysis',B/'web_result_review.md':'web_result_review.md',B/'web_design_review.md':'web_design_review.md'}
tools=['build_results.py','helper_common.py','monitor.py','download_reports.py','publish_frozen.py','register_and_deploy.py','remote_control.py','seal_and_download.py','verify_github.py','curate_results.py','upload_derived.py']
mapping.update({B/n:'analysis_tools/'+n for n in tools})
token=json.loads((B/'creation.json').read_text())['token']
records={}
for src,dest in mapping.items():
 assert src.exists(),str(src)
 for p in ([src] if src.is_file() else src.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():
   n=dest if p==src else dest+'/'+str(p.relative_to(src));records[n]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
archive=B/'derived-upload.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for src,dest in mapping.items():t.add(src,arcname=dest)
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','dgx-127-slurm']
if Path('/tmp/pmr-v5-dgx-20261010.sock').exists():ssh[1:1]=['-S','/tmp/pmr-v5-dgx-20261010.sock']
check='import json;from pathlib import Path;p=Path('+repr(REMOTE)+');c=json.loads((p/"creation.json").read_text());assert c["token"]=='+repr(token)+';assert not any((p/n).exists() for n in ["archive_manifest.json","analysis","web_result_review.md","web_design_review.md","analysis_tools"]);assert json.loads((p/"evaluation_controller.json").read_text())["status"]=="repair_diagnostic_complete"'
subprocess.run(ssh+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(check))],check=True)
with archive.open('rb') as f:subprocess.run(ssh+['bash -lc '+shlex.quote('tar -xzf - -C '+REMOTE)],stdin=f,check=True)
verify='import hashlib,json;from pathlib import Path;p=Path('+repr(REMOTE)+');r='+repr(records)+';assert all((p/n).stat().st_size==v["bytes"] and hashlib.sha256((p/n).read_bytes()).hexdigest()==v["sha256"] for n,v in r.items());print(json.dumps({"verified_files":len(r)}))'
r=subprocess.run(ssh+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(verify))],capture_output=True,text=True);assert r.returncode==0,r.stderr;print(r.stdout)
(B/'derived_upload_receipt.json').write_text(json.dumps(dict(files=records,remote_verified=json.loads(r.stdout)),indent=2)+'\n')
