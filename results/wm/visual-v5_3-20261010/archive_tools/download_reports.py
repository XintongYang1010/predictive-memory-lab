import json,shlex,subprocess,tarfile
from pathlib import Path,PurePosixPath
B=Path('execution/visual-v5_3-20261010');remote='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v53';out=B/'report-snapshot';out.mkdir(exist_ok=False)
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','dgx-127-slurm']
if Path('/tmp/pmr-v5-dgx-20261010.sock').exists():ssh[1:1]=['-S','/tmp/pmr-v5-dgx-20261010.sock']
check='import json;from pathlib import Path;p=Path('+repr(remote)+');s=json.loads((p/"evaluation_controller.json").read_text());assert s["status"]=="repair_diagnostic_complete";print("complete")'
subprocess.run(ssh+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(check))],check=True)
paths=['anchor_report','bridge_report','fit_cache/receipt.json','anchor_cache/receipt.json','bridge_cache/receipt.json','reader/freeze_receipt.json','reader/training_config.json','fit_controller.json','evaluation_controller.json','registration_publication.json','freeze_publication.json','ledger/budget.json']
paths += [f'bridge_cache/shards/{i:06d}.json' for i in range(384,640)]
archive=B/'report-snapshot.tar.gz'
with archive.open('xb') as f:subprocess.run(ssh+['bash -lc '+shlex.quote('tar -czf - -C '+remote+' '+shlex.join(paths))],stdout=f,check=True)
with tarfile.open(archive,'r:gz') as t:
 for m in t:
  p=PurePosixPath(m.name);assert not p.is_absolute() and '..' not in p.parts and (m.isdir() or m.isfile())
  dest=out/m.name
  if m.isdir():dest.mkdir(parents=True,exist_ok=True)
  else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(t.extractfile(m).read())
print(str(out))
