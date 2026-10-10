"""Read-only recovery of transport failures; reuse only complete local ranges."""
import concurrent.futures,json,shlex,subprocess,time
from pathlib import Path
import register_deploy as h
export=json.loads((h.HERE/'remote_export.json').read_text());sealed=json.loads((h.HERE/'remote_seal_receipt.json').read_text())
expected=dict(sealed['seal']['files']);expected['archive_manifest.json']=sealed['manifest_record']
code='''import hashlib,json
from pathlib import Path
p=Path(EXPORT);s=p.stat();assert not p.is_symlink() and (s.st_ino,s.st_dev,s.st_size)==IDENTITY
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==DIGEST
print(json.dumps(dict(export_reverified=True)))
'''.replace('EXPORT',repr(export['path'])).replace('IDENTITY',repr((export['inode'],export['device'],export['bytes']))).replace('DIGEST',repr(export['sha256']))
print(json.dumps(h.remote(code)),flush=True)
retry=h.HERE/'download_recovery';retry.mkdir(exist_ok=False);chunk=128*1024**2;events=[]
ranges=[(i,offset,min(chunk,export['bytes']-offset)) for i,offset in enumerate(range(0,export['bytes'],chunk))]
def download(row):
 i,offset,length=row;old=h.HERE/'download_parts'/f'{i:04d}.part'
 if old.is_file() and old.stat().st_size==length:return old
 for attempt in range(3):
  p=retry/f'{i:04d}.attempt{attempt}.part'
  code='''import sys
with open(EXPORT,'rb') as f:
 f.seek(OFFSET);remaining=LENGTH
 while remaining:
  b=f.read(min(1048576,remaining))
  if not b:raise RuntimeError('Unexpected export EOF')
  sys.stdout.buffer.write(b);remaining-=len(b)
'''.replace('EXPORT',repr(export['path'])).replace('OFFSET',str(offset)).replace('LENGTH',str(length))
  args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=60','dgx-127-slurm','bash -lc '+shlex.quote('python3 -c '+shlex.quote(code))]
  with p.open('xb') as out:result=subprocess.run(args,stdout=out,stderr=subprocess.PIPE)
  events.append(dict(part=i,attempt=attempt,returncode=result.returncode,bytes=p.stat().st_size,stderr=result.stderr.decode()))
  if result.returncode==0 and p.stat().st_size==length:
   print(json.dumps(dict(recovered_part=i,bytes=length)),flush=True);return p
  time.sleep(5)
 raise RuntimeError('Failed recovery part '+str(i))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:parts=list(pool.map(download,ranges))
archive=h.HERE/'visual-v5_5-20261010-remote-scene.tar.gz'
with archive.open('xb') as out:
 for part in parts:
  with part.open('rb') as src:
   for b in iter(lambda:src.read(1048576),b''):out.write(b)
assert archive.stat().st_size==export['bytes'] and h.sha(archive)==export['sha256']
h.helper.extract_verified(archive,h.HERE/'diagnostic-20261010-v55',expected)
h.save(h.HERE/'transport_recovery.json',dict(status='recovered_and_fully_verified',initial_failure='SSH banner exchange timeout and connection closures in8-connection export download',remote_writes=False,remote_export_hash_reverified=True,events=events,selected_parts=[str(p.relative_to(h.HERE)) for p in parts],full_archive_sha256=h.sha(archive)))
h.save(h.HERE/'local_archive_verification.json',dict(archive_sha256=h.sha(archive),archive_bytes=archive.stat().st_size,files=len(expected),all_files_verified=True,manifest_sha256=sealed['manifest_record']['sha256'],transport_recovery='transport_recovery.json'))
print(json.dumps(dict(status='complete_and_verified',files=len(expected),bytes=archive.stat().st_size)),flush=True)
