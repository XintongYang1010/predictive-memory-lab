"""Publish segmented full scene and re-read every byte/file from GitHub."""
import concurrent.futures,hashlib,json,subprocess,time
from pathlib import Path
import register_deploy as h
TAG='visual-v5_5-20261010'
RESULT=h.ROOT/'results/wm'/TAG
def main():
 archive=h.HERE/(TAG+'-remote-scene.tar.gz')
 seal=json.loads((h.HERE/'remote_seal_receipt.json').read_text())
 verified=json.loads((h.HERE/'local_archive_verification.json').read_text())
 assert verified['all_files_verified'] and h.sha(archive)==verified['archive_sha256']
 parts=h.HERE/'release-parts';parts.mkdir(exist_ok=False);items=[]
 with archive.open('rb') as src:
  i=0
  while True:
   b=src.read(512*1024**2)
   if not b:break
   p=parts/(archive.name+'.part'+str(i).zfill(3));p.write_bytes(b)
   items.append(dict(name=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()));i+=1
 meta=dict(schema='v55.split-archive.v1',archive=dict(name=archive.name,bytes=archive.stat().st_size,sha256=h.sha(archive)),parts=items,concatenation='lexical order part000, part001, ...; then verify full SHA256 and extract tar.gz',manifest_sha256=seal['manifest_record']['sha256'],release_url='https://github.com/'+h.REPO+'/releases/tag/'+TAG)
 h.save(parts/'archive_parts.json',meta);h.save(RESULT/'release_assets.json',meta)
 subprocess.run(['git','add','--','results/wm/'+TAG],cwd=h.ROOT,check=True)
 subprocess.run(['git','commit','-m','Archive V5.5 full object-intervention diagnosis and matched-batch verification'],cwd=h.ROOT,check=True)
 commit=h.cmd(['git','rev-parse','HEAD']);subprocess.run(['git','push','origin','HEAD'],cwd=h.ROOT,check=True)
 note=parts/'release-notes.md'
 note.write_text('V5.5 completed diagnostic: reader intervention lowers original horizontal MSE by29.78%; object-specific motion response remains weak and susceptible to other objects. Complete paired scene, canonical/raw caches, frozen lane model, logs, source, reports and plots. All4 steps415453.40/.41/.42/.43 completed; new0.8068233269 GPUh. No Test or P0/memory pass. Official encoder and prior V5.4 scene are pinned dependencies.\n\nDownload all `.partNNN` assets and `archive_parts.json`; concatenate parts in numeric order, verify the assembled SHA256, and extract tar.gz. Every scene member is listed in archive_manifest.json.\n\nResults: https://github.com/'+h.REPO+'/tree/'+commit+'/results/wm/'+TAG+'\n\nArchive SHA256: '+meta['archive']['sha256']+'\n')
 subprocess.run(['gh','release','create',TAG,'--repo',h.REPO,'--target',commit,'--title','V5.5: object response and reader bottleneck diagnosis','--notes-file',str(note)],cwd=h.ROOT,check=True)
 def upload(p):
  subprocess.run(['gh','release','upload',TAG,str(p),'--repo',h.REPO],cwd=h.ROOT,check=True)
  print(json.dumps(dict(uploaded=p.name)),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(upload,[parts/v['name'] for v in items]+[parts/'archive_parts.json']))
 release=h.api('releases/tags/'+TAG);readback=h.HERE/'github-readback';readback.mkdir(exist_ok=False)
 expected_assets={v['name']:v for v in items};expected_assets['archive_parts.json']=dict(name='archive_parts.json',bytes=(parts/'archive_parts.json').stat().st_size,sha256=h.sha(parts/'archive_parts.json'))
 assert not release['draft'] and {a['name'] for a in release['assets']}==set(expected_assets)
 def download(a):
  v=expected_assets[a['name']];assert a['state']=='uploaded' and a['size']==v['bytes'] and a.get('digest') in (None,'sha256:'+v['sha256'])
  p=readback/a['name']
  with p.open('xb') as f:subprocess.run(['gh','api','repos/'+h.REPO+'/releases/assets/'+str(a['id']),'-H','Accept: application/octet-stream'],stdout=f,check=True)
  assert p.stat().st_size==v['bytes'] and h.sha(p)==v['sha256'],a['name']
  print(json.dumps(dict(downloaded_and_verified=a['name'])),flush=True)
  return dict(name=a['name'],id=a['id'],bytes=v['bytes'],sha256=v['sha256'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:assets=list(pool.map(download,release['assets']))
 assert json.loads((readback/'archive_parts.json').read_text())==meta
 assembled=readback/archive.name
 with assembled.open('xb') as out:
  for v in items:
   with (readback/v['name']).open('rb') as src:
    for b in iter(lambda:src.read(1048576),b''):out.write(b)
 assert assembled.stat().st_size==meta['archive']['bytes'] and h.sha(assembled)==meta['archive']['sha256']
 expected=dict(seal['seal']['files']);expected['archive_manifest.json']=seal['manifest_record']
 h.helper.extract_verified(assembled,readback/'scene',expected)
 files=h.old.github_files(commit,[p for p in RESULT.rglob('*') if p.is_file()])
 receipt=dict(schema='v55.github.readback.v1',verified_unix=time.time(),result_commit=commit,release_url=release['html_url'],release_tag=TAG,assets=assets,archive=meta['archive'],manifest_sha256=seal['manifest_record']['sha256'],full_github_download=True,all_archive_member_hashes_verified=True,exact_inventory=True,archive_member_count=len(expected),git_files=files,cleanup_authorized_scope=h.REMOTE+' and its owned export only')
 h.save(h.HERE/'github_verification.json',receipt);h.save(RESULT/'github_verification.json',receipt)
 subprocess.run(['git','add','--','results/wm/'+TAG+'/github_verification.json'],cwd=h.ROOT,check=True)
 subprocess.run(['git','commit','-m','Verify every V5.5 GitHub archive member and result blob'],cwd=h.ROOT,check=True)
 subprocess.run(['git','push','origin','HEAD'],cwd=h.ROOT,check=True)
 reccommit=h.cmd(['git','rev-parse','HEAD']);check=h.old.github_files(reccommit,[RESULT/'github_verification.json'])
 h.save(h.HERE/'verification_receipt_readback.json',dict(commit=reccommit,files=check))
 print(json.dumps(dict(status='fully_verified',members=len(expected),git_files=len(files),release=release['html_url'],commit=reccommit)),flush=True)
if __name__=='__main__':main()
