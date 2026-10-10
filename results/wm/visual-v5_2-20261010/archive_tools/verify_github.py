"""Read every curated Git blob and every Release archive file back from GitHub."""
import argparse,base64,concurrent.futures,hashlib,json,subprocess,tarfile,time
from pathlib import Path,PurePosixPath

p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
repo='XintongYang1010/predictive-memory-lab'
prefix='results/wm/visual-v5_2-20261010/'
base=Path('execution/visual-v5_2-20261010');out=base/'github-readback';out.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def api(endpoint):
 for attempt in range(3):
  r=subprocess.run(['gh','api','repos/'+repo+'/'+endpoint],capture_output=True)
  if r.returncode==0:return json.loads(r.stdout)
  if attempt==2:raise RuntimeError(r.stderr.decode())
  time.sleep(2)
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
tree=api('git/trees/'+a.commit+'?recursive=1');assert not tree['truncated']
entries=[v for v in tree['tree'] if v['type']=='blob' and v['path'].startswith(prefix)]
assert {v['path'] for v in entries}=={str(p) for p in Path(prefix).rglob('*') if p.is_file()}
def fetch(entry):
 blob=api('git/blobs/'+entry['sha']);assert blob['encoding']=='base64'
 data=base64.b64decode(blob['content']);assert len(data)==entry['size']
 assert data==Path(entry['path']).read_bytes(),entry['path']
 dest=out/'repo'/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 return entry['path'],dict(git_blob_sha=entry['sha'],bytes=len(data),sha256=sha(data))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:records=dict(pool.map(fetch,entries))
save(out/'git-verification.json',dict(commit=a.commit,verified_readback=True,curated_tree_exact=True,transport='GitHub Git blob API, decoded base64',files=records))
print(json.dumps(dict(git_files_read_back=len(records),commit=a.commit)),flush=True)
release=api('releases/tags/visual-v5_2-20261010');assert not release['draft']
save(out/'release-metadata.json',release)
meta=json.loads((out/'repo'/prefix/'release_asset.json').read_text())
asset=[v for v in release['assets'] if v['name']==meta['file']]
assert len(asset)==1;asset=asset[0]
assert asset['state']=='uploaded' and asset['size']==meta['bytes']
assert asset.get('digest') in (None,'sha256:'+meta['sha256'])
dest=out/meta['file']
with dest.open('xb') as f:
 r=subprocess.run(['gh','api','repos/'+repo+'/releases/assets/'+str(asset['id']),'-H','Accept: application/octet-stream'],stdout=f,stderr=subprocess.PIPE)
assert r.returncode==0,r.stderr.decode()
assert dest.stat().st_size==meta['bytes'] and sha(dest.read_bytes())==meta['sha256']
manifest_bytes=(out/'repo'/prefix/'remote_scene_manifest.json').read_bytes();manifest=json.loads(manifest_bytes)
expected=dict(manifest['files']);expected['archive_manifest.json']=dict(bytes=len(manifest_bytes),sha256=sha(manifest_bytes))
seen=set();all_members=set();directories=0
with tarfile.open(dest,'r|gz') as archive:
 for m in archive:
  q=PurePosixPath(m.name)
  assert m.name not in all_members;all_members.add(m.name)
  assert not q.is_absolute() and '..' not in q.parts and q.parts[0]=='diagnostic-20261010-v52'
  if m.isdir():directories+=1;continue
  assert m.isfile() and not m.issparse()
  name='/'.join(q.parts[1:]);assert name in expected,name
  data=archive.extractfile(m).read();assert len(data)==expected[name]['bytes'] and sha(data)==expected[name]['sha256'],name
  seen.add(name)
assert seen==set(expected)
receipt=dict(schema='v5.2.github.archive.readback.v1',verified_unix=time.time(),result_commit=a.commit,
 git_artifacts=dict(verified_files=len(records),transport='GitHub Git blob API, decoded base64',curated_tree_exact=True,evidence_sha256=sha((out/'git-verification.json').read_bytes())),
 release=dict(url=release['html_url'],tag=release['tag_name'],asset_id=asset['id'],name=asset['name'],bytes=meta['bytes'],sha256=meta['sha256'],server_digest=asset.get('digest'),downloaded_from_github_and_rehashed=True),
 scene=dict(inventoried_files=len(manifest['files']),inventoried_bytes=sum(v['bytes'] for v in manifest['files'].values()),manifest_sha256=sha(manifest_bytes),all_member_hashes_match=True,exact_file_inventory=True,regular_file_members=len(seen),directory_members=directories,unsafe_special_duplicate_members=0),
 conclusion='Yellow nuisance intervention reduced horizontal MSE by 85.63% on known bridge scenes; original gates unchanged, Test unopened, P0 not started.',
 cleanup='Remote scientific failure investigation scenes retained; existing WM and IndexMem preserved.')
save(out/'verification-draft.json',receipt)
print(json.dumps(receipt,indent=2),flush=True)
