#!/usr/bin/env python3
"""Read every curated Git blob and every Release archive member from GitHub.

Read-only GitHub access. Writes only to a fresh local readback directory; no
remote deletion, jobs, or scientific execution occurs.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import time


def sha(data):return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def safe_path(name):
    q=PurePosixPath(name)
    if not name or q.is_absolute() or '..' in q.parts or str(q)!=name:
        raise ValueError('Unsafe noncanonical relative path: '+name)
    return q


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True,help='Exact results commit already pushed to GitHub')
    parser.add_argument('--source-commit',default='875be45515276cc8559ffcdb21c7412c6048d5ad')
    parser.add_argument('--repo',default='XintongYang1010/predictive-memory-lab')
    parser.add_argument('--prefix',default='results/wm/visual-v5_3-20261010')
    parser.add_argument('--release-tag',default='visual-v5_3-20261010')
    parser.add_argument('--scene-name',default='diagnostic-20261010-v53')
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'github-readback')
    parser.add_argument('--conclusion',required=True,help='Actual reviewed conclusion; no invented pass status')
    args=parser.parse_args()
    if any(re.fullmatch('[0-9a-f]{40}',v) is None for v in (args.commit,args.source_commit)):
        raise ValueError('Use exact 40-character source and result commit hashes')
    prefix=str(safe_path(args.prefix))+'/'
    if len(safe_path(args.scene_name).parts)!=1:raise ValueError('Scene name must have one path component')
    if args.out.exists():raise FileExistsError('Existing GitHub readback preserved; choose a fresh --out')
    args.out.mkdir(parents=True)
    def api(endpoint):
        for attempt in range(3):
            result=subprocess.run(['gh','api','repos/'+args.repo+'/'+endpoint],capture_output=True)
            if result.returncode==0:return json.loads(result.stdout)
            if attempt==2:raise RuntimeError(result.stderr.decode())
            time.sleep(2)
    tree=api('git/trees/'+args.commit+'?recursive=1')
    if tree.get('truncated'):raise ValueError('GitHub tree is truncated')
    entries=[row for row in tree['tree'] if row['type']=='blob' and row['path'].startswith(prefix)]
    local_root=args.workspace/args.prefix;local_files=set()
    for path in local_root.rglob('*'):
        if path.is_symlink():raise ValueError('Curated local artifact is a symlink')
        if path.is_file():local_files.add(path.relative_to(args.workspace).as_posix())
    if not entries or {row['path'] for row in entries}!=local_files:
        raise ValueError('Curated Git tree differs from exact local file inventory')
    def fetch(entry):
        safe_path(entry['path'])
        if entry.get('mode') not in ('100644','100755'):raise ValueError('Nonregular Git artifact')
        blob=api('git/blobs/'+entry['sha'])
        if blob['encoding']!='base64':raise ValueError('Unexpected Git blob encoding')
        data=base64.b64decode(''.join(blob['content'].split()),validate=True)
        git_sha=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
        if len(data)!=entry['size'] or git_sha!=entry['sha'] or data!=(args.workspace/entry['path']).read_bytes():
            raise ValueError('Git blob content mismatch: '+entry['path'])
        path=args.out/'repo'/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
        return entry['path'],dict(git_blob_sha=git_sha,bytes=len(data),sha256=sha(data))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:records=dict(pool.map(fetch,entries))
    save(args.out/'git-verification.json',dict(commit=args.commit,verified_readback=True,
         curated_tree_exact=True,transport='GitHub Git blob API, decoded base64',files=records))
    print(json.dumps(dict(git_files_read_back=len(records),commit=args.commit)),flush=True)
    release=api('releases/tags/'+args.release_tag)
    if release['draft'] or release['tag_name']!=args.release_tag:raise ValueError('Wrong/unpublished release')
    save(args.out/'release-metadata.json',release)
    curated=args.out/'repo'/args.prefix
    meta=json.loads((curated/'release_asset.json').read_text());safe_path(meta['file'])
    if len(PurePosixPath(meta['file']).parts)!=1:raise ValueError('Asset filename must be a basename')
    assets=[row for row in release['assets'] if row['name']==meta['file']]
    if len(assets)!=1:raise ValueError('Expected one matching release archive asset')
    asset=assets[0]
    if asset['state']!='uploaded' or asset['size']!=meta['bytes'] or asset.get('digest') not in (None,'sha256:'+meta['sha256']):
        raise ValueError('Release asset metadata/digest mismatch')
    dest=args.out/meta['file']
    with dest.open('xb') as stream:
        result=subprocess.run(['gh','api','repos/'+args.repo+'/releases/assets/'+str(asset['id']),
             '-H','Accept: application/octet-stream'],stdout=stream,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stderr.decode())
    if dest.stat().st_size!=meta['bytes'] or file_sha(dest)!=meta['sha256']:
        raise ValueError('Downloaded Release archive differs from curated checksum')
    manifest_bytes=(curated/'remote_scene_manifest.json').read_bytes();manifest=json.loads(manifest_bytes)
    if manifest['source_commit']!=args.source_commit or PurePosixPath(manifest['root']).name!=args.scene_name:
        raise ValueError('Release source or scene identity mismatch')
    expected=dict(manifest['files'])
    if 'archive_manifest.json' in expected:raise ValueError('Seal manifest cannot list itself')
    for name in expected:safe_path(name)
    expected['archive_manifest.json']=dict(bytes=len(manifest_bytes),sha256=sha(manifest_bytes))
    seen=set();all_members=set();directories=0
    with tarfile.open(dest,'r|gz') as archive:
        for member in archive:
            path=PurePosixPath(member.name);canonical=str(path)
            if (canonical in all_members or path.is_absolute() or '..' in path.parts
                    or not path.parts or path.parts[0]!=args.scene_name):
                raise ValueError('Unsafe/duplicate archive member: '+member.name)
            all_members.add(canonical)
            if member.isdir():directories+=1;continue
            if not member.isfile() or member.issparse():raise ValueError('Nonregular archive member')
            name='/'.join(path.parts[1:]);record=expected.get(name)
            if record is None or member.size!=record['bytes']:raise ValueError('Unexpected archive member: '+name)
            source=archive.extractfile(member);h=hashlib.sha256();count=0
            for chunk in iter(lambda:source.read(1048576),b''):h.update(chunk);count+=len(chunk)
            if count!=record['bytes'] or h.hexdigest()!=record['sha256']:raise ValueError('Archive member hash mismatch: '+name)
            seen.add(name)
    if seen!=set(expected):raise ValueError('Archive is missing files from seal inventory')
    receipt=dict(schema='v5.3.github.archive.readback.v1',verified_unix=time.time(),result_commit=args.commit,
        source_commit=args.source_commit,
        git_artifacts=dict(verified_files=len(records),transport='GitHub Git blob API, decoded base64',curated_tree_exact=True,
            evidence_sha256=file_sha(args.out/'git-verification.json')),
        release=dict(url=release['html_url'],tag=release['tag_name'],asset_id=asset['id'],name=asset['name'],
            bytes=meta['bytes'],sha256=meta['sha256'],server_digest=asset.get('digest'),downloaded_from_github_and_rehashed=True),
        scene=dict(inventoried_files=len(manifest['files']),inventoried_bytes=sum(row['bytes'] for row in manifest['files'].values()),
            manifest_sha256=sha(manifest_bytes),all_member_hashes_match=True,exact_file_inventory=True,
            regular_file_members=len(seen),directory_members=directories,unsafe_special_duplicate_members=0),
        conclusion=args.conclusion,
        cleanup='Investigation scenes retained; no deletion or job cancellation. Original V5/V5.1 gates unchanged.')
    save(args.out/'verification-draft.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
