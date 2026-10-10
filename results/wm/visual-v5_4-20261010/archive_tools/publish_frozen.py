#!/usr/bin/env python3
"""Publish and fully read back the V5.4 N/W freeze; --execute is required."""
import argparse
import base64
import gc
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
import time
from helper_common import (BASE, REPO_ROOT, REMOTE, GITHUB, PLAN, PROTOCOL, api,
    command, completed_gate, download_snapshot, extract_verified, frozen_source,
    inventory, read, save_new, sha, ssh)

TAG = 'visual-v5_4-fit-20261010'
RESULT = REPO_ROOT / 'results/wm/visual-v5_4-20261010-freeze'


def validate_complete_freeze(directory, identity):
    """No fitting/scoring: validate all saved coefficients, basis arrays and receipts."""
    directory = directory.resolve()
    freeze = read(directory/'reader/freeze_receipt.json')
    if (freeze['cache_identity']['source_commit'] != identity['source_commit'] or
            freeze['cache_identity']['plan_sha256'] != identity['plan_sha256'] or
            freeze['narrow_reference']['manifest_sha256'] != identity['narrow_manifest_sha256']):
        raise ValueError('Frozen model, source registration, or historical N identity differs')
    if sha(directory/'source_manifest.json') != sha(BASE/'source_manifest.json'):
        raise ValueError('Remote source manifest differs from local source freeze')
    registration = read(directory/'registration_publication.json')
    if (registration.get('verified_readback') is not True or
            any(registration.get(k) != identity[k] for k in ('source_commit','plan_sha256','protocol_sha256'))):
        raise ValueError('Source/protocol GitHub readback is incomplete')
    if (freeze['fit_receipt_utf8']['W'].encode() != (directory/'fit_cache/receipt.json').read_bytes() or
            hashlib.sha256(freeze['fit_receipt_utf8']['N'].encode()).hexdigest() != freeze['narrow_reference']['receipt_sha256']):
        raise ValueError('Both original FIT receipts must be present exactly')
    # This is the source-hash-checked reader, not a new fit. The saved joint
    # reference contains 512 worlds / 8192 views, including all N and W features.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO_ROOT/'scripts'))
    import v5_4_reader
    arrays, verified = v5_4_reader.load_frozen(directory/'reader')
    summary = dict(union_feature_shape=list(arrays['fit_features'].shape),
                   union_target_shape=list(arrays['fit_targets'].shape),
                   reference_views=len(arrays['fit_view_indices']),
                   coefficient_shapes={arm:list(arrays[arm+'_concat__coefficients'].shape) for arm in ('N','W')},
                   normalization_reference_views=len(arrays['concat__linear_columns']),
                   both_fit_receipts_verified=True,
                   union_reference_fully_verified=True,
                   matched_label_audit=verified['matched_label_audit'])
    del arrays
    gc.collect()
    return freeze, summary


def git_publish(paths, message):
    """Only commit our own paths, preserving unrelated staged/unstaged edits."""
    relative = [str(p.relative_to(REPO_ROOT)) for p in paths]
    subprocess.run(['git','add','--'] + relative, cwd=REPO_ROOT, check=True)
    subprocess.run(['git','commit','--only','-m',message,'--'] + relative, cwd=REPO_ROOT, check=True)
    subprocess.run(['git','push'], cwd=REPO_ROOT, check=True)
    return command(['git','rev-parse','HEAD'])


def verify_git_files(directory, commit):
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            rel = path.relative_to(REPO_ROOT).as_posix()
            blob = api('contents/'+rel+'?ref='+commit)
            if blob.get('encoding') != 'base64' or base64.b64decode(blob['content']) != path.read_bytes():
                raise ValueError('GitHub Git receipt differs: '+rel)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        print('Preview: verify both FIT steps; download complete N/W reader; publish Release '+TAG+
              '; re-download all basis/coefficients/receipts; publish and read back the release receipt; only then install remote freeze_publication.json.')
        return
    identity = frozen_source()
    for path in (RESULT, BASE/'frozen', BASE/'frozen-github-readback', BASE/'freeze_publication.json'):
        if path.exists():
            raise FileExistsError('Preserve existing publication evidence; inspect before retry: '+str(path))
    gate = completed_gate(['fit'])
    if gate['source_commit'] != identity['source_commit']:
        raise ValueError('FIT controller source differs from deployment')
    save_new(BASE/'freeze-step-verification.json', gate)
    source_archive = BASE/'frozen-remote-snapshot.tar.gz'
    directory = BASE/'frozen'
    paths = ['reader','fit_cache/receipt.json','fit_cache/manifest.json','source_manifest.json',
             'registration_publication.json','freshness_audit.json','fit_controller.json','fit_extraction.json']
    download_snapshot(paths, source_archive, directory)
    freeze, summary = validate_complete_freeze(directory, identity)
    # The original N receipt already lives verbatim inside the freeze. Expose
    # a byte-identical standalone copy in the Release for simple offline review.
    narrow = directory/'narrow_reference'
    narrow.mkdir()
    (narrow/'receipt.json').write_bytes(freeze['fit_receipt_utf8']['N'].encode())
    save_new(narrow/'identity.json', freeze['narrow_reference'])
    save_new(directory/'freeze_step_verification.json', gate)
    records = inventory(directory)
    archive = BASE/'visual-v5_4-20261010-frozen-readers.tar.gz'
    with archive.open('xb') as output, tarfile.open(fileobj=output, mode='w:gz') as tar:
        for name in records:
            tar.add(directory/name, arcname=name, recursive=False)
    meta = dict(file=archive.name, sha256=sha(archive), bytes=archive.stat().st_size,
                tag=TAG, source_commit=identity['source_commit'])
    RESULT.mkdir(parents=True, exist_ok=False)
    for name in ('reader/freeze_receipt.json','reader/training_config.json','fit_cache/receipt.json',
                 'fit_controller.json','narrow_reference/receipt.json','narrow_reference/identity.json',
                 'freeze_step_verification.json'):
        out = RESULT/name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((directory/name).read_bytes())
    save_new(RESULT/'release_asset.json', meta)
    save_new(RESULT/'complete_model_validation.json', summary)
    (RESULT/'README.md').write_text('# V5.4 complete N/W reader freeze\n\n'
        'The narrow and wide readers each use 4096 FIT views. Their common normalization reference '
        'contains all 8192 N/W views. The Release asset includes the complete union feature basis, '
        'coefficients, intercepts, normalization moments, labels, and exact original N/W FIT receipts. '
        'This is a model freeze before fresh anchor/bridge generation, not an evaluation result. '
        'Evaluation release requires complete asset and receipt readback.\n')
    freeze_commit = git_publish([RESULT], 'Publish complete V5.4 N/W reader freeze before evaluation')
    notes = BASE/'freeze-release-notes.md'
    notes.write_text('Complete V5.4 N/W frozen readers with the shared 8192-view reference, both FIT '
                     'receipts, all coefficients and normalization. No evaluation data accessed.\n\n'
                     'Source: '+identity['source_commit']+'\nAsset SHA256: '+meta['sha256']+'\n')
    subprocess.run(['gh','release','create',TAG,str(archive),'--repo',GITHUB,'--target',freeze_commit,
        '--title','V5.4 frozen N/W FIT readers before evaluation','--notes-file',str(notes)],
        cwd=REPO_ROOT,check=True)
    release = api('releases/tags/'+TAG)
    matching = [v for v in release['assets'] if v['name']==archive.name]
    if len(matching)!=1 or matching[0]['size']!=meta['bytes'] or matching[0]['state']!='uploaded':
        raise ValueError('Uploaded model asset metadata differs')
    asset = matching[0]
    readback = BASE/'frozen-github-readback.tar.gz'
    with readback.open('xb') as f:
        subprocess.run(['gh','api','repos/'+GITHUB+'/releases/assets/'+str(asset['id']),
                        '-H','Accept: application/octet-stream'],stdout=f,check=True)
    if readback.stat().st_size!=meta['bytes'] or sha(readback)!=meta['sha256']:
        raise ValueError('Complete Release asset failed byte/hash readback')
    extracted = BASE/'frozen-github-readback'
    extract_verified(readback, extracted, records)
    _, readback_summary = validate_complete_freeze(extracted, identity)
    if readback_summary != summary:
        raise ValueError('Re-downloaded joint reference/model validation differs')
    verify_git_files(RESULT, freeze_commit)
    # Require no evaluation generation even after the full upload/readback delay.
    if completed_gate(['fit']) != gate:
        raise ValueError('Remote FIT/step state changed during publication')
    publication = dict(verified_readback=True, frozen_reader_sha256=sha(directory/'reader/freeze_receipt.json'),
        source_commit=identity['source_commit'], plan_sha256=identity['plan_sha256'],
        protocol_sha256=identity['protocol_sha256'], verified_unix=time.time(), freeze_commit=freeze_commit,
        release_url=release['html_url'],asset_id=asset['id'],asset_sha256=meta['sha256'],asset_bytes=meta['bytes'],
        all_model_and_receipt_files_downloaded_and_verified=True,
        both_fit_receipts_verified=True,union_reference_fully_verified=True,
        narrow_reference=freeze['narrow_reference'],complete_model_validation=summary,files=records)
    save_new(BASE/'freeze_publication.json', publication)
    save_new(RESULT/'freeze_publication.json', publication)
    pub_commit = git_publish([RESULT/'freeze_publication.json'], 'Verify complete V5.4 N/W freeze before evaluation release')
    verify_git_files(RESULT, pub_commit)
    # The only remote mutation here is installing this fully verified release
    # receipt. noclobber prevents replacing an earlier release on any failure.
    shell = 'set -o noclobber; cat > '+shlex.quote(REMOTE+'/freeze_publication.json')
    with (BASE/'freeze_publication.json').open('rb') as f:
        subprocess.run(ssh()+['bash -lc '+shlex.quote(shell)],stdin=f,check=True)
    print(json.dumps(publication,indent=2),flush=True)


if __name__ == '__main__':
    main()
