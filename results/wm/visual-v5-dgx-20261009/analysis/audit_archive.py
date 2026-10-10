import csv
import hashlib
import json
from pathlib import Path

root = Path('results/wm/visual-v5-dgx-20261009')
def read(name):
    return json.loads((root / name).read_text())
def meta(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

manifest = read('run/run_manifest.json')
source = manifest['configuration']['source']
assert source['git_commit'] == 'd0b4fef25e98eefb58bfba737997cb0145f27daa'
for name, digest in source['files'].items():
    assert meta(root/'source'/name)['sha256'] == digest, name
inventory = read('workspace/remote_run_inventory.json')
omitted = {}
copied = 0
for name, entry in inventory['files'].items():
    rel = Path(name).relative_to('run1')
    if rel.parts[0] == 'probe_features' and rel.suffix == '.pt':
        cache = read('run/' + str(rel.parent/'manifest.json'))
        assert cache['files'][rel.name] == entry, name
        omitted[name] = entry
    elif str(rel) == 'active.lock':
        assert entry['bytes'] == 0
    else:
        assert meta(root/'run'/rel) == entry, name
        copied += 1
assert len(omitted) == 256
for split in ('train','dev'):
    cache = read(f'run/probe_features/{split}/manifest.json')
    assert cache['status'] == 'complete' and cache['validated_records'] == 128
    assert len(cache['files']) == 128
workers = [json.loads(p.read_text()) for p in (root/'run/workers').glob('*.json')]
assert len(workers) == 8 and len({w['visible_device'] for w in workers}) == 8
for worker in workers:
    assert worker['status'] == 'complete' and worker['step_id'] == '15'
    assert len(worker['tasks']) == 2
    assert all(t['records'] == 16 and t['status'] == 'complete' for t in worker['tasks'])
assert read('run/feature_gate.json')['passed'] is False
assert read('run/profile.json')['status'] == 'complete'
assert read('run/profile.json')['throughput_gate_passed'] is True
assert read('run/feature_diagnostics/verification.json')['matches_recorded_feature_gate'] is True
assert read('run/report_status.json')['test_evaluated'] is False
assert read('workspace/controller_status.json')['stage'] == 'failed'
budget = read('run/budget.json')
assert len(budget['leases']) == 2
assert abs(budget['charged_gpuh'] - sum(x['charged_gpuh'] for x in budget['leases'])) < 1e-9
slurm = list(csv.DictReader((root/'logs/slurm-accounting-final.txt').open(), delimiter='|'))
jobs = {row['JobID']:row for row in slurm}
assert jobs['415453.14']['State'] == 'COMPLETED'
assert jobs['415453.15']['State'] == 'FAILED' and jobs['415453.15']['ExitCode'] == '1:0'
assert jobs['415453.5']['State'] == 'RUNNING'
assert read('archive_manifest.json')['checkpoint_archive'] is None
assert not list((root/'run').rglob('*.pt'))
result = dict(status='verified_partial_evidence', science_status='partial_feature_gate_failed',
    source_commit=source['git_commit'], source_files_verified=len(source['files']),
    remote_run_files=len(inventory['files']), downloaded_run_files_verified=copied,
    omitted_feature_tensors=dict(files=len(omitted),bytes=sum(x['bytes'] for x in omitted.values()),
        reason='Reproducible caches omitted under handoff policy; all original SHA256/size entries retained and matched against remote inventory.'),
    omitted_transient_files=['run1/active.lock'],
    probe_episodes=dict(train=128,dev=128,test=0), workers_verified=8,
    diagnostic_matches_recorded_gate=True, cumulative_selected_gpu_hours=budget['charged_gpuh'],
    main_checkpoints=0, checkpoint_reason='P0/adaptation/replay were not started after failed prerequisite.',
    slurm_terminal=dict(profile=jobs['415453.14'],features=jobs['415453.15']),
    existing_indexmem_status=jobs['415453.5'],
    remote_cleanup=dict(performed=False,reason='Experiment incomplete; retain scene as requested.'),
    verification_scope='Local artifact and remote-inventory reconciliation. Separate immutable GitHub readback is required and recorded outside this self-hashed archive.')
(root/'archival_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
