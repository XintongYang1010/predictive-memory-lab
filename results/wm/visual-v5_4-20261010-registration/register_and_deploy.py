#!/usr/bin/env python3
"""Publish/read back V5.4 registration, then deploy into a newly owned directory.

This helper is not scientific source. It never renders/encodes a sample, fits a
reader, submits a Slurm step, cancels a job or deletes a file. Executing it DOES
publish a scoped Git commit and create/deploy a new remote working directory.
All existing scientific artifacts are read-only. Any failure preserves state;
there is no automatic resume or replacement of existing registration outputs.
"""
import argparse
import base64
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REPOSITORY='XintongYang1010/predictive-memory-lab'
REMOTE_PARENT='/aifs4su/guhao/KVCache/WM'
REMOTE=REMOTE_PARENT+'/diagnostic-20261010-v54'
OLD_REMOTE=REMOTE_PARENT+'/diagnostic-20261010-v53'
BASELINE=3.573925619125368
OLD_MANIFEST='bfa3006ad1af90869deab7fa24d69b309bfaf76ca9b885532373145acb95ea13'
OLD_SOURCE='875be45515276cc8559ffcdb21c7412c6048d5ad'
SCHEMA='v5.4-spatial-support-1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def record(path):return dict(bytes=Path(path).stat().st_size,sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def command(args,**kwargs):
    return subprocess.check_output(args,cwd=ROOT,text=True,**kwargs).strip()


def api(path):return json.loads(command(['gh','api','repos/'+REPOSITORY+'/'+path]))


def full_hash(value):
    if re.fullmatch('[0-9a-f]{40}',value) is None:raise ValueError('Immutable 40-character Git commit required')
    return value


def github_files(commit,paths):
    """Every requested local file must equal its immutable GitHub blob bytes."""
    tree=api('git/trees/'+full_hash(commit)+'?recursive=1')
    if tree.get('truncated'):raise ValueError('GitHub tree is truncated')
    entries={x['path']:x for x in tree['tree'] if x['type']=='blob'}
    def check(path):
        path=Path(path);relative=path.resolve().relative_to(ROOT).as_posix()
        entry=entries[relative]
        if entry['mode'] not in ('100644','100755'):raise ValueError('Non-regular Git blob '+relative)
        blob=api('git/blobs/'+entry['sha'])
        if blob.get('encoding')!='base64':raise ValueError('Unexpected GitHub blob encoding')
        actual=base64.b64decode(blob['content']);expected=path.read_bytes()
        if actual!=expected:raise ValueError('GitHub/local bytes differ: '+relative)
        return relative,dict(git_blob_sha=entry['sha'],sha256=hashlib.sha256(actual).hexdigest(),bytes=len(actual))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        return dict(pool.map(check,paths))


def scene_inventory(scene):
    scene=Path(scene).resolve();manifest_path=scene/'archive_manifest.json'
    if sha(manifest_path)!=OLD_MANIFEST:raise ValueError('Wrong sealed V5.3 manifest')
    manifest=read(manifest_path)
    if manifest.get('source_commit')!=OLD_SOURCE:raise ValueError('Wrong sealed V5.3 source')
    actual={}
    for p in scene.rglob('*'):
        if p.is_symlink():raise ValueError('Symlink in sealed local scene')
        if p.is_file() and p!=manifest_path:actual[p.relative_to(scene).as_posix()]=record(p)
    if actual!=manifest['files']:raise ValueError('Local old scene inventory/hash mismatch')
    budget=read(scene/'ledger/budget.json');final=read(scene/'final_accounting.json')
    validate_budget(budget)
    if (abs(final.get('cumulative_charged_gpuh',-1)-BASELINE)>1e-12 or
            final.get('all_new_steps_complete') is not True or final.get('active_new_leases') is not False):
        raise ValueError('Old final accounting is incomplete or disagrees with exact ledger')
    import numpy as np
    with np.load(scene/'bridge_report/predictions.npz',allow_pickle=False) as z:
        if not np.array_equal(z['pair_ids'],np.arange(384,640)):raise ValueError('Old evaluated ID range differs')
    return dict(manifest_sha256=OLD_MANIFEST,files=len(actual),bytes=sum(v['bytes'] for v in actual.values()),
        ledger=record(scene/'ledger/budget.json'),final_accounting=record(scene/'final_accounting.json'),
        prior_bridge_predictions=record(scene/'bridge_report/predictions.npz'),prior_bridge_ids=[384,639])


def validate_budget(budget):
    if abs(budget['charged_gpuh']-BASELINE)>1e-12:raise ValueError('Wrong exact V5.3 baseline')
    if any(x.get('status')!='closed' for x in budget.get('controller_envelopes',[])):
        raise ValueError('Unclosed prior controller envelope')
    if any(x.get('status')=='reserved_or_interrupted' for x in budget['leases']):
        raise ValueError('Unreconciled prior lease')
    if any(k.startswith('v54_') for k in budget['stages']):raise ValueError('Prior ledger already contains V5.4')


def old_github_verification(path):
    value=read(path)
    if value.get('schema')!='v5.3.github.archive.readback.v1':raise ValueError('Final V5.3 GitHub readback receipt required')
    scene=value['scene'];release=value['release']
    if (scene.get('manifest_sha256')!=OLD_MANIFEST or scene.get('all_member_hashes_match') is not True or
        scene.get('exact_file_inventory') is not True or scene.get('unsafe_special_duplicate_members',0)!=0 or
        release.get('downloaded_from_github_and_rehashed') is not True or
        not release.get('url','').startswith('https://github.com/'+REPOSITORY+'/releases/')):
        raise ValueError('Old complete scene lacks verified GitHub release readback')
    if scene.get('inventoried_files')!=1799 or scene.get('inventoried_bytes')!=465241564:
        raise ValueError('Old scene count/bytes mismatch')
    return value


def seed_audit():
    def old_seed(split,i):return int.from_bytes(hashlib.sha256(f'v5-rgb-pairs-1/{split}/{i}'.encode()).digest()[:4],'little')
    def new_seed(schema,role,i):return int.from_bytes(hashlib.sha256(f'{schema}/{role}/{i}'.encode()).digest()[:8],'little')
    fresh={old_seed('dev',i) for i in range(640,896)}
    consumed={old_seed('dev',i) for i in range(640)}|{old_seed('train',i) for i in range(1024)}
    anchor={new_seed(SCHEMA,'anchor',i) for i in range(256)}
    old_roles=set().union(*[{new_seed(s,r,i) for i in range(256)} for s,roles in
        [('v5.3-nuisance-repair-1',('fit','anchor','software_test')),('v5-temporal-confirm-1',('fit','confirm','software_test'))] for r in roles])
    if len(fresh)!=256 or fresh&consumed or len(anchor)!=256 or anchor&old_roles:raise ValueError('Seed audit failed')
    return dict(new_bridge_unique_seed_count=256,new_bridge_collisions_with_old_dev_or_planned_train=0,
        fresh_anchor_unique_seed_count=256,anchor_collisions_with_known_old_namespaces=0,
        fit_latents='deliberate reuse of V5.3/fit IDs0..255; not claimed fresh',
        software_latents='V5.3/software_test only',scientific_rng_draws_or_images_generated=False)


def ssh_command(socket):
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3']
    socket=Path(socket)
    if socket.exists() and stat.S_ISSOCK(socket.stat().st_mode):cmd+=['-S',str(socket)]
    return cmd+['dgx-127-slurm']


REMOTE_AUDIT=r'''
import base64,hashlib,json,re,subprocess,time
from pathlib import Path
c=json.loads(base64.b64decode('CONFIG'))
parent=Path(c['parent']);old=Path(c['old']);new=Path(c['new'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rec(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
if not parent.is_dir() or parent.is_symlink() or old.is_symlink() or new.exists() or new.is_symlink():raise RuntimeError('Parent/old/new directory guard failed')
m=old/'archive_manifest.json'
if sha(m)!=c['old_manifest']:raise RuntimeError('Remote old manifest changed')
manifest=json.loads(m.read_text());actual={}
for p in old.rglob('*'):
 if p.is_symlink():raise RuntimeError('Old scene contains a symlink')
 if p.is_file() and p!=m:actual[p.relative_to(old).as_posix()]=rec(p)
if actual!=manifest['files'] or manifest['source_commit']!=c['old_source']:raise RuntimeError('Remote old scene changed')
ledger=old/'ledger/budget.json';b=json.loads(ledger.read_text())
if sha(ledger)!=c['ledger_sha256'] or abs(b['charged_gpuh']-c['baseline'])>1e-12:raise RuntimeError('Exact old ledger differs')
if any(v.get('status')!='closed' for v in b.get('controller_envelopes',[])) or any(v.get('status')=='reserved_or_interrupted' for v in b['leases']):raise RuntimeError('Prior accounting unresolved')
final=json.loads((old/'final_accounting.json').read_text())
if final.get('all_new_steps_complete') is not True or final.get('active_new_leases') is not False or abs(final['cumulative_charged_gpuh']-c['baseline'])>1e-12:raise RuntimeError('Final accounting mismatch')
def run(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(command=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
job=run(['scontrol','show','job',c['allocation_job'],'-o'])
fields=dict(v.split('=',1) for v in job['stdout'].split() if '=' in v)
if job['returncode'] or fields.get('JobState')!='RUNNING' or not fields.get('UserId','').startswith('guhao(') or fields.get('Account')!=c['account'] or fields.get('Partition')!=c['partition'] or fields.get('QOS')!=c['qos'] or 'gres/gpu=8' not in fields.get('AllocTRES',''):raise RuntimeError('Existing eight-GPU allocation identity is not valid')
indexmem=run(['scontrol','show','step',c['allocation_job']+'.5','-o'])
inventory={}
for folder in sorted(parent.iterdir()):
 if folder.is_symlink() or not folder.is_dir():continue
 if folder.name not in ('run1','continuation-20261010') and not folder.name.startswith('diagnostic-'):continue
 rows=[]
 for p in sorted(folder.rglob('*')):
  if p.is_symlink():continue
  if not p.is_file():continue
  rel=p.relative_to(parent).as_posix();rows.append(rel)
  if '/bridge_cache/shards/' in rel and p.stem.isdigit() and 640<=int(p.stem)<=895:raise RuntimeError('New bridge ID already present in known remote cache inventory')
 inventory[folder.name]=rows
print(json.dumps(dict(checked_unix=time.time(),remote_root_absent=True,old_scene_verified=True,old_manifest_sha256=sha(m),old_files=len(actual),old_bytes=sum(x['bytes'] for x in actual.values()),old_ledger=rec(ledger),source_commit=manifest['source_commit'],parent_children=[dict(name=p.name,directory=p.is_dir(),symlink=p.is_symlink()) for p in sorted(parent.iterdir())],known_scientific_file_inventory=inventory,allocation=job,indexmem_observation=indexmem,disk=run(['df','-h',str(parent)]),new_slurm_steps=False,remote_files_written=False)))
'''


REMOTE_CREATE=r'''
import base64,hashlib,json,time,uuid
from pathlib import Path
c=json.loads(base64.b64decode('CONFIG'));p=Path(c['new']);old=Path(c['old'])
if p.exists() or p.is_symlink() or p.parent.is_symlink():raise RuntimeError('New path already exists; preserve it')
data=(old/'ledger/budget.json').read_bytes();b=json.loads(data)
if hashlib.sha256(data).hexdigest()!=c['ledger_sha256'] or abs(b['charged_gpuh']-c['baseline'])>1e-12:raise RuntimeError('Old budget changed since audit')
if hashlib.sha256((old/'archive_manifest.json').read_bytes()).hexdigest()!=c['old_manifest']:raise RuntimeError('Old manifest changed since audit')
if any(v.get('status')!='closed' for v in b.get('controller_envelopes',[])) or any(v.get('status')=='reserved_or_interrupted' for v in b['leases']):raise RuntimeError('Unreconciled old budget')
p.mkdir(mode=0o700);s=p.stat()
creation=dict(root=str(p),created_unix=time.time(),inode=s.st_ino,device=s.st_dev,uid=s.st_uid,token=uuid.uuid4().hex,preexisted=False,parent_preserved=True)
with (p/'creation.json').open('x') as f:f.write(json.dumps(creation,indent=2)+'\n')
for name in ('code','logs','ledger'):(p/name).mkdir()
for name in ('inherited_budget.json','budget.json'):
 with (p/'ledger'/name).open('xb') as f:f.write(data)
with (p/'ledger/inheritance.json').open('x') as f:f.write(json.dumps(dict(path=str(old/'ledger/budget.json'),sha256=c['ledger_sha256'],charged_gpuh=c['baseline'],old_archive_manifest_sha256=c['old_manifest'],extra_management_gpu_charge=0),indent=2)+'\n')
print(json.dumps(creation))
'''


REMOTE_DEPLOY=r'''
import base64,hashlib,json,sys,tarfile
from pathlib import Path,PurePosixPath
c=json.loads(base64.b64decode('CONFIG'));root=Path(c['new']);creation=json.loads((root/'creation.json').read_text());s=root.stat()
if creation!=c['creation'] or creation['inode']!=s.st_ino or creation['device']!=s.st_dev or creation['uid']!=s.st_uid:raise RuntimeError('New directory identity changed')
if (root/'source_manifest.json').exists():raise RuntimeError('Source already deployed; do not overwrite')
seen=set()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as tar:
 for member in tar:
  rel=PurePosixPath(member.name)
  if not member.isfile() or rel.is_absolute() or '..' in rel.parts or member.name in seen:raise RuntimeError('Unsafe/duplicate deployment member')
  seen.add(member.name);dst=root/rel
  for component in dst.parents:
   if component==root:break
   if component.is_symlink():raise RuntimeError('Symlink in deployment path')
  dst.parent.mkdir(parents=True,exist_ok=True)
  with dst.open('xb') as f:
   src=tar.extractfile(member)
   for block in iter(lambda:src.read(1048576),b''):f.write(block)
  dst.chmod(member.mode & 0o777)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((root/'source_manifest.json').read_text())
if m['commit']!=c['source_commit']:raise RuntimeError('Wrong source commit')
for name,digest in m['files'].items():
 if sha(root/'code'/name)!=digest:raise RuntimeError('Deployed source checksum mismatch '+name)
for name,value in c['bundle_files'].items():
 p=root/name
 if p.stat().st_size!=value['bytes'] or sha(p)!=value['sha256']:raise RuntimeError('Deployed registration checksum mismatch '+name)
if sha(root/'ledger/budget.json')!=c['ledger_sha256'] or sha(root/'ledger/inherited_budget.json')!=c['ledger_sha256']:raise RuntimeError('Ledger copy changed during deployment')
print(json.dumps(dict(status='deployed_verified_no_science_started',source_commit=m['commit'],verified_source_files=len(m['files']),verified_registration_files=len(c['bundle_files']),creation=creation,ledger_sha256=c['ledger_sha256'],new_slurm_steps=False,existing_files_modified=False)))
'''


def remote(ssh,script,config,stdin=None):
    code=script.replace('CONFIG',base64.b64encode(json.dumps(config).encode()).decode())
    argv=ssh+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(code))]
    result=subprocess.run(argv,stdin=stdin,capture_output=True,text=stdin is None,timeout=600)
    if result.returncode:
        err=result.stderr.decode() if isinstance(result.stderr,bytes) else result.stderr
        raise RuntimeError('Remote operation failed; preserve any created state: '+err)
    return json.loads(result.stdout)


def copy_new(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(Path(source).read_bytes())


def run(args):
    source=full_hash(args.source_commit or command(['git','rev-parse','HEAD']))
    prior_ref=full_hash(args.prior_verification_ref or source)
    plan=read(ROOT/'scripts/v5_4_plan.json')
    if (plan['schema']!=SCHEMA or plan['narrow_reference']['manifest_sha256']!=OLD_MANIFEST or
        abs(plan['execution']['inherited_cumulative_gpuh']-BASELINE)>1e-12):raise ValueError('Unexpected registered plan')
    for path in (args.precommit_log,args.web_review,args.profile_reuse,args.prior_verification):
        if not path.is_file() or path.is_symlink():raise ValueError('Required existing evidence file missing: '+str(path))
    previous=old_github_verification(args.prior_verification)
    prior_local=scene_inventory(args.prior_scene)
    old_readback=github_files(prior_ref,[args.prior_verification])
    # Preserve unrelated work/staging: validate frozen paths only, never reset.
    names=command(['git','ls-tree','-r','--name-only',source,'--','pmr','scripts','docs','tests','START_HERE.md','pyproject.toml','requirements.txt']).splitlines()
    if not names or 'scripts/v5_4_plan.json' not in names:raise ValueError('Source commit lacks V5.4')
    paths=[ROOT/name for name in names]
    for path in paths:
        if not path.is_file() or path.is_symlink():raise ValueError('Nonregular/missing source '+str(path))
        frozen=subprocess.check_output(['git','show',source+':'+path.relative_to(ROOT).as_posix()],cwd=ROOT)
        if path.read_bytes()!=frozen:raise ValueError('Working source differs from frozen commit: '+str(path))
    source_readback=github_files(source,paths)
    output=args.work_dir
    output.mkdir(parents=True,exist_ok=False)
    save(output/'source_manifest.json',dict(commit=source,files={p:source_readback[p]['sha256'] for p in names}))
    publication=dict(verified_readback=True,source_commit=source,plan_sha256=sha(ROOT/'scripts/v5_4_plan.json'),
        protocol_sha256=sha(ROOT/'docs/V5_4_EXECUTION_PROTOCOL.md'),verified_unix=time.time(),files=source_readback)
    save(output/'registration_publication.json',publication)
    save(output/'prior_scene_verification_readback.json',dict(commit=prior_ref,files=old_readback,
        complete_scene_manifest_sha256=OLD_MANIFEST,local_full_scene_verification=prior_local,
        release=previous['release']))
    ssh=ssh_command(args.ssh_socket)
    config=dict(parent=REMOTE_PARENT,old=OLD_REMOTE,new=REMOTE,old_manifest=OLD_MANIFEST,old_source=OLD_SOURCE,
        baseline=BASELINE,ledger_sha256=prior_local['ledger']['sha256'],allocation_job=args.allocation_job,
        account=args.account,partition=args.partition,qos=args.qos)
    audited=remote(ssh,REMOTE_AUDIT,config)
    save(output/'preflight.json',audited)
    evidence=dict(prior_local_scene=prior_local,prior_verification=record(args.prior_verification),
        prior_verification_commit=prior_ref,remote_audit=record(output/'preflight.json'),
        old_original_dev_ids_conservatively_consumed=[0,639],seed_hash_only_audit=seed_audit(),
        source_test_paths=[p for p in names if p.startswith('tests/')],source_manifest=record(output/'source_manifest.json'),
        scope='Known local sealed artifacts and source/tests, exact old remote archive plus known WM cache filename inventory; no claim about unknown external experiments',
        scientific_samples_generated=False)
    save(output/'freshness_evidence.json',evidence)
    freshness=dict(schema='v5.4-freshness-audit-1',verified=True,test_opened=False,
        unseen_namespaces=[SCHEMA+'/anchor'],fit_latents_reused_from='v5.3-nuisance-repair-1/fit',
        unseen_original_dev_pair_ids=list(range(640,896)),source_commit=source,plan_sha256=publication['plan_sha256'],
        verified_unix=time.time(),evidence=evidence,scientific_samples_generated_before_registration=False,
        new_wide_fit_not_generated_before_registration=True,scope=evidence['scope'])
    save(output/'freshness_audit.json',freshness)
    creation=remote(ssh,REMOTE_CREATE,config)
    save(output/'creation.json',creation)
    registration=ROOT/'results/wm/visual-v5_4-20261010-registration'
    registration.mkdir(parents=True,exist_ok=False)
    for path in output.iterdir():
        if path.is_file():copy_new(path,registration/path.name)
    for path,name in [(args.precommit_log,'precommit-tests.log'),(args.web_review,'web_design_review.md'),
        (args.profile_reuse,'profile_reuse.json'),(args.prior_verification,'prior_v53_github_verification.json'),
        (Path(__file__),'register_and_deploy.py')]:copy_new(path,registration/name)
    fixture=HERE/'software_fixture'
    if fixture.is_dir():
        for path in sorted(fixture.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in ('.py','.json','.log','.md'):
                copy_new(path,registration/'software_fixture'/path.relative_to(fixture))
    (registration/'README.md').write_text(
        '# V5.4 registration: matched spatial coverage\n\n'
        'N reuses the verified V5.3 narrow FIT features. W replays the same latent draws with only wider target centers. '
        'Both readers are refit with joint N+W unlabeled FIT kernel normalization, each4096 supervised rows, fixed quadratic alpha7.8125e-7 and a free intercept. '
        'The sole primary is fresh Dev640–895 N−W horizontal MSE; wide-anchor/negative controls and registered narrow-envelope subgroups are supporting evidence. '
        'No Test, P0, tuning or old-gate rewrite.\n\n'
        'Complete new models and the joint basis must be separately published and read back before either evaluation is generated. '
        'This record authorizes the bounded FIT stage only after every prerequisite; it does not start a Slurm step.\n\n'
        'Inherited exact charge3.573925619125368 GPUh; two1.2GPUh envelopes cap this version at2.4 additional GPUh. '
        'No extra GPU profile/query, automatic retry or borrowing from the54GPUh reserve. profile_reuse.json is an extrapolation from recorded V5.3 timing, not a fresh GPU measurement.\n\n'
        'Precommit tests, software fixtures and web review are attached as their actual logs/artifacts; they are not scientific outputs. '
        'The source readback pins source commit '+source+'. Whole-model freezes remain pending.\n')
    relative=registration.relative_to(ROOT).as_posix()
    subprocess.run(['git','add','--',relative],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','Publish V5.4 source readback and bounded coverage registration','--',relative],cwd=ROOT,check=True)
    subprocess.run(['git','push'],cwd=ROOT,check=True)
    registration_commit=full_hash(command(['git','rev-parse','HEAD']))
    registered_files=[p for p in sorted(registration.rglob('*')) if p.is_file()]
    registration_readback=github_files(registration_commit,registered_files)
    save(output/'registration_artifact_readback.json',dict(verified_readback=True,commit=registration_commit,
        verified_unix=time.time(),files=registration_readback))
    bundle_files={}
    archive=output/'source-deploy.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        def add(path,name,executable=False):
            data=Path(path).read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o755 if executable else 0o644
            tar.addfile(info,io.BytesIO(data))
            if not name.startswith('code/'):bundle_files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        for p in paths:add(p,'code/'+p.relative_to(ROOT).as_posix(),bool(p.stat().st_mode & 0o111))
        for name in ('source_manifest.json','registration_publication.json','freshness_audit.json','freshness_evidence.json',
                     'prior_scene_verification_readback.json','registration_artifact_readback.json','preflight.json'):
            add(output/name,name)
        for p in registered_files:add(p,'registration/'+p.relative_to(registration).as_posix())
    config.update(source_commit=source,creation=creation,bundle_files=bundle_files)
    with archive.open('rb') as stream:deployed=remote(ssh,REMOTE_DEPLOY,config,stdin=stream)
    save(output/'deployment.json',dict(source_commit=source,registration_commit=registration_commit,
        deployed_unix=time.time(),archive=record(archive),verified_remote=deployed,
        next_action='Root may invoke the frozen controller FIT stage; helper submitted no job.'))
    print(json.dumps(dict(status='registered_and_deployed_no_science_started',work_dir=str(output),
        source_commit=source,registration_commit=registration_commit,remote_root=REMOTE,creation=creation),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',help='Frozen 40-character source commit; default local HEAD')
    parser.add_argument('--prior-scene',type=Path,required=True,help='Complete locally downloaded sealed V5.3 scene')
    parser.add_argument('--prior-verification',type=Path,default=ROOT/'results/wm/visual-v5_3-20261010-verification.json')
    parser.add_argument('--prior-verification-ref',help='Immutable GitHub commit containing final old verification receipt; default source commit')
    parser.add_argument('--work-dir',type=Path,default=HERE/('registration_attempt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    parser.add_argument('--precommit-log',type=Path,default=HERE/'precommit-tests.log')
    parser.add_argument('--web-review',type=Path,default=HERE/'web_design_review.md')
    parser.add_argument('--profile-reuse',type=Path,default=HERE/'profile_reuse.json')
    parser.add_argument('--ssh-socket',default='/tmp/pmr-v5-dgx-20261010.sock')
    parser.add_argument('--allocation-job',default='415453')
    parser.add_argument('--account',default='hansirui')
    parser.add_argument('--partition',default='llm')
    parser.add_argument('--qos',default='llm')
    args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        if args.work_dir.is_dir() and not (args.work_dir/'failure.json').exists():
            save(args.work_dir/'failure.json',dict(status='failed_preserve_state',error=str(exc),time_unix=time.time(),
                remote_root=REMOTE,no_automatic_retry=True,no_cleanup_performed=True))
        raise


if __name__=='__main__':main()
