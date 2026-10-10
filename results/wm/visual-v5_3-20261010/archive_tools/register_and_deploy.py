import base64,concurrent.futures,hashlib,json,shlex,subprocess,tarfile,time
from pathlib import Path
B=Path('execution/visual-v5_3-20261010');R=Path('results/wm/visual-v5_3-20261010-registration');R.mkdir(parents=True,exist_ok=True)
REMOTE='/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v53'
REPO='XintongYang1010/predictive-memory-lab'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def cmd(a):return subprocess.check_output(a,text=True).strip()
def api(p):return json.loads(cmd(['gh','api','repos/'+REPO+'/'+p]))
commit=cmd(['git','rev-parse','HEAD']);assert not cmd(['git','status','--porcelain'])
paths=cmd(['git','ls-tree','-r','--name-only',commit,'--','pmr','scripts','docs','tests','START_HERE.md','pyproject.toml','requirements.txt']).splitlines()
manifest=dict(commit=commit,files={p:sha(p) for p in paths})
save(B/'source_manifest.json',manifest)
tree=api('git/trees/'+commit+'?recursive=1');assert not tree['truncated']
entries={v['path']:v for v in tree['tree'] if v['type']=='blob'}
def readback(p):
 b=api('git/blobs/'+entries[p]['sha']);data=base64.b64decode(b['content']);assert data==Path(p).read_bytes(),p
 return p,dict(git_blob_sha=entries[p]['sha'],sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:records=dict(pool.map(readback,paths))
publication=dict(verified_readback=True,source_commit=commit,plan_sha256=sha('scripts/v5_3_plan.json'),protocol_sha256=sha('docs/V5_3_EXECUTION_PROTOCOL.md'),verified_unix=time.time(),files=records)
save(B/'registration_publication.json',publication)
fresh=dict(schema='v5.3-freshness-audit-1',verified=True,test_opened=False,unseen_namespaces=['v5.3-nuisance-repair-1/fit','v5.3-nuisance-repair-1/anchor'],unseen_original_dev_pair_ids=list(range(384,640)),source_commit=commit,plan_sha256=sha('scripts/v5_3_plan.json'),verified_unix=time.time(),evidence=json.loads((B/'freshness_evidence.json').read_text()),scope='Known local project artifacts, prior verified archives, source/tests, and remote file inventory; no statement about unknown external experiments',scientific_samples_generated_before_registration=False)
save(B/'freshness_audit.json',fresh)
for name in ['source_manifest.json','registration_publication.json','freshness_audit.json','freshness_evidence.json','prior_file_inventory.json','creation.json','preflight.json','precommit-tests.log']:(R/name).write_bytes((B/name).read_bytes())
(R/'software_fixture').mkdir(exist_ok=True)
for src in [B/'software_fixture/README.md',B/'software_fixture/run_fullscale.py',B/'software_fixture/run_001/result.json']:
 if src.is_file():(R/'software_fixture'/src.name).write_bytes(src.read_bytes())
(R/'README.md').write_text('''# V5.3 preregistration and no-access record\n\nStatic-supervision S and nuisance-balanced B use the same frozen encoder, image pool, shared unlabeled FIT normalization, regularization and world weights. Both arms fit an unpenalized intercept. Six readers are frozen and published before fresh anchor and original-generator Dev384–639 evaluation. No Test or P0.\n\nThe original web conversation reviewed the design before generation, identifying the S-submatrix intercept issue. This was fixed and tested against the augmented KRR system; query normalization retains all B reference columns. Negative control squares each nuisance-direction prediction before aggregation. Source commit: '''+commit+'''\n\nValidation: 133 visual-memory tests passed before the final standalone budget boundary test; final V5.3 tests are recorded in precommit-tests.log. Full 4096-view/six-reader CPU software fixture used only software_test scenes and random 64D features. macOS NumPy emitted RuntimeWarnings; finite outputs and equivalence checks passed. This is software evidence, not scientific training or GPU throughput.\n\nAll new scientific generation occurs after this record. Prior original V5 gates and V5.1 bridge remain failed. Bootstrap intervals are conditional on one frozen FIT, not training-set replicates.\n''')
subprocess.run(['git','add',str(R)],check=True);subprocess.run(['git','commit','-m','Publish V5.3 source readback and pre-generation freshness audit'],check=True);subprocess.run(['git','push'],check=True)
registration=cmd(['git','rev-parse','HEAD'])
published=api('contents/'+str(R/'freshness_audit.json')+'?ref='+registration)
assert base64.b64decode(published['content'])==(B/'freshness_audit.json').read_bytes()
archive=B/'source-deploy.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in paths:t.add(p,arcname='code/'+p)
 for n in ['source_manifest.json','registration_publication.json','freshness_audit.json'] :t.add(B/n,arcname=n)
 t.add(R,arcname='registration')
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','dgx-127-slurm']
remote='test ! -e '+REMOTE+'/source_manifest.json && tar -xzf - -C '+REMOTE
with archive.open('rb') as f:subprocess.run(ssh+['bash -lc '+shlex.quote(remote)],stdin=f,check=True)
verify='import hashlib,json;from pathlib import Path;p=Path('+repr(REMOTE)+');m=json.loads((p/"source_manifest.json").read_text());assert all(hashlib.sha256((p/"code"/n).read_bytes()).hexdigest()==h for n,h in m["files"].items());print(json.dumps({"source_commit":m["commit"],"verified_files":len(m["files"])}))'
result=cmd(ssh+['bash -lc '+shlex.quote('python3 -c '+shlex.quote(verify))]);print(result,flush=True)
save(B/'deployment.json',dict(source_commit=commit,registration_commit=registration,verified_remote=json.loads(result),deployed_unix=time.time()))
