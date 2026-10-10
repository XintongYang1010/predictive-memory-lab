"""Local execution helpers; importing this module performs no I/O or networking."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]
REMOTE = '/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v54'
NARROW = '/aifs4su/guhao/KVCache/WM/diagnostic-20261010-v53'
PYTHON = '/aifs4su/guhao/KVCache/WM/.venv/bin/python'
ASSETS = '/aifs4su/guhao/KVCache/WM/assets'
GITHUB = 'XintongYang1010/predictive-memory-lab'
PLAN = 'scripts/v5_4_plan.json'
PROTOCOL = 'docs/V5_4_EXECUTION_PROTOCOL.md'
STAGES = {'fit': ['v54_fit', 'v54_fit_reader'],
          'evaluation': ['v54_eval', 'v54_anchor_reader', 'v54_bridge_reader']}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name.rstrip('/'):
        raise ValueError('Unsafe or noncanonical archive path: ' + name)
    return p


def ssh():
    args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    socket = Path('/tmp/pmr-v5-dgx-20261010.sock')
    if socket.exists():
        args += ['-S', str(socket)]
    return args + ['dgx-127-slurm']


def remote_python(code, **kwargs):
    return subprocess.run(ssh() + ['bash -lc ' + shlex.quote('python3 -c ' + shlex.quote(code))],
                          check=True, **kwargs)


def command(args):
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()


def api(path):
    return json.loads(command(['gh', 'api', 'repos/' + GITHUB + '/' + path]))


def frozen_source():
    deployment = read(BASE / 'deployment.json')
    manifest = read(BASE / 'source_manifest.json')
    commit = deployment['source_commit']
    if re.fullmatch('[0-9a-f]{40}', commit) is None or manifest['commit'] != commit:
        raise ValueError('Deployment/source freeze mismatch')
    for name, digest in manifest['files'].items():
        safe_name(name)
        if sha(REPO_ROOT / name) != digest:
            raise ValueError('Local frozen source changed: ' + name)
    plan = read(REPO_ROOT / PLAN)
    return dict(source_commit=commit, plan_sha256=manifest['files'][PLAN],
                protocol_sha256=manifest['files'][PROTOCOL],
                narrow_manifest_sha256=plan['narrow_reference']['manifest_sha256'])


def completed_gate(actions):
    """Read-only remote check: every expected science step must be COMPLETED/0:0."""
    code = '''import json,re,subprocess
from pathlib import Path
p=Path(REMOTE); expected=STAGES; records={}; commits=set()
for action in ACTIONS:
 s=json.loads((p/(action+'_controller.json')).read_text())
 commits.add(s['source_commit'])
 status='reader_frozen_awaiting_publication' if action=='fit' else 'repair_diagnostic_complete'
 assert s['status']==status and [j['stage'] for j in s['jobs']]==expected[action]
 for j in s['jobs']:
  assert j['status']=='complete' and j['returncode']==0
  ids={re.search(r'_(\d+\.\d+)\.log$', n).group(1) for n in j['logs'] if re.search(r'_(\d+\.\d+)\.log$',n)}
  assert len(ids)==1, ('Ambiguous exact step',j)
  step=ids.pop()
  rows=subprocess.check_output(['sacct','-j',step,'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True).splitlines()
  matches=[r.split('|') for r in rows if r.split('|')[0]==step]
  assert len(matches)==1 and matches[0][1:3]==['COMPLETED','0:0'], (step,rows)
  records[j['stage']]={'step_id':step,'state':matches[0][1],'exit_code':matches[0][2]}
 if action=='fit' and ACTIONS==['fit']:
  assert not (p/'anchor_cache').exists() and not (p/'bridge_cache').exists() and not (p/'evaluation_controller.json').exists()
assert len(commits)==1, 'Controller source commits differ'
assert len({v['step_id'] for v in records.values()})==len(records), 'Science steps must be distinct'
print(json.dumps({'source_commit':commits.pop(),'steps':records}))
'''.replace('REMOTE', repr(REMOTE)).replace('STAGES', repr(STAGES)).replace('ACTIONS', repr(actions))
    return json.loads(remote_python(code, capture_output=True, text=True).stdout)


def inventory(directory):
    result = {}
    for p in sorted(Path(directory).rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlinks are forbidden: ' + str(p))
        if p.is_file():
            result[p.relative_to(directory).as_posix()] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
        elif not p.is_dir():
            raise ValueError('Special file: ' + str(p))
    return result


def extract_verified(archive, dest, expected):
    """No links, traversal, duplicate names, overwritten files, or extra payload."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    seen, files = set(), set()
    with tarfile.open(archive, 'r:gz') as t:
        for member in t:
            path = safe_name(member.name)
            key = str(path)
            if key in seen or not (member.isfile() or member.isdir()):
                raise ValueError('Duplicate/link/special tar entry: ' + member.name)
            seen.add(key)
            out = dest / key
            if member.isdir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            if key not in expected or member.size != expected[key]['bytes']:
                raise ValueError('Unexpected file or size: ' + key)
            files.add(key)
            out.parent.mkdir(parents=True, exist_ok=True)
            with t.extractfile(member) as src, out.open('xb') as target:
                for chunk in iter(lambda: src.read(1024 * 1024), b''):
                    target.write(chunk)
            if sha(out) != expected[key]['sha256']:
                raise ValueError('Downloaded SHA mismatch: ' + key)
    if files != set(expected) or inventory(dest) != expected:
        raise ValueError('Downloaded inventory mismatch')


def download_snapshot(paths, archive, dest):
    """Remote reads only; hash selected files before streaming their tar to local disk."""
    for name in paths:
        safe_name(name)
    code = '''import hashlib,json
from pathlib import Path
root=Path(REMOTE); files={}
for name in PATHS:
 p=root/name
 assert p.exists() and not p.is_symlink(),name
 entries=sorted(p.rglob('*')) if p.is_dir() else [p]
 for q in entries:
  assert not q.is_symlink(),str(q)
  if q.is_dir():continue
  assert q.is_file(),str(q)
  h=hashlib.sha256()
  with q.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
  name=q.relative_to(root).as_posix()
  assert name not in files,name
  files[name]={'bytes':q.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(files))
'''.replace('REMOTE', repr(REMOTE)).replace('PATHS', repr(paths))
    expected = json.loads(remote_python(code, capture_output=True, text=True).stdout)
    tarcmd = shlex.join(['tar', '-czf', '-', '-C', REMOTE, '--'] + paths)
    with Path(archive).open('xb') as f:
        subprocess.run(ssh() + ['bash -lc ' + shlex.quote(tarcmd)], stdout=f, check=True)
    extract_verified(archive, dest, expected)
    return expected
