"""Bundle a completed run, candidates, checksums and optional console log."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def package(run, log=None):
    run=Path(run).resolve()
    stats=json.loads((run/'run_stats.json').read_text())
    if stats.get('status')!='complete':raise ValueError('Only completed runs can be packaged')
    output=Path(str(run)+'.tar.gz')
    paths=[(p,p.relative_to(run).as_posix()) for p in sorted(run.rglob('*'))
           if p.is_file() and 'synthetic_data' not in p.relative_to(run).parts]
    if log is not None:paths.append((Path(log),'console.log'))
    if len({name for _,name in paths})!=len(paths):raise ValueError('Duplicate archive names')
    metadata={}
    for path,name in paths:
        if path.is_symlink():raise ValueError(f'Symlinks are not packaged: {path}')
        data=path.read_bytes()
        metadata[name]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    with tarfile.open(output,'x:gz') as archive:
        for path,name in paths:archive.add(path,arcname=f'{run.name}/{name}',recursive=False)
        content=(json.dumps({'files':metadata},indent=2)+'\n').encode()
        info=tarfile.TarInfo(f'{run.name}/package_manifest.json');info.size=len(content)
        archive.addfile(info,io.BytesIO(content))
    checksum=hashlib.sha256(output.read_bytes()).hexdigest()
    Path(str(output)+'.sha256').write_text(f'{checksum}  {output.name}\n')
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'sha256':checksum,'files':len(paths)},indent=2))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--log')
    args=parser.parse_args()
    package(args.run,args.log)
