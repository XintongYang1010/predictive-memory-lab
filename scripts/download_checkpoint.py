"""Download the public checkpoint linked by the pinned RealMotion README."""
import argparse
from pathlib import Path
import json
import gdown
from pmr.common import sha256,write_json

p=argparse.ArgumentParser()
p.add_argument("--out",default="checkpoints/realmotion.pth")
args=p.parse_args()
EXPECTED_SHA256="5f3c9f9cffecd094ee0fccce42a54d1797c77b2b3491393855f1846a71630f1c"
out=Path(args.out)
out.parent.mkdir(parents=True,exist_ok=True)
if not out.exists():
    temp=out.with_suffix(".download")
    result=gdown.download(id="1qyT0HHTMtpsvGy6YFo-jlp-1b-oNGbMr",output=str(temp),quiet=False)
    if result is None:
        raise SystemExit("Download unavailable. Get remotion.pth from the official README and pass CHECKPOINT=/path/to/file.")
    if temp.stat().st_size<100000:
        raise SystemExit("Download too small to be the model checkpoint; refusing it")
    if sha256(temp)!=EXPECTED_SHA256:
        raise SystemExit("Official checkpoint content changed or download corrupted; refusing to use it")
    temp.replace(out)
if sha256(out)!=EXPECTED_SHA256:
    raise SystemExit("Existing file differs from the pinned official checkpoint")
write_json(out.with_suffix(".source.json"),{"source":"https://github.com/fudan-zvg/RealMotion",
           "download_id":"1qyT0HHTMtpsvGy6YFo-jlp-1b-oNGbMr","sha256":sha256(out)})
print(out.resolve())
