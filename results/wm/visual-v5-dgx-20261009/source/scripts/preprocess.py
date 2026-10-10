"""Fail-fast AV2 preprocessing with a per-file manifest and bounded workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import torch
from tqdm import tqdm
from pmr.upstream import activate
from pmr.common import write_json,sha256


def process_one(job):
    activate()
    from realmotion.datamodules.av2_extractor import Av2Extractor
    raw,out,mode=job
    out=Path(out)
    # Existing outputs are reused; final run manifest fingerprints each cache.
    if not out.exists():
        data=Av2Extractor(mode=mode).get_data(Path(raw))
        tmp=out.with_suffix(".tmp")
        torch.save(data,tmp)
        tmp.replace(out)
    return {"raw":raw,"output":str(out),"sha256":sha256(out)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw",required=True)
    p.add_argument("--out",default="data/realmotion_processed")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--limit-train",type=int,default=6000)
    p.add_argument("--limit-val",type=int,default=1000)
    a=p.parse_args()
    jobs=[]
    for split,limit in (("train",a.limit_train),("val",a.limit_val)):
        files=sorted((Path(a.raw)/split).rglob("scenario_*.parquet"))
        files=files[:limit] if limit else files
        if not files: raise SystemExit(f"No scenario parquet files in {a.raw}/{split}")
        folder=Path(a.out)/split
        folder.mkdir(parents=True,exist_ok=True)
        jobs.extend((str(f),str(folder/(f.stem+".pt")),split) for f in files)
    if a.workers<1: raise SystemExit('--workers must be positive')
    if a.workers==1:
        rows=[process_one(j) for j in tqdm(jobs)]
    else:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            rows=list(tqdm(pool.map(process_one,jobs,chunksize=8),total=len(jobs)))
    write_json(Path(a.out)/"preprocessing_manifest.json",rows)


if __name__=="__main__": main()
