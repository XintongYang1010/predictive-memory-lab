from pathlib import Path
import hashlib
import torch
from .common import clone, move, sha256, write_json
from .upstream import activate


def bucket(scene_id):
    return int(hashlib.sha256(scene_id.encode()).hexdigest()[:8],16)%10


def scene_id(path):
    return Path(path).stem.removeprefix("scenario_")


def make_manifest(root,out,train_limit=1000,dev_limit=200,calib_limit=200,eval_limit=1000):
    root=Path(root).resolve()
    all_train=sorted((root/"train").glob("*.pt"))
    all_val=sorted((root/"val").glob("*.pt"))
    groups={"train":[],"dev":[],"calib":[],"eval":all_val}
    for path in all_train:
        group="calib" if bucket(scene_id(path))==0 else "dev" if bucket(scene_id(path))==1 else "train"
        groups[group].append(path)
    limits=dict(train=train_limit,dev=dev_limit,calib=calib_limit,eval=eval_limit)
    result={}
    seen=set()
    for group,files in groups.items():
        files=files[:limits[group]] if limits[group] else files
        if not files:
            raise ValueError(f"Empty {group} split. Need processed train and val; train is scene-hashed into train/dev/calib.")
        rows=[]
        for file in files:
            sid=scene_id(file)
            if sid in seen:
                raise ValueError(f"Duplicate/cross-split scenario: {sid}")
            seen.add(sid)
            rows.append({"path":str(file),"scene_id":sid,"sha256":sha256(file)})
        result[group]=rows
    write_json(out,{"schema":1,"root":str(root),"splits":result})
    return result


def load_sequence(record,device="cpu",trusted=False):
    _,dataset_module=activate()
    path=Path(record["path"])
    if sha256(path)!=record["sha256"]:
        raise ValueError(f"Data changed since manifest: {path}")
    raw=torch.load(path,map_location="cpu",weights_only=not trusted)
    if str(raw["scenario_id"])!=record["scene_id"]:
        raise ValueError("Scene ID does not match filename/manifest")
    dataset=dataset_module.Av2Dataset(path.parent.parent,split="train",num_historical_steps=30,split_points=[30,40,50])
    seq=dataset_module.collate_fn([dataset.process(raw)])
    return move(seq,device)


def synthetic_raw(sid,seed):
    """Small deterministic fixture; generated solely for software validation."""
    gen=torch.Generator().manual_seed(seed)
    time=torch.arange(110)*.1
    speed=3.+float(torch.rand((),generator=gen))*8
    x=speed*time
    curvature=(float(torch.rand((),generator=gen))-.5)*.025
    y=curvature*(time-2).clamp_min(0).square()
    positions=torch.stack((torch.stack((x,y),-1),torch.stack((x+12,y+3.5),-1)))
    positions+=torch.randn(positions.shape,generator=gen)*.02
    headings=torch.atan2(torch.gradient(positions,dim=1)[0][...,1],torch.gradient(positions,dim=1)[0][...,0])
    lanes=torch.stack([torch.stack((torch.linspace(-50,200,20),torch.full((20,),v)),-1) for v in (0.,3.5,-3.5)])
    return {"x_positions":positions,"x_attr":torch.tensor([[0,3,0],[0,2,0]],dtype=torch.uint8),
            "x_angles":headings,"x_velocity":torch.full((2,110),speed),
            "x_valid_mask":torch.ones(2,110,dtype=torch.bool),"lane_positions":lanes,
            "lane_attr":torch.tensor([[0,3.5,0]]*3),"is_intersections":torch.zeros(3),
            "scenario_id":sid,"agent_ids":["focal","other"],"focal_idx":0,"scored_idx":[1],"city":"SMOKE"}


def create_synthetic(root):
    root=Path(root)
    # Ensure each hash partition is represented without relying on lucky IDs.
    counts={"train":0,"dev":0,"calib":0}
    i=0
    while min(counts.values())<5:
        sid=f"synthetic-train-{i:04d}"
        b=bucket(sid)
        group="calib" if b==0 else "dev" if b==1 else "train"
        if counts[group]<5:
            path=root/"train"/f"scenario_{sid}.pt"
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(synthetic_raw(sid,i),path)
            counts[group]+=1
        i+=1
    for i in range(4):
        sid=f"synthetic-val-{i:04d}"
        path=root/"val"/f"scenario_{sid}.pt"
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(synthetic_raw(sid,1000+i),path)
