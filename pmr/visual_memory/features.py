"""Freeze the official I-JEPA encoder; no random fallback in a scientific run."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F
from .common import atomic_json,atomic_save,digest
from .data import episode,specs,pooled_masks,SCHEMA

IJEPA_COMMIT = "52c1ae95d05f743e000e8f10a1f3a79b10cff048"


class Encoder:
    def __init__(self, code, checkpoint, device="cuda", grid=4):
        code, checkpoint = Path(code).resolve(),Path(checkpoint).resolve()
        sha = subprocess.check_output(["git","-C",str(code),"rev-parse","HEAD"],text=True).strip()
        if sha != IJEPA_COMMIT:
            raise ValueError(f"Expected pinned official I-JEPA {IJEPA_COMMIT}, got {sha}")
        if subprocess.check_output(["git","-C",str(code),"status","--porcelain"],text=True).strip():
            raise ValueError("I-JEPA checkout is dirty; use the pinned clean source")
        sys.path.insert(0,str(code))
        from src.models.vision_transformer import vit_huge
        self.model = vit_huge(patch_size=14,img_size=[224])
        blob = torch.load(checkpoint,map_location="cpu",weights_only=True)
        weights = blob["target_encoder"]
        weights = {k.removeprefix("module."):v for k,v in weights.items()}
        self.model.load_state_dict(weights,strict=True)
        del weights,blob
        self.model.requires_grad_(False).eval().to(device)
        self.device,self.grid = device,grid
        self.identity = dict(encoder="official-IJEPA-ViT-H14-224-target_encoder",code_commit=sha,
                             checkpoint_sha256=digest(checkpoint),pretrained=True,grid=grid)

    @torch.inference_mode()
    def __call__(self, rgb, raw=False):
        x = torch.from_numpy(rgb.copy()).permute(0,3,1,2).to(self.device).float()/255
        mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
        std = x.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=str(self.device).startswith("cuda")):
            z = self.model((x-mean)/std).float()
        side = int(z.shape[1]**.5)
        if side*side != z.shape[1]:
            raise ValueError("Expected spatial tokens without CLS")
        pooled = F.adaptive_avg_pool2d(z.transpose(1,2).reshape(len(z),-1,side,side),self.grid)
        pooled = pooled.flatten(2).transpose(1,2).cpu().to(torch.bfloat16)
        return (pooled,z.cpu().to(torch.bfloat16)) if raw else pooled


def cache_split(encoder, root, split, count, budget, batch_frames=32, probe_pairs=64):
    root = Path(root); folder = root/split; folder.mkdir(parents=True,exist_ok=True)
    expected = dict(schema=SCHEMA,split=split,count=count,encoder=encoder.identity,
                    seed_scheme="sha256(schema/split/pair), variants share split",horizons=[1,4,8])
    manifest = folder/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["configuration"] != expected:
        raise ValueError("Feature cache identity mismatch; choose a fresh output root")
    atomic_json(manifest,dict(configuration=expected,status="partial"))
    for spec in specs(split,count):
        budget.check()
        path = folder/f"{spec[1]:06d}-{spec[2]}.pt"
        if path.exists():
            continue
        ep = episode(*spec)
        images = np.concatenate([ep.rgb,ep.target_rgb])
        out = []
        for start in range(0,len(images),batch_frames):
            budget.check(); out.append(encoder(images[start:start+batch_frames]))
        feats = torch.cat(out)
        # Raw-vs-pooled probes use the same two visible early observations.
        raw_cue = None
        if spec[1] < probe_pairs:
            _,raw_cue = encoder(ep.rgb[6:8],raw=True)
        atomic_save(path,dict(features=feats[:-3],targets=feats[-3:],
                             times=torch.from_numpy(ep.times),shutters=torch.from_numpy(ep.shutters),
                             target_masks=torch.from_numpy(pooled_masks(ep.target_masks,encoder.grid)).float(),
                             probe_state=torch.from_numpy(ep.probe_state),raw_cue=raw_cue,metadata=ep.metadata))
        print(json.dumps(dict(stage="features",episode=ep.metadata["episode_id"],frames=len(images))),flush=True)
    atomic_json(manifest,dict(configuration=expected,status="complete"))


def load_records(root, split, limit=None):
    folder = Path(root)/split
    manifest = json.loads((folder/"manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"{split} cache incomplete")
    paths = sorted(folder.glob("*.pt"))
    return [torch.load(p,map_location="cpu",weights_only=True) for p in (paths if limit is None else paths[:limit])]


def probe(train, dev):
    """Train-only kernel ridge, held-out Dev R2; no state enters predictor."""
    result = {}
    for mode in ("raw","pooled"):
        tr = [r for r in train if r["raw_cue"] is not None]
        dv = [r for r in dev if r["raw_cue"] is not None]
        def x(rows):
            return torch.stack([(r["raw_cue"] if mode=="raw" else r["features"][6:8]).float().flatten() for r in rows]).double()
        a,b=x(tr),x(dv)
        mean = a.mean(0); a=a-mean; b=b-mean
        scale = a.square().sum(1).mean().clamp_min(1e-12)
        y=torch.stack([r["probe_state"] for r in tr]).double()
        target=torch.stack([r["probe_state"] for r in dv]).double()
        ym=y.mean(0); k=a@a.T/scale
        coef=torch.linalg.solve(k+.01*torch.eye(len(k)),y-ym)
        pred=(b@a.T/scale)@coef+ym
        error=(pred-target).square().mean(0)
        baseline=(target-ym).square().mean(0).clamp_min(1e-12)
        result[mode]=dict(r2_per_dimension=(1-error/baseline).tolist(),mse=float(error.mean()),
                          position_r2=float((1-error/baseline)[:4].mean()),velocity_r2=float((1-error/baseline)[4:].mean()))
    result["passed"] = result["pooled"]["position_r2"]>0.1 and result["pooled"]["velocity_r2"]>0.
    result["criterion"]="pilot gate: held-out position R2>0.1 and velocity R2>0; inspect per-dimension values"
    return result
