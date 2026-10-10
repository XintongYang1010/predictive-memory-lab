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


def cache_configuration(encoder_identity, split, count):
    if split not in ("train","dev"):
        raise ValueError("Scientific feature extraction only opens Train/Dev")
    specs(split,count)
    return dict(schema=SCHEMA,split=split,count=count,encoder=encoder_identity,
                seed_scheme="sha256(schema/split/pair), variants share split",horizons=[1,4,8])


def begin_cache(root, split, count, encoder_identity):
    expected=cache_configuration(encoder_identity,split,count)
    folder=Path(root)/split;folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["configuration"]!=expected:
        raise ValueError("Feature cache identity mismatch; choose a fresh output root")
    atomic_json(manifest,dict(configuration=expected,status="partial"))
    return folder


def validate_record(path, spec):
    row=torch.load(path,map_location="cpu",weights_only=True)
    split,pair,variant=spec
    metadata=row.get("metadata",{})
    if (metadata.get("episode_id")!=f"{split}-{pair:06d}-{variant}"
            or metadata.get("pair_id")!=f"{split}-{pair:06d}"
            or metadata.get("split")!=split or metadata.get("variant")!=variant):
        raise ValueError(f"Cached episode identity mismatch: {path}")
    n=len(row["features"])
    if (n<4 or len(row["times"])!=n or len(row["shutters"])!=n
            or n!=metadata.get("observation_count")
            or len(row["targets"])!=3 or len(row["target_masks"])!=3
            or row["targets"].shape[1:]!=row["features"].shape[1:]):
        raise ValueError(f"Cached episode tensor shape mismatch: {path}")
    return row


def cache_records(encoder, folder, selected_specs, budget, batch_frames=32, probe_pairs=64):
    """Write disjoint episode files; only the parent owns cache manifests."""
    if batch_frames<1:raise ValueError("frame-batch must be positive")
    folder=Path(folder)
    for spec in selected_specs:
        budget.check()
        path=folder/f"{spec[1]:06d}-{spec[2]}.pt"
        if path.exists():
            validate_record(path,spec)
            continue
        ep=episode(*spec)
        images=np.concatenate([ep.rgb,ep.target_rgb])
        out=[]
        for start in range(0,len(images),batch_frames):
            budget.check();out.append(encoder(images[start:start+batch_frames]))
        feats=torch.cat(out)
        raw_cue=None
        if spec[1]<probe_pairs:
            budget.check();_,raw_cue=encoder(ep.rgb[6:8],raw=True)
        atomic_save(path,dict(features=feats[:-3],targets=feats[-3:],
                             times=torch.from_numpy(ep.times),shutters=torch.from_numpy(ep.shutters),
                             target_masks=torch.from_numpy(pooled_masks(ep.target_masks,encoder.grid)).float(),
                             probe_state=torch.from_numpy(ep.probe_state),raw_cue=raw_cue,metadata=ep.metadata))
        print(json.dumps(dict(stage="features",episode=ep.metadata["episode_id"],frames=len(images))),flush=True)


def finish_cache(root, split, count, encoder_identity, budget):
    folder=Path(root)/split
    expected=cache_configuration(encoder_identity,split,count)
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["configuration"]!=expected:
        raise ValueError("Feature cache identity changed while workers were running")
    requested=specs(split,count)
    paths={folder/f"{sp[1]:06d}-{sp[2]}.pt" for sp in requested}
    if set(folder.glob("*.pt"))!=paths:
        raise ValueError("Feature cache record inventory does not match requested paired split")
    inventory={}
    for spec in requested:
        budget.check()
        path=folder/f"{spec[1]:06d}-{spec[2]}.pt"
        validate_record(path,spec)
        inventory[path.name]=dict(bytes=path.stat().st_size,sha256=digest(path))
    atomic_json(folder/"manifest.json",dict(configuration=expected,status="complete",validated_records=count,
                                           files=inventory))


def cache_split(encoder, root, split, count, budget, batch_frames=32, probe_pairs=64):
    folder=begin_cache(root,split,count,encoder.identity)
    cache_records(encoder,folder,specs(split,count),budget,batch_frames,probe_pairs)
    finish_cache(root,split,count,encoder.identity,budget)


def load_records(root, split, limit=None):
    folder = Path(root)/split
    manifest = json.loads((folder/"manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"{split} cache incomplete")
    requested=specs(split,manifest["configuration"]["count"])
    paths=[folder/f"{spec[1]:06d}-{spec[2]}.pt" for spec in requested]
    if set(folder.glob("*.pt"))!=set(paths):
        raise ValueError(f"{split} cache inventory mismatch")
    pairs=list(zip(paths,requested))
    records=[]
    for path,spec in (pairs if limit is None else pairs[:limit]):
        if "files" in manifest and manifest["files"].get(path.name)!=dict(bytes=path.stat().st_size,sha256=digest(path)):
            raise ValueError(f"{split} cache file hash mismatch: {path.name}")
        records.append(validate_record(path,spec))
    return records


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
