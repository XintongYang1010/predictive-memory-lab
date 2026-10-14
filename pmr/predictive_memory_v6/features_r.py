"""R layer: frozen DINOv2 ViT-S/14 features of Memory Maze frames (S5).

One fixed configuration, run once (AGENTS.md: no encoder sweeps): crop the
2 px target-colour border (64 -> 60 px), bicubic resize to 224, ImageNet
normalisation, CLS (384) + 2x2 average-pooled patch tokens (4 x 384) = 1920
dims, PCA-64 fitted on ~1M train frames, stored as float16 ``rfeat.npy`` next
to the token cache.  The SMOKE encoder is a fixed random projection and can
only be selected by the CPU SMOKE stage.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import data as Dm
from . import maze as M
from . import pipeline as P
from . import prereg

DINOV2_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
FEAT_DIM = 1920


class SmokeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("W", torch.randn(192, FEAT_DIM, generator=g) / 192 ** 0.5)

    def forward(self, x):                       # x [B,3,224,224]
        z = F.adaptive_avg_pool2d(x, 8).flatten(1)
        return torch.tanh(z @ self.W)


class Dino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ck = os.environ.get("PMR_DINOV2_CKPT")
        self.m = torch.hub.load(os.environ.get("PMR_DINOV2_REPO", DINOV2_REPO), "dinov2_vits14",
                                pretrained=ck is None, trust_repo=True)
        if ck:
            self.m.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)
        self.m.eval()

    def forward(self, x):
        out = self.m.forward_features(x)
        cls = out["x_norm_clstoken"]
        pt = out["x_norm_patchtokens"]
        B, N, C = pt.shape
        s = int(round(N ** 0.5))
        grid = pt.transpose(1, 2).reshape(B, C, s, s)
        pooled = F.adaptive_avg_pool2d(grid, 2).flatten(2).transpose(1, 2).reshape(B, 4 * C)
        return torch.cat([cls, pooled], -1)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(imgs, dev):
    b = prereg.R_GATE["crop_border_px"]
    x = torch.as_tensor(np.ascontiguousarray(imgs[:, b:-b, b:-b]), device=dev).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=prereg.R_GATE["image_px"], mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - MEAN.to(dev)) / STD.to(dev)


def encode_episode(enc, path, dev, chunk=256):
    imgs = M.load_images(path)
    outs = []
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for s in range(0, len(imgs), chunk):
            outs.append(enc(preprocess(imgs[s:s + chunk], dev)).float())
    return torch.cat(outs)


def _paths(spec, split):
    ids = json.loads((Path(spec["cache"]) / split / "ids.json").read_text())
    return [str(Path(spec["data"]) / i) for i in ids]


def _encoder(spec, dev):
    return (SmokeEncoder() if spec.get("smoke_encoder") else Dino()).to(dev).eval()


def fit_pca(spec):
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    dev = P.device()
    enc = _encoder(spec, dev)
    paths = _paths(spec, "train")
    n_eps = max(1, prereg.R_GATE["pca_fit_frames"] // 1001)
    n_eps = min(n_eps, spec.get("pca_episodes", n_eps), len(paths))
    s1 = torch.zeros(FEAT_DIM, dtype=torch.float64, device=dev)
    s2 = torch.zeros(FEAT_DIM, FEAT_DIM, dtype=torch.float64, device=dev)
    n = 0
    for p in paths[:n_eps]:
        f = encode_episode(enc, p, dev).double()
        s1 += f.sum(0)
        s2 += f.T @ f
        n += f.shape[0]
    mean = s1 / n
    cov = s2 / n - torch.outer(mean, mean)
    ev, evec = torch.linalg.eigh(cov)
    k = prereg.R_GATE["pca_dim"]
    comps = evec[:, -k:].flip(-1)
    proj_std = ev[-k:].flip(-1).clamp_min(1e-12).sqrt()
    np.savez(Path(spec["cache"]) / "pca.npz", mean=mean.cpu().numpy(), comps=comps.cpu().numpy(),
             std=proj_std.cpu().numpy(), explained=float(ev[-k:].sum() / ev.sum()), n_frames=n)
    for split in spec.get("alloc_splits", ["train", "dev"]):
        N = len(json.loads((Path(spec["cache"]) / split / "ids.json").read_text()))
        f = Path(spec["cache"]) / split / "rfeat.npy"
        if not f.exists():
            np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(N, 1001, k))
    P.write_json(out / "status.json", {"status": "complete", "n_frames": n,
                                       "explained": float(ev[-k:].sum() / ev.sum())})
    return {"status": "complete"}


def project(spec):
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    dev = P.device()
    enc = _encoder(spec, dev)
    pca = np.load(Path(spec["cache"]) / "pca.npz")
    mean = torch.as_tensor(pca["mean"], device=dev, dtype=torch.float32)
    comps = torch.as_tensor(pca["comps"], device=dev, dtype=torch.float32)
    std = torch.as_tensor(pca["std"], device=dev, dtype=torch.float32)
    split = spec["split"]
    paths = _paths(spec, split)
    rf = np.load(Path(spec["cache"]) / split / "rfeat.npy", mmap_mode="r+")
    i0, n_sh = spec.get("shard", 0), spec.get("n_shards", 1)
    done = 0
    for i in range(i0, len(paths), n_sh):
        f = encode_episode(enc, paths[i], dev)
        z = ((f - mean) @ comps) / std            # whitened PCA-64
        rf[i] = z.cpu().numpy().astype(np.float16)
        done += 1
    rf.flush()
    P.write_json(out / "status.json", {"status": "complete", "episodes": done})
    return {"status": "complete"}


def r_gate(spec):
    """Pre-registered single-run sanity gate for the R layer."""
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    tr, dv = Dm.Store(spec["cache"], "train"), Dm.Store(spec["cache"], "dev")

    def xy(store, n):
        X, Yv = [], []
        for i in store.order[:n]:
            e = store.episode(int(i))
            for g in range(store.G):
                v = e["vis"][:, g]
                X.append(e["x"][v])
                Yv.append(e["tvec"][v, g])
        return np.concatenate(X), np.concatenate(Yv)
    Xt, Yt = xy(tr, spec.get("n_train", 2000))
    Xd, Yd = xy(dv, spec.get("n_dev", 500))
    A = np.concatenate([Xt, np.ones((len(Xt), 1))], 1)
    Wr = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ Yt)
    pred = np.concatenate([Xd, np.ones((len(Xd), 1))], 1) @ Wr
    mse = float(((pred - Yd) ** 2).sum(1).mean())
    const = float(((Yt.mean(0) - Yd) ** 2).sum(1).mean())
    slow_r2 = P.load_json(spec["slow_status"]).get("dev_r2")
    res = {"probe_mse": mse, "const_mse": const, "ratio": mse / max(const, 1e-12),
           "A0_next_latent_r2": slow_r2,
           "pass": bool(mse / max(const, 1e-12) <= prereg.R_GATE["probe_mse_frac_of_const_max"]
                        and slow_r2 is not None and slow_r2 >= prereg.R_GATE["A0_next_latent_r2_min"])}
    P.write_json(out / "r_gate.json", res)
    P.write_json(out / "status.json", {"status": "complete", **res})
    return {"status": "complete"}
