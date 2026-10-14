"""Memoryless latent-frame-prediction (LFP) analogue of Cambrian-S (torch).

Cambrian-S's LFP head is Linear-GELU-Linear on LLM hidden states, trained with
masked MSE + cosine-embedding loss to predict next-frame SigLIP2 patch
features (cambrian_arch.py L262-272; cambrian_qwen2.py L323-335).  Here the
"frame" is a rasterized token grid (surprise.rasterize) and the head predicts
the next frame from the current one only, i.e. it sees no memory.  It is
frozen at evaluation, as in Cambrian-S.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import make_world, geometry_of, stable_seed
from .surprise import rasterize, frame_surprise


class LFPAnalog(nn.Module):
    def __init__(self, patches=64, feat=18, hidden=512):
        super().__init__()
        self.patches, self.feat = patches, feat
        self.net = nn.Sequential(nn.Linear(patches * feat, hidden), nn.GELU(),
                                 nn.Linear(hidden, patches * feat))

    def forward(self, Fr):  # [B,T,P,D] -> prediction of frame t+1 at index t
        B, T, Pn, D = Fr.shape
        return self.net(Fr.reshape(B, T, Pn * D)).view(B, T, Pn, D)


def lfp_loss(pred, target):
    active = (target.norm(dim=-1) > 1e-3).float()
    mse = ((pred - target) ** 2).mean(-1)
    cos = 1 - F.cosine_similarity(pred, target, dim=-1, eps=1e-6)
    return ((mse + cos) * active).sum() / active.sum().clamp(min=1) + 0.1 * mse.mean()


def train_lfp(world_spec, T, steps, device, seed=0, batch=128, grid=8, log=None):
    torch.manual_seed(seed)
    probe = make_world(world_spec, 2, T, stable_seed("lfp-probe", world_spec))
    feat = probe["app"].shape[-1] + 2
    model = LFPAnalog(grid * grid, feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for step in range(steps):
        d = make_world(world_spec, batch, T, stable_seed("lfp-train", world_spec, seed, step))
        Fr = torch.as_tensor(rasterize(d, geometry_of(d), grid), device=device)
        pred = model(Fr[:, :-1])
        loss = lfp_loss(pred, Fr[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log is not None and step % 200 == 0:
            log({"lfp_step": step, "lfp_loss": float(loss)})
    model.eval()
    return model


@torch.no_grad()
def lfp_surprise(model, data, device, grid=8):
    Fr = torch.as_tensor(rasterize(data, geometry_of(data), grid), device=device)
    pred = torch.zeros_like(Fr)
    pred[:, 1:] = model(Fr[:, :-1])
    return frame_surprise(pred.cpu().numpy(), Fr.cpu().numpy())
