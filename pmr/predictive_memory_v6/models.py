"""Small torch modules for V6.1: f_theta, key net phi, learned gates, probes.

Nothing here is large: f_theta and phi are two-layer MLPs (hidden 256), the
gates have <= 10k parameters, and BF-Delta itself has no parameters.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Standardize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x):
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-3))

    def forward(self, x):
        return (x - self.mean) / self.std


def mlp(i, h, o, act=nn.GELU):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, o))


class SlowModel(nn.Module):
    """Memoryless one-step predictor of the next value vector (the prior)."""

    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, out_dim=out_dim, hidden=hidden)
        self.norm = Standardize(in_dim)
        self.net = mlp(in_dim, hidden, out_dim)
        self.register_buffer("sigma", torch.ones(out_dim))   # residual s.d. on train (raw units)

    def forward(self, z):
        return self.net(self.norm(z))

    def residual(self, z, y, o):
        """Standardized residual, zero where unobserved."""
        return (y - self(z)) / self.sigma * o


class KeyNet(nn.Module):
    def __init__(self, in_dim, dk, hidden=256):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, dk=dk, hidden=hidden)
        self.norm = Standardize(in_dim)
        self.net = mlp(in_dim, hidden, dk)

    def forward(self, z):
        return F.normalize(self.net(self.norm(z)), dim=-1, eps=1e-6)


class GateA2(nn.Module):
    """Gated DeltaNet gates (alpha, beta) = sigmoid(MLP(input))."""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, hidden=hidden)
        self.norm = Standardize(in_dim)
        self.net = mlp(in_dim, hidden, 2)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([4.0, 0.0]))

    def forward(self, z):
        g = torch.sigmoid(self.net(self.norm(z)))
        return g


class GateA2P(nn.Module):
    """Strongest fair learned gate: same sufficient statistics as BF-Delta
    (|e|^2/D, |r|^2/D, log c) plus supervision with change data."""

    def __init__(self, hidden=64):
        super().__init__()
        self.cfg = dict(hidden=hidden)
        self.net = mlp(3, hidden, 3)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([4.0, -1.0, 3.0]))

    def forward(self, feats):
        g = torch.sigmoid(self.net(feats))
        return g[:, 0], g[:, 1], g[:, 2]


class ScalarGate(nn.Module):
    """A3 / A4: beta = sigmoid(a * x + b) with two scalars."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))
        self.cfg = {}


class Probe(nn.Module):
    """Memory Maze probe: 4 x 1024 MLP, LayerNorm + ELU; input gradients stopped."""

    def __init__(self, in_dim, out_dim, layers=4, hidden=1024):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ELU()]
            d = hidden
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x.detach())


def save(path, module, extra=None):
    torch.save({"cls": type(module).__name__, "cfg": getattr(module, "cfg", {}),
                "state": module.state_dict(), "extra": extra or {}}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cls = {"SlowModel": SlowModel, "KeyNet": KeyNet, "GateA2": GateA2, "GateA2P": GateA2P,
           "ScalarGate": ScalarGate}[ck["cls"]]
    m = cls(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck.get("extra", {})
