"""Portable implementations of the narrow layer APIs used by RealMotion.

No custom CUDA extension. Neighborhood windows shift at boundaries; they are
not zero-padded centered windows. Only dilation=1, noncausal NA is supported.
Parameter names match NATTEN 0.17.3 (whose default has no relative bias).
"""
import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class NeighborhoodAttention1D(nn.Module):
    def __init__(self, dim, num_heads, kernel_size, dilation=1,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rel_pos_bias=False, is_causal=False):
        super().__init__()
        if dilation not in (1, None) or is_causal:
            raise ValueError("Portable backend supports noncausal dilation=1 only")
        if kernel_size % 2 != 1 or dim % num_heads:
            raise ValueError("Odd kernel and divisible head dimension required")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.kernel_size = kernel_size
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.register_parameter("rpb", nn.Parameter(torch.zeros(num_heads, 2 * kernel_size - 1))
                                if rel_pos_bias else None)

    def forward(self, x):
        b, length, dim = x.shape
        ksize = self.kernel_size
        if length < ksize:
            raise ValueError(f"NA sequence length {length} < kernel {ksize}")
        q, k, v = self.qkv(x).reshape(b, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        t = torch.arange(length, device=x.device)
        start = (t - ksize // 2).clamp(0, length - ksize)
        indices = start[:, None] + torch.arange(ksize, device=x.device)
        keys, values = k[:, :, indices, :], v[:, :, indices, :]
        logits = ((q * self.scale).unsqueeze(-2) * keys).sum(-1)
        if self.rpb is not None:
            logits = logits + self.rpb[:, indices - t[:, None] + ksize - 1]
        out = (self.attn_drop(logits.softmax(-1)).unsqueeze(-1) * values).sum(-2)
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, length, dim)))
