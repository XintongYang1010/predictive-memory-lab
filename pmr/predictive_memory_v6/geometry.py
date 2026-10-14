"""World geometry: torus (L0), reflecting box (L1) and plane (real tracks).

All model-side coordinates are normalized:

* torus  -- world units / period, readout ``h(z) = z mod 1``;
* box    -- ``(x - lo) / (hi - lo)``, readout ``h`` = triangle-wave fold onto
            [0, 1] (unfold linear motion, fold back into the box);
* plane  -- ``(x - center) / scale``, readout identity.

``fold_sign`` is the Jacobian of the fold (+1 on rising, -1 on falling
segments).  The V6 write multiplies the observation residual by it so that a
residual measured in folded coordinates corrects the unfolded state in the
right direction.  Numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def torus_diff(a, b, period=1.0):
    """Shortest signed difference ``a - b`` on a circle of the given period."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return d - period * np.round(d / period)


def fold(x, lo=0.0, hi=1.0):
    """Triangle-wave fold of an unfolded coordinate into ``[lo, hi]``."""
    span = hi - lo
    r = np.mod(np.asarray(x, dtype=np.float64) - lo, 2.0 * span)
    return lo + span - np.abs(r - span)


def fold_sign(x, lo=0.0, hi=1.0):
    """d fold / d x: +1 while moving up the triangle, -1 while moving down."""
    span = hi - lo
    r = np.mod(np.asarray(x, dtype=np.float64) - lo, 2.0 * span)
    return np.where(r < span, 1.0, -1.0)


@dataclass(frozen=True)
class Geometry:
    kind: str                 # "torus" | "box" | "plane"
    period: float = 1.0       # torus period in world units
    lo: float = 0.0           # box lower wall (reflection point) in world units
    hi: float = 1.0           # box upper wall (reflection point) in world units
    center_x: float = 0.0     # plane normalization
    center_y: float = 0.0
    scale: float = 1.0

    def __post_init__(self):
        if self.kind not in ("torus", "box", "plane"):
            raise ValueError(f"unknown geometry {self.kind!r}")
        if self.kind == "torus" and self.period <= 0:
            raise ValueError("torus period must be positive")
        if self.kind == "box" and not self.hi > self.lo:
            raise ValueError("box needs hi > lo")
        if self.kind == "plane" and self.scale <= 0:
            raise ValueError("plane scale must be positive")

    # ---------------------------------------------------------------- units
    @property
    def unit(self) -> float:
        """World units per normalized unit."""
        if self.kind == "torus":
            return self.period
        if self.kind == "box":
            return self.hi - self.lo
        return self.scale

    def normalize(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        if self.kind == "torus":
            return pos / self.period
        if self.kind == "box":
            return (pos - self.lo) / (self.hi - self.lo)
        center = np.array([self.center_x, self.center_y])
        return (pos - center) / self.scale

    def denormalize(self, z):
        z = np.asarray(z, dtype=np.float64)
        if self.kind == "torus":
            return z * self.period
        if self.kind == "box":
            return self.lo + z * (self.hi - self.lo)
        center = np.array([self.center_x, self.center_y])
        return z * self.scale + center

    def normalize_vector(self, v):
        return np.asarray(v, dtype=np.float64) / self.unit

    # ------------------------------------------------------- world-unit ops
    def diff(self, a, b):
        """Error ``a - b`` in world units (shortest on the torus)."""
        if self.kind == "torus":
            return torus_diff(a, b, self.period)
        return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)

    def readout(self, z):
        """Map an unfolded normalized coordinate to the observed normalized one."""
        if self.kind == "torus":
            return np.mod(z, 1.0)
        if self.kind == "box":
            return fold(z, 0.0, 1.0)
        return np.asarray(z, dtype=np.float64)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Geometry(**d)
