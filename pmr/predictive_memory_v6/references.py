"""Non-learned reference trackers with known object identity (numpy).

Every function returns PRE-WRITE predictions ``[B, T, N, 2]`` in world units:
the estimate for step ``t`` uses observations strictly before ``t``.  Steps
before an object's first sighting are NaN and are never scored.

* ``pause``          copy the last observation (the failure V6 targets);
* ``cv_kf``          per-object constant-velocity Kalman filter, geometry
                     aware innovations, no knowledge of walls;
* ``cv_kf_oru``      cv_kf plus OC-SORT observation-centric re-update: on a
                     reveal, replay virtual linear observations across the gap;
* ``analytic_kf``    cv_kf in unfolded coordinates with the known wall fold
                     (identical to cv_kf on torus and plane);
* ``interp_oracle``  linear interpolation between the last observation before
                     and the first after a hidden run.  Uses the FUTURE; it is
                     the ceiling for gap closure, never a competitor.
"""
from __future__ import annotations

import numpy as np

from .geometry import Geometry, fold, fold_sign


class _KF:
    """Independent-axis constant-velocity Kalman filter for B x N objects."""

    def __init__(self, shape, q, r, v0):
        self.x = np.zeros(shape + (2,))
        self.v = np.zeros(shape + (2,))
        self.pxx = np.zeros(shape + (2,))
        self.pxv = np.zeros(shape + (2,))
        self.pvv = np.zeros(shape + (2,))
        self.init = np.zeros(shape, dtype=bool)
        self.q, self.r, self.v0 = q, r, v0

    def predict(self, dt):
        dt = dt[..., None]
        q = self.q
        self.x = self.x + self.v * dt
        pxx = self.pxx + 2 * dt * self.pxv + dt * dt * self.pvv + q * dt ** 3 / 3
        pxv = self.pxv + dt * self.pvv + q * dt ** 2 / 2
        pvv = self.pvv + q * dt
        self.pxx, self.pxv, self.pvv = pxx, pxv, pvv

    def first(self, mask, y):
        m = mask[..., None]
        self.x = np.where(m, y, self.x)
        self.v = np.where(m, 0.0, self.v)
        self.pxx = np.where(m, self.r, self.pxx)
        self.pxv = np.where(m, 0.0, self.pxv)
        self.pvv = np.where(m, self.v0 ** 2, self.pvv)
        self.init |= mask

    def update(self, mask, e):
        m = mask[..., None]
        s = self.pxx + self.r
        kx, kv = self.pxx / s, self.pxv / s
        self.x = np.where(m, self.x + kx * e, self.x)
        self.v = np.where(m, self.v + kv * e, self.v)
        pxx, pxv, pvv = (1 - kx) * self.pxx, (1 - kx) * self.pxv, self.pvv - kv * self.pxv
        self.pxx = np.where(m, pxx, self.pxx)
        self.pxv = np.where(m, pxv, self.pxv)
        self.pvv = np.where(m, pvv, self.pvv)

    def snapshot(self):
        return {k: getattr(self, k).copy() for k in ("x", "v", "pxx", "pxv", "pvv")}

    def restore(self, snap, mask):
        m = mask[..., None]
        for k, v in snap.items():
            setattr(self, k, np.where(m, v, getattr(self, k)))


def _readout(geom: Geometry, x, walls: bool):
    if geom.kind == "torus":
        return np.mod(x, geom.period)
    if geom.kind == "box" and walls:
        return fold(x, geom.lo, geom.hi)
    return x


def _innovation(geom: Geometry, y, x, walls: bool):
    if geom.kind == "torus":
        return geom.diff(y, np.mod(x, geom.period))
    if geom.kind == "box" and walls:
        return fold_sign(x, geom.lo, geom.hi) * (y - fold(x, geom.lo, geom.hi))
    return y - x


def _run_kf(data, geom, q, r, v0, walls=False, oru=False):
    vis, obs, dt = data["vis"], data["obs"], data["dt"]
    B, T, N = vis.shape
    kf = _KF((B, N), q, r, v0)
    pred = np.full((B, T, N, 2), np.nan)
    last_t = np.full((B, N), -1)
    last_y = np.zeros((B, N, 2))
    snaps = {}  # posterior snapshot at each object's last visible step (for ORU)
    post = None
    for t in range(T):
        if t > 0:
            kf.predict(np.broadcast_to(dt[:, t, None], (B, N)))
        pred[:, t] = np.where(kf.init[..., None], _readout(geom, kf.x, walls), np.nan)
        v = vis[:, t]
        new = v & ~kf.init
        upd = v & kf.init
        if oru and upd.any():
            gap = t - last_t
            reveal = upd & (gap > 1)
            if reveal.any():
                # OC-SORT ORU: rewind to the last real posterior, then replay
                # virtual observations on the straight line to the new one.
                kf.restore(post, reveal)
                step_y = obs[:, t] - last_y
                if geom.kind == "torus":
                    step_y = geom.diff(obs[:, t], last_y)
                bidx = np.arange(B)[:, None]
                for k in range(1, int(gap[reveal].max())):
                    active = reveal & (k <= gap - 1)
                    if not active.any():
                        continue
                    when = np.clip(last_t + k, 0, T - 1)
                    kf.predict(np.where(active, dt[bidx, when], 0.0))
                    z = last_y + step_y * (k / np.maximum(gap, 1))[..., None]
                    if geom.kind == "torus":
                        z = np.mod(z, geom.period)
                    kf.update(active, _innovation(geom, z, kf.x, walls))
                kf.predict(np.where(reveal, dt[:, t, None], 0.0))
        kf.first(new, obs[:, t])
        if upd.any():
            kf.update(upd, _innovation(geom, obs[:, t], kf.x, walls))
        if v.any():
            post = kf.snapshot() if post is None else {
                k: np.where(v[..., None], val, post[k]) for k, val in kf.snapshot().items()}
            last_t = np.where(v, t, last_t)
            last_y = np.where(v[..., None], obs[:, t], last_y)
    return pred


def default_kf_params(geom: Geometry, obs_noise=0.0, vel_scale=2.0):
    r = max(obs_noise, 1e-3 * geom.unit / 50.0) ** 2
    q = (1e-4 * geom.unit / 50.0) ** 2
    return q, r, 4.0 * vel_scale


def pause(data, geom: Geometry):
    vis, obs = data["vis"], data["obs"]
    B, T, N = vis.shape
    pred = np.full((B, T, N, 2), np.nan)
    last = np.full((B, N, 2), np.nan)
    for t in range(T):
        pred[:, t] = last
        last = np.where(vis[:, t, ..., None], obs[:, t], last)
    return pred


def cv_kf(data, geom, obs_noise=0.0, vel_scale=2.0):
    q, r, v0 = default_kf_params(geom, obs_noise, vel_scale)
    return _run_kf(data, geom, q, r, v0, walls=False)


def cv_kf_oru(data, geom, obs_noise=0.0, vel_scale=2.0):
    q, r, v0 = default_kf_params(geom, obs_noise, vel_scale)
    return _run_kf(data, geom, q, r, v0, walls=False, oru=True)


def analytic_kf(data, geom, obs_noise=0.0, vel_scale=2.0):
    q, r, v0 = default_kf_params(geom, obs_noise, vel_scale)
    return _run_kf(data, geom, q, r, v0, walls=True)


def interp_oracle(data, geom: Geometry):
    """Ceiling: linear interpolation across each hidden run that ends in a reveal.

    On the torus the displacement across a long gap is ambiguous modulo the
    period, so the oracle reads the unwrapped ground truth when the generator
    provides it (it is an oracle either way).
    """
    vis, obs = data["vis"], data["obs"]
    unwrapped = data.get("pos_unwrapped") if geom.kind == "torus" else None
    B, T, N = vis.shape
    pred = np.full((B, T, N, 2), np.nan)  # defined only inside reveal-terminated runs
    for b in range(B):
        for i in range(N):
            row = vis[b, :, i]
            t = 0
            while t < T:
                if not row[t] and t > 0 and row[:t].any():
                    s = t
                    while t < T and not row[t]:
                        t += 1
                    if t < T:
                        tl = s - 1
                        while not row[tl]:
                            tl -= 1
                        y0, y1 = obs[b, tl, i], obs[b, t, i]
                        d = (unwrapped[b, t, i] - unwrapped[b, tl, i]) if unwrapped is not None \
                            else geom.diff(y1, y0)
                        for k in range(s, t):
                            z = y0 + d * (k - tl) / (t - tl)
                            pred[b, k, i] = np.mod(z, geom.period) if geom.kind == "torus" else z
                else:
                    t += 1
    return pred


REFERENCES = {
    "pause": pause,
    "cv_kf": cv_kf,
    "cv_kf_oru": cv_kf_oru,
    "analytic_kf": analytic_kf,
    "interp_oracle": interp_oracle,
}


def run_reference(name, data, geom, **kw):
    fn = REFERENCES[name]
    if name in ("pause", "interp_oracle"):
        return fn(data, geom)
    return fn(data, geom, **kw)
