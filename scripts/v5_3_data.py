#!/usr/bin/env python3
"""Registered V5.3 six-image blocks; labels and identity remain analysis-only.

Tests must use software_test. FIT and anchor are separately gated by the
extractor before this deterministic generator is called in production.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from v5_temporal_confirmation import render

SCHEMA = "v5.3-nuisance-repair-1"
ROLES = ("software_test", "fit", "anchor")
ORDERS = np.array(((0, 1), (2, 1), (1, 2), (1, 0)), dtype=np.int64)
PATHS = np.array(((0, 0), (1, 1), (0, 1), (1, 0)), dtype=np.int64)
UNIQUE_ORDER = ("p0a", "p0b", "p1a", "p1b", "p2a", "p2b")
PATH_NAMES = ("aa", "bb", "ab", "ba")
VIEW_NAMES = tuple(f"{view}/{path}" for view in ("end+", "end-", "start+", "start-")
                   for path in PATH_NAMES)
VIEW_INDICES = np.array([[2*start+a, 2*end+b] for start, end in ORDERS for a, b in PATHS], dtype=np.int64)
STATIC_VIEW_INDICES = np.array([4*view+path for view in range(4) for path in (0, 1)], dtype=np.int64)
NEGATIVE_INDICES = np.array(((2, 3), (3, 2)), dtype=np.int64)
BRIDGE_PAIR_IDS = tuple(range(384, 640))


@dataclass
class Block:
    unique_rgb: np.ndarray
    positions: np.ndarray
    targets: np.ndarray
    negative_targets: np.ndarray
    metadata: dict

    def actor_inputs(self):
        """Only rendered observations, public cue times, and visibility."""
        return dict(rgb=self.unique_rgb[VIEW_INDICES],
                    times=np.tile(np.array([6., 7.], dtype=np.float32), (16, 1)),
                    shutters=np.ones((16, 2, 2), dtype=np.float32))


def seed_for(role, block):
    if role not in ROLES or type(block) is not int or not 0 <= block < 256:
        raise ValueError("V5.3 roles are software_test/fit/anchor and block IDs 0..255; no Test role")
    return int.from_bytes(hashlib.sha256(f"{SCHEMA}/{role}/{block}".encode()).digest()[:8], "little")


def make_block(role, block):
    rng = np.random.default_rng(seed_for(role, block))
    center = rng.uniform([.20, .20, .20, .65], [.80, .35, .80, .80])
    displacement = rng.uniform(.008, .018, 4)*rng.choice([-1, 1], 4)
    center = np.round(center*2**20)/2**20
    displacement = np.round(displacement*2**20)/2**20
    positions = np.stack((center-displacement, center, center+displacement))
    colors = [tuple(int(x) for x in c) for c in rng.integers(60, 240, (2, 3))]
    colors[0] = (235, colors[0][1]//2, 60)
    colors[1] = (60, colors[1][1]//2, 235)
    # Independent draws; no rejection of coincident raster positions.
    nuisance_x = rng.uniform(.1, .9, 2)
    images = np.stack([render(p, colors, x) for p in positions for x in nuisance_x])
    start, end = positions[ORDERS[:, 0]], positions[ORDERS[:, 1]]
    target = np.concatenate((end, end-start), axis=1).astype(np.float32)
    targets = np.repeat(target, 4, axis=0)
    negative = np.tile(np.concatenate((center, np.zeros(4))).astype(np.float32), (2, 1))
    return Block(images, positions, targets, negative,
                 dict(schema=SCHEMA, role=role, block=block, group_id=f"{SCHEMA}/{role}/{block:06d}",
                      cue_reflections=False, nuisance_x=nuisance_x.tolist(),
                      nuisance_pixel_centers=[int(x*224) for x in nuisance_x],
                      nuisance_coincident_pixels=bool(int(nuisance_x[0]*224)==int(nuisance_x[1]*224)),
                      unique_order=list(UNIQUE_ORDER), view_indices=VIEW_INDICES.tolist(),
                      negative_indices=NEGATIVE_INDICES.tolist()))
