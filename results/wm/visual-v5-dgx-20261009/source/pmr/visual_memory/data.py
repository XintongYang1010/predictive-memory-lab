"""Deterministic paired RGB worlds; latent state is evaluation-only.

Time gap and observation/write count are independent. Shutter schedules and
camera observations never depend on a target's hidden velocity or pair variant.
This pilot has two moving targets plus visual distractors, not natural videos.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "v5-rgb-pairs-1"
HORIZONS = (1., 4., 8.)


def seed_for(split, pair):
    return int.from_bytes(hashlib.sha256(f"{SCHEMA}/{split}/{pair}".encode()).digest()[:4], "little")


@dataclass
class Episode:
    rgb: np.ndarray
    times: np.ndarray
    shutters: np.ndarray
    target_rgb: np.ndarray
    target_masks: np.ndarray
    probe_state: np.ndarray
    metadata: dict


def reflect(x, low, high):
    span = high - low
    return low + span - np.abs(np.mod(x - low, 2 * span) - span)


def episode(split, pair, variant, size=224, gap=None, writes=None, family=None):
    if variant not in (0, 1):
        raise ValueError("A pair has exactly two variants")
    rng = np.random.default_rng(seed_for(split, pair))
    family = family or ("hold", "interference", "update")[pair % 3]
    gap = gap or (8, 16, 32)[(pair // 3) % 3]
    writes = (0, 8, 32)[(pair // 9) % 3] if writes is None else writes
    query = 7. + gap
    correction = 7. + gap * .4
    early = np.arange(8, dtype=np.float32)
    recent = query + np.array([-.75, -.5, -.25, 0], dtype=np.float32)
    distractor = np.linspace(7.3, query - 1., writes, dtype=np.float32)
    update_obs = correction + np.arange(4, dtype=np.float32) * .2
    times = np.unique(np.concatenate([early, distractor, update_obs, recent])).astype(np.float32)
    p = rng.uniform([.2, .23], [.8, .32], (2, 2))
    p[1, 1] += .42
    v = rng.uniform(.008, .018, (2, 2)) * rng.choice([-1, 1], (2, 2))
    v *= (1 if variant == 0 else -1)
    revised = rng.uniform(.008, .018, 2) * rng.choice([-1, 1], 2) * (1 if variant == 0 else -1)
    colors = [tuple(int(x) for x in c) for c in rng.integers(60, 240, (2, 3))]
    colors[0] = (235, colors[0][1] // 2, 60)
    colors[1] = (60, colors[1][1] // 2, 235)
    lanes = [(.15, .40), (.60, .85)]

    def position(t, obj):
        pos = p[obj] + v[obj] * t
        if family == "update" and obj == 0 and t >= correction:
            # Reflect only after continuous integration, so the update is visible
            # as a changed motion rule. No hidden future random event is scored.
            pos = p[obj] + v[obj] * correction + revised * (t - correction)
        return np.array([reflect(pos[0], .10, .90), reflect(pos[1], *lanes[obj])])

    def render(t, open_shutters, distract=True):
        im = Image.new("RGB", (size, size), (30, 35, 42))
        draw = ImageDraw.Draw(im)
        for xx in range(0, size, max(1, size // 16)):
            draw.line((xx, 0, xx, size), fill=(42, 47, 54))
        masks = np.zeros((2, size, size), dtype=np.uint8)
        radius = size * .06
        for obj in range(2):
            x, y = position(t, obj) * size
            box = (int(x-radius), int(y-radius), int(x+radius), int(y+radius))
            draw.ellipse(box, fill=colors[obj], outline=(250, 250, 240), width=max(1,size//100))
            draw.line((x-radius/2, y, x+radius/2, y), fill=(255,255,255), width=max(1,size//60))
            if obj == 1:
                draw.line((x,y-radius/2,x,y+radius/2), fill=(255,255,255), width=max(1,size//60))
            mask = Image.new("L", (size,size), 0)
            ImageDraw.Draw(mask).ellipse(box, fill=1)
            masks[obj] = np.asarray(mask)
            if not open_shutters[obj]:
                lo, hi = ((.04,.47) if obj == 0 else (.53,.96))
                draw.rectangle((0,int(size*lo),size,int(size*hi)), fill=(80,85,91))
                masks[obj] = 0
        if distract:
            x = int((.5 + .4*np.sin(t*2.3 + pair)) * size)
            draw.rectangle((x-5,size*.48,x+5,size*.52), fill=(240,220,50))
        return np.asarray(im).copy(), masks

    shutters = []
    frames = []
    for t in times:
        opened = [bool(t <= 7), bool(t <= 7)]
        if family == "update" and np.any(np.isclose(t, update_obs, atol=1e-5)):
            opened[0] = True
        shutters.append(opened)
        frames.append(render(float(t), opened)[0])
    targets = [render(query + h, [True,True]) for h in HORIZONS]
    meta = dict(episode_id=f"{split}-{pair:06d}-{variant}", pair_id=f"{split}-{pair:06d}",
                split=split, variant=variant, family=family, gap=gap, interference_writes=writes,
                observation_count=len(times), query_time=query, update_time=correction,
                updated_object=0 if family == "update" else None)
    return Episode(np.stack(frames), times, np.array(shutters,dtype=np.float32),
                   np.stack([x[0] for x in targets]), np.stack([x[1] for x in targets]),
                   np.concatenate([position(7,0), position(7,1),
                                   position(7,0)-position(6,0),position(7,1)-position(6,1)]).astype(np.float32), meta)


def specs(split, count):
    if count < 2 or count % 2:
        raise ValueError("Episode count must be even; never split a paired world")
    return [(split, p, variant) for p in range(count // 2) for variant in (0,1)]


def pooled_masks(masks, grid=4):
    h, o, height, width = masks.shape
    if height % grid or width % grid:
        raise ValueError("Image dimensions must be divisible by grid")
    return masks.reshape(h,o,grid,height//grid,grid,width//grid).mean((3,5)).reshape(h,o,grid*grid)
