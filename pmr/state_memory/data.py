"""Causal, reproducible controlled data for state-memory mechanism experiments.

This is a synthetic diagnostic task, not a RealMotion or visual-world-model
benchmark. Positions have arbitrary distance units; this suite requires
``dt=1.0`` in arbitrary time units per frame. Velocities are noisy *current sensor measurements*, never finite
differences involving future frames. Arrays named state/targets and all role
metadata are supervision/diagnostics only; a model receives observation only.

Two objects become hidden after the first two frames. One receives an
unannounced position/velocity change when fresh evidence reappears. The other
stays hidden, testing preservation of unrelated state. Other objects remain
visible as irrelevant repeated context. Object roles are randomized per scene.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np


_VERSION = "state-memory-v4.1"
_FIELDS = ("position_x", "position_y", "velocity_x", "velocity_y", "visible")


@dataclass(frozen=True)
class TaskConfig:
    n_objects: int = 3
    length: int = 24
    reveal_step: int = 12
    dt: float = 1.0
    observation_noise: float = 0.01
    horizons: tuple[int, ...] = (0, 1, 4)
    reveal_jitter: int = 1
    reveal_visible_steps: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", tuple(self.horizons))
        for name in ("n_objects", "length", "reveal_step", "reveal_jitter", "reveal_visible_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.n_objects < 3:
            raise ValueError("At least three objects are needed for changed/preserved/context roles")
        if self.length < 5 or self.reveal_jitter < 0 or self.reveal_visible_steps < 1:
            raise ValueError("Invalid length, reveal_jitter or reveal_visible_steps")
        if self.reveal_step - self.reveal_jitter < 2:
            raise ValueError("Reveal must occur after the two initial evidence frames")
        if self.reveal_step + self.reveal_jitter + self.reveal_visible_steps > self.length:
            raise ValueError("All reveal evidence frames must fit within length")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.dt != 1.0:
            raise ValueError("V4 state-memory models and metrics require dt=1.0")
        if not np.isfinite(self.observation_noise) or self.observation_noise < 0:
            raise ValueError("observation_noise must be finite and nonnegative")
        if not self.horizons or any(
            isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 0
            for h in self.horizons
        ):
            raise ValueError("horizons must contain nonnegative integer frame offsets")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")


def _digest(*parts: Any) -> str:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rng(seed: int, split: str, index: int, component: str) -> np.random.Generator:
    # Do not use Python's process-randomized hash(). Separate streams prevent a
    # change to a future intervention from changing earlier observation noise.
    return np.random.default_rng(int(_digest(_VERSION, seed, split, index, component)[:16], 16))


def _observations(state: np.ndarray, visible: np.ndarray, noise: np.ndarray) -> np.ndarray:
    out = np.zeros((*visible.shape, 5), dtype=np.float32)
    out[..., :4] = np.where(visible[..., None], state + noise, 0.0)
    out[..., 4] = visible
    return out


def make_batch(config: TaskConfig, count: int, seed: int, split: str = "train") -> dict[str, Any]:
    """Return paired episodes, with supervision strictly separate from inputs.

    observation: [B,T,N,5]; state: [B,T,N,4]; targets: [B,T,N,H,2].
    A target is the position at t+h, including positions beyond the observation
    sequence. Only targets for the changed object whose horizon crosses its
    not-yet-observed jump are masked. h=0 is reconstruction, not forecasting.

    ``paired_*`` removes only the local jump; all other objects are bitwise
    identical. ``history_pair_*`` instead changes only the preserved object's
    initial position/velocity. Its first two observations and true future differ,
    while observations at every t>=2 are identical to the original episode.

    Seeds are namespaced by split and episode index. Same seed/split/config is
    reproducible and increasing count preserves earlier episodes. Same
    seed/split with different lengths deliberately shares latent scenes for
    controlled comparisons: use distinct split names for independent holdouts.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a nonempty namespace")
    seed, count = int(seed), int(count)
    b, t, n, h = count, config.length, config.n_objects, len(config.horizons)
    extended_length = t + max(config.horizons)
    times = np.arange(extended_length, dtype=np.float64) * config.dt
    all_state = np.empty((b, extended_length, n, 4), dtype=np.float32)
    all_paired = np.empty_like(all_state)
    all_history = np.empty_like(all_state)
    visible = np.ones((b, t, n), dtype=bool)
    noise = np.empty((b, t, n, 4), dtype=np.float32)
    changed = np.empty(b, dtype=np.int64)
    preserved = np.empty(b, dtype=np.int64)
    reveal_steps = np.empty(b, dtype=np.int64)
    config_dict = asdict(config)
    config_hash = _digest(config_dict)
    episode_ids: list[str] = []
    latent_ids: list[str] = []

    for i in range(b):
        initial_rng = _rng(seed, split, i, "initial")
        initial_position = initial_rng.uniform(-1.0, 1.0, size=(n, 2))
        velocity = initial_rng.uniform(-0.08, 0.08, size=(n, 2))
        role_rng = _rng(seed, split, i, "roles")
        roles = role_rng.permutation(n)
        c, p = int(roles[0]), int(roles[1])
        changed[i], preserved[i] = c, p
        event_rng = _rng(seed, split, i, "event")
        r = config.reveal_step + int(event_rng.integers(-config.reveal_jitter, config.reveal_jitter + 1))
        reveal_steps[i] = r
        position_jump = event_rng.uniform(0.35, 0.70, size=2) * event_rng.choice((-1.0, 1.0), size=2)
        velocity_jump = event_rng.uniform(-0.12, 0.12, size=2)

        base = np.empty((extended_length, n, 4), dtype=np.float64)
        base[..., :2] = initial_position[None] + times[:, None, None] * velocity[None]
        base[..., 2:] = velocity[None]
        revised = base.copy()
        since_event = times[r:] - times[r]
        revised[r:, c, :2] += position_jump + since_event[:, None] * velocity_jump
        revised[r:, c, 2:] += velocity_jump

        alternate_history = revised.copy()
        history_rng = _rng(seed, split, i, "history_pair")
        # Resample from the SAME priors as ordinary episodes. Adding a fixed
        # offset would turn this memory diagnostic into an OOD-state test.
        alternate_position = history_rng.uniform(-1.0, 1.0, size=2)
        alternate_velocity = history_rng.uniform(-0.08, 0.08, size=2)
        alternate_history[:, p, :2] = alternate_position + times[:, None] * alternate_velocity
        alternate_history[:, p, 2:] = alternate_velocity
        all_state[i], all_paired[i], all_history[i] = revised, base, alternate_history
        visible[i, 2:, p] = False
        visible[i, 2:, c] = False
        visible[i, r:r + config.reveal_visible_steps, c] = True
        noise[i] = _rng(seed, split, i, "sensor_noise").normal(
            0.0, config.observation_noise, size=(t, n, 4)
        )
        latent_id = _digest(_VERSION, seed, split, i)
        latent_ids.append(latent_id)
        episode_ids.append(f"{split}:{_digest(latent_id, config_hash)[:24]}")

    target_indices = np.arange(t)[:, None] + np.asarray(config.horizons)[None]

    def gather(states: np.ndarray) -> np.ndarray:
        return states[:, target_indices, :, :2].transpose(0, 1, 3, 2, 4).copy()

    target_mask = np.ones((b, t, n, h), dtype=bool)
    for i in range(b):
        crosses_unannounced_jump = (
            (np.arange(t)[:, None] < reveal_steps[i]) & (target_indices >= reveal_steps[i])
        )
        target_mask[i, :, changed[i], :] = ~crosses_unannounced_jump

    observations = _observations(all_state[:, :t], visible, noise)
    paired_observations = _observations(all_paired[:, :t], visible, noise)
    history_observations = _observations(all_history[:, :t], visible, noise)
    manifest = {
        "schema": _VERSION,
        "config": config_dict,
        "seed": seed,
        "split": split,
        "count": count,
        "config_hash": config_hash,
        "episode_ids": episode_ids,
        "latent_episode_ids": latent_ids,
        "dataset_hash": _digest(_VERSION, config_dict, seed, split, latent_ids),
        "observation_fields": list(_FIELDS),
        "units": {"position": "arbitrary distance", "time": "arbitrary time", "velocity": "distance/time", "horizons": "frames"},
        "noise": {"distribution": "independent zero-mean Gaussian", "std_per_observed_state_coordinate": config.observation_noise},
        "initial_evidence_frames": 2,
        "jump": "One position and velocity change; fresh measurements arrive at the same step",
        "target_rule": "t+h from simulation extended through T+max(h); changed-object horizons crossing an unannounced jump are invalid",
        "model_input": "observation only; state, targets, masks, role indices and event metadata are not inputs",
        "holdout_rule": "Use distinct split namespaces for independent scenes, including length holdouts",
        "scope": "Controlled synthetic state-memory mechanism diagnostic, not a visual world model benchmark",
    }
    return {
        "observation": observations,
        "targets": gather(all_state),
        "target_mask": target_mask,
        "state": all_state[:, :t].copy(),
        "changed_object": changed,
        "preserved_object": preserved,
        "reveal_step": reveal_steps,
        "episode_id": episode_ids,
        "paired_observation": paired_observations,
        "paired_targets": gather(all_paired),
        "paired_target_mask": np.ones_like(target_mask),
        "paired_state": all_paired[:, :t].copy(),
        "history_pair_observation": history_observations,
        "history_pair_targets": gather(all_history),
        "history_pair_target_mask": target_mask.copy(),
        "history_pair_state": all_history[:, :t].copy(),
        "manifest": manifest,
    }


def corrupt_history(observation: np.ndarray, mode: str, seed: int = 0) -> np.ndarray:
    """Change ONLY the first two evidence frames, never labels or later reveal.

    correct/none copies the input. delete removes initial evidence and its
    visibility flags. shuffle exchanges whole two-frame histories between
    episodes using a derangement (for a single episode, between objects).
    stale repeats frame zero's measurements at frame one; it tests outdated
    initial evidence, not refusal to accept a later correction. All modes are
    evaluation interventions unless explicitly mixed into training by a caller.
    """
    array = np.asarray(observation)
    if array.ndim != 4 or array.shape[-1] != 5 or array.shape[1] < 2 or array.shape[0] < 1:
        raise ValueError("observation must have shape [B,T,N,5], with B>=1 and T>=2")
    result = array.copy()
    if mode in ("correct", "none"):
        return result
    if mode == "delete":
        result[:, :2] = 0
    elif mode == "stale":
        result[:, 1] = array[:, 0]
    elif mode == "shuffle":
        rng = np.random.default_rng(seed)
        if array.shape[0] > 1:
            order = rng.permutation(array.shape[0])
            sources = np.empty_like(order)
            sources[order] = np.roll(order, 1)
            result[:, :2] = array[sources, :2]
        elif array.shape[2] > 1:
            order = rng.permutation(array.shape[2])
            sources = np.empty_like(order)
            sources[order] = np.roll(order, 1)
            result[0, :2] = array[0, :2][:, sources]
        else:
            raise ValueError("shuffle needs at least two episodes or two objects")
    else:
        raise ValueError(f"Unknown history corruption: {mode!r}")
    # Maintain the input contract even for externally supplied missing values.
    result[..., :4] = np.where(result[..., 4:5] > 0, result[..., :4], 0)
    return result


def history_pairs(
    config: TaskConfig,
    count: int,
    seed: int,
    split: str = "test",
    window: int = 4,
) -> dict[str, Any]:
    """Paired worlds indistinguishable from their last ``window`` observations.

    Their preserved object's history and required future differ. An observation
    window beginning at t>=2 cannot distinguish them; a model with correct
    retained history can. This supplies both worlds' labels for paired scoring.
    """
    if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= config.length - 2:
        raise ValueError("window must leave the first two evidence frames outside the recent window")
    batch = make_batch(config, count, seed, split)
    return {
        "observation": batch["observation"],
        "paired_observation": batch["history_pair_observation"],
        "targets": batch["targets"],
        "paired_targets": batch["history_pair_targets"],
        "target_mask": batch["target_mask"],
        "paired_target_mask": batch["history_pair_target_mask"],
        "preserved_object": batch["preserved_object"],
        "episode_id": batch["episode_id"],
        "window": window,
        "manifest": batch["manifest"],
    }
