"""Causal, candidate-only retention policies, independent of any checkpoint.

The caller moves frames out of its recent buffer in timestamp order and calls
``observe`` once per eligible frame. A policy can inspect only the current K
entries, the one incoming entry, and explicitly supplied current/recent context.
The class retains no observation archive or callback outputs after a decision.

Engineering choices (not extra proposal requirements): timestamps are integer
frame indices; IDs are canonical episode/time pairs; exact score ties discard
the oldest candidate. Temporal buckets have geometric age boundaries beginning
at 8 frames by default. Diversity maximizes minimum then mean cosine distance.
Missing surprise at warm-up is lower priority than any observed finite score.

An attention policy is *only* a selection adapter until a predictor attention
callback is supplied. This module neither fabricates attention nor implements a
predictor. The callback contract prohibits future targets/actions and state or
evidence labels; callers must separately audit what the callback closes over.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Sequence

import numpy as np


POLICIES = ("fifo", "reservoir", "temporal_buckets", "visual_diversity",
            "prediction_surprise", "attention")


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


@dataclass(frozen=True, eq=False)
class MemoryEntry:
    episode_id: str
    timestamp: int
    representation: np.ndarray
    surprise: float | None = None

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a nonempty string")
        object.__setattr__(self, "timestamp", _integer(self.timestamp, "timestamp"))
        z = np.asarray(self.representation)
        if z.ndim != 1 or not z.size or z.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("representation must be a nonempty 1-D float32/float64 array")
        if not np.isfinite(z).all():
            raise ValueError("representation must be finite")
        # A bytes backing store also prevents callers from re-enabling writes
        # with setflags; a read-only owning ndarray would allow that mutation.
        z = np.frombuffer(z.tobytes(order="C"), dtype=z.dtype)
        object.__setattr__(self, "representation", z)
        if self.surprise is not None:
            score = float(self.surprise)
            if not np.isfinite(score) or score < 0:
                raise ValueError("stored surprise must be a finite nonnegative past-computed error")
            object.__setattr__(self, "surprise", score)

    @property
    def entry_id(self) -> tuple[str, int]:
        """Canonical IDs make duplicate/reintroduced old frames detectable in O(1)."""
        return self.episode_id, self.timestamp

    @property
    def payload_bytes(self) -> int:
        # Logical serialized payload, not Python allocator/RSS overhead.
        return self.representation.nbytes + 8 + len(self.episode_id.encode("utf-8")) + (8 if self.surprise is not None else 0)


@dataclass(frozen=True)
class PolicyContext:
    current_time: int
    recent: tuple[MemoryEntry, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "current_time", _integer(self.current_time, "current_time"))
        object.__setattr__(self, "recent", tuple(self.recent))
        if any(not isinstance(e, MemoryEntry) for e in self.recent):
            raise ValueError("recent must contain MemoryEntry instances")
        times = [e.timestamp for e in self.recent]
        if times != sorted(set(times)) or any(t > self.current_time for t in times):
            raise ValueError("recent timestamps must be distinct, ordered, and no later than current_time")


ScoreCallback = Callable[[tuple[MemoryEntry, ...], PolicyContext], Sequence[float]]


@dataclass(frozen=True)
class Transition:
    kept_ids: tuple[tuple[str, int], ...]
    deleted_id: tuple[str, int] | None
    arrivals_seen: int


class BoundedMemory:
    """A single irreversible stream; ``reset`` starts a new episode/RNG stream.

    Reservoir uses Algorithm R: for arrival n>K draw j uniformly in [0,n),
    replace slot j if j<K, otherwise reject the incoming observation. Every
    arrival increments n, including rejections. This differs from repeatedly
    deleting uniformly among K+1 candidates.

    Score callbacks receive immutable candidate tuples and legal context only.
    They return one scalar per candidate, where larger means retain. For stored
    prediction surprise, compute the scalar causally outside this class before
    constructing the entry. Any forecast queues used to compute it must be
    included separately in memory/latency reports.
    """

    def __init__(self, capacity: int, policy: str, *, seed: int = 0,
                 score_callback: ScoreCallback | None = None,
                 temporal_bucket_width: int = 8):
        self.capacity = _integer(capacity, "capacity", 1)
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy {policy!r}; choose one of {POLICIES}")
        if score_callback is not None and policy not in ("prediction_surprise", "attention"):
            raise ValueError("score callbacks are supported only for surprise and attention")
        if policy == "attention" and score_callback is None:
            raise ValueError("attention requires an explicit audited predictor-attention callback")
        if score_callback is not None and not callable(score_callback):
            raise ValueError("score_callback must be callable")
        self.policy = policy
        self.seed = _integer(seed, "seed")
        self.score_callback = score_callback
        self.temporal_bucket_width = _integer(temporal_bucket_width, "temporal_bucket_width", 1)
        self.reset()

    def reset(self, episode_id: str | None = None, *, seed: int | None = None) -> None:
        if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
            raise ValueError("episode_id must be a nonempty string")
        self._rng = np.random.default_rng(self.seed if seed is None else _integer(seed, "seed"))
        self._entries: tuple[MemoryEntry, ...] = ()
        self._episode_id = episode_id
        self._latest_timestamp = -1
        self._latest_context_time = -1
        self._feature_spec: tuple | None = None
        self.arrivals_seen = 0

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return self._entries

    def observe(self, entry: MemoryEntry, context: PolicyContext) -> Transition:
        if not isinstance(entry, MemoryEntry) or not isinstance(context, PolicyContext):
            raise ValueError("observe requires a MemoryEntry and PolicyContext")
        if self._episode_id is not None and entry.episode_id != self._episode_id:
            raise ValueError("reset is required before changing episodes")
        if entry.timestamp <= self._latest_timestamp:
            raise ValueError("duplicate, out-of-order, or resurrected observation")
        if entry.timestamp > context.current_time or context.current_time < self._latest_context_time:
            raise ValueError("future observation or backwards current_time")
        spec = (entry.representation.shape, entry.representation.dtype.str)
        if self._feature_spec is not None and spec != self._feature_spec:
            raise ValueError("feature shape/dtype changed within an episode")
        for recent in context.recent:
            if recent.episode_id != entry.episode_id or recent.timestamp <= entry.timestamp:
                raise ValueError("recent and historical roles must be distinct and from the same episode")
            if (recent.representation.shape, recent.representation.dtype.str) != spec:
                raise ValueError("recent feature shape/dtype mismatch")

        # Work locally so failed scoring does not partially advance stream state.
        candidates = self._entries + (entry,)
        n = self.arrivals_seen + 1
        deletion = self._select_deletion(candidates, context, n) if len(candidates) > self.capacity else None
        kept = tuple(e for i, e in enumerate(candidates) if i != deletion)
        self._entries = kept
        self._episode_id = entry.episode_id
        self._feature_spec = spec
        self._latest_timestamp = entry.timestamp
        self._latest_context_time = context.current_time
        self.arrivals_seen = n
        return Transition(tuple(e.entry_id for e in kept),
                          None if deletion is None else candidates[deletion].entry_id, n)

    def _select_deletion(self, entries: tuple[MemoryEntry, ...], context: PolicyContext, n: int) -> int:
        if self.policy == "fifo":
            return 0
        if self.policy == "reservoir":
            j = int(self._rng.integers(n))
            return j if j < self.capacity else self.capacity
        if self.policy in ("attention", "prediction_surprise"):
            if self.score_callback is not None:
                scores = np.asarray(self.score_callback(entries, context), dtype=np.float64)
                if scores.shape != (len(entries),) or not np.isfinite(scores).all():
                    raise ValueError("callback must return exactly one finite score per candidate")
            else:
                scores = np.array([-np.inf if e.surprise is None else e.surprise for e in entries])
            return int(np.argmin(scores))
        # Enumerating K+1 deletions ensures candidate-only equal-capacity choices.
        best_index, best_value = 0, None
        for dropped in range(len(entries)):
            retained = tuple(e for i, e in enumerate(entries) if i != dropped)
            if self.policy == "temporal_buckets":
                boundaries = [self.temporal_bucket_width * 2**j for j in range(self.capacity - 1)]
                buckets = np.searchsorted(boundaries, [context.current_time - e.timestamp for e in retained], side="right")
                value = (len(set(buckets.tolist())), sum(e.timestamp for e in retained))
            else:
                z = np.stack([e.representation for e in retained]).astype(np.float64)
                norms = np.linalg.norm(z, axis=1)
                unit = z / np.maximum(norms[:, None], np.finfo(np.float64).tiny)
                similarity = np.clip(unit @ unit.T, -1., 1.)
                similarity[np.logical_and(norms[:, None] == 0, norms[None, :] == 0)] = 1.
                distances = (1. - similarity)[np.triu_indices(len(retained), 1)]
                value = (float(distances.min()) if len(distances) else 0.,
                         float(distances.mean()) if len(distances) else 0.,
                         sum(e.timestamp for e in retained))
            if best_value is None or value > best_value:
                best_index, best_value = dropped, value
        return best_index

    def accounting(self, context: PolicyContext | None = None) -> dict:
        """Logical payload accounting; explicitly excludes Python and model RSS.

        Callback model weights/temporary workspaces/forecast queues are owned by
        the caller. Their sizes must be added by the experiment runner; they are
        not silently claimed to be zero here.
        """
        features = sum(e.representation.nbytes for e in self._entries)
        entries = sum(e.payload_bytes for e in self._entries)
        rng = len(json.dumps(self._rng.bit_generator.state, sort_keys=True).encode())
        # n, last observation time, last context time, K, seed, bucket width.
        spec = 0 if self._feature_spec is None else 8 * len(self._feature_spec[0]) + len(self._feature_spec[1].encode())
        policy = 6 * 8 + rng + spec + len((self._episode_id or "").encode()) + len(self.policy.encode())
        recent = sum(e.payload_bytes for e in context.recent) if context is not None else 0
        return {"real_entries": len(self._entries), "capacity": self.capacity,
                "feature_bytes": features, "entry_metadata_bytes": entries - features,
                "policy_metadata_bytes": policy, "recent_payload_bytes": recent,
                "logical_payload_bytes": entries + policy + recent,
                "includes_python_allocator_overhead": False,
                "external_callback_storage_must_be_accounted": self.score_callback is not None}


@dataclass(frozen=True)
class StreamingSnapshot:
    """A frozen value returned to the caller, never an internal history archive.

    ``transition`` describes this push's historical admission/deletion, or None
    during recent-buffer warm-up. ``StreamingMemory.snapshot`` omits that event.
    Saving snapshots for audit is the caller's responsibility; old snapshots
    must not be exposed as an inference-time observation source.
    """

    current_time: int | None
    recent: tuple[MemoryEntry, ...]
    retained: tuple[MemoryEntry, ...]
    observations_seen: int
    transition: Transition | None = None

    @property
    def current(self) -> MemoryEntry | None:
        return self.recent[-1] if self.recent else None


class StreamingMemory:
    """Own the recent R-buffer and irreversible K-entry memory together.

    A push first forms the next recent buffer. If the buffer overflows, its
    oldest entry becomes the sole eligible historical candidate, and retention
    sees the *new* current/recent context. Thus historical and recent entries
    never overlap. At most R+K observations persist, with one additional real
    observation temporarily present while making an overflow decision.

    By default arrivals must be consecutive integer frame indices; the first
    timestamp may be any nonnegative index. ``require_contiguous=False`` allows
    explicitly subsampled but strictly increasing streams. In that mode R means
    R observations, not R elapsed frames; original timestamps remain unchanged.
    Surprise, when supplied, is frozen at observation creation and travels with
    the same entry until eviction. This wrapper never recalculates it later.
    """

    def __init__(self, recent_length: int, capacity: int, policy: str, *, seed: int = 0,
                 score_callback: ScoreCallback | None = None,
                 temporal_bucket_width: int = 8, require_contiguous: bool = True):
        self.recent_length = _integer(recent_length, "recent_length", 1)
        if not isinstance(require_contiguous, bool):
            raise ValueError("require_contiguous must be a bool")
        self.require_contiguous = require_contiguous
        self._memory = BoundedMemory(capacity, policy, seed=seed,
                                     score_callback=score_callback,
                                     temporal_bucket_width=temporal_bucket_width)
        self._recent: tuple[MemoryEntry, ...] = ()
        self._episode_id: str | None = None
        self.observations_seen = 0

    @property
    def snapshot(self) -> StreamingSnapshot:
        return StreamingSnapshot(self._recent[-1].timestamp if self._recent else None,
                                 self._recent, self._memory.entries,
                                 self.observations_seen)

    def push(self, observation: MemoryEntry) -> StreamingSnapshot:
        if not isinstance(observation, MemoryEntry):
            raise ValueError("push requires a MemoryEntry")
        if self._episode_id is not None and observation.episode_id != self._episode_id:
            raise ValueError("reset is required before changing episodes")
        if self._recent:
            latest = self._recent[-1]
            if observation.timestamp <= latest.timestamp:
                raise ValueError("duplicate, out-of-order, or resurrected observation")
            if self.require_contiguous and observation.timestamp != latest.timestamp + 1:
                raise ValueError("contiguous observations required; explicitly opt in to sparse arrivals")
            if (observation.representation.shape != latest.representation.shape
                    or observation.representation.dtype != latest.representation.dtype):
                raise ValueError("feature shape/dtype changed within an episode")

        # Keep all work local until the retention callback succeeds. In
        # particular, failed scoring must not consume or drop a recent frame.
        recent = self._recent + (observation,)
        transition = None
        if len(recent) > self.recent_length:
            leaving, recent = recent[0], recent[1:]
            context = PolicyContext(observation.timestamp, recent)
            transition = self._memory.observe(leaving, context)
        self._recent = recent
        self._episode_id = observation.episode_id
        self.observations_seen += 1
        return StreamingSnapshot(observation.timestamp, recent, self._memory.entries,
                                 self.observations_seen, transition)

    def reset(self, episode_id: str | None = None, *, seed: int | None = None) -> None:
        # Validate and reset the underlying RNG before committing wrapper state.
        self._memory.reset(episode_id, seed=seed)
        self._recent = ()
        self._episode_id = episode_id
        self.observations_seen = 0

    def accounting(self) -> dict:
        context = PolicyContext(self._recent[-1].timestamp if self._recent else 0, self._recent)
        result = self._memory.accounting(context)
        # R, observation count, contiguous flag, wrapper episode identity.
        wrapper = 2 * 8 + 1 + len((self._episode_id or "").encode())
        result["stream_metadata_bytes"] = wrapper
        result["logical_payload_bytes"] += wrapper
        result.update(recent_length=self.recent_length,
                      resident_observations=len(self._recent) + len(self._memory.entries),
                      max_transient_observations=self.recent_length + self._memory.capacity + 1,
                      current_shares_recent_representation=True)
        return result
