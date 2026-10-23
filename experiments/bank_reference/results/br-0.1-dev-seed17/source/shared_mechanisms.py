"""Explicit bounded-memory adaptations of Worth Remembering and PaFu-KV.

This is NOT a reproduction of either complete paper system. WR reuses the
caller's resident recent features instead of adding a 64-frame V-JEPA encoder
window or a latent archive. Its standardized L1 score and past-only MAD gate
retain the paper's surprise mechanism. A one-observation-delayed local peak and
greedy causal NMS replace any offline peak search. Confirmed events outrank
non-events; event surprise breaks ties, and otherwise recent entries fill K.

PaFu's student consumes only causal frame-level set features. Its offline
teacher probabilities must come from the actual reader's temporal attention
probes, NOT deletion losses, cosine similarity, or task evidence. Teacher
extraction and any bidirectional-probe distribution shift belong to the runner.
Normalizing these frame scores to probabilities is an explicit adaptation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .memory import MemoryEntry, PolicyContext


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _immutable(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class SurpriseRecord:
    """Scalar-only admission metadata; never an observation representation."""

    episode_id: str
    timestamp: int
    raw_score: float
    threshold: float | None
    valid: bool
    confirmed_event: bool = False

    @property
    def entry_id(self):
        return self.episode_id, self.timestamp

    @property
    def logical_bytes(self):
        # timestamp, score, optional threshold, presence flag, two bools, ID.
        return 8 + 8 + (8 if self.threshold is not None else 0) + 3 + len(self.episode_id.encode())


class CausalSurpriseTracker:
    """WR-inspired scoring with no retained feature arrays or feature archive.

    Call ``observe(new, past_window=snapshot.recent)`` before every stream push,
    then ``retain_only(new_snapshot.recent + new_snapshot.retained)`` after it.
    ``score`` is the BoundedMemory keep-score callback. Pruning is mandatory
    before the next observation, so scalar records cannot silently accumulate.

    Gaussian statistics use at most ``window`` preceding resident observations;
    population variance is floored by ``sigma_floor**2``. The current feature is
    excluded. The score threshold is median + gamma*1.4826*MAD of ONLY preceding
    valid scores. At least ``min_history`` features and scores are required.
    Peak t is confirmed at t+1 iff s[t]>s[t-1], s[t]>=s[t+1], and s[t]>tau[t].
    Causal NMS accepts the first qualifying peak, suppressing subsequent peaks
    within ``nms_radius`` frames; it never revises an earlier decision.

    No peak is flushed at end-of-stream: doing so would use an artificial future
    boundary. A pending peak has ordinary non-event priority until confirmed.
    """

    def __init__(self, *, window=4, score_window=64, min_history=2,
                 nms_radius=3, gamma=1.0, sigma_floor=1e-4):
        self.window = _integer(window, "window", 2)
        self.score_window = _integer(score_window, "score_window", 2)
        self.min_history = _integer(min_history, "min_history", 2)
        self.nms_radius = _integer(nms_radius, "nms_radius")
        self.gamma, self.sigma_floor = float(gamma), float(sigma_floor)
        if self.min_history > min(self.window, self.score_window):
            raise ValueError("min_history must fit both bounded windows")
        if not np.isfinite(self.gamma) or self.gamma < 0 or not np.isfinite(self.sigma_floor) or self.sigma_floor <= 0:
            raise ValueError("gamma must be nonnegative and sigma_floor positive, both finite")
        self.reset()

    def reset(self):
        self._records = {}
        self._scores = deque(maxlen=self.score_window)
        self._episode_id = None
        self._last_time = None
        self._feature_spec = None
        self._seen = 0
        self._older_score = None
        self._last_peak_time = None
        self._needs_prune = False

    @property
    def records(self):
        return tuple(sorted(self._records.values(), key=lambda r: r.timestamp))

    def observe(self, entry: MemoryEntry, past_window: Sequence[MemoryEntry]):
        if not isinstance(entry, MemoryEntry):
            raise ValueError("observe requires a MemoryEntry")
        if self._needs_prune:
            raise ValueError("retain_only must prune resident scalar records after every push")
        if self._episode_id is not None and entry.episode_id != self._episode_id:
            raise ValueError("reset is required before changing episodes")
        if self._last_time is not None and entry.timestamp != self._last_time + 1:
            raise ValueError("duplicate, resurrected, or non-contiguous observation")
        spec = (entry.representation.shape, entry.representation.dtype.str)
        if self._feature_spec is not None and spec != self._feature_spec:
            raise ValueError("feature shape/dtype changed within an episode")
        supplied = tuple(past_window)
        if any(not isinstance(e, MemoryEntry) for e in supplied):
            raise ValueError("past_window must contain MemoryEntry values")
        times = [e.timestamp for e in supplied]
        if times != sorted(set(times)) or any(t >= entry.timestamp for t in times):
            raise ValueError("past_window must be strictly past and timestamp ordered")
        if any(e.episode_id != entry.episode_id or e.entry_id not in self._records for e in supplied):
            raise ValueError("past_window contains nonresident or cross-episode history")
        if any((e.representation.shape, e.representation.dtype.str) != spec for e in supplied):
            raise ValueError("past_window feature shape/dtype mismatch")
        past = supplied[-self.window:]
        required = min(self.window, self._seen)
        expected = list(range(entry.timestamp - required, entry.timestamp))
        if [e.timestamp for e in past] != expected:
            raise ValueError("resident recent buffer is too short for the declared Gaussian window")

        valid = len(past) >= self.min_history
        score = 0.0
        threshold = None
        if valid:
            z = np.stack([e.representation for e in past]).astype(np.float64)
            sigma = np.sqrt(np.maximum(z.var(axis=0), self.sigma_floor ** 2))
            score = float(np.abs((entry.representation.astype(np.float64) - z.mean(axis=0)) / sigma).mean())
            if not np.isfinite(score):
                raise ValueError("surprise overflowed; inspect the feature scale")
            if len(self._scores) >= self.min_history:
                history = np.asarray(self._scores, np.float64)
                median = float(np.median(history))
                threshold = median + self.gamma * 1.4826 * float(np.median(np.abs(history - median)))
        record = SurpriseRecord(entry.episode_id, entry.timestamp, score, threshold, valid)

        previous = self._records.get((entry.episode_id, self._last_time))
        if previous is not None and previous.valid and valid and self._older_score is not None:
            peak = (previous.threshold is not None and previous.raw_score > previous.threshold
                    and previous.raw_score > self._older_score and previous.raw_score >= score)
            separated = self._last_peak_time is None or previous.timestamp - self._last_peak_time > self.nms_radius
            if peak and separated:
                self._records[previous.entry_id] = replace(previous, confirmed_event=True)
                self._last_peak_time = previous.timestamp
        self._older_score = previous.raw_score if previous is not None and previous.valid else None
        if valid:
            self._scores.append(score)
        self._records[entry.entry_id] = record
        self._episode_id, self._last_time = entry.episode_id, entry.timestamp
        self._feature_spec = spec
        self._seen += 1
        self._needs_prune = True
        return record

    def retain_only(self, entries: Iterable[MemoryEntry]):
        """Discard scalar metadata for evicted observations; never recreate it."""
        entries = tuple(entries)
        if not self._needs_prune:
            raise ValueError("retain_only must follow one observation and stream push")
        if any(not isinstance(e, MemoryEntry) for e in entries):
            raise ValueError("resident entries must be MemoryEntry values")
        ids = {e.entry_id for e in entries}
        if len(ids) != len(entries) or not ids <= self._records.keys():
            raise ValueError("duplicate or resurrected resident entry")
        if (self._episode_id, self._last_time) not in ids:
            raise ValueError("current observation must remain in the recent buffer")
        self._records = {key: self._records[key] for key in ids}
        self._needs_prune = False

    def score(self, candidates, context: PolicyContext):
        """Confirmed events first, surprise within events, recency otherwise."""
        candidates = tuple(candidates)
        if not isinstance(context, PolicyContext) or context.current_time != self._last_time:
            raise ValueError("score requires the current causal context")
        if not candidates or any(not isinstance(e, MemoryEntry) or e.entry_id not in self._records for e in candidates):
            raise ValueError("score received missing or resurrected candidates")
        all_entries = candidates + context.recent
        if (len({e.entry_id for e in all_entries}) != len(all_entries)
                or any(e.episode_id != self._episode_id or e.timestamp > context.current_time for e in all_entries)
                or (context.recent and max(e.timestamp for e in candidates) >= context.recent[0].timestamp)):
            raise ValueError("candidate/recent identities cross the causal boundary")
        values = []
        for entry in candidates:
            record = self._records[entry.entry_id]
            values.append(2.0 + record.raw_score / (1.0 + record.raw_score)
                          if record.confirmed_event else entry.timestamp / (1.0 + context.current_time))
        return np.asarray(values, np.float64)

    def accounting(self):
        records = sum(record.logical_bytes for record in self._records.values())
        scores = 8 * len(self._scores)
        # Four integer configuration fields; two float configurations; seen,
        # optional timestamps/previous score with presence flags; prune flag;
        # feature specification and episode identity. Logical packed bytes only.
        optional = sum(1 + (8 if value is not None else 0)
                       for value in (self._last_time, self._older_score, self._last_peak_time))
        spec = 0 if self._feature_spec is None else 8 * len(self._feature_spec[0]) + len(self._feature_spec[1].encode())
        metadata = 4 * 8 + 2 * 8 + 8 + optional + 1 + spec + len((self._episode_id or "").encode())
        return {"auxiliary_logical_bytes": records + scores + metadata,
                "resident_scalar_record_bytes": records, "resident_scalar_records": len(self._records),
                "score_ring_bytes": scores, "score_ring_length": len(self._scores),
                "configuration_and_state_bytes": metadata, "feature_archive_bytes": 0,
                "peak_confirmation_delay": 1, "includes_python_allocator_overhead": False}


def make_student_inputs(candidate_projected, recent_projected, candidate_times,
                        recent_times, current_time):
    """C x (5D+1): candidate,current,delta,candidate mean,recent mean,log1p age.

    These are set-equivariant causal features. The caller may apply a frozen,
    train-only projection upstream; this function fits no normalization and
    receives no teacher target, future action, episode ID, or task evidence.
    """
    current_time = _integer(current_time, "current_time")
    candidate, recent = np.asarray(candidate_projected), np.asarray(recent_projected)
    if (candidate.ndim != 2 or recent.ndim != 2 or min(candidate.shape) < 1 or min(recent.shape) < 1
            or candidate.shape[1] != recent.shape[1] or candidate.dtype != np.float32 or recent.dtype != np.float32
            or not np.isfinite(candidate).all() or not np.isfinite(recent).all()):
        raise ValueError("candidate/recent features must be finite nonempty float32 matrices of equal width")
    ct, rt = np.asarray(candidate_times), np.asarray(recent_times)
    if (ct.shape != (len(candidate),) or rt.shape != (len(recent),)
            or not np.issubdtype(ct.dtype, np.integer) or not np.issubdtype(rt.dtype, np.integer)
            or np.any(ct < 0) or np.any(rt < 0) or len(np.unique(ct)) != len(ct)
            or np.any(np.diff(rt) <= 0) or rt[-1] != current_time or np.any(ct >= rt[0])):
        raise ValueError("timestamps must be unique; candidates precede ordered recent context ending now")
    current = np.broadcast_to(recent[-1], candidate.shape)
    candidate_mean = np.broadcast_to(candidate.mean(axis=0), candidate.shape)
    recent_mean = np.broadcast_to(recent.mean(axis=0), candidate.shape)
    ages = np.log1p(current_time - ct).astype(np.float32)[:, None]
    values = np.concatenate([candidate, current, candidate - current, candidate_mean, recent_mean, ages], axis=1)
    if not np.isfinite(values).all():
        raise ValueError("student features overflowed")
    return _immutable(values.astype(np.float32, copy=False))


def balanced_attention_targets(attention, query_times, candidate_times, *, decision_time=None):
    """Offline PaFu-inspired temporal attention target, never an online scorer.

    ``attention[query,key]`` must be actual normalized reader weights. For EACH
    key j, partition query timestamps into past/same/future relative to key j;
    take the maximum in each nonempty block and average available block maxima.
    Finally normalize across candidate keys. Missing blocks are omitted, never
    padded with fictitious zeros. At least one genuinely later query is required
    per key; otherwise this is not a future-attention target. The runner should
    also pass ``decision_time``: this requires a query strictly after the ONLINE
    decision, not merely after an old key timestamp. The runner must separately
    fix the maximum future horizon before label generation.

    An earlier query reading a later key is a bidirectional teacher probe, not a
    causal rollout of a predictor trained only with past keys. The caller must
    report that distribution shift. The function neither invents attention nor
    accepts losses in its place; trace/provenance audits remain runner duties.
    """
    a = np.asarray(attention)
    qt, kt = np.asarray(query_times), np.asarray(candidate_times)
    if (a.ndim != 2 or min(a.shape) < 1 or a.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(a).all() or np.any(a < 0)
            or not np.allclose(a.sum(axis=1), 1., rtol=0., atol=1e-5)):
        raise ValueError("attention must contain finite nonnegative normalized reader-weight rows")
    if (qt.shape != (a.shape[0],) or kt.shape != (a.shape[1],)
            or not np.issubdtype(qt.dtype, np.integer) or not np.issubdtype(kt.dtype, np.integer)
            or np.any(qt < 0) or np.any(kt < 0) or len(np.unique(qt)) != len(qt)
            or len(np.unique(kt)) != len(kt)):
        raise ValueError("query/key times must be nonnegative unique integer vectors of matching size")
    if decision_time is not None:
        decision_time = _integer(decision_time, "decision_time")
        if np.any(kt > decision_time) or not np.any(qt > decision_time):
            raise ValueError("teacher needs past/current keys and an actual query after the online decision")
    salience = np.empty(len(kt), np.float64)
    for j, time in enumerate(kt):
        groups = (qt < time, qt == time, qt > time)
        if not groups[-1].any():
            raise ValueError("every key needs an actual future attention query")
        salience[j] = np.mean([a[mask, j].max() for mask in groups if mask.any()])
    if not np.isfinite(salience).all() or salience.sum() <= 0:
        raise ValueError("attention salience cannot be normalized")
    return _immutable((salience / salience.sum()).astype(np.float32))


class SalienceStudent(nn.Module):
    """Shared two-layer scalar MLP; cross-candidate context is explicit in input."""

    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.input_dim = _integer(input_dim, "input_dim", 1)
        self.hidden = _integer(hidden, "hidden", 1)
        self.network = nn.Sequential(nn.Linear(self.input_dim, self.hidden), nn.SiLU(), nn.Linear(self.hidden, 1))

    def forward(self, inputs):
        if (inputs.ndim != 2 or inputs.shape[1] != self.input_dim or inputs.shape[0] < 1
                or inputs.dtype != torch.float32 or not torch.isfinite(inputs).all()):
            raise ValueError("student inputs must be a finite float32 C x input_dim tensor")
        return self.network(inputs).squeeze(-1)


@dataclass(frozen=True, eq=False)
class SalienceTrainingRecord:
    inputs: np.ndarray
    teacher_probability: np.ndarray
    group_id: str
    split: str = "train"
    target_kind: str = "future_attention"

    def __post_init__(self):
        inputs, target = np.asarray(self.inputs), np.asarray(self.teacher_probability)
        if inputs.ndim != 2 or min(inputs.shape) < 1 or inputs.dtype != np.float32 or not np.isfinite(inputs).all():
            raise ValueError("inputs must be a finite nonempty float32 matrix")
        if (target.shape != (len(inputs),) or target.dtype != np.float32 or not np.isfinite(target).all()
                or np.any(target < 0) or not np.isclose(target.sum(dtype=np.float64), 1., rtol=0., atol=1e-5)):
            raise ValueError("teacher_probability must be finite nonnegative float32 probabilities summing to one")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("a nonempty independent group_id is required")
        if self.split not in ("train", "validation", "test") or self.target_kind != "future_attention":
            raise ValueError("only explicit split names and future_attention targets are accepted; never deletion losses")
        object.__setattr__(self, "inputs", _immutable(inputs))
        object.__setattr__(self, "teacher_probability", _immutable(target))


def student_probabilities(model: SalienceStudent, inputs):
    """Pure causal inference; owns no stream, labels, or history archive."""
    if not isinstance(model, SalienceStudent):
        raise ValueError("a SalienceStudent is required")
    values = np.asarray(inputs)
    if values.dtype != np.float32:
        raise ValueError("inputs must be float32")
    with torch.inference_mode():
        tensor = torch.from_numpy(values.copy()).to(next(model.parameters()).device)
        return torch.softmax(model(tensor), dim=0).cpu().numpy().copy()


def train_salience_student(records: Sequence[SalienceTrainingRecord], *, seed,
                           epochs=100, hidden=32, lr=1e-3, train_group_ids=None):
    """Train-only full-batch SmoothL1 on probabilities; return FINAL model, curve.

    Each state contributes equally regardless of candidate count. Curve[0] is
    the initial training loss and curve[e] is loss after e updates. No validation
    checkpoint selection, deletion loss input, future feature input, or hidden
    normalization is performed. The returned CPU student is frozen and in eval
    mode. ``train_group_ids`` optionally enforces an externally fixed split.
    """
    seed, epochs = _integer(seed, "seed"), _integer(epochs, "epochs", 1)
    hidden = _integer(hidden, "hidden", 1)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    records = tuple(records)
    if not records or any(not isinstance(record, SalienceTrainingRecord) for record in records):
        raise ValueError("nonempty SalienceTrainingRecord sequence required")
    if any(record.split != "train" or record.target_kind != "future_attention" for record in records):
        raise ValueError("only train-split future_attention records may train the student")
    if train_group_ids is not None:
        permitted = set(train_group_ids)
        if any(record.group_id not in permitted for record in records):
            raise ValueError("record group lies outside the fixed train group IDs")
    width = records[0].inputs.shape[1]
    if any(record.inputs.shape[1] != width for record in records):
        raise ValueError("all training states must share an input width")
    tensors = torch.from_numpy(np.concatenate([record.inputs for record in records]).copy())
    sizes = [len(record.inputs) for record in records]
    targets = [torch.from_numpy(record.teacher_probability.copy()) for record in records]

    # CPU RNG state is restored, so fitting a baseline does not perturb another
    # method's independently fixed initialization or episode generation seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SalienceStudent(width, hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

        def objective():
            logits = model(tensors).split(sizes)
            return torch.stack([F.smooth_l1_loss(torch.softmax(logit, dim=0), target)
                                for logit, target in zip(logits, targets)]).mean()

        with torch.no_grad():
            curve = [float(objective())]
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite salience training loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                curve.append(float(objective()))
    model.eval()
    model.requires_grad_(False)
    return model, curve


def student_accounting(model: SalienceStudent):
    parameters = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return {"student_parameter_bytes": parameters, "student_buffer_bytes": buffers,
            "student_logical_bytes": parameters + buffers + 16,
            "stream_archive_bytes": 0, "includes_python_allocator_overhead": False}
