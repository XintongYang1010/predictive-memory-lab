"""Causal memory baselines for the controlled state-memory task.

Every forward call starts a new scene. The only input is ``obs[B,T,N,5]``:
position, velocity, and a visibility flag. Hidden values are explicitly masked.
Outputs are current state estimates ``[B,T,N,4]``; future targets belong solely
to the runner. One observation step is one unit of physical time.

Window and retrieval use the same reader architecture, but different bounded
histories. Retrieval retains the most recent K *observed object events*, not K
frames. Simultaneous events are ordered by object index; if the budget cuts a
frame, higher indices survive. Slot has known object correspondence, a strong
structural prior, and is a baseline rather than a claimed new architecture.

Global GRU and slot have N*hidden_dim recurrent scalars. This matches their
hidden-state budgets, NOT their parameter counts. Metadata reports both.
Memory accounting excludes parameters, transient activations, returned output,
and the training autograd graph; it counts the conceptual per-scene online
state, with a boolean or integer counted as one scalar, not one float's bytes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StateMemoryModel(nn.Module):
    """Common contract; implementations keep no scene state on the module."""

    model_name: str
    memory_scalars_per_scene: int
    memory_description: str

    def __init__(self, n_objects: int, hidden_dim: int) -> None:
        super().__init__()
        if n_objects < 1 or hidden_dim < 1:
            raise ValueError("n_objects and hidden_dim must be positive")
        self.n_objects = n_objects
        self.hidden_dim = hidden_dim

    def _observations(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        if obs.ndim != 4 or obs.shape[2:] != (self.n_objects, 5):
            raise ValueError(
                f"expected obs[B,T,{self.n_objects},5], got {tuple(obs.shape)}"
            )
        if obs.shape[1] == 0:
            raise ValueError("at least one observation step is required")
        if not obs.is_floating_point():
            raise TypeError("observations must have floating point dtype")
        visible = obs[..., 4] > 0.5
        state = torch.where(visible[..., None], obs[..., :4], 0.0)
        return state, visible

    def metadata(self) -> dict[str, int | str | bool]:
        return {
            "name": self.model_name,
            "n_objects": self.n_objects,
            "hidden_dim": self.hidden_dim,
            "memory_scalars_per_scene": self.memory_scalars_per_scene,
            "memory_description": self.memory_description,
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "object_correspondence_given": True,
            "scene_state_reset_each_forward": True,
        }


class _EventReader(nn.Module):
    """Shared reader architecture for bounded frame and event histories."""

    def __init__(self, n_objects: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_objects = n_objects
        self.query = nn.Parameter(torch.randn(n_objects, hidden_dim) * 0.02)
        self.encoder = nn.Sequential(
            nn.Linear(n_objects + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4)
        )

    def forward(self, state: Tensor, ids: Tensor, age: Tensor, valid: Tensor) -> Tensor:
        identity = F.one_hot(ids, self.n_objects).to(state.dtype)
        features = torch.cat(
            (state, identity, (age / 32.0)[..., None], torch.log1p(age)[..., None]),
            dim=-1,
        )
        encoded = self.encoder(features)
        score = torch.einsum("bkh,nh->bnk", encoded, self.query)
        score = score / math.sqrt(encoded.shape[-1])
        score = score.masked_fill(~valid[:, None, :], torch.finfo(score.dtype).min)
        # Multiplication makes an entirely unobserved history return zero context
        # without all-masked softmax NaNs. Queries still provide a learned prior.
        weight = score.softmax(dim=-1) * valid[:, None, :].to(score.dtype)
        context = torch.einsum("bnk,bkh->bnh", weight, encoded)
        query = self.query[None].expand(state.shape[0], -1, -1)
        return self.decoder(torch.cat((context, query), dim=-1))


class WindowModel(StateMemoryModel):
    model_name = "window"

    def __init__(self, n_objects: int, hidden_dim: int, window: int) -> None:
        super().__init__(n_objects, hidden_dim)
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self.reader = _EventReader(n_objects, hidden_dim)
        self.memory_scalars_per_scene = window * n_objects * 5
        self.memory_description = "W recent frames, each N*(state4+visible1); age/id implicit"

    def forward(self, obs: Tensor) -> Tensor:
        state, visible = self._observations(obs)
        batch, steps = obs.shape[:2]
        outputs = []
        for t in range(steps):
            start = max(0, t + 1 - self.window)
            count = t + 1 - start
            ids = torch.arange(self.n_objects, device=obs.device).repeat(count)
            ids = ids[None].expand(batch, -1)
            age = torch.arange(count - 1, -1, -1, device=obs.device, dtype=obs.dtype)
            age = age.repeat_interleave(self.n_objects)[None].expand(batch, -1)
            outputs.append(
                self.reader(
                    state[:, start : t + 1].reshape(batch, -1, 4),
                    ids,
                    age,
                    visible[:, start : t + 1].reshape(batch, -1),
                )
            )
        return torch.stack(outputs, dim=1)

    def metadata(self) -> dict[str, int | str | bool]:
        return {**super().metadata(), "window": self.window}


class RetrievalModel(StateMemoryModel):
    model_name = "retrieval"

    def __init__(self, n_objects: int, hidden_dim: int, memory_slots: int) -> None:
        super().__init__(n_objects, hidden_dim)
        if memory_slots < 1:
            raise ValueError("memory_slots must be positive")
        self.memory_slots = memory_slots
        self.reader = _EventReader(n_objects, hidden_dim)
        self.memory_scalars_per_scene = memory_slots * 7
        self.memory_description = "K observed events: state4+object_id1+age1+valid1; FIFO eviction"

    def forward(self, obs: Tensor) -> Tensor:
        state, visible = self._observations(obs)
        batch, steps = obs.shape[:2]
        k = self.memory_slots
        memory = obs.new_zeros(batch, k, 4)
        ids = torch.zeros(batch, k, device=obs.device, dtype=torch.long)
        age = obs.new_zeros(batch, k)
        valid = torch.zeros(batch, k, device=obs.device, dtype=torch.bool)
        current_ids = torch.arange(self.n_objects, device=obs.device)[None].expand(batch, -1)
        # All tensors retained across steps stay K long. The temporary K+N
        # candidate buffer cannot grow with scene length.
        priority = torch.arange(k + self.n_objects, device=obs.device)[None].expand(batch, -1)
        outputs = []
        for t in range(steps):
            candidate_state = torch.cat((memory, state[:, t]), dim=1)
            candidate_ids = torch.cat((ids, current_ids), dim=1)
            candidate_age = torch.cat((age + 1, obs.new_zeros(batch, self.n_objects)), dim=1)
            candidate_valid = torch.cat((valid, visible[:, t]), dim=1)
            rank = priority.masked_fill(~candidate_valid, -1)
            # Ascending indices retain chronological order for the next update.
            selected = rank.topk(k, dim=1).indices.sort(dim=1).values
            memory = candidate_state.gather(1, selected[..., None].expand(-1, -1, 4))
            ids = candidate_ids.gather(1, selected)
            age = candidate_age.gather(1, selected)
            valid = candidate_valid.gather(1, selected)
            outputs.append(self.reader(memory, ids, age, valid))
        return torch.stack(outputs, dim=1)

    def metadata(self) -> dict[str, int | str | bool]:
        return {**super().metadata(), "memory_slots": self.memory_slots}


class GlobalGRUModel(StateMemoryModel):
    model_name = "gru"

    def __init__(self, n_objects: int, hidden_dim: int) -> None:
        super().__init__(n_objects, hidden_dim)
        self.recurrent_dim = n_objects * hidden_dim
        self.cell = nn.GRUCell(n_objects * 5, self.recurrent_dim)
        self.decoder = nn.Linear(self.recurrent_dim, n_objects * 4)
        self.memory_scalars_per_scene = self.recurrent_dim
        self.memory_description = "one global recurrent hidden vector of N*hidden_dim scalars"

    def forward(self, obs: Tensor) -> Tensor:
        state, visible = self._observations(obs)
        batch, steps = obs.shape[:2]
        masked = torch.cat((state, visible[..., None].to(obs.dtype)), dim=-1)
        hidden = obs.new_zeros(batch, self.recurrent_dim)
        outputs = []
        for t in range(steps):
            hidden = self.cell(masked[:, t].reshape(batch, -1), hidden)
            outputs.append(self.decoder(hidden).reshape(batch, self.n_objects, 4))
        return torch.stack(outputs, dim=1)


class SlotModel(StateMemoryModel):
    """Known per-object slots with shared learned writing and latent dynamics.

    Objects never attend to or overwrite another object's slot. Occluded slots
    still advance via learned dynamics; they are not frozen caches. This task's
    provided object IDs make that factorization a substantive structural prior.
    """

    model_name = "slot"

    def __init__(self, n_objects: int, hidden_dim: int) -> None:
        super().__init__(n_objects, hidden_dim)
        self.writer = nn.GRUCell(4, hidden_dim)
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Begin with identity latent evolution, then learn the transition.
        nn.init.zeros_(self.dynamics[-1].weight)
        nn.init.zeros_(self.dynamics[-1].bias)
        self.decoder = nn.Linear(hidden_dim, 4)
        self.memory_scalars_per_scene = n_objects * hidden_dim
        self.memory_description = "N object slots of hidden_dim scalars; shared writer/dynamics"

    def forward(self, obs: Tensor) -> Tensor:
        state, visible = self._observations(obs)
        batch, steps = obs.shape[:2]
        hidden = obs.new_zeros(batch * self.n_objects, self.hidden_dim)
        outputs = []
        for t in range(steps):
            prior = hidden if t == 0 else hidden + self.dynamics(hidden)
            proposed = self.writer(state[:, t].reshape(-1, 4), prior)
            hidden = torch.where(visible[:, t].reshape(-1, 1), proposed, prior)
            outputs.append(self.decoder(hidden).reshape(batch, self.n_objects, 4))
        return torch.stack(outputs, dim=1)


class AnalyticModel(StateMemoryModel):
    """Zero-parameter dead reckoning from the latest visible state only.

    With noiseless observations and piecewise constant velocity, this is the
    solvability reference. Unseen changes during occlusion remain unknowable;
    it is not a hindsight oracle or a universal performance upper bound.
    """

    model_name = "analytic"

    def __init__(self, n_objects: int, hidden_dim: int) -> None:
        super().__init__(n_objects, hidden_dim)
        self.memory_scalars_per_scene = n_objects * 4
        self.memory_description = "N latest state estimates (xy,vxy), advanced at dt=1"

    def forward(self, obs: Tensor) -> Tensor:
        state, visible = self._observations(obs)
        estimate = obs.new_zeros(obs.shape[0], self.n_objects, 4)
        outputs = []
        for t in range(obs.shape[1]):
            advanced = torch.cat((estimate[..., :2] + estimate[..., 2:], estimate[..., 2:]), dim=-1)
            estimate = torch.where(visible[:, t, :, None], state[:, t], advanced)
            outputs.append(estimate)
        return torch.stack(outputs, dim=1)


def make_model(
    name: str,
    n_objects: int = 3,
    hidden_dim: int = 64,
    window: int = 4,
    memory_slots: int = 8,
) -> StateMemoryModel:
    """Construct a fresh model; all models accept observations and nothing else."""
    if name == "window":
        return WindowModel(n_objects, hidden_dim, window)
    if name == "retrieval":
        return RetrievalModel(n_objects, hidden_dim, memory_slots)
    if name == "gru":
        return GlobalGRUModel(n_objects, hidden_dim)
    if name == "slot":
        return SlotModel(n_objects, hidden_dim)
    if name == "analytic":
        return AnalyticModel(n_objects, hidden_dim)
    raise ValueError(f"unknown state memory model: {name!r}")
