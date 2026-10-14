"""V6: predictive associative memory for world models.

One recurrence per step, studied on object tokens only (no perception):

* predict  -- evolve every stored content on the VALUE side with a learned,
              action/dt-conditioned linear generator that contains a shear
              (position += velocity * dt): ``S <- expm(dt * L(u)) S``;
* compare  -- residual between an observed token and the memory's own
              pre-write prediction; the same residual is the surprise signal;
* correct  -- delta-rule write of that residual, ``S <- S + beta G J^T e k^T``,
              which is one preconditioned TTT/SGD step on the memory's own
              observation-prediction loss.

The package is split so that everything except the learned memories runs on
numpy only (world generators, references, metrics, loaders, surprise logic).
Torch is imported lazily by ``memories``, ``train`` and ``evaluate``.

Protocol, hypotheses and budgets: ``docs/V6_PREDICTIVE_MEMORY.md`` and
``docs/PREREGISTRATION_V6.md``.
"""

SCHEMA = "pmr-v6-1"
