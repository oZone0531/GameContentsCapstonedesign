"""Mask-compliant reproducible random agent."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Sample uniformly from enabled actions using an agent-local RNG stream."""

    def __init__(
        self,
        agent_id: str | int = "random",
        seed: int | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, seed=seed)
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator or None")
        if rng is not None and seed is not None:
            raise ValueError("provide either seed or rng, not both")
        self._rng = rng if rng is not None else np.random.default_rng(self.action_seed())

    @property
    def rng(self) -> np.random.Generator:
        """Return the private action RNG; no global RNG is consulted."""

        return self._rng

    def on_episode_start(self, context: Any) -> None:
        super().on_episode_start(context)
        if self.episode_seed is not None:
            self._rng = np.random.default_rng(self.action_seed(self.episode_seed))

    def act(self, observation: Any, action_mask: Any) -> int:
        mask = self._validate_act_inputs(observation, action_mask)
        valid_actions = np.flatnonzero(mask)
        selected = int(self._rng.choice(valid_actions))
        return self._validate_selected_action(selected, mask)

    # ``select_action`` is a descriptive alias useful to non-AEC callers.
    select_action = act


__all__ = ["RandomAgent"]
