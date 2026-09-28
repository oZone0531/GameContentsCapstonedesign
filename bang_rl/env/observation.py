"""Canonical Gymnasium observation-space definition for Bang agents."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import gymnasium.spaces as gym_spaces

from .types import ACTION_SPACE_SIZE, PLAYER_COUNT, PublicEnvironmentState, Role


class PublicObservationSpace(gym_spaces.Space):
    """Gymnasium space for the flat canonical public observation mapping.

    PettingZoo's generic API helper reserves ``Dict`` observations for the
    nested ``{"observation": ..., "action_mask": ...}`` convention. Bang's
    canonical observation is intentionally flat, so this remains a regular
    Gymnasium ``Space`` while retaining component lookup and containment.
    """

    def __init__(self, spaces: dict[str, gym_spaces.Space]) -> None:
        super().__init__(shape=None, dtype=None)
        self.spaces = spaces

    def __getitem__(self, key: str) -> gym_spaces.Space:
        return self.spaces[key]

    def contains(self, x) -> bool:
        if isinstance(x, PublicEnvironmentState):
            values = x.to_observation_dict()
        elif isinstance(x, Mapping):
            values = x
        else:
            return False
        return all(key in values and space.contains(values[key]) for key, space in self.spaces.items())

    def sample(self, mask=None):
        del mask
        return {key: space.sample() for key, space in self.spaces.items()}

    def seed(self, seed=None):
        return [seed]


def make_observation_space(*, max_episode_steps: int = np.iinfo(np.uint32).max) -> gym_spaces.Dict:
    """Return the fixed-shape portion of the canonical observation contract.

    Event history is intentionally raw and variable length, so it is exposed by
    the observation object but is not padded or truncated by this space factory.
    """

    if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int) or max_episode_steps < 1:
        raise ValueError("max_episode_steps must be a positive integer")
    uint32_max = np.iinfo(np.uint32).max
    return PublicObservationSpace(
        {
            "current_player_idx": gym_spaces.Box(0, PLAYER_COUNT - 1, shape=(1,), dtype=np.uint8),
            "environment_step": gym_spaces.Box(0, min(max_episode_steps, uint32_max), shape=(1,), dtype=np.uint32),
            "round_index": gym_spaces.Box(0, uint32_max, shape=(1,), dtype=np.uint32),
            "healths": gym_spaces.Box(0, 255, shape=(PLAYER_COUNT,), dtype=np.uint8),
            "alive": gym_spaces.MultiBinary(PLAYER_COUNT),
            "turn_order": gym_spaces.Box(0, PLAYER_COUNT - 1, shape=(PLAYER_COUNT,), dtype=np.uint8),
            "sheriff_idx": gym_spaces.Box(0, PLAYER_COUNT - 1, shape=(1,), dtype=np.uint8),
            "self_role": gym_spaces.Box(0, len(Role) - 1, shape=(1,), dtype=np.uint8),
            "role_visibility": gym_spaces.Box(0, 3, shape=(PLAYER_COUNT,), dtype=np.uint8),
            "revealed_roles": gym_spaces.Box(0, 255, shape=(PLAYER_COUNT,), dtype=np.uint8),
            "confirmed_role_distribution": gym_spaces.Box(
                0.0, 1.0, shape=(PLAYER_COUNT, len(Role)), dtype=np.float32
            ),
            "action_mask": gym_spaces.Box(0, 1, shape=(ACTION_SPACE_SIZE,), dtype=np.uint8),
        }
    )


__all__ = ["make_observation_space"]
