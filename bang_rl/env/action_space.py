"""Discrete(6) action space, legal-mask helpers, and validation boundary.

Mapping: index 0 = PASS; index 1..5 = BANG targeting player 0..4 (target =
index - 1).  A player can never target itself, a dead player, or act in a
terminal state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium.spaces as gym_spaces
import numpy as np

from .types import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    PLAYER_COUNT,
    Action,
    ActionType,
)


ACTION_MASK_SHAPE: tuple[int, ...] = (ACTION_SPACE_SIZE,)
ACTION_SPACE = gym_spaces.Discrete(ACTION_SPACE_SIZE, start=0)
ACTION_MASK_SPACE = gym_spaces.Box(low=0, high=1, shape=ACTION_MASK_SHAPE, dtype=np.uint8)


def encode_action(action: Action) -> int:
    if action.action_type is ActionType.PASS:
        return 0
    if action.target_idx is None:
        raise ValueError("a BANG action requires a target")
    return int(action.target_idx) + 1


def decode_action(index: Any, *, actor_idx: int) -> Action:
    if isinstance(index, bool):
        raise TypeError("action index must be an integer, not a bool")
    if not isinstance(index, (int, np.integer)):
        raise TypeError("action index must be an integer")
    value = int(index)
    if value < 0 or value >= ACTION_SPACE_SIZE:
        raise ValueError(f"action index {value} is out of range")
    if value == 0:
        return Action.pass_action(actor_idx)
    return Action.bang(actor_idx=actor_idx, target_idx=value - 1)


def legal_action_mask(
    alive: Sequence[Any],
    acting_player_idx: int,
    *,
    terminal: bool = False,
) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    alive_flags = [bool(x) for x in alive]
    if terminal or not (0 <= acting_player_idx < len(alive_flags)) or not alive_flags[acting_player_idx]:
        return mask
    mask[0] = 1
    for target_idx, is_alive in enumerate(alive_flags):
        if target_idx != acting_player_idx and is_alive:
            mask[target_idx + 1] = 1
    return mask


def action_mask_for_state(state: Any) -> np.ndarray:
    terminal = bool(getattr(state, "terminated", False) or getattr(state, "truncated", False))
    return legal_action_mask(state.alive, int(state.agent_selection), terminal=terminal)


def valid_action_indices(mask: Sequence[Any]) -> tuple[int, ...]:
    array = np.asarray(mask).reshape(-1)
    return tuple(int(i) for i in np.flatnonzero(array))


def select_deterministic_action(logits: Sequence[float], mask: Sequence[Any]) -> int:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    mask_array = np.asarray(mask).reshape(-1)
    valid = np.flatnonzero(mask_array)
    if valid.size == 0:
        raise ValueError("no legal action available in the mask")
    best = valid[int(np.argmax(values[valid]))]
    # Break ties on the lowest index to stay deterministic.
    best_value = values[best]
    for index in valid:
        if np.isclose(values[index], best_value):
            return int(index)
    return int(best)


def serialize_action_mask(mask: Sequence[Any]) -> dict[str, Any]:
    array = np.asarray(mask, dtype=np.uint8).reshape(-1)
    return {"shape": [int(array.shape[0])], "dtype": "uint8", "mask": [int(x) for x in array.tolist()]}


def deserialize_action_mask(data: Any) -> np.ndarray:
    shape = tuple(int(x) for x in data["shape"])
    if shape != ACTION_MASK_SHAPE:
        raise ValueError(f"action mask shape must be {list(ACTION_MASK_SHAPE)}")
    values = [int(x) for x in data["mask"]]
    if any(value not in (0, 1) for value in values):
        raise ValueError("action mask must contain only 0 or 1")
    array = np.asarray(values, dtype=np.uint8)
    if array.shape != ACTION_MASK_SHAPE:
        raise ValueError("action mask values do not match the declared shape")
    return array


@dataclass(frozen=True)
class ActionMask:
    values: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: Any) -> "ActionMask":
        array = deserialize_action_mask(data)
        return cls(tuple(int(x) for x in array.tolist()))

    def to_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.uint8)

    def to_dict(self) -> dict[str, Any]:
        return serialize_action_mask(self.to_array())


def action_space_spec() -> dict[str, Any]:
    mapping = {"0": "PASS"}
    for target in range(PLAYER_COUNT):
        mapping[str(target + 1)] = f"PLAYER_{target}_BANG"
    return {
        "version": ACTION_SPACE_VERSION,
        "n": ACTION_SPACE_SIZE,
        "mapping": mapping,
        "mask": {"shape": [ACTION_SPACE_SIZE], "dtype": "uint8", "values": [0, 1]},
    }


def make_action_observation_space() -> gym_spaces.Dict:
    return gym_spaces.Dict({"action_mask": gym_spaces.Box(low=0, high=1, shape=ACTION_MASK_SHAPE, dtype=np.uint8)})


class ActionValidationBoundary:
    """Pure validation seam that only inspects a supplied state."""

    def __init__(self, state: Any) -> None:
        self.state = state

    def current_action_mask(self) -> np.ndarray:
        return action_mask_for_state(self.state)

    def validate(self, action: Any) -> int:
        if isinstance(action, bool):
            raise TypeError("action must be an integer, not a bool")
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        value = int(action)
        mask = self.current_action_mask()
        if value < 0 or value >= ACTION_SPACE_SIZE:
            raise ValueError(f"action {value} is out of range")
        if not mask[value]:
            raise ValueError(f"action {value} is disallowed by the current mask")
        return value


__all__ = [
    "ACTION_MASK_SHAPE",
    "ACTION_MASK_SPACE",
    "ACTION_SPACE",
    "ActionMask",
    "ActionValidationBoundary",
    "action_mask_for_state",
    "action_space_spec",
    "decode_action",
    "deserialize_action_mask",
    "encode_action",
    "legal_action_mask",
    "make_action_observation_space",
    "select_deterministic_action",
    "serialize_action_mask",
    "valid_action_indices",
]
