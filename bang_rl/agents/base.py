"""Common agent contract and boundary validation.

Agents communicate with the game through the canonical observation mapping and
an explicit action mask.  This module intentionally does not import the
environment implementation or privileged environment state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from bang_rl.runtime.seed import derive_seed

ACTION_SPACE_SIZE = 6


class AgentInterruptedError(RuntimeError):
    """Raised when a waiting human action is interrupted."""


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    """Public per-episode context supplied to an agent lifecycle callback."""

    episode_id: str
    seed: int | None = None
    worker_id: int = 0
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Public episode result supplied to an agent lifecycle callback."""

    episode_id: str | None = None
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] | None = None


def _context_value(context: Any, name: str, default: Any = None) -> Any:
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def _validate_seed(seed: Any) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None")
    return int(seed)


def validate_action_mask(action_mask: Any) -> np.ndarray:
    """Validate and detach the canonical six-entry binary action mask."""

    try:
        values = np.asarray(action_mask)
    except (TypeError, ValueError) as exc:
        raise TypeError("action_mask must be array-like") from exc
    if values.shape != (ACTION_SPACE_SIZE,):
        raise ValueError(f"action_mask must have shape ({ACTION_SPACE_SIZE},)")
    if values.dtype.kind not in "biu":
        raise TypeError("action_mask values must be integer or boolean")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("action_mask values must be only 0 or 1")
    return values.astype(np.uint8, copy=True)


def validate_action(action: Any, action_mask: Any) -> int:
    """Validate an action index and ensure it is enabled by ``action_mask``."""

    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
        raise TypeError("action must be an integer in [0, 5]")
    normalized = int(action)
    if not 0 <= normalized < ACTION_SPACE_SIZE:
        raise ValueError(f"action must be in [0, {ACTION_SPACE_SIZE - 1}]")
    mask = validate_action_mask(action_mask)
    if not bool(mask[normalized]):
        raise ValueError(f"action {normalized} is disallowed by action_mask")
    return normalized


def _validate_public_observation(observation: Any) -> None:
    """Reject non-observation and obviously privileged state inputs.

    ``PublicEnvironmentState`` is intentionally a lightweight mapping-like
    object rather than a dependency imported from the environment package, so
    the check uses only the canonical mapping protocol.
    """

    if observation is None:
        raise TypeError("observation must be a canonical observation mapping")
    if isinstance(observation, Mapping):
        keys = set(observation.keys())
    elif hasattr(observation, "keys") and hasattr(observation, "__getitem__"):
        keys = set(observation.keys())
    else:
        raise TypeError("observation must be a canonical observation mapping")
    forbidden = {
        "roles",
        "true_roles",
        "actual_roles",
        "hidden_roles",
        "privileged_state",
        "privileged_observation",
        "private_observation",
        "emotion_label",
        "reward",
        "reward_breakdown",
    }
    leaked = sorted(forbidden.intersection(keys))
    if leaked:
        raise ValueError(f"privileged observation fields are not accepted: {', '.join(leaked)}")


class BaseAgent(ABC):
    """Abstract interface shared by HumanAgent, RandomAgent, and future agents."""

    def __init__(self, agent_id: str | int = "agent", seed: int | None = None) -> None:
        if isinstance(agent_id, bool) or not isinstance(agent_id, (str, int, np.integer)):
            raise TypeError("agent_id must be a non-empty string or integer")
        if isinstance(agent_id, str) and not agent_id.strip():
            raise ValueError("agent_id must not be empty")
        if isinstance(agent_id, (int, np.integer)) and int(agent_id) < 0:
            raise ValueError("agent_id must be non-negative")
        self.agent_id = agent_id
        self.seed = _validate_seed(seed)
        self.episode_id: str | None = None
        self.episode_seed: int | None = None
        self._last_episode_context: Any | None = None
        self._last_episode_result: Any | None = None

    @abstractmethod
    def act(self, observation: Any, action_mask: Any) -> int:
        """Choose one enabled action from a canonical observation boundary."""

    def _validate_act_inputs(self, observation: Any, action_mask: Any) -> np.ndarray:
        _validate_public_observation(observation)
        mask = validate_action_mask(action_mask)
        if "action_mask" in observation:
            observed_mask = validate_action_mask(observation["action_mask"])
            if not np.array_equal(mask, observed_mask):
                raise ValueError("action_mask argument does not match observation action_mask")
        if not np.any(mask):
            raise ValueError("cannot select an action from an empty action mask")
        return mask

    def _validate_selected_action(self, action: Any, action_mask: Any) -> int:
        return validate_action(action, action_mask)

    def on_episode_start(self, context: Any) -> None:
        """Record public episode context; subclasses may reset their streams."""

        episode_id = _context_value(context, "episode_id")
        if episode_id is not None and not isinstance(episode_id, str):
            episode_id = str(episode_id)
        self.episode_id = episode_id
        episode_seed = _context_value(context, "seed", _context_value(context, "episode_seed"))
        self.episode_seed = _validate_seed(episode_seed)
        self._last_episode_context = context

    def on_episode_end(self, result: Any) -> None:
        """Record the public episode result without inspecting environment state."""

        self._last_episode_result = result

    def action_seed(self, seed: int | None = None) -> int:
        """Return a stable action-stream seed scoped by agent identity."""

        root = self.seed if seed is None else _validate_seed(seed)
        if root is None:
            # ``default_rng(None)`` remains intentionally nondeterministic, but
            # an explicit integer seed always follows the stable path above.
            root = int(np.random.SeedSequence().entropy)
        return derive_seed(root, "agent", self.agent_id, "action")


__all__ = [
    "ACTION_SPACE_SIZE",
    "AgentInterruptedError",
    "BaseAgent",
    "EpisodeContext",
    "EpisodeResult",
    "validate_action",
    "validate_action_mask",
]
