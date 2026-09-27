"""Replay-safe pending MainAgent transition ledger.

The ledger is deliberately independent from the PPO updater.  It bridges the
long-lived transition boundary (one MainAgent action to the next MainAgent
opportunity or episode end) and the shorter environment-step boundary.  A
validated environment result is recorded once, while invalid attempts,
retries, worker faults, timeouts, and cancellations never create training
transitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import json
from math import isfinite
from typing import Any, ClassVar

import numpy as np


PENDING_LEDGER_STATE_VERSION = "1.0"
_ACTION_MASK_SIZE = 6
_FORBIDDEN_PUBLIC_KEYS = {
    "roles", "true_roles", "actual_roles", "hidden_roles", "private_state",
    "privileged_state", "privileged_observation", "private_observation",
    "reward", "reward_breakdown", "terminal_reward", "emotion", "emotions",
    "emotion_state", "emotion_label", "emotion_labels", "true_emotion",
}


class PendingTransitionLedgerError(RuntimeError):
    """Raised when a transition boundary violates ledger invariants."""


LedgerError = PendingTransitionLedgerError


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "value"):
        return _json_safe(value.value)
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("ledger values must be finite")
        return value
    return str(value)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_safe(value)
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return _json_safe(result)
    if hasattr(value, "to_public_dict"):
        result = value.to_public_dict()
        if isinstance(result, Mapping):
            return _json_safe(result)
    raise TypeError(f"{name} must be a mapping or serializable value")


def _reject_private(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    lowered = {str(key).lower() for key in value}
    leaked = sorted(key for key in _FORBIDDEN_PUBLIC_KEYS if key in lowered)
    if leaked:
        raise PendingTransitionLedgerError(
            f"{name} contains fields that cannot enter the public action path: {', '.join(leaked)}"
        )
    return deepcopy(dict(value))


def _integer(value: Any, name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mask(value: Any) -> tuple[int, ...]:
    values = tuple(int(item) for item in np.asarray(value, dtype=np.uint8).reshape(-1).tolist())
    if len(values) != _ACTION_MASK_SIZE or any(item not in (0, 1) for item in values):
        raise ValueError("action_mask must contain six binary values")
    return values


def _action_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe(value)
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    return _json_safe(value)


def _result_values(result: Any) -> dict[str, Any]:
    values = _mapping(result, "action result")
    if "environment_step" not in values:
        raise TypeError("validated action result requires environment_step")
    if "action" not in values and "actor_idx" not in values:
        raise TypeError("validated action result requires action or actor_idx")
    info = values.get("info", {})
    if not isinstance(info, Mapping):
        info = {}
    values["info"] = dict(info)
    values.setdefault("events", ())
    values.setdefault("terminated", False)
    values.setdefault("truncated", False)
    return values


def _result_actor(values: Mapping[str, Any], fallback: int | None = None) -> int | None:
    action = values.get("action")
    if isinstance(action, Mapping):
        actor = action.get("actor_idx", action.get("actor"))
    else:
        actor = getattr(action, "actor_idx", None)
    if actor is None:
        actor = values.get("actor_idx", fallback)
    return None if actor is None else _integer(actor, "result actor_idx")


def _events(values: Any) -> tuple[dict[str, Any], ...]:
    if values is None:
        return ()
    if isinstance(values, Mapping) or hasattr(values, "event_id"):
        values = (values,)
    result: list[dict[str, Any]] = []
    for event in values:
        mapped = _mapping(event, "event")
        if mapped.get("public", True) is False:
            raise PendingTransitionLedgerError("private events cannot enter a transition")
        result.append(mapped)
    return tuple(result)


def _breakdown(value: Any, actor_idx: int | None = None) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping) and "rewards" in value:
        rewards = value.get("rewards", {})
        if actor_idx is not None:
            value = rewards.get(actor_idx, rewards.get(str(actor_idx), value))
    if isinstance(value, Mapping):
        return _json_safe(value)
    return {"value": _finite_float(value, "reward breakdown")}


def _reward_from_result(values: Mapping[str, Any], actor_idx: int | None) -> tuple[float, dict[str, Any] | None]:
    info = values.get("info", {})
    breakdown_value = values.get("reward_breakdown")
    if breakdown_value is None and isinstance(info, Mapping):
        breakdown_value = info.get("reward_breakdown")
    breakdown = _breakdown(breakdown_value, actor_idx)
    if "reward" in values and values["reward"] is not None:
        reward = _finite_float(values["reward"], "reward")
    elif breakdown is not None and "total" in breakdown:
        reward = _finite_float(breakdown["total"], "reward breakdown total")
    else:
        reward = 0.0
    return reward, breakdown


def _terminal_from_breakdown(value: Mapping[str, Any] | None) -> float:
    if value is None:
        return 0.0
    return _finite_float(value.get("terminal_reward", 0.0), "terminal reward")


@dataclass(frozen=True, order=True, slots=True)
class TransitionKey:
    worker_id: int
    episode_id: str
    environment_step: int
    actor_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "episode_id": self.episode_id,
            "environment_step": self.environment_step,
            "actor_idx": self.actor_idx,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentStepReward:
    environment_step: int
    actor_idx: int | None
    reward: float
    reward_breakdown: Mapping[str, Any] | None = None
    events: tuple[Mapping[str, Any], ...] = ()

    @property
    def terminal_reward(self) -> float:
        return _terminal_from_breakdown(self.reward_breakdown)

    @property
    def shaping_reward(self) -> float:
        return self.reward - self.terminal_reward

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "environment_step": self.environment_step,
            "actor_idx": self.actor_idx,
            "reward": self.reward,
            "reward_breakdown": self.reward_breakdown,
            "events": self.events,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentStepReward":
        return cls(
            environment_step=_integer(value["environment_step"], "environment_step"),
            actor_idx=None if value.get("actor_idx") is None else _integer(value["actor_idx"], "actor_idx"),
            reward=_finite_float(value.get("reward", 0.0), "reward"),
            reward_breakdown=None if value.get("reward_breakdown") is None else dict(value["reward_breakdown"]),
            events=tuple(dict(item) for item in value.get("events", ())),
        )


@dataclass(frozen=True, slots=True)
class MainAgentTransition:
    """A committed transition, immutable after its episode boundary."""

    worker_id: int
    episode_id: str
    transition_id: int
    environment_step: int
    actor_idx: int
    observation: Mapping[str, Any]
    action_mask: tuple[int, ...]
    action: Any
    log_prob: float | None
    value: float | None
    state: Mapping[str, Any] | None
    emotion_labels: Any
    environment_step_rewards: tuple[EnvironmentStepReward, ...]
    accumulated_reward: float
    terminal_reward: float
    terminal_reward_breakdown: Mapping[str, Any] | None
    events: tuple[Mapping[str, Any], ...]
    next_observation: Mapping[str, Any] | None
    terminated: bool
    truncated: bool
    main_agent_eliminated: bool
    trainable: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def start_environment_step(self) -> int:
        return self.environment_step

    @property
    def environment_steps_spanned(self) -> int:
        return len(self.environment_step_rewards)

    @property
    def reward(self) -> float:
        return self.accumulated_reward

    @property
    def action_metadata(self) -> dict[str, Any]:
        return {"log_prob": self.log_prob, "value": self.value}

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "worker_id": self.worker_id,
            "episode_id": self.episode_id,
            "transition_id": self.transition_id,
            "environment_step": self.environment_step,
            "actor_idx": self.actor_idx,
            "observation": self.observation,
            "action_mask": list(self.action_mask),
            "action": self.action,
            "log_prob": self.log_prob,
            "value": self.value,
            "state": self.state,
            "emotion_labels": self.emotion_labels,
            "environment_step_rewards": [item.to_dict() for item in self.environment_step_rewards],
            "accumulated_reward": self.accumulated_reward,
            "terminal_reward": self.terminal_reward,
            "terminal_reward_breakdown": self.terminal_reward_breakdown,
            "events": self.events,
            "next_observation": self.next_observation,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "main_agent_eliminated": self.main_agent_eliminated,
            "trainable": self.trainable,
            "metadata": self.metadata,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MainAgentTransition":
        return cls(
            worker_id=_integer(value["worker_id"], "worker_id"),
            episode_id=str(value["episode_id"]),
            transition_id=_integer(value["transition_id"], "transition_id"),
            environment_step=_integer(value["environment_step"], "environment_step"),
            actor_idx=_integer(value["actor_idx"], "actor_idx"),
            observation=dict(value.get("observation", {})),
            action_mask=_mask(value["action_mask"]),
            action=deepcopy(value.get("action")),
            log_prob=None if value.get("log_prob") is None else _finite_float(value["log_prob"], "log_prob"),
            value=None if value.get("value") is None else _finite_float(value["value"], "value"),
            state=None if value.get("state") is None else dict(value["state"]),
            emotion_labels=deepcopy(value.get("emotion_labels")),
            environment_step_rewards=tuple(EnvironmentStepReward.from_dict(item) for item in value.get("environment_step_rewards", ())),
            accumulated_reward=_finite_float(value.get("accumulated_reward", 0.0), "accumulated_reward"),
            terminal_reward=_finite_float(value.get("terminal_reward", 0.0), "terminal_reward"),
            terminal_reward_breakdown=None if value.get("terminal_reward_breakdown") is None else dict(value["terminal_reward_breakdown"]),
            events=tuple(dict(item) for item in value.get("events", ())),
            next_observation=None if value.get("next_observation") is None else dict(value["next_observation"]),
            terminated=bool(value.get("terminated", False)),
            truncated=bool(value.get("truncated", False)),
            main_agent_eliminated=bool(value.get("main_agent_eliminated", False)),
            trainable=bool(value.get("trainable", False)),
            metadata=dict(value.get("metadata", {})),
        )


# Design and downstream PPO code use both names.
CommittedTransition = MainAgentTransition


@dataclass
class _PendingBuffer:
    key: TransitionKey
    transition_id: int
    observation: Mapping[str, Any]
    action_mask: tuple[int, ...]
    action: Any
    log_prob: float | None
    value: float | None
    state: Mapping[str, Any] | None
    emotion_labels: Any
    metadata: Mapping[str, Any]
    rewards: list[EnvironmentStepReward] = field(default_factory=list)
    events: list[Mapping[str, Any]] = field(default_factory=list)
    result_keys: dict[tuple[int, int, int, int], str] = field(default_factory=dict)
    last_result_step: int | None = None
    terminated: bool = False
    truncated: bool = False
    main_agent_eliminated: bool = False
    terminal_reward_override: float = 0.0
    terminal_reward_breakdown: Mapping[str, Any] | None = None


class PendingTransitionLedger:
    """Per-worker, retry-safe ledger for MainAgent transition boundaries.

    ``begin_action`` captures a pre-action public boundary.  Each subsequent
    validated result is attached with ``commit_result``.  ``begin_action`` for
    the next action or ``finish_episode`` emits the completed immutable
    transition.  The ledger never creates a transition for an invalid action,
    missing post-action result, eliminated-agent PASS, or worker cleanup.
    """

    state_version: ClassVar[str] = PENDING_LEDGER_STATE_VERSION

    def __init__(self, *, logger: Any | None = None) -> None:
        self.logger = logger
        self._pending: dict[TransitionKey, _PendingBuffer] = {}
        self._committed: list[MainAgentTransition] = []
        self._diagnostics: list[dict[str, Any]] = []
        self._seen_results: dict[tuple[int, str, int, int], str] = {}
        self._next_ids: dict[tuple[int, str], int] = {}
        self._closed: dict[tuple[int, str], str] = {}

    @property
    def pending(self) -> tuple[MainAgentTransition, ...]:
        return tuple(self._as_transition(item, None, False, False, False, False) for item in self._ordered_pending())

    @property
    def pending_transitions(self) -> tuple[MainAgentTransition, ...]:
        return self.pending

    @property
    def committed(self) -> tuple[MainAgentTransition, ...]:
        return tuple(sorted(self._committed, key=lambda item: (item.worker_id, item.episode_id, item.environment_step, item.actor_idx, item.transition_id)))

    @property
    def transitions(self) -> tuple[MainAgentTransition, ...]:
        return self.committed

    @property
    def records(self) -> tuple[MainAgentTransition, ...]:
        return self.committed

    @property
    def trainable(self) -> tuple[MainAgentTransition, ...]:
        return tuple(item for item in self.committed if item.trainable and not item.truncated)

    @property
    def diagnostics(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(deepcopy(item) for item in self._diagnostics)

    @property
    def size(self) -> int:
        return len(self._pending)

    def _diag(self, kind: str, *, worker_id: int | None = None, episode_id: str | None = None, reason: str | None = None, **fields: Any) -> None:
        item = {"sequence": len(self._diagnostics), "kind": kind}
        if worker_id is not None:
            item["worker_id"] = int(worker_id)
        if episode_id is not None:
            item["episode_id"] = str(episode_id)
        if reason is not None:
            item["reason"] = str(reason)
        item.update(_json_safe(fields))
        self._diagnostics.append(item)
        if self.logger is not None and hasattr(self.logger, "event"):
            self.logger.event("pending_transition_ledger", "ledger diagnostic", **item)

    @staticmethod
    def _worker(value: Any) -> int:
        return _integer(value, "worker_id")

    @staticmethod
    def _episode(value: Any) -> str:
        if value is None or str(value) == "":
            raise ValueError("episode_id is required")
        return str(value)

    def _key(self, worker_id: Any, episode_id: Any, environment_step: Any, actor_idx: Any) -> TransitionKey:
        return TransitionKey(self._worker(worker_id), self._episode(episode_id), _integer(environment_step, "environment_step"), _integer(actor_idx, "actor_idx"))

    def start_episode(self, worker_id: Any, episode_id: Any) -> None:
        worker = self._worker(worker_id)
        episode = self._episode(episode_id)
        if any(key.worker_id == worker and key.episode_id == episode for key in self._pending):
            raise PendingTransitionLedgerError("episode already has pending transitions")
        self._closed.pop((worker, episode), None)
        self._next_ids.setdefault((worker, episode), 0)

    reset_episode = start_episode
    on_episode_start = start_episode

    def _ensure_open(self, worker_id: int, episode_id: str) -> None:
        status = self._closed.get((worker_id, episode_id))
        if status is not None:
            raise PendingTransitionLedgerError(f"episode {episode_id} is already closed ({status})")

    def _find_pending(self, worker: int, episode: str, actor: int, step: int) -> _PendingBuffer | None:
        key = TransitionKey(worker, episode, step, actor)
        exact = self._pending.get(key)
        if exact is not None:
            return exact
        # Prefer the pending boundary owned by this acting actor.  This keeps
        # simultaneous per-agent captures aligned while still allowing a
        # MainAgent transition to absorb intervening actions by other actors.
        owned = [
            item for item in self._pending.values()
            if item.key.worker_id == worker and item.key.episode_id == episode
            and item.key.actor_idx == actor
            and ((item.last_result_step is None and item.key.environment_step + 1 == step)
                 or (item.last_result_step is not None and item.last_result_step + 1 == step))
        ]
        if len(owned) == 1:
            return owned[0]
        if len(owned) > 1:
            raise PendingTransitionLedgerError("validated result matches multiple pending boundaries for its actor")
        candidates = [
            item for item in self._pending.values()
            if item.key.worker_id == worker and item.key.episode_id == episode and item.last_result_step is not None and item.last_result_step + 1 == step
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise PendingTransitionLedgerError("validated result matches multiple pending actor boundaries")
        # The first result for a pending action is always its next env step.
        candidates = [
            item for item in self._pending.values()
            if item.key.worker_id == worker and item.key.episode_id == episode and item.last_result_step is None and item.key.environment_step + 1 == step
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def begin_action(
        self,
        worker_id: Any,
        episode_id: Any,
        environment_step: Any,
        actor_idx: Any,
        observation: Any,
        action_mask: Any,
        action: Any,
        *,
        log_prob: Any | None = None,
        value: Any | None = None,
        state: Any | None = None,
        emotion_labels: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MainAgentTransition | None:
        worker = self._worker(worker_id)
        episode = self._episode(episode_id)
        step = _integer(environment_step, "environment_step")
        actor = _integer(actor_idx, "actor_idx")
        self._ensure_open(worker, episode)
        public_observation = _reject_private(_mapping(observation, "observation"), "observation")
        public_state = None if state is None else _reject_private(_mapping(state, "state"), "state")
        mask = _mask(action_mask)
        resolved_log_prob = None if log_prob is None else _finite_float(log_prob, "log_prob")
        resolved_value = None if value is None else _finite_float(value, "value")
        key = TransitionKey(worker, episode, step, actor)
        existing = self._pending.get(key)
        if existing is not None:
            # The key identifies the already-captured action boundary.  A
            # worker retry may carry a rebuilt/transient observation or state;
            # never replace the original boundary or create another one.
            self._diag("retry_ignored", worker_id=worker, episode_id=episode, environment_step=step, actor_idx=actor)
            return None
        prior_key = next((item.key for item in self._pending.values() if item.key.worker_id == worker and item.key.episode_id == episode and item.key.actor_idx == actor and item.key.environment_step < step), None)
        finalized: MainAgentTransition | None = None
        if prior_key is not None:
            prior = self._pending[prior_key]
            if prior.last_result_step is None:
                self._diag("boundary_without_validated_result", worker_id=worker, episode_id=episode, environment_step=step, actor_idx=actor)
                raise PendingTransitionLedgerError("cannot close a pending action without a validated post-action result")
            finalized = self._finalize(prior, next_observation=public_observation)
        transition_id = self._next_ids.get((worker, episode), 0)
        self._next_ids[(worker, episode)] = transition_id + 1
        self._pending[key] = _PendingBuffer(
            key=key, transition_id=transition_id, observation=public_observation,
            action_mask=mask, action=_action_dict(action), log_prob=resolved_log_prob,
            value=resolved_value, state=public_state, emotion_labels=deepcopy(emotion_labels),
            metadata=deepcopy(dict(metadata or {})),
        )
        return finalized

    capture = begin_action
    prepare_action = begin_action
    open_transition = begin_action
    on_action = begin_action

    def record_invalid_action(self, worker_id: Any, episode_id: Any, environment_step: Any, actor_idx: Any, action: Any, *, reason: str | None = None, action_mask: Any | None = None) -> None:
        self._diag("invalid_action", worker_id=self._worker(worker_id), episode_id=self._episode(episode_id), environment_step=_integer(environment_step, "environment_step"), actor_idx=_integer(actor_idx, "actor_idx"), action=_action_dict(action), action_mask=None if action_mask is None else list(_mask(action_mask)), reason=reason or "invalid_action")

    on_invalid_action = record_invalid_action
    handle_invalid_action = record_invalid_action

    def _result_fingerprint(self, values: Mapping[str, Any], actor_idx: int) -> str:
        """Return a canonical identity for one validated environment result.

        Transport-level omissions and equivalent empty containers must not turn
        a retry into a second reward/event append.  Validation flags are
        control metadata, not part of the validated result payload; all
        transition-affecting fields remain covered by the fingerprint.
        """
        reward, breakdown = _reward_from_result(values, actor_idx)
        canonical = dict(values)
        canonical.pop("valid", None)
        canonical.pop("invalid", None)
        canonical["actor_idx"] = actor_idx
        canonical["action"] = _action_dict(values.get("action"))
        canonical["events"] = _events(values.get("events"))
        canonical["reward"] = reward
        canonical["reward_breakdown"] = breakdown
        return json.dumps(_json_safe(canonical), sort_keys=True, separators=(",", ":"))

    def commit_result(
        self,
        worker_id: Any,
        episode_id: Any,
        result: Any,
        *,
        valid: bool = True,
        actor_idx: Any | None = None,
    ) -> MainAgentTransition | None:
        worker = self._worker(worker_id)
        episode = self._episode(episode_id)
        values = _result_values(result)
        step = _integer(values["environment_step"], "environment_step")
        resolved_actor = _result_actor(values, None if actor_idx is None else _integer(actor_idx, "actor_idx"))
        if not valid or bool(values.get("valid", True)) is False or bool(values.get("invalid", False)):
            self._diag("invalid_result_ignored", worker_id=worker, episode_id=episode, environment_step=step, actor_idx=resolved_actor)
            return None
        if bool(values.get("terminated", False)) and bool(values.get("truncated", False)):
            raise PendingTransitionLedgerError("terminated and truncated cannot both be true")
        if resolved_actor is None:
            raise TypeError("validated action result requires actor_idx")
        result_key = (worker, episode, step, resolved_actor)
        fingerprint = self._result_fingerprint(values, resolved_actor)
        if result_key in self._seen_results:
            if self._seen_results[result_key] != fingerprint:
                raise PendingTransitionLedgerError("validated result key was reused with different data")
            self._diag("duplicate_result_ignored", worker_id=worker, episode_id=episode, environment_step=step, actor_idx=resolved_actor)
            return None
        pending = self._find_pending(worker, episode, resolved_actor, step)
        if pending is None:
            self._diag("orphan_result", worker_id=worker, episode_id=episode, environment_step=step, actor_idx=resolved_actor)
            raise PendingTransitionLedgerError("validated result has no matching pending action")
        expected = pending.key.environment_step + 1 if pending.last_result_step is None else pending.last_result_step + 1
        if step != expected:
            raise PendingTransitionLedgerError(f"environment steps must be contiguous: expected {expected}, got {step}")
        reward, breakdown = _reward_from_result(values, resolved_actor)
        event_values = _events(values.get("events"))
        event_ids = {int(item["event_id"]) for item in pending.events if "event_id" in item}
        for event in event_values:
            if "event_id" in event and int(event["event_id"]) in event_ids:
                continue
            pending.events.append(deepcopy(event))
            if "event_id" in event:
                event_ids.add(int(event["event_id"]))
        pending.rewards.append(EnvironmentStepReward(step, resolved_actor, reward, breakdown, event_values))
        pending.last_result_step = step
        pending.result_keys[result_key] = fingerprint
        pending.terminated = bool(values.get("terminated", False))
        pending.truncated = bool(values.get("truncated", False))
        pending.main_agent_eliminated = bool(values.get("main_agent_eliminated", False)) or values.get("eliminated_idx") == pending.key.actor_idx
        self._seen_results[result_key] = fingerprint
        return None

    commit_post_action = commit_result
    record_post_action = commit_result
    on_action_result = commit_result
    after_action = commit_result

    def _as_transition(self, pending: _PendingBuffer, next_observation: Mapping[str, Any] | None, terminated: bool, truncated: bool, eliminated: bool, trainable: bool) -> MainAgentTransition:
        reward_total = _finite_float(sum(item.reward for item in pending.rewards) + pending.terminal_reward_override, "accumulated reward")
        terminal_reward = _finite_float(sum(item.terminal_reward for item in pending.rewards) + pending.terminal_reward_override, "terminal reward")
        return MainAgentTransition(
            worker_id=pending.key.worker_id, episode_id=pending.key.episode_id,
            transition_id=pending.transition_id, environment_step=pending.key.environment_step,
            actor_idx=pending.key.actor_idx, observation=deepcopy(pending.observation),
            action_mask=pending.action_mask, action=deepcopy(pending.action), log_prob=pending.log_prob,
            value=pending.value, state=deepcopy(pending.state), emotion_labels=deepcopy(pending.emotion_labels),
            environment_step_rewards=tuple(deepcopy(pending.rewards)), accumulated_reward=reward_total,
            terminal_reward=terminal_reward, terminal_reward_breakdown=deepcopy(pending.terminal_reward_breakdown), events=tuple(deepcopy(pending.events)),
            next_observation=None if next_observation is None else deepcopy(next_observation),
            terminated=bool(terminated), truncated=bool(truncated),
            main_agent_eliminated=bool(eliminated or pending.main_agent_eliminated),
            trainable=bool(trainable and not truncated), metadata=deepcopy(pending.metadata),
        )

    def _finalize(self, pending: _PendingBuffer, *, next_observation: Mapping[str, Any] | None, terminated: bool | None = None, truncated: bool | None = None, eliminated: bool = False, trainable: bool | None = None) -> MainAgentTransition:
        resolved_terminated = pending.terminated if terminated is None else bool(terminated)
        resolved_truncated = pending.truncated if truncated is None else bool(truncated)
        if resolved_terminated and resolved_truncated:
            raise PendingTransitionLedgerError("terminated and truncated cannot both be true")
        resolved_trainable = (not resolved_truncated) if trainable is None else bool(trainable)
        transition = self._as_transition(pending, next_observation, resolved_terminated, resolved_truncated, eliminated, resolved_trainable)
        self._pending.pop(pending.key, None)
        self._committed.append(transition)
        return transition

    def finish_episode(
        self,
        worker_id: Any,
        episode_id: Any,
        *,
        terminated: bool | None = None,
        truncated: bool | None = None,
        final_observation: Any | None = None,
        main_agent_eliminated: bool = False,
        terminal_reward: Any | None = None,
        terminal_reward_breakdown: Any | None = None,
        reason: str | None = None,
    ) -> tuple[MainAgentTransition, ...]:
        worker = self._worker(worker_id)
        episode = self._episode(episode_id)
        self._ensure_open(worker, episode)
        if terminated is not None and truncated is not None and terminated and truncated:
            raise PendingTransitionLedgerError("terminated and truncated cannot both be true")
        matching = self._ordered_pending(worker, episode)
        resolved_terminated = bool(terminated) if terminated is not None else any(item.terminated for item in matching)
        resolved_truncated = bool(truncated) if truncated is not None else any(item.truncated for item in matching)
        if resolved_terminated and resolved_truncated:
            raise PendingTransitionLedgerError("terminated and truncated cannot both be true")
        final = None if final_observation is None else _reject_private(_mapping(final_observation, "final_observation"), "final_observation")
        finalized: list[MainAgentTransition] = []
        for pending in matching:
            # No terminal signal can turn an action with no validated result
            # into a transition.  Cleanup is explicit and diagnostic.
            if pending.last_result_step is None:
                self._diag("pending_discarded_without_result", worker_id=worker, episode_id=episode, environment_step=pending.key.environment_step, actor_idx=pending.key.actor_idx, reason=reason or "episode_end")
                self._pending.pop(pending.key, None)
                continue
            eliminated = bool(main_agent_eliminated or pending.main_agent_eliminated)
            if terminal_reward is not None and not resolved_truncated:
                if not pending.rewards or all(item.terminal_reward == 0.0 for item in pending.rewards):
                    pending.terminal_reward_override = _finite_float(terminal_reward, "terminal_reward")
                    pending.terminal_reward_breakdown = _breakdown(terminal_reward_breakdown, pending.key.actor_idx)
            finalized.append(self._finalize(pending, next_observation=None if eliminated else final, terminated=resolved_terminated, truncated=resolved_truncated, eliminated=eliminated, trainable=resolved_terminated and not resolved_truncated))
        self._closed[(worker, episode)] = "terminated" if resolved_terminated else "truncated" if resolved_truncated else reason or "closed"
        return tuple(finalized)

    finalize_episode = finish_episode
    end_episode = finish_episode
    on_episode_end = finish_episode

    def eliminate_agent(self, worker_id: Any, episode_id: Any, *, terminated: bool = False, truncated: bool = False, terminal_reward: Any | None = None, terminal_reward_breakdown: Any | None = None, reason: str = "main_agent_eliminated") -> tuple[MainAgentTransition, ...]:
        """Mark the MainAgent boundary without closing a still-running episode.

        Elimination removes the MainAgent's future action opportunity, not the
        environment episode.  The actual pending transition stays parked until
        the environment reports normal/truncated completion, so no fabricated
        PASS, next state, or terminal reward is created here.
        """

        worker = self._worker(worker_id)
        episode = self._episode(episode_id)
        self._ensure_open(worker, episode)
        if terminated and truncated:
            raise PendingTransitionLedgerError("terminated and truncated cannot both be true")
        if terminal_reward is not None and not (terminated or truncated):
            raise PendingTransitionLedgerError("terminal reward requires an environment termination")
        if terminated or truncated:
            return self.finish_episode(
                worker,
                episode,
                terminated=terminated,
                truncated=truncated,
                main_agent_eliminated=True,
                terminal_reward=terminal_reward,
                terminal_reward_breakdown=terminal_reward_breakdown,
                reason=reason,
            )
        for pending in self._ordered_pending(worker, episode):
            pending.main_agent_eliminated = True
            self._diag(
                "main_agent_eliminated_parked",
                worker_id=worker,
                episode_id=episode,
                environment_step=pending.key.environment_step,
                actor_idx=pending.key.actor_idx,
                reason=reason,
            )
        return ()

    on_agent_eliminated = eliminate_agent

    def cleanup(self, worker_id: Any, episode_id: Any, *, reason: str = "cleanup", status: str = "cancelled") -> None:
        worker = self._worker(worker_id)
        episode = self._episode(episode_id)
        for pending in self._ordered_pending(worker, episode):
            self._diag("pending_discarded", worker_id=worker, episode_id=episode, environment_step=pending.key.environment_step, actor_idx=pending.key.actor_idx, reason=reason, status=status)
            self._pending.pop(pending.key, None)
        self._closed[(worker, episode)] = str(status)

    cancel = cleanup
    timeout = cleanup
    fault = cleanup
    on_worker_fault = cleanup
    on_timeout = cleanup
    on_cancel = cleanup

    def _ordered_pending(self, worker: int | None = None, episode: str | None = None) -> list[_PendingBuffer]:
        values = [item for item in self._pending.values() if (worker is None or item.key.worker_id == worker) and (episode is None or item.key.episode_id == episode)]
        return sorted(values, key=lambda item: (item.key.worker_id, item.key.episode_id, item.key.environment_step, item.key.actor_idx, item.transition_id))

    def state_dict(self) -> dict[str, Any]:
        pending_values = []
        for item in self._ordered_pending():
            pending_values.append({
                "key": item.key.to_dict(), "transition_id": item.transition_id,
                "observation": item.observation, "action_mask": list(item.action_mask),
                "action": item.action, "log_prob": item.log_prob, "value": item.value,
                "state": item.state, "emotion_labels": item.emotion_labels,
                "metadata": item.metadata, "rewards": [reward.to_dict() for reward in item.rewards],
                "events": item.events, "result_keys": [{"key": list(key), "fingerprint": fingerprint} for key, fingerprint in sorted(item.result_keys.items())],
                "last_result_step": item.last_result_step, "terminated": item.terminated,
                "truncated": item.truncated, "main_agent_eliminated": item.main_agent_eliminated,
                "terminal_reward_override": item.terminal_reward_override,
                "terminal_reward_breakdown": item.terminal_reward_breakdown,
            })

        seen = [{"key": list(key), "fingerprint": value} for key, value in sorted(self._seen_results.items())]
        return _json_safe({
            "state_version": self.state_version, "pending": pending_values,
            "committed": [item.to_dict() for item in self.committed],
            "diagnostics": self._diagnostics, "seen_results": seen,
            "next_ids": [{"worker_id": key[0], "episode_id": key[1], "next_id": value} for key, value in sorted(self._next_ids.items())],
            "closed": [{"worker_id": key[0], "episode_id": key[1], "status": value} for key, value in sorted(self._closed.items())],
        })

    to_dict = state_dict
    serialize = state_dict

    def to_json(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any], *, logger: Any | None = None) -> "PendingTransitionLedger":
        if state.get("state_version", cls.state_version) != cls.state_version:
            raise ValueError("unsupported pending transition ledger state schema")
        ledger = cls(logger=logger)
        ledger._committed = [MainAgentTransition.from_dict(item) for item in state.get("committed", ())]
        ledger._diagnostics = [dict(item) for item in state.get("diagnostics", ())]
        ledger._next_ids = {(int(item["worker_id"]), str(item["episode_id"])): int(item["next_id"]) for item in state.get("next_ids", ())}
        ledger._closed = {(int(item["worker_id"]), str(item["episode_id"])): str(item["status"]) for item in state.get("closed", ())}
        for raw in state.get("seen_results", ()):
            key = tuple(raw["key"])
            ledger._seen_results[(int(key[0]), str(key[1]), int(key[2]), int(key[3]))] = str(raw["fingerprint"])
        for raw in state.get("pending", ()):
            key_value = raw["key"]
            key = TransitionKey(int(key_value["worker_id"]), str(key_value["episode_id"]), int(key_value["environment_step"]), int(key_value["actor_idx"]))
            pending = _PendingBuffer(
                key=key, transition_id=int(raw["transition_id"]), observation=dict(raw.get("observation", {})),
                action_mask=_mask(raw["action_mask"]), action=deepcopy(raw.get("action")),
                log_prob=None if raw.get("log_prob") is None else _finite_float(raw["log_prob"], "log_prob"),
                value=None if raw.get("value") is None else _finite_float(raw["value"], "value"),
                state=None if raw.get("state") is None else dict(raw["state"]),
                emotion_labels=deepcopy(raw.get("emotion_labels")), metadata=dict(raw.get("metadata", {})),
                rewards=[EnvironmentStepReward.from_dict(item) for item in raw.get("rewards", ())],
                events=[dict(item) for item in raw.get("events", ())],
                result_keys={(int(item["key"][0]), str(item["key"][1]), int(item["key"][2]), int(item["key"][3])): str(item["fingerprint"]) for item in raw.get("result_keys", ())},
                last_result_step=None if raw.get("last_result_step") is None else int(raw["last_result_step"]),
                terminated=bool(raw.get("terminated", False)), truncated=bool(raw.get("truncated", False)),
                main_agent_eliminated=bool(raw.get("main_agent_eliminated", False)),
                terminal_reward_override=_finite_float(raw.get("terminal_reward_override", 0.0), "terminal_reward_override"),
                terminal_reward_breakdown=None if raw.get("terminal_reward_breakdown") is None else dict(raw["terminal_reward_breakdown"]),
            )
            if key in ledger._pending:
                raise ValueError("duplicate pending transition key in serialized state")
            ledger._pending[key] = pending
        return ledger

    from_dict = from_state_dict
    deserialize = from_state_dict

    @classmethod
    def from_json(cls, value: str, *, logger: Any | None = None) -> "PendingTransitionLedger":
        if not isinstance(value, str):
            raise TypeError("serialized state must be a JSON string")
        return cls.from_state_dict(json.loads(value), logger=logger)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = self.from_state_dict(state, logger=self.logger)
        self.__dict__.update(restored.__dict__)


PendingTransition = MainAgentTransition

__all__ = [
    "PENDING_LEDGER_STATE_VERSION", "PendingTransitionLedger", "PendingTransitionLedgerError", "LedgerError",
    "TransitionKey", "EnvironmentStepReward", "PendingTransition", "MainAgentTransition", "CommittedTransition",
]
