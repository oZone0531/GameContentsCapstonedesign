"""Canonical episode collection primitives for Phase 1 dataset capture.

This module deliberately stops at an in-memory canonical record.  It does not
implement a Parquet shard writer (that is TASK-5-2); instead it exposes a small
writer lifecycle so a later writer can consume completed episodes without
changing collection semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
from math import isfinite
from typing import Any, Protocol

import numpy as np

from bang_rl.env.types import CANONICAL_PLAYER_ORDER, PASS_ACTION


CANONICAL_DATASET_SCHEMA_VERSION = "1.0"
ACTION_SPACE_VERSION = "1.0"


class DatasetWriter(Protocol):
    """Minimal sink consumed by :class:`EpisodeCollector`.

    A writer is intentionally episode-oriented.  It may persist an episode in
    ``write_episode`` and must not rely on a per-step cumulative event history.
    ``flush`` and ``finalize`` are optional in practice and are called when
    present.
    """

    def write_episode(self, episode: "CanonicalEpisode") -> None: ...

    def flush(self) -> None: ...

    def finalize(self) -> None: ...


class CollectorStateError(RuntimeError):
    """Raised when collection lifecycle or step alignment is invalid."""


def _json_safe(value: Any) -> Any:
    """Convert runtime/domain values into deterministic JSON-compatible data."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if hasattr(value, "value"):
        return _json_safe(value.value)
    if hasattr(value, "to_public_dict"):
        return _json_safe(value.to_public_dict())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("canonical records require finite floating-point values")
        return value
    return str(value)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_safe(value)
    if hasattr(value, "to_public_dict"):
        return _json_safe(value.to_public_dict())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    raise TypeError(f"{name} must be a mapping or serializable domain value")


def _event_mapping(event: Any) -> dict[str, Any]:
    if hasattr(event, "to_public_dict"):
        value = event.to_public_dict()
    elif isinstance(event, Mapping):
        value = dict(event)
    elif hasattr(event, "to_dict"):
        value = event.to_dict()
    else:
        raise TypeError("events must be serializable GameEvent values")
    value = _json_safe(value)
    if value.get("public", True) is False:
        raise ValueError("private events cannot enter the canonical public event stream")
    if "event_id" not in value or "environment_step" not in value:
        raise ValueError("canonical events require event_id and environment_step")
    return value


def _observation(value: Any) -> dict[str, Any]:
    """Store a single-time observation projection without cumulative history."""

    result = _as_mapping(value, "observation")
    # Event objects are stored once in CanonicalEpisode.events.  Keeping them
    # in each step would make dataset size grow quadratically with episode
    # length and would make replay depend on duplicated representations.
    for key in ("event_sequence", "event_history", "event_log", "events"):
        result.pop(key, None)
    forbidden = {"roles", "true_roles", "actual_roles", "hidden_roles", "private_state", "privileged_state"}
    for key in forbidden:
        result.pop(key, None)
    return result


def _event_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        values = value.get("event_ids", value.get("input_event_ids", ()))
    else:
        values = getattr(value, "input_event_ids", ())
    return tuple(int(item) for item in values)


def _scalar_int(value: Any, name: str) -> int:
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise CollectorStateError(f"{name} must be a scalar")
    return int(values[0])


def _mask(value: Any, *, observation: Mapping[str, Any] | None = None) -> tuple[int, ...]:
    raw = value
    if raw is None and observation is not None:
        raw = observation.get("action_mask")
    if raw is None:
        return (1, 1, 1, 1, 1, 1)
    values = tuple(int(item) for item in np.asarray(raw).reshape(-1).tolist())
    if len(values) != 6 or any(item not in (0, 1) for item in values):
        raise ValueError("action_mask must contain six binary values")
    return values


def _result_values(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        values = dict(result)
    elif hasattr(result, "to_dict"):
        values = dict(result.to_dict())
    else:
        values = {
            name: getattr(result, name)
            for name in ("action", "environment_step", "events", "hp_before", "hp_after",
                         "eliminated_idx", "terminated", "truncated", "info", "observation", "reward")
            if hasattr(result, name)
        }
    if "action" not in values or "environment_step" not in values:
        raise TypeError("a valid action result requires action and environment_step")
    values.setdefault("events", ())
    values.setdefault("info", {})
    values.setdefault("terminated", False)
    values.setdefault("truncated", False)
    values.setdefault("reward", 0.0)
    if bool(values["terminated"]) and bool(values["truncated"]):
        raise ValueError("terminated and truncated cannot both be true")
    return values


def _action_values(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        return dict(action)
    if hasattr(action, "to_dict"):
        return dict(action.to_dict())
    return {
        "action_type": getattr(action, "action_type", None),
        "actor_idx": getattr(action, "actor_idx", None),
        "target_idx": getattr(action, "target_idx", None),
    }


def _action_fields(action: Any) -> tuple[int, int, int | None, str]:
    values = _action_values(action)
    actor = int(values["actor_idx"])
    target = values.get("target_idx")
    target = None if target is None else int(target)
    kind = values.get("action_type", values.get("type"))
    token = str(getattr(kind, "value", kind)).rsplit(".", 1)[-1].lower()
    is_pass = token in {"0", "pass"}
    action_index = PASS_ACTION if is_pass else target + 1 if target is not None else -1
    if not 0 <= action_index < 6:
        raise ValueError("action does not map to Discrete(6)")
    return action_index, actor, target, "pass" if is_pass else "bang"


def _reduced_emotion_label(label: Any) -> dict[str, Any]:
    """Keep target values and time boundary, not a duplicated event sequence."""

    values = _as_mapping(label, "emotion label")
    public = values.get("public_observation", {})
    reduced = {
        "episode_id": values.get("episode_id"),
        "environment_step": int(values["environment_step"]),
        "player_idx": int(values["player_idx"]),
        "values": list(values.get("values", values.get("emotion_values", ()))),
        "emotion_names": list(values.get("emotion_names", ())),
        "emotion_schema_version": str(values.get("emotion_schema_version", "")),
        "input_event_ids": list(values.get("input_event_ids", _event_ids(label))),
        "public_observation": _observation(public) if public else {},
    }
    return _json_safe(reduced)


def _emotion_record(value: Any) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    if value is None:
        return {}, None, None
    labels = getattr(value, "labels", None)
    if labels is None and isinstance(value, Mapping):
        labels = value.get("labels", value)
    labels = labels or {}
    reduced: dict[str, Any] = {}
    for index, label in sorted(labels.items(), key=lambda item: int(item[0])):
        reduced[str(int(index))] = _reduced_emotion_label(label)
    schema = None
    if reduced:
        schema = next(iter(reduced.values())).get("emotion_schema_version") or None
    public = getattr(value, "public_observation", None)
    if public is None and isinstance(value, Mapping):
        public = value.get("public_observation")
    # ``EmotionRuntime`` exposes ``public_observation`` as a *method* (its
    # per-boundary projection is produced on demand), so a bare attribute probe
    # yields a bound method rather than a stored mapping.  Treat such callables
    # (and any value that is not a mapping/serializable projection) as "no
    # stored observation" so a runtime handed straight to the collector never
    # feeds a non-serializable object into ``_observation`` -> ``_as_mapping``.
    if callable(public):
        public = None
    elif public is not None and not (
        isinstance(public, Mapping)
        or hasattr(public, "to_public_dict")
        or hasattr(public, "to_dict")
    ):
        public = None
    return reduced, schema, None if public is None else _observation(public)


def _reward_record(value: Any, result_values: Mapping[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    if value is None:
        info = result_values.get("info", {})
        value = info.get("reward_breakdown") if isinstance(info, Mapping) else None
    if value is None:
        return {}, None, None
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("reward breakdown must be a RewardBatch or mapping")
    if "rewards" in value:
        rewards = value.get("rewards", {})
        version = value.get("reward_config_version")
        identifier = value.get("reward_config_id")
    else:
        rewards = value
        version = None
        identifier = None
    normalized = {str(int(index)): _json_safe(item) for index, item in sorted(rewards.items(), key=lambda item: int(item[0]))}
    if normalized and version is None:
        version = next(iter(normalized.values())).get("reward_config_version")
        identifier = next(iter(normalized.values())).get("reward_config_id")
    return normalized, None if version is None else str(version), None if identifier is None else str(identifier)


def _snapshot_metadata(snapshot: Any) -> dict[str, Any]:
    if snapshot is None:
        return {}
    values = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
    versions = values.get("versions", {})
    selected = {
        "job_id": values.get("job_id"),
        "config_version": values.get("config_version"),
        "environment_version": versions.get("environment_version"),
        "observation_schema_version": versions.get("observation_schema_version"),
        "action_space_version": versions.get("action_space_version"),
        "emotion_schema_version": versions.get("emotion_schema_version"),
        "reward_config_version": versions.get("reward_config_version"),
        "preprocessing_version": versions.get("preprocessing_version"),
        "preset_version": versions.get("dummy_preset_version"),
        "seed_snapshot": values.get("seed"),
    }
    return {key: value for key, value in selected.items() if value is not None}


@dataclass(frozen=True)
class CanonicalStep:
    """One validated environment action, aligned to its pre-action inputs."""

    environment_step: int
    actor_idx: int
    action: int
    action_type: str
    target_idx: int | None
    action_mask: tuple[int, ...]
    pre_observation: Mapping[str, Any]
    post_observation: Mapping[str, Any] | None
    pre_event_ids: tuple[int, ...]
    event_ids: tuple[int, ...]
    hp_before: int | None
    hp_after: int | None
    eliminated_idx: int | None
    reward: float
    reward_breakdowns: Mapping[str, Any] = field(default_factory=dict)
    emotion_labels: Mapping[str, Any] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "environment_step": self.environment_step,
            "actor_idx": self.actor_idx,
            "action": self.action,
            "action_type": self.action_type,
            "target_idx": self.target_idx,
            "action_mask": list(self.action_mask),
            "pre_observation": dict(self.pre_observation),
            "post_observation": None if self.post_observation is None else dict(self.post_observation),
            "pre_event_ids": list(self.pre_event_ids),
            "event_ids": list(self.event_ids),
            "hp_before": self.hp_before,
            "hp_after": self.hp_after,
            "eliminated_idx": self.eliminated_idx,
            "reward": self.reward,
            "reward_breakdowns": dict(self.reward_breakdowns),
            "emotion_labels": dict(self.emotion_labels),
            "terminated": self.terminated,
            "truncated": self.truncated,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalStep":
        return cls(
            environment_step=int(value["environment_step"]), actor_idx=int(value["actor_idx"]),
            action=int(value["action"]), action_type=str(value["action_type"]),
            target_idx=None if value.get("target_idx") is None else int(value["target_idx"]),
            action_mask=tuple(int(item) for item in value["action_mask"]),
            pre_observation=dict(value.get("pre_observation", {})),
            post_observation=None if value.get("post_observation") is None else dict(value["post_observation"]),
            pre_event_ids=tuple(int(item) for item in value.get("pre_event_ids", [])),
            event_ids=tuple(int(item) for item in value.get("event_ids", [])),
            hp_before=value.get("hp_before"), hp_after=value.get("hp_after"),
            eliminated_idx=value.get("eliminated_idx"), reward=float(value.get("reward", 0.0)),
            reward_breakdowns=dict(value.get("reward_breakdowns", {})),
            emotion_labels=dict(value.get("emotion_labels", {})),
            terminated=bool(value.get("terminated", False)),
            truncated=bool(value.get("truncated", False)),
        )


@dataclass(frozen=True)
class CanonicalEpisode:
    """Complete canonical source record; events are stored exactly once."""

    episode_id: str
    seed: int
    agent_order: tuple[int, ...]
    metadata: Mapping[str, Any]
    initial_state: Mapping[str, Any]
    steps: tuple[CanonicalStep, ...]
    events: tuple[Mapping[str, Any], ...]
    termination_status: str
    termination_reason: str | None
    terminal_result: Mapping[str, Any] | None
    final_state: Mapping[str, Any] | None
    role_labels: tuple[int, ...] | None = None
    invalid_actions: tuple[Mapping[str, Any], ...] = ()

    @property
    def environment_steps(self) -> int:
        return len(self.steps)

    @property
    def roles(self) -> tuple[int, ...] | None:
        """Private role labels, when the collector was allowed to retain them."""

        return self.role_labels

    @property
    def environment_step_count(self) -> int:
        return len(self.steps)

    @property
    def step_rows(self) -> tuple[CanonicalStep, ...]:
        return self.steps

    @property
    def episode_row(self) -> "CanonicalEpisode":
        return self

    def to_dict(self, *, include_private: bool = True) -> dict[str, Any]:
        value = {
            "dataset_schema_version": CANONICAL_DATASET_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "agent_order": list(self.agent_order),
            "metadata": dict(self.metadata),
            "initial_state": dict(self.initial_state),
            "steps": [step.to_dict() for step in self.steps],
            "events": [dict(event) for event in self.events],
            "termination_status": self.termination_status,
            "termination_reason": self.termination_reason,
            "terminal_result": None if self.terminal_result is None else dict(self.terminal_result),
            "final_state": None if self.final_state is None else dict(self.final_state),
            "invalid_actions": [dict(item) for item in self.invalid_actions],
        }
        if include_private:
            value["role_labels"] = None if self.role_labels is None else list(self.role_labels)
        return _json_safe(value)

    def to_public_dict(self) -> dict[str, Any]:
        return self.to_dict(include_private=False)

    def to_json(self, *, include_private: bool = True) -> str:
        return json.dumps(self.to_dict(include_private=include_private), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalEpisode":
        return cls(
            episode_id=str(value["episode_id"]), seed=int(value["seed"]),
            agent_order=tuple(int(item) for item in value.get("agent_order", CANONICAL_PLAYER_ORDER)),
            metadata=dict(value.get("metadata", {})), initial_state=dict(value.get("initial_state", {})),
            steps=tuple(CanonicalStep.from_dict(item) for item in value.get("steps", [])),
            events=tuple(dict(item) for item in value.get("events", [])),
            termination_status=str(value["termination_status"]),
            termination_reason=value.get("termination_reason"),
            terminal_result=None if value.get("terminal_result") is None else dict(value["terminal_result"]),
            final_state=None if value.get("final_state") is None else dict(value["final_state"]),
            role_labels=None if value.get("role_labels") is None else tuple(int(item) for item in value["role_labels"]),
            invalid_actions=tuple(dict(item) for item in value.get("invalid_actions", [])),
        )


class EpisodeCollector:
    """Collect validated transitions into deterministic canonical episodes.

    ``before_action`` is optional but recommended: it captures the observation
    and emotion label at the exact pre-action boundary.  Invalid actions must
    use ``record_invalid_action`` and never call ``record_step``.  A valid step
    is accepted exactly once, in increasing environment-step order.
    """

    def __init__(
        self,
        *,
        writer: DatasetWriter | None = None,
        job_snapshot: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_private_roles: bool = True,
        reward_calculator: Any | None = None,
        reward_runtime: Any | None = None,
        emotion_runtime: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        self.writer = writer
        self.reward_calculator = reward_calculator
        self.reward_runtime = reward_runtime
        self.emotion_runtime = emotion_runtime
        self.logger = logger
        if reward_calculator is not None and reward_runtime is not None:
            raise ValueError("provide reward_calculator or reward_runtime, not both")
        self.include_private_roles = bool(include_private_roles)
        self._base_metadata = _snapshot_metadata(job_snapshot)
        self._base_metadata.update(_json_safe(metadata or {}))
        self._active: dict[str, Any] | None = None
        self._completed: list[CanonicalEpisode] = []
        self._flushed_count = 0
        self._finalized = False

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is None:
            return
        try:
            if hasattr(self.logger, "event"):
                self.logger.event(event, **fields)
            elif callable(self.logger):
                self.logger({"event": event, **fields})
        except Exception:
            # Telemetry must not change canonical collection semantics.
            return

    @property
    def active_episode_id(self) -> str | None:
        return None if self._active is None else self._active["episode_id"]

    @property
    def episodes(self) -> tuple[CanonicalEpisode, ...]:
        return tuple(self._completed)

    @property
    def records(self) -> tuple[CanonicalEpisode, ...]:
        return self.episodes

    @property
    def completed_episodes(self) -> tuple[CanonicalEpisode, ...]:
        return self.episodes

    @property
    def current_episode(self) -> Mapping[str, Any] | None:
        return None if self._active is None else self._active

    def start_episode(
        self,
        environment: Any | None = None,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        roles: Iterable[Any] | Mapping[Any, Any] | None = None,
        initial_state: Any | None = None,
        initial_observation: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CanonicalEpisode | None:
        if self._finalized:
            raise CollectorStateError("collector has been finalized")
        if self._active is not None:
            raise CollectorStateError("finish the active episode before starting another")
        state = getattr(environment, "state", None) if environment is not None else None
        if state is None and initial_state is not None:
            state = initial_state
        resolved_id = episode_id or getattr(state, "episode_id", None) or getattr(environment, "episode_id", None)
        if resolved_id is None:
            raise ValueError("episode_id is required when no environment/state is supplied")
        resolved_seed = seed if seed is not None else getattr(state, "seed", None)
        if resolved_seed is None:
            raise ValueError("seed is required when no environment/state is supplied")
        resolved_seed = int(resolved_seed)
        if roles is None and state is not None:
            roles = getattr(state, "roles", None)
        role_values: tuple[int, ...] | None
        if roles is None:
            role_values = None
        elif isinstance(roles, Mapping):
            role_values = tuple(int(roles[index]) for index in CANONICAL_PLAYER_ORDER)
        else:
            role_values = tuple(int(getattr(role, "value", role)) for role in roles)
            if len(role_values) != 5:
                raise ValueError("roles must contain exactly five canonical role labels")
        if initial_observation is None and environment is not None and hasattr(environment, "observe"):
            initial_observation = environment.observe()
        if initial_observation is None and state is not None and hasattr(state, "to_public"):
            initial_observation = state.to_public(getattr(state, "agent_selection", 0)).to_dict()
        initial_public = {} if initial_observation is None else _observation(initial_observation)
        if state is not None:
            initial_public.setdefault("healths", list(getattr(state, "health", ())))
            initial_public.setdefault("alive", list(getattr(state, "alive", ())))
            initial_public.setdefault("turn_order", list(CANONICAL_PLAYER_ORDER))
        episode_metadata = dict(self._base_metadata)
        episode_metadata.update(_json_safe(metadata or {}))
        episode_metadata.setdefault("dataset_schema_version", CANONICAL_DATASET_SCHEMA_VERSION)
        episode_metadata.setdefault("action_space_version", ACTION_SPACE_VERSION)
        episode_metadata.setdefault("environment_version", getattr(environment, "metadata", {}).get("version", "1.0") if environment is not None else "1.0")
        episode_metadata.setdefault("observation_schema_version", getattr(environment, "observation_schema_version", "1.0"))
        if environment is not None and getattr(environment, "max_episode_steps", None) is not None:
            episode_metadata.setdefault("max_episode_steps", int(environment.max_episode_steps))
        if roles is not None:
            episode_metadata.setdefault("role_label_schema", "role_index_uint8")
        self._active = {
            "episode_id": str(resolved_id), "seed": resolved_seed, "agent_order": tuple(CANONICAL_PLAYER_ORDER),
            "metadata": episode_metadata, "initial_state": initial_public,
            "roles": role_values, "steps": [], "events": [], "event_ids": set(),
            "invalid_actions": [], "pending": None, "environment": environment,
            "final_state": None, "terminal_result": None,
            "termination_status": None, "termination_reason": None,
            "emotion_schema_version": episode_metadata.get("emotion_schema_version"),
            "reward_config_version": episode_metadata.get("reward_config_version"),
            "reward_config_id": episode_metadata.get("reward_config_id"),
        }
        self._log(
            "episode_started",
            episode_id=str(resolved_id),
            seed=resolved_seed,
            environment_step=0,
            metadata=episode_metadata,
        )
        return None
    begin_episode = start_episode
    start = start_episode

    def before_action(
        self,
        observation: Any | None = None,
        *,
        action_mask: Any | None = None,
        emotion_record: Any | None = None,
        actor_idx: int | None = None,
    ) -> Mapping[str, Any]:
        if self._active is None:
            raise CollectorStateError("start an episode before capturing an action boundary")
        if self._active["pending"] is not None:
            pending = self._active["pending"]
            if actor_idx is not None and pending["actor_idx"] != int(actor_idx):
                raise CollectorStateError("actor changed while retrying a pre-action boundary")
            return dict(pending["observation"])
        if observation is None and self._active["environment"] is not None:
            environment = self._active["environment"]
            if hasattr(environment, "observe"):
                observation = environment.observe()
        pre = {} if observation is None else _observation(observation)
        if emotion_record is None and self.emotion_runtime is not None:
            emotion_record = self.emotion_runtime
        labels, schema, label_observation = _emotion_record(emotion_record)
        if label_observation is not None and not pre:
            pre = label_observation
        raw_step = pre.get("environment_step", len(self._active["steps"]))
        step = _scalar_int(raw_step, "pre-action environment_step")
        actor = actor_idx
        if actor is None:
            actor = pre.get("current_player_idx")
        actor = None if actor is None else _scalar_int(actor, "pre-action actor_idx")
        self._active["pending"] = {
            "observation": pre, "action_mask": _mask(action_mask, observation=pre),
            "emotion_labels": labels, "emotion_schema_version": schema,
            "pre_event_ids": _event_ids(pre) or tuple(int(item) for item in self._active["event_ids"]),
            "environment_step": step, "actor_idx": actor,
        }
        if schema is not None:
            self._active["emotion_schema_version"] = schema
            self._active["metadata"].setdefault("emotion_schema_version", schema)
        return dict(pre)

    prepare_action = before_action

    def record_invalid_action(
        self,
        action: Any,
        *,
        actor_idx: int | None = None,
        action_mask: Any | None = None,
        reason: str | None = None,
    ) -> None:
        if self._active is None:
            raise CollectorStateError("start an episode before recording diagnostics")
        if isinstance(action, np.generic):
            action = action.item()
        if isinstance(action, (int, np.integer)) and not isinstance(action, bool):
            action_value: Any = int(action)
        else:
            action_value = _json_safe(action)
        pending = self._active.get("pending")
        if actor_idx is None and pending is not None:
            actor_idx = pending.get("actor_idx")
        if action_mask is None and pending is not None:
            action_mask = pending.get("action_mask")
        self._active["invalid_actions"].append(_json_safe({
            "attempt_index": len(self._active["invalid_actions"]),
            "environment_step": len(self._active["steps"]),
            "actor_idx": actor_idx,
            "action": action_value,
            "action_mask": list(_mask(action_mask)),
            "reason": reason,
        }))
        self._log(
            "invalid_action",
            episode_id=self._active["episode_id"],
            environment_step=len(self._active["steps"]),
            agent_id=actor_idx,
            action=action_value,
            action_mask=list(_mask(action_mask)),
            reason=reason,
        )

    on_invalid_action = record_invalid_action

    def record_step(
        self,
        result: Any,
        *,
        observation: Any | None = None,
        post_observation: Any | None = None,
        action_mask: Any | None = None,
        emotion_record: Any | None = None,
        reward_batch: Any | None = None,
    ) -> CanonicalStep:
        if self._active is None:
            raise CollectorStateError("start an episode before recording a step")
        values = _result_values(result)
        step_number = _scalar_int(values["environment_step"], "environment_step")
        expected = len(self._active["steps"]) + 1
        if step_number != expected:
            raise CollectorStateError(f"valid environment steps must be contiguous: expected {expected}, got {step_number}")
        pending = self._active.get("pending")
        if observation is None and pending is not None:
            observation = pending["observation"]
        if emotion_record is None and self.emotion_runtime is not None:
            emotion_record = self.emotion_runtime
        labels, emotion_schema, label_observation = _emotion_record(emotion_record)
        if not labels and pending is not None:
            labels = dict(pending["emotion_labels"])
            emotion_schema = pending.get("emotion_schema_version")
        if observation is None and label_observation is not None:
            observation = label_observation
        pre = {} if observation is None else _observation(observation)
        action_index, actor_idx, target_idx, action_type = _action_fields(values["action"])
        pre_step_value = pending.get("environment_step") if pending is not None else pre.get("environment_step")
        if pre_step_value is not None and _scalar_int(pre_step_value, "pre-action environment_step") != step_number - 1:
            raise CollectorStateError("pre-action observation must immediately precede the validated result step")
        for label in labels.values():
            if int(label["environment_step"]) != step_number - 1:
                raise CollectorStateError("emotion labels must use the validated action's pre-action step")
        if pending is not None and pending.get("actor_idx") is not None and pending["actor_idx"] != actor_idx:
            raise CollectorStateError("pre-action actor does not match the validated action actor")
        mask = _mask(action_mask, observation=pre if pre else None)
        if pending is not None and action_mask is None:
            mask = pending["action_mask"]
        post_source = post_observation if post_observation is not None else values.get("observation")
        if post_source is None and self._active["environment"] is not None and hasattr(self._active["environment"], "observe"):
            post_source = self._active["environment"].observe()
        post = None if post_source is None else _observation(post_source)
        if reward_batch is None:
            role_context = self._active["roles"]
            terminal_context = None
            if isinstance(values.get("info"), Mapping):
                terminal_context = values["info"].get("terminal_result")
            if self.reward_runtime is not None:
                reward_batch = self.reward_runtime.on_action_result(
                    values, roles=role_context, terminal_result=terminal_context
                )
            elif self.reward_calculator is not None:
                reward_batch = self.reward_calculator.calculate_all(
                    values, roles=role_context, terminal_result=terminal_context
                )
        rewards, reward_version, reward_id = _reward_record(reward_batch, values)
        if reward_version is not None:
            self._active["reward_config_version"] = reward_version
            self._active["metadata"].setdefault("reward_config_version", reward_version)
        if reward_id is not None:
            self._active["reward_config_id"] = reward_id
            self._active["metadata"].setdefault("reward_config_id", reward_id)
        acting_reward = float(values.get("reward", 0.0))
        acting_breakdown = rewards.get(str(actor_idx))
        if isinstance(acting_breakdown, Mapping) and "total" in acting_breakdown:
            # RewardRuntime/RewardCalculator is authoritative when present;
            # ActionResult.reward is often the environment transport default.
            acting_reward = float(acting_breakdown["total"])
        event_values = tuple(_event_mapping(event) for event in values.get("events", ()))
        ids: list[int] = []
        for event in event_values:
            event_id = int(event["event_id"])
            ids.append(event_id)
            if event_id not in self._active["event_ids"]:
                self._active["event_ids"].add(event_id)
                self._active["events"].append(event)
        pre_ids = tuple(pending["pre_event_ids"]) if pending is not None else _event_ids(pre)
        if not pre_ids and self._active["events"]:
            pre_ids = tuple(int(event["event_id"]) for event in self._active["events"] if int(event["environment_step"]) < step_number)
        step = CanonicalStep(
            environment_step=step_number, actor_idx=actor_idx, action=action_index,
            action_type=action_type, target_idx=target_idx, action_mask=mask,
            pre_observation=pre, post_observation=post, pre_event_ids=pre_ids,
            event_ids=tuple(ids), hp_before=values.get("hp_before"), hp_after=values.get("hp_after"),
            eliminated_idx=values.get("eliminated_idx"), reward=acting_reward,
            reward_breakdowns=rewards, emotion_labels=labels,
            terminated=bool(values.get("terminated", False)),
            truncated=bool(values.get("truncated", False)),
        )
        self._active["steps"].append(step)
        self._active["pending"] = None
        self._log(
            "environment_step",
            episode_id=self._active["episode_id"],
            environment_step=step_number,
            agent_id=actor_idx,
            action=action_index,
            action_type=action_type,
            target_idx=target_idx,
            reward=acting_reward,
            reward_breakdown=rewards,
            terminated=step.terminated,
            truncated=step.truncated,
        )
        if emotion_schema is not None:
            self._active["emotion_schema_version"] = emotion_schema
            self._active["metadata"].setdefault("emotion_schema_version", emotion_schema)
        if bool(values.get("terminated")) or bool(values.get("truncated")):
            self.finish_episode(result=values, final_state=post_source)
        return step

    on_action_result = record_step
    record_transition = record_step
    record_valid_action = record_step
    add_step = record_step
    collect = record_step

    def finish_episode(
        self,
        result: Any | None = None,
        *,
        final_state: Any | None = None,
        terminal_result: Any | None = None,
    ) -> CanonicalEpisode:
        if self._active is None:
            if self._completed:
                return self._completed[-1]
            raise CollectorStateError("no active episode to finish")
        values = {} if result is None else _result_values(result)
        environment = self._active["environment"]
        state = getattr(environment, "state", None) if environment is not None else None
        terminated = bool(values.get("terminated", getattr(state, "terminated", False)))
        truncated = bool(values.get("truncated", getattr(state, "truncated", False)))
        if terminated and truncated:
            raise ValueError("terminated and truncated cannot both be true")
        if not terminated and not truncated:
            raise CollectorStateError("only normal or truncated episodes can be finalized")
        status = "normal" if terminated else "truncated"
        if final_state is None:
            final_state = state
        if final_state is None:
            final_state = values.get("observation")
        if final_state is state and state is not None and hasattr(state, "to_public"):
            final_state = state.to_public(getattr(state, "agent_selection", 0)).to_dict()
        final_public = None if final_state is None else _observation(final_state)
        resolved_terminal = terminal_result
        if resolved_terminal is None:
            resolved_terminal = getattr(state, "terminal_result", None) if state is not None else None
        if resolved_terminal is None and isinstance(values.get("info"), Mapping):
            resolved_terminal = values["info"].get("terminal_result")
        terminal_dict = None if resolved_terminal is None else _as_mapping(resolved_terminal, "terminal_result")
        reason = None
        if terminal_dict is not None:
            reason = terminal_dict.get("reason")
        if reason is None and truncated:
            reason = "max_episode_steps_reached"
        if state is not None:
            self._active["metadata"].setdefault("max_episode_steps", getattr(environment, "max_episode_steps", None))
            self._active["metadata"].setdefault("final_health", list(getattr(state, "health", ())))
            self._active["metadata"].setdefault("final_alive", list(getattr(state, "alive", ())))
        self._active["metadata"].setdefault("termination_status", status)
        if truncated:
            self._active["metadata"].setdefault("max_step_diagnostics", {
                "environment_step": len(self._active["steps"]),
                "max_episode_steps": self._active["metadata"].get("max_episode_steps"),
                "truncated": True,
            })
        episode = CanonicalEpisode(
            episode_id=self._active["episode_id"], seed=self._active["seed"],
            agent_order=self._active["agent_order"], metadata=self._active["metadata"],
            initial_state=self._active["initial_state"], steps=tuple(self._active["steps"]),
            events=tuple(self._active["events"]), termination_status=status,
            termination_reason=None if reason is None else str(reason), terminal_result=terminal_dict,
            final_state=final_public, role_labels=self._active["roles"] if self.include_private_roles else None,
            invalid_actions=tuple(self._active["invalid_actions"]),
        )
        self._completed.append(episode)
        self._log(
            "episode_terminated" if terminated else "episode_truncated",
            episode_id=episode.episode_id,
            environment_step=episode.environment_steps,
            termination_status=episode.termination_status,
            termination_reason=episode.termination_reason,
            event_count=len(episode.events),
            invalid_action_count=len(episode.invalid_actions),
            counters={"environment_steps": episode.environment_steps},
            metadata=episode.metadata,
        )
        self._active = None
        return episode

    on_episode_end = finish_episode
    end_episode = finish_episode
    finalize_episode = finish_episode

    def flush(self) -> tuple[CanonicalEpisode, ...]:
        """Send completed episodes to the configured writer, retaining records."""

        if self.writer is None:
            return self.episodes
        pending = self._completed[self._flushed_count :]
        if not pending:
            return ()
        for episode in pending:
            if hasattr(self.writer, "write_episode"):
                self.writer.write_episode(episode)
            elif hasattr(self.writer, "append"):
                self.writer.append(episode)
            elif hasattr(self.writer, "write"):
                self.writer.write(episode)
            else:
                raise TypeError("dataset writer must provide write_episode, append, or write")
        self._flushed_count = len(self._completed)
        if hasattr(self.writer, "flush"):
            self.writer.flush()
        return tuple(pending)

    def finalize(self) -> tuple[CanonicalEpisode, ...]:
        if self._finalized:
            return self.episodes
        if self._active is not None:
            raise CollectorStateError("cannot finalize with an unfinished episode")
        self.flush()
        if self.writer is not None and hasattr(self.writer, "finalize"):
            self.writer.finalize()
        self._finalized = True
        return self.episodes

    close = finalize

    def to_dict(self, *, include_private: bool = True) -> dict[str, Any]:
        if self._active is not None:
            raise CollectorStateError("serialize only completed episodes")
        return _json_safe({
            "dataset_schema_version": CANONICAL_DATASET_SCHEMA_VERSION,
            "episodes": [episode.to_dict(include_private=include_private) for episode in self._completed],
        })

    def to_json(self, *, include_private: bool = True) -> str:
        return json.dumps(self.to_dict(include_private=include_private), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, writer: DatasetWriter | None = None) -> "EpisodeCollector":
        collector = cls(writer=writer)
        collector._completed = [CanonicalEpisode.from_dict(item) for item in value.get("episodes", [])]
        collector._flushed_count = 0
        return collector

    @classmethod
    def from_json(cls, value: str, *, writer: DatasetWriter | None = None) -> "EpisodeCollector":
        return cls.from_dict(json.loads(value), writer=writer)


__all__ = [
    "ACTION_SPACE_VERSION",
    "CANONICAL_DATASET_SCHEMA_VERSION",
    "CanonicalEpisode",
    "CanonicalStep",
    "CollectorStateError",
    "DatasetWriter",
    "EpisodeCollector",
]
