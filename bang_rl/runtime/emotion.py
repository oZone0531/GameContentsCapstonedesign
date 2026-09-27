"""Execution-time coordination for DummyAgent emotion state and labels.

The environment owns public transitions and events; this module owns the
private emotion modules associated with agent slots.  It deliberately keeps
those boundaries separate: public events are staged as soon as a validated
transition is observed, but they are not applied to emotion state until the
next action boundary.  Consequently a label is always a snapshot of the
state and public history available *before* the action it labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from math import isfinite
from typing import Any, ClassVar

import numpy as np


EMOTION_RUNTIME_STATE_VERSION = "1.0"
PLAYER_COUNT = 5


class EmotionRuntimeError(RuntimeError):
    """Raised when an emotion update is attempted at an invalid boundary."""


@dataclass(frozen=True, slots=True)
class EmotionLabelSnapshot:
    """A detached, pre-action emotion target for one player.

    ``public_event_sequence`` is intentionally the only input context stored
    with a label.  The actual vector is a training target and is never added
    to the public observation returned by :class:`EmotionRuntime`.
    """

    episode_id: str | None
    environment_step: int
    player_idx: int
    values_tuple: tuple[float, ...]
    emotion_names: tuple[str, ...]
    emotion_schema_version: str
    input_event_ids: tuple[int, ...]
    public_event_sequence: tuple[dict[str, Any], ...]
    public_observation: Mapping[str, Any]

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.values_tuple, dtype=np.float32).copy()

    @property
    def emotion_values(self) -> np.ndarray:
        return self.values

    @property
    def state(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(self.emotion_names, self.values_tuple)
        }

    @property
    def event_sequence(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self.public_event_sequence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "environment_step": self.environment_step,
            "player_idx": self.player_idx,
            "values": list(self.values_tuple),
            "emotion_names": list(self.emotion_names),
            "emotion_schema_version": self.emotion_schema_version,
            "input_event_ids": list(self.input_event_ids),
            "public_event_sequence": [dict(event) for event in self.public_event_sequence],
            "public_observation": _json_safe(self.public_observation),
        }


@dataclass(frozen=True, slots=True)
class EmotionTransitionRecord:
    """Collector-facing separation of public features and private targets."""

    label_step: int
    actor_idx: int | None
    labels: Mapping[int, EmotionLabelSnapshot]
    public_observation: Mapping[str, Any]
    post_action_event_ids: tuple[int, ...] = ()
    result_step: int | None = None
    terminated: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_step": self.label_step,
            "actor_idx": self.actor_idx,
            "labels": {
                str(index): label.to_dict() for index, label in self.labels.items()
            },
            "public_observation": _json_safe(self.public_observation),
            "post_action_event_ids": list(self.post_action_event_ids),
            "result_step": self.result_step,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _int(value: Any, name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_public_dict"):
        result = value.to_public_dict()
        if isinstance(result, Mapping):
            return result
    if hasattr(value, "to_dict"):
        # ActionResult.to_dict is safe for event/result metadata.  GameEvent
        # is handled by to_public_dict first, so private role data cannot leak.
        result = value.to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError(f"{label} must be a mapping or a public serializable value")


def _event_mapping(event: Any) -> dict[str, Any]:
    if hasattr(event, "public") and getattr(event, "public") is False:
        raise ValueError("private events cannot enter the public emotion stream")
    if isinstance(event, Mapping):
        values = dict(event)
        if values.get("public", True) is False:
            raise ValueError("private events cannot enter the public emotion stream")
    elif hasattr(event, "to_public_dict"):
        values = dict(event.to_public_dict())
    else:
        values = dict(_mapping(event, "event"))
        if values.get("public", True) is False:
            raise ValueError("private events cannot enter the public emotion stream")
    forbidden = {
        "roles", "true_roles", "actual_roles", "hidden_roles",
        "privileged_state", "privileged_observation", "private_observation",
        "emotion_label", "reward_breakdown",
    }
    leaked = sorted(forbidden.intersection(str(key) for key in values))
    if leaked:
        raise ValueError("privileged event fields are not accepted: " + ", ".join(leaked))
    event_id = _int(values.get("event_id"), "event_id")
    event_type = values.get("event_type", values.get("type", values.get("event")))
    if event_type is None:
        raise ValueError("public emotion events require an event_type")
    if hasattr(event_type, "value"):
        event_type = event_type.value
    values["event_id"] = event_id
    values["event_type"] = str(event_type).rsplit(".", 1)[-1].lower()
    values["public"] = True
    return _json_safe(values)


def _event_list(events: Any) -> tuple[dict[str, Any], ...]:
    if events is None:
        return ()
    if isinstance(events, Mapping) or hasattr(events, "event_id"):
        events = (events,)
    try:
        return tuple(_event_mapping(event) for event in events)
    except TypeError as exc:
        raise TypeError("events must be an iterable of public events") from exc


def _module_from_agent(value: Any) -> Any:
    # Importing EmotionModule here avoids a package-import cycle: agents.base
    # imports bang_rl.runtime.seed during package initialization.
    from bang_rl.agents.dummy_agent.emotion_module import EmotionModule

    module = value
    if not isinstance(module, EmotionModule):
        module = getattr(value, "emotion_module", None)
    if not isinstance(module, EmotionModule):
        raise TypeError("each runtime agent must expose an EmotionModule")
    return module


class EmotionRuntime:
    """Own and coordinate one private ``EmotionModule`` per agent slot.

    A normal execution loop is::

        labels = runtime.before_action(step, public_events, observation=obs,
                                       actor_idx=environment.state.agent_selection)
        result = environment.step(action)
        runtime.on_action_result(result)

    Invalid action paths must call ``on_invalid_action`` (or simply skip the
    result callback); they never advance this runtime.  ``on_action_result``
    accepts only a validated ``ActionResult``-shaped value and performs action,
    reward, event staging, and terminal updates in that order.
    """

    state_version: ClassVar[str] = EMOTION_RUNTIME_STATE_VERSION

    def __init__(
        self,
        agents: Mapping[Any, Any] | Iterable[Any] | None = None,
        *,
        emotion_modules: Mapping[Any, Any] | Iterable[Any] | None = None,
        modules: Mapping[Any, Any] | Iterable[Any] | None = None,
    ) -> None:
        supplied = [value is not None for value in (agents, emotion_modules, modules)]
        if sum(supplied) > 1:
            raise ValueError("provide only one of agents, emotion_modules, or modules")
        source = agents if agents is not None else emotion_modules if emotion_modules is not None else modules
        self._modules = self._coerce_modules(source)
        self._validate_schema()
        self.episode_id: str | None = None
        self.episode_seed: int | None = None
        self.environment_step: int | None = None
        self._prepared_step: int | None = None
        self._prepared_actor_idx: int | None = None
        self._prepared_labels: dict[int, EmotionLabelSnapshot] = {}
        self._prepared_event_ids: tuple[int, ...] = ()
        self._prepared_action_mask: np.ndarray | None = None
        self._action_committed = False
        self._last_result_step: int | None = None
        self._terminated = False
        self._truncated = False
        self._alive = {index: True for index in self._modules}
        self._event_history: list[dict[str, Any]] = []
        self._event_by_id: dict[int, dict[str, Any]] = {}
        self._pending_event_ids: list[int] = []
        self._records: list[EmotionTransitionRecord] = []
        self._reset_done = False

    @staticmethod
    def _coerce_modules(source: Any) -> dict[int, Any]:
        if source is None:
            return {}
        if isinstance(source, Mapping):
            items = source.items()
        else:
            items = enumerate(source)
        modules: dict[int, Any] = {}
        for raw_index, value in items:
            index = _int(raw_index, "player_idx")
            if index >= PLAYER_COUNT:
                raise ValueError("player_idx must be in [0, 4]")
            if index in modules:
                raise ValueError(f"duplicate emotion module for player {index}")
            modules[index] = _module_from_agent(value)
        return dict(sorted(modules.items()))

    def _validate_schema(self) -> None:
        if not self._modules:
            return
        first = next(iter(self._modules.values()))
        names = tuple(first.emotion_names)
        version = str(first.config.schema_version)
        count = int(first.emotion_count)
        for index, module in self._modules.items():
            if int(module.emotion_count) != count or tuple(module.emotion_names) != names:
                raise ValueError(f"emotion schema shape mismatch for player {index}")
            if str(module.config.schema_version) != version:
                raise ValueError(f"emotion schema version mismatch for player {index}")
            self._assert_module_bounds(module, index)

    @staticmethod
    def _assert_module_bounds(module: Any, index: int) -> None:
        values = np.asarray(module.values, dtype=np.float32)
        if values.ndim != 1 or values.shape != (int(module.emotion_count),):
            raise ValueError(f"emotion state for player {index} has the wrong shape")
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"emotion state for player {index} must be finite and in [0, 1]")

    @property
    def player_indices(self) -> tuple[int, ...]:
        return tuple(self._modules)

    @property
    def emotion_schema_version(self) -> str | None:
        if not self._modules:
            return None
        return str(next(iter(self._modules.values())).config.schema_version)

    @property
    def emotion_names(self) -> tuple[str, ...]:
        if not self._modules:
            return ()
        return tuple(next(iter(self._modules.values())).emotion_names)

    @property
    def emotion_count(self) -> int:
        return len(self.emotion_names)

    @property
    def labels(self) -> dict[int, EmotionLabelSnapshot]:
        return dict(self._prepared_labels)

    @property
    def label_snapshots(self) -> dict[int, EmotionLabelSnapshot]:
        return self.labels

    @property
    def records(self) -> tuple[EmotionTransitionRecord, ...]:
        return tuple(self._records)

    @property
    def training_records(self) -> tuple[EmotionTransitionRecord, ...]:
        return self.records

    @property
    def pending_event_ids(self) -> tuple[int, ...]:
        return tuple(self._pending_event_ids)

    @property
    def processed_event_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._event_by_id))

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def private_state(self) -> dict[int, np.ndarray]:
        """Return detached target vectors; never use this for an observation."""

        return {
            index: np.asarray(module.values, dtype=np.float32).copy()
            for index, module in self._modules.items()
        }

    @property
    def states(self) -> dict[int, np.ndarray]:
        return self.private_state

    def reset(self, context: Any | None = None, *, episode_id: Any = None, seed: Any = None) -> dict[int, np.ndarray]:
        if context is not None and (episode_id is not None or seed is not None):
            raise ValueError("provide context or episode_id/seed, not both")
        if context is not None:
            if isinstance(context, Mapping):
                episode_id = context.get("episode_id")
                seed = context.get("seed", context.get("episode_seed"))
            else:
                episode_id = getattr(context, "episode_id", None)
                seed = getattr(context, "seed", getattr(context, "episode_seed", None))
        if episode_id is not None:
            episode_id = str(episode_id)
        if seed is not None:
            seed = _int(seed, "seed", nonnegative=False)
        self.episode_id = episode_id
        self.episode_seed = seed
        self.environment_step = None
        self._prepared_step = None
        self._prepared_actor_idx = None
        self._prepared_labels = {}
        self._prepared_event_ids = ()
        self._prepared_action_mask = None
        self._action_committed = False
        self._last_result_step = None
        self._terminated = False
        self._truncated = False
        self._event_history = []
        self._event_by_id = {}
        self._pending_event_ids = []
        self._records = []
        self._alive = {index: True for index in self._modules}
        context_value = {"episode_id": episode_id, "seed": seed}
        for module in self._modules.values():
            module.on_episode_start(context_value)
            self._assert_module_bounds(module, int(module.player_idx) if module.player_idx is not None else 0)
        self._reset_done = True
        return self.private_state

    on_episode_start = reset

    def receive_events(self, events: Any) -> tuple[int, ...]:
        """Stage public events without applying them before a valid boundary."""

        values = _event_list(events)
        return self._stage_events(values)

    ingest_events = receive_events

    def _stage_events(self, events: tuple[dict[str, Any], ...]) -> tuple[int, ...]:
        staged: list[int] = []
        for event in events:
            event_id = int(event["event_id"])
            previous = self._event_by_id.get(event_id)
            if previous is not None:
                if previous != event:
                    raise ValueError(f"event_id {event_id} was reused with different public data")
                continue
            self._event_by_id[event_id] = dict(event)
            self._event_history.append(dict(event))
            self._pending_event_ids.append(event_id)
            staged.append(event_id)
        return tuple(staged)

    def _apply_pending_events(self, alive_players: Iterable[int] | None = None) -> None:
        if not self._pending_event_ids:
            return
        allowed = None if alive_players is None else {_int(index, "alive player") for index in alive_players}
        pending = tuple(self._pending_event_ids)
        for event_id in pending:
            event = self._event_by_id[event_id]
            for index, module in self._modules.items():
                if allowed is None or index in allowed:
                    module.apply_event(event)
                    self._assert_module_bounds(module, index)
        self._pending_event_ids = []

    def public_observation(self, *, observation: Any | None = None) -> dict[str, Any]:
        """Return public context only; private emotion values/labels are absent."""

        values: dict[str, Any] = {
            "episode_id": self.episode_id,
            "environment_step": 0 if self.environment_step is None else self.environment_step,
            "event_sequence": tuple(dict(event) for event in self._event_history),
            "event_ids": tuple(int(event["event_id"]) for event in self._event_history),
            "emotion_schema_version": self.emotion_schema_version,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }
        if observation is not None:
            source = _mapping(observation, "observation")
            forbidden = {
                "emotion", "emotions", "emotion_state", "emotion_label",
                "emotion_labels", "true_emotion", "reward", "reward_breakdown",
                "roles", "true_roles", "actual_roles", "hidden_roles",
                "privileged_state", "privileged_observation", "private_observation",
            }
            leaked = sorted(forbidden.intersection(str(key) for key in source))
            if leaked:
                raise ValueError("privileged fields are not accepted: " + ", ".join(leaked))
            reserved = {
                "episode_id", "environment_step", "event_sequence", "event_ids",
                "emotion_schema_version", "terminated", "truncated",
            }
            values.update({
                str(key): _json_safe(item)
                for key, item in source.items()
                if str(key) not in reserved
            })
        for key in tuple(values):
            if key.lower().startswith("emotion") and key != "emotion_schema_version":
                values.pop(key, None)
        return values

    observation = public_observation

    def before_action(
        self,
        environment_step: Any,
        events: Any = (),
        *,
        actor_idx: Any | None = None,
        observation: Any | None = None,
        action_mask: Any | None = None,
        alive_players: Iterable[int] | None = None,
    ) -> dict[int, EmotionLabelSnapshot]:
        """Advance one valid action boundary and snapshot all agent labels."""

        step = _int(environment_step, "environment_step")
        actor = None if actor_idx is None else _int(actor_idx, "actor_idx")
        incoming = _event_list(events)
        incoming_ids = tuple(int(event["event_id"]) for event in incoming)
        for event_value in incoming:
            event_step = event_value.get("environment_step")
            if event_step is not None and _int(event_step, "event environment_step") > step:
                raise EmotionRuntimeError("future events cannot enter a pre-action label")
        if self._terminated or self._truncated:
            raise EmotionRuntimeError("cannot prepare an action after episode termination")
        if self._prepared_step == step:
            if any(event_id not in self._event_by_id for event_id in incoming_ids):
                raise EmotionRuntimeError("a second event boundary was supplied for the same step")
            if actor != self._prepared_actor_idx and actor is not None:
                raise EmotionRuntimeError("actor changed while retrying the same action")
            return self.labels
        if self._prepared_step is not None and step <= self._prepared_step:
            raise EmotionRuntimeError("environment_step must advance at a valid action boundary")
        if self._prepared_step is not None and step != self._prepared_step + 1:
            raise EmotionRuntimeError("environment_step cannot skip a valid action boundary")
        self._stage_events(incoming)
        if alive_players is not None:
            alive_set = {_int(index, "alive player") for index in alive_players}
            self._alive = {index: index in alive_set for index in self._modules}
        self._apply_pending_events(self._alive)
        for index, module in self._modules.items():
            module.advance_step(step)
            self._assert_module_bounds(module, index)
        self.environment_step = step
        public = self.public_observation(observation=observation)
        event_ids = tuple(int(event["event_id"]) for event in self._event_history)
        sequence = tuple(dict(event) for event in self._event_history)
        self._prepared_labels = {
            index: EmotionLabelSnapshot(
                episode_id=self.episode_id,
                environment_step=step,
                player_idx=index,
                values_tuple=tuple(float(value) for value in module.values),
                emotion_names=tuple(module.emotion_names),
                emotion_schema_version=str(module.config.schema_version),
                input_event_ids=event_ids,
                public_event_sequence=sequence,
                public_observation=public,
            )
            for index, module in self._modules.items()
        }
        self._prepared_step = step
        self._prepared_actor_idx = actor
        self._prepared_event_ids = event_ids
        self._prepared_action_mask = None if action_mask is None else np.asarray(action_mask, dtype=np.uint8).copy()
        self._action_committed = False
        return self.labels

    prepare_action = before_action
    pre_action = before_action
    snapshot_labels = before_action
    label_snapshot = before_action

    def _result_values(self, result: Any) -> Mapping[str, Any]:
        values = _mapping(result, "action result")
        if "action" not in values or "environment_step" not in values:
            raise TypeError("action result must contain action and environment_step")
        return values

    @staticmethod
    def _action_values(action: Any) -> Mapping[str, Any]:
        if isinstance(action, Mapping):
            return action
        if hasattr(action, "to_dict"):
            return action.to_dict()
        raise TypeError("action result action must be serializable")

    def _action_index(self, action: Any) -> int:
        values = self._action_values(action)
        action_type = values.get("action_type", values.get("type"))
        if hasattr(action_type, "value"):
            action_type = action_type.value
        if isinstance(action_type, str):
            normalized = action_type.rsplit(".", 1)[-1].lower()
            if normalized in {"pass", "0"}:
                return 0
            target = values.get("target_idx")
            if target is None:
                raise ValueError("BANG action requires target_idx")
            return _int(target, "target_idx") + 1
        kind = _int(action_type, "action_type")
        if kind == 0:
            return 0
        target = values.get("target_idx")
        if target is None:
            raise ValueError("BANG action requires target_idx")
        return _int(target, "target_idx") + 1

    @staticmethod
    def _validate_reward_value(value: Any) -> None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ) or not isfinite(float(value)):
            raise ValueError("reward must be a finite number")

    def _validate_result_rewards(self, values: Mapping[str, Any]) -> None:
        info = values.get("info", {})
        reward_map = info.get("rewards", info.get("reward_by_agent")) if isinstance(info, Mapping) else None
        if isinstance(reward_map, Mapping):
            for raw_index, reward in reward_map.items():
                _int(raw_index, "reward player")
                self._validate_reward_value(reward)
        elif "reward" in values:
            self._validate_reward_value(values.get("reward"))

    def _apply_result_rewards(self, values: Mapping[str, Any], actor_idx: int) -> None:
        info = values.get("info", {})
        if not isinstance(info, Mapping):
            info = {}
        reward_map = info.get("rewards", info.get("reward_by_agent"))
        if isinstance(reward_map, Mapping):
            for raw_index, reward in reward_map.items():
                index = _int(raw_index, "reward player")
                module = self._modules.get(index)
                if module is not None:
                    module.apply_reward(reward, valid=True)
                    self._assert_module_bounds(module, index)
            return
        if actor_idx in self._modules and "reward" in values:
            self._modules[actor_idx].apply_reward(values.get("reward", 0.0), valid=True)
            self._assert_module_bounds(self._modules[actor_idx], actor_idx)

    def apply_reward(self, player_idx: Any, reward: Any, *, valid: bool = True) -> np.ndarray | None:
        """Apply a training reward only when the caller confirms validity."""

        if not valid:
            return None if _int(player_idx, "player_idx") not in self._modules else self._modules[_int(player_idx, "player_idx")].values
        index = _int(player_idx, "player_idx")
        module = self._modules.get(index)
        if module is None:
            raise KeyError(f"no emotion module registered for player {index}")
        values = module.apply_reward(reward, valid=True)
        self._assert_module_bounds(module, index)
        return values

    def apply_terminal(self, result: Any, *, valid: bool = True) -> dict[int, np.ndarray]:
        if not valid:
            return self.private_state
        values = _mapping(result, "terminal result") if result is not None else {}
        terminated = bool(values.get("terminated", False))
        truncated = bool(values.get("truncated", False))
        if terminated and truncated:
            raise ValueError("terminated and truncated cannot both be true")
        if not terminated and not truncated:
            return self.private_state
        self._terminated = terminated
        self._truncated = truncated
        for index, module in self._modules.items():
            module.apply_terminal(values)
            self._assert_module_bounds(module, index)
        return self.private_state

    def on_action_result(self, result: Any, *, valid: bool = True) -> EmotionTransitionRecord | None:
        """Commit one validated environment transition after its label snapshot."""

        if not valid:
            return None
        values = self._result_values(result)
        result_step = _int(values["environment_step"], "result environment_step")
        if self._prepared_step is None:
            raise EmotionRuntimeError("an action result requires before_action first")
        if result_step != self._prepared_step + 1:
            raise EmotionRuntimeError("action result is not the next environment transition")
        if self._action_committed:
            if self._last_result_step == result_step:
                return self._records[-1] if self._records else None
            raise EmotionRuntimeError("only one valid action result is allowed per boundary")
        action = values["action"]
        action_values = self._action_values(action)
        actor_idx = _int(action_values.get("actor_idx"), "actor_idx")
        if actor_idx not in self._modules:
            raise ValueError(f"no emotion module registered for actor {actor_idx}")
        action_index = self._action_index(action)
        events = _event_list(values.get("events", ()))
        for event_value in events:
            event_step = event_value.get("environment_step")
            if event_step is not None and _int(event_step, "event environment_step") != result_step:
                raise EmotionRuntimeError("action result events must belong to its environment step")
        terminated = bool(values.get("terminated", False))
        truncated = bool(values.get("truncated", False))
        if terminated and truncated:
            raise ValueError("terminated and truncated cannot both be true")
        self._validate_result_rewards(values)
        mask = self._prepared_action_mask
        self._modules[actor_idx].apply_action(action_index, valid=True, action_mask=mask)
        self._assert_module_bounds(self._modules[actor_idx], actor_idx)
        self._apply_result_rewards(values, actor_idx)
        post_ids = self._stage_events(events)
        info = values.get("info", {})
        if isinstance(info, Mapping):
            alive = info.get("alive_players")
            if alive is not None:
                self._alive = {index: index in set(_int(item, "alive player") for item in alive) for index in self._modules}
        eliminated = values.get("eliminated_idx")
        if eliminated is not None and _int(eliminated, "eliminated_idx") in self._alive:
            self._alive[_int(eliminated, "eliminated_idx")] = False
        self.environment_step = result_step
        self._last_result_step = result_step
        self._action_committed = True
        self.apply_terminal(values, valid=True)
        record = EmotionTransitionRecord(
            label_step=int(self._prepared_step),
            actor_idx=self._prepared_actor_idx if self._prepared_actor_idx is not None else actor_idx,
            labels=self.labels,
            public_observation=dict(
                next(iter(self._prepared_labels.values())).public_observation
            ) if self._prepared_labels else self.public_observation(),
            post_action_event_ids=post_ids,
            result_step=result_step,
            terminated=bool(values.get("terminated", False)),
            truncated=bool(values.get("truncated", False)),
        )
        self._records.append(record)
        return record

    commit = on_action_result
    after_action = on_action_result
    process_action_result = on_action_result
    on_valid_transition = on_action_result

    def on_invalid_action(self, *args: Any, **kwargs: Any) -> dict[int, EmotionLabelSnapshot]:
        """Explicit invalid/retry hook; it is intentionally a strict no-op."""

        del args, kwargs
        return self.labels

    handle_invalid_action = on_invalid_action

    def on_episode_end(self, result: Any | None = None) -> dict[int, np.ndarray]:
        if result is None:
            result = {"terminated": self._terminated, "truncated": self._truncated}
        return self.apply_terminal(result)

    def state_dict(self) -> dict[str, Any]:
        self._validate_schema()
        return {
            "state_version": self.state_version,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "environment_step": self.environment_step,
            "prepared_step": self._prepared_step,
            "prepared_actor_idx": self._prepared_actor_idx,
            "prepared_event_ids": list(self._prepared_event_ids),
            "prepared_action_mask": None if self._prepared_action_mask is None else self._prepared_action_mask.tolist(),
            "action_committed": self._action_committed,
            "last_result_step": self._last_result_step,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "alive": {str(index): bool(value) for index, value in self._alive.items()},
            "event_history": [dict(event) for event in self._event_history],
            "pending_event_ids": list(self._pending_event_ids),
            "modules": {str(index): module.state_dict() for index, module in self._modules.items()},
            "labels": {str(index): label.to_dict() for index, label in self._prepared_labels.items()},
            "records": [record.to_dict() for record in self._records],
        }

    to_dict = state_dict
    serialize = state_dict

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EmotionRuntime":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("state_version", cls.state_version) != cls.state_version:
            raise ValueError("unsupported emotion runtime state schema")
        from bang_rl.agents.dummy_agent.emotion_module import EmotionModule

        modules = {
            int(index): EmotionModule.from_state_dict(module_state)
            for index, module_state in state.get("modules", {}).items()
        }
        runtime = cls(emotion_modules=modules)
        runtime.episode_id = state.get("episode_id")
        runtime.episode_seed = state.get("episode_seed")
        runtime.environment_step = state.get("environment_step")
        runtime._prepared_step = state.get("prepared_step")
        runtime._prepared_actor_idx = state.get("prepared_actor_idx")
        runtime._prepared_event_ids = tuple(int(value) for value in state.get("prepared_event_ids", []))
        mask = state.get("prepared_action_mask")
        runtime._prepared_action_mask = None if mask is None else np.asarray(mask, dtype=np.uint8)
        runtime._action_committed = bool(state.get("action_committed", False))
        runtime._last_result_step = state.get("last_result_step")
        runtime._terminated = bool(state.get("terminated", False))
        runtime._truncated = bool(state.get("truncated", False))
        runtime._alive = {int(index): bool(value) for index, value in state.get("alive", {}).items()}
        runtime._event_history = [dict(event) for event in state.get("event_history", [])]
        runtime._event_by_id = {int(event["event_id"]): dict(event) for event in runtime._event_history}
        runtime._pending_event_ids = [int(value) for value in state.get("pending_event_ids", [])]
        runtime._prepared_labels = {
            int(index): runtime._label_from_dict(label)
            for index, label in state.get("labels", {}).items()
        }
        runtime._records = [runtime._record_from_dict(record) for record in state.get("records", [])]
        runtime._reset_done = True
        runtime._validate_schema()
        return runtime

    from_dict = from_state_dict
    deserialize = from_state_dict

    def _label_from_dict(self, value: Mapping[str, Any]) -> EmotionLabelSnapshot:
        values = tuple(float(item) for item in value.get("values", []))
        if len(values) != self.emotion_count or not all(isfinite(item) for item in values):
            raise ValueError("serialized emotion label has the wrong shape or non-finite values")
        if any(item < 0.0 or item > 1.0 for item in values):
            raise ValueError("serialized emotion label values must be in [0, 1]")
        names = tuple(str(item) for item in value.get("emotion_names", []))
        if names != self.emotion_names:
            raise ValueError("serialized emotion label names do not match the runtime schema")
        version = str(value.get("emotion_schema_version", self.emotion_schema_version))
        if version != self.emotion_schema_version:
            raise ValueError("serialized emotion label schema version does not match the runtime")
        return EmotionLabelSnapshot(
            episode_id=value.get("episode_id"),
            environment_step=_int(value["environment_step"], "label environment_step"),
            player_idx=_int(value["player_idx"], "label player_idx"),
            values_tuple=values,
            emotion_names=names,
            emotion_schema_version=version,
            input_event_ids=tuple(int(item) for item in value.get("input_event_ids", [])),
            public_event_sequence=tuple(dict(item) for item in value.get("public_event_sequence", [])),
            public_observation=dict(value.get("public_observation", {})),
        )

    def _record_from_dict(self, value: Mapping[str, Any]) -> EmotionTransitionRecord:
        labels = {
            int(index): self._label_from_dict(label)
            for index, label in value.get("labels", {}).items()
        }
        return EmotionTransitionRecord(
            label_step=_int(value["label_step"], "record label_step"),
            actor_idx=None if value.get("actor_idx") is None else _int(value["actor_idx"], "record actor_idx"),
            labels=labels,
            public_observation=dict(value.get("public_observation", {})),
            post_action_event_ids=tuple(int(item) for item in value.get("post_action_event_ids", [])),
            result_step=None if value.get("result_step") is None else _int(value["result_step"], "record result_step"),
            terminated=bool(value.get("terminated", False)),
            truncated=bool(value.get("truncated", False)),
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = self.from_state_dict(state)
        self.__dict__.update(restored.__dict__)

    def to_json(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "EmotionRuntime":
        if not isinstance(value, str):
            raise TypeError("serialized state must be a JSON string")
        return cls.from_state_dict(json.loads(value))


__all__ = [
    "EMOTION_RUNTIME_STATE_VERSION",
    "EmotionLabelSnapshot",
    "EmotionRuntime",
    "EmotionRuntimeError",
    "EmotionTransitionRecord",
]
