"""Deterministic, public-input-only emotion state for ``DummyAgent``.

The module is deliberately independent from ``bang_rl.env``.  The execution
layer feeds it public events and valid actions; the environment never reaches
into this private state.  Emotion effects are exposed for the later action
selector task, but are not applied to intent selection here.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any, ClassVar

import numpy as np

from bang_rl.config import EmotionConfig


EMOTION_MODULE_STATE_VERSION = "1.0"

# These names are rejected at the agent boundary.  Public role-reveal fields
# such as ``revealed_role`` are intentionally not included in this list.
_PRIVATE_FIELDS = {
    "roles",
    "true_roles",
    "actual_roles",
    "hidden_roles",
    "privileged_state",
    "privileged_observation",
    "private_observation",
    "emotion_label",
    "reward_breakdown",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Convert a public event/action-like value without importing the env."""

    if isinstance(value, Mapping):
        result = value
    elif hasattr(value, "to_public_dict"):
        result = value.to_public_dict()
    elif hasattr(value, "to_dict"):
        result = value.to_dict()
    else:
        result = {
            name: getattr(value, name)
            for name in ("event_id", "event_type", "public", "action", "action_type", "valid")
            if hasattr(value, name)
        }
    if not isinstance(result, Mapping):
        raise TypeError(f"{label} must be a public mapping or public event/action value")
    leaked = sorted(_PRIVATE_FIELDS.intersection(result.keys()))
    if leaked:
        raise ValueError(f"privileged fields are not accepted: {', '.join(leaked)}")
    return result


def _text(value: Any) -> str:
    """Return the stable wire spelling for strings and enum-like values."""

    if hasattr(value, "value"):
        value = value.value
    return str(value).rsplit(".", 1)[-1].strip().lower()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


class EmotionModule:
    """Manage N configurable normalized emotion values deterministically.

    ``event_id`` is the idempotency key for public events.  ``advance_step``
    applies configured recovery/decay at most once for a supplied environment
    step.  Decay is a linear amount toward the configured initial value; all
    mutations are clamped to ``[0, 1]``.
    """

    schema_version: ClassVar[str] = EMOTION_MODULE_STATE_VERSION

    def __init__(
        self,
        config: EmotionConfig | Mapping[str, Any] | Any | None = None,
        *,
        player_idx: int | None = None,
    ) -> None:
        self.config = self._coerce_config(config)
        if player_idx is not None:
            if isinstance(player_idx, (bool, np.bool_)) or not 0 <= int(player_idx) < 5:
                raise ValueError("player_idx must be in [0, 4] or None")
            player_idx = int(player_idx)
        self.player_idx = player_idx
        self._definitions = tuple(self.config.emotions)
        self._names = tuple(definition.name for definition in self._definitions)
        self._baselines = np.asarray(
            [
                self.config.initial_baseline
                if definition.initial_value is None
                else definition.initial_value
                for definition in self._definitions
            ],
            dtype=np.float32,
        )
        self._values = self._baselines.copy()
        self.episode_id: str | None = None
        self.episode_seed: int | None = None
        self.last_event_id: int | None = None
        self.last_environment_step: int | None = None
        self.terminated = False
        self.truncated = False
        self._processed_event_ids: set[int] = set()

    @staticmethod
    def _coerce_config(config: EmotionConfig | Mapping[str, Any] | Any | None) -> EmotionConfig:
        if config is None:
            return EmotionConfig()
        if isinstance(config, EmotionConfig):
            return config
        if hasattr(config, "model_dump"):
            config = config.model_dump(mode="python")
        if isinstance(config, Mapping):
            return EmotionConfig.model_validate(dict(config))
        raise TypeError("config must be EmotionConfig, a mapping, or None")

    def update_config(self, config: EmotionConfig | Mapping[str, Any] | Any) -> None:
        """Apply a preset at a decision boundary without changing current state."""

        updated = self._coerce_config(config)
        old_names = self.emotion_names
        new_names = tuple(item.name for item in updated.emotions)
        if old_names != new_names and updated.schema_version == self.config.schema_version:
            raise ValueError("emotion schema changes require a new schema_version")
        old_values = dict(zip(old_names, self._values))
        self.config = updated
        self._definitions = tuple(updated.emotions)
        self._names = new_names
        self._baselines = np.asarray(
            [
                updated.initial_baseline
                if definition.initial_value is None
                else definition.initial_value
                for definition in self._definitions
            ],
            dtype=np.float32,
        )
        self._values = np.asarray(
            [old_values.get(name, baseline) for name, baseline in zip(self._names, self._baselines)],
            dtype=np.float32,
        )
        self._values = np.clip(self._values, 0.0, 1.0).astype(np.float32)

    @property
    def emotion_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def emotion_count(self) -> int:
        return len(self._names)

    @property
    def values(self) -> np.ndarray:
        """Return a detached float32 vector in schema order."""

        return self._values.copy()

    @property
    def state(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self._names, self._values)}

    @property
    def baseline(self) -> np.ndarray:
        return self._baselines.copy()

    @property
    def processed_event_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._processed_event_ids))

    @property
    def effects(self) -> dict[str, dict[str, float]]:
        return {
            definition.name: dict(sorted(definition.effects.items()))
            for definition in self._definitions
        }

    def effects_for_action(self, action: Any) -> dict[str, float]:
        """Return configured action effects without mutating state.

        Action-selection integration is intentionally deferred to TASK-2-5.
        """

        key, action_type = self._action_keys(action)
        result: dict[str, float] = {}
        for definition in self._definitions:
            for candidate in (key, action_type, "action", "*"):
                if candidate is not None and candidate in definition.effects:
                    result[definition.name] = float(definition.effects[candidate])
                    break
        return result

    def reset(self, context: Any | None = None) -> np.ndarray:
        """Reset values, event deduplication, step clock, and lifecycle flags."""

        self._values = self._baselines.copy()
        self.last_event_id = None
        self.last_environment_step = None
        self._processed_event_ids.clear()
        self.terminated = False
        self.truncated = False
        self.episode_id = None
        self.episode_seed = None
        if context is not None:
            if isinstance(context, Mapping):
                self.episode_id = context.get("episode_id")
                seed = context.get("seed", context.get("episode_seed"))
            else:
                self.episode_id = getattr(context, "episode_id", None)
                seed = getattr(context, "seed", getattr(context, "episode_seed", None))
            if self.episode_id is not None:
                self.episode_id = str(self.episode_id)
            if seed is not None:
                self.episode_seed = int(seed)
        return self.values

    def on_episode_start(self, context: Any | None = None) -> np.ndarray:
        return self.reset(context)

    def _change(self, deltas: list[float]) -> np.ndarray:
        self._values = np.clip(
            self._values + np.asarray(deltas, dtype=np.float32), 0.0, 1.0
        ).astype(np.float32)
        return self.values

    @staticmethod
    def _rule_value(rules: Mapping[str, float], keys: tuple[str | None, ...]) -> float | None:
        for key in keys:
            if key is not None and key in rules:
                return float(rules[key])
        return None

    def _event_value(self, definition: Any, event_type: str) -> float:
        aliases = {
            "elimination": "player_eliminated",
            "role_reveal": "role_revealed",
        }
        normalized = aliases.get(event_type, event_type)
        value = self._rule_value(
            definition.event_updates,
            (event_type, normalized, "*"),
        )
        return definition.change_rate if value is None else value

    def apply_event(self, event: Any) -> np.ndarray:
        """Apply one public event once; duplicate IDs are a no-op."""

        values = _mapping(event, label="event")
        if values.get("public", True) is False:
            return self.values
        event_id = values.get("event_id")
        if isinstance(event_id, (bool, np.bool_)) or not isinstance(event_id, (int, np.integer)):
            raise ValueError("public emotion events require an integer event_id")
        event_id = int(event_id)
        if event_id < 0:
            raise ValueError("event_id must be non-negative")
        if event_id in self._processed_event_ids:
            return self.values
        event_type = _text(values.get("event_type", values.get("type", values.get("event", ""))))
        if not event_type:
            raise ValueError("public emotion events require an event_type")
        deltas = [self._event_value(definition, event_type) for definition in self._definitions]
        self._processed_event_ids.add(event_id)
        self.last_event_id = event_id if self.last_event_id is None else max(self.last_event_id, event_id)
        return self._change(deltas)

    update_from_event = apply_event
    apply_public_event = apply_event

    def apply_events(self, events: Any) -> np.ndarray:
        if events is None:
            return self.values
        if isinstance(events, Mapping) or hasattr(events, "event_id"):
            events = (events,)
        for event in events:
            self.apply_event(event)
        return self.values

    update_from_events = apply_events

    def advance_step(self, environment_step: Any) -> np.ndarray:
        """Apply decay/recovery once for this environment step."""

        if isinstance(environment_step, (bool, np.bool_)) or not isinstance(environment_step, (int, np.integer)):
            raise TypeError("environment_step must be a non-negative integer")
        step = int(environment_step)
        if step < 0:
            raise ValueError("environment_step must be non-negative")
        if self.last_environment_step is not None and step < self.last_environment_step:
            raise ValueError("environment_step cannot move backwards")
        if self.last_environment_step == step:
            return self.values
        if self.last_environment_step is not None or step > 0:
            deltas = []
            for definition, current, baseline in zip(self._definitions, self._values, self._baselines):
                amount = max(0.0, float(definition.decay_rate))
                distance = float(baseline) - float(current)
                deltas.append(np.sign(distance) * min(abs(distance), amount))
            self._change(deltas)
        self.last_environment_step = step
        return self.values

    decay = advance_step
    apply_decay = advance_step

    def prepare_action(self, environment_step: int, events: Any = ()) -> np.ndarray:
        """Apply pending public events and step decay, then snapshot the label."""

        self.apply_events(events)
        self.advance_step(environment_step)
        return self.values

    label_snapshot = prepare_action

    @staticmethod
    def _action_keys(action: Any) -> tuple[str, str]:
        if isinstance(action, Mapping):
            if "action" in action and not isinstance(action["action"], Mapping):
                raw = action["action"]
            else:
                raw = action.get("action_type", action.get("type", action.get("index")))
            if action.get("valid", True) is False:
                return "invalid", "invalid"
        else:
            raw = action
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError("action must be a public integer or action mapping")
        if isinstance(raw, (int, np.integer)):
            number = int(raw)
            if number < 0:
                raise ValueError("action must be non-negative")
            kind = "pass" if number == 0 else "bang" if number == 1 else "bang"
            return str(number), kind
        text = _text(raw)
        return text, text

    def apply_action(
        self,
        action: Any,
        *,
        valid: bool = True,
        action_mask: Any | None = None,
    ) -> np.ndarray:
        """Apply a configured action rule only for a validated game action."""

        if not valid:
            return self.values
        if action_mask is not None:
            mask = np.asarray(action_mask)
            if mask.shape != (6,) or mask.dtype.kind not in "biu" or np.any((mask != 0) & (mask != 1)):
                raise ValueError("action_mask must be a six-entry binary mask")
            key, _ = self._action_keys(action)
            try:
                action_index = int(key)
            except ValueError:
                action_index = None
            if action_index is not None and (action_index < 0 or action_index >= 6 or not bool(mask[action_index])):
                return self.values
        key, action_type = self._action_keys(action)
        deltas = []
        for definition in self._definitions:
            value = self._rule_value(
                definition.action_updates,
                (key, action_type, "*"),
            )
            deltas.append(0.0 if value is None else value)
        return self._change(deltas)

    update_from_action = apply_action
    record_action = apply_action

    def apply_reward(self, reward: Any, *, valid: bool = True) -> np.ndarray:
        """Apply a configured reward rule; invalid/retried actions are no-ops."""

        if not valid:
            return self.values
        amount = _finite_number(reward, "reward")
        sign = "positive" if amount > 0 else "negative" if amount < 0 else "zero"
        exact = str(amount).rstrip("0").rstrip(".") if "." in str(amount) else str(amount)
        deltas = []
        for definition in self._definitions:
            value = self._rule_value(definition.reward_updates, (exact, sign, "*"))
            deltas.append(0.0 if value is None else value)
        return self._change(deltas)

    update_from_reward = apply_reward

    def apply_terminal(self, result: Any) -> np.ndarray:
        """Apply terminal/truncated rules from a public lifecycle result."""

        values = _mapping(result, label="terminal result") if result is not None else {}
        terminated = bool(values.get("terminated", False))
        truncated = bool(values.get("truncated", False))
        if terminated and truncated:
            raise ValueError("terminated and truncated cannot both be true")
        self.terminated = terminated
        self.truncated = truncated
        status = "truncated" if truncated else "terminated" if terminated else "running"
        result_name = values.get("result", values.get("outcome"))
        if isinstance(result_name, Mapping):
            result_name = result_name.get("result", result_name.get("outcome"))
        result_name = None if result_name is None else _text(result_name)
        deltas = []
        for definition in self._definitions:
            value = self._rule_value(
                definition.terminal_updates,
                (result_name, status, "normal" if terminated else None, "*")
            )
            deltas.append(0.0 if value is None else value)
        return self._change(deltas)

    update_from_terminal = apply_terminal

    def on_episode_end(self, result: Any | None = None) -> np.ndarray:
        return self.apply_terminal(result or {})

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "emotion_schema_version": self.config.schema_version,
            "player_idx": self.player_idx,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "config": self.config.model_dump(mode="json"),
            "emotion_names": list(self._names),
            "values": [float(value) for value in self._values],
            "baseline": [float(value) for value in self._baselines],
            "last_event_id": self.last_event_id,
            "processed_event_ids": list(self.processed_event_ids),
            "last_environment_step": self.last_environment_step,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    to_dict = state_dict
    serialize = state_dict

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = self.from_state_dict(state)
        self.__dict__.update(restored.__dict__)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EmotionModule":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("schema_version", cls.schema_version) != cls.schema_version:
            raise ValueError("unsupported emotion module state schema")
        module = cls(state.get("config"), player_idx=state.get("player_idx"))
        names = tuple(state.get("emotion_names", module.emotion_names))
        if names != module.emotion_names:
            raise ValueError("emotion state names do not match its config schema")
        values = np.asarray(state.get("values", module.values), dtype=np.float32)
        if values.shape != (module.emotion_count,) or not np.all(np.isfinite(values)):
            raise ValueError("emotion state values have the wrong shape")
        module._values = np.clip(values, 0.0, 1.0).astype(np.float32)
        module.episode_id = state.get("episode_id")
        module.episode_seed = state.get("episode_seed")
        module.last_event_id = state.get("last_event_id")
        module.last_environment_step = state.get("last_environment_step")
        module.terminated = bool(state.get("terminated", False))
        module.truncated = bool(state.get("truncated", False))
        module._processed_event_ids = {int(item) for item in state.get("processed_event_ids", [])}
        return module

    from_dict = from_state_dict
    deserialize = from_state_dict

    def to_json(self) -> str:
        import json

        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "EmotionModule":
        import json

        if not isinstance(value, str):
            raise TypeError("serialized state must be a JSON string")
        return cls.from_state_dict(json.loads(value))


__all__ = ["EMOTION_MODULE_STATE_VERSION", "EmotionModule"]
