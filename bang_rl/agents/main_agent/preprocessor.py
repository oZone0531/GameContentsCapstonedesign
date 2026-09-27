"""Canonical public-observation preprocessing for MainAgent inference.

The environment deliberately exposes a variable-length public event history.  This
module is the model boundary: it validates the public mapping, converts fixed
fields to stable tensors, and creates one fixed-length stream for each player
other than the viewer.  No environment or privileged state is imported here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from math import isfinite
from typing import Any, ClassVar

import numpy as np

from ...env.types import (
    ACTION_SPACE_SIZE,
    HIDDEN_ROLE_SENTINEL,
    OBSERVATION_SCHEMA_VERSION,
    PLAYER_COUNT,
    PASS_TARGET_SENTINEL,
    Role,
)


PREPROCESSING_VERSION = "1.0"
ACTION_SPACE_VERSION = "1.0"
MODEL_HISTORY_LENGTH = 64
MAX_HEALTH = 5.0
PLAYER_INDEX_MAX = float(PLAYER_COUNT - 1)
EVENT_FEATURE_SIZE = 17
STATE_VECTOR_SIZE = 9
PLAYER_FEATURE_SIZE = 8
OTHER_PLAYER_COUNT = PLAYER_COUNT - 1

# These names are intentionally stable artifact metadata.  They make a saved
# MainAgent input contract inspectable without importing this implementation.
FEATURE_SCHEMA = {
    "state_vector": {"shape": [STATE_VECTOR_SIZE], "dtype": "float32"},
    "player_state_matrix": {
        "shape": [PLAYER_COUNT, PLAYER_FEATURE_SIZE],
        "dtype": "float32",
    },
    "role_beliefs": {"shape": [PLAYER_COUNT, len(Role)], "dtype": "float32"},
    "event_history": {
        "shape": [OTHER_PLAYER_COUNT, MODEL_HISTORY_LENGTH, EVENT_FEATURE_SIZE],
        "dtype": "float32",
    },
    "event_valid_mask": {
        "shape": [OTHER_PLAYER_COUNT, MODEL_HISTORY_LENGTH],
        "dtype": "bool",
    },
    "action_mask": {"shape": [ACTION_SPACE_SIZE], "dtype": "uint8"},
}

_FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {
        "roles",
        "role_labels",
        "hidden_roles",
        "true_roles",
        "privileged_state",
        "privileged_observation",
        "private_observation",
        "emotion_label",
        "emotion_labels",
        "reward",
        "reward_breakdown",
        "privileged_diagnostics",
    }
)
_FORBIDDEN_EVENT_KEYS = frozenset(
    {
        "roles",
        "role",
        "true_role",
        "true_roles",
        "hidden_role",
        "hidden_roles",
        "emotion",
        "emotion_label",
        "reward",
        "reward_breakdown",
    }
)
_REQUIRED_FIELDS = (
    "current_player_idx",
    "environment_step",
    "round_index",
    "healths",
    "alive",
    "turn_order",
    "sheriff_idx",
    "self_role",
    "role_visibility",
    "revealed_roles",
    "confirmed_role_distribution",
    "action_mask",
)


def _json_ready(value: Any) -> Any:
    """Convert arrays and scalar numpy values to deterministic JSON values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _scalar(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.shape not in [(), (1,)]:
        raise ValueError(f"{name} must be a scalar or shape (1,)")
    item = array.reshape(-1)[0].item()
    if not _is_integer(item):
        raise TypeError(f"{name} must be an integer")
    return int(item)


def _integer_array(
    value: Any, shape: tuple[int, ...], name: str, *, allow_bool: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if allow_bool and np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.int64, copy=False)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain integers")
    return array.astype(np.int64, copy=False)


def _float_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain numbers")
    result = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _bounded_unit(value: float, name: str) -> float:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class ObservationPreprocessorConfig:
    """Immutable, serializable input contract for MainAgent preprocessing."""

    model_history_length: int = MODEL_HISTORY_LENGTH
    max_episode_steps: int = 2000
    max_health: int = 5
    preprocessing_version: str = PREPROCESSING_VERSION
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    action_space_version: str = ACTION_SPACE_VERSION
    feature_schema: Mapping[str, Any] = field(
        default_factory=lambda: _json_ready(FEATURE_SCHEMA)
    )

    def __post_init__(self) -> None:
        if isinstance(self.model_history_length, bool) or not isinstance(self.model_history_length, int):
            raise TypeError("model_history_length must be an integer")
        if self.model_history_length != MODEL_HISTORY_LENGTH:
            raise ValueError(f"model_history_length is fixed at {MODEL_HISTORY_LENGTH}")
        if isinstance(self.max_episode_steps, bool) or not isinstance(self.max_episode_steps, int):
            raise TypeError("max_episode_steps must be an integer")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if isinstance(self.max_health, bool) or not isinstance(self.max_health, int):
            raise TypeError("max_health must be an integer")
        if self.max_health <= 0:
            raise ValueError("max_health must be positive")
        for name in (
            "preprocessing_version",
            "observation_schema_version",
            "action_space_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "feature_schema", _json_ready(self.feature_schema))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_history_length": self.model_history_length,
            "max_episode_steps": self.max_episode_steps,
            "max_health": self.max_health,
            "preprocessing_version": self.preprocessing_version,
            "observation_schema_version": self.observation_schema_version,
            "action_space_version": self.action_space_version,
            "feature_schema": _json_ready(self.feature_schema),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationPreprocessorConfig":
        if not isinstance(value, Mapping):
            raise TypeError("preprocessor config must be a mapping")
        return cls(**dict(value))

    @classmethod
    def from_json(cls, value: str) -> "ObservationPreprocessorConfig":
        if not isinstance(value, str):
            raise TypeError("preprocessor config JSON must be a string")
        return cls.from_dict(json.loads(value))


# Short name used by the design document and likely artifact consumers.
PreprocessorConfig = ObservationPreprocessorConfig


@dataclass(frozen=True, slots=True)
class ProcessedObservation:
    """Fixed-shape tensors consumed by the MainAgent inference modules."""

    state_vector: np.ndarray
    player_state_matrix: np.ndarray
    role_beliefs: np.ndarray
    event_history: np.ndarray
    event_valid_mask: np.ndarray
    action_mask: np.ndarray
    metadata: Mapping[str, Any]

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "state_vector",
        "player_state_matrix",
        "role_beliefs",
        "event_history",
        "event_valid_mask",
        "action_mask",
        "metadata",
    )

    def __post_init__(self) -> None:
        expected = {
            "state_vector": ((STATE_VECTOR_SIZE,), np.float32),
            "player_state_matrix": ((PLAYER_COUNT, PLAYER_FEATURE_SIZE), np.float32),
            "role_beliefs": ((PLAYER_COUNT, len(Role)), np.float32),
            "event_history": (
                (OTHER_PLAYER_COUNT, MODEL_HISTORY_LENGTH, EVENT_FEATURE_SIZE),
                np.float32,
            ),
            "event_valid_mask": ((OTHER_PLAYER_COUNT, MODEL_HISTORY_LENGTH), np.bool_),
            "action_mask": ((ACTION_SPACE_SIZE,), np.uint8),
        }
        for name, (shape, dtype) in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if np.any((self.action_mask != 0) & (self.action_mask != 1)):
            raise ValueError("action_mask must contain only 0 or 1")
        if np.any(self.role_beliefs < 0.0) or np.any(self.role_beliefs > 1.0):
            raise ValueError("role_beliefs must be in [0, 1]")

    @property
    def action_history(self) -> np.ndarray:
        """Compatibility alias for the model's actor-separated event input."""

        return self.event_history

    @property
    def valid_mask(self) -> np.ndarray:
        return self.event_valid_mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_vector": self.state_vector.tolist(),
            "player_state_matrix": self.player_state_matrix.tolist(),
            "role_beliefs": self.role_beliefs.tolist(),
            "event_history": self.event_history.tolist(),
            "event_valid_mask": self.event_valid_mask.tolist(),
            "action_mask": self.action_mask.tolist(),
            "metadata": _json_ready(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def __getitem__(self, key: str) -> Any:
        if key == "action_history":
            key = "event_history"
        if key == "valid_mask":
            key = "event_valid_mask"
        if key not in self.FIELD_NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def keys(self):
        return self.FIELD_NAMES

    def items(self):
        return ((key, self[key]) for key in self.FIELD_NAMES)


class ObservationPreprocessor:
    """Validate and convert one or a batch of canonical public observations."""

    def __init__(self, config: ObservationPreprocessorConfig | Mapping[str, Any] | None = None) -> None:
        if config is None:
            config = ObservationPreprocessorConfig()
        elif isinstance(config, Mapping):
            config = ObservationPreprocessorConfig.from_dict(config)
        if not isinstance(config, ObservationPreprocessorConfig):
            raise TypeError("config must be ObservationPreprocessorConfig or a mapping")
        self.config = config

    @property
    def input_dim(self) -> int:
        return EVENT_FEATURE_SIZE

    @property
    def output_schema(self) -> Mapping[str, Any]:
        return _json_ready(FEATURE_SCHEMA)

    def metadata(self) -> dict[str, Any]:
        return {
            "preprocessing_version": self.config.preprocessing_version,
            "observation_schema_version": self.config.observation_schema_version,
            "action_space_version": self.config.action_space_version,
            "model_history_length": self.config.model_history_length,
            "feature_schema": self.output_schema,
            "config": self.config.to_dict(),
        }

    def serialize_config(self) -> str:
        return self.config.to_json()

    def preprocess(self, observation: Any) -> ProcessedObservation | dict[str, Any]:
        """Process one observation, or a sequence into stacked batch tensors."""

        if self._is_observation_mapping(observation):
            return self._process_one(observation)
        if isinstance(observation, Sequence) and not isinstance(observation, (str, bytes, bytearray)):
            return self.preprocess_batch(observation)
        raise TypeError("observation must be a canonical mapping or a sequence of mappings")

    transform = preprocess
    process = preprocess

    def preprocess_batch(self, observations: Sequence[Any]) -> dict[str, Any]:
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes, bytearray)):
            raise TypeError("observations must be a sequence")
        processed = [self._process_one(item) for item in observations]
        if not processed:
            return self._empty_batch()
        return {
            "state_vector": np.stack([item.state_vector for item in processed]).astype(np.float32),
            "player_state_matrix": np.stack([item.player_state_matrix for item in processed]).astype(np.float32),
            "role_beliefs": np.stack([item.role_beliefs for item in processed]).astype(np.float32),
            "event_history": np.stack([item.event_history for item in processed]).astype(np.float32),
            "event_valid_mask": np.stack([item.event_valid_mask for item in processed]).astype(np.bool_),
            "action_mask": np.stack([item.action_mask for item in processed]).astype(np.uint8),
            "metadata": self.metadata(),
        }

    def _empty_batch(self) -> dict[str, Any]:
        return {
            "state_vector": np.empty((0, STATE_VECTOR_SIZE), dtype=np.float32),
            "player_state_matrix": np.empty((0, PLAYER_COUNT, PLAYER_FEATURE_SIZE), dtype=np.float32),
            "role_beliefs": np.empty((0, PLAYER_COUNT, len(Role)), dtype=np.float32),
            "event_history": np.empty(
                (0, OTHER_PLAYER_COUNT, self.config.model_history_length, EVENT_FEATURE_SIZE),
                dtype=np.float32,
            ),
            "event_valid_mask": np.empty(
                (0, OTHER_PLAYER_COUNT, self.config.model_history_length), dtype=np.bool_
            ),
            "action_mask": np.empty((0, ACTION_SPACE_SIZE), dtype=np.uint8),
            "metadata": self.metadata(),
        }

    def _process_one(self, observation: Any) -> ProcessedObservation:
        values = self._mapping(observation)
        self._reject_privileged(values)
        self._validate_required(values)
        input_schema_version = values.get("observation_schema_version")
        if input_schema_version is not None and input_schema_version != self.config.observation_schema_version:
            raise ValueError(
                "observation_schema_version does not match the preprocessor configuration"
            )
        current_player = _scalar(values["current_player_idx"], "current_player_idx")
        viewer_idx = _scalar(values.get("viewer_idx", current_player), "viewer_idx")
        environment_step = _scalar(values["environment_step"], "environment_step")
        round_index = _scalar(values["round_index"], "round_index")
        sheriff_idx = _scalar(values["sheriff_idx"], "sheriff_idx")
        self_role = _scalar(values["self_role"], "self_role")
        for name, item, upper in (
            ("current_player_idx", current_player, PLAYER_COUNT - 1),
            ("viewer_idx", viewer_idx, PLAYER_COUNT - 1),
            ("sheriff_idx", sheriff_idx, PLAYER_COUNT - 1),
            ("self_role", self_role, len(Role) - 1),
        ):
            if not 0 <= item <= upper:
                raise ValueError(f"{name} is out of range")
        if environment_step < 0 or round_index < 0:
            raise ValueError("environment_step and round_index must be non-negative")

        healths = _integer_array(values["healths"], (PLAYER_COUNT,), "healths")
        alive = _integer_array(values["alive"], (PLAYER_COUNT,), "alive", allow_bool=True)
        turn_order = _integer_array(values["turn_order"], (PLAYER_COUNT,), "turn_order")
        visibility = _integer_array(values["role_visibility"], (PLAYER_COUNT,), "role_visibility")
        revealed = _integer_array(values["revealed_roles"], (PLAYER_COUNT,), "revealed_roles")
        role_beliefs = _float_array(
            values["confirmed_role_distribution"], (PLAYER_COUNT, len(Role)), "confirmed_role_distribution"
        )
        action_mask = _integer_array(values["action_mask"], (ACTION_SPACE_SIZE,), "action_mask")

        if np.any((healths < 0) | (healths > self.config.max_health)):
            raise ValueError(f"healths must be in [0, {self.config.max_health}]")
        if np.any((alive < 0) | (alive > 1)):
            raise ValueError("alive must contain only 0 or 1")
        if not np.array_equal(alive, healths > 0):
            raise ValueError("alive must agree with healths > 0")
        if not np.array_equal(turn_order, np.arange(PLAYER_COUNT)):
            raise ValueError("turn_order must be canonical player order")
        if np.any((visibility < 0) | (visibility > 3)):
            raise ValueError("role_visibility must contain values in [0, 3]")
        if np.any((revealed != HIDDEN_ROLE_SENTINEL) & ((revealed < 0) | (revealed >= len(Role)))):
            raise ValueError("revealed_roles must contain role indices or 255")
        if np.any((action_mask < 0) | (action_mask > 1)):
            raise ValueError("action_mask must contain only 0 or 1")
        if not np.all(np.isfinite(role_beliefs)) or np.any((role_beliefs < 0.0) | (role_beliefs > 1.0)):
            raise ValueError("confirmed_role_distribution must be in [0, 1]")
        row_sums = role_beliefs.sum(axis=1)
        if not np.all(np.isclose(row_sums, 0.0) | np.isclose(row_sums, 1.0)):
            raise ValueError("confirmed_role_distribution rows must be zero or normalized")

        # State vector: all ordinal values are scaled to [0, 1]; step values
        # use the configured episode limit and are clipped only at inference.
        viewer_health = float(healths[viewer_idx]) / float(self.config.max_health)
        state_vector = np.asarray(
            [
                current_player / PLAYER_INDEX_MAX,
                min(environment_step / self.config.max_episode_steps, 1.0),
                min(round_index / self.config.max_episode_steps, 1.0),
                sheriff_idx / PLAYER_INDEX_MAX,
                self_role / PLAYER_INDEX_MAX,
                viewer_health,
                float(alive[current_player]),
                float(alive.sum()) / PLAYER_COUNT,
                float(values.get("viewer_idx", current_player)) / PLAYER_INDEX_MAX,
            ],
            dtype=np.float32,
        )

        # The player matrix is deliberately public: health/alive and confirmed
        # distribution only.  The visibility code is a public status, not a role.
        player_matrix = np.concatenate(
            [
                (healths.astype(np.float32) / self.config.max_health)[:, None],
                alive.astype(np.float32)[:, None],
                role_beliefs.astype(np.float32),
                (visibility.astype(np.float32) / 3.0)[:, None],
            ],
            axis=1,
        ).astype(np.float32)

        events = self._events(values)
        event_history, valid_mask = self._event_features(events, viewer_idx)
        metadata = self.metadata()
        metadata["viewer_idx"] = viewer_idx
        return ProcessedObservation(
            state_vector=state_vector,
            player_state_matrix=player_matrix,
            role_beliefs=role_beliefs.astype(np.float32),
            event_history=event_history,
            event_valid_mask=valid_mask,
            action_mask=action_mask.astype(np.uint8),
            metadata=metadata,
        )

    @staticmethod
    def _is_observation_mapping(value: Any) -> bool:
        return isinstance(value, Mapping) or (
            hasattr(value, "keys") and hasattr(value, "__getitem__")
        )

    @staticmethod
    def _mapping(observation: Any) -> Mapping[str, Any]:
        if isinstance(observation, Mapping):
            return observation
        if hasattr(observation, "to_observation_dict"):
            values = dict(observation.to_observation_dict())
            if hasattr(observation, "viewer_idx"):
                values["viewer_idx"] = observation.viewer_idx
            return values
        if hasattr(observation, "keys") and hasattr(observation, "__getitem__"):
            values = {key: observation[key] for key in observation.keys()}
            if hasattr(observation, "viewer_idx"):
                values["viewer_idx"] = observation.viewer_idx
            return values
        raise TypeError("observation must be a canonical observation mapping")

    @staticmethod
    def _reject_privileged(values: Mapping[str, Any]) -> None:
        leaked = sorted(_FORBIDDEN_OBSERVATION_KEYS.intersection(values.keys()))
        if leaked:
            raise ValueError(f"privileged observation fields are not accepted: {', '.join(leaked)}")

    @staticmethod
    def _validate_required(values: Mapping[str, Any]) -> None:
        missing = [name for name in _REQUIRED_FIELDS if name not in values]
        if missing:
            raise ValueError(f"observation is missing required fields: {', '.join(missing)}")

    @staticmethod
    def _events(values: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        raw = values.get("event_sequence", values.get("event_history", ()))
        if raw is None:
            return []
        if isinstance(raw, (str, bytes, bytearray)):
            raise TypeError("event_sequence must be an iterable of public event mappings")
        try:
            events = list(raw)
        except TypeError as exc:
            raise TypeError("event_sequence must be iterable") from exc
        normalized: list[Mapping[str, Any]] = []
        for index, event in enumerate(events):
            if hasattr(event, "to_public_dict"):
                event = event.to_public_dict()
            if not isinstance(event, Mapping):
                raise TypeError(f"event_sequence[{index}] must be a mapping")
            leaked = sorted(_FORBIDDEN_EVENT_KEYS.intersection(event.keys()))
            if leaked:
                raise ValueError(
                    f"privileged event fields are not accepted: {', '.join(leaked)}"
                )
            if event.get("public", True) is not True:
                raise ValueError("private events are not accepted by MainAgent preprocessing")
            normalized.append(event)
        return sorted(
            normalized,
            key=lambda event: (
                _event_int(event, "environment_step", nonnegative=True),
                _event_int(event, "event_id", nonnegative=True),
            ),
        )

    def _event_features(
        self, events: Sequence[Mapping[str, Any]], viewer_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        history = np.zeros(
            (OTHER_PLAYER_COUNT, self.config.model_history_length, EVENT_FEATURE_SIZE),
            dtype=np.float32,
        )
        valid = np.zeros(
            (OTHER_PLAYER_COUNT, self.config.model_history_length), dtype=np.bool_
        )
        other_players = tuple(index for index in range(PLAYER_COUNT) if index != viewer_idx)
        player_events: dict[int, list[Mapping[str, Any]]] = {index: [] for index in other_players}
        for event in events:
            actor = _optional_player(event.get("actor_idx"), "actor_idx")
            if actor is None:
                # Elimination/reveal events have no actor.  Attribute them to
                # their public subject so they remain in the correct stream.
                actor = _optional_player(
                    event.get("eliminated_idx", event.get("target_idx")), "event subject"
                )
            if actor is None:
                raise ValueError("event must have actor_idx or a public event subject")
            if actor == viewer_idx or actor not in player_events:
                continue
            player_events[actor].append(event)

        for stream, player_idx in enumerate(other_players):
            selected = player_events[player_idx][-self.config.model_history_length :]
            start = self.config.model_history_length - len(selected)
            for offset, event in enumerate(selected):
                history[stream, start + offset] = self._encode_event(event)
                valid[stream, start + offset] = True
        return history, valid

    def _encode_event(self, event: Mapping[str, Any]) -> np.ndarray:
        event_type = event.get("event_type")
        event_names = ("pass", "bang", "player_eliminated", "role_revealed")
        type_index = event_names.index(str(event_type)) if str(event_type) in event_names else None
        if type_index is None:
            raise ValueError("event_type must be one of pass, bang, player_eliminated, role_revealed")
        type_one_hot = np.zeros(4, dtype=np.float32)
        if type_index is not None:
            type_one_hot[type_index] = 1.0
        target = _optional_player(event.get("target_idx"), "target_idx")
        eliminated = _optional_player(event.get("eliminated_idx"), "eliminated_idx")
        revealed = event.get("revealed_role")
        if revealed is None or revealed == HIDDEN_ROLE_SENTINEL or revealed == PASS_TARGET_SENTINEL:
            revealed_index = None
        else:
            if not _is_integer(revealed) or not 0 <= int(revealed) < len(Role):
                raise ValueError("revealed_role must be null, 255, or a role index")
            revealed_index = int(revealed)
        role_one_hot = np.zeros(len(Role), dtype=np.float32)
        if revealed_index is not None:
            role_one_hot[revealed_index] = 1.0
        hp_before = _optional_bounded_int(event.get("hp_before"), "hp_before", self.config.max_health)
        hp_after = _optional_bounded_int(event.get("hp_after"), "hp_after", self.config.max_health)
        before = 0.0 if hp_before is None else hp_before / self.config.max_health
        after = 0.0 if hp_after is None else hp_after / self.config.max_health
        delta = 0.0 if hp_before is None or hp_after is None else (hp_after - hp_before) / self.config.max_health
        event_step = _event_int(event, "environment_step", nonnegative=True)
        event_id = _event_int(event, "event_id", nonnegative=True)
        return np.asarray(
            [
                *type_one_hot,
                0.0 if target is None else target / PLAYER_INDEX_MAX,
                float(target is not None),
                before,
                after,
                delta,
                min(event_step / self.config.max_episode_steps, 1.0),
                min(event_id / self.config.max_episode_steps, 1.0),
                float(eliminated is not None),
                *role_one_hot,
            ],
            dtype=np.float32,
        )


def _event_int(event: Mapping[str, Any], name: str, *, nonnegative: bool = False) -> int:
    value = event.get(name)
    if not _is_integer(value):
        raise TypeError(f"event {name} must be an integer")
    value = int(value)
    if nonnegative and value < 0:
        raise ValueError(f"event {name} must be non-negative")
    return value


def _optional_player(value: Any, name: str) -> int | None:
    if value is None or value == PASS_TARGET_SENTINEL:
        return None
    if not _is_integer(value):
        raise TypeError(f"{name} must be null, 255, or a player index")
    value = int(value)
    if not 0 <= value < PLAYER_COUNT:
        raise ValueError(f"{name} must be null, 255, or a player index")
    return value


def _optional_bounded_int(value: Any, name: str, maximum: int) -> int | None:
    if value is None:
        return None
    if not _is_integer(value):
        raise TypeError(f"event {name} must be null or an integer")
    value = int(value)
    if not 0 <= value <= maximum:
        raise ValueError(f"event {name} must be in [0, {maximum}]")
    return value


# Public aliases preserve the design/task vocabulary.
ObsPreprocessor = ObservationPreprocessor
ProcessedObs = ProcessedObservation

__all__ = [
    "ACTION_SPACE_VERSION",
    "EVENT_FEATURE_SIZE",
    "FEATURE_SCHEMA",
    "MODEL_HISTORY_LENGTH",
    "ObservationPreprocessor",
    "ObservationPreprocessorConfig",
    "ObsPreprocessor",
    "PreprocessorConfig",
    "ProcessedObservation",
    "ProcessedObs",
    "PLAYER_FEATURE_SIZE",
    "STATE_VECTOR_SIZE",
]
