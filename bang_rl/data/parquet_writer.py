"""Canonical Parquet shard persistence for Phase 1 episode records.

The writer is intentionally episode oriented.  A canonical event is written
once in the episode ``events`` list; step rows refer to those events by ID.
This keeps the source dataset lossless without copying the cumulative event
history into every transition.  No tensor conversion belongs in this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
import json
import os
from pathlib import Path
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from bang_rl.config import DatasetConfig, SystemConfig
from bang_rl.exceptions import DataError

from .collector import (
    ACTION_SPACE_VERSION,
    CANONICAL_DATASET_SCHEMA_VERSION,
    CanonicalEpisode,
    CanonicalStep,
)


PARQUET_WRITER_SCHEMA_VERSION = "1.0"
_SHARD_RE = re.compile(r"^shard-(?P<index>\d{6})\.parquet$")
_TMP_SUFFIX = ".parquet.tmp"
_PRIVATE_KEYS = {
    "roles",
    "true_roles",
    "actual_roles",
    "hidden_roles",
    "private_state",
    "privileged_state",
}


def _nullable(kind: pa.DataType) -> pa.Field:
    return pa.field("value", kind, nullable=True)


# Fixed-size arrays are represented as Arrow fixed-size lists, not opaque JSON
# or NumPy blobs.  This is the stable on-disk contract for canonical data.
_OBSERVATION = pa.struct(
    [
        pa.field("current_player_idx", pa.uint8(), nullable=True),
        pa.field("environment_step", pa.uint32(), nullable=True),
        pa.field("round_index", pa.uint32(), nullable=True),
        pa.field("healths", pa.list_(pa.uint8()), nullable=True),
        pa.field("alive", pa.list_(pa.bool_()), nullable=True),
        pa.field("turn_order", pa.list_(pa.uint8()), nullable=True),
        pa.field("sheriff_idx", pa.uint8(), nullable=True),
        pa.field("self_role", pa.uint8(), nullable=True),
        pa.field("role_visibility", pa.list_(pa.uint8()), nullable=True),
        pa.field("revealed_roles", pa.list_(pa.uint8()), nullable=True),
        pa.field("confirmed_role_distribution", pa.list_(pa.list_(pa.float32())), nullable=True),
        pa.field("action_mask", pa.list_(pa.uint8(), 6), nullable=True),
        pa.field("observation_schema_version", pa.string(), nullable=True),
        # Unknown extension fields remain lossless, while the fields above keep
        # their fixed types and remain directly queryable.
        pa.field("extra_json", pa.string(), nullable=True),
    ]
)

_EVENT = pa.struct(
    [
        pa.field("event_id", pa.uint32(), nullable=False),
        pa.field("environment_step", pa.uint32(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("actor_idx", pa.uint8(), nullable=True),
        pa.field("target_idx", pa.uint8(), nullable=True),
        pa.field("hp_before", pa.uint8(), nullable=True),
        pa.field("hp_after", pa.uint8(), nullable=True),
        pa.field("eliminated_idx", pa.uint8(), nullable=True),
        pa.field("revealed_role", pa.uint8(), nullable=True),
        pa.field("public", pa.bool_(), nullable=False),
    ]
)

_LABEL = pa.struct(
    [
        pa.field("episode_id", pa.string(), nullable=True),
        pa.field("environment_step", pa.uint32(), nullable=False),
        pa.field("player_idx", pa.uint8(), nullable=False),
        pa.field("values", pa.list_(pa.float32()), nullable=True),
        pa.field("emotion_names", pa.list_(pa.string()), nullable=True),
        pa.field("emotion_schema_version", pa.string(), nullable=True),
        pa.field("input_event_ids", pa.list_(pa.uint32()), nullable=True),
        pa.field("public_observation", _OBSERVATION, nullable=True),
    ]
)

_STEP = pa.struct(
    [
        pa.field("environment_step", pa.uint32(), nullable=False),
        pa.field("actor_idx", pa.uint8(), nullable=False),
        pa.field("action", pa.uint8(), nullable=False),
        pa.field("action_type", pa.string(), nullable=False),
        pa.field("target_idx", pa.uint8(), nullable=True),
        pa.field("action_mask", pa.list_(pa.uint8(), 6), nullable=False),
        pa.field("pre_observation", _OBSERVATION, nullable=False),
        pa.field("post_observation", _OBSERVATION, nullable=True),
        pa.field("pre_event_ids", pa.list_(pa.uint32()), nullable=False),
        pa.field("event_ids", pa.list_(pa.uint32()), nullable=False),
        pa.field("hp_before", pa.uint8(), nullable=True),
        pa.field("hp_after", pa.uint8(), nullable=True),
        pa.field("eliminated_idx", pa.uint8(), nullable=True),
        pa.field("reward", pa.float32(), nullable=False),
        # Reward values are labels/diagnostics, not policy input.  The value
        # JSON is nested under a typed player entry to retain arbitrary DSL
        # components without making the Parquet schema dynamic.
        pa.field(
            "reward_breakdowns",
            pa.list_(pa.struct([
                pa.field("player_idx", pa.uint8(), nullable=False),
                pa.field("value_json", pa.string(), nullable=False),
            ])),
            nullable=False,
        ),
        pa.field("emotion_labels", pa.list_(pa.struct([
            pa.field("player_idx", pa.uint8(), nullable=False),
            pa.field("label", _LABEL, nullable=False),
        ])), nullable=False),
        pa.field("terminated", pa.bool_(), nullable=False),
        pa.field("truncated", pa.bool_(), nullable=False),
    ]
)

_TERMINAL = pa.struct(
    [
        pa.field("terminated", pa.bool_(), nullable=False),
        pa.field("winners", pa.list_(pa.uint8()), nullable=False),
        pa.field("losers", pa.list_(pa.uint8()), nullable=False),
        pa.field("winning_team", pa.uint8(), nullable=True),
        pa.field("reason", pa.string(), nullable=False),
    ]
)

_INVALID = pa.struct(
    [
        pa.field("attempt_index", pa.uint32(), nullable=False),
        pa.field("environment_step", pa.uint32(), nullable=False),
        pa.field("actor_idx", pa.uint8(), nullable=True),
        pa.field("action_json", pa.string(), nullable=False),
        pa.field("action_mask", pa.list_(pa.uint8(), 6), nullable=False),
        pa.field("reason", pa.string(), nullable=True),
    ]
)


CANONICAL_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("dataset_schema_version", pa.string(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("termination_status", pa.string(), nullable=False),
        pa.field("termination_reason", pa.string(), nullable=True),
        pa.field("environment_version", pa.string(), nullable=True),
        pa.field("observation_schema_version", pa.string(), nullable=True),
        pa.field("action_space_version", pa.string(), nullable=True),
        pa.field("emotion_schema_version", pa.string(), nullable=True),
        pa.field("reward_config_version", pa.string(), nullable=True),
        pa.field("reward_config_id", pa.string(), nullable=True),
        pa.field("preset_version", pa.string(), nullable=True),
        pa.field("job_id", pa.string(), nullable=True),
        pa.field("agent_order", pa.list_(pa.uint8()), nullable=False),
        pa.field("initial_state", _OBSERVATION, nullable=False),
        pa.field("events", pa.list_(_EVENT), nullable=False),
        pa.field("steps", pa.list_(_STEP), nullable=False),
        pa.field("role_labels", pa.list_(pa.uint8()), nullable=True),
        pa.field("invalid_actions", pa.list_(_INVALID), nullable=False),
        pa.field("terminal_result", _TERMINAL, nullable=True),
        pa.field("final_state", _OBSERVATION, nullable=True),
        pa.field("metadata_json", pa.string(), nullable=False),
    ],
    metadata={
        b"bang_rl.writer_schema_version": PARQUET_WRITER_SCHEMA_VERSION.encode(),
        b"bang_rl.dataset_schema_version": CANONICAL_DATASET_SCHEMA_VERSION.encode(),
        b"bang_rl.action_space_version": ACTION_SPACE_VERSION.encode(),
        b"bang_rl.compression": b"zstd",
        b"bang_rl.event_storage": b"episode_events_once_step_event_ids",
        b"bang_rl.private_roles": b"explicit_role_labels_column",
    },
)


def _json(value: Any) -> str:
    return json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe(item)
            for key, item in value.items()
            if str(key) not in _PRIVATE_KEYS and str(key) != "role"
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except (AttributeError, ValueError):
            pass
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _safe(value.value)
    return value


def _int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DataError(f"expected an integer value, got {value!r}") from exc


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _public_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_public_dict"):
        value = value.to_public_dict()
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise DataError("canonical public state must be a mapping")
    return _safe(value)


def _observation(value: Any) -> dict[str, Any]:
    source = _public_mapping(value)
    aliases = {"health": "healths", "fixed_turn_order": "turn_order"}
    normalized = {aliases.get(key, key): item for key, item in source.items()}
    known = {
        "current_player_idx", "environment_step", "round_index", "healths", "alive",
        "turn_order", "sheriff_idx", "self_role", "role_visibility", "revealed_roles",
        "confirmed_role_distribution", "action_mask", "observation_schema_version",
    }
    result: dict[str, Any] = {
        "current_player_idx": _int(normalized.get("current_player_idx")),
        "environment_step": _int(normalized.get("environment_step")),
        "round_index": _int(normalized.get("round_index")),
        "healths": [int(item) for item in _list(normalized.get("healths"))] or None,
        "alive": [bool(item) for item in _list(normalized.get("alive"))] or None,
        "turn_order": [int(item) for item in _list(normalized.get("turn_order"))] or None,
        "sheriff_idx": _int(normalized.get("sheriff_idx")),
        "self_role": _int(normalized.get("self_role")),
        "role_visibility": [int(item) for item in _list(normalized.get("role_visibility"))] or None,
        "revealed_roles": [int(item) for item in _list(normalized.get("revealed_roles"))] or None,
        "confirmed_role_distribution": (
            [[float(item) for item in _list(row)] for row in _list(normalized.get("confirmed_role_distribution"))]
            or None
        ),
        # PyArrow 19 cannot round-trip a null nested fixed-size list: it writes
        # the null value as a zero-length list and then rejects it on read.
        # Keep the fixed six-slot on-disk shape by encoding an absent mask as
        # six nullable elements; _from_observation removes this storage-only
        # placeholder when restoring the sparse canonical observation.
        "action_mask": (
            [None] * 6
            if normalized.get("action_mask") is None
            else [int(item) for item in _list(normalized.get("action_mask"))]
        ),
        "observation_schema_version": (
            None if normalized.get("observation_schema_version") is None
            else str(normalized["observation_schema_version"])
        ),
        "extra_json": _json({
            key: item for key, item in normalized.items()
            if key not in known and key not in {"event_sequence", "event_history", "event_log", "events"}
        }) if any(key not in known for key in normalized if key not in {"event_sequence", "event_history", "event_log", "events"}) else None,
    }
    # Event sequences are deliberately absent from the struct and extra JSON.
    if result["action_mask"] is not None and len(result["action_mask"]) != 6:
        raise DataError("observation action_mask must contain exactly six values")
    return result


def _event(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("public", True) is False:
        raise DataError("private events cannot enter the canonical public event stream")
    return {
        "event_id": _required_uint(value.get("event_id"), "event_id", 32),
        "environment_step": _required_uint(value.get("environment_step"), "environment_step", 32),
        "event_type": str(value.get("event_type", "")),
        "actor_idx": _uint_or_none(value.get("actor_idx"), 8),
        "target_idx": _uint_or_none(value.get("target_idx"), 8),
        "hp_before": _uint_or_none(value.get("hp_before"), 8),
        "hp_after": _uint_or_none(value.get("hp_after"), 8),
        "eliminated_idx": _uint_or_none(value.get("eliminated_idx"), 8),
        "revealed_role": _uint_or_none(value.get("revealed_role"), 8),
        "public": bool(value.get("public", True)),
    }


def _required_uint(value: Any, name: str, bits: int) -> int:
    converted = _uint_or_none(value, bits)
    if converted is None:
        raise DataError(f"{name} is required")
    return converted


def _uint_or_none(value: Any, bits: int) -> int | None:
    if value is None:
        return None
    converted = _int(value)
    if converted is None or converted < 0 or converted >= 2**bits:
        raise DataError(f"integer {value!r} does not fit uint{bits}")
    return converted


def _ids(value: Iterable[Any] | None, name: str) -> list[int]:
    return [_required_uint(item, name, 32) for item in (value or ())]


def _label(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": None if value.get("episode_id") is None else str(value["episode_id"]),
        "environment_step": _required_uint(value.get("environment_step"), "label.environment_step", 32),
        "player_idx": _required_uint(value.get("player_idx"), "label.player_idx", 8),
        "values": [float(item) for item in _list(value.get("values", value.get("emotion_values")))],
        "emotion_names": [str(item) for item in _list(value.get("emotion_names"))],
        "emotion_schema_version": (
            None if value.get("emotion_schema_version") is None else str(value["emotion_schema_version"])
        ),
        "input_event_ids": _ids(value.get("input_event_ids", ()), "label.input_event_ids"),
        "public_observation": _observation(value.get("public_observation")) if value.get("public_observation") else None,
    }


def _indexed_json(value: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not value:
        return []
    result = []
    for key, item in sorted(value.items(), key=lambda pair: int(pair[0])):
        result.append({"player_idx": _required_uint(key, "player_idx", 8), "value_json": _json(item)})
    return result


def _labels(value: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not value:
        return []
    result = []
    for key, item in sorted(value.items(), key=lambda pair: int(pair[0])):
        result.append({"player_idx": _required_uint(key, "player_idx", 8), "label": _label(item)})
    return result


def _step(step: CanonicalStep) -> dict[str, Any]:
    mask = [int(item) for item in step.action_mask]
    if len(mask) != 6 or any(item not in (0, 1) for item in mask):
        raise DataError("step action_mask must contain exactly six binary values")
    return {
        "environment_step": _required_uint(step.environment_step, "step.environment_step", 32),
        "actor_idx": _required_uint(step.actor_idx, "step.actor_idx", 8),
        "action": _required_uint(step.action, "step.action", 8),
        "action_type": str(step.action_type),
        "target_idx": _uint_or_none(step.target_idx, 8),
        "action_mask": mask,
        "pre_observation": _observation(step.pre_observation),
        "post_observation": None if step.post_observation is None else _observation(step.post_observation),
        "pre_event_ids": _ids(step.pre_event_ids, "step.pre_event_ids"),
        "event_ids": _ids(step.event_ids, "step.event_ids"),
        "hp_before": _uint_or_none(step.hp_before, 8),
        "hp_after": _uint_or_none(step.hp_after, 8),
        "eliminated_idx": _uint_or_none(step.eliminated_idx, 8),
        "reward": float(step.reward),
        "reward_breakdowns": _indexed_json(step.reward_breakdowns),
        "emotion_labels": _labels(step.emotion_labels),
        "terminated": bool(step.terminated),
        "truncated": bool(step.truncated),
    }


def _terminal(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "terminated": bool(value.get("terminated", True)),
        "winners": [_required_uint(item, "winner", 8) for item in value.get("winners", ())],
        "losers": [_required_uint(item, "loser", 8) for item in value.get("losers", ())],
        "winning_team": _uint_or_none(value.get("winning_team"), 8),
        "reason": str(value.get("reason", "")),
    }


def _invalid(value: Mapping[str, Any]) -> dict[str, Any]:
    mask = [int(item) for item in _list(value.get("action_mask"))]
    if len(mask) != 6:
        raise DataError("invalid action action_mask must contain exactly six values")
    return {
        "attempt_index": _required_uint(value.get("attempt_index", 0), "attempt_index", 32),
        "environment_step": _required_uint(value.get("environment_step", 0), "environment_step", 32),
        "actor_idx": _uint_or_none(value.get("actor_idx"), 8),
        "action_json": _json(value.get("action")),
        "action_mask": mask,
        "reason": None if value.get("reason") is None else str(value["reason"]),
    }


def _row(episode: CanonicalEpisode, *, include_private_roles: bool) -> dict[str, Any]:
    metadata = _safe(dict(episode.metadata))
    events = sorted((_event(dict(item)) for item in episode.events), key=lambda item: (item["event_id"], item["environment_step"]))
    if len({item["event_id"] for item in events}) != len(events):
        raise DataError(f"episode {episode.episode_id!r} contains duplicate event IDs")
    steps = [_step(item) for item in episode.steps]
    if [item["environment_step"] for item in steps] != sorted(item["environment_step"] for item in steps):
        raise DataError(f"episode {episode.episode_id!r} steps are not ordered by environment_step")
    status = str(getattr(episode.termination_status, "value", episode.termination_status))
    if status not in {"normal", "truncated"}:
        raise DataError(f"episode {episode.episode_id!r} has invalid termination_status {status!r}")
    dataset_version = str(metadata.get("dataset_schema_version", CANONICAL_DATASET_SCHEMA_VERSION))
    if dataset_version != CANONICAL_DATASET_SCHEMA_VERSION:
        raise DataError(
            f"episode {episode.episode_id!r} has unsupported dataset_schema_version {dataset_version!r}"
        )
    return {
        "dataset_schema_version": dataset_version,
        "episode_id": str(episode.episode_id),
        "seed": int(episode.seed),
        "termination_status": status,
        "termination_reason": None if episode.termination_reason is None else str(episode.termination_reason),
        "environment_version": _metadata_str(metadata, "environment_version"),
        "observation_schema_version": _metadata_str(metadata, "observation_schema_version"),
        "action_space_version": _metadata_str(metadata, "action_space_version", ACTION_SPACE_VERSION),
        "emotion_schema_version": _metadata_str(metadata, "emotion_schema_version"),
        "reward_config_version": _metadata_str(metadata, "reward_config_version"),
        "reward_config_id": _metadata_str(metadata, "reward_config_id"),
        "preset_version": _metadata_str(metadata, "preset_version"),
        "job_id": _metadata_str(metadata, "job_id"),
        "agent_order": [_required_uint(item, "agent_order", 8) for item in episode.agent_order],
        "initial_state": _observation(episode.initial_state),
        "events": events,
        "steps": steps,
        "role_labels": (
            None if not include_private_roles or episode.role_labels is None
            else [_required_uint(item, "role_label", 8) for item in episode.role_labels]
        ),
        "invalid_actions": [_invalid(dict(item)) for item in episode.invalid_actions],
        "terminal_result": _terminal(episode.terminal_result),
        "final_state": None if episode.final_state is None else _observation(episode.final_state),
        "metadata_json": _json(metadata),
    }


def _metadata_str(metadata: Mapping[str, Any], key: str, default: str | None = None) -> str | None:
    value = metadata.get(key, default)
    return None if value is None else str(value)


def _table(episodes: Iterable[CanonicalEpisode], *, include_private_roles: bool) -> pa.Table:
    rows = [_row(episode, include_private_roles=include_private_roles) for episode in episodes]
    table = pa.Table.from_pylist(rows, schema=CANONICAL_PARQUET_SCHEMA)
    metadata = dict(CANONICAL_PARQUET_SCHEMA.metadata or {})
    metadata[b"bang_rl.contains_private_role_labels"] = (
        b"true" if include_private_roles else b"false"
    )
    return table.replace_schema_metadata(metadata)


def _serialized(table: pa.Table, *, row_group_size: int) -> bytes:
    import io

    target = io.BytesIO()
    pq.write_table(
        table,
        target,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=max(1, int(row_group_size)),
        version="2.6",
        data_page_version="1.0",
    )
    return target.getvalue()


class ParquetCanonicalShardWriter:
    """Append-only canonical shard writer with atomic commit and recovery.

    ``write_episode``/``append`` accept one completed :class:`CanonicalEpisode`
    at a time.  ``flush`` commits the current shard; ``finalize`` flushes and
    closes the writer.  Existing valid shards are treated as committed and a
    resumed writer continues at the next shard number, so it never mutates a
    previously committed file or silently overwrites an episode.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str] | DatasetConfig | SystemConfig | None = None,
        config: DatasetConfig | SystemConfig | None = None,
        *,
        dataset_config: DatasetConfig | None = None,
        include_private_roles: bool = True,
        row_group_max_episodes: int | None = None,
        recover: bool = True,
    ) -> None:
        if isinstance(output_dir, (DatasetConfig, SystemConfig)):
            if config is not None or dataset_config is not None:
                raise TypeError("configuration was supplied more than once")
            config = output_dir
            output_dir = None
        resolved = dataset_config or config
        if isinstance(resolved, SystemConfig):
            resolved = resolved.dataset
        if resolved is None:
            resolved = DatasetConfig()
        if not isinstance(resolved, DatasetConfig):
            raise TypeError("config must be DatasetConfig or SystemConfig")
        self.config = resolved
        self.output_dir = Path(output_dir or resolved.canonical_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_episodes = int(resolved.canonical_shard_max_episodes)
        self.max_bytes = int(resolved.canonical_shard_max_bytes)
        self.include_private_roles = bool(include_private_roles)
        self.row_group_max_episodes = int(row_group_max_episodes or self.max_episodes)
        if self.row_group_max_episodes < 1:
            raise ValueError("row_group_max_episodes must be positive")
        self._closed = False
        self._pending: list[CanonicalEpisode] = []
        self._pending_serialized: bytes | None = None
        self._episode_ids: set[str] = set()
        self._unreported_paths: list[Path] = []
        self._next_index = 0
        if recover:
            self._recover_temps()
        self._load_existing_shards()

    @property
    def pending_episodes(self) -> tuple[CanonicalEpisode, ...]:
        return tuple(self._pending)

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self.output_dir.glob("shard-*.parquet")))

    def _recover_temps(self) -> None:
        for temp in sorted(self.output_dir.glob(f"shard-*{_TMP_SUFFIX}")):
            final = Path(str(temp)[: -len(".tmp")])
            if final.exists():
                temp.unlink()
                continue
            try:
                table = pq.read_table(temp)
                self._validate_table(table, temp)
            except Exception as exc:
                raise DataError(f"orphan temporary shard is corrupt: {temp}") from exc
            os.replace(temp, final)
            _fsync_directory(self.output_dir)

    def _load_existing_shards(self) -> None:
        paths = sorted(self.output_dir.glob("shard-*.parquet"))
        for path in paths:
            match = _SHARD_RE.match(path.name)
            if match is None:
                raise DataError(f"invalid canonical shard filename: {path.name}")
            index = int(match.group("index"))
            self._next_index = max(self._next_index, index + 1)
            try:
                table = pq.read_table(path)
                self._validate_table(table, path)
                ids = table.column("episode_id").to_pylist()
            except DataError:
                raise
            except Exception as exc:
                raise DataError(f"cannot read canonical shard {path}") from exc
            for episode_id in ids:
                if episode_id in self._episode_ids:
                    raise DataError(f"duplicate episode_id across canonical shards: {episode_id!r}")
                self._episode_ids.add(str(episode_id))

    @staticmethod
    def _validate_table(table: pa.Table, path: Path) -> None:
        if table.schema.remove_metadata() != CANONICAL_PARQUET_SCHEMA.remove_metadata():
            raise DataError(f"canonical shard schema mismatch: {path}")
        if table.num_rows < 1:
            raise DataError(f"canonical shard contains no episode rows: {path}")
        metadata = table.schema.metadata or {}
        for key, expected in ((b"bang_rl.writer_schema_version", PARQUET_WRITER_SCHEMA_VERSION),
                              (b"bang_rl.dataset_schema_version", CANONICAL_DATASET_SCHEMA_VERSION)):
            if metadata.get(key) != expected.encode():
                raise DataError(f"canonical shard metadata mismatch for {key.decode()}: {path}")

    def _ensure_open(self) -> None:
        if self._closed:
            raise DataError("canonical shard writer has been finalized")

    def write_episode(self, episode: CanonicalEpisode) -> None:
        self._ensure_open()
        if not isinstance(episode, CanonicalEpisode):
            raise TypeError("write_episode expects CanonicalEpisode")
        detached = CanonicalEpisode.from_dict(copy.deepcopy(episode.to_dict(include_private=True)))
        if detached.episode_id in self._episode_ids or any(item.episode_id == detached.episode_id for item in self._pending):
            raise DataError(f"duplicate episode_id: {detached.episode_id!r}")
        candidate = self._pending + [detached]
        if self._pending and len(candidate) > self.max_episodes:
            self._commit_pending()
            candidate = [detached]
        pending_serialized: bytes | None = None
        if self._pending:
            # Measure the actual compressed candidate.  Retain those bytes when
            # it fits so a later rollover/flush can commit the same payload
            # without repeating Arrow/Zstandard serialization.
            pending_serialized = _serialized(
                _table(candidate, include_private_roles=self.include_private_roles),
                row_group_size=self.row_group_max_episodes,
            )
            if len(pending_serialized) > self.max_bytes:
                self._commit_pending()
                candidate = [detached]
                pending_serialized = None
        self._pending = candidate
        self._pending_serialized = pending_serialized

    append = write_episode
    write = write_episode

    def _commit_pending(self, serialized: bytes | None = None) -> Path:
        """Atomically commit pending rows and retain the path for flush reporting."""
        episodes = tuple(self._pending)
        table = _table(episodes, include_private_roles=self.include_private_roles)
        data = (
            self._pending_serialized
            if serialized is None and self._pending_serialized is not None
            else serialized
        )
        if data is None:
            data = _serialized(table, row_group_size=min(self.row_group_max_episodes, len(episodes)))
        final = self.output_dir / f"shard-{self._next_index:06d}.parquet"
        temp = Path(str(final) + ".tmp")
        try:
            _atomic_write(temp, final, data)
        except Exception as exc:
            if temp.exists():
                temp.unlink()
            raise DataError(f"failed to atomically write canonical shard {final}") from exc
        self._next_index += 1
        self._episode_ids.update(item.episode_id for item in episodes)
        self._pending = []
        self._pending_serialized = None
        self._unreported_paths.append(final)
        _free_episode_references(episodes)
        return final

    def flush(self) -> tuple[Path, ...]:
        self._ensure_open()
        if self._pending:
            self._commit_pending()
        paths = tuple(self._unreported_paths)
        self._unreported_paths.clear()
        return paths

    def finalize(self) -> tuple[Path, ...]:
        if self._closed:
            return ()
        paths = self.flush()
        self._closed = True
        return paths

    close = finalize

    def __enter__(self) -> "ParquetCanonicalShardWriter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.finalize()

    @classmethod
    def read_shard(cls, path: str | os.PathLike[str], *, include_private_roles: bool = True) -> tuple[CanonicalEpisode, ...]:
        path = Path(path)
        try:
            table = pq.read_table(path)
            cls._validate_table(table, path)
        except DataError:
            raise
        except Exception as exc:
            raise DataError(f"cannot read canonical shard {path}") from exc
        return tuple(_episode_from_row(row, include_private_roles=include_private_roles) for row in table.to_pylist())

    @classmethod
    def read_dataset(cls, directory: str | os.PathLike[str]) -> tuple[CanonicalEpisode, ...]:
        writer = cls(directory)
        episodes: list[CanonicalEpisode] = []
        for path in writer.shard_paths:
            episodes.extend(cls.read_shard(path, include_private_roles=writer.include_private_roles))
        return tuple(episodes)


def _atomic_write(temp: Path, final: Path, data: bytes) -> None:
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, final)
    _fsync_directory(final.parent)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _free_episode_references(episodes: tuple[CanonicalEpisode, ...]) -> None:
    # The writer owns only the temporary tuple used for serialization.  Clear
    # the local reference explicitly to make the lifecycle/memory boundary
    # visible without mutating caller-owned CanonicalEpisode objects.
    del episodes


def _from_observation(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Restore the sparse canonical observation representation.

    Arrow structs materialize nullable fields even when the source observation
    omitted them.  For an otherwise empty observation this is especially
    visible for the nullable fixed-size action mask: compatible Arrow readers
    may expose its null value as ``None`` or an empty list.  Drop only those
    absent values so a Parquet round-trip remains canonically equivalent to the
    input, while preserving every populated field (including a real six-value
    mask) and the fixed on-disk schema.
    """
    if value is None:
        return None
    result = dict(value)
    extra = result.pop("extra_json", None)
    if extra:
        result.update(json.loads(extra))
    return {
        key: item
        for key, item in result.items()
        if item is not None
        and not (
            key == "action_mask"
            and isinstance(item, (list, tuple))
            and (not item or all(entry is None for entry in item))
        )
    }


def _episode_from_row(row: Mapping[str, Any], *, include_private_roles: bool) -> CanonicalEpisode:
    metadata = json.loads(row["metadata_json"])
    steps: list[CanonicalStep] = []
    for item in row["steps"]:
        reward_breakdowns = {str(entry["player_idx"]): json.loads(entry["value_json"]) for entry in item["reward_breakdowns"]}
        emotion_labels = {str(entry["player_idx"]): _label_to_dict(entry["label"]) for entry in item["emotion_labels"]}
        steps.append(CanonicalStep(
            environment_step=item["environment_step"], actor_idx=item["actor_idx"], action=item["action"],
            action_type=item["action_type"], target_idx=item["target_idx"], action_mask=tuple(item["action_mask"]),
            pre_observation=_from_observation(item["pre_observation"]) or {},
            post_observation=_from_observation(item["post_observation"]),
            pre_event_ids=tuple(item["pre_event_ids"]), event_ids=tuple(item["event_ids"]),
            hp_before=item["hp_before"], hp_after=item["hp_after"], eliminated_idx=item["eliminated_idx"],
            reward=item["reward"], reward_breakdowns=reward_breakdowns, emotion_labels=emotion_labels,
            terminated=item["terminated"], truncated=item["truncated"],
        ))
    terminal = row["terminal_result"]
    final_state = _from_observation(row["final_state"])
    roles = row["role_labels"] if include_private_roles else None
    return CanonicalEpisode(
        episode_id=row["episode_id"], seed=row["seed"], agent_order=tuple(row["agent_order"]),
        metadata=metadata, initial_state=_from_observation(row["initial_state"]) or {}, steps=tuple(steps),
        events=tuple(dict(event) for event in row["events"]), termination_status=row["termination_status"],
        termination_reason=row["termination_reason"], terminal_result=None if terminal is None else dict(terminal),
        final_state=final_state, role_labels=None if roles is None else tuple(roles),
        invalid_actions=tuple(dict(item) for item in row["invalid_actions"]),
    )


def _label_to_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": value["episode_id"], "environment_step": value["environment_step"],
        "player_idx": value["player_idx"], "values": value["values"], "emotion_names": value["emotion_names"],
        "emotion_schema_version": value["emotion_schema_version"], "input_event_ids": value["input_event_ids"],
        "public_observation": _from_observation(value["public_observation"]),
    }


# Short alias for callers that use the task wording.
ParquetShardWriter = ParquetCanonicalShardWriter

__all__ = [
    "CANONICAL_PARQUET_SCHEMA",
    "PARQUET_WRITER_SCHEMA_VERSION",
    "ParquetCanonicalShardWriter",
    "ParquetShardWriter",
]
