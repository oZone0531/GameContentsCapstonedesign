"""Deterministic, derived NPZ tensor caches for the supervised/RL phases.

Parquet remains the canonical source of truth.  This module only materializes
normal (non-truncated) episode transitions into fixed-shape tensors.  The
cache contains no raw observations or private diagnostic records; labels are
stored as targets and are never returned by the default model-input API.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np

from ..agents.main_agent.preprocessor import (
    EVENT_FEATURE_SIZE,
    MODEL_HISTORY_LENGTH,
    ObservationPreprocessor,
    ObservationPreprocessorConfig,
    OTHER_PLAYER_COUNT,
    PLAYER_FEATURE_SIZE,
    STATE_VECTOR_SIZE,
)
from ..env.types import ACTION_SPACE_SIZE, HIDDEN_ROLE_SENTINEL, PASS_TARGET_SENTINEL, PLAYER_COUNT
from ..exceptions import DataError
from .collector import CANONICAL_DATASET_SCHEMA_VERSION, CanonicalEpisode, CanonicalStep
from .splitter import EpisodeSplitResult, SplitPartition


CACHE_FORMAT_VERSION = "1.0"
CACHE_SCHEMA_VERSION = "1.0"
DEFAULT_CACHE_PREFIX = "cache"

MODEL_INPUT_KEYS = (
    "state_vector",
    "player_state_matrix",
    "role_beliefs",
    "event_history",
    "event_valid_mask",
    "action_mask",
)
TARGET_KEYS = (
    "role_labels",
    "role_label_mask",
    "emotion_labels",
    "emotion_label_mask",
)
TRANSITION_KEYS = (
    "actions",
    "actor_indices",
    "target_indices",
    "environment_steps",
    "event_id_starts",
    "event_id_ends",
    "event_counts",
    "episode_indices",
    "rewards",
    "terminated",
    "truncated",
)

# Every cache tensor is described here.  Emotion dimensions are filled at
# generation time because Emotion Schema is intentionally configurable.
FIXED_TENSOR_SCHEMA: dict[str, dict[str, Any]] = {
    "state_vector": {"dtype": "float32", "shape": [None, STATE_VECTOR_SIZE]},
    "player_state_matrix": {"dtype": "float32", "shape": [None, PLAYER_COUNT, PLAYER_FEATURE_SIZE]},
    "role_beliefs": {"dtype": "float32", "shape": [None, PLAYER_COUNT, 5]},
    "event_history": {"dtype": "float32", "shape": [None, OTHER_PLAYER_COUNT, MODEL_HISTORY_LENGTH, EVENT_FEATURE_SIZE]},
    "event_valid_mask": {"dtype": "bool", "shape": [None, OTHER_PLAYER_COUNT, MODEL_HISTORY_LENGTH]},
    "action_mask": {"dtype": "uint8", "shape": [None, ACTION_SPACE_SIZE]},
    "role_labels": {"dtype": "uint8", "shape": [None, PLAYER_COUNT]},
    "role_label_mask": {"dtype": "bool", "shape": [None, PLAYER_COUNT]},
    "emotion_labels": {"dtype": "float32", "shape": [None, PLAYER_COUNT, None]},
    "emotion_label_mask": {"dtype": "bool", "shape": [None, PLAYER_COUNT]},
    "actions": {"dtype": "uint8", "shape": [None]},
    "actor_indices": {"dtype": "uint8", "shape": [None]},
    "target_indices": {"dtype": "uint8", "shape": [None]},
    "environment_steps": {"dtype": "uint32", "shape": [None]},
    "event_id_starts": {"dtype": "uint32", "shape": [None]},
    "event_id_ends": {"dtype": "uint32", "shape": [None]},
    "event_counts": {"dtype": "uint32", "shape": [None]},
    "episode_indices": {"dtype": "uint32", "shape": [None]},
    "rewards": {"dtype": "float32", "shape": [None]},
    "terminated": {"dtype": "bool", "shape": [None]},
    "truncated": {"dtype": "bool", "shape": [None]},
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _config_payload(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return _json_safe(value)


def _hash(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_episode(value: Any) -> CanonicalEpisode:
    if isinstance(value, CanonicalEpisode):
        return value
    if isinstance(value, Mapping):
        try:
            return CanonicalEpisode.from_dict(value)
        except Exception as exc:
            raise DataError("canonical episode mapping is malformed") from exc
    raise DataError("tensor cache input must contain CanonicalEpisode records")


def _episodes(value: Any) -> tuple[CanonicalEpisode, ...]:
    if isinstance(value, SplitPartition):
        values = value.episodes
    elif isinstance(value, EpisodeSplitResult):
        values = value.train.episodes
    elif isinstance(value, (str, os.PathLike)):
        from .parquet_writer import ParquetCanonicalShardWriter
        path = Path(value)
        if not path.exists():
            raise DataError(f"canonical dataset path does not exist: {path}")
        values = ParquetCanonicalShardWriter.read_shard(path) if path.is_file() else ParquetCanonicalShardWriter.read_dataset(path)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise DataError("episodes must be iterable") from exc
    result = tuple(_as_episode(item) for item in values)
    ids = [str(item.episode_id) for item in result]
    if len(ids) != len(set(ids)):
        raise DataError("duplicate episode_id in tensor cache input")
    return tuple(sorted(result, key=lambda item: str(item.episode_id)))


def _manifest_from(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, EpisodeSplitResult):
        return value.manifest
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read split manifest {path}") from exc
    if not isinstance(loaded, Mapping):
        raise DataError("split manifest must be a JSON object")
    return loaded


def _source_hash(episodes: Sequence[CanonicalEpisode], supplied: str | None = None) -> str:
    if supplied:
        return str(supplied)
    records = [
        {"episode_id": str(item.episode_id), "record_hash": _hash(item.to_dict(include_private=True))}
        for item in sorted(episodes, key=lambda item: str(item.episode_id))
    ]
    return _hash(records)


def _version(episodes: Sequence[CanonicalEpisode], key: str, default: str) -> str:
    values = {str(item.metadata.get(key)) for item in episodes if item.metadata.get(key) is not None}
    if len(values) > 1:
        raise DataError(f"tensor cache source has multiple {key} values: {sorted(values)}")
    return next(iter(values), default)


def _normal_episodes(episodes: Sequence[CanonicalEpisode]) -> tuple[CanonicalEpisode, ...]:
    return tuple(item for item in episodes if str(item.termination_status) == "normal")


def _step_event_ids(step: CanonicalStep) -> tuple[int, ...]:
    return tuple(int(item) for item in step.event_ids)


def _emotion_vector(label: Any) -> np.ndarray | None:
    if not isinstance(label, Mapping):
        return None
    values = label.get("values")
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise DataError("emotion labels must be finite values in [0, 1]")
    return array


def _emotion_dimension(episodes: Sequence[CanonicalEpisode], configured: int | None) -> int:
    dimensions: set[int] = set()
    for episode in episodes:
        for step in episode.steps:
            for label in step.emotion_labels.values():
                vector = _emotion_vector(label)
                if vector is not None:
                    dimensions.add(int(vector.size))
    if configured is not None:
        if isinstance(configured, bool) or int(configured) < 0:
            raise DataError("emotion dimension must be a non-negative integer")
        dimensions.add(int(configured))
    if len(dimensions) > 1:
        raise DataError(f"emotion label dimension mismatch: {sorted(dimensions)}")
    return next(iter(dimensions), 0)


def _normalization_stats(tensors: Mapping[str, np.ndarray]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key in ("state_vector", "player_state_matrix", "role_beliefs", "event_history"):
        array = np.asarray(tensors[key], dtype=np.float32)
        if array.size == 0:
            stats[key] = {"count": 0, "mean": [], "std": []}
            continue
        flattened = array.reshape(-1, array.shape[-1]) if array.ndim > 1 else array.reshape(-1, 1)
        mean = flattened.mean(axis=0, dtype=np.float64)
        std = flattened.std(axis=0, dtype=np.float64)
        stats[key] = {
            "count": int(flattened.shape[0]),
            "mean": mean.astype(np.float32).tolist(),
            "std": np.where(std == 0.0, 1.0, std).astype(np.float32).tolist(),
        }
    return stats


def _npz_bytes(tensors: Mapping[str, np.ndarray]) -> bytes:
    """Create a byte-stable NPZ (numpy's default ZipInfo timestamp is not stable)."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key in sorted(tensors):
            array_buffer = io.BytesIO()
            np.save(array_buffer, np.asarray(tensors[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    except Exception as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise DataError(f"failed to atomically write tensor cache {path}") from exc


@dataclass(frozen=True)
class TensorCacheArtifact:
    """Paths and immutable manifest produced for one split."""

    split: str
    tensor_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]

    @property
    def path(self) -> Path:
        return self.tensor_path

    @property
    def cache_hash(self) -> str:
        return str(self.manifest["cache_content_hash"])


class NPZTensorCacheWriter:
    """Build and atomically persist one or all split tensor caches."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        preprocessor: ObservationPreprocessor | ObservationPreprocessorConfig | Mapping[str, Any] | None = None,
        preprocessing_version: str | None = None,
        observation_schema_version: str | None = None,
        emotion_schema_version: str | None = None,
        emotion_dimension: int | None = None,
        split_manifest: Any | None = None,
        source_shard_ids: Iterable[str] | None = None,
        config: Any | None = None,
        config_hash: str | None = None,
        resume: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        if preprocessor is None:
            self.preprocessor = ObservationPreprocessor()
        elif isinstance(preprocessor, ObservationPreprocessor):
            self.preprocessor = preprocessor
        else:
            self.preprocessor = ObservationPreprocessor(preprocessor)
        self.preprocessing_version = preprocessing_version or self.preprocessor.config.preprocessing_version
        self.observation_schema_version = observation_schema_version or self.preprocessor.config.observation_schema_version
        self.emotion_schema_version = emotion_schema_version
        self.emotion_dimension = emotion_dimension
        self.split_manifest = _manifest_from(split_manifest)
        self.source_shard_ids = tuple(sorted(str(item) for item in (source_shard_ids or ())))
        self.config_payload = _config_payload(config)
        self.config_hash = config_hash or (_hash(self.config_payload) if config is not None else None)
        self.resume = bool(resume)

    def write(
        self,
        split: str | EpisodeSplitResult | Mapping[str, Any],
        episodes: Iterable[CanonicalEpisode] | None = None,
        *,
        train_episodes: Iterable[CanonicalEpisode] | None = None,
        split_manifest: Any | None = None,
        source_content_hash: str | None = None,
        source_shard_ids: Iterable[str] | None = None,
        split_seed: int | None = None,
        config_hash: str | None = None,
    ) -> TensorCacheArtifact | dict[str, TensorCacheArtifact]:
        if isinstance(split, EpisodeSplitResult):
            return self.write_all(split, split_manifest=split.manifest, train_episodes=split.train.episodes)
        if isinstance(split, Mapping) and "splits" in split:
            manifest = split
            return self.write_all(manifest, split_manifest=manifest, train_episodes=train_episodes)
        name = str(split)
        if name not in {"train", "validation", "test"}:
            raise DataError("cache split must be train, validation, or test")
        if episodes is None:
            raise DataError("episodes are required when writing one cache split")
        manifest = _manifest_from(split_manifest) or self.split_manifest
        records = _episodes(episodes)
        train_records = _episodes(train_episodes) if train_episodes is not None else (records if name == "train" else ())
        return self._write_one(
            name, records, train_records, manifest=manifest,
            source_content_hash=source_content_hash,
            source_shard_ids=source_shard_ids,
            split_seed=split_seed,
            config_hash=config_hash,
        )

    def write_all(
        self,
        split_result: EpisodeSplitResult | Mapping[str, Any],
        *,
        split_manifest: Any | None = None,
        train_episodes: Iterable[CanonicalEpisode] | None = None,
    ) -> dict[str, TensorCacheArtifact]:
        manifest = _manifest_from(split_manifest) or _manifest_from(split_result)
        if isinstance(split_result, EpisodeSplitResult):
            partitions = split_result.partitions
            training = _episodes(train_episodes) if train_episodes is not None else _episodes(split_result.train.episodes)
        elif isinstance(split_result, Mapping):
            raise DataError("a split manifest alone cannot supply episode records; pass episodes to write")
        else:
            raise DataError("write_all expects EpisodeSplitResult")
        source_hash = None if manifest is None else manifest.get("source_content_hash")
        split_seed = None if manifest is None else manifest.get("split_seed")
        config_hash = None if manifest is None else manifest.get("config_hash")
        return {
            name: self._write_one(
                name, _episodes(partitions[name].episodes), training,
                manifest=manifest, source_content_hash=source_hash,
                source_shard_ids=self.source_shard_ids, split_seed=split_seed,
                config_hash=config_hash,
            )
            for name in ("train", "validation", "test")
        }

    build = write
    write_split = write

    def _write_one(
        self,
        name: str,
        records: Sequence[CanonicalEpisode],
        train_records: Sequence[CanonicalEpisode],
        *,
        manifest: Mapping[str, Any] | None,
        source_content_hash: str | None,
        source_shard_ids: Iterable[str] | None,
        split_seed: int | None,
        config_hash: str | None,
    ) -> TensorCacheArtifact:
        all_records = tuple(records)
        normal_records = _normal_episodes(all_records)
        normal_train = _normal_episodes(tuple(train_records))
        if name != "train" and not normal_train:
            raise DataError("validation/test cache requires normal training episodes for statistics")
        effective_manifest = manifest or {}
        source_hash = _source_hash(all_records, str(source_content_hash) if source_content_hash else effective_manifest.get("source_content_hash"))
        schema_version = self.emotion_schema_version or _version(all_records, "emotion_schema_version", "unknown")
        obs_version = self.observation_schema_version or _version(all_records, "observation_schema_version", "1.0")
        emotion_dim = _emotion_dimension(normal_records or normal_train, self.emotion_dimension)
        tensors = self._materialize(normal_records, emotion_dim)
        train_tensors = self._materialize(normal_train, emotion_dim)
        stats = _normalization_stats(train_tensors)
        split_ids = [str(item.episode_id) for item in normal_records]
        manifest_source_metadata = effective_manifest.get("source_metadata", {})
        if not isinstance(manifest_source_metadata, Mapping):
            manifest_source_metadata = {}
        manifest_shards = manifest_source_metadata.get(
            "source_shard_ids", manifest_source_metadata.get("source_shards", ())
        )
        if not manifest_shards:
            manifest_shards = effective_manifest.get("source_shard_ids", effective_manifest.get("source_shards", ()))
        source_shards = tuple(sorted(str(item) for item in (
            source_shard_ids or self.source_shard_ids or manifest_shards or ()
        )))
        split_manifest_hash = _hash(effective_manifest) if effective_manifest else None
        descriptor = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "source_content_hash": source_hash,
            "source_shard_ids": list(source_shards),
            "split_manifest_hash": split_manifest_hash,
            "split": name,
            "split_seed": None if split_seed is None else int(split_seed),
            "config_hash": config_hash or self.config_hash,
            "preprocessor_config_hash": _hash(self.preprocessor.config.to_dict()),
            "preprocessing_version": self.preprocessing_version,
            "preprocessor_config": self.preprocessor.config.to_dict(),
            "observation_schema_version": obs_version,
            "emotion_schema_version": schema_version,
            "schema_hash": _hash({"observation_schema_version": obs_version, "emotion_schema_version": schema_version, "preprocessor_config_hash": _hash(self.preprocessor.config.to_dict())}),
            "emotion_dimension": emotion_dim,
            "episode_ids": split_ids,
            "included_episode_count": len(split_ids),
            "excluded_truncated_episode_count": sum(str(item.termination_status) == "truncated" for item in all_records),
            "tensor_schema": self._schema(tensors),
            "model_input_keys": list(MODEL_INPUT_KEYS),
            "target_keys": list(TARGET_KEYS),
            "transition_keys": list(TRANSITION_KEYS),
            "normalization": {"method": "preprocessor", "applied": False, "statistics_source": "train", "statistics": stats},
            "train_statistics": stats,
        }
        cache_hash = _hash(descriptor)
        tensor_path = self.output_dir / f"{DEFAULT_CACHE_PREFIX}-{name}.npz"
        manifest_path = self.output_dir / f"{DEFAULT_CACHE_PREFIX}-{name}.manifest.json"
        existing = self._existing_if_compatible(tensor_path, manifest_path, descriptor, cache_hash)
        if existing is not None and self.resume:
            return existing
        npz_data = _npz_bytes(tensors)
        final_manifest = dict(descriptor)
        final_manifest.update({
            "cache_content_hash": cache_hash,
            "npz_file": tensor_path.name,
            "manifest_file": manifest_path.name,
            "npz_sha256": sha256(npz_data).hexdigest(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        _atomic_write(tensor_path, npz_data)
        _atomic_write(manifest_path, _canonical_json(final_manifest) + b"\n")
        return TensorCacheArtifact(name, tensor_path, manifest_path, final_manifest)

    @staticmethod
    def _schema(tensors: Mapping[str, np.ndarray]) -> dict[str, Any]:
        return {
            key: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for key, value in sorted(tensors.items())
        }

    def _materialize(self, records: Sequence[CanonicalEpisode], emotion_dim: int) -> dict[str, np.ndarray]:
        rows: list[dict[str, Any]] = []
        for episode_index, episode in enumerate(sorted(records, key=lambda item: str(item.episode_id))):
            if str(episode.termination_status) != "normal":
                continue
            for step in sorted(episode.steps, key=lambda item: int(item.environment_step)):
                if step.pre_observation is None:
                    raise DataError(f"step in episode {episode.episode_id!r} has no pre_observation")
                try:
                    processed = self.preprocessor.preprocess(step.pre_observation)
                except Exception as exc:
                    raise DataError(f"cannot preprocess episode {episode.episode_id!r} step {step.environment_step}") from exc
                if not hasattr(processed, "state_vector"):
                    raise DataError("preprocessor returned an unexpected batch value")
                event_ids = _step_event_ids(step)
                roles = episode.role_labels
                role_values = np.full(PLAYER_COUNT, HIDDEN_ROLE_SENTINEL, dtype=np.uint8)
                role_mask = np.zeros(PLAYER_COUNT, dtype=np.bool_)
                if roles is not None:
                    if len(roles) != PLAYER_COUNT:
                        raise DataError("role_labels must contain five players")
                    role_values[:] = np.asarray(roles, dtype=np.uint8)
                    if np.any(role_values >= 5):
                        raise DataError("role_labels must contain role indices in [0, 4]")
                    role_mask[:] = True
                emotion_values = np.zeros((PLAYER_COUNT, emotion_dim), dtype=np.float32)
                emotion_mask = np.zeros(PLAYER_COUNT, dtype=np.bool_)
                for player_key, label in step.emotion_labels.items():
                    player = int(player_key)
                    if not 0 <= player < PLAYER_COUNT:
                        raise DataError("emotion label player index is out of range")
                    vector = _emotion_vector(label)
                    if vector is None:
                        continue
                    if vector.size != emotion_dim:
                        raise DataError("emotion label dimension does not match Emotion Schema")
                    emotion_values[player] = vector
                    emotion_mask[player] = True
                rows.append({
                    "state_vector": processed.state_vector,
                    "player_state_matrix": processed.player_state_matrix,
                    "role_beliefs": processed.role_beliefs,
                    "event_history": processed.event_history,
                    "event_valid_mask": processed.event_valid_mask,
                    "action_mask": processed.action_mask,
                    "role_labels": role_values,
                    "role_label_mask": role_mask,
                    "emotion_labels": emotion_values,
                    "emotion_label_mask": emotion_mask,
                    "actions": int(step.action),
                    "actor_indices": int(step.actor_idx),
                    "target_indices": PASS_TARGET_SENTINEL if step.target_idx is None else int(step.target_idx),
                    "environment_steps": int(step.environment_step),
                    "event_id_starts": min(event_ids) if event_ids else 0,
                    "event_id_ends": max(event_ids) if event_ids else 0,
                    "event_counts": len(event_ids),
                    "episode_indices": episode_index,
                    "rewards": float(step.reward),
                    "terminated": bool(step.terminated),
                    "truncated": bool(step.truncated),
                })
        if not rows:
            return self._empty_tensors(emotion_dim)
        arrays: dict[str, np.ndarray] = {}
        for key in MODEL_INPUT_KEYS + TARGET_KEYS:
            arrays[key] = np.stack([row[key] for row in rows]).astype(np.dtype(FIXED_TENSOR_SCHEMA[key]["dtype"]), copy=False)
        for key in TRANSITION_KEYS:
            arrays[key] = np.asarray([row[key] for row in rows], dtype=np.dtype(FIXED_TENSOR_SCHEMA[key]["dtype"]))
        return arrays

    @staticmethod
    def _empty_tensors(emotion_dim: int) -> dict[str, np.ndarray]:
        schema = {
            **{key: tuple(value["shape"]) for key, value in FIXED_TENSOR_SCHEMA.items() if key != "emotion_labels"},
            "emotion_labels": (None, PLAYER_COUNT, emotion_dim),
        }
        arrays: dict[str, np.ndarray] = {}
        for key, shape in schema.items():
            resolved = tuple(0 if item is None and index == 0 else item for index, item in enumerate(shape))
            arrays[key] = np.empty(resolved, dtype=np.dtype(FIXED_TENSOR_SCHEMA[key]["dtype"]))
        return arrays

    @staticmethod
    def _existing_if_compatible(
        tensor_path: Path, manifest_path: Path, descriptor: Mapping[str, Any], cache_hash: str
    ) -> TensorCacheArtifact | None:
        if not tensor_path.exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping) or manifest.get("cache_content_hash") != cache_hash:
                return None
            loader = NPZTensorCacheLoader(tensor_path, manifest_path=manifest_path)
            loader.validate()
            return TensorCacheArtifact(str(descriptor["split"]), tensor_path, manifest_path, manifest)
        except (OSError, json.JSONDecodeError, DataError, ValueError, TypeError):
            return None


class NPZTensorCacheLoader:
    """Integrity-checking indexed/batched access to a derived tensor cache."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        manifest_path: str | os.PathLike[str] | None = None,
        expected: Mapping[str, Any] | None = None,
    ) -> None:
        candidate = Path(path)
        if candidate.is_dir():
            split = str((expected or {}).get("split", "train"))
            candidate = candidate / f"{DEFAULT_CACHE_PREFIX}-{split}.npz"
        self.tensor_path = candidate
        self.manifest_path = Path(manifest_path) if manifest_path is not None else candidate.with_name(candidate.stem + ".manifest.json")
        self.expected = dict(expected or {})
        self._tensors: dict[str, np.ndarray] | None = None
        self._manifest: Mapping[str, Any] | None = None

    @property
    def manifest(self) -> Mapping[str, Any]:
        self._load_manifest()
        assert self._manifest is not None
        return self._manifest

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(MODEL_INPUT_KEYS)

    @property
    def all_keys(self) -> tuple[str, ...]:
        return tuple(sorted(FIXED_TENSOR_SCHEMA))

    def _load_manifest(self) -> None:
        if self._manifest is not None:
            return
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"cannot read tensor cache manifest {self.manifest_path}") from exc
        if not isinstance(value, Mapping) or value.get("cache_format_version") != CACHE_FORMAT_VERSION:
            raise DataError("unsupported or malformed tensor cache manifest")
        self._manifest = value

    def _load_tensors(self) -> None:
        if self._tensors is not None:
            return
        self._load_manifest()
        assert self._manifest is not None
        if not self.tensor_path.exists():
            raise DataError(f"tensor cache file does not exist: {self.tensor_path}")
        expected_hash = self._manifest.get("npz_sha256")
        if not isinstance(expected_hash, str) or _file_hash(self.tensor_path) != expected_hash:
            raise DataError("tensor cache checksum mismatch")
        try:
            with np.load(self.tensor_path, allow_pickle=False) as archive:
                names = tuple(sorted(archive.files))
                expected_names = tuple(sorted(FIXED_TENSOR_SCHEMA))
                if names != expected_names:
                    raise DataError("tensor cache keys do not match the fixed schema")
                self._tensors = {name: np.array(archive[name], copy=True) for name in names}
        except DataError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise DataError(f"cannot load tensor cache {self.tensor_path}") from exc
        self._validate_tensors()

    def _validate_tensors(self) -> None:
        assert self._tensors is not None
        manifest_schema = self.manifest.get("tensor_schema")
        if not isinstance(manifest_schema, Mapping):
            raise DataError("tensor cache manifest is missing tensor_schema")
        for key in sorted(FIXED_TENSOR_SCHEMA):
            array = self._tensors[key]
            schema = manifest_schema.get(key)
            if not isinstance(schema, Mapping) or schema.get("dtype") != str(array.dtype) or schema.get("shape") != list(array.shape):
                raise DataError(f"tensor cache shape/dtype validation failed for {key}")
            if key == "action_mask" and (np.any((array != 0) & (array != 1))):
                raise DataError("action_mask tensor must contain only 0 and 1")
            if key == "emotion_labels" and (np.any(~np.isfinite(array)) or np.any((array < 0) | (array > 1))):
                raise DataError("emotion_labels tensor must be normalized to [0, 1]")
        leading = {int(array.shape[0]) for array in self._tensors.values()}
        if len(leading) != 1:
            raise DataError("tensor cache tensors have inconsistent batch dimensions")

    def validate(self) -> None:
        self._load_tensors()
        self._validate_expected()

    def _validate_expected(self) -> None:
        for key, expected in self.expected.items():
            if key in {"split", "split_seed", "source_content_hash", "split_manifest_hash", "preprocessing_version", "observation_schema_version", "emotion_schema_version", "config_hash", "preprocessor_config_hash", "schema_hash", "cache_content_hash"}:
                if self.manifest.get(key) != expected:
                    raise DataError(f"tensor cache compatibility mismatch for {key}")

    def __len__(self) -> int:
        self._load_tensors()
        assert self._tensors is not None
        return int(self._tensors[MODEL_INPUT_KEYS[0]].shape[0])

    def __getitem__(self, index: int | slice) -> dict[str, np.ndarray]:
        return self.get(index)

    def get(self, index: int | slice, *, include_labels: bool = False) -> dict[str, np.ndarray]:
        self._load_tensors()
        assert self._tensors is not None
        keys = MODEL_INPUT_KEYS + (TARGET_KEYS if include_labels else ())
        return {key: np.array(self._tensors[key][index], copy=True) for key in keys}

    def model_inputs(self, index: int | slice = slice(None)) -> dict[str, np.ndarray]:
        return self.get(index, include_labels=False)

    def labels(self, index: int | slice = slice(None)) -> dict[str, np.ndarray]:
        self._load_tensors()
        assert self._tensors is not None
        return {key: np.array(self._tensors[key][index], copy=True) for key in TARGET_KEYS}

    def transition(self, index: int | slice = slice(None)) -> dict[str, np.ndarray]:
        self._load_tensors()
        assert self._tensors is not None
        return {key: np.array(self._tensors[key][index], copy=True) for key in TRANSITION_KEYS}

    def sample(self, index: int | slice = slice(None), *, include_labels: bool = True) -> dict[str, np.ndarray]:
        """Return model inputs, transition bookkeeping, and optional targets."""

        self._load_tensors()
        assert self._tensors is not None
        keys = MODEL_INPUT_KEYS + TRANSITION_KEYS + (TARGET_KEYS if include_labels else ())
        return {key: np.array(self._tensors[key][index], copy=True) for key in keys}

    def all_tensors(self, index: int | slice = slice(None)) -> dict[str, np.ndarray]:
        """Return every validated tensor, including targets, for inspection."""

        self._load_tensors()
        assert self._tensors is not None
        return {key: np.array(self._tensors[key][index], copy=True) for key in sorted(self._tensors)}

    def batch_all(self, start: int = 0, stop: int | None = None, *, include_labels: bool = True) -> dict[str, np.ndarray]:
        """Batch form of :meth:`sample`; model inputs remain label-free by default."""

        if stop is None:
            stop = int(start) + 1
        self._load_tensors()
        return self.sample(slice(int(start), int(stop)), include_labels=include_labels)

    def batch(self, start: int | Sequence[int] = 0, stop: int | None = None, *, include_labels: bool = False) -> dict[str, np.ndarray]:
        if isinstance(start, Sequence) and not isinstance(start, (str, bytes, bytearray)):
            index: Any = np.asarray(list(start), dtype=np.int64)
        elif stop is None:
            index = slice(int(start), int(start) + 1)
        else:
            index = slice(int(start), int(stop))
        return self.get(index, include_labels=include_labels)

    def iter_batches(self, batch_size: int, *, include_labels: bool = False):
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), int(batch_size)):
            yield self.get(slice(start, min(start + int(batch_size), len(self))), include_labels=include_labels)

    iter_batch = iter_batches

    def __iter__(self):
        for index in range(len(self)):
            yield self.get(index)

    def close(self) -> None:
        self._tensors = None


# Short aliases follow the wording used by DatasetService and the task.
TensorCacheWriter = NPZTensorCacheWriter
TensorCacheLoader = NPZTensorCacheLoader
NPZCacheWriter = NPZTensorCacheWriter
NPZCacheLoader = NPZTensorCacheLoader

__all__ = [
    "CACHE_FORMAT_VERSION",
    "CACHE_SCHEMA_VERSION",
    "FIXED_TENSOR_SCHEMA",
    "MODEL_INPUT_KEYS",
    "TARGET_KEYS",
    "TRANSITION_KEYS",
    "TensorCacheArtifact",
    "NPZTensorCacheWriter",
    "NPZTensorCacheLoader",
    "TensorCacheWriter",
    "TensorCacheLoader",
    "NPZCacheWriter",
    "NPZCacheLoader",
]
