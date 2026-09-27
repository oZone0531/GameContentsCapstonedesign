"""Deterministic, episode-level dataset splitting for the Phase 1.5 pipeline.

The canonical Parquet dataset stores one row per complete episode.  This module
keeps that boundary intact: a split contains complete ``CanonicalEpisode``
objects, never individual steps or labels.  A JSON manifest is the durable
source of truth for the assignment and is validated before it is resumed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from math import floor, isfinite
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from bang_rl.exceptions import DataError

from .collector import CANONICAL_DATASET_SCHEMA_VERSION, CanonicalEpisode
from .parquet_writer import ParquetCanonicalShardWriter

SPLIT_MANIFEST_VERSION = "1.0"
SPLIT_CONFIG_VERSION = "1.0"
_SPLIT_NAMES = ("train", "validation", "test")


def _json_safe(value: Any) -> Any:
    """Convert common values to deterministic JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _termination_status(episode: CanonicalEpisode) -> str:
    status = str(getattr(episode.termination_status, "value", episode.termination_status))
    if status not in {"normal", "truncated"}:
        raise DataError(f"episode {episode.episode_id!r} has invalid termination_status {status!r}")
    return status


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate a total with largest-remainder rounding and stable tie breaks."""

    raw = [total * ratio for ratio in ratios]
    counts = [floor(value) for value in raw]
    remaining = total - sum(counts)
    # Split order is intentionally stable.  This makes a tie (e.g. one item)
    # deterministic without relying on dictionary or input iteration order.
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(int(value) for value in counts)  # type: ignore[return-value]


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("split ratios must contain train, validation, and test values")
    resolved = tuple(float(value) for value in ratios)
    if any(not isfinite(value) or value < 0.0 for value in resolved):
        raise ValueError("split ratios must be finite and non-negative")
    if sum(resolved) <= 0.0 or abs(sum(resolved) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")
    return resolved  # type: ignore[return-value]


@dataclass(frozen=True)
class EpisodeSplitConfig:
    """Configuration captured in a split manifest.

    ``counts`` takes precedence over ratios when supplied, and is useful for
    small datasets where exact requested set sizes are more important than a
    rounded ratio.  All three counts must be supplied together.
    """

    seed: int = 0
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    counts: tuple[int, int, int] | None = None
    stratify_by_termination: bool = False
    split_version: str = SPLIT_CONFIG_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "ratios", _validate_ratios(self.ratios))
        if self.counts is not None:
            if len(self.counts) != 3 or any(isinstance(value, bool) or int(value) < 0 for value in self.counts):
                raise ValueError("split counts must contain three non-negative integers")
            object.__setattr__(self, "counts", tuple(int(value) for value in self.counts))
        if not self.split_version:
            raise ValueError("split_version must not be empty")

    @property
    def names(self) -> tuple[str, str, str]:
        return _SPLIT_NAMES

    def target_counts(self, total: int) -> tuple[int, int, int]:
        if total < 0:
            raise ValueError("total episode count cannot be negative")
        if self.counts is not None:
            if sum(self.counts) != total:
                raise ValueError("explicit split counts must sum to the episode count")
            return self.counts
        return _allocate_counts(total, self.ratios)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_version": self.split_version,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "counts": None if self.counts is None else list(self.counts),
            "stratify_by_termination": self.stratify_by_termination,
        }

    @property
    def config_hash(self) -> str:
        return _hash(self.to_dict())


# Short name for callers using the task wording.
SplitConfig = EpisodeSplitConfig


@dataclass(frozen=True)
class SplitPartition:
    """One manifest-backed split and its complete episode records."""

    name: str
    episodes: tuple[CanonicalEpisode, ...]
    episode_ids: tuple[str, ...]
    termination_counts: Mapping[str, int]
    environment_steps: int

    @property
    def count(self) -> int:
        return len(self.episodes)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "split": self.name,
            "episode_count": self.count,
            "environment_step_count": self.environment_steps,
            "termination_counts": dict(self.termination_counts),
            "episode_ids": list(self.episode_ids),
        }


@dataclass(frozen=True)
class EpisodeSplitResult:
    """Result of a validated split operation."""

    train: SplitPartition
    validation: SplitPartition
    test: SplitPartition
    manifest: Mapping[str, Any]
    manifest_path: Path | None = None

    @property
    def partitions(self) -> Mapping[str, SplitPartition]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    @property
    def episode_ids(self) -> Mapping[str, tuple[str, ...]]:
        return {name: partition.episode_ids for name, partition in self.partitions.items()}

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.manifest.get("output_metadata", {})

    def __getitem__(self, name: str) -> SplitPartition:
        return self.partitions[name]


# Friendly aliases for external dataset-service callers.
SplitResult = EpisodeSplitResult


class EpisodeSplitter:
    """Split canonical episodes without row, step, or label leakage."""

    def __init__(
        self,
        config: EpisodeSplitConfig | None = None,
        *,
        seed: int | None = None,
        ratios: Sequence[float] | None = None,
        counts: Sequence[int] | None = None,
        stratify_by_termination: bool | None = None,
        manifest_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if config is not None and any(value is not None for value in (seed, ratios, counts, stratify_by_termination)):
            raise TypeError("provide config or split configuration keyword arguments, not both")
        if config is None:
            config = EpisodeSplitConfig(
                seed=0 if seed is None else seed,
                ratios=(0.8, 0.1, 0.1) if ratios is None else tuple(ratios),
                counts=None if counts is None else tuple(counts),
                stratify_by_termination=False if stratify_by_termination is None else stratify_by_termination,
            )
        if not isinstance(config, EpisodeSplitConfig):
            raise TypeError("config must be EpisodeSplitConfig")
        self.config = config
        self.manifest_path = None if manifest_path is None else Path(manifest_path)

    @staticmethod
    def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"cannot read split manifest {path}") from exc
        if not isinstance(value, dict) or value.get("manifest_version") != SPLIT_MANIFEST_VERSION:
            raise DataError(f"unsupported or malformed split manifest: {path}")
        return value

    def split(
        self,
        episodes: Iterable[CanonicalEpisode] | str | os.PathLike[str],
        *,
        manifest_path: str | os.PathLike[str] | None = None,
        resume: bool = False,
        statistics_fn: Callable[[tuple[CanonicalEpisode, ...]], Mapping[str, Any]] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> EpisodeSplitResult:
        records = self._load_episodes(episodes)
        self._validate_episodes(records)
        path = self.manifest_path if manifest_path is None else Path(manifest_path)
        source_hash = self._source_hash(records)
        existing = self.load_manifest(path) if path is not None and path.exists() else None
        if existing is not None:
            self._validate_manifest_compatibility(existing, records, source_hash)
            # A compatible existing manifest is safe to resume even when the
            # caller did not set resume explicitly; it cannot silently change.
            return self._result_from_manifest(existing, records, path)
        result_manifest = self._create_manifest(records, source_hash, statistics_fn, source_metadata)
        if path is not None:
            self._write_manifest(path, result_manifest)
        return self._result_from_manifest(result_manifest, records, path)

    def split_dataset(self, source: str | os.PathLike[str], **kwargs: Any) -> EpisodeSplitResult:
        """Read a canonical shard/dataset directory and split its episodes."""

        return self.split(source, **kwargs)

    def write_split_datasets(
        self,
        result: EpisodeSplitResult,
        output_dir: str | os.PathLike[str],
        *,
        include_private_roles: bool = True,
    ) -> dict[str, tuple[Path, ...]]:
        """Persist complete episode records per split without mixing partitions."""

        root = Path(output_dir)
        paths: dict[str, tuple[Path, ...]] = {}
        for name, partition in result.partitions.items():
            writer = ParquetCanonicalShardWriter(
                root / name,
                include_private_roles=include_private_roles,
            )
            paths[name] = writer.finalize() if not partition.episodes else self._write_partition(writer, partition)
        return paths

    @staticmethod
    def _write_partition(writer: ParquetCanonicalShardWriter, partition: SplitPartition) -> tuple[Path, ...]:
        for episode in partition.episodes:
            writer.append(episode)
        return writer.finalize()

    @staticmethod
    def _load_episodes(
        source: Iterable[CanonicalEpisode] | str | os.PathLike[str],
    ) -> tuple[CanonicalEpisode, ...]:
        if isinstance(source, (str, os.PathLike)):
            path = Path(source)
            if path.is_file():
                return ParquetCanonicalShardWriter.read_shard(path)
            if path.is_dir():
                return ParquetCanonicalShardWriter.read_dataset(path)
            raise DataError(f"canonical dataset path does not exist: {path}")
        records: list[CanonicalEpisode] = []
        for item in source:
            if isinstance(item, CanonicalEpisode):
                records.append(item)
            elif isinstance(item, Mapping):
                try:
                    records.append(CanonicalEpisode.from_dict(item))
                except Exception as exc:
                    raise DataError("canonical episode mapping is malformed") from exc
            else:
                raise DataError("episode split input must contain CanonicalEpisode records")
        return tuple(records)

    @staticmethod
    def _validate_episodes(records: Sequence[CanonicalEpisode]) -> None:
        ids: set[str] = set()
        schema_values: dict[str, set[str]] = {}
        for episode in records:
            episode_id = str(episode.episode_id).strip()
            if not episode_id or episode_id.lower() in {"none", "null"}:
                raise DataError("canonical episode is missing episode_id")
            if episode_id in ids:
                raise DataError(f"duplicate episode_id in split input: {episode_id!r}")
            ids.add(episode_id)
            metadata = episode.metadata or {}
            dataset_version = str(metadata.get("dataset_schema_version", CANONICAL_DATASET_SCHEMA_VERSION))
            if dataset_version != CANONICAL_DATASET_SCHEMA_VERSION:
                raise DataError(
                    f"episode {episode_id!r} has incompatible dataset schema {dataset_version!r}"
                )
            _termination_status(episode)
            for key in ("observation_schema_version", "action_space_version", "emotion_schema_version", "reward_config_version"):
                if key in metadata and metadata[key] is not None:
                    schema_values.setdefault(key, set()).add(str(metadata[key]))
        mismatches = {key: sorted(values) for key, values in schema_values.items() if len(values) > 1}
        if mismatches:
            raise DataError(f"canonical episode schema mismatch: {mismatches}")

    @staticmethod
    def _source_hash(records: Sequence[CanonicalEpisode]) -> str:
        # Sorting by ID makes reproducibility independent of Parquet shard/input order.
        payload = [
            {"episode_id": episode.episode_id, "record_hash": _hash(episode.to_dict(include_private=True))}
            for episode in sorted(records, key=lambda item: str(item.episode_id))
        ]
        return _hash(payload)

    def _assignment(self, records: Sequence[CanonicalEpisode]) -> dict[str, str]:
        target_counts = self.config.target_counts(len(records))
        groups: dict[str, list[CanonicalEpisode]] = {"normal": [], "truncated": []}
        if self.config.stratify_by_termination:
            for episode in records:
                groups[_termination_status(episode)].append(episode)
        else:
            groups = {"all": list(records)}

        # First allocate each group by its ideal share, then distribute the
        # largest remainders into globally available target slots.  This keeps
        # exact requested totals while making status composition as proportional
        # as integer counts permit.
        initial: dict[str, list[int]] = {}
        remainders: list[tuple[float, str, int]] = []
        for group_name in sorted(groups):
            group_size = len(groups[group_name])
            raw = [group_size * target / len(records) if records else 0.0 for target in target_counts]
            base = [floor(value) for value in raw]
            initial[group_name] = base
            for split_index, value in enumerate(raw):
                remainders.append((value - base[split_index], group_name, split_index))
        capacities = [target_counts[index] - sum(initial[group][index] for group in initial) for index in range(3)]
        remaining_by_group = {group: len(items) - sum(initial[group]) for group, items in groups.items()}
        for _, group_name, split_index in sorted(remainders, key=lambda item: (-item[0], item[1], item[2])):
            if remaining_by_group[group_name] and capacities[split_index]:
                initial[group_name][split_index] += 1
                remaining_by_group[group_name] -= 1
                capacities[split_index] -= 1
        # The fallback handles pathological zero-ratio/rounding combinations
        # while retaining exact totals; it is deterministic and never drops an
        # episode.
        for group_name in sorted(groups):
            for split_index in range(3):
                while remaining_by_group[group_name] and capacities[split_index]:
                    initial[group_name][split_index] += 1
                    remaining_by_group[group_name] -= 1
                    capacities[split_index] -= 1
        if any(remaining_by_group.values()) or any(capacities):
            raise DataError("could not allocate all episodes to requested split counts")

        assignment: dict[str, str] = {}
        for group_name in sorted(groups):
            ordered = sorted(
                groups[group_name],
                key=lambda episode: hashlib.sha256(
                    f"{self.config.seed}\0{group_name}\0{episode.episode_id}".encode("utf-8")
                ).hexdigest(),
            )
            cursor = 0
            for split_index, split_name in enumerate(_SPLIT_NAMES):
                for episode in ordered[cursor : cursor + initial[group_name][split_index]]:
                    assignment[str(episode.episode_id)] = split_name
                cursor += initial[group_name][split_index]
        if set(assignment) != {str(episode.episode_id) for episode in records}:
            raise DataError("split assignment does not cover every episode")
        return assignment

    def _create_manifest(
        self,
        records: Sequence[CanonicalEpisode],
        source_hash: str,
        statistics_fn: Callable[[tuple[CanonicalEpisode, ...]], Mapping[str, Any]] | None,
        source_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        assignment = self._assignment(records)
        by_id = {str(episode.episode_id): episode for episode in records}
        split_ids = {name: sorted(episode_id for episode_id, split in assignment.items() if split == name) for name in _SPLIT_NAMES}
        train_episodes = tuple(by_id[episode_id] for episode_id in split_ids["train"])
        train_statistics = (
            _json_safe(dict(statistics_fn(train_episodes))) if statistics_fn is not None else self._default_train_statistics(train_episodes)
        )
        partitions = {
            name: self._partition_metadata(name, tuple(by_id[episode_id] for episode_id in split_ids[name]))
            for name in _SPLIT_NAMES
        }
        metadata_values = {
            key: sorted({str(episode.metadata[key]) for episode in records if key in episode.metadata and episode.metadata[key] is not None})
            for key in ("dataset_schema_version", "observation_schema_version", "action_space_version", "emotion_schema_version", "reward_config_version")
        }
        manifest: dict[str, Any] = {
            "manifest_version": SPLIT_MANIFEST_VERSION,
            "split_config_version": self.config.split_version,
            "split_seed": self.config.seed,
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash,
            "source_content_hash": source_hash,
            "source_metadata": _json_safe(dict(source_metadata or {})),
            "dataset_schema_versions": metadata_values,
            "source_episode_count": len(records),
            "episode_ids": sorted(str(episode.episode_id) for episode in records),
            "splits": {name: partitions[name]["episode_ids"] for name in _SPLIT_NAMES},
            "split_metadata": partitions,
            "train_statistics": train_statistics,
            "statistics_source": "train",
            "output_metadata": {
                "split_seed": self.config.seed,
                "config_hash": self.config.config_hash,
                "source_content_hash": source_hash,
                "episode_count": len(records),
                "split_counts": {name: partitions[name]["episode_count"] for name in _SPLIT_NAMES},
                "termination_counts": {name: partitions[name]["termination_counts"] for name in _SPLIT_NAMES},
                "dataset_schema_versions": metadata_values,
            },
        }
        return manifest

    @staticmethod
    def _default_train_statistics(episodes: Sequence[CanonicalEpisode]) -> dict[str, Any]:
        return {
            "episode_count": len(episodes),
            "environment_step_count": sum(episode.environment_step_count for episode in episodes),
            "termination_counts": dict(Counter(_termination_status(episode) for episode in episodes)),
        }

    @staticmethod
    def _partition_metadata(name: str, episodes: Sequence[CanonicalEpisode]) -> dict[str, Any]:
        return {
            "split": name,
            "episode_count": len(episodes),
            "episode_ids": sorted(str(episode.episode_id) for episode in episodes),
            "environment_step_count": sum(episode.environment_step_count for episode in episodes),
            "termination_counts": dict(sorted(Counter(_termination_status(episode) for episode in episodes).items())),
        }

    def _validate_manifest_compatibility(
        self,
        manifest: Mapping[str, Any],
        records: Sequence[CanonicalEpisode],
        source_hash: str,
    ) -> None:
        if manifest.get("config_hash") != self.config.config_hash:
            raise DataError("split manifest configuration does not match requested configuration")
        if manifest.get("source_content_hash") != source_hash:
            raise DataError("split manifest source dataset does not match input episodes")
        expected_ids = sorted(str(episode.episode_id) for episode in records)
        if manifest.get("episode_ids") != expected_ids:
            raise DataError("split manifest contains missing or unexpected episode IDs")
        splits = manifest.get("splits")
        if not isinstance(splits, Mapping) or set(splits) != set(_SPLIT_NAMES):
            raise DataError("split manifest is missing train/validation/test assignments")
        seen: list[str] = []
        for name in _SPLIT_NAMES:
            ids = splits[name]
            if not isinstance(ids, list) or ids != sorted(ids):
                raise DataError(f"split manifest IDs for {name} are malformed")
            seen.extend(str(item) for item in ids)
        if len(seen) != len(set(seen)) or sorted(seen) != expected_ids:
            raise DataError("split manifest has duplicate, missing, or overlapping episode IDs")

    def _result_from_manifest(
        self,
        manifest: Mapping[str, Any],
        records: Sequence[CanonicalEpisode],
        path: Path | None,
    ) -> EpisodeSplitResult:
        by_id = {str(episode.episode_id): episode for episode in records}
        partitions = {}
        for name in _SPLIT_NAMES:
            ids = tuple(manifest["splits"][name])
            if any(episode_id not in by_id for episode_id in ids):
                raise DataError(f"split manifest references a missing episode ID in {name}")
            selected = tuple(by_id[episode_id] for episode_id in ids)
            metadata = self._partition_metadata(name, selected)
            partitions[name] = SplitPartition(
                name=name,
                episodes=selected,
                episode_ids=ids,
                termination_counts=metadata["termination_counts"],
                environment_steps=metadata["environment_step_count"],
            )
        if set(partitions["train"].episode_ids) & set(partitions["validation"].episode_ids) or set(partitions["train"].episode_ids) & set(partitions["test"].episode_ids) or set(partitions["validation"].episode_ids) & set(partitions["test"].episode_ids):
            raise DataError("episode leakage detected between split partitions")
        return EpisodeSplitResult(
            train=partitions["train"], validation=partitions["validation"], test=partitions["test"],
            manifest=manifest, manifest_path=path,
        )

    @staticmethod
    def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (_canonical_json(manifest) + "\n").encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception as exc:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise DataError(f"failed to atomically write split manifest {path}") from exc


# Functional API for DatasetService and small callers.
def split_episodes(
    episodes: Iterable[CanonicalEpisode] | str | os.PathLike[str],
    *,
    seed: int = 0,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    counts: Sequence[int] | None = None,
    stratify_by_termination: bool = False,
    manifest_path: str | os.PathLike[str] | None = None,
    statistics_fn: Callable[[tuple[CanonicalEpisode, ...]], Mapping[str, Any]] | None = None,
) -> EpisodeSplitResult:
    return EpisodeSplitter(
        seed=seed,
        ratios=ratios,
        counts=counts,
        stratify_by_termination=stratify_by_termination,
        manifest_path=manifest_path,
    ).split(episodes, statistics_fn=statistics_fn)


__all__ = [
    "SPLIT_CONFIG_VERSION",
    "SPLIT_MANIFEST_VERSION",
    "EpisodeSplitConfig",
    "SplitConfig",
    "SplitPartition",
    "EpisodeSplitResult",
    "SplitResult",
    "EpisodeSplitter",
    "split_episodes",
]
