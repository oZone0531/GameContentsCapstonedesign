"""Immutable, persisted snapshots for collection and training jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from pydantic import BaseModel

from .schema import SystemConfig

SNAPSHOT_FORMAT_VERSION = "1.0"
DEFAULT_ENVIRONMENT_VERSION = "1.0"
DEFAULT_OBSERVATION_SCHEMA_VERSION = "1.0"
DEFAULT_ACTION_SPACE_VERSION = "1.0"
DEFAULT_PREPROCESSING_VERSION = "1.0"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _dump_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _canonical_copy(value: Any) -> dict[str, Any]:
    # JSON round-tripping also normalizes enum/tuple values and guarantees the
    # snapshot contains no references into the mutable Pydantic config object.
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class JobSnapshot:
    """Deeply immutable resolved configuration captured at job start."""

    job_id: str
    mode: str
    config_version: str
    seed: int
    split_seed: int
    config_snapshot: Mapping[str, Any]
    preset_snapshot: Mapping[str, Any]
    reward_snapshot: Mapping[str, Any]
    emotion_schema_snapshot: Mapping[str, Any]
    model_architecture: Mapping[str, Any]
    optimizers: Mapping[str, Any]
    versions: Mapping[str, Any]
    artifact_compatibility: Mapping[str, Any]
    split: Mapping[str, Any]
    captured_at: str | None = None

    @property
    def resolved_config(self) -> Mapping[str, Any]:
        return self.config_snapshot

    @property
    def model_architecture_snapshot(self) -> Mapping[str, Any]:
        return self.model_architecture

    @property
    def optimizer_snapshot(self) -> Mapping[str, Any]:
        return self.optimizers

    @property
    def seed_snapshot(self) -> int:
        return self.seed

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy for persistence or logging."""

        result = {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "job_id": self.job_id,
            "mode": self.mode,
            "config_version": self.config_version,
            "seed": self.seed,
            "split_seed": self.split_seed,
            "config_snapshot": _thaw(self.config_snapshot),
            "preset_snapshot": _thaw(self.preset_snapshot),
            "reward_snapshot": _thaw(self.reward_snapshot),
            "emotion_schema_snapshot": _thaw(self.emotion_schema_snapshot),
            "model_architecture": _thaw(self.model_architecture),
            "optimizers": _thaw(self.optimizers),
            "versions": _thaw(self.versions),
            "artifact_compatibility": _thaw(self.artifact_compatibility),
            "split": _thaw(self.split),
        }
        if self.captured_at is not None:
            result["captured_at"] = self.captured_at
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobSnapshot":
        required = {
            "job_id",
            "mode",
            "config_version",
            "seed",
            "split_seed",
            "config_snapshot",
            "preset_snapshot",
            "reward_snapshot",
            "emotion_schema_snapshot",
            "model_architecture",
            "optimizers",
            "versions",
            "artifact_compatibility",
            "split",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"Job snapshot is missing required fields: {', '.join(missing)}")
        return cls(
            job_id=str(value["job_id"]),
            mode=str(value["mode"]),
            config_version=str(value["config_version"]),
            seed=int(value["seed"]),
            split_seed=int(value["split_seed"]),
            config_snapshot=_freeze(value["config_snapshot"]),
            preset_snapshot=_freeze(value["preset_snapshot"]),
            reward_snapshot=_freeze(value["reward_snapshot"]),
            emotion_schema_snapshot=_freeze(value["emotion_schema_snapshot"]),
            model_architecture=_freeze(value["model_architecture"]),
            optimizers=_freeze(value["optimizers"]),
            versions=_freeze(value["versions"]),
            artifact_compatibility=_freeze(value["artifact_compatibility"]),
            split=_freeze(value["split"]),
            captured_at=value.get("captured_at"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "JobSnapshot":
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read job snapshot {source}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"Job snapshot {source} must contain a JSON object")
        return cls.from_dict(value)

    @classmethod
    def capture(
        cls,
        config: SystemConfig,
        *,
        mode: str = "rl",
        seed: int | None = None,
        split_seed: int | None = None,
        job_id: str | None = None,
        environment_version: str = DEFAULT_ENVIRONMENT_VERSION,
        observation_schema_version: str = DEFAULT_OBSERVATION_SCHEMA_VERSION,
        action_space_version: str = DEFAULT_ACTION_SPACE_VERSION,
        preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
        captured_at: str | None = None,
    ) -> "JobSnapshot":
        if not isinstance(config, SystemConfig):
            raise TypeError("JobSnapshot.capture requires a validated SystemConfig")
        resolved = _canonical_copy(config.model_dump(mode="json"))
        actual_seed = config.seed if seed is None else int(seed)
        actual_split_seed = actual_seed if split_seed is None else int(split_seed)
        reward = _dump_model(config.reward_config)
        preset = _dump_model(config.preset)
        emotion = _dump_model(config.emotion)
        model_architecture = _dump_model(config.model_architecture)
        optimizers = _dump_model(config.optimizer_snapshot)
        versions = {
            "config_version": config.config_version,
            "environment_version": environment_version,
            "observation_schema_version": observation_schema_version,
            "action_space_version": action_space_version,
            "reward_config_version": config.reward_config.version,
            "emotion_schema_version": config.emotion.schema_version,
            "inference_model_artifact_format_version": (
                config.artifact.inference_model_artifact_format_version
            ),
            "dummy_preset_version": config.preset.version,
            "preprocessing_version": preprocessing_version,
            "dataset_split_seed": actual_split_seed,
        }
        compatibility = {
            "environment_version": environment_version,
            "observation_schema_version": observation_schema_version,
            "action_space_version": action_space_version,
            "reward_config_version": config.reward_config.version,
            "emotion_schema_version": config.emotion.schema_version,
            "inference_model_artifact_format_version": (
                config.artifact.inference_model_artifact_format_version
            ),
            "model_architecture": model_architecture,
        }
        return cls(
            job_id=job_id or str(uuid4()),
            mode=mode,
            config_version=config.config_version,
            seed=actual_seed,
            split_seed=actual_split_seed,
            config_snapshot=_freeze(resolved),
            preset_snapshot=_freeze(preset),
            reward_snapshot=_freeze(reward),
            emotion_schema_snapshot=_freeze(emotion),
            model_architecture=_freeze(model_architecture),
            optimizers=_freeze(optimizers),
            versions=_freeze(versions),
            artifact_compatibility=_freeze(compatibility),
            split=_freeze({"seed": actual_split_seed}),
            captured_at=captured_at,
        )


@dataclass(frozen=True)
class TrainingJob:
    """A job handle whose snapshot cannot change during execution."""

    snapshot: JobSnapshot

    @classmethod
    def start(
        cls,
        config: SystemConfig,
        *,
        mode: str = "rl",
        seed: int | None = None,
        split_seed: int | None = None,
        job_id: str | None = None,
        environment_version: str = DEFAULT_ENVIRONMENT_VERSION,
        observation_schema_version: str = DEFAULT_OBSERVATION_SCHEMA_VERSION,
        action_space_version: str = DEFAULT_ACTION_SPACE_VERSION,
        preprocessing_version: str = DEFAULT_PREPROCESSING_VERSION,
        captured_at: str | None = None,
    ) -> "TrainingJob":
        return cls(
            snapshot=JobSnapshot.capture(
                config,
                mode=mode,
                seed=seed,
                split_seed=split_seed,
                job_id=job_id,
                environment_version=environment_version,
                observation_schema_version=observation_schema_version,
                action_space_version=action_space_version,
                preprocessing_version=preprocessing_version,
                captured_at=captured_at,
            )
        )

    @property
    def job_id(self) -> str:
        return self.snapshot.job_id

    def save_snapshot(self, path: str | Path) -> Path:
        return self.snapshot.save(path)


__all__ = [
    "DEFAULT_ACTION_SPACE_VERSION",
    "DEFAULT_ENVIRONMENT_VERSION",
    "DEFAULT_OBSERVATION_SCHEMA_VERSION",
    "DEFAULT_PREPROCESSING_VERSION",
    "JobSnapshot",
    "SNAPSHOT_FORMAT_VERSION",
    "TrainingJob",
]
