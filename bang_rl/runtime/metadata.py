"""Shared, deterministic metadata and report-linkage primitives.

The training, collection, evaluation, application, and artifact layers use the
same small envelope.  It contains identifiers and hashes only; private roles,
labels, and runtime state are intentionally not part of the public envelope.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return json_safe(value.model_dump(mode="json"))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize metadata deterministically for reports, hashes, and tests."""

    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
        return result if isinstance(result, Mapping) else None
    return None


def _snapshot_values(snapshot: Any) -> Mapping[str, Any] | None:
    return _mapping(snapshot)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Stable context shared by every pipeline report and structured record."""

    run_id: str
    job_id: str | None = None
    episode_id: str | None = None
    worker_id: int | str | None = None
    environment_step: int | None = None
    main_agent_transition: int | None = None
    ppo_update: int | None = None
    agent_id: str | int | None = None
    component: str | None = None
    config_version: str | None = None
    snapshot_id: str | None = None
    versions: Mapping[str, Any] = field(default_factory=dict)
    hashes: Mapping[str, Any] = field(default_factory=dict)
    source_hashes: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    hardware: Mapping[str, Any] = field(default_factory=dict)
    counters: Mapping[str, Any] = field(default_factory=dict)
    active_training_seconds: float = 0.0
    paused_seconds: float = 0.0
    paused_time_excluded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return json_safe({
            "run_id": self.run_id,
            "job_id": self.job_id,
            "episode_id": self.episode_id,
            "worker_id": self.worker_id,
            "environment_step": self.environment_step,
            "main_agent_transition": self.main_agent_transition,
            "ppo_update": self.ppo_update,
            "agent_id": self.agent_id,
            "component": self.component,
            "config_version": self.config_version,
            "snapshot_id": self.snapshot_id,
            "versions": dict(self.versions),
            "hashes": dict(self.hashes),
            "source_hashes": dict(self.source_hashes),
            "artifact_id": self.artifact_id,
            "hardware": dict(self.hardware),
            "counters": dict(self.counters),
            "active_training_seconds": float(self.active_training_seconds),
            "paused_seconds": float(self.paused_seconds),
            "paused_time_excluded": bool(self.paused_time_excluded),
        })

    def serialize(self) -> str:
        return canonical_json(self.to_dict())


def build_run_metadata(
    *,
    run_id: str,
    job_id: str | None = None,
    episode_id: str | None = None,
    worker_id: int | str | None = None,
    environment_step: int | None = None,
    main_agent_transition: int | None = None,
    ppo_update: int | None = None,
    agent_id: str | int | None = None,
    component: str | None = None,
    snapshot: Any | None = None,
    config: Any | None = None,
    artifact: Any | None = None,
    source_hashes: Mapping[str, Any] | None = None,
    counters: Mapping[str, Any] | None = None,
    active_training_seconds: float = 0.0,
    paused_seconds: float = 0.0,
    hardware: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical envelope from a JobSnapshot/config/artifact.

    Hashes are over canonical JSON.  Artifact linkage uses the artifact file
    digest when a path is supplied, otherwise the declared artifact metadata.
    """

    snap = _snapshot_values(snapshot)
    config_values = _mapping(config)
    if config_values is None and snap is not None:
        config_values = _mapping(snap.get("config_snapshot"))
    versions: dict[str, Any] = {}
    if snap is not None and isinstance(snap.get("versions"), Mapping):
        versions.update(json_safe(snap["versions"]))
    if config_values is not None:
        versions.setdefault("config_version", config_values.get("config_version"))
        reward = config_values.get("reward_config", config_values.get("reward", {}))
        if isinstance(reward, Mapping):
            versions.setdefault("reward_config_version", reward.get("version"))
        emotion = config_values.get("emotion", {})
        if isinstance(emotion, Mapping):
            versions.setdefault("emotion_schema_version", emotion.get("schema_version"))
    versions = {key: value for key, value in versions.items() if value is not None}
    hashes: dict[str, Any] = {}
    if config_values is not None:
        hashes["config_snapshot_sha256"] = content_hash(config_values)
    if snap is not None:
        hashes["job_snapshot_sha256"] = content_hash(snap)
    sources = dict(json_safe(source_hashes or {}))
    artifact_id = None
    if artifact is not None:
        artifact_path = getattr(artifact, "path", artifact if isinstance(artifact, (str, Path)) else None)
        if artifact_path is not None and Path(artifact_path).is_file():
            artifact_id = file_hash(artifact_path)
            sources.setdefault("artifact_sha256", artifact_id)
        else:
            artifact_values = _mapping(getattr(artifact, "metadata", artifact))
            if artifact_values is not None:
                artifact_id = str(artifact_values.get("artifact_id") or content_hash(artifact_values))
                declared = artifact_values.get("source_hashes")
                if isinstance(declared, Mapping):
                    sources.update(json_safe(declared))
    result = RunMetadata(
        run_id=str(run_id), job_id=None if job_id is None else str(job_id),
        episode_id=None if episode_id is None else str(episode_id), worker_id=worker_id,
        environment_step=environment_step, main_agent_transition=main_agent_transition,
        ppo_update=ppo_update, agent_id=agent_id, component=component,
        config_version=(None if snap is None else snap.get("config_version")) or versions.get("config_version"),
        snapshot_id=None if snap is None else snap.get("job_id"), versions=versions,
        hashes=hashes, source_hashes=sources, artifact_id=artifact_id,
        hardware=json_safe(hardware or (config_values or {}).get("hardware", {})),
        counters=json_safe(counters or {}), active_training_seconds=float(active_training_seconds),
        paused_seconds=float(paused_seconds), paused_time_excluded=True,
    )
    return result.to_dict()


__all__ = ["RunMetadata", "build_run_metadata", "canonical_json", "content_hash", "file_hash", "json_safe"]
