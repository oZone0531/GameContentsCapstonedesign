"""Strict compatibility policy for inference artifacts.

This module is intentionally independent from model reconstruction.  It validates
what may be executed before a loader creates torch modules, and it never guesses
at migrations or silently changes a snapshot/config contract.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from bang_rl.config.schema import ModelArchitectureSnapshot, OptimizerSnapshot, SystemConfig
from bang_rl.config.snapshot import (
    DEFAULT_ACTION_SPACE_VERSION,
    DEFAULT_ENVIRONMENT_VERSION,
    DEFAULT_OBSERVATION_SCHEMA_VERSION,
    DEFAULT_PREPROCESSING_VERSION,
    JobSnapshot,
)
from bang_rl.exceptions import ArtifactCompatibilityError

SUPPORTED_ARTIFACT_TYPE = "inference"
SUPPORTED_ARTIFACT_FORMAT_VERSION = "1.0"
REQUIRED_BUNDLE_FILES = ("manifest.json", "metadata.json", "model.safetensors")
VERSION_FIELDS = (
    "environment_version",
    "observation_schema_version",
    "action_space_version",
    "reward_config_version",
    "emotion_schema_version",
    "inference_model_artifact_format_version",
    "preprocessing_version",
)
_REQUIRED_METADATA_FIELDS = (
    "artifact_type",
    "artifact_format_version",
    "inference_model_artifact_format_version",
    "architecture",
    "modules",
    "preprocessing",
    "role_fusion",
    "versions",
    "model_architecture",
    "optimizers",
    "training",
    "evaluation",
    "random_seed",
    "dataset_split_seed",
    "hardware",
    "execution_statistics",
    "contracts",
    "training_state",
    "version_migration_policy",
    "config_hash",
    "hashes",
    "run_metadata",
)
_REQUIRED_MODULES = ("intent", "emotion", "ppo")
_FORBIDDEN_STATE_KEYS = frozenset(
    {
        "optimizer_state",
        "optimizer_state_dict",
        "rollout_state",
        "rollout_buffer",
        "buffer",
        "checkpoint_state",
        "environment_state",
        "worker_state",
        "resume_state",
    }
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def snapshot_mapping(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, JobSnapshot):
        return value.to_dict()
    if isinstance(value, Mapping):
        return _json_safe(dict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return _json_safe(dict(result))
    raise TypeError(f"{name} must be a mapping, JobSnapshot, or model with model_dump()")


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value)).hexdigest()


def _path_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _mismatch(field: str, actual: Any, expected: Any, source: Path | None = None) -> None:
    raise ArtifactCompatibilityError(
        field,
        actual,
        expected,
        artifact=str(source) if source is not None else None,
    )


def _expect_field(metadata: Mapping[str, Any], field: str, expected: Any, source: Path | None) -> None:
    actual = _path_value(metadata, field)
    if actual != expected:
        _mismatch(field, actual, expected, source)


def _fixed_architecture() -> dict[str, Any]:
    return ModelArchitectureSnapshot().model_dump(mode="json")


def _fixed_optimizers() -> dict[str, Any]:
    values = OptimizerSnapshot().model_dump(mode="json")
    # Model metadata calls the optimizer identifier ``type`` while the
    # Pydantic snapshot calls it ``name``; these are the same contract field.
    return {
        key: {**{field: value for field, value in item.items() if field != "name"}, "type": item["name"]}
        for key, item in values.items()
    }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _find_forbidden_state_keys(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            child = f"{path}.{name}" if path else name
            if name in _FORBIDDEN_STATE_KEYS:
                return child
            found = _find_forbidden_state_keys(item, child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_forbidden_state_keys(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _config_contract(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return expected versions and fixed contracts from a SystemConfig."""
    if not isinstance(value, SystemConfig):
        raise TypeError("config must be a validated SystemConfig")
    return (
        {
            "config_version": value.config_version,
            "environment_version": DEFAULT_ENVIRONMENT_VERSION,
            "observation_schema_version": DEFAULT_OBSERVATION_SCHEMA_VERSION,
            "action_space_version": DEFAULT_ACTION_SPACE_VERSION,
            "reward_config_version": value.reward_config.version,
            "emotion_schema_version": value.emotion.schema_version,
            "inference_model_artifact_format_version": SUPPORTED_ARTIFACT_FORMAT_VERSION,
            "preprocessing_version": DEFAULT_PREPROCESSING_VERSION,
        },
        {
            "model_architecture": value.model_architecture.model_dump(mode="json"),
            "optimizers": value.optimizer_snapshot.model_dump(mode="json"),
        },
    )


@dataclass(frozen=True)
class CompatibilityExpectation:
    """Resolved runtime expectations used by the strict validator."""

    versions: Mapping[str, Any]
    model_architecture: Mapping[str, Any] | None = None
    optimizers: Mapping[str, Any] | None = None
    config_snapshot: Mapping[str, Any] | None = None
    job_snapshot: Mapping[str, Any] | None = None
    additional: Mapping[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str | None:
        return content_hash(self.config_snapshot) if self.config_snapshot is not None else None

    @property
    def job_snapshot_hash(self) -> str | None:
        return content_hash(self.job_snapshot) if self.job_snapshot is not None else None


def build_expectation(
    *,
    config: SystemConfig | None = None,
    expected_snapshot: JobSnapshot | Mapping[str, Any] | None = None,
    config_snapshot: Mapping[str, Any] | Any | None = None,
    expected: Mapping[str, Any] | None = None,
) -> CompatibilityExpectation:
    versions: dict[str, Any] = {"inference_model_artifact_format_version": SUPPORTED_ARTIFACT_FORMAT_VERSION}
    architecture: Mapping[str, Any] | None = None
    optimizers: Mapping[str, Any] | None = None
    resolved_config: Mapping[str, Any] | None = None
    resolved_job: Mapping[str, Any] | None = None
    additional: dict[str, Any] = {}

    if config is not None:
        config_versions, contracts = _config_contract(config)
        versions.update(config_versions)
        architecture = contracts["model_architecture"]
        optimizers = contracts["optimizers"]
        resolved_config = snapshot_mapping(config, "config")
    if config_snapshot is not None:
        resolved_config = snapshot_mapping(config_snapshot, "config_snapshot")
    if expected_snapshot is not None:
        resolved_job = snapshot_mapping(expected_snapshot, "expected_snapshot")
        snapshot_versions = resolved_job.get("versions", {})
        if isinstance(snapshot_versions, Mapping):
            versions.update({str(key): value for key, value in snapshot_versions.items() if key in VERSION_FIELDS or key == "config_version"})
        compatibility = resolved_job.get("artifact_compatibility", {})
        if isinstance(compatibility, Mapping):
            versions.update({str(key): value for key, value in compatibility.items() if key in VERSION_FIELDS})
            if isinstance(compatibility.get("model_architecture"), Mapping):
                architecture = _json_safe(compatibility["model_architecture"])
        if isinstance(resolved_job.get("model_architecture"), Mapping):
            architecture = _json_safe(resolved_job["model_architecture"])
        if isinstance(resolved_job.get("optimizers"), Mapping):
            optimizers = _json_safe(resolved_job["optimizers"])
        if resolved_config is None and isinstance(resolved_job.get("config_snapshot"), Mapping):
            resolved_config = _json_safe(resolved_job["config_snapshot"])

    for field, required in (expected or {}).items():
        normalized = str(field)
        if normalized.startswith("versions."):
            versions[normalized.split(".", 1)[1]] = required
        elif normalized in VERSION_FIELDS or normalized == "config_version":
            versions[normalized] = required
        elif normalized == "model_architecture":
            architecture = required
        elif normalized == "optimizers":
            optimizers = required
        else:
            additional[normalized] = required

    return CompatibilityExpectation(versions, architecture, optimizers, resolved_config, resolved_job, additional)


class ArtifactCompatibilityValidator:
    """Validate final inference metadata against a runtime contract."""

    def __init__(self, expectation: CompatibilityExpectation | None = None, *, source: Path | None = None) -> None:
        self.expectation = expectation or CompatibilityExpectation(
            versions={"inference_model_artifact_format_version": SUPPORTED_ARTIFACT_FORMAT_VERSION},
            model_architecture=None,
            optimizers=None,
        )
        self.source = source

    def validate_manifest(self, manifest: Mapping[str, Any], actual_files: tuple[str, ...] | None = None) -> None:
        if not isinstance(manifest, Mapping):
            _mismatch("manifest", type(manifest).__name__, "JSON object", self.source)
        _expect_field(manifest, "artifact_type", SUPPORTED_ARTIFACT_TYPE, self.source)
        _expect_field(manifest, "inference_model_artifact_format_version", SUPPORTED_ARTIFACT_FORMAT_VERSION, self.source)
        _expect_field(manifest, "artifact_format_version", SUPPORTED_ARTIFACT_FORMAT_VERSION, self.source)
        _expect_field(manifest, "weights_file", "model.safetensors", self.source)
        _expect_field(manifest, "metadata_file", "metadata.json", self.source)
        _expect_field(manifest, "files", list(REQUIRED_BUNDLE_FILES), self.source)
        _expect_field(manifest, "weight_namespaces", ["intent", "emotion", "ppo"], self.source)
        for field in ("weights_sha256", "metadata_sha256"):
            if not _is_sha256(manifest.get(field)):
                _mismatch(f"manifest.{field}", manifest.get(field), "64 lowercase hexadecimal characters", self.source)
        if actual_files is not None and tuple(actual_files) != REQUIRED_BUNDLE_FILES:
            _mismatch("bundle_files", list(actual_files), list(REQUIRED_BUNDLE_FILES), self.source)

    def validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        if not isinstance(metadata, Mapping):
            _mismatch("metadata", type(metadata).__name__, "JSON object", self.source)
        for field in _REQUIRED_METADATA_FIELDS:
            if field not in metadata:
                _mismatch(field, None, "required metadata field", self.source)
        for field, required in (
            ("artifact_type", SUPPORTED_ARTIFACT_TYPE),
            ("artifact_format_version", SUPPORTED_ARTIFACT_FORMAT_VERSION),
            ("inference_model_artifact_format_version", SUPPORTED_ARTIFACT_FORMAT_VERSION),
        ):
            _expect_field(metadata, field, required, self.source)
        state = metadata.get("training_state")
        if not isinstance(state, Mapping):
            _mismatch("training_state", state, "inference-only state flags", self.source)
        for field in ("optimizer_state_included", "rollout_state_included", "checkpoint_state_included", "environment_state_included", "worker_state_included", "resume_supported"):
            if state.get(field) is not False:
                _mismatch(f"training_state.{field}", state.get(field), False, self.source)
        forbidden = _find_forbidden_state_keys(metadata)
        if forbidden is not None:
            _mismatch(forbidden, "present", "absent (inference-only artifact)", self.source)
        contracts = metadata.get("contracts")
        if not isinstance(contracts, Mapping) or contracts.get("inference_only") is not True:
            _mismatch("contracts.inference_only", _path_value(metadata, "contracts.inference_only"), True, self.source)
        migration = metadata.get("version_migration_policy")
        if not isinstance(migration, Mapping) or migration.get("automatic_migration") is not False:
            _mismatch("version_migration_policy.automatic_migration", _path_value(metadata, "version_migration_policy.automatic_migration"), False, self.source)

        modules = metadata.get("modules")
        if not isinstance(modules, Mapping):
            _mismatch("modules", modules, "intent/emotion/ppo mappings", self.source)
        for name in _REQUIRED_MODULES:
            module = modules.get(name)
            if not isinstance(module, Mapping):
                _mismatch(f"modules.{name}", module, "module metadata", self.source)
            if module.get("identifier") != {"intent": "IntentInferenceModule", "emotion": "EmotionInferenceModule", "ppo": "PPOModule"}[name]:
                _mismatch(f"modules.{name}.identifier", module.get("identifier"), {"intent": "IntentInferenceModule", "emotion": "EmotionInferenceModule", "ppo": "PPOModule"}[name], self.source)
            if not isinstance(module.get("metadata"), Mapping):
                _mismatch(f"modules.{name}.metadata", module.get("metadata"), "mapping", self.source)

        versions = metadata.get("versions")
        if not isinstance(versions, Mapping):
            _mismatch("versions", versions, "version mapping", self.source)
        for field, required in self.expectation.versions.items():
            if required is not None:
                _expect_field(metadata, f"versions.{field}", required, self.source)
        for field in VERSION_FIELDS:
            if field not in versions:
                _mismatch(f"versions.{field}", None, "required version", self.source)
        for field, required in self.expectation.additional.items():
            _expect_field(metadata, field, required, self.source)

        hashes = metadata.get("hashes")
        if not isinstance(hashes, Mapping):
            _mismatch("hashes", hashes, "config/job snapshot hash mapping", self.source)
        if not _is_sha256(metadata.get("config_hash")):
            _mismatch("config_hash", metadata.get("config_hash"), "64 lowercase hexadecimal characters", self.source)
        if hashes.get("config_snapshot_sha256") != metadata.get("config_hash"):
            _mismatch("hashes.config_snapshot_sha256", hashes.get("config_snapshot_sha256"), metadata.get("config_hash"), self.source)
        run_metadata = metadata.get("run_metadata")
        if not isinstance(run_metadata, Mapping):
            _mismatch("run_metadata", run_metadata, "shared run metadata envelope", self.source)
        run_hashes = run_metadata.get("hashes", {})
        if isinstance(run_hashes, Mapping) and run_hashes.get("config_snapshot_sha256") != metadata.get("config_hash"):
            _mismatch("run_metadata.hashes.config_snapshot_sha256", run_hashes.get("config_snapshot_sha256"), metadata.get("config_hash"), self.source)
        if self.expectation.config_hash is not None:
            _expect_field(metadata, "config_hash", self.expectation.config_hash, self.source)
            _expect_field(metadata, "hashes.config_snapshot_sha256", self.expectation.config_hash, self.source)
        if self.expectation.job_snapshot_hash is not None:
            _expect_field(metadata, "job_snapshot_hash", self.expectation.job_snapshot_hash, self.source)
            _expect_field(metadata, "hashes.job_snapshot_sha256", self.expectation.job_snapshot_hash, self.source)

        if self.expectation.model_architecture is not None:
            _expect_field(metadata, "model_architecture", self.expectation.model_architecture, self.source)
        if self.expectation.optimizers is not None:
            _expect_field(metadata, "optimizers", self.expectation.optimizers, self.source)
        self._validate_fixed_contracts(metadata)

    def _validate_fixed_contracts(self, metadata: Mapping[str, Any]) -> None:
        """Check architecture/optimizer invariants even without a config."""
        modules = metadata["modules"]
        fixed = _fixed_architecture()
        fixed_optimizer = _fixed_optimizers()
        checks = (
            ("modules.intent.metadata.architecture.encoder.encoder_type", "GRU"),
            ("modules.intent.metadata.architecture.encoder.hidden_size", fixed["intent_hidden_size"]),
            ("modules.intent.metadata.architecture.encoder.num_layers", fixed["intent_num_layers"]),
            ("modules.intent.metadata.architecture.encoder.batch_first", fixed["intent_batch_first"]),
            ("modules.intent.metadata.architecture.encoder.dropout", fixed["intent_dropout"]),
            ("modules.emotion.metadata.architecture.encoder.encoder_type", "GRU"),
            ("modules.emotion.metadata.architecture.encoder.hidden_size", fixed["emotion_hidden_size"]),
            ("modules.emotion.metadata.architecture.encoder.num_layers", fixed["emotion_num_layers"]),
            ("modules.emotion.metadata.architecture.encoder.batch_first", fixed["emotion_batch_first"]),
            ("modules.emotion.metadata.architecture.encoder.dropout", fixed["emotion_dropout"]),
            ("modules.ppo.metadata.architecture.shared_trunk.layers.0.out_features", 256),
            ("modules.ppo.metadata.architecture.shared_trunk.layers.2.out_features", 256),
            ("modules.ppo.metadata.architecture.policy_head.layers.0.out_features", 6),
            ("modules.ppo.metadata.architecture.value_head.layers.0.out_features", 1),
        )
        for field, required in checks:
            _expect_field(metadata, field, required, self.source)
        for name in _REQUIRED_MODULES:
            actual_optimizer = _path_value(metadata, f"modules.{name}.metadata.architecture.optimizer")
            expected_optimizer = fixed_optimizer["ppo" if name == "ppo" else name]
            if actual_optimizer != expected_optimizer:
                _mismatch(f"modules.{name}.metadata.architecture.optimizer", actual_optimizer, expected_optimizer, self.source)


def validate_final_artifact_metadata(
    metadata: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    config: SystemConfig | None = None,
    expected_snapshot: JobSnapshot | Mapping[str, Any] | None = None,
    config_snapshot: Mapping[str, Any] | Any | None = None,
    expected: Mapping[str, Any] | None = None,
    source: Path | None = None,
) -> CompatibilityExpectation:
    expectation = build_expectation(
        config=config,
        expected_snapshot=expected_snapshot,
        config_snapshot=config_snapshot,
        expected=expected,
    )
    validator = ArtifactCompatibilityValidator(expectation, source=source)
    if manifest is not None:
        validator.validate_manifest(manifest)
    validator.validate_metadata(metadata)
    return expectation


class InferenceCompatibilityService:
    """Application-facing compatibility gate for final game inference."""

    def validate(self, path: str | Path, **kwargs: Any) -> dict[str, Any]:
        from .final_inference_artifact import read_final_artifact_metadata

        manifest, metadata = read_final_artifact_metadata(path)
        validate_final_artifact_metadata(metadata, manifest=manifest, source=Path(path), **kwargs)
        return metadata

    def load(self, path: str | Path, **kwargs: Any) -> Any:
        from .final_inference_artifact import load_final_inference_artifact

        return load_final_inference_artifact(path, **kwargs)


CompatibilityService = InferenceCompatibilityService

__all__ = [
    "ArtifactCompatibilityValidator",
    "CompatibilityExpectation",
    "CompatibilityService",
    "InferenceCompatibilityService",
    "SUPPORTED_ARTIFACT_FORMAT_VERSION",
    "SUPPORTED_ARTIFACT_TYPE",
    "build_expectation",
    "canonical_json",
    "content_hash",
    "validate_final_artifact_metadata",
]
