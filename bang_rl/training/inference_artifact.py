"""Unified, inference-only Phase-2 module artifact.

The Phase-2 bundle is separate from the final Phase-3 ``.bangmodel``. It
contains only supervised inference weights and public-input contracts; it
never contains optimizer, checkpoint, rollout, environment, or worker state.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from bang_rl.agents.main_agent import (
    EMOTION_INFERENCE_FORMAT_VERSION,
    INTENT_INFERENCE_FORMAT_VERSION,
    EmotionInferenceModule,
    IntentInferenceModule,
    ObservationPreprocessor,
    RoleFusion,
)
from bang_rl.config.snapshot import JobSnapshot
from bang_rl.exceptions import ArtifactCompatibilityError, TrainingError

PHASE2_ARTIFACT_TYPE = "phase2_inference_module"
PHASE2_ARTIFACT_FORMAT_VERSION = "1.0"
PHASE2_ARTIFACT_EXTENSION = ".phase2model"
_MANIFEST_FILE = "manifest.json"
_METADATA_FILE = "metadata.json"
_WEIGHTS_FILE = "model.safetensors"
_BUNDLE_FILES = frozenset({_MANIFEST_FILE, _METADATA_FILE, _WEIGHTS_FILE})
VERSION_MIGRATION_POLICY = {
    "current_version": PHASE2_ARTIFACT_FORMAT_VERSION,
    "supported_source_versions": [PHASE2_ARTIFACT_FORMAT_VERSION],
    "automatic_migration": False,
    "incompatible_versions": "reject with ArtifactCompatibilityError",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, JobSnapshot):
        return value.to_dict()
    if isinstance(value, Mapping):
        return _json_safe(dict(value))
    raise TypeError(f"{name} must be a mapping or JobSnapshot")


def _path_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _mismatch(field: str, actual: Any, expected: Any, source: Path | None = None) -> None:
    raise ArtifactCompatibilityError(field, actual, expected, artifact=str(source) if source else None)


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _module_metadata(module: Any, name: str) -> dict[str, Any]:
    if not hasattr(module, "to_dict") or not hasattr(module, "metadata"):
        raise TypeError(f"{name} does not expose versioned inference metadata")
    return _json_safe(module.to_dict())


def _trainer_metadata(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "artifact_metadata"):
        value = value.artifact_metadata
    if not isinstance(value, Mapping):
        raise TypeError("trainer metadata must be a mapping or TrainingResult")
    return value


def _preprocessor_config(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingError("Phase-2 preprocessing metadata is malformed")
    config = value.get("config", value)
    if not isinstance(config, Mapping):
        raise TrainingError("Phase-2 preprocessing config is malformed")
    return config


def _build_metadata(
    intent: IntentInferenceModule,
    emotion: EmotionInferenceModule,
    *,
    role_fusion: RoleFusion,
    preprocessor: ObservationPreprocessor,
    intent_metadata: Any,
    emotion_metadata: Any,
    job_snapshot: JobSnapshot | Mapping[str, Any] | None,
    config_snapshot: Mapping[str, Any] | Any | None,
    config_hash: str | None,
    source_hashes: Mapping[str, Any] | None,
    extra_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    snapshot = None if job_snapshot is None else _mapping(job_snapshot, "job_snapshot")
    if config_snapshot is None:
        config = None
    elif hasattr(config_snapshot, "model_dump"):
        config = _json_safe(config_snapshot.model_dump(mode="json"))
    elif hasattr(config_snapshot, "to_dict"):
        config = _json_safe(config_snapshot.to_dict())
    else:
        config = _mapping(config_snapshot, "config_snapshot")
    config_hash = config_hash or _sha256(_canonical_json(snapshot if snapshot is not None else config or {}))
    intent_source = _trainer_metadata(intent_metadata)
    emotion_source = _trainer_metadata(emotion_metadata)
    merged_hashes: dict[str, Any] = {}
    if intent_source is not None and isinstance(intent_source.get("source_hashes"), Mapping):
        merged_hashes["intent"] = _json_safe(intent_source["source_hashes"])
    if emotion_source is not None and isinstance(emotion_source.get("source_hashes"), Mapping):
        merged_hashes["emotion"] = _json_safe(emotion_source["source_hashes"])
    if source_hashes is not None:
        merged_hashes.update(_json_safe(dict(source_hashes)))

    versions: dict[str, Any] = {}
    if snapshot is not None and isinstance(snapshot.get("versions"), Mapping):
        versions.update(_json_safe(snapshot["versions"]))
    versions.setdefault("inference_model_artifact_format_version", PHASE2_ARTIFACT_FORMAT_VERSION)
    versions.setdefault("observation_schema_version", preprocessor.config.observation_schema_version)
    versions.setdefault("action_space_version", preprocessor.config.action_space_version)
    versions.setdefault("preprocessing_version", preprocessor.config.preprocessing_version)
    versions.setdefault("emotion_schema_version", emotion.emotion_schema_version)
    if config is not None:
        versions.setdefault("config_version", config.get("config_version"))
        reward = config.get("reward_config", config.get("reward", {}))
        if isinstance(reward, Mapping):
            versions.setdefault("reward_config_version", reward.get("version"))

    result: dict[str, Any] = {
        "artifact_type": PHASE2_ARTIFACT_TYPE,
        "artifact_format_version": PHASE2_ARTIFACT_FORMAT_VERSION,
        "inference_model_artifact_format_version": PHASE2_ARTIFACT_FORMAT_VERSION,
        "phase": "phase2_supervised_inference",
        "modules": {
            "intent": {
                "identifier": IntentInferenceModule.module_type,
                "format_version": INTENT_INFERENCE_FORMAT_VERSION,
                "metadata": _module_metadata(intent, "IntentInferenceModule"),
                "source_metadata": _json_safe(intent_source or {}),
            },
            "emotion": {
                "identifier": EmotionInferenceModule.module_type,
                "format_version": EMOTION_INFERENCE_FORMAT_VERSION,
                "metadata": _module_metadata(emotion, "EmotionInferenceModule"),
                "source_metadata": _json_safe(emotion_source or {}),
            },
        },
        "role_fusion": {"identifier": RoleFusion.module_type, "metadata": _json_safe(role_fusion.to_dict())},
        "preprocessing": _json_safe(preprocessor.metadata()),
        "emotion_schema": {
            "version": emotion.emotion_schema_version,
            "count": emotion.emotion_count,
            "names": list(emotion.emotion_names),
            "initial_baseline": emotion.initial_baseline,
        },
        "versions": versions,
        "config_hash": config_hash,
        "config_snapshot": config,
        "job_snapshot": snapshot,
        "source_hashes": merged_hashes,
        "contracts": {
            "intent_optimizer": intent.optimizer_metadata,
            "emotion_optimizer": emotion.optimizer_metadata,
            "intent_architecture": intent.architecture_metadata,
            "emotion_architecture": emotion.architecture_metadata,
            "role_fusion": "confirmed-role hard override; role-count projection not implemented",
            "inputs": ["public event history", "event_valid_mask"],
            "privileged_inputs": "forbidden",
            "inference_only": True,
            "frozen_on_load": True,
        },
        "training_state": {
            "optimizer_state_included": False,
            "checkpoint_state_included": False,
            "rollout_state_included": False,
            "environment_state_included": False,
            "worker_state_included": False,
            "resume_supported": False,
        },
        "version_migration_policy": VERSION_MIGRATION_POLICY,
    }
    if extra_metadata:
        protected = set(result) | {"contracts", "training_state", "modules", "versions"}
        result.update({key: _json_safe(value) for key, value in extra_metadata.items() if key not in protected})
    return result


def _state_dict(intent: IntentInferenceModule, emotion: EmotionInferenceModule) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for prefix, module in (("intent", intent), ("emotion", emotion)):
        for name, tensor in module.state_dict().items():
            if not isinstance(tensor, torch.Tensor):
                raise TrainingError(f"{prefix} state contains a non-tensor value")
            result[f"{prefix}.{name}"] = tensor.detach().cpu().contiguous()
    return result


def _split_state(state: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    intent, emotion = {}, {}
    for name, tensor in state.items():
        if name.startswith("intent."):
            intent[name[7:]] = tensor
        elif name.startswith("emotion."):
            emotion[name[8:]] = tensor
        else:
            raise TrainingError(f"unexpected Phase-2 weight namespace: {name}")
    if not intent or not emotion:
        raise TrainingError("Phase-2 weights must contain both intent and emotion namespaces")
    return intent, emotion


def _validate_metadata(metadata: Mapping[str, Any], source: Path | None = None) -> None:
    if metadata.get("artifact_type") != PHASE2_ARTIFACT_TYPE:
        _mismatch("artifact_type", metadata.get("artifact_type"), PHASE2_ARTIFACT_TYPE, source)
    if metadata.get("artifact_format_version") != PHASE2_ARTIFACT_FORMAT_VERSION:
        _mismatch("artifact_format_version", metadata.get("artifact_format_version"), PHASE2_ARTIFACT_FORMAT_VERSION, source)
    if metadata.get("inference_model_artifact_format_version") != PHASE2_ARTIFACT_FORMAT_VERSION:
        _mismatch("inference_model_artifact_format_version", metadata.get("inference_model_artifact_format_version"), PHASE2_ARTIFACT_FORMAT_VERSION, source)
    state = metadata.get("training_state")
    forbidden = ("optimizer_state_included", "checkpoint_state_included", "rollout_state_included", "environment_state_included", "worker_state_included", "resume_supported")
    if not isinstance(state, Mapping) or any(state.get(field) is not False for field in forbidden):
        raise ArtifactCompatibilityError("training_state", state, "inference-only/no-resume-state", artifact=str(source) if source else None)
    modules = metadata.get("modules")
    if not isinstance(modules, Mapping) or not all(isinstance(modules.get(key), Mapping) for key in ("intent", "emotion")):
        raise TrainingError("Phase-2 metadata must contain intent and emotion modules")
    schema = metadata.get("emotion_schema")
    if not isinstance(schema, Mapping) or not schema.get("version"):
        raise TrainingError("Phase-2 metadata is missing Emotion Schema")


def _atomic_write(destination: Path, files: Mapping[str, bytes]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with ZipFile(raw, "w", compression=ZIP_STORED) as archive:
                for name in (_MANIFEST_FILE, _METADATA_FILE, _WEIGHTS_FILE):
                    archive.writestr(name, files[name])
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise TrainingError(f"failed to atomically write Phase-2 artifact {destination}") from exc


@dataclass
class Phase2InferenceArtifact:
    """Loaded Phase-2 modules and their compatibility metadata."""
    intent: IntentInferenceModule
    emotion: EmotionInferenceModule
    role_fusion: RoleFusion
    preprocessor: ObservationPreprocessor
    metadata: Mapping[str, Any]
    path: Path | None = None

    def __post_init__(self) -> None:
        _freeze(self.intent)
        _freeze(self.emotion)

    @property
    def intent_inference(self) -> IntentInferenceModule:
        return self.intent

    @property
    def emotion_inference(self) -> EmotionInferenceModule:
        return self.emotion

    @property
    def emotion_schema(self) -> Mapping[str, Any]:
        return self.metadata["emotion_schema"]

    def validate_compatibility(self, expected: Mapping[str, Any]) -> None:
        for field, required in expected.items():
            actual = _path_value(self.metadata, str(field))
            if actual != required:
                _mismatch(str(field), actual, required, self.path)


def save_phase2_inference_artifact(
    path: str | os.PathLike[str],
    intent: IntentInferenceModule,
    emotion: EmotionInferenceModule,
    *,
    role_fusion: RoleFusion | None = None,
    preprocessor: ObservationPreprocessor | None = None,
    intent_metadata: Any = None,
    emotion_metadata: Any = None,
    job_snapshot: JobSnapshot | Mapping[str, Any] | None = None,
    config_snapshot: Mapping[str, Any] | Any | None = None,
    config_hash: str | None = None,
    source_hashes: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a Phase-2 bundle with safetensors weights only."""
    if not isinstance(intent, IntentInferenceModule) or not isinstance(emotion, EmotionInferenceModule):
        raise TypeError("intent and emotion must be their inference module types")
    role_fusion = role_fusion or RoleFusion()
    preprocessor = preprocessor or ObservationPreprocessor()
    if not isinstance(role_fusion, RoleFusion) or not isinstance(preprocessor, ObservationPreprocessor):
        raise TypeError("role_fusion and preprocessor have incompatible types")
    if intent.input_dim != emotion.input_dim or intent.input_dim != preprocessor.input_dim:
        raise ArtifactCompatibilityError("input_dim", (intent.input_dim, emotion.input_dim), preprocessor.input_dim)
    destination = Path(path)
    bundle_metadata = _build_metadata(intent, emotion, role_fusion=role_fusion, preprocessor=preprocessor, intent_metadata=intent_metadata, emotion_metadata=emotion_metadata, job_snapshot=job_snapshot, config_snapshot=config_snapshot, config_hash=config_hash, source_hashes=source_hashes, extra_metadata=metadata)
    _validate_metadata(bundle_metadata, destination)
    weights = save_safetensors(_state_dict(intent, emotion))
    metadata_bytes = _canonical_json(bundle_metadata)
    manifest = {
        "artifact_type": PHASE2_ARTIFACT_TYPE,
        "inference_model_artifact_format_version": PHASE2_ARTIFACT_FORMAT_VERSION,
        "artifact_format_version": PHASE2_ARTIFACT_FORMAT_VERSION,
        "weights_file": _WEIGHTS_FILE,
        "metadata_file": _METADATA_FILE,
        "weights_sha256": _sha256(weights),
        "metadata_sha256": _sha256(metadata_bytes),
        "weight_namespaces": ["intent", "emotion"],
        "files": [_MANIFEST_FILE, _METADATA_FILE, _WEIGHTS_FILE],
    }
    _atomic_write(destination, {_MANIFEST_FILE: _canonical_json(manifest), _METADATA_FILE: metadata_bytes, _WEIGHTS_FILE: weights})
    return destination


def _read_bundle(source: Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    try:
        with ZipFile(source, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != _BUNDLE_FILES:
                raise ArtifactCompatibilityError("bundle_files", names, sorted(_BUNDLE_FILES), artifact=str(source))
            if archive.testzip() is not None:
                raise TrainingError(f"corrupt Phase-2 artifact {source}")
            manifest_bytes, metadata_bytes, weights = (archive.read(name) for name in (_MANIFEST_FILE, _METADATA_FILE, _WEIGHTS_FILE))
    except BadZipFile as exc:
        raise TrainingError(f"cannot read Phase-2 artifact {source}") from exc
    try:
        manifest, metadata = json.loads(manifest_bytes), json.loads(metadata_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Phase-2 artifact JSON is malformed: {source}") from exc
    if not isinstance(manifest, Mapping) or not isinstance(metadata, Mapping):
        raise TrainingError("Phase-2 manifest and metadata must be JSON objects")
    if manifest.get("artifact_type") != PHASE2_ARTIFACT_TYPE:
        _mismatch("manifest.artifact_type", manifest.get("artifact_type"), PHASE2_ARTIFACT_TYPE, source)
    if manifest.get("weights_file") != _WEIGHTS_FILE or manifest.get("metadata_file") != _METADATA_FILE:
        raise ArtifactCompatibilityError("manifest.file_names", manifest, "model.safetensors/metadata.json", artifact=str(source))
    if manifest.get("inference_model_artifact_format_version") != PHASE2_ARTIFACT_FORMAT_VERSION:
        _mismatch("manifest.inference_model_artifact_format_version", manifest.get("inference_model_artifact_format_version"), PHASE2_ARTIFACT_FORMAT_VERSION, source)
    if manifest.get("weights_sha256") != _sha256(weights):
        _mismatch("weights_sha256", manifest.get("weights_sha256"), _sha256(weights), source)
    if manifest.get("metadata_sha256") != _sha256(metadata_bytes):
        _mismatch("metadata_sha256", manifest.get("metadata_sha256"), _sha256(metadata_bytes), source)
    _validate_metadata(metadata, source)
    return dict(manifest), dict(metadata), weights


def load_phase2_inference_artifact(path: str | os.PathLike[str], *, expected: Mapping[str, Any] | None = None, map_location: str | torch.device = "cpu") -> Phase2InferenceArtifact:
    """Checksum-verify, reconstruct, compatibility-check, and freeze a bundle."""
    source = Path(path)
    if not source.is_file():
        raise TrainingError(f"Phase-2 artifact does not exist: {source}")
    _, metadata, weights_bytes = _read_bundle(source)
    if expected:
        for field, required in expected.items():
            actual = _path_value(metadata, str(field))
            if actual != required:
                _mismatch(str(field), actual, required, source)
    try:
        intent = IntentInferenceModule.from_dict(metadata["modules"]["intent"]["metadata"])
        emotion = EmotionInferenceModule.from_dict(metadata["modules"]["emotion"]["metadata"])
        preprocessor = ObservationPreprocessor(_preprocessor_config(metadata["preprocessing"]))
        intent.preprocessor = preprocessor
        emotion.preprocessor = preprocessor
        # ``safetensors.torch.load`` returns CPU tensors in the supported
        # project version; the reconstructed modules are moved below.
        states = load_safetensors(weights_bytes)
        intent_state, emotion_state = _split_state(states)
        intent.load_state_dict(intent_state, strict=True)
        emotion.load_state_dict(emotion_state, strict=True)
        role_fusion = RoleFusion.from_dict(metadata["role_fusion"]["metadata"])
    except ArtifactCompatibilityError:
        raise
    except Exception as exc:
        raise ArtifactCompatibilityError("module_state", "malformed", "compatible Phase-2 modules", artifact=str(source)) from exc
    artifact = Phase2InferenceArtifact(intent, emotion, role_fusion, preprocessor, metadata, source)
    artifact.intent.to(map_location)
    artifact.emotion.to(map_location)
    return artifact


def migrate_phase2_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the Phase-2 version policy without guessing schema migrations.

    Version 1.0 is already canonical. Any other version is rejected until an
    explicit migration handler is reviewed and added; tensor or schema
    changes are never silently reinterpretated.
    """
    value = _json_safe(dict(metadata))
    version = value.get("artifact_format_version")
    if version != PHASE2_ARTIFACT_FORMAT_VERSION:
        _mismatch("artifact_format_version", version, PHASE2_ARTIFACT_FORMAT_VERSION)
    return value


def save_phase2_artifact(*args: Any, **kwargs: Any) -> Path:
    return save_phase2_inference_artifact(*args, **kwargs)


def load_phase2_artifact(*args: Any, **kwargs: Any) -> Phase2InferenceArtifact:
    return load_phase2_inference_artifact(*args, **kwargs)


InferenceModuleArtifact = Phase2InferenceArtifact
save_inference_module_artifact = save_phase2_inference_artifact
load_inference_module_artifact = load_phase2_inference_artifact
save_artifact = save_phase2_inference_artifact
load_artifact = load_phase2_inference_artifact

__all__ = [
    "InferenceModuleArtifact", "PHASE2_ARTIFACT_EXTENSION", "PHASE2_ARTIFACT_FORMAT_VERSION", "PHASE2_ARTIFACT_TYPE", "Phase2InferenceArtifact", "VERSION_MIGRATION_POLICY", "load_artifact", "load_inference_module_artifact", "load_phase2_artifact", "load_phase2_inference_artifact", "migrate_phase2_metadata", "save_artifact", "save_inference_module_artifact", "save_phase2_artifact", "save_phase2_inference_artifact",
]
