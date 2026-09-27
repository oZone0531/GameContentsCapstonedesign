"""Final Phase-3 inference-only ``.bangmodel`` artifact.

The final bundle is deliberately separate from the Phase-2 supervised module
bundle.  It is the only disk representation of the trained game policy: a
three-file ZIP containing manifest metadata and safetensors weights.  No
optimizer, rollout, worker, environment, checkpoint, or resume state is
serialized.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
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
    PPO_MODULE_FORMAT_VERSION,
    EmotionInferenceModule,
    IntentInferenceModule,
    ObservationPreprocessor,
    PPOModule,
    RoleFusion,
)
from bang_rl.config.snapshot import JobSnapshot
from bang_rl.exceptions import ArtifactCompatibilityError, TrainingError
from bang_rl.runtime.metadata import build_run_metadata

FINAL_ARTIFACT_TYPE = "inference"
FINAL_ARTIFACT_FORMAT_VERSION = "1.0"
FINAL_ARTIFACT_EXTENSION = ".bangmodel"
_MANIFEST_FILE = "manifest.json"
_METADATA_FILE = "metadata.json"
_WEIGHTS_FILE = "model.safetensors"
_BUNDLE_FILES = frozenset({_MANIFEST_FILE, _METADATA_FILE, _WEIGHTS_FILE})

VERSION_MIGRATION_POLICY = {
    "current_version": FINAL_ARTIFACT_FORMAT_VERSION,
    "supported_source_versions": [FINAL_ARTIFACT_FORMAT_VERSION],
    "automatic_migration": False,
    "incompatible_versions": "reject with ArtifactCompatibilityError",
}


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
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
    raise TypeError(f"{name} must be a mapping, JobSnapshot, model_dump(), or to_dict object")


def _path_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _mismatch(field: str, actual: Any, expected: Any, source: Path | None = None) -> None:
    raise ArtifactCompatibilityError(field, actual, expected, artifact=str(source) if source else None)


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def freeze_inference_modules(*modules: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    """Put inference modules in eval mode and disable all gradients."""

    if not modules:
        raise ValueError("at least one inference module is required")
    return tuple(_freeze(module) for module in modules)


def _log(logger: Any | None, event: str, **fields: Any) -> None:
    if logger is None:
        return
    try:
        if hasattr(logger, "event"):
            logger.event(event, **fields)
        elif hasattr(logger, "info"):
            logger.info(event, **fields)
        elif callable(logger):
            logger({"event": event, **fields})
    except Exception:
        # Artifact logging must not turn a successful atomic save into a
        # failed save (and must never hide the original save exception).
        pass


def _module_metadata(module: Any, name: str) -> dict[str, Any]:
    if not isinstance(module, torch.nn.Module) or not hasattr(module, "to_dict"):
        raise TypeError(f"{name} must be a versioned torch module")
    return _json_safe(module.to_dict())


def _phase2_components(
    phase2_artifact: Any | None,
    phase2_path: str | os.PathLike[str] | None,
) -> tuple[IntentInferenceModule, EmotionInferenceModule, RoleFusion, ObservationPreprocessor, Mapping[str, Any] | None, Path | None]:
    if phase2_artifact is None and phase2_path is not None:
        from .inference_artifact import load_phase2_inference_artifact

        phase2_artifact = load_phase2_inference_artifact(phase2_path)
    if phase2_artifact is None:
        raise ValueError("phase2_artifact or phase2_path is required")
    intent = getattr(phase2_artifact, "intent", getattr(phase2_artifact, "intent_inference", None))
    emotion = getattr(phase2_artifact, "emotion", getattr(phase2_artifact, "emotion_inference", None))
    role_fusion = getattr(phase2_artifact, "role_fusion", None) or RoleFusion()
    preprocessor = getattr(phase2_artifact, "preprocessor", None) or ObservationPreprocessor()
    metadata = getattr(phase2_artifact, "metadata", None)
    source = getattr(phase2_artifact, "path", None)
    if not isinstance(intent, IntentInferenceModule) or not isinstance(emotion, EmotionInferenceModule):
        raise TypeError("phase2_artifact does not expose IntentInference and EmotionInference modules")
    if not isinstance(role_fusion, RoleFusion) or not isinstance(preprocessor, ObservationPreprocessor):
        raise TypeError("phase2_artifact has incompatible role fusion or preprocessor")
    return intent, emotion, role_fusion, preprocessor, metadata, Path(source) if source else None


def _state_dict(intent: IntentInferenceModule, emotion: EmotionInferenceModule, ppo: PPOModule) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for prefix, module in (("intent", intent), ("emotion", emotion), ("ppo", ppo)):
        state = module.state_dict()
        if not state:
            raise TrainingError(f"{prefix} module has no generated weights")
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise TrainingError(f"{prefix} state contains a non-tensor value")
            result[f"{prefix}.{name}"] = tensor.detach().cpu().contiguous()
    return result


def _split_state(state: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    parts = {"intent": {}, "emotion": {}, "ppo": {}}
    for name, tensor in state.items():
        for prefix in parts:
            marker = prefix + "."
            if name.startswith(marker):
                parts[prefix][name[len(marker):]] = tensor
                break
        else:
            raise TrainingError(f"unexpected final artifact weight namespace: {name}")
    if any(not value for value in parts.values()):
        raise TrainingError("final artifact weights must contain intent, emotion, and ppo namespaces")
    return parts["intent"], parts["emotion"], parts["ppo"]


def _snapshot_versions(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    versions = {}
    if snapshot and isinstance(snapshot.get("versions"), Mapping):
        versions.update(_json_safe(snapshot["versions"]))
    return versions


def _build_metadata(
    intent: IntentInferenceModule,
    emotion: EmotionInferenceModule,
    ppo: PPOModule,
    *,
    role_fusion: RoleFusion,
    preprocessor: ObservationPreprocessor,
    phase2_metadata: Mapping[str, Any] | None,
    phase2_path: Path | None,
    job_snapshot: JobSnapshot | Mapping[str, Any] | None,
    config_snapshot: Mapping[str, Any] | Any | None,
    trainer: Any | None,
    training_metadata: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    snapshot = None if job_snapshot is None else _as_mapping(job_snapshot, "job_snapshot")
    config = None if config_snapshot is None else _as_mapping(config_snapshot, "config_snapshot")
    if config is None and snapshot is not None:
        config = snapshot.get("config_snapshot")
    versions = _snapshot_versions(snapshot)
    if phase2_metadata and isinstance(phase2_metadata.get("versions"), Mapping):
        for key, value in phase2_metadata["versions"].items():
            versions.setdefault(str(key), _json_safe(value))
    versions.setdefault("environment_version", "1.0")
    versions.setdefault("observation_schema_version", preprocessor.config.observation_schema_version)
    versions.setdefault("action_space_version", preprocessor.config.action_space_version)
    versions.setdefault("preprocessing_version", preprocessor.config.preprocessing_version)
    versions.setdefault("emotion_schema_version", emotion.emotion_schema_version)
    versions.setdefault("inference_model_artifact_format_version", FINAL_ARTIFACT_FORMAT_VERSION)
    if config:
        versions.setdefault("config_version", config.get("config_version"))
        reward = config.get("reward_config", config.get("reward", {}))
        if isinstance(reward, Mapping):
            versions.setdefault("reward_config_version", reward.get("version"))
    created_at = datetime.now(timezone.utc).isoformat()
    trainer_meta: dict[str, Any] = {}
    if trainer is not None:
        value = trainer.metadata() if hasattr(trainer, "metadata") else trainer
        if isinstance(value, Mapping):
            trainer_meta = _json_safe(dict(value))
    if training_metadata:
        trainer_meta.update(_json_safe(dict(training_metadata)))
    counters = trainer_meta.get("counters", {}) if isinstance(trainer_meta.get("counters"), Mapping) else {}
    snapshot_hardware = config.get("hardware", {}) if isinstance(config, Mapping) else {}
    preset = snapshot.get("preset_snapshot", {}) if snapshot else {}
    evaluation = trainer_meta.get("evaluation", {}) if isinstance(trainer_meta.get("evaluation"), Mapping) else {}
    source_hashes: dict[str, Any] = {}
    if phase2_path is not None and phase2_path.is_file():
        source_hashes["phase2_artifact_sha256"] = _file_sha256(phase2_path)
    if phase2_metadata and isinstance(phase2_metadata.get("source_hashes"), Mapping):
        source_hashes.update(_json_safe(dict(phase2_metadata["source_hashes"])))
    execution_statistics = {
        key: value
        for key, value in trainer_meta.items()
        if key not in {
            "checkpoint", "rollout", "optimizer_state", "optimizer_state_dict",
            "rollout_state", "buffer", "rollout_buffer", "environment_state", "worker_state",
        }
    }

    architecture = {
        "intent": _module_metadata(intent, "IntentInferenceModule"),
        "emotion": _module_metadata(emotion, "EmotionInferenceModule"),
        "ppo": _module_metadata(ppo, "PPOModule"),
        "role_fusion": _json_safe(role_fusion.to_dict()),
        "preprocessing": _json_safe(preprocessor.metadata()),
    }
    # Hashes are over canonical JSON, not Python reprs.  A missing JobSnapshot
    # is represented explicitly by ``None`` and can never be mistaken for a
    # compatible snapshot when a runtime supplies one at load time.
    config_hash = _sha256(_canonical_json(config if config is not None else {}))
    job_snapshot_hash = _sha256(_canonical_json(snapshot)) if snapshot is not None else None
    result: dict[str, Any] = {
        "artifact_type": FINAL_ARTIFACT_TYPE,
        "artifact_format_version": FINAL_ARTIFACT_FORMAT_VERSION,
        "inference_model_artifact_format_version": FINAL_ARTIFACT_FORMAT_VERSION,
        "phase": "phase3_ppo_inference",
        "created_at": created_at,
        "architecture": architecture,
        "modules": {
            "intent": {"identifier": intent.module_type, "format_version": INTENT_INFERENCE_FORMAT_VERSION, "metadata": architecture["intent"]},
            "emotion": {"identifier": emotion.module_type, "format_version": EMOTION_INFERENCE_FORMAT_VERSION, "metadata": architecture["emotion"]},
            "ppo": {"identifier": ppo.module_type, "format_version": PPO_MODULE_FORMAT_VERSION, "metadata": architecture["ppo"]},
        },
        "preprocessing": architecture["preprocessing"],
        "role_fusion": architecture["role_fusion"],
        "versions": versions,
        "environment_version": versions.get("environment_version"),
        "observation_schema_version": versions.get("observation_schema_version"),
        "action_space_version": versions.get("action_space_version"),
        "reward_config_version": versions.get("reward_config_version"),
        "emotion_schema_version": versions.get("emotion_schema_version"),
        "preprocessing_version": versions.get("preprocessing_version"),
        "source_compatibility": {
            "phase2_artifact_type": phase2_metadata.get("artifact_type") if phase2_metadata else "phase2_inference_module",
            "phase2_artifact_format_version": phase2_metadata.get("artifact_format_version") if phase2_metadata else "1.0",
            "phase2_versions": _json_safe(phase2_metadata.get("versions", {})) if phase2_metadata else {},
            "hashes": source_hashes,
        },
        "source_hashes": source_hashes,
        "config_hash": config_hash,
        "job_snapshot_hash": job_snapshot_hash,
        "run_metadata": build_run_metadata(
            run_id=str((snapshot or {}).get("job_id") or job_snapshot_hash or config_hash),
            job_id=(snapshot or {}).get("job_id"),
            component="final_inference_artifact",
            snapshot=snapshot,
            config=config,
            source_hashes=source_hashes,
            artifact=None,
            counters=counters,
            active_training_seconds=float(trainer_meta.get("total_training_time_seconds", trainer_meta.get("training_time_seconds", 0.0))),
            hardware=snapshot_hardware,
        ),
        "hashes": {
            "config_snapshot_sha256": config_hash,
            "job_snapshot_sha256": job_snapshot_hash,
        },
        "config_snapshot": config,
        "job_snapshot": snapshot,
        "model_architecture": snapshot.get("model_architecture", architecture) if snapshot else architecture,
        "optimizers": snapshot.get("optimizers", {"ppo": ppo.optimizer_metadata}) if snapshot else {"ppo": ppo.optimizer_metadata},
        "training": {
            "ppo_config": _json_safe(getattr(ppo.config, "model_dump", lambda **_: {}) (mode="json")),
            "optimizer": ppo.optimizer_metadata,
            "counters": counters,
            "total_environment_steps": counters.get("environment_steps", 0),
            "total_main_agent_transitions": counters.get("main_agent_transitions", 0),
            "total_training_time_seconds": trainer_meta.get("total_training_time_seconds", trainer_meta.get("training_time_seconds", 0.0)),
            "paused_time_excluded": True,
            "opponent": trainer_meta.get("opponent", trainer_meta.get("training_opponent")),
            "preset": _json_safe(preset),
        },
        "evaluation": evaluation,
        "random_seed": snapshot.get("seed") if snapshot else trainer_meta.get("seed"),
        "dataset_split_seed": snapshot.get("split_seed") if snapshot else trainer_meta.get("split_seed"),
        "dummy_preset": _json_safe(preset),
        "training_opponent": trainer_meta.get("opponent", trainer_meta.get("training_opponent")),
        "total_environment_steps": counters.get("environment_steps", 0),
        "total_main_agent_transitions": counters.get("main_agent_transitions", 0),
        "total_training_time_seconds": trainer_meta.get("total_training_time_seconds", trainer_meta.get("training_time_seconds", 0.0)),
        "hardware": _json_safe(snapshot_hardware),
        "execution_statistics": execution_statistics,
        "contracts": {
            "inference_only": True,
            "action_mask": "required; binary shape (6); zero-probability invalid actions",
            "public_inputs_only": True,
            "privileged_roles": "forbidden",
            "emotion_labels": "forbidden",
            "reward_inputs": "forbidden",
            "phase2_artifact_is_initialization_only": True,
        },
        "training_state": {
            "optimizer_state_included": False,
            "rollout_state_included": False,
            "checkpoint_state_included": False,
            "environment_state_included": False,
            "worker_state_included": False,
            "resume_supported": False,
        },
        "version_migration_policy": VERSION_MIGRATION_POLICY,
    }
    if metadata:
        protected = set(result) | {"training_state", "modules", "architecture", "versions", "source_compatibility", "run_metadata"}
        result.update({str(key): _json_safe(value) for key, value in metadata.items() if str(key) not in protected})
    return result


def _validate_metadata(metadata: Mapping[str, Any], source: Path | None = None) -> None:
    """Run the shared strict structural/contract validator."""

    from .compatibility import validate_final_artifact_metadata

    validate_final_artifact_metadata(metadata, source=source)


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
        raise TrainingError(f"failed to atomically write final inference artifact {destination}") from exc


@dataclass
class PPOInferenceArtifact:
    """Loaded final model with inference-only APIs and no training state."""

    intent: IntentInferenceModule
    emotion: EmotionInferenceModule
    ppo: PPOModule
    role_fusion: RoleFusion
    preprocessor: ObservationPreprocessor
    metadata: Mapping[str, Any]
    path: Path | None = None

    def __post_init__(self) -> None:
        freeze_inference_modules(self.intent, self.emotion, self.ppo)

    @property
    def intent_inference(self) -> IntentInferenceModule:
        return self.intent

    @property
    def emotion_inference(self) -> EmotionInferenceModule:
        return self.emotion

    @property
    def ppo_module(self) -> PPOModule:
        return self.ppo

    @property
    def inference_only(self) -> bool:
        return True

    def __call__(self, inputs: Any, action_mask: Any | None = None, **kwargs: Any) -> Any:
        return self.ppo(inputs, action_mask, **kwargs)

    def forward(self, inputs: Any, action_mask: Any | None = None, **kwargs: Any) -> Any:
        return self.ppo(inputs, action_mask, **kwargs)

    def act(self, inputs: Any, action_mask: Any | None = None, **kwargs: Any) -> Any:
        return self.ppo.act(inputs, action_mask, **kwargs)

    predict = act
    select_action = act

    def evaluate_actions(self, inputs: Any, actions: Any, action_mask: Any | None = None, **kwargs: Any) -> Any:
        return self.ppo.evaluate_actions(inputs, actions, action_mask, **kwargs)

    def validate_compatibility(self, expected: Mapping[str, Any]) -> None:
        for field, required in expected.items():
            actual = _path_value(self.metadata, str(field))
            if actual != required:
                _mismatch(str(field), actual, required, self.path)


def save_final_inference_artifact(
    path: str | os.PathLike[str],
    ppo_module: PPOModule,
    *,
    phase2_artifact: Any | None = None,
    phase2_path: str | os.PathLike[str] | None = None,
    intent: IntentInferenceModule | None = None,
    emotion: EmotionInferenceModule | None = None,
    role_fusion: RoleFusion | None = None,
    preprocessor: ObservationPreprocessor | None = None,
    job_snapshot: JobSnapshot | Mapping[str, Any] | None = None,
    config_snapshot: Mapping[str, Any] | Any | None = None,
    trainer: Any | None = None,
    training_metadata: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    logger: Any | None = None,
) -> Path:
    """Atomically save a final inference-only ``.bangmodel`` bundle.

    Phase-2 modules may be supplied as a loaded artifact or by path.  Direct
    ``intent``/``emotion`` arguments are supported for integrations that have
    already loaded the Phase-2 artifact and are validated identically.
    """

    destination = Path(path)
    try:
        if not isinstance(ppo_module, PPOModule):
            raise TypeError("ppo_module must be a PPOModule")
        if phase2_artifact is not None or phase2_path is not None:
            phase2_intent, phase2_emotion, phase2_role_fusion, phase2_preprocessor, phase2_metadata, source = _phase2_components(phase2_artifact, phase2_path)
            intent = intent or phase2_intent
            emotion = emotion or phase2_emotion
            role_fusion = role_fusion or phase2_role_fusion
            preprocessor = preprocessor or phase2_preprocessor
        else:
            phase2_metadata, source = None, None
        if not isinstance(intent, IntentInferenceModule) or not isinstance(emotion, EmotionInferenceModule):
            raise ValueError("Phase-2 IntentInference and EmotionInference modules are required")
        role_fusion = role_fusion or RoleFusion()
        preprocessor = preprocessor or ObservationPreprocessor()
        if not isinstance(role_fusion, RoleFusion) or not isinstance(preprocessor, ObservationPreprocessor):
            raise TypeError("role_fusion and preprocessor have incompatible types")
        freeze_inference_modules(intent, emotion, ppo_module)
        weights = save_safetensors(_state_dict(intent, emotion, ppo_module))
        bundle_metadata = _build_metadata(intent, emotion, ppo_module, role_fusion=role_fusion, preprocessor=preprocessor, phase2_metadata=phase2_metadata, phase2_path=source, job_snapshot=job_snapshot, config_snapshot=config_snapshot, trainer=trainer, training_metadata=training_metadata, metadata=metadata)
        _validate_metadata(bundle_metadata, destination)
        metadata_bytes = _canonical_json(bundle_metadata)
        manifest = {
            "artifact_type": FINAL_ARTIFACT_TYPE,
            "inference_model_artifact_format_version": FINAL_ARTIFACT_FORMAT_VERSION,
            "artifact_format_version": FINAL_ARTIFACT_FORMAT_VERSION,
            "weights_file": _WEIGHTS_FILE,
            "metadata_file": _METADATA_FILE,
            "weights_sha256": _sha256(weights),
            "metadata_sha256": _sha256(metadata_bytes),
            "weight_namespaces": ["intent", "emotion", "ppo"],
            "files": [_MANIFEST_FILE, _METADATA_FILE, _WEIGHTS_FILE],
        }
        _atomic_write(destination, {_MANIFEST_FILE: _canonical_json(manifest), _METADATA_FILE: metadata_bytes, _WEIGHTS_FILE: weights})
        _log(logger, "inference_artifact_saved", path=str(destination), artifact_type=FINAL_ARTIFACT_TYPE)
        return destination
    except Exception as exc:
        _log(logger, "inference_artifact_save_failed", path=str(destination), error=str(exc))
        raise


def _read_bundle(source: Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    try:
        with ZipFile(source, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != _BUNDLE_FILES:
                raise ArtifactCompatibilityError("bundle_files", names, sorted(_BUNDLE_FILES), artifact=str(source))
            if archive.testzip() is not None:
                raise ArtifactCompatibilityError("bundle", "corrupt ZIP member", "readable ZIP members", artifact=str(source))
            manifest_bytes = archive.read(_MANIFEST_FILE)
            metadata_bytes = archive.read(_METADATA_FILE)
            weights = archive.read(_WEIGHTS_FILE)
    except BadZipFile as exc:
        raise ArtifactCompatibilityError("bundle", "invalid ZIP archive", "ZIP archive", artifact=str(source)) from exc
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError("manifest", "malformed JSON", "JSON object", artifact=str(source)) from exc
    try:
        metadata = json.loads(metadata_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError("metadata", "malformed JSON", "JSON object", artifact=str(source)) from exc
    if not isinstance(manifest, Mapping) or not isinstance(metadata, Mapping):
        raise ArtifactCompatibilityError("manifest", {"manifest": type(manifest).__name__, "metadata": type(metadata).__name__}, "JSON objects", artifact=str(source))
    from .compatibility import ArtifactCompatibilityValidator

    ArtifactCompatibilityValidator(source=source).validate_manifest(manifest)
    if manifest.get("weights_sha256") != _sha256(weights):
        _mismatch("weights_sha256", manifest.get("weights_sha256"), _sha256(weights), source)
    if manifest.get("metadata_sha256") != _sha256(metadata_bytes):
        _mismatch("metadata_sha256", manifest.get("metadata_sha256"), _sha256(metadata_bytes), source)
    _validate_metadata(metadata, source)
    return dict(manifest), dict(metadata), weights


def read_final_artifact_metadata(path: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and checksum-verify only the manifest and metadata contract."""

    source = Path(path)
    if not source.is_file():
        raise TrainingError(f"final inference artifact does not exist: {source}")
    manifest, metadata, _ = _read_bundle(source)
    return manifest, metadata


def load_final_inference_artifact(
    path: str | os.PathLike[str],
    *,
    expected: Mapping[str, Any] | None = None,
    expected_snapshot: JobSnapshot | Mapping[str, Any] | None = None,
    config: Any | None = None,
    config_snapshot: Mapping[str, Any] | Any | None = None,
    map_location: str | torch.device = "cpu",
) -> PPOInferenceArtifact:
    """Load, checksum-verify, compatibility-check, and freeze a bundle."""

    source = Path(path)
    if not source.is_file():
        raise TrainingError(f"final inference artifact does not exist: {source}")
    manifest, metadata, weights_bytes = _read_bundle(source)
    from .compatibility import validate_final_artifact_metadata

    validate_final_artifact_metadata(
        metadata,
        manifest=manifest,
        config=config,
        expected_snapshot=expected_snapshot,
        config_snapshot=config_snapshot,
        expected=expected,
        source=source,
    )
    try:
        intent = IntentInferenceModule.from_dict(metadata["modules"]["intent"]["metadata"])
        emotion = EmotionInferenceModule.from_dict(metadata["modules"]["emotion"]["metadata"])
        preprocessor_config = metadata["preprocessing"].get("config", metadata["preprocessing"])
        preprocessor = ObservationPreprocessor(preprocessor_config)
        intent.preprocessor = preprocessor
        emotion.preprocessor = preprocessor
        ppo = PPOModule.from_dict(metadata["modules"]["ppo"]["metadata"])
        states = load_safetensors(weights_bytes)
        intent_state, emotion_state, ppo_state = _split_state(states)
        intent.load_state_dict(intent_state, strict=True)
        emotion.load_state_dict(emotion_state, strict=True)
        ppo.load_state_dict(ppo_state, strict=True)
        role_fusion = RoleFusion.from_dict(metadata["role_fusion"])
    except ArtifactCompatibilityError:
        raise
    except Exception as exc:
        raise ArtifactCompatibilityError("module_state", "malformed", "compatible final inference modules", artifact=str(source)) from exc
    intent.to(map_location)
    emotion.to(map_location)
    ppo.to(map_location)
    return PPOInferenceArtifact(intent, emotion, ppo, role_fusion, preprocessor, metadata, source)


def migrate_final_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Reject unknown versions; no implicit tensor/schema migration exists."""

    value = _json_safe(dict(metadata))
    if value.get("artifact_format_version") != FINAL_ARTIFACT_FORMAT_VERSION:
        _mismatch("artifact_format_version", value.get("artifact_format_version"), FINAL_ARTIFACT_FORMAT_VERSION)
    if value.get("inference_model_artifact_format_version") != FINAL_ARTIFACT_FORMAT_VERSION:
        _mismatch("inference_model_artifact_format_version", value.get("inference_model_artifact_format_version"), FINAL_ARTIFACT_FORMAT_VERSION)
    return value


# Explicit names are preferred; these aliases make the format convenient for
# callers that refer to the extension or to a model bundle rather than phase.
INFERENCE_ARTIFACT_FORMAT_VERSION = FINAL_ARTIFACT_FORMAT_VERSION
INFERENCE_ARTIFACT_TYPE = FINAL_ARTIFACT_TYPE
BANGMODEL_FORMAT_VERSION = FINAL_ARTIFACT_FORMAT_VERSION
save_bangmodel = save_final_inference_artifact
load_bangmodel = load_final_inference_artifact
save_final_artifact = save_final_inference_artifact
load_final_artifact = load_final_inference_artifact
FinalInferenceArtifact = PPOInferenceArtifact

__all__ = [
    "FINAL_ARTIFACT_EXTENSION",
    "FINAL_ARTIFACT_FORMAT_VERSION",
    "FINAL_ARTIFACT_TYPE",
    "INFERENCE_ARTIFACT_FORMAT_VERSION",
    "INFERENCE_ARTIFACT_TYPE",
    "BANGMODEL_FORMAT_VERSION",
    "FinalInferenceArtifact",
    "PPOInferenceArtifact",
    "VERSION_MIGRATION_POLICY",
    "freeze_inference_modules",
    "load_bangmodel",
    "load_final_artifact",
    "load_final_inference_artifact",
    "migrate_final_metadata",
    "read_final_artifact_metadata",
    "save_bangmodel",
    "save_final_artifact",
    "save_final_inference_artifact",
]
