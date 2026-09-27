from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import EmotionInferenceModule, IntentInferenceModule, ObservationPreprocessor
from bang_rl.config import EmotionConfig
from bang_rl.exceptions import ArtifactCompatibilityError, TrainingError
from bang_rl.training import (
    PHASE2_ARTIFACT_FORMAT_VERSION,
    PHASE2_ARTIFACT_TYPE,
    VERSION_MIGRATION_POLICY,
    load_phase2_inference_artifact,
    save_phase2_inference_artifact,
)


def _models() -> tuple[IntentInferenceModule, EmotionInferenceModule]:
    return (
        IntentInferenceModule(seed=41).eval(),
        EmotionInferenceModule(config=EmotionConfig(schema_version="emotion-v1", emotion_count=2), seed=41).eval(),
    )


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(-1.0, 1.0, 4 * 64 * 17, dtype=np.float32).reshape(4, 64, 17)
    mask = np.zeros((4, 64), dtype=np.bool_)
    mask[:, -3:] = True
    return values, mask


def _save(path: Path) -> tuple[IntentInferenceModule, EmotionInferenceModule]:
    intent, emotion = _models()
    save_phase2_inference_artifact(
        path,
        intent,
        emotion,
        config_snapshot={"config_version": "cfg-v1", "reward_config": {"version": "reward-v1"}},
        source_hashes={"canonical": "source-sha"},
    )
    return intent, emotion


def test_bundle_has_only_manifest_metadata_and_safetensors_with_checksums(tmp_path: Path) -> None:
    artifact = tmp_path / "phase2.phase2model"
    _save(artifact)

    with ZipFile(artifact) as bundle:
        assert set(bundle.namelist()) == {"manifest.json", "metadata.json", "model.safetensors"}
        manifest = json.loads(bundle.read("manifest.json"))
        metadata_bytes = bundle.read("metadata.json")
        weights = bundle.read("model.safetensors")
        metadata = json.loads(metadata_bytes)

    assert manifest["artifact_type"] == PHASE2_ARTIFACT_TYPE
    assert manifest["inference_model_artifact_format_version"] == PHASE2_ARTIFACT_FORMAT_VERSION
    assert manifest["weights_sha256"] == sha256(weights).hexdigest()
    assert manifest["metadata_sha256"] == sha256(metadata_bytes).hexdigest()
    assert metadata["modules"]["intent"]["identifier"] == "IntentInferenceModule"
    assert metadata["modules"]["emotion"]["identifier"] == "EmotionInferenceModule"
    assert metadata["role_fusion"]["identifier"] == "RoleFusion"
    assert metadata["preprocessing"]["config"]["model_history_length"] == 64
    assert metadata["emotion_schema"] == {
        "version": "emotion-v1",
        "count": 2,
        "names": ["emotion_0", "emotion_1"],
        "initial_baseline": 0.0,
    }
    assert metadata["config_hash"]
    assert metadata["source_hashes"] == {"canonical": "source-sha"}


def test_round_trip_restores_outputs_metadata_and_frozen_eval_modules(tmp_path: Path) -> None:
    artifact = tmp_path / "phase2.phase2model"
    intent, emotion = _save(artifact)
    sequence, mask = _inputs()
    expected_intent = intent(sequence, mask)
    expected_emotion = emotion(sequence, mask)

    loaded = load_phase2_inference_artifact(artifact)

    torch.testing.assert_close(loaded.intent(sequence, mask), expected_intent)
    torch.testing.assert_close(loaded.emotion(sequence, mask), expected_emotion)
    assert loaded.role_fusion.metadata() == loaded.metadata["role_fusion"]["metadata"]["architecture"]
    assert loaded.preprocessor.metadata() == loaded.metadata["preprocessing"]
    assert loaded.emotion_schema["version"] == "emotion-v1"
    assert not loaded.intent.training and not loaded.emotion.training
    assert all(not parameter.requires_grad for parameter in loaded.intent.parameters())
    assert all(not parameter.requires_grad for parameter in loaded.emotion.parameters())


def test_checksum_tampering_and_compatibility_mismatch_are_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "phase2.phase2model"
    _save(artifact)
    with ZipFile(artifact) as bundle:
        files = {name: bundle.read(name) for name in bundle.namelist()}
    files["model.safetensors"] = files["model.safetensors"][:-1] + bytes([files["model.safetensors"][-1] ^ 1])
    tampered = tmp_path / "tampered.phase2model"
    with ZipFile(tampered, "w", compression=ZIP_STORED) as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    with pytest.raises(ArtifactCompatibilityError, match="weights_sha256"):
        load_phase2_inference_artifact(tampered)

    with pytest.raises(ArtifactCompatibilityError, match="emotion_schema"):
        load_phase2_inference_artifact(artifact, expected={"emotion_schema.version": "other-schema"})


def test_atomic_failure_does_not_replace_existing_bundle(tmp_path: Path) -> None:
    artifact = tmp_path / "phase2.phase2model"
    _save(artifact)
    original = artifact.read_bytes()
    with pytest.raises(TypeError):
        save_phase2_inference_artifact(artifact, object(), _models()[1])  # type: ignore[arg-type]
    assert artifact.read_bytes() == original
    assert not list(tmp_path.glob(f".{artifact.name}.*.tmp"))


def test_training_and_optimizer_state_are_not_serialized_and_privacy_boundary_holds(tmp_path: Path) -> None:
    artifact = tmp_path / "phase2.phase2model"
    _save(artifact)
    loaded = load_phase2_inference_artifact(artifact)

    with ZipFile(artifact) as bundle:
        metadata = json.loads(bundle.read("metadata.json"))
        weights = bundle.read("model.safetensors")
    assert all(value is False for value in metadata["training_state"].values())
    assert b"optimizer" not in weights and b"rollout" not in weights
    assert metadata["training_state"]["optimizer_state_included"] is False
    assert metadata["training_state"]["resume_supported"] is False
    assert VERSION_MIGRATION_POLICY["automatic_migration"] is False

    sequence, mask = _inputs()
    with pytest.raises(ValueError, match="privileged"):
        loaded.intent({"event_history": sequence, "event_valid_mask": mask, "true_roles": np.zeros(5)})
    with pytest.raises(ValueError, match="privileged"):
        loaded.emotion({"event_history": sequence, "event_valid_mask": mask, "actual_emotions": np.zeros(2)})


def test_unknown_artifact_version_is_rejected_without_migration(tmp_path: Path) -> None:
    artifact = tmp_path / "phase2.phase2model"
    _save(artifact)
    with ZipFile(artifact) as bundle:
        files = {name: bundle.read(name) for name in bundle.namelist()}
    metadata = json.loads(files["metadata.json"])
    metadata["artifact_format_version"] = "0.9"
    files["metadata.json"] = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    manifest = json.loads(files["manifest.json"])
    manifest["metadata_sha256"] = sha256(files["metadata.json"]).hexdigest()
    files["manifest.json"] = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    migrated = tmp_path / "legacy.phase2model"
    with ZipFile(migrated, "w", compression=ZIP_STORED) as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    with pytest.raises(ArtifactCompatibilityError, match="artifact_format_version"):
        load_phase2_inference_artifact(migrated)
