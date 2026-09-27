from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import EmotionInferenceModule, IntentInferenceModule, PPOModule
from bang_rl.config import EmotionConfig
from bang_rl.exceptions import ArtifactCompatibilityError, TrainingError
from bang_rl.training import (
    PPOInferenceArtifact,
    PPOTrainer,
    load_final_inference_artifact,
    save_final_inference_artifact,
)
from bang_rl.training.inference_artifact import load_phase2_inference_artifact, save_phase2_inference_artifact


def _phase2(tmp_path: Path):
    intent = IntentInferenceModule(seed=13).eval()
    emotion = EmotionInferenceModule(
        config=EmotionConfig(schema_version="emotion-v1", emotion_count=2), seed=13
    ).eval()
    phase2_path = tmp_path / "phase2.phase2model"
    save_phase2_inference_artifact(phase2_path, intent, emotion)
    return load_phase2_inference_artifact(phase2_path)


def _save(tmp_path: Path) -> tuple[Path, PPOModule]:
    phase2 = _phase2(tmp_path)
    ppo = PPOModule(input_dim=4, seed=17)
    path = tmp_path / "model.bangmodel"
    save_final_inference_artifact(
        path,
        ppo,
        phase2_artifact=phase2,
        job_snapshot={
            "job_id": "job-1",
            "seed": 7,
            "split_seed": 8,
            "versions": {
                "environment_version": "env-v1",
                "observation_schema_version": "obs-v1",
                "action_space_version": "action-v1",
                "reward_config_version": "reward-v1",
                "emotion_schema_version": "emotion-v1",
                "preprocessing_version": "pre-v1",
            },
            "artifact_compatibility": {},
            "model_architecture": {"ppo": ppo.metadata()},
            "optimizers": {"ppo": ppo.optimizer_metadata},
            "config_snapshot": {},
            "preset_snapshot": {"preset_id": "dummy", "version": "1.0"},
            "reward_snapshot": {"version": "reward-v1"},
            "emotion_schema_snapshot": {"schema_version": "emotion-v1"},
            "split": {"seed": 8},
        },
        training_metadata={"total_training_time_seconds": 12.5, "final_win_rate": 0.6},
    )
    return path, ppo


def test_final_bundle_has_exact_files_checksums_and_no_training_state(tmp_path: Path) -> None:
    path, _ = _save(tmp_path)
    with ZipFile(path) as bundle:
        assert set(bundle.namelist()) == {"manifest.json", "metadata.json", "model.safetensors"}
        manifest_bytes = bundle.read("manifest.json")
        metadata_bytes = bundle.read("metadata.json")
        weights = bundle.read("model.safetensors")
    manifest = json.loads(manifest_bytes)
    metadata = json.loads(metadata_bytes)
    assert manifest["artifact_type"] == "inference"
    assert manifest["inference_model_artifact_format_version"] == "1.0"
    assert manifest["weights_sha256"] == sha256(weights).hexdigest()
    assert manifest["metadata_sha256"] == sha256(metadata_bytes).hexdigest()
    assert manifest["weight_namespaces"] == ["intent", "emotion", "ppo"]
    assert metadata["modules"]["ppo"]["identifier"] == "PPOModule"
    assert metadata["versions"]["reward_config_version"] == "reward-v1"
    assert metadata["training"]["total_training_time_seconds"] == 12.5
    assert all(value is False for value in metadata["training_state"].values())
    assert metadata["training_state"]["resume_supported"] is False
    assert b"optimizer_state_dict" not in weights
    assert b"rollout_buffer" not in weights


def test_round_trip_restores_policy_and_freezes_all_modules(tmp_path: Path) -> None:
    path, original = _save(tmp_path)
    features = torch.arange(4, dtype=torch.float32)
    mask = np.asarray([1, 1, 0, 1, 0, 1], dtype=np.uint8)
    expected = original(features, mask)
    loaded = load_final_inference_artifact(path, expected={"versions.environment_version": "env-v1"})

    assert isinstance(loaded, PPOInferenceArtifact)
    torch.testing.assert_close(loaded(features, mask).logits, expected.logits)
    torch.testing.assert_close(loaded(features, mask).value, expected.value)
    assert not loaded.intent.training and not loaded.emotion.training and not loaded.ppo.training
    assert all(not p.requires_grad for module in (loaded.intent, loaded.emotion, loaded.ppo) for p in module.parameters())


def test_inference_api_preserves_mask_and_privacy_boundary(tmp_path: Path) -> None:
    path, _ = _save(tmp_path)
    loaded = load_final_inference_artifact(path)
    features = torch.ones(4)
    mask = np.asarray([1, 1, 0, 1, 0, 1], dtype=np.uint8)
    action = loaded.act(features, mask, deterministic=True)
    assert mask[int(action.action)] == 1
    assert torch.all(action.distribution.probs[torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.bool)] == 0)
    with pytest.raises(ValueError, match="privileged"):
        loaded({"features": features, "action_mask": mask, "reward": 1.0})
    with pytest.raises(ValueError, match="privileged"):
        loaded.intent({"event_history": np.zeros((64, 17), dtype=np.float32), "event_valid_mask": np.ones(64), "true_roles": np.zeros(5)})
    with pytest.raises(ValueError, match="privileged"):
        loaded.emotion({"event_history": np.zeros((64, 17), dtype=np.float32), "event_valid_mask": np.ones(64), "actual_emotions": np.zeros(2)})


def test_checksum_mismatch_version_mismatch_and_atomic_failure_are_rejected(tmp_path: Path) -> None:
    path, _ = _save(tmp_path)
    original = path.read_bytes()
    with ZipFile(path) as bundle:
        files = {name: bundle.read(name) for name in bundle.namelist()}
    files["model.safetensors"] = files["model.safetensors"][:-1] + bytes([files["model.safetensors"][-1] ^ 1])
    tampered = tmp_path / "tampered.bangmodel"
    with ZipFile(tampered, "w", compression=ZIP_STORED) as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    with pytest.raises(ArtifactCompatibilityError, match="weights_sha256"):
        load_final_inference_artifact(tampered)
    with pytest.raises(ArtifactCompatibilityError, match="emotion_schema_version"):
        load_final_inference_artifact(path, expected={"versions.emotion_schema_version": "other"})
    with pytest.raises(TypeError):
        save_final_inference_artifact(path, object())  # type: ignore[arg-type]
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_missing_phase2_or_unknown_format_never_creates_bundle(tmp_path: Path) -> None:
    path = tmp_path / "missing.bangmodel"
    with pytest.raises(ValueError, match="Phase-2"):
        save_final_inference_artifact(path, PPOModule(input_dim=4))
    assert not path.exists()
    with pytest.raises(TrainingError):
        load_final_inference_artifact(tmp_path / "does-not-exist.bangmodel")


def test_ppo_trainer_freezes_phase2_modules_and_keeps_ppo_trainable(tmp_path: Path) -> None:
    phase2 = _phase2(tmp_path)
    ppo = PPOModule(input_dim=4, seed=23)
    trainer = PPOTrainer(ppo, phase2_artifact=phase2)
    assert all(not parameter.requires_grad for module in (phase2.intent, phase2.emotion) for parameter in module.parameters())
    assert any(parameter.requires_grad for parameter in ppo.parameters())
    assert trainer.metadata()["inference_modules"]["frozen"] is True
