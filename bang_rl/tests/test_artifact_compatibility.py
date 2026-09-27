from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from bang_rl.app import ArtifactCompatibilityService, CompatibilityService
from bang_rl.config import SystemConfig, TrainingJob
from bang_rl.exceptions import ArtifactCompatibilityError
from bang_rl.training import (
    content_hash,
    load_final_inference_artifact,
    save_final_inference_artifact,
)
from bang_rl.training.inference_artifact import (
    load_phase2_inference_artifact,
    save_phase2_inference_artifact,
)
from bang_rl.agents.main_agent import EmotionInferenceModule, IntentInferenceModule, PPOModule


def _artifact(tmp_path: Path) -> tuple[Path, SystemConfig, TrainingJob]:
    config = SystemConfig()
    job = TrainingJob.start(config, mode="rl", seed=17, split_seed=19, job_id="compat-job")
    intent = IntentInferenceModule(seed=3)
    emotion = EmotionInferenceModule(config=config.emotion, seed=3)
    phase2_path = tmp_path / "phase2.phase2model"
    save_phase2_inference_artifact(phase2_path, intent, emotion, job_snapshot=job.snapshot)
    phase2 = load_phase2_inference_artifact(phase2_path)
    path = tmp_path / "model.bangmodel"
    save_final_inference_artifact(
        path,
        PPOModule(seed=3),
        phase2_artifact=phase2,
        job_snapshot=job.snapshot,
        config_snapshot=config,
    )
    return path, config, job


def _rewrite(path: Path, mutate) -> Path:
    with ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    metadata = json.loads(files["metadata.json"])
    mutate(metadata)
    files["metadata.json"] = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    manifest = json.loads(files["manifest.json"])
    manifest["metadata_sha256"] = sha256(files["metadata.json"]).hexdigest()
    files["manifest.json"] = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    destination = path.with_name(f"changed-{path.name}")
    with ZipFile(destination, "w", compression=ZIP_STORED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return destination


def test_matching_config_snapshot_and_service_runtime_load(tmp_path: Path) -> None:
    path, config, job = _artifact(tmp_path)

    loaded = load_final_inference_artifact(path, config=config, expected_snapshot=job.snapshot)
    assert loaded.inference_only is True
    assert ArtifactCompatibilityService().validate(path, config=config, expected_snapshot=job.snapshot)["artifact_type"] == "inference"
    assert CompatibilityService().load(path, config=config, snapshot=job.snapshot).inference_only
    assert CompatibilityService().load(path, config=config).inference_only


def test_config_and_job_snapshot_hash_mismatches_are_field_level(tmp_path: Path) -> None:
    path, config, job = _artifact(tmp_path)
    changed_job = TrainingJob.start(config, mode="rl", seed=18, split_seed=19, job_id="compat-job").snapshot
    with pytest.raises(ArtifactCompatibilityError) as error:
        load_final_inference_artifact(path, expected_snapshot=changed_job)
    assert error.value.field == "config_hash" or error.value.field == "job_snapshot_hash"
    assert error.value.current_value is not None
    assert error.value.expected_value is not None

    changed_snapshot = config.model_dump(mode="json")
    changed_snapshot["seed"] = 999
    with pytest.raises(ArtifactCompatibilityError) as error:
        load_final_inference_artifact(path, config_snapshot=changed_snapshot)
    assert error.value.field == "config_hash"
    payload = json.loads(error.value.to_json())
    assert payload["field"] == "config_hash"
    assert payload["current_value"] != payload["expected_value"]


def test_version_policy_rejects_mismatch_without_migration(tmp_path: Path) -> None:
    path, _, _ = _artifact(tmp_path)
    with pytest.raises(ArtifactCompatibilityError, match="versions.emotion_schema_version"):
        load_final_inference_artifact(path, expected={"versions.emotion_schema_version": "emotion-legacy"})
    changed = _rewrite(path, lambda metadata: metadata.update({"inference_model_artifact_format_version": "0.9"}))
    with pytest.raises(ArtifactCompatibilityError, match="inference_model_artifact_format_version"):
        load_final_inference_artifact(changed)


def test_fixed_architecture_and_optimizer_contracts_are_checked_before_reconstruction(tmp_path: Path) -> None:
    path, _, _ = _artifact(tmp_path)
    architecture_mismatch = _rewrite(
        path,
        lambda metadata: metadata["modules"]["ppo"]["metadata"]["architecture"]["policy_head"]["layers"][0].update({"out_features": 5}),
    )
    with pytest.raises(ArtifactCompatibilityError, match="modules.ppo.metadata.architecture.policy_head.layers.0.out_features"):
        load_final_inference_artifact(architecture_mismatch)

    optimizer_mismatch = _rewrite(
        path,
        lambda metadata: metadata["modules"]["intent"]["metadata"]["architecture"]["optimizer"].update({"learning_rate": 0.5}),
    )
    with pytest.raises(ArtifactCompatibilityError, match="optimizer"):
        load_final_inference_artifact(optimizer_mismatch)


def test_missing_schema_fields_forbidden_resume_state_and_malformed_bundle_fail_clearly(tmp_path: Path) -> None:
    path, _, _ = _artifact(tmp_path)
    missing = _rewrite(path, lambda metadata: metadata.pop("hardware"))
    with pytest.raises(ArtifactCompatibilityError, match="hardware"):
        load_final_inference_artifact(missing)
    forbidden = _rewrite(path, lambda metadata: metadata.update({"rollout_state": {"pending": True}}))
    with pytest.raises(ArtifactCompatibilityError, match="rollout_state"):
        load_final_inference_artifact(forbidden)

    malformed = tmp_path / "malformed.bangmodel"
    malformed.write_bytes(b"not a zip")
    with pytest.raises(ArtifactCompatibilityError, match="bundle"):
        load_final_inference_artifact(malformed)


def test_hash_helper_matches_metadata_contract(tmp_path: Path) -> None:
    path, config, job = _artifact(tmp_path)
    with ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata["config_hash"] == content_hash(config.model_dump(mode="json"))
    assert metadata["job_snapshot_hash"] == content_hash(job.snapshot.to_dict())
    assert metadata["hashes"]["config_snapshot_sha256"] == metadata["config_hash"]
