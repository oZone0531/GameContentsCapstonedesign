"""Focused tests for TASK-0-4 config loading and job snapshots."""

import json

import pytest

from bang_rl.config import (
    ConfigFileError,
    ConfigLoader,
    ConfigValidationError,
    JobSnapshot,
    SystemConfig,
    TrainingJob,
)


def test_yaml_loader_resolves_defaults_and_validates_nested_values(tmp_path) -> None:
    path = tmp_path / "training.yaml"
    path.write_text(
        """
config_version: test-v2
seed: 17
env:
  max_episode_steps: 900
  warn_turn_threshold: 20
  critical_turn_threshold: 50
reward_config:
  version: reward-v2
""",
        encoding="utf-8",
    )

    config = ConfigLoader().load(path)

    assert isinstance(config, SystemConfig)
    assert config.config_version == "test-v2"
    assert config.seed == 17
    assert config.env.max_episode_steps == 900
    assert config.reward_config.version == "reward-v2"
    assert config.dataset.canonical_shard_max_bytes == 256_000_000


def test_json_loader_and_explicit_missing_path_have_clear_errors(tmp_path) -> None:
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"config_version": "json-v1", "seed": 4}), encoding="utf-8")
    assert ConfigLoader().load(path).seed == 4

    with pytest.raises(ConfigFileError, match="does not exist"):
        ConfigLoader().load(tmp_path / "missing.yaml")


def test_loader_reports_field_path_for_invalid_config(tmp_path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "env:\n  warn_turn_threshold: 100\n  critical_turn_threshold: 100\n  max_episode_steps: 200\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="env:.*thresholds"):
        ConfigLoader().load(path)


def test_default_loader_uses_schema_defaults_when_no_file_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = ConfigLoader().load()
    assert config.env.max_episode_steps == 2_000


def test_training_job_snapshot_is_deeply_immutable_and_captured_once() -> None:
    config = SystemConfig(config_version="v1", seed=11)
    job = TrainingJob.start(config, mode="collection", split_seed=99, job_id="job-1")
    original = job.snapshot.to_dict()

    config.config_version = "v2"
    config.seed = 23
    config.reward_config.version = "reward-v2"
    config.preset.version = "preset-v2"

    assert job.snapshot.to_dict() == original
    assert job.snapshot.seed == 11
    assert job.snapshot.split_seed == 99
    assert job.snapshot.versions["reward_config_version"] == "1.0"
    with pytest.raises(TypeError):
        job.snapshot.config_snapshot["seed"] = 23

    next_job = TrainingJob.start(config, mode="collection", split_seed=99, job_id="job-2")
    assert next_job.snapshot.config_version == "v2"
    assert next_job.snapshot.seed == 23
    assert next_job.snapshot.versions["reward_config_version"] == "reward-v2"


def test_snapshot_records_contracts_compatibility_and_persists_json(tmp_path) -> None:
    config = SystemConfig(seed=7)
    snapshot = JobSnapshot.capture(config, split_seed=13, job_id="job-2")

    assert snapshot.model_architecture["intent_hidden_size"] == 128
    assert snapshot.optimizers["ppo"]["learning_rate"] == 0.0003
    assert snapshot.preset_snapshot["version"] == "1.0"
    assert snapshot.reward_snapshot["version"] == "1.0"
    assert snapshot.emotion_schema_snapshot["schema_version"] == "1.0"
    assert snapshot.artifact_compatibility["action_space_version"] == "1.0"
    assert snapshot.versions["dataset_split_seed"] == 13

    path = tmp_path / "job_snapshot.json"
    snapshot.save(path)
    loaded = JobSnapshot.load(path)
    assert loaded.to_dict() == snapshot.to_dict()
