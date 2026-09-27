"""Focused TASK-6-2 tests for EmotionInference supervised training."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import EVENT_FEATURE_SIZE, EmotionInferenceModule
from bang_rl.config import EmotionConfig
from bang_rl.data.tensor_cache import CACHE_FORMAT_VERSION, FIXED_TENSOR_SCHEMA
from bang_rl.exceptions import ArtifactCompatibilityError, DataError
from bang_rl.training import (
    EmotionInferenceSLTrainer,
    EmotionSLConfig,
    build_emotion_batch,
    load_emotion_artifact,
)


def _cache(tmp_path: Path, split: str, *, schema: str = "1.0", seed: int = 0) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    count = 3 if split == "train" else 2
    rng = np.random.default_rng(seed)
    tensors = {
        "state_vector": np.zeros((count, 9), dtype=np.float32),
        "player_state_matrix": np.zeros((count, 5, 8), dtype=np.float32),
        "role_beliefs": np.zeros((count, 5, 5), dtype=np.float32),
        "event_history": rng.normal(size=(count, 4, 64, EVENT_FEATURE_SIZE)).astype(np.float32),
        "event_valid_mask": np.zeros((count, 4, 64), dtype=np.bool_),
        "action_mask": np.ones((count, 6), dtype=np.uint8),
        "role_labels": np.zeros((count, 5), dtype=np.uint8),
        "role_label_mask": np.ones((count, 5), dtype=np.bool_),
        "emotion_labels": np.zeros((count, 5, 2), dtype=np.float32),
        "emotion_label_mask": np.ones((count, 5), dtype=np.bool_),
        "actions": np.arange(count, dtype=np.uint8),
        "actor_indices": np.zeros(count, dtype=np.uint8),
        "target_indices": np.full(count, 255, dtype=np.uint8),
        "environment_steps": np.arange(count, dtype=np.uint32),
        "event_id_starts": np.arange(count, dtype=np.uint32),
        "event_id_ends": np.arange(count, dtype=np.uint32),
        "event_counts": np.zeros(count, dtype=np.uint32),
        "episode_indices": np.arange(count, dtype=np.uint32),
        "rewards": np.zeros(count, dtype=np.float32),
        "terminated": np.zeros(count, dtype=np.bool_),
        "truncated": np.zeros(count, dtype=np.bool_),
    }
    viewers = [0, 2, 4, 1][:count]
    tensors["state_vector"][:, 8] = np.asarray(viewers, dtype=np.float32) / 4.0
    tensors["event_valid_mask"][:, :, -2:] = True
    for row in range(count):
        for player in range(5):
            tensors["emotion_labels"][row, player] = [
                (row * 10 + player) / 100.0,
                (row * 10 + player + 1) / 100.0,
            ]
    destination = tmp_path / f"cache-{split}.npz"
    np.savez(destination, **tensors)
    tensor_schema = {
        key: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for key, value in sorted(tensors.items())
    }
    manifest = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_schema_version": "1.0",
        "cache_content_hash": f"cache-{split}-{schema}",
        "npz_sha256": sha256(destination.read_bytes()).hexdigest(),
        "source_content_hash": "source-hash",
        "source_shard_ids": ["source-0"],
        "split": split,
        "split_seed": 11,
        "preprocessing_version": "1.0",
        "preprocessor_config_hash": "preprocessor-hash",
        "observation_schema_version": "1.0",
        "emotion_schema_version": schema,
        "emotion_dimension": 2,
        "schema_hash": f"schema-{schema}",
        "episode_ids": [f"episode-{split}-{index}" for index in range(count)],
        "tensor_schema": tensor_schema,
    }
    manifest_path = destination.with_name(destination.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return destination


def _trainer(tmp_path: Path, *, config: EmotionSLConfig | None = None, test: bool = False) -> EmotionInferenceSLTrainer:
    train = _cache(tmp_path, "train", seed=1)
    validation = _cache(tmp_path, "validation", seed=2)
    test_cache = _cache(tmp_path, "test", seed=3) if test else None
    return EmotionInferenceSLTrainer(
        train,
        validation,
        test_cache=test_cache,
        config=config or EmotionSLConfig(epochs=1, batch_size=3, seed=7),
        emotion_config=EmotionConfig(schema_version="1.0", emotion_count=2),
    )


def test_batch_aligns_four_targets_to_public_viewer_and_exposes_only_public_inputs(tmp_path: Path) -> None:
    cache = _cache(tmp_path, "train")
    batch = build_emotion_batch(cache, [0, 1])

    # Row 0 viewer=0 -> players 1,2,3,4; row 1 viewer=2 -> 0,1,3,4.
    torch.testing.assert_close(
        batch.emotion_labels[0, :, 0], torch.tensor([0.01, 0.02, 0.03, 0.04])
    )
    torch.testing.assert_close(
        batch.emotion_labels[1, :, 0], torch.tensor([0.10, 0.11, 0.13, 0.14])
    )
    assert set(batch.model_inputs) == {"event_history", "event_valid_mask"}
    assert "actions" not in batch.model_inputs
    assert "emotion_labels" not in batch.model_inputs


def test_one_step_uses_pre_action_targets_and_updates_weights_with_masked_mse(tmp_path: Path) -> None:
    config = EmotionSLConfig(epochs=1, batch_size=3, seed=5)
    train = _cache(tmp_path, "train", seed=1)
    validation = _cache(tmp_path, "validation", seed=2)
    model = EmotionInferenceModule(config=EmotionConfig(emotion_count=2), seed=5)
    batch = build_emotion_batch(train, [0, 1, 2])
    with torch.no_grad():
        initial = model(batch.model_inputs["event_history"], batch.valid_mask)
        expected = model.loss(initial[batch.emotion_label_mask], batch.emotion_labels[batch.emotion_label_mask])
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trainer = EmotionInferenceSLTrainer(
        train,
        validation,
        config=config,
        model=model,
        emotion_config=None,
    )
    result = trainer.fit()

    assert result.history[0]["train_loss"] == pytest.approx(expected.item())
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert result.history[0]["train_valid_targets"] == 12
    assert result.history[0]["train_valid_values"] == 24


def test_outputs_are_bounded_and_masked_rows_are_excluded_from_loss(tmp_path: Path) -> None:
    cache = _cache(tmp_path, "train")
    batch = build_emotion_batch(cache, [0])
    batch.emotion_label_mask[0, 0] = False
    trainer = _trainer(tmp_path)
    loss, targets, values = trainer._loss_and_metrics(batch)

    assert torch.isfinite(loss)
    assert targets == 3
    assert values == 6
    predictions = trainer.model(batch.model_inputs["event_history"], batch.valid_mask)
    assert bool(torch.all((predictions >= 0.0) & (predictions <= 1.0)))


def test_metrics_checkpoint_early_stopping_and_artifact_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "emotion.checkpoint.pt"
    metrics = tmp_path / "metrics.jsonl"
    trainer = _trainer(
        tmp_path,
        config=EmotionSLConfig(
            epochs=3,
            batch_size=2,
            seed=9,
            early_stopping_patience=1,
            checkpoint_path=checkpoint,
            metrics_path=metrics,
        ),
        test=True,
    )
    result = trainer.fit()

    assert checkpoint.exists()
    logged = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert {entry["split"] for entry in logged if entry["event"] == "epoch"} >= {"validation", "test"}
    assert result.test_metrics is not None
    metadata = result.artifact_metadata
    assert metadata["emotion_schema"] == {
        "version": "1.0",
        "count": 2,
        "names": ["emotion_0", "emotion_1"],
        "initial_baseline": 0.0,
    }
    assert metadata["source_hashes"]["train"]["source_content_hash"] == "source-hash"
    assert metadata["cache_metadata"]["validation"]["cache_content_hash"] == "cache-validation-1.0"
    assert metadata["training"]["label_alignment"].startswith("pre-action")

    artifact = tmp_path / "emotion.phase2.pt"
    trainer.save_artifact(artifact)
    payload = torch.load(artifact, weights_only=False)
    assert "optimizer_state" not in payload
    restored, restored_metadata = load_emotion_artifact(artifact)
    assert restored.emotion_schema_version == "1.0"
    assert restored_metadata["emotion_schema_version"] == "1.0"


def test_resume_reproduces_uninterrupted_training(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.checkpoint.pt"
    config = EmotionSLConfig(epochs=2, batch_size=2, seed=13, checkpoint_path=checkpoint)
    full = _trainer(tmp_path / "full", config=config)
    full_result = full.fit()
    partial = _trainer(tmp_path / "partial", config=config)
    partial.fit(stop_after_epoch=1)
    resumed = _trainer(tmp_path / "partial", config=config)
    resumed_result = resumed.fit()

    assert len(resumed_result.history) == len(full_result.history) == 2
    for left, right in zip(full_result.history, resumed_result.history):
        assert left == right
    for name, value in full_result.model.state_dict().items():
        torch.testing.assert_close(value, resumed_result.model.state_dict()[name])


def test_schema_and_artifact_compatibility_reject_mismatch_and_privacy_leakage(tmp_path: Path) -> None:
    train = _cache(tmp_path, "train", schema="1.0")
    validation = _cache(tmp_path, "validation", schema="2.0")
    with pytest.raises(DataError, match="Schema Version"):
        EmotionInferenceSLTrainer(train, validation)

    trainer = _trainer(tmp_path)
    artifact = tmp_path / "emotion.pt"
    trainer.save_artifact(artifact)
    with pytest.raises(ArtifactCompatibilityError):
        load_emotion_artifact(artifact, expected={"emotion_schema_version": "2.0"})
    with pytest.raises(ValueError, match="privileged|fields"):
        trainer.model({
            "event_history": np.zeros((4, 64, EVENT_FEATURE_SIZE), dtype=np.float32),
            "event_valid_mask": np.ones((4, 64), dtype=np.bool_),
            "action": np.asarray([1]),
        })
