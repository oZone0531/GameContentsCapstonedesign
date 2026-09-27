"""Focused TASK-6-1 tests for IntentInference supervised training."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from bang_rl.data import CanonicalEpisode, CanonicalStep, NPZTensorCacheWriter
from bang_rl.exceptions import ArtifactCompatibilityError
from bang_rl.training import (
    IntentInferenceSLTrainer,
    IntentSLConfig,
    build_intent_batch,
    load_intent_artifact,
)


def _observation(*, viewer: int = 0, step: int = 0) -> dict:
    beliefs = np.zeros((5, 5), dtype=np.float32)
    beliefs[0, 0] = 1.0
    return {
        "current_player_idx": np.asarray([viewer], dtype=np.uint8),
        "viewer_idx": viewer,
        "environment_step": np.asarray([step], dtype=np.uint32),
        "round_index": np.asarray([step], dtype=np.uint32),
        "healths": np.asarray([5, 4, 4, 4, 4], dtype=np.uint8),
        "alive": np.asarray([True] * 5, dtype=np.bool_),
        "turn_order": np.arange(5, dtype=np.uint8),
        "sheriff_idx": np.asarray([0], dtype=np.uint8),
        "self_role": np.asarray([0], dtype=np.uint8),
        "role_visibility": np.asarray([2, 0, 0, 0, 0], dtype=np.uint8),
        "revealed_roles": np.asarray([0, 255, 255, 255, 255], dtype=np.uint8),
        "confirmed_role_distribution": beliefs,
        "event_sequence": (),
        "action_mask": np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8),
    }


def _episode(index: int, *, viewer: int = 0) -> CanonicalEpisode:
    before = _observation(viewer=viewer)
    step = CanonicalStep(
        environment_step=1,
        actor_idx=viewer,
        action=0,
        action_type="pass",
        target_idx=None,
        action_mask=(1, 0, 1, 1, 1, 1),
        pre_observation=before,
        post_observation=_observation(viewer=viewer, step=1),
        pre_event_ids=(),
        event_ids=(),
        hp_before=None,
        hp_after=None,
        eliminated_idx=None,
        reward=0.0,
        emotion_labels={},
        terminated=True,
        truncated=False,
    )
    return CanonicalEpisode(
        episode_id=f"episode-{index}",
        seed=index,
        agent_order=(0, 1, 2, 3, 4),
        metadata={
            "observation_schema_version": "1.0",
            "emotion_schema_version": "emotion-v1",
            "reward_config_version": "reward-v1",
        },
        initial_state=before,
        steps=(step,),
        events=(),
        termination_status="normal",
        termination_reason="finished",
        terminal_result=None,
        final_state=_observation(viewer=viewer, step=1),
        role_labels=(0, 1, 2, 3, 4),
    )


def _caches(tmp_path):
    writer = NPZTensorCacheWriter(tmp_path)
    train_records = [_episode(1), _episode(2, viewer=2)]
    validation_records = [_episode(3, viewer=1)]
    test_records = [_episode(4, viewer=4)]
    train = writer.write("train", train_records, split_seed=19)
    validation = writer.write(
        "validation", validation_records, train_episodes=train_records, split_seed=19
    )
    test = writer.write("test", test_records, train_episodes=train_records, split_seed=19)
    return train, validation, test


def test_batch_selects_four_non_viewer_targets_and_keeps_inputs_public(tmp_path) -> None:
    train, _, _ = _caches(tmp_path)
    batch = build_intent_batch(train.tensor_path, [1])

    assert tuple(batch.model_inputs) == ("event_history", "event_valid_mask")
    assert batch.model_inputs["event_history"].shape == (1, 4, 64, 17)
    assert batch.model_inputs["event_valid_mask"].dtype == torch.bool
    # episode-2 is viewed by slot 2; target order is slots 0, 1, 3, 4.
    torch.testing.assert_close(batch.role_labels, torch.tensor([[0, 1, 3, 4]]))
    assert "role_labels" not in batch.model_inputs
    assert "true_roles" not in batch.model_inputs


def test_one_step_uses_role_cross_entropy_logits_and_updates_parameters(tmp_path) -> None:
    train, validation, _ = _caches(tmp_path)
    trainer = IntentInferenceSLTrainer(
        train.tensor_path,
        validation.tensor_path,
        config=IntentSLConfig(epochs=1, batch_size=2, seed=7),
    )
    before = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}

    metrics = trainer._train_epoch(1)

    assert metrics["valid_targets"] == 8
    assert metrics["gradient_norm"] > 0.0
    assert any(not torch.equal(before[name], value) for name, value in trainer.model.named_parameters())
    trainer.model.eval()
    batch = build_intent_batch(train.tensor_path, [0])
    logits = trainer.model.logits(
        batch.model_inputs["event_history"], batch.model_inputs["event_valid_mask"]
    )
    probabilities = trainer.model.predict_proba(
        batch.model_inputs["event_history"], batch.model_inputs["event_valid_mask"]
    )
    torch.testing.assert_close(probabilities, torch.softmax(logits, dim=-1))


def test_validation_metrics_do_not_add_optimizer_updates_and_logs_checkpoint_artifact(tmp_path) -> None:
    train, validation, test = _caches(tmp_path / "cache")
    checkpoint = tmp_path / "intent.checkpoint.pt"
    metrics_path = tmp_path / "training.jsonl"
    artifact = tmp_path / "intent.phase2.pt"
    trainer = IntentInferenceSLTrainer(
        train.tensor_path,
        validation.tensor_path,
        test_cache=test.tensor_path,
        config=IntentSLConfig(
            epochs=2,
            batch_size=1,
            seed=11,
            checkpoint_path=checkpoint,
            metrics_path=metrics_path,
        ),
    )
    result = trainer.fit()
    trainer.save_artifact(artifact, test_metrics=result.test_metrics)

    assert checkpoint.exists()
    assert artifact.exists()
    assert result.test_metrics is not None
    assert result.test_metrics["split"] == "test"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert any(record.get("event") == "batch" and "gradient_norm" in record for record in records)
    assert sum(record.get("split") == "validation" for record in records) == 2
    assert result.artifact_metadata["source_hashes"]["train"]["cache_content_hash"]
    assert result.artifact_metadata["schema_versions"]["intent_inference_artifact_format_version"] == "1.0"

    loaded, metadata = load_intent_artifact(
        artifact,
        expected={"artifact_type": "phase2_intent_inference", "config_hash": result.artifact_metadata["config_hash"]},
    )
    assert loaded.training is False
    assert metadata["module_type"] == "IntentInferenceModule"

    with pytest.raises(ArtifactCompatibilityError, match="config_hash"):
        load_intent_artifact(artifact, expected={"config_hash": "wrong"})


def test_checkpoint_resume_reproduces_uninterrupted_training(tmp_path) -> None:
    train, validation, _ = _caches(tmp_path / "cache")
    checkpoint = tmp_path / "resume.checkpoint.pt"
    full = IntentInferenceSLTrainer(
        train.tensor_path,
        validation.tensor_path,
        config=IntentSLConfig(epochs=2, batch_size=1, seed=23),
    )
    full.fit()

    partial = IntentInferenceSLTrainer(
        train.tensor_path,
        validation.tensor_path,
        config=IntentSLConfig(
            epochs=2, batch_size=1, seed=23, checkpoint_path=checkpoint
        ),
    )
    partial.fit(stop_after_epoch=1)
    resumed = IntentInferenceSLTrainer(
        train.tensor_path,
        validation.tensor_path,
        config=IntentSLConfig(epochs=2, batch_size=1, seed=23),
        resume_from=checkpoint,
    )
    resumed.fit()

    for name, expected in full.model.state_dict().items():
        torch.testing.assert_close(expected, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert resumed.history == full.history
    assert resumed.best_epoch == full.best_epoch
