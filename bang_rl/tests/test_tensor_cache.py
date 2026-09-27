"""Focused TASK-5-4 tests for deterministic NPZ tensor caches."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bang_rl.data import (
    CanonicalEpisode,
    CanonicalStep,
    NPZTensorCacheLoader,
    NPZTensorCacheWriter,
)
from bang_rl.exceptions import DataError


def observation(*, step: int = 0, events=(), health: int = 5) -> dict:
    return {
        "current_player_idx": np.asarray([0], dtype=np.uint8),
        "environment_step": np.asarray([step], dtype=np.uint32),
        "round_index": np.asarray([step], dtype=np.uint32),
        "healths": np.asarray([health, 4, 4, 4, 4], dtype=np.uint8),
        "alive": np.asarray([health > 0, True, True, True, True], dtype=np.bool_),
        "turn_order": np.arange(5, dtype=np.uint8),
        "sheriff_idx": np.asarray([0], dtype=np.uint8),
        "self_role": np.asarray([0], dtype=np.uint8),
        "role_visibility": np.asarray([1, 0, 0, 0, 0], dtype=np.uint8),
        "revealed_roles": np.asarray([0, 255, 255, 255, 255], dtype=np.uint8),
        "confirmed_role_distribution": np.asarray(
            [[1, 0, 0, 0, 0]] + [[0, 0, 0, 0, 0]] * 4, dtype=np.float32
        ),
        "event_sequence": tuple(events),
        "action_mask": np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8),
        "viewer_idx": 0,
    }


def event(event_id: int = 1) -> dict:
    return {
        "event_id": event_id,
        "environment_step": event_id,
        "event_type": "bang",
        "actor_idx": 1,
        "target_idx": 2,
        "hp_before": 4,
        "hp_after": 3,
        "eliminated_idx": None,
        "revealed_role": None,
        "public": True,
    }


def episode(index: int, *, status: str = "normal", pre=None) -> CanonicalEpisode:
    before = observation(events=() if pre is None else pre)
    step = CanonicalStep(
        environment_step=1,
        actor_idx=0,
        action=0,
        action_type="pass",
        target_idx=None,
        action_mask=(1, 0, 1, 1, 1, 1),
        pre_observation=before,
        post_observation=observation(step=1),
        pre_event_ids=(),
        event_ids=(),
        hp_before=None,
        hp_after=None,
        eliminated_idx=None,
        reward=0.25,
        emotion_labels={
            "0": {"values": [0.25, 0.75], "environment_step": 0},
            "1": {"values": [0.5, 0.5], "environment_step": 0},
        },
        terminated=status == "normal",
        truncated=status == "truncated",
    )
    return CanonicalEpisode(
        episode_id=f"episode-{index:03d}",
        seed=index,
        agent_order=(0, 1, 2, 3, 4),
        metadata={
            "dataset_schema_version": "1.0",
            "observation_schema_version": "1.0",
            "action_space_version": "1.0",
            "emotion_schema_version": "emotion-v1",
            "reward_config_version": "reward-v1",
        },
        initial_state=before,
        steps=(step,),
        events=tuple(() if pre is None else pre),
        termination_status=status,
        termination_reason="finished" if status == "normal" else "max_episode_steps_reached",
        terminal_result=None,
        final_state=observation(step=1),
        role_labels=(0, 1, 2, 3, 4),
    )


def test_fixed_tensors_shapes_dtypes_and_normal_only_filter(tmp_path) -> None:
    writer = NPZTensorCacheWriter(tmp_path)
    artifact = writer.write("train", [episode(2), episode(1, status="truncated")])
    loader = NPZTensorCacheLoader(artifact.tensor_path)
    loader.validate()

    assert len(loader) == 1
    batch = loader.model_inputs()
    assert batch["state_vector"].shape == (1, 9)
    assert batch["player_state_matrix"].shape == (1, 5, 8)
    assert batch["role_beliefs"].shape == (1, 5, 5)
    assert batch["event_history"].shape == (1, 4, 64, 17)
    assert batch["event_valid_mask"].shape == (1, 4, 64)
    assert batch["action_mask"].shape == (1, 6)
    assert batch["action_mask"].dtype == np.uint8
    np.testing.assert_allclose(loader.transition()["rewards"], [0.25])
    assert batch["environment_steps"].dtype == np.uint32 if "environment_steps" in batch else True
    assert loader.manifest["excluded_truncated_episode_count"] == 1

    labels = loader.labels()
    assert labels["role_labels"].dtype == np.uint8
    assert labels["emotion_labels"].shape == (1, 5, 2)
    assert labels["emotion_labels"].dtype == np.float32


def test_preprocessor_padding_and_valid_mask_are_saved_exactly(tmp_path) -> None:
    artifact = NPZTensorCacheWriter(tmp_path).write(
        "train", [episode(1, pre=(event(1), event(2), event(3)))]
    )
    loader = NPZTensorCacheLoader(artifact.tensor_path)
    tensors = loader.model_inputs()
    valid = tensors["event_valid_mask"][0]
    assert valid[0, -3:].all()
    assert not valid[0, :-3].any()
    np.testing.assert_allclose(tensors["event_history"][0, 0, -3:, 10], [1 / 2000, 2 / 2000, 3 / 2000])


def test_normalization_statistics_are_train_only(tmp_path) -> None:
    train = episode(1)
    validation = episode(2, pre=(event(99),))
    writer = NPZTensorCacheWriter(tmp_path)
    artifact = writer.write("validation", [validation], train_episodes=[train])
    stats = artifact.manifest["normalization"]
    assert stats["statistics_source"] == "train"
    assert stats["statistics"]["state_vector"]["count"] == 1
    assert artifact.manifest["episode_ids"] == ["episode-002"]


def test_order_and_npz_bytes_are_deterministic(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    records = [episode(3), episode(1), episode(2)]
    one = NPZTensorCacheWriter(first).write("train", records)
    two = NPZTensorCacheWriter(second).write("train", list(reversed(records)))
    assert one.tensor_path.read_bytes() == two.tensor_path.read_bytes()
    assert one.manifest["episode_ids"] == ["episode-001", "episode-002", "episode-003"]


def test_manifest_compatibility_and_source_change_invalidate_cache(tmp_path) -> None:
    records = [episode(1)]
    writer = NPZTensorCacheWriter(tmp_path, source_shard_ids=["shard-a"])
    first = writer.write("train", records, source_content_hash="source-a", split_seed=3)
    resumed = writer.write("train", records, source_content_hash="source-a", split_seed=3)
    assert resumed.manifest["generated_at"] == first.manifest["generated_at"]
    changed = writer.write("train", records, source_content_hash="source-b", split_seed=3)
    assert changed.manifest["cache_content_hash"] != first.manifest["cache_content_hash"]
    with pytest.raises(DataError, match="compatibility"):
        NPZTensorCacheLoader(
            changed.tensor_path,
            expected={"source_content_hash": "source-a"},
        ).validate()


def test_corruption_is_rejected_and_writer_recovers_atomically(tmp_path) -> None:
    artifact = NPZTensorCacheWriter(tmp_path).write("train", [episode(1)])
    artifact.tensor_path.write_bytes(b"corrupt")
    with pytest.raises(DataError, match="checksum"):
        NPZTensorCacheLoader(artifact.tensor_path).validate()
    recovered = NPZTensorCacheWriter(tmp_path).write("train", [episode(1)])
    NPZTensorCacheLoader(recovered.tensor_path).validate()
    assert not list(tmp_path.glob("*.tmp"))


def test_private_role_labels_are_targets_not_default_model_inputs(tmp_path) -> None:
    artifact = NPZTensorCacheWriter(tmp_path).write("train", [episode(1)])
    loader = NPZTensorCacheLoader(artifact.tensor_path)
    assert "role_labels" not in loader.model_inputs()
    assert "emotion_labels" not in loader.model_inputs()
    assert np.array_equal(loader.labels()["role_labels"][0], [0, 1, 2, 3, 4])
    assert list(loader.iter_batches(1))[0]["state_vector"].shape == (1, 9)
