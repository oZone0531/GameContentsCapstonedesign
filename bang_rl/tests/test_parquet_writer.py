"""Focused TASK-5-2 tests for canonical Parquet persistence."""

from __future__ import annotations

import shutil

import pyarrow.parquet as pq
import pytest

from bang_rl.config import DatasetConfig
from bang_rl.data import (
    CANONICAL_PARQUET_SCHEMA,
    CanonicalEpisode,
    CanonicalStep,
    ParquetCanonicalShardWriter,
)
from bang_rl.exceptions import DataError



def episode(index: int, *, status: str = "normal", include_hidden_role: bool = False) -> CanonicalEpisode:
    event = {
        "event_id": index + 1,
        "environment_step": 1,
        "event_type": "pass",
        "actor_idx": 0,
        "target_idx": None,
        "hp_before": None,
        "hp_after": None,
        "eliminated_idx": None,
        "revealed_role": None,
        "public": True,
    }
    public_state = {
        "environment_step": [0],
        "current_player_idx": [0],
        "healths": [5, 4, 4, 4, 4],
        "alive": [True] * 5,
        "turn_order": [0, 1, 2, 3, 4],
        "action_mask": [1, 0, 1, 1, 1, 1],
        "role_visibility": [1, 0, 0, 0, 0],
        "revealed_roles": [0, 255, 255, 255, 255],
        "confirmed_role_distribution": [[1.0, 0.0, 0.0, 0.0, 0.0]] + [[0.0] * 5] * 4,
        "self_role": 0,
    }
    if include_hidden_role:
        public_state["roles"] = [0, 1, 2, 3, 4]
        public_state["players"] = [{"player_idx": 1, "role": 4}]
    step = CanonicalStep(
        environment_step=1,
        actor_idx=0,
        action=0,
        action_type="pass",
        target_idx=None,
        action_mask=(1, 0, 1, 1, 1, 1),
        pre_observation=public_state,
        post_observation={**public_state, "environment_step": [1]},
        pre_event_ids=(),
        event_ids=(index + 1,),
        hp_before=None,
        hp_after=None,
        eliminated_idx=None,
        reward=0.25,
        reward_breakdowns={"0": {"total": 0.25}},
        emotion_labels={
            "0": {
                "episode_id": f"episode-{index}",
                "environment_step": 0,
                "player_idx": 0,
                "values": [0.2],
                "emotion_names": ["fear"],
                "emotion_schema_version": "emotion-v1",
                "input_event_ids": [],
                "public_observation": public_state,
            }
        },
        terminated=status == "normal",
        truncated=status == "truncated",
    )
    return CanonicalEpisode(
        episode_id=f"episode-{index}",
        seed=index,
        agent_order=(0, 1, 2, 3, 4),
        metadata={
            "dataset_schema_version": "1.0",
            "environment_version": "env-v1",
            "observation_schema_version": "obs-v1",
            "action_space_version": "1.0",
            "emotion_schema_version": "emotion-v1",
            "reward_config_version": "reward-v1",
            "preset_version": "preset-v1",
            "custom": {"index": index},
        },
        initial_state=public_state,
        steps=(step,),
        events=(event,),
        termination_status=status,
        termination_reason="finished" if status == "normal" else "max_episode_steps_reached",
        terminal_result=(
            {"terminated": True, "winners": [0], "losers": [1, 2, 3, 4], "winning_team": 0, "reason": "finished"}
            if status == "normal" else None
        ),
        final_state={**public_state, "environment_step": [1]},
        role_labels=(0, 1, 2, 3, 4),
    )


def test_schema_round_trip_has_fixed_nested_types_and_preserves_both_statuses(tmp_path) -> None:
    writer = ParquetCanonicalShardWriter(
        tmp_path,
        DatasetConfig(canonical_shard_max_episodes=10, canonical_shard_max_bytes=100_000_000),
    )
    writer.append(episode(1, status="normal"))
    writer.append(episode(2, status="truncated"))
    paths = writer.finalize()

    table = pq.read_table(paths[0])
    assert table.schema.remove_metadata() == CANONICAL_PARQUET_SCHEMA.remove_metadata()
    assert table.schema.field("events").type.value_type[0].name == "event_id"
    assert table.schema.field("steps").type.value_type[0].name == "environment_step"
    assert table.schema.field("steps").type.value_type[6].name == "pre_observation"
    assert table.schema.field("steps").type.value_type[15].name == "emotion_labels"
    assert table.column("termination_status").to_pylist() == ["normal", "truncated"]
    assert table.schema.metadata[b"bang_rl.compression"] == b"zstd"
    assert table.schema.metadata[b"bang_rl.contains_private_role_labels"] == b"true"

    restored = ParquetCanonicalShardWriter.read_shard(paths[0])
    assert [item.episode_id for item in restored] == ["episode-1", "episode-2"]
    assert restored[0].events == (episode(1).events[0],)
    assert restored[1].termination_status == "truncated"
    assert restored[0].steps[0].event_ids == (2,)


def test_episode_and_byte_limits_rotate_shards(tmp_path) -> None:
    by_count = ParquetCanonicalShardWriter(
        tmp_path / "count",
        DatasetConfig(canonical_shard_max_episodes=2, canonical_shard_max_bytes=100_000_000),
    )
    for index in range(3):
        by_count.append(episode(index))
    paths = by_count.finalize()
    assert [pq.read_table(path).num_rows for path in paths] == [2, 1]

    by_bytes = ParquetCanonicalShardWriter(
        tmp_path / "bytes",
        DatasetConfig(canonical_shard_max_episodes=100, canonical_shard_max_bytes=1),
    )
    by_bytes.append(episode(10))
    by_bytes.append(episode(11))
    byte_paths = by_bytes.finalize()
    assert len(byte_paths) == 2
    assert all(path.stat().st_size > 1 for path in byte_paths)


def test_pending_serialized_candidate_is_reused_at_flush(tmp_path, monkeypatch) -> None:
    import bang_rl.data.parquet_writer as parquet_writer

    calls = 0
    original = parquet_writer._serialized

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(parquet_writer, "_serialized", counted)
    writer = ParquetCanonicalShardWriter(
        tmp_path,
        DatasetConfig(canonical_shard_max_episodes=10, canonical_shard_max_bytes=100_000_000),
    )
    writer.append(episode(20))
    writer.append(episode(21))
    paths = writer.finalize()

    assert len(paths) == 1
    assert calls == 1
    assert pq.read_table(paths[0]).num_rows == 2



    first_dir, second_dir = tmp_path / "one", tmp_path / "two"
    for directory in (first_dir, second_dir):
        writer = ParquetCanonicalShardWriter(directory, include_private_roles=False)
        writer.append(episode(1, include_hidden_role=True))
        writer.finalize()

    first_bytes = (first_dir / "shard-000000.parquet").read_bytes()
    second_bytes = (second_dir / "shard-000000.parquet").read_bytes()
    assert first_bytes == second_bytes
    table = pq.read_table(first_dir / "shard-000000.parquet")
    row = table.to_pylist()[0]
    assert row["role_labels"] is None
    assert table.schema.metadata[b"bang_rl.contains_private_role_labels"] == b"false"
    assert "roles" not in row["metadata_json"]
    assert "\"role\"" not in row["metadata_json"]
    assert row["initial_state"]["extra_json"] == '{"players":[{"player_idx":1}]}'
    assert '"role"' not in row["initial_state"]["extra_json"]


def test_atomic_failure_does_not_leave_final_or_temp_file(tmp_path, monkeypatch) -> None:
    writer = ParquetCanonicalShardWriter(tmp_path)
    writer.append(episode(1))

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("bang_rl.data.parquet_writer._atomic_write", fail)
    with pytest.raises(DataError, match="atomically"):
        writer.flush()
    assert list(tmp_path.glob("*.parquet")) == []
    assert list(tmp_path.glob("*.parquet.tmp")) == []
    assert len(writer.pending_episodes) == 1


def test_resume_recovers_valid_temp_and_rejects_corruption(tmp_path) -> None:
    original = ParquetCanonicalShardWriter(tmp_path)
    original.append(episode(1))
    original.finalize()
    final = tmp_path / "shard-000000.parquet"
    temp = tmp_path / "shard-000001.parquet.tmp"
    shutil.copyfile(final, temp)
    final.unlink()

    resumed = ParquetCanonicalShardWriter(tmp_path)
    assert (tmp_path / "shard-000001.parquet").exists()
    assert resumed.shard_paths == (tmp_path / "shard-000001.parquet",)
    with pytest.raises(DataError, match="duplicate episode_id"):
        resumed.append(episode(1))

    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "shard-000000.parquet").write_bytes(b"not parquet")
    with pytest.raises(DataError, match="cannot read canonical shard"):
        ParquetCanonicalShardWriter(corrupt_dir)


def test_lifecycle_is_idempotent_and_rejects_writes_after_finalize(tmp_path) -> None:
    writer = ParquetCanonicalShardWriter(tmp_path)
    writer.append(episode(1, status="truncated"))
    assert len(writer.flush()) == 1
    assert writer.flush() == ()
    assert writer.finalize() == ()
    assert writer.finalize() == ()
    with pytest.raises(DataError, match="finalized"):
        writer.append(episode(2))
