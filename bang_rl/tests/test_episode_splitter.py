"""Focused TASK-5-3 tests for deterministic episode-level splitting."""

from __future__ import annotations

import json

import pytest

from bang_rl.data import (
    CanonicalEpisode,
    EpisodeSplitConfig,
    EpisodeSplitter,
    ParquetCanonicalShardWriter,
)
from bang_rl.exceptions import DataError


def make_episode(index: int, *, status: str = "normal", observation_schema: str = "obs-v1") -> CanonicalEpisode:
    return CanonicalEpisode(
        episode_id=f"episode-{index:03d}",
        seed=index,
        agent_order=(0, 1, 2, 3, 4),
        metadata={
            "dataset_schema_version": "1.0",
            "observation_schema_version": observation_schema,
            "action_space_version": "1.0",
            "emotion_schema_version": "emotion-v1",
            "reward_config_version": "reward-v1",
        },
        initial_state={"episode_index": index},
        steps=(),
        events=(),
        termination_status=status,
        termination_reason="finished" if status == "normal" else "max_episode_steps_reached",
        terminal_result=None,
        final_state={"episode_index": index},
    )


def test_default_split_has_exact_80_10_10_episode_counts_and_no_leakage(tmp_path) -> None:
    episodes = tuple(make_episode(index) for index in range(20))
    result = EpisodeSplitter(seed=17, manifest_path=tmp_path / "split.json").split(episodes)

    assert (result.train.count, result.validation.count, result.test.count) == (16, 2, 2)
    assert set(result.train.episode_ids).isdisjoint(result.validation.episode_ids)
    assert set(result.train.episode_ids).isdisjoint(result.test.episode_ids)
    assert set(result.validation.episode_ids).isdisjoint(result.test.episode_ids)
    assert set(result.train.episode_ids + result.validation.episode_ids + result.test.episode_ids) == {
        episode.episode_id for episode in episodes
    }
    # Complete records, including all observations/sequences/labels, remain together.
    assert all(episode.episode_id.startswith("episode-") for episode in result.train.episodes)
    assert result.manifest["statistics_source"] == "train"
    assert result.manifest["train_statistics"]["episode_count"] == 16


def test_same_seed_is_input_order_independent_and_different_seed_changes_assignment() -> None:
    episodes = tuple(make_episode(index) for index in range(30))
    first = EpisodeSplitter(seed=99).split(episodes)
    repeated = EpisodeSplitter(seed=99).split(reversed(episodes))
    different = EpisodeSplitter(seed=100).split(episodes)

    assert first.episode_ids == repeated.episode_ids
    assert first.episode_ids != different.episode_ids


def test_explicit_counts_and_termination_stratification_are_recorded() -> None:
    episodes = tuple(
        make_episode(index, status="truncated" if index % 3 == 0 else "normal")
        for index in range(12)
    )
    config = EpisodeSplitConfig(seed=5, counts=(6, 3, 3), stratify_by_termination=True)
    result = EpisodeSplitter(config).split(episodes)

    assert [result[name].count for name in ("train", "validation", "test")] == [6, 3, 3]
    assert sum(partition.termination_counts.get("truncated", 0) for partition in result.partitions.values()) == 4
    assert result.manifest["output_metadata"]["termination_counts"] == {
        name: dict(result[name].termination_counts) for name in ("train", "validation", "test")
    }


def test_manifest_persists_and_resume_rejects_changed_source_or_config(tmp_path) -> None:
    path = tmp_path / "split-manifest.json"
    episodes = tuple(make_episode(index) for index in range(10))
    splitter = EpisodeSplitter(seed=7, manifest_path=path)
    first = splitter.split(episodes)
    assert path.exists()
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["config_hash"] == first.manifest["config_hash"]
    assert persisted["split_seed"] == 7

    resumed = EpisodeSplitter(seed=7, manifest_path=path).split(tuple(reversed(episodes)), resume=True)
    assert resumed.episode_ids == first.episode_ids

    with pytest.raises(DataError, match="configuration"):
        EpisodeSplitter(seed=8, manifest_path=path).split(episodes, resume=True)
    with pytest.raises(DataError, match="source dataset"):
        EpisodeSplitter(seed=7, manifest_path=path).split(episodes[:-1], resume=True)


def test_invalid_duplicate_missing_and_schema_mismatch_inputs_are_rejected() -> None:
    with pytest.raises(DataError, match="duplicate episode_id"):
        EpisodeSplitter().split([make_episode(1), make_episode(1)])
    missing = make_episode(1)
    object.__setattr__(missing, "episode_id", "")
    with pytest.raises(DataError, match="missing episode_id"):
        EpisodeSplitter().split([missing])
    with pytest.raises(DataError, match="schema mismatch"):
        EpisodeSplitter().split([make_episode(1), make_episode(2, observation_schema="obs-v2")])


def test_schema_round_trip_and_split_dataset_path(tmp_path) -> None:
    source = tmp_path / "canonical"
    writer = ParquetCanonicalShardWriter(source)
    for index in range(10):
        writer.append(make_episode(index))
    writer.finalize()

    result = EpisodeSplitter(seed=3).split_dataset(source)
    assert result.train.count == 8
    assert len(result.train.episode_ids + result.validation.episode_ids + result.test.episode_ids) == 10
