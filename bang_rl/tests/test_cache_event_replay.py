"""The tensor cache must give the model the same event history it sees live.

Stored step observations carry no event sequence (each event is stored once
per episode), so the cache rebuilds it from ``pre_event_ids``. Without that the
role-inference input (``event_history``) was all zeros for collected data.
"""

from __future__ import annotations

import numpy as np
import pytest

from bang_rl.agents.main_agent.preprocessor import ObservationPreprocessor
from bang_rl.data import (
    EpisodeCollector,
    EpisodeSplitter,
    NPZTensorCacheLoader,
    NPZTensorCacheWriter,
    ParquetCanonicalShardWriter,
)
from bang_rl.data.game_collection import (
    FORMULA_LABEL_SCHEMA,
    build_game,
    collect_games,
    drive_game,
    load_collection_config,
)
from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES


@pytest.fixture(scope="module")
def config():
    return load_collection_config()


def _live_histories(config, seed: int):
    """Play one game and record the preprocessed live event history per step."""

    preprocessor = ObservationPreprocessor()
    env, agents, episode_id = build_game(config, seed=seed)
    live: dict[int, np.ndarray] = {}
    for agent in agents:
        original = agent.act

        def act(observation, mask, _act=original, **kwargs):
            step = int(np.asarray(observation["environment_step"]).reshape(-1)[0])
            live[step] = preprocessor.preprocess(observation).event_history
            return _act(observation, mask, **kwargs)

        agent.act = act
    episode = drive_game(env, agents, config, EpisodeCollector(), episode_id=episode_id, seed=seed)
    return episode, live


def test_stored_observations_have_no_events(config):
    episode, _ = _live_histories(config, seed=1)
    for step in episode.steps:
        assert "event_sequence" not in step.pre_observation


def test_cache_event_history_matches_live_play(config, tmp_path):
    episode, live = _live_histories(config, seed=2)
    split = EpisodeSplitter(counts=(1, 0, 0), seed=0).split([episode])
    artifacts = NPZTensorCacheWriter(
        tmp_path / "cache", emotion_schema_version=FORMULA_LABEL_SCHEMA, emotion_dimension=len(LABEL_NAMES)
    ).write_all(split)
    cache = NPZTensorCacheLoader(artifacts["train"].tensor_path).all_tensors()
    steps = cache["environment_steps"].tolist()
    assert np.count_nonzero(cache["event_history"]) > 0
    assert cache["event_valid_mask"].any()
    assert len(steps) == len(live)
    for row, step in enumerate(steps):
        # Cache rows are numbered by the step the action produced; the actor
        # observed the boundary one step earlier.
        assert np.array_equal(cache["event_history"][row], live[step - 1])


def test_cache_from_parquet_round_trip_has_events(config, tmp_path):
    collect_games(config, 4, tmp_path / "canonical")
    episodes = ParquetCanonicalShardWriter.read_dataset(tmp_path / "canonical")
    split = EpisodeSplitter(counts=(2, 1, 1), seed=0).split(episodes)
    artifacts = NPZTensorCacheWriter(
        tmp_path / "cache", emotion_schema_version=FORMULA_LABEL_SCHEMA, emotion_dimension=len(LABEL_NAMES)
    ).write_all(split)
    for name in ("train", "validation", "test"):
        tensors = NPZTensorCacheLoader(artifacts[name].tensor_path).all_tensors()
        valid = tensors["event_valid_mask"]
        assert valid.any(), name
        # The very first step of a game has seen no events; later ones have.
        later = tensors["environment_steps"] > 1
        assert valid[later].reshape(int(later.sum()), -1).any(axis=1).mean() > 0.9
