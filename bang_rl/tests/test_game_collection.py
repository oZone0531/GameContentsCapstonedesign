"""Phase 1 collection from real all-dummy games (bang_rl.data.game_collection)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bang_rl.agents.main_agent.preprocessor import ObservationPreprocessor
from bang_rl.app.services import GameService, SessionHandle, _Session
from bang_rl.data import (
    EpisodeCollector,
    EpisodeSplitter,
    NPZTensorCacheLoader,
    NPZTensorCacheWriter,
    ParquetCanonicalShardWriter,
)
from bang_rl.data.collector import CanonicalEpisode, CanonicalStep
from bang_rl.data.game_collection import (
    FORMULA_LABEL_SCHEMA,
    build_game,
    collect_games,
    drive_game,
    formula_label_emotion_config,
    load_collection_config,
    play_and_collect_game,
)

LABEL_DIM = 9


@pytest.fixture(scope="module")
def formula_config():
    return load_collection_config()


def _labels_of(agents):
    return [agent.emotion_module.label().tolist() for agent in agents]


def _play_through_game_service(config, seed, mode):
    """Step an identical game with the real GameService.step (the GUI path)."""

    env, agents, _ = build_game(config, seed=seed, action_mode=mode)
    service = GameService()
    handle = SessionHandle("parity", "dummy_five", seed, None)
    service._sessions["parity"] = _Session(handle, config, env, agents)
    actions = []
    while not (env.state.terminated or env.state.truncated):
        result = service.step(handle)
        if result.action is not None:
            actions.append(int(result.action))
    return actions, _labels_of(agents)


@pytest.mark.parametrize("mode", ["deterministic", "stochastic"])
def test_collection_plays_the_same_game_as_game_service_step(formula_config, mode):
    # Reading labels must never change a decision or an emotion trajectory.
    for seed in range(8):
        expected_actions, expected_labels = _play_through_game_service(formula_config, seed, mode)
        env, agents, episode_id = build_game(formula_config, seed=seed, action_mode=mode)
        episode = drive_game(env, agents, formula_config, EpisodeCollector(), episode_id=episode_id, seed=seed)
        assert [step.action for step in episode.steps] == expected_actions
        assert _labels_of(agents) == expected_labels


def test_same_seed_replays_the_same_game(formula_config):
    first, _, _ = play_and_collect_game(formula_config, EpisodeCollector(), seed=11, action_mode="stochastic")
    second, _, _ = play_and_collect_game(formula_config, EpisodeCollector(), seed=11, action_mode="stochastic")
    assert first.role_labels == second.role_labels
    assert [step.action for step in first.steps] == [step.action for step in second.steps]
    assert [step.emotion_labels for step in first.steps] == [step.emotion_labels for step in second.steps]


def test_formula_labels_are_pre_action_nine_entries_for_living_dummies_only(formula_config):
    episode, _, _ = play_and_collect_game(formula_config, EpisodeCollector(), seed=3)
    assert episode.termination_status == "normal"
    saw_dead = False
    for step in episode.steps:
        alive = step.pre_observation["alive"]
        assert sorted(int(key) for key in step.emotion_labels) == [i for i in range(5) if alive[i]]
        saw_dead = saw_dead or not all(alive)
        for key, label in step.emotion_labels.items():
            values = np.asarray(label["values"])
            assert values.shape == (LABEL_DIM,)
            assert np.all((values >= 0.0) & (values <= 1.0))
            assert label["emotion_schema_version"] == FORMULA_LABEL_SCHEMA
            assert label["environment_step"] == step.environment_step - 1
            assert values[4 + int(key)] == pytest.approx(0.5)  # relation to self
            for seat in range(5):
                if not alive[seat]:
                    assert values[4 + seat] == pytest.approx(0.5)  # relation to the dead
    assert saw_dead


def test_pre_observation_is_flat_and_accepted_by_the_preprocessor(formula_config):
    episode, _, _ = play_and_collect_game(formula_config, EpisodeCollector(), seed=4)
    preprocessor = ObservationPreprocessor()
    for step in episode.steps:
        observation = step.pre_observation
        assert "players" not in observation and "observation" not in observation
        assert observation["current_player_idx"] == [step.actor_idx]
        processed = preprocessor.preprocess(observation)
        assert processed.action_mask.tolist() == list(step.action_mask)


def test_rule_table_mode_keeps_the_config_emotion_schema():
    config = load_collection_config(emotion_file=None)
    episode, _, agents = play_and_collect_game(config, EpisodeCollector(), seed=2)
    assert not any(agent.uses_emotion_formula for agent in agents)
    label = next(iter(episode.steps[0].emotion_labels.values()))
    assert label["emotion_schema_version"] == config.preset.emotion.schema_version
    assert len(label["values"]) == config.preset.emotion.emotion_count


def test_label_without_public_observation_round_trips_through_parquet(tmp_path):
    # PyArrow 19 wrote a null label observation struct but then failed to read
    # the shard back ("Expected all lists to be of size=6").
    label = {
        "episode_id": "e", "environment_step": 0, "player_idx": 0, "values": [0.5, 0.25],
        "emotion_names": ["a", "b"], "emotion_schema_version": "x", "input_event_ids": [],
    }
    observation = {"current_player_idx": [0], "environment_step": [0], "action_mask": [1, 1, 1, 1, 1, 1]}
    steps = tuple(
        CanonicalStep(
            environment_step=index + 1, actor_idx=0, action=0, action_type="pass", target_idx=None,
            action_mask=(1, 0, 0, 0, 0, 0), pre_observation=observation, post_observation=observation,
            pre_event_ids=(), event_ids=(), hp_before=None, hp_after=None, eliminated_idx=None,
            reward=0.0, reward_breakdowns={},
            emotion_labels={"0": dict(label, environment_step=index), "1": dict(label, player_idx=1, environment_step=index)},
            terminated=index == 1, truncated=False,
        )
        for index in range(2)
    )
    episode = CanonicalEpisode(
        episode_id="e", seed=0, agent_order=(0, 1, 2, 3, 4), metadata={}, initial_state=observation,
        steps=steps, events=(), termination_status="normal", termination_reason="test",
        terminal_result=None, final_state=observation, role_labels=(0, 1, 2, 3, 4), invalid_actions=(),
    )
    writer = ParquetCanonicalShardWriter(tmp_path)
    writer.write_episode(episode)
    (path,) = writer.finalize()
    (restored,) = ParquetCanonicalShardWriter.read_shard(path)
    restored_label = restored.steps[1].emotion_labels["1"]
    assert restored_label["values"] == pytest.approx([0.5, 0.25])
    assert restored_label["public_observation"] == {}


def test_collect_games_writes_shards_and_summary(formula_config, tmp_path):
    out = tmp_path / "formula"
    summary = collect_games(formula_config, 4, out, base_seed=100, keep_episodes=True)
    episodes = summary.pop("episodes")
    saved = json.loads((out / "collection_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert summary["games"] == 4 and summary["seeds"] == [100, 103]
    assert summary["emotion_mode"] == ["formula"]
    assert summary["emotion_label_schema"] == [FORMULA_LABEL_SCHEMA]
    assert summary["emotion_label_dimension"] == LABEL_DIM
    assert sum(summary["termination_status"].values()) == 4
    read_back = ParquetCanonicalShardWriter.read_dataset(out)
    assert sorted(ep.episode_id for ep in read_back) == sorted(ep.episode_id for ep in episodes)
    assert [ep.episode_id for ep in episodes] == [f"game-{seed:08d}" for seed in range(100, 104)]


def test_collect_games_refuses_a_non_empty_folder(formula_config, tmp_path):
    (tmp_path / "old.parquet").write_bytes(b"")
    with pytest.raises(FileExistsError):
        collect_games(formula_config, 1, tmp_path)


def test_formula_dataset_runs_through_parquet_cache_and_both_sl_trainers(formula_config, tmp_path):
    from bang_rl.training.emotion_sl_trainer import EmotionInferenceSLTrainer, EmotionSLConfig
    from bang_rl.training.sl_trainer import IntentInferenceSLTrainer, IntentSLConfig

    collect_games(formula_config, 12, tmp_path / "canonical")
    episodes = ParquetCanonicalShardWriter.read_dataset(tmp_path / "canonical")
    split = EpisodeSplitter(counts=(8, 2, 2), seed=0, manifest_path=tmp_path / "split.json").split(episodes)
    emotion_config = formula_label_emotion_config()
    assert emotion_config.emotion_count == LABEL_DIM
    artifacts = NPZTensorCacheWriter(
        tmp_path / "cache", emotion_schema_version=FORMULA_LABEL_SCHEMA, emotion_dimension=LABEL_DIM
    ).write_all(split)
    train, validation, test = (
        NPZTensorCacheLoader(artifacts[name].tensor_path) for name in ("train", "validation", "test")
    )
    intent = IntentInferenceSLTrainer(
        train, validation, test_cache=test, config=IntentSLConfig(epochs=1, batch_size=32)
    ).fit()
    emotion = EmotionInferenceSLTrainer(
        train, validation, test_cache=test, config=EmotionSLConfig(epochs=1, batch_size=32),
        emotion_config=emotion_config,
    ).fit()
    assert len(intent.history) == 1 and len(emotion.history) == 1


def test_cli_collects_into_an_empty_folder(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "collect_data.py"
    spec = importlib.util.spec_from_file_location("collect_data_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "cli"
    emotion_file = Path(__file__).resolve().parents[2] / "emotion_formula.example.json"
    assert module.main(["--games", "2", "--out", str(out), "--emotion-file", str(emotion_file)]) == 0
    assert (out / "collection_summary.json").is_file()
    assert module.main(["--games", "1", "--out", str(out), "--emotion-file", str(emotion_file)]) == 2
    assert "not empty" in capsys.readouterr().err
