"""Dummies using the main agent's trained role inference as beliefs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bang_rl.agents.dummy_agent.belief_source import describe_belief_spec, parse_belief_spec
from bang_rl.agents.main_agent import EmotionInferenceModule, IntentInferenceModule
from bang_rl.agents.main_agent.dummy_belief_adapter import LearnedBeliefSource, _to_canonical
from bang_rl.data import EpisodeCollector
from bang_rl.data.game_collection import (
    build_game,
    collect_games,
    drive_game,
    formula_label_emotion_config,
    load_collection_config,
    play_and_collect_game,
)
from bang_rl.training.inference_artifact import save_phase2_inference_artifact


@pytest.fixture(scope="module")
def config():
    return load_collection_config()


@pytest.fixture(scope="module")
def phase2_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("learned")
    return save_phase2_inference_artifact(
        root / "untrained.phase2model",
        IntentInferenceModule(seed=1),
        EmotionInferenceModule(config=formula_label_emotion_config(), seed=2),
    )


def test_learned_spec_keeps_path_case_and_drive_colons():
    assert parse_belief_spec("learned:C:/Runs/Model.phase2model") == ("learned", (1.0,))
    assert parse_belief_spec("LEARNED@0.5:x.phase2model") == ("learned", (0.5,))
    assert describe_belief_spec("learned@0.5:C:/Runs/Model.phase2model") == "learned:0.5:Model.phase2model"
    for bad in ("learned", "learned:", "learned@2:x", "learned@x:y"):
        with pytest.raises(ValueError):
            parse_belief_spec(bad)


def test_four_way_rows_split_outlaw_mass():
    assert _to_canonical([0.1, 0.2, 0.4, 0.3]).tolist() == [0.1, 0.2, 0.2, 0.2, 0.3]
    assert _to_canonical([0.2] * 5).tolist() == [0.2] * 5


def test_learned_beliefs_drive_dummies_without_privileged_input(config, phase2_model):
    env, agents, episode_id = build_game(config, seed=3, belief=f"learned:{phase2_model}")
    for agent in agents:
        assert isinstance(agent.belief_source, LearnedBeliefSource)
    episode = drive_game(env, agents, config, EpisodeCollector(), episode_id=episode_id, seed=3)
    assert episode.termination_status == "normal"
    assert episode.metadata["belief_source"] == f"learned:1:{Path(phase2_model).name}"
    for seat, agent in enumerate(agents):
        beliefs = agent.role_beliefs
        assert np.allclose(beliefs.sum(axis=1), 1.0, atol=1e-5)
        assert beliefs[seat].tolist() == agent.rule_beliefs[seat].tolist()  # own row from the rule estimator


def test_learned_beliefs_keep_roles_the_rule_estimator_rules_out():
    source = LearnedBeliefSource(None, weight=1.0)
    source.on_episode_start(1, None)  # seat 1, a deputy
    source._cached_step = 7
    source._cached_rows = np.full((5, 5), 0.2)  # model spreads mass everywhere
    rule = np.zeros((5, 5))
    rule[0] = [1, 0, 0, 0, 0]  # public sheriff
    rule[1] = [0, 1, 0, 0, 0]  # self
    rule[2:] = [0, 0, 0.35, 0.35, 0.3]  # no second sheriff or deputy
    beliefs = source.beliefs(rule, {"environment_step": 7})
    assert beliefs[0].tolist() == [1, 0, 0, 0, 0]
    assert beliefs[1].tolist() == [0, 1, 0, 0, 0]
    for seat in (2, 3, 4):
        assert beliefs[seat][0] == 0 and beliefs[seat][1] == 0
        assert np.allclose(beliefs[seat][2:], 1 / 3, atol=1e-6)


def test_learned_beliefs_are_reproducible_and_blend_with_weight(config, phase2_model):
    first, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=6, belief=f"learned:{phase2_model}")
    second, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=6, belief=f"learned:{phase2_model}")
    assert [s.action for s in first.steps] == [s.action for s in second.steps]
    rule, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=6)
    zero, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=6, belief=f"learned@0:{phase2_model}")
    assert [s.action for s in rule.steps] == [s.action for s in zero.steps]  # weight 0 == rule


def test_collect_records_learned_source(config, phase2_model, tmp_path):
    summary = collect_games(config, 2, tmp_path / "learned", belief=f"learned:{phase2_model}")
    assert summary["belief_source"] == f"learned:1:{Path(phase2_model).name}"


def test_iteration_cli_runs_two_rounds(tmp_path):
    script = Path(__file__).resolve().parents[2] / "iterate_dummies.py"
    spec = importlib.util.spec_from_file_location("iterate_dummies_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    emotion_file = Path(__file__).resolve().parents[2] / "emotion_formula.example.json"
    out = tmp_path / "iterate"
    assert module.main([
        "--rounds", "2", "--games", "12", "--eval-games", "3", "--epochs", "1",
        "--emotion-file", str(emotion_file), "--out", str(out),
    ]) == 0
    rounds = json.loads((out / "rounds.json").read_text(encoding="utf-8"))
    assert [row["round"] for row in rounds] == ["start", 0, 1]
    assert rounds[1]["collected_with"] == "rule"
    assert rounds[2]["collected_with"].startswith("learned:1:model.phase2model")
    assert (out / "round_1" / "model.phase2model").is_file()
    assert module.main(["--rounds", "1", "--games", "2", "--out", str(out)]) == 2  # refuses a used folder
