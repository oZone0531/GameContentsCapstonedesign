"""Pluggable dummy belief sources (rule / oracle warm start)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bang_rl.agents.dummy_agent.belief_source import (
    BeliefSource,
    HintBeliefSource,
    OracleMixBeliefSource,
    make_belief_source,
    parse_belief_spec,
    truth_matrix,
)
from bang_rl.data import EpisodeCollector
from bang_rl.data.game_collection import (
    build_game,
    collect_games,
    drive_game,
    load_collection_config,
    play_and_collect_game,
)


@pytest.fixture(scope="module")
def config():
    return load_collection_config()


def test_truth_matrix_splits_outlaw_mass_and_hides_outlaw_label():
    matrix = truth_matrix([0, 1, 2, 3, 4])
    assert matrix[0].tolist() == [1, 0, 0, 0, 0]
    assert matrix[2].tolist() == matrix[3].tolist() == [0, 0, 0.5, 0.5, 0]
    assert matrix[4].tolist() == [0, 0, 0, 0, 1]


def test_spec_parsing():
    assert parse_belief_spec(None) == ("rule", ())
    assert parse_belief_spec("rule") == ("rule", ())
    assert parse_belief_spec("oracle:0.3") == ("oracle", (0.3,))
    assert parse_belief_spec("hint:0.7") == ("hint", (0.7, 0.5))
    assert parse_belief_spec("hint:0.7:0.3") == ("hint", (0.7, 0.3))
    assert make_belief_source("rule", [0, 1, 2, 3, 4]) is None
    assert isinstance(make_belief_source("oracle:1", [0, 1, 2, 3, 4]), BeliefSource)
    assert isinstance(make_belief_source("hint:0.7", [0, 1, 2, 3, 4], seed=1), HintBeliefSource)
    for bad in ("oracle:1.5", "oracle:x", "learned", "hint", "oracle:0.1:0.2", "hint:0.5:0.5:0.5"):
        with pytest.raises(ValueError):
            parse_belief_spec(bad)


def test_oracle_mix_keeps_own_row_and_rows_sum_to_one():
    source = OracleMixBeliefSource([0, 1, 2, 3, 4], alpha=0.4)
    source.on_episode_start(1, None)
    rule = np.full((5, 5), 0.2)
    rule[1] = [0, 1, 0, 0, 0]
    mixed = source.beliefs(rule, None)
    assert np.allclose(mixed.sum(axis=1), 1.0)
    assert mixed[1].tolist() == rule[1].tolist()
    assert mixed[4, 4] == pytest.approx(0.6 * 0.2 + 0.4)


def test_rule_and_oracle_zero_play_identical_games(config):
    for seed in range(5):
        rule, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=seed)
        zero, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=seed, belief="oracle:0")
        assert [s.action for s in rule.steps] == [s.action for s in zero.steps]
        assert [s.emotion_labels for s in rule.steps] == [s.emotion_labels for s in zero.steps]
        assert rule.metadata["belief_source"] == "rule"
        assert zero.metadata["belief_source"] == "oracle:0"


def test_full_oracle_beliefs_reach_intent_and_emotion(config):
    env, agents, episode_id = build_game(config, seed=3, belief="oracle:1")
    roles = list(env.state.roles)
    truth = truth_matrix(roles)
    for index, agent in enumerate(agents):
        beliefs = agent.role_beliefs
        others = [seat for seat in range(5) if seat != index]
        assert np.allclose(beliefs[others], truth[others])
        assert not np.allclose(agent.rule_beliefs[others], truth[others])
    episode = drive_game(env, agents, config, EpisodeCollector(), episode_id=episode_id, seed=3)
    assert episode.termination_status == "normal"


def test_oracle_changes_play(config):
    changed = 0
    for seed in range(20):
        rule, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=seed)
        full, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=seed, belief="oracle:1")
        changed += [s.action for s in rule.steps] != [s.action for s in full.steps]
    assert changed > 0


def test_collect_games_records_belief_source(config, tmp_path):
    summary = collect_games(config, 2, tmp_path / "o", belief="hint:0.7", keep_episodes=True)
    assert summary["belief_source"] == "hint:0.7:0.5"
    assert summary["episodes"][0].metadata["belief_source"] == "hint:0.7:0.5"
    saved = json.loads((tmp_path / "o" / "collection_summary.json").read_text(encoding="utf-8"))
    assert saved["belief_source"] == "hint:0.7:0.5"
    with pytest.raises(ValueError):
        collect_games(config, 1, tmp_path / "bad", belief="oracle:2")


ROLES = [0, 1, 2, 3, 4]


def _hint(accuracy, seed=0, player=0, strength=0.5):
    source = HintBeliefSource(ROLES, accuracy, strength=strength, rng=np.random.default_rng(seed))
    source.on_episode_start(player, None)
    return source


def test_hint_accuracy_one_points_at_truth_and_zero_never_does():
    right = _hint(1.0)
    assert right.hint_correct.all()
    assert np.allclose(right.hint, truth_matrix(ROLES))
    for seed in range(20):
        wrong = _hint(0.0, seed=seed)
        others = [1, 2, 3, 4]
        assert not wrong.hint_correct[others].any()
        assert wrong.hint_correct[0]  # the sheriff is public: never hinted wrong
        for seat in others:
            assert not np.allclose(wrong.hint[seat], truth_matrix(ROLES)[seat])
            assert wrong.hint[seat, 0] == 0.0  # a wrong hint never names the sheriff


def test_hint_rate_matches_accuracy_and_is_reproducible():
    rate = np.mean([_hint(0.7, seed=seed).hint_correct[1:].mean() for seed in range(400)])
    assert rate == pytest.approx(0.7, abs=0.04)
    first, second = _hint(0.5, seed=9), _hint(0.5, seed=9)
    assert np.array_equal(first.hint, second.hint)


def test_hint_keeps_own_row_and_public_facts():
    source = _hint(0.0, seed=3, player=2)
    rule = np.full((5, 5), 0.2)
    rule[2] = [0, 0, 1, 0, 0]  # own role
    rule[4] = [0, 0, 0, 0, 1]  # publicly revealed traitor
    beliefs = source.beliefs(rule, None)
    assert beliefs[2].tolist() == rule[2].tolist()
    assert beliefs[4].tolist() == rule[4].tolist()
    assert np.allclose(beliefs.sum(axis=1), 1.0)


def test_hint_games_are_reproducible_and_keep_mistakes(config):
    first, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=4, belief="hint:0.7")
    second, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=4, belief="hint:0.7")
    assert [s.action for s in first.steps] == [s.action for s in second.steps]
    team = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
    friendly = 0
    for seed in range(40):
        episode, _, _ = play_and_collect_game(config, EpisodeCollector(), seed=seed, belief="hint:0.5")
        roles = episode.role_labels
        friendly += sum(
            1 for step in episode.steps
            if step.action > 0 and roles[step.actor_idx] != 4
            and team[roles[step.actor_idx]] == team[roles[step.action - 1]]
        )
    assert friendly > 0  # an imperfect hint still produces friendly fire
