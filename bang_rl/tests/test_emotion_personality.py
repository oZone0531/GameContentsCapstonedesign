"""Tests for emotion personality presets (formula block personality_presets / personality_mode)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from bang_rl.agents import DummyAgent
from bang_rl.agents.dummy_agent.emotion_formula import FormulaEmotionModule, draw_personality
from bang_rl.config import EmotionConfig, SystemConfig
from bang_rl.data.game_collection import build_game, load_collection_config, personality_labels

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "emotion_formula.example.json"
PERSONALITY = ROOT / "emotion_formula.personality.json"


def _doc(path: Path = EXAMPLE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _module(doc: dict, player_idx: int = 0, seed: int | None = 3) -> FormulaEmotionModule:
    config = EmotionConfig.model_validate(doc)
    module = FormulaEmotionModule(config, player_idx=player_idx, seed=seed)
    module.on_episode_start({"episode_id": "e"}, seed=seed)
    return module


def test_personality_file_matches_example_except_mode():
    example, personality = _doc(EXAMPLE), _doc(PERSONALITY)
    assert example["formula"]["personality_mode"] == "role"
    assert personality["formula"]["personality_mode"] == "random"
    personality["formula"]["personality_mode"] = "role"
    assert personality == example


def test_role_mode_assigns_no_preset_and_keeps_base_params():
    doc = _doc(EXAMPLE)
    module = _module(doc)
    assert module.personality is None
    assert module.params.p_A == pytest.approx(doc["formula"]["params"]["p_A"])


def test_random_mode_applies_the_drawn_preset_on_top_of_base_params():
    doc = _doc(PERSONALITY)
    presets = doc["formula"]["personality_presets"]
    for seed in range(20):
        module = _module(doc, seed=seed)
        assert module.personality in presets
        expected = {**doc["formula"]["params"], **presets[module.personality]}
        for name, value in expected.items():
            assert getattr(module.params, name) == pytest.approx(value)


def test_preset_overrides_role_overrides():
    doc = _doc(PERSONALITY)
    doc["formula"]["personality_presets"] = {"only": {"b_A": 0.4}}
    doc["formula"]["personality_by_role"] = {name: {"b_A": 0.2, "b_F": 0.3} for name in ("sheriff", "deputy", "outlaw", "traitor")}
    config = EmotionConfig.model_validate(doc)
    module = FormulaEmotionModule(config, player_idx=1, seed=0)
    module.on_episode_start({"episode_id": "e"}, seed=0)
    module._set_role(1)
    assert module.params.b_A == pytest.approx(0.4)   # preset wins
    assert module.params.b_F == pytest.approx(0.3)   # role value kept where the preset is silent


def test_draw_is_reproducible_and_covers_every_preset():
    presets = {"a": {}, "b": {}, "c": {}, "d": {}}
    draws = [draw_personality(presets, seed) for seed in range(400)]
    assert draws == [draw_personality(presets, seed) for seed in range(400)]
    counts = Counter(draws)
    assert set(counts) == set(presets)
    assert min(counts.values()) > 60  # roughly uniform (expected 100 each)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"personality_mode": "sometimes"}, "personality_mode"),
        ({"personality_presets": {}}, "personality_presets"),
        ({"personality_presets": {"bad": {"not_a_param": 1.0}}}, "unknown emotion formula params"),
    ],
)
def test_invalid_personality_settings_are_rejected(change, message):
    doc = _doc(PERSONALITY)
    doc["formula"].update(change)
    with pytest.raises(ValidationError, match=message):
        EmotionConfig.model_validate(doc)


def test_collection_games_draw_presets_independently_of_role_and_record_them():
    config = load_collection_config(emotion_file=PERSONALITY)
    by_role: dict[int, Counter] = {}
    for seed in range(60):
        env, agents, _ = build_game(config, seed=seed)
        names = personality_labels(agents)
        assert all(name is not None for name in names)
        again = personality_labels(build_game(config, seed=seed)[1])
        assert names == again  # same seed -> same personalities
        for role, name in zip(env.state.roles, names):
            by_role.setdefault(int(role), Counter())[name] += 1
    presets = set(_doc(PERSONALITY)["formula"]["personality_presets"])
    for counts in by_role.values():
        assert len(counts) >= 3 and set(counts) <= presets  # every role sees several presets


def test_example_file_games_record_no_personality():
    config = load_collection_config(emotion_file=EXAMPLE)
    _, agents, _ = build_game(config, seed=1)
    assert personality_labels(agents) == [None] * 5


def test_state_round_trip_keeps_the_personality():
    config = SystemConfig.model_validate({"preset": {"emotion": _doc(PERSONALITY)}})
    _, agents, _ = build_game(config, seed=4)
    for agent in agents:
        copy = DummyAgent.from_state_dict(json.loads(json.dumps(agent.state_dict())))
        assert copy.emotion_module.personality == agent.emotion_module.personality
        assert copy.emotion_module.params == agent.emotion_module.params
