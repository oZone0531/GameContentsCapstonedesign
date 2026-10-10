"""Tests for the emotion formula module and its DummyAgent / GameService wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from bang_rl.agents import DummyAgent
from bang_rl.agents.dummy_agent.emotion_formula import (
    EMOTION_NAMES,
    EmotionParams,
    EmotionState,
    FormulaEmotionModule,
)
from bang_rl.agents.dummy_agent.emotion_module import EmotionModule
from bang_rl.app import GameService, SessionConfig
from bang_rl.config import EmotionConfig, SystemConfig
from bang_rl.gui.widgets.spectator_overlay import emotion_bars_for_seat

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "emotion_formula.example.json"


def _example(**params: float) -> dict:
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc["formula"]["params"].update(params)
    return doc


def _config(emotion: dict | None) -> SystemConfig:
    return SystemConfig.model_validate({"preset": {"emotion": emotion}} if emotion else {})


def _session(cfg: SystemConfig, env_seed: int):
    """GameService dummy_five session pinned to a fixed environment seed."""

    service = GameService()
    handle = service.create_session(SessionConfig(mode="dummy_five", config=cfg, seed=0))
    session = service._session(handle)
    env = session.environment
    episode = f"e{env_seed}"
    env.reset(seed=env_seed, episode_id=episode)
    for index, agent in enumerate(session.agents):
        agent.on_episode_start({"episode_id": episode, "seed": env_seed, "observation": env.observe(index)})
    return service, handle, session


def _play(service, handle, session, max_steps: int | None = None) -> list[int]:
    env = session.environment
    actions = []
    while not (env.state.terminated or env.state.truncated):
        if max_steps is not None and len(actions) >= max_steps:
            break
        actions.append(service.step(handle).action)
    return actions


# -- formula math (spec section 8, example 1) --------------------------------
def test_spec_example_one_hit_by_seat_three():
    beliefs = np.zeros((5, 5))
    beliefs[0, 0] = beliefs[1, 1] = 1.0
    beliefs[2:, 2:4] = 1.0 / 3.0
    beliefs[2:, 4] = 1.0 / 3.0
    state = EmotionState(1, EmotionParams(sigma=0.0, sigma_R=0.0), np.random.default_rng(0))
    event = {"event_id": 1, "event_type": "bang", "actor_idx": 3, "target_idx": 1, "hp_after": 3}
    label = state.step(1, [event], beliefs=beliefs, self_role=1, sheriff_idx=0,
                       healths=[5, 3, 4, 4, 4], alive=[True] * 5)
    assert state.R[3] == pytest.approx(-0.15)
    # Team decision 2026-10-06: internal R=-0.15 <-> stored label 0.425.
    assert label[4 + 3] == pytest.approx(0.425)
    np.testing.assert_allclose(label, [0.803, 0.055, 0.197, 0.153, 0.5, 0.5, 0.5, 0.425, 0.5], atol=1e-3)
    contribution = state.action_contribution()
    assert int(np.argmax(contribution)) == 1 + 3  # shooting seat 3 scores highest


# -- config selection ---------------------------------------------------------
def test_default_config_keeps_rule_table_module():
    agent = DummyAgent("player_1", seed=1, player_idx=1, emotion_config=EmotionConfig())
    assert isinstance(agent.emotion_module, EmotionModule)
    assert not agent.uses_emotion_formula
    agent.observe_public_boundary({"environment_step": [0]})  # no-op, must not raise


def test_formula_block_selects_formula_module():
    agent = DummyAgent("player_2", seed=1, player_idx=2, emotion_config=EmotionConfig.model_validate(_example()))
    assert isinstance(agent.emotion_module, FormulaEmotionModule)
    assert agent.emotion_module.emotion_names == EMOTION_NAMES


def test_unknown_formula_names_are_rejected():
    with pytest.raises(ValidationError):
        EmotionConfig.model_validate({"formula": {"params": {"beta_bangg": 0.1}}})
    with pytest.raises(ValidationError):
        EmotionConfig.model_validate({"formula": {"personality_by_role": {"cowboy": {"p_A": 1.2}}}})


def test_personality_by_role_is_applied_from_own_role():
    doc = _example()
    doc["formula"]["personality_by_role"] = {"outlaw": {"b_A": 0.3}}
    service, handle, session = _session(_config(doc), env_seed=5)
    roles = [int(r) for r in session.environment.state.roles]
    for index, agent in enumerate(session.agents):
        expected = 0.3 if roles[index] in (2, 3) else 0.1
        assert agent.emotion_module.params.b_A == pytest.approx(expected)
        assert agent.emotion_module.self_role == roles[index]


# -- integration ----------------------------------------------------------------
@pytest.mark.parametrize("env_seed", [0, 1, 2, 3, 4])
def test_formula_with_zero_effect_plays_the_same_game_as_default(env_seed):
    default = _play(*_session(_config(None), env_seed))
    zeroed = _play(*_session(_config(_example(lambda_E=0.0, sigma=0.0, sigma_R=0.0)), env_seed))
    assert zeroed == default


def test_every_living_dummy_updates_every_step_from_its_own_view():
    service, handle, session = _session(_config(_example()), env_seed=7)
    roles = [int(r) for r in session.environment.state.roles]
    _play(service, handle, session, max_steps=3)
    step = int(session.environment.state.environment_step)
    for index, agent in enumerate(session.agents):
        module = agent.emotion_module
        assert module.self_role == roles[index]          # no role leaked from another seat
        if session.environment.state.alive[index]:
            # observed at least at the previous boundary even when it did not act
            assert module.last_environment_step is not None and module.last_environment_step >= step - 1


def test_state_round_trip_continues_identically():
    service, handle, session = _session(_config(_example()), env_seed=11)
    _play(service, handle, session, max_steps=4)
    saved = [agent.state_dict() for agent in session.agents]
    restored = [DummyAgent.from_state_dict(json.loads(json.dumps(s))) for s in saved]
    for original, copy in zip(session.agents, restored):
        assert isinstance(copy.emotion_module, FormulaEmotionModule)
        np.testing.assert_allclose(copy.emotion_module.values, original.emotion_module.values)
        np.testing.assert_allclose(copy.emotion_module.relation, original.emotion_module.relation)
        # identical RNG stream -> identical future noise
        assert copy.emotion_module._state.rng.normal() == original.emotion_module._state.rng.normal()


def test_gui_bars_and_label_fields_are_exported():
    service, handle, session = _session(_config(_example()), env_seed=3)
    _play(service, handle, session, max_steps=3)
    internals = service.export_agent_internals(handle)
    descriptor = internals["seats"]["0"]
    bars = emotion_bars_for_seat(descriptor)
    assert [bar.name for bar in bars] == list(EMOTION_NAMES)
    assert all(bar.description for bar in bars)
    state = descriptor["state"]["emotion_module"]
    assert len(state["label"]) == 9 and len(state["label_mask"]) == 9
    rel, rel01 = np.asarray(state["relation"]), np.asarray(state["relation_01"])
    active = np.asarray(state["label_mask"][4:])
    np.testing.assert_allclose(rel01[active], (rel[active] + 1.0) / 2.0)
