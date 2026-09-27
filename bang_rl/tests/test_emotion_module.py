"""Focused tests for TASK-2-4 Dummy EmotionModule."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from bang_rl.agents import DummyAgent, EmotionModule
from bang_rl.config import EmotionConfig


def emotion_config() -> EmotionConfig:
    return EmotionConfig(
        schema_version="emotion-test-v1",
        initial_baseline=0.25,
        emotions=[
            {
                "name": "fear",
                "initial_value": 0.8,
                "change_rate": 0.3,
                "decay_rate": 0.1,
                "event_updates": {"bang": 0.4, "player_eliminated": -0.2},
                "action_updates": {"pass": 0.05, "bang": -0.05},
                "reward_updates": {"positive": 0.1, "negative": -0.15},
                "terminal_updates": {"terminated": -0.4, "truncated": 0.2},
                "effects": {"pass": 0.2, "bang": -0.3},
            },
            {
                "name": "trust",
                "initial_value": 0.25,
                "decay_rate": 0.2,
                "event_updates": {"bang": -0.3},
            },
        ],
    )


def test_config_defines_dynamic_baseline_and_state_shape() -> None:
    module = EmotionModule(emotion_config(), player_idx=2)

    assert module.emotion_names == ("fear", "trust")
    assert module.emotion_count == 2
    np.testing.assert_allclose(module.values, [0.8, 0.25])
    np.testing.assert_allclose(module.baseline, [0.8, 0.25])
    assert module.effects_for_action("pass") == {"fear": 0.2}


def test_event_updates_are_idempotent_and_values_are_clamped() -> None:
    module = EmotionModule(emotion_config())
    event = {"event_id": 4, "event_type": "bang", "public": True}

    np.testing.assert_allclose(module.apply_event(event), [1.0, 0.0])
    first = module.values
    np.testing.assert_array_equal(module.apply_event(event), first)
    assert module.processed_event_ids == (4,)

    module.apply_event({"event_id": 5, "event_type": "player_eliminated"})
    np.testing.assert_allclose(module.values, [0.8, 0.0])


def test_decay_is_applied_at_most_once_per_environment_step() -> None:
    module = EmotionModule(emotion_config())
    module.apply_event({"event_id": 1, "event_type": "bang"})

    # Initial step zero has no previous step to decay.
    np.testing.assert_allclose(module.advance_step(0), [1.0, 0.0])
    np.testing.assert_allclose(module.advance_step(1), [0.9, 0.2])
    np.testing.assert_allclose(module.advance_step(1), [0.9, 0.2])

    with pytest.raises(ValueError):
        module.advance_step(0)


def test_action_reward_terminal_rules_and_invalid_mask_paths() -> None:
    module = EmotionModule(emotion_config())
    initial = module.values

    np.testing.assert_array_equal(module.apply_action(1, valid=False), initial)
    np.testing.assert_array_equal(
        module.apply_action(1, action_mask=np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)),
        initial,
    )
    np.testing.assert_allclose(
        module.apply_action(0, action_mask=np.ones(6, dtype=np.uint8)), [0.85, 0.25]
    )
    np.testing.assert_allclose(module.apply_reward(1.0), [0.95, 0.25])
    np.testing.assert_allclose(module.apply_terminal({"terminated": True}), [0.55, 0.25], atol=1e-6)

    terminal_state = module.values
    np.testing.assert_array_equal(module.apply_reward(-1.0, valid=False), terminal_state)


def test_terminal_lifecycle_reset_and_agent_integration() -> None:
    module = EmotionModule(emotion_config(), player_idx=1)
    module.on_episode_start({"episode_id": "episode-a", "seed": 22})
    module.apply_event({"event_id": 1, "event_type": "bang"})
    module.on_episode_end({"terminated": False, "truncated": True})
    assert module.episode_id == "episode-a"
    assert module.truncated is True
    np.testing.assert_allclose(module.values, [1.0, 0.0])

    agent = DummyAgent(
        agent_id="player_1",
        player_idx=1,
        emotion_config=emotion_config(),
    )
    agent.on_episode_start({"episode_id": "episode-b", "seed": 23})
    np.testing.assert_allclose(agent.emotion_state, [0.8, 0.25])
    agent.on_episode_end({"terminated": True})
    assert agent.emotion_module.terminated is True
    agent.emotion_module.reset()
    np.testing.assert_allclose(agent.emotion_state, [0.8, 0.25])


def test_deterministic_replay_and_json_persistence() -> None:
    events = [
        {"event_id": 1, "event_type": "bang"},
        {"event_id": 2, "event_type": "player_eliminated"},
    ]
    first = EmotionModule(emotion_config(), player_idx=3)
    first.prepare_action(0, events[:1])
    first.prepare_action(1, events[1:])
    first.apply_action("pass")
    first.apply_reward(-1.0)
    expected = first.values

    restored = EmotionModule.from_json(first.to_json())
    np.testing.assert_array_equal(restored.values, expected)
    assert restored.to_json() == first.to_json()
    np.testing.assert_array_equal(restored.apply_event(events[1]), expected)

    replay = EmotionModule(emotion_config(), player_idx=3)
    replay.prepare_action(0, events[:1])
    replay.prepare_action(1, events[1:])
    replay.apply_action("pass")
    replay.apply_reward(-1.0)
    np.testing.assert_array_equal(replay.values, expected)


def test_public_privacy_and_schema_versioning() -> None:
    module = EmotionModule(emotion_config())
    with pytest.raises(ValueError, match="privileged"):
        module.apply_event({"event_id": 1, "event_type": "bang", "true_roles": [0, 1, 2, 3, 4]})

    encoded = module.state_dict()
    assert encoded["schema_version"] == "1.0"
    assert encoded["emotion_schema_version"] == "emotion-test-v1"
    assert "true_roles" not in module.to_json()
    assert "reward_breakdown" not in module.to_json()
    assert "BangEnvironment" not in inspect.getsource(EmotionModule)


def test_emotion_config_can_add_and_remove_items_without_code_changes() -> None:
    one = EmotionConfig(emotions={"calm": {"initial": 0.4}})
    two = EmotionConfig(
        emotions={"calm": {"initial": 0.4}, "alert": {"initial": 0.9}},
    )

    assert EmotionModule(one).emotion_names == ("calm",)
    assert EmotionModule(two).emotion_names == ("calm", "alert")
