"""Focused tests for TASK-4-1 EmotionRuntime boundaries and persistence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bang_rl.agents import EmotionModule
from bang_rl.config import EmotionConfig
from bang_rl.env import Action, ActionResult, EventType, GameEvent
from bang_rl.runtime import EmotionRuntime, EmotionRuntimeError


def config() -> EmotionConfig:
    return EmotionConfig(
        schema_version="runtime-emotion-v1",
        initial_baseline=0.5,
        emotions=[
            {
                "name": "fear",
                "initial_value": 0.5,
                "change_rate": 0.2,
                "decay_rate": 0.05,
                "event_updates": {"bang": 0.2, "player_eliminated": 0.1},
                "action_updates": {"pass": 0.1, "bang": -0.1},
                "reward_updates": {"positive": 0.05, "negative": -0.05},
                "terminal_updates": {"terminated": 0.2, "truncated": -0.2},
            },
            {
                "name": "calm",
                "initial_value": 0.25,
                "decay_rate": 0.1,
                "terminal_updates": {"terminated": 0.2, "truncated": -0.2},
            },
        ],
    )


def modules() -> dict[int, EmotionModule]:
    return {index: EmotionModule(config(), player_idx=index) for index in range(5)}


def event(
    event_id: int,
    event_type: EventType = EventType.BANG,
    *,
    environment_step: int | None = None,
) -> GameEvent:
    return GameEvent(
        event_id=event_id,
        environment_step=event_id if environment_step is None else environment_step,
        event_type=event_type,
        actor_idx=0,
        target_idx=1 if event_type is EventType.BANG else None,
    )


def result(step: int, *, terminal: bool = False, truncated: bool = False) -> ActionResult:
    return ActionResult(
        action=Action.pass_action(0),
        environment_step=step,
        events=(event(step),),
        reward=1.0,
        terminated=terminal,
        truncated=truncated,
    )


def test_valid_boundary_applies_events_decay_label_action_reward_and_terminal() -> None:
    runtime = EmotionRuntime(emotion_modules=modules())
    runtime.reset({"episode_id": "episode-1", "seed": 7})

    labels = runtime.before_action(0, action_mask=np.ones(6, dtype=np.uint8), actor_idx=0)
    np.testing.assert_allclose(labels[0].values, [0.5, 0.25])
    committed = runtime.on_action_result(result(1))
    assert committed is not None
    # The result event is staged, not part of the just-completed action label.
    assert committed.labels[0].input_event_ids == ()
    assert committed.post_action_event_ids == (1,)
    np.testing.assert_allclose(runtime.private_state[0], [0.65, 0.25])

    next_labels = runtime.before_action(1, actor_idx=1)
    assert next_labels[0].input_event_ids == (1,)
    # Event (+.2), then one decay step toward .5 (-.05).
    np.testing.assert_allclose(next_labels[0].values, [0.8, 0.25])

    terminal_result = result(2, terminal=True)
    runtime.on_action_result(terminal_result)
    assert runtime.terminated is True
    assert runtime.truncated is False
    np.testing.assert_allclose(runtime.private_state[0], [1.0, 0.45])


def test_invalid_action_and_retry_do_not_advance_runtime() -> None:
    runtime = EmotionRuntime(emotion_modules=modules())
    runtime.reset({"episode_id": "episode-2", "seed": 8})
    labels = runtime.before_action(0, actor_idx=0)
    before = runtime.to_json()

    assert runtime.on_invalid_action("bad-action") == labels
    assert runtime.on_action_result(result(1), valid=False) is None
    assert runtime.to_json() == before
    assert runtime.before_action(0, actor_idx=0) == labels

    with pytest.raises(EmotionRuntimeError):
        runtime.on_action_result(result(2))


def test_labels_align_all_players_to_same_pre_action_public_history() -> None:
    runtime = EmotionRuntime(emotion_modules=modules())
    runtime.reset({"episode_id": "episode-3", "seed": 9})
    runtime.before_action(0, actor_idx=0)
    runtime.on_action_result(result(1))
    labels = runtime.before_action(1, actor_idx=1)

    assert {label.environment_step for label in labels.values()} == {1}
    assert {label.input_event_ids for label in labels.values()} == {(1,)}
    for label in labels.values():
        assert label.public_event_sequence[0]["event_id"] == 1
        assert "emotion_label" not in label.public_observation
        assert "emotion_state" not in label.public_observation


def test_bounds_privacy_and_persistence_are_enforced() -> None:
    runtime = EmotionRuntime(emotion_modules=modules())
    runtime.reset({"episode_id": "episode-4", "seed": 10})
    runtime.before_action(0, actor_idx=0)

    public = runtime.public_observation()
    encoded_public = json.dumps(public)
    assert "emotion_state" not in encoded_public
    assert "emotion_label" not in encoded_public
    with pytest.raises(ValueError, match="privileged"):
        runtime.public_observation(observation={"emotion_label": [0.1, 0.2]})

    payload = runtime.state_dict()
    assert payload["state_version"] == "1.0"
    assert payload["modules"]["0"]["emotion_schema_version"] == "runtime-emotion-v1"
    restored = EmotionRuntime.from_json(runtime.to_json())
    assert restored.to_json() == runtime.to_json()
    np.testing.assert_array_equal(restored.private_state[0], runtime.private_state[0])

    with pytest.raises(ValueError):
        EmotionRuntime(emotion_modules={0: EmotionModule(config(), player_idx=0), 1: EmotionModule(EmotionConfig(schema_version="other"), player_idx=1)})


def test_reset_starts_a_new_deterministic_episode_and_truncated_has_no_normal_terminal() -> None:
    runtime = EmotionRuntime(emotion_modules=modules())
    runtime.reset({"episode_id": "episode-a", "seed": 11})
    runtime.before_action(0, actor_idx=0)
    runtime.on_action_result(result(1, truncated=True))
    assert runtime.truncated is True
    assert runtime.terminated is False

    runtime.reset({"episode_id": "episode-b", "seed": 12})
    assert runtime.episode_id == "episode-b"
    assert runtime.episode_seed == 12
    assert runtime.processed_event_ids == ()
    assert runtime.pending_event_ids == ()
    np.testing.assert_allclose(runtime.private_state[0], [0.5, 0.25])
    runtime.before_action(0, actor_idx=0)
    assert runtime.to_json() == EmotionRuntime.from_json(runtime.to_json()).to_json()
