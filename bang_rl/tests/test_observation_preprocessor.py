"""Focused TASK-3-1 tests for MainAgent observation preprocessing."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from bang_rl.agents.main_agent import (
    MODEL_HISTORY_LENGTH,
    ObservationPreprocessor,
    ObservationPreprocessorConfig,
)
from bang_rl.env import BangEnvironment, EnvironmentState, Role


def _observation(*, events=(), viewer_idx=0, current_player_idx=0) -> dict:
    beliefs = np.zeros((5, 5), dtype=np.float32)
    beliefs[0, int(Role.SHERIFF)] = 1.0
    if viewer_idx != 0:
        beliefs[viewer_idx, int(Role.OUTLAW_A)] = 1.0
    return {
        "current_player_idx": np.asarray([current_player_idx], dtype=np.uint8),
        "environment_step": np.asarray([1000], dtype=np.uint32),
        "round_index": np.asarray([500], dtype=np.uint32),
        "healths": np.asarray([5, 4, 3, 2, 1], dtype=np.uint8),
        "alive": np.asarray([True, True, True, True, True], dtype=np.bool_),
        "turn_order": np.arange(5, dtype=np.uint8),
        "sheriff_idx": np.asarray([0], dtype=np.uint8),
        "self_role": np.asarray(
            [int(Role.SHERIFF if viewer_idx == 0 else Role.OUTLAW_A)], dtype=np.uint8
        ),
        "role_visibility": np.asarray([2, 1, 0, 0, 0], dtype=np.uint8),
        "revealed_roles": np.asarray([0, 2, 255, 255, 255], dtype=np.uint8),
        "confirmed_role_distribution": beliefs,
        "event_sequence": tuple(events),
        "action_mask": np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8),
        "viewer_idx": viewer_idx,
    }


def _event(event_id: int, *, actor: int | None = 1, event_type: str = "bang") -> dict:
    return {
        "event_id": event_id,
        "environment_step": event_id,
        "event_type": event_type,
        "actor_idx": actor,
        "target_idx": 2 if event_type == "bang" else None,
        "hp_before": 4 if event_type == "bang" else None,
        "hp_after": 3 if event_type == "bang" else None,
        "eliminated_idx": None,
        "revealed_role": None,
        "public": True,
    }


def test_fixed_output_contract_normalization_and_mask() -> None:
    processed = ObservationPreprocessor().preprocess(_observation(viewer_idx=2, current_player_idx=4))

    assert processed.state_vector.shape == (9,)
    assert processed.state_vector.dtype == np.float32
    assert processed.player_state_matrix.shape == (5, 8)
    assert processed.player_state_matrix.dtype == np.float32
    assert processed.role_beliefs.shape == (5, 5)
    assert processed.role_beliefs.dtype == np.float32
    assert processed.event_history.shape == (4, 64, 17)
    assert processed.event_history.dtype == np.float32
    assert processed.event_valid_mask.shape == (4, 64)
    assert processed.event_valid_mask.dtype == np.bool_
    assert processed.action_mask.shape == (6,)
    assert processed.action_mask.dtype == np.uint8
    np.testing.assert_array_equal(processed.action_mask, [1, 0, 1, 1, 1, 1])

    # current player, environment step, round, sheriff, self role, self HP,
    # self alive, alive count, and viewer are all normalized deterministically.
    np.testing.assert_allclose(
        processed.state_vector,
        [1.0, 0.5, 0.25, 0.0, 0.5, 0.6, 1.0, 1.0, 0.5],
    )
    np.testing.assert_allclose(processed.player_state_matrix[:, 0], [1.0, 0.8, 0.6, 0.4, 0.2])
    np.testing.assert_array_equal(processed.role_beliefs[0], [1, 0, 0, 0, 0])


def test_empty_history_has_left_padding_and_no_valid_events() -> None:
    processed = ObservationPreprocessor().preprocess(_observation())

    assert not np.any(processed.event_valid_mask)
    assert not np.any(processed.event_history)


def test_events_are_split_by_non_self_actor_and_preserve_order() -> None:
    events = (_event(1, actor=0), _event(2, actor=1), _event(3, actor=3))
    processed = ObservationPreprocessor().preprocess(_observation(events=events, viewer_idx=0))

    # Other players are streams [1, 2, 3, 4]; actor 0 is excluded.
    assert processed.event_valid_mask.sum(axis=1).tolist() == [1, 0, 1, 0]
    assert processed.event_valid_mask[:, -1].tolist() == [True, False, True, False]
    # A bang event has the event-type one-hot at the last timestep.
    assert processed.event_history[0, -1, 1] == 1.0
    assert processed.event_history[2, -1, 1] == 1.0


def test_history_keeps_latest_64_events_and_left_pads_short_streams() -> None:
    events = tuple(_event(index, actor=1) for index in range(1, 66))
    processed = ObservationPreprocessor().preprocess(_observation(events=events))

    stream = processed.event_history[0]
    valid = processed.event_valid_mask[0]
    assert valid.all()
    # Event-id feature is index 10 and is normalized/clipped by max_episode_steps.
    assert stream[0, 10] == pytest.approx(2 / 2000)
    assert stream[-1, 10] == pytest.approx(65 / 2000)

    short = ObservationPreprocessor().preprocess(_observation(events=events[:3]))
    assert not np.any(short.event_valid_mask[0, : MODEL_HISTORY_LENGTH - 3])
    assert short.event_valid_mask[0, -3:].all()
    np.testing.assert_allclose(short.event_history[0, -3:, 10], [1 / 2000, 2 / 2000, 3 / 2000])


def test_public_role_reveal_is_featured_but_privileged_role_data_is_rejected() -> None:
    reveal = _event(1, actor=None, event_type="role_revealed")
    reveal["eliminated_idx"] = 1
    reveal["revealed_role"] = int(Role.DEPUTY)
    processed = ObservationPreprocessor().preprocess(_observation(events=(reveal,)))
    assert processed.event_valid_mask[0, -1]
    np.testing.assert_array_equal(processed.event_history[0, -1, 12:17], [0, 1, 0, 0, 0])

    private = copy.deepcopy(_observation())
    private["roles"] = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="privileged"):
        ObservationPreprocessor().preprocess(private)

    private_event = _event(1)
    private_event["public"] = False
    with pytest.raises(ValueError, match="private events"):
        ObservationPreprocessor().preprocess(_observation(events=(private_event,)))


def test_malformed_fixed_fields_and_action_mask_are_rejected() -> None:
    malformed = _observation()
    malformed["healths"] = np.zeros(4, dtype=np.uint8)
    with pytest.raises(ValueError, match="healths must have shape"):
        ObservationPreprocessor().preprocess(malformed)

    malformed = _observation()
    malformed["action_mask"] = np.asarray([1, 2, 0, 0, 0, 0], dtype=np.uint8)
    with pytest.raises(ValueError, match="action_mask"):
        ObservationPreprocessor().preprocess(malformed)

    malformed = _observation()
    malformed["event_sequence"] = ({"event_id": 1, "environment_step": 1},)
    with pytest.raises(ValueError, match="actor_idx or a public event subject"):
        ObservationPreprocessor().preprocess(malformed)


def test_batch_and_raw_environment_observation_compatibility() -> None:
    preprocessor = ObservationPreprocessor()
    first = _observation()
    second = _observation(viewer_idx=1, current_player_idx=3)
    batch = preprocessor.preprocess_batch([first, second])
    assert batch["state_vector"].shape == (2, 9)
    assert batch["player_state_matrix"].shape == (2, 5, 8)
    assert batch["event_history"].shape == (2, 4, 64, 17)
    assert batch["event_valid_mask"].dtype == np.bool_
    assert batch["action_mask"].dtype == np.uint8

    state = EnvironmentState.initial("preprocessor", 7)
    environment = BangEnvironment(state=state, max_episode_steps=None)
    raw = preprocessor.preprocess(environment.observe(agent=1))
    assert raw.metadata["viewer_idx"] == 1
    assert raw.event_history.shape == (4, 64, 17)


def test_config_and_processed_observation_serialization_are_deterministic() -> None:
    config = ObservationPreprocessorConfig()
    assert config.to_json() == ObservationPreprocessorConfig.from_json(config.to_json()).to_json()

    processed = ObservationPreprocessor(config).preprocess(_observation(events=(_event(1),)))
    assert processed.to_json() == processed.to_json()
    assert processed.metadata["preprocessing_version"] == "1.0"
    assert processed.metadata["model_history_length"] == 64

    with pytest.raises(ValueError, match="fixed at 64"):
        ObservationPreprocessorConfig(model_history_length=32)
