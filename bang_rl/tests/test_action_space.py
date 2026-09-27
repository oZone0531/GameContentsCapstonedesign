"""Focused tests for TASK-1-2 action-space and mask contracts."""

from itertools import product

import gymnasium.spaces as gym_spaces
import numpy as np
import pytest

from bang_rl.env import (
    ACTION_MASK_SHAPE,
    ACTION_MASK_SPACE,
    ACTION_SPACE,
    Action,
    ActionMask,
    EnvironmentState,
    Role,
    TerminalResult,
    action_mask_for_state,
    action_space_spec,
    decode_action,
    deserialize_action_mask,
    encode_action,
    legal_action_mask,
    make_action_observation_space,
    select_deterministic_action,
    serialize_action_mask,
    valid_action_indices,
)


def test_all_agents_share_the_fixed_gymnasium_action_contract() -> None:
    assert isinstance(ACTION_SPACE, gym_spaces.Discrete)
    assert ACTION_SPACE.n == 6
    assert ACTION_SPACE.start == 0
    assert ACTION_MASK_SPACE.shape == ACTION_MASK_SHAPE
    assert ACTION_MASK_SPACE.dtype == np.uint8

    observation_space = make_action_observation_space()
    assert isinstance(observation_space, gym_spaces.Dict)
    assert observation_space["action_mask"].shape == (6,)
    assert observation_space["action_mask"].dtype == np.uint8
    assert observation_space.contains({"action_mask": np.ones(6, dtype=np.uint8)})


def test_action_mapping_round_trips_pass_and_each_target() -> None:
    assert encode_action(Action.pass_action(2)) == 0
    assert decode_action(0, actor_idx=2) == Action.pass_action(2)

    for target_idx in range(5):
        action = Action.bang(actor_idx=3, target_idx=target_idx)
        assert encode_action(action) == target_idx + 1
        assert decode_action(target_idx + 1, actor_idx=3) == action

    with pytest.raises(ValueError):
        decode_action(6, actor_idx=0)
    with pytest.raises(TypeError):
        decode_action(True, actor_idx=0)


def test_mask_matches_every_running_alive_state_and_actor() -> None:
    for alive_values in product((False, True), repeat=5):
        for actor_idx in range(5):
            expected = np.zeros(6, dtype=np.uint8)
            if alive_values[actor_idx]:
                expected[0] = 1
                for target_idx, is_alive in enumerate(alive_values):
                    if target_idx != actor_idx and is_alive:
                        expected[target_idx + 1] = 1
            actual = legal_action_mask(alive_values, actor_idx)
            assert actual.dtype == np.uint8
            assert actual.shape == (6,)
            assert set(actual.tolist()) <= {0, 1}
            np.testing.assert_array_equal(actual, expected)


def test_terminal_and_dead_actor_masks_authorize_no_action() -> None:
    alive = (True, True, False, True, False)
    np.testing.assert_array_equal(
        legal_action_mask(alive, acting_player_idx=0, terminal=True),
        np.zeros(6, dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        legal_action_mask(alive, acting_player_idx=2),
        np.zeros(6, dtype=np.uint8),
    )

    state = EnvironmentState.initial("terminal-mask", 9)
    terminal_state = EnvironmentState(
        episode_id=state.episode_id,
        seed=state.seed,
        roles=state.roles,
        health=state.health,
        alive=state.alive,
        terminated=True,
        terminal_result=TerminalResult(
            terminated=True,
            winners=(state.sheriff_idx,),
            losers=tuple(index for index in range(5) if index != state.sheriff_idx),
            winning_team=0,
            reason="test",
        ),
        termination_status="normal",
    )
    np.testing.assert_array_equal(action_mask_for_state(terminal_state), np.zeros(6, dtype=np.uint8))


def test_action_mask_serialization_and_validation_are_round_trip_safe() -> None:
    mask = legal_action_mask((True, False, True, False, True), acting_player_idx=0)
    serialized = serialize_action_mask(mask)
    assert serialized == {
        "shape": [6],
        "dtype": "uint8",
        "mask": [1, 0, 0, 1, 0, 1],
    }
    restored = deserialize_action_mask(serialized)
    assert restored.dtype == np.uint8
    np.testing.assert_array_equal(restored, mask)
    assert ActionMask.from_dict(serialized).to_array().dtype == np.uint8

    with pytest.raises(ValueError):
        deserialize_action_mask({"shape": [5], "dtype": "uint8", "mask": [1, 0, 0, 0, 0, 1]})
    with pytest.raises(ValueError):
        deserialize_action_mask({"shape": [6], "dtype": "uint8", "mask": [1, 2, 0, 0, 0, 1]})


def test_deterministic_selection_never_chooses_masked_action() -> None:
    mask = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    assert valid_action_indices(mask) == (0, 2, 5)
    assert select_deterministic_action([0.9, 10.0, 0.9, 20.0, 30.0, 0.1], mask) == 0
    assert select_deterministic_action([0.0, 10.0, 1.0, 20.0, 30.0, 1.0], mask) == 2
    assert select_deterministic_action([0.0, 10.0, 0.0, 20.0, 30.0, 0.0], mask) == 0

    with pytest.raises(ValueError):
        select_deterministic_action(np.zeros(6), np.zeros(6, dtype=np.uint8))


def test_action_space_spec_is_json_safe_and_mapping_is_explicit() -> None:
    spec = action_space_spec()
    assert spec["version"] == "discrete-6-v1"
    assert spec["n"] == 6
    assert spec["mapping"]["0"] == "PASS"
    assert spec["mapping"]["5"] == "PLAYER_4_BANG"
    assert spec["mask"] == {"shape": [6], "dtype": "uint8", "values": [0, 1]}
