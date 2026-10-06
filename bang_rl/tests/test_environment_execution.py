"""Focused execution tests for TASK-1-4 immutable BANG/PASS transitions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bang_rl.env import (
    ActionResult,
    BangEnvironment,
    EnvironmentState,
    EventType,
    Role,
    TerminationStatus,
    apply_action,
)
from bang_rl.exceptions import InvalidActionError, TerminalStateActionError


ROLES = (Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR)


def make_state(**overrides) -> EnvironmentState:
    values = {
        "episode_id": "execution-episode",
        "seed": 7,
        "roles": ROLES,
        "health": (5, 4, 4, 4, 4),
        "alive": (True, True, True, True, True),
        "agent_selection": 0,
    }
    values.update(overrides)
    return EnvironmentState(**values)


def test_pass_is_one_step_with_targetless_event_and_detached_result() -> None:
    environment = BangEnvironment(state=make_state(), max_episode_steps=None)
    before = environment.state

    result = environment.step(0)

    assert isinstance(result, ActionResult)
    assert result is not None
    assert result.action.actor_idx == 0
    assert result.action.target_idx is None
    assert result.environment_step == 1
    assert result.hp_before is None
    assert result.hp_after is None
    assert result.events == (
        result.events[0],
    )
    event = result.events[0]
    assert event.event_type is EventType.PASS
    assert event.actor_idx == 0
    assert event.target_idx is None
    assert event.environment_step == 1
    assert environment.state.environment_step == 1
    assert environment.state.agent_selection == 1
    assert environment.state.round_index == 0
    assert environment.state.health == before.health
    assert environment.state.event_log == result.events
    assert before.environment_step == 0
    assert before.event_log == ()
    assert result.observation.current_player_idx == 1
    assert result.reward == 0.0
    assert result.terminated is False
    assert result.truncated is False


def test_bang_hit_decrements_only_target_and_records_hp_delta() -> None:
    environment = BangEnvironment(state=make_state(), max_episode_steps=None)

    result = environment.step(2)  # player 0 -> player 1

    assert result.action.target_idx == 1
    assert result.hp_before == 4
    assert result.hp_after == 3
    assert result.eliminated_idx is None
    assert result.events[0].event_type is EventType.BANG
    assert result.events[0].target_idx == 1
    assert result.events[0].hp_before == 4
    assert result.events[0].hp_after == 3
    assert result.events[0].eliminated_idx is None
    assert environment.state.health == (5, 3, 4, 4, 4)
    assert environment.state.alive == (True, True, True, True, True)
    assert environment.state.agent_selection == 1
    assert environment.environment_step == 1


def test_bang_kill_marks_target_dead_and_advances_over_it() -> None:
    state = make_state(health=(5, 1, 4, 4, 4), alive=(True, True, True, True, True))
    environment = BangEnvironment(state=state, max_episode_steps=None)

    result = environment.step(2)  # player 0 -> player 1

    assert environment.state.health == (5, 0, 4, 4, 4)
    assert environment.state.alive == (True, False, True, True, True)
    assert result.eliminated_idx == 1
    assert [event.event_type for event in result.events] == [
        EventType.BANG,
        EventType.PLAYER_ELIMINATED,
        EventType.ROLE_REVEALED,
    ]
    elimination = result.events[1]
    assert elimination.eliminated_idx == 1
    assert elimination.target_idx == 1
    assert elimination.hp_before == 0
    assert elimination.hp_after == 0
    assert environment.state.agent_selection == 2
    assert 1 not in environment.agents


def test_turn_rotation_uses_fixed_order_and_increments_round_on_wrap() -> None:
    environment = BangEnvironment(state=make_state(), max_episode_steps=None)

    for expected_next in (1, 2, 3, 4):
        result = environment.step(0)
        assert result.info["next_agent_selection"] == expected_next
        assert environment.round_index == 0
    result = environment.step(0)

    assert result.info["next_agent_selection"] == 0
    assert environment.agent_selection == 0
    assert environment.environment_step == 5
    assert environment.round_index == 1


def test_dead_slots_are_skipped_without_changing_fixed_order() -> None:
    state = make_state(
        health=(5, 0, 4, 4, 4),
        alive=(True, False, True, True, True),
        agent_selection=0,
    )
    environment = BangEnvironment(state=state, max_episode_steps=None)

    environment.step(0)

    assert environment.agent_selection == 2
    assert environment.state.fixed_turn_order == (0, 1, 2, 3, 4)


def test_invalid_action_and_cleanup_do_not_count_as_environment_steps() -> None:
    environment = BangEnvironment(state=make_state(), max_episode_steps=None)
    before = environment.state

    with pytest.raises(InvalidActionError):
        environment.step(1)  # self-target
    assert environment.state == before
    assert environment.environment_step == 0
    assert environment.last_result is None

    with pytest.raises(InvalidActionError):
        environment.step(None)
    assert environment.environment_step == 0
    assert environment.cleanup_calls == 0

    result = environment.step(0)
    assert result.environment_step == 1
    assert environment.cleanup_calls == 0


def test_pure_transition_does_not_mutate_input_state() -> None:
    state = make_state()

    next_state, result = apply_action(state, 0, max_episode_steps=None)

    assert state == make_state()
    assert next_state != state
    assert next_state.event_log == result.events
    assert result.action.target_idx is None


def test_truncated_boundary_has_no_winner_and_rejects_following_action() -> None:
    environment = BangEnvironment(state=make_state(), max_episode_steps=3, warn_turn_threshold=1, critical_turn_threshold=2)

    result = environment.step(0)
    environment.step(0)
    result = environment.step(0)

    assert result.truncated is True
    assert result.terminated is False
    assert environment.state.termination_status is TerminationStatus.TRUNCATED
    assert environment.state.terminal_result is None
    assert environment.terminations == {index: False for index in range(5)}
    assert all(value is True for value in environment.truncations.values())
    with pytest.raises(TerminalStateActionError):
        environment.step(0)
    assert environment.environment_step == 3


def test_terminal_cleanup_is_not_a_game_action() -> None:
    from bang_rl.env import TerminalResult

    terminal = make_state(
        terminated=True,
        terminal_result=TerminalResult(
            terminated=True,
            winners=(0,),
            losers=(1, 2, 3, 4),
            winning_team=0,
            reason="fixture",
        ),
        termination_status=TerminationStatus.NORMAL,
    )
    environment = BangEnvironment(state=terminal, max_episode_steps=None)

    assert environment.step(None) is None
    assert environment.cleanup_calls == 1
    assert environment.environment_step == 0


def test_sheriff_team_wins_before_truncation_and_shares_result_with_deputy() -> None:
    state = make_state(
        health=(5, 0, 1, 0, 0),
        alive=(True, False, True, False, False),
        agent_selection=0,
    )
    environment = BangEnvironment(state=state, max_episode_steps=3, warn_turn_threshold=1, critical_turn_threshold=2)

    result = environment.step(3)  # sheriff eliminates the last outlaw survivor

    assert result.terminated is True
    assert result.truncated is False
    assert environment.state.termination_status is TerminationStatus.NORMAL
    assert environment.state.terminal_result is not None
    assert environment.state.terminal_result.winning_team.value == 0
    assert environment.state.terminal_result.winners == (0, 1)
    assert environment.state.terminal_result.losers == (2, 3, 4)
    assert result.info["termination_status"] == "normal"
    assert result.info["terminal_result"]["winners"] == [0, 1]


def test_outlaw_team_wins_when_sheriff_dies_with_any_non_traitor_survivor() -> None:
    state = make_state(
        health=(1, 4, 0, 0, 0),
        alive=(True, True, False, False, False),
        agent_selection=1,
    )
    environment = BangEnvironment(state=state, max_episode_steps=None)

    result = environment.step(1)  # deputy eliminates the sheriff

    assert result.terminated is True
    assert result.truncated is False
    assert environment.state.terminal_result is not None
    assert environment.state.terminal_result.winning_team.value == 1
    assert environment.state.terminal_result.winners == (2, 3)
    assert environment.state.terminal_result.losers == (0, 1, 4)


def test_traitor_wins_when_it_is_the_only_survivor_after_sheriff_elimination() -> None:
    state = make_state(
        health=(1, 0, 0, 0, 4),
        alive=(True, False, False, False, True),
        agent_selection=4,
    )
    environment = BangEnvironment(state=state, max_episode_steps=None)

    result = environment.step(1)  # traitor eliminates the sheriff

    assert result.terminated is True
    assert environment.state.terminal_result is not None
    assert environment.state.terminal_result.winning_team.value == 2
    assert environment.state.terminal_result.winners == (4,)
    assert environment.state.terminal_result.losers == (0, 1, 2, 3)


def test_sheriff_traitor_duel_continues_without_a_forced_terminal_state() -> None:
    state = make_state(
        health=(5, 0, 0, 0, 4),
        alive=(True, False, False, False, True),
        agent_selection=0,
    )
    environment = BangEnvironment(state=state, max_episode_steps=None)

    result = environment.step(0)

    assert result.terminated is False
    assert result.truncated is False
    assert environment.state.termination_status is TerminationStatus.RUNNING
    assert environment.state.terminal_result is None
    assert environment.agent_selection == 4


def test_repeated_pass_visits_seats_clockwise_from_the_sheriff() -> None:
    # Item 2: starting from EnvironmentState.initial (Sheriff seat) and passing
    # every turn must visit seats in ascending (clockwise) order, wrapping back
    # to the Sheriff after a full lap.
    state = EnvironmentState.initial("clockwise", 42)
    sheriff_seat = state.roles.index(Role.SHERIFF)
    assert state.agent_selection == sheriff_seat
    environment = BangEnvironment(state=state, max_episode_steps=None)
    visited = [environment.state.agent_selection]
    for _ in range(5):
        environment.step(0)  # PASS
        visited.append(environment.state.agent_selection)
    expected = [(sheriff_seat + offset) % 5 for offset in range(6)]
    assert visited == expected
