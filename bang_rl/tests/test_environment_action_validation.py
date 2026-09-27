"""Runtime validation tests for TASK-1-3."""

from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json

import numpy as np
import pytest

from bang_rl.env import (
    ActionValidationBoundary,
    BangEnvironment,
    EnvironmentState,
    Role,
    TerminationStatus,
    TerminalResult,
)
from bang_rl.exceptions import AgentOrderError, InvalidActionError, TerminalStateActionError
from bang_rl.runtime.logging import LogChannel, StructuredLogger


ROLES = (Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR)


def make_state(**overrides) -> EnvironmentState:
    values = {
        "episode_id": "validation-episode",
        "seed": 17,
        "roles": ROLES,
        "health": (5, 4, 4, 4, 4),
        "alive": (True, True, True, True, True),
        "agent_selection": 0,
    }
    values.update(overrides)
    return EnvironmentState(**values)


def test_invalid_scalar_type_and_range_raise_shared_error_with_context() -> None:
    environment = BangEnvironment(state=make_state())
    before = environment.state

    for action, expected_reason in ((True, "action_type"), ("0", "action_type"), (6, "action_range"), (-1, "action_range")):
        with pytest.raises(InvalidActionError) as captured:
            environment.step(action)
        error = captured.value
        assert error.reason == expected_reason
        assert error.episode_id == before.episode_id
        assert error.environment_step == before.environment_step
        assert error.agent_index == before.agent_selection
        assert error.action_mask == [1, 0, 1, 1, 1, 1]
        assert error.state_summary["alive"] == [True, True, True, True, True]
        assert error.context["agent_selection"] == 0
        assert error.context["termination_status"] == "running"
        assert environment.state == before


def test_order_terminal_and_truncated_boundaries_use_distinct_aec_errors() -> None:
    environment = BangEnvironment(state=make_state())

    with pytest.raises(AgentOrderError) as captured:
        environment.step(0, agent_index=1)
    assert captured.value.expected_agent == 0
    assert captured.value.actual_agent == 1
    assert captured.value.context["environment_step"] == 0
    assert environment.state == make_state()

    terminal = make_state(
        terminated=True,
        terminal_result=TerminalResult(
            terminated=True,
            winners=(0, 1),
            losers=(2, 3, 4),
            winning_team=0,
            reason="test terminal",
        ),
        termination_status=TerminationStatus.NORMAL,
    )
    environment.load_state(terminal)
    with pytest.raises(TerminalStateActionError) as captured:
        environment.step(0)
    assert captured.value.context["termination_status"] == "normal"
    assert captured.value.context["terminated"] is True
    assert environment.state == terminal

    truncated = make_state(truncated=True, termination_status=TerminationStatus.TRUNCATED)
    environment.load_state(truncated)
    with pytest.raises(TerminalStateActionError) as captured:
        environment.step(0)
    assert captured.value.context["termination_status"] == "truncated"
    assert captured.value.context["truncated"] is True
    assert environment.state == truncated


def test_mask_self_and_dead_target_rejections_are_immutable() -> None:
    environment = BangEnvironment(state=make_state())
    before = environment.state

    with pytest.raises(InvalidActionError) as captured:
        environment.step(1)  # player 0 targets itself
    assert captured.value.reason == "self_target"
    assert captured.value.context["target_index"] == 0
    assert environment.state == before

    dead = make_state(
        health=(5, 4, 0, 4, 4),
        alive=(True, True, False, True, True),
    )
    environment.load_state(dead)
    with pytest.raises(InvalidActionError) as captured:
        environment.step(3)  # player 0 targets dead player 2
    assert captured.value.reason == "dead_target"
    assert captured.value.context["target_index"] == 2
    assert environment.state == dead

    with pytest.raises(InvalidActionError) as captured:
        environment.step(2, agent_index=0, action_mask=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8))
    assert captured.value.reason == "mask_disallowed"
    assert captured.value.action_mask == [1, 0, 0, 0, 0, 0]
    assert environment.state == dead


def test_invalid_action_is_logged_on_separate_channel_with_state_context() -> None:
    stream = StringIO()
    logger = StructuredLogger(
        LogChannel.INVALID_ACTION,
        run_id="run-test",
        job_id="job-test",
        episode_id="validation-episode",
        component="environment",
        stream=stream,
    )
    try:
        environment = BangEnvironment(state=make_state(), logger=logger)
        with pytest.raises(InvalidActionError):
            environment.step(1)
        record = json.loads(stream.getvalue().strip())
    finally:
        logger.close()

    assert record["channel"] == "invalid_action"
    assert record["episode_id"] == "validation-episode"
    assert record["step"] == 0
    assert record["reason"] == "self_target"
    assert record["action"] == 1
    assert record["action_mask"] == [1, 0, 1, 1, 1, 1]
    assert record["state_summary"]["alive"] == [True, True, True, True, True]


def test_cleanup_none_is_separate_and_valid_actions_reach_boundary() -> None:
    environment = BangEnvironment(state=make_state())
    initial = environment.state

    # Validation remains a pure boundary; execution is covered by the
    # TASK-1-4 tests and is intentionally not conflated with validation.
    for action in (0, 2, 3, 4, 5):
        assert environment.validate_action(action) == action
        assert environment.state == initial
        assert environment.environment_step == 0
        assert environment.agent_selection == 0

    with pytest.raises(InvalidActionError) as captured:
        environment.step(None)
    assert captured.value.context["cleanup"] is True
    assert environment.cleanup_calls == 0

    dead = make_state(
        health=(5, 0, 4, 4, 4),
        alive=(True, False, True, True, True),
        agent_selection=1,
    )
    environment.load_state(dead)
    assert environment.step(None) is None
    assert environment.cleanup_calls == 1
    assert environment.state == dead


def test_boundary_direct_api_uses_current_state_mask_and_returns_normalized_numpy_int() -> None:
    state = make_state()
    boundary = ActionValidationBoundary(state)
    assert boundary.validate(np.int64(0)) == 0
    assert boundary.validate(np.int64(2)) == 2
    np.testing.assert_array_equal(boundary.current_action_mask(), np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8))
