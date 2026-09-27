"""Focused tests for TASK-4-2 Reward DSL and transition boundaries."""

from __future__ import annotations

from io import StringIO

import numpy as np

from bang_rl.config import RewardConfig
from bang_rl.env import Action, ActionResult, EnvironmentState, EventType, GameEvent, Role, Team
from bang_rl.runtime import (
    LogChannel,
    RewardBatch,
    RewardBreakdown,
    RewardCalculator,
    RewardRuntime,
    StructuredLogger,
    attach_rewards,
)


ROLES = (Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR)


def state(**overrides) -> EnvironmentState:
    values = {
        "episode_id": "reward-episode",
        "seed": 11,
        "roles": ROLES,
        "health": (5, 4, 4, 4, 4),
        "alive": (True, True, True, True, True),
        "agent_selection": 2,
    }
    values.update(overrides)
    return EnvironmentState(**values)


def bang_result(*, actor: int = 2, target: int = 0, step: int = 1, killed: bool = False, terminated: bool = False, truncated: bool = False) -> ActionResult:
    event = GameEvent(
        event_id=step,
        environment_step=step,
        event_type=EventType.BANG,
        actor_idx=actor,
        target_idx=target,
        hp_before=1 if killed else 4,
        hp_after=0 if killed else 3,
        eliminated_idx=target if killed else None,
    )
    return ActionResult(
        action=Action.bang(actor, target),
        environment_step=step,
        events=(event,),
        terminated=terminated,
        truncated=truncated,
        info={
            "terminal_result": {
                "terminated": True,
                "winners": [0, 1],
                "losers": [2, 3, 4],
                "winning_team": int(Team.SHERIFF_TEAM),
                "reason": "fixture",
            }
            if terminated
            else None
        },
    )


def pass_result(step: int = 1, *, truncated: bool = False) -> ActionResult:
    return ActionResult(
        action=Action.pass_action(0),
        environment_step=step,
        events=(GameEvent(event_id=step, environment_step=step, event_type=EventType.PASS, actor_idx=0),),
        truncated=truncated,
    )


def test_each_namespace_condition_and_multiple_components_are_evaluated() -> None:
    config = RewardConfig(
        version="reward-v4",
        terminal={
            "components": [
                {"id": "terminal_win", "enabled": True, "condition": {"type": "terminal_result", "result": "win"}, "value": 10},
                {"id": "terminal_loss", "enabled": True, "condition": {"type": "terminal_result", "result": "loss"}, "value": -10},
            ]
        },
        shaping={
            "components": [
                {"id": "bang_hit", "enabled": True, "condition": {"type": "event", "event": "bang_hit"}, "value": 1},
                {"id": "kill_enemy", "enabled": True, "condition": {"type": "event", "event": "kill_enemy"}, "value": 2},
                {"id": "attack_sheriff_outlaw", "enabled": True, "condition": {"type": "role_attack", "actor_role": "outlaw_a", "target_role": "sheriff"}, "value": 3},
                {"id": "role_or_faction_attack_terms", "enabled": True, "condition": {"type": "role_attack", "actor_role": "outlaw_a", "target_role": "sheriff"}, "value": 4},
            ]
        },
    )
    # IDs are unique within a namespace, so use a second config for the
    # faction condition while keeping the condition-specific assertions small.
    calculator = RewardCalculator(config)
    result = bang_result(actor=2, target=0, killed=True, terminated=True)
    breakdown = calculator.calculate_for_agent(result, 2, state=state(), transition_id="t-1")

    assert breakdown.terminal_reward == -10
    assert breakdown.terminal_components == {"terminal_win": 0.0, "terminal_loss": -10.0}
    assert breakdown.shaping["bang_hit"] == 1
    assert breakdown.shaping["kill_enemy"] == 2
    assert breakdown.shaping["attack_sheriff_outlaw"] == 3
    assert breakdown.shaping["role_or_faction_attack_terms"] == 4
    assert breakdown.total == 0
    assert breakdown.main_agent_transition_id == "t-1"

    faction_config = config.model_copy(deep=True)
    faction_config.shaping.components = [
        {"id": "role_or_faction_attack_terms", "enabled": True, "condition": {"type": "faction_attack", "actor_faction": "outlaw_team", "target_faction": "sheriff_team"}, "value": 5}
    ]
    faction = RewardCalculator(faction_config).calculate_for_agent(result, 2, roles=ROLES)
    assert faction.shaping_reward == 5


def test_pass_condition_and_disabled_or_missing_rules_are_safe() -> None:
    config = RewardConfig(
        version="reward-empty-v1",
        shaping={
            "components": [
                {"id": "pass_penalty", "enabled": True, "condition": {"type": "pass"}, "value": -0.25},
                {"id": "bang_hit", "enabled": False, "condition": {"type": "event", "event": "bang_hit"}, "value": 99},
            ]
        },
    )
    calculator = RewardCalculator(config)
    result = calculator.calculate(pass_result())
    assert isinstance(result, RewardBatch)
    assert result[0].shaping["pass_penalty"] == -0.25
    assert result[0].shaping["bang_hit"] == 0
    assert all(value.total == (-0.25 if index == 0 else 0) for index, value in result.rewards.items())

    empty = RewardCalculator(RewardConfig(version="no-rules")).calculate(bang_result())
    assert all(value.total == 0 for value in empty.rewards.values())


def test_terminal_and_truncated_precedence_is_per_agent() -> None:
    config = RewardConfig(
        version="terminal-v1",
        terminal={
            "components": [
                {"id": "terminal_win", "enabled": True, "condition": {"type": "terminal_result", "result": "win"}, "value": 7},
                {"id": "terminal_loss", "enabled": True, "condition": {"type": "terminal_result", "result": "loss"}, "value": -8},
            ]
        },
        shaping={
            "components": [
                {"id": "bang_hit", "enabled": True, "condition": {"type": "event", "event": "bang_hit"}, "value": 0.5},
            ]
        },
    )
    calculator = RewardCalculator(config)
    normal = calculator.calculate_all(bang_result(terminated=True), state=state())
    assert normal[0].terminal_reward == 7
    assert normal[1].terminal_reward == 7
    assert normal[2].terminal_reward == -8
    assert normal[2].shaping_reward == 0.5

    truncated = calculator.calculate_all(bang_result(truncated=True), state=state())
    assert all(value.terminal_reward == 0 for value in truncated.rewards.values())
    assert truncated[2].total == 0.5


def test_clipping_is_explicit_and_does_not_change_breakdown_components() -> None:
    config = RewardConfig(
        version="clip-v1",
        clipping={"enabled": True, "min": -1, "max": 1},
        shaping={
            "components": [
                {"id": "bang_hit", "enabled": True, "condition": {"type": "event", "event": "bang_hit"}, "value": 3},
            ]
        },
    )
    breakdown = RewardCalculator(config).calculate_for_agent(bang_result(), 2)
    assert breakdown.unclipped_total == 3
    assert breakdown.total == 1
    assert breakdown.clipped is True


def test_logging_privacy_deterministic_replay_and_serialization() -> None:
    stream = StringIO()
    logger = StructuredLogger(LogChannel.REWARD_BREAKDOWN, stream=stream)
    config = RewardConfig(
        version="persist-v1",
        shaping={
            "components": [
                {"id": "bang_hit", "enabled": True, "condition": {"type": "event", "event": "bang_hit"}, "value": 0.75},
            ]
        },
    )
    calculator = RewardCalculator(config, logger=logger)
    result = bang_result()
    first = calculator.calculate_all(result, roles=ROLES, main_agent_transition_id=4)
    replay = calculator.calculate_all(result, roles=ROLES, main_agent_transition_id=4)
    assert first.to_dict() == replay.to_dict()
    assert RewardCalculator.from_json(calculator.to_json()).to_json() == calculator.to_json()
    logged = stream.getvalue()
    assert "persist-v1" in logged
    assert "reward_breakdown" in logged
    assert '"roles"' not in logged
    assert '"true_roles"' not in logged
    logger.close()


def test_reward_runtime_invalid_boundary_deduplicates_and_persists() -> None:
    calculator = RewardCalculator(RewardConfig(version="runtime-v1"))
    runtime = RewardRuntime(calculator)
    runtime.reset({"episode_id": "episode-r", "seed": 42})
    result = pass_result()
    assert runtime.on_invalid_action(result) is None
    batch = runtime.on_action_result(result, roles=ROLES)
    assert batch is not None
    assert runtime.on_action_result(result, roles=ROLES) is batch
    restored = RewardRuntime.from_json(runtime.to_json())
    assert restored.records[0].to_dict() == batch.to_dict()

    attached = attach_rewards(result, batch)
    assert attached.reward == 0
    assert attached.info["rewards"][0] == 0
    assert "reward_breakdown" in attached.info
    assert np.isfinite(attached.reward)
    assert isinstance(batch[0], RewardBreakdown)


def test_deputy_role_term_and_elimination_event_variant() -> None:
    config = RewardConfig(
        version="condition-v1",
        shaping={
            "components": [
                {"id": "kill_enemy", "enabled": True, "condition": {"type": "event", "event": "kill_enemy"}, "value": 2},
                {"id": "attack_sheriff_deputy", "enabled": True, "condition": {"type": "role_attack", "actor_role": "deputy", "target_role": "sheriff"}, "value": -3},
            ]
        },
    )
    event = GameEvent(
        event_id=1,
        environment_step=1,
        event_type=EventType.PLAYER_ELIMINATED,
        actor_idx=1,
        target_idx=0,
        eliminated_idx=0,
    )
    result = ActionResult(action=Action.bang(1, 0), environment_step=1, events=(event,))
    breakdown = RewardCalculator(config).calculate_for_agent(result, 1, roles=ROLES)
    assert breakdown.shaping["kill_enemy"] == 2
    assert breakdown.shaping["attack_sheriff_deputy"] == -3
    assert breakdown.total == -1


def test_yaml_dsl_wrapper_is_loadable() -> None:
    calculator = RewardCalculator.from_yaml(
        "reward_config:\n"
        "  version: yaml-v1\n"
        "  shaping:\n"
        "    components:\n"
        "      - id: pass_penalty\n"
        "        enabled: true\n"
        "        condition: {type: pass}\n"
        "        value: -0.1\n"
    )
    assert calculator.config_version == "yaml-v1"
    assert calculator.calculate_for_agent(pass_result(), 0).total == -0.1
