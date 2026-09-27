from __future__ import annotations

from pettingzoo.test import api_test

from bang_rl.env import BangEnvironment, EnvironmentState, Role


ROLES = (Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR)


class SilentLogger:
    def event(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def make_state(**overrides) -> EnvironmentState:
    values = {
        "episode_id": "aec-episode",
        "seed": 23,
        "roles": ROLES,
        "health": (5, 4, 4, 4, 4),
        "alive": (True, True, True, True, True),
        "agent_selection": 0,
    }
    values.update(overrides)
    return EnvironmentState(**values)


def aec_step(environment: BangEnvironment, action) -> None:
    iterator = environment.agent_iter(1)
    next(iterator)
    assert environment.step(action) is None


def test_pettingzoo_aec_api_and_space_contract() -> None:
    environment = BangEnvironment(
        max_episode_steps=8,
        warn_turn_threshold=2,
        critical_turn_threshold=4,
        logger=SilentLogger(),
        event_logger=SilentLogger(),
        termination_logger=SilentLogger(),
    )
    api_test(environment, num_cycles=1)


def test_aec_uses_canonical_agent_order_and_dead_agent_cleanup() -> None:
    environment = BangEnvironment(
        state=make_state(health=(5, 1, 4, 4, 4), alive=(True, True, True, True, True)),
        max_episode_steps=None,
    )

    assert environment.possible_agents == ["player_0", "player_1", "player_2", "player_3", "player_4"]
    assert environment.agents == environment.possible_agents
    assert environment.agent_selection == "player_0"

    aec_step(environment, 2)
    assert environment.environment_step == 1
    assert environment.agent_selection == "player_1"
    assert environment.terminations["player_1"] is True
    assert "player_1" in environment.agents

    aec_step(environment, None)
    assert environment.environment_step == 1
    assert "player_1" not in environment.agents
    assert environment.agent_selection == "player_2"


def test_terminal_cleanup_removes_every_agent_without_extra_environment_steps() -> None:
    environment = BangEnvironment(max_episode_steps=3, warn_turn_threshold=1, critical_turn_threshold=2)
    environment.reset(seed=91)

    aec_step(environment, 0)
    aec_step(environment, 0)
    aec_step(environment, 0)
    assert environment.state.truncated is True
    assert environment.environment_step == 3

    while environment.agents:
        selected = environment.agent_selection
        assert selected in environment.agents
        assert environment.truncations[selected] is True
        aec_step(environment, None)

    assert environment.agents == []
    assert environment.environment_step == 3
    assert environment.cleanup_calls == 5
