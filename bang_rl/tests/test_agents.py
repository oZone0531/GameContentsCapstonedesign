"""Focused contract tests for the base, human, and random agents."""

from __future__ import annotations

import inspect
import threading
import time

import numpy as np
import pytest

from bang_rl.agents import (
    AgentInterruptedError,
    HumanAgent,
    RandomAgent,
    validate_action,
    validate_action_mask,
)
from bang_rl.env import BangEnvironment


def observation(mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    return {"action_mask": np.asarray(mask if mask is not None else [1, 1, 0, 1, 0, 0], dtype=np.uint8)}


def test_random_agent_is_reproducible_and_uses_only_enabled_actions() -> None:
    mask = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    first = RandomAgent(agent_id="player_2", seed=123)
    second = RandomAgent(agent_id="player_2", seed=123)

    first_actions = [first.act(observation(mask), mask) for _ in range(40)]
    second_actions = [second.act(observation(mask), mask) for _ in range(40)]

    assert first_actions == second_actions
    assert set(first_actions).issubset({0, 2, 5})
    assert all(mask[action] == 1 for action in first_actions)


def test_masks_and_selected_actions_are_validated_without_pass_fallback() -> None:
    with pytest.raises(ValueError):
        validate_action_mask([1, 0, 0])
    with pytest.raises(TypeError):
        validate_action_mask([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="disallowed"):
        validate_action(1, [1, 0, 1, 0, 0, 0])
    with pytest.raises(TypeError):
        validate_action(True, [1, 1, 0, 0, 0, 0])

    agent = RandomAgent(seed=7)
    with pytest.raises(ValueError, match="does not match"):
        agent.act(observation([1, 0, 0, 0, 0, 0]), [1, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="privileged"):
        agent.act({"roles": [0, 1, 2, 3, 4], "action_mask": [1, 1, 0, 0, 0, 0]}, [1, 1, 0, 0, 0, 0])


def test_human_agent_queue_pause_resume_and_interrupt() -> None:
    agent = HumanAgent(agent_id="player_0", poll_interval=0.005)
    mask = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    agent.submit_action(1)
    agent.pause()
    result: list[object] = []

    def paused_action() -> None:
        try:
            result.append(agent.act(observation(mask), mask, timeout=1.0))
        except BaseException as exc:  # capture the worker result for the assertion below
            result.append(exc)

    worker = threading.Thread(target=paused_action)
    worker.start()
    time.sleep(0.03)
    assert worker.is_alive()
    assert agent.input_queue.qsize() == 1
    agent.resume()
    worker.join(timeout=1.0)
    assert result == [1]

    interrupted: list[object] = []

    def waiting_action() -> None:
        try:
            interrupted.append(agent.act(observation(mask), mask))
        except BaseException as exc:
            interrupted.append(exc)

    worker = threading.Thread(target=waiting_action)
    worker.start()
    time.sleep(0.03)
    agent.interrupt()
    worker.join(timeout=1.0)
    assert len(interrupted) == 1
    assert isinstance(interrupted[0], AgentInterruptedError)


def test_agent_lifecycle_resets_random_stream_and_records_result() -> None:
    context = {"episode_id": "episode-9", "seed": 321}
    mask = np.asarray([1, 1, 0, 1, 0, 0], dtype=np.uint8)
    first = RandomAgent(agent_id="player_1", seed=10)
    second = RandomAgent(agent_id="player_1", seed=10)
    first.on_episode_start(context)
    second.on_episode_start(context)
    assert [first.act(observation(mask), mask) for _ in range(12)] == [
        second.act(observation(mask), mask) for _ in range(12)
    ]
    result = {"episode_id": "episode-9", "terminated": True}
    first.on_episode_end(result)
    assert first.episode_id == "episode-9"
    assert first.episode_seed == 321
    assert first._last_episode_result == result


def test_agents_drive_the_aec_environment_through_public_observation_only() -> None:
    env = BangEnvironment(episode_id="agent-aec", seed=42, max_episode_steps=20, warn_turn_threshold=5, critical_turn_threshold=10)
    env.reset()
    agents = {
        name: RandomAgent(agent_id=name, seed=99)
        for name in env.possible_agents
    }
    for selected in env.agent_iter(max_iter=4):
        public_observation = env.observe(selected)
        # infos carries a plain list (see BangEnvironment.infos); normalise it.
        mask = np.asarray(env.infos[selected]["action_mask"], dtype=np.uint8)
        action = agents[selected].act(public_observation, mask)
        assert mask[action] == 1
        env.step(action)
    assert env.environment_step == 4
    assert all("roles" not in env.observe(name).keys() for name in env.possible_agents)


def test_agent_modules_do_not_import_environment_implementation() -> None:
    source = inspect.getsource(RandomAgent) + inspect.getsource(HumanAgent)
    assert "BangEnvironment" not in source
