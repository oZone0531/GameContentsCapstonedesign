from __future__ import annotations

from io import StringIO
import json
import threading
import time

import numpy as np
import pytest

from bang_rl.agents import DummyAgent
from bang_rl.config import HardwareConfig, SystemConfig
from bang_rl.env import BangEnvironment, EnvironmentState, Role
from bang_rl.runtime.logging import LogChannel, StructuredLogger
from bang_rl.runtime.seed import SeedManager
from bang_rl.training import (
    DummyVecEnv,
    ResourceGuard,
    WorkerFault,
    WorkerRuntime,
)
from bang_rl.runtime.pause_coordinator import WorkerCancelled


ROLES = (Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR)


def short_env(*, worker_id=None, seed=0, episode_id="episode", **_kwargs):
    del worker_id
    return BangEnvironment(max_episode_steps=3, warn_turn_threshold=1, critical_turn_threshold=2, seed=seed, episode_id=episode_id)


class TerminalEnvironment(BangEnvironment):
    def __init__(self, *, seed=0, episode_id="terminal", **kwargs):
        state = EnvironmentState(
            episode_id=episode_id,
            seed=seed,
            roles=ROLES,
            health=(5, 1, 0, 0, 0),
            alive=(True, True, False, False, False),
            agent_selection=0,
        )
        self._preset_state = state
        super().__init__(state=state, max_episode_steps=3, warn_turn_threshold=1, critical_turn_threshold=2, **kwargs)

    def reset(self, seed=None, options=None, *, episode_id="terminal"):
        state = self._preset_state
        if seed is not None or options and "episode_id" in options or episode_id != "terminal":
            resolved_id = str(options.get("episode_id", episode_id)) if options else episode_id
            state = EnvironmentState(
                episode_id=resolved_id,
                seed=state.seed if seed is None else int(seed),
                roles=state.roles,
                health=state.health,
                alive=state.alive,
                agent_selection=state.agent_selection,
            )
        self.load_state(state)
        return self.observe()


def terminal_env(*, worker_id=None, seed=0, episode_id="terminal", **_kwargs):
    del worker_id
    return TerminalEnvironment(seed=seed, episode_id=episode_id)


def test_vector_shape_fixed_order_and_user_resource_metadata() -> None:
    config = SystemConfig(
        hardware=HardwareConfig(cpu_cores=2, ram_gb=2, workers=2, batch_size=8, buffer_size=32),
    )
    vector = DummyVecEnv([short_env, short_env], config=config, seed=41, job_id="job")
    try:
        observations = vector.reset()
        assert vector.n_envs == vector.num_workers == 2
        assert [observation.current_player_idx for observation in observations] == [0, 0]
        assert vector.metadata["batch_size"] == 8
        assert vector.metadata["buffer_size"] == 32
        assert vector.metadata["hardware"]["workers"] == 2
        assert vector.workers[0].fixed_agent_order == tuple(f"player_{i}" for i in range(5))
        next_observations, rewards, terminated, truncated, infos = vector.step([0, 0])
        assert len(next_observations) == len(rewards) == len(terminated) == len(truncated) == len(infos) == 2
        assert all(info["worker_id"] in (0, 1) for info in infos)
    finally:
        vector.close()


def test_worker_agent_graphs_are_isolated_and_replayable() -> None:
    templates = tuple(DummyAgent(f"player_{index}", seed=101 + index, player_idx=index) for index in range(5))
    first = DummyVecEnv([short_env, short_env], seed=77, agents=templates, job_id="agent-isolation")
    second = DummyVecEnv([short_env, short_env], seed=77, agents=templates, job_id="agent-isolation")
    try:
        for index in range(5):
            left = first.workers[0].agents[index]
            right = first.workers[1].agents[index]
            assert left is not right
            assert left.role_estimator is not right.role_estimator
            assert left.emotion_module is not right.emotion_module
            assert left.rng is not right.rng

        first.workers[0].agents[0].emotion_module.apply_event(
            {"event_id": 999, "event_type": "bang", "actor_idx": 0, "target_idx": 1}
        )
        assert 999 in first.workers[0].agents[0].emotion_module.processed_event_ids
        assert 999 not in first.workers[1].agents[0].emotion_module.processed_event_ids

        first_records = first.run_episodes(2)
        second_records = second.run_episodes(2)
        assert [record.to_dict() for record in first_records] == [record.to_dict() for record in second_records]
    finally:
        first.close()
        second.close()


def test_worker_and_episode_seeds_are_isolated_and_reproducible() -> None:
    first = DummyVecEnv([short_env, short_env], seed=77, job_id="replay")
    second = DummyVecEnv([short_env, short_env], seed=77, job_id="replay")
    try:
        assert first.metadata["worker_seeds"] == second.metadata["worker_seeds"]
        assert first.metadata["worker_seeds"][0] != first.metadata["worker_seeds"][1]
        assert [first.assignment_for_episode(i) for i in range(6)] == [second.assignment_for_episode(i) for i in range(6)]
        first_records = first.run_episodes(2)
        second_records = second.run_episodes(2)
        assert [record.to_dict() for record in first_records] == [record.to_dict() for record in second_records]
        assert {record.worker_id for record in first_records} == {0, 1}
    finally:
        first.close()
        second.close()


def test_normal_termination_has_no_truncated_flag_and_truncated_is_distinct() -> None:
    normal = DummyVecEnv([terminal_env], seed=3)
    truncated = DummyVecEnv([short_env], seed=3)
    try:
        normal_record = normal.run_episodes(1, action_provider=lambda _observation, _mask: 2)[0]
        truncated_record = truncated.run_episodes(1)[0]
        assert normal_record.terminated is True
        assert normal_record.truncated is False
        assert normal_record.termination_status == "normal"
        assert truncated_record.terminated is False
        assert truncated_record.truncated is True
        assert truncated_record.termination_status == "truncated"
        assert truncated_record.terminal_result is None
    finally:
        normal.close()
        truncated.close()


def test_pause_freezes_worker_until_resume_and_cancel_is_cooperative() -> None:
    vector = DummyVecEnv([short_env], seed=9)
    completed: list[object] = []

    def slow_action(observation, mask):
        time.sleep(0.005)
        return 0

    def run() -> None:
        completed.extend(vector.run_episodes(1, action_provider=slow_action))

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 2
    while vector.workers[0].environment is None or vector.workers[0].environment_step < 1:
        if time.monotonic() >= deadline:
            pytest.fail("worker did not start")
        time.sleep(0.005)
    assert vector.pause(timeout=2) is True
    paused_step = vector.workers[0].environment_step
    time.sleep(0.03)
    assert vector.workers[0].environment_step == paused_step
    assert vector.pause_coordinator.state.value == "paused"
    vector.resume()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert completed and completed[0].truncated

    cancelling = DummyVecEnv([short_env], seed=10)
    try:
        thread = threading.Thread(target=lambda: cancelling.run_episodes(1, action_provider=slow_action))
        thread.start()
        time.sleep(0.02)
        cancelling.cancel()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert cancelling.pause_coordinator.state.value == "cancelled"
    finally:
        vector.close()
        cancelling.close()


def test_worker_fault_propagates_with_identity() -> None:
    vector = DummyVecEnv([short_env], seed=5)
    try:
        def fail(_observation, _mask):
            raise RuntimeError("agent exploded")

        with pytest.raises(WorkerFault) as captured:
            vector.run_episodes(1, action_provider=fail)
        assert captured.value.worker_id == 0
        assert captured.value.episode_id == "episode-0"
        assert isinstance(captured.value.cause, RuntimeError)
    finally:
        vector.close()


def test_structured_worker_logs_keep_public_and_privileged_channels_separate() -> None:
    public_stream = StringIO()
    privileged_stream = StringIO()
    public_logger = StructuredLogger(LogChannel.TRAINING, run_id="run", job_id="job", stream=public_stream)
    privileged_logger = StructuredLogger(LogChannel.PRIVILEGED_DIAGNOSTICS, run_id="run", job_id="job", stream=privileged_stream)
    worker = WorkerRuntime(
        0,
        seed_manager=SeedManager(13),
        environment_factory=short_env,
        logger=public_logger,
        privileged_logger=privileged_logger,
        job_id="job",
    )
    try:
        record = worker.run_episode(worker.assignment(0))
        assert record.public
        public_records = [json.loads(line) for line in public_stream.getvalue().splitlines()]
        privileged_records = [json.loads(line) for line in privileged_stream.getvalue().splitlines()]
        assert public_records and privileged_records
        assert all("roles" not in item for item in public_records)
        assert any("roles" in item for item in privileged_records)
        assert all(item["worker_id"] == 0 for item in public_records)
    finally:
        worker.close()
        public_logger.close()
        privileged_logger.close()


def test_resource_guard_rejects_worker_mismatch_without_optimization() -> None:
    guard = ResourceGuard(HardwareConfig(workers=2, batch_size=4, buffer_size=16))
    with pytest.raises(Exception):
        guard.validate_workers(1)
    assert guard.snapshot_metadata()["auto_optimization"] is False
    assert guard.snapshot_metadata()["benchmark_executed"] is False
