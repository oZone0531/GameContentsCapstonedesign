"""Focused integration tests for TASK-1-7 long-game diagnostics."""

from __future__ import annotations

import io
import json

import pytest

from bang_rl.config import EnvConfig
from bang_rl.env import BangEnvironment
from bang_rl.runtime.logging import LogChannel, StructuredLogger


def _records(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_environment_does_not_close_caller_owned_loggers() -> None:
    class CallerLogger:
        channel = LogChannel.PRIVILEGED_DIAGNOSTICS.value
        episode_id = "caller-owned"

        def close(self):
            self.close_count += 1

        def __init__(self):
            self.close_count = 0

    validation = CallerLogger()
    events = CallerLogger()
    termination = CallerLogger()
    diagnostics = CallerLogger()
    environment = BangEnvironment(
        max_episode_steps=6,
        warn_turn_threshold=2,
        critical_turn_threshold=4,
        logger=validation,
        event_logger=events,
        termination_logger=termination,
        privileged_diagnostics_logger=diagnostics,
    )
    environment.close()
    assert [logger.close_count for logger in (validation, events, termination, diagnostics)] == [0, 0, 0, 0]


def test_direct_constructor_rejects_unreachable_default_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        BangEnvironment(max_episode_steps=6)


def test_direct_constructor_valid_short_thresholds_emit_each_crossing_once() -> None:
    stream = io.StringIO()
    diagnostics_logger = StructuredLogger(
        LogChannel.PRIVILEGED_DIAGNOSTICS,
        episode_id="direct-short",
        stream=stream,
    )
    environment = BangEnvironment(
        max_episode_steps=6,
        warn_turn_threshold=2,
        critical_turn_threshold=4,
        privileged_diagnostics_logger=diagnostics_logger,
    )
    try:
        for _ in range(6):
            environment.step(0)
        records = _records(stream)
        assert [(record["level"], record["threshold"]) for record in records] == [
            ("WARNING", 2),
            ("CRITICAL", 4),
        ]
    finally:
        environment.close()
        diagnostics_logger.close()


    event_stream = io.StringIO()
    termination_stream = io.StringIO()
    diagnostics_stream = io.StringIO()
    validation_stream = io.StringIO()
    event_logger = StructuredLogger(
        LogChannel.EPISODE_EVENTS,
        run_id="run-7",
        job_id="job-7",
        episode_id="long-episode",
        stream=event_stream,
    )
    validation_logger = StructuredLogger(
        LogChannel.INVALID_ACTION,
        run_id="run-7",
        job_id="job-7",
        episode_id="long-episode",
        stream=validation_stream,
    )
    termination_logger = StructuredLogger(
        LogChannel.TERMINATION,
        run_id="run-7",
        job_id="job-7",
        episode_id="long-episode",
        stream=termination_stream,
    )
    diagnostics_logger = StructuredLogger(
        LogChannel.PRIVILEGED_DIAGNOSTICS,
        run_id="run-7",
        job_id="job-7",
        episode_id="long-episode",
        stream=diagnostics_stream,
    )
    environment = BangEnvironment(
        config=EnvConfig(
            max_episode_steps=6,
            warn_turn_threshold=2,
            critical_turn_threshold=4,
        ),
        episode_id="long-episode",
        seed=123,
        logger=validation_logger,
        event_logger=event_logger,
        termination_logger=termination_logger,
        privileged_diagnostics_logger=diagnostics_logger,
        diagnostics_metadata={
            "environment_version": "env-v1",
            "agent_preset": "preset-v2",
            "reward_config_version": "reward-v3",
            "experiment_id": "experiment-4",
        },
    )

    try:
        for _ in range(6):
            environment.step(0)

        records = _records(diagnostics_stream)
        assert [record["level"] for record in records] == ["WARNING", "CRITICAL"]
        assert [record["threshold"] for record in records] == [2, 4]
        assert [record["step"] for record in records] == [3, 5]
        assert all(record["episode_id"] == "long-episode" for record in records)
        assert all(record["run_id"] == "run-7" for record in records)
        assert all(record["job_id"] == "job-7" for record in records)
        assert all(record["seed"] == 123 for record in records)
        assert all(record["alive_players"] == [0, 1, 2, 3, 4] for record in records)
        assert all(
            sorted(record["health"].values()) == [4, 4, 4, 4, 5]
            for record in records
        )
        assert all(len(record["recent_events"]) == record["step"] for record in records)
        assert all(record["state_snapshot"]["environment_step"] == record["step"] for record in records)
        assert all(len(record["roles"]) == 5 for record in records)
        assert all(record["diagnostic_metadata"]["experiment_id"] == "experiment-4" for record in records)

        # Steps after the critical crossing do not repeat either threshold.
        assert len(records) == 2

        termination_records = _records(termination_stream)
        assert len(termination_records) == 1
        termination = termination_records[0]
        assert termination["termination_status"] == "truncated"
        assert termination["initial_state"]["environment_step"] == 0
        assert termination["final_state"]["environment_step"] == 6
        assert len(termination["public_events"]) == 6
        assert "roles" not in termination["initial_state"]
        assert "roles" not in termination["final_state"]
        assert "roles" not in termination
        assert "roles" not in event_stream.getvalue()
    finally:
        environment.close()
        event_logger.close()
        termination_logger.close()
        diagnostics_logger.close()
        validation_logger.close()


def test_reset_starts_a_new_episode_threshold_ledger() -> None:
    stream = io.StringIO()
    validation_stream = io.StringIO()
    event_stream = io.StringIO()
    termination_stream = io.StringIO()
    logger = StructuredLogger(
        LogChannel.PRIVILEGED_DIAGNOSTICS,
        run_id="run-1",
        job_id="job-1",
        episode_id="episode-1",
        stream=stream,
    )
    validation_logger = StructuredLogger(
        LogChannel.INVALID_ACTION,
        run_id="run-1",
        job_id="job-1",
        episode_id="episode-1",
        stream=validation_stream,
    )
    event_logger = StructuredLogger(
        LogChannel.EPISODE_EVENTS,
        run_id="run-1",
        job_id="job-1",
        episode_id="episode-1",
        stream=event_stream,
    )
    termination_logger = StructuredLogger(
        LogChannel.TERMINATION,
        run_id="run-1",
        job_id="job-1",
        episode_id="episode-1",
        stream=termination_stream,
    )
    environment = BangEnvironment(
        config=EnvConfig(max_episode_steps=4, warn_turn_threshold=1, critical_turn_threshold=2),
        logger=validation_logger,
        event_logger=event_logger,
        termination_logger=termination_logger,
        privileged_diagnostics_logger=logger,
    )
    try:
        for _ in range(3):
            environment.step(0)
        environment.reset(episode_id="episode-2", seed=99)
        for _ in range(3):
            environment.step(0)

        records = _records(stream)
        assert [(record["episode_id"], record["threshold"]) for record in records] == [
            ("episode-1", 1),
            ("episode-1", 2),
            ("episode-2", 1),
            ("episode-2", 2),
        ]
        assert records[-1]["seed"] == 99
    finally:
        environment.close()
        logger.close()
        validation_logger.close()
        event_logger.close()
        termination_logger.close()
