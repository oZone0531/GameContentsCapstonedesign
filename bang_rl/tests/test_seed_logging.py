"""Focused tests for TASK-0-5 deterministic seeds and structured logging."""

from __future__ import annotations

import io
import json
import random

import numpy as np

from bang_rl.runtime.logging import (
    LOG_CHANNELS,
    LogChannel,
    LogManager,
    LongGameDiagnostics,
    StructuredLogger,
)
from bang_rl.runtime.seed import SeedManager, derive_seed, seed_everything


def test_global_seed_api_repeats_python_numpy_and_torch_draws() -> None:
    snapshot = {"config_version": "v1", "reward": {"version": "r1"}}

    seed_everything(41, snapshot=snapshot)
    first = (random.random(), float(np.random.random()))
    try:
        import torch
    except ImportError:  # pragma: no cover - project dependency normally exists
        torch_value = None
    else:
        torch_value = float(torch.rand(1).item())

    seed_everything(41, snapshot=snapshot)
    second = (random.random(), float(np.random.random()))
    if torch_value is not None:
        import torch

        assert torch_value == float(torch.rand(1).item())
    assert first == second


def test_named_environment_and_action_streams_are_reproducible_and_independent() -> None:
    first = SeedManager(123, snapshot={"job": "same"}).streams("worker", 2, "episode", 7)
    second = SeedManager(123, snapshot={"job": "same"}).streams("worker", 2, "episode", 7)
    assert first.seed == second.seed
    assert first.environment.integers(0, 1_000, size=8).tolist() == second.environment.integers(0, 1_000, size=8).tolist()
    assert first.action.integers(0, 6, size=8).tolist() == second.action.integers(0, 6, size=8).tolist()
    assert first.environment is not first.action
    assert SeedManager(123).environment_seed("episode-1") != SeedManager(123).action_seed("agent-1", episode_id="episode-1")


def test_derived_seeds_are_stable_across_snapshot_mapping_order() -> None:
    left = {"alpha": 1, "nested": {"x": 2, "y": 3}}
    right = {"nested": {"y": 3, "x": 2}, "alpha": 1}
    assert derive_seed(9, "worker", 1, "episode", 2, snapshot=left) == derive_seed(
        9, "worker", 1, "episode", 2, snapshot=right
    )
    assert derive_seed(9, "worker", 1) != derive_seed(9, "worker", 2)
    assert derive_seed(9, "worker", 1, snapshot=left) != derive_seed(9, "worker", 1, snapshot={"alpha": 2})


def test_structured_json_log_has_context_channel_and_component_fields() -> None:
    stream = io.StringIO()
    logger = StructuredLogger(
        LogChannel.EPISODE_EVENTS,
        run_id="run-1",
        job_id="job-1",
        episode_id="episode-1",
        agent_id="agent-2",
        component="environment",
        stream=stream,
    )
    logger.info("valid action", step=12, action=0, event_type="pass")
    logger.close()

    record = json.loads(stream.getvalue())
    assert record["channel"] == "episode_events"
    assert record["run_id"] == "run-1"
    assert record["job_id"] == "job-1"
    assert record["episode_id"] == "episode-1"
    assert record["step"] == 12
    assert record["agent_id"] == "agent-2"
    assert record["component"] == "environment"
    assert record["event_type"] == "pass"


def test_secret_fields_and_secret_assignments_are_redacted() -> None:
    stream = io.StringIO()
    logger = StructuredLogger("training", stream=stream)
    logger.info(
        "request token=super-secret",
        password="hunter2",
        api_key="key-value",
        ordinary="safe-value",
    )
    logger.close()

    encoded = stream.getvalue()
    assert "hunter2" not in encoded
    assert "key-value" not in encoded
    assert "super-secret" not in encoded
    assert "safe-value" in encoded
    assert "[REDACTED]" in encoded


def test_channels_are_defined_and_privileged_logs_are_physically_separate(tmp_path) -> None:
    assert set(LOG_CHANNELS) == {
        "episode_events",
        "termination",
        "reward_breakdown",
        "invalid_action",
        "training",
        "evaluation",
        "privileged_diagnostics",
    }
    with LogManager(tmp_path, run_id="run-1", job_id="job-1") as manager:
        public = manager.logger("episode_events", episode_id="episode-1")
        privileged = manager.logger("privileged_diagnostics", episode_id="episode-1")
        public.info("public state", step=1)
        privileged.info("private state", step=1, true_roles=["sheriff"])

    assert (tmp_path / "episode_events.jsonl").is_file()
    assert (tmp_path / "privileged_diagnostics" / "privileged_diagnostics.jsonl").is_file()
    assert "true_roles" not in (tmp_path / "episode_events.jsonl").read_text(encoding="utf-8")
    assert "true_roles" in (tmp_path / "privileged_diagnostics" / "privileged_diagnostics.jsonl").read_text(encoding="utf-8")


def test_long_game_diagnostics_emit_each_threshold_once_and_include_state(tmp_path) -> None:
    with LogManager(tmp_path, run_id="run-1", job_id="job-1") as manager:
        logger = manager.logger("privileged_diagnostics", episode_id="episode-1")
        diagnostics = LongGameDiagnostics(
            logger,
            warn_turn_threshold=2,
            critical_turn_threshold=4,
            max_episode_steps=10,
        )
        for step in (3, 5, 6):
            diagnostics.observe(
                step,
                alive_players=[0, 2],
                health={0: 5, 2: 3},
                recent_events=[{"event": "pass"}],
                seed=99,
                roles=["sheriff", "outlaw_a"],
            )

    records = [
        json.loads(line)
        for line in (tmp_path / "privileged_diagnostics" / "privileged_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["level"] for record in records] == ["WARNING", "CRITICAL"]
    assert {record["threshold"] for record in records} == {2, 4}
    assert all(record["alive_players"] == [0, 2] for record in records)
    assert all(record["roles"] == ["sheriff", "outlaw_a"] for record in records)
