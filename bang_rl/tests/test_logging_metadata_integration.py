from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from bang_rl.app import TrainingReport, TrainingProgress
from bang_rl.config import SystemConfig, TrainingJob
from bang_rl.runtime.logging import LogChannel, LogManager, StructuredLogger, read_jsonl
from bang_rl.runtime.metadata import build_run_metadata, canonical_json, content_hash


def test_shared_context_fields_and_public_privileged_separation() -> None:
    public_stream = io.StringIO()
    private_stream = io.StringIO()
    public = StructuredLogger(
        LogChannel.TRAINING,
        run_id="run-1", job_id="job-1", episode_id="episode-1", worker_id=2,
        main_agent_transition=7, ppo_update=3, agent_id="agent-0",
        component="ppo", stream=public_stream,
    )
    private = StructuredLogger(
        LogChannel.PRIVILEGED_DIAGNOSTICS,
        run_id="run-1", job_id="job-1", episode_id="episode-1", worker_id=2,
        component="evaluator", stream=private_stream,
    )
    public.info("metric", step=11, true_roles=["TRAITOR"], reward=0.25)
    private.info("diagnostic", step=11, true_roles=["TRAITOR"])
    public.close()
    private.close()

    public_record = json.loads(public_stream.getvalue())
    private_record = json.loads(private_stream.getvalue())
    assert public_record["run_id"] == private_record["run_id"] == "run-1"
    assert public_record["job_id"] == "job-1"
    assert public_record["episode_id"] == "episode-1"
    assert public_record["worker_id"] == 2
    assert public_record["step"] == public_record["environment_step"] == 11
    assert public_record["main_agent_transition"] == 7
    assert public_record["ppo_update"] == 3
    assert public_record["agent"] == public_record["agent_id"] == "agent-0"
    assert public_record["component"] == "ppo"
    assert "true_roles" not in public_record
    assert "TRAITOR" not in public_stream.getvalue()
    assert private_record["true_roles"] == ["TRAITOR"]


def test_metadata_hashes_and_serialization_are_order_independent() -> None:
    config = SystemConfig(config_version="cfg-1", seed=12)
    snapshot = TrainingJob.start(config, mode="rl", seed=12, job_id="job-1").snapshot
    first = build_run_metadata(run_id="run-1", snapshot=snapshot, config=config)
    reordered = {"config_version": "cfg-1", "seed": 12, "nested": {"b": 2, "a": 1}}
    reordered_again = {"nested": {"a": 1, "b": 2}, "seed": 12, "config_version": "cfg-1"}
    assert canonical_json(reordered) == canonical_json(reordered_again)
    assert content_hash(reordered) == content_hash(reordered_again)
    assert first["hashes"]["config_snapshot_sha256"] == content_hash(config.model_dump(mode="json"))
    assert first["hashes"]["job_snapshot_sha256"] == content_hash(snapshot.to_dict())
    assert first["config_version"] == "cfg-1"
    assert first["versions"]["reward_config_version"] == config.reward_config.version


def test_jsonl_is_append_and_resume_safe(tmp_path: Path) -> None:
    with LogManager(tmp_path, run_id="run-1", job_id="job-1") as manager:
        manager.logger("training", component="trainer").info("first", step=1)
    with LogManager(tmp_path, run_id="run-1", job_id="job-1") as manager:
        manager.logger("training", component="trainer").info("second", step=2)

    records = read_jsonl(tmp_path / "training.jsonl")
    assert [record["message"] for record in records] == ["first", "second"]
    assert [record["step"] for record in records] == [1, 2]


def test_jsonl_preserves_middle_errors_and_tolerates_only_final_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]

    path.write_text('{"id": 1}\nnot-json\n{"id": 3}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)

    path.write_text('{"id": 1}\n{"id": 2}\npartial', encoding="utf-8")
    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path, tolerate_trailing_partial=False)


def test_training_report_exposes_linked_metadata_and_pause_contract() -> None:
    config = SystemConfig(config_version="cfg-report")
    snapshot = TrainingJob.start(config, job_id="job-report").snapshot
    progress = TrainingProgress("job-report", "finished", 12, 4, 1, 2, 1, 3.5)
    metadata = build_run_metadata(
        run_id="job-report", job_id="job-report", component="training-service",
        snapshot=snapshot, counters={"environment_steps": 12, "main_agent_transitions": 4, "ppo_updates": 1},
        active_training_seconds=3.5, paused_seconds=2.0,
    )
    report = TrainingReport("job-report", "finished", snapshot, progress, None, None, (), metadata)
    payload = report.to_dict()
    assert payload["metadata"]["job_id"] == "job-report"
    assert payload["metadata"]["counters"]["environment_steps"] == 12
    assert payload["metadata"]["paused_seconds"] == 2.0
    assert payload["metadata"]["paused_time_excluded"] is True
