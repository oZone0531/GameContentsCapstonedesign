from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pytest

from bang_rl.app import (
    CompatibilityService,
    ConfigService,
    EvaluationRunConfig,
    EvaluationService,
    GameService,
    SessionConfig,
    TrainingConfig,
    TrainingService,
)
from bang_rl.config import SystemConfig
from bang_rl.exceptions import ResourceLimitError, TrainingError
from bang_rl.runtime.seed import SeedManager


class PassAgent:
    def on_episode_start(self, context):
        self.context = context

    def on_episode_end(self, result):
        self.result = result

    def act(self, observation, action_mask):
        assert "roles" not in observation
        return 0


def _pass_agents():
    return tuple(PassAgent() for _ in range(5))


def test_config_snapshot_is_fixed_and_public_game_state_is_read_only() -> None:
    config = SystemConfig(config_version="v1", seed=17)
    service = ConfigService()
    snapshot = service.snapshot(config, mode="rl", job_id="job-fixed")
    config.config_version = "v2"
    assert snapshot.config_version == "v1"
    assert snapshot.config_snapshot["config_version"] == "v1"

    game = GameService()
    handle = game.create_session(SessionConfig(config=config, agents=_pass_agents(), seed=17))
    state = game.get_state(handle)
    assert "roles" not in state.observation
    with pytest.raises(TypeError):
        state.observation["environment_step"] = 999  # type: ignore[index]
    game.close_session(handle)


def test_game_service_steps_through_injected_agents_without_private_state() -> None:
    config = SystemConfig(seed=3)
    game = GameService()
    handle = game.create_session(SessionConfig(config=config, agents=_pass_agents()))
    first = game.get_state(handle)
    result = game.step(handle)
    assert result.action == 0
    assert result.state.environment_step == first.environment_step + 1
    assert result.state.action_mask[0] == 1
    assert result.state.termination_status == "running"
    game.close_session(handle)


def test_seed_manager_is_reproducible_for_same_snapshot() -> None:
    snapshot = ConfigService().snapshot(SystemConfig(seed=41), mode="rl", job_id="fixed-job")
    first = SeedManager(41, snapshot=snapshot)
    second = SeedManager(41, snapshot=snapshot)
    assert first.worker_seed(0) == second.worker_seed(0)
    assert first.environment_seed("episode-0", worker_id=1) == second.environment_seed("episode-0", worker_id=1)
    assert first.action_seed("player_2", episode_id="episode-0", worker_id=1) == second.action_seed("player_2", episode_id="episode-0", worker_id=1)


def test_training_runner_lifecycle_pause_resume_cancel_and_report(tmp_path: Path) -> None:
    started = False

    def runner(handle, run):
        nonlocal started
        started = True
        time.sleep(0.05)
        return {"environment_steps": 4, "transitions": 2}

    service = TrainingService()
    handle = service.start(TrainingConfig(config=SystemConfig(), runner=runner, report_path=tmp_path / "report.json"))
    for _ in range(20):
        if started:
            break
        time.sleep(0.01)
    service.pause(handle)
    assert service.progress(handle).status == "paused"
    service.resume(handle)
    assert service.progress(handle).status == "running"
    service.cancel(handle)
    report = service.wait(handle, timeout=2)
    assert report.status == "cancelled"
    assert report.progress.environment_steps == 4
    assert report.report_path == str(tmp_path / "report.json")
    assert (tmp_path / "report.json").is_file()


def test_default_training_rejects_unwired_pipeline_without_reporting_success() -> None:
    with pytest.raises(TrainingError, match="declared collection/SL/PPO/artifact pipeline"):
        TrainingService().start(TrainingConfig(config=SystemConfig(), episodes=1))


def test_training_resource_validation_and_fault_are_structured() -> None:
    with pytest.raises(Exception):
        SystemConfig.model_validate({"hardware": {"workers": 0}})

    def faulty_runner(handle, run):
        raise RuntimeError("worker exploded")

    service = TrainingService()
    handle = service.start(TrainingConfig(config=SystemConfig(), runner=faulty_runner))
    report = service.wait(handle, timeout=2)
    assert report.status == "faulted"
    assert report.progress.error is not None
    assert report.progress.error["error_type"] == "RuntimeError"


def test_compatibility_service_rejects_missing_artifact_and_evaluation_reports_na(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bangmodel"
    result = CompatibilityService().check(missing, config=SystemConfig())
    assert result.compatible is False
    assert result.errors[0]["error_type"] in {"TrainingError", "FileNotFoundError"}

    report = EvaluationService().run_diagnostic(EvaluationRunConfig(config=SystemConfig(), games=1, runner=lambda options, seeds: {"normal_games": 0, "truncated_games": 1, "normal_wins": 0}, report_path=tmp_path / "evaluation.json"))
    assert report.win_rate == "N/A"
    assert report.truncated_rate == 1.0
    assert report.seed_set == (0,)
    assert (tmp_path / "evaluation.json").is_file()


def test_public_evaluation_service_delegates_artifact_evaluation(monkeypatch) -> None:
    from bang_rl.evaluation import protocol

    calls = {}

    class FakePrimary:
        def __init__(self, artifact, **kwargs):
            calls["artifact"] = artifact
            calls["kwargs"] = kwargs

        def run(self, evaluation):
            calls["evaluation"] = evaluation
            return "protocol-report"

    monkeypatch.setattr(protocol, "PrimaryEvaluationProtocol", FakePrimary)
    artifact = object()
    result = EvaluationService(artifact=artifact).run(EvaluationRunConfig(config=SystemConfig()))
    assert result == "protocol-report"
    assert calls["artifact"] is artifact
    assert calls["evaluation"].games == 500
    assert calls["evaluation"].games_per_seat == 100
    assert calls["evaluation"].seat_policy == "balanced"


def test_application_service_has_no_gui_imports() -> None:
    source = Path(__file__).parents[1] / "app" / "services.py"
    text = source.read_text(encoding="utf-8").lower()
    assert "pyqt" not in text
    assert "pyqtgraph" not in text
