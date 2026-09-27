from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from bang_rl.app import TrainingProgress
from bang_rl.config import ConfigService, SystemConfig
from bang_rl.gui.settings import SettingsEditor, SettingsValidationError
from bang_rl.gui.training import (
    TrainingMonitorController,
    TrainingSetup,
    public_training_metadata,
)


@dataclass
class FakeTrainingService:
    progress_value: TrainingProgress

    def __post_init__(self) -> None:
        self.requests = []
        self.calls = []
        self.snapshot = ConfigService().snapshot(SystemConfig(seed=12), mode="rl", seed=12, job_id="job-gui")
        self.handle = SimpleNamespace(job_id="job-gui", snapshot=self.snapshot)

    def start(self, request):
        self.requests.append(request)
        return self.handle

    def progress(self, handle):
        return self.progress_value

    def pause(self, handle, timeout=None):
        self.calls.append(("pause", timeout))
        self.progress_value = TrainingProgress("job-gui", "paused", 4, 2, 0, 1, 1, 0.1)

    def resume(self, handle):
        self.calls.append(("resume", None))
        self.progress_value = TrainingProgress("job-gui", "running", 4, 2, 0, 1, 1, 0.1)

    def cancel(self, handle):
        self.calls.append(("cancel", None))
        self.progress_value = TrainingProgress("job-gui", "cancelled", 4, 2, 0, 1, 1, 0.1)


def progress(status: str = "running") -> TrainingProgress:
    return TrainingProgress(
        "job-gui",
        status,
        8,
        5,
        2,
        2,
        1,
        1.25,
        artifact_path="models/result.bangmodel",
        report_path="logs/report.json",
        truncated_episodes=1,
        normal_episodes=1,
        win_rate=0.75,
        truncated_rate=0.5,
        metrics={"loss": 0.2},
        logs=({"message": "public update"}, {"privileged_diagnostics": "secret"}),
    )


def test_settings_editor_validates_editable_training_and_resource_values(tmp_path: Path) -> None:
    editor = SettingsEditor(SystemConfig())
    state = editor.update(
        ppo_clip_epsilon=0.15,
        emotion_initial_baseline=0.4,
        cpu_cores=2,
        ram_gb=4,
        use_gpu=True,
        workers=3,
        batch_size=32,
        buffer_size=512,
    )
    assert state.ppo_clip_epsilon == pytest.approx(0.15)
    assert state.emotion_initial_baseline == pytest.approx(0.4)
    assert state.hardware["workers"] == 3
    assert state.hardware["use_gpu"] is True
    destination = editor.save(tmp_path / "settings.json")
    assert destination.dirty is False
    assert (tmp_path / "settings.json").is_file()


def test_settings_editor_rejects_invalid_values_before_staging() -> None:
    editor = SettingsEditor(SystemConfig())
    before = editor.snapshot
    with pytest.raises(SettingsValidationError, match="initial_baseline"):
        editor.update(emotion_initial_baseline=1.1)
    assert editor.snapshot.config == before.config


def test_monitor_wires_start_pause_resume_stop_and_displays_public_progress() -> None:
    service = FakeTrainingService(progress())
    controller = TrainingMonitorController(service)
    state = controller.start(TrainingSetup(config=SystemConfig(), episodes=2, target_transitions=10))
    assert service.requests[0].target_transitions == 10
    assert state.progress is service.progress_value
    assert state.snapshot.job_id == "job-gui"
    assert state.progress.truncated_episodes == 1
    assert state.progress.report_path == "logs/report.json"
    assert state.win_rate_points == (0.75,)
    assert state.logs == ("public update",)

    controller.pause(timeout=30.0)
    controller.resume()
    controller.stop()
    assert [call[0] for call in service.calls] == ["pause", "resume", "cancel"]
    assert controller.snapshot.progress.status == "cancelled"


def test_privileged_metadata_is_not_renderable_in_public_monitor_data() -> None:
    metadata = public_training_metadata(
        {
            "environment_version": "1.0",
            "roles": ["traitor"],
            "privileged_diagnostics": {"actual_roles": ["secret"]},
            "emotion_schema_version": "1.0",
        }
    )
    assert metadata == {"environment_version": "1.0", "emotion_schema_version": "1.0"}


def test_training_import_boundary_excludes_gui_dependencies_from_runtime() -> None:
    root = Path(__file__).parents[1]
    for relative in ("app", "env", "agents", "training"):
        for path in (root / relative).rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            assert "pyqt6" not in text
            assert "pyqtgraph" not in text


def test_monitor_does_not_create_resume_or_checkpoint_controls() -> None:
    source = (Path(__file__).parents[1] / "gui" / "training.py").read_text(encoding="utf-8").lower()
    assert "checkpoint" not in source
    assert "benchmark" not in source
