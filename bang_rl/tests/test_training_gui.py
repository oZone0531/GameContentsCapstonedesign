from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from bang_rl.app import TrainingProgress
from bang_rl.config import ConfigService, EmotionConfig, SystemConfig
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


def test_settings_snapshot_exposes_dummy_emotion_state() -> None:
    editor = SettingsEditor(SystemConfig())
    snapshot = editor.snapshot
    assert snapshot.dummy_emotion_count == len(snapshot.config.preset.emotion.emotions)
    assert snapshot.dummy_emotion_count == snapshot.config.preset.emotion.emotion_count
    assert len(snapshot.dummy_emotions) == snapshot.dummy_emotion_count
    assert snapshot.dummy_emotion["emotions"][0]["name"] == (
        snapshot.config.preset.emotion.emotions[0].name
    )


def test_settings_editor_applies_per_emotion_initial_values() -> None:
    editor = SettingsEditor(SystemConfig())
    emotion = dict(editor.snapshot.dummy_emotion)
    definitions = [dict(item) for item in emotion["emotions"]]
    definitions[0]["initial_value"] = 0.42
    emotion["emotions"] = definitions
    state = editor.apply_dummy_emotion(emotion)
    assert state.config.preset.emotion.emotions[0].initial_value == pytest.approx(0.42)


def test_settings_editor_applies_named_dummy_emotions() -> None:
    editor = SettingsEditor(SystemConfig())
    emotion = {
        "schema_version": "2.0",
        "emotion_count": 3,
        "emotions": [
            {"name": "fear", "initial_value": 0.1, "change_rate": 0.2},
            {"name": "anger", "initial_value": 0.5, "action_updates": {"bang": 0.3}},
            {"name": "calm", "initial_value": 0.9, "effects": {"aggression": -0.2}},
        ],
    }
    state = editor.apply_dummy_emotion(emotion)
    names = [item.name for item in state.config.preset.emotion.emotions]
    assert names == ["fear", "anger", "calm"]
    assert state.config.preset.emotion.emotion_count == 3
    assert state.config.preset.emotion.schema_version == "2.0"


def test_settings_editor_preserves_reward_and_terminal_updates() -> None:
    # A preset that supplies reward_updates/terminal_updates must survive a
    # dialog round-trip; editing one emotion must never silently wipe them.
    editor = SettingsEditor(SystemConfig())
    emotion = {
        "schema_version": "2.0",
        "emotion_count": 1,
        "emotions": [
            {
                "name": "fear",
                "initial_value": 0.3,
                "change_rate": 0.1,
                "decay_rate": 0.05,
                "event_updates": {"hit": 0.2},
                "action_updates": {"bang": 0.1},
                "reward_updates": {"win": -0.4},
                "terminal_updates": {"eliminated": 0.6},
                "effects": {"aggression": 0.2},
            }
        ],
    }
    staged = editor.apply_dummy_emotion(emotion)

    # Simulate the dialog reading the staged config back out, serializing each
    # row, and re-applying (the destructive path the review flagged).
    reloaded = dict(staged.dummy_emotion)
    definitions = [dict(item) for item in reloaded["emotions"]]
    round_tripped = editor.apply_dummy_emotion({**reloaded, "emotions": definitions})

    definition = round_tripped.config.preset.emotion.emotions[0]
    assert definition.reward_updates == {"win": -0.4}
    assert definition.terminal_updates == {"eliminated": 0.6}
    assert definition.event_updates == {"hit": 0.2}
    assert definition.action_updates == {"bang": 0.1}
    assert definition.effects == {"aggression": 0.2}


def test_dialog_apply_preserves_non_zero_initial_baseline() -> None:
    # The dummy-AI dialog's _collect_emotion now carries initial_baseline, so a
    # non-zero baseline must survive an Apply (previously it was silently reset
    # to EmotionConfig's 0.0 default because the collected dict omitted it).
    editor = SettingsEditor(SystemConfig())
    # Stage a config that carries a non-zero baseline (as if loaded from a file).
    editor.apply_dummy_emotion(
        {
            "schema_version": "1.0",
            "emotion_count": 1,
            "initial_baseline": 0.6,
            "emotions": [{"name": "fear", "initial_value": 0.2}],
        }
    )
    assert editor.snapshot.config.preset.emotion.initial_baseline == pytest.approx(0.6)

    # Reproduce exactly what DummyAISettingsDialog._collect_emotion builds: the
    # header fields + per-row definitions + the baseline read from its spin.
    snapshot = editor.snapshot
    emotion = dict(snapshot.dummy_emotion)
    collected = {
        "schema_version": str(emotion.get("schema_version", "1.0")),
        "emotion_count": int(emotion.get("emotion_count", 1)),
        "initial_baseline": float(emotion.get("initial_baseline", 0.0)),
        "emotions": [dict(item) for item in emotion["emotions"]],
    }
    applied = editor.apply_dummy_emotion(collected)
    # The baseline is preserved rather than reset to 0.0.
    assert applied.config.preset.emotion.initial_baseline == pytest.approx(0.6)


def test_non_zero_initial_baseline_survives_save_load_round_trip(tmp_path: Path) -> None:
    source = SettingsEditor(SystemConfig())
    source.apply_dummy_emotion(
        {
            "schema_version": "1.0",
            "emotion_count": 1,
            "initial_baseline": 0.35,
            "emotions": [{"name": "fear", "initial_value": 0.2}],
        }
    )
    path = tmp_path / "emotions.json"
    source.export_dummy_emotion(path)

    target = SettingsEditor(SystemConfig())
    staged = target.import_dummy_emotion(path)
    assert staged.config.preset.emotion.initial_baseline == pytest.approx(0.35)


def test_settings_editor_rejects_duplicate_emotion_names() -> None:
    editor = SettingsEditor(SystemConfig())
    emotion = {
        "emotions": [
            {"name": "fear", "initial_value": 0.1},
            {"name": "fear", "initial_value": 0.2},
        ],
    }
    with pytest.raises(SettingsValidationError, match="unique"):
        editor.apply_dummy_emotion(emotion)


def test_settings_editor_rejects_out_of_range_emotion_initial_value() -> None:
    editor = SettingsEditor(SystemConfig())
    emotion = {"emotions": [{"name": "fear", "initial_value": 1.5}]}
    with pytest.raises(SettingsValidationError, match=r"initial_value|\[0, 1\]"):
        editor.apply_dummy_emotion(emotion)


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_export_import_dummy_emotion_round_trips(tmp_path: Path, suffix: str) -> None:
    source = SettingsEditor(SystemConfig())
    source.apply_dummy_emotion(
        {
            "schema_version": "2.0",
            "emotion_count": 2,
            "emotions": [
                {"name": "fear", "initial_value": 0.25, "change_rate": 0.1},
                {"name": "anger", "initial_value": 0.75, "action_updates": {"bang": 0.3}},
            ],
        }
    )
    original = source.snapshot.dummy_emotion

    path = tmp_path / f"emotions{suffix}"
    written = source.export_dummy_emotion(path)
    assert written.is_file()

    # The exported file contains ONLY the emotion config, not a full SystemConfig.
    document = ConfigService().load(path, EmotionConfig)
    assert set(document.model_dump(mode="json")) == set(original)

    target = SettingsEditor(SystemConfig())
    staged = target.import_dummy_emotion(path)

    assert staged.dummy_emotion == original
    names = [item.name for item in staged.config.preset.emotion.emotions]
    assert names == ["fear", "anger"]
    assert staged.config.preset.emotion.emotion_count == 2
    initial_values = [item.initial_value for item in staged.config.preset.emotion.emotions]
    assert initial_values == pytest.approx([0.25, 0.75])


def test_import_dummy_emotion_rejects_invalid_file(tmp_path: Path) -> None:
    editor = SettingsEditor(SystemConfig())
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema_version": "1.0", "emotion_count": 2, "emotions": '
        '[{"name": "fear", "initial_value": 0.1}, {"name": "fear", "initial_value": 0.2}]}',
        encoding="utf-8",
    )
    with pytest.raises(SettingsValidationError, match="unique"):
        editor.import_dummy_emotion(path)


def test_import_dummy_emotion_rejects_out_of_range_initial_value(tmp_path: Path) -> None:
    editor = SettingsEditor(SystemConfig())
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: '1.0'\n"
        "emotion_count: 1\n"
        "emotions:\n"
        "  - name: fear\n"
        "    initial_value: 1.5\n",
        encoding="utf-8",
    )
    with pytest.raises(SettingsValidationError, match=r"initial_value|\[0, 1\]"):
        editor.import_dummy_emotion(path)


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


def test_training_start_wires_to_service_start_orchestrator() -> None:
    """The 'AI 학습' Start button reaches TrainingMonitorController.start -> service.start."""

    service = FakeTrainingService(progress())
    controller = TrainingMonitorController(service)
    setup = TrainingSetup(config=SystemConfig(), episodes=3, target_transitions=7)
    controller.start(setup)
    assert len(service.requests) == 1
    request = service.requests[0]
    assert request.episodes == 3
    assert request.target_transitions == 7
    assert controller.handle is service.handle


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
