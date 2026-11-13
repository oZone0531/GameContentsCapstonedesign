"""Epoch vs time limits, stop/pause handling and graph-point sampling for
Phase-2 module training."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest

from bang_rl.app import TrainingConfig, TrainingService
from bang_rl.config import SystemConfig
from bang_rl.gui.qt_compat import QT_AVAILABLE
from bang_rl.training import pipeline_runner
from bang_rl.training.pipeline_runner import _TrainingStopController, _publish_module_metric


# --- graph point sampling -----------------------------------------------------


def _batch(phase: str, accuracy: float) -> dict[str, object]:
    return {"event": "batch", "phase": phase, "accuracy": accuracy}


def test_graph_points_are_sampled_every_interval_batches() -> None:
    run = SimpleNamespace(
        lock=RLock(), metrics={}, config=SimpleNamespace(graph_point_interval=3)
    )
    for index in range(7):
        _publish_module_metric(run, _batch("phase2_supervised_intent", index / 10))
    # Batches 0, 3 and 6 are kept (the first batch always is).
    assert run.metrics["intent_accuracy_points"] == [0.0, 0.3, 0.6]

    # Each module has its own counter.
    for index in range(4):
        _publish_module_metric(run, _batch("phase2_supervised_emotion", index / 10))
    assert run.metrics["emotion_accuracy_points"] == [0.0, 0.3]


def test_every_batch_is_kept_without_an_interval() -> None:
    run = SimpleNamespace(lock=RLock(), metrics={})
    for index in range(5):
        _publish_module_metric(run, _batch("phase2_supervised_intent", index / 10))
    assert len(run.metrics["intent_accuracy_points"]) == 5


# --- stop controller ----------------------------------------------------------


def test_stop_controller_handles_cancel_deadline_and_pause() -> None:
    run = SimpleNamespace(lock=RLock(), status="running")

    assert _TrainingStopController(run).should_stop() is False
    run.status = "cancelling"
    assert _TrainingStopController(run).should_stop() is True

    run.status = "running"
    assert _TrainingStopController(run, budget_seconds=0.0).should_stop() is True

    # Pausing blocks (it is not a stop) and the paused time is not charged to
    # the budget: 0.4 s of budget survives a 0.6 s pause...
    controller = _TrainingStopController(run, budget_seconds=0.4)
    run.status = "paused"
    threading.Timer(0.6, lambda: setattr(run, "status", "running")).start()
    started = time.monotonic()
    assert controller.should_stop() is False
    assert time.monotonic() - started >= 0.5
    # ...and only runs out once real training time has passed.
    time.sleep(0.8)
    assert controller.should_stop() is True

    # Stop pressed while paused ends the wait.
    run.status = "paused"
    threading.Timer(0.2, lambda: setattr(run, "status", "cancelling")).start()
    assert _TrainingStopController(run).should_stop() is True


# --- end to end: epoch mode, time mode, stop -------------------------------------


@pytest.fixture(scope="module")
def collected(tmp_path_factory: pytest.TempPathFactory) -> tuple[SystemConfig, Path]:
    root = tmp_path_factory.mktemp("limits")
    config = SystemConfig.model_validate(
        {
            "dataset": {
                "canonical_path": str(root / "canonical"),
                "cache_path": str(root / "cache"),
            },
            "sl_training": {"epochs": 1, "batch_size": 32},
        }
    )
    service = TrainingService()
    handle = service.start(
        TrainingConfig(
            config=config,
            mode="data_collection",
            episodes=12,
            report_path=root / "collect.json",
            output_path=root / "out",
        )
    )
    assert service.wait(handle, timeout=300).status == "finished"
    return config, root


def _module_request(config: SystemConfig, root: Path, name: str, **kwargs: object) -> TrainingConfig:
    return TrainingConfig(
        config=config,
        mode="module_training",
        episodes=1,
        report_path=root / f"{name}.json",
        output_path=root / "out",
        **kwargs,
    )


def test_epoch_mode_trains_exactly_the_requested_epochs(
    collected: tuple[SystemConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = collected
    monkeypatch.setattr(pipeline_runner, "PROJECT_ROOT", root)
    epochs_config = config.model_copy(
        update={"sl_training": config.sl_training.model_copy(update={"epochs": 2})}
    )
    service = TrainingService()
    handle = service.start(_module_request(epochs_config, root, "epochs"))
    report = service.wait(handle, timeout=300)

    assert report.status == "finished"
    metrics = report.progress.metrics
    assert metrics["train_by_time"] is False
    assert metrics["intent_epochs_completed"] == 2
    assert metrics["emotion_epochs_completed"] == 2


def test_time_mode_ignores_the_epoch_count_and_stops_at_the_budget(
    collected: tuple[SystemConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = collected  # sl_training.epochs == 1
    monkeypatch.setattr(pipeline_runner, "PROJECT_ROOT", root)
    service = TrainingService()
    started = time.monotonic()
    handle = service.start(
        _module_request(config, root, "timed", train_by_time=True, max_training_seconds=4.0)
    )
    report = service.wait(handle, timeout=300)

    assert report.status == "finished"
    metrics = report.progress.metrics
    assert metrics["train_by_time"] is True
    # Both modules trained past the configured single epoch, in turn.
    assert metrics["intent_epochs_completed"] >= 2
    assert metrics["emotion_epochs_completed"] >= 1
    assert Path(metrics["intent_artifact_path"]).is_file()
    assert Path(metrics["emotion_artifact_path"]).is_file()
    assert time.monotonic() - started < 120


def test_unlimited_time_mode_runs_until_stop_and_saves_the_model(
    collected: tuple[SystemConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = collected
    monkeypatch.setattr(pipeline_runner, "PROJECT_ROOT", root)
    service = TrainingService()
    handle = service.start(_module_request(config, root, "unlimited", train_by_time=True))

    deadline = time.monotonic() + 120
    while True:
        progress = service.progress(handle)
        assert progress.status != "faulted", progress.error
        if progress.metrics.get("intent_validation_accuracy") is not None:
            break  # at least one intent epoch finished
        assert time.monotonic() < deadline, "no epoch finished in time"
        time.sleep(0.1)
    service.cancel(handle)
    report = service.wait(handle, timeout=120)

    assert report.status == "cancelled"
    metrics = report.progress.metrics
    assert metrics["intent_epochs_completed"] >= 1
    assert Path(metrics["intent_artifact_path"]).is_file()


# --- GUI panel -----------------------------------------------------------------


_APP = None


@pytest.fixture
def qapp():
    """A QApplication kept alive for the whole module.

    The reference must be held: a temporary QApplication is destroyed at once
    and constructing a widget afterwards aborts the whole test process.
    """

    global _APP
    from bang_rl.gui.qt_compat import QApplication

    _APP = QApplication.instance() or QApplication([])
    return _APP


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")
def test_setup_panel_epochs_and_time_are_exclusive(qapp, tmp_path: Path) -> None:
    from bang_rl.gui.screens.training_monitor import TrainingSetupPanel

    panel = TrainingSetupPanel(config=SystemConfig(), model_root=tmp_path)
    panel.max_minutes.setText("90")
    panel.graph_interval.setValue(250)

    panel.epochs.setValue(5)
    setup = panel.setup()
    assert setup.train_by_time is False
    assert setup.max_training_seconds is None  # time is ignored when epochs is set
    assert setup.config.sl_training.epochs == 5
    assert setup.graph_point_interval == 250
    assert not panel.max_minutes.isEnabled()

    panel.epochs.setValue(0)  # blank => limit by time
    setup = panel.setup()
    assert setup.train_by_time is True
    assert setup.max_training_seconds == 90 * 60.0
    assert setup.config.sl_training.epochs == 1  # schema minimum, ignored by the runner
    assert panel.max_minutes.isEnabled()

    panel.max_minutes.setText("")
    assert panel.setup().max_training_seconds is None  # unlimited


# --- saved modules: listing, performance summary, continue training ------------


def test_saved_modules_are_listed_summarized_and_can_continue_training(
    collected: tuple[SystemConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from bang_rl.gui.dialogs.module_performance import format_module_run_summary
    from bang_rl.training.module_artifacts import (
        list_saved_module_runs,
        summarize_module_artifact,
        summarize_module_run,
    )

    config, root = collected
    monkeypatch.setattr(pipeline_runner, "PROJECT_ROOT", root)
    model_root = root / "bang_rl" / "models"
    service = TrainingService()

    first = service.wait(service.start(_module_request(config, root, "base")), timeout=300)
    assert first.status == "finished"
    base_intent = Path(first.progress.metrics["intent_artifact_path"])
    base_emotion = Path(first.progress.metrics["emotion_artifact_path"])

    runs = {run.job_id: run for run in list_saved_module_runs(model_root)}
    assert first.job_id in runs
    run = runs[first.job_id]
    assert run.intent_path == base_intent and run.emotion_path == base_emotion

    summary = summarize_module_run(run, reports_dir=root)
    intent, emotion = summary["intent"], summary["emotion"]
    assert intent["kind"] == "intent" and emotion["kind"] == "emotion"
    assert 0.0 <= intent["test_accuracy"] <= 1.0
    assert intent["names"] == ["sheriff", "deputy", "outlaw", "traitor"]
    assert len(intent["test_by_class"]) == 4
    assert len(emotion["names"]) == 9 and len(emotion["test_by_class"]) == 9
    assert emotion["test_mse"] is not None
    assert intent["epochs_completed"] == 1 and intent["total_epochs"] == 1
    assert intent["resumed_from"] is None
    text = format_module_run_summary(summary)
    assert first.job_id in text and "sheriff" in text and "calm" in text

    # Continue training both modules from the saved run.
    resumed = service.wait(
        service.start(
            _module_request(
                config, root, "resumed",
                resume_intent_artifact_path=str(base_intent),
                resume_emotion_artifact_path=str(base_emotion),
            )
        ),
        timeout=300,
    )
    assert resumed.status == "finished", resumed.progress.error
    metrics = resumed.progress.metrics
    assert metrics["resumed_intent_from"] == str(base_intent)
    assert metrics["resumed_emotion_from"] == str(base_emotion)
    assert metrics["intent_total_epochs"] == 2
    assert metrics["emotion_total_epochs"] == 2
    continued = summarize_module_artifact(metrics["intent_artifact_path"])
    assert continued["resumed_from"] == str(base_intent)
    assert continued["prior_epochs"] == 1 and continued["total_epochs"] == 2
    assert continued["best_epoch"] in (0, 1)  # 0 = starting model was kept


def test_continuing_from_a_wrong_module_type_fails_clearly(
    collected: tuple[SystemConfig, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from bang_rl.training.module_artifacts import list_saved_module_runs

    config, root = collected
    monkeypatch.setattr(pipeline_runner, "PROJECT_ROOT", root)
    runs = [run for run in list_saved_module_runs(root / "bang_rl" / "models") if run.intent_path]
    if not runs:
        service = TrainingService()
        report = service.wait(service.start(_module_request(config, root, "seed")), timeout=300)
        runs = [run for run in list_saved_module_runs(root / "bang_rl" / "models") if run.intent_path]
        assert report.status == "finished" and runs
    service = TrainingService()
    # An intent file handed in as the emotion module must be rejected.
    report = service.wait(
        service.start(
            _module_request(
                config, root, "wrong", resume_emotion_artifact_path=str(runs[0].intent_path)
            )
        ),
        timeout=300,
    )
    assert report.status == "faulted"
    assert report.progress.error


def test_summary_reports_unreadable_files_without_raising(tmp_path: Path) -> None:
    from bang_rl.training.module_artifacts import list_saved_module_runs, summarize_module_run

    (tmp_path / "intent").mkdir()
    (tmp_path / "intent" / "intent-broken-job.phase2model").write_bytes(b"not a model")
    (run,) = list_saved_module_runs(tmp_path)
    assert run.job_id == "broken-job" and run.emotion_path is None
    summary = summarize_module_run(run, reports_dir=tmp_path)
    assert summary["intent"]["error"]
    assert summary["emotion"] is None


def _fake_runs(root: Path, *job_ids: str) -> None:
    for kind in ("intent", "emotion"):
        (root / kind).mkdir(parents=True, exist_ok=True)
        for job_id in job_ids:
            (root / kind / f"{kind}-{job_id}.phase2model").write_bytes(b"")


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")
def test_setup_panel_can_select_a_saved_run_to_continue(qapp, tmp_path: Path) -> None:
    from bang_rl.gui.screens.training_monitor import TrainingSetupPanel

    _fake_runs(tmp_path, "job-a")
    panel = TrainingSetupPanel(config=SystemConfig(), model_root=tmp_path)
    assert panel.resume_module.count() == 2  # "none" + job-a
    setup = panel.setup()
    assert setup.resume_intent_artifact_path is None  # default: train from scratch

    assert panel.select_resume_job("job-a")
    setup = panel.setup()
    assert setup.resume_intent_artifact_path == str(tmp_path / "intent" / "intent-job-a.phase2model")
    assert setup.resume_emotion_artifact_path == str(tmp_path / "emotion" / "emotion-job-a.phase2model")

    # A run saved later shows up after refresh, keeping the current choice.
    _fake_runs(tmp_path, "job-b")
    panel.refresh_resume_runs()
    assert panel.resume_module.count() == 3
    assert panel.resume_module.currentData() == "job-a"
    assert not panel.select_resume_job("missing")


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")
def test_performance_dialog_shows_details_and_picks_a_run(qapp, tmp_path: Path) -> None:
    from bang_rl.gui.dialogs.module_performance import ModulePerformanceDialog

    _fake_runs(tmp_path, "job-a")

    def fake_summary(run, reports_dir=None):
        return {
            "job_id": run.job_id,
            "modified": run.modified_text,
            "report_path": None,
            "intent": {"kind": "intent", "epochs_completed": 3, "total_epochs": 3, "best_epoch": 2,
                       "test_accuracy": 0.712, "names": ["sheriff", "deputy", "outlaw", "traitor"],
                       "test_by_class": [0.98, 0.51, 0.88, 0.33]},
            "emotion": None,
        }

    dialog = ModulePerformanceDialog(model_root=tmp_path, summarize=fake_summary)
    assert dialog.list.count() == 1
    text = dialog.details.toPlainText()
    assert "job-a" in text and "0.712" in text and "sheriff 0.980" in text
    dialog._use_for_resume()
    assert dialog.selected_job_id == "job-a"
