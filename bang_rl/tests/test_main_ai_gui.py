"""Main-AI training GUI: setup panel, module compatibility, monitor and saved-model views."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bang_rl.config import SystemConfig
from bang_rl.data.game_collection import FORMULA_LABEL_SCHEMA
from bang_rl.gui import main_ai_view
from bang_rl.gui.i18n import _STRINGS, tr
from bang_rl.gui.qt_compat import QT_AVAILABLE
from bang_rl.gui.training import (
    TRAINING_MODE_MAIN_AI,
    TRAINING_MODE_MODULE_TRAINING,
    TrainingMonitorSnapshot,
)
from bang_rl.training import main_ai_artifacts, module_artifacts
from bang_rl.training.module_artifacts import (
    MAIN_AI_REASON_EMOTION_SCHEMA,
    MAIN_AI_REASON_INTENT_ROLES,
    MAIN_AI_REASON_MISSING,
    MAIN_AI_REASON_UNREADABLE,
    list_saved_module_runs,
    main_ai_compatibility,
)

_APP = None
ROLES = ("sheriff", "deputy", "outlaw", "traitor")


@pytest.fixture
def qapp():
    """Keep a module-level QApplication reference (a temporary one aborts the process)."""

    global _APP
    from bang_rl.gui.qt_compat import QApplication

    _APP = QApplication.instance() or QApplication([])
    return _APP


# --- fake saved modules -------------------------------------------------------


def _write_intent(root: Path, job: str, *, role_count: int = 4, mtime: float | None = None) -> None:
    folder = root / "intent"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"intent-{job}.phase2model"
    torch.save(
        {
            "artifact_type": "phase2_intent_inference",
            "metadata": {"architecture": {"role_count": role_count}},
        },
        path,
    )
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _write_emotion(
    root: Path, job: str, *, version: str = FORMULA_LABEL_SCHEMA, count: int = 9, mtime: float | None = None
) -> None:
    folder = root / "emotion"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"emotion-{job}.phase2model"
    torch.save(
        {
            "artifact_type": "phase2_emotion_inference",
            "metadata": {"emotion_schema": {"version": version, "count": count}},
        },
        path,
    )
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _runs(root: Path) -> dict[str, module_artifacts.SavedModuleRun]:
    return {run.job_id: run for run in list_saved_module_runs(root)}


def test_main_ai_compatibility_checks_role_count_and_emotion_schema(tmp_path: Path) -> None:
    _write_intent(tmp_path, "good")
    _write_emotion(tmp_path, "good")
    _write_intent(tmp_path, "five-roles", role_count=5)
    _write_emotion(tmp_path, "five-roles")
    _write_intent(tmp_path, "old-emotion")
    _write_emotion(tmp_path, "old-emotion", version="formula-1.0", count=5)
    _write_intent(tmp_path, "intent-only")
    (tmp_path / "emotion" / "emotion-broken.phase2model").write_bytes(b"not a model")
    _write_intent(tmp_path, "broken")
    runs = _runs(tmp_path)

    assert main_ai_compatibility(runs["good"]) == (True, None)
    assert main_ai_compatibility(runs["five-roles"]) == (False, MAIN_AI_REASON_INTENT_ROLES)
    assert main_ai_compatibility(runs["old-emotion"]) == (False, MAIN_AI_REASON_EMOTION_SCHEMA)
    assert main_ai_compatibility(runs["intent-only"]) == (False, MAIN_AI_REASON_MISSING)
    assert main_ai_compatibility(runs["broken"]) == (False, MAIN_AI_REASON_UNREADABLE)


# --- saved main-AI models -----------------------------------------------------


def _main_ai_metadata() -> dict:
    by_role = {name: {"games": 50, "wins": 20, "win_rate": rate} for name, rate in zip(ROLES, (0.7, 0.6, 0.5, 0.4))}
    base = lambda rate, scale: {  # noqa: E731
        "win_rate": rate,
        "by_role": {name: {"games": 50, "wins": 1, "win_rate": value * scale} for name, value in zip(ROLES, (0.6, 0.5, 0.4, 0.3))},
    }
    return {
        "created_at": "2025-01-02T03:04:05+00:00",
        "total_training_time_seconds": 321.0,
        "main_ai_training": {
            "training_games": 1500,
            "ppo_updates": 42,
            "source": {"origin": "latest_saved_modules", "module_job_id": "b8ecacce-1"},
            "evaluation": {
                "games": 250,
                "win_rate": 0.567,
                "by_role": by_role,
                "role_accuracy": 0.81,
                "emotion_mse": 0.0123,
                "truncated_games": 2,
            },
            "normalized_scores": {"mean": 1.25, "by_role": dict(zip(ROLES, (1.5, 1.2, 1.1, None)))},
            "baselines": {"dummy": base(0.45, 1.0), "random": base(0.3, 0.5)},
        },
    }


def test_main_ai_artifacts_are_listed_and_summarized_from_metadata(tmp_path: Path, monkeypatch) -> None:
    newer = tmp_path / "job-new.bangmodel"
    older = tmp_path / "job-old.bangmodel"
    older.write_bytes(b"x")
    newer.write_bytes(b"x")
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    (tmp_path / "ignore.txt").write_text("x")

    models = main_ai_artifacts.list_saved_main_ai_models([tmp_path, tmp_path])  # duplicate folder
    assert [model.job_id for model in models] == ["job-new", "job-old"]

    monkeypatch.setattr(
        main_ai_artifacts, "read_final_artifact_metadata", lambda path: ({}, _main_ai_metadata())
    )
    summary = main_ai_artifacts.summarize_main_ai_artifact(newer)
    assert summary["is_main_ai_training"] is True
    assert summary["training_games"] == 1500 and summary["ppo_updates"] == 42
    assert summary["eval_win_rate"] == pytest.approx(0.567)
    assert summary["eval_by_role"]["sheriff"] == pytest.approx(0.7)
    assert summary["normalized_mean"] == pytest.approx(1.25)
    assert summary["normalized_by_role"]["traitor"] is None
    assert summary["baselines"]["dummy"]["win_rate"] == pytest.approx(0.45)
    assert summary["baselines"]["random"]["by_role"]["sheriff"] == pytest.approx(0.3)
    assert summary["source_module_job_id"] == "b8ecacce-1"


def test_unreadable_main_ai_artifact_reports_an_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.bangmodel"
    broken.write_bytes(b"not a bundle")
    summary = main_ai_artifacts.summarize_main_ai_artifact(broken)
    assert summary["error"] and summary["job_id"] == "broken"


def test_main_ai_model_summary_text_is_readable(tmp_path: Path, monkeypatch) -> None:
    from bang_rl.gui.dialogs.module_performance import format_main_ai_model_summary

    path = tmp_path / "job-a.bangmodel"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        main_ai_artifacts, "read_final_artifact_metadata", lambda _path: ({}, _main_ai_metadata())
    )
    text = format_main_ai_model_summary(main_ai_artifacts.summarize_main_ai_artifact(path))
    for expected in ("job-a", "1500", "42", "0.567", "1.250", "0.450", "0.300", "b8ecacce-1", "0.0123", "0.70 / 0.60 / 0.30"):
        assert expected in text
    assert "—" in text  # the missing traitor score is shown as a dash

    other = format_main_ai_model_summary({"job_id": "x", "path": "p", "error": "bad file"})
    assert "bad file" in other
    plain = format_main_ai_model_summary({"job_id": "x", "path": "p", "is_main_ai_training": False})
    assert tr("perf.main_ai_not_trained") in plain


# --- pure monitor helpers -----------------------------------------------------


def _metrics(*, truncated_rate: float = 0.0) -> dict:
    by_role = dict(zip(ROLES, (0.7, 0.6, 0.5, 0.4)))
    baseline_roles = lambda scale: {  # noqa: E731
        name: {"games": 50, "wins": 5, "win_rate": value * scale} for name, value in zip(ROLES, (0.6, 0.5, 0.4, 0.3))
    }
    row = {
        "ppo_update": 20,
        "training_games": 600,
        "win_rate": 0.62,
        "normalized_score": 1.1,
        "by_role": by_role,
        "role_accuracy": 0.8,
        "emotion_mse": 0.02,
    }
    return {
        "main_games_completed": 600,
        "main_games_total": 1000,
        "main_graph_point_interval": 100,
        "truncated_rate": truncated_rate,
        "main_training_win_rate": 0.55,
        "main_win_rate_points": [0.2, 0.3, 0.4, 0.5],
        "main_reward_points": [-0.6, -0.4, -0.2, 0.0],
        "main_policy_loss_points": [0.1, 0.05],
        "main_value_loss_points": [0.9, 0.7],
        "main_entropy_points": [1.5, 1.4],
        "main_approx_kl_points": [0.01, 0.02],
        "main_eval_win_rate_points": [0.4, 0.62],
        "main_eval_score_points": [0.5, 1.1],
        "main_eval_updates": [10, 20],
        "main_eval_games": [300, 600],
        "main_eval_latest": row,
        "main_best_eval": row,
        "main_baselines": {
            "dummy": {"win_rate": 0.45, "by_role": baseline_roles(1.0)},
            "random": {"win_rate": 0.3, "by_role": baseline_roles(0.5)},
        },
    }


def test_main_ai_view_series_axes_and_truncation_warning() -> None:
    metrics = _metrics()
    assert main_ai_view.series(metrics, "main_win_rate_points") == [0.2, 0.3, 0.4, 0.5]
    assert main_ai_view.series(metrics, "missing") == []
    assert main_ai_view.series({"x": "text"}, "x") == []
    assert main_ai_view.sampled_axis(3, 100) == [1, 101, 201]
    assert main_ai_view.sampled_axis(2, None) == [1, 2]
    assert main_ai_view.baseline_win_rates(metrics) == {"dummy": 0.45, "random": 0.3}
    assert main_ai_view.baseline_win_rates({}) == {"dummy": None, "random": None}
    assert not main_ai_view.truncation_is_high(0.01)  # exactly 1% is not above the limit
    assert main_ai_view.truncation_is_high(0.0101)
    assert not main_ai_view.truncation_is_high(None)


def test_main_ai_summary_lists_eval_baselines_and_saved_model() -> None:
    texts = main_ai_view.format_main_ai_summary(_metrics(), ppo_updates=21, artifact_path="C:/models/job.bangmodel")
    assert "600" in texts["progress"] and "21" in texts["progress"] and "0.550" in texts["progress"]
    assert "0.450" in texts["baselines"] and "0.300" in texts["baselines"]
    for key in ("latest", "best"):
        assert "0.620" in texts[key] and "1.100" in texts[key] and "0.800" in texts[key] and "0.0200" in texts[key]
        assert "0.70 / 0.60 / 0.30" in texts[key]  # main / dummy / random for the first role
    assert "C:/models/job.bangmodel" in texts["saved"]

    empty = main_ai_view.format_main_ai_summary({}, ppo_updates=0, artifact_path=None)
    assert tr("mainai.no_eval") in empty["latest"]
    assert empty["saved"] == tr("mainai.not_saved")
    assert empty["baselines"] == tr("mainai.baselines_pending")


def test_every_translation_has_both_languages_and_tooltips_have_no_em_dash() -> None:
    for key, entry in _STRINGS.items():
        assert entry.get("en") and entry.get("ko"), key
        if key.startswith("tooltip."):
            assert "—" not in entry["en"] and "—" not in entry["ko"], key


# --- setup panel --------------------------------------------------------------


def _main_ai_panel(tmp_path: Path):
    from bang_rl.gui.screens.training_monitor import TrainingSetupPanel

    # ok-old < ok-new < bad-newest by time; the newest compatible one is ok-new.
    _write_intent(tmp_path, "ok-old", mtime=1000)
    _write_emotion(tmp_path, "ok-old", mtime=1000)
    _write_intent(tmp_path, "ok-new", mtime=2000)
    _write_emotion(tmp_path, "ok-new", mtime=2000)
    _write_intent(tmp_path, "bad-newest", role_count=5, mtime=3000)
    _write_emotion(tmp_path, "bad-newest", mtime=3000)
    panel = TrainingSetupPanel(config=SystemConfig(), model_root=tmp_path)
    panel.mode.setCurrentIndex(panel.mode.findData(TRAINING_MODE_MAIN_AI))
    return panel


def _row_label(panel, field) -> str:
    return panel._form.labelForField(field).text()


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")
def test_panel_switches_labels_and_visible_inputs_per_mode(qapp, tmp_path: Path) -> None:
    panel = _main_ai_panel(tmp_path)

    assert _row_label(panel, panel.resume_row) == tr("training.resume_module_main")
    assert not panel.games.isHidden()
    assert panel.episodes.isHidden() and panel.epochs.isHidden() and panel.target.isHidden()
    assert _row_label(panel, panel.graph_interval) == tr("training.graph_interval_main")
    assert panel.resume_module.itemText(0) == tr("training.resume_module_auto")
    assert panel.resume_artifact.toolTip() == tr("tooltip.training_resume_artifact_path_main")
    assert "—" not in panel.resume_module.toolTip()

    panel.mode.setCurrentIndex(panel.mode.findData(TRAINING_MODE_MODULE_TRAINING))
    assert _row_label(panel, panel.resume_row) == tr("training.resume_module")
    # Module training reads epochs (not episodes, games or the legacy target).
    assert panel.games.isHidden() and not panel.epochs.isHidden() and panel.episodes.isHidden()
    assert panel.target.isHidden() and panel.resume_artifact.isHidden() and panel.artifact.isHidden()
    assert _row_label(panel, panel.graph_interval) == tr("training.graph_interval")
    assert panel.resume_module.itemText(0) == tr("training.resume_module_none")
    assert panel.resume_module.currentData() is None  # module training starts from scratch by default
    assert all(panel.resume_module.model().item(i).isEnabled() for i in range(panel.resume_module.count()))


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")
def test_panel_marks_incompatible_runs_and_defaults_to_newest_compatible(qapp, tmp_path: Path) -> None:
    panel = _main_ai_panel(tmp_path)
    combo = panel.resume_module
    marker = tr("training.resume_module_incompatible")

    states = {combo.itemData(i): (combo.itemText(i), combo.model().item(i).isEnabled()) for i in range(combo.count())}
    assert states["bad-newest"][0].endswith(marker) and states["bad-newest"][1] is False
    assert marker not in states["ok-new"][0] and states["ok-new"][1] is True
    assert states[None][1] is True  # the automatic option is always usable
    assert combo.currentData() == "ok-new"

    assert not panel.select_resume_job("bad-newest")  # unusable run cannot be chosen
    assert combo.currentData() == "ok-new"
    assert panel.select_resume_job("ok-old")
    assert panel.select_resume_job(None)
    setup = panel.setup()
    assert setup.resume_intent_artifact_path is None and setup.resume_emotion_artifact_path is None  # auto

    # Choices are remembered per mode when switching back and forth.
    assert panel.select_resume_job("ok-old")
    panel.mode.setCurrentIndex(panel.mode.findData(TRAINING_MODE_MODULE_TRAINING))
    assert panel.select_resume_job("bad-newest")  # allowed for module training
    panel.mode.setCurrentIndex(panel.mode.findData(TRAINING_MODE_MAIN_AI))
    assert combo.currentData() == "ok-old"


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")
def test_panel_builds_main_ai_request_with_games_or_time(qapp, tmp_path: Path) -> None:
    panel = _main_ai_panel(tmp_path)
    panel.games.setValue(30)
    panel.max_minutes.setText("2")
    panel.target.setValue(7)  # hidden legacy field must not leak into a main-AI request

    setup = panel.setup()
    assert setup.mode == TRAINING_MODE_MAIN_AI
    assert setup.episodes == 30 and setup.train_by_time is False
    assert setup.max_training_seconds is None  # time applies only when games is blank
    assert setup.target_transitions is None
    assert setup.resume_intent_artifact_path == str(tmp_path / "intent" / "intent-ok-new.phase2model")
    assert setup.resume_emotion_artifact_path == str(tmp_path / "emotion" / "emotion-ok-new.phase2model")
    assert setup.resume_artifact_path is None  # the default artifacts folder is not a continue source
    assert not panel.max_minutes.isEnabled()
    request = setup.to_request()
    assert request.mode == "main_ai_training" and request.episodes == 30 and request.train_by_time is False

    panel.games.setValue(0)
    setup = panel.setup()
    assert setup.train_by_time is True and setup.max_training_seconds == 120.0
    assert panel.max_minutes.isEnabled()
    assert setup.to_request().train_by_time is True
    panel.max_minutes.setText("")
    assert panel.setup().max_training_seconds is None  # unlimited


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")
def test_bangmodel_takes_precedence_over_the_module_pair(qapp, tmp_path: Path) -> None:
    panel = _main_ai_panel(tmp_path)
    model = tmp_path / "continue.bangmodel"
    model.write_bytes(b"x")

    panel.resume_artifact.setText(str(model))
    setup = panel.setup()
    assert setup.resume_artifact_path == str(model)
    assert setup.resume_intent_artifact_path is None and setup.resume_emotion_artifact_path is None
    assert not panel.resume_module.isEnabled()
    assert tr("training.main_ai_bangmodel_active") in panel.resume_module.toolTip()

    # A folder or a missing file is not a continue source.
    for value in (str(tmp_path), str(tmp_path / "missing.bangmodel"), ""):
        panel.resume_artifact.setText(value)
        setup = panel.setup()
        assert setup.resume_artifact_path is None
        assert setup.resume_intent_artifact_path is not None
        assert panel.resume_module.isEnabled()

    # Module training never continues from a .bangmodel: the (hidden) field is not sent.
    panel.mode.setCurrentIndex(panel.mode.findData(TRAINING_MODE_MODULE_TRAINING))
    panel.resume_artifact.setText(str(tmp_path))
    assert panel.setup().resume_artifact_path is None
    assert panel.resume_module.isEnabled()


@pytest.mark.skipif(not QT_AVAILABLE, reason="PyQt6 is not installed")
def test_each_mode_shows_only_the_inputs_it_reads(qapp) -> None:
    from bang_rl.gui.screens.training_monitor import TrainingSetupPanel
    from bang_rl.gui.training import TRAINING_MODE_DATA_COLLECTION

    panel = TrainingSetupPanel(config=SystemConfig())
    expected = {
        TRAINING_MODE_DATA_COLLECTION: {"episodes", "players"},
        TRAINING_MODE_MODULE_TRAINING: {"epochs", "max_minutes", "graph_interval", "resume_row", "datasets_row", "max_games"},
        TRAINING_MODE_MAIN_AI: {"games", "max_minutes", "graph_interval", "resume_row", "resume_artifact", "artifact"},
    }
    fields = (
        "episodes", "players", "epochs", "games", "max_minutes", "graph_interval",
        "resume_row", "resume_artifact", "artifact", "datasets_row", "target", "output", "max_games",
    )
    for mode, visible in expected.items():
        panel.mode.setCurrentIndex(panel.mode.findData(mode))
        shown = {name for name in fields if not getattr(panel, name).isHidden()}
        assert shown == visible, mode
        assert panel.clip.isHidden() is (mode != TRAINING_MODE_MAIN_AI)


@pytest.mark.skipif(not QT_AVAILABLE, reason="PyQt6 is not installed")
def test_max_games_is_module_training_only_and_reaches_the_request(qapp) -> None:
    from bang_rl.gui.screens.training_monitor import TrainingSetupPanel
    from bang_rl.gui.training import TRAINING_MODE_DATA_COLLECTION

    panel = TrainingSetupPanel(config=SystemConfig())
    panel.mode.setCurrentIndex(panel.mode.findData(TRAINING_MODE_DATA_COLLECTION))
    assert panel.max_games.isHidden()
    panel.mode.setCurrentIndex(panel.mode.findData(TRAINING_MODE_MODULE_TRAINING))
    assert not panel.max_games.isHidden()
    assert panel.max_games.value() == 10_000
    panel.max_games.setValue(2000)
    setup = panel.setup()
    assert setup.max_games == 2000
    assert setup.to_request().max_games == 2000
    panel.max_games.setValue(0)
    assert panel.setup().max_games is None


# --- monitor ------------------------------------------------------------------


def _main_ai_state(metrics: dict, *, status: str = "running", artifact: str | None = None):
    progress = SimpleNamespace(
        status=status,
        ppo_updates=21,
        episodes_completed=600,
        environment_steps=12345,
        truncated_episodes=int(600 * metrics.get("truncated_rate", 0.0)),
        truncated_rate=metrics.get("truncated_rate", 0.0),
        metrics=metrics,
        report_path=None,
        output_path=None,
        artifact_path=artifact,
        main_agent_transitions=900,
        win_rate=0.62,
    )
    snapshot = SimpleNamespace(
        mode="main_ai_training", job_id="job-main", config_version="1", seed=3, config_snapshot={}
    )
    return TrainingMonitorSnapshot(handle=None, progress=progress, snapshot=snapshot, logs=("stage message",))


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")
def test_monitor_renders_main_ai_metrics(qapp, tmp_path: Path) -> None:
    from bang_rl.gui.screens.training_monitor import PYQTGRAPH_AVAILABLE, TrainingMonitor, TrainingSetupPanel

    panel = TrainingSetupPanel(config=SystemConfig(), model_root=tmp_path)
    monitor = TrainingMonitor(setup_panel=panel)
    try:
        monitor._render(_main_ai_state(_metrics(), artifact="C:/models/job.bangmodel"))
        assert not monitor.main_ai_panel.isHidden() and monitor.validation_summary.isHidden()
        assert "600" in monitor.main_ai_labels["progress"].text()
        assert "0.620" in monitor.main_ai_labels["latest"].text()
        assert "C:/models/job.bangmodel" in monitor.main_ai_labels["saved"].text()
        assert monitor.truncated.styleSheet() == ""  # 0% truncated: no warning
        assert monitor.main_ai_warning.isHidden()
        assert monitor.progress_bar.maximum() == 1000 and monitor.progress_bar.value() == 600
        assert monitor.graph_stack.currentIndex() == monitor._main_ai_graph_index

        if PYQTGRAPH_AVAILABLE:
            names = [name for name, _curve, _color in monitor._active_curves]
            assert names == [
                tr("mainai.curve.train_win"),
                tr("mainai.curve.eval_win"),
                tr("mainai.curve.dummy"),
                tr("mainai.curve.random"),
                tr("mainai.curve.policy_loss"),
                tr("mainai.curve.value_loss"),
                tr("mainai.curve.entropy"),
            ]
            assert monitor._dashed_curves == {tr("mainai.curve.dummy"), tr("mainai.curve.random")}
            assert len(monitor._legend_swatches) == 7
            train_curve = monitor._active_curves[0][1]
            assert list(train_curve.getData()[0]) == [1, 101, 201, 301]  # sampled every 100 games
            eval_curve = monitor._active_curves[1][1]
            assert list(eval_curve.getData()[0]) == [300, 600]
            monitor._highlight(tr("mainai.curve.eval_win"))  # hover must not fail on dashed lines
            monitor._highlight(None)
            monitor._render(_main_ai_state(_metrics()))  # unchanged data: no redraw, no error

        high = _main_ai_state(_metrics(truncated_rate=0.05))
        monitor._render(high)
        assert "e08a00" in monitor.truncated.styleSheet()
        assert not monitor.main_ai_warning.isHidden() and monitor.main_ai_warning.text()

        # Early snapshot: no points, baselines or evaluations yet.
        monitor._render(_main_ai_state({"main_games_completed": 0, "main_games_total": 0}))
        assert tr("mainai.baselines_pending") == monitor.main_ai_labels["baselines"].text()

        # Another workflow hides the main-AI summary again.
        other = _main_ai_state({})
        other.snapshot.mode = "module_training"
        monitor._render(other)
        assert monitor.main_ai_panel.isHidden() and not monitor.validation_summary.isHidden()
        assert monitor.truncated.styleSheet() == ""
    finally:
        monitor._timer.stop()


# --- performance dialog -------------------------------------------------------


@pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")
def test_performance_dialog_lists_main_ai_models_after_module_runs(qapp, tmp_path: Path) -> None:
    from bang_rl.gui.dialogs.module_performance import ModulePerformanceDialog

    _write_intent(tmp_path, "job-a")
    _write_emotion(tmp_path, "job-a")
    model = main_ai_artifacts.SavedMainAiModel(path=tmp_path / "main-job.bangmodel", modified=2000.0)

    def module_summary(run, reports_dir=None):
        return {"job_id": run.job_id, "modified": run.modified_text, "report_path": None, "intent": None, "emotion": None}

    dialog = ModulePerformanceDialog(
        model_root=tmp_path,
        summarize=module_summary,
        list_main_ai=lambda _dirs: [model],
        summarize_main_ai=lambda path: {
            "job_id": "main-job",
            "path": str(path),
            "modified": "2025-01-01 00:00",
            "is_main_ai_training": True,
            "training_games": 777,
            "ppo_updates": 9,
            "eval_games": 250,
            "eval_win_rate": 0.512,
            "normalized_mean": 0.9,
            "baselines": {"dummy": {"win_rate": 0.4, "by_role": {}}, "random": {"win_rate": 0.2, "by_role": {}}},
        },
    )
    assert dialog.list.count() == 2
    assert dialog.resume_button.isEnabled()  # module run selected first
    dialog.list.setCurrentRow(1)
    text = dialog.details.toPlainText()
    assert "main-job" in text and "777" in text and "0.512" in text
    assert not dialog.resume_button.isEnabled()  # a main-AI model cannot be a module resume source
    dialog.list.setCurrentRow(0)
    assert dialog.resume_button.isEnabled()
    dialog._use_for_resume()
    assert dialog.selected_job_id == "job-a"
