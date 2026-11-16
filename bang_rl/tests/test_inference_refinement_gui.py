"""GUI side of inference refinement: setup panel, monitor, evaluation dialog."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bang_rl.config import SystemConfig
from bang_rl.data.game_collection import collect_games
from bang_rl.gui import main_ai_view
from bang_rl.gui.i18n import tr
from bang_rl.gui.qt_compat import QT_AVAILABLE
from bang_rl.gui.training import (
    TRAINING_MODE_DATA_COLLECTION,
    TRAINING_MODE_INFERENCE_REFINEMENT,
    TRAINING_MODE_MODULE_TRAINING,
    TrainingMonitorSnapshot,
    TrainingSetup,
    public_training_metadata,
)
from bang_rl.gui.dialogs.inference_evaluation import InferenceEvaluationJob
from bang_rl.tests.test_inference_refinement import _policy, _save_base_model
from bang_rl.tests.test_main_ai_training import _write_phase2
from bang_rl.training.final_inference_artifact import load_final_inference_artifact
from bang_rl.training.main_ai_collection import collect_main_ai_games, policy_from_artifact

if QT_AVAILABLE:
    from bang_rl.gui.qt_compat import QApplication

    _APP = QApplication.instance() or QApplication([])

needs_qt = pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")


# --- fixtures -------------------------------------------------------------------


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("refine-gui")
    model_root = root / "models"
    main_ai_dir = model_root / "main_ai"
    main_ai_dir.mkdir(parents=True)
    model = _save_base_model(main_ai_dir / "good.bangmodel")
    (main_ai_dir / "broken.bangmodel").write_bytes(b"not a model")
    canonical = root / "canonical"
    dummy = canonical / "dummy-ds"
    collect_games(SystemConfig(), 3, dummy, base_seed=7, episode_id_prefix="gui-d5")
    main = canonical / "main-ds"
    collect_main_ai_games(
        SystemConfig(), policy_from_artifact(load_final_inference_artifact(model)), 2, main, base_seed=3, source_model_path=model
    )
    return SimpleNamespace(root=root, model_root=model_root, main_ai_dir=main_ai_dir, model=model, canonical=canonical, dummy=dummy, main=main)


def _panel(workspace: SimpleNamespace):
    from bang_rl.gui.screens.training_monitor import TrainingSetupPanel

    return TrainingSetupPanel(
        config=SystemConfig(),
        model_root=workspace.model_root,
        main_ai_dirs=[workspace.main_ai_dir],
        dataset_root=workspace.canonical,
    )


def _select_mode(panel, mode: str) -> None:
    panel.mode.setCurrentIndex(panel.mode.findData(mode))


# --- pure view helpers ------------------------------------------------------------


def _history() -> list[dict]:
    return [
        {"round": 0, "accepted": True, "reason": "base_reference",
         "metrics": {"hidden_role_accuracy": 0.5, "role_accuracy": 0.6, "emotion_mse": 0.05, "win_rate": 0.4}},
        {"round": 1, "accepted": True, "reason": "improved_both",
         "metrics": {"hidden_role_accuracy": 0.55, "role_accuracy": 0.62, "emotion_mse": 0.04, "win_rate": 0.41}},
        {"round": 2, "accepted": False, "reason": "no_meaningful_improvement",
         "metrics": {"hidden_role_accuracy": 0.551, "role_accuracy": None, "emotion_mse": 0.0399, "win_rate": 0.4}},
    ]


def _refine_metrics(**extra) -> dict:
    metrics = {
        "refine_total_rounds": 5,
        "refine_round": 2,
        "refine_best_round": 1,
        "refine_stage": "refine_training",
        "refine_history": _history(),
        "refine_hidden_role_accuracy_points": [0.5, 0.55, 0.551],
        "refine_role_accuracy_points": [0.6, 0.62, None],
        "refine_emotion_mse_points": [0.05, 0.04, 0.0399],
        "refine_win_rate_points": [0.4, 0.41, 0.4],
        "refine_accepted_points": [1, 1, 0],
        "episodes_completed": 400,
    }
    metrics.update(extra)
    return metrics


def test_refinement_summary_text_lists_each_round_with_reason() -> None:
    texts = main_ai_view.format_refinement_summary(_refine_metrics(stop_reason="patience"), artifact_path="C:/m/x.bangmodel")
    assert "2" in texts["progress"] and "5" in texts["progress"] and "400" in texts["progress"]
    assert "0.500" in texts["base"] and "0.0500" in texts["base"]
    assert "1" in texts["best"] and "0.550" in texts["best"]
    lines = texts["rounds"].splitlines()
    assert tr("refine.verdict.accepted") in lines[0] and tr("refine.reason.improved_both") in lines[0]
    assert tr("refine.verdict.rejected") in lines[1] and tr("refine.reason.no_meaningful_improvement") in lines[1]
    assert lines[-1] == tr("refine.stop.patience")
    assert "C:/m/x.bangmodel" in texts["saved"] and texts["rule"] == tr("refine.rule")
    assert main_ai_view.format_refinement_summary({}, artifact_path=None)["saved"] == tr("refine.not_saved")


def test_refinement_series_keep_unmeasured_rounds_in_place() -> None:
    series = main_ai_view.refine_series(_refine_metrics())
    assert series["role_accuracy"] == [0.6, 0.62, None]
    assert len(series["hidden_role_accuracy"]) == len(series["emotion_mse"]) == len(series["win_rate"]) == 3
    # The public-metrics filter must not shift later points when one is None.
    public = public_training_metadata(_refine_metrics())
    assert main_ai_view.refine_series(public)["role_accuracy"] == [0.6, 0.62, None]


def test_inference_rows_have_one_cell_per_column() -> None:
    rows = main_ai_view.inference_rows(
        [
            {"label": "base", "hidden_role_accuracy": 0.5, "role_accuracy": 0.6, "emotion_mse": 0.05, "win_rate": 0.4,
             "role_accuracy_by_target_role": {"sheriff": 1.0, "deputy": None, "outlaw": 0.5, "traitor": 0.0}},
            {"label": "pair-b", "hidden_role_accuracy": None},
        ]
    )
    assert [len(row) for row in rows] == [6, 6]
    assert rows[0][0] == tr("infer.base_label") and rows[1][0] == "pair-b"
    assert "1.00" in rows[0][5] and rows[1][1] == "—"


def test_training_setup_carries_the_new_fields_into_the_request() -> None:
    setup = TrainingSetup(
        config=SystemConfig(),
        mode=TRAINING_MODE_INFERENCE_REFINEMENT,
        episodes=40,
        collection_players="main1-dummy4",
        main_action_mode="stochastic",
        base_model_path="x.bangmodel",
        dataset_paths=("a", "b"),
        refine_rounds=3,
        refine_epochs=2,
        refine_patience=2,
        refine_eval_games=30,
    )
    request = setup.to_request()
    assert request.mode == "inference_refinement" and request.episodes == 40
    assert (request.collection_players, request.main_action_mode, request.base_model_path) == (
        "main1-dummy4", "stochastic", "x.bangmodel",
    )
    assert tuple(request.dataset_paths) == ("a", "b")
    assert (request.refine_rounds, request.refine_epochs, request.refine_patience, request.refine_eval_games) == (3, 2, 2, 30)


# --- setup panel ------------------------------------------------------------------


@needs_qt
def test_panel_builds_the_refinement_request(workspace: SimpleNamespace) -> None:
    panel = _panel(workspace)
    _select_mode(panel, TRAINING_MODE_INFERENCE_REFINEMENT)
    # Refinement inputs are shown; unrelated ones are hidden.
    for field in (panel.refine_games, panel.refine_rounds, panel.refine_epochs, panel.refine_patience,
                  panel.refine_eval_games, panel.base_model_row, panel.action_mode, panel.datasets_row):
        assert not field.isHidden()
    for field in (panel.episodes, panel.epochs, panel.games, panel.players, panel.resume_row, panel.graph_interval):
        assert field.isHidden()
    assert panel.refine_games.value() == 5000 and panel.refine_rounds.value() == 5
    assert panel.refine_epochs.value() == 5 and panel.refine_patience.value() == 1
    assert panel.refine_eval_games.value() == 250

    # The compatible model is listed and selected; the unreadable one is disabled.
    states = {panel.base_model.itemData(i): panel.base_model.model().item(i).isEnabled() for i in range(panel.base_model.count())}
    assert states[str(workspace.model)] is True and states[str(workspace.main_ai_dir / "broken.bangmodel")] is False
    assert panel.base_model.currentData() == str(workspace.model)

    # Datasets: readable labels with players and games; nothing checked = all.
    labels = [panel.datasets.item(i).text() for i in range(panel.datasets.count())]
    assert any("dummy5" in text and "3판" in text for text in labels)
    assert any("main1-dummy4" in text and "2판" in text for text in labels)
    assert panel.setup().dataset_paths is None
    from bang_rl.gui.qt_compat import Qt

    for index in range(panel.datasets.count()):
        if str(workspace.dummy) == panel.datasets.item(index).data(Qt.ItemDataRole.UserRole):
            panel.datasets.item(index).setCheckState(Qt.CheckState.Checked)

    panel.refine_games.setValue(40)
    panel.refine_rounds.setValue(3)
    panel.refine_epochs.setValue(2)
    panel.refine_patience.setValue(2)
    panel.refine_eval_games.setValue(30)
    panel.action_mode.setCurrentIndex(panel.action_mode.findData("stochastic"))
    panel.max_minutes.setText("90")
    setup = panel.setup()
    assert setup.mode == "inference_refinement" and setup.episodes == 40
    assert (setup.refine_rounds, setup.refine_epochs, setup.refine_patience, setup.refine_eval_games) == (3, 2, 2, 30)
    assert setup.base_model_path == str(workspace.model) and setup.main_action_mode == "stochastic"
    assert setup.dataset_paths == (str(workspace.dummy),)
    assert setup.resume_intent_artifact_path is None and setup.resume_emotion_artifact_path is None
    assert setup.resume_artifact_path is None
    assert setup.train_by_time is True and setup.max_training_seconds == 90 * 60.0
    assert setup.to_request().refine_rounds == 3


@needs_qt
def test_panel_data_collection_offers_main1_dummy4_with_a_model(workspace: SimpleNamespace) -> None:
    panel = _panel(workspace)
    _select_mode(panel, TRAINING_MODE_DATA_COLLECTION)
    assert not panel.players.isHidden()
    assert panel.base_model_row.isHidden() and panel.action_mode.isHidden()  # dummy5 needs no model
    setup = panel.setup()
    assert setup.collection_players == "dummy5" and setup.base_model_path is None

    panel.players.setCurrentIndex(panel.players.findData("main1-dummy4"))
    assert not panel.base_model_row.isHidden() and not panel.action_mode.isHidden()
    assert panel.datasets_row.isHidden()
    setup = panel.setup()
    assert setup.mode == "data_collection" and setup.collection_players == "main1-dummy4"
    assert setup.base_model_path == str(workspace.model) and setup.main_action_mode == "deterministic"
    assert setup.dataset_paths is None


@needs_qt
def test_panel_module_training_selects_datasets(workspace: SimpleNamespace) -> None:
    from bang_rl.gui.qt_compat import Qt

    panel = _panel(workspace)
    _select_mode(panel, TRAINING_MODE_MODULE_TRAINING)
    assert not panel.datasets_row.isHidden() and panel.base_model_row.isHidden()
    assert panel.setup().dataset_paths is None
    for index in range(panel.datasets.count()):
        panel.datasets.item(index).setCheckState(Qt.CheckState.Checked)
    assert set(panel.setup().dataset_paths) == {str(workspace.dummy), str(workspace.main)}
    # Refreshing keeps the checks.
    panel.refresh_datasets()
    assert len(panel.checked_dataset_paths()) == 2


# --- monitor ----------------------------------------------------------------------


def _refine_state(metrics: dict, *, status: str = "running", artifact: str | None = None):
    progress = SimpleNamespace(
        status=status, ppo_updates=0, episodes_completed=400, environment_steps=9000, truncated_episodes=0,
        truncated_rate=0.0, metrics=metrics, report_path=None, output_path=None, artifact_path=artifact,
        main_agent_transitions=0, win_rate=0.4,
    )
    snapshot = SimpleNamespace(
        mode="inference_refinement", job_id="job-refine", config_version="1", seed=3, config_snapshot={}
    )
    return TrainingMonitorSnapshot(handle=None, progress=progress, snapshot=snapshot, logs=("stage message",))


@needs_qt
def test_monitor_renders_a_refinement_snapshot(workspace: SimpleNamespace) -> None:
    from bang_rl.gui.screens.training_monitor import PYQTGRAPH_AVAILABLE, TrainingMonitor

    monitor = TrainingMonitor(setup_panel=_panel(workspace))
    try:
        monitor._render(_refine_state(public_training_metadata(_refine_metrics()), artifact="C:/m/refined.bangmodel"))
        assert not monitor.refine_panel.isHidden()
        assert monitor.main_ai_panel.isHidden() and monitor.validation_summary.isHidden()
        assert tr("refine.verdict.rejected") in monitor.refine_labels["rounds"].text()
        assert "C:/m/refined.bangmodel" in monitor.refine_labels["saved"].text()
        assert monitor.progress_bar.maximum() == 5 and monitor.progress_bar.value() == 2  # finished rounds
        assert monitor.graph_stack.currentIndex() == monitor._refine_graph_index
        if PYQTGRAPH_AVAILABLE:
            names = [name for name, _curve, _color in monitor._active_curves]
            assert names == [
                tr("refine.curve.hidden"), tr("refine.curve.role"), tr("refine.curve.win"), tr("refine.curve.mse"),
            ]
            assert monitor._dashed_curves == {tr("refine.curve.win")}  # win rate is the secondary line
            hidden_curve = monitor._active_curves[0][1]
            assert list(hidden_curve.getData()[0]) == [0, 1, 2]
            role_curve = monitor._active_curves[1][1]
            assert list(role_curve.getData()[0]) == [0, 1]  # the unmeasured round is skipped, not shifted
            monitor._highlight(tr("refine.curve.win"))
            monitor._highlight(None)
            monitor._render(_refine_state(public_training_metadata(_refine_metrics())))  # no change: no redraw

        # Another workflow hides the refinement summary again.
        other = _refine_state({})
        other.snapshot.mode = "module_training"
        monitor._render(other)
        assert monitor.refine_panel.isHidden() and not monitor.validation_summary.isHidden()
    finally:
        monitor._timer.stop()


@needs_qt
def test_refinement_outputs_can_be_saved(tmp_path: Path) -> None:
    from bang_rl.saved_artifacts import plans_for_training_result

    model = _save_base_model(tmp_path / "refined.bangmodel")
    plans = plans_for_training_result("inference_refinement", artifact_path=str(model), saved_root=tmp_path / "saved")
    assert [plan.category for plan in plans] == ["main_ai"]


# --- evaluation job and dialog ---------------------------------------------------------


def _pair_files(root: Path) -> tuple[Path, Path]:
    pair = _policy(seed=9)
    intent_path = root / "intent" / "intent-pair-a.phase2model"
    emotion_path = root / "emotion" / "emotion-pair-a.phase2model"
    _write_phase2(
        intent_path, pair.intent, "phase2_intent_inference",
        {"architecture": {"role_count": 4}, "training": {"total_epochs": 1}},
    )
    _write_phase2(
        emotion_path, pair.emotion, "phase2_emotion_inference",
        {
            "emotion_schema": {"version": pair.emotion.emotion_schema_version, "count": pair.emotion.emotion_count},
            "training": {"total_epochs": 1},
        },
    )
    return intent_path, emotion_path


def test_evaluation_job_runs_base_and_pairs_on_a_thread_and_can_stop(tmp_path: Path) -> None:
    model = _save_base_model(tmp_path / "m.bangmodel")
    intent_path, emotion_path = _pair_files(tmp_path)
    job = InferenceEvaluationJob(model, [("other", str(intent_path), str(emotion_path))], games=3)
    job.start()
    job.join(300)
    status, done, total, _current, rows, error = job.snapshot()
    assert (status, done, total, error) == ("finished", 2, 2, None)
    assert [row["label"] for row in rows] == ["base", "other"]
    assert all(row["hidden_role_accuracy"] is None or 0.0 <= row["hidden_role_accuracy"] <= 1.0 for row in rows)

    stopped = InferenceEvaluationJob(model, [], games=50)
    stopped.stop()
    stopped.start()
    stopped.join(300)
    assert stopped.snapshot()[0] == "stopped" and not stopped.rows

    broken = InferenceEvaluationJob(tmp_path / "missing.bangmodel", [], games=2)
    broken.start()
    broken.join(60)
    assert broken.snapshot()[0] == "failed" and broken.error


@needs_qt
def test_evaluation_dialog_lists_models_and_pairs_and_fills_the_table(workspace: SimpleNamespace, tmp_path: Path) -> None:
    from bang_rl.gui.dialogs.inference_evaluation import InferenceEvaluationDialog
    from bang_rl.gui.qt_compat import Qt

    _pair_files(workspace.model_root)
    dialog = InferenceEvaluationDialog(
        model_root=workspace.model_root, main_ai_dirs=[workspace.main_ai_dir]
    )
    try:
        assert dialog.model.count() == 1  # the unreadable model is not offered
        assert dialog.pairs.count() == 1
        dialog.include_base.setChecked(False)
        dialog.run()  # nothing chosen: a message, no job
        assert dialog.status.text() == tr("infer.nothing") and dialog._job is None
        dialog.include_base.setChecked(True)
        dialog.pairs.item(0).setCheckState(Qt.CheckState.Checked)
        dialog.games.setValue(3)
        dialog.run()
        dialog._job.join(300)
        dialog._poll()
        assert dialog.table.rowCount() == 2 and dialog.table.columnCount() == 6
        assert dialog.status.text() == tr("infer.finished")
        assert dialog.run_button.isEnabled()
    finally:
        dialog._timer.stop()
