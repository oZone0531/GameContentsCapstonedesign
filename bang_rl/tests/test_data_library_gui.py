"""Qt tests of the data library screen and of applying its items to the training form."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bang_rl import data_library as lib
from bang_rl.artifact_info import read_info
from bang_rl.config import SystemConfig
from bang_rl.gui import library_view as view
from bang_rl.gui.i18n import tr
from bang_rl.gui.qt_compat import QT_AVAILABLE
from bang_rl.gui.training import (
    PresetResult,
    TrainingMonitorController,
    TrainingPreset,
    controller_status,
    training_is_active,
)
from bang_rl.tests.test_data_library import (  # noqa: F401 - fixtures
    DATASET_ID,
    MODEL_ID,
    RUN_BAD,
    RUN_GOOD,
    _dataset,
    _report,
    bangmodel,
    stored,
)

if QT_AVAILABLE:
    from bang_rl.gui.qt_compat import QApplication, QUrl, Qt

    _APP = QApplication.instance() or QApplication([])

needs_qt = pytest.mark.skipif(not QT_AVAILABLE, reason="Qt is not available")


def _library(stored: SimpleNamespace) -> lib.Library:
    return lib.build_library(stored.loc)


def _entry(library: lib.Library, kind: str, id: str | None = None) -> lib.LibraryEntry:
    return next(e for e in library.entries if e.kind == kind and (id is None or e.id == id))


def _dialog(stored: SimpleNamespace, **kwargs):
    from bang_rl.gui.dialogs.data_library import DataLibraryDialog

    return DataLibraryDialog(locations=stored.loc, **kwargs)


def _panel(stored: SimpleNamespace):
    from bang_rl.gui.screens.training_monitor import TrainingSetupPanel

    return TrainingSetupPanel(
        config=SystemConfig(),
        model_root=stored.loc.model_root,
        main_ai_dirs=list(stored.loc.main_ai_dirs),
        dataset_root=stored.loc.canonical_root,
    )


class _Service:
    """Training service stub that only reports a status."""

    def __init__(self, status: str = "running") -> None:
        self.status = status

    def progress(self, _handle):
        return SimpleNamespace(status=self.status)


def _busy_controller(status: str = "running") -> TrainingMonitorController:
    controller = TrainingMonitorController(_Service(status))
    controller.handle = object()
    return controller


# --- the dialog -------------------------------------------------------------------------------


@needs_qt
def test_dialog_lists_every_entry_newest_first(stored: SimpleNamespace) -> None:
    dialog = _dialog(stored)
    assert dialog.table.rowCount() == len(dialog.library.entries) == 6
    assert dialog.table.columnCount() == len(view.TABLE_COLUMNS)
    first = dialog.table.item(0, 0)
    assert first.text() == tr("library.kind.replay") and first.data(Qt.ItemDataRole.UserRole) == dialog.visible_entries[0].key
    last_row = dialog.table.rowCount() - 1
    assert dialog.table.item(last_row, 0).text() == tr("library.kind.dataset")
    assert dialog.table.item(last_row, 6).text() == "첫 번째 데이터"  # first line of the note
    assert dialog.count_label.text() == tr("library.count", shown=6, total=6)
    assert dialog.empty_label.isHidden() and dialog.last_build_seconds >= 0


@needs_qt
def test_empty_library_shows_the_hint(tmp_path: Path) -> None:
    from bang_rl.gui.dialogs.data_library import DataLibraryDialog

    dialog = DataLibraryDialog(locations=lib.LibraryLocations.under(tmp_path))
    assert dialog.table.rowCount() == 0 and not dialog.empty_label.isHidden()
    assert dialog.selected_entries() == []
    assert all(not button.isEnabled() for button in dialog.buttons.values())


@needs_qt
def test_selecting_a_row_shows_details_and_per_kind_buttons(stored: SimpleNamespace) -> None:
    dialog = _dialog(stored)
    library = dialog.library
    model = _entry(library, "main_ai")
    assert dialog.select_entry(model.key)
    assert [entry.key for entry in dialog.selected_entries()] == [model.key]
    html = dialog.details.toHtml()
    assert model.name in html and tr("library.detail.performance") in html
    plain = dialog.details.toPlainText()
    assert "120" in plain and model.id in plain  # training games and id appear
    for action in (view.ACTION_PLAY, view.ACTION_CONTINUE_PPO, view.ACTION_REFINE_BASE, view.ACTION_COLLECT,
                   view.ACTION_EVALUATE, view.ACTION_EDIT_NOTES, view.ACTION_SAVE_COPY, view.ACTION_OPEN_FOLDER):
        assert not dialog.buttons[action].isHidden() and dialog.buttons[action].isEnabled(), action
    for action in (view.ACTION_MODULE_TRAINING_DATA, view.ACTION_VIEW_REPLAY, view.ACTION_MAIN_AI_TRAIN):
        assert dialog.buttons[action].isHidden(), action

    dialog.select_entry(_entry(library, "module_pair", RUN_BAD).key)
    button = dialog.buttons[view.ACTION_MAIN_AI_TRAIN]
    assert not button.isHidden() and not button.isEnabled()
    assert tr("training.resume_module_reason.intent_roles") in button.toolTip()
    assert dialog.buttons[view.ACTION_RESUME_MODULES].isEnabled()

    dialog.select_entry(_entry(library, "replay").key)
    assert dialog.buttons[view.ACTION_VIEW_REPLAY].isEnabled() and not dialog.buttons[view.ACTION_SAVE_COPY].isEnabled()
    dialog.select_entry(_entry(library, "report").key)
    assert dialog.buttons[view.ACTION_OPEN_REPORT].isEnabled()


@needs_qt
def test_filters_narrow_the_table(stored: SimpleNamespace) -> None:
    dialog = _dialog(stored)
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("module_pair"))
    assert dialog.table.rowCount() == 2
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("models"))
    assert dialog.table.rowCount() == 3
    dialog.kind_combo.setCurrentIndex(0)
    dialog.location_combo.setCurrentIndex(dialog.location_combo.findData("saved"))
    assert dialog.table.rowCount() == 0
    dialog.location_combo.setCurrentIndex(0)
    dialog.search.setText("좋은 모델")
    assert dialog.table.rowCount() == 1 and dialog.visible_entries[0].kind == "main_ai"
    assert dialog.count_label.text() == tr("library.count", shown=1, total=6)
    dialog.search.setText("")
    assert dialog.table.rowCount() == 6


@needs_qt
def test_performance_entry_point_filters_to_saved_models(stored: SimpleNamespace) -> None:
    dialog = _dialog(stored, kind_filter=view.KIND_FILTER_MODELS)
    assert dialog.kind_combo.currentData() == "models" and dialog.table.rowCount() == 3
    assert {entry.kind for entry in dialog.visible_entries} == {"module_pair", "main_ai"}


@needs_qt
def test_links_select_the_linked_entry_and_reset_hiding_filters(stored: SimpleNamespace) -> None:
    dialog = _dialog(stored)
    model = _entry(dialog.library, "main_ai")
    good = _entry(dialog.library, "module_pair", RUN_GOOD)
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("main_ai"))
    dialog.select_entry(model.key)
    assert f'entry:{good.key}' in dialog.details.toHtml()
    dialog._on_link(QUrl(f"entry:{good.key}"))  # the module pair is hidden by the kind filter
    assert dialog.kind_combo.currentData() == view.KIND_FILTER_ALL
    assert [entry.key for entry in dialog.selected_entries()] == [good.key]
    assert model.name in dialog.details.toPlainText()  # "used by" lists the model
    assert not dialog.select_entry("no-such-key")


@needs_qt
def test_apply_actions_hand_the_choice_back_and_close(stored: SimpleNamespace) -> None:
    dialog = _dialog(stored)
    good = _entry(dialog.library, "module_pair", RUN_GOOD)
    dialog.select_entry(good.key)
    assert dialog.trigger(view.ACTION_MAIN_AI_TRAIN)
    assert dialog.chosen == view.LibraryChoice(view.ACTION_MAIN_AI_TRAIN, (good,))

    blocked = _dialog(stored)
    blocked.select_entry(_entry(blocked.library, "module_pair", RUN_BAD).key)
    assert not blocked.trigger(view.ACTION_MAIN_AI_TRAIN) and blocked.chosen is None
    assert blocked.message.text()  # the reason is shown


@needs_qt
def test_several_datasets_can_be_applied_together(stored: SimpleNamespace) -> None:
    _dataset(stored.loc.canonical_root / "ds-two", id="d2222222-second", games=7)
    dialog = _dialog(stored)
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("dataset"))
    dialog.table.selectAll()
    assert len(dialog.selected_entries()) == 2
    assert dialog.buttons[view.ACTION_MODULE_TRAINING_DATA].isEnabled() and not dialog.buttons[view.ACTION_EDIT_NOTES].isEnabled()
    assert dialog.trigger(view.ACTION_REFINEMENT_DATA)
    assert {entry.id for entry in dialog.chosen.entries} == {DATASET_ID, "d2222222-second"}


@needs_qt
def test_busy_training_blocks_apply_but_not_other_actions(stored: SimpleNamespace, monkeypatch) -> None:
    dialog = _dialog(stored, training_busy=lambda: True)
    dataset = _entry(dialog.library, "dataset")
    dialog.select_entry(dataset.key)
    assert not dialog.trigger(view.ACTION_MODULE_TRAINING_DATA)
    assert dialog.chosen is None and dialog.message.text() == tr("library.busy")
    opened: list[Path] = []
    monkeypatch.setattr(dialog, "_open_path", lambda path: opened.append(path) or True)
    assert dialog.trigger(view.ACTION_OPEN_FOLDER) and opened == [stored.dataset]


@needs_qt
def test_double_click_runs_the_usual_action_per_kind(stored: SimpleNamespace, monkeypatch) -> None:
    from bang_rl.gui.dialogs import data_library as module

    expected = {
        ("dataset", None): view.ACTION_MODULE_TRAINING_DATA,
        ("module_pair", RUN_GOOD): view.ACTION_MAIN_AI_TRAIN,
        ("module_pair", RUN_BAD): view.ACTION_RESUME_MODULES,
        ("main_ai", None): view.ACTION_PLAY,
        ("replay", None): view.ACTION_VIEW_REPLAY,
    }
    for (kind, identifier), action in expected.items():
        dialog = _dialog(stored)
        monkeypatch.setattr(dialog, "_choose_game_mode", lambda _entry: module.PLAY_MODE_SPECTATE)
        entry = _entry(dialog.library, kind, identifier)
        dialog.select_entry(entry.key)
        row = dialog.visible_entries.index(entry)
        dialog.table.cellDoubleClicked.emit(row, 1)
        assert dialog.chosen is not None and dialog.chosen.action == action, kind
        assert dialog.chosen.entries == (entry,)
        if action == view.ACTION_PLAY:
            assert dialog.chosen.params == {"mode": module.PLAY_MODE_SPECTATE}

    # The report is shown inside the library instead of leaving it.
    shown: list[lib.LibraryEntry] = []

    class _Viewer:
        def __init__(self, entry, _parent=None, **_kwargs) -> None:
            shown.append(entry)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(module, "ReportViewDialog", _Viewer)
    dialog = _dialog(stored)
    report = _entry(dialog.library, "report")
    dialog.select_entry(report.key)
    assert dialog.activate_default() and shown == [report] and dialog.chosen is None


@needs_qt
def test_play_offers_two_modes_and_can_be_cancelled(stored: SimpleNamespace, monkeypatch) -> None:
    from bang_rl.gui.dialogs import data_library as module

    dialog = _dialog(stored)
    dialog.select_entry(_entry(dialog.library, "main_ai").key)
    answers = iter([module.PLAY_MODE_HUMAN, None])
    monkeypatch.setattr(dialog, "_choose_game_mode", lambda _entry: next(answers))
    assert dialog.trigger(view.ACTION_PLAY) and dialog.chosen.params == {"mode": "human_main_dummy"}
    again = _dialog(stored)
    again.select_entry(_entry(again.library, "main_ai").key)
    monkeypatch.setattr(again, "_choose_game_mode", lambda _entry: None)
    assert not again.trigger(view.ACTION_PLAY) and again.chosen is None


@needs_qt
def test_notes_editing_persists_to_every_file_of_the_entry(stored: SimpleNamespace) -> None:
    dialog = _dialog(stored)
    for kind, identifier in (("module_pair", RUN_GOOD), ("dataset", None), ("main_ai", None), ("report", None), ("replay", None)):
        entry = _entry(dialog.library, kind, identifier)
        assert dialog.set_notes(entry, f"새 비고 {kind}\n둘째 줄") == len(entry.info_paths)
        for path in entry.info_paths:
            assert read_info(path).notes == f"새 비고 {kind}\n둘째 줄"
        refreshed = _entry(dialog.library, kind, identifier)
        assert refreshed.notes == f"새 비고 {kind}\n둘째 줄" and refreshed.first_note_line == f"새 비고 {kind}"
    # The edit is visible in the table without reopening the library.
    row = dialog.visible_entries.index(_entry(dialog.library, "dataset"))
    assert dialog.table.item(row, 6).text() == "새 비고 dataset"


@needs_qt
def test_open_folder_evaluate_and_save_use_the_existing_helpers(stored: SimpleNamespace, monkeypatch) -> None:
    from bang_rl.gui.dialogs import inference_evaluation, save_outputs

    dialog = _dialog(stored)
    opened: list[Path] = []
    monkeypatch.setattr(dialog, "_open_path", lambda path: opened.append(path) or True)
    model = _entry(dialog.library, "main_ai")
    dialog.select_entry(model.key)
    assert dialog.trigger(view.ACTION_OPEN_FOLDER) and opened == [model.path.parent]

    created: list[dict] = []

    class _Evaluation:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(inference_evaluation, "InferenceEvaluationDialog", _Evaluation)
    assert dialog.trigger(view.ACTION_EVALUATE) and created[-1]["select_model"] == model.path
    good = _entry(dialog.library, "module_pair", RUN_GOOD)
    dialog.select_entry(good.key)
    assert dialog.trigger(view.ACTION_EVALUATE) and created[-1]["select_pair"] == (good.intent_path, good.emotion_path)

    plans: list = []

    class _Save:
        def __init__(self, plan, _parent=None) -> None:
            plans.append(plan)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(save_outputs, "SaveOutputsDialog", _Save)
    assert dialog.trigger(view.ACTION_SAVE_COPY) and plans[0].category == "modules"
    assert [item.source for item in plans[0].items][:2] == [good.intent_path, good.emotion_path]


@needs_qt
def test_library_refreshes_when_a_training_job_ends(stored: SimpleNamespace) -> None:
    statuses = ["running"]
    dialog = _dialog(stored, status_provider=lambda: statuses[-1])
    assert dialog.table.rowCount() == 6 and dialog._timer.isActive()
    _report(stored.loc.report_dirs[0] / "new-report.json", "z9999999-new", created_at="2026-11-14T10:00:00+09:00")
    dialog._poll_status()  # still running: nothing changes
    assert dialog.table.rowCount() == 6
    statuses.append("finished")
    dialog._poll_status()
    assert dialog.table.rowCount() == 7 and dialog.visible_entries[0].id == "z9999999-new"
    dialog._poll_status()  # no new transition
    dialog.reject()
    assert not dialog._timer.isActive()


@needs_qt
def test_refresh_keeps_filters_and_selection(stored: SimpleNamespace) -> None:
    dialog = _dialog(stored)
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData("report"))
    report = _entry(dialog.library, "report")
    dialog.select_entry(report.key)
    dialog.refresh()
    assert dialog.table.rowCount() == 1 and [e.key for e in dialog.selected_entries()] == [report.key]


# --- applying a preset to the training form ----------------------------------------------------------


@needs_qt
def test_preset_module_training_checks_exactly_the_given_datasets(stored: SimpleNamespace) -> None:
    second = _dataset(stored.loc.canonical_root / "ds-two", id="d2222222-second", games=7)
    panel = _panel(stored)
    result = panel.apply_preset(TrainingPreset("module_training", dataset_paths=(str(stored.dataset),)))
    assert result.ok and tr("library.applied_datasets", count=1) in result.message
    setup = panel.setup()
    assert setup.mode == "module_training" and setup.dataset_paths == (str(stored.dataset),)
    panel.apply_preset(TrainingPreset("module_training", dataset_paths=(str(second), str(stored.dataset))))
    assert set(panel.setup().dataset_paths) == {str(stored.dataset), str(second)}
    panel.apply_preset(TrainingPreset("module_training", dataset_paths=(str(second),)))
    assert panel.setup().dataset_paths == (str(second),)  # the earlier one is unchecked again


@needs_qt
def test_preset_adds_a_dataset_outside_the_listed_folder(stored: SimpleNamespace, tmp_path: Path) -> None:
    elsewhere = _dataset(tmp_path / "elsewhere" / "kept", id="d3333333-elsewhere", games=3)
    panel = _panel(stored)
    result = panel.apply_preset(TrainingPreset("module_training", dataset_paths=(str(elsewhere),)))
    assert result.ok and panel.setup().dataset_paths == (str(elsewhere),)
    gone = panel.apply_preset(TrainingPreset("module_training", dataset_paths=(str(tmp_path / "missing"),)))
    assert not gone.ok and gone.problems and panel.setup().dataset_paths is None  # nothing checked = all datasets


@needs_qt
def test_preset_resume_modules_selects_the_pair(stored: SimpleNamespace) -> None:
    panel = _panel(stored)
    result = panel.apply_preset(TrainingPreset("module_training", module_run_id=RUN_GOOD))
    assert result.ok
    setup = panel.setup()
    assert setup.mode == "module_training"
    assert (setup.resume_intent_artifact_path, setup.resume_emotion_artifact_path) == tuple(str(p) for p in stored.good)


@needs_qt
def test_preset_main_ai_pair_overrides_a_typed_bangmodel(stored: SimpleNamespace) -> None:
    panel = _panel(stored)
    panel.apply_preset(TrainingPreset("main_ai_training", bangmodel_path=str(stored.model)))
    assert panel.setup().resume_artifact_path == str(stored.model) and panel.setup().resume_intent_artifact_path is None
    result = panel.apply_preset(TrainingPreset("main_ai_training", module_run_id=RUN_GOOD))
    assert result.ok
    setup = panel.setup()
    assert setup.mode == "main_ai_training" and setup.resume_artifact_path is None
    assert (setup.resume_intent_artifact_path, setup.resume_emotion_artifact_path) == tuple(str(p) for p in stored.good)


@needs_qt
def test_preset_rejects_an_incompatible_pair_for_main_ai(stored: SimpleNamespace) -> None:
    panel = _panel(stored)
    result = panel.apply_preset(TrainingPreset("main_ai_training", module_run_id=RUN_BAD))
    assert not result.ok and tr("library.apply_module_unusable", job=RUN_BAD) in result.message
    assert panel.setup().mode == "main_ai_training"  # the mode was still switched


@needs_qt
def test_preset_continue_ppo_sets_the_model_file(stored: SimpleNamespace) -> None:
    panel = _panel(stored)
    result = panel.apply_preset(TrainingPreset("main_ai_training", bangmodel_path=str(stored.model)))
    assert result.ok and panel.setup().resume_artifact_path == str(stored.model)
    missing = panel.apply_preset(TrainingPreset("main_ai_training", bangmodel_path=str(stored.root / "none.bangmodel")))
    assert not missing.ok and missing.problems


@needs_qt
def test_preset_refinement_base_model_and_datasets(stored: SimpleNamespace) -> None:
    panel = _panel(stored)
    result = panel.apply_preset(TrainingPreset("inference_refinement", dataset_paths=(str(stored.dataset),),
                                               bangmodel_path=str(stored.model)))
    assert result.ok
    setup = panel.setup()
    assert setup.mode == "inference_refinement" and setup.base_model_path == str(stored.model)
    assert setup.dataset_paths == (str(stored.dataset),)


@needs_qt
def test_preset_collection_with_main_ai_and_four_dummies(stored: SimpleNamespace) -> None:
    panel = _panel(stored)
    result = panel.apply_preset(TrainingPreset("data_collection", players="main1-dummy4", bangmodel_path=str(stored.model)))
    assert result.ok
    setup = panel.setup()
    assert setup.mode == "data_collection" and setup.collection_players == "main1-dummy4"
    assert setup.base_model_path == str(stored.model)
    assert not panel.base_model_row.isHidden()


@needs_qt
def test_preset_with_unknown_mode_changes_nothing(stored: SimpleNamespace) -> None:
    panel = _panel(stored)
    before = panel.setup().mode
    result = panel.apply_preset(TrainingPreset("no_such_mode"))
    assert isinstance(result, PresetResult) and not result.ok and panel.setup().mode == before


# --- MainWindow and TrainingMonitor wiring ---------------------------------------------------------------


@pytest.fixture
def window(stored: SimpleNamespace, monkeypatch):
    from bang_rl.gui.dialogs import data_library as module
    from bang_rl.gui.main_window import MainWindow

    real_build = module.build_library
    monkeypatch.setattr(module, "build_library", lambda _locations=None, config=None: real_build(stored.loc))
    return MainWindow(training_service=_Service("finished"))


def _choose_in_library(monkeypatch, window, kind: str, identifier: str | None = None, *, mode: str | None = None) -> None:
    """Make the library "open" and double-click the entry of ``kind``."""

    from bang_rl.gui.dialogs import data_library as module

    def fake_exec(self) -> int:
        entry = _entry(self.library, kind, identifier)
        self.select_entry(entry.key)
        if mode is not None:
            monkeypatch.setattr(self, "_choose_game_mode", lambda _entry: mode)
        self.activate_default()
        return 1

    monkeypatch.setattr(module.DataLibraryDialog, "exec", fake_exec)


@needs_qt
def test_menu_has_a_library_button_that_opens_the_library(window, monkeypatch) -> None:
    from bang_rl.gui.dialogs import data_library as module

    opened: list[dict] = []

    class _Fake:
        chosen = None

        def __init__(self, **kwargs) -> None:
            opened.append(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(module, "DataLibraryDialog", _Fake)
    assert window._library_button.text() == tr("menu.library")
    window._library_button.click()
    assert len(opened) == 1 and opened[0]["kind_filter"] is None
    assert opened[0]["training_busy"]() is False and opened[0]["status_provider"]() is None


@needs_qt
def test_double_click_dataset_opens_module_training_with_that_dataset(window, stored, monkeypatch) -> None:
    got: list[TrainingPreset] = []
    monkeypatch.setattr(window, "open_training", lambda preset=None: got.append(preset))
    _choose_in_library(monkeypatch, window, "dataset")
    window._library_button.click()
    assert got == [TrainingPreset("module_training", dataset_paths=(str(stored.dataset),))]


@needs_qt
def test_double_click_module_pair_opens_main_ai_training(window, stored, monkeypatch) -> None:
    got: list[TrainingPreset] = []
    monkeypatch.setattr(window, "open_training", lambda preset=None: got.append(preset))
    _choose_in_library(monkeypatch, window, "module_pair", RUN_GOOD)
    window._library_button.click()
    assert got == [TrainingPreset("main_ai_training", module_run_id=RUN_GOOD)]


@needs_qt
def test_double_click_main_ai_starts_a_game_with_the_chosen_mode(window, stored, monkeypatch) -> None:
    from bang_rl.gui.game import GameMode

    started: list[tuple] = []
    monkeypatch.setattr(window, "_start_game", lambda mode, path: started.append((mode, path)))
    option = SimpleNamespace(path=str(stored.model), compatible=True, metadata={"version": "v-test"})
    monkeypatch.setattr(window, "_selection", SimpleNamespace(discover=lambda paths, config=None: (option,)))
    _choose_in_library(monkeypatch, window, "main_ai", mode="human_main_dummy")
    window._library_button.click()
    assert started == [(GameMode.HUMAN_MAIN_DUMMY, str(stored.model))]
    assert window.controller.ai_metadata == {"version": "v-test"}  # recorded for the saved replay

    started.clear()
    _choose_in_library(monkeypatch, window, "main_ai", mode="main_one_dummy_four")
    window._library_button.click()
    assert started == [(GameMode.MAIN_ONE_DUMMY_FOUR, str(stored.model))]


@needs_qt
def test_incompatible_model_does_not_start_a_game(window, stored, monkeypatch) -> None:
    from bang_rl.gui.game import GameMode

    started: list = []
    monkeypatch.setattr(window, "_start_game", lambda mode, path: started.append((mode, path)))
    option = SimpleNamespace(path=str(stored.model), compatible=False, metadata={})
    monkeypatch.setattr(window, "_selection", SimpleNamespace(discover=lambda paths, config=None: (option,)))
    window._play_with_model(GameMode.MAIN_ONE_DUMMY_FOUR, str(stored.model))
    assert started == [] and window.statusBar().currentMessage()


@needs_qt
def test_double_click_replay_opens_the_replay_screen_on_that_file(window, stored, monkeypatch) -> None:
    got: list = []
    monkeypatch.setattr(window, "_show_replay_file", lambda path: got.append(path))
    _choose_in_library(monkeypatch, window, "replay")
    window._library_button.click()
    assert got == [stored.replay]


@needs_qt
def test_replay_file_really_loads_in_the_replay_screen(window, stored) -> None:
    window._show_replay_file(stored.replay)
    assert window._replay_screen is not None and window._replay_screen._loaded_path == stored.replay


@needs_qt
def test_open_training_applies_the_preset_and_confirms(window, stored) -> None:
    result = window.open_training(TrainingPreset("module_training", dataset_paths=(str(stored.dataset),)))
    assert result is not None and result.ok
    monitor = window._training_monitor
    assert monitor is not None and window.centralWidget() is monitor
    assert monitor.setup_panel.setup().mode == "module_training"
    assert monitor.setup_panel.setup().dataset_paths == (str(stored.dataset),)
    assert monitor.message.text() == result.message and tr("library.applied_datasets", count=1) in result.message
    assert window.statusBar().currentMessage() == result.message.splitlines()[0]


@needs_qt
def test_open_training_without_a_preset_just_opens_the_screen(window) -> None:
    assert window.open_training() is None and window._training_monitor is not None
    assert window.centralWidget() is window._training_monitor


@needs_qt
def test_open_training_does_not_apply_while_a_job_runs(window, stored) -> None:
    window.training_controller = _busy_controller("running")
    assert training_is_active(window.training_controller) and controller_status(window.training_controller) == "running"
    result = window.open_training(TrainingPreset("module_training", dataset_paths=(str(stored.dataset),)))
    assert result is not None and not result.ok and result.message == tr("library.busy")
    monitor = window._training_monitor
    assert monitor is not None and monitor.message.text() == tr("library.busy")
    assert monitor.setup_panel.setup().dataset_paths is None  # the form was left alone
    window.training_controller = _busy_controller("finished")
    assert not training_is_active(window.training_controller)


@needs_qt
def test_training_screen_applies_library_choices_to_its_own_form(stored, monkeypatch) -> None:
    from bang_rl.gui.screens.training_monitor import TrainingMonitor

    passed: list = []
    monitor = TrainingMonitor(TrainingMonitorController(_Service("finished")), setup_panel=_panel(stored),
                              on_library_choice=lambda choice: passed.append(choice))
    library = _library(stored)
    good = _entry(library, "module_pair", RUN_GOOD)
    monitor.handle_library_choice(view.LibraryChoice(view.ACTION_MAIN_AI_TRAIN, (good,)))
    setup = monitor.setup_panel.setup()
    assert setup.mode == "main_ai_training" and setup.resume_intent_artifact_path == str(stored.good[0])
    assert tr("library.applied_module", job=RUN_GOOD[:8]) in monitor.message.text()

    replay = _entry(library, "replay")
    monitor.handle_library_choice(view.LibraryChoice(view.ACTION_VIEW_REPLAY, (replay,)))
    assert [choice.action for choice in passed] == [view.ACTION_VIEW_REPLAY]

    monitor.set_notice("")
    monitor.controller = _busy_controller("running")
    dataset = _entry(library, "dataset")
    monitor.handle_library_choice(view.LibraryChoice(view.ACTION_MODULE_TRAINING_DATA, (dataset,)))
    assert monitor.message.text() == tr("library.busy") and monitor.setup_panel.setup().mode == "main_ai_training"


@needs_qt
def test_training_screen_buttons_open_the_library(stored, monkeypatch) -> None:
    from bang_rl.gui.dialogs import data_library as module
    from bang_rl.gui.screens.training_monitor import TrainingMonitor

    opened: list[dict] = []

    class _Fake:
        chosen = None

        def __init__(self, **kwargs) -> None:
            opened.append(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(module, "DataLibraryDialog", _Fake)
    monitor = TrainingMonitor(TrainingMonitorController(_Service("finished")), setup_panel=_panel(stored))
    monitor.library_button.click()
    monitor.performance_button.click()
    assert [item["kind_filter"] for item in opened] == [None, view.KIND_FILTER_MODELS]
    locations = opened[0]["locations"]
    assert locations.model_root == stored.loc.model_root and locations.canonical_root == stored.loc.canonical_root
    assert tuple(locations.main_ai_dirs) == tuple(stored.loc.main_ai_dirs)


@needs_qt
def test_library_strings_exist_in_both_languages_without_em_dashes() -> None:
    from bang_rl.gui.i18n import LANGUAGES, _STRINGS

    keys = [key for key in _STRINGS if key.startswith(("library.", "tooltip.library", "menu.library"))]
    assert len(keys) > 100
    for key in keys:
        for language in LANGUAGES:
            text = _STRINGS[key].get(language, "")
            assert text.strip(), f"{key} empty in {language}"
            assert "\u2014" not in text, f"em dash in {key}"
    for action in view.ALL_ACTIONS:
        assert f"library.action.{action}" in _STRINGS and f"tooltip.library_action.{action}" in _STRINGS
    assert tr("menu.library") == "Data library" or tr("menu.library") == "데이터 보관함"
