"""Data library ("데이터 보관함"): browse every stored artifact and apply it directly.

One table lists datasets, module pairs, main-AI models, reports and replays
(see :mod:`bang_rl.data_library`). The right side shows everything about the
selected entry, with links to the entries it was made from and the ones that
used it. The buttons depend on the selection: they hand a
:class:`~bang_rl.gui.library_view.LibraryChoice` to whoever opened the library
(the main window or the training screen), which applies it to the training
setup, starts a game or opens a replay.

What may be shown and which button is usable is decided in
:mod:`bang_rl.gui.library_view`, which has no Qt dependency and is unit tested.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bang_rl.data_library import Library, LibraryEntry, LibraryLocations, build_library

from .. import library_view as view
from ..i18n import tr
from ..qt_compat import (
    QAbstractItemView,
    QComboBox,
    QDesktopServices,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTimer,
    Qt,
    QUrl,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from ..training import ACTIVE_TRAINING_STATUSES

# Game modes the "use in a game" button offers (values of ``GameMode``).
PLAY_MODE_SPECTATE = "main_one_dummy_four"
PLAY_MODE_HUMAN = "human_main_dummy"

_STATUS_POLL_MS = 1500


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class ReportViewDialog(QDialog):
        """Read-only view of one report file."""

        def __init__(self, entry: LibraryEntry, parent: QWidget | None = None, *, open_external: Callable[[Path], bool]) -> None:
            require_qt()
            super().__init__(parent)
            self.setWindowTitle(entry.name)
            self.resize(760, 560)
            layout = QVBoxLayout(self)
            self.text = QPlainTextEdit(self)
            self.text.setReadOnly(True)
            self.text.setPlainText(view.report_text(entry.path))
            layout.addWidget(self.text, 1)
            buttons = QHBoxLayout()
            buttons.addStretch(1)
            self.external_button = QPushButton(tr("library.open_external"), self)
            self.external_button.clicked.connect(lambda: open_external(entry.path))
            close = QPushButton(tr("perf.close"), self)
            close.clicked.connect(self.accept)
            buttons.addWidget(self.external_button)
            buttons.addWidget(close)
            layout.addLayout(buttons)

    class DataLibraryDialog(QDialog):
        """Filter bar, table, details pane and context-sensitive action buttons."""

        def __init__(
            self,
            *,
            locations: LibraryLocations | None = None,
            config: Any = None,
            kind_filter: str | None = None,
            training_busy: Callable[[], bool] | None = None,
            status_provider: Callable[[], str | None] | None = None,
            build: Callable[[], Library] | None = None,
            parent: QWidget | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self._locations = locations
            self._config = config
            self._training_busy = training_busy
            self._status_provider = status_provider
            self._build = build or (lambda: build_library(self._locations, config=self._config))
            self.library: Library = Library([])
            self.visible_entries: list[LibraryEntry] = []
            self.chosen: view.LibraryChoice | None = None
            self.last_build_seconds = 0.0
            self._details_cache: dict[str, str] = {}
            self._last_status: str | None = status_provider() if status_provider is not None else None
            self._filling = False

            self.setWindowTitle(tr("library.title"))
            self.resize(1240, 720)

            self.kind_combo = QComboBox(self)
            for value in view.KIND_FILTERS:
                self.kind_combo.addItem(self._kind_filter_label(value), value)
            self.location_combo = QComboBox(self)
            self.location_combo.addItem(tr("library.location.all"), None)
            self.location_combo.addItem(tr("library.location.local"), "local")
            self.location_combo.addItem(tr("library.location.saved"), "saved")
            self.search = QLineEdit(self)
            self.search.setPlaceholderText(tr("library.search_placeholder"))
            self.search.setClearButtonEnabled(True)
            self.search.setToolTip(tr("tooltip.library_search"))
            self.refresh_button = QPushButton(tr("perf.refresh"), self)
            self.refresh_button.setToolTip(tr("tooltip.library_refresh"))
            self.count_label = QLabel("", self)
            self.kind_combo.setToolTip(tr("tooltip.library_kind"))
            self.location_combo.setToolTip(tr("tooltip.library_location"))

            self.table = QTableWidget(0, len(view.TABLE_COLUMNS), self)
            self.table.setHorizontalHeaderLabels([tr(key) for key in view.TABLE_COLUMNS])
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setWordWrap(False)
            self.table.verticalHeader().setVisible(False)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(True)
            for column, width in enumerate((96, 230, 120, 190, 230, 64, 160)):
                self.table.setColumnWidth(column, width)
            self.details = QTextBrowser(self)
            self.details.setOpenLinks(False)
            self.details.setOpenExternalLinks(False)
            self.empty_label = QLabel(tr("library.empty"), self)
            self.empty_label.setWordWrap(True)
            self.message = QLabel("", self)
            self.message.setWordWrap(True)

            split = QSplitter(Qt.Orientation.Horizontal, self)
            split.addWidget(self.table)
            split.addWidget(self.details)
            split.setStretchFactor(0, 3)
            split.setStretchFactor(1, 2)
            split.setSizes([760, 440])

            filters = QHBoxLayout()
            filters.addWidget(QLabel(tr("library.filter_kind"), self))
            filters.addWidget(self.kind_combo)
            filters.addWidget(QLabel(tr("library.filter_location"), self))
            filters.addWidget(self.location_combo)
            filters.addWidget(self.search, 1)
            filters.addWidget(self.refresh_button)
            filters.addWidget(self.count_label)

            self.buttons: dict[str, QPushButton] = {}
            action_row = QHBoxLayout()
            for action in view.ALL_ACTIONS:
                button = QPushButton(tr(f"library.action.{action}"), self)
                button.clicked.connect(lambda _checked=False, name=action: self.trigger(name))
                self.buttons[action] = button
                action_row.addWidget(button)
            action_row.addStretch(1)
            self.close_button = QPushButton(tr("perf.close"), self)
            self.close_button.clicked.connect(self.reject)
            action_row.addWidget(self.close_button)

            layout = QVBoxLayout(self)
            layout.addLayout(filters)
            layout.addWidget(self.empty_label)
            layout.addWidget(split, 1)
            layout.addWidget(self.message)
            layout.addLayout(action_row)

            if kind_filter is not None:
                index = self.kind_combo.findData(kind_filter)
                if index >= 0:
                    self.kind_combo.setCurrentIndex(index)
            self.kind_combo.currentIndexChanged.connect(lambda _index: self.apply_filters())
            self.location_combo.currentIndexChanged.connect(lambda _index: self.apply_filters())
            self.search.textChanged.connect(lambda _text: self.apply_filters())
            self.refresh_button.clicked.connect(self.refresh)
            self.table.itemSelectionChanged.connect(self._on_selection_changed)
            self.table.cellDoubleClicked.connect(lambda _row, _column: self.activate_default())
            self.details.anchorClicked.connect(self._on_link)

            self._timer = QTimer(self)
            self._timer.setInterval(_STATUS_POLL_MS)
            self._timer.timeout.connect(self._poll_status)
            if status_provider is not None:
                self._timer.start()
            self.refresh()

        # -- filters and table -------------------------------------------------------

        @staticmethod
        def _kind_filter_label(value: str) -> str:
            if value == view.KIND_FILTER_ALL:
                return tr("library.kind.all")
            return tr(f"library.kind.{value}")

        def refresh(self) -> None:
            """Rescan every folder, keep the filters and the selection."""

            keys = [entry.key for entry in self.selected_entries()]
            started = time.perf_counter()
            self.library = self._build()
            self.last_build_seconds = time.perf_counter() - started
            self._details_cache.clear()
            self.apply_filters(select_keys=keys)

        def apply_filters(self, *, select_keys: list[str] | None = None) -> None:
            if select_keys is None:
                select_keys = [entry.key for entry in self.selected_entries()]
            kinds = view.kinds_for_filter(self.kind_combo.currentData())
            self.visible_entries = self.library.filtered(
                kinds=kinds, location=self.location_combo.currentData(), text=self.search.text()
            )
            self._filling = True
            self.table.setUpdatesEnabled(False)
            try:
                self.table.clearContents()
                self.table.setRowCount(len(self.visible_entries))
                for row, entry in enumerate(self.visible_entries):
                    data = view.table_row(entry)
                    for column, text in enumerate(data.cells):
                        item = QTableWidgetItem(text)
                        item.setToolTip(data.tooltip)
                        if column == 0:
                            item.setData(Qt.ItemDataRole.UserRole, entry.key)
                        self.table.setItem(row, column, item)
                wanted = set(select_keys)
                self.table.clearSelection()
                for row, entry in enumerate(self.visible_entries):
                    if entry.key in wanted:
                        self.table.selectRow(row)
            finally:
                self.table.setUpdatesEnabled(True)
                self._filling = False
            total = len(self.library.entries)
            self.count_label.setText(tr("library.count", shown=len(self.visible_entries), total=total))
            self.empty_label.setVisible(total == 0)
            if not self.table.selectedItems() and self.visible_entries and not select_keys:
                self.table.selectRow(0)
            self._on_selection_changed()

        def selected_entries(self) -> list[LibraryEntry]:
            rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()}) if self.table.selectionModel() else []
            return [self.visible_entries[row] for row in rows if 0 <= row < len(self.visible_entries)]

        def select_entry(self, key: str) -> bool:
            """Select the entry with ``key``; filters that hide it are reset first."""

            entry = self.library.get(key)
            if entry is None:
                return False
            if entry not in self.visible_entries:
                self.kind_combo.blockSignals(True)
                self.location_combo.blockSignals(True)
                self.search.blockSignals(True)
                self.kind_combo.setCurrentIndex(0)
                self.location_combo.setCurrentIndex(0)
                self.search.clear()
                self.kind_combo.blockSignals(False)
                self.location_combo.blockSignals(False)
                self.search.blockSignals(False)
                self.apply_filters(select_keys=[key])
            self.table.clearSelection()
            for row, item in enumerate(self.visible_entries):
                if item.key == key:
                    self.table.selectRow(row)
                    self.table.scrollToItem(self.table.item(row, 1))
                    break
            self._on_selection_changed()
            return True

        # -- details and buttons -----------------------------------------------------

        def _on_selection_changed(self) -> None:
            if self._filling:
                return
            entries = self.selected_entries()
            self._show_details(entries)
            self._update_buttons(entries)

        def _show_details(self, entries: list[LibraryEntry]) -> None:
            if not entries:
                self.details.setHtml(f"<p>{tr('library.select_prompt')}</p>")
                return
            if len(entries) > 1:
                self.details.setHtml(view.multi_details_html(entries))
                return
            entry = entries[0]
            html = self._details_cache.get(entry.key)
            if html is None:
                try:
                    performance = view.performance_text(entry)
                except Exception as exc:  # noqa: BLE001 - a broken file must not break the pane
                    performance = tr("perf.error", error=str(exc))
                html = view.entry_details_html(entry, performance=performance)
                self._details_cache[entry.key] = html
            self.details.setHtml(html)

        def _update_buttons(self, entries: list[LibraryEntry]) -> None:
            states = view.action_states(entries)
            kinds = {entry.kind for entry in entries}
            relevant: set[str] = set(view.GENERAL_ACTIONS)
            if len(kinds) == 1:
                relevant |= set(view.ACTIONS_BY_KIND.get(next(iter(kinds)), ()))
            for action, button in self.buttons.items():
                state = states[action]
                button.setVisible(action in relevant)
                button.setEnabled(state.enabled)
                tip = tr(f"tooltip.library_action.{action}")
                button.setToolTip(tip if state.enabled or not state.reason else f"{state.reason}\n\n{tip}")

        def _on_link(self, url: Any) -> None:
            text = url.toString() if hasattr(url, "toString") else str(url)
            if text.startswith("entry:"):
                self.select_entry(text[len("entry:"):])

        def _say(self, text: str) -> None:
            self.message.setText(text)

        # -- actions -------------------------------------------------------------------

        def activate_default(self) -> bool:
            """Double click: run the usual action of the selected entry's kind."""

            action = view.default_action(self.selected_entries())
            if action is None:
                return False
            return self.trigger(action)

        def trigger(self, action: str) -> bool:
            """Run ``action`` on the selection. Returns True when it was carried out."""

            entries = tuple(self.selected_entries())
            state = view.action_states(entries).get(action)
            if state is None or not state.enabled:
                self._say(state.reason if state is not None else "")
                return False
            self._say("")
            if action in view.APPLY_ACTIONS:
                if self._training_busy is not None and self._training_busy():
                    self._say(tr("library.busy"))
                    return False
                return self._choose(view.LibraryChoice(action, entries))
            if action == view.ACTION_PLAY:
                mode = self._choose_game_mode(entries[0])
                if mode is None:
                    return False
                return self._choose(view.LibraryChoice(action, entries, {"mode": mode}))
            if action == view.ACTION_VIEW_REPLAY:
                return self._choose(view.LibraryChoice(action, entries))
            if action == view.ACTION_DELETE:
                return self._delete(entries)
            entry = entries[0]
            if action == view.ACTION_EVALUATE:
                self._open_evaluation(entry)
            elif action == view.ACTION_OPEN_REPORT:
                self._open_report(entry)
            elif action == view.ACTION_EDIT_NOTES:
                self._edit_notes(entry)
            elif action == view.ACTION_SAVE_COPY:
                self._save_copy(entry)
            elif action == view.ACTION_OPEN_FOLDER:
                self._open_path(entry.folder)
            return True

        def _choose(self, choice: view.LibraryChoice) -> bool:
            self.chosen = choice
            self.accept()
            return True

        def _choose_game_mode(self, entry: LibraryEntry) -> str | None:
            """Ask whether to watch the model play or to play against it."""

            box = QMessageBox(self)
            box.setWindowTitle(tr("library.play_title"))
            box.setText(tr("library.play_prompt", name=entry.name))
            spectate = box.addButton(tr("library.play_spectate"), QMessageBox.ButtonRole.ActionRole)
            human = box.addButton(tr("library.play_human"), QMessageBox.ButtonRole.ActionRole)
            box.addButton(tr("save.cancel"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is spectate:
                return PLAY_MODE_SPECTATE
            if clicked is human:
                return PLAY_MODE_HUMAN
            return None

        def _open_path(self, path: Path) -> bool:
            return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))))

        def _open_report(self, entry: LibraryEntry) -> None:
            ReportViewDialog(entry, self, open_external=self._open_path).exec()

        def _open_evaluation(self, entry: LibraryEntry) -> None:
            from .inference_evaluation import InferenceEvaluationDialog

            locations = self._locations
            kwargs: dict[str, Any] = {}
            if entry.kind == view.KIND_MAIN_AI:
                kwargs["select_model"] = entry.path
            else:
                kwargs["select_pair"] = (entry.intent_path, entry.emotion_path)
            dialog = InferenceEvaluationDialog(
                config=self._config,
                model_root=None if locations is None else locations.model_root,
                main_ai_dirs=None if locations is None else locations.main_ai_dirs,
                parent=self,
                **kwargs,
            )
            dialog.exec()

        def set_notes(self, entry: LibraryEntry, text: str) -> int:
            """Write ``text`` as the note of ``entry`` (every file holding its info); returns files updated."""

            from bang_rl.artifact_info import read_info, update_notes

            updated = 0
            for path in entry.info_paths:
                if read_info(path) is not None:
                    update_notes(path, text)
                    updated += 1
            self.refresh()
            return updated

        def _edit_notes(self, entry: LibraryEntry) -> None:
            from .save_outputs import NotesDialog

            dialog = NotesDialog(entry.name, entry.notes, self)
            if dialog.exec():
                self.set_notes(entry, dialog.text)

        def confirm_delete(self, entries: tuple[LibraryEntry, ...]) -> bool:
            """Ask before moving entries to the trash (tests replace this)."""

            answer = QMessageBox.question(
                self,
                tr("library.delete_title"),
                view.delete_confirmation_text(entries),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return answer == QMessageBox.StandardButton.Yes

        def delete_entries(self, entries: tuple[LibraryEntry, ...], *, trash_dir: Path | None = None,
                           allowed_roots: Any = None) -> list[Path]:
            """Move each entry to the trash; returns the trash folders. Errors are reported, not raised."""

            from bang_rl.artifact_trash import move_to_trash

            if self._training_busy is not None and self._training_busy():
                self._say(tr("library.busy"))
                return []
            folders: list[Path] = []
            errors: list[str] = []
            for entry in entries:
                try:
                    result = move_to_trash(
                        view.entry_delete_paths(entry),
                        name=entry.name,
                        include_replay_trace=entry.kind == view.KIND_REPLAY,
                        trash_dir=trash_dir,
                        allowed_roots=allowed_roots,
                    )
                    folders.append(result.folder)
                except Exception as exc:  # noqa: BLE001 - one failure must not stop the rest
                    errors.append(f"{entry.name}: {exc}")
            self.refresh()
            if errors:
                self._say(tr("library.delete_failed", errors="; ".join(errors)))
            else:
                self._say(tr("library.delete_done", count=len(folders)))
            return folders

        def _delete(self, entries: tuple[LibraryEntry, ...]) -> bool:
            if not entries or not self.confirm_delete(entries):
                return False
            self.delete_entries(entries)
            return True

        def _save_copy(self, entry: LibraryEntry) -> None:
            from .save_outputs import SaveOutputsDialog

            plan = view.save_plan_for(entry)
            if plan is None:
                self._say(tr("library.reason.no_copy_plan"))
                return
            SaveOutputsDialog(plan, self).exec()
            self.refresh()

        # -- auto refresh ------------------------------------------------------------------

        def _poll_status(self) -> None:
            """Reload when a training job that was running has just ended."""

            if self._status_provider is None:
                return
            status = self._status_provider()
            if status != self._last_status:
                finished = status is not None and status not in ACTIVE_TRAINING_STATUSES
                self._last_status = status
                if finished:
                    self.refresh()

        def done(self, result: int) -> None:  # noqa: D401 - Qt signature
            self._timer.stop()
            super().done(result)

else:

    class ReportViewDialog:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

    class DataLibraryDialog:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["DataLibraryDialog", "PLAY_MODE_HUMAN", "PLAY_MODE_SPECTATE", "ReportViewDialog"]
