"""Graphical replay viewer.

A recorded episode plays back through the SAME graphical board and left game
log used for human play (FEAT-003/FEAT-004), so replays look identical to a
live game.  By default the replay auto-plays from the first turn to the last on
a QTimer whose interval is set by the bottom speed slider (reusing the AI-turn
speed bounds).  The viewer can pause auto-play and step one turn at a time with
the right (next) and left (prev) arrow buttons.

Hovering a log row shows the game state at that recorded moment: the board
seeks to the moment's snapshot while the pointer stays on the row and restores
the current playback position when the pointer leaves.  All the reconstruction
and the row -> moment mapping live in the pure :mod:`bang_rl.gui.replay`
``ReplayModel`` so they are unit-testable headlessly.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..i18n import tr
from ..qt_compat import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    QT_AVAILABLE,
    require_qt,
)
from ..replay import ReplayModel
from .. import replay_store
from ..widgets.game_widgets import GameLog, GraphicalBoard
from .game_screen import AI_SPEED_DEFAULT_MS, AI_SPEED_MAX_MS, AI_SPEED_MIN_MS


# The human seat used when rendering a replay. Recordings are captured from the
# canonical player order with seat 0 as the human, matching live play.
REPLAY_HUMAN_PLAYER = 0


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class ReplayScreen(QWidget):
        def __init__(
            self,
            model: ReplayModel | None,
            parent: QWidget | None = None,
            *,
            on_back: Callable[[], None] | None = None,
            replay_directory: str | Path | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.model = model
            self.on_back = on_back
            self._alive = True
            # Current folder the browser lists replays from. When set the left
            # browser is populated via the Qt-free replay_store.list_replays.
            self._replay_directory: Path | None = (
                Path(replay_directory) if replay_directory is not None else None
            )
            # Snapshot index to restore to after a log-row hover preview ends.
            self._hover_restore_index: int | None = None
            self.setWindowTitle(tr("replay.title"))
            self.setMinimumWidth(1024)

            self.board = GraphicalBoard(self, human_player=REPLAY_HUMAN_PLAYER)
            self.log = GameLog(self)
            self.status = QLabel(self)

            # Left-side file browser: lists saved replays from the current
            # folder. Selecting one loads it onto the board/log. The re-point
            # button opens a native folder picker to change the load folder.
            self.browser_label = QLabel(tr("replay.browser"), self)
            self.browser = QListWidget(self)
            self.browser.itemActivated.connect(self._on_replay_selected)
            self.browser.itemClicked.connect(self._on_replay_selected)
            self.repath_button = QPushButton(tr("replay.repath"), self)
            self.repath_button.clicked.connect(self._on_repath_clicked)
            self.open_file_button = QPushButton(tr("replay.open_file"), self)
            self.open_file_button.clicked.connect(self._on_open_file_clicked)
            self._replay_paths: list[Path] = []

            # Auto-play timer; the slider sets its interval. Guarded by _alive so
            # a torn-down screen never renders into deleted C++ objects.
            self._play_timer = QTimer(self)
            self._play_timer.setSingleShot(False)
            self._play_timer.timeout.connect(self._advance_one_turn)

            self.back_button = QPushButton(tr("common.back"), self)
            self.back_button.clicked.connect(self._go_back)

            # Static label describing that the viewer auto-plays by default.
            self.autoplay_label = QLabel(tr("replay.autoplay"), self)
            self.play_button = QPushButton(tr("replay.pause"), self)
            self.play_button.clicked.connect(self._toggle_play)
            self.prev_button = QPushButton(tr("replay.prev"), self)
            self.prev_button.clicked.connect(self._step_prev)
            self.next_button = QPushButton(tr("replay.next"), self)
            self.next_button.clicked.connect(self._step_next)

            self.speed_label = QLabel(self)
            self.speed_slider = QSlider(Qt.Orientation.Horizontal, self)
            self.speed_slider.setMinimum(AI_SPEED_MIN_MS)
            self.speed_slider.setMaximum(AI_SPEED_MAX_MS)
            self.speed_slider.setValue(AI_SPEED_DEFAULT_MS)
            self.speed_slider.setSingleStep(100)
            self.speed_slider.valueChanged.connect(self._on_speed_changed)
            self._update_speed_label(AI_SPEED_DEFAULT_MS)

            # Left = folder browser, then log, center = graphical board.
            browser_column = QVBoxLayout()
            browser_column.addWidget(self.browser_label)
            browser_column.addWidget(self.browser, 1)
            browser_column.addWidget(self.repath_button)
            browser_column.addWidget(self.open_file_button)

            top = QHBoxLayout()
            top.addLayout(browser_column, 1)
            top.addWidget(self.log, 1)
            top.addWidget(self.board, 3)

            controls = QHBoxLayout()
            controls.addWidget(self.autoplay_label)
            controls.addWidget(self.prev_button)
            controls.addWidget(self.play_button)
            controls.addWidget(self.next_button)

            speed_row = QHBoxLayout()
            speed_row.addWidget(self.speed_label)
            speed_row.addWidget(self.speed_slider, 1)

            layout = QVBoxLayout(self)
            layout.addWidget(self.back_button)
            layout.addLayout(top, 1)
            layout.addLayout(controls)
            layout.addLayout(speed_row)
            layout.addWidget(self.status)

            # Log-row hover shows the game state at that recorded moment. Mouse
            # tracking + itemEntered fire per-row; leaving the list restores the
            # current playback position.
            self.log.list.setMouseTracking(True)
            self.log.list.itemEntered.connect(self._on_log_row_hovered)
            self.log.list.viewport().installEventFilter(self)

            # Populate the browser from the initial folder (if any). This does
            # not auto-load a file, so an in-memory model (if supplied) stays
            # shown; with no model the first listed replay can be selected.
            self._reload_browser()

            self._render_current()
            # Default behaviour: auto-play from the first turn to the last.
            self._start_autoplay()

        # -- folder browser ----------------------------------------------
        def _reload_browser(self) -> None:
            """Refresh the file list from the current replay directory."""

            if not self._alive:
                return
            self.browser.clear()
            self._replay_paths = []
            if self._replay_directory is None:
                return
            try:
                paths = replay_store.list_replays(self._replay_directory)
            except OSError:
                paths = []
            self._replay_paths = list(paths)
            for path in self._replay_paths:
                item = QListWidgetItem(self._browser_label_for(path))
                self.browser.addItem(item)

        def _browser_label_for(self, path: Path) -> str:
            """Build a display label from a lightweight replay header read."""

            label = path.name
            try:
                record = replay_store.load_replay(path)
            except (OSError, ValueError):
                return label
            summary = replay_store.replay_summary(record)
            mode_label = summary.get("mode_label")
            outcome = summary.get("outcome")
            parts = [path.name]
            if mode_label:
                parts.append(str(mode_label))
            if outcome:
                parts.append(str(outcome))
            return "  |  ".join(parts)

        def _on_replay_selected(self, item: Any) -> None:
            if not self._alive or item is None:
                return
            row = self.browser.row(item)
            if not (0 <= row < len(self._replay_paths)):
                return
            path = self._replay_paths[row]
            try:
                record = replay_store.load_replay(path)
                model = ReplayModel(record)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.status.setText(str(exc) or tr("replay.no_episode"))
                return
            self._load_model(model)

        def _load_model(self, model: ReplayModel) -> None:
            """Swap in a freshly loaded model and restart playback."""

            self._play_timer.stop()
            self._hover_restore_index = None
            self.model = model
            self._render_current()
            self._start_autoplay()

        def _on_repath_clicked(self) -> None:
            start_dir = str(self._replay_directory) if self._replay_directory else ""
            directory = QFileDialog.getExistingDirectory(
                self, tr("replay.folder_title"), start_dir
            )
            if not directory:
                return
            self._replay_directory = Path(directory)
            self._reload_browser()

        def _on_open_file_clicked(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self, tr("replay.open_file"), "", tr("replay.file_filter")
            )
            if not path:
                return
            try:
                record = replay_store.load_replay(path)
                model = ReplayModel(record)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.status.setText(str(exc) or tr("replay.no_episode"))
                return
            self._load_model(model)

        # -- navigation --------------------------------------------------
        def _go_back(self) -> None:
            self._detach()
            if self.on_back is not None:
                self.on_back()

        def _detach(self) -> None:
            """Stop rendering/timing into this (about to be torn down) screen."""

            self._alive = False
            self._play_timer.stop()
            self.board.detach()

        # -- playback ----------------------------------------------------
        def _start_autoplay(self) -> None:
            if not self._alive or self.model is None or self.model.is_empty():
                self.play_button.setText(tr("replay.play"))
                return
            if self.model.is_at_end():
                # Nothing left to auto-play; leave the button as Play.
                self.play_button.setText(tr("replay.play"))
                return
            self.play_button.setText(tr("replay.pause"))
            self._play_timer.setInterval(int(self.speed_slider.value()))
            self._play_timer.start()

        def _toggle_play(self) -> None:
            if self._play_timer.isActive():
                self._play_timer.stop()
                self.play_button.setText(tr("replay.play"))
            else:
                self._start_autoplay()

        def _advance_one_turn(self) -> None:
            if not self._alive or self.model is None:
                self._play_timer.stop()
                return
            if self.model.is_at_end():
                self._play_timer.stop()
                self.play_button.setText(tr("replay.play"))
                return
            self.model.next()
            self._render_current()
            if self.model.is_at_end():
                self._play_timer.stop()
                self.play_button.setText(tr("replay.play"))

        def _step_next(self) -> None:
            if self.model is None:
                return
            self._play_timer.stop()
            self.play_button.setText(tr("replay.play"))
            self.model.next()
            self._render_current()

        def _step_prev(self) -> None:
            if self.model is None:
                return
            self._play_timer.stop()
            self.play_button.setText(tr("replay.play"))
            self.model.prev()
            self._render_current()

        # -- speed slider ------------------------------------------------
        def _on_speed_changed(self, value: int) -> None:
            self._update_speed_label(value)
            if self._play_timer.isActive():
                self._play_timer.setInterval(int(value))

        def _update_speed_label(self, value: int) -> None:
            self.speed_label.setText(tr("replay.speed", seconds=int(value) / 1000.0))

        # -- log-row hover -----------------------------------------------
        def _on_log_row_hovered(self, item: Any) -> None:
            if not self._alive or self.model is None:
                return
            row = self.log.list.row(item)
            index = self.model.index_for_log_row(row)
            if index is None:
                return
            # Remember where playback was so leaving the row restores it.
            if self._hover_restore_index is None:
                self._hover_restore_index = self.model.index
            snapshot = self.model.snapshot_at(index)
            if snapshot is not None:
                self._render_snapshot(snapshot)

        def _end_hover_preview(self) -> None:
            if self._hover_restore_index is None:
                return
            index = self._hover_restore_index
            self._hover_restore_index = None
            if self.model is not None:
                self.model.seek(index)
            self._render_current()

        def eventFilter(self, source: Any, event: Any) -> bool:
            # Restore the playback position when the pointer leaves the log.
            if (
                self._alive
                and source is self.log.list.viewport()
                and event.type() == event.Type.Leave
            ):
                self._end_hover_preview()
            return super().eventFilter(source, event)

        # -- rendering ---------------------------------------------------
        def _render_current(self) -> None:
            if self.model is None or self.model.is_empty():
                self.board.setEnabled(False)
                self.status.setText(tr("replay.no_episode"))
                self._set_controls_enabled(False)
                return
            snapshot = self.model.current()
            if snapshot is not None:
                self._render_snapshot(snapshot)
            self._set_controls_enabled(True)
            self.status.setText(
                tr("replay.status", index=self.model.index + 1, total=len(self.model))
            )

        def _render_snapshot(self, snapshot: Any) -> None:
            if not self._alive:
                return
            self.board.update_state(snapshot)
            self.log.update_state(snapshot)

        def _set_controls_enabled(self, enabled: bool) -> None:
            self.play_button.setEnabled(enabled)
            self.prev_button.setEnabled(enabled)
            self.next_button.setEnabled(enabled)
            self.speed_slider.setEnabled(enabled)

        def closeEvent(self, event: Any) -> None:
            self._detach()
            event.accept()

else:

    class ReplayScreen:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["REPLAY_HUMAN_PLAYER", "ReplayScreen"]
