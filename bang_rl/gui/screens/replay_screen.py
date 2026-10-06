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
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
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
from ..playback_speed import is_manual_speed, speed_to_interval_ms
from ..widgets.game_widgets import GameLog, GraphicalBoard
from .game_screen import AI_SPEED_MAX_MS, AI_SPEED_MIN_MS


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
            # The currently loaded replay record + its on-disk path (when it was
            # loaded from the browser). Used by rename/delete (item 10).
            self._loaded_record: Any = None
            self._loaded_path: Path | None = None
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
            # FEAT-002: enable the spectator / replay overlays (emotion bars +
            # role-inference hover) for EVERY replay, including human-played
            # ones. A replay is a post-hoc review of a finished game, not live
            # hidden-info play, so revealing the recorded emotions / inferred
            # roles here leaks nothing. A spectator replay already has
            # human_player=None (overlays on by layout); this makes the human
            # replay case consistent.
            #
            # Item 6 (FEAT-004): this forced flag survives every _load_model
            # (set_human_player never clears it), and _render_snapshot feeds the
            # current snapshot's internals via set_seat_internals on EVERY cursor
            # change (auto-advance, Prev / Next, click, hover), so the inverted
            # role-inference overlay draws on hover for a FULL-format replay that
            # carries per-turn internals. A PUBLIC (observations-only) replay has
            # internals=None on every snapshot, so the overlay shows nothing on
            # hover; that is expected (no internals were recorded).
            self.board.set_overlays_enabled(True)
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
            # Item 10: rename / delete the currently selected browser replay.
            self.rename_button = QPushButton(tr("replay.rename"), self)
            self.rename_button.clicked.connect(self._on_rename_clicked)
            self.delete_button = QPushButton(tr("replay.delete"), self)
            self.delete_button.clicked.connect(self._on_delete_clicked)
            self._replay_paths: list[Path] = []

            # Item 6: read-only info panel describing the loaded record (mode,
            # outcome/winner, seed, and the per-seat AI + version breakdown from
            # record['ai_versions']). Populated on every load and cleared when no
            # record is loaded. The Qt-free extraction lives in replay_store; the
            # panel only localizes the structured entries via tr().
            self.info_title = QLabel(tr("replay.info_title"), self)
            self.info_panel = QLabel(self)
            self.info_panel.setWordWrap(True)
            self.info_panel.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._update_info_panel()

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
            # Item 3 (FEAT-004): the replay opens in MANUAL mode (slider at the
            # minimum). A loaded replay therefore starts PAUSED on the first
            # frame and the user steps with Prev / Next; auto-play only begins
            # once the slider is dragged above the minimum. _start_autoplay
            # returns early while the slider sits at the minimum
            # (speed_to_interval_ms is None), so the Play button stays on Play.
            self.speed_slider.setValue(AI_SPEED_MIN_MS)
            self.speed_slider.setSingleStep(100)
            self.speed_slider.valueChanged.connect(self._on_speed_changed)
            # Item 3 (FEAT-002): a static "Manual" tick label pinned near the
            # slider minimum. In manual mode auto-play stops and the user steps
            # with the existing Next turn / Prev buttons (one turn = model.next
            # / prev + render, which _step_next / _step_prev already do).
            self.speed_manual_label = QLabel(tr("replay.speed_manual_label"), self)
            # Item 3 (FEAT-004): label reflects the manual default slider value.
            self._update_speed_label(AI_SPEED_MIN_MS)

            # Left = folder browser, then log, center = graphical board.
            browser_column = QVBoxLayout()
            browser_column.addWidget(self.browser_label)
            browser_column.addWidget(self.browser, 1)
            browser_column.addWidget(self.repath_button)
            browser_column.addWidget(self.open_file_button)
            manage_row = QHBoxLayout()
            manage_row.addWidget(self.rename_button)
            manage_row.addWidget(self.delete_button)
            browser_column.addLayout(manage_row)
            # Item 6: info panel sits below the manage buttons in the left
            # column, visible during playback without crowding the board.
            browser_column.addWidget(self.info_title)
            browser_column.addWidget(self.info_panel)

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
            speed_row.addWidget(self.speed_manual_label)
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
            # Item 4 (FEAT-003): clicking a log row PINS the view to that turn
            # and forces manual mode (see _on_log_row_clicked). This coexists
            # with the hover-preview above: hover = preview, click = pin.
            self.log.list.itemClicked.connect(self._on_log_row_clicked)
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
            # Item 5: surface whether this is a full-state (resumable) or public
            # (training-only) replay so the user knows before selecting it. The
            # format label resolves through the single Qt-free helper so the
            # browser row and the info panel can never disagree.
            parts.append(tr(replay_store.replay_format_label_tag(summary)))
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
            self._loaded_record = record
            self._loaded_path = path
            self._load_model(model)

        def _load_model(self, model: ReplayModel) -> None:
            """Swap in a freshly loaded model and restart playback."""

            self._play_timer.stop()
            self._hover_restore_index = None
            self.model = model
            # ITEM 2: a spectator (all-AI) replay has no human, so render the
            # board with no dedicated "나"/"You" seat; every seat is a Player.
            # The recorded observations already carry all-revealed roles for a
            # spectator game (the service marked them REVEALED during play), so
            # dropping the human seat is enough to show all roles here too.
            # Item 6 (FEAT-004) regression guard: set_human_player only updates
            # the human seat + clears the hover seat; it does NOT touch
            # GraphicalBoard._overlays_forced. So the __init__ set_overlays_enabled
            # (True) call keeps the role-inference / emotion overlays ON across
            # every _load_model, for BOTH a spectator replay (human_player None,
            # overlays on by layout) and a human replay (human_player 0, overlays
            # kept on by the forced flag). Do not call set_overlays_enabled here;
            # the forced flag must survive model swaps.
            if replay_store.is_spectator_record(self._loaded_record):
                self.board.set_human_player(None)
            else:
                self.board.set_human_player(REPLAY_HUMAN_PLAYER)
            self._render_current()
            self._update_info_panel()
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
            self._loaded_record = record
            self._loaded_path = Path(path)
            self._load_model(model)

        # -- rename / delete (item 10) -----------------------------------
        def _selected_browser_path(self) -> Path | None:
            # Resolve the selected row robustly: prefer the explicitly selected
            # / current item (which stays correct after a click even if the
            # currentRow index drifts), then fall back to currentRow(). This
            # fixes delete/rename acting on the wrong (or no) file when the row
            # cursor and the visual selection are out of sync.
            row = -1
            item = self.browser.currentItem()
            if item is None:
                selected = self.browser.selectedItems()
                if selected:
                    item = selected[0]
            if item is not None:
                row = self.browser.row(item)
            # Delegate the final "row + paths -> Path" bounds mapping to the
            # pure, Qt-free replay_store.select_replay_path seam so the bounds
            # logic is unit-testable in-sandbox and revert-protected.
            path = replay_store.select_replay_path(self._replay_paths, row)
            if path is None:
                row = self.browser.currentRow()
                path = replay_store.select_replay_path(self._replay_paths, row)
            if path is None:
                self.status.setText(tr("replay.select_first"))
                return None
            return path

        def _on_rename_clicked(self) -> None:
            path = self._selected_browser_path()
            if path is None:
                return
            new_name, accepted = QInputDialog.getText(
                self, tr("replay.rename_title"), tr("replay.rename_prompt"), text=path.name
            )
            if not accepted or not str(new_name).strip():
                return
            try:
                new_path = replay_store.rename_replay(path, new_name)
            except (OSError, ValueError, FileExistsError, FileNotFoundError) as exc:
                self.status.setText(str(exc) or tr("replay.no_episode"))
                return
            # Keep a loaded record's path in sync when it was the renamed file.
            if self._loaded_path is not None and self._loaded_path == path:
                self._loaded_path = new_path
            self._reload_browser()

        def _on_delete_clicked(self) -> None:
            path = self._selected_browser_path()
            if path is None:
                return
            confirm = QMessageBox.question(
                self,
                tr("replay.delete_title"),
                tr("replay.delete_confirm"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            try:
                replay_store.delete_replay(path)
            except OSError as exc:
                self.status.setText(str(exc) or tr("replay.no_episode"))
                return
            # Defensive: confirm the file is actually gone. If something kept it
            # on disk (e.g. a stale path), surface it instead of silently
            # pretending the delete succeeded.
            try:
                still_present = Path(path).exists()
            except OSError:
                still_present = False
            if still_present:
                self.status.setText(tr("replay.delete_failed"))
                self._reload_browser()
                return
            # If the deleted file was the loaded one, clear the board / log.
            if self._loaded_path is not None and self._loaded_path == path:
                self._play_timer.stop()
                self.model = None
                self._loaded_record = None
                self._loaded_path = None
                self._render_current()
                self._update_info_panel()
            self._reload_browser()

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
            # Item 3 (FEAT-002): in manual mode (slider at minimum) never start
            # the auto-play timer; the user advances with the Next turn / Prev
            # buttons. Leave the Play button showing Play.
            interval = speed_to_interval_ms(self.speed_slider.value())
            if interval is None:
                self._play_timer.stop()
                self.play_button.setText(tr("replay.play"))
                return
            if self.model.is_at_end():
                # Nothing left to auto-play; leave the button as Play.
                self.play_button.setText(tr("replay.play"))
                return
            self.play_button.setText(tr("replay.pause"))
            self._play_timer.setInterval(interval)
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
            if is_manual_speed(value):
                # Item 3 (FEAT-002): entering manual mode stops auto-play; the
                # user steps with the Next turn / Prev buttons. Guard with
                # _alive so a torn-down screen never touches dead C++ objects.
                if self._alive:
                    self._play_timer.stop()
                    self.play_button.setText(tr("replay.play"))
                return
            interval = speed_to_interval_ms(value)
            if interval is None:  # pragma: no cover - manual handled above
                return
            if self._play_timer.isActive():
                self._play_timer.setInterval(interval)
            elif (
                self._alive
                and self.model is not None
                and not self.model.is_empty()
                and not self.model.is_at_end()
            ):
                # Leaving manual mode (and not at the end): resume auto-play.
                self._start_autoplay()

        def _update_speed_label(self, value: int) -> None:
            # At the minimum tick the label reads "Manual" so the user sees that
            # auto-play is off and they must step with Next turn / Prev.
            if is_manual_speed(value):
                self.speed_label.setText(tr("replay.speed_manual"))
            else:
                self.speed_label.setText(
                    tr("replay.speed", seconds=int(value) / 1000.0)
                )

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
                # Items 2 & 3: preview the hovered moment on the BOARD only and
                # highlight the hovered row via preview_moment. The full log
                # stays visible (later rows kept, de-emphasized) and the
                # viewport is not auto-scrolled, so the row stays under the
                # cursor. self.log.list.row(item) is the item's index in the
                # FULL (non-shrunk) list, so model.index_for_log_row(row)
                # resolves the same moment regardless of scroll position.
                self.board.update_state(snapshot)
                # FEAT-002: the hovered moment's own per-turn internals drive
                # the board overlays so bars / inference match the previewed
                # snapshot, not the current playback cursor.
                self.board.set_seat_internals(getattr(snapshot, "internals", None))
                self.log.preview_moment(row)

        def _on_log_row_clicked(self, item: Any) -> None:
            # Item 4 (FEAT-003): clicking a log row PINS the board + full log to
            # that recorded turn and switches into the FEAT-002 manual mode, so
            # auto-play stops and does not resume on its own. From there the user
            # advances with the Next turn button (self._step_next) and steps back
            # with Prev (self._step_prev). This mirrors the hover handler's
            # row -> moment mapping (index_for_log_row) so a click resolves the
            # same moment the hover preview showed, regardless of scroll.
            if not self._alive or self.model is None:
                return
            row = self.log.list.row(item)
            index = self.model.index_for_log_row(row)
            if index is None:
                return
            # Clear hover-restore state so a subsequent pointer-Leave restores to
            # the PINNED turn (the model's current index) rather than snapping
            # back to the pre-click auto-play position. Also drop the hover
            # row de-emphasis now that the full log is pinned.
            self._hover_restore_index = None
            self.log.clear_preview()
            # Stop auto-play and pin the board + full log to the clicked turn.
            self._play_timer.stop()
            self.play_button.setText(tr("replay.play"))
            self.model.seek(index)
            self._render_current()
            # Item 5 (FEAT-004): keep the clicked / pinned row highlighted while
            # playback is paused there. _render_current already pins the current
            # step to log_row_for_index(index) (the first row of that moment);
            # re-pin to the actual clicked row so a mid-turn row the user picked
            # stays emphasized rather than snapping to the turn's opening row.
            self.log.set_current_step(row)
            # Force manual mode: setting the slider to its minimum fires
            # _on_speed_changed which (via is_manual_speed) stops the timer and
            # shows the Manual label. The FEAT-002 gating guarantees no code path
            # restarts the timer while the slider sits at the minimum, so
            # playback stays paused until the user steps with Next turn / Prev.
            self.speed_slider.setValue(AI_SPEED_MIN_MS)

        def _end_hover_preview(self) -> None:
            if self._hover_restore_index is None:
                return
            index = self._hover_restore_index
            self._hover_restore_index = None
            if self.model is not None:
                self.model.seek(index)
            self._render_current()
            self.log.clear_preview()

        def eventFilter(self, source: Any, event: Any) -> bool:
            # Restore the playback position when the pointer leaves the log.
            if (
                self._alive
                and source is self.log.list.viewport()
                and event.type() == event.Type.Leave
            ):
                self._end_hover_preview()
            return super().eventFilter(source, event)

        # -- info panel (item 6) -----------------------------------------
        def _ai_descriptor_text(self, kind: str, version: Any) -> str:
            """Localize one seat's AI kind + version into display text."""

            if kind == "random":
                return tr("replay.ai_random")
            if kind == "dummy":
                return tr("replay.ai_dummy")
            if kind == "main":
                shown = "" if version is None else str(version)
                return tr("replay.ai_main", version=shown).strip()
            return tr("replay.ai_unknown")

        def _update_info_panel(self) -> None:
            """Populate the read-only info panel from the loaded record.

            Shows the localized mode label, outcome / winning team, seed and the
            per-seat AI + version breakdown. Clears to empty text when no record
            is loaded (before the first load and after a delete clears the
            board). All extraction stays in the Qt-free replay_store helpers.
            """

            if not self._alive:
                return
            record = self._loaded_record
            if not isinstance(record, dict):
                self.info_panel.setText("")
                return
            summary = replay_store.replay_summary(record)
            lines: list[str] = []
            mode_label = summary.get("mode_label") or summary.get("mode")
            if mode_label:
                lines.append(tr("replay.info_mode", mode=str(mode_label)))
            # FEAT-001: surface the recorded format (full vs public) in the
            # info panel too, not only in the easy-to-miss browser row. Resolve
            # the label through the same Qt-free helper the browser row uses so
            # the two can never disagree.
            format_name = tr(replay_store.replay_format_label_tag(summary))
            lines.append(tr("replay.info_format", format=format_name))
            outcome = summary.get("winning_team") or summary.get("outcome")
            if outcome:
                lines.append(tr("replay.info_outcome", outcome=str(outcome)))
            seed = record.get("seed")
            if seed is not None:
                lines.append(tr("replay.info_seed", seed=str(seed)))
            for _tag, seat, kind, version in replay_store.describe_ai_versions(
                record.get("ai_versions")
            ):
                ai_text = self._ai_descriptor_text(kind, version)
                lines.append(tr("replay.info_seat", seat=seat, ai=ai_text))
            self.info_panel.setText("\n".join(lines))

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
            # Item 2: the BOARD plays from turn 1 via the stepped snapshot, but
            # the LOG shows the ENTIRE episode log from the first frame. Feeding
            # the full-log snapshot (instead of the stepped one) makes every
            # turn visible at once; GameLog.update_state never shrinks on a
            # shorter incoming sequence, so the full log stays stable as the
            # board advances / rewinds. index_for_log_row still resolves hovered
            # rows against full_events, so hover-preview is unaffected.
            self.board.update_state(snapshot)
            # FEAT-002: feed the current snapshot's recorded per-turn internals
            # so the board can draw emotion bars and the role-inference hover
            # overlay. Overlays are enabled for ALL replays (see _render_current
            # / __init__): a replay is a post-hoc review of a finished game, not
            # live hidden-info play, so a human-played replay may also show them.
            self.board.set_seat_internals(getattr(snapshot, "internals", None))
            if self.model is not None:
                self.log.update_state(self.model.full_log_snapshot())
            else:
                self.log.update_state(snapshot)
            # Item 5 (FEAT-004): highlight the currently playing / seeked step in
            # the log so the row the board is showing is always emphasized, like
            # the live game's current-step highlight. ReplayModel.log_row_for_index
            # is the Qt-free inverse of index_for_log_row: it returns the first
            # full-log row representing the current snapshot's moment. Every
            # cursor change (auto-advance, Prev / Next, click, hover-restore)
            # routes through here via _render_current, so the highlight tracks
            # the cursor. set_current_step COEXISTS with the hover preview
            # (FEAT-002 _apply_emphasis), so a hover never erases it.
            if self.model is not None:
                row = self.model.log_row_for_index(self.model.index)
                self.log.set_current_step(row)

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
