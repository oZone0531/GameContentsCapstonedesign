"""Interactive graphical game screen.

The human plays on a graphical board (five gun-holding circles, the human seat
at the bottom) with the game log on the left.  Clicking an opponent circle
BANGs it; clicking the human's own circle passes.  After the human acts, AI
turns play out one at a time on a QTimer whose delay is set by the bottom
AI-turn-speed slider, so the player can watch who banged whom before control
returns.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..game import GameSessionController
from ..i18n import tr
from ..playback_speed import (
    AI_SPEED_DEFAULT_MS,
    AI_SPEED_MAX_MS,
    AI_SPEED_MIN_MS,
    is_manual_speed,
    play_pause_is_playing,
    speed_to_interval_ms,
)
from ..qt_compat import (
    QHBoxLayout,
    QLabel,
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
from ..widgets.game_widgets import ActionPanel, GameLog, GraphicalBoard


# AI-turn-speed slider bounds, in milliseconds per AI turn. The single source of
# truth now lives in bang_rl/gui/playback_speed.py (a Qt-free module so the pure
# slider/manual-mode test can path-load it headlessly); the names are re-exported
# here so replay_screen's existing `from .game_screen import AI_SPEED_...` keeps
# working. See __all__ at the bottom of the module.


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class GameScreen(QWidget):
        def __init__(
            self,
            controller: GameSessionController,
            parent: QWidget | None = None,
            *,
            on_back: Callable[[], None] | None = None,
            on_restart: Callable[[], None] | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.controller = controller
            self.on_back = on_back
            self.on_restart = on_restart
            self._alive = True
            # Item 6: in a spectator (all-AI) game the human can only watch, so
            # the action buttons are never enabled and board clicks are ignored.
            self._spectator = bool(getattr(controller.mode, "is_spectator", False))
            # True while the pointer hovers a log row and the board is showing
            # that recorded moment as a read-only preview (item 8).
            self._log_hover_preview = False
            # Item 4 (FEAT-003): when a terminal SPECTATOR game is being
            # reviewed (the user clicked a log step after game-over), this holds
            # a ReplayModel over the recorded observations + internals and drives
            # the prev/pause-play/next controls. None means not in review mode
            # (live play / human game). Set in _enter_review_mode.
            self._review_model: ReplayModel | None = None
            # Tracks whether the last _render saw a terminal snapshot, so a
            # spectator log click after game-over enters review mode.
            self._game_over = False
            self.setWindowTitle(tr("game.title"))
            self.setMinimumWidth(1024)

            # ITEM 2: a spectator (all-AI) game has no human, so the board is
            # built with human_player=None and renders no dedicated "나"/"You"
            # seat; seat 0 is captioned as an ordinary Player like the rest.
            board_human_player = None if self._spectator else controller.HUMAN_PLAYER
            self.board = GraphicalBoard(self, human_player=board_human_player)
            self.log = GameLog(self)
            self.status = QLabel(self)
            # Transient hit / elimination status line shown during the shot
            # animation, plus a click hint shown on the human's own turn.
            self.event_status = QLabel(self)
            self.hint = QLabel(self)
            self.actions = ActionPanel(self)
            self.board.playerHit.connect(self._on_player_hit)
            self.board.playerEliminated.connect(self._on_player_eliminated)

            # AI turns advance one at a time on this timer; the slider controls
            # its interval. Guarded by _alive so a torn-down screen never
            # renders into deleted C++ objects.
            self._ai_timer = QTimer(self)
            self._ai_timer.setSingleShot(False)
            self._ai_timer.timeout.connect(self._advance_one_ai_turn)

            self.back_button = QPushButton(tr("common.back"), self)
            self.back_button.clicked.connect(self._go_back)

            # Item 7: Restart / Return-to-menu buttons shown under the board
            # only once the game is over (terminated or truncated), for both
            # human and spectator games. Hidden while a game is in progress.
            self.restart_button = QPushButton(tr("game.restart"), self)
            self.restart_button.clicked.connect(self._on_restart_clicked)
            self.return_menu_button = QPushButton(tr("game.return_menu"), self)
            self.return_menu_button.clicked.connect(self._go_back)
            self.restart_button.setVisible(False)
            self.return_menu_button.setVisible(False)

            # Static descriptive label for the slider ("AI turn speed").
            self.speed_title = QLabel(tr("game.ai_speed"), self)
            self.speed_label = QLabel(self)
            self.speed_slider = QSlider(Qt.Orientation.Horizontal, self)
            self.speed_slider.setMinimum(AI_SPEED_MIN_MS)
            self.speed_slider.setMaximum(AI_SPEED_MAX_MS)
            # Item 3 (FEAT-003): BOTH spectator and human games now start in
            # MANUAL playback (slider pinned at the minimum). The manual-mode
            # gating (is_manual_speed / speed_to_interval_ms -> None) then makes
            # _maybe_start_ai_sequence a no-op, so entering a game is paused and
            # the user advances AI turns with the Next-turn button. The user can
            # drag the slider off the minimum at any time to auto-advance.
            self.speed_slider.setValue(AI_SPEED_MIN_MS)
            self.speed_slider.setSingleStep(100)
            self.speed_slider.valueChanged.connect(self._on_speed_changed)
            # Item 3 (FEAT-002): a static "Manual" tick label pinned at the
            # slider minimum, plus a Next-turn button that advances exactly one
            # AI turn per press while in manual mode. The button is enabled only
            # in manual mode (toggled in _on_speed_changed / _maybe_start_ai_
            # sequence); above the minimum the cadence is driven by the timer.
            self.speed_manual_label = QLabel(tr("game.speed_manual_label"), self)
            self.next_turn_button = QPushButton(tr("game.next_turn"), self)
            self.next_turn_button.clicked.connect(self._manual_advance)
            # Item 3 (FEAT-003): spectator playback controls (prev turn /
            # pause-play / next turn) mirror the replay screen. They are only
            # added to the layout in spectator mode (see below); the next-turn
            # button is shared with manual human play. The prev/pause-play
            # buttons drive the FEAT-003 terminal review / recorded-turn
            # stepping. Created unconditionally so the attributes always exist.
            self.prev_turn_button = QPushButton(tr("game.prev_turn"), self)
            self.prev_turn_button.clicked.connect(self._spectator_prev_turn)
            self.play_pause_button = QPushButton(tr("game.play"), self)
            self.play_pause_button.clicked.connect(self._toggle_play_pause)
            self._update_speed_label(AI_SPEED_MIN_MS)
            self._update_manual_controls(AI_SPEED_MIN_MS)

            # Left = log, center = graphical board.
            top = QHBoxLayout()
            top.addWidget(self.log, 1)
            top.addWidget(self.board, 3)

            speed_row = QHBoxLayout()
            speed_row.addWidget(self.speed_title)
            speed_row.addWidget(self.speed_label)
            speed_row.addWidget(self.speed_manual_label)
            speed_row.addWidget(self.speed_slider, 1)
            # Item 3 (FEAT-003): in a human game the Next-turn button lives in
            # the speed row next to the slider (manual mode). In a spectator game
            # it moves into the dedicated playback-control row (prev / pause-play
            # / next) below, so it is not added here.
            if not self._spectator:
                speed_row.addWidget(self.next_turn_button)

            # Item 3 (FEAT-003): a spectator game gets a replay-style control row
            # (prev turn / pause-play / next turn) and the ActionPanel bang/pass
            # buttons are hidden entirely. These mirror the replay screen's
            # controls (same i18n labels, manual default, same semantics). A
            # human game keeps the ActionPanel and never shows this row.
            self.spectator_controls_row = QHBoxLayout()
            self.spectator_controls_row.addStretch(1)
            self.spectator_controls_row.addWidget(self.prev_turn_button)
            self.spectator_controls_row.addWidget(self.play_pause_button)
            self.spectator_controls_row.addWidget(self.next_turn_button)
            self.spectator_controls_row.addStretch(1)

            # Item 7: game-over button row sits beneath the board.
            self.game_over_row = QHBoxLayout()
            self.game_over_row.addStretch(1)
            self.game_over_row.addWidget(self.restart_button)
            self.game_over_row.addWidget(self.return_menu_button)
            self.game_over_row.addStretch(1)

            layout = QVBoxLayout(self)
            layout.addWidget(self.back_button)
            layout.addLayout(top, 1)
            layout.addWidget(self.hint)
            layout.addWidget(self.event_status)
            layout.addLayout(self.game_over_row)
            if self._spectator:
                # Item 3 (FEAT-003): spectator playback controls replace the
                # ActionPanel. The AI-speed slider stays visible so the viewer
                # can leave manual mode.
                layout.addLayout(self.spectator_controls_row)
            layout.addLayout(speed_row)
            layout.addWidget(self.actions)
            layout.addWidget(self.status)

            # Item 6: a spectator human never banks/passes, so the board's
            # action clicks and the ActionPanel buttons are not wired to submit
            # an action. The AI-speed slider and log hover stay fully active.
            # Item 3 (FEAT-003): in spectator mode the ActionPanel (bang/pass
            # buttons) is hidden entirely; its pause control is replaced by the
            # play_pause_button in the spectator control row.
            if not self._spectator:
                self.board.actionRequested.connect(self._submit_action)
                self.actions.actionRequested.connect(self._submit_action)
                self.actions.pauseRequested.connect(self._toggle_pause)
            else:
                self.actions.setVisible(False)

            # Item 8: hovering a left-log row previews that moment on the board;
            # clicking a row rewinds the game to that moment so play continues
            # from there. Mirrors the ReplayScreen log-hover wiring.
            self.log.list.setMouseTracking(True)
            self.log.list.itemEntered.connect(self._on_log_row_hovered)
            self.log.list.itemClicked.connect(self._on_log_row_clicked)
            self.log.list.viewport().installEventFilter(self)

            controller.on_update = self._render
            self._render(controller.snapshot)
            # Item 6: a spectator game has no human action to kick off the AI
            # cadence, so start driving the all-AI turns right away.
            # Item 1 (FEAT-003): a HUMAN game whose first turn is an AI seat
            # (Sheriff-first) also needs the cadence kicked off here so the
            # initial AI run-up ANIMATES one turn at a time into the human's
            # first turn (the controller was started with advance_ai=False so it
            # is still parked on the AI turn). When the human is genuinely first,
            # is_ai_turn() is False and the timer stays stopped (no double-run).
            self._maybe_start_ai_sequence()

        # -- navigation --------------------------------------------------
        def _go_back(self) -> None:
            self._detach()
            self.controller.close()
            if self.on_back is not None:
                self.on_back()

        def _detach(self) -> None:
            """Stop rendering/timing into this (about to be torn down) screen.

            ``setCentralWidget`` deletes the underlying C++ objects of the
            previous central widget, so any later ``controller.on_update``
            publish or timer callback must not touch this screen's now-dead
            widgets.
            """

            self._alive = False
            self._ai_timer.stop()
            # Stop the board's own shot-animation timer so a late frame never
            # repaints this soon-to-be-deleted widget.
            self.board.detach()
            if getattr(self.controller, "on_update", None) is self._render:
                self.controller.on_update = None

        # -- speed slider ------------------------------------------------
        def _on_speed_changed(self, value: int) -> None:
            self._update_speed_label(value)
            self._update_manual_controls(value)
            if is_manual_speed(value):
                # Item 3 (FEAT-002): entering manual mode stops auto-advance;
                # the user drives turns with the Next-turn button. Guard with
                # _alive so a torn-down screen never touches dead C++ objects.
                if self._alive:
                    self._ai_timer.stop()
                    self._update_play_pause_label()
                return
            # Leaving manual mode (or just changing a non-manual value): if it
            # is an AI turn, (re)start/resume the timer at the new interval.
            interval = speed_to_interval_ms(value)
            if interval is None:  # pragma: no cover - manual handled above
                return
            if self._ai_timer.isActive():
                # Apply the new cadence live to the running AI sequence.
                self._ai_timer.setInterval(interval)
            elif self._review_model is not None:
                # Item 4 (FEAT-003): leaving manual while reviewing a terminal
                # spectator game resumes auto-advance of the review model (not a
                # live AI turn) at the new interval, unless we are at the end.
                if not self._review_model.is_at_end():
                    self._ai_timer.setInterval(interval)
                    self._ai_timer.start()
            else:
                self._maybe_start_ai_sequence()
            # Item 3 (FEAT-003): keep the spectator pause-play label in sync with
            # the (possibly now running) timer.
            self._update_play_pause_label()

        def _update_speed_label(self, value: int) -> None:
            # At the minimum tick the label reads "Manual" instead of a per-turn
            # seconds value, so the user sees that auto-advance is off.
            if is_manual_speed(value):
                self.speed_label.setText(tr("game.ai_speed_manual"))
            else:
                self.speed_label.setText(
                    tr("game.ai_speed_value", seconds=int(value) / 1000.0)
                )

        def _update_manual_controls(self, value: int) -> None:
            """Enable the Next-turn button (human: only in manual mode).

            Item 3 (FEAT-003): in SPECTATOR mode the prev / next controls stay
            enabled regardless of the slider position so the viewer can always
            step the recording / review (matching the replay screen, which keeps
            its prev / next buttons enabled). In a human game the Next-turn
            button is only meaningful in manual mode, so it is enabled only then.
            """

            if self._spectator:
                self.next_turn_button.setEnabled(True)
                self.prev_turn_button.setEnabled(True)
                return
            self.next_turn_button.setEnabled(is_manual_speed(value))

        def _manual_advance(self) -> None:
            """Advance exactly one turn per press (Next-turn button).

            Item 4 (FEAT-003): while reviewing a terminal spectator game the
            Next-turn button steps the recorded ReplayModel forward one turn and
            re-renders that past snapshot (with its own internals). Otherwise it
            mirrors a single _ai_timer tick but is user-driven: it only acts when
            the slider is at the manual (minimum) position and it is an AI turn,
            advancing one turn via step_single_ai_turn().
            """

            if not self._alive:
                return
            if self._review_model is not None:
                self._review_step(forward=True)
                return
            if not is_manual_speed(self.speed_slider.value()):
                return
            if not self.controller.is_ai_turn():
                return
            try:
                snapshot = self.controller.step_single_ai_turn()
            except Exception:
                snapshot = self.controller.snapshot
            self._render(snapshot)

        # -- spectator playback controls (item 3) ------------------------
        def _spectator_prev_turn(self) -> None:
            """Prev-turn control: step one recorded turn BACKWARD (item 3).

            When reviewing (terminal spectator, or after any spectator log
            click) this steps the recorded ReplayModel back one turn. If no
            review model exists yet it builds one seeded at the current live
            tail, so pressing Prev on a live spectator game begins a backward
            review from the latest recorded turn.
            """

            if not self._alive or not self._spectator:
                return
            if self._review_model is None:
                model = self._log_model()
                if model is None or model.is_empty():
                    return
                self._ai_timer.stop()
                self._review_model = model
                # Seed at the live tail so the first Prev steps back from "now".
                model.seek(len(model) - 1)
            self._review_step(forward=False)

        def _toggle_play_pause(self) -> None:
            """Pause-play control for spectator mode (item 3).

            Mirrors the replay screen's pause-play semantics via the shared
            Qt-free play_pause_is_playing helper: while the AI timer runs the
            button reads Pause and a click stops it; while stopped a click
            resumes auto-advance at the slider interval. In manual mode (slider
            at the minimum) play is a no-op and the label stays Play, matching
            replay. In terminal review mode play auto-advances the review via the
            same timer.
            """

            if not self._alive or not self._spectator:
                return
            if play_pause_is_playing(self._ai_timer.isActive()):
                self._ai_timer.stop()
                self._update_play_pause_label()
                return
            interval = speed_to_interval_ms(self.speed_slider.value())
            if interval is None:
                # Manual mode: play does nothing, label stays Play.
                self._update_play_pause_label()
                return
            if self._review_model is not None:
                self._ai_timer.setInterval(interval)
                self._ai_timer.start()
            else:
                self._maybe_start_ai_sequence()
            self._update_play_pause_label()

        def _update_play_pause_label(self) -> None:
            if not self._spectator:
                return
            playing = play_pause_is_playing(self._ai_timer.isActive())
            self.play_pause_button.setText(
                tr("game.pause") if playing else tr("game.play")
            )

        def _review_step(self, *, forward: bool) -> None:
            """Step the terminal-review ReplayModel one turn and re-render.

            Item 4 (FEAT-003): advances / rewinds the recorded model, renders the
            sought snapshot on the board, re-supplies that turn's internals so
            emotion bars + role inference show for the past turn, and highlights
            the current step in the log.
            """

            model = self._review_model
            if model is None or model.is_empty():
                return
            snapshot = model.next() if forward else model.prev()
            if snapshot is None:
                return
            self._render_review_snapshot(snapshot)

        def _render_review_snapshot(self, snapshot: Any) -> None:
            """Render one review snapshot: board + internals + step highlight."""

            model = self._review_model
            if model is None:
                return
            self.board.update_state(snapshot)
            # Item 4: re-supply the past turn's own internals so the overlay
            # (emotion bars + role inference) shows instead of "no inference".
            self.board.set_seat_internals(getattr(snapshot, "internals", None))
            # Item 5: highlight the reviewed step in the log.
            self.log.set_current_step(model.index)

        # -- transient hit / elimination status --------------------------
        def _on_player_hit(self, player: int) -> None:
            if not self._alive:
                return
            self.event_status.setText(tr("board.hit", player=int(player)))

        def _on_player_eliminated(self, player: int) -> None:
            if not self._alive:
                return
            self.event_status.setText(tr("board.eliminated", player=int(player)))

        # -- game over (item 7) ------------------------------------------
        def _on_restart_clicked(self) -> None:
            self._detach()
            self.controller.close()
            if self.on_restart is not None:
                self.on_restart()

        # -- actions -----------------------------------------------------
        def _submit_action(self, action: int) -> None:
            # Item 6: guard so a spectator can never influence the game even if
            # a stray signal reaches here.
            if self._spectator:
                return
            try:
                # advance_ai=False: this screen drives the AI cadence itself so
                # each AI turn is shown one at a time via the QTimer below.
                snapshot = self.controller.submit_human_action(action, advance_ai=False)
            except Exception:
                snapshot = self.controller.snapshot
            self._render(snapshot)
            self._maybe_start_ai_sequence()

        def _toggle_pause(self) -> None:
            if self.controller.paused:
                snapshot = self.controller.resume()
                self._render(snapshot)
                self._maybe_start_ai_sequence()
            else:
                self._ai_timer.stop()
                snapshot = self.controller.pause()
                self._render(snapshot)

        def _maybe_start_ai_sequence(self) -> None:
            if not self._alive:
                return
            # Item 3 (FEAT-002): in manual mode (slider at minimum) never start
            # the timer; the user advances AI turns with the Next-turn button.
            interval = speed_to_interval_ms(self.speed_slider.value())
            if interval is None:
                self._ai_timer.stop()
                return
            if self.controller.is_ai_turn():
                self._ai_timer.setInterval(interval)
                self._ai_timer.start()
            else:
                self._ai_timer.stop()

        def _advance_one_ai_turn(self) -> None:
            if not self._alive:
                return
            # Item 4 (FEAT-003): in terminal review mode the timer auto-advances
            # the recorded ReplayModel (pause-play "playing") rather than driving
            # a live AI turn. Stop at the last recorded turn.
            if self._review_model is not None:
                model = self._review_model
                if model.is_at_end():
                    self._ai_timer.stop()
                    self._update_play_pause_label()
                    return
                self._review_step(forward=True)
                if model.is_at_end():
                    self._ai_timer.stop()
                    self._update_play_pause_label()
                return
            if not self.controller.is_ai_turn():
                self._ai_timer.stop()
                return
            try:
                snapshot = self.controller.step_single_ai_turn()
            except Exception:
                snapshot = self.controller.snapshot
            self._render(snapshot)
            if not self.controller.is_ai_turn():
                self._ai_timer.stop()

        # -- log-row time travel (item 8) --------------------------------
        def _log_model(self) -> ReplayModel | None:
            """Build a ReplayModel over the recorded observations + internals.

            Item 4 (FEAT-003): the model carries the recorded per-turn internals
            index-parallel (ReplayModel._attach_internals) so a hovered / sought
            past turn can re-supply its own emotion bars + role inference to the
            board. Mirrors replay_screen's internals-aware model so the spectator
            hover / terminal review shows the same overlays a replay would.
            """

            observations = getattr(self.controller, "_replay_observations", None)
            if not observations:
                return None
            internals = getattr(self.controller, "_replay_internals", None)
            try:
                return ReplayModel(
                    {
                        "observations": list(observations),
                        "internals": list(internals) if internals else [],
                    }
                )
            except (TypeError, ValueError):
                return None

        def _on_log_row_hovered(self, item: Any) -> None:
            if not self._alive or item is None:
                return
            model = self._log_model()
            if model is None or model.is_empty():
                return
            row = self.log.list.row(item)
            index = model.index_for_log_row(row)
            if index is None:
                return
            snapshot = model.snapshot_at(index)
            if snapshot is None:
                return
            # Read-only preview of the hovered moment; the live view is
            # restored when the pointer leaves the log (eventFilter Leave).
            # Items 2 & 3: preview the moment on the BOARD only and highlight the
            # hovered row in the log via preview_moment -- the full log stays
            # visible (later rows are kept, just de-emphasized) and the viewport
            # is NOT auto-scrolled, so the row stays under the cursor. The step
            # the row maps to is unchanged: self.log.list.row(item) is the item's
            # index in the FULL (non-shrunk) list, so model.index_for_log_row(row)
            # resolves the same moment regardless of scroll position. A CLICK (see
            # _on_log_row_clicked) is what actually rewinds/forks the game from
            # this moment and drops the later steps.
            self._log_hover_preview = True
            self.board.update_state(snapshot)
            # Item 4 (FEAT-003): re-supply the hovered turn's OWN internals so
            # the overlay shows emotion bars + role inference for that past turn.
            # update_state resets _seat_internals to None (clearing them), so
            # without this the overlay falls back to "no inference info" after a
            # hover (and after the game ends). For a human game overlays are
            # disabled so set_seat_internals is a safe no-op. Mirrors
            # replay_screen._on_log_row_hovered.
            self.board.set_seat_internals(getattr(snapshot, "internals", None))
            self.log.preview_moment(row)

        def _end_log_hover_preview(self) -> None:
            if not self._alive or not self._log_hover_preview:
                return
            self._log_hover_preview = False
            # Item 4 (FEAT-003): when reviewing a (terminal) spectator game,
            # leaving the log restores the PINNED review turn (board + its own
            # internals + step highlight) rather than snapping back to the live
            # terminal snapshot. Otherwise restore whatever the controller is
            # currently showing (the live turn or a committed time-travel view).
            if self._review_model is not None:
                snapshot = self._review_model.current()
                if snapshot is not None:
                    self._render_review_snapshot(snapshot)
                self.log.clear_preview()
                return
            self._render(self.controller.snapshot)
            self.log.clear_preview()

        def _on_log_row_clicked(self, item: Any) -> None:
            if not self._alive or item is None:
                return
            row = self.log.list.row(item)
            self._log_hover_preview = False
            self._ai_timer.stop()
            if self._spectator:
                # Item 4 (FEAT-003): clicking a log row in a spectator game
                # enters a replay-like REVIEW of the recorded turns. After the
                # game is over this is the only way to look back at past turns
                # (the live session is terminal); during live play it pins the
                # view to that recorded turn and the prev/pause-play/next
                # controls step through the recording. The ReplayModel carries
                # per-turn internals so emotion bars + role inference show for
                # the sought turn (fixing the "no inference info" bug).
                self._enter_review_mode(row)
                return
            # Item 2 (FEAT-003): a HUMAN click forks a live, continuable session
            # from that moment. restore_to_moment truncates the replay buffers to
            # [:row+1]; advance_ai=False so the restored AI run-up animates one
            # turn at a time on the QTimer below instead of fast-forwarding.
            snapshot = self.controller.restore_to_moment(row, advance_ai=False)
            self._render(snapshot)
            # Item 2 (FEAT-003): the resume-click FORKS the timeline, so the log
            # rows AFTER the restored step never happened and must be deleted.
            # _render re-grows the log from the (now shorter) observation, but
            # GameLog.update_state never SHRINKS (that is the hover-preview
            # contract), so explicitly truncate the view to the restored row.
            self.log.truncate_after(row)
            # Item 5: pin the current-step highlight to the restored (now last)
            # step after truncation.
            self.log.set_current_step(row)
            # If the fork resumed a live game with AI seats to act, restart the
            # AI cadence so play continues from the restored moment.
            self._maybe_start_ai_sequence()

        def _enter_review_mode(self, row: int) -> None:
            """Enter replay-like review of recorded spectator turns (item 4)."""

            model = self._log_model()
            if model is None or model.is_empty():
                # No recording to review (e.g. empty buffers): fall back to the
                # Qt-free time-travel preview so the click is not a dead no-op.
                snapshot = self.controller.time_travel_to(row)
                self._render(snapshot)
                self.log.set_current_step(row)
                return
            index = model.index_for_log_row(row)
            if index is None:
                return
            self._review_model = model
            snapshot = model.seek(index)
            if snapshot is None:
                return
            self._update_play_pause_label()
            self._render_review_snapshot(snapshot)

        def eventFilter(self, source: Any, event: Any) -> bool:
            if (
                self._alive
                and source is self.log.list.viewport()
                and event.type() == event.Type.Leave
            ):
                self._end_log_hover_preview()
            return super().eventFilter(source, event)

        # -- rendering ---------------------------------------------------
        def _render(self, snapshot: Any) -> None:
            if not self._alive:
                return
            if snapshot.state is not None:
                self.board.update_state(snapshot.state)
                # FEAT-002: feed live per-seat internals to the board ONLY in
                # spectator mode (overlays are already enabled there because the
                # spectator board has human_player=None). In a human live game
                # this is skipped and the internals stay None, so no opponent
                # emotion / inferred role is ever shown to the human player.
                if self._spectator:
                    self.board.set_seat_internals(self.controller.live_internals())
                self.log.update_state(snapshot.state)
                # Item 6: in a spectator game the action buttons stay disabled
                # (active=False) so the human can never bang/pass.
                self.actions.update_mask(
                    snapshot.state.action_mask,
                    paused=snapshot.paused,
                    active=snapshot.active and not self._spectator,
                )
                if self._spectator:
                    # Item 6: show a spectator hint instead of the click hint.
                    self.hint.setText(tr("game.spectating"))
                else:
                    # Show the click hint only while it is the human's live turn.
                    is_human_turn = (
                        snapshot.active
                        and not snapshot.paused
                        and snapshot.state.current_player == self.controller.HUMAN_PLAYER
                    )
                    self.hint.setText(tr("game.click_hint") if is_human_turn else "")
                status = snapshot.state.termination_status
                self.status.setText(
                    tr(
                        "game.status_line",
                        step=snapshot.state.environment_step,
                        player=snapshot.state.current_player,
                        status=status,
                    )
                )
                # Item 7: reveal the Restart / Return-to-menu buttons only when
                # the game is over (terminated or truncated).
                game_over = bool(
                    getattr(snapshot.state, "terminated", False)
                    or getattr(snapshot.state, "truncated", False)
                )
                # Item 4 (FEAT-003): remember game-over so a later spectator log
                # click knows to enter review mode (the live session is now
                # terminal and can only be reviewed, not continued).
                self._game_over = game_over
                self.restart_button.setVisible(game_over)
                self.return_menu_button.setVisible(game_over)
            else:
                self.actions.update_mask((0, 0, 0, 0, 0, 0), paused=snapshot.paused, active=snapshot.active)
                self.hint.setText("")
                self.status.setText(tr("game.no_session"))
                self.restart_button.setVisible(False)
                self.return_menu_button.setVisible(False)
            self.actions.set_message(snapshot.message)

        def closeEvent(self, event: Any) -> None:
            self._detach()
            self.controller.close()
            event.accept()

else:

    class GameScreen:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


# Re-export the slider bounds (sourced from bang_rl.gui.playback_speed) so
# replay_screen's existing `from .game_screen import AI_SPEED_...` keeps working.
__all__ = ["AI_SPEED_DEFAULT_MS", "AI_SPEED_MAX_MS", "AI_SPEED_MIN_MS", "GameScreen"]
