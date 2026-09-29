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


# AI-turn-speed slider bounds, in milliseconds per AI turn. Defined outside the
# Qt gate so they can be referenced/tested headlessly. Default is ~1s per turn.
AI_SPEED_MIN_MS = 200
AI_SPEED_MAX_MS = 3000
AI_SPEED_DEFAULT_MS = 1000


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
            self.speed_slider.setValue(AI_SPEED_DEFAULT_MS)
            self.speed_slider.setSingleStep(100)
            self.speed_slider.valueChanged.connect(self._on_speed_changed)
            self._update_speed_label(AI_SPEED_DEFAULT_MS)

            # Left = log, center = graphical board.
            top = QHBoxLayout()
            top.addWidget(self.log, 1)
            top.addWidget(self.board, 3)

            speed_row = QHBoxLayout()
            speed_row.addWidget(self.speed_title)
            speed_row.addWidget(self.speed_label)
            speed_row.addWidget(self.speed_slider, 1)

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
            layout.addLayout(speed_row)
            layout.addWidget(self.actions)
            layout.addWidget(self.status)

            # Item 6: a spectator human never banks/passes, so the board's
            # action clicks and the ActionPanel buttons are not wired to submit
            # an action. The AI-speed slider and log hover stay fully active.
            if not self._spectator:
                self.board.actionRequested.connect(self._submit_action)
                self.actions.actionRequested.connect(self._submit_action)
                self.actions.pauseRequested.connect(self._toggle_pause)
            else:
                # Keep pause available so the viewer can freeze the AI cadence.
                self.actions.pauseRequested.connect(self._toggle_pause)

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
            if self._spectator:
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
            if self._ai_timer.isActive():
                # Apply the new cadence live to the running AI sequence.
                self._ai_timer.setInterval(int(value))

        def _update_speed_label(self, value: int) -> None:
            self.speed_label.setText(tr("game.ai_speed_value", seconds=int(value) / 1000.0))

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
            if self.controller.is_ai_turn():
                self._ai_timer.setInterval(int(self.speed_slider.value()))
                self._ai_timer.start()
            else:
                self._ai_timer.stop()

        def _advance_one_ai_turn(self) -> None:
            if not self._alive:
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
            """Build a ReplayModel over the recorded per-turn observations."""

            observations = getattr(self.controller, "_replay_observations", None)
            if not observations:
                return None
            try:
                return ReplayModel(list(observations))
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
            self._log_hover_preview = True
            self.board.update_state(snapshot)
            self.log.update_state(snapshot)

        def _end_log_hover_preview(self) -> None:
            if not self._alive or not self._log_hover_preview:
                return
            self._log_hover_preview = False
            # Restore whatever the controller is currently showing (the live
            # turn, or a committed time-travel preview if a row was clicked).
            self._render(self.controller.snapshot)

        def _on_log_row_clicked(self, item: Any) -> None:
            if not self._alive or item is None:
                return
            row = self.log.list.row(item)
            self._log_hover_preview = False
            self._ai_timer.stop()
            # Item 5: in a human game a click forks a live, continuable session
            # from that moment (restore_to_moment; falls back to a read-only
            # review when the service lacks restore support). A spectator game
            # only reviews the moment (time_travel_to) since the human never
            # plays. Both are Qt-free controller calls.
            if self._spectator:
                snapshot = self.controller.time_travel_to(row)
            else:
                snapshot = self.controller.restore_to_moment(row)
            self._render(snapshot)
            # If the fork resumed a live game with AI seats to act, restart the
            # AI cadence so play continues from the restored moment.
            self._maybe_start_ai_sequence()

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


__all__ = ["AI_SPEED_DEFAULT_MS", "AI_SPEED_MAX_MS", "AI_SPEED_MIN_MS", "GameScreen"]
