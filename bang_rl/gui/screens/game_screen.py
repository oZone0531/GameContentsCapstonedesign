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
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.controller = controller
            self.on_back = on_back
            self._alive = True
            self.setWindowTitle(tr("game.title"))
            self.setMinimumWidth(1024)

            self.board = GraphicalBoard(self, human_player=controller.HUMAN_PLAYER)
            self.log = GameLog(self)
            self.status = QLabel(self)
            self.actions = ActionPanel(self)

            # AI turns advance one at a time on this timer; the slider controls
            # its interval. Guarded by _alive so a torn-down screen never
            # renders into deleted C++ objects.
            self._ai_timer = QTimer(self)
            self._ai_timer.setSingleShot(False)
            self._ai_timer.timeout.connect(self._advance_one_ai_turn)

            self.back_button = QPushButton(tr("common.back"), self)
            self.back_button.clicked.connect(self._go_back)

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
            speed_row.addWidget(self.speed_label)
            speed_row.addWidget(self.speed_slider, 1)

            layout = QVBoxLayout(self)
            layout.addWidget(self.back_button)
            layout.addLayout(top, 1)
            layout.addLayout(speed_row)
            layout.addWidget(self.actions)
            layout.addWidget(self.status)

            self.board.actionRequested.connect(self._submit_action)
            self.actions.actionRequested.connect(self._submit_action)
            self.actions.pauseRequested.connect(self._toggle_pause)
            controller.on_update = self._render
            self._render(controller.snapshot)

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

        # -- actions -----------------------------------------------------
        def _submit_action(self, action: int) -> None:
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

        # -- rendering ---------------------------------------------------
        def _render(self, snapshot: Any) -> None:
            if not self._alive:
                return
            if snapshot.state is not None:
                self.board.update_state(snapshot.state)
                self.log.update_state(snapshot.state)
                self.actions.update_mask(
                    snapshot.state.action_mask, paused=snapshot.paused, active=snapshot.active
                )
                status = snapshot.state.termination_status
                self.status.setText(
                    tr(
                        "game.status_line",
                        step=snapshot.state.environment_step,
                        player=snapshot.state.current_player,
                        status=status,
                    )
                )
            else:
                self.actions.update_mask((0, 0, 0, 0, 0, 0), paused=snapshot.paused, active=snapshot.active)
                self.status.setText(tr("game.no_session"))
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
