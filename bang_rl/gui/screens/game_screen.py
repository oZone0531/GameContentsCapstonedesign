"""Interactive game screen for TASK-9-2."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..game import GameSessionController
from ..i18n import tr
from ..qt_compat import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from ..widgets.game_widgets import ActionPanel, DebugHUD, GameBoard, GameLog


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
            # The board | HUD | log panels sit side by side in the top row, so
            # give the screen enough width for the Opponents panel to show in
            # full without horizontal scrolling. A minimum width (rather than a
            # hardcoded top-level geometry) keeps MainWindow._set_screen's
            # adjustSize() behavior intact.
            self.setMinimumWidth(1024)
            self.back_button = QPushButton(tr("common.back"), self)
            self.back_button.clicked.connect(self._go_back)
            self.status = QLabel(self)
            self.board = GameBoard(self, human_player=controller.HUMAN_PLAYER)
            self.actions = ActionPanel(self)
            self.hud = DebugHUD(self, human_player=controller.HUMAN_PLAYER)
            self.log = GameLog(self)
            top = QHBoxLayout()
            top.addWidget(self.board)
            top.addWidget(self.hud)
            top.addWidget(self.log)
            layout = QVBoxLayout(self)
            layout.addWidget(self.back_button)
            layout.addWidget(self.status)
            layout.addLayout(top)
            layout.addWidget(self.actions)
            self.actions.actionRequested.connect(self._submit_action)
            self.actions.pauseRequested.connect(self._toggle_pause)
            controller.on_update = self._render
            self._render(controller.snapshot)

        def _go_back(self) -> None:
            self._detach()
            self.controller.close()
            if self.on_back is not None:
                self.on_back()

        def _detach(self) -> None:
            """Stop rendering into this (about to be torn down) screen.

            ``setCentralWidget`` deletes the underlying C++ objects of the
            previous central widget, so any later ``controller.on_update``
            publish must not touch this screen's now-dead widgets.
            """

            self._alive = False
            if getattr(self.controller, "on_update", None) is self._render:
                self.controller.on_update = None

        def _submit_action(self, action: int) -> None:
            try:
                snapshot = self.controller.submit_human_action(action)
            except Exception:
                snapshot = self.controller.snapshot
            self._render(snapshot)

        def _toggle_pause(self) -> None:
            snapshot = self.controller.resume() if self.controller.paused else self.controller.pause()
            self._render(snapshot)

        def _render(self, snapshot: Any) -> None:
            if not self._alive:
                return
            if snapshot.state is not None:
                self.board.update_state(snapshot.state)
                self.hud.update_state(snapshot.state)
                self.log.update_state(snapshot.state)
                self.actions.update_mask(snapshot.state.action_mask, paused=snapshot.paused, active=snapshot.active)
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


__all__ = ["GameScreen"]
