"""Interactive game screen for TASK-9-2."""
from __future__ import annotations

from typing import Any

from ..game import GameSessionController
from ..qt_compat import QHBoxLayout, QLabel, QVBoxLayout, QWidget, QT_AVAILABLE, require_qt
from ..widgets.game_widgets import ActionPanel, DebugHUD, GameBoard


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class GameScreen(QWidget):
        def __init__(self, controller: GameSessionController, parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(parent)
            self.controller = controller
            self.setWindowTitle("Bang RL — Game")
            self.status = QLabel(self)
            self.board = GameBoard(self)
            self.actions = ActionPanel(self)
            self.hud = DebugHUD(self, human_player=controller.HUMAN_PLAYER)
            top = QHBoxLayout()
            top.addWidget(self.board)
            top.addWidget(self.hud)
            layout = QVBoxLayout(self)
            layout.addWidget(self.status)
            layout.addLayout(top)
            layout.addWidget(self.actions)
            self.actions.actionRequested.connect(self._submit_action)
            self.actions.pauseRequested.connect(self._toggle_pause)
            controller.on_update = self._render
            self._render(controller.snapshot)

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
            if snapshot.state is not None:
                self.board.update_state(snapshot.state)
                self.hud.update_state(snapshot.state)
                self.actions.update_mask(snapshot.state.action_mask, paused=snapshot.paused, active=snapshot.active)
                status = snapshot.state.termination_status
                self.status.setText(
                    f"Step {snapshot.state.environment_step} · turn Player {snapshot.state.current_player} · {status}"
                )
            else:
                self.actions.update_mask((0, 0, 0, 0, 0, 0), paused=snapshot.paused, active=snapshot.active)
                self.status.setText("No game session")
            self.actions.set_message(snapshot.message)

        def closeEvent(self, event: Any) -> None:
            self.controller.close()
            event.accept()

else:

    class GameScreen:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["GameScreen"]
