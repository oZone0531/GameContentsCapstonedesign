"""PyQt6 game widgets fed exclusively by public DTO snapshots."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..i18n import tr
from ..qt_compat import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Qt,
    QT_AVAILABLE,
    pyqtSignal,
    require_qt,
)
from ..roles import hidden_role_label, role_name


def opponent_public_rows(state: Any, *, human_player: int = 0) -> tuple[dict[str, Any], ...]:
    """Build a privacy-safe four-opponent projection from a public DTO."""

    observation = state.observation
    healths = _sequence(observation.get("healths"), 5, 0)
    alive = _sequence(observation.get("alive"), 5, False)
    visibility = _sequence(observation.get("role_visibility"), 5, 0)
    revealed = _sequence(observation.get("revealed_roles"), 5, 255)
    rows = []
    for player in range(5):
        if player == human_player:
            continue
        known_role = int(revealed[player]) if int(visibility[player]) >= 2 and int(revealed[player]) != 255 else None
        rows.append({
            "player": player,
            "health": healths[player],
            "alive": bool(alive[player]),
            "revealed_role": known_role,
        })
    return tuple(rows)


if QT_AVAILABLE:

    class GameBoard(QGroupBox):
        def __init__(self, parent: QWidget | None = None, *, human_player: int = 0) -> None:
            require_qt()
            super().__init__(tr("game.public_state"), parent)
            self.human_player = human_player
            self.your_role = QLabel(parent=self)
            self._labels = [QLabel(parent=self) for _ in range(5)]
            layout = QVBoxLayout(self)
            layout.addWidget(self.your_role)
            for label in self._labels:
                layout.addWidget(label)

        def update_state(self, state: Any) -> None:
            observation = state.observation
            healths = _sequence(observation.get("healths"), 5, 0)
            alive = _sequence(observation.get("alive"), 5, False)
            revealed = _sequence(observation.get("revealed_roles"), 5, 255)
            visibility = _sequence(observation.get("role_visibility"), 5, 0)
            self.your_role.setText(
                tr("game.you_are", player=self.human_player, role=_self_role(observation, self.human_player))
            )
            hp = tr("game.hp")
            for index, label in enumerate(self._labels):
                status = tr("game.alive") if bool(alive[index]) else tr("game.eliminated")
                you = tr("game.you_suffix") if index == self.human_player else ""
                if index == self.human_player or int(visibility[index]) >= 2:
                    known = _self_role(observation, index) if index == self.human_player else role_name(revealed[index])
                else:
                    known = hidden_role_label()
                label.setText(f"Player {index}{you}: {hp} {healths[index]} — {status} — {known}")


    class ActionPanel(QWidget):
        actionRequested = pyqtSignal(int)
        pauseRequested = pyqtSignal()

        def __init__(self, parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(parent)
            layout = QVBoxLayout(self)
            buttons_layout = QHBoxLayout()
            self.buttons: list[QPushButton] = []
            labels = [tr("game.action.pass")] + [tr("game.action.bang", player=index) for index in range(5)]
            for action, label in enumerate(labels):
                button = QPushButton(label, self)
                button.clicked.connect(lambda _checked=False, selected=action: self.actionRequested.emit(selected))
                buttons_layout.addWidget(button)
                self.buttons.append(button)
            layout.addLayout(buttons_layout)
            self.pause_button = QPushButton(tr("game.pause"), self)
            self.pause_button.clicked.connect(self.pauseRequested.emit)
            layout.addWidget(self.pause_button)
            self.message = QLabel(self)
            self.message.setWordWrap(True)
            layout.addWidget(self.message)

        def update_mask(self, mask: Sequence[int], *, paused: bool = False, active: bool = True) -> None:
            for button, allowed in zip(self.buttons, mask):
                button.setEnabled(bool(allowed) and active and not paused)
            self.pause_button.setText(tr("game.resume") if paused else tr("game.pause"))
            self.pause_button.setEnabled(active)

        def set_message(self, message: str | None) -> None:
            self.message.setText(message or "")


    class DebugHUD(QGroupBox):
        """Privacy-safe opponent HUD: exactly four rows, never the human seat."""

        def __init__(self, parent: QWidget | None = None, *, human_player: int = 0) -> None:
            require_qt()
            super().__init__(tr("game.opponents"), parent)
            self.human_player = human_player
            self.table = QTableWidget(4, 4, self)
            self.table.setHorizontalHeaderLabels(
                [tr("hud.player"), tr("hud.hp"), tr("hud.status"), tr("hud.role")]
            )
            # Keep the four opponent rows compact so the panel stops wasting
            # horizontal space: the three short columns (Player/HP/Status) size
            # to their contents while the role column takes the remaining width.
            header = self.table.horizontalHeader()
            for column in (0, 1, 2):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            # The row index adds no information for a fixed four-opponent table.
            self.table.verticalHeader().setVisible(False)
            # Four rows always fit, so a vertical scrollbar only adds clutter.
            self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            layout = QVBoxLayout(self)
            layout.addWidget(self.table)

        def update_state(self, state: Any) -> None:
            for row, opponent in enumerate(opponent_public_rows(state, human_player=self.human_player)):
                self.table.setItem(row, 0, QTableWidgetItem(f"Player {opponent['player']}"))
                self.table.setItem(row, 1, QTableWidgetItem(str(opponent["health"])))
                status = tr("game.alive") if opponent["alive"] else tr("game.eliminated")
                self.table.setItem(row, 2, QTableWidgetItem(status))
                role_text = hidden_role_label() if opponent["revealed_role"] is None else role_name(opponent["revealed_role"])
                self.table.setItem(row, 3, QTableWidgetItem(role_text))
            self.table.resizeColumnsToContents()


    class GameLog(QGroupBox):
        """Chronological, human-readable log of public game events."""

        def __init__(self, parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(tr("game.log"), parent)
            self.list = QListWidget(self)
            layout = QVBoxLayout(self)
            layout.addWidget(self.list)

        def update_state(self, state: Any) -> None:
            observation = getattr(state, "observation", None)
            if observation is None:
                return
            events = observation.get("event_sequence") or ()
            self.list.clear()
            for event in events:
                if isinstance(event, Mapping):
                    self.list.addItem(format_event(event))
            count = self.list.count()
            if count:
                self.list.setCurrentRow(count - 1)
                self.list.scrollToBottom()


else:

    class GameBoard:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

    class ActionPanel:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

    class DebugHUD:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

    class GameLog:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


def _self_role(observation: Mapping[str, Any], human_player: int) -> str:
    """Resolve the viewer's own role label from the public observation."""

    self_role = observation.get("self_role")
    if self_role is not None:
        try:
            values = list(self_role)
        except TypeError:
            values = [self_role]
        if values:
            return role_name(values[0])
    revealed = _sequence(observation.get("revealed_roles"), 5, 255)
    return role_name(revealed[human_player]) if 0 <= human_player < len(revealed) else hidden_role_label()


def format_event(event: Mapping[str, Any]) -> str:
    """Render a single public event row as a readable, localized log line."""

    step = event.get("environment_step")
    prefix = tr("event.step", step=step) if step is not None else ""
    event_type = str(event.get("event_type", "")).strip()
    name = _event_label(event_type)
    actor = event.get("actor_idx")
    target = event.get("target_idx")
    eliminated = event.get("eliminated_idx")
    revealed = event.get("revealed_role")
    parts: list[str] = [name]
    if actor is not None:
        parts.append(tr("event.by", player=int(actor)))
    if target is not None:
        parts.append(tr("event.target", player=int(target)))
    if eliminated is not None:
        parts.append(tr("event.eliminated", player=int(eliminated)))
    if revealed is not None:
        parts.append(tr("event.role", role=role_name(revealed)))
    return prefix + " ".join(parts)


def _event_label(event_type: str) -> str:
    known = {
        "bang", "pass", "player_eliminated", "role_revealed",
        "game_started", "game_ended", "turn_started",
    }
    key = event_type.lower()
    if key in known:
        return tr(f"event.{key}")
    return " ".join(part.capitalize() for part in key.replace("-", "_").split("_") if part) or "Event"


def _sequence(value: Any, length: int, default: Any) -> list[Any]:
    if value is None:
        return [default] * length
    try:
        values = list(value)
    except TypeError:
        values = [value]
    return (values + [default] * length)[:length]


__all__ = ["ActionPanel", "DebugHUD", "GameBoard", "GameLog", "format_event", "opponent_public_rows"]
