"""PyQt6 game widgets fed exclusively by public DTO snapshots."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..i18n import tr
from ..qt_compat import (
    QBrush,
    QColor,
    QFont,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPainter,
    QPen,
    QPointF,
    QPushButton,
    QRectF,
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
from .board_layout import (
    action_for_seat,
    circle_radius,
    is_action_legal,
    muzzle_point_and_angle,
    seat_at_point,
    seat_fill_color,
    seat_positions,
)


# Maximum pixel width of the Opponents ('상대') HUD panel. Defined outside the
# ``if QT_AVAILABLE:`` gate so it is importable/testable headlessly. Keeps the
# panel from over-expanding while still fitting four short opponent rows.
OPPONENT_PANEL_MAX_WIDTH = 340


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
            # Bound the panel width so the Opponents ('상대') group stops
            # consuming excessive horizontal space. The four rows only hold a
            # short player id, HP, status and a role label, so a modest cap is
            # plenty and leaves room for the board and log panels.
            self.setMaximumWidth(OPPONENT_PANEL_MAX_WIDTH)
            self.table = QTableWidget(4, 4, self)
            self.table.setMaximumWidth(OPPONENT_PANEL_MAX_WIDTH)
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


    class GraphicalBoard(QWidget):
        """Canvas board: five gun-holding circles, the human at the bottom.

        Each seat renders as a circle with a small black rectangle 'gun' and a
        name/HP/role caption beneath, honoring the same role-privacy rule as the
        table board (only the viewer's own role and roles with
        ``role_visibility >= 2`` are shown; everything else is hidden).  Clicking
        a legal opponent circle emits ``actionRequested(player_index + 1)``
        (BANG) and clicking the human's own circle emits ``actionRequested(0)``
        (PASS).  Illegal targets per the current action mask are ignored.

        The geometry (seat positions, hit-testing, muzzle/gun angle) comes from
        the pure ``board_layout`` helpers so FEAT-004 can layer a shot animation
        on top without re-deriving coordinates.
        """

        actionRequested = pyqtSignal(int)

        _MIN_WIDTH = 560
        _MIN_HEIGHT = 520

        def __init__(self, parent: QWidget | None = None, *, human_player: int = 0) -> None:
            require_qt()
            super().__init__(parent)
            self.human_player = human_player
            self._state: Any = None
            self._mask: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
            self._active = False
            self.setMinimumSize(self._MIN_WIDTH, self._MIN_HEIGHT)
            self.setMouseTracking(True)

        def update_state(self, state: Any) -> None:
            self._state = state
            mask = getattr(state, "action_mask", None)
            self._mask = tuple(int(value) for value in mask) if mask is not None else (0,) * 6
            terminated = bool(getattr(state, "terminated", False) or getattr(state, "truncated", False))
            self._active = not terminated and getattr(state, "current_player", None) == self.human_player
            self.update()

        # -- interaction -------------------------------------------------
        def mousePressEvent(self, event: Any) -> None:
            if self._state is None or not self._active:
                return
            position = event.position()
            seat = seat_at_point(
                position.x(),
                position.y(),
                self.width(),
                self.height(),
                human_player=self.human_player,
            )
            if seat is None:
                return
            action = action_for_seat(seat, human_player=self.human_player)
            if is_action_legal(action, self._mask):
                self.actionRequested.emit(int(action))

        # -- rendering ---------------------------------------------------
        def paintEvent(self, event: Any) -> None:
            painter = QPainter(self)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                self._paint_background(painter)
                if self._state is None:
                    return
                self._paint_seats(painter)
            finally:
                painter.end()

        def _paint_background(self, painter: QPainter) -> None:
            painter.fillRect(self.rect(), QColor(24, 28, 36))

        def _paint_seats(self, painter: QPainter) -> None:
            observation = getattr(self._state, "observation", {}) or {}
            healths = _sequence(observation.get("healths"), 5, 0)
            alive = _sequence(observation.get("alive"), 5, False)
            visibility = _sequence(observation.get("role_visibility"), 5, 0)
            revealed = _sequence(observation.get("revealed_roles"), 5, 255)
            width = self.width()
            height = self.height()
            radius = circle_radius(width, height)
            positions = seat_positions(width, height, human_player=self.human_player)
            for player in range(5):
                cx, cy = positions[player]
                self._paint_circle(painter, player, cx, cy, radius, healths, alive)
                self._paint_gun(painter, player, cx, cy, radius)
                self._paint_caption(
                    painter, player, cx, cy, radius, observation, healths, alive, visibility, revealed
                )

        def _paint_circle(
            self,
            painter: QPainter,
            player: int,
            cx: float,
            cy: float,
            radius: float,
            healths: Sequence[int],
            alive: Sequence[bool],
        ) -> None:
            color = QColor(*seat_fill_color(player, healths, alive))
            painter.setBrush(QBrush(color))
            highlight = (
                self._active
                and is_action_legal(action_for_seat(player, human_player=self.human_player), self._mask)
            )
            pen = QPen(QColor(250, 210, 80) if highlight else QColor(20, 20, 20))
            pen.setWidth(3 if highlight else 1)
            painter.setPen(pen)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

        def _paint_gun(
            self, painter: QPainter, player: int, cx: float, cy: float, radius: float
        ) -> None:
            # A neutral gun points outward from the board centre; FEAT-004 can
            # rotate it toward a target using muzzle_point_and_angle.
            painter.save()
            painter.setBrush(QBrush(QColor(10, 10, 10)))
            painter.setPen(QPen(QColor(10, 10, 10)))
            gun_length = radius * 0.9
            gun_height = max(4.0, radius * 0.18)
            if player == self.human_player:
                # Human gun points up, toward the opponents.
                rect = QRectF(cx - gun_height / 2, cy - radius - gun_length * 0.4, gun_height, gun_length)
            else:
                # Opponent guns point down, toward the human seat.
                rect = QRectF(cx - gun_height / 2, cy + radius - gun_length * 0.6, gun_height, gun_length)
            painter.drawRect(rect)
            painter.restore()

        def _paint_caption(
            self,
            painter: QPainter,
            player: int,
            cx: float,
            cy: float,
            radius: float,
            observation: Mapping[str, Any],
            healths: Sequence[int],
            alive: Sequence[bool],
            visibility: Sequence[int],
            revealed: Sequence[int],
        ) -> None:
            if player == self.human_player:
                name = tr("game.you_name")
            else:
                name = tr("game.player_name", player=player)
            if player == self.human_player or int(visibility[player]) >= 2:
                role = _self_role(observation, player) if player == self.human_player else role_name(revealed[player])
            else:
                role = hidden_role_label()
            status = tr("game.alive") if bool(alive[player]) else tr("game.eliminated")
            hp = tr("game.hp_label", health=healths[player])
            painter.setPen(QPen(QColor(235, 235, 235)))
            font = QFont()
            font.setBold(True)
            painter.setFont(font)
            top = cy + radius + 6
            line_height = radius * 0.34 + 6
            caption_width = radius * 3
            left = cx - caption_width / 2
            self._draw_centered(painter, name, left, top, caption_width)
            font.setBold(False)
            painter.setFont(font)
            self._draw_centered(painter, f"{hp} · {status}", left, top + line_height, caption_width)
            self._draw_centered(painter, role, left, top + 2 * line_height, caption_width)

        @staticmethod
        def _draw_centered(painter: QPainter, text: str, left: float, top: float, width: float) -> None:
            rect = QRectF(left, top, width, 22)
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop), text)


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

    class GraphicalBoard:  # type: ignore[no-redef]
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


__all__ = [
    "ActionPanel",
    "DebugHUD",
    "GameBoard",
    "GameLog",
    "GraphicalBoard",
    "OPPONENT_PANEL_MAX_WIDTH",
    "format_event",
    "opponent_public_rows",
]
