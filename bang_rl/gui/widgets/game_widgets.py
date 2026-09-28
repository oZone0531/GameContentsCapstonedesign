"""PyQt6 game widgets fed exclusively by public DTO snapshots."""
from __future__ import annotations

import math
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
    QListWidgetItem,
    QPainter,
    QPen,
    QPointF,
    QPushButton,
    QRectF,
    QTableWidget,
    QTableWidgetItem,
    QTimer,
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
    health_to_color,
    is_action_legal,
    muzzle_point_and_angle,
    new_animation_events,
    seat_at_point,
    seat_fill_color,
    seat_positions,
)


# Bang! shot-animation timing. Kept outside the Qt gate so it is importable
# headlessly. Phase 1 (bright shot + target shake) then phase 2 (grey smoke
# fade); each frame advances the QTimer once.
BANG_FRAME_MS = 40
BANG_SHOT_FRAMES = 8
BANG_SMOKE_FRAMES = 10
BANG_TOTAL_FRAMES = BANG_SHOT_FRAMES + BANG_SMOKE_FRAMES


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
                    item = QListWidgetItem(format_event(event))
                    # Hovering a row reveals the game moment (step / round) at
                    # which the action was recorded.
                    item.setToolTip(format_log_moment(event))
                    self.list.addItem(item)
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
        # Emitted when a newly-appended bang hits a target / an elimination is
        # recorded, so the screen can show a transient localized status line
        # (board.hit / board.eliminated) alongside the visual shot animation.
        playerHit = pyqtSignal(int)
        playerEliminated = pyqtSignal(int)

        _MIN_WIDTH = 560
        _MIN_HEIGHT = 520

        def __init__(self, parent: QWidget | None = None, *, human_player: int = 0) -> None:
            require_qt()
            super().__init__(parent)
            self.human_player = human_player
            self._state: Any = None
            self._mask: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
            self._active = False
            self._alive = True
            # Track the events already rendered so update_state can detect only
            # the newly-appended bang/elimination events to animate.
            self._seen_events: tuple[Mapping, ...] = ()
            # Active shot animation state (None when the board is settled).
            self._shot: dict[str, Any] | None = None
            self._anim_timer = QTimer(self)
            self._anim_timer.setSingleShot(False)
            self._anim_timer.timeout.connect(self._advance_shot)
            self.setMinimumSize(self._MIN_WIDTH, self._MIN_HEIGHT)
            self.setMouseTracking(True)

        def update_state(self, state: Any) -> None:
            self._state = state
            mask = getattr(state, "action_mask", None)
            self._mask = tuple(int(value) for value in mask) if mask is not None else (0,) * 6
            terminated = bool(getattr(state, "terminated", False) or getattr(state, "truncated", False))
            self._active = not terminated and getattr(state, "current_player", None) == self.human_player
            self._trigger_new_events(state)
            self.update()

        def _trigger_new_events(self, state: Any) -> None:
            observation = getattr(state, "observation", {}) or {}
            events = tuple(
                event for event in (observation.get("event_sequence") or ()) if isinstance(event, Mapping)
            )
            new_events = new_animation_events(self._seen_events, events)
            self._seen_events = events
            for event in new_events:
                event_type = str(event.get("event_type", "")).strip().lower()
                if event_type == "player_eliminated":
                    eliminated = event.get("eliminated_idx")
                    if eliminated is None:
                        eliminated = event.get("target_idx")
                    if eliminated is not None:
                        self.playerEliminated.emit(int(eliminated))
                    continue
                if event_type != "bang":
                    continue
                actor = event.get("actor_idx")
                target = event.get("target_idx")
                if actor is None or target is None:
                    continue
                self.play_bang(int(actor), int(target))
                self.playerHit.emit(int(target))
                # Animate the most recent bang; earlier ones in the same batch
                # are reflected by the resulting HP/eliminated styling.

        # -- bang! shot animation ---------------------------------------
        def play_bang(self, shooter_idx: int, target_idx: int) -> None:
            """Animate a bang! shot from ``shooter_idx`` toward ``target_idx``.

            Phase 1 draws a yellow line with a red tint from the shooter's gun
            muzzle to the target circle while the target shakes; phase 2 fades
            the line to grey (smoke) before the gun resets to neutral.  Driven
            by a lifecycle-guarded QTimer so a torn-down board never repaints
            deleted C++ objects.
            """

            if not self._alive:
                return
            if shooter_idx == target_idx:
                return
            self._shot = {
                "shooter": int(shooter_idx),
                "target": int(target_idx),
                "frame": 0,
            }
            self._anim_timer.start(BANG_FRAME_MS)
            self.update()

        def _advance_shot(self) -> None:
            if not self._alive or self._shot is None:
                self._anim_timer.stop()
                return
            self._shot["frame"] += 1
            if self._shot["frame"] >= BANG_TOTAL_FRAMES:
                self._shot = None
                self._anim_timer.stop()
            self.update()

        def stop_animation(self) -> None:
            """Halt any in-flight shot animation and settle the board."""

            self._shot = None
            self._anim_timer.stop()

        def detach(self) -> None:
            """Mark the board dead so the timer never repaints a torn-down view.

            Mirrors the ``_alive``/``_detach`` lifecycle used by GameScreen:
            once the central widget is replaced its C++ object is deleted, so a
            late animation frame must not touch it.
            """

            self._alive = False
            self._anim_timer.stop()
            self._shot = None

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
                self._paint_shot(painter)
            finally:
                painter.end()

        def _paint_background(self, painter: QPainter) -> None:
            painter.fillRect(self.rect(), QColor(24, 28, 36))

        def _shake_offset(self, player: int) -> tuple[float, float]:
            """Return a small oscillating (dx, dy) for a hit target in phase 1."""

            shot = self._shot
            if shot is None or player != shot["target"]:
                return (0.0, 0.0)
            frame = int(shot["frame"])
            if frame >= BANG_SHOT_FRAMES:
                return (0.0, 0.0)
            # Decaying horizontal shake so the hit target visibly rattles.
            amplitude = 6.0 * (1.0 - frame / max(1, BANG_SHOT_FRAMES))
            return (amplitude * math.sin(frame * 2.2), 0.0)

        def _hit_flash(self, player: int) -> bool:
            """True when ``player`` should show the phase-1 hit flash outline."""

            shot = self._shot
            if shot is None or player != shot["target"]:
                return False
            return int(shot["frame"]) < BANG_SHOT_FRAMES

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
                base_x, base_y = positions[player]
                dx, dy = self._shake_offset(player)
                cx, cy = base_x + dx, base_y + dy
                self._paint_circle(painter, player, cx, cy, radius, healths, alive)
                self._paint_gun(painter, player, base_x, base_y, radius)
                self._paint_caption(
                    painter, player, base_x, base_y, radius, observation, healths, alive, visibility, revealed
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
            if self._hit_flash(player):
                # Bright flashing outline on the seat currently being hit.
                pen = QPen(QColor(255, 90, 60))
                pen.setWidth(5)
            elif highlight:
                pen = QPen(QColor(250, 210, 80))
                pen.setWidth(3)
            else:
                pen = QPen(QColor(20, 20, 20))
                pen.setWidth(1)
            painter.setPen(pen)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

        def _paint_gun(
            self, painter: QPainter, player: int, cx: float, cy: float, radius: float
        ) -> None:
            # A neutral gun points outward from the board centre; during a bang
            # animation the shooter's gun rotates to point along the shooter ->
            # target direction (angle from muzzle_point_and_angle).
            painter.save()
            painter.setBrush(QBrush(QColor(10, 10, 10)))
            painter.setPen(QPen(QColor(10, 10, 10)))
            gun_length = radius * 0.9
            gun_height = max(4.0, radius * 0.18)
            firing_angle = self._gun_firing_angle(player)
            if firing_angle is not None:
                # Draw the gun as a barrel starting at the circle edge and
                # extending outward along the firing angle.
                painter.translate(cx, cy)
                painter.rotate(math.degrees(firing_angle))
                rect = QRectF(radius - gun_length * 0.2, -gun_height / 2, gun_length, gun_height)
                painter.drawRect(rect)
            elif player == self.human_player:
                # Human gun points up, toward the opponents.
                rect = QRectF(cx - gun_height / 2, cy - radius - gun_length * 0.4, gun_height, gun_length)
                painter.drawRect(rect)
            else:
                # Opponent guns point down, toward the human seat.
                rect = QRectF(cx - gun_height / 2, cy + radius - gun_length * 0.6, gun_height, gun_length)
                painter.drawRect(rect)
            painter.restore()

        def _gun_firing_angle(self, player: int) -> float | None:
            """Return the firing angle (radians) if ``player`` is the shooter."""

            shot = self._shot
            if shot is None or player != shot["shooter"]:
                return None
            result = muzzle_point_and_angle(
                shot["shooter"],
                shot["target"],
                self.width(),
                self.height(),
                human_player=self.human_player,
            )
            return None if result is None else result[1]

        def _paint_shot(self, painter: QPainter) -> None:
            """Draw the bang! shot line: bright phase, then grey smoke fade."""

            shot = self._shot
            if shot is None:
                return
            result = muzzle_point_and_angle(
                shot["shooter"],
                shot["target"],
                self.width(),
                self.height(),
                human_player=self.human_player,
            )
            if result is None:
                return
            (mx, my), _angle = result
            positions = seat_positions(self.width(), self.height(), human_player=self.human_player)
            if shot["target"] not in positions:
                return
            tx, ty = positions[shot["target"]]
            frame = int(shot["frame"])
            painter.save()
            if frame < BANG_SHOT_FRAMES:
                # Phase 1: yellow line with a red tint from muzzle to target.
                color = QColor(255, 210, 60, 235)
                pen = QPen(color)
                pen.setWidth(5)
                painter.setPen(pen)
                painter.drawLine(QPointF(mx, my), QPointF(tx, ty))
                # A thinner red core reinforces the red tint.
                red_pen = QPen(QColor(230, 70, 40, 200))
                red_pen.setWidth(2)
                painter.setPen(red_pen)
                painter.drawLine(QPointF(mx, my), QPointF(tx, ty))
            else:
                # Phase 2: grey smoke fading out (decreasing alpha).
                smoke_frame = frame - BANG_SHOT_FRAMES
                progress = smoke_frame / max(1, BANG_SMOKE_FRAMES)
                alpha = int(max(0, round(160 * (1.0 - progress))))
                pen = QPen(QColor(170, 170, 170, alpha))
                pen.setWidth(6)
                painter.setPen(pen)
                painter.drawLine(QPointF(mx, my), QPointF(tx, ty))
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


def format_log_moment(event: Mapping[str, Any]) -> str:
    """Return a localized 'game moment' string for a log-row hover tooltip.

    Uses the event's ``environment_step`` (and ``round_index`` when present) so
    hovering a log row reveals exactly when the action was recorded.  Pure and
    Qt-free so it is localizable and unit-testable in CI.
    """

    step = event.get("environment_step")
    if step is None:
        return tr("log.moment_unknown")
    round_index = event.get("round_index")
    if round_index is not None:
        return tr("log.moment_round", step=int(step), round=int(round_index))
    return tr("log.moment", step=int(step))


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
    "format_log_moment",
    "opponent_public_rows",
]
