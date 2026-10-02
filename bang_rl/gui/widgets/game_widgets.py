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
from ..replay import merge_rendered_log
from ..roles import faction_name, hidden_role_label, role_name
from .board_layout import (
    HIDDEN_ROLE_SENTINEL,
    action_for_seat,
    circle_radius,
    facing_angle,
    gun_anchor_and_angle,
    health_to_color,
    highlight_color,
    interpolate_angle,
    is_action_legal,
    muzzle_point_and_angle,
    new_animation_events,
    role_for_display,
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

# Minimum drawable size of the GraphicalBoard, in pixels. Defined outside the
# ``if QT_AVAILABLE:`` gate so the layout tests can reason about the realistic
# smallest board size headlessly (item 3 caption-fit at short/wide sizes). The
# board can never render smaller than this inside the fixed main window, so it
# is the tightest case for whether the human's caption lines fit below the seat.
GRAPHICAL_BOARD_MIN_WIDTH = 560
GRAPHICAL_BOARD_MIN_HEIGHT = 520


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


    # Foreground brushes for the GameLog hover preview. Rows at or before the
    # hovered row keep the normal (bright) colour; rows AFTER it are drawn in a
    # dimmer colour so they stay visible (items 2 & 3) but are visibly
    # de-emphasized as "future" relative to the hovered moment.
    _LOG_NORMAL_FG = QColor(235, 235, 235)
    _LOG_DIMMED_FG = QColor(120, 120, 120)


    class GameLog(QGroupBox):
        """Chronological, human-readable log of public game events.

        The list always renders the FULL cumulative log. A hover preview NEVER
        clears or shrinks the list (items 2 & 3): instead it highlights the
        hovered row and dims the rows after it via :meth:`preview_moment`, and
        :meth:`clear_preview` restores the normal emphasis. ``update_state``
        only rebuilds (and auto-scrolls) when the log actually GROWS during live
        play; a shorter (hover-preview) ``event_sequence`` is treated as a
        preview and does not drop the later rows.
        """

        def __init__(self, parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(tr("game.log"), parent)
            self.list = QListWidget(self)
            layout = QVBoxLayout(self)
            layout.addWidget(self.list)
            # Number of event rows currently rendered. Tracks the FULL log so a
            # shorter hover-preview sequence does not shrink the view.
            self._rendered_count = 0

        def update_state(self, state: Any) -> None:
            observation = getattr(state, "observation", None)
            if observation is None:
                return
            events = [
                event
                for event in (observation.get("event_sequence") or ())
                if isinstance(event, Mapping)
            ]
            current = self.list.count()
            # Items 2 & 3: the shared pure helper decides whether the incoming
            # (possibly truncated hover-preview) sequence shrinks the log. A
            # shorter sequence keeps the full rendered log and is treated as a
            # hover preview instead of clearing it.
            _rows, grew = merge_rendered_log(range(current), events)
            if len(events) < current:
                self.preview_moment(len(events) - 1 if events else None)
                return
            # The log grew (or was first populated): rebuild from the full
            # cumulative sequence and reset any lingering hover de-emphasis.
            self.list.clear()
            for event in events:
                item = QListWidgetItem(format_event(event))
                item.setForeground(QBrush(_LOG_NORMAL_FG))
                self.list.addItem(item)
            self._rendered_count = len(events)
            if events:
                self.list.setCurrentRow(len(events) - 1)
                if grew:
                    # Only auto-scroll when new rows were appended for live play
                    # so a hover preview never jumps the viewport (item 3).
                    self.list.scrollToBottom()

        def preview_moment(self, row: int | None) -> None:
            """Preview the hovered moment WITHOUT clearing/shrinking the list.

            Highlights ``row`` (the hovered log row) and dims every row AFTER it
            so the later records stay visible (items 2 & 3) but are shown as
            "future" relative to the hovered moment. Passing ``None`` clears the
            selection. The scroll position is left untouched so the row under
            the cursor stays under the cursor.
            """

            count = self.list.count()
            if count == 0:
                return
            if row is None:
                self.list.clearSelection()
                self.list.setCurrentRow(-1)
            else:
                row = max(0, min(int(row), count - 1))
                self.list.setCurrentRow(row)
            threshold = count if row is None else row
            for index in range(count):
                item = self.list.item(index)
                if item is None:
                    continue
                dim = index > threshold
                item.setForeground(QBrush(_LOG_DIMMED_FG if dim else _LOG_NORMAL_FG))

        def clear_preview(self) -> None:
            """Restore normal emphasis after a hover preview ends."""

            count = self.list.count()
            for index in range(count):
                item = self.list.item(index)
                if item is not None:
                    item.setForeground(QBrush(_LOG_NORMAL_FG))
            if count:
                self.list.setCurrentRow(count - 1)


    class GraphicalBoard(QWidget):
        """Canvas board: five gun-holding circles, the human at the bottom.

        ITEM 2: when constructed with ``human_player=None`` (spectating an
        all-AI game) there is no human seat, so all five seats fill the ring and
        every seat is captioned as Player N (no "나"/"You").

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

        _MIN_WIDTH = GRAPHICAL_BOARD_MIN_WIDTH
        _MIN_HEIGHT = GRAPHICAL_BOARD_MIN_HEIGHT

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            human_player: int | None = 0,
        ) -> None:
            require_qt()
            super().__init__(parent)
            # ITEM 2: ``human_player=None`` means a spectator (all-AI) board with
            # no dedicated bottom "you" seat. All five seats are then laid out in
            # the ring and captioned as Player N, and no seat is interactive.
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
            # Seat currently under the pointer during the human's turn (item 5).
            self._hover_seat: int | None = None
            self._anim_timer = QTimer(self)
            self._anim_timer.setSingleShot(False)
            self._anim_timer.timeout.connect(self._advance_shot)
            self.setMinimumSize(self._MIN_WIDTH, self._MIN_HEIGHT)
            self.setMouseTracking(True)

        def set_human_player(self, human_player: int | None) -> None:
            """Reconfigure which seat (if any) is the human (item 2).

            ``None`` switches the board to spectator layout (no "you" seat).
            Used by the replay screen when a loaded record is a spectator game.
            """

            if human_player == self.human_player:
                return
            self.human_player = human_player
            self._hover_seat = None
            self.update()

        def update_state(self, state: Any) -> None:
            self._state = state
            mask = getattr(state, "action_mask", None)
            self._mask = tuple(int(value) for value in mask) if mask is not None else (0,) * 6
            terminated = bool(getattr(state, "terminated", False) or getattr(state, "truncated", False))
            self._active = not terminated and getattr(state, "current_player", None) == self.human_player
            if not self._active:
                # Hover highlighting is only meaningful on the human's turn.
                self._hover_seat = None
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
            # Record the shooter's resting facing angle (toward the board
            # centre) and the aim angle toward the target so the barrel can
            # rotate facing -> target during phase 1 and ease back to facing
            # over the remaining frames.
            start_facing = facing_angle(
                int(shooter_idx), self.width(), self.height(), human_player=self.human_player
            )
            aim = muzzle_point_and_angle(
                int(shooter_idx),
                int(target_idx),
                self.width(),
                self.height(),
                human_player=self.human_player,
            )
            aim_angle = start_facing if aim is None else aim[1]
            self._shot = {
                "shooter": int(shooter_idx),
                "target": int(target_idx),
                "frame": 0,
                "start_facing": float(start_facing),
                "aim_angle": float(aim_angle),
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

        def mouseMoveEvent(self, event: Any) -> None:
            # Item 5: on the human's turn, hovering a seat highlights it. Track
            # the hovered seat and repaint only when it changes.
            if not self._active or self._state is None:
                if self._hover_seat is not None:
                    self._hover_seat = None
                    self.update()
                return
            position = event.position()
            seat = seat_at_point(
                position.x(),
                position.y(),
                self.width(),
                self.height(),
                human_player=self.human_player,
            )
            if seat != self._hover_seat:
                self._hover_seat = seat
                self.update()

        def leaveEvent(self, event: Any) -> None:
            if self._hover_seat is not None:
                self._hover_seat = None
                self.update()

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
                self._paint_game_over_banner(painter)
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
            aim_in_frames = max(1, BANG_SHOT_FRAMES // 2)
            # The target only rattles once the shot actually fires (barrel aimed).
            if frame < aim_in_frames or frame >= BANG_SHOT_FRAMES:
                return (0.0, 0.0)
            # Decaying horizontal shake so the hit target visibly rattles.
            hit_frame = frame - aim_in_frames
            span = max(1, BANG_SHOT_FRAMES - aim_in_frames)
            amplitude = 6.0 * (1.0 - hit_frame / span)
            return (amplitude * math.sin(hit_frame * 2.2), 0.0)

        def _hit_flash(self, player: int) -> bool:
            """True when ``player`` should show the phase-1 hit flash outline."""

            shot = self._shot
            if shot is None or player != shot["target"]:
                return False
            aim_in_frames = max(1, BANG_SHOT_FRAMES // 2)
            return aim_in_frames <= int(shot["frame"]) < BANG_SHOT_FRAMES

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
            base_fill = seat_fill_color(player, healths, alive)
            # Item 5: on the human's turn, the hovered seat (a legal target or
            # the human's own seat) is filled with a brighter colour.
            hovered = (
                self._active
                and self._hover_seat == player
                and (
                    player == self.human_player
                    or is_action_legal(
                        action_for_seat(player, human_player=self.human_player), self._mask
                    )
                )
            )
            if hovered:
                color = QColor(*highlight_color(base_fill))
            else:
                color = QColor(*base_fill)
            painter.setBrush(QBrush(color))
            highlight = (
                self._active
                and is_action_legal(action_for_seat(player, human_player=self.human_player), self._mask)
            )
            if self._hit_flash(player):
                # Bright flashing outline on the seat currently being hit.
                pen = QPen(QColor(255, 90, 60))
                pen.setWidth(5)
            elif hovered:
                # Emphasize the hovered selection with a thick bright outline
                # on top of the brighter fill.
                pen = QPen(QColor(255, 240, 150))
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
            # Item 4: every character holds a gun on its RIGHT side while facing
            # the board centre. A settled character's barrel points along its
            # facing angle. The shooter's barrel rotates facing -> target during
            # phase 1, fires when aimed, then eases back to facing.
            facing = facing_angle(player, self.width(), self.height(), human_player=self.human_player)
            barrel_angle = self._gun_barrel_angle(player, facing)
            anchor = gun_anchor_and_angle(
                player, self.width(), self.height(), facing=facing, human_player=self.human_player
            )
            if anchor is None:
                return
            (ax, ay), _facing_angle = anchor
            painter.save()
            painter.setBrush(QBrush(QColor(10, 10, 10)))
            painter.setPen(QPen(QColor(10, 10, 10)))
            gun_length = radius * 0.9
            gun_height = max(4.0, radius * 0.18)
            # Mount the barrel at the right-side anchor and orient it along the
            # current barrel angle (facing when settled, aim while firing).
            painter.translate(ax, ay)
            painter.rotate(math.degrees(barrel_angle))
            rect = QRectF(0.0, -gun_height / 2, gun_length, gun_height)
            painter.drawRect(rect)
            painter.restore()

        def _gun_barrel_angle(self, player: int, facing: float) -> float:
            """Return the current barrel angle (radians) for ``player``.

            Non-shooters (and every seat when settled) point along ``facing``.
            The shooter animates facing -> aim across phase-1 frames, holds the
            aim through the shot, then eases aim -> facing over phase 2.
            """

            shot = self._shot
            if shot is None or player != shot["shooter"]:
                return facing
            start = float(shot.get("start_facing", facing))
            aim = float(shot.get("aim_angle", facing))
            frame = int(shot["frame"])
            # Rotate to the target across the first half of phase 1, hold aim
            # for the flash, then ease back to facing during phase 2.
            aim_in_frames = max(1, BANG_SHOT_FRAMES // 2)
            if frame < aim_in_frames:
                return interpolate_angle(start, aim, frame / aim_in_frames)
            if frame < BANG_SHOT_FRAMES:
                return aim
            smoke = frame - BANG_SHOT_FRAMES
            return interpolate_angle(aim, start, smoke / max(1, BANG_SMOKE_FRAMES))

        def _shot_is_aimed(self) -> bool:
            """True when the barrel has finished rotating onto the target."""

            shot = self._shot
            if shot is None:
                return False
            aim_in_frames = max(1, BANG_SHOT_FRAMES // 2)
            return aim_in_frames <= int(shot["frame"]) < BANG_SHOT_FRAMES

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
            aim_in_frames = max(1, BANG_SHOT_FRAMES // 2)
            painter.save()
            if frame < aim_in_frames:
                # Barrel is still rotating onto the target: no shot yet.
                painter.restore()
                return
            if frame < BANG_SHOT_FRAMES:
                # Phase 1 (aimed): yellow line with a red tint from muzzle to
                # target, fired the moment the barrel is aimed at the target.
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

        def _is_game_over(self) -> bool:
            """True when the current state has terminated or truncated."""

            state = self._state
            if state is None:
                return False
            return bool(getattr(state, "terminated", False) or getattr(state, "truncated", False))

        def _paint_game_over_banner(self, painter: QPainter) -> None:
            """Item 7: large centered winning-team (or no-winner) banner."""

            state = self._state
            if state is None or not self._is_game_over():
                return
            terminated = bool(getattr(state, "terminated", False))
            if terminated:
                faction = getattr(state, "winning_team", None)
                text = tr("game.winner_banner", faction=faction_name(faction))
            else:
                text = tr("game.no_winner_banner")
            width = self.width()
            height = self.height()
            # Scale the banner to the board but keep it comfortably large.
            point_size = max(28, int(min(width, height) * 0.06))
            font = QFont()
            font.setBold(True)
            font.setPointSize(point_size)
            painter.save()
            painter.setFont(font)
            rect = QRectF(0.0, 0.0, float(width), float(height))
            # Shadow for legibility over the coloured seats, then the banner.
            painter.setPen(QPen(QColor(0, 0, 0, 200)))
            shadow = QRectF(2.0, 2.0, float(width), float(height))
            painter.drawText(shadow, int(Qt.AlignmentFlag.AlignCenter), text)
            painter.setPen(QPen(QColor(255, 235, 120)))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)
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
            # ITEM 2: in spectator mode (human_player is None) no seat is the
            # human, so every seat is captioned as Player N and none shows the
            # "you" label.
            if self.human_player is not None and player == self.human_player:
                name = tr("game.you_name")
            else:
                name = tr("game.player_name", player=player)
            game_over = self._is_game_over()
            # ITEM 1: in a spectator session the DTO carries reveal_all_roles so
            # every seat shows its true role during play. Human sessions leave
            # this False and keep the sheriff-only-until-terminal privacy.
            reveal_all = bool(getattr(self._state, "reveal_all_roles", False))
            # Item 7: at game end reveal EVERY seat's role from the terminal
            # state's final_roles (all players are shown once the game is over).
            # During play, and whenever final_roles is absent (a replay snapshot
            # or a test stub may not carry it), fall back to the public reveal
            # logic so no living role ever leaks. The helper also always shows
            # the human's own known role for their seat (item 6).
            final_roles = getattr(self._state, "final_roles", ())
            role_index = role_for_display(
                player,
                observation,
                final_roles,
                game_over,
                human_player=self.human_player,
                reveal_all=reveal_all,
            )
            role = (
                hidden_role_label()
                if int(role_index) == HIDDEN_ROLE_SENTINEL
                else role_name(role_index)
            )
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

        def preview_moment(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

        def clear_preview(self, *_args: Any, **_kwargs: Any) -> None:
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
    "GRAPHICAL_BOARD_MIN_HEIGHT",
    "GRAPHICAL_BOARD_MIN_WIDTH",
    "OPPONENT_PANEL_MAX_WIDTH",
    "format_event",
    "opponent_public_rows",
]
