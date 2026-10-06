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
    QToolTip,
    QVBoxLayout,
    QWidget,
    Qt,
    QT_AVAILABLE,
    pyqtSignal,
    require_qt,
)
from ..replay import merge_rendered_log
from ..roles import faction_name, hidden_role_label, role_name
from .spectator_overlay import (
    bar_rect_at_point,
    emotion_bar_rects,
    emotion_bar_string,
    emotion_bars_for_seat,
    format_percent,
    inverted_role_inference,
    sheriff_seat_from_observation,
)
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

# FEAT-002: how many emotion bars at most are stacked under a seat. The three
# caption lines (name / HP+status / role) already sit below each circle, so the
# bars are drawn beneath them; capping the count keeps the stack inside the
# seat's area and avoids one talkative dummy's bars overlapping the seat below.
# Any extra emotions beyond the cap are simply not drawn (the hover tooltip
# still describes the drawn ones). Chosen as 6, comfortably above the default
# emotion set while bounding vertical growth.
MAX_EMOTION_BARS = 6

# FEAT-002: map the merged inference role_key produced by
# ``spectator_overlay.role_inference_items`` to a localized label. Sheriff is
# never produced (always public) and the two outlaw slots are merged into one
# ``outlaw`` bucket, so there is no single canonical role index for it: deputy
# and traitor reuse their existing role labels while the merged outlaw uses the
# new ``role.outlaw_merged`` string.
_INFERENCE_ROLE_I18N = {
    "deputy": "role.deputy",
    "traitor": "role.traitor",
    "outlaw": "role.outlaw_merged",
}


def inference_role_label(role_key: str) -> str:
    """Return the localized label for an inference ``role_key``.

    Keeps the role_key -> i18n mapping testable and Qt-light. An unknown key
    falls back to the hidden-role label rather than raising.
    """

    key = _INFERENCE_ROLE_I18N.get(str(role_key))
    return tr(key) if key is not None else hidden_role_label()


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
    # FEAT-002 (item 5): a persistent "current step" highlight that COEXISTS
    # with the hover-preview dimming. The current row that is actually shown on
    # the play surface (live current turn, pinned replay/spectator seek target)
    # is drawn in a warm highlight with a subtle background so it stays visible
    # even while a hover preview dims the later rows.
    _LOG_CURRENT_FG = QColor(255, 210, 110)
    _LOG_CURRENT_BG = QColor(70, 55, 20)


    class GameLog(QGroupBox):
        """Chronological, human-readable log of public game events.

        The list always renders the FULL cumulative log. A hover preview NEVER
        clears or shrinks the list (items 2 & 3): instead it highlights the
        hovered row and dims the rows after it via :meth:`preview_moment`, and
        :meth:`clear_preview` restores the normal emphasis. ``update_state``
        only rebuilds (and auto-scrolls) when the log actually GROWS during live
        play; a shorter (hover-preview) ``event_sequence`` is treated as a
        preview and does not drop the later rows.

        FEAT-002 (item 5): a SEPARATE persistent "current step" highlight marks
        the row that is actually shown on the play surface (the live current
        turn, or a pinned replay / spectator seek target set via
        :meth:`set_current_step`). It COEXISTS with the hover preview: the two
        concerns are unified in :meth:`_apply_emphasis`, so starting or ending a
        hover never erases the current-step highlight.
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
            # FEAT-002 (item 5): the persistent current-step row (None = none).
            self._current_step: int | None = None

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
                # Item 5: during live play the current turn is always the last
                # row, so pin the current-step highlight there after rebuilding
                # (set_current_step re-applies emphasis so the highlight survives
                # the clear + re-add above).
                self.set_current_step(len(events) - 1)
                if grew:
                    # Only auto-scroll when new rows were appended for live play
                    # so a hover preview never jumps the viewport (item 3).
                    self.list.scrollToBottom()
            else:
                self.set_current_step(None)

        def set_current_step(self, row: int | None) -> None:
            """Pin the persistent current-step highlight to ``row`` (item 5).

            Records the row that is actually shown on the play surface (the live
            current turn, or a pinned replay / spectator seek target) and
            repaints emphasis so the row is highlighted even when it is not the
            hovered row. Passing ``None`` clears the highlight. The highlight is
            SEPARATE from the hover preview and persists across hover start/end;
            see :meth:`_apply_emphasis`. The scroll position is left untouched.
            """

            count = self.list.count()
            if row is None or count == 0:
                self._current_step = None
            else:
                self._current_step = max(0, min(int(row), count - 1))
            # Re-apply with no active hover preview (hover re-applies its own
            # dimming on top via preview_moment).
            self._apply_emphasis(None)

        def _apply_emphasis(self, preview_row: int | None) -> None:
            """Re-paint every row's emphasis for the two coexisting concerns.

            FEAT-002 (item 5): for each row decide, in order, whether to DIM it
            (a hover preview is active and the row is AFTER ``preview_row``),
            HIGHLIGHT it (it is the persistent ``_current_step``), or paint it
            NORMAL. The current-step highlight is applied AFTER the preview
            decision so a hovered-and-dimmed future row still shows the
            current-step color if it happens to be the current step; dimming
            only affects rows that are neither the current step nor at/before the
            preview row. Both :meth:`preview_moment` and :meth:`clear_preview`
            call through here, so the current-step highlight PERSISTS across
            hover start and end (hover never erases it).
            """

            count = self.list.count()
            for index in range(count):
                item = self.list.item(index)
                if item is None:
                    continue
                is_current = (
                    self._current_step is not None and index == self._current_step
                )
                dim = (
                    preview_row is not None
                    and index > preview_row
                    and not is_current
                )
                if is_current:
                    item.setForeground(QBrush(_LOG_CURRENT_FG))
                    item.setBackground(QBrush(_LOG_CURRENT_BG))
                else:
                    item.setForeground(QBrush(_LOG_DIMMED_FG if dim else _LOG_NORMAL_FG))
                    item.setBackground(QBrush())
                font = item.font()
                font.setBold(is_current)
                item.setFont(font)

        def preview_moment(self, row: int | None) -> None:
            """Preview the hovered moment WITHOUT clearing/shrinking the list.

            Highlights ``row`` (the hovered log row) and dims every row AFTER it
            so the later records stay visible (items 2 & 3) but are shown as
            "future" relative to the hovered moment. Passing ``None`` clears the
            selection. The scroll position is left untouched so the row under
            the cursor stays under the cursor. The persistent current-step
            highlight is preserved via :meth:`_apply_emphasis` (item 5).
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
            self._apply_emphasis(row)

        def clear_preview(self) -> None:
            """Restore normal emphasis after a hover preview ends.

            Clears the hover dimming but PRESERVES the persistent current-step
            highlight via :meth:`_apply_emphasis` (item 5).
            """

            count = self.list.count()
            self._apply_emphasis(None)
            if count:
                self.list.setCurrentRow(count - 1)

        def truncate_after(self, row: int | None) -> None:
            """Delete the log rows AFTER ``row`` and shrink the view (item 2).

            FEAT-003 (item 2): the never-shrink contract in :meth:`update_state`
            is correct for a HOVER preview (a shorter incoming sequence is a
            read-only look at a past moment, so the later rows are kept). But a
            HUMAN resume-click FORKS the timeline: the controller has truncated
            its replay buffers to ``[:row+1]`` and the later events never
            happened, so those rows must genuinely disappear. This removes every
            list item with index > ``row`` and updates ``_rendered_count`` so the
            shorter log becomes authoritative; a subsequent :meth:`update_state`
            that re-grows from the forked timeline appends from the new count.
            Passing ``None`` clears the whole log. The current-step highlight is
            re-pinned to the surviving last row via :meth:`set_current_step`.
            """

            count = self.list.count()
            if count == 0:
                return
            if row is None:
                self.list.clear()
                self._rendered_count = 0
                self.set_current_step(None)
                return
            keep = max(0, min(int(row), count - 1))
            for index in range(count - 1, keep, -1):
                item = self.list.takeItem(index)
                del item
            self._rendered_count = self.list.count()
            # Item 5: the resume target is now the last surviving row.
            self.set_current_step(self.list.count() - 1 if self.list.count() else None)


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
            # FEAT-002: per-turn AI internals ({"health": [...], "seats":
            # {seat_str: descriptor}}) supplied by the spectator / replay screen
            # for the current state. None means "no overlay data" (every human
            # live game, and any snapshot without internals). Reset on each
            # update_state; the screen re-supplies it right after (documented
            # ordering below).
            self._seat_internals: Mapping[str, Any] | None = None
            # FEAT-002 (hidden-information safety): emotion bars and the role-
            # inference hover overlay render ONLY when overlays are enabled.
            # They are enabled when there is no human seat (spectator layout,
            # human_player is None) OR when a replay screen explicitly turns them
            # on. In a human-played LIVE game this stays False, so even if
            # internals were somehow set the overlay paint methods are no-ops and
            # no opponent emotion / inferred role can leak to the human player.
            self._overlays_forced = False
            # Cached bar-rect geometry per seat from the last paint, keyed by
            # seat index, so mouseMoveEvent can hit-test the painted bars for the
            # hover tooltip without re-deriving coordinates.
            self._bar_rects: dict[int, tuple[Any, ...]] = {}
            self._bars_by_seat: dict[int, tuple[Any, ...]] = {}
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

        # -- FEAT-002 overlay seam ---------------------------------------
        @property
        def _overlays_enabled(self) -> bool:
            """True only in spectator layout or when a replay forced overlays.

            Hidden-information rationale: emotion bars and the role-inference
            hover overlay expose every seat's (otherwise private) emotion state
            and the dummy AI's inferred role distribution. That is safe to show
            only when there is no human whose hidden information must be
            protected (spectator layout, ``human_player is None``) or during a
            replay of an already-finished game (``_overlays_forced`` set by the
            replay screen). In a human-played LIVE game this returns False.
            """

            return self.human_player is None or self._overlays_forced

        def set_overlays_enabled(self, enabled: bool) -> None:
            """Force the spectator / replay overlays on (replay screen only).

            A spectator board already has ``human_player is None`` so overlays
            are on without this; the replay screen calls this so a HUMAN-played
            replay (which keeps a human seat) also shows the overlays, since a
            replay is a post-hoc review of a finished game, not live hidden-info
            play. Never call this from a live human game screen.
            """

            forced = bool(enabled)
            if forced == self._overlays_forced:
                return
            self._overlays_forced = forced
            self.update()

        def set_seat_internals(self, internals: Mapping[str, Any] | None) -> None:
            """Supply the current per-turn AI internals for overlay rendering.

            Ordering contract (consistent across both screens): the screen calls
            ``update_state`` first (which clears any stale internals) and then
            ``set_seat_internals`` with the internals for THIS state. ``None``
            clears the overlay data. The internals are only ever consulted by
            the paint methods when ``_overlays_enabled`` is True, so passing
            them in a human live game would still render nothing.
            """

            self._seat_internals = internals if isinstance(internals, Mapping) else None
            self.update()

        def _descriptor_for_seat(self, player: int) -> Any:
            """Return seat ``player``'s descriptor from the current internals.

            ``export_agent_internals`` keys seats by STRING index under
            ``"seats"``; returns ``None`` when overlays are disabled, there are
            no internals, or the seat is absent. Never raises.
            """

            if not self._overlays_enabled:
                return None
            internals = self._seat_internals
            if not isinstance(internals, Mapping):
                return None
            seats = internals.get("seats")
            if not isinstance(seats, Mapping):
                return None
            return seats.get(str(int(player)))

        def update_state(self, state: Any) -> None:
            self._state = state
            # FEAT-002 ordering: a new state clears the stale internals; the
            # spectator / replay screen re-supplies them via set_seat_internals
            # immediately after this call. A human live game never re-supplies
            # them, so they stay None (and overlays stay disabled regardless).
            self._seat_internals = None
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
            # Item 5: on the human's turn, hovering a seat highlights it.
            # FEAT-002: in spectator / replay (overlays enabled) hovering a seat
            # also shows that seat's role inference, and hovering a painted
            # emotion bar shows that emotion's description as a tooltip. The two
            # behaviours are independent: the overlay hover works even though the
            # board is not "active" (no human turn) in spectator / replay.
            position = event.position()
            if self._state is None:
                if self._hover_seat is not None:
                    self._hover_seat = None
                    self.update()
                return
            if self._overlays_enabled:
                # FEAT-002 (item 4 enabler): this branch sets _hover_seat and
                # returns BEFORE the `if not self._active:` guard below, so
                # overlay hover (emotion bars + role inference) keeps working
                # after the game is terminal. update_state clears _hover_seat and
                # sets _active=False on a terminal state, but it does not disable
                # overlays, and the paint methods gate only on _overlays_enabled
                # plus _hover_seat (never on _active / terminated). Reviewing a
                # finished spectator game by hovering seats therefore still draws
                # the overlays.
                #
                # FEAT-001: resolve and assign _hover_seat (and request a repaint
                # on change) FIRST, so the role-inference overlay always gets its
                # hovered seat. The emotion-bar tooltip hit-test is a secondary,
                # best-effort nicety: run it AFTER and wrapped in try/except so a
                # tooltip failure (QToolTip / cached-rect geometry) can never
                # abort the handler before _hover_seat is set and so can never
                # block the role-inference overlay from painting.
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
                try:
                    self._update_bar_tooltip(position, event)
                except Exception:
                    pass
                return
            if not self._active:
                if self._hover_seat is not None:
                    self._hover_seat = None
                    self.update()
                return
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

        def _update_bar_tooltip(self, position: Any, event: Any) -> None:
            """Show an emotion's description when the pointer is over its bar.

            FEAT-002: hit-tests the pointer against the cached bar rects (the
            same geometry drawn in ``_paint_emotion_bars``) and, when over a bar
            whose EmotionBar.description is non-empty, shows it via QToolTip. An
            empty description shows nothing (hide the tip) to avoid noise. Only
            runs when overlays are enabled.
            """

            if QToolTip is None:
                return
            x, y = position.x(), position.y()
            for seat, rects in self._bar_rects.items():
                index = bar_rect_at_point(rects, x, y)
                if index is None:
                    continue
                bars = self._bars_by_seat.get(seat, ())
                if 0 <= index < len(bars):
                    description = bars[index].description
                    if description:
                        QToolTip.showText(event.globalPosition().toPoint(), description, self)
                        return
                # Over a bar with no description: hide any lingering tooltip.
                QToolTip.hideText()
                return
            QToolTip.hideText()

        def leaveEvent(self, event: Any) -> None:
            if QToolTip is not None:
                QToolTip.hideText()
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
                # FEAT-001: the role-inference hover overlay must paint LAST, so
                # it is drawn ON TOP of the game-over banner. On the terminal
                # replay frame (where reviewers commonly land) the large centred
                # banner would otherwise overdraw the inference plates near the
                # board centre, making the overlay look like it never rendered.
                # Still only runs here because paintEvent returns early above when
                # _state is None, and the overlay stays gated by
                # _paint_role_inference's own _overlays_enabled + _hover_seat
                # checks.
                self._paint_role_inference(painter)
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
            # FEAT-002: rebuild the per-seat bar-rect / bar caches each paint so
            # the hover tooltip hit-test uses the SAME geometry that was drawn.
            self._bar_rects = {}
            self._bars_by_seat = {}
            for player in range(5):
                base_x, base_y = positions[player]
                dx, dy = self._shake_offset(player)
                cx, cy = base_x + dx, base_y + dy
                self._paint_circle(painter, player, cx, cy, radius, healths, alive)
                self._paint_gun(painter, player, base_x, base_y, radius)
                self._paint_caption(
                    painter, player, base_x, base_y, radius, observation, healths, alive, visibility, revealed
                )
                # FEAT-002: emotion bars render below the caption lines, gated
                # to spectator / replay via _paint_emotion_bars itself.
                self._paint_emotion_bars(painter, player, base_x, base_y, radius)
            # FEAT-001: the role-inference hover overlay is NOT painted here any
            # more. It is now drawn as the LAST step in paintEvent, after
            # _paint_game_over_banner, so the terminal-frame banner can never
            # overdraw the inference plates. _paint_seats still draws the
            # circles / guns / captions / emotion bars exactly as before.

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

        # -- FEAT-002 part 2: emotion bars -------------------------------
        def _emotion_bar_layout(
            self, player: int, cx: float, cy: float, radius: float
        ) -> tuple[float, float, float, float]:
            """Return ``(left, top, width, line_height)`` for the bar stack.

            The bars sit directly below the three caption lines the board draws
            under each seat (see ``_paint_caption``: top = cy + radius + 6, three
            lines of ``line_height = radius*0.34 + 6``). Kept as its own helper
            so the painter and the hover hit-test share one geometry source.
            """

            line_height = radius * 0.34 + 6
            caption_top = cy + radius + 6
            # Three caption lines are drawn; bars start below them.
            top = caption_top + 3 * line_height
            width = radius * 3
            left = cx - width / 2
            return left, top, width, line_height

        def _paint_emotion_bars(
            self, painter: QPainter, player: int, cx: float, cy: float, radius: float
        ) -> None:
            """Draw proportional emotion bars under a dummy seat (spectator/replay).

            No-op unless overlays are enabled (hidden-information guard) and the
            seat has a dummy descriptor. Each bar is one row ``name ====----``.
            The drawn count is capped at ``MAX_EMOTION_BARS`` so a talkative
            dummy's bars stay within the seat area; the matching bar rects are
            cached for the hover tooltip hit-test.
            """

            if not self._overlays_enabled:
                return
            descriptor = self._descriptor_for_seat(player)
            if descriptor is None:
                return
            bars = emotion_bars_for_seat(descriptor)
            if not bars:
                return
            # Cap the number of drawn bars to bound vertical growth.
            bars = bars[:MAX_EMOTION_BARS]
            left, top, width, line_height = self._emotion_bar_layout(player, cx, cy, radius)
            rects = emotion_bar_rects(
                len(bars), left=left, top=top, width=width, line_height=line_height
            )
            self._bar_rects[player] = rects
            self._bars_by_seat[player] = bars
            painter.save()
            painter.setPen(QPen(QColor(210, 220, 235)))
            font = QFont()
            font.setBold(False)
            # A touch smaller than the caption text so a full bar fits the width.
            point = font.pointSize()
            if point > 0:
                font.setPointSize(max(7, point - 1))
            font.setFamily("monospace")
            painter.setFont(font)
            for bar in bars:
                line = f"{bar.name} {emotion_bar_string(bar.value)}"
                rect = QRectF(left, top, width, line_height)
                painter.drawText(
                    rect,
                    int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                    line,
                )
                top += line_height
            painter.restore()

        # -- FEAT-001: inverted role-inference hover overlay -------------
        def _paint_role_inference(self, painter: QPainter) -> None:
            """Draw the hovered seat's OWN role inference over the OTHER seats.

            No-op unless overlays are enabled and a seat is hovered. Hovering
            seat S shows S's own current role inferences drawn OVER each OTHER
            seat T's circle (T != S and T != the sheriff seat, which is derived
            from revealed_roles), using row T of S's
            own beliefs matrix reduced by the pure helpers (sheriff dropped,
            outlaws merged, descending). Nothing is drawn on S's own circle.
            Each per-target plate shows localized ``role N%`` lines. When S is
            not a dummy or has no usable beliefs, the localized "no inference
            info" fallback is drawn once over S's own circle so the user still
            gets feedback. This NEVER shows a seat's true hidden role, only S's
            own inferred distribution ABOUT others.
            """

            if not self._overlays_enabled:
                return
            seat = self._hover_seat
            if seat is None:
                return
            width = self.width()
            height = self.height()
            radius = circle_radius(width, height)
            positions = seat_positions(width, height, human_player=self.human_player)
            if seat not in positions:
                return
            # Defense in depth: never read internals when overlays are off, so a
            # human live game can never surface another seat's inference even if
            # internals were somehow set.
            internals = self._seat_internals if self._overlays_enabled else None
            # Derive the sheriff SEAT from the current state's observation so the
            # inverted projection excludes whichever seat actually holds the
            # Sheriff role (roles are shuffled; it is not necessarily seat 0).
            sheriff = sheriff_seat_from_observation(
                getattr(self._state, "observation", None)
            )
            projection = inverted_role_inference(internals, int(seat), sheriff_seat=sheriff)
            if not projection:
                # No usable inference for the source seat: show the fallback
                # once over the hovered seat's own circle.
                cx, cy = positions[seat]
                self._paint_inference_plate(
                    painter, cx, cy, radius, [tr("board.no_inference")]
                )
                return
            # Draw S's inference about each OTHER non-sheriff target T over T's
            # circle. The source seat's own circle is never drawn on.
            for target_seat, items in projection:
                if target_seat not in positions:
                    continue
                tx, ty = positions[target_seat]
                lines = [
                    f"{inference_role_label(key)} {format_percent(prob)}%"
                    for key, prob in items
                ]
                self._paint_inference_plate(painter, tx, ty, radius, lines)

        def _paint_inference_plate(
            self,
            painter: QPainter,
            cx: float,
            cy: float,
            radius: float,
            lines: list[str],
        ) -> None:
            """Draw one translucent role-inference plate centered on a circle.

            Shared by the per-target projection and the no-info fallback so the
            plate style (dark rounded rect + localized ``role N%`` text lines)
            stays identical. Pure presentation; all math is done upstream.
            """

            painter.save()
            font = QFont()
            font.setBold(True)
            point = font.pointSize()
            if point > 0:
                font.setPointSize(max(8, point - 1))
            painter.setFont(font)
            text = "\n".join(lines)
            box_width = radius * 1.9
            box_height = radius * 1.6
            rect = QRectF(cx - box_width / 2, cy - box_height / 2, box_width, box_height)
            # Dark translucent plate so the overlay text stays legible over the
            # seat fill, then the text itself.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(10, 12, 18, 205)))
            painter.drawRoundedRect(rect, 6.0, 6.0)
            painter.setPen(QPen(QColor(245, 240, 210)))
            painter.drawText(
                rect,
                int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                text,
            )
            painter.restore()

        def _inverted_inference_lines(self, source_seat: int) -> dict[int, list[str]]:
            """Return per-target localized inference lines for a hovered seat.

            Delegates all math to ``spectator_overlay.inverted_role_inference``
            (hovering seat ``source_seat`` S yields S's own beliefs ABOUT each
            OTHER non-sheriff seat T) and only maps each ``role_key`` to its
            localized label via ``inference_role_label`` + ``format_percent`` for
            display. Returns a ``{target_seat: [lines]}`` mapping; an empty dict
            when S is not a dummy or has no usable beliefs (the caller then shows
            the "no inference info" fallback). The sheriff seat (derived from
            revealed_roles) is excluded both as a target and from every
            per-target list (``role_inference_items`` drops the sheriff role
            column), and S's own seat never appears.
            """

            internals = self._seat_internals if self._overlays_enabled else None
            sheriff = sheriff_seat_from_observation(
                getattr(self._state, "observation", None)
            )
            projection = inverted_role_inference(
                internals, int(source_seat), sheriff_seat=sheriff
            )
            return {
                target_seat: [
                    f"{inference_role_label(key)} {format_percent(prob)}%"
                    for key, prob in items
                ]
                for target_seat, items in projection
            }


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

        def set_current_step(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

        def truncate_after(self, *_args: Any, **_kwargs: Any) -> None:
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
    "MAX_EMOTION_BARS",
    "OPPONENT_PANEL_MAX_WIDTH",
    "format_event",
    "inference_role_label",
    "opponent_public_rows",
]
