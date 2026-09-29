"""Pure-Python geometry helpers for the graphical game board.

These functions are intentionally free of any Qt import so that seat
placement, click hit-testing, fill-color and muzzle/gun-angle math can be
unit-tested headlessly in CI (where PyQt6 is unavailable).  The Qt widget in
``game_widgets.py`` consumes plain ``(x, y)`` tuples produced here and only
converts them to ``QPointF``/``QRectF`` at draw time.

Action contract (shared with :class:`bang_rl.gui.game.GameSessionController`):
``0`` = PASS, ``1..5`` = BANG player_0..player_4.  Clicking opponent player
index ``p`` maps to action ``p + 1``; clicking the human's own seat maps to
action ``0`` (PASS).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

PLAYER_COUNT = 5

# A comfortable fraction of the smaller board dimension, used for each seat
# circle radius.  Kept modest so five circles plus their labels fit inside a
# fixed window without overlapping.
_RADIUS_FRACTION = 0.11
_MIN_RADIUS = 24.0
_MAX_RADIUS = 60.0

# The human seat sits centred along the bottom edge; the four opponents are
# arranged clockwise around the top and sides.  Values are normalized (0..1)
# offsets within the drawable area, later scaled by width/height.
#
# ITEM 3 (reduced top margin): the human seat used to sit at y=0.84, so the
# three caption lines the board draws BELOW the human circle (name, "HP .
# status", role) fell off the bottom once the circle grew toward _MAX_RADIUS.
# Raising the human seat leaves >= 3 caption lines + the radius below it within
# the board height, and lowering the top opponent row from y=0.20 to y=0.14
# reclaims the freed top margin so the five seats stay well spread out.
#
# REVIEW FIX (short/wide windows): item 3 was originally verified only at a
# square 1000x1000 board where the seat radius clamps to its _MAX_RADIUS. The
# tightest realistic case is the board's own minimum height (GraphicalBoard
# _MIN_HEIGHT = 520 inside the fixed 1280x860 window), where the radius is
# clamped by min(w, h) instead. The board draws the last caption line at
#   cy + radius + 6 + 2 * (radius*0.34 + 6) + 22   (see game_widgets._paint_caption),
# with cy = y * height. At y=0.72 that leaves only ~9.5px below a 520px board;
# lowering the human seat to y=0.70 widens the worst-case margin to ~20px so the
# three caption lines fit comfortably at every realistic (non-square, short and
# wide) board size, not just the square where the radius is maxed out. It stays
# < 0.84 so the top-margin reduction that item 3 introduced is preserved.
_HUMAN_SLOT = (0.5, 0.70)
_OPPONENT_SLOTS = (
    (0.16, 0.56),  # left-lower
    (0.24, 0.14),  # left-upper
    (0.76, 0.14),  # right-upper
    (0.84, 0.56),  # right-lower
)


def circle_radius(width: float, height: float) -> float:
    """Return the seat-circle radius for a board of ``width`` x ``height``."""

    base = min(float(width), float(height)) * _RADIUS_FRACTION
    return max(_MIN_RADIUS, min(_MAX_RADIUS, base))


def seat_positions(
    width: float,
    height: float,
    *,
    human_player: int = 0,
    player_count: int = PLAYER_COUNT,
) -> dict[int, tuple[float, float]]:
    """Map each player index to a ``(center_x, center_y)`` on the board.

    The ``human_player`` seat is always placed at the bottom-centre; the other
    players fill the surrounding slots in ascending index order.  Returns a
    dict keyed by player index so callers never rely on list ordering.
    """

    width = float(width)
    height = float(height)
    positions: dict[int, tuple[float, float]] = {}
    positions[human_player] = (_HUMAN_SLOT[0] * width, _HUMAN_SLOT[1] * height)
    others = [player for player in range(player_count) if player != human_player]
    for player, slot in zip(others, _OPPONENT_SLOTS):
        positions[player] = (slot[0] * width, slot[1] * height)
    return positions


def seat_at_point(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    human_player: int = 0,
    player_count: int = PLAYER_COUNT,
) -> int | None:
    """Return the player index whose circle contains ``(x, y)`` or ``None``."""

    radius = circle_radius(width, height)
    positions = seat_positions(width, height, human_player=human_player, player_count=player_count)
    best: tuple[float, int] | None = None
    for player, (cx, cy) in positions.items():
        distance = math.hypot(float(x) - cx, float(y) - cy)
        if distance <= radius and (best is None or distance < best[0]):
            best = (distance, player)
    return None if best is None else best[1]


def action_for_seat(player: int, *, human_player: int = 0) -> int:
    """Map a clicked seat to its action index (0=PASS, p+1=BANG player p)."""

    return 0 if player == human_player else int(player) + 1


def seat_for_action(action: int, *, human_player: int = 0) -> int | None:
    """Inverse of :func:`action_for_seat`; ``None`` when out of range."""

    action = int(action)
    if action == 0:
        return human_player
    seat = action - 1
    return seat if 0 <= seat < PLAYER_COUNT else None


def is_action_legal(action: int, mask: Sequence[int]) -> bool:
    """Return ``True`` when ``action`` is enabled in the current mask."""

    action = int(action)
    if action < 0 or action >= len(mask):
        return False
    return bool(mask[action])


# Fully-faded eliminated seat: near-black with a low alpha so it visibly
# recedes behind the living, coloured seats.
_ELIMINATED_COLOR = (24, 24, 28, 90)


def health_to_color(
    health: int,
    max_health: int,
    alive: bool,
) -> tuple[int, int, int, int]:
    """Return an ``(r, g, b, a)`` fill for a seat from its health/alive state.

    A living seat fades from a calm full-HP blue toward a *vivid* red as its
    health drops: at full HP the colour is clearly not red, at ~1 HP it is a
    near-pure vivid red.  An eliminated seat (``alive`` False or ``health<=0``)
    renders as a faded near-black (low alpha) so it is visually distinct from
    the living, coloured players.  Kept Qt-free so FEAT-004 colour behaviour is
    unit-testable headlessly.
    """

    current = int(health)
    if not bool(alive) or current <= 0:
        return _ELIMINATED_COLOR
    ceiling = max(1, int(max_health))
    ratio = max(0.0, min(1.0, current / ceiling))
    # Vivid red at low HP -> calm blue at full HP. ``damage`` is how hurt the
    # seat is (1.0 at 1/ceiling health, 0.0 at full health).
    damage = 1.0 - ratio
    red = int(round(70 + 185 * damage))
    green = int(round(150 - 120 * damage))
    blue = int(round(210 - 170 * damage))
    return (
        max(0, min(255, red)),
        max(0, min(255, green)),
        max(0, min(255, blue)),
        255,
    )


def seat_fill_color(
    player: int,
    healths: Sequence[int],
    alive: Sequence[bool],
    *,
    max_health: int = 5,
) -> tuple[int, int, int, int]:
    """Return an ``(r, g, b, a)`` fill for ``player`` based on health/alive.

    Thin adapter over :func:`health_to_color` that reads the per-seat values
    out of the public ``healths``/``alive`` sequences.  A living seat reddens
    with damage; an eliminated seat is faded near-black.
    """

    is_alive = 0 <= player < len(alive) and bool(alive[player])
    current = int(healths[player]) if 0 <= player < len(healths) else 0
    return health_to_color(current, max_health, is_alive)


def new_animation_events(
    prev_events: Sequence[Mapping] | None,
    current_events: Sequence[Mapping] | None,
) -> list[Mapping]:
    """Return the bang/elimination events added since ``prev_events``.

    ``event_sequence`` grows monotonically, so the freshly-appended tail is
    ``current_events[len(prev_events):]``.  From that tail we keep only the
    events worth animating: ``bang`` shots and ``player_eliminated``
    eliminations.  Pure and Qt-free so the trigger diff is unit-testable.
    """

    prev = list(prev_events or ())
    current = list(current_events or ())
    start = len(prev) if len(current) >= len(prev) else 0
    tail = current[start:]
    animated = {"bang", "player_eliminated"}
    result: list[Mapping] = []
    for event in tail:
        if not isinstance(event, Mapping):
            continue
        if str(event.get("event_type", "")).strip().lower() in animated:
            result.append(event)
    return result


def muzzle_point(
    shooter: int,
    target: int,
    width: float,
    height: float,
    *,
    human_player: int = 0,
    player_count: int = PLAYER_COUNT,
) -> tuple[float, float] | None:
    """Return the point on the shooter's circle edge facing the target.

    Also usable as the origin of a FEAT-004 shot animation.  Returns ``None``
    when either seat is missing or the two seats coincide.
    """

    result = muzzle_point_and_angle(
        shooter,
        target,
        width,
        height,
        human_player=human_player,
        player_count=player_count,
    )
    return None if result is None else result[0]


def gun_angle(
    shooter: int,
    target: int,
    width: float,
    height: float,
    *,
    human_player: int = 0,
    player_count: int = PLAYER_COUNT,
) -> float | None:
    """Return the gun angle (radians) from ``shooter`` toward ``target``."""

    result = muzzle_point_and_angle(
        shooter,
        target,
        width,
        height,
        human_player=human_player,
        player_count=player_count,
    )
    return None if result is None else result[1]


def facing_angle(
    player: int,
    width: float,
    height: float,
    *,
    human_player: int = 0,
    player_count: int = PLAYER_COUNT,
) -> float:
    """Return the default facing angle (radians) for ``player``.

    Each character normally faces the CENTRE of the board ``(width/2,
    height/2)``.  The angle is measured from the positive x-axis (math
    convention, ``atan2(dy, dx)``).  A seat sitting exactly on the centre (or a
    missing seat) faces straight up (``-pi/2``) as a stable fallback so the
    helper never raises.  Qt-free so FEAT-004 facing geometry is unit-testable.
    """

    positions = seat_positions(width, height, human_player=human_player, player_count=player_count)
    if player not in positions:
        return -math.pi / 2
    sx, sy = positions[player]
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    dx = cx - sx
    dy = cy - sy
    if dx == 0 and dy == 0:
        return -math.pi / 2
    return math.atan2(dy, dx)


def gun_anchor_and_angle(
    player: int,
    width: float,
    height: float,
    *,
    facing: float,
    human_player: int = 0,
    player_count: int = PLAYER_COUNT,
) -> tuple[tuple[float, float], float] | None:
    """Return ``((anchor_x, anchor_y), angle)`` for the gun a character holds.

    The gun sits on the character's *visually*-RIGHT side and its barrel points
    along ``facing``.  Because every seat faces the board centre, the previous
    "right of facing" offset ``(sin(f), -cos(f))`` (facing rotated by
    ``-90 degrees``) rendered the gun on the seat's visual LEFT.  ITEM 1 flips
    the anchor to the opposite perpendicular side, i.e. facing rotated by
    ``+90 degrees`` -> ``(-sin(f), cos(f))``, so the gun now appears on the
    character's visual right.  The returned angle stays equal to ``facing`` so
    the barrel still points along ``facing``.  Returns ``None`` when the seat is
    missing.  Pure and Qt-free so seat/gun placement is unit-testable.
    """

    positions = seat_positions(width, height, human_player=human_player, player_count=player_count)
    if player not in positions:
        return None
    cx, cy = positions[player]
    radius = circle_radius(width, height)
    # Visual-right of the character (ITEM 1): rotate facing by +90 degrees ->
    # (cos(f + pi/2), sin(f + pi/2)) == (-sin(f), cos(f)); the opposite
    # perpendicular of the old (sin(f), -cos(f)) offset that appeared on the left.
    right_x = -math.sin(facing)
    right_y = math.cos(facing)
    anchor = (cx + radius * right_x, cy + radius * right_y)
    return anchor, float(facing)


def interpolate_angle(a: float, b: float, t: float) -> float:
    """Shortest-path angular interpolation from ``a`` to ``b`` at ``t``.

    ``t`` is clamped to ``[0, 1]`` (``t=0`` -> ``a``, ``t=1`` -> ``b``).  The
    step from ``a`` to ``b`` is taken along the shortest direction across the
    ``+/-pi`` wrap, so easing a barrel from its facing angle to the aim angle
    and back never spins the long way round.  Qt-free and unit-testable.
    """

    t = max(0.0, min(1.0, float(t)))
    delta = (float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi
    return float(a) + delta * t


def highlight_color(base_rgba: Sequence[int]) -> tuple[int, int, int, int]:
    """Return a brighter ``(r, g, b, a)`` for a hovered seat's fill.

    Each colour channel is lightened toward ``255`` by ~35% (clamped) so a
    hovered target visibly stands out over the normal seat fill; the alpha is
    preserved.  Pure and Qt-free so the highlight math is unit-testable.
    """

    values = list(base_rgba) + [0, 0, 0, 255]
    r, g, b, a = (int(values[0]), int(values[1]), int(values[2]), int(values[3]))

    def _lighten(channel: int) -> int:
        channel = max(0, min(255, channel))
        return max(0, min(255, int(round(channel + (255 - channel) * 0.35))))

    return (_lighten(r), _lighten(g), _lighten(b), max(0, min(255, a)))


HIDDEN_ROLE_SENTINEL = 255


def _self_role_index(observation: Mapping, human_player: int) -> int:
    """Return the viewer's own role index from ``self_role`` (255 if unknown).

    The public observation carries ``self_role`` (the viewer's own role, always
    known to them) either as a scalar or a one-element sequence.  Returns the
    hidden sentinel when it is absent or unreadable.  Pure and Qt-free.
    """

    self_role = observation.get("self_role")
    if self_role is None:
        return HIDDEN_ROLE_SENTINEL
    try:
        values = list(self_role)
    except TypeError:
        values = [self_role]
    if not values:
        return HIDDEN_ROLE_SENTINEL
    try:
        return int(values[0])
    except (TypeError, ValueError):
        return HIDDEN_ROLE_SENTINEL


def role_for_display(
    player: int,
    observation: Mapping,
    final_roles: Sequence[int] | None,
    game_over: bool,
    *,
    human_player: int = 0,
) -> int:
    """Return the role index to display beneath ``player`` (255 == hidden).

    The single source of truth for item 7's "reveal all at end" decision, kept
    Qt-free so it is unit-testable headlessly:

    * The human's own seat always shows its real role from ``self_role``.
    * When the game is over AND a non-empty ``final_roles`` is supplied (only
      the terminal :class:`PublicGameState` carries it), every seat shows its
      true role from ``final_roles`` so all players are revealed at game end.
    * Otherwise the seat follows the public reveal logic: a role visible in the
      public ``revealed_roles`` (visibility >= 2) is shown, else it stays
      hidden.  No living role ever leaks mid-game.
    """

    if player == int(human_player):
        return _self_role_index(observation, human_player)

    finals = list(final_roles or ())
    if game_over and finals and 0 <= player < len(finals):
        try:
            return int(finals[player])
        except (TypeError, ValueError):
            return HIDDEN_ROLE_SENTINEL

    revealed = observation.get("revealed_roles")
    revealed_values = list(revealed) if revealed is not None else []
    visibility = observation.get("role_visibility")
    visibility_values = list(visibility) if visibility is not None else []

    def _revealed_index() -> int:
        if 0 <= player < len(revealed_values):
            try:
                return int(revealed_values[player])
            except (TypeError, ValueError):
                return HIDDEN_ROLE_SENTINEL
        return HIDDEN_ROLE_SENTINEL

    if game_over:
        return _revealed_index()
    if 0 <= player < len(visibility_values):
        try:
            visible = int(visibility_values[player]) >= 2
        except (TypeError, ValueError):
            visible = False
        if visible:
            return _revealed_index()
    return HIDDEN_ROLE_SENTINEL


def muzzle_point_and_angle(
    shooter: int,
    target: int,
    width: float,
    height: float,
    *,
    human_player: int = 0,
    player_count: int = PLAYER_COUNT,
) -> tuple[tuple[float, float], float] | None:
    """Return ``((muzzle_x, muzzle_y), angle_radians)`` for a (shooter, target).

    The muzzle sits on the shooter circle edge along the line to the target;
    the angle is measured from the positive x-axis (math convention).  Returns
    ``None`` when a seat is missing or the two seats coincide.
    """

    positions = seat_positions(width, height, human_player=human_player, player_count=player_count)
    if shooter not in positions or target not in positions:
        return None
    sx, sy = positions[shooter]
    tx, ty = positions[target]
    dx = tx - sx
    dy = ty - sy
    distance = math.hypot(dx, dy)
    if distance == 0:
        return None
    radius = circle_radius(width, height)
    angle = math.atan2(dy, dx)
    muzzle = (sx + radius * dx / distance, sy + radius * dy / distance)
    return muzzle, angle


__all__ = [
    "HIDDEN_ROLE_SENTINEL",
    "PLAYER_COUNT",
    "action_for_seat",
    "circle_radius",
    "facing_angle",
    "gun_anchor_and_angle",
    "gun_angle",
    "health_to_color",
    "highlight_color",
    "interpolate_angle",
    "is_action_legal",
    "muzzle_point",
    "muzzle_point_and_angle",
    "new_animation_events",
    "role_for_display",
    "seat_at_point",
    "seat_fill_color",
    "seat_for_action",
    "seat_positions",
]
