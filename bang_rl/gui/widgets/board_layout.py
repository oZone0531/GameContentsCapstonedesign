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
_HUMAN_SLOT = (0.5, 0.84)
_OPPONENT_SLOTS = (
    (0.16, 0.60),  # left-lower
    (0.24, 0.20),  # left-upper
    (0.76, 0.20),  # right-upper
    (0.84, 0.60),  # right-lower
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
    "PLAYER_COUNT",
    "action_for_seat",
    "circle_radius",
    "gun_angle",
    "health_to_color",
    "is_action_legal",
    "muzzle_point",
    "muzzle_point_and_angle",
    "new_animation_events",
    "seat_at_point",
    "seat_fill_color",
    "seat_for_action",
    "seat_positions",
]
