"""Qt-free, numpy-free, pydantic-free display helpers for the spectator and
replay overlays.

The GUI board (``game_widgets.GraphicalBoard``) and the replay / game screens
consume these helpers to render, under each seat, a proportional emotion bar
(for example ``분노 ========--``) with the per-emotion natural-language
description shown on hover, plus a per-seat role-inference list shown when the
cursor hovers a seat in spectator mode or replay.

All logic lives here (not in the Qt widget) so it can be unit-tested headlessly
in the restricted sandbox, where numpy / pydantic / PyQt6 are not importable.
This module therefore imports ONLY the standard library and must stay that way:
the pure tests path-load it directly (importlib.util.spec_from_file_location),
bypassing ``bang_rl/gui/__init__`` which would otherwise pull in Qt.

Two design decisions are baked into the role-inference helpers and worth
stating up front:

* Sheriff is DROPPED from every inference list. The sheriff role is always
  public in this game, so an AI never has to infer it and showing a sheriff
  probability would be meaningless. ``role_inference_items`` discards index 0.
* The two outlaw seats (``outlaw_a`` + ``outlaw_b``) are MERGED into a single
  ``outlaw`` bucket by summing their probabilities. The user-facing example
  shows a single ``무법자`` entry, and from a spectator's point of view "how
  likely is this player an outlaw" is the meaningful question, not which of the
  two outlaw slots. ``deputy`` and ``traitor`` stay separate.

The canonical belief order used across the project is
``(sheriff, deputy, outlaw_a, outlaw_b, traitor)`` = indices ``0..4``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

# Canonical role order of a belief row / matrix column (indices 0..4).
CANONICAL_ROLE_ORDER = ("sheriff", "deputy", "outlaw_a", "outlaw_b", "traitor")
# The number of roles / seats in a belief row.
ROLE_COUNT = 5
# Stable role keys produced by ``role_inference_items`` after the sheriff is
# dropped and the two outlaw slots are merged, in the fixed order used to break
# ties when two buckets share a probability.
INFERENCE_ROLE_KEYS = ("deputy", "outlaw", "traitor")
_INFERENCE_TIE_ORDER = {key: index for index, key in enumerate(INFERENCE_ROLE_KEYS)}
# Entries at or below this probability are dropped from the inference list.
_PROBABILITY_EPSILON = 1e-9


@dataclass(frozen=True)
class EmotionBar:
    """One emotion's render-ready bar for a seat.

    ``value`` is the clamped [0, 1] intensity, ``description`` is the author's
    natural-language text (empty when none was authored), and ``filled`` /
    ``empty`` are the pre-computed block counts so the caller can build the bar
    string (or draw it) without re-deriving the proportion.
    """

    name: str
    value: float
    description: str
    filled: int
    empty: int


def _clamp_unit(value: float) -> float:
    """Clamp a finite float to the inclusive [0, 1] range."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _coerce_unit_value(value: Any) -> float | None:
    """Return ``value`` coerced to a finite [0, 1] float, or ``None``.

    Booleans and non-numeric / non-finite inputs yield ``None`` so a malformed
    entry can be skipped by the caller rather than raising.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not isfinite(number):
        return None
    return _clamp_unit(number)


def emotion_bar_cells(value: Any, total: int = 10) -> tuple[int, int]:
    """Return ``(filled, empty)`` block counts for ``value`` over ``total``.

    ``value`` is clamped to [0, 1]; ``filled`` is ``round(clamp(value) * total)``
    and ``empty`` is the remainder, so the two always sum to ``total``. A
    non-numeric / non-finite ``value`` is treated as ``0.0`` (an empty bar). A
    non-positive ``total`` yields ``(0, 0)``.
    """

    if total <= 0:
        return (0, 0)
    unit = _coerce_unit_value(value)
    if unit is None:
        unit = 0.0
    filled = int(round(unit * total))
    if filled < 0:
        filled = 0
    if filled > total:
        filled = total
    return (filled, total - filled)


def emotion_bar_string(
    value: Any,
    total: int = 10,
    filled_char: str = "=",
    empty_char: str = "-",
) -> str:
    """Return a proportional bar such as ``========--`` for ``value``.

    The caller builds the user-facing ``분노 ========--`` style line as
    ``f"{name} {bar}"``; this helper only renders the bar itself so the label
    and spacing stay a presentation concern.
    """

    filled, empty = emotion_bar_cells(value, total)
    return filled_char * filled + empty_char * empty


def _unwrap_state(descriptor: Any) -> Mapping[str, Any] | None:
    """Return the inner dummy ``state`` dict from a seat descriptor.

    Accepts either the full ``{'kind': 'dummy', 'state': {...}}`` wrapper or the
    inner state dict directly. Returns ``None`` for anything that is not a dummy
    seat (``kind`` present and not ``'dummy'``) or not a mapping at all.
    """

    if not isinstance(descriptor, Mapping):
        return None
    kind = descriptor.get("kind")
    if kind is not None:
        if kind != "dummy":
            return None
        state = descriptor.get("state")
        if isinstance(state, Mapping):
            return state
        return None
    # No ``kind`` key: treat the mapping as an already-unwrapped state dict.
    return descriptor


def _description_lookup(config: Any) -> dict[str, str]:
    """Build a ``{emotion_name: description}`` map from an emotion config dict.

    Tolerant of a missing / malformed ``config`` or ``emotions`` list and of
    individual entries that are not dicts or lack a usable name.
    """

    lookup: dict[str, str] = {}
    if not isinstance(config, Mapping):
        return lookup
    emotions = config.get("emotions")
    if not isinstance(emotions, Sequence) or isinstance(emotions, (str, bytes)):
        return lookup
    for entry in emotions:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = entry.get("description")
        if isinstance(description, str):
            lookup[name] = description
        else:
            lookup[name] = ""
    return lookup


def emotion_bars_for_seat(descriptor: Any, total: int = 10) -> tuple[EmotionBar, ...]:
    """Return the render-ready emotion bars for a dummy seat descriptor.

    ``descriptor`` is a dummy-seat descriptor shaped
    ``{'kind': 'dummy', 'state': {'emotion_module': {'emotion_names': [...],
    'values': [...], 'config': {'emotions': [{'name': ..., 'description': ...},
    ...]}}}}`` (the full wrapper or just the inner state dict is accepted).

    Each value is paired by index with its emotion name (``emotion_names``) and
    with the per-emotion description looked up by name from
    ``config['emotions']`` (falling back to ``""`` when the description is
    absent / empty or the emotion is not found). A non-dummy descriptor, or one
    missing ``emotion_module`` / ``values``, yields an empty tuple so non-dummy
    seats render no bars. Malformed entries (short lists, non-numeric values)
    are skipped rather than raising.
    """

    state = _unwrap_state(descriptor)
    if state is None:
        return ()
    emotion_module = state.get("emotion_module")
    if not isinstance(emotion_module, Mapping):
        return ()
    values = emotion_module.get("values")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    names = emotion_module.get("emotion_names")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        names = ()
    descriptions = _description_lookup(emotion_module.get("config"))

    bars: list[EmotionBar] = []
    for index, raw_value in enumerate(values):
        unit = _coerce_unit_value(raw_value)
        if unit is None:
            continue
        if index < len(names) and isinstance(names[index], str) and names[index]:
            name = names[index]
        else:
            name = f"emotion_{index}"
        description = descriptions.get(name, "")
        if not isinstance(description, str):
            description = ""
        filled, empty = emotion_bar_cells(unit, total)
        bars.append(
            EmotionBar(
                name=name,
                value=unit,
                description=description,
                filled=filled,
                empty=empty,
            )
        )
    return tuple(bars)


def _coerce_belief_row(belief_row: Any) -> tuple[float, ...] | None:
    """Return ``belief_row`` as exactly ``ROLE_COUNT`` finite floats, or ``None``."""

    if belief_row is None:
        return None
    if not isinstance(belief_row, Sequence) or isinstance(belief_row, (str, bytes)):
        return None
    if len(belief_row) != ROLE_COUNT:
        return None
    row: list[float] = []
    for item in belief_row:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not isfinite(number):
            return None
        row.append(number)
    return tuple(row)


def role_inference_items(belief_row: Any) -> tuple[tuple[str, float], ...]:
    """Return descending ``(role_key, probability)`` inference items.

    ``belief_row`` is a length-5 sequence of probabilities in canonical order
    ``(sheriff, deputy, outlaw_a, outlaw_b, traitor)``. The sheriff (index 0) is
    DROPPED because that role is always public, and the two outlaw slots
    (indices 2 and 3) are MERGED into a single ``outlaw`` bucket by summing
    their probabilities. ``deputy`` and ``traitor`` are kept separate. The
    result is sorted by probability descending (ties broken by the fixed
    ``INFERENCE_ROLE_KEYS`` order), and buckets with probability at or below a
    tiny epsilon are dropped.

    Percentage formatting is intentionally left to the caller (see
    ``format_percent``) so this pure math stays independent of presentation.
    A ``None`` / wrong-length / non-numeric ``belief_row`` yields an empty tuple
    so the caller can fall back to a "no inference info" message.
    """

    row = _coerce_belief_row(belief_row)
    if row is None:
        return ()
    # Index 0 (sheriff) is deliberately never read: it is always public.
    buckets = {
        "deputy": row[1],
        "outlaw": row[2] + row[3],
        "traitor": row[4],
    }
    items = [
        (key, probability)
        for key, probability in buckets.items()
        if probability > _PROBABILITY_EPSILON
    ]
    items.sort(key=lambda item: (-item[1], _INFERENCE_TIE_ORDER[item[0]]))
    return tuple(items)


def role_belief_row_for_seat(descriptor: Any, seat_index: Any) -> tuple[float, ...] | None:
    """Return seat ``seat_index``'s belief row from a dummy seat descriptor.

    ``descriptor`` is a dummy-seat descriptor whose state carries
    ``role_estimator['beliefs']`` as a 5x5 list. The belief matrix is this dummy
    AI's belief about EVERY seat; row ``seat_index`` is that AI's probability
    distribution over seat ``seat_index``'s role.

    Per the user's request (hovering a player shows that player's own AI
    inference), the caller hovers seat S and passes ``seat_index=S`` to read
    seat S's distribution; the caller is free to choose which row it wants.

    Returns the validated length-5 row (five finite numbers) or ``None`` when
    the descriptor is not a dummy, lacks ``role_estimator`` / ``beliefs``, or the
    requested row is missing or malformed. Never raises on bad input.
    """

    state = _unwrap_state(descriptor)
    if state is None:
        return None
    if isinstance(seat_index, bool) or not isinstance(seat_index, int):
        return None
    estimator = state.get("role_estimator")
    if not isinstance(estimator, Mapping):
        return None
    beliefs = estimator.get("beliefs")
    if not isinstance(beliefs, Sequence) or isinstance(beliefs, (str, bytes)):
        return None
    if seat_index < 0 or seat_index >= len(beliefs):
        return None
    return _coerce_belief_row(beliefs[seat_index])


def format_percent(probability: Any) -> int:
    """Return ``round(probability * 100)`` as an int for display.

    Kept tiny and separate from ``role_inference_items`` so the display rounding
    (used to build the user's ``무법자 87%`` style line) is testable and
    revert-sensitive. A non-numeric / non-finite probability yields ``0``.
    """

    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        return 0
    number = float(probability)
    if not isfinite(number):
        return 0
    return int(round(number * 100))
