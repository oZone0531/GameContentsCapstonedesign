"""Qt-free playback / AI-turn speed mapping (single source of truth).

This stdlib-only module owns the AI-turn-speed slider bounds and the two tiny
mapping rules used by BOTH the live/spectator game screen
(``bang_rl/gui/screens/game_screen.py``) and the replay screen
(``bang_rl/gui/screens/replay_screen.py``):

* :data:`AI_SPEED_MIN_MS` / :data:`AI_SPEED_MAX_MS` / :data:`AI_SPEED_DEFAULT_MS`
  are the slider's minimum / maximum / default interval in milliseconds. The
  slider minimum doubles as the "manual" position.
* :func:`is_manual_speed` answers "is this slider value the manual position?".
  Manual means the slider is at (or below) its minimum tick: auto-advance stops
  and the user must press a Next-turn button to advance exactly one turn.
* :func:`speed_to_interval_ms` maps a slider value to the QTimer interval, or to
  ``None`` in manual mode (meaning "do not auto-advance").

Keeping these here (not inside the Qt-gated screen modules) lets the pure test
``bang_rl/tests/test_playback_speed_pure.py`` path-load this file with
``importlib.util.spec_from_file_location`` and run in the restricted sandbox
where PyQt6 is unavailable. ``game_screen.py`` re-exports the constants so the
existing ``from .game_screen import AI_SPEED_...`` in ``replay_screen.py`` keeps
working.
"""
from __future__ import annotations

# AI-turn-speed slider bounds, in milliseconds per AI turn. Default is ~1s per
# turn. The minimum tick is also the "manual" position (see is_manual_speed).
AI_SPEED_MIN_MS = 200
AI_SPEED_MAX_MS = 3000
AI_SPEED_DEFAULT_MS = 1000


def _coerce_int(value: object) -> int | None:
    """Best-effort int() of a slider value; None when non-numeric.

    Qt hands us an ``int`` in practice, but tolerate floats / numeric strings
    and never raise. ``bool`` is intentionally rejected (``True``/``False`` are
    not meaningful slider positions) so a stray boolean does not read as 0/1.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def is_manual_speed(value: object) -> bool:
    """Return True when the slider value means "manual" (no auto-advance).

    Manual is the slider's minimum position: ``int(value) <= AI_SPEED_MIN_MS``.
    Using ``<=`` (not ``<``) means the exact minimum tick IS manual, and any
    out-of-range value below the minimum is treated as manual too. A value that
    cannot be coerced to an int degrades to manual (the safe choice: it stops
    the timer rather than auto-advancing at an unknown cadence).
    """

    coerced = _coerce_int(value)
    if coerced is None:
        return True
    return coerced <= AI_SPEED_MIN_MS


def speed_to_interval_ms(value: object) -> int | None:
    """Map a slider value to a QTimer interval, or None in manual mode.

    Returns ``None`` whenever :func:`is_manual_speed` is True (the caller must
    then NOT start its timer). Otherwise returns the interval clamped into
    ``[AI_SPEED_MIN_MS, AI_SPEED_MAX_MS]``.
    """

    if is_manual_speed(value):
        return None
    coerced = _coerce_int(value)
    if coerced is None:  # pragma: no cover - is_manual_speed already caught this
        return None
    if coerced < AI_SPEED_MIN_MS:
        return AI_SPEED_MIN_MS
    if coerced > AI_SPEED_MAX_MS:
        return AI_SPEED_MAX_MS
    return coerced


def play_pause_is_playing(timer_active: object) -> bool:
    """Return True when the play/pause control should read as "playing".

    FEAT-003 (item 3): the spectator game screen and the replay screen share the
    same play/pause semantics. The control reads as "playing" (so its label is
    the Pause string and a click should stop) exactly when the auto-advance
    timer is running. Pulling this trivial decision into the Qt-free module lets
    the pure test assert the spectator and replay controls agree without
    instantiating a QWidget, and keeps the two screens from diverging.
    """

    return bool(timer_active)


def should_enter_spectator_review(is_spectator: object, game_over: object) -> bool:
    """Return True when a spectator log-click should enter terminal review mode.

    FEAT-003 (item 4) and the user request scope the replay-like review ("게임이
    종료되었을 때 로그를 클릭하면") to AFTER a SPECTATOR game is terminal. A click
    (or a prev-turn press) during live spectator play must NOT build and pin a
    review model, because the terminal-review model hijacks the live AI timer
    (``_advance_one_ai_turn`` steps the recording instead of the game) and the
    screen has no path to leave it. Gating entry here keeps a live spectator
    click on the Qt-free, non-hijacking time-travel preview.

    Pulling this gate into the Qt-free module lets the pure test characterize
    the review-entry contract (spectator AND game-over) without a QWidget: the
    defect the reviewer confirmed (review entered on ANY click regardless of the
    ``_game_over`` flag) is exactly a wrong answer from this function.
    """

    return bool(is_spectator) and bool(game_over)


__all__ = [
    "AI_SPEED_DEFAULT_MS",
    "AI_SPEED_MAX_MS",
    "AI_SPEED_MIN_MS",
    "is_manual_speed",
    "play_pause_is_playing",
    "should_enter_spectator_review",
    "speed_to_interval_ms",
]
