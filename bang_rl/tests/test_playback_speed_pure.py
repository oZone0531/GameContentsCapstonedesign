"""Sandbox-executable tests for the Qt-free playback / AI-speed helper.

These exercise ``bang_rl/gui/playback_speed.py`` which owns the single source of
truth for the AI-turn-speed slider bounds and the two tiny mapping rules used by
both the live/spectator game screen and the replay screen for FEAT-002's manual
turn-advance mode. The module imports only the standard library, so (like
test_emotion_bar_role_inference_pure.py) this test path-loads it straight from
its file with ``importlib.util.spec_from_file_location`` to bypass
``bang_rl/gui/__init__`` (which pulls in pydantic / PyQt6 / numpy, absent in the
restricted sandbox).

Several assertions are REVERT-SENSITIVE: they fail if ``is_manual_speed`` is
weakened from ``<= AI_SPEED_MIN_MS`` to ``< AI_SPEED_MIN_MS``, or if
``speed_to_interval_ms`` stops returning ``None`` at the minimum. Those cases
are called out inline.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(relative: str, alias: str):
    path = Path(__file__).resolve().parents[1] / "gui" / relative
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


speed = _load_module("playback_speed.py", "_playback_speed_isolated")

MIN = speed.AI_SPEED_MIN_MS
MAX = speed.AI_SPEED_MAX_MS
DEFAULT = speed.AI_SPEED_DEFAULT_MS


# -- constants are a sane single source of truth ---------------------------
def test_constants_ordered_with_default_in_range() -> None:
    assert MIN < MAX
    assert MIN <= DEFAULT <= MAX
    # Pin the documented values so a drift is caught.
    assert (MIN, MAX, DEFAULT) == (200, 3000, 1000)


# -- is_manual_speed: minimum tick (and below) is manual -------------------
def test_is_manual_speed_true_at_and_below_minimum() -> None:
    # REVERT-SENSITIVE: the boundary at exactly the minimum MUST be manual.
    # If `<= AI_SPEED_MIN_MS` is weakened to `<`, this exact-minimum assertion
    # fails loudly.
    assert speed.is_manual_speed(MIN) is True
    # Anything below the minimum is manual too (clamped-manual).
    assert speed.is_manual_speed(MIN - 1) is True
    assert speed.is_manual_speed(0) is True
    assert speed.is_manual_speed(-500) is True


def test_is_manual_speed_false_above_minimum() -> None:
    # REVERT-SENSITIVE companion: one step above the minimum is NOT manual.
    # If the comparison became `>=` (or the None default flipped) this fails.
    assert speed.is_manual_speed(MIN + 1) is False
    assert speed.is_manual_speed(DEFAULT) is False
    assert speed.is_manual_speed(MAX) is False
    # Above the maximum is still a (clamped) auto-advance value, not manual.
    assert speed.is_manual_speed(MAX + 1000) is False


def test_is_manual_speed_tolerates_non_int() -> None:
    # Numeric strings / floats coerce; non-numeric degrades to manual (safe:
    # stop the timer rather than auto-advance at an unknown cadence).
    assert speed.is_manual_speed(float(MIN)) is True
    assert speed.is_manual_speed(float(DEFAULT)) is False
    assert speed.is_manual_speed(str(DEFAULT)) is False
    assert speed.is_manual_speed(str(MIN)) is True
    assert speed.is_manual_speed("oops") is True
    assert speed.is_manual_speed(None) is True
    # A bool is not a valid slider position and must not read as 0/1.
    assert speed.is_manual_speed(True) is True
    assert speed.is_manual_speed(False) is True


# -- speed_to_interval_ms: None in manual, clamped int otherwise -----------
def test_speed_to_interval_none_at_minimum() -> None:
    # REVERT-SENSITIVE: at the minimum the helper MUST return None so the
    # caller does not start its QTimer. If this stops returning None at the
    # minimum (e.g. the manual guard is removed) this fails.
    assert speed.speed_to_interval_ms(MIN) is None
    assert speed.speed_to_interval_ms(MIN - 1) is None
    assert speed.speed_to_interval_ms(0) is None


def test_speed_to_interval_returns_value_above_minimum() -> None:
    assert speed.speed_to_interval_ms(MIN + 1) == MIN + 1
    assert speed.speed_to_interval_ms(DEFAULT) == DEFAULT
    assert speed.speed_to_interval_ms(MAX) == MAX


def test_speed_to_interval_clamps_above_maximum() -> None:
    # Out-of-range-high clamps to the maximum (and is never manual).
    assert speed.speed_to_interval_ms(MAX + 5000) == MAX


def test_speed_to_interval_tolerates_non_int() -> None:
    assert speed.speed_to_interval_ms(float(DEFAULT)) == DEFAULT
    assert speed.speed_to_interval_ms(str(DEFAULT)) == DEFAULT
    # Non-numeric is manual -> None (never raises).
    assert speed.speed_to_interval_ms("oops") is None
    assert speed.speed_to_interval_ms(None) is None
    assert speed.speed_to_interval_ms(True) is None


# -- play_pause_is_playing: shared spectator / replay control state --------
def test_play_pause_is_playing_mirrors_timer_active() -> None:
    # FEAT-003 (item 3): the spectator game screen and the replay screen share
    # the SAME play/pause semantics via this helper. The control reads as
    # "playing" (label = Pause, a click stops) exactly when the auto-advance
    # timer is running. Pulling this into the Qt-free module lets the two
    # screens agree without instantiating a QWidget.
    assert speed.play_pause_is_playing(True) is True
    assert speed.play_pause_is_playing(False) is False
    # Tolerates truthy/falsy inputs (QTimer.isActive() returns a bool, but a
    # None / 0 / "" must read as "not playing", never raise).
    assert speed.play_pause_is_playing(None) is False
    assert speed.play_pause_is_playing(0) is False
    assert speed.play_pause_is_playing("") is False
    assert speed.play_pause_is_playing(1) is True
    assert speed.play_pause_is_playing("running") is True


def test_manual_and_interval_agree() -> None:
    # The two helpers must stay consistent: interval is None exactly when the
    # value is manual, for a sweep across and around the range.
    for value in (-10, 0, MIN - 1, MIN, MIN + 1, DEFAULT, MAX, MAX + 1):
        if speed.is_manual_speed(value):
            assert speed.speed_to_interval_ms(value) is None
        else:
            interval = speed.speed_to_interval_ms(value)
            assert interval is not None
            assert MIN <= interval <= MAX
