"""Headless-pure tests for the FEAT-004 graphical-board effects.

None of these import PyQt6: they exercise the pure ``board_layout`` helpers
(HP-based colour, muzzle/gun-angle geometry, new-event diff) and the
``format_log_moment`` log-hover text, all of which live outside the
``if QT_AVAILABLE:`` gates so they run in CI without the optional ``[gui]``
extra.
"""
from __future__ import annotations

import math

from bang_rl.gui.i18n import LANGUAGES, current_language, set_language
from bang_rl.gui.i18n import _STRINGS
from bang_rl.gui.widgets.board_layout import (
    circle_radius,
    health_to_color,
    muzzle_point_and_angle,
    new_animation_events,
    seat_fill_color,
    seat_positions,
)
from bang_rl.gui.widgets.game_widgets import format_log_moment


# -- HP-based circle colour -------------------------------------------------

def test_full_hp_is_not_red() -> None:
    r, g, b, a = health_to_color(5, 5, alive=True)
    assert a == 255
    # A calm full-HP colour: blue clearly dominates red.
    assert b > r
    assert not (r > 150 and g < 90 and b < 90), "full HP should not look red"


def test_low_hp_is_vivid_red() -> None:
    r, g, b, a = health_to_color(1, 5, alive=True)
    assert a == 255
    assert r > 200, "1 HP should be a vivid red"
    assert r > g and r > b


def test_color_reddens_monotonically_as_hp_drops() -> None:
    reds = [health_to_color(hp, 5, alive=True)[0] for hp in (5, 4, 3, 2, 1)]
    assert reds == sorted(reds), "red channel should rise as HP falls"


def test_eliminated_is_black_and_faded() -> None:
    r, g, b, a = health_to_color(0, 5, alive=True)
    assert (r, g, b) == (24, 24, 28)
    assert a < 128, "eliminated seat should be faded (low alpha)"
    # alive=False also collapses to the same faded near-black.
    assert health_to_color(3, 5, alive=False) == (24, 24, 28, 90)


def test_seat_fill_color_reads_sequences() -> None:
    healths = [5, 1, 3, 0, 4]
    alive = [True, True, True, False, True]
    assert seat_fill_color(0, healths, alive) == health_to_color(5, 5, alive=True)
    assert seat_fill_color(1, healths, alive) == health_to_color(1, 5, alive=True)
    # Eliminated seat (alive False) -> faded near-black regardless of health.
    assert seat_fill_color(3, healths, alive) == (24, 24, 28, 90)
    # Out-of-range seat is treated as not-alive.
    assert seat_fill_color(9, healths, alive) == (24, 24, 28, 90)


# -- muzzle / gun-angle geometry -------------------------------------------

def test_muzzle_sits_on_shooter_circle_edge_toward_target() -> None:
    width, height = 800.0, 800.0
    radius = circle_radius(width, height)
    positions = seat_positions(width, height)
    for shooter, target in ((0, 2), (3, 1)):
        result = muzzle_point_and_angle(shooter, target, width, height)
        assert result is not None
        (mx, my), angle = result
        sx, sy = positions[shooter]
        # Muzzle is exactly one radius from the shooter centre.
        assert math.isclose(math.hypot(mx - sx, my - sy), radius, rel_tol=1e-6)
        # Angle points from shooter toward target.
        tx, ty = positions[target]
        expected = math.atan2(ty - sy, tx - sx)
        assert math.isclose(angle, expected, abs_tol=1e-9)


def test_muzzle_none_for_missing_or_coincident_seats() -> None:
    assert muzzle_point_and_angle(0, 0, 800, 800) is None
    assert muzzle_point_and_angle(0, 99, 800, 800) is None


# -- new-event diff ---------------------------------------------------------

def _bang(step: int, actor: int, target: int) -> dict:
    return {"event_type": "bang", "environment_step": step, "actor_idx": actor, "target_idx": target}


def test_new_animation_events_returns_only_new_bang_and_elimination() -> None:
    prev = (_bang(1, 0, 1), {"event_type": "turn_started", "environment_step": 2})
    current = prev + (
        {"event_type": "turn_started", "environment_step": 3},
        _bang(4, 2, 3),
        {"event_type": "player_eliminated", "environment_step": 4, "eliminated_idx": 3},
    )
    new = new_animation_events(prev, current)
    assert [e["event_type"] for e in new] == ["bang", "player_eliminated"]
    assert new[0]["actor_idx"] == 2 and new[0]["target_idx"] == 3


def test_new_animation_events_empty_when_no_growth() -> None:
    events = (_bang(1, 0, 1),)
    assert new_animation_events(events, events) == []


def test_new_animation_events_handles_none_and_reset() -> None:
    assert new_animation_events(None, None) == []
    current = (_bang(1, 0, 1),)
    # A shorter current (reset) rescans the whole current sequence.
    assert new_animation_events((_bang(1, 0, 1), _bang(2, 1, 2)), current) == [current[0]]


# -- log-moment hover text --------------------------------------------------

def test_format_log_moment_uses_step_in_both_languages() -> None:
    original = current_language()
    try:
        set_language("en")
        assert format_log_moment({"environment_step": 7}) == "Recorded at step 7"
        set_language("ko")
        assert format_log_moment({"environment_step": 7}) == "스텝 7 시점에 기록됨"
    finally:
        set_language(original)


def test_format_log_moment_includes_round_when_present() -> None:
    original = current_language()
    try:
        set_language("en")
        text = format_log_moment({"environment_step": 7, "round_index": 2})
        assert "step 7" in text and "round 2" in text
        set_language("ko")
        text_ko = format_log_moment({"environment_step": 7, "round_index": 2})
        assert "7" in text_ko and "2" in text_ko
    finally:
        set_language(original)


def test_format_log_moment_unknown_when_no_step() -> None:
    original = current_language()
    try:
        set_language("en")
        assert format_log_moment({}) == "Recorded moment unavailable"
    finally:
        set_language(original)


# -- i18n key presence ------------------------------------------------------

def test_new_i18n_keys_exist_in_both_languages() -> None:
    keys = ("log.moment", "log.moment_round", "log.moment_unknown", "board.hit", "board.eliminated")
    for key in keys:
        assert key in _STRINGS, f"missing key {key}"
        for language in LANGUAGES:
            assert _STRINGS[key][language].strip(), f"{key} empty for {language}"


def test_no_em_dash_in_new_board_strings() -> None:
    keys = ("log.moment", "log.moment_round", "log.moment_unknown", "board.hit", "board.eliminated")
    for key in keys:
        for text in _STRINGS[key].values():
            assert "\u2014" not in text
