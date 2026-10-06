"""Headless-pure tests for the FEAT-004 graphical-board effects.

None of these import PyQt6: they exercise the pure ``board_layout`` helpers
(HP-based colour, muzzle/gun-angle geometry, new-event diff), all of which live
outside the ``if QT_AVAILABLE:`` gates so they run in CI without the optional
``[gui]`` extra.
"""
from __future__ import annotations

import math

from bang_rl.gui.i18n import LANGUAGES
from bang_rl.gui.i18n import _STRINGS
from bang_rl.gui.widgets.board_layout import (
    circle_radius,
    facing_angle,
    gun_anchor_and_angle,
    health_to_color,
    highlight_color,
    interpolate_angle,
    muzzle_point_and_angle,
    new_animation_events,
    seat_fill_color,
    seat_positions,
)


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


# -- FEAT-004 facing / gun-on-right / angle-lerp / highlight geometry ------

def test_facing_angle_points_each_seat_toward_board_center() -> None:
    width, height = 800.0, 800.0
    cx, cy = width / 2.0, height / 2.0
    positions = seat_positions(width, height)
    for player in range(5):
        sx, sy = positions[player]
        angle = facing_angle(player, width, height)
        # The unit facing vector must point from the seat toward the centre:
        # its dot product with (centre - seat) is positive.
        fx, fy = math.cos(angle), math.sin(angle)
        dx, dy = cx - sx, cy - sy
        assert fx * dx + fy * dy > 0, f"seat {player} should face the board centre"


def test_gun_sits_on_the_right_of_the_facing_direction() -> None:
    width, height = 800.0, 800.0
    positions = seat_positions(width, height)
    for player in range(5):
        facing = facing_angle(player, width, height)
        result = gun_anchor_and_angle(player, width, height, facing=facing)
        assert result is not None
        (ax, ay), angle = result
        # The barrel points along the facing angle.
        assert math.isclose(angle, facing, abs_tol=1e-9)
        cx, cy = positions[player]
        # Offset of the anchor from the seat centre lies on the RIGHT of the
        # facing direction. Right-of-facing in screen coords (y down) is
        # (sin f, -cos f); the dot product with the offset is positive.
        off_x, off_y = ax - cx, ay - cy
        right_x, right_y = math.sin(facing), -math.cos(facing)
        assert off_x * right_x + off_y * right_y > 0


def test_gun_anchor_none_for_missing_seat() -> None:
    assert gun_anchor_and_angle(99, 800, 800, facing=0.0) is None


def test_interpolate_angle_endpoints_and_shortest_path() -> None:
    assert math.isclose(interpolate_angle(0.5, 1.5, 0.0), 0.5, abs_tol=1e-9)
    assert math.isclose(interpolate_angle(0.5, 1.5, 1.0), 1.5, abs_tol=1e-9)
    # Halfway is the midpoint for a small, non-wrapping span.
    assert math.isclose(interpolate_angle(0.0, 1.0, 0.5), 0.5, abs_tol=1e-9)
    # Across the +/-pi wrap the shortest path steps the SHORT way: from
    # 3.0 rad to -3.0 rad the shortest move is +~0.283 rad (through +pi),
    # not the -6.0 rad long way.
    result = interpolate_angle(3.0, -3.0, 0.5)
    delta = (-3.0 - 3.0 + math.pi) % (2.0 * math.pi) - math.pi
    assert math.isclose(result, 3.0 + delta * 0.5, abs_tol=1e-9)
    assert delta > 0, "shortest path from 3.0 to -3.0 goes the short (positive) way"


def test_interpolate_angle_clamps_t() -> None:
    assert math.isclose(interpolate_angle(0.2, 0.8, -1.0), 0.2, abs_tol=1e-9)
    assert math.isclose(interpolate_angle(0.2, 0.8, 5.0), 0.8, abs_tol=1e-9)


def test_highlight_color_brightens_each_channel_and_clamps() -> None:
    r, g, b, a = highlight_color((70, 150, 210, 255))
    assert r > 70 and g > 150 and b > 210
    assert a == 255
    # Every channel stays within range and a bright input clamps at 255.
    assert all(0 <= channel <= 255 for channel in (r, g, b, a))
    assert highlight_color((255, 255, 255, 255)) == (255, 255, 255, 255)
    # Short inputs are tolerated (missing alpha defaults to opaque).
    r2, g2, b2, a2 = highlight_color((0, 0, 0))
    assert (r2, g2, b2) > (0, 0, 0) and a2 == 255


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


# -- i18n key presence ------------------------------------------------------

def test_new_i18n_keys_exist_in_both_languages() -> None:
    keys = ("board.hit", "board.eliminated")
    for key in keys:
        assert key in _STRINGS, f"missing key {key}"
        for language in LANGUAGES:
            assert _STRINGS[key][language].strip(), f"{key} empty for {language}"


def test_no_em_dash_in_new_board_strings() -> None:
    keys = ("board.hit", "board.eliminated")
    for key in keys:
        for text in _STRINGS[key].values():
            assert "\u2014" not in text


def test_feat004_game_i18n_keys_exist_in_both_languages() -> None:
    keys = (
        "game.winner_banner",
        "game.no_winner_banner",
        "game.time_travel_hint",
        "game.time_travel_hint_terminal",
    )
    for key in keys:
        assert key in _STRINGS, f"missing key {key}"
        for language in LANGUAGES:
            assert _STRINGS[key][language].strip(), f"{key} empty for {language}"
            assert "\u2014" not in _STRINGS[key][language]
