"""Headless-pure tests for the global GUI layout polish (FEAT-001).

None of these require PyQt6: they exercise the module-level constants and the
i18n tooltip strings, which live outside the ``if QT_AVAILABLE:`` gates so they
import and run in CI without the optional ``[gui]`` extra.
"""
from __future__ import annotations

import pytest

from bang_rl.gui.i18n import LANGUAGES, current_language, set_language, tr
from bang_rl.gui.i18n import _STRINGS
from bang_rl.gui.main_window import (
    MENU_BUTTON_MIN_HEIGHT,
    MENU_BUTTON_WIDTH,
    MIN_FONT_POINT_SIZE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from bang_rl.gui.widgets.game_widgets import OPPONENT_PANEL_MAX_WIDTH


# Every Settings field/menu item that must expose a mouse-over help tooltip.
# Includes the AI-training settings menu (tooltip.training_*) and the dummy-AI
# emotion settings menu (tooltip.emotion_*) so item 3 is covered end to end.
TOOLTIP_KEYS = (
    "tooltip.language",
    "tooltip.ppo_clip",
    "tooltip.cpu",
    "tooltip.ram",
    "tooltip.workers",
    "tooltip.batch",
    "tooltip.buffer",
    "tooltip.gpu",
    "tooltip.reward_json",
    "tooltip.emotion_initial_values",
    "tooltip.dummy_ai",
    # AI-training settings menu (TrainingSetupPanel) labels + fields
    "tooltip.training_mode",
    "tooltip.training_episodes",
    "tooltip.training_target",
    "tooltip.training_seed",
    "tooltip.training_report_path",
    "tooltip.training_output_path",
    "tooltip.training_artifact_path",
    "tooltip.training_resume_artifact_path",
    # Dummy-AI emotion settings menu (DummyAISettingsDialog) labels + fields
    "tooltip.emotion_schema_version",
    "tooltip.emotion_count",
    "tooltip.emotion_name",
    "tooltip.emotion_change_rate",
    "tooltip.emotion_decay_rate",
    "tooltip.emotion_event_updates",
    "tooltip.emotion_action_updates",
    "tooltip.emotion_reward_updates",
    "tooltip.emotion_terminal_updates",
    "tooltip.emotion_effects",
)


def test_window_is_large_and_fixed_font_is_at_least_10pt() -> None:
    assert WINDOW_WIDTH >= 1024
    assert WINDOW_HEIGHT >= 720
    assert MIN_FONT_POINT_SIZE >= 10


def test_opponent_panel_width_is_bounded() -> None:
    assert 0 < OPPONENT_PANEL_MAX_WIDTH <= WINDOW_WIDTH // 2


def test_menu_button_width_is_about_half_the_window() -> None:
    # User item 1: menu buttons should be about half the window width.
    assert MENU_BUTTON_WIDTH == WINDOW_WIDTH // 2
    assert WINDOW_WIDTH * 0.4 <= MENU_BUTTON_WIDTH <= WINDOW_WIDTH * 0.6
    assert MENU_BUTTON_MIN_HEIGHT >= 40


def test_every_settings_tooltip_key_exists_in_both_languages() -> None:
    for key in TOOLTIP_KEYS:
        assert key in _STRINGS, f"missing tooltip key: {key}"
        entry = _STRINGS[key]
        for language in LANGUAGES:
            assert language in entry, f"{key} missing language {language}"
            assert entry[language].strip(), f"{key} empty for language {language}"


def test_every_settings_tooltip_resolves_via_tr_in_each_language() -> None:
    original = current_language()
    try:
        for language in LANGUAGES:
            set_language(language)
            for key in TOOLTIP_KEYS:
                text = tr(key)
                # tr() returns the key unchanged when no translation is found;
                # a real tooltip must resolve to non-key text.
                assert text != key, f"{key} did not resolve in {language}"
                assert text.strip()
    finally:
        set_language(original)


def test_cpu_tooltip_matches_exact_korean_wording() -> None:
    original = current_language()
    try:
        set_language("ko")
        assert tr("tooltip.cpu") == "학습에 사용되는 CPU 코어의 수를 의미합니다."
    finally:
        set_language(original)


def test_no_em_dash_in_new_tooltip_strings() -> None:
    for key in TOOLTIP_KEYS:
        for text in _STRINGS[key].values():
            assert "\u2014" not in text, f"em dash found in {key}"


# --- Item 3: human caption fit at REALISTIC (non-square) board sizes ------
#
# The reviewer flagged that item 3's caption fit was only checked at a square
# 1000x1000 board (where the seat radius clamps to its max). These tests use the
# SAME geometry the Qt board draws with -- pure board_layout seat placement plus
# the caption metrics copied from GraphicalBoard._paint_caption -- to prove the
# human's three caption lines fit at the realistic minimum/short-wide board
# sizes, not just a square.

from bang_rl.gui.widgets import board_layout
from bang_rl.gui.widgets.game_widgets import (
    GRAPHICAL_BOARD_MIN_HEIGHT,
    GRAPHICAL_BOARD_MIN_WIDTH,
)


def _caption_bottom(width: float, height: float) -> float:
    """Bottom y of the human's third caption line for a width x height board.

    Mirrors GraphicalBoard._paint_caption exactly:
      top = cy + radius + 6
      line_height = radius * 0.34 + 6
      third line drawn at top + 2 * line_height, in a 22px-tall text rect.
    with cy = human seat centre y from the pure board_layout geometry.
    """

    radius = board_layout.circle_radius(width, height)
    _, cy = board_layout.seat_positions(width, height, human_player=0)[0]
    top = cy + radius + 6
    line_height = radius * 0.34 + 6
    caption_text_rect_height = 22
    return top + 2 * line_height + caption_text_rect_height


# Realistic board sizes. The GraphicalBoard enforces a minimum size, and it
# lives inside the fixed WINDOW_WIDTH x WINDOW_HEIGHT window, so the board can
# never render shorter than its own _MIN_HEIGHT. The tightest case is that
# minimum height paired with a wide board (radius clamped by height), which the
# square 1000x1000 check never exercised.
_REALISTIC_BOARD_SIZES = (
    (GRAPHICAL_BOARD_MIN_WIDTH, GRAPHICAL_BOARD_MIN_HEIGHT),  # smallest allowed
    (960, GRAPHICAL_BOARD_MIN_HEIGHT),  # wide + shortest allowed height
    (WINDOW_WIDTH * 3 // 4, GRAPHICAL_BOARD_MIN_HEIGHT),  # ~board share, short
    (WINDOW_WIDTH, GRAPHICAL_BOARD_MIN_HEIGHT),  # extreme wide-short
    (1000, 1000),  # the original square (still must fit)
)


def test_human_caption_fits_at_realistic_board_sizes() -> None:
    for width, height in _REALISTIC_BOARD_SIZES:
        bottom = _caption_bottom(width, height)
        assert bottom <= height, (
            f"caption overflows at {width}x{height}: bottom={bottom:.1f} > {height}"
        )


def test_human_caption_has_comfortable_margin_at_min_board_height() -> None:
    # At the board's own minimum height the three caption lines must not merely
    # touch the edge; keep a visible margin so descenders/anti-aliasing show.
    width = WINDOW_WIDTH
    height = GRAPHICAL_BOARD_MIN_HEIGHT
    margin = height - _caption_bottom(width, height)
    assert margin >= 12, f"only {margin:.1f}px below the third caption line"


def test_human_seat_stays_above_the_old_low_slot() -> None:
    # The seat was raised from the old 0.84 fraction to reclaim top margin AND
    # fit the caption; it must remain clearly above that old position.
    width, height = 1000.0, 1000.0
    _, human_y = board_layout.seat_positions(width, height, human_player=0)[0]
    assert human_y <= 0.72 * height
