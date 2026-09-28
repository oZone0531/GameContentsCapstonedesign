"""Headless-pure tests for the global GUI layout polish (FEAT-001).

None of these require PyQt6: they exercise the module-level constants and the
i18n tooltip strings, which live outside the ``if QT_AVAILABLE:`` gates so they
import and run in CI without the optional ``[gui]`` extra.
"""
from __future__ import annotations

import pytest

from bang_rl.gui.i18n import LANGUAGES, current_language, set_language, tr
from bang_rl.gui.i18n import _STRINGS
from bang_rl.gui.main_window import MIN_FONT_POINT_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH
from bang_rl.gui.widgets.game_widgets import OPPONENT_PANEL_MAX_WIDTH


# Every Settings field/menu item that must expose a mouse-over help tooltip.
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
)


def test_window_is_large_and_fixed_font_is_at_least_10pt() -> None:
    assert WINDOW_WIDTH >= 1024
    assert WINDOW_HEIGHT >= 720
    assert MIN_FONT_POINT_SIZE >= 10


def test_opponent_panel_width_is_bounded() -> None:
    assert 0 < OPPONENT_PANEL_MAX_WIDTH <= WINDOW_WIDTH // 2


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
