"""Regression test for the turn-based log step label (user item 1).

CI-ONLY: this imports ``bang_rl.gui.i18n``, and importing anything under
``bang_rl.gui`` executes ``bang_rl/gui/__init__`` which pulls in PyQt6 and
pydantic. Those dependencies are unavailable in the restricted sandbox, so this
test can only run in CI where the full GUI stack is installed.

It pins the wording change for the per-row log prefix produced by the
``event.step`` key: Korean now reads ``"{step}턴 "`` (number first, then 턴)
instead of the old ``"[스텝 {step}] "``, and English reads ``"Turn {step} "``.
The single trailing space is load-bearing because ``format_event()`` builds a
row as ``prefix + " ".join(parts)``.
"""
from __future__ import annotations

from bang_rl.gui.i18n import _STRINGS

# U+2014 EM DASH is banned repo-wide; assert the label never reintroduces it.
_EM_DASH = "\u2014"


def test_event_step_label_korean_is_turn_based() -> None:
    ko = _STRINGS["event.step"]["ko"]
    rendered = ko.format(step=3)
    assert "3턴" in rendered
    assert "스텝" not in rendered
    # The trailing space that format_event() relies on must survive.
    assert rendered.endswith(" ")


def test_event_step_label_english_is_turn_based() -> None:
    en = _STRINGS["event.step"]["en"]
    rendered = en.format(step=3)
    assert "Turn 3" in rendered
    assert rendered.endswith(" ")


def test_event_step_label_has_no_em_dash() -> None:
    entry = _STRINGS["event.step"]
    assert _EM_DASH not in entry["ko"]
    assert _EM_DASH not in entry["en"]
