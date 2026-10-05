"""Sandbox-executable tests for the Qt-free spectator / replay overlay helpers.

These exercise ``bang_rl/gui/widgets/spectator_overlay.py``, which backs the
emotion bars under each seat and the per-seat role-inference list shown on hover
in spectator mode and replay. The module imports only the standard library, so
(like test_replay_internals_pure.py) this test path-loads it straight from its
file with ``importlib.util.spec_from_file_location`` to bypass
``bang_rl/gui/__init__`` (which pulls in pydantic / PyQt6 / numpy, absent in the
restricted sandbox).

Several assertions are REVERT-SENSITIVE: they fail if the sheriff-drop or the
outlaw-merge in ``role_inference_items`` is removed, or if the bar proportion
math is reverted. Those cases are called out inline.
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


overlay = _load_module("widgets/spectator_overlay.py", "_spectator_overlay_isolated")


def _dummy_descriptor() -> dict:
    return {
        "kind": "dummy",
        "state": {
            "emotion_module": {
                "emotion_names": ["분노", "공포", "기쁨"],
                "values": [1.0, 0.5, 0.0],
                "config": {
                    "emotions": [
                        {"name": "분노", "description": "상대의 공격에 대한 분노"},
                        {"name": "공포", "description": "위협을 느낄 때의 공포"},
                        {"name": "기쁨", "description": ""},
                    ]
                },
            },
            "role_estimator": {
                "beliefs": [
                    [0.6, 0.1, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.45, 0.42, 0.13],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.1, 0.2, 0.3, 0.2, 0.2],
                    [0.0, 0.5, 0.0, 0.0, 0.5],
                ],
            },
        },
    }


# -- emotion_bar_cells / emotion_bar_string proportions --------------------
def test_emotion_bar_cells_at_extremes_and_mid() -> None:
    # Revert-sensitive: exact filled/empty counts for the proportion math.
    assert overlay.emotion_bar_cells(0.0, total=10) == (0, 10)
    assert overlay.emotion_bar_cells(1.0, total=10) == (10, 0)
    assert overlay.emotion_bar_cells(0.5, total=10) == (5, 5)
    # Rounding: 0.47 * 10 = 4.7 -> 5 filled.
    assert overlay.emotion_bar_cells(0.47, total=10) == (5, 5)
    # Clamping out-of-range values.
    assert overlay.emotion_bar_cells(-3.0, total=10) == (0, 10)
    assert overlay.emotion_bar_cells(2.0, total=10) == (10, 0)
    # Non-numeric degrades to an empty bar, never raises.
    assert overlay.emotion_bar_cells("oops", total=10) == (0, 10)
    assert overlay.emotion_bar_cells(0.5, total=0) == (0, 0)


def test_emotion_bar_string_renders_fill_and_empty() -> None:
    assert overlay.emotion_bar_string(1.0, total=10) == "=" * 10
    assert overlay.emotion_bar_string(0.0, total=10) == "-" * 10
    # Mid value: 5 filled, 5 empty -> the user's '분노 ========--' style.
    assert overlay.emotion_bar_string(0.5, total=10) == "=====-----"
    assert overlay.emotion_bar_string(0.3, total=10, filled_char="#", empty_char=".") == "###......."


# -- emotion_bars_for_seat --------------------------------------------------
def test_emotion_bars_for_seat_pairs_names_values_descriptions() -> None:
    bars = overlay.emotion_bars_for_seat(_dummy_descriptor(), total=10)
    assert len(bars) == 3
    assert [bar.name for bar in bars] == ["분노", "공포", "기쁨"]
    assert [bar.value for bar in bars] == [1.0, 0.5, 0.0]
    assert bars[0].description == "상대의 공격에 대한 분노"
    assert bars[1].description == "위협을 느낄 때의 공포"
    # Graceful degrade: an empty / absent description surfaces as "".
    assert bars[2].description == ""
    assert (bars[0].filled, bars[0].empty) == (10, 0)
    assert (bars[1].filled, bars[1].empty) == (5, 5)


def test_emotion_bars_for_seat_accepts_inner_state_dict() -> None:
    inner = _dummy_descriptor()["state"]
    bars = overlay.emotion_bars_for_seat(inner)
    assert [bar.name for bar in bars] == ["분노", "공포", "기쁨"]


def test_emotion_bars_for_seat_empty_for_non_dummy_or_random() -> None:
    assert overlay.emotion_bars_for_seat({"kind": "random"}) == ()
    assert overlay.emotion_bars_for_seat({"kind": "human"}) == ()
    assert overlay.emotion_bars_for_seat({"kind": "main"}) == ()
    assert overlay.emotion_bars_for_seat(None) == ()
    assert overlay.emotion_bars_for_seat(42) == ()
    # Dummy but no emotion_module / values -> no bars.
    assert overlay.emotion_bars_for_seat({"kind": "dummy", "state": {}}) == ()


def test_emotion_bars_for_seat_skips_malformed_values() -> None:
    descriptor = {
        "kind": "dummy",
        "state": {
            "emotion_module": {
                "emotion_names": ["a", "b", "c"],
                "values": [0.5, "bad", 0.25],
            }
        },
    }
    bars = overlay.emotion_bars_for_seat(descriptor)
    # The non-numeric middle entry is skipped; the other two survive.
    assert [bar.name for bar in bars] == ["a", "c"]
    # Missing config -> every description degrades to "".
    assert all(bar.description == "" for bar in bars)


# -- role_inference_items: sheriff drop + outlaw merge ---------------------
def test_role_inference_drops_sheriff_and_merges_outlaws() -> None:
    # Canonical order: sheriff, deputy, outlaw_a, outlaw_b, traitor.
    row = [0.6, 0.1, 0.1, 0.1, 0.1]
    items = overlay.role_inference_items(row)
    keys = [key for key, _ in items]
    # Revert-sensitive (sheriff-drop): sheriff must never appear.
    assert "sheriff" not in keys
    # Revert-sensitive (outlaw-merge): exactly one merged 'outlaw' bucket whose
    # probability is the SUM of outlaw_a + outlaw_b (0.1 + 0.1 = 0.2), which
    # would be wrong (0.1, and two separate entries) if the merge were removed.
    outlaw = dict(items)["outlaw"]
    assert abs(outlaw - 0.2) < 1e-9
    assert keys.count("outlaw") == 1
    # Merged outlaw (0.2) now outranks deputy (0.1) and traitor (0.1).
    assert keys[0] == "outlaw"
    assert set(keys) == {"deputy", "outlaw", "traitor"}


def test_role_inference_sorts_descending_and_drops_zero() -> None:
    row = [0.0, 0.5, 0.0, 0.0, 0.5]
    items = overlay.role_inference_items(row)
    # deputy 0.5 and traitor 0.5; outlaw sums to 0.0 and is dropped.
    assert [key for key, _ in items] == ["deputy", "traitor"]
    assert all(prob > 0 for _, prob in items)
    # Descending order holds for a clear ranking.
    row2 = [0.0, 0.05, 0.4, 0.45, 0.1]
    items2 = overlay.role_inference_items(row2)
    assert [key for key, _ in items2] == ["outlaw", "traitor", "deputy"]
    probs = [prob for _, prob in items2]
    assert probs == sorted(probs, reverse=True)
    assert abs(dict(items2)["outlaw"] - 0.85) < 1e-9


def test_role_inference_graceful_degrade() -> None:
    assert overlay.role_inference_items(None) == ()
    assert overlay.role_inference_items([0.5, 0.5]) == ()  # wrong length
    assert overlay.role_inference_items([0.2, 0.2, 0.2, 0.2, "x"]) == ()  # non-numeric
    assert overlay.role_inference_items("nope") == ()


# -- role_belief_row_for_seat ----------------------------------------------
def test_role_belief_row_for_seat_returns_requested_row() -> None:
    descriptor = _dummy_descriptor()
    row = overlay.role_belief_row_for_seat(descriptor, 1)
    assert row == (0.0, 0.0, 0.45, 0.42, 0.13)
    # Feeding it into the inference pipeline merges the two outlaw slots.
    items = overlay.role_inference_items(row)
    assert items[0][0] == "outlaw"
    assert abs(dict(items)["outlaw"] - 0.87) < 1e-9


def test_role_belief_row_for_seat_none_for_non_dummy_or_bad_index() -> None:
    assert overlay.role_belief_row_for_seat({"kind": "random"}, 0) is None
    assert overlay.role_belief_row_for_seat(None, 0) is None
    assert overlay.role_belief_row_for_seat(_dummy_descriptor(), 99) is None
    assert overlay.role_belief_row_for_seat(_dummy_descriptor(), -1) is None
    assert overlay.role_belief_row_for_seat(_dummy_descriptor(), True) is None
    # Dummy but missing role_estimator -> None.
    assert overlay.role_belief_row_for_seat({"kind": "dummy", "state": {}}, 0) is None


# -- format_percent ---------------------------------------------------------
def test_format_percent_rounds() -> None:
    assert overlay.format_percent(0.87) == 87
    assert overlay.format_percent(0.0) == 0
    assert overlay.format_percent(1.0) == 100
    assert overlay.format_percent(0.034) == 3
    assert overlay.format_percent(0.125) == 12  # round-half-to-even: 12.5 -> 12
    assert overlay.format_percent("bad") == 0
