"""CI-only tests for the FEAT-002 spectator / replay overlay on GraphicalBoard.

These construct a real :class:`GraphicalBoard` (so they need PyQt6, which is
absent in the restricted sandbox and skipped there) and verify, through the
board's own gating + the pure overlay helpers it delegates to, that:

* emotion bars and the role-inference hover overlay render ONLY in spectator
  layout (``human_player is None``) or when a replay explicitly enables them,
  and NEVER in a human live game (hidden-information safety);
* FEAT-001 inverted projection: hovering a dummy seat S projects S's OWN beliefs
  ABOUT each OTHER non-sheriff seat T (drawn over T's circle), each a descending
  localized list with the sheriff dropped and the two outlaws merged; the
  sheriff seat 0 and S itself are never targets, and a non-dummy / unavailable
  source seat yields an empty projection so the "no inference info" fallback is
  drawn over S instead.

The exhaustive pure assertions live in test_emotion_bar_role_inference_pure.py
(the helper math) and this file only proves the board wires them up and honours
the overlay gate.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _spectator_state() -> SimpleNamespace:
    observation = {
        "healths": [5, 5, 5, 5, 5],
        "alive": [True] * 5,
        "role_visibility": [2, 2, 2, 2, 2],
        "revealed_roles": [0, 1, 2, 3, 4],
        "self_role": [255],
    }
    return SimpleNamespace(
        observation=observation,
        action_mask=(0, 0, 0, 0, 0, 0),
        current_player=1,
        environment_step=1,
        terminated=False,
        truncated=False,
        termination_status="running",
        final_roles=(),
        reveal_all_roles=True,
    )


def _dummy_internals() -> dict:
    dummy = {
        "kind": "dummy",
        "state": {
            "emotion_module": {
                "emotion_names": ["분노", "공포"],
                "values": [1.0, 0.5],
                "config": {
                    "emotions": [
                        {"name": "분노", "description": "상대의 공격에 대한 분노"},
                        {"name": "공포", "description": ""},
                    ]
                },
            },
            # Seat 1's OWN beliefs about every seat. Row T is seat 1's
            # distribution over seat T's role. The inverted hover projection
            # reads these rows (row 0 for the sheriff target is never drawn, and
            # row 1 for seat 1 itself is never a target).
            "role_estimator": {
                "beliefs": [
                    [0.2, 0.2, 0.2, 0.2, 0.2],  # row 0: about sheriff (not a target)
                    [0.2, 0.2, 0.2, 0.2, 0.2],  # row 1: about self (not a target)
                    [0.0, 0.03, 0.45, 0.42, 0.1],  # row 2: outlaw-heavy
                    [0.0, 0.6, 0.1, 0.1, 0.2],  # row 3: deputy-heavy
                    [0.0, 0.1, 0.1, 0.1, 0.7],  # row 4: traitor-heavy
                ]
            },
        },
    }
    return {
        "health": [5, 5, 5, 5, 5],
        "seats": {
            "0": {"kind": "random"},
            "1": dummy,
            "2": {"kind": "random"},
            "3": {"kind": "random"},
            "4": {"kind": "random"},
        },
    }


_APP = None


def _app():
    # Keep a module-level reference: a QApplication that is only returned and
    # discarded is garbage-collected at once, and the next QWidget aborts Python.
    global _APP
    from bang_rl.gui.qt_compat import QApplication

    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_overlays_disabled_in_human_live_game() -> None:
    pytest.importorskip("PyQt6")
    _app()
    from bang_rl.gui.widgets.game_widgets import GraphicalBoard

    board = GraphicalBoard(human_player=0)
    board.update_state(_spectator_state())
    board.set_seat_internals(_dummy_internals())
    # Hidden-information safety: a human seat keeps overlays off even if
    # internals were set, and the descriptor lookup refuses to expose them.
    assert board._overlays_enabled is False
    assert board._descriptor_for_seat(1) is None
    # The inverted projection reads no internals when overlays are off.
    assert board._inverted_inference_lines(1) == {}


def test_overlays_enabled_in_spectator_layout() -> None:
    pytest.importorskip("PyQt6")
    _app()
    from bang_rl.gui.widgets.game_widgets import GraphicalBoard

    board = GraphicalBoard(human_player=None)
    board.update_state(_spectator_state())
    board.set_seat_internals(_dummy_internals())
    assert board._overlays_enabled is True
    descriptor = board._descriptor_for_seat(1)
    assert descriptor is not None and descriptor["kind"] == "dummy"


def test_spectator_inverted_inference_draws_source_beliefs_over_others() -> None:
    pytest.importorskip("PyQt6")
    _app()
    from bang_rl.gui.i18n import tr
    from bang_rl.gui.widgets.game_widgets import GraphicalBoard

    board = GraphicalBoard(human_player=None)
    board.update_state(_spectator_state())
    board.set_seat_internals(_dummy_internals())
    # Hovering seat 1 (S) projects S's OWN beliefs ABOUT each other non-sheriff
    # seat T onto T's circle. Targets are 2, 3, 4; never 0 (sheriff) and never
    # 1 (the hovered seat itself).
    projection = board._inverted_inference_lines(1)
    assert sorted(projection.keys()) == [2, 3, 4]
    assert 0 not in projection
    assert 1 not in projection
    # Target 2: seat 1's row 2 merges outlaw_a + outlaw_b (0.45 + 0.42 = 0.87)
    # and drops the sheriff, so the merged outlaw entry leads at 87%.
    lines2 = projection[2]
    assert lines2[0] == f"{tr('role.outlaw_merged')} 87%"
    assert lines2[1] == f"{tr('role.traitor')} 10%"
    assert lines2[2] == f"{tr('role.deputy')} 3%"
    # Target 3 is deputy-heavy and target 4 is traitor-heavy, proving each
    # target reads its OWN row of the SOURCE's matrix (not a single shared row).
    assert projection[3][0] == f"{tr('role.deputy')} 60%"
    assert projection[4][0] == f"{tr('role.traitor')} 70%"
    # Sheriff never appears in any per-target list.
    for lines in projection.values():
        assert all(tr("role.sheriff") not in line for line in lines)


def test_non_dummy_source_seat_yields_empty_projection() -> None:
    pytest.importorskip("PyQt6")
    _app()
    from bang_rl.gui.widgets.game_widgets import GraphicalBoard

    board = GraphicalBoard(human_player=None)
    board.update_state(_spectator_state())
    board.set_seat_internals(_dummy_internals())
    # Seat 2 is a random seat (no beliefs matrix) -> empty projection so the
    # painter shows the localized "no inference info" fallback over seat 2.
    assert board._inverted_inference_lines(2) == {}


def test_replay_can_force_overlays_on_with_human_seat() -> None:
    pytest.importorskip("PyQt6")
    _app()
    from bang_rl.gui.widgets.game_widgets import GraphicalBoard

    # A human-played replay keeps a human seat but the replay screen forces
    # overlays on (a replay is a post-hoc review, not live hidden-info play).
    board = GraphicalBoard(human_player=0)
    assert board._overlays_enabled is False
    board.set_overlays_enabled(True)
    assert board._overlays_enabled is True
    board.update_state(_spectator_state())
    board.set_seat_internals(_dummy_internals())
    assert board._descriptor_for_seat(1) is not None
