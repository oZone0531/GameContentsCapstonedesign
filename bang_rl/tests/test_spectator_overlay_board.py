"""CI-only tests for the FEAT-002 spectator / replay overlay on GraphicalBoard.

These construct a real :class:`GraphicalBoard` (so they need PyQt6, which is
absent in the restricted sandbox and skipped there) and verify, through the
board's own gating + the pure overlay helpers it delegates to, that:

* emotion bars and the role-inference hover overlay render ONLY in spectator
  layout (``human_player is None``) or when a replay explicitly enables them,
  and NEVER in a human live game (hidden-information safety);
* a hovered dummy seat produces a descending localized role-inference list with
  the sheriff dropped and the two outlaws merged; a non-dummy / unavailable seat
  shows the localized "no inference info" fallback.

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
            "role_estimator": {
                "beliefs": [
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.0, 0.03, 0.45, 0.42, 0.1],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                ]
            },
        },
    }
    return {
        "health": [5, 5, 5, 5, 5],
        "seats": {"1": dummy, "2": {"kind": "random"}},
    }


def _app():
    from bang_rl.gui.qt_compat import QApplication

    return QApplication.instance() or QApplication([])


def test_overlays_disabled_in_human_live_game() -> None:
    pytest.importorskip("PyQt6")
    _app()
    from bang_rl.gui.i18n import tr
    from bang_rl.gui.widgets.game_widgets import GraphicalBoard

    board = GraphicalBoard(human_player=0)
    board.update_state(_spectator_state())
    board.set_seat_internals(_dummy_internals())
    # Hidden-information safety: a human seat keeps overlays off even if
    # internals were set, and the descriptor lookup refuses to expose them.
    assert board._overlays_enabled is False
    assert board._descriptor_for_seat(1) is None
    assert board._role_inference_lines(1) == [tr("board.no_inference")]


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


def test_spectator_role_inference_lines_are_descending_and_localized() -> None:
    pytest.importorskip("PyQt6")
    _app()
    from bang_rl.gui.i18n import tr
    from bang_rl.gui.widgets.game_widgets import GraphicalBoard

    board = GraphicalBoard(human_player=None)
    board.update_state(_spectator_state())
    board.set_seat_internals(_dummy_internals())
    lines = board._role_inference_lines(1)
    # Seat 1's belief row merges outlaw_a + outlaw_b (0.45 + 0.42 = 0.87) and
    # drops the sheriff, so the merged outlaw entry leads at 87%.
    assert lines[0] == f"{tr('role.outlaw_merged')} 87%"
    # Descending: traitor (10%) then deputy (3%). Sheriff never appears.
    assert lines[1] == f"{tr('role.traitor')} 10%"
    assert lines[2] == f"{tr('role.deputy')} 3%"
    assert all(tr("role.sheriff") not in line for line in lines)


def test_non_dummy_seat_shows_no_inference_fallback() -> None:
    pytest.importorskip("PyQt6")
    _app()
    from bang_rl.gui.i18n import tr
    from bang_rl.gui.widgets.game_widgets import GraphicalBoard

    board = GraphicalBoard(human_player=None)
    board.update_state(_spectator_state())
    board.set_seat_internals(_dummy_internals())
    # Seat 2 is a random seat (no beliefs) -> localized fallback.
    assert board._role_inference_lines(2) == [tr("board.no_inference")]


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
