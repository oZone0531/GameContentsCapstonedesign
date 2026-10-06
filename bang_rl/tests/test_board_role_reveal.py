"""Headless tests for item 7's "reveal all roles at game end" decision.

These import ONLY the Qt-free ``board_layout`` module (no PyQt6, numpy, or
pydantic) so they collect and run in the sandbox as well as in CI. They pin the
pure ``role_for_display`` helper that :class:`GraphicalBoard._paint_caption`
delegates to, proving that every seat is revealed once the game is over while no
living role leaks during play.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_board_layout():
    """Import ``board_layout`` directly from its file.

    ``import bang_rl.gui.widgets.board_layout`` executes ``bang_rl/gui/__init__``
    which pulls in PyQt6/pydantic-backed modules that are absent in the sandbox.
    ``board_layout`` is itself Qt-free, so we load it straight from its path so
    this test collects and runs headlessly (and identically in CI).
    """

    path = Path(__file__).resolve().parents[1] / "gui" / "widgets" / "board_layout.py"
    spec = importlib.util.spec_from_file_location("_board_layout_isolated", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_board_layout = _load_board_layout()
HIDDEN_ROLE_SENTINEL = _board_layout.HIDDEN_ROLE_SENTINEL
role_for_display = _board_layout.role_for_display


# A public observation where seat 1 (a living non-sheriff, non-self player) is
# still hidden mid-game: revealed_roles[1] == 255, visibility[1] == 0.
_OBSERVATION = {
    "self_role": [0],  # the human at seat 0 knows their own role (Sheriff).
    "role_visibility": [3, 0, 0, 3, 0],
    "revealed_roles": [0, 255, 255, 2, 255],
}

# The terminal PublicGameState carries the true role of every seat.
_FINAL_ROLES = (0, 4, 2, 3, 1)


def test_human_seat_always_shows_own_role() -> None:
    # During play and at game end the human's own role is always known.
    assert role_for_display(0, _OBSERVATION, (), False, human_player=0) == 0
    assert role_for_display(0, _OBSERVATION, _FINAL_ROLES, True, human_player=0) == 0


def test_living_hidden_seat_stays_hidden_during_play() -> None:
    # Mid-game a living non-sheriff, non-self seat must NOT leak its role.
    assert (
        role_for_display(1, _OBSERVATION, (), False, human_player=0)
        == HIDDEN_ROLE_SENTINEL
    )


def test_publicly_visible_seat_shows_during_play() -> None:
    # A seat the public projection already reveals (visibility >= 2) shows.
    assert role_for_display(3, _OBSERVATION, (), False, human_player=0) == 2


def test_game_over_reveals_every_seat_from_final_roles() -> None:
    # Once the game is over final_roles reveals even the still-hidden living
    # seat, overriding the public hidden logic.
    assert role_for_display(1, _OBSERVATION, _FINAL_ROLES, True, human_player=0) == 4
    assert role_for_display(2, _OBSERVATION, _FINAL_ROLES, True, human_player=0) == 2


def test_game_over_without_final_roles_falls_back_to_public() -> None:
    # A replay snapshot or test stub without final_roles must not crash and
    # falls back to the public revealed_roles at game end.
    assert (
        role_for_display(1, _OBSERVATION, (), True, human_player=0)
        == HIDDEN_ROLE_SENTINEL
    )
    assert role_for_display(3, _OBSERVATION, (), True, human_player=0) == 2


def test_helper_reads_final_roles_from_a_state_like_namespace() -> None:
    # Mirror how GraphicalBoard reads getattr(state, "final_roles", ()): a
    # terminal PublicGameState-like object exposes the tuple; a snapshot stub
    # without it defaults to empty and never raises.
    terminal = SimpleNamespace(terminated=True, truncated=False, final_roles=_FINAL_ROLES)
    snapshot = SimpleNamespace(terminated=True, truncated=False)
    assert (
        role_for_display(
            1, _OBSERVATION, getattr(terminal, "final_roles", ()), True, human_player=0
        )
        == 4
    )
    assert (
        role_for_display(
            1, _OBSERVATION, getattr(snapshot, "final_roles", ()), True, human_player=0
        )
        == HIDDEN_ROLE_SENTINEL
    )
