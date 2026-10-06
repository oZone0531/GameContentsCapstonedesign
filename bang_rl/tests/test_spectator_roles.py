"""Headless tests for spectator role reveal and the no-human-seat layout.

Covers user items 1 and 2 at the Qt-free ``board_layout`` boundary:

* ``role_for_display(..., reveal_all=True)`` shows every seat's true role
  DURING play (spectator, all-AI game), while ``reveal_all=False`` keeps the
  existing human-game privacy byte-for-byte unchanged.
* ``seat_positions(..., human_player=None)`` lays out all five seats with no
  dedicated bottom "you" slot so the spectator board never implies a human.

These import ONLY the Qt-free ``board_layout`` module (no PyQt6/numpy/pydantic)
so they collect and run in the sandbox as well as in CI, mirroring
``test_board_role_reveal.py``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_board_layout():
    path = Path(__file__).resolve().parents[1] / "gui" / "widgets" / "board_layout.py"
    spec = importlib.util.spec_from_file_location("_board_layout_spectator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_board_layout = _load_board_layout()
HIDDEN_ROLE_SENTINEL = _board_layout.HIDDEN_ROLE_SENTINEL
role_for_display = _board_layout.role_for_display
seat_positions = _board_layout.seat_positions
_HUMAN_SLOT = _board_layout._HUMAN_SLOT


# A SPECTATOR observation the service produces: projected from a stable viewer
# (seat 0) and rewritten so every seat is REVEALED with its true role. There is
# no human whose hidden info must be protected.
_SPECTATOR_OBS = {
    "self_role": [0],
    "role_visibility": [2, 2, 2, 2, 2],
    "revealed_roles": [0, 4, 2, 3, 1],
}

# A HUMAN observation mid-game: seat 1 (living non-sheriff) stays hidden.
_HUMAN_OBS = {
    "self_role": [0],
    "role_visibility": [1, 0, 0, 2, 0],
    "revealed_roles": [0, 255, 255, 2, 255],
}
_FINAL_ROLES = (0, 4, 2, 3, 1)


def test_reveal_all_shows_every_seat_true_role_during_play() -> None:
    # Item 1: with reveal_all=True every seat (including seat 0) shows its true
    # role during play, independent of role_visibility and game_over.
    for player, expected in enumerate(_SPECTATOR_OBS["revealed_roles"]):
        assert (
            role_for_display(
                player,
                _SPECTATOR_OBS,
                (),
                False,
                human_player=None,
                reveal_all=True,
            )
            == expected
        )


def test_reveal_all_falls_back_to_final_roles_when_revealed_absent() -> None:
    # If the observation lacks a seat's revealed role, reveal_all falls back to
    # final_roles rather than hiding it.
    obs = {"revealed_roles": [0, 255, 255, 255, 255]}
    assert (
        role_for_display(1, obs, _FINAL_ROLES, True, human_player=None, reveal_all=True)
        == _FINAL_ROLES[1]
    )


def test_reveal_all_hides_only_when_truly_absent() -> None:
    obs = {"revealed_roles": [0]}
    assert (
        role_for_display(3, obs, (), False, human_player=None, reveal_all=True)
        == HIDDEN_ROLE_SENTINEL
    )


def test_default_reveal_all_false_keeps_human_privacy_unchanged() -> None:
    # Item 7 privacy invariant: with reveal_all=False (default) the human seat
    # shows self_role, a living non-sheriff seat stays hidden mid-game, and a
    # publicly-visible seat shows.
    assert role_for_display(0, _HUMAN_OBS, (), False, human_player=0) == 0
    assert (
        role_for_display(1, _HUMAN_OBS, (), False, human_player=0)
        == HIDDEN_ROLE_SENTINEL
    )
    assert role_for_display(3, _HUMAN_OBS, (), False, human_player=0) == 2
    # And at terminal every seat is revealed from final_roles.
    assert role_for_display(1, _HUMAN_OBS, _FINAL_ROLES, True, human_player=0) == 4


def test_spectator_seat_positions_have_no_dedicated_human_slot() -> None:
    # Item 2: with human_player=None all five seats are laid out and none sits
    # in the bottom-centre "you" slot alone as a lone human seat. All five
    # normalized slots are distinct.
    positions = seat_positions(1000.0, 800.0, human_player=None)
    assert set(positions.keys()) == {0, 1, 2, 3, 4}
    assert len(set(positions.values())) == 5


def test_human_seat_positions_unchanged_by_default() -> None:
    # The default human layout still places seat 0 at the bottom-centre slot.
    positions = seat_positions(1000.0, 800.0)
    assert positions[0] == (_HUMAN_SLOT[0] * 1000.0, _HUMAN_SLOT[1] * 800.0)
