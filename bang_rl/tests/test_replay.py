"""Headless tests for the pure replay model (no Qt required)."""
from __future__ import annotations

from bang_rl.gui.replay import ReplayModel, ReplaySnapshot
from bang_rl.gui.widgets.board_layout import new_animation_events


def _events() -> list[dict]:
    """A small public event stream: two bangs and an elimination."""

    return [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 0, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 1, "target_idx": 0},
        {
            "environment_step": 2,
            "event_type": "player_eliminated",
            "actor_idx": 2,
            "target_idx": 3,
            "eliminated_idx": 3,
        },
    ]


def _mapping_episode() -> dict:
    """A CanonicalEpisode.to_dict-like mapping with three turns + a final state."""

    def observation(healths, alive, current):
        return {
            "healths": tuple(healths),
            "alive": tuple(alive),
            "role_visibility": (1, 0, 0, 0, 0),
            "revealed_roles": (0, 255, 255, 255, 255),
            "self_role": (0,),
            "current_player_idx": (current,),
            "environment_step": current if current < 3 else 0,
        }

    steps = [
        {
            "environment_step": 0,
            "actor_idx": 0,
            "action": 2,
            "action_mask": (1, 1, 0, 0, 0, 0),
            "pre_observation": observation((5, 5, 5, 5, 5), (True,) * 5, 0),
            "terminated": False,
            "truncated": False,
        },
        {
            "environment_step": 1,
            "actor_idx": 1,
            "action": 1,
            "action_mask": (1, 1, 0, 0, 0, 0),
            "pre_observation": observation((5, 4, 5, 5, 5), (True,) * 5, 1),
            "terminated": False,
            "truncated": False,
        },
        {
            "environment_step": 2,
            "actor_idx": 2,
            "action": 4,
            "action_mask": (1, 0, 0, 1, 0, 0),
            "pre_observation": observation((4, 4, 5, 5, 5), (True,) * 5, 2),
            "terminated": False,
            "truncated": False,
        },
    ]
    final_state = {
        "healths": (4, 4, 5, 0, 5),
        "alive": (True, True, True, False, True),
        "role_visibility": (1, 0, 0, 0, 0),
        "revealed_roles": (0, 255, 255, 255, 255),
        "self_role": (0,),
    }
    return {
        "steps": steps,
        "events": _events(),
        "final_state": final_state,
        "termination_status": "normal",
    }


def test_len_matches_turns_plus_final_state() -> None:
    model = ReplayModel(_mapping_episode())
    # Three recorded steps + the appended final settled state.
    assert len(model) == 4


def test_next_and_prev_clamp_at_the_ends() -> None:
    model = ReplayModel(_mapping_episode())
    assert model.index == 0
    model.prev()  # already at start -> clamped
    assert model.index == 0
    for _ in range(10):
        model.next()
    assert model.index == len(model) - 1
    model.next()  # clamped at the last turn
    assert model.index == len(model) - 1
    model.prev()
    assert model.index == len(model) - 2


def test_seek_moves_the_cursor_and_returns_snapshot() -> None:
    model = ReplayModel(_mapping_episode())
    snapshot = model.seek(2)
    assert model.index == 2
    assert isinstance(snapshot, ReplaySnapshot)
    assert snapshot.environment_step == 2
    # Out-of-range seeks clamp rather than raise.
    model.seek(99)
    assert model.index == len(model) - 1
    model.seek(-5)
    assert model.index == 0


def test_snapshot_carries_cumulative_event_sequence() -> None:
    model = ReplayModel(_mapping_episode())
    first = model.snapshot_at(0)
    assert first is not None
    # At step 0 only the game_started + first bang are visible.
    first_events = first.observation["event_sequence"]
    assert [event["event_type"] for event in first_events] == ["game_started", "bang"]
    # At step 2 every event up to and including the elimination is visible.
    third = model.snapshot_at(2)
    assert third is not None
    assert any(event["event_type"] == "player_eliminated" for event in third.observation["event_sequence"])


def test_snapshot_at_step_returns_expected_healths_and_current_player() -> None:
    model = ReplayModel(_mapping_episode())
    index = model.index_for_step(1)
    assert index == 1
    snapshot = model.snapshot_at(index)
    assert snapshot is not None
    assert snapshot.current_player == 1
    assert tuple(snapshot.observation["healths"]) == (5, 4, 5, 5, 5)
    # The final state shows the eliminated player.
    final = model.snapshot_at(len(model) - 1)
    assert final is not None
    assert tuple(final.observation["alive"]) == (True, True, True, False, True)
    assert final.terminated is True


def test_moment_lookup_maps_an_event_to_its_snapshot() -> None:
    model = ReplayModel(_mapping_episode())
    elimination = _events()[-1]
    snapshot = model.snapshot_for_moment(elimination)
    assert snapshot is not None
    assert snapshot.environment_step == 2
    # A step-int moment resolves identically.
    assert model.index_for_step(2) == 2


def test_log_row_hover_maps_to_the_moment_snapshot() -> None:
    model = ReplayModel(_mapping_episode())
    # Move to the last turn so the cumulative log contains every event.
    model.seek(len(model) - 1)
    events = model.current_events()
    assert len(events) == len(_events())
    # Row 1 is the first bang recorded at step 0 -> snapshot index 0.
    assert model.index_for_log_row(1) == 0
    # The elimination row (last) maps to the step-2 snapshot.
    assert model.index_for_log_row(len(events) - 1) == 2
    # Out-of-range rows return None.
    assert model.index_for_log_row(999) is None


def test_log_row_hover_maps_full_log_with_cursor_at_start() -> None:
    # Regression for the item-8 bug: the live GameLog renders the FULL
    # cumulative log, but the freshly built ReplayModel sits at cursor 0 whose
    # own event_sequence is the SHORTEST log. Rows past the opening turn must
    # still map to the right snapshot WITHOUT first seeking to the end.
    model = ReplayModel(_mapping_episode())
    assert model.index == 0
    # The full log the user sees has every recorded event.
    full = model.full_events()
    assert len(full) == len(_events())
    # The short current-cursor log only holds the opening turn's events.
    assert len(model.current_events()) < len(full)
    # A row PAST the opening turn (the elimination) still resolves correctly
    # even though the cursor never left index 0.
    assert model.index_for_log_row(len(full) - 1) == 2
    # The step-1 bang row maps to the step-1 snapshot.
    assert model.index_for_log_row(2) == 1
    # Resolving the row did not disturb the cursor.
    assert model.index == 0


def test_event_diff_between_consecutive_snapshots_yields_new_bangs() -> None:
    model = ReplayModel(_mapping_episode())
    prev = model.snapshot_at(0).observation["event_sequence"]
    current = model.snapshot_at(1).observation["event_sequence"]
    new_events = new_animation_events(prev, current)
    # The second bang (step 1) is the freshly-appended animatable event.
    assert [event["event_type"] for event in new_events] == ["bang"]
    assert new_events[0]["actor_idx"] == 1
    assert new_events[0]["target_idx"] == 0


def test_plain_observation_list_source_is_supported() -> None:
    observations = [
        {
            "healths": (5, 5, 5, 5, 5),
            "alive": (True,) * 5,
            "current_player_idx": (0,),
            "environment_step": 0,
            "event_sequence": ({"environment_step": 0, "event_type": "bang", "actor_idx": 0, "target_idx": 1},),
        },
        {
            "healths": (5, 4, 5, 5, 5),
            "alive": (True,) * 5,
            "current_player_idx": (1,),
            "environment_step": 1,
            "event_sequence": (
                {"environment_step": 0, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
                {"environment_step": 1, "event_type": "bang", "actor_idx": 1, "target_idx": 0},
            ),
        },
    ]
    model = ReplayModel(observations)
    assert len(model) == 2
    assert model.snapshot_at(1).current_player == 1
    # Cumulative event sequence at turn 0 contains only the first bang.
    assert len(model.snapshot_at(0).observation["event_sequence"]) == 1


def test_replay_i18n_keys_exist_in_both_languages() -> None:
    from bang_rl.gui.i18n import _STRINGS

    for key in (
        "menu.replay",
        "replay.title",
        "replay.play",
        "replay.pause",
        "replay.next",
        "replay.prev",
        "replay.autoplay",
        "replay.speed",
        "replay.status",
        "replay.no_episode",
    ):
        assert key in _STRINGS
        assert "en" in _STRINGS[key]
        assert "ko" in _STRINGS[key]


def test_empty_episode_is_safe() -> None:
    model = ReplayModel([])
    assert len(model) == 0
    assert model.is_empty()
    assert model.current() is None
    assert model.next() is None
    assert model.index_for_step(0) is None


def _terminal_reveal_episode() -> dict:
    """A replay record with a top-level final_roles so the END reveals all."""

    def observation(current):
        return {
            "healths": (5, 5, 5, 5, 5),
            "alive": (True,) * 5,
            # Mid-game only the sheriff (seat 0) is publicly revealed; every
            # living non-sheriff seat is hidden (255).
            "role_visibility": (3, 0, 0, 0, 0),
            "revealed_roles": (0, 255, 255, 255, 255),
            "self_role": (0,),
            "current_player_idx": (current,),
            "environment_step": current,
        }

    steps = [
        {
            "environment_step": step,
            "actor_idx": step,
            "action_mask": (1, 1, 0, 0, 0, 0),
            "pre_observation": observation(step),
            "terminated": False,
            "truncated": False,
        }
        for step in range(3)
    ]
    final_state = {
        "healths": (5, 5, 5, 0, 5),
        "alive": (True, True, True, False, True),
        "role_visibility": (3, 0, 0, 3, 0),
        "revealed_roles": (0, 255, 255, 3, 255),
        "self_role": (0,),
    }
    return {
        "steps": steps,
        "events": _events(),
        "final_state": final_state,
        "termination_status": "normal",
        # Item 1: the finished game's terminal role of every seat.
        "final_roles": [0, 2, 1, 3, 4],
    }


def test_replay_board_reveals_every_role_at_terminal_snapshot() -> None:
    # CI-only (needs the real GraphicalBoard / Qt): at the FINAL replay snapshot
    # every seat caption resolves to a concrete role, while at a MID snapshot a
    # known-hidden living seat still shows hidden. Skipped in the sandbox where
    # PyQt6 is absent.
    import pytest

    pytest.importorskip("PyQt6")
    from bang_rl.gui.qt_compat import QApplication
    from bang_rl.gui.widgets.board_layout import HIDDEN_ROLE_SENTINEL, role_for_display
    from bang_rl.gui.widgets.game_widgets import GraphicalBoard

    app = QApplication.instance() or QApplication([])
    assert app is not None

    model = ReplayModel(_terminal_reveal_episode())
    board = GraphicalBoard()
    board.human_player = 0

    def _caption_role(seat: int) -> int:
        # Resolve a seat's displayed role exactly as GraphicalBoard._paint_caption
        # does: from the live board state (self._state) that update_state set,
        # using the snapshot's observation + final_roles + game_over flag.
        state = board._state
        observation = getattr(state, "observation", {}) or {}
        return role_for_display(
            seat,
            observation,
            getattr(state, "final_roles", ()),
            board._is_game_over(),
            human_player=board.human_player,
            reveal_all=bool(getattr(state, "reveal_all_roles", False)),
        )

    # Terminal snapshot: game_over is True and final_roles is threaded through,
    # so every seat caption resolves to a concrete role (none hidden).
    terminal = model.snapshot_at(len(model) - 1)
    assert terminal is not None
    assert terminal.terminated or terminal.truncated
    board.update_state(terminal)
    assert board._is_game_over() is True
    for seat, expected in enumerate([0, 2, 1, 3, 4]):
        resolved = _caption_role(seat)
        assert resolved != HIDDEN_ROLE_SENTINEL, f"seat {seat} must be revealed at end"
        assert resolved == expected

    # Mid snapshot: a known-hidden living non-sheriff seat (1) stays hidden.
    mid = model.snapshot_at(0)
    assert mid is not None
    assert mid.final_roles == ()
    board.update_state(mid)
    assert board._is_game_over() is False
    assert _caption_role(1) == HIDDEN_ROLE_SENTINEL
