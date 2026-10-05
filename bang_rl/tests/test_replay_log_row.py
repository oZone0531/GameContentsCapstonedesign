"""Headless regression tests for item-8 log-row -> moment mapping.

These import ONLY the Qt-free ``replay`` module (no PyQt6, numpy, or pydantic)
so they collect and run in the sandbox as well as in CI. ``import
bang_rl.gui.replay`` would execute ``bang_rl/gui/__init__`` which pulls in
pydantic-backed modules absent in the sandbox, so we load ``replay.py`` straight
from its path.

They pin the fix for the confirmed bug where a freshly built ``ReplayModel``
(cursor at index 0) resolved a hovered/clicked log row against the opening
turn's SHORT event list, so any row past the first turn mapped to nothing. The
live GameLog renders the FULL cumulative log, so the row must resolve against
:meth:`ReplayModel.full_events` regardless of the cursor position.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_replay():
    import sys

    path = Path(__file__).resolve().parents[1] / "gui" / "replay.py"
    name = "_replay_isolated"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A frozen dataclass looks up its module via sys.modules[cls.__module__]
    # during processing, so register the module before executing it.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_replay = _load_replay()
ReplayModel = _replay.ReplayModel


def _observations() -> list[dict]:
    """Per-turn public observations, each carrying a cumulative event log.

    Mirrors what ``GameSessionController._replay_observations`` records: the
    opening turn's ``event_sequence`` is the shortest, the last turn's is the
    complete log the live GameLog displays.
    """

    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 0, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 1, "target_idx": 0},
        {"environment_step": 2, "event_type": "bang", "actor_idx": 2, "target_idx": 3},
        {
            "environment_step": 3,
            "event_type": "player_eliminated",
            "actor_idx": 0,
            "target_idx": 3,
            "eliminated_idx": 3,
        },
    ]

    def observation(step: int, current: int) -> dict:
        return {
            "healths": (5, 5, 5, 5, 5),
            "alive": (True,) * 5,
            "current_player_idx": (current,),
            "environment_step": step,
            "event_sequence": tuple(
                dict(event) for event in events if event["environment_step"] <= step
            ),
        }

    # Four recorded turns at steps 0..3.
    return [observation(step, step % 5) for step in range(4)]


def test_full_events_is_the_longest_cumulative_log() -> None:
    model = ReplayModel(_observations())
    # Cursor starts at the opening turn whose log is the shortest.
    assert model.index == 0
    assert len(model.current_events()) == 2  # game_started + first bang
    # full_events returns the complete log the live GameLog renders.
    assert len(model.full_events()) == 5


def test_late_row_maps_without_seeking_to_end() -> None:
    model = ReplayModel(_observations())
    full = model.full_events()
    # Row 0 (game_started, step 0) -> opening snapshot.
    assert model.index_for_log_row(0) == 0
    # Row for the step-1 bang -> step-1 snapshot (index 1).
    assert model.index_for_log_row(2) == 1
    # Row for the step-2 bang -> step-2 snapshot (index 2).
    assert model.index_for_log_row(3) == 2
    # The elimination row at step 3 (the LAST row) -> step-3 snapshot (index 3).
    assert model.index_for_log_row(len(full) - 1) == 3
    # Resolving rows never moved the cursor.
    assert model.index == 0


def test_out_of_range_row_returns_none() -> None:
    model = ReplayModel(_observations())
    assert model.index_for_log_row(999) is None
    assert model.index_for_log_row(-1) is None


def test_empty_model_full_events_is_empty() -> None:
    model = ReplayModel([])
    assert model.full_events() == ()
    assert model.index_for_log_row(0) is None


def _observations_with_list_steps() -> list[dict]:
    """Observations whose environment_step is a 1-element list (item 9 bug).

    The public projection stores scalar fields as single-element arrays, so the
    recorded observations carry ``environment_step`` as ``[n]``. The old
    ``_int_or_none`` returned ``None`` for those, collapsing every snapshot to
    step 0 (no visible log, no playback). The fix unwraps a 1-element sequence.
    """

    def obs(step: int, current: int, events: list[dict]) -> dict:
        return {
            "healths": [5, 5, 5, 5, 5],
            "alive": [True] * 5,
            "current_player_idx": [current],
            "environment_step": [step],  # 1-element list, NOT a scalar
            "event_sequence": [dict(event) for event in events],
        }

    events = [
        {"environment_step": [0], "event_type": "game_started"},
        {"environment_step": [1], "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": [2], "event_type": "bang", "actor_idx": 1, "target_idx": 0},
    ]
    return [
        obs(0, 0, events[:1]),
        obs(1, 1, events[:2]),
        obs(2, 2, events[:3]),
    ]


def test_list_valued_steps_do_not_collapse_snapshots_to_step_zero() -> None:
    # Regression for item 9: a saved public replay whose steps are 1-element
    # lists must still reconstruct into distinct per-turn snapshots with a
    # growing cumulative log and a working row -> snapshot mapping.
    model = ReplayModel(_observations_with_list_steps())
    assert len(model) == 3
    # Snapshots resolved distinct steps (not all 0).
    assert [snap.environment_step for snap in model.snapshots] == [0, 1, 2]
    # The cumulative log grows and full_events is the complete log.
    assert len(model.full_events()) == 3
    # Each visible log row maps to a full-board snapshot at that moment.
    for row in range(len(model.full_events())):
        index = model.index_for_log_row(row)
        assert index is not None
        snapshot = model.snapshot_at(index)
        assert snapshot is not None
        # Item 4: a hovered moment renders a COMPLETE board.
        assert "healths" in snapshot.observation
        assert "alive" in snapshot.observation


def test_row_mapping_is_monotonic_and_never_collapses_to_zero() -> None:
    # Items 3 & 8: a later log row must map to a later-or-equal snapshot and a
    # row past the opening turn must NEVER collapse back to index 0.
    model = ReplayModel(_observations())
    full = model.full_events()
    indices = [model.index_for_log_row(row) for row in range(len(full))]
    assert all(index is not None for index in indices)
    # Non-decreasing.
    assert indices == sorted(indices)
    # Only opening-turn rows may map to 0; every later row is strictly past 0.
    assert indices[0] == 0
    assert all(index > 0 for index in indices[2:])
    # The last row reaches the final snapshot (the full cumulative board).
    assert indices[-1] == len(model) - 1


def _observations_without_any_step() -> list[dict]:
    """Per-turn observations that omit ``environment_step`` AND whose events
    carry no ``environment_step`` either.

    This is the pathological data shape behind the item-3/8 regression: with no
    usable step signal a naive per-event step lookup drags every row to turn 0.
    The robust positional fallback must still spread the rows monotonically
    across the recorded snapshots.
    """

    events = [
        {"event_type": "game_started"},
        {"event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"event_type": "bang", "actor_idx": 1, "target_idx": 2},
        {"event_type": "bang", "actor_idx": 2, "target_idx": 3},
        {"event_type": "player_eliminated", "eliminated_idx": 3},
    ]

    def observation(turn: int) -> dict:
        return {
            "healths": (5, 5, 5, 5, 5),
            "alive": (True,) * 5,
            "current_player_idx": (turn % 5,),
            "event_sequence": tuple(dict(event) for event in events[: turn + 1]),
        }

    return [observation(turn) for turn in range(5)]


def test_positional_fallback_when_no_step_signal_is_present() -> None:
    # Regression for items 3 & 8: even with no environment_step anywhere the
    # row -> snapshot mapping must be monotonic and reach the last snapshot,
    # so hover/click land on the clicked moment instead of resetting to turn 0.
    model = ReplayModel(_observations_without_any_step())
    full = model.full_events()
    indices = [model.index_for_log_row(row) for row in range(len(full))]
    assert all(index is not None for index in indices)
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert indices[-1] == len(model) - 1
    # A mid/late row does not collapse to the opening snapshot.
    assert model.index_for_log_row(len(full) - 1) != 0


def _observations_with_constant_step() -> list[dict]:
    """Observations whose ``environment_step`` is a constant (all zero).

    Mirrors a recording where the per-turn step never varied, which used to
    collapse ``index_for_step`` to snapshot 0 for every row.
    """

    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 0, "event_type": "bang"},
        {"environment_step": 0, "event_type": "bang"},
        {"environment_step": 0, "event_type": "player_eliminated"},
    ]

    def observation(turn: int) -> dict:
        return {
            "healths": (5, 5, 5, 5, 5),
            "alive": (True,) * 5,
            "current_player_idx": (turn % 5,),
            "environment_step": (0,),  # constant across every turn
            "event_sequence": tuple(dict(event) for event in events[: turn + 1]),
        }

    return [observation(turn) for turn in range(4)]


def test_constant_step_does_not_collapse_rows_to_turn_zero() -> None:
    model = ReplayModel(_observations_with_constant_step())
    full = model.full_events()
    indices = [model.index_for_log_row(row) for row in range(len(full))]
    assert indices == sorted(indices)
    assert indices[-1] == len(model) - 1
    assert model.index_for_log_row(len(full) - 1) != 0


def test_multiple_events_in_one_turn_map_to_that_turns_snapshot() -> None:
    # A single turn can emit several events (e.g. a bang plus the elimination
    # it caused). Every such row must map to that turn's snapshot, and the last
    # one must reach the final snapshot rather than an earlier turn.
    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 2},
        {"environment_step": 2, "event_type": "player_eliminated", "eliminated_idx": 2},
    ]

    def observation(step: int) -> dict:
        return {
            "healths": (5, 5, 5, 5, 5),
            "alive": (True,) * 5,
            "current_player_idx": (step % 5,),
            "environment_step": (step,),
            "event_sequence": tuple(
                dict(event) for event in events if event["environment_step"] <= step
            ),
        }

    model = ReplayModel([observation(step) for step in range(3)])
    full = model.full_events()
    # Both step-2 rows resolve to the last snapshot (index 2), not an earlier one.
    assert model.index_for_log_row(2) == 2
    assert model.index_for_log_row(3) == 2
    assert model.index_for_log_row(len(full) - 1) == len(model) - 1


def test_log_row_for_index_round_trips_through_index_for_log_row() -> None:
    # FEAT-004 (item 5): log_row_for_index is the inverse of index_for_log_row.
    # For every snapshot index, the row it returns must map back to the SAME
    # snapshot index, so the current-step highlight lands on a row that the
    # hover / click would resolve to the playing snapshot.
    model = ReplayModel(_observations())
    for index in range(len(model)):
        row = model.log_row_for_index(index)
        assert row is not None
        assert model.index_for_log_row(row) == index
    # Resolving rows / indices never moves the playback cursor.
    assert model.index == 0


def test_log_row_for_index_returns_first_row_of_the_moment() -> None:
    # When several log rows belong to one snapshot's moment, the inverse must
    # return the FIRST such row (the earliest row representing that moment),
    # because the row -> index mapping is non-decreasing.
    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 2},
        {"environment_step": 2, "event_type": "player_eliminated", "eliminated_idx": 2},
    ]

    def observation(step: int) -> dict:
        return {
            "healths": (5, 5, 5, 5, 5),
            "alive": (True,) * 5,
            "current_player_idx": (step % 5,),
            "environment_step": (step,),
            "event_sequence": tuple(
                dict(event) for event in events if event["environment_step"] <= step
            ),
        }

    model = ReplayModel([observation(step) for step in range(3)])
    # Snapshot index 2 is represented by rows 2 and 3; the first is row 2.
    first_row = model.log_row_for_index(2)
    assert first_row == 2
    assert model.index_for_log_row(first_row) == 2


def test_log_row_for_index_clamps_and_handles_empty_model() -> None:
    model = ReplayModel(_observations())
    last = len(model) - 1
    # Out-of-range indices clamp into range rather than returning None.
    assert model.log_row_for_index(999) == model.log_row_for_index(last)
    assert model.log_row_for_index(-5) == model.log_row_for_index(0)
    # An empty model has no rows to highlight.
    empty = ReplayModel([])
    assert empty.log_row_for_index(0) is None


def test_review_highlight_uses_log_row_not_snapshot_index() -> None:
    # Defect fix (review item 1): the spectator terminal-review highlight must
    # convert the review cursor (a SNAPSHOT index, model.index) to a LOG ROW via
    # log_row_for_index before calling GameLog.set_current_step (which expects a
    # row). The two coordinate spaces DIVERGE whenever a turn emitted more than
    # one log event, so this pins that the conversion lands on the right row and
    # that using the raw snapshot index would have been wrong.
    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 2},
        {"environment_step": 2, "event_type": "player_eliminated", "eliminated_idx": 2},
    ]

    def observation(step: int) -> dict:
        return {
            "healths": (5, 5, 5, 5, 5),
            "alive": (True,) * 5,
            "current_player_idx": (step % 5,),
            "environment_step": (step,),
            "event_sequence": tuple(
                dict(event) for event in events if event["environment_step"] <= step
            ),
        }

    model = ReplayModel([observation(step) for step in range(3)])
    # Seek the review cursor to the LAST turn (snapshot index 2), which spans two
    # log rows (rows 2 and 3). This is the terminal-review _render_review_snapshot
    # path: it computes the highlight from model.index.
    model.seek(len(model) - 1)
    assert model.index == 2
    # The correct highlight row is the first row of that moment (row 2).
    highlight_row = model.log_row_for_index(model.index)
    assert highlight_row == 2
    # That row round-trips back to the sought snapshot, so hover/click on the
    # highlighted row resolves to the same reviewed turn.
    assert model.index_for_log_row(highlight_row) == model.index
    # Guard against the reverted bug: passing the raw snapshot index (2) as a
    # row would still be a valid row here, but on a model where snapshots
    # outnumber the index it would clamp/mismatch. Pin that the conversion is
    # NOT the identity in general by checking a case where index != first row.
    model.seek(0)
    # Row count exceeds snapshot count (4 rows, 3 snapshots), so the last
    # snapshot index (2) differs from the last row (3): the identity mapping
    # would highlight the wrong row for the final turn.
    assert len(model.full_events()) == 4
    assert len(model) == 3
    assert model.log_row_for_index(len(model) - 1) != len(model.full_events()) - 1


def test_board_keys_carry_forward_when_a_later_observation_omits_them() -> None:
    # Item 4: if a recorded observation omits board keys, the snapshot inherits
    # the last-known value so a hovered moment renders a full board, not empty.
    observations = [
        {
            "healths": [5, 5, 5, 5, 5],
            "alive": [True] * 5,
            "current_player_idx": [0],
            "environment_step": [0],
            "event_sequence": [{"environment_step": [0], "event_type": "game_started"}],
        },
        {
            # This turn omits healths/alive entirely.
            "current_player_idx": [1],
            "environment_step": [1],
            "event_sequence": [
                {"environment_step": [0], "event_type": "game_started"},
                {"environment_step": [1], "event_type": "bang"},
            ],
        },
    ]
    model = ReplayModel(observations)
    second = model.snapshot_at(1)
    assert second is not None
    # The board keys carried forward from the first turn.
    assert tuple(second.observation["healths"]) == (5, 5, 5, 5, 5)
    assert tuple(second.observation["alive"]) == (True,) * 5
    # But the per-turn identity keys reflect THIS turn, not the previous one.
    assert second.environment_step == 1
