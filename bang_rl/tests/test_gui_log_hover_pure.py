"""Headless regression tests for items 2 & 3 (log-hover log retention + stable
row -> step mapping).

These import ONLY the Qt-free ``replay`` module (no PyQt6, numpy, or pydantic)
so they collect and run in the restricted sandbox as well as in CI. ``import
bang_rl.gui.replay`` would execute ``bang_rl/gui/__init__`` which pulls in the
pydantic-backed package, so we path-load ``replay.py`` straight from its file
(same pattern as ``test_replay_log_row.py``).

They pin the fix for the confirmed bug where hovering a log row called
``GameLog.update_state`` with the HOVERED moment's (truncated) ``event_sequence``,
so the QListWidget was cleared and rebuilt from only the events up to that step:
every row AFTER the hovered one disappeared (item 2) and, because the list
shrank and always ``scrollToBottom()``-ed, the physical row under the cursor no
longer matched the step the ReplayModel mapped it to (item 3).

The ReplayModel row -> moment mapping was ALREADY correct (test_replay_log_row.py)
and is unchanged here. The behavioural fix lives in the GameLog render contract,
captured headlessly by :func:`bang_rl.gui.replay.merge_rendered_log`:

* a SHORTER incoming (hover-preview) sequence keeps the FULL rendered log and
  reports no growth (so the view is not cleared and not auto-scrolled);
* a longer/equal sequence (live play) renders the full sequence and reports
  growth only when rows were actually appended.

``merge_rendered_log`` is the single source of truth that ``GameLog.update_state``
also calls, so these assertions FAIL against the old truncating implementation
and PASS with the fix.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_replay():
    path = Path(__file__).resolve().parents[1] / "gui" / "replay.py"
    name = "_replay_isolated_hover"
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
merge_rendered_log = _replay.merge_rendered_log


def _observations() -> list[dict]:
    """A long, multi-turn observation stream (>= 8 rows across >= 4 snapshots).

    Each turn carries a cumulative event log: the opening turn's is the
    shortest, the last turn's is the complete log the live GameLog renders.
    """

    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 0, "event_type": "turn_started", "actor_idx": 0},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 1, "event_type": "turn_started", "actor_idx": 1},
        {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 2},
        {"environment_step": 2, "event_type": "turn_started", "actor_idx": 2},
        {"environment_step": 3, "event_type": "bang", "actor_idx": 2, "target_idx": 3},
        {
            "environment_step": 3,
            "event_type": "player_eliminated",
            "actor_idx": 2,
            "target_idx": 3,
            "eliminated_idx": 3,
        },
    ]

    def observation(step: int) -> dict:
        return {
            "healths": (5, 5, 5, 5, 5),
            "alive": (True,) * 5,
            "current_player_idx": (step % 5,),
            "environment_step": step,
            "event_sequence": tuple(
                dict(event) for event in events if event["environment_step"] <= step
            ),
        }

    # Four recorded turns at steps 0..3.
    return [observation(step) for step in range(4)]


# -- item 2: the log never shrinks ----------------------------------------
def test_full_events_length_is_constant_regardless_of_cursor() -> None:
    model = ReplayModel(_observations())
    full_len = len(model.full_events())
    assert full_len >= 8  # the complete cumulative log is long
    # The full log is independent of which snapshot the cursor is on: it stays
    # the longest cumulative log whether we sit on the opening turn or seek to
    # the final one. (Pure analogue of item 2: the log never shrinks.)
    model.seek(0)
    assert len(model.full_events()) == full_len
    model.seek(len(model) - 1)
    assert len(model.full_events()) == full_len


def test_merge_keeps_full_log_on_a_shorter_hover_sequence() -> None:
    # The GameLog render contract: once the full log of N rows is rendered,
    # hovering step n (whose snapshot carries only the first k < N events) must
    # NOT shrink the rendered log and must NOT report growth (so no auto-scroll).
    full = tuple(range(8))  # 8 rows already rendered
    hover_truncated = tuple(range(3))  # hover preview carries only 3 events
    rows, grew = merge_rendered_log(full, hover_truncated)
    assert rows == full, "hover must keep every rendered row (item 2)"
    assert grew is False, "hover must not report growth so it never scrolls (item 3)"


def test_merge_grows_and_scrolls_only_on_live_append() -> None:
    rendered = tuple(range(3))
    # Live play appends two new rows: render the full sequence and report growth.
    rows, grew = merge_rendered_log(rendered, tuple(range(5)))
    assert rows == tuple(range(5))
    assert grew is True
    # An equal-length refresh re-renders the same rows but does NOT scroll.
    rows, grew = merge_rendered_log(rendered, tuple(range(3)))
    assert rows == tuple(range(3))
    assert grew is False


def test_old_truncating_behavior_would_shrink_the_log() -> None:
    # Documents the regression the fix prevents at the widget layer: the OLD
    # code rebuilt the list from ONLY the hovered step's events, so a
    # ReplayModel built from just those truncated events has a SHORTER log than
    # the full episode. The fix keeps rendering the full log instead.
    observations = _observations()
    full_model = ReplayModel(observations)
    hovered_step_events = observations[1]["event_sequence"]  # events up to step 1
    truncated_model = ReplayModel([{**observations[1]}])
    assert len(truncated_model.full_events()) == len(hovered_step_events)
    assert len(truncated_model.full_events()) < len(full_model.full_events())


# -- item 3: row -> step mapping is scroll/cursor independent ---------------
def test_index_for_log_row_is_independent_of_cursor_position() -> None:
    model = ReplayModel(_observations())
    full = model.full_events()
    model.seek(0)
    at_start = [model.index_for_log_row(row) for row in range(len(full))]
    # Seeking to the last snapshot (as if the view scrolled) must not change
    # the row -> moment mapping (pure analogue of item 3's scroll independence).
    model.seek(len(model) - 1)
    at_end = [model.index_for_log_row(row) for row in range(len(full))]
    assert at_start == at_end
    # Resolving rows never relies on / mutates toward a particular cursor.
    assert model.index == len(model) - 1


def test_row_mapping_is_monotonic_and_last_row_is_final_snapshot() -> None:
    model = ReplayModel(_observations())
    full = model.full_events()
    indices = [model.index_for_log_row(row) for row in range(len(full))]
    assert all(index is not None for index in indices)
    # Non-decreasing: a later row never maps to an earlier snapshot.
    assert indices == sorted(indices)
    # The opening rows map to the first snapshot; the last row maps to the
    # final snapshot (the complete cumulative board).
    assert indices[0] == 0
    assert indices[-1] == len(model) - 1
