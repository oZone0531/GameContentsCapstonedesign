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
