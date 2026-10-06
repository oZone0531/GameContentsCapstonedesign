"""Headless regression tests for item-2 full-log-from-start replay behaviour.

On replay playback the LEFT game log must show the ENTIRE episode log from the
very first frame (every turn visible at once) while the game BOARD plays from
turn 1 and advances via auto-play / speed slider / prev / next. Before the fix
``ReplayScreen._render_snapshot`` fed the SAME stepped snapshot to both the
board and the log, so the log revealed incrementally. The fix adds a pure,
Qt-free ``ReplayModel.full_log_snapshot()`` whose ``observation`` carries the
COMPLETE cumulative log independent of the cursor, which the screen renders into
the log while the board still renders the stepped snapshot.

These import ONLY the Qt-free ``replay`` module (no PyQt6, numpy, or pydantic),
loading ``replay.py`` straight from its path, so they collect and run in the
sandbox as well as in CI. The ``full_log_snapshot`` assertions FAIL against the
pre-change code (which has no such method) and PASS after.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_replay():
    import sys

    path = Path(__file__).resolve().parents[1] / "gui" / "replay.py"
    name = "_replay_isolated_full_log"
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

    The opening turn's ``event_sequence`` is the shortest, the last turn's is
    the complete log the live GameLog must display from the first frame.
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


def test_full_log_snapshot_is_the_complete_log_at_index_zero() -> None:
    model = ReplayModel(_observations())
    # The board/current snapshot at the opening turn carries the SHORT log.
    assert model.index == 0
    assert len(model.current_events()) < len(model.full_events())
    # The full-log snapshot carries the COMPLETE log from the first frame.
    snapshot = model.full_log_snapshot()
    sequence = snapshot.observation["event_sequence"]
    assert tuple(sequence) == tuple(model.full_events())
    assert len(sequence) == 5


def test_full_log_snapshot_is_independent_of_cursor_index() -> None:
    model = ReplayModel(_observations())
    full_length = len(model.full_events())
    # Advancing the cursor steps the board but must NOT change the full log.
    model.next()
    assert model.index == 1
    assert len(model.full_log_snapshot().observation["event_sequence"]) == full_length
    model.seek(len(model) - 1)
    assert len(model.full_log_snapshot().observation["event_sequence"]) == full_length
    assert tuple(model.full_log_snapshot().observation["event_sequence"]) == tuple(
        model.full_events()
    )


def test_full_log_snapshot_duck_types_what_gamelog_reads() -> None:
    # GameLog.update_state reads getattr(state, "observation", None) then
    # observation.get("event_sequence"); the snapshot must satisfy both.
    model = ReplayModel(_observations())
    snapshot = model.full_log_snapshot()
    observation = getattr(snapshot, "observation", None)
    assert observation is not None
    assert observation.get("event_sequence") == model.full_events()


def test_empty_model_full_log_snapshot_is_empty_sequence() -> None:
    model = ReplayModel([])
    snapshot = model.full_log_snapshot()
    assert snapshot.observation["event_sequence"] == ()


if __name__ == "__main__":
    test_full_log_snapshot_is_the_complete_log_at_index_zero()
    test_full_log_snapshot_is_independent_of_cursor_index()
    test_full_log_snapshot_duck_types_what_gamelog_reads()
    test_empty_model_full_log_snapshot_is_empty_sequence()
    print("all full-log-from-start tests passed")
