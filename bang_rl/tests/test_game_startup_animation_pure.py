"""Sandbox-executable controller tests for FEAT-003 (items 1 and 2).

The GUI controller (``bang_rl/gui/game.py``) is Qt-free and its heavy imports
are lazy, but ``import bang_rl.gui.game`` would execute ``bang_rl/gui/__init__``
which pulls in pydantic / PyQt6 (absent in the restricted sandbox). This module
path-loads ``game.py`` directly, and first path-loads ``replay.py`` and registers
it under the real name ``bang_rl.gui.replay`` so the controller's lazy
``from bang_rl.gui.replay import ReplayModel`` resolves to it in the sandbox. A
tiny in-file ``FakeService`` (mirroring the one in the CI-only test_game_gui.py)
drives the controller without any numpy / Qt / pydantic dependency.

Covered here (all pure), matching the item-1 / item-2 acceptance criteria:

* item 1: starting a human game whose first turn is an AI (the Sheriff), with
  ``advance_ai=False``, leaves the controller on the AI turn (``is_ai_turn()``
  True) instead of fast-forwarding, and stepping single AI turns walks
  Sheriff-first one at a time until the human seat.
* item 1 regression: the default ``start(advance_ai=True)`` still fast-forwards
  to the human seat.
* item 2: after ``restore_to_moment`` (with ``advance_ai=False``) on a moment
  whose current player is an AI seat, ``is_ai_turn()`` is True so the screen
  would animate the run-up; ``advance_ai=True`` still fast-forwards.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load(filename: str, alias: str):
    path = Path(__file__).resolve().parents[1] / "gui" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# Register replay under its real dotted name so the controller's lazy import
# resolves without executing the pydantic-backed package __init__.
_load("replay.py", "bang_rl.gui.replay")
_game = _load("game.py", "_game_startup_isolated")
GameSessionController = _game.GameSessionController
GameMode = _game.GameMode


def _state(*, step: int, current_player: int, events: tuple) -> Any:
    observation = {
        "healths": (5, 5, 5, 5, 5),
        "alive": (True,) * 5,
        "role_visibility": (1, 0, 0, 0, 0),
        "revealed_roles": (0, 255, 255, 255, 255),
        "current_player_idx": (current_player,),
        "environment_step": (step,),
        "event_sequence": events,
        "action_mask": (1, 1, 0, 0, 0, 0),
    }
    return SimpleNamespace(
        session_id="session-1",
        episode_id="episode-1",
        observation=observation,
        action_mask=(1, 1, 0, 0, 0, 0),
        environment_step=step,
        current_player=current_player,
        terminated=False,
        truncated=False,
        termination_status="running",
        winning_team=None,
        winners=(),
    )


# Seat 0 is the human. The Sheriff-first run-up here has seats 1..4 acting
# before control reaches the human at seat 0 on the fifth state.
_EVENTS = [
    {"environment_step": 0, "event_type": "game_started"},
    {"environment_step": 1, "event_type": "bang", "actor_idx": 1, "target_idx": 2},
    {"environment_step": 2, "event_type": "bang", "actor_idx": 2, "target_idx": 3},
    {"environment_step": 3, "event_type": "bang", "actor_idx": 3, "target_idx": 4},
    {"environment_step": 4, "event_type": "bang", "actor_idx": 4, "target_idx": 0},
]


def _runup_states() -> list[Any]:
    # Initial snapshot is an AI (seat 1) turn; each step moves to the next seat;
    # the final state hands control to the human (seat 0).
    return [
        _state(step=0, current_player=1, events=tuple(_EVENTS[:1])),
        _state(step=1, current_player=2, events=tuple(_EVENTS[:2])),
        _state(step=2, current_player=3, events=tuple(_EVENTS[:3])),
        _state(step=3, current_player=4, events=tuple(_EVENTS[:4])),
        _state(step=4, current_player=0, events=tuple(_EVENTS[:5])),
    ]


@dataclass
class FakeService:
    """Minimal service seam whose states walk Sheriff-first to the human."""

    states: list[Any]

    def __post_init__(self) -> None:
        self.created: list[Any] = []
        self.steps: list[Any] = []
        self.closed: list[Any] = []
        self.restored: list[Any] = []
        self.handle = SimpleNamespace(session_id="session-1")
        self._cursor = 0

    def create_session(self, request):
        self.created.append(request)
        self._cursor = 0
        return self.handle

    def get_state(self, handle):
        return self.states[self._cursor]

    def step(self, handle, action=None):
        self.steps.append(action)
        if self._cursor < len(self.states) - 1:
            self._cursor += 1
        state = self.states[self._cursor]
        return SimpleNamespace(state=state, reward=0.0, action=action, info={})

    def close_session(self, handle):
        self.closed.append(handle)

    def full_state(self, handle):
        return {"episode_id": "e", "environment_step": self._cursor}

    def export_agent_internals(self, handle):
        return {
            "health": [5, 5, 5, 5, 5],
            "seats": {"1": {"kind": "dummy", "state": {"turn": self._cursor}}},
        }

    def restore_session(self, state_dict, *, internals=None):
        # Rebuild a live session parked on the SAME moment the state came from.
        # full_state() recorded the cursor as ``environment_step``, so restore
        # the cursor to that index; get_state(handle) then returns the restored
        # moment's state (an AI seat, in the item-2 test) and the forked run-up
        # walks forward from there.
        self.restored.append(state_dict)
        try:
            self._cursor = int(state_dict["environment_step"])
        except (KeyError, TypeError, ValueError):
            pass
        return self.handle


def test_start_advance_ai_false_parks_on_the_ai_turn() -> None:
    # Item 1: with advance_ai=False the controller stays on the Sheriff-first AI
    # turn (not fast-forwarded), so the screen can animate the run-up.
    controller = GameSessionController(FakeService(_runup_states()))
    controller.start(GameMode.HUMAN_RANDOM, advance_ai=False)
    assert controller.active is True
    assert controller.is_ai_turn() is True
    assert controller.state.current_player == 1
    # No AI turns were stepped during start (only the initial get_state).
    assert controller.service.steps == []


def test_start_advance_ai_false_then_single_steps_sheriff_first() -> None:
    # Item 1: stepping single AI turns walks one seat at a time to the human.
    controller = GameSessionController(FakeService(_runup_states()))
    controller.start(GameMode.HUMAN_RANDOM, advance_ai=False)
    seen = [controller.state.current_player]
    for _ in range(10):
        if not controller.is_ai_turn():
            break
        controller.step_single_ai_turn()
        seen.append(controller.state.current_player)
    # The run-up walked seats 1 -> 2 -> 3 -> 4 and then reached the human (0),
    # one AI turn per call, and stopped there.
    assert seen == [1, 2, 3, 4, 0]
    assert controller.is_ai_turn() is False
    assert controller.state.current_player == controller.HUMAN_PLAYER


def test_default_start_still_fast_forwards_to_the_human() -> None:
    # Item 1 regression: the default start (advance_ai=True) fast-forwards the
    # whole run-up synchronously so the controller lands on the human seat.
    controller = GameSessionController(FakeService(_runup_states()))
    controller.start(GameMode.HUMAN_RANDOM)
    assert controller.active is True
    assert controller.is_ai_turn() is False
    assert controller.state.current_player == controller.HUMAN_PLAYER


def _restore_states() -> list[Any]:
    # A short human game whose recorded turns include an AI-turn moment. The
    # initial turn is the human (seat 0); a human action hands off to an AI seat
    # (seat 2) so a restore can land on an AI turn.
    return [
        _state(step=0, current_player=0, events=tuple(_EVENTS[:1])),
        _state(step=1, current_player=2, events=tuple(_EVENTS[:2])),
        _state(step=2, current_player=3, events=tuple(_EVENTS[:3])),
    ]


def test_restore_to_ai_moment_advance_ai_false_reports_ai_turn() -> None:
    # Item 2: restoring to a moment whose current player is an AI seat, with
    # advance_ai=False, leaves is_ai_turn() True so the screen would animate.
    controller = GameSessionController(FakeService(_restore_states()))
    controller.start(GameMode.HUMAN_RANDOM, advance_ai=False)
    # Human acts, handing off to the AI run (advance_ai=False so no fast-fwd).
    controller.submit_human_action(1, advance_ai=False)  # records the seat-2 AI moment
    controller.step_single_ai_turn()  # records the seat-3 AI moment
    # Row 1 corresponds to the step-1 AI (seat 2) moment.
    controller.restore_to_moment(1, advance_ai=False)
    assert controller.active is True
    assert controller.previewing is False
    assert controller.is_ai_turn() is True
    assert controller.state.current_player != controller.HUMAN_PLAYER


def test_restore_default_advance_ai_true_fast_forwards() -> None:
    # Item 2 regression: the default restore still fast-forwards to the human.
    states = [
        _state(step=0, current_player=0, events=tuple(_EVENTS[:1])),
        _state(step=1, current_player=2, events=tuple(_EVENTS[:2])),
        _state(step=2, current_player=0, events=tuple(_EVENTS[:3])),
    ]
    controller = GameSessionController(FakeService(states))
    controller.start(GameMode.HUMAN_RANDOM, advance_ai=False)
    controller.submit_human_action(1, advance_ai=False)
    controller.restore_to_moment(1)
    assert controller.active is True
    assert controller.is_ai_turn() is False
    assert controller.state.current_player == controller.HUMAN_PLAYER
