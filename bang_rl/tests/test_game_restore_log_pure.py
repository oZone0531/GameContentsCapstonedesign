"""Sandbox-executable controller tests for item-3 (log preservation) and item-4.

The GUI controller (``bang_rl/gui/game.py``) is Qt-free and its heavy imports
are lazy, but ``import bang_rl.gui.game`` would execute ``bang_rl/gui/__init__``
which pulls in pydantic / PyQt6 (absent in the restricted sandbox). This module
path-loads ``game.py`` directly, and first path-loads ``replay.py`` and registers
it under the real name ``bang_rl.gui.replay`` so the controller's lazy
``from bang_rl.gui.replay import ReplayModel`` resolves to it in the sandbox. A
tiny in-file ``FakeService`` (mirroring the one in the CI-only test_game_gui.py)
drives the controller without any numpy / Qt / pydantic dependency.

Covered here (all pure):

* item 3: clicking a mid log row to resume PRESERVES the earlier log history
  (the events before the clicked point are NOT deleted).
* item 4: the third per-turn buffer (``_replay_internals``) stays index-parallel
  to the observation / state buffers and its matching entry is passed to a
  restorer that accepts an ``internals`` keyword.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
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
# (``from bang_rl.gui.replay import ReplayModel``) resolves without executing
# the pydantic-backed package __init__.
_load("replay.py", "bang_rl.gui.replay")
_game = _load("game.py", "_game_restore_isolated")
GameSessionController = _game.GameSessionController
GameMode = _game.GameMode


@dataclass
class FakeService:
    """Minimal service seam with per-turn distinct, log-carrying states."""

    states: list[Any]

    def __post_init__(self) -> None:
        self.created: list[Any] = []
        self.steps: list[Any] = []
        self.closed: list[Any] = []
        self.restored: list[Any] = []
        self.restored_internals: list[Any] = []
        self.handle = SimpleNamespace(session_id="session-1")
        self._cursor = 0

    def create_session(self, request):
        self.created.append(request)
        self._cursor = 0
        return self.handle

    def get_state(self, handle):
        # After a restore_session fork the controller calls get_state(handle) on
        # the forked handle; return the forked state (carrying the cumulative
        # event_sequence) so the on-screen render path can be asserted.
        if getattr(handle, "session_id", None) == "forked":
            forked = getattr(self, "_forked_state", None)
            if forked is not None:
                return forked
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
        # Mirror the real GameService.get_full_state: env.state.to_dict() carries
        # the CUMULATIVE event_log up to the current turn (the environment
        # accumulates state.event_log + new events every step). Recording the
        # cumulative prefix here lets a later restore_session rebuild a forked
        # observation whose event_sequence spans game start .. that turn, exactly
        # like EnvironmentState.from_dict + env.load_state does in production.
        return {
            "episode_id": "e",
            "environment_step": self._cursor,
            "event_log": [dict(event) for event in _EVENTS[: self._cursor + 1]],
        }

    def export_agent_internals(self, handle):
        return {
            "health": [5, max(1, 5 - self._cursor), 5, 5, 5],
            "seats": {"1": {"kind": "dummy", "state": {"turn": self._cursor}}},
        }

    def restore_session(self, state_dict, *, internals=None):
        self.restored.append(state_dict)
        self.restored_internals.append(internals)
        # Fork a live session whose PUBLIC observation carries the cumulative
        # event_sequence restored from the state dict's event_log (this is what
        # the real GameService does: load_state(EnvironmentState.from_dict(...))
        # then _state projects env.state.to_public(0).to_observation_dict(),
        # whose event_sequence is the full public log up to the moment). This is
        # the log the on-screen GameScreen._render feeds to GameLog.update_state.
        events = tuple(dict(event) for event in state_dict.get("event_log", ()))
        step = int(state_dict.get("environment_step", 0))
        forked = _logged_state(step, step=step, events=events)
        self._forked_state = forked
        return SimpleNamespace(session_id="forked")

    def _forked_get_state(self):
        return getattr(self, "_forked_state", None)


def _logged_state(turn: int, *, step: int, events: tuple) -> Any:
    observation = {
        "healths": (5, max(1, 5 - turn), 5, 5, 5),
        "alive": (True,) * 5,
        "role_visibility": (1, 0, 0, 0, 0),
        "revealed_roles": (0, 255, 255, 255, 255),
        "current_player_idx": (0,),
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
        current_player=0,
        terminated=False,
        truncated=False,
        termination_status="running",
        winning_team=None,
        winners=(),
    )


_EVENTS = [
    {"environment_step": 0, "event_type": "game_started"},
    {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
    {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 2},
]


def _three_turn_states() -> list[Any]:
    return [
        _logged_state(0, step=0, events=tuple(_EVENTS[:1])),
        _logged_state(1, step=1, events=tuple(_EVENTS[:2])),
        _logged_state(2, step=2, events=tuple(_EVENTS[:3])),
    ]


def _drive(controller: GameSessionController) -> None:
    controller.start(GameMode.HUMAN_RANDOM)  # turn 0 recorded
    controller.submit_human_action(1)  # turn 1 recorded
    controller.submit_human_action(1)  # turn 2 recorded


def _rendered_log_event_types(controller: GameSessionController) -> list[str]:
    """Return the event_type list the ReplayModel renders for the current log."""

    from bang_rl.gui.replay import ReplayModel

    model = ReplayModel(list(controller._replay_observations))
    return [str(event.get("event_type")) for event in model.full_events()]


def test_three_buffers_stay_index_parallel_with_internals() -> None:
    controller = GameSessionController(FakeService(_three_turn_states()))
    _drive(controller)
    assert len(controller._replay_observations) == 3
    assert len(controller._replay_states) == len(controller._replay_observations)
    assert len(controller._replay_internals) == len(controller._replay_observations)
    # Every per-turn internals entry carries HP and the dummy-seat descriptor.
    for entry in controller._replay_internals:
        assert entry is not None
        assert len(entry["health"]) == 5
        assert entry["seats"]["1"]["kind"] == "dummy"


def test_clicking_mid_row_preserves_earlier_log_history() -> None:
    # Item 3: resuming from a mid log row must NOT delete the earlier events.
    controller = GameSessionController(FakeService(_three_turn_states()))
    _drive(controller)
    before = _rendered_log_event_types(controller)
    assert "game_started" in before and before.count("bang") >= 2
    # Click the row for the step-1 bang (row index 1 in the cumulative log).
    controller.restore_to_moment(1)
    assert controller.active is True
    assert controller.previewing is False
    after = _rendered_log_event_types(controller)
    # The earlier events (through the clicked moment) are still present.
    assert "game_started" in after
    assert "bang" in after
    # The prefix up to and including the clicked moment is intact: the log did
    # not shrink below the game start + first bang the user had already seen.
    assert after[0] == "game_started"
    assert len(after) >= 2


def _snapshot_state_log_event_types(snapshot: Any) -> list[str]:
    """Return the event_type list of the RENDERED (on-screen) log after resume.

    GameScreen._render feeds ``snapshot.state`` to GameLog.update_state, which
    reads ``observation['event_sequence']`` -- so this mirrors exactly what the
    user sees on screen after a log-click resume, NOT the controller buffer the
    other item-3 test checks.
    """

    state = snapshot.state
    assert state is not None
    observation = state.observation
    events = observation.get("event_sequence") or ()
    return [str(event.get("event_type")) for event in events]


def test_resume_render_snapshot_preserves_full_prior_log() -> None:
    # Item 3 (review issue 1): the ON-SCREEN log after a resume is rendered from
    # the FORKED service state's observation event_sequence (what
    # GameScreen._render feeds to GameLog.update_state), not from the controller
    # buffer. Because the recorded per-turn full_state carries the CUMULATIVE
    # event_log and restore_session forks a public observation from it, the
    # rendered log after clicking a mid row still contains every event from game
    # start through the clicked moment (the earlier entries are NOT deleted).
    service = FakeService(_three_turn_states())
    controller = GameSessionController(service)
    _drive(controller)
    # advance_ai=False so the returned snapshot IS the forked-moment state (no
    # follow-on AI stepping mutates it) -- exactly what the screen renders first.
    snapshot = controller.restore_to_moment(1, advance_ai=False)
    rendered = _snapshot_state_log_event_types(snapshot)
    # The forked observation's log spans the game start through the clicked
    # (step-1) bang: the earlier history survives the resume on screen.
    assert rendered[0] == "game_started"
    assert "bang" in rendered
    assert len(rendered) >= 2


def test_recorded_full_state_carries_cumulative_event_log() -> None:
    # Item 3 (review issue 1): each recorded authoritative state dict carries the
    # CUMULATIVE event_log up to its turn, so restoring from ANY row rebuilds a
    # forked observation whose event_sequence includes the earliest events. This
    # is the invariant the on-screen-log preservation rests on.
    controller = GameSessionController(FakeService(_three_turn_states()))
    _drive(controller)
    states = controller._replay_states
    assert len(states) == 3
    # The last recorded state's cumulative log holds every event; earlier ones
    # hold a growing prefix, and each includes the game start.
    for turn, state_dict in enumerate(states):
        log = state_dict["event_log"]
        assert log[0]["event_type"] == "game_started"
        assert len(log) == turn + 1


def test_internals_snapshot_is_no_op_for_public_only_line() -> None:
    # Review issue 3: _record_internals_snapshot is self-enforcing -- for a
    # public-only line (_save_full_replay False) it appends nothing, so the
    # internals buffer stays empty in lock-step with _replay_states and AI
    # internals never leak into a public/training line.
    controller = GameSessionController(FakeService(_three_turn_states()))
    controller.start(GameMode.HUMAN_RANDOM)
    # Force the controller onto a public-only line and re-run the capture path.
    controller._save_full_replay = False
    controller._replay_internals = []
    controller._replay_states = []
    controller._record_internals_snapshot()
    assert controller._replay_internals == []
    controller._record_full_state()
    assert controller._replay_states == []
    assert controller._replay_internals == []


def test_restore_passes_matching_internals_to_the_service() -> None:
    # Item 4: the restorer receives the per-turn internals for the clicked row.
    service = FakeService(_three_turn_states())
    controller = GameSessionController(service)
    _drive(controller)
    controller.restore_to_moment(1)
    assert service.restored  # a fork happened
    assert service.restored_internals  # internals were passed
    passed = service.restored_internals[-1]
    assert passed is not None
    assert passed["seats"]["1"]["kind"] == "dummy"
    assert len(passed["health"]) == 5
