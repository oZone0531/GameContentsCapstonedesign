"""Revert-sensitive regression test for the empty-replay (frozen observation) bug.

THE BUG
-------
``GameSessionController._record_replay_snapshot`` captures the live PUBLIC
observation into ``_replay_observations`` so a human game can be re-watched.
The application ``GameService`` freezes every observation via ``_freeze`` into a
tree of ``types.MappingProxyType`` (nested dicts) and tuples -- and the
observation's cumulative ``event_sequence`` is a TUPLE of ``MappingProxyType``
event dicts. The pre-fix code did ``copy.deepcopy(dict(observation))`` inside a
broad ``try/except Exception: return``. ``copy.deepcopy`` cannot pickle a
``mappingproxy`` and raises ``TypeError('cannot pickle mappingproxy object')``;
the except swallowed it and returned WITHOUT appending. Every snapshot carrying
real event progress (a non-empty ``event_sequence`` of frozen event dicts) was
therefore silently dropped, so saved replays showed no game progression. The fix
replaced the deepcopy with a module-level ``_plain_copy`` helper that thaws
``MappingProxyType`` -> ``dict`` and tuple/list -> ``list``.

WHY THE EXISTING *_pure.py TESTS MISSED IT
------------------------------------------
``test_game_restore_log_pure.py`` (and the other pure tests) build their
``FakeService`` observations as PLAIN dicts with tuple values -- no
``MappingProxyType`` anywhere. ``copy.deepcopy`` happily copies a plain dict, so
the pre-fix code never raised and the bug was invisible. This test closes that
gap by FREEZING the FakeService observations exactly like the real service
(``MappingProxyType`` observation whose ``event_sequence`` is a tuple of
``MappingProxyType`` event dicts), which is what makes the pre-fix deepcopy fail.

This test is INTENTIONALLY REVERT-SENSITIVE: it PASSES with ``_plain_copy`` in
place and FAILS if the deepcopy-based capture is restored.

It imports NO numpy / pydantic / PyQt6 and runs headlessly, either via pytest or
as ``python bang_rl/tests/test_replay_capture_frozen_pure.py`` (``__main__``
runner at the bottom). It path-loads ``bang_rl/gui/replay.py`` (registered under
its real dotted name so the controller's lazy import resolves) and
``bang_rl/gui/game.py`` under an isolated alias, mirroring the existing pure
tests, and imports the Qt-free ``bang_rl/gui/replay_store.py`` the same way.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
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
# (``from bang_rl.gui.replay import ReplayModel``) resolves without executing the
# pydantic-backed package __init__. replay_store is Qt/numpy-free; load it under
# its real name too so a direct import inside it (if any) resolves cleanly.
_load("replay.py", "bang_rl.gui.replay")
_replay_store = _load("replay_store.py", "bang_rl.gui.replay_store")
_game = _load("game.py", "_game_capture_frozen_isolated")
GameSessionController = _game.GameSessionController
GameMode = _game.GameMode


def _freeze(value: Any) -> Any:
    """Replicate bang_rl/app/services.py _freeze WITHOUT importing it (numpy).

    Mappings become ``types.MappingProxyType`` (recursively), and lists/tuples
    become tuples of frozen items. Scalars pass through. This yields the exact
    read-only observation shape the real service hands the controller, so a
    pre-fix ``copy.deepcopy`` of it raises ``cannot pickle mappingproxy``.
    """

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


# A growing cumulative public event log (one more event each turn).
_EVENTS = [
    {"environment_step": 0, "event_type": "game_started"},
    {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
    {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 2},
]


def _frozen_state(turn: int) -> Any:
    """Build a PublicGameState-like object whose observation is FROZEN.

    The ``event_sequence`` is a tuple of ``MappingProxyType`` event dicts and the
    whole observation is a ``MappingProxyType`` -- mirroring the real service.
    """

    raw_observation = {
        "healths": [5, max(1, 5 - turn), 5, 5, 5],
        "alive": [True] * 5,
        "role_visibility": [1, 0, 0, 0, 0],
        "revealed_roles": [0, 255, 255, 255, 255],
        "current_player_idx": [0],
        "environment_step": [turn],
        "event_sequence": [dict(event) for event in _EVENTS[: turn + 1]],
        "action_mask": [1, 1, 0, 0, 0, 0],
    }
    observation = _freeze(raw_observation)
    return SimpleNamespace(
        session_id="session-1",
        episode_id="episode-1",
        observation=observation,
        action_mask=(1, 1, 0, 0, 0, 0),
        environment_step=turn,
        current_player=0,
        terminated=False,
        truncated=False,
        termination_status="running",
        winning_team=None,
        winners=(),
    )


@dataclass
class FakeService:
    """Service seam whose per-turn observations are FROZEN like the real one."""

    states: list[Any]

    def __post_init__(self) -> None:
        self.created: list[Any] = []
        self.steps: list[Any] = []
        self.closed: list[Any] = []
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
        # Mirror GameService.get_full_state: a plain-dict authoritative state
        # carrying the cumulative event_log (NOT frozen; the service exporters
        # return _json_value output).
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


def _three_turn_states() -> list[Any]:
    return [_frozen_state(0), _frozen_state(1), _frozen_state(2)]


def _drive(controller: GameSessionController) -> None:
    controller.start(GameMode.HUMAN_RANDOM)  # turn 0 recorded
    controller.submit_human_action(1)  # turn 1 recorded
    controller.submit_human_action(1)  # turn 2 recorded


def _is_mappingproxy_anywhere(value: Any) -> bool:
    """Return True if any MappingProxyType survives anywhere in ``value``."""

    if isinstance(value, MappingProxyType):
        return True
    if isinstance(value, dict):
        return any(_is_mappingproxy_anywhere(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_is_mappingproxy_anywhere(item) for item in value)
    return False


def test_frozen_observations_are_captured_not_dropped() -> None:
    # Test 1 (capture): the frozen, event-carrying observations must be recorded,
    # NOT silently dropped by a deepcopy-of-mappingproxy failure.
    controller = GameSessionController(FakeService(_three_turn_states()))
    _drive(controller)

    # Pre-fix: deepcopy raises on the frozen observation, so every event-carrying
    # turn is dropped and this is empty. Post-fix: all three turns are retained.
    assert len(controller._replay_observations) >= 3, (
        "frozen event-carrying observations were dropped -- deepcopy regression"
    )

    last = controller._replay_observations[-1]
    # Each recorded snapshot must be a PLAIN dict (JSON-safe for the record
    # builder), with no MappingProxyType surviving anywhere in the tree.
    for snapshot in controller._replay_observations:
        assert isinstance(snapshot, dict)
        assert not isinstance(snapshot, MappingProxyType)
        assert not _is_mappingproxy_anywhere(snapshot)

    events = last["event_sequence"]
    assert isinstance(events, list)
    assert len(events) == 3, "the last turn must carry the full cumulative log"
    assert [event["event_type"] for event in events] == ["game_started", "bang", "bang"]
    # Real event dicts (plain) survived with their payload intact.
    assert events[1]["actor_idx"] == 0 and events[1]["target_idx"] == 1


def test_record_build_and_replay_model_reconstruct() -> None:
    # Test 2 (record build + ReplayModel reconstruction end-to-end).
    from bang_rl.gui.replay import ReplayModel

    controller = GameSessionController(FakeService(_three_turn_states()))
    _drive(controller)

    record = _replay_store.build_replay_record(
        mode="human_random",
        mode_label="Human vs Random",
        ai_versions={},
        winning_team=None,
        winners=(),
        human_player=0,
        seed=None,
        observations=controller._replay_observations,
    )
    assert record["observations"], "record carried no observations (empty replay)"
    assert len(record["observations"]) >= 3, "record is not multi-turn"

    model = ReplayModel(record)
    assert len(model) > 1, "ReplayModel reconstructed only one (or zero) snapshot"

    full_events = model.full_events()
    assert full_events, "reconstructed log is empty"
    assert len(full_events) >= 3, "reconstructed log did not grow across turns"
    assert [str(event.get("event_type")) for event in full_events] == [
        "game_started",
        "bang",
        "bang",
    ]

    # A mid/late log row must map to a NON-zero snapshot (its own moment), not
    # collapse back to turn 0 -- proving per-row moment mapping over real data.
    late_index = model.index_for_log_row(len(full_events) - 1)
    assert late_index is not None and late_index > 0, (
        "a late log row collapsed to turn 0 -- degenerate single-snapshot model"
    )


def test_plain_copy_thaws_frozen_observation_mutation_safely() -> None:
    # Test 3 (focused guard on the helper): a MappingProxyType observation with
    # MappingProxyType event entries yields a mutation-safe plain-dict deep copy;
    # mutating the copy does not touch the source and no mappingproxy survives.
    # This documents the exact pre-fix failure mode (deepcopy cannot pickle a
    # mappingproxy) that _plain_copy replaces.
    import copy

    observation = _freeze(
        {
            "environment_step": [2],
            "event_sequence": [dict(event) for event in _EVENTS],
        }
    )

    # The pre-fix capture path would raise here; prove the frozen shape is what
    # breaks copy.deepcopy, so the test's frozen fixtures really reproduce the bug.
    raised = False
    try:
        copy.deepcopy(dict(observation))
    except TypeError:
        raised = True
    assert raised, "frozen observation did not break copy.deepcopy as expected"

    plain = _game._plain_copy(observation)
    assert isinstance(plain, dict)
    assert not _is_mappingproxy_anywhere(plain)
    assert [event["event_type"] for event in plain["event_sequence"]] == [
        "game_started",
        "bang",
        "bang",
    ]

    # Mutation safety: changing the copy must not alter the frozen source.
    plain["event_sequence"].append({"event_type": "injected"})
    assert len(observation["event_sequence"]) == 3


def _run_all() -> int:
    tests = [
        test_frozen_observations_are_captured_not_dropped,
        test_record_build_and_replay_model_reconstruct,
        test_plain_copy_thaws_frozen_observation_mutation_safely,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface unexpected errors
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
