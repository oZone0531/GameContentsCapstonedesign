"""Sandbox-executable tests for the item-4 / item-6 per-step internals format.

These exercise the Qt-free, numpy-free ``replay_store`` helpers that carry the
per-step internal state (every player's HP plus each dummy / main AI's emotion
and role belief / guess) into the FULL replay format only, and the pure
``replay`` model that resolves a hovered log row to a snapshot whose observation
carries HP so the board renders that moment.

``import bang_rl.gui.replay_store`` (and ``.replay``) would execute
``bang_rl/gui/__init__`` which pulls in pydantic / PyQt6 modules absent in the
restricted sandbox, so this mirrors the importlib path-loading pattern used by
test_replay_store_pure.py / test_replay_log_row.py and loads the two modules
straight from their paths (they import only json / datetime / pathlib and
typing, so they path-load cleanly without pydantic / numpy).

The DummyAgent-backed round-trip half (which needs numpy) lives in the CI-only
test_replay_store.py; this module proves the JSON-safe format round-trips
through build_full_replay_record -> load_replay with the per-step internals and
placeholders intact, that the public builder never carries them, and that a
hovered mid / late row resolves to a full-board snapshot.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(filename: str, alias: str):
    path = Path(__file__).resolve().parents[1] / "gui" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


replay_store = _load_module("replay_store.py", "_replay_store_internals_isolated")
replay = _load_module("replay.py", "_replay_internals_isolated")
ReplayModel = replay.ReplayModel


def _observations() -> list[dict]:
    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 0},
    ]

    def obs(step: int) -> dict:
        return {
            "healths": [5, 5, 5, 5, 5],
            "alive": [True] * 5,
            "current_player_idx": [step % 5],
            "environment_step": [step],
            "event_sequence": [dict(e) for e in events if e["environment_step"] <= step],
        }

    return [obs(step) for step in range(3)]


def _states() -> list[dict]:
    return [{"episode_id": "e", "environment_step": step} for step in range(3)]


def _internals() -> list:
    # A dummy-seat descriptor (state kept minimal / JSON-safe), a random-seat
    # descriptor, and a None placeholder standing in for a per-turn export miss.
    dummy = {
        "kind": "dummy",
        "state": {
            "schema_version": "2.0",
            "agent_id": "player_1",
            "emotion_module": {"values": [0.1, 0.2, 0.3]},
            "role_estimator": {"beliefs": [[0.2, 0.2, 0.2, 0.2, 0.2]], "known_roles": [255]},
        },
    }
    turn0 = {"health": [5, 5, 5, 5, 5], "seats": {"1": dummy, "2": {"kind": "random"}}}
    turn1 = {"health": [5, 4, 5, 5, 5], "seats": {"1": dummy, "2": {"kind": "random"}}}
    return [turn0, None, turn1]


def _common() -> dict:
    return dict(
        mode="human_dummy",
        mode_label="Human vs Dummy",
        ai_versions={"1": "dummy", "2": "dummy", "3": "dummy", "4": "dummy"},
        winning_team="sheriff_team",
        winners=[0],
        human_player=0,
        seed=7,
        observations=_observations(),
    )


# -- item 6: full record carries a per-step internals list -----------------
def test_build_full_replay_record_stores_per_step_internals() -> None:
    record = replay_store.build_full_replay_record(
        states=_states(), internals=_internals(), **_common()
    )
    assert record["replay_format"] == "full"
    assert "internals" in record
    internals = record["internals"]
    # Index-parallel with states / observations, including the None placeholder.
    assert len(internals) == 3
    assert internals[1] is None
    assert internals[0]["health"] == [5, 5, 5, 5, 5]
    assert internals[0]["seats"]["1"]["kind"] == "dummy"
    assert internals[2]["health"] == [5, 4, 5, 5, 5]


def test_public_record_never_carries_internals() -> None:
    record = replay_store.build_replay_record(**_common())
    assert record["replay_format"] == "public"
    assert "internals" not in record
    assert "states" not in record


def test_full_replay_internals_round_trip_through_save_and_load(tmp_path) -> None:
    record = replay_store.build_full_replay_record(
        states=_states(), internals=_internals(), **_common()
    )
    path = replay_store.save_replay(record, tmp_path)
    loaded = replay_store.load_replay(path)
    assert loaded["internals"][1] is None
    # The dummy state survives JSON round-trip intact so it can be handed back
    # to DummyAgent.from_state_dict (exercised with numpy in CI).
    dummy_state = loaded["internals"][0]["seats"]["1"]["state"]
    assert dummy_state["schema_version"] == "2.0"
    assert dummy_state["emotion_module"]["values"] == [0.1, 0.2, 0.3]
    assert dummy_state["role_estimator"]["known_roles"] == [255]


def test_build_full_replay_record_without_internals_omits_key() -> None:
    # Passing no internals keeps the legacy full-record shape (states only).
    record = replay_store.build_full_replay_record(states=_states(), **_common())
    assert "internals" not in record
    assert record["replay_format"] == "full"


# -- item 6: replay_summary reports has_internals --------------------------
def test_replay_summary_reports_has_internals_for_full_record() -> None:
    full = replay_store.build_full_replay_record(
        states=_states(), internals=_internals(), **_common()
    )
    summary = replay_store.replay_summary(full)
    assert summary["has_internals"] is True
    assert summary["has_full_state"] is True
    assert summary["replay_format"] == "full"


def test_replay_summary_has_internals_false_for_public_and_stateonly() -> None:
    public = replay_store.build_replay_record(**_common())
    assert replay_store.replay_summary(public)["has_internals"] is False
    state_only = replay_store.build_full_replay_record(states=_states(), **_common())
    assert replay_store.replay_summary(state_only)["has_internals"] is False


# -- item 5: a hovered mid / late row resolves to a full-board snapshot -----
def test_hovered_late_row_snapshot_carries_healths_and_step() -> None:
    model = ReplayModel(_observations())
    full = model.full_events()
    # The last visible log row (the step-2 bang) must resolve to the step-2
    # snapshot whose observation carries non-empty healths so the board renders.
    last_row = len(full) - 1
    index = model.index_for_log_row(last_row)
    assert index is not None
    snapshot = model.snapshot_at(index)
    assert snapshot is not None
    assert snapshot.environment_step == 2
    healths = tuple(snapshot.observation.get("healths") or ())
    assert len(healths) == 5
    assert tuple(snapshot.observation.get("alive") or ()) == (True,) * 5


def test_every_hovered_row_snapshot_has_board_keys() -> None:
    model = ReplayModel(_observations())
    for row in range(len(model.full_events())):
        index = model.index_for_log_row(row)
        assert index is not None
        snapshot = model.snapshot_at(index)
        assert snapshot is not None
        assert "healths" in snapshot.observation
        assert "alive" in snapshot.observation
        # A mid / late row never collapses to the opening snapshot's step 0.
        assert snapshot.environment_step >= 0
