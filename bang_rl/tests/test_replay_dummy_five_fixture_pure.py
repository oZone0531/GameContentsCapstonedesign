"""Fixture-driven pure test for always-on inference with a real-shaped replay.

This file loads a full-format dummy_five replay fixture from disk
(bang_rl/tests/data/replay_dummy_five_3turns.json) and verifies that the
always-on inference pipeline works end-to-end:

  fixture loads -> ReplayModel -> _attach_internals -> snapshot.internals ->
  always_on_all_seats / always_on_inference_items -> non-empty items per
  non-sheriff dummy seat per turn.

The fixture mirrors the confirmed real file shape: 5 dummies (no human), string
seat keys, 5x5 beliefs whose rows sum to 1.0, sheriff at seat 4
(revealed_roles[4]==0). This locks down the chain so a regression that broke
any step -- record shape, internals attachment, aggregation math, or role
reduction -- is caught headlessly.

Uses the importlib path-loading pattern (see
test_replay_overlay_hover_render_pure.py) to bypass bang_rl/gui/__init__ which
would pull in PyQt6/pydantic. Stays Qt-free and numpy-free so it runs in the
restricted sandbox.

NOTE: test_replay_always_on_inference_pure.py (FEAT-003) tests the same helpers
with INLINE synthetic data and is NOT modified here. This file is a SEPARATE
fixture-driven complement.
"""
from __future__ import annotations

import importlib.util
import json
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


replay_store = _load_module("replay_store.py", "_replay_store_fixture_isolated")
replay = _load_module("replay.py", "_replay_fixture_isolated")
spectator_overlay = _load_module(
    "widgets/spectator_overlay.py", "_spectator_overlay_fixture_isolated"
)
ReplayModel = replay.ReplayModel

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "data" / "replay_dummy_five_3turns.json"
)


def _load_fixture() -> dict:
    """Load and return the on-disk fixture record."""
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _sheriff_seat_for(observation: dict) -> int:
    """Derive the sheriff seat from an observation."""
    seat = spectator_overlay.sheriff_seat_from_observation(observation)
    assert seat is not None, "could not derive sheriff seat from observation"
    return seat


# ---- Tests ------------------------------------------------------------------


def test_fixture_loads_as_replay_model() -> None:
    """Load fixture, construct ReplayModel, assert len >= 3."""
    record = _load_fixture()
    model = ReplayModel(record)
    assert len(model) >= 3, f"expected at least 3 snapshots, got {len(model)}"


def test_fixture_internals_attached_per_turn() -> None:
    """For each turn, snapshot.internals is not None with 5 string seat keys."""
    record = _load_fixture()
    model = ReplayModel(record)
    for index in range(min(3, len(model))):
        snapshot = model.snapshot_at(index)
        assert snapshot is not None, f"snapshot at {index} is None"
        assert snapshot.internals is not None, (
            f"turn {index}: snapshot.internals is None (not attached)"
        )
        seats = snapshot.internals.get("seats")
        assert isinstance(seats, dict), f"turn {index}: no 'seats' dict in internals"
        seat_keys = set(seats.keys())
        assert len(seat_keys) == 5, (
            f"turn {index}: expected 5 seat keys, got {seat_keys}"
        )
        # All keys must be string digits 0..4.
        expected = {"0", "1", "2", "3", "4"}
        assert seat_keys == expected, (
            f"turn {index}: seat keys {seat_keys} != {expected}"
        )


def test_fixture_always_on_produces_items_per_dummy_seat_per_turn() -> None:
    """For each turn, always_on_all_seats returns 4 entries (5 - 1 sheriff).

    Every non-sheriff dummy seat has non-empty items. The sheriff seat (4) is
    never in the result.
    """
    record = _load_fixture()
    model = ReplayModel(record)
    for index in range(min(3, len(model))):
        snapshot = model.snapshot_at(index)
        assert snapshot is not None and snapshot.internals is not None
        sheriff = _sheriff_seat_for(snapshot.observation)
        assert sheriff == 4, f"turn {index}: expected sheriff at seat 4, got {sheriff}"

        seat_items = spectator_overlay.always_on_all_seats(
            snapshot.internals, sheriff_seat=sheriff
        )
        assert isinstance(seat_items, dict), f"turn {index}: result is not a dict"
        # 5 dummies minus 1 sheriff = 4 non-sheriff seats expected.
        assert len(seat_items) == 4, (
            f"turn {index}: expected 4 entries, got {len(seat_items)} "
            f"(keys={set(seat_items)})"
        )
        # Sheriff seat must be excluded.
        assert sheriff not in seat_items, (
            f"turn {index}: sheriff seat {sheriff} should not be in result"
        )
        # Every non-sheriff seat has non-empty items.
        for seat, items in seat_items.items():
            assert items, (
                f"turn {index}, seat {seat}: always-on items are empty"
            )


def test_fixture_always_on_single_seat_detail() -> None:
    """For seat 0, turn 0, always_on_inference_items returns valid items.

    Asserts 1-3 items. Role keys are a subset of {deputy, outlaw, traitor}.
    Each probability is strictly between 0 and 1.
    """
    record = _load_fixture()
    model = ReplayModel(record)
    snapshot = model.snapshot_at(0)
    assert snapshot is not None and snapshot.internals is not None
    sheriff = _sheriff_seat_for(snapshot.observation)

    items = spectator_overlay.always_on_inference_items(
        snapshot.internals, 0, sheriff_seat=sheriff
    )
    assert isinstance(items, tuple)
    assert 1 <= len(items) <= 3, f"expected 1-3 items, got {len(items)}"

    valid_role_keys = {"deputy", "outlaw", "traitor"}
    for role_key, prob in items:
        assert role_key in valid_role_keys, (
            f"unexpected role key {role_key!r}, expected one of {valid_role_keys}"
        )
        assert 0 < prob < 1, (
            f"probability {prob} for {role_key!r} is not in (0, 1)"
        )


def test_fixture_removing_internals_produces_empty() -> None:
    """A ReplayModel built without internals yields empty always-on results."""
    record = _load_fixture()
    # Remove internals from the record before constructing the model.
    record.pop("internals", None)
    model = ReplayModel(record)
    snapshot = model.snapshot_at(0)
    assert snapshot is not None
    assert snapshot.internals is None, (
        "snapshot.internals should be None when record has no internals"
    )
    result = spectator_overlay.always_on_all_seats(snapshot.internals)
    assert result == {}, f"expected empty dict, got {result}"
