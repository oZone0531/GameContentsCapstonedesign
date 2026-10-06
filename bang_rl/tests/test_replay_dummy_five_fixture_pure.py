"""Fixture-driven pure test for the hover role-inference overlay with a real-shaped replay.

This file loads a full-format dummy_five replay fixture from disk
(bang_rl/tests/data/replay_dummy_five_3turns.json) and verifies that the
HOVER role-inference pipeline works end-to-end:

  fixture loads -> ReplayModel -> _attach_internals -> snapshot.internals ->
  inverted_role_inference(source_seat, sheriff_seat) -> non-empty projection
  over the OTHER non-sheriff seats (source seat and sheriff seat excluded).

The fixture mirrors the confirmed real file shape: 5 dummies (no human), string
seat keys, 5x5 beliefs whose rows sum to 1.0, sheriff at seat 4
(revealed_roles[4]==0). This locks down the chain so a regression that broke
any step -- record shape, internals attachment, or role inference math -- is
caught headlessly.

Uses the importlib path-loading pattern (see
test_replay_overlay_hover_render_pure.py) to bypass bang_rl/gui/__init__ which
would pull in PyQt6/pydantic. Stays Qt-free and numpy-free so it runs in the
restricted sandbox.
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


def test_fixture_hover_produces_projection_per_dummy_seat_per_turn() -> None:
    """For each turn, hovering each non-sheriff dummy seat yields a non-empty
    projection over the OTHER non-sheriff seats (source and sheriff excluded).

    The fixture has 5 dummies with sheriff at seat 4. Hovering any of seats
    0..3 should produce targets that exclude the source seat and the sheriff.
    """
    record = _load_fixture()
    model = ReplayModel(record)
    for index in range(min(3, len(model))):
        snapshot = model.snapshot_at(index)
        assert snapshot is not None and snapshot.internals is not None
        sheriff = _sheriff_seat_for(snapshot.observation)
        assert sheriff == 4, f"turn {index}: expected sheriff at seat 4, got {sheriff}"

        # Hover each non-sheriff dummy seat and check projection.
        for source_seat in range(5):
            if source_seat == sheriff:
                continue
            projection = spectator_overlay.inverted_role_inference(
                snapshot.internals, source_seat, sheriff_seat=sheriff
            )
            assert projection, (
                f"turn {index}, source seat {source_seat}: "
                f"hover overlay produced no inference (invisible overlay)"
            )
            targets = sorted(t for t, _ in projection)
            # Source seat and sheriff are excluded from targets.
            assert source_seat not in targets, (
                f"turn {index}: source seat {source_seat} found in its own targets"
            )
            assert sheriff not in targets, (
                f"turn {index}: sheriff seat {sheriff} found in targets"
            )
            # Every other non-sheriff seat should be a target.
            expected_targets = sorted(
                s for s in range(5) if s != source_seat and s != sheriff
            )
            assert targets == expected_targets, (
                f"turn {index}, source {source_seat}: "
                f"targets {targets} != expected {expected_targets}"
            )


def test_fixture_hover_single_seat_detail() -> None:
    """For source seat 0, turn 0, inverted_role_inference returns valid items
    for each target. Role keys are a subset of {deputy, outlaw, traitor}.
    Each probability is strictly between 0 and 1.
    """
    record = _load_fixture()
    model = ReplayModel(record)
    snapshot = model.snapshot_at(0)
    assert snapshot is not None and snapshot.internals is not None
    sheriff = _sheriff_seat_for(snapshot.observation)

    projection = spectator_overlay.inverted_role_inference(
        snapshot.internals, 0, sheriff_seat=sheriff
    )
    assert projection, "hover projection for seat 0 is empty"

    valid_role_keys = {"deputy", "outlaw", "traitor"}
    for target_seat, items in projection:
        assert items, f"target seat {target_seat}: items are empty"
        assert 1 <= len(items) <= 3, (
            f"target seat {target_seat}: expected 1-3 items, got {len(items)}"
        )
        for role_key, prob in items:
            assert role_key in valid_role_keys, (
                f"target {target_seat}: unexpected role key {role_key!r}"
            )
            assert 0 < prob < 1, (
                f"target {target_seat}: probability {prob} for {role_key!r} "
                f"is not in (0, 1)"
            )


def test_fixture_removing_internals_produces_empty_projection() -> None:
    """A ReplayModel built without internals yields empty hover projection.

    This is the revert guard: if internals stop being attached, the hover
    overlay has nothing to draw. Pins down the "no overlay" case.
    """
    record = _load_fixture()
    # Remove internals from the record before constructing the model.
    record.pop("internals", None)
    model = ReplayModel(record)
    snapshot = model.snapshot_at(0)
    assert snapshot is not None
    assert snapshot.internals is None, (
        "snapshot.internals should be None when record has no internals"
    )
    projection = spectator_overlay.inverted_role_inference(
        snapshot.internals, 0, sheriff_seat=4
    )
    assert projection == (), f"expected empty projection, got {projection}"
