"""Sandbox-executable regression for the FEAT-003 always-on inference overlay.

The reported bug: the replay role-inference overlay was HOVER-ONLY, so a
reviewer who did not hover a seat saw nothing and reported the overlay as
"not visible". FEAT-003 adds an ALWAYS-ON compact per-seat inference plate that
renders whenever overlays are enabled and per-turn internals are present, while
KEEPING the existing hover-to-expand behavior.

The always-on math is two new Qt-free, numpy-free helpers in spectator_overlay:

  * ``always_on_inference_items(internals, target_seat, sheriff_seat=...)`` --
    the aggregated (averaged across all OTHER dummy AIs) consensus about
    ``target_seat``, reduced the usual way (sheriff role dropped, outlaws
    merged, descending) and capped at top-N.
  * ``always_on_all_seats(internals, sheriff_seat=...)`` -- a dict of
    ``{seat: items}`` for every NON-sheriff dummy seat with inference.

Both are pure composition of existing helpers (``aggregate_role_inference_row``
+ ``role_inference_items``); the Qt paint call site lives under
``if QT_AVAILABLE:`` in GraphicalBoard and cannot run in the restricted sandbox
(PyQt6 / numpy / pydantic are not installable). This module locks in the exact
data the paint method consumes, mirroring the importlib path-loading pattern of
test_replay_overlay_hover_render_pure.py so bang_rl/gui/__init__ (which pulls in
Qt) is bypassed.
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


replay_store = _load_module("replay_store.py", "_replay_store_always_on_isolated")
replay = _load_module("replay.py", "_replay_always_on_isolated")
spectator_overlay = _load_module(
    "widgets/spectator_overlay.py", "_spectator_overlay_always_on_isolated"
)
ReplayModel = replay.ReplayModel


# The sheriff sits at a NON-zero seat so the test is revert-sensitive to any
# "seat 0 is always the sheriff" assumption: the sheriff is seat 2 here.
_SHERIFF_SEAT = 2
_HUMAN_SEAT = 0


def _eliminated_beliefs_for_seat(seat: int, sheriff: int = _SHERIFF_SEAT) -> list:
    """A realistic FEAT-001-style 5x5 beliefs matrix (see hover test)."""

    rows: list[list[float]] = []
    for target in range(5):
        if target == seat:
            row = [0.0, 0.0, 1.0, 0.0, 0.0]
        elif target == sheriff:
            row = [1.0, 0.0, 0.0, 0.0, 0.0]
        else:
            row = [0.0, 1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0]
        rows.append(row)
    return rows


def _dummy_seat_descriptor(seat: int) -> dict:
    return {
        "kind": "dummy",
        "state": {
            "schema_version": "2.0",
            "agent_id": f"player_{seat}",
            "emotion_module": {"values": [0.1, 0.2, 0.3]},
            "role_estimator": {
                "beliefs": _eliminated_beliefs_for_seat(seat),
                "known_roles": [255],
            },
        },
    }


def _full_internals_entry() -> dict:
    seats: dict[str, dict] = {str(_HUMAN_SEAT): {"kind": "human"}}
    for seat in range(5):
        if seat == _HUMAN_SEAT:
            continue
        seats[str(seat)] = _dummy_seat_descriptor(seat)
    return {"health": [5, 5, 5, 5, 5], "seats": seats}


def _realistic_observations() -> list[dict]:
    revealed = [255, 255, 255, 255, 255]
    revealed[_SHERIFF_SEAT] = 0  # SHERIFF_ROLE_VALUE

    def obs(step: int) -> dict:
        return {
            "healths": [5, 5, 5, 5, 5],
            "alive": [True] * 5,
            "current_player_idx": [step % 5],
            "environment_step": [step],
            "revealed_roles": list(revealed),
            "event_sequence": [],
        }

    return [obs(step) for step in range(3)]


def _states() -> list[dict]:
    return [{"episode_id": "e", "environment_step": step} for step in range(3)]


def _common() -> dict:
    return dict(
        mode="human_dummy",
        mode_label="Human vs Dummy",
        ai_versions={"1": "dummy", "2": "dummy", "3": "dummy", "4": "dummy"},
        winning_team="sheriff_team",
        winners=[0],
        human_player=_HUMAN_SEAT,
        seed=7,
    )


def _full_record() -> dict:
    return replay_store.build_full_replay_record(
        states=_states(),
        internals=[_full_internals_entry() for _ in range(3)],
        observations=_realistic_observations(),
        **_common(),
    )


def _json_roundtrip(record: dict) -> dict:
    return json.loads(json.dumps(record))


def _internals_at(index: int = 0) -> dict:
    model = ReplayModel(_json_roundtrip(_full_record()))
    snapshot = model.snapshot_at(index)
    assert snapshot is not None and snapshot.internals is not None
    return snapshot.internals


def test_always_on_items_returns_aggregated_reduced_tuple() -> None:
    internals = _internals_at(0)
    # Aggregate the OTHER dummies' beliefs about a non-sheriff dummy seat (3).
    items = spectator_overlay.always_on_inference_items(internals, 3)
    assert isinstance(items, tuple)
    assert items, "always-on produced no inference (invisible overlay)"
    assert len(items) <= 3
    keys = [key for key, _ in items]
    # Sheriff role is dropped; outlaws merged into a single 'outlaw' bucket.
    assert "sheriff" not in keys
    assert "outlaw_a" not in keys and "outlaw_b" not in keys
    assert set(keys) <= {"deputy", "outlaw", "traitor"}
    # Descending by probability.
    probs = [prob for _, prob in items]
    assert probs == sorted(probs, reverse=True)


def test_always_on_items_capped_at_max_items() -> None:
    internals = _internals_at(0)
    items = spectator_overlay.always_on_inference_items(internals, 3, max_items=2)
    assert len(items) <= 2
    # Non-positive max_items yields empty.
    assert spectator_overlay.always_on_inference_items(internals, 3, max_items=0) == ()


def test_always_on_items_empty_when_no_other_dummy_beliefs() -> None:
    # Human seat has no estimator and no OTHER dummy infers about it with a
    # usable row? It does: others DO carry a row for seat 0. But a seat with no
    # internals at all yields empty.
    empty: dict = {"health": [], "seats": {}}
    assert spectator_overlay.always_on_inference_items(empty, 0) == ()


def test_always_on_all_seats_excludes_sheriff_and_covers_dummies() -> None:
    internals = _internals_at(0)
    sheriff = spectator_overlay.sheriff_seat_from_observation(
        _realistic_observations()[0]
    )
    assert sheriff == _SHERIFF_SEAT
    seat_items = spectator_overlay.always_on_all_seats(
        internals, sheriff_seat=sheriff
    )
    assert isinstance(seat_items, dict)
    # The sheriff seat is excluded; the human seat (no estimator) is excluded.
    assert sheriff not in seat_items
    assert _HUMAN_SEAT not in seat_items
    # Every remaining dummy seat is present with non-empty items.
    assert set(seat_items) == {1, 3, 4}
    for items in seat_items.values():
        assert items


def test_always_on_all_seats_without_sheriff_hint_keeps_all_dummies() -> None:
    internals = _internals_at(0)
    # With sheriff_seat=None no seat is excluded for being the sheriff, so
    # seat 2 is ELIGIBLE, but its aggregated row is 100 % sheriff (every other
    # dummy knows seat 2 is the sheriff) and role_inference_items drops the
    # sheriff ROLE column, leaving no items. Seat 2 therefore has nothing to
    # draw and is omitted from the result even when not explicitly excluded.
    seat_items = spectator_overlay.always_on_all_seats(internals, sheriff_seat=None)
    assert set(seat_items) == {1, 3, 4}


def test_always_on_all_seats_at_every_recorded_turn() -> None:
    model = ReplayModel(_json_roundtrip(_full_record()))
    for index in range(len(model)):
        snapshot = model.snapshot_at(index)
        assert snapshot is not None and snapshot.internals is not None
        sheriff = spectator_overlay.sheriff_seat_from_observation(
            snapshot.observation
        )
        seat_items = spectator_overlay.always_on_all_seats(
            snapshot.internals, sheriff_seat=sheriff
        )
        assert seat_items, f"turn {index} produced no always-on inference"
        assert sheriff not in seat_items


def test_public_record_has_no_internals_and_empty_always_on() -> None:
    # REVERT GUARD: a PUBLIC replay never carries internals, so the overlay has
    # nothing to draw -- pins down the "no overlay" case.
    public = replay_store.build_replay_record(
        observations=_realistic_observations(), **_common()
    )
    assert "internals" not in public
    model = ReplayModel(_json_roundtrip(public))
    snapshot = model.snapshot_at(0)
    assert snapshot is not None
    assert snapshot.internals is None
    assert spectator_overlay.always_on_all_seats(snapshot.internals) == {}
