"""Sandbox-executable tests for the Qt-free spectator / replay overlay helpers.

These exercise ``bang_rl/gui/widgets/spectator_overlay.py``, which backs the
emotion bars under each seat and the per-seat role-inference list shown on hover
in spectator mode and replay. The module imports only the standard library, so
(like test_replay_internals_pure.py) this test path-loads it straight from its
file with ``importlib.util.spec_from_file_location`` to bypass
``bang_rl/gui/__init__`` (which pulls in pydantic / PyQt6 / numpy, absent in the
restricted sandbox).

Several assertions are REVERT-SENSITIVE: they fail if the sheriff-drop or the
outlaw-merge in ``role_inference_items`` is removed, or if the bar proportion
math is reverted. Those cases are called out inline.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(relative: str, alias: str):
    path = Path(__file__).resolve().parents[1] / "gui" / relative
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


overlay = _load_module("widgets/spectator_overlay.py", "_spectator_overlay_isolated")
# FEAT-002: replay.py path-loads cleanly (stdlib + dataclasses only) so the
# internals-threading can be asserted headlessly, like test_replay_internals_pure.
replay = _load_module("replay.py", "_replay_overlay_isolated")
ReplayModel = replay.ReplayModel
ReplaySnapshot = replay.ReplaySnapshot


def _dummy_descriptor() -> dict:
    return {
        "kind": "dummy",
        "state": {
            "emotion_module": {
                "emotion_names": ["분노", "공포", "기쁨"],
                "values": [1.0, 0.5, 0.0],
                "config": {
                    "emotions": [
                        {"name": "분노", "description": "상대의 공격에 대한 분노"},
                        {"name": "공포", "description": "위협을 느낄 때의 공포"},
                        {"name": "기쁨", "description": ""},
                    ]
                },
            },
            "role_estimator": {
                "beliefs": [
                    [0.6, 0.1, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.45, 0.42, 0.13],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                    [0.1, 0.2, 0.3, 0.2, 0.2],
                    [0.0, 0.5, 0.0, 0.0, 0.5],
                ],
            },
        },
    }


# -- emotion_bar_cells / emotion_bar_string proportions --------------------
def test_emotion_bar_cells_at_extremes_and_mid() -> None:
    # Revert-sensitive: exact filled/empty counts for the proportion math.
    assert overlay.emotion_bar_cells(0.0, total=10) == (0, 10)
    assert overlay.emotion_bar_cells(1.0, total=10) == (10, 0)
    assert overlay.emotion_bar_cells(0.5, total=10) == (5, 5)
    # Rounding: 0.47 * 10 = 4.7 -> 5 filled.
    assert overlay.emotion_bar_cells(0.47, total=10) == (5, 5)
    # Clamping out-of-range values.
    assert overlay.emotion_bar_cells(-3.0, total=10) == (0, 10)
    assert overlay.emotion_bar_cells(2.0, total=10) == (10, 0)
    # Non-numeric degrades to an empty bar, never raises.
    assert overlay.emotion_bar_cells("oops", total=10) == (0, 10)
    assert overlay.emotion_bar_cells(0.5, total=0) == (0, 0)


def test_emotion_bar_string_renders_fill_and_empty() -> None:
    assert overlay.emotion_bar_string(1.0, total=10) == "=" * 10
    assert overlay.emotion_bar_string(0.0, total=10) == "-" * 10
    # Mid value: 5 filled, 5 empty -> the user's '분노 ========--' style.
    assert overlay.emotion_bar_string(0.5, total=10) == "=====-----"
    assert overlay.emotion_bar_string(0.3, total=10, filled_char="#", empty_char=".") == "###......."


# -- emotion_bars_for_seat --------------------------------------------------
def test_emotion_bars_for_seat_pairs_names_values_descriptions() -> None:
    bars = overlay.emotion_bars_for_seat(_dummy_descriptor(), total=10)
    assert len(bars) == 3
    assert [bar.name for bar in bars] == ["분노", "공포", "기쁨"]
    assert [bar.value for bar in bars] == [1.0, 0.5, 0.0]
    assert bars[0].description == "상대의 공격에 대한 분노"
    assert bars[1].description == "위협을 느낄 때의 공포"
    # Graceful degrade: an empty / absent description surfaces as "".
    assert bars[2].description == ""
    assert (bars[0].filled, bars[0].empty) == (10, 0)
    assert (bars[1].filled, bars[1].empty) == (5, 5)


def test_emotion_bars_for_seat_accepts_inner_state_dict() -> None:
    inner = _dummy_descriptor()["state"]
    bars = overlay.emotion_bars_for_seat(inner)
    assert [bar.name for bar in bars] == ["분노", "공포", "기쁨"]


def test_emotion_bars_for_seat_empty_for_non_dummy_or_random() -> None:
    assert overlay.emotion_bars_for_seat({"kind": "random"}) == ()
    assert overlay.emotion_bars_for_seat({"kind": "human"}) == ()
    assert overlay.emotion_bars_for_seat({"kind": "main"}) == ()
    assert overlay.emotion_bars_for_seat(None) == ()
    assert overlay.emotion_bars_for_seat(42) == ()
    # Dummy but no emotion_module / values -> no bars.
    assert overlay.emotion_bars_for_seat({"kind": "dummy", "state": {}}) == ()


def test_emotion_bars_for_seat_skips_malformed_values() -> None:
    descriptor = {
        "kind": "dummy",
        "state": {
            "emotion_module": {
                "emotion_names": ["a", "b", "c"],
                "values": [0.5, "bad", 0.25],
            }
        },
    }
    bars = overlay.emotion_bars_for_seat(descriptor)
    # The non-numeric middle entry is skipped; the other two survive.
    assert [bar.name for bar in bars] == ["a", "c"]
    # Missing config -> every description degrades to "".
    assert all(bar.description == "" for bar in bars)


# -- role_inference_items: sheriff drop + outlaw merge ---------------------
def test_role_inference_drops_sheriff_and_merges_outlaws() -> None:
    # Canonical order: sheriff, deputy, outlaw_a, outlaw_b, traitor.
    row = [0.6, 0.1, 0.1, 0.1, 0.1]
    items = overlay.role_inference_items(row)
    keys = [key for key, _ in items]
    # Revert-sensitive (sheriff-drop): sheriff must never appear.
    assert "sheriff" not in keys
    # Revert-sensitive (outlaw-merge): exactly one merged 'outlaw' bucket whose
    # probability is the SUM of outlaw_a + outlaw_b (0.1 + 0.1 = 0.2), which
    # would be wrong (0.1, and two separate entries) if the merge were removed.
    outlaw = dict(items)["outlaw"]
    assert abs(outlaw - 0.2) < 1e-9
    assert keys.count("outlaw") == 1
    # Merged outlaw (0.2) now outranks deputy (0.1) and traitor (0.1).
    assert keys[0] == "outlaw"
    assert set(keys) == {"deputy", "outlaw", "traitor"}


def test_role_inference_sorts_descending_and_drops_zero() -> None:
    row = [0.0, 0.5, 0.0, 0.0, 0.5]
    items = overlay.role_inference_items(row)
    # deputy 0.5 and traitor 0.5; outlaw sums to 0.0 and is dropped.
    assert [key for key, _ in items] == ["deputy", "traitor"]
    assert all(prob > 0 for _, prob in items)
    # Descending order holds for a clear ranking.
    row2 = [0.0, 0.05, 0.4, 0.45, 0.1]
    items2 = overlay.role_inference_items(row2)
    assert [key for key, _ in items2] == ["outlaw", "traitor", "deputy"]
    probs = [prob for _, prob in items2]
    assert probs == sorted(probs, reverse=True)
    assert abs(dict(items2)["outlaw"] - 0.85) < 1e-9


def test_role_inference_graceful_degrade() -> None:
    assert overlay.role_inference_items(None) == ()
    assert overlay.role_inference_items([0.5, 0.5]) == ()  # wrong length
    assert overlay.role_inference_items([0.2, 0.2, 0.2, 0.2, "x"]) == ()  # non-numeric
    assert overlay.role_inference_items("nope") == ()


# -- role_belief_row_for_seat ----------------------------------------------
def test_role_belief_row_for_seat_returns_requested_row() -> None:
    descriptor = _dummy_descriptor()
    row = overlay.role_belief_row_for_seat(descriptor, 1)
    assert row == (0.0, 0.0, 0.45, 0.42, 0.13)
    # Feeding it into the inference pipeline merges the two outlaw slots.
    items = overlay.role_inference_items(row)
    assert items[0][0] == "outlaw"
    assert abs(dict(items)["outlaw"] - 0.87) < 1e-9


def test_role_belief_row_for_seat_none_for_non_dummy_or_bad_index() -> None:
    assert overlay.role_belief_row_for_seat({"kind": "random"}, 0) is None
    assert overlay.role_belief_row_for_seat(None, 0) is None
    assert overlay.role_belief_row_for_seat(_dummy_descriptor(), 99) is None
    assert overlay.role_belief_row_for_seat(_dummy_descriptor(), -1) is None
    assert overlay.role_belief_row_for_seat(_dummy_descriptor(), True) is None
    # Dummy but missing role_estimator -> None.
    assert overlay.role_belief_row_for_seat({"kind": "dummy", "state": {}}, 0) is None


# -- aggregate_role_inference_row: OTHER AIs' beliefs about a seat ----------
def _estimator_descriptor(beliefs: list[list[float]]) -> dict:
    """A dummy seat descriptor carrying only a role_estimator belief matrix."""
    return {"kind": "dummy", "state": {"role_estimator": {"beliefs": beliefs}}}


def _one_hot_row(role_index: int) -> list[float]:
    """A one-hot belief row on ``role_index`` (canonical 0..4 order)."""
    row = [0.0, 0.0, 0.0, 0.0, 0.0]
    row[role_index] = 1.0
    return row


def test_aggregate_averages_other_seats_rows_about_target() -> None:
    # Target is seat 0. Two OTHER seats (1 and 2) each carry a belief row ABOUT
    # seat 0 at row index 0 of their matrices. The aggregate is their equal-
    # weight average. (Rows about other seats are filler and must be ignored.)
    filler = [0.2, 0.2, 0.2, 0.2, 0.2]
    seat1 = _estimator_descriptor(
        [[0.0, 0.2, 0.4, 0.2, 0.2], filler, filler, filler, filler]
    )
    seat2 = _estimator_descriptor(
        [[0.0, 0.0, 0.6, 0.2, 0.2], filler, filler, filler, filler]
    )
    internals = {"seats": {"0": _estimator_descriptor([filler] * 5), "1": seat1, "2": seat2}}
    row = overlay.aggregate_role_inference_row(internals, 0)
    assert row is not None
    # Equal-weight average of the two row-0 distributions.
    expected = (0.0, 0.1, 0.5, 0.2, 0.2)
    assert all(abs(a - b) < 1e-9 for a, b in zip(row, expected))
    # Reduced inference merges the two outlaw slots (0.5 + 0.2 = 0.7) and sorts
    # descending: outlaw 0.7, then deputy/traitor tie at 0.1 and 0.2.
    items = overlay.role_inference_items(row)
    assert items[0][0] == "outlaw"
    assert abs(dict(items)["outlaw"] - 0.7) < 1e-9
    assert [key for key, _ in items] == ["outlaw", "traitor", "deputy"]


def test_aggregate_does_not_leak_targets_true_role_one_hot_self_row() -> None:
    # REALISTIC revert-sensitive case: in a real game seat 0's OWN estimator
    # confirms its own role from self_role, so its self-row (row 0 of its own
    # matrix) is one-hot on its TRUE hidden role (here outlaw_a, index 2).
    # The OTHER AIs instead hold an uncertain belief about seat 0. The
    # aggregated-inference path must average the OTHERS' rows and must NOT
    # surface the true role at 100%. If the code reverts to reading the self-
    # row, this fails loudly.
    true_role = 2  # outlaw_a
    self_descriptor = _estimator_descriptor(
        [_one_hot_row(true_role)] + [[0.2, 0.2, 0.2, 0.2, 0.2]] * 4
    )
    other1 = _estimator_descriptor(
        [[0.0, 0.3, 0.3, 0.1, 0.3]] + [[0.2, 0.2, 0.2, 0.2, 0.2]] * 4
    )
    other2 = _estimator_descriptor(
        [[0.0, 0.1, 0.3, 0.3, 0.3]] + [[0.2, 0.2, 0.2, 0.2, 0.2]] * 4
    )
    internals = {"seats": {"0": self_descriptor, "1": other1, "2": other2}}
    row = overlay.aggregate_role_inference_row(internals, 0)
    assert row is not None
    # The one-hot self-row (value 1.0 at index 2) must NOT dominate the result.
    assert row[true_role] < 0.5
    items = overlay.role_inference_items(row)
    # No bucket is pinned at 100% (the leak signature).
    assert all(prob < 0.999 for _, prob in items)
    # Sanity: the averaged distribution is the mean of the two OTHER rows only.
    expected = (0.0, 0.2, 0.3, 0.2, 0.3)
    assert all(abs(a - b) < 1e-9 for a, b in zip(row, expected))


def test_aggregate_single_dummy_or_no_other_estimator_returns_none() -> None:
    # (c) Only the target seat has an estimator -> no OTHER inference available.
    only_self = {"seats": {"0": _estimator_descriptor([[0.2, 0.2, 0.2, 0.2, 0.2]] * 5)}}
    assert overlay.aggregate_role_inference_row(only_self, 0) is None
    # All OTHER seats are random / main / human (no estimator) -> None.
    mixed = {
        "seats": {
            "0": _estimator_descriptor([[0.2, 0.2, 0.2, 0.2, 0.2]] * 5),
            "1": {"kind": "random"},
            "2": {"kind": "main"},
            "3": {"kind": "human"},
        }
    }
    assert overlay.aggregate_role_inference_row(mixed, 0) is None
    # Degenerate / malformed internals never raise.
    assert overlay.aggregate_role_inference_row(None, 0) is None
    assert overlay.aggregate_role_inference_row({"seats": {}}, 0) is None
    assert overlay.aggregate_role_inference_row({"nope": 1}, 0) is None
    assert overlay.aggregate_role_inference_row({"seats": {"1": _estimator_descriptor([[0.2, 0.2, 0.2, 0.2, 0.2]] * 5)}}, True) is None


def test_aggregate_can_include_self_when_not_excluded() -> None:
    # The exclude_self flag defaults to True (the safe behavior). When a caller
    # explicitly opts in, the self-row is included in the average; this guards
    # the flag's wiring so a future caller cannot silently flip it on.
    self_descriptor = _estimator_descriptor(
        [_one_hot_row(2)] + [[0.2, 0.2, 0.2, 0.2, 0.2]] * 4
    )
    other = _estimator_descriptor(
        [[0.0, 0.0, 0.0, 0.0, 1.0]] + [[0.2, 0.2, 0.2, 0.2, 0.2]] * 4
    )
    internals = {"seats": {"0": self_descriptor, "1": other}}
    excluded = overlay.aggregate_role_inference_row(internals, 0)
    included = overlay.aggregate_role_inference_row(internals, 0, exclude_self=False)
    # Excluding self -> only the other row (one-hot traitor).
    assert excluded == (0.0, 0.0, 0.0, 0.0, 1.0)
    # Including self -> average of the one-hot outlaw_a self-row and the
    # one-hot traitor other-row.
    assert included == (0.0, 0.0, 0.5, 0.0, 0.5)


# -- format_percent ---------------------------------------------------------
def test_format_percent_rounds() -> None:
    assert overlay.format_percent(0.87) == 87
    assert overlay.format_percent(0.0) == 0
    assert overlay.format_percent(1.0) == 100
    assert overlay.format_percent(0.034) == 3
    assert overlay.format_percent(0.125) == 12  # round-half-to-even: 12.5 -> 12
    assert overlay.format_percent("bad") == 0


# -- FEAT-002: shared bar-rect geometry (painter + hover hit-test agree) ----
def test_emotion_bar_rects_stack_vertically() -> None:
    rects = overlay.emotion_bar_rects(3, left=10.0, top=100.0, width=60.0, line_height=20.0)
    assert len(rects) == 3
    assert [r.index for r in rects] == [0, 1, 2]
    # Rows stack by line_height from the top.
    assert [r.top for r in rects] == [100.0, 120.0, 140.0]
    assert all(r.left == 10.0 and r.width == 60.0 and r.height == 20.0 for r in rects)
    # Degenerate inputs yield no rects (never raise).
    assert overlay.emotion_bar_rects(0, left=0, top=0, width=10, line_height=10) == ()
    assert overlay.emotion_bar_rects(2, left=0, top=0, width=10, line_height=0) == ()
    assert overlay.emotion_bar_rects(True, left=0, top=0, width=10, line_height=10) == ()


def test_bar_rect_at_point_hits_the_matching_row() -> None:
    rects = overlay.emotion_bar_rects(3, left=10.0, top=100.0, width=60.0, line_height=20.0)
    # A point inside the second row maps to index 1.
    assert overlay.bar_rect_at_point(rects, 40.0, 125.0) == 1
    # Inside the first row.
    assert overlay.bar_rect_at_point(rects, 10.0, 100.0) == 0
    # Outside every row -> None.
    assert overlay.bar_rect_at_point(rects, 200.0, 125.0) is None
    assert overlay.bar_rect_at_point(rects, 40.0, 500.0) is None


# -- FEAT-002: ReplaySnapshot carries internals; ReplayModel threads them ----
def _record_with_internals(internals_key: bool = True) -> dict:
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

    dummy = {"kind": "dummy", "state": {"emotion_module": {"values": [0.1]}}}
    record: dict = {
        "replay_format": "full",
        "observations": [obs(step) for step in range(3)],
        "events": events,
    }
    if internals_key:
        # Index-parallel to observations, with a None placeholder in the middle.
        record["internals"] = [
            {"health": [5, 5, 5, 5, 5], "seats": {"1": dummy}},
            None,
            {"health": [5, 4, 5, 5, 5], "seats": {"1": dummy}},
        ]
    return record


def test_replay_snapshot_defaults_internals_to_none() -> None:
    # A bare snapshot (and every legacy construction) carries internals=None.
    snapshot = ReplaySnapshot(
        observation={}, action_mask=(0,) * 6, environment_step=0, current_player=None
    )
    assert snapshot.internals is None


def test_replay_model_attaches_internals_index_parallel() -> None:
    model = ReplayModel(_record_with_internals(internals_key=True))
    snapshots = model.snapshots
    assert len(snapshots) == 3
    # Turn 0 and turn 2 carry their mapping; the None placeholder leaves None.
    assert snapshots[0].internals is not None
    assert snapshots[0].internals["seats"]["1"]["kind"] == "dummy"
    assert snapshots[1].internals is None
    assert snapshots[2].internals is not None
    assert snapshots[2].internals["health"] == [5, 4, 5, 5, 5]
    # The threaded internals feed the pure overlay helpers end to end.
    bars = overlay.emotion_bars_for_seat(snapshots[0].internals["seats"]["1"])
    assert len(bars) == 1


def test_replay_model_legacy_record_without_internals_is_none() -> None:
    model = ReplayModel(_record_with_internals(internals_key=False))
    assert len(model.snapshots) == 3
    assert all(snapshot.internals is None for snapshot in model.snapshots)


def test_replay_model_observation_stream_has_no_internals() -> None:
    # A plain observation list (no record wrapper) carries no internals.
    record = _record_with_internals(internals_key=True)
    model = ReplayModel(record["observations"])
    assert all(snapshot.internals is None for snapshot in model.snapshots)
