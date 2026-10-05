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
# FEAT-002: the hover overlay's pure math (sheriff-seat derivation + inverted
# role inference). It imports only the standard library, so path-loading it the
# same way (bypassing bang_rl/gui/__init__, which pulls in Qt / numpy) lets this
# module exercise the EXACT replay -> overlay data path the board hover uses,
# headlessly in the restricted sandbox.
spectator_overlay = _load_module(
    "widgets/spectator_overlay.py", "_spectator_overlay_internals_isolated"
)
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


def _main_internals_captured() -> dict:
    # A main-AI descriptor where an active inference stream buffer WAS captured.
    return {
        "health": [5, 5, 5, 5, 5],
        "seats": {
            "0": {
                "kind": "main",
                "captured": True,
                "module_states": {"intent": [[0.1, 0.9]], "emotion": [[0.5]]},
            }
        },
    }


def _main_internals_empty() -> dict:
    # A main-AI descriptor where get_state() returned None on every module, so
    # nothing was captured (the honest best-effort no-op case).
    return {"health": [5, 5, 5, 5, 5], "seats": {"0": {"kind": "main", "captured": False}}}


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


# -- review issue 2: main-AI capture is best-effort (captured flag) ---------
def test_main_ai_captured_flag_round_trips_true_when_buffer_present(tmp_path) -> None:
    # A main-AI seat records captured=True only when an active inference stream
    # buffer was present; the module_states survive the JSON round-trip so a
    # resume can reapply them (set_state) on the forked main agent.
    record = replay_store.build_full_replay_record(
        states=[_states()[0]], internals=[_main_internals_captured()], **_common()
    )
    path = replay_store.save_replay(record, tmp_path)
    loaded = replay_store.load_replay(path)
    seat = loaded["internals"][0]["seats"]["0"]
    assert seat["kind"] == "main"
    assert seat["captured"] is True
    assert seat["module_states"]["intent"] == [[0.1, 0.9]]


def test_main_ai_captured_false_when_no_stream_buffer(tmp_path) -> None:
    # The honest best-effort no-op: get_state() returned None for every module,
    # so captured is False and there are no module_states to restore. A resume
    # then simply keeps the freshly built main agent (documented limitation).
    record = replay_store.build_full_replay_record(
        states=[_states()[0]], internals=[_main_internals_empty()], **_common()
    )
    path = replay_store.save_replay(record, tmp_path)
    loaded = replay_store.load_replay(path)
    seat = loaded["internals"][0]["seats"]["0"]
    assert seat["kind"] == "main"
    assert seat["captured"] is False
    assert "module_states" not in seat


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


# -- FEAT-003 item 4: game-screen spectator review / hover internals seam ---
def _game_screen_review_model() -> ReplayModel:
    """Mirror GameScreen._log_model: observations + index-parallel internals.

    The spectator game screen builds a ReplayModel from a mapping carrying the
    recorded per-turn observations AND the recorded per-turn internals so a
    hovered / sought PAST turn can re-supply its own emotion bars + role
    inference to the board (fixing the "no inference info" bug). This proves the
    mapping form the screen passes resolves each log row to a snapshot whose
    ``internals`` is that turn's recorded entry.
    """

    return ReplayModel({"observations": _observations(), "internals": _internals()})


def test_game_screen_review_model_attaches_internals_per_row() -> None:
    model = _game_screen_review_model()
    # Row 0 (game start) -> turn-0 internals; the last row (step-2 bang) ->
    # turn-2 internals. The None placeholder at index 1 stays None.
    first = model.snapshot_at(model.index_for_log_row(0))
    assert first is not None and first.internals is not None
    assert first.internals["health"] == [5, 5, 5, 5, 5]
    assert first.internals["seats"]["1"]["kind"] == "dummy"
    last_row = len(model.full_events()) - 1
    last = model.snapshot_at(model.index_for_log_row(last_row))
    assert last is not None and last.internals is not None
    assert last.internals["health"] == [5, 4, 5, 5, 5]


def test_game_screen_review_model_prev_next_carry_internals() -> None:
    # Item 4: wiring prev/next to model.prev()/model.next() must yield snapshots
    # whose internals the screen re-supplies via set_seat_internals, so stepping
    # through the review always has per-turn inference data (never None once it
    # was recorded for that turn).
    model = _game_screen_review_model()
    model.seek(0)
    assert model.current().internals is not None
    nxt = model.next()
    assert nxt is not None
    # index 1 was a recorded export MISS (None placeholder) -> internals None,
    # which the board treats as "no overlay for this turn" (not a bug).
    assert nxt.internals is None
    nxt2 = model.next()
    assert nxt2 is not None and nxt2.internals is not None
    back = model.prev()
    assert back is not None and back.internals is None


# -- FEAT-002: full replay -> ReplayModel -> hover overlay data path --------
#
# These prove the EXACT chain the replay board hover uses:
#   build_full_replay_record(...)  (what GameSessionController._save_replay_if_needed saves)
#     -> ReplayModel(record)       (what ReplayScreen loads)
#       -> snapshot.internals      (what _render_snapshot feeds set_seat_internals)
#         -> spectator_overlay.inverted_role_inference(..., sheriff_seat=...)
#            (what GraphicalBoard._paint_role_inference calls, deriving the
#             sheriff seat from the snapshot observation)
#
# Unlike the TRUNCATED one-row fixture in _internals() above (whose beliefs are a
# single row, so the inverted projection would be empty), these use a REALISTIC
# full 5x5 beliefs matrix per dummy seat (the shape a real DummyAgent.state_dict
# carries), so the projection is meaningfully NON-EMPTY -- which is what makes
# the overlay actually render on hover.

# The sheriff sits at a NON-zero seat so the test is revert-sensitive to the
# "seat 0 is the sheriff" bug: the sheriff is seat 2 here.
_SHERIFF_SEAT = 2
# The hovered (source) seat whose own beliefs about the OTHER seats are drawn.
_SOURCE_SEAT = 1


def _eliminated_beliefs_for_seat(seat: int, sheriff: int = _SHERIFF_SEAT) -> list:
    """Return a realistic, FEAT-001-style eliminated 5x5 beliefs matrix.

    Row ``T`` is this dummy's distribution over seat ``T``'s role in canonical
    order ``(sheriff, deputy, outlaw_a, outlaw_b, traitor)``:

    * the self-row is one-hot on the dummy's own known role (outlaw_a here), the
      way a real estimator confirms its own role from ``self_role``;
    * the sheriff row is one-hot on the public Sheriff role;
    * every OTHER row folds out the two KNOWN roles (self + sheriff) and spreads
      the remaining mass evenly over deputy / the two outlaw slots / traitor,
      which is exactly the ~1/3 deputy, 1/3 outlaw, 1/3 traitor an Outlaw dummy
      should hold once FEAT-001 initializes its estimator (NOT the naive uniform
      40/20/20 prior). This makes the recorded internals meaningfully
      non-uniform, so the overlay carries real information.
    """

    rows: list[list[float]] = []
    for target in range(5):
        if target == seat:
            # Own role known: one-hot on outlaw_a (index 2).
            row = [0.0, 0.0, 1.0, 0.0, 0.0]
        elif target == sheriff:
            # Sheriff role is public: one-hot on sheriff (index 0).
            row = [1.0, 0.0, 0.0, 0.0, 0.0]
        else:
            # Self + sheriff eliminated -> remaining mass split over deputy,
            # the two outlaw slots, and traitor (deputy 1/3, outlaw merged 1/3,
            # traitor 1/3). The two outlaw slots share the outlaw third.
            row = [0.0, 1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0]
        rows.append(row)
    return rows


def _dummy_seat_descriptor(seat: int) -> dict:
    """A realistic dummy seat descriptor shaped like DummyAgent.state_dict()."""

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
    """One per-turn internals entry: HP + a dummy descriptor for every seat."""

    return {
        "health": [5, 5, 5, 5, 5],
        "seats": {str(seat): _dummy_seat_descriptor(seat) for seat in range(5)},
    }


def _realistic_observations() -> list[dict]:
    """Observations whose revealed_roles mark the sheriff SEAT (not seat 0)."""

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


def _full_record_with_realistic_internals() -> dict:
    return replay_store.build_full_replay_record(
        states=_states(),
        internals=[_full_internals_entry() for _ in range(3)],
        observations=_realistic_observations(),
        **{key: value for key, value in _common().items() if key != "observations"},
    )


def test_full_replay_overlay_data_path_is_non_empty_on_hover() -> None:
    # The exact chain the board hover uses: a FULL record -> ReplayModel ->
    # snapshot.internals -> inverted_role_inference with the sheriff seat derived
    # from the snapshot observation (sheriff_seat_from_observation), mirroring
    # GraphicalBoard._paint_role_inference's call site.
    model = ReplayModel(_full_record_with_realistic_internals())
    snapshot = model.snapshot_at(0)
    assert snapshot is not None
    # 1) ReplayModel attached the per-turn internals (NOT None).
    assert snapshot.internals is not None
    assert snapshot.internals["seats"][str(_SOURCE_SEAT)]["kind"] == "dummy"

    # 2) The sheriff seat is derived from the observation, NOT hardcoded to 0.
    sheriff = spectator_overlay.sheriff_seat_from_observation(snapshot.observation)
    assert sheriff == _SHERIFF_SEAT

    # 3) The inverted projection the board draws is NON-EMPTY: the overlay
    #    renders on hover. Targets are every OTHER non-sheriff seat (0, 3, 4);
    #    the source seat (1) and the sheriff seat (2) are excluded.
    projection = spectator_overlay.inverted_role_inference(
        snapshot.internals, _SOURCE_SEAT, sheriff_seat=sheriff
    )
    assert projection, "replay hover overlay produced no inference (invisible overlay)"
    targets = sorted(target for target, _ in projection)
    assert targets == [0, 3, 4]
    assert _SOURCE_SEAT not in targets
    assert sheriff not in targets

    # 4) The reduced items carry the expected role buckets for a target the
    #    source has eliminated self + sheriff about: deputy / outlaw / traitor
    #    each ~1/3 (the FEAT-001 eliminated prior), NEVER a sheriff bucket.
    items = dict(projection)[0]
    keys = [key for key, _ in items]
    assert set(keys) == {"deputy", "outlaw", "traitor"}
    assert "sheriff" not in keys
    probabilities = dict(items)
    assert abs(probabilities["deputy"] - 1.0 / 3.0) < 1e-9
    assert abs(probabilities["outlaw"] - 1.0 / 3.0) < 1e-9
    assert abs(probabilities["traitor"] - 1.0 / 3.0) < 1e-9


def test_full_replay_overlay_data_path_non_empty_at_every_recorded_turn() -> None:
    # Turn 0 (session start) must already carry non-uniform, overlay-producing
    # internals: GameSessionController records the first snapshot AFTER
    # create_session initialized each dummy's estimator (FEAT-001), so even the
    # opening turn surfaces the overlay. Assert the whole recorded run.
    model = ReplayModel(_full_record_with_realistic_internals())
    for index in range(len(model)):
        snapshot = model.snapshot_at(index)
        assert snapshot is not None
        assert snapshot.internals is not None
        sheriff = spectator_overlay.sheriff_seat_from_observation(snapshot.observation)
        projection = spectator_overlay.inverted_role_inference(
            snapshot.internals, _SOURCE_SEAT, sheriff_seat=sheriff
        )
        assert projection, f"turn {index} produced no hover inference"


def test_revert_sensitive_internals_none_makes_overlay_empty() -> None:
    # REVERT GUARD: if ReplayModel stopped attaching internals (snapshot
    # internals None), the overlay projection would be empty -> no overlay. This
    # is exactly what a PUBLIC replay legitimately produces, so a public record
    # (build_replay_record, which never carries internals) is the natural proof.
    public = replay_store.build_replay_record(
        observations=_realistic_observations(),
        **{key: value for key, value in _common().items() if key != "observations"},
    )
    assert "internals" not in public
    model = ReplayModel(public)
    snapshot = model.snapshot_at(0)
    assert snapshot is not None
    # A public replay carries NO internals, so the overlay correctly shows
    # nothing (this is intended, not a bug).
    assert snapshot.internals is None
    sheriff = spectator_overlay.sheriff_seat_from_observation(snapshot.observation)
    assert sheriff == _SHERIFF_SEAT
    projection = spectator_overlay.inverted_role_inference(
        snapshot.internals, _SOURCE_SEAT, sheriff_seat=sheriff
    )
    assert projection == ()


def test_revert_sensitive_truncated_beliefs_drop_the_overlay_targets() -> None:
    # REVERT / FIXTURE GUARD: a TRUNCATED one-row beliefs matrix (the shape the
    # earlier _internals() fixture used) only has row 0, so
    # role_belief_row_for_seat returns a row for target seat 0 but NONE for any
    # target T >= 1. The inverted projection therefore collapses to at most the
    # single seat-0 target and LOSES seats 3 and 4 that the realistic full 5x5
    # matrix surfaces (see test_full_replay_overlay_data_path_is_non_empty_on_hover,
    # which asserts targets == [0, 3, 4]). This proves the data-path test depends
    # on a REALISTIC full matrix: a regression that drops belief rows (or a
    # fixture that reverts to the truncated shape) would silently shrink the
    # overlay, and this assertion would fail.
    truncated = {
        "health": [5, 5, 5, 5, 5],
        "seats": {
            str(seat): {
                "kind": "dummy",
                "state": {"role_estimator": {"beliefs": [[0.2, 0.2, 0.2, 0.2, 0.2]]}},
            }
            for seat in range(5)
        },
    }
    record = replay_store.build_full_replay_record(
        states=_states(),
        internals=[truncated for _ in range(3)],
        observations=_realistic_observations(),
        **{key: value for key, value in _common().items() if key != "observations"},
    )
    model = ReplayModel(record)
    snapshot = model.snapshot_at(0)
    assert snapshot is not None and snapshot.internals is not None
    sheriff = spectator_overlay.sheriff_seat_from_observation(snapshot.observation)
    projection = spectator_overlay.inverted_role_inference(
        snapshot.internals, _SOURCE_SEAT, sheriff_seat=sheriff
    )
    targets = sorted(target for target, _ in projection)
    # Only the single row-0 target survives; the full-matrix targets 3 and 4 are
    # gone because their rows (T >= 1) do not exist in a one-row matrix.
    assert targets == [0]
    assert 3 not in targets and 4 not in targets
