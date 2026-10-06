"""Sandbox-executable regression for the FEAT-001 replay hover overlay fix.

The reported bug was "리플레이 포맷이 full인데도 표시되지 않아": in a FULL-format
replay the inverted role-inference hover overlay (the plates drawn over the OTHER
players' circles when a reviewer hovers a seat) did not render, even though the
always-on emotion bars did. The DATA path and the pure overlay math are correct
(proven here and in test_replay_internals_pure.py); the runtime break lived in
the Qt hover / paint layer of GraphicalBoard:

  (A) mouseMoveEvent assigned ``self._hover_seat`` only AFTER a best-effort
      emotion-bar tooltip hit-test, so any exception in that hit-test aborted the
      handler before ``_hover_seat`` was set and the overlay never painted; and
  (B) ``_paint_game_over_banner`` ran AFTER ``_paint_role_inference``, so on the
      terminal replay frame the large centred banner overdrew the inference
      plates.

Those two fixes are pure Qt ordering / robustness and cannot be exercised in the
restricted sandbox (PyQt6 / numpy / pydantic are not installable and
GraphicalBoard lives under ``if QT_AVAILABLE:``). What this module locks in is
the EXACT replay -> overlay data the fixed ``_paint_role_inference`` call site
consumes, so a regression that broke the data path (and would make the overlay
invisible for a different reason) is still caught headlessly.

Like test_replay_internals_pure.py, importing ``bang_rl.gui.replay_store`` /
``.replay`` / ``.widgets.spectator_overlay`` directly would execute
``bang_rl/gui/__init__`` which pulls in pydantic / PyQt6, so this mirrors the
importlib path-loading pattern and loads the three Qt-free, numpy-free modules
straight from their paths.
"""
from __future__ import annotations

import ast
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


replay_store = _load_module("replay_store.py", "_replay_store_overlay_hover_isolated")
replay = _load_module("replay.py", "_replay_overlay_hover_isolated")
spectator_overlay = _load_module(
    "widgets/spectator_overlay.py", "_spectator_overlay_hover_isolated"
)
ReplayModel = replay.ReplayModel


# The sheriff sits at a NON-zero seat so the test is revert-sensitive to any
# "seat 0 is always the sheriff" assumption: the sheriff is seat 2 here.
_SHERIFF_SEAT = 2
# The hovered (source) seat whose own beliefs about the OTHER seats are drawn.
_SOURCE_SEAT = 1
# The human seat: hovering it must yield an empty projection (human is not a
# dummy, so there is no estimator to invert) -- the revert-sensitive fallback.
_HUMAN_SEAT = 0


def _eliminated_beliefs_for_seat(seat: int, sheriff: int = _SHERIFF_SEAT) -> list:
    """A realistic, FEAT-001-style eliminated 5x5 beliefs matrix.

    Row ``T`` is this dummy's distribution over seat ``T``'s role in canonical
    order ``(sheriff, deputy, outlaw_a, outlaw_b, traitor)``: the self-row is
    one-hot on the dummy's own known role (outlaw_a), the sheriff row is one-hot
    on the public Sheriff role, and every OTHER row folds out the two known roles
    (self + sheriff) and splits the remaining mass evenly over deputy / the two
    outlaw slots / traitor (deputy 1/3, outlaw merged 1/3, traitor 1/3). This is
    the same shape used by test_replay_internals_pure.py.
    """

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
    """One per-turn internals entry: HP, a human seat 0, dummy seats 1..4.

    Seat 0 is the human (``{'kind': 'human'}``) and seats 1..4 are dummies with
    the realistic 5x5 beliefs. This mirrors a real human-vs-dummy FULL record so
    hovering the human seat exercises the empty-projection fallback.
    """

    seats: dict[str, dict] = {str(_HUMAN_SEAT): {"kind": "human"}}
    for seat in range(5):
        if seat == _HUMAN_SEAT:
            continue
        seats[str(seat)] = _dummy_seat_descriptor(seat)
    return {"health": [5, 5, 5, 5, 5], "seats": seats}


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
    """Mimic save/load: seat keys become strings and tuples become lists."""

    return json.loads(json.dumps(record))


def test_full_replay_hover_overlay_renders_for_hovered_dummy_seat() -> None:
    # The EXACT chain GraphicalBoard._paint_role_inference consumes: a FULL
    # record (JSON-roundtripped like save/load) -> ReplayModel -> snapshot
    # internals -> inverted_role_inference with the sheriff seat derived from the
    # observation. A NON-EMPTY projection is what makes the overlay actually
    # paint on hover.
    model = ReplayModel(_json_roundtrip(_full_record()))
    snapshot = model.snapshot_at(0)
    assert snapshot is not None
    assert snapshot.internals is not None
    assert snapshot.internals["seats"][str(_SOURCE_SEAT)]["kind"] == "dummy"

    sheriff = spectator_overlay.sheriff_seat_from_observation(snapshot.observation)
    assert sheriff == _SHERIFF_SEAT

    projection = spectator_overlay.inverted_role_inference(
        snapshot.internals, _SOURCE_SEAT, sheriff_seat=sheriff
    )
    assert projection, "replay hover overlay produced no inference (invisible overlay)"
    targets = sorted(target for target, _ in projection)
    # Every OTHER non-sheriff seat; the source seat and the sheriff seat are
    # both excluded.
    assert targets == [0, 3, 4]
    assert _SOURCE_SEAT not in targets
    assert sheriff not in targets


def test_full_replay_hover_overlay_renders_at_every_recorded_turn() -> None:
    model = ReplayModel(_json_roundtrip(_full_record()))
    for index in range(len(model)):
        snapshot = model.snapshot_at(index)
        assert snapshot is not None
        assert snapshot.internals is not None
        sheriff = spectator_overlay.sheriff_seat_from_observation(snapshot.observation)
        projection = spectator_overlay.inverted_role_inference(
            snapshot.internals, _SOURCE_SEAT, sheriff_seat=sheriff
        )
        assert projection, f"turn {index} produced no hover inference"


def test_hovering_the_human_seat_yields_empty_projection() -> None:
    # REVERT GUARD: hovering the HUMAN seat (0) must produce an EMPTY projection
    # (there is no dummy estimator to invert for a human seat), so the overlay
    # correctly draws nothing there -- not a bug. This proves the non-empty
    # result above is specific to the dummy SOURCE seat, not an artefact.
    model = ReplayModel(_json_roundtrip(_full_record()))
    snapshot = model.snapshot_at(0)
    assert snapshot is not None and snapshot.internals is not None
    sheriff = spectator_overlay.sheriff_seat_from_observation(snapshot.observation)
    projection = spectator_overlay.inverted_role_inference(
        snapshot.internals, _HUMAN_SEAT, sheriff_seat=sheriff
    )
    assert projection == ()


def _render_current_source() -> str:
    """Return the source of ReplayScreen._render_current without importing Qt.

    ``replay_screen`` imports PyQt6 at module load (via ``..qt_compat``), which
    is not installable in the restricted sandbox, so parse the file as text and
    extract the method body with ast. This lets a pure test lock in the Qt
    enabled-state wiring that the real bug lived in.
    """

    source = (
        Path(__file__).resolve().parents[1] / "gui" / "screens" / "replay_screen.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_render_current":
            return ast.get_source_segment(source, node) or ""
    raise AssertionError("_render_current not found in replay_screen.py")


def test_render_current_reenables_board_on_model_present_path() -> None:
    # REGRESSION GUARD for the real (Qt-interaction) bug: a DISABLED Qt widget
    # never receives mouseMoveEvent, so GraphicalBoard._hover_seat stays None and
    # the role-inference overlay never paints. _render_current disables the board
    # on the empty-model branch (setEnabled(False)); the model-present branch
    # MUST re-enable it (setEnabled(True)) so hover works once a replay loads.
    # Qt itself cannot run in-sandbox, so assert the wiring at the source level.
    body = _render_current_source()
    assert "self.board.setEnabled(False)" in body, (
        "expected the empty-model branch to disable the board"
    )
    assert "self.board.setEnabled(True)" in body, (
        "model-present path must re-enable the board so it receives "
        "mouseMoveEvent and the hover overlay can paint"
    )
    # The re-enable must come BEFORE the snapshot render so the board is live for
    # the first frame's hover, mirroring the always-enabled spectator board.
    assert body.index("self.board.setEnabled(True)") < body.index(
        "self._render_snapshot(snapshot)"
    )


def test_public_record_has_no_internals_and_empty_projection() -> None:
    # REVERT GUARD: a PUBLIC replay (build_replay_record, which never carries
    # internals) resolves to snapshot.internals is None, so the overlay projection
    # is empty. This is exactly what the board would show if ReplayModel stopped
    # attaching internals -- i.e. it pins down the "no overlay" case.
    public = replay_store.build_replay_record(
        observations=_realistic_observations(), **_common()
    )
    assert "internals" not in public
    model = ReplayModel(_json_roundtrip(public))
    snapshot = model.snapshot_at(0)
    assert snapshot is not None
    assert snapshot.internals is None
    sheriff = spectator_overlay.sheriff_seat_from_observation(snapshot.observation)
    assert sheriff == _SHERIFF_SEAT
    projection = spectator_overlay.inverted_role_inference(
        snapshot.internals, _SOURCE_SEAT, sheriff_seat=sheriff
    )
    assert projection == ()
