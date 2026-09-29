from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from types import SimpleNamespace
from bang_rl.gui import GameGuiError, GameMode, GameSessionController, GuiDependencyError, QT_AVAILABLE
from bang_rl.gui.dialogs.model_select import ArtifactOption, ModelSelection
from bang_rl.gui.widgets import opponent_public_rows
from bang_rl.gui.widgets import board_layout


@dataclass
class FakeService:
    states: list[object]

    def __post_init__(self) -> None:
        self.created: list[object] = []
        self.steps: list[object] = []
        self.closed: list[object] = []
        self.handle = SimpleNamespace(session_id="session-1")

    def create_session(self, request):
        self.created.append(request)
        return self.handle

    def get_state(self, handle):
        return self.states[0]

    def step(self, handle, action=None):
        self.steps.append(action)
        if len(self.states) > 1:
            self.states.pop(0)
        return SimpleNamespace(state=self.states[0], reward=0.0, action=action, info={})

    def close_session(self, handle):
        self.closed.append(handle)


def state(*, current=0, mask=(1, 1, 0, 0, 0, 0), terminated=False, truncated=False, step=0, winning_team=None):
    observation = {
        "healths": (5, 4, 4, 4, 4),
        "alive": (True, True, True, True, True),
        "role_visibility": (1, 0, 0, 0, 0),
        "revealed_roles": (0, 255, 255, 255, 255),
        "current_player_idx": (current,),
        "action_mask": mask,
    }
    return SimpleNamespace(session_id="session-1", episode_id="episode-1", observation=observation, action_mask=tuple(mask), environment_step=step, current_player=current, terminated=terminated, truncated=truncated, termination_status="normal" if terminated else "truncated" if truncated else "running", winning_team=winning_team, winners=())


def test_all_modes_are_stable_and_main_modes_require_artifact() -> None:
    assert {mode.value for mode in GameMode} == {
        "human_random",
        "human_dummy",
        "human_main_dummy",
        "random_five",
        "dummy_five",
        "main_one_dummy_four",
    }
    # Both Main-AI modes (human-played and spectator) require an artifact.
    controller = GameSessionController(FakeService([state()]))
    with pytest.raises(GameGuiError, match="compatible"):
        controller.start(GameMode.HUMAN_MAIN_DUMMY)
    controller = GameSessionController(FakeService([state()]))
    with pytest.raises(GameGuiError, match="compatible"):
        controller.start(GameMode.MAIN_ONE_DUMMY_FOUR)


def test_mode_spectator_and_human_helpers() -> None:
    assert GameMode.RANDOM_FIVE.is_spectator
    assert GameMode.DUMMY_FIVE.is_spectator
    assert GameMode.MAIN_ONE_DUMMY_FOUR.is_spectator
    assert not GameMode.HUMAN_RANDOM.is_spectator
    assert GameMode.HUMAN_RANDOM.is_human
    assert GameMode.HUMAN_MAIN_DUMMY.is_human
    assert not GameMode.RANDOM_FIVE.is_human
    # Every mode carries a non-empty label.
    for mode in GameMode:
        assert mode.label


def test_controller_wires_service_and_never_submits_masked_action() -> None:
    service = FakeService([state(), state(step=1)])
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1)
    assert service.created[0].mode == "human_random"
    assert service.steps == [1]
    before = len(service.steps)
    snapshot = controller.submit_human_action(2)
    assert len(service.steps) == before
    assert snapshot.message and "not legal" in snapshot.message


def test_controller_pause_resume_and_terminal_cleanup_are_idempotent() -> None:
    service = FakeService([state(), state(step=1, terminated=True, winning_team="sheriff_team")])
    controller = GameSessionController(service)
    controller.start("human_dummy")
    controller.pause()
    controller.submit_human_action(1)
    assert service.steps == []
    controller.resume()
    assert service.steps == []
    snapshot = controller.submit_human_action(1)
    assert service.steps == [1]
    assert not controller.active
    # The normal-termination message names the winning faction (localized).
    assert snapshot.message and "finished normally" in snapshot.message
    assert "Sheriff team" in snapshot.message
    controller.close()
    assert len(service.closed) == 1


def test_terminal_message_is_localized_in_korean() -> None:
    from bang_rl.gui.i18n import current_language, set_language

    original = current_language()
    try:
        set_language("ko")
        service = FakeService([state(), state(step=1, terminated=True, winning_team="outlaw_team")])
        controller = GameSessionController(service)
        controller.start("human_dummy")
        snapshot = controller.submit_human_action(1)
        assert snapshot.message is not None
        assert "정상적으로 종료" in snapshot.message
        assert "무법자 팀" in snapshot.message
    finally:
        set_language(original)


def test_truncated_state_is_closed_and_has_no_winner_view() -> None:
    service = FakeService([state(), state(step=1, truncated=True)])
    controller = GameSessionController(service)
    snapshot = controller.start("human_random")
    assert snapshot.active
    result = controller.submit_human_action(1)
    assert result.state is not None and result.state.truncated
    assert not result.active
    assert result.message and "episode limit" in result.message
    assert len(service.closed) == 1


def test_start_does_not_require_a_live_on_update_callback() -> None:
    """Starting with on_update=None must not raise (callback detached on teardown)."""

    service = FakeService([state()])
    controller = GameSessionController(service)
    controller.on_update = None
    snapshot = controller.start(GameMode.HUMAN_RANDOM)
    assert snapshot.active
    assert controller.on_update is None


def test_start_after_detaching_a_stale_callback_does_not_invoke_it() -> None:
    """Simulate reopening a game: a stale screen's callback is detached before restart."""

    calls: list[object] = []
    service = FakeService([state(), state(step=1)])
    controller = GameSessionController(service)
    # First "screen" binds its render callback.
    controller.on_update = lambda snap: calls.append(snap)
    controller.start(GameMode.HUMAN_RANDOM)
    assert len(calls) == 1
    # Screen is torn down -> callback detached (mirrors main_window._select_mode
    # and GameScreen._detach resetting controller.on_update = None).
    controller.on_update = None
    controller.close()
    before = len(calls)
    # A new session starts without invoking the stale callback.
    controller.start(GameMode.HUMAN_RANDOM)
    assert len(calls) == before


def test_model_selection_keeps_incompatible_artifacts_unselectable() -> None:
    class Compatibility:
        def check(self, path, **kwargs):
            compatible = path.endswith("good.bangmodel")
            return type("Result", (), {"compatible": compatible, "metadata": {"path": path}, "errors": () if compatible else ({"message": "version mismatch"},)})()

    options = ModelSelection(Compatibility()).discover(["bad.bangmodel", "good.bangmodel"])
    assert ModelSelection.selectable(options) == (options[1],)
    assert options[0].compatibility_text == "Not compatible"


def test_gui_dependency_is_graceful_and_cross_layer_imports_stay_clean() -> None:
    if not QT_AVAILABLE:
        from bang_rl.gui.main_window import MainWindow

        with pytest.raises(GuiDependencyError):
            MainWindow()
    root = Path(__file__).parents[1]
    for relative in ("env", "agents", "training", "app"):
        for path in (root / relative).rglob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            assert "pyqt6" not in source
            assert "pyqtgraph" not in source


def test_public_opponent_hud_has_four_rows_and_hides_unrevealed_roles() -> None:
    rows = opponent_public_rows(state(), human_player=0)
    assert tuple(row["player"] for row in rows) == (1, 2, 3, 4)
    assert all(row["revealed_role"] is None for row in rows)


class _FailingStepService(FakeService):
    """Fake service whose step() fails, to exercise the error-message path."""

    def __init__(self, states, *, error: Exception) -> None:
        super().__init__(states)
        self._error = error

    def step(self, handle, action=None):
        self.steps.append(action)
        raise self._error


def test_failing_operation_populates_a_visible_error_message() -> None:
    """A failed step must surface a non-empty message on the snapshot.

    GameScreen renders snapshot.message into ActionPanel.set_message, so a
    populated message is what keeps the bottom error text visible (item 2).
    """

    service = _FailingStepService([state()], error=RuntimeError("boom"))
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    with pytest.raises(GameGuiError):
        controller.submit_human_action(1)
    assert controller.snapshot.message
    assert "boom" in controller.snapshot.message


def test_error_message_falls_back_to_localized_generic_when_exception_is_empty() -> None:
    from bang_rl.gui.i18n import current_language, set_language, tr

    original = current_language()
    try:
        set_language("en")
        service = _FailingStepService([state()], error=RuntimeError(""))
        controller = GameSessionController(service)
        controller.start(GameMode.HUMAN_RANDOM)
        with pytest.raises(GameGuiError):
            controller.submit_human_action(1)
        assert controller.snapshot.message == tr("game.error_generic")
    finally:
        set_language(original)


# --- Graphical board layout helpers (pure, Qt-free) ----------------------

def test_human_seat_is_at_the_bottom_and_seats_are_distinct() -> None:
    width, height = 800.0, 600.0
    positions = board_layout.seat_positions(width, height, human_player=0)
    assert set(positions) == {0, 1, 2, 3, 4}
    human_x, human_y = positions[0]
    # The human seat is centred horizontally and near the bottom edge.
    assert human_x == pytest.approx(width / 2)
    assert human_y > height * 0.75
    # Every opponent sits above (smaller y) the human seat.
    for player in (1, 2, 3, 4):
        assert positions[player][1] < human_y


def test_click_on_opponent_maps_to_bang_and_self_maps_to_pass() -> None:
    width, height = 800.0, 600.0
    positions = board_layout.seat_positions(width, height, human_player=0)
    # Clicking the human's own circle -> PASS (action 0).
    hx, hy = positions[0]
    seat = board_layout.seat_at_point(hx, hy, width, height, human_player=0)
    assert seat == 0
    assert board_layout.action_for_seat(seat, human_player=0) == 0
    # Clicking opponent player 3 -> BANG action index 4 (player_index + 1).
    ox, oy = positions[3]
    opp_seat = board_layout.seat_at_point(ox, oy, width, height, human_player=0)
    assert opp_seat == 3
    assert board_layout.action_for_seat(opp_seat, human_player=0) == 4


def test_click_outside_any_circle_returns_none() -> None:
    assert board_layout.seat_at_point(0.0, 0.0, 800.0, 600.0, human_player=0) is None


def test_illegal_targets_are_rejected_against_the_mask() -> None:
    mask = (1, 0, 1, 0, 0, 0)  # PASS legal, BANG player_1 legal, others not.
    assert board_layout.is_action_legal(0, mask) is True
    assert board_layout.is_action_legal(2, mask) is True
    assert board_layout.is_action_legal(1, mask) is False
    assert board_layout.is_action_legal(5, mask) is False
    # Out-of-range action indices are rejected safely.
    assert board_layout.is_action_legal(9, mask) is False


def test_seat_fill_color_reflects_health_and_elimination() -> None:
    healths = (5, 1, 0, 4, 4)
    alive = (True, True, False, True, True)
    # An eliminated seat is a faded near-black RGBA regardless of stored health.
    assert board_layout.seat_fill_color(2, healths, alive) == (24, 24, 28, 90)
    full = board_layout.seat_fill_color(0, healths, alive)
    low = board_layout.seat_fill_color(1, healths, alive)
    # Living seats return an opaque RGBA fill.
    assert full[3] == 255
    assert low[3] == 255
    # Lower health shifts the fill redder (more red, less blue) than full health.
    assert low[0] > full[0]
    assert low[2] < full[2]


def test_muzzle_point_and_angle_point_from_shooter_toward_target() -> None:
    width, height = 800.0, 600.0
    result = board_layout.muzzle_point_and_angle(1, 3, width, height, human_player=0)
    assert result is not None
    (mx, my), angle = result
    positions = board_layout.seat_positions(width, height, human_player=0)
    sx, sy = positions[1]
    radius = board_layout.circle_radius(width, height)
    # The muzzle sits one radius from the shooter centre.
    assert ((mx - sx) ** 2 + (my - sy) ** 2) ** 0.5 == pytest.approx(radius, rel=1e-6)
    # A shooter cannot fire at itself.
    assert board_layout.muzzle_point_and_angle(1, 1, width, height) is None


# --- Controller single-AI-step seam --------------------------------------

def test_single_ai_step_advances_exactly_one_service_step() -> None:
    # Human at slot 0 acts (advance_ai=False), then an AI seat (player 1) is
    # active, then control returns to the human. step_single_ai_turn must
    # advance exactly one service.step at a time and stop at the human seat.
    service = FakeService([
        state(current=0, mask=(1, 1, 0, 0, 0, 0)),
        state(current=1, mask=(1, 0, 0, 0, 0, 0), step=1),
        state(current=0, mask=(1, 1, 0, 0, 0, 0), step=2),
    ])
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1, advance_ai=False)
    assert service.steps == [1]
    assert controller.is_ai_turn() is True
    controller.step_single_ai_turn()
    # Exactly one AI step ran and control returned to the human seat.
    assert service.steps == [1, None]
    assert controller.is_ai_turn() is False
    # Polling again at the human seat is a safe no-op (no extra step).
    controller.step_single_ai_turn()
    assert service.steps == [1, None]


def test_submit_human_action_can_defer_ai_advancement() -> None:
    # With advance_ai=False the controller must not fast-forward AI turns;
    # the caller (graphical screen) drives them one at a time instead.
    service = FakeService([
        state(current=0, mask=(1, 1, 0, 0, 0, 0)),
        state(current=1, mask=(1, 0, 0, 0, 0, 0), step=1),
        state(current=0, mask=(1, 1, 0, 0, 0, 0), step=2),
    ])
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1, advance_ai=False)
    # Only the human's own step ran; the AI seat is now pending.
    assert service.steps == [1]
    assert controller.is_ai_turn() is True


def test_ai_speed_slider_bounds_include_the_one_second_default() -> None:
    from bang_rl.gui.screens.game_screen import (
        AI_SPEED_DEFAULT_MS,
        AI_SPEED_MAX_MS,
        AI_SPEED_MIN_MS,
    )

    assert AI_SPEED_MIN_MS <= AI_SPEED_DEFAULT_MS <= AI_SPEED_MAX_MS
    assert AI_SPEED_DEFAULT_MS == 1000


# --- Item 6: spectator (all-AI) modes ------------------------------------

def test_spectator_mode_gates_human_actions_to_a_no_op() -> None:
    # Two turns so the game can advance; seat 0 is AI-driven in spectator mode.
    service = FakeService([state(current=0), state(current=1, step=1)])
    controller = GameSessionController(service)
    snapshot = controller.start(GameMode.DUMMY_FIVE)
    assert snapshot.active
    # Every seat is an AI turn in spectator mode (including seat 0).
    assert controller.is_ai_turn() is True
    # Submitting a human action does NOT step the game; it only sets a notice.
    before = list(service.steps)
    result = controller.submit_human_action(1)
    assert service.steps == before  # no advance
    assert result.message  # localized spectator notice
    # The screen still drives AI turns one at a time.
    controller.step_single_ai_turn()
    assert len(service.steps) >= 1


def test_spectator_derives_ai_versions_for_all_five_seats() -> None:
    controller = GameSessionController(FakeService([state()]))
    controller.mode = GameMode.RANDOM_FIVE
    versions = controller._derive_ai_versions(GameMode.RANDOM_FIVE)
    assert versions == {str(seat): "random" for seat in range(5)}
    dummy = controller._derive_ai_versions(GameMode.DUMMY_FIVE)
    assert dummy == {str(seat): "dummy" for seat in range(5)}
    main = controller._derive_ai_versions(GameMode.MAIN_ONE_DUMMY_FOUR)
    assert set(main) == {"0", "1", "2", "3", "4"}
    assert isinstance(main["0"], dict) and main["0"]["kind"] == "main"
    assert all(main[str(seat)] == "dummy" for seat in range(1, 5))


def test_spectator_game_flags_full_replay_format() -> None:
    controller = GameSessionController(FakeService([state()]))
    controller.start(GameMode.DUMMY_FIVE)
    assert controller._save_full_replay is True
    human = GameSessionController(FakeService([state()]))
    human.start(GameMode.HUMAN_RANDOM)
    assert human._save_full_replay is True


# --- Item 5: full-state rewind-and-continue ------------------------------

class RestoreService(FakeService):
    """A FakeService exposing full-state export + restore support."""

    def __init__(self, states) -> None:
        super().__init__(states)
        self.restored: list[object] = []

    def full_state(self, handle):  # noqa: D401 - test seam
        return {"episode_id": "e", "environment_step": len(self.steps)}

    def restore_session(self, state_dict):
        self.restored.append(state_dict)
        return SimpleNamespace(session_id="forked")


def test_restore_to_moment_forks_a_continuable_session_when_supported() -> None:
    service = RestoreService([
        state(current=0, step=0),
        state(current=0, step=1),
    ])
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1)
    # Full state was captured for the human game.
    assert controller._replay_states
    controller.restore_to_moment(0)
    assert service.restored  # the service was asked to rebuild a session
    assert controller.active is True
    assert controller.previewing is False
    # A fresh line can be saved again from the fork.
    assert controller._replay_saved is False


def test_restore_to_moment_falls_back_to_review_without_restore_support() -> None:
    # Plain FakeService exposes no full_state / restore_session methods.
    service = FakeService([state(current=0, step=0), state(current=0, step=1)])
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1)
    # No authoritative state captured -> falls back to read-only review.
    assert controller._replay_states == []
    controller.restore_to_moment(0)
    assert controller.previewing is True
    assert controller.active is False


def _turn_state(turn: int, *, step: int) -> object:
    """A human-turn state with a per-turn-distinct observation.

    Health of seat 1 varies by ``turn`` so consecutive observations differ and
    are NOT de-duplicated by ``_record_replay_snapshot`` (which drops repeats).
    Seat 0 (the human) stays active with a legal BANG so the human can keep
    acting across turns.
    """

    healths = [5, max(1, 4 - turn), 4, 4, 4]
    observation = {
        "healths": tuple(healths),
        "alive": (True, True, True, True, True),
        "role_visibility": (1, 0, 0, 0, 0),
        "revealed_roles": (0, 255, 255, 255, 255),
        "current_player_idx": (0,),
        "environment_step": (step,),
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


class MissingExportRestoreService(RestoreService):
    """Restore service whose full_state export misses on ONE chosen turn.

    Simulates a transient authoritative-state export failure: on the target
    turn ``full_state`` returns ``None`` (or raises), on every other turn it
    returns a dict tagged with that turn's index so a test can prove which turn
    a fork actually resumed from. ``full_state`` is called once per recorded
    turn (via ``_record_full_state``), so the internal counter tracks turns.
    """

    def __init__(self, states, *, miss_turn: int, raises: bool = False) -> None:
        super().__init__(states)
        self._miss_turn = int(miss_turn)
        self._raises = bool(raises)
        self._turn = -1

    def full_state(self, handle):  # noqa: D401 - test seam
        self._turn += 1
        if self._turn == self._miss_turn:
            if self._raises:
                raise RuntimeError("export failed for this turn")
            return None
        return {"episode_id": "e", "turn_tag": self._turn}


def _three_turn_service(*, miss_turn: int, raises: bool = False) -> MissingExportRestoreService:
    return MissingExportRestoreService(
        [_turn_state(0, step=0), _turn_state(1, step=1), _turn_state(2, step=2)],
        miss_turn=miss_turn,
        raises=raises,
    )


def _drive_three_turns(controller: GameSessionController) -> None:
    """Advance the controller across three distinct recorded turns."""

    controller.start(GameMode.HUMAN_RANDOM)  # turn 0 recorded
    controller.submit_human_action(1)  # turn 1 recorded
    controller.submit_human_action(1)  # turn 2 recorded


def test_export_miss_keeps_state_buffer_index_parallel() -> None:
    # A missed export on turn 1 must NOT desync the buffers: _record_full_state
    # appends a None placeholder so both buffers stay one per recorded turn.
    service = _three_turn_service(miss_turn=1)
    controller = GameSessionController(service)
    _drive_three_turns(controller)
    # Three distinct turns were recorded and both buffers advanced in lock-step
    # (one entry per recorded turn) even though turn 1's export returned None.
    assert len(controller._replay_observations) == 3
    assert len(controller._replay_states) == len(controller._replay_observations)
    assert controller._replay_states[1] is None
    assert isinstance(controller._replay_states[0], Mapping)
    assert isinstance(controller._replay_states[2], Mapping)


def test_restore_after_export_miss_targets_the_correct_turn() -> None:
    # After a turn-1 export miss, restoring to turn 2 must fork from turn 2's
    # authoritative state (not turn 0 or 1), proving indices stayed aligned.
    service = _three_turn_service(miss_turn=1)
    controller = GameSessionController(service)
    _drive_three_turns(controller)
    controller.restore_to_moment(2)
    assert controller.active is True
    assert controller.previewing is False
    assert service.restored  # a fork happened
    # The fork used turn 2's tagged state, not a misaligned earlier turn.
    assert service.restored[-1]["turn_tag"] == 2


def test_restore_to_missed_turn_falls_back_to_review_not_wrong_state() -> None:
    # Restoring to the exact turn whose export missed must fall back to the
    # read-only review rather than forking from a wrong/absent state.
    service = _three_turn_service(miss_turn=1)
    controller = GameSessionController(service)
    _drive_three_turns(controller)
    controller.restore_to_moment(1)
    # No fork: safe read-only preview instead.
    assert service.restored == []
    assert controller.previewing is True
    assert controller.active is False


def test_restore_after_export_that_raises_still_targets_correct_turn() -> None:
    # An exporter that RAISES on a turn behaves like a miss (None placeholder),
    # so a later restore still forks from the correct, aligned turn.
    service = _three_turn_service(miss_turn=0, raises=True)
    controller = GameSessionController(service)
    _drive_three_turns(controller)
    assert len(controller._replay_states) == len(controller._replay_observations)
    assert controller._replay_states[0] is None
    controller.restore_to_moment(2)
    assert controller.active is True
    assert service.restored[-1]["turn_tag"] == 2


# --- Items 3 & 8: a log ROW resolves to its own moment (not turn 0) -------

def _logged_turn_state(turn: int, *, step: int, events: tuple) -> object:
    """A human-turn state whose observation carries a cumulative event log.

    Mirrors the REAL recorded shape (unlike ``_turn_state``, which omits
    ``event_sequence``): each event carries an ``environment_step`` and the
    observation's ``event_sequence`` is the cumulative log up to this turn, so
    the GameLog row index the GUI hands to the controller resolves through
    ``ReplayModel.index_for_log_row``.
    """

    observation = {
        "healths": (5, max(1, 5 - turn), 5, 5, 5),
        "alive": (True, True, True, True, True),
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


def _logged_three_turn_states() -> list[object]:
    events = [
        {"environment_step": 0, "event_type": "game_started"},
        {"environment_step": 1, "event_type": "bang", "actor_idx": 0, "target_idx": 1},
        {"environment_step": 2, "event_type": "bang", "actor_idx": 1, "target_idx": 2},
    ]
    return [
        _logged_turn_state(0, step=0, events=tuple(events[:1])),
        _logged_turn_state(1, step=1, events=tuple(events[:2])),
        _logged_turn_state(2, step=2, events=tuple(events[:3])),
    ]


def test_time_travel_to_a_late_log_row_previews_that_moment_not_turn_zero() -> None:
    # Item 3/8 regression: clicking/hovering the LAST visible log row must move
    # the board to that moment (environment_step 2 here), never reset to turn 0.
    service = FakeService(_logged_three_turn_states())
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1)  # turn 1 recorded
    controller.submit_human_action(1)  # turn 2 recorded
    assert len(controller._replay_observations) == 3
    # Row 2 is the last log row (the step-2 bang).
    controller.time_travel_to(2)
    assert controller.previewing is True
    assert controller.state.environment_step == 2  # NOT 0


def test_restore_to_moment_by_log_row_forks_from_that_row_state() -> None:
    # Item 8: restoring a mid/late log row forks from that row's authoritative
    # state, proving the row -> state index mapping does not collapse to 0.
    service = RestoreService(_logged_three_turn_states())
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1)
    controller.submit_human_action(1)
    controller.restore_to_moment(2)
    assert service.restored  # a fork happened
    # The fork used the step-2 turn's captured state, not turn 0.
    assert service.restored[-1]["environment_step"] != 0


# --- item 4: per-step internals capture, index-parallel + restore ---------

class InternalsRestoreService(RestoreService):
    """RestoreService that also exports per-turn AI internals and accepts them."""

    def __init__(self, states) -> None:
        super().__init__(states)
        self.restored_internals: list[object] = []

    def export_agent_internals(self, handle):
        return {
            "health": [5, max(1, 5 - len(self.steps)), 5, 5, 5],
            "seats": {"1": {"kind": "dummy", "state": {"turn": len(self.steps)}}},
        }

    def restore_session(self, state_dict, *, internals=None):
        self.restored_internals.append(internals)
        return super().restore_session(state_dict)


def test_controller_records_third_internals_buffer_index_parallel() -> None:
    service = InternalsRestoreService(_logged_three_turn_states())
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1)
    controller.submit_human_action(1)
    assert len(controller._replay_internals) == len(controller._replay_observations)
    assert len(controller._replay_states) == len(controller._replay_internals)
    for entry in controller._replay_internals:
        assert entry is not None
        assert len(entry["health"]) == 5
        assert entry["seats"]["1"]["kind"] == "dummy"


def test_restore_to_moment_passes_matching_internals_when_supported() -> None:
    service = InternalsRestoreService(_logged_three_turn_states())
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1)
    controller.submit_human_action(1)
    controller.restore_to_moment(1)
    assert service.restored  # a fork happened
    assert service.restored_internals  # internals were forwarded to the restorer
    assert service.restored_internals[-1] is not None
    assert service.restored_internals[-1]["seats"]["1"]["kind"] == "dummy"


def test_restore_still_forks_when_restorer_rejects_internals_kwarg() -> None:
    # A legacy RestoreService.restore_session takes ONLY the state dict; the
    # controller must not pass internals to it and must still fork.
    service = RestoreService(_logged_three_turn_states())

    def export_agent_internals(handle):
        return {"health": [5, 5, 5, 5, 5], "seats": {}}

    service.export_agent_internals = export_agent_internals  # type: ignore[attr-defined]
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1)
    controller.submit_human_action(1)
    controller.restore_to_moment(1)
    assert service.restored  # the state-only restorer still forked a session
    assert controller.active is True


def test_log_history_before_clicked_row_is_preserved_on_resume() -> None:
    # Item 3: clicking a mid log row to resume must NOT delete the earlier log.
    service = InternalsRestoreService(_logged_three_turn_states())
    controller = GameSessionController(service)
    controller.start(GameMode.HUMAN_RANDOM)
    controller.submit_human_action(1)
    controller.submit_human_action(1)
    from bang_rl.gui.replay import ReplayModel

    before = [
        e.get("event_type")
        for e in ReplayModel(list(controller._replay_observations)).full_events()
    ]
    assert "game_started" in before
    controller.restore_to_moment(1)
    after = [
        e.get("event_type")
        for e in ReplayModel(list(controller._replay_observations)).full_events()
    ]
    # The events at / before the clicked moment survived (not wiped).
    assert after[0] == "game_started"
    assert "bang" in after


# --- Item 1 + 3: gun to the visual right, reduced top margin -------------

def test_gun_anchor_sits_on_the_opposite_perpendicular_side() -> None:
    import math

    width, height = 1000.0, 1000.0
    for facing in (0.0, math.pi / 2, 1.2, -0.7):
        result = board_layout.gun_anchor_and_angle(0, width, height, facing=facing)
        assert result is not None
        (ax, ay), angle = result
        cx, cy = board_layout.seat_positions(width, height)[0]
        radius = board_layout.circle_radius(width, height)
        # New offset is the +90 degree perpendicular (-sin, cos); the OLD
        # implementation used the opposite (sin, -cos). Assert the sign flipped.
        old = (math.sin(facing), -math.cos(facing))
        new_x = (ax - cx) / radius
        new_y = (ay - cy) / radius
        assert new_x == pytest.approx(-old[0], abs=1e-9)
        assert new_y == pytest.approx(-old[1], abs=1e-9)
        # The barrel still points along facing.
        assert angle == pytest.approx(facing)


def test_human_seat_moved_up_to_fit_three_caption_lines() -> None:
    width, height = 1000.0, 1000.0
    human_x, human_y = board_layout.seat_positions(width, height)[0]
    # Human seat raised from the old 0.84 fraction so the 3 caption lines fit.
    assert human_y < 0.84 * height
    # There must be room below the human circle for radius + 3 caption lines.
    radius = board_layout.circle_radius(width, height)
    line_height = 18.0  # generous per-caption-line estimate
    assert human_y + radius + 3 * line_height <= height
    # The top opponent row was raised to reclaim the freed top margin.
    positions = board_layout.seat_positions(width, height)
    top_y = min(positions[p][1] for p in (1, 2, 3, 4))
    assert top_y <= 0.16 * height


def test_new_ai_speed_i18n_keys_exist_in_both_languages() -> None:
    from bang_rl.gui.i18n import _STRINGS

    for key in (
        "game.ai_speed",
        "game.ai_speed_value",
        "game.you_name",
        "game.player_name",
        # Item 6: spectator (all-AI) no-action notice.
        "game.spectator_no_actions",
        # Keys wired into the graphical board / screens (no longer dead).
        "game.click_hint",
        "board.hit",
        "board.eliminated",
        "replay.autoplay",
        "settings.emotion_initial_baseline",
        "tooltip.emotion_initial_baseline_field",
    ):
        assert key in _STRINGS
        assert "en" in _STRINGS[key]
        assert "ko" in _STRINGS[key]
