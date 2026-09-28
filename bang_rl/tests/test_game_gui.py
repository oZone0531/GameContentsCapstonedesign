from __future__ import annotations

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


def test_all_three_modes_are_stable_and_main_mode_requires_artifact() -> None:
    assert {mode.value for mode in GameMode} == {"human_random", "human_dummy", "human_main_dummy"}
    controller = GameSessionController(FakeService([state()]))
    with pytest.raises(GameGuiError, match="compatible"):
        controller.start(GameMode.HUMAN_MAIN_DUMMY)


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
    # Eliminated seat is dark grey regardless of stored health.
    assert board_layout.seat_fill_color(2, healths, alive) == (70, 70, 70)
    full = board_layout.seat_fill_color(0, healths, alive)
    low = board_layout.seat_fill_color(1, healths, alive)
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


def test_new_ai_speed_i18n_keys_exist_in_both_languages() -> None:
    from bang_rl.gui.i18n import _STRINGS

    for key in ("game.ai_speed", "game.ai_speed_value", "game.you_name", "game.player_name"):
        assert key in _STRINGS
        assert "en" in _STRINGS[key]
        assert "ko" in _STRINGS[key]
