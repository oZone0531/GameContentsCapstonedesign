from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from types import SimpleNamespace
from bang_rl.gui import GameGuiError, GameMode, GameSessionController, GuiDependencyError, QT_AVAILABLE
from bang_rl.gui.dialogs.model_select import ArtifactOption, ModelSelection
from bang_rl.gui.widgets import opponent_public_rows


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
