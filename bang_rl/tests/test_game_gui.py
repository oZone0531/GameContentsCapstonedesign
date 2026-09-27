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


def state(*, current=0, mask=(1, 1, 0, 0, 0, 0), terminated=False, truncated=False, step=0):
    observation = {
        "healths": (5, 4, 4, 4, 4),
        "alive": (True, True, True, True, True),
        "role_visibility": (1, 0, 0, 0, 0),
        "revealed_roles": (0, 255, 255, 255, 255),
        "current_player_idx": (current,),
        "action_mask": mask,
    }
    return SimpleNamespace(session_id="session-1", episode_id="episode-1", observation=observation, action_mask=tuple(mask), environment_step=step, current_player=current, terminated=terminated, truncated=truncated, termination_status="normal" if terminated else "truncated" if truncated else "running")


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
    service = FakeService([state(), state(step=1, terminated=True)])
    controller = GameSessionController(service)
    controller.start("human_dummy")
    controller.pause()
    controller.submit_human_action(1)
    assert service.steps == []
    controller.resume()
    assert service.steps == []
    controller.submit_human_action(1)
    assert service.steps == [1]
    assert not controller.active
    controller.close()
    assert len(service.closed) == 1


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
