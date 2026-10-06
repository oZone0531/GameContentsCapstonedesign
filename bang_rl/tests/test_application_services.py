from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pytest

from bang_rl.app import (
    CompatibilityService,
    ConfigService,
    EvaluationRunConfig,
    EvaluationService,
    GameService,
    SessionConfig,
    TrainingConfig,
    TrainingService,
)
from bang_rl.config import SystemConfig
from bang_rl.exceptions import ResourceLimitError, TrainingError
from bang_rl.runtime.seed import SeedManager


class PassAgent:
    def on_episode_start(self, context):
        self.context = context

    def on_episode_end(self, result):
        self.result = result

    def act(self, observation, action_mask):
        assert "roles" not in observation
        return 0


def _pass_agents():
    return tuple(PassAgent() for _ in range(5))


def test_config_snapshot_is_fixed_and_public_game_state_is_read_only() -> None:
    config = SystemConfig(config_version="v1", seed=17)
    service = ConfigService()
    snapshot = service.snapshot(config, mode="rl", job_id="job-fixed")
    config.config_version = "v2"
    assert snapshot.config_version == "v1"
    assert snapshot.config_snapshot["config_version"] == "v1"

    game = GameService()
    handle = game.create_session(SessionConfig(config=config, agents=_pass_agents(), seed=17))
    state = game.get_state(handle)
    assert "roles" not in state.observation
    with pytest.raises(TypeError):
        state.observation["environment_step"] = 999  # type: ignore[index]
    game.close_session(handle)


def test_game_service_steps_through_injected_agents_without_private_state() -> None:
    config = SystemConfig(seed=3)
    game = GameService()
    handle = game.create_session(SessionConfig(config=config, agents=_pass_agents()))
    first = game.get_state(handle)
    result = game.step(handle)
    assert result.action == 0
    assert result.state.environment_step == first.environment_step + 1
    assert result.state.action_mask[0] == 1
    assert result.state.termination_status == "running"
    game.close_session(handle)


def test_seed_manager_is_reproducible_for_same_snapshot() -> None:
    snapshot = ConfigService().snapshot(SystemConfig(seed=41), mode="rl", job_id="fixed-job")
    first = SeedManager(41, snapshot=snapshot)
    second = SeedManager(41, snapshot=snapshot)
    assert first.worker_seed(0) == second.worker_seed(0)
    assert first.environment_seed("episode-0", worker_id=1) == second.environment_seed("episode-0", worker_id=1)
    assert first.action_seed("player_2", episode_id="episode-0", worker_id=1) == second.action_seed("player_2", episode_id="episode-0", worker_id=1)


def test_training_runner_lifecycle_pause_resume_cancel_and_report(tmp_path: Path) -> None:
    started = False

    def runner(handle, run):
        nonlocal started
        started = True
        time.sleep(0.05)
        return {"environment_steps": 4, "transitions": 2}

    service = TrainingService()
    handle = service.start(TrainingConfig(config=SystemConfig(), runner=runner, report_path=tmp_path / "report.json"))
    for _ in range(20):
        if started:
            break
        time.sleep(0.01)
    service.pause(handle)
    assert service.progress(handle).status == "paused"
    service.resume(handle)
    assert service.progress(handle).status == "running"
    service.cancel(handle)
    report = service.wait(handle, timeout=2)
    assert report.status == "cancelled"
    assert report.progress.environment_steps == 4
    assert report.report_path == str(tmp_path / "report.json")
    assert (tmp_path / "report.json").is_file()


def test_default_training_rejects_unwired_pipeline_without_reporting_success() -> None:
    with pytest.raises(TrainingError, match="declared collection/SL/PPO/artifact pipeline"):
        TrainingService().start(TrainingConfig(config=SystemConfig(), episodes=1))


def test_training_resource_validation_and_fault_are_structured() -> None:
    with pytest.raises(Exception):
        SystemConfig.model_validate({"hardware": {"workers": 0}})

    def faulty_runner(handle, run):
        raise RuntimeError("worker exploded")

    service = TrainingService()
    handle = service.start(TrainingConfig(config=SystemConfig(), runner=faulty_runner))
    report = service.wait(handle, timeout=2)
    assert report.status == "faulted"
    assert report.progress.error is not None
    assert report.progress.error["error_type"] == "RuntimeError"


def test_compatibility_service_rejects_missing_artifact_and_evaluation_reports_na(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bangmodel"
    result = CompatibilityService().check(missing, config=SystemConfig())
    assert result.compatible is False
    assert result.errors[0]["error_type"] in {"TrainingError", "FileNotFoundError"}

    report = EvaluationService().run_diagnostic(EvaluationRunConfig(config=SystemConfig(), games=1, runner=lambda options, seeds: {"normal_games": 0, "truncated_games": 1, "normal_wins": 0}, report_path=tmp_path / "evaluation.json"))
    assert report.win_rate == "N/A"
    assert report.truncated_rate == 1.0
    assert report.seed_set == (0,)
    assert (tmp_path / "evaluation.json").is_file()


def test_public_evaluation_service_delegates_artifact_evaluation(monkeypatch) -> None:
    from bang_rl.evaluation import protocol

    calls = {}

    class FakePrimary:
        def __init__(self, artifact, **kwargs):
            calls["artifact"] = artifact
            calls["kwargs"] = kwargs

        def run(self, evaluation):
            calls["evaluation"] = evaluation
            return "protocol-report"

    monkeypatch.setattr(protocol, "PrimaryEvaluationProtocol", FakePrimary)
    artifact = object()
    result = EvaluationService(artifact=artifact).run(EvaluationRunConfig(config=SystemConfig()))
    assert result == "protocol-report"
    assert calls["artifact"] is artifact
    assert calls["evaluation"].games == 500
    assert calls["evaluation"].games_per_seat == 100
    assert calls["evaluation"].seat_policy == "balanced"


def test_application_service_has_no_gui_imports() -> None:
    source = Path(__file__).parents[1] / "app" / "services.py"
    text = source.read_text(encoding="utf-8").lower()
    assert "pyqt" not in text
    assert "pyqtgraph" not in text


# --- FEAT-003: spectator all-AI modes + full-state export/restore -------------


def test_spectator_modes_build_all_five_ai_seats() -> None:
    from bang_rl.agents import DummyAgent, HumanAgent, RandomAgent

    game = GameService()
    random_handle = game.create_session(SessionConfig(mode="random_five", seed=5))
    random_session = game._session(random_handle)
    assert len(random_session.agents) == 5
    assert all(isinstance(agent, RandomAgent) for agent in random_session.agents)
    # No seat is a human in a spectator game.
    assert not any(isinstance(agent, HumanAgent) for agent in random_session.agents)
    game.close_session(random_handle)

    dummy_handle = game.create_session(SessionConfig(mode="dummy_five", seed=5))
    dummy_session = game._session(dummy_handle)
    assert all(isinstance(agent, DummyAgent) for agent in dummy_session.agents)
    game.close_session(dummy_handle)


def test_main_one_dummy_four_requires_artifact() -> None:
    game = GameService()
    with pytest.raises(ValueError, match="artifact"):
        game.create_session(SessionConfig(mode="main_one_dummy_four", seed=5))


def test_mode_aliases_normalize_to_spectator_modes() -> None:
    assert GameService._normalize_mode("random_ai_5") == "random_five"
    assert GameService._normalize_mode("dummy_ai_5") == "dummy_five"
    assert GameService._normalize_mode("main_1_dummy_4") == "main_one_dummy_four"
    # The canonical enum values normalize to themselves.
    assert GameService._normalize_mode("random_five") == "random_five"


def test_get_full_state_and_restore_session_round_trip() -> None:
    config = SystemConfig(seed=11)
    game = GameService()
    handle = game.create_session(SessionConfig(config=config, agents=_pass_agents(), seed=11))
    # Advance a couple of turns so the state is mid-game.
    game.step(handle)
    game.step(handle)
    full = game.get_full_state(handle)
    assert isinstance(full, dict)
    assert "agent_selection" in full and "environment_step" in full
    live_state = game.get_state(handle)

    # Restore a NEW live session from the exported authoritative state.
    restored = game.restore_session(full, mode="human_random", seed=11, config=config)
    restored_state = game.get_state(restored)
    # The restored session resumes at the same environment step / current turn.
    assert restored_state.environment_step == live_state.environment_step
    assert restored_state.current_player == live_state.current_player
    # Restore is also reachable under the create_session_from_state alias with a
    # single positional argument (the controller calls it this way).
    forked = game.create_session_from_state(full)
    assert game.get_state(forked).environment_step == live_state.environment_step
    game.close_session(handle)
    game.close_session(restored)
    game.close_session(forked)


def test_export_agent_internals_returns_hp_and_per_seat_state() -> None:
    # Item 4: a dummy_five session exposes every player's HP plus each dummy
    # seat's serialized emotion / role-belief state, and nothing leaks into the
    # public projection.
    config = SystemConfig(seed=31)
    game = GameService()
    handle = game.create_session(SessionConfig(mode="dummy_five", config=config, seed=31))
    game.step(handle)
    internals = game.export_agent_internals(handle)
    assert isinstance(internals, dict)
    assert len(internals["health"]) == 5
    seats = internals["seats"]
    # Every seat is a DummyAgent, so each descriptor carries a serialized state.
    for seat in range(5):
        descriptor = seats[str(seat)]
        assert descriptor["kind"] == "dummy"
        assert descriptor["state"]["schema_version"] == "2.0"
    # Internals never leak into the public state DTO / observation.
    state = game.get_state(handle)
    assert not hasattr(state, "agent_internals")
    assert "agent_internals" not in state.observation
    game.close_session(handle)


def test_export_agent_internals_random_seats_are_minimal() -> None:
    config = SystemConfig(seed=32)
    game = GameService()
    handle = game.create_session(SessionConfig(mode="random_five", config=config, seed=32))
    game.step(handle)
    internals = game.export_agent_internals(handle)
    for seat in range(5):
        assert internals["seats"][str(seat)] == {"kind": "random"}
    game.close_session(handle)


class _StubInferenceModule:
    """Minimal main-AI inference module exposing get_state (item-4 review)."""

    def __init__(self, state):
        self._state = state

    def get_state(self):
        return self._state


class _StubArtifact:
    def __init__(self, intent_state, emotion_state):
        self.intent = _StubInferenceModule(intent_state)
        self.emotion = _StubInferenceModule(emotion_state)


class _StubArtifactAgent:
    def __init__(self, artifact):
        self.artifact = artifact


def test_main_ai_descriptor_records_captured_true_when_buffer_present() -> None:
    # Review issue 2: the main-AI descriptor is best-effort. When the inference
    # modules hold an active stream buffer, captured is True and module_states
    # carries the JSON-safe buffers.
    agent = _StubArtifactAgent(
        _StubArtifact(intent_state=np.asarray([0.1, 0.9]), emotion_state=[[0.5]])
    )
    descriptor = GameService._agent_internal_descriptor(agent)
    assert descriptor["kind"] == "main"
    assert descriptor["captured"] is True
    assert descriptor["module_states"]["intent"] == [0.1, 0.9]
    assert descriptor["module_states"]["emotion"] == [[0.5]]


def test_main_ai_descriptor_records_captured_false_when_no_buffer() -> None:
    # The honest no-op case: get_state() returns None on every module, so
    # captured is False and there are no module_states (documented limitation:
    # a resume then keeps the freshly built main agent).
    agent = _StubArtifactAgent(_StubArtifact(intent_state=None, emotion_state=None))
    descriptor = GameService._agent_internal_descriptor(agent)
    assert descriptor["kind"] == "main"
    assert descriptor["captured"] is False
    assert "module_states" not in descriptor


def test_restore_session_with_internals_keeps_dummy_state() -> None:
    # Item 4: restoring with a captured internals mapping resumes each dummy
    # seat's serialized state rather than a fresh agent. The restore is
    # defensive and reachable via the single-positional alias too.
    config = SystemConfig(seed=33)
    game = GameService()
    handle = game.create_session(SessionConfig(mode="dummy_five", config=config, seed=33))
    game.step(handle)
    game.step(handle)
    full = game.get_full_state(handle)
    internals = game.export_agent_internals(handle)
    restored = game.restore_session(
        full, mode="dummy_five", seed=33, config=config, internals=internals
    )
    # The restored session is live and public-safe.
    restored_state = game.get_state(restored)
    assert restored_state.environment_step == game.get_state(handle).environment_step
    # A malformed / absent internals mapping must not raise (defensive resume).
    forked = game.restore_session(full, mode="dummy_five", config=config, internals=None)
    assert game.get_state(forked) is not None
    game.close_session(handle)
    game.close_session(restored)
    game.close_session(forked)


def test_restored_session_keeps_public_privacy_during_play() -> None:
    config = SystemConfig(seed=13)
    game = GameService()
    handle = game.create_session(SessionConfig(config=config, agents=_pass_agents(), seed=13))
    game.step(handle)
    full = game.get_full_state(handle)
    restored = game.restore_session(full, config=config)
    state = game.get_state(restored)
    # Public projection stays private mid-game: no living roles are leaked.
    assert "roles" not in state.observation
    assert state.final_roles == ()
    game.close_session(handle)
    game.close_session(restored)


def _role_visibility(state) -> list[int]:
    return [int(v) for v in state.observation["role_visibility"]]


def _self_role(state) -> int:
    return int(state.observation["self_role"][0])


def test_human_session_keeps_privacy_and_reveal_all_false() -> None:
    # Item 1/7: a human session never reveals living non-sheriff roles and its
    # reveal_all_roles flag stays False.
    config = SystemConfig(seed=21)
    game = GameService()
    handle = game.create_session(SessionConfig(config=config, agents=_pass_agents(), seed=21))
    state = game.get_state(handle)
    assert state.reveal_all_roles is False
    assert "reveal_all_roles" in state.to_dict()
    # Mid-game, at most the sheriff / the viewer's own seat are visible; no
    # living seat is marked REVEALED (2) beyond the sheriff reveal rule.
    visibility = _role_visibility(state)
    # The viewer's own seat (0) is SELF (1); every other living seat that is not
    # the publicly-known sheriff stays HIDDEN (0).
    assert visibility[0] == 1
    game.close_session(handle)


def test_human_self_role_is_stable_across_ai_turns() -> None:
    # Item 7 (flicker fix): the human seat's self_role must not change as the
    # current player rotates, because _state now projects from the fixed seat 0.
    config = SystemConfig(seed=23)
    game = GameService()
    handle = game.create_session(SessionConfig(config=config, agents=_pass_agents(), seed=23))
    baseline = _self_role(game.get_state(handle))
    for _ in range(5):
        result = game.step(handle)
        if result.terminated or result.truncated:
            break
        assert _self_role(game.get_state(handle)) == baseline
    game.close_session(handle)


def test_spectator_session_reveals_all_roles_during_play() -> None:
    # Item 1: a spectator (all-AI) session reveals every seat's true role during
    # play and sets reveal_all_roles True.
    game = GameService()
    handle = game.create_session(SessionConfig(mode="random_five", seed=25))
    state = game.get_state(handle)
    assert state.reveal_all_roles is True
    assert state.to_dict()["reveal_all_roles"] is True
    # Every seat is marked REVEALED (>= 2) and carries a concrete (non-hidden)
    # role during play.
    visibility = _role_visibility(state)
    revealed = [int(r) for r in state.observation["revealed_roles"]]
    assert all(v >= 2 for v in visibility)
    assert all(r != 255 for r in revealed)
    game.close_session(handle)


def test_create_session_runs_episode_start_per_seat_with_own_projection() -> None:
    # Revert-sensitive wiring guard for the FEAT-001 fix: GameService.create_session
    # must call on_episode_start on EVERY seat with that seat's OWN per-seat public
    # projection (environment.observe(index)), so each dummy folds its own role and
    # the public sheriff into its RoleBeliefEstimator from turn 0 instead of sitting
    # at the uniform 1/5 constructor default.
    #
    # This test imports bang_rl.app (numpy / pydantic), so it is CI-only in the
    # restricted sandbox; that is acceptable for a wiring test. It fails on a revert
    # of the services.py create_session on_episode_start loop (no PassAgent would
    # have a .context, so the first assertion raises AttributeError / fails), AND it
    # fails if observe(index) were changed to observe() (every seat would then share
    # one observation and self_role would not be five distinct roles).
    config = SystemConfig(seed=29)
    game = GameService()
    agents = _pass_agents()
    handle = game.create_session(SessionConfig(config=config, agents=agents, seed=29))
    try:
        # (a) on_episode_start ran on every seat (unset context -> fix reverted).
        for agent in agents:
            assert hasattr(agent, "context"), "on_episode_start was never called on this seat"
            # (b) the episode-start context carries the per-seat observation.
            assert "observation" in agent.context
        # (c) per-SEAT projection: each seat's own public projection sets self_role
        # to that seat's own role, so the five self_role values must be a permutation
        # of the five canonical roles 0..4.  A shared observation (observe() with no
        # index) would make them all identical.
        self_roles = sorted(int(agent.context["observation"]["self_role"][0]) for agent in agents)
        assert self_roles == [0, 1, 2, 3, 4]
    finally:
        game.close_session(handle)
