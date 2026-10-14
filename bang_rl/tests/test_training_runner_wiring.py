"""Pure wiring tests for the default training runner seam (FEAT-002).

These tests prove the *wiring* that makes pressing "AI 학습" actually run the
end-to-end pipeline, without exercising the real torch training path:

* Test A: when NO runner is injected, ``TrainingService.start`` resolves and
  runs ``bang_rl.training.pipeline_runner.default_training_runner`` (the default
  runner is engaged, and its sentinel result is folded into the report).
* Test B: when an explicit runner IS injected, that runner overrides the
  default (the injectable seam is intact).
* Test C: a dependency-free guard that the orchestration module exists, exports
  a callable default runner, and stays Qt-free.

Test C runs with NONE of numpy / pydantic / torch imported, so it executes in
the restricted sandbox.  Tests A and B construct a ``SystemConfig`` and thus
need pydantic; they are guarded with ``pytest.importorskip('pydantic')`` and are
CI-only in the sandbox.  The REAL torch end-to-end training is covered by
``test_pipeline_integration.py`` (CI-only); it is never run here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# --- Test C: pure, dependency-free -------------------------------------- #


def _runner_module_path() -> Path:
    return Path(__file__).parents[1] / "training" / "pipeline_runner.py"


def test_pipeline_runner_module_exists_is_callable_and_qt_free() -> None:
    """Revert-sensitive guard: the orchestration module exists and is Qt-free.

    This assertion must run in the restricted sandbox, so it inspects the module
    SOURCE directly via its filesystem path instead of importing it (importing
    through ``bang_rl.training`` would pull torch / numpy / PyYAML, none of which
    are installable here, and the runner imports those lazily on purpose).  It
    (1) confirms the module exists, (2) parses it and confirms it defines a
    ``default_training_runner`` function plus a ``build_default_training_runner``
    factory, and (3) greps the source for Qt bindings to keep the training layer
    free of cross-layer GUI imports.
    """

    import ast

    path = _runner_module_path()
    assert path.is_file(), "bang_rl/training/pipeline_runner.py must exist"

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "default_training_runner" in defined
    assert "build_default_training_runner" in defined

    lowered = source.lower()
    assert "pyqt" not in lowered
    assert "pyqtgraph" not in lowered


def test_training_service_start_imports_the_default_runner_symbol() -> None:
    """The monkeypatch target in Tests A/B matches the symbol start() imports.

    ``TrainingService.start`` does ``from bang_rl.training.pipeline_runner import
    default_training_runner`` at call time, so patching
    ``bang_rl.training.pipeline_runner.default_training_runner`` is the correct
    seam.  This static guard fails if someone renames that symbol or moves the
    import, which would silently break the default-runner wiring.
    """

    services_source = (Path(__file__).parents[1] / "app" / "services.py").read_text(encoding="utf-8")
    assert "from bang_rl.training.pipeline_runner import default_training_runner" in services_source


# --- Tests A/B: need pydantic-backed SystemConfig (CI-only in sandbox) --- #


def _wait_for(service, handle, timeout: float = 2.0):
    return service.wait(handle, timeout=timeout)


def test_default_runner_is_used_when_none_is_injected(monkeypatch, tmp_path: Path) -> None:
    """Test A: starting with NO runner engages the default pipeline runner."""

    pytest.importorskip("pydantic")
    pytest.importorskip("numpy")

    from bang_rl.app import TrainingConfig, TrainingService
    from bang_rl.config import SystemConfig
    import bang_rl.training.pipeline_runner as pipeline_runner

    calls: dict[str, object] = {}

    def fake_default_runner(handle, run):
        calls["handle"] = handle
        calls["run"] = run
        return {
            "environment_steps": 7,
            "main_agent_transitions": 3,
            "episodes_completed": 2,
            "win_rate": 0.5,
            "metrics": {"sentinel": "default-runner"},
            "artifact_path": str(tmp_path / "default.bangmodel"),
        }

    # start() imports default_training_runner from this module at call time.
    monkeypatch.setattr(pipeline_runner, "default_training_runner", fake_default_runner)

    service = TrainingService()
    handle = service.start(TrainingConfig(config=SystemConfig(), report_path=tmp_path / "report.json"))
    report = _wait_for(service, handle)

    assert "handle" in calls, "the default runner was never invoked"
    assert report.status == "finished"
    assert report.progress.environment_steps == 7
    assert report.progress.main_agent_transitions == 3
    assert report.progress.metrics["sentinel"] == "default-runner"


def test_injected_runner_overrides_the_default(monkeypatch, tmp_path: Path) -> None:
    """Test B: an explicit injected runner runs instead of the default."""

    pytest.importorskip("pydantic")
    pytest.importorskip("numpy")

    from bang_rl.app import TrainingConfig, TrainingService
    from bang_rl.config import SystemConfig
    import bang_rl.training.pipeline_runner as pipeline_runner

    default_ran = {"value": False}

    def fake_default_runner(handle, run):
        default_ran["value"] = True
        return {"metrics": {"sentinel": "default-runner"}}

    monkeypatch.setattr(pipeline_runner, "default_training_runner", fake_default_runner)

    injected_calls: dict[str, object] = {}

    def injected_runner(handle, run):
        injected_calls["ran"] = True
        return {"environment_steps": 11, "metrics": {"sentinel": "injected-runner"}}

    service = TrainingService()
    handle = service.start(
        TrainingConfig(config=SystemConfig(), runner=injected_runner, report_path=tmp_path / "report.json")
    )
    report = _wait_for(service, handle)

    assert injected_calls.get("ran") is True
    assert default_ran["value"] is False, "the default runner must NOT run when a runner is injected"
    assert report.status == "finished"
    assert report.progress.environment_steps == 11
    assert report.progress.metrics["sentinel"] == "injected-runner"


# --- Real end-to-end composition test (CI-only) ------------------------- #


def test_default_training_runner_runs_the_real_pipeline_end_to_end(tmp_path: Path) -> None:
    """CI-ONLY: execute the REAL ``default_training_runner`` body end to end.

    Every other test in this module fakes the runner or inspects its source, and
    ``test_pipeline_integration.py`` composes the stages inline but never calls
    the runner.  So the exact composition that must be correct for the "AI 학습"
    button to work had no executable coverage, which let a real-API mismatch
    (the observation-schema bug: feeding ``PublicEnvironmentState`` instead of
    its ``to_observation_dict()`` into collection, breaking the SL tensor cache)
    pass every in-repo test and only fail the first time a user pressed the
    button.

    This test closes that gap: it starts ``TrainingService`` with NO injected
    runner (so the real default runner is resolved and executed on the daemon
    thread) against the REAL stage components with a tiny budget, and asserts
    that the full collect -> split -> cache -> SL -> phase2 -> PPO -> artifact
    -> evaluation composition finishes and writes a ``.bangmodel``.  It is the
    regression guard for the real composition.

    CI-ONLY: it needs numpy / pydantic / torch, which are NOT installable in the
    restricted sandbox, so it SKIPS there.  The ``importorskip`` guards keep
    collection import-error-free in-sandbox.
    """

    pytest.importorskip("numpy")
    pytest.importorskip("pydantic")
    pytest.importorskip("torch")

    from bang_rl.app import TrainingConfig, TrainingService
    from bang_rl.config import SystemConfig

    service = TrainingService()
    handle = service.start(
        TrainingConfig(
            config=SystemConfig(),
            episodes=1,  # the runner floors this to its small collection budget
            report_path=tmp_path / "report.json",
            artifact_path=tmp_path / "model.bangmodel",
            resume_artifact_path=tmp_path / "no-resume",
        )
    )
    # The real pipeline (collection + SL + PPO + a short custom evaluation) can
    # take a while on CI, so allow a generous timeout.
    report = service.wait(handle, timeout=600.0)

    assert report.status == "finished", f"real pipeline did not finish: {report.status}"
    assert report.progress.environment_steps > 0
    assert report.progress.artifact_path is not None
    assert Path(report.progress.artifact_path).exists(), "a .bangmodel artifact must be written"
    # The SL stage only runs if the observation schema fed into collection was
    # the canonical to_observation_dict() shape; a schema mismatch would have
    # raised DataError and faulted the run before reaching PPO/artifact save.
    assert report.progress.metrics is not None
    # FEAT-001 regression guard: the per-seat DummyAgent collection policy must
    # produce at least one NORMAL (non-truncated) episode.  Before the fix the
    # collection loop hard-selected the first legal action (always PASS), so no
    # HP ever changed, every episode only truncated, and the tensor cache raised
    # DataError('validation/test cache requires normal training episodes for
    # statistics'), faulting the run with zero transitions.  A normal episode
    # here proves games actually play out and reach a win condition.
    assert report.progress.normal_episodes >= 1, (
        "collection must yield at least one normal (non-truncated) episode"
    )
