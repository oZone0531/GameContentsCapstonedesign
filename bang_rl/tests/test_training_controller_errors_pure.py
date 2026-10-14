"""Pure, Qt-free tests for TrainingMonitorController error surfacing.

These tests deliberately import ONLY ``bang_rl.gui.training`` and
``bang_rl.gui.i18n`` so they run in-sandbox without PyQt6, torch, numpy, or
pydantic. They use hand-rolled fakes for the service port and the progress DTO
instead of the real ``bang_rl.app`` types (which pull in pydantic/torch).

They lock in the FEAT-003 behavior: a start failure, a lifecycle failure, and a
faulted run each surface a NON-EMPTY, LOCALIZED message (never a raw English
exception string once the Korean locale is active), so the fix is
revert-sensitive without needing a display server.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


# ``bang_rl.gui.__init__`` eagerly imports MainWindow, which pulls in pydantic /
# PyQt6 and therefore cannot be imported in the restricted sandbox. The two
# modules under test (``bang_rl.gui.i18n`` and ``bang_rl.gui.training``) are both
# Qt-free, so we load them by file path under their real dotted names after
# inserting a lightweight stub for the ``bang_rl.gui`` package. That stub stops
# Python from executing the heavy package ``__init__`` while still letting the
# submodules' own ``from bang_rl.gui.i18n import tr`` resolve.
_GUI_DIR = Path(__file__).parents[1] / "gui"


def _load_gui_module(name: str) -> types.ModuleType:
    if "bang_rl.gui" not in sys.modules:
        stub = types.ModuleType("bang_rl.gui")
        stub.__path__ = [str(_GUI_DIR)]
        sys.modules["bang_rl.gui"] = stub
    dotted = f"bang_rl.gui.{name}"
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, _GUI_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


i18n = _load_gui_module("i18n")
tr = i18n.tr
TrainingMonitorController = _load_gui_module("training").TrainingMonitorController


# A bare request stand-in. Passing a non-Mapping, non-TrainingSetup object makes
# the controller take its `else: request = setup` branch, so start() never calls
# TrainingSetup.to_request() / ConcreteTrainingConfig(...), which would import
# bang_rl.app (pydantic/torch) and break in-sandbox.
_FAKE_REQUEST = SimpleNamespace(episodes=1, mode="rl")


@pytest.fixture(autouse=True)
def _restore_language() -> None:
    """Keep each test isolated from the persisted/active language choice."""

    original = i18n.current_language()
    try:
        yield
    finally:
        i18n.translator._language = original


class _RaisingStartService:
    """Service whose start() always raises a plain exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def start(self, request):  # noqa: ANN001 - test fake
        raise self._exc

    def progress(self, handle):  # pragma: no cover - never reached
        raise AssertionError("progress must not be called after a failed start")


class _FaultedProgressService:
    """Service that starts fine but reports a faulted (errored) progress."""

    def __init__(self, error, *, status: str = "faulted") -> None:  # noqa: ANN001 - test fake
        self._error = error
        self._status = status
        self.handle = SimpleNamespace(job_id="job-fault", snapshot=None)

    def start(self, request):  # noqa: ANN001 - test fake
        return self.handle

    def progress(self, handle):  # noqa: ANN001 - test fake
        return SimpleNamespace(
            status=self._status,
            win_rate=None,
            logs=(),
            error=self._error,
        )


def test_start_failure_publishes_localized_start_failed_message() -> None:
    i18n.set_language("en")
    detail = "runner is not wired to the pipeline"
    controller = TrainingMonitorController(_RaisingStartService(RuntimeError(detail)))

    snapshot = controller.start(_FAKE_REQUEST)

    assert controller.handle is None
    assert snapshot.message
    assert snapshot.message == tr("training.start_failed", detail=detail)
    assert detail in snapshot.message


def test_faulted_progress_publishes_localized_run_failed_message() -> None:
    i18n.set_language("en")
    detail = "exploded during rollout"
    service = _FaultedProgressService({"message": detail, "error_type": "RuntimeError"})
    controller = TrainingMonitorController(service)

    controller.start(_FAKE_REQUEST)
    snapshot = controller.refresh()

    assert snapshot.message
    assert snapshot.message == tr("training.run_failed", detail=detail)
    assert detail in snapshot.message


def test_faulted_status_without_error_payload_still_surfaces_run_failed() -> None:
    # Regression guard: a run can report status == "faulted" with an empty
    # error payload. The monitor must still show a NON-EMPTY localized
    # run_failed message instead of staying silent (the user-reported symptom:
    # the GUI showed no error when the run faulted). Revert-sensitive: if
    # _publish only keys off ``error`` this assertion fails because the fake
    # error below is falsy.
    i18n.set_language("en")
    service = _FaultedProgressService(None, status="faulted")
    controller = TrainingMonitorController(service)

    controller.start(_FAKE_REQUEST)
    snapshot = controller.refresh()

    assert snapshot.message
    assert snapshot.message == tr("training.run_failed", detail=tr("training.run_failed_unknown"))


def test_korean_locale_start_message_is_not_raw_english_exception() -> None:
    i18n.set_language("ko")
    detail = "default execution is not wired"
    controller = TrainingMonitorController(_RaisingStartService(RuntimeError(detail)))

    snapshot = controller.start(_FAKE_REQUEST)

    assert snapshot.message
    # The localized Korean prefix must be present and the message must not be
    # the bare English exception text (the revert-sensitive localization guard).
    assert snapshot.message != detail
    assert snapshot.message.startswith("학습을 시작할 수 없습니다")
    assert snapshot.message == tr("training.start_failed", detail=detail)


def test_korean_locale_run_failed_message_is_localized() -> None:
    i18n.set_language("ko")
    detail = "학습 중 예외"
    service = _FaultedProgressService(f"RuntimeError: {detail}")
    controller = TrainingMonitorController(service)

    controller.start(_FAKE_REQUEST)
    snapshot = controller.refresh()

    assert snapshot.message
    assert snapshot.message.startswith("학습이 실패했습니다")


def test_refresh_without_job_reports_localized_no_job_running() -> None:
    i18n.set_language("ko")
    controller = TrainingMonitorController(service=None)

    snapshot = controller.refresh()

    assert snapshot.message == tr("training.no_job_running")
    assert snapshot.message == "진행 중인 학습 작업이 없습니다."
