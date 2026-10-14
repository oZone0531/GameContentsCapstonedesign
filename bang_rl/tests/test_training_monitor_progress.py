"""Pure, Qt-free tests for TrainingMonitorController run-in-progress state.

These tests import ONLY ``bang_rl.gui.training`` and ``bang_rl.gui.i18n`` so
they run in-sandbox without PyQt6, torch, numpy, or pydantic. They use
hand-rolled fakes for the service port and the progress DTO instead of the real
``bang_rl.app`` types (which pull in pydantic/torch).

They lock in the FEAT-002 behavior: once a run starts, the win-rate graph has a
visible seeded baseline point (so it is never empty until the first evaluation),
and a running progress is observable via ``snapshot.active`` / ``progress.status``
so the Qt screen can show a localized "Training in progress" indicator.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


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
# the controller take its ``else: request = setup`` branch, so start() never
# calls TrainingSetup.to_request() / ConcreteTrainingConfig(...), which would
# import bang_rl.app (pydantic/torch) and break in-sandbox. The injected fake
# service.start() ignores the request entirely.
_FAKE_REQUEST = SimpleNamespace(episodes=1, mode="rl")


class _RunningService:
    """Service that starts fine and reports a live running progress.

    ``progress()`` returns ``win_rate=None`` to prove the non-empty graph comes
    from the seeded baseline point, not from a real evaluation win-rate.
    """

    def __init__(self) -> None:
        self.handle = SimpleNamespace(job_id="job-run", snapshot=None)

    def start(self, request):  # noqa: ANN001 - test fake
        return self.handle

    def progress(self, handle):  # noqa: ANN001 - test fake
        return SimpleNamespace(
            status="running",
            win_rate=None,
            logs=(),
            error=None,
        )


def test_start_seeds_baseline_win_rate_point() -> None:
    controller = TrainingMonitorController(_RunningService())

    snapshot = controller.start(_FAKE_REQUEST)

    # The seeded baseline makes the graph non-empty from the start of the run,
    # even though no evaluation win-rate has arrived yet.
    assert snapshot.win_rate_points
    assert snapshot.win_rate_points[0] == 0.0


def test_running_progress_is_active() -> None:
    controller = TrainingMonitorController(_RunningService())

    controller.start(_FAKE_REQUEST)
    snapshot = controller.refresh()

    assert snapshot.progress is not None
    assert snapshot.progress.status == "running"
    assert snapshot.active is True
    # The seeded baseline point survives through a refresh with no real win-rate.
    assert snapshot.win_rate_points
    assert snapshot.win_rate_points[0] == 0.0


def test_in_progress_string_is_localized_both_locales() -> None:
    # The in-progress indicator the Qt screen renders must be localized in both
    # locales (no raw UI text outside i18n.py).
    original = i18n.current_language()
    try:
        i18n.set_language("en")
        en_text = tr("training.in_progress")
        i18n.set_language("ko")
        ko_text = tr("training.in_progress")
    finally:
        i18n.translator._language = original
    assert en_text
    assert ko_text
    assert en_text != ko_text
