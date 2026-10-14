"""Pure, dependency-free tests for the collection step-cap seam (FEAT-001).

These tests prove the fix for PROBLEM A (pressing "AI 학습" -> 시작 logs only
``training_started`` then goes silent): the default training runner now emits an
early ``collection_started`` event, bounds each collection episode with a modest
per-episode step cap, and logs ``episode_finished`` per episode so a run is never
silent after ``training_started``.

They run in the restricted sandbox WITHOUT numpy / pydantic / torch / PyQt6.
``bang_rl.training.pipeline_runner`` imports those heavy dependencies LAZILY
inside the runner body, but a plain ``import bang_rl.training.pipeline_runner``
would execute the ``bang_rl.training`` package ``__init__`` which pulls torch /
numpy.  So we load the runner module by file path under a lightweight stub
``bang_rl`` / ``bang_rl.training`` package (mirroring the Qt-free path-load
pattern in ``test_training_monitor_progress.py`` and the source-grep pattern in
``test_training_runner_wiring.py``).  The module body itself only imports
``collections.abc`` / ``pathlib`` / ``typing`` at import time, so loading it this
way is cheap and dependency-free.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


_RUNNER_PATH = Path(__file__).parents[1] / "training" / "pipeline_runner.py"


def _load_runner_module() -> types.ModuleType:
    """Load pipeline_runner.py by file path under stub parent packages."""

    if "bang_rl" not in sys.modules:
        root = types.ModuleType("bang_rl")
        root.__path__ = [str(Path(__file__).parents[1])]
        sys.modules["bang_rl"] = root
    if "bang_rl.training" not in sys.modules:
        training = types.ModuleType("bang_rl.training")
        training.__path__ = [str(_RUNNER_PATH.parent)]
        sys.modules["bang_rl.training"] = training
    dotted = "bang_rl.training.pipeline_runner"
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


_runner = _load_runner_module()


# --- collection_step_cap ------------------------------------------------- #


def test_collection_step_cap_defaults_when_env_config_is_none() -> None:
    assert _runner.collection_step_cap(None) == _runner._COLLECTION_STEP_CAP


def test_collection_step_cap_defaults_when_no_override() -> None:
    env_config = SimpleNamespace()
    assert _runner.collection_step_cap(env_config) == _runner._COLLECTION_STEP_CAP


def test_collection_step_cap_honours_positive_override() -> None:
    env_config = SimpleNamespace(collection_step_cap=37)
    assert _runner.collection_step_cap(env_config) == 37


def test_collection_step_cap_ignores_non_positive_or_bool_override() -> None:
    assert _runner.collection_step_cap(SimpleNamespace(collection_step_cap=0)) == (
        _runner._COLLECTION_STEP_CAP
    )
    assert _runner.collection_step_cap(SimpleNamespace(collection_step_cap=-5)) == (
        _runner._COLLECTION_STEP_CAP
    )
    # bool is a subclass of int but must not be treated as a cap value.
    assert _runner.collection_step_cap(SimpleNamespace(collection_step_cap=True)) == (
        _runner._COLLECTION_STEP_CAP
    )


def test_collection_step_cap_clamps_to_env_max_episode_steps_when_smaller() -> None:
    env_config = SimpleNamespace(max_episode_steps=50)
    assert _runner.collection_step_cap(env_config) == 50


def test_collection_step_cap_does_not_grow_beyond_default_when_max_is_larger() -> None:
    env_config = SimpleNamespace(max_episode_steps=5000)
    assert _runner.collection_step_cap(env_config) == _runner._COLLECTION_STEP_CAP


def test_collection_step_cap_override_still_clamped_to_env_max() -> None:
    env_config = SimpleNamespace(collection_step_cap=400, max_episode_steps=120)
    assert _runner.collection_step_cap(env_config) == 120


def test_collection_step_cap_custom_default_argument_is_honoured() -> None:
    assert _runner.collection_step_cap(None, default=12) == 12


# --- should_stop_collection --------------------------------------------- #


def test_should_stop_collection_is_false_below_cap() -> None:
    assert _runner.should_stop_collection(0, 200) is False
    assert _runner.should_stop_collection(199, 200) is False


def test_should_stop_collection_is_true_at_and_after_cap() -> None:
    assert _runner.should_stop_collection(200, 200) is True
    assert _runner.should_stop_collection(201, 200) is True


def test_should_stop_collection_false_for_non_positive_cap() -> None:
    assert _runner.should_stop_collection(1000, 0) is False
    assert _runner.should_stop_collection(1000, -1) is False


# --- source-level assertions on the collection loop --------------------- #


def test_runner_emits_collection_started_before_the_loop() -> None:
    """collection_started is logged BEFORE the per-episode for-loop."""

    source = _RUNNER_PATH.read_text(encoding="utf-8")
    started_at = source.index('"collection_started"')
    loop_at = source.index("for episode_index in range(episode_budget):")
    assert started_at < loop_at, "collection_started must be logged before the collection loop"


def test_runner_emits_episode_finished_inside_the_loop() -> None:
    """episode_finished is logged inside the per-episode loop, before collection_finished."""

    source = _RUNNER_PATH.read_text(encoding="utf-8")
    loop_at = source.index("for episode_index in range(episode_budget):")
    # The log calls may be written across multiple lines, so match the quoted
    # event name itself rather than a single-line "log(\"...\"" prefix.
    episode_finished_at = source.index('"episode_finished"')
    collection_finished_at = source.index('"collection_finished"')
    assert loop_at < episode_finished_at < collection_finished_at


def test_runner_breaks_on_the_collection_step_cap() -> None:
    """The collection loop stops a passive standoff at the step cap."""

    source = _RUNNER_PATH.read_text(encoding="utf-8")
    assert "should_stop_collection(int(env.environment_step), step_cap)" in source
    assert "step_cap = collection_step_cap(env_config)" in source


def test_runner_module_defines_the_pure_helpers() -> None:
    """A parse-level guard that the Qt/torch-free helpers exist."""

    tree = ast.parse(_RUNNER_PATH.read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "collection_step_cap" in defined
    assert "should_stop_collection" in defined
