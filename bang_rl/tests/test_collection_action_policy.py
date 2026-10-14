"""Pure tests for the collection action-selection helper (FEAT-001).

These tests prove that ``select_collection_action`` in
``bang_rl/training/pipeline_runner.py`` prefers a non-PASS legal action over
PASS so DummyAgent-driven collection reaches a win condition and terminates
'normal' instead of every seat PASSing forever (which only ever truncated and
produced zero normal episodes, faulting the run with the downstream DataError).

They run in the restricted sandbox: NONE of numpy / pydantic / torch is
imported.  Because importing through ``bang_rl.training`` would pull those heavy
deps (the runner imports them lazily on purpose, but the package __init__ does
not), the helper is loaded directly from the module's filesystem path via
importlib, mirroring the source-inspection pattern already used by
``test_training_runner_wiring.py`` test C.  The helper itself operates on plain
Python ints only, so it is importable and callable without numpy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_pipeline_runner_module() -> ModuleType:
    """Load pipeline_runner.py directly from its path (no heavy imports).

    The module's top-level imports are stdlib-only (collections.abc, pathlib,
    typing); numpy / torch / DummyAgent are imported lazily inside the runner
    body, so loading the module here does not require those packages.
    """

    path = Path(__file__).parents[1] / "training" / "pipeline_runner.py"
    spec = importlib.util.spec_from_file_location("_pipeline_runner_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_pipeline_runner_module()
select_collection_action = _MODULE.select_collection_action


def test_prefers_non_pass_legal_action_when_no_agent_action() -> None:
    # mask: PASS legal plus BANG targets at indices 2..5; must NOT pick PASS (0).
    action = select_collection_action([1, 0, 1, 1, 1, 1])
    assert action != 0
    assert action == 2


def test_honours_a_legal_agent_action_verbatim() -> None:
    # Index 3 is legal and chosen by the agent; return it unchanged even though
    # index 2 is also a legal non-PASS action.
    action = select_collection_action([1, 0, 1, 1, 1, 1], agent_action=3)
    assert action == 3


def test_illegal_agent_action_falls_back_to_non_pass() -> None:
    # Index 1 is NOT legal in this mask, so the illegal agent action is ignored
    # and the helper falls back to the first non-PASS legal action (index 2).
    action = select_collection_action([1, 0, 1, 1, 1, 1], agent_action=1)
    assert action == 2


def test_pass_only_mask_returns_pass() -> None:
    # Only PASS is legal; there is no non-PASS option, so 0 is returned.
    assert select_collection_action([1, 0, 0, 0, 0, 0]) == 0


def test_empty_all_zero_mask_returns_zero() -> None:
    # No legal action at all; the helper returns 0 defensively.
    assert select_collection_action([0, 0, 0, 0, 0, 0]) == 0


def test_agent_action_out_of_range_falls_back() -> None:
    # An out-of-range agent action index is treated as illegal and ignored.
    action = select_collection_action([1, 0, 1, 0, 0, 0], agent_action=99)
    assert action == 2
