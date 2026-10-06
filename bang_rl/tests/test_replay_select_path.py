"""Headless regression test for the replay browser's row -> path seam.

This pins the item-10 delete/rename fix: the ReplayScreen resolves which row
the user acted on from the Qt selection, but the final "selected row + the
list_replays() paths -> the Path to act on" bounds mapping is factored into the
pure, Qt-free helper ``replay_store.select_replay_path`` so a stale / -1 /
out-of-range row can never make delete target the wrong (or a nonexistent)
file. Reverting that bounds mapping (so an out-of-range row returns a path
anyway) makes these tests fail.

Like ``test_replay_log_row.py`` this loads ``replay_store.py`` straight from its
path via importlib so it runs in the sandbox too: ``import
bang_rl.gui.replay_store`` would execute ``bang_rl/gui/__init__`` which pulls in
PyQt6 / pydantic (absent in the sandbox). ``replay_store`` itself imports no Qt
and no numpy.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_replay_store():
    path = Path(__file__).resolve().parents[1] / "gui" / "replay_store.py"
    name = "_replay_store_isolated"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_store = _load_replay_store()
select_replay_path = _store.select_replay_path


def _paths() -> list[Path]:
    return [Path("a.json"), Path("b.json"), Path("c.json")]


def test_valid_row_returns_the_matching_path() -> None:
    paths = _paths()
    assert select_replay_path(paths, 0) == Path("a.json")
    assert select_replay_path(paths, 1) == Path("b.json")
    assert select_replay_path(paths, 2) == Path("c.json")


def test_out_of_range_row_returns_none() -> None:
    paths = _paths()
    assert select_replay_path(paths, 3) is None
    assert select_replay_path(paths, 99) is None


def test_negative_row_returns_none() -> None:
    # ``currentRow()`` on a list with no selection returns -1; that must not
    # resolve to the last element (which would delete the wrong file).
    assert select_replay_path(_paths(), -1) is None


def test_empty_paths_returns_none_for_any_row() -> None:
    assert select_replay_path([], 0) is None
    assert select_replay_path([], -1) is None


def test_non_int_row_returns_none() -> None:
    assert select_replay_path(_paths(), None) is None
