"""Sandbox-executable tests for the Qt-free replay_store item-5/item-6 helpers.

``import bang_rl.gui.replay_store`` executes ``bang_rl/gui/__init__`` which pulls
in pydantic/PyQt6-backed modules absent in the restricted sandbox, so those
imports cannot be collected here. This module mirrors the importlib.util
path-loading pattern used by test_replay_log_row.py / test_spectator_roles.py to
load ``replay_store.py`` straight from its path (it imports only json / datetime
/ pathlib, so it path-loads cleanly without pydantic) and exercises
``build_replay_filename`` + ``describe_ai_versions`` directly.

The same assertions also live in test_replay_store.py for CI; this module exists
so the item-5/item-6 logic is actually runnable in the sandbox.
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


replay_store = _load_replay_store()


# -- item 5: build_replay_filename -----------------------------------------
def test_build_replay_filename_encodes_mode() -> None:
    name = replay_store.build_replay_filename("human_random")
    assert name.startswith("replay-human_random-")
    assert name.endswith(".json")
    assert "human_random" in name


def test_build_replay_filename_falls_back_when_mode_missing() -> None:
    for bad in (None, "", "   ", "///"):
        name = replay_store.build_replay_filename(bad)
        assert name.startswith("replay-")
        assert name.endswith(".json")
        assert bad not in ("///",) or "/" not in name


def test_build_replay_filename_sanitizes_mode_no_path_leak() -> None:
    name = replay_store.build_replay_filename("../../evil/mode.json")
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name
    assert name.endswith(".json")


def test_build_replay_filename_lowercases_and_keeps_session() -> None:
    name = replay_store.build_replay_filename("Human_Random", session="Sess 1!")
    assert "human_random" in name
    # Session is sanitized (spaces / punctuation dropped) and included.
    assert "Sess1" in name


# -- item 6: describe_ai_versions ------------------------------------------
def test_describe_ai_versions_random_five() -> None:
    entries = replay_store.describe_ai_versions(
        {str(seat): "random" for seat in range(5)}
    )
    assert len(entries) == 5
    assert all(kind == "random" and version is None for _t, _s, kind, version in entries)
    assert [seat for _t, seat, _k, _v in entries] == ["0", "1", "2", "3", "4"]


def test_describe_ai_versions_main_with_version_and_dummies() -> None:
    entries = replay_store.describe_ai_versions(
        {
            "0": "dummy",
            "1": {"kind": "main", "version": "1.0", "metadata": {"x": 1}},
            "2": "dummy",
            "3": "dummy",
            "4": "dummy",
        }
    )
    kinds = {seat: (kind, version) for _t, seat, kind, version in entries}
    assert kinds["1"] == ("main", "1.0")
    assert kinds["0"] == ("dummy", None)
    assert kinds["4"] == ("dummy", None)


def test_describe_ai_versions_main_without_version() -> None:
    entries = replay_store.describe_ai_versions({"0": {"kind": "main", "version": None}})
    assert entries == [("seat", "0", "main", None)]


def test_describe_ai_versions_legacy_and_empty_safe() -> None:
    assert replay_store.describe_ai_versions({}) == []
    assert replay_store.describe_ai_versions(None) == []
    entries = replay_store.describe_ai_versions({"0": 123, "1": {"kind": "wat"}})
    kinds = {seat: kind for _t, seat, kind, _v in entries}
    assert kinds["0"] == "unknown"
    assert kinds["1"] == "unknown"


def test_save_replay_auto_name_uses_record_mode(tmp_path) -> None:
    record = {"mode": "dummy_five", "observations": []}
    path = replay_store.save_replay(record, tmp_path)
    assert "dummy_five" in path.name
    assert path.exists()
    # An explicit filename overrides the auto mode name.
    path2 = replay_store.save_replay(record, tmp_path, filename="explicit.json")
    assert path2.name == "explicit.json"
