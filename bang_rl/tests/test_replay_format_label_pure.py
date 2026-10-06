"""Sandbox-executable tests for the Qt-free replay-format label seam (FEAT-001).

``import bang_rl.gui.replay_store`` executes ``bang_rl/gui/__init__`` which pulls
in pydantic/PyQt6-backed modules absent in the restricted sandbox, so those
imports cannot be collected here. This module mirrors the importlib.util
path-loading pattern used by test_replay_store_pure.py to load ``replay_store.py``
and the ``i18n._STRINGS`` dict straight from their paths (both import only the
stdlib so they path-load cleanly without pydantic / numpy / PyQt6).

It covers ``replay_format_label_tag`` for full / public / legacy records and
asserts the new ``replay.info_format`` i18n key exists in both languages with no
em dash (U+2014).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(filename: str, isolated_name: str):
    path = Path(__file__).resolve().parents[1] / "gui" / filename
    spec = importlib.util.spec_from_file_location(isolated_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[isolated_name] = module
    spec.loader.exec_module(module)
    return module


replay_store = _load_module("replay_store.py", "_replay_store_fmt_isolated")
i18n = _load_module("i18n.py", "_i18n_fmt_isolated")


# -- replay_format_label_tag ------------------------------------------------
def test_label_tag_full_record() -> None:
    assert replay_store.replay_format_label_tag({"replay_format": "full"}) == "replay.format_full"


def test_label_tag_public_record() -> None:
    assert replay_store.replay_format_label_tag({"replay_format": "public"}) == "replay.format_public"


def test_label_tag_legacy_record_defaults_to_public() -> None:
    # A legacy record written before the format tag existed has no key; it must
    # default to public, matching replay_summary's existing default.
    assert replay_store.replay_format_label_tag({}) == "replay.format_public"
    assert replay_store.replay_format_label_tag(None) == "replay.format_public"


def test_label_tag_accepts_summary() -> None:
    # replay_summary results carry the same replay_format key.
    full_summary = replay_store.replay_summary({"replay_format": "full", "states": [{}]})
    assert replay_store.replay_format_label_tag(full_summary) == "replay.format_full"
    public_summary = replay_store.replay_summary({"observations": []})
    assert replay_store.replay_format_label_tag(public_summary) == "replay.format_public"


def test_label_tag_in_all() -> None:
    assert "replay_format_label_tag" in replay_store.__all__


# -- i18n: replay.info_format ----------------------------------------------
def test_info_format_key_exists_in_both_languages() -> None:
    strings = i18n._STRINGS
    assert "replay.info_format" in strings
    entry = strings["replay.info_format"]
    assert "en" in entry and "ko" in entry
    assert "{format}" in entry["en"]
    assert "{format}" in entry["ko"]


def test_info_format_key_has_no_em_dash() -> None:
    entry = i18n._STRINGS["replay.info_format"]
    for lang in ("en", "ko"):
        assert "\u2014" not in entry[lang]
