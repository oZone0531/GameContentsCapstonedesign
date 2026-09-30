"""Pure, dependency-free unit tests for the HP-context bucket helper.

These tests exercise the pure helper functions ``_seat_index``, ``_hp_number``
and ``_hp_event_bucket_keys`` from
``bang_rl/agents/dummy_agent/emotion_module.py``.

Why the unusual loading strategy: ``emotion_module.py`` performs a top-level
``import numpy`` (and ``from bang_rl.config import EmotionConfig`` which pulls in
pydantic).  Those packages are intentionally NOT installable in the restricted
sandbox where this suite must run, so a plain ``import`` of the module would
fail before the helpers could be reached.  The helpers, however, depend on
nothing but builtins.  We therefore parse the module source with :mod:`ast`,
extract ONLY those function definitions, and ``exec`` them in an isolated
namespace.  This mirrors ``test_emotion_relation_vector.py`` so the test is
genuinely runnable here (real local signal) without importing numpy or pydantic,
while still testing the exact source that ships in the module.

This file is fully runnable both via ``pytest`` and directly with
``python bang_rl/tests/test_emotion_hp_buckets.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "agents"
    / "dummy_agent"
    / "emotion_module.py"
)
_PURE_HELPERS = ("_seat_index", "_hp_number", "_hp_event_bucket_keys")


def _load_pure_helpers() -> dict[str, object]:
    """Extract the pure helper functions from the source without importing numpy."""

    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _PURE_HELPERS
    ]
    missing = set(_PURE_HELPERS) - {node.name for node in wanted}
    if missing:
        raise AssertionError(f"pure helpers not found in source: {sorted(missing)}")
    # ``from __future__ import annotations`` makes the annotations strings, so the
    # extracted defs never evaluate ``int | None`` at definition time; a plain
    # builtins-only namespace is therefore sufficient.
    namespace: dict[str, object] = {"__future__annotations__": True}
    module = ast.Module(body=wanted, type_ignores=[])
    exec(compile(module, filename=str(_MODULE_PATH), mode="exec"), namespace)
    return namespace


_HELPERS = _load_pure_helpers()
_hp_number = _HELPERS["_hp_number"]  # type: ignore[assignment]
_hp_event_bucket_keys = _HELPERS["_hp_event_bucket_keys"]  # type: ignore[assignment]


# --- _hp_number ------------------------------------------------------------


def test_hp_number_accepts_genuine_int() -> None:
    assert _hp_number(0) == 0
    assert _hp_number(4) == 4


def test_hp_number_rejects_bool_and_non_int() -> None:
    assert _hp_number(True) is None
    assert _hp_number(False) is None
    assert _hp_number(None) is None
    assert _hp_number("2") is None
    assert _hp_number(2.0) is None


# --- default (no thresholds configured) is a strict no-op ------------------


def test_no_thresholds_yields_no_keys() -> None:
    # Both thresholds None -> HP buckets never fire (byte-for-byte backward compat).
    assert (
        _hp_event_bucket_keys("bang", 0, 3, None, 1, None, None) == ()
    )
    assert (
        _hp_event_bucket_keys("bang", 1, 3, None, 1, None, None) == ()
    )


def test_unknown_hp_after_yields_no_keys() -> None:
    # hp_after unknown -> no bucket even when thresholds are configured.
    assert _hp_event_bucket_keys("bang", None, 3, None, 1, 1, 0) == ()
    assert _hp_event_bucket_keys("bang", True, 3, None, 1, 1, 0) == ()


def test_unknown_affected_seat_yields_no_keys() -> None:
    # No target and no eliminated_idx -> nothing to bucket.
    assert _hp_event_bucket_keys("bang", 0, None, None, 1, 1, 0) == ()


# --- target buckets (affected seat is a DIFFERENT seat) --------------------


def test_target_low_hp_bucket() -> None:
    # Seat 3 took damage to 1 HP; low<=1, lethal<=0 -> only low fires.
    assert _hp_event_bucket_keys("bang", 1, 3, None, 0, 1, 0) == ("target_low_hp",)


def test_target_lethal_and_low_ordering() -> None:
    # hp_after 0 <= lethal 0 and <= low 1 -> lethal precedes low.
    keys = _hp_event_bucket_keys("bang", 0, 3, None, 0, 1, 0)
    assert keys == ("target_lethal", "target_low_hp")
    assert keys.index("target_lethal") < keys.index("target_low_hp")


def test_target_full_hp_yields_no_keys() -> None:
    # 4 HP is above both thresholds.
    assert _hp_event_bucket_keys("bang", 4, 3, None, 0, 1, 0) == ()


# --- self buckets (affected seat is our OWN seat) --------------------------


def test_self_low_hp_bucket() -> None:
    # We (seat 1) are the target and dropped to 1 HP.
    assert _hp_event_bucket_keys("bang", 1, 1, None, 1, 1, 0) == ("self_low_hp",)


def test_self_lethal_bucket_uses_eliminated_idx_when_no_target() -> None:
    # No target seat, but eliminated_idx names our own seat with hp_after 0.
    keys = _hp_event_bucket_keys("player_eliminated", 0, None, 2, 2, 1, 0)
    assert keys == ("self_lethal", "self_low_hp")


def test_target_idx_takes_precedence_over_eliminated_idx() -> None:
    # When target_idx is present it is the affected seat, not eliminated_idx.
    assert _hp_event_bucket_keys("bang", 1, 3, 4, 0, 1, 0) == ("target_low_hp",)


def test_only_lethal_configured_fires_only_lethal() -> None:
    # low_threshold None -> only the lethal bucket can fire.
    assert _hp_event_bucket_keys("bang", 0, 3, None, 0, None, 0) == ("target_lethal",)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
