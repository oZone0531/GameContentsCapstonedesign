"""Pure, dependency-free unit tests for the relationship-key derivation.

These tests exercise the pure helper functions ``_seat_index`` and
``_relationship_event_keys`` from
``bang_rl/agents/dummy_agent/emotion_module.py``.

Why the unusual loading strategy: ``emotion_module.py`` performs a top-level
``import numpy`` (and ``from bang_rl.config import EmotionConfig`` which pulls in
pydantic).  Those packages are intentionally NOT installable in the restricted
sandbox where this suite must run, so a plain ``import`` of the module would
fail before the helpers could be reached.  Both helpers, however, depend on
nothing but builtins.  We therefore parse the module source with :mod:`ast`,
extract ONLY those two function definitions, and ``exec`` them in an isolated
namespace.  This keeps the test genuinely runnable here (real local signal)
without importing numpy or pydantic, while still testing the exact source that
ships in the module.

This file is fully runnable both via ``pytest`` and directly with
``python bang_rl/tests/test_emotion_relationship_keys.py``.
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
_PURE_HELPERS = ("_seat_index", "_relationship_event_keys")


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
_seat_index = _HELPERS["_seat_index"]  # type: ignore[assignment]
_relationship_event_keys = _HELPERS["_relationship_event_keys"]  # type: ignore[assignment]


# --- _seat_index -----------------------------------------------------------


def test_seat_index_accepts_genuine_int() -> None:
    assert _seat_index(0) == 0
    assert _seat_index(3) == 3


def test_seat_index_rejects_bool_and_non_int() -> None:
    assert _seat_index(True) is None
    assert _seat_index(False) is None
    assert _seat_index(None) is None
    assert _seat_index("1") is None
    assert _seat_index(1.0) is None


# --- _relationship_event_keys ----------------------------------------------


def test_no_player_idx_yields_no_keys() -> None:
    assert _relationship_event_keys("bang", 1, 3, None) == ()


def test_actor_me_target_other() -> None:
    assert _relationship_event_keys("bang", 1, 3, 1) == ("bang_by_me", "bang_at_other")


def test_actor_other_target_me() -> None:
    assert _relationship_event_keys("bang", 2, 1, 1) == ("bang_by_other", "bang_at_me")


def test_actor_me_target_me_self_edge() -> None:
    assert _relationship_event_keys("bang", 1, 1, 1) == ("bang_by_me", "bang_at_me")


def test_role_revealed_no_actor() -> None:
    # actor_idx is None (role reveals have no actor) so no _by_* key is emitted.
    assert _relationship_event_keys("role_revealed", None, 1, 1) == (
        "role_revealed_at_me",
    )


def test_player_eliminated_by_me_at_other() -> None:
    assert _relationship_event_keys("player_eliminated", 1, 3, 1) == (
        "player_eliminated_by_me",
        "player_eliminated_at_other",
    )


def test_actor_before_target_ordering() -> None:
    # The actor-relative (_by_*) key must always precede the target-relative
    # (_at_*) key so it wins first-match precedence.
    keys = _relationship_event_keys("bang", 4, 2, 0)
    assert keys == ("bang_by_other", "bang_at_other")
    assert keys.index("bang_by_other") < keys.index("bang_at_other")


def test_bool_inputs_treated_as_unknown() -> None:
    # Booleans must never be coerced to seat 0/1; they are treated as unknown
    # for actor, target, and player_idx (and must not crash).
    assert _relationship_event_keys("bang", True, False, 1) == ()
    assert _relationship_event_keys("bang", 1, 3, True) == ()


def test_non_int_inputs_treated_as_unknown() -> None:
    # A known seat but string/None actor & target -> those sides are omitted.
    assert _relationship_event_keys("bang", "actor", None, 1) == ()
    assert _relationship_event_keys("bang", None, "target", 1) == ()
    # A string player_idx is unknown -> no relationship keys at all.
    assert _relationship_event_keys("bang", 1, 3, "1") == ()


def test_only_known_side_is_emitted() -> None:
    # Only the actor is a known int -> only the _by_* key.
    assert _relationship_event_keys("bang", 1, None, 1) == ("bang_by_me",)
    # Only the target is a known int -> only the _at_* key.
    assert _relationship_event_keys("bang", None, 2, 1) == ("bang_at_other",)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
