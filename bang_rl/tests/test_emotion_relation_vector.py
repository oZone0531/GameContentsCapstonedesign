"""Pure, dependency-free unit tests for the relation-implication helper.

These tests exercise the pure helper functions ``_seat_index`` and
``_relation_event_seats`` from
``bang_rl/agents/dummy_agent/emotion_module.py``.

Why the unusual loading strategy: ``emotion_module.py`` performs a top-level
``import numpy`` (and ``from bang_rl.config import EmotionConfig`` which pulls in
pydantic).  Those packages are intentionally NOT installable in the restricted
sandbox where this suite must run, so a plain ``import`` of the module would
fail before the helpers could be reached.  The helpers, however, depend on
nothing but builtins.  We therefore parse the module source with :mod:`ast`,
extract ONLY those function definitions, and ``exec`` them in an isolated
namespace.  This mirrors ``test_emotion_relationship_keys.py`` so the test is
genuinely runnable here (real local signal) without importing numpy or pydantic,
while still testing the exact source that ships in the module.

This file is fully runnable both via ``pytest`` and directly with
``python bang_rl/tests/test_emotion_relation_vector.py``.
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
_PURE_HELPERS = ("_seat_index", "_relation_event_seats")


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
_relation_event_seats = _HELPERS["_relation_event_seats"]  # type: ignore[assignment]


# --- unknown / no-op cases -------------------------------------------------


def test_no_player_idx_yields_no_seats() -> None:
    # Without a known seat, relation must never be implicated.
    assert _relation_event_seats("bang", 2, 1, None) == ()
    assert _relation_event_seats("bang", 2, 1, True) == ()
    assert _relation_event_seats("bang", 2, 1, "1") == ()


def test_self_directed_event_implicates_no_other_seat() -> None:
    # actor == target == us implicates no other seat.
    assert _relation_event_seats("bang", 1, 1, 1) == ()


# --- "someone acted on me" -> "{kind}_by" toward that seat -----------------


def test_shot_by_other_seat_toward_actor() -> None:
    # Seat 3 banged us (seat 1) -> grudge toward seat 3, kind shot_by.
    assert _relation_event_seats("bang", 3, 1, 1) == ((3, "shot_by"),)


def test_eliminated_by_other_seat_toward_actor() -> None:
    assert _relation_event_seats("player_eliminated", 4, 0, 0) == ((4, "eliminated_by"),)


# --- "I acted on someone" -> "{kind}_at" toward that seat ------------------


def test_shot_at_other_seat_toward_target() -> None:
    # We (seat 1) banged seat 3 -> kind shot_at toward seat 3.
    assert _relation_event_seats("bang", 1, 3, 1) == ((3, "shot_at"),)


def test_eliminated_at_other_seat_toward_target() -> None:
    assert _relation_event_seats("player_eliminated", 2, 4, 2) == ((4, "eliminated_at"),)


# --- third-party interactions (neither side is us) -------------------------


def test_third_party_bang_implicates_both_other_seats() -> None:
    # Seat 4 banged seat 2, we are seat 0.  Both are "other".
    seats = _relation_event_seats("bang", 4, 2, 0)
    assert seats == ((4, "shot_by_other"), (2, "shot_at_other"))


def test_only_known_other_side_is_emitted() -> None:
    # Only the actor is a known other seat -> only the _by_other pair.
    assert _relation_event_seats("bang", 4, None, 0) == ((4, "shot_by_other"),)
    # Only the target is a known other seat -> only the _at_other pair.
    assert _relation_event_seats("bang", None, 2, 0) == ((2, "shot_at_other"),)


# --- kind spelling ---------------------------------------------------------


def test_unmapped_event_type_uses_its_own_spelling() -> None:
    assert _relation_event_seats("role_revealed", None, 3, 1) == ((3, "role_revealed_at_other"),)


def test_bool_indices_treated_as_unknown() -> None:
    # Booleans must not be coerced to seat 0/1.
    assert _relation_event_seats("bang", True, False, 1) == ()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
