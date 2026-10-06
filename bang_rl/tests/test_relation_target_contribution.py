"""Pure, dependency-free unit tests for the per-target relation term.

These tests exercise the pure helper ``_relation_target_contribution`` from
``bang_rl/agents/dummy_agent/dummy_agent.py`` (added in FEAT-004).

Why the unusual loading strategy: ``dummy_agent.py`` performs a top-level
``import numpy`` and imports ``EmotionModule`` / ``IntentPolicyConfig`` (which
pull in numpy and pydantic).  Those packages are intentionally NOT installable
in the restricted sandbox where this suite must run, so a plain ``import`` of
the module would fail before the helper could be reached.  The helper, however,
depends on nothing but builtins.  We therefore parse the module source with
:mod:`ast`, extract ONLY that function definition, and ``exec`` it in an
isolated namespace.  This mirrors ``test_emotion_relation_vector.py`` so the
test is genuinely runnable here (real local signal) without importing numpy or
pydantic, while still testing the exact source that ships in the module.

This file is fully runnable both via ``pytest`` and directly with
``python bang_rl/tests/test_relation_target_contribution.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "agents"
    / "dummy_agent"
    / "dummy_agent.py"
)
_PURE_HELPERS = ("_relation_target_contribution",)


def _load_pure_helpers() -> dict[str, object]:
    """Extract the pure helper function from the source without importing numpy."""

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
    namespace: dict[str, object] = {}
    module = ast.Module(body=wanted, type_ignores=[])
    exec(compile(module, filename=str(_MODULE_PATH), mode="exec"), namespace)
    return namespace


_HELPERS = _load_pure_helpers()
_relation_target_contribution = _HELPERS["_relation_target_contribution"]  # type: ignore[assignment]


# --- weight 0 is an exact no-op -------------------------------------------


def test_zero_weight_is_all_zero() -> None:
    relation = [-1.0, 0.5, -0.25, 1.0, -0.75]
    assert _relation_target_contribution(relation, 0.0, 0) == [0.0] * 6


def test_zero_weight_ignores_relation_entirely() -> None:
    # Even an extreme relation vector produces an exact zero vector at weight 0.
    relation = [1.0, -1.0, 1.0, -1.0, 1.0]
    assert _relation_target_contribution(relation, 0.0, 2) == [0.0] * 6


# --- pass action never gets a term ----------------------------------------


def test_pass_action_index_zero_never_gets_a_term() -> None:
    relation = [-1.0, -1.0, -1.0, -1.0, -1.0]
    result = _relation_target_contribution(relation, 1.0, None)
    assert result[0] == 0.0


# --- sign convention: grudge (negative) -> positive bump at that seat ------


def test_grudge_toward_seat_raises_that_bang_score() -> None:
    # Seat 3 is resented (negative relation); with a positive weight the bang
    # action for seat 3 (action index 4) must get a POSITIVE contribution.
    relation = [0.0, 0.0, 0.0, -0.5, 0.0]
    result = _relation_target_contribution(relation, 2.0, 0)
    # term = -weight * relation[s] = -2.0 * -0.5 = 1.0 at action index 4.
    assert result == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_trust_toward_seat_lowers_that_bang_score() -> None:
    # Positive (trust) relation lowers the bang score for that seat.
    relation = [0.0, 0.8, 0.0, 0.0, 0.0]
    result = _relation_target_contribution(relation, 1.5, 0)
    # action index 2 == bang seat 1: -1.5 * 0.8 = -1.2
    assert result[2] == pytest.approx(-1.2)
    assert result[0] == 0.0


def test_action_to_seat_mapping_is_offset_by_one() -> None:
    # relation[s] maps to action index s + 1 for every non-self seat.
    relation = [-0.1, -0.2, -0.3, -0.4, -0.5]
    result = _relation_target_contribution(relation, 1.0, None)
    # -weight * relation[s] with weight 1.0 -> +0.1, +0.2, ...
    assert result == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


# --- self slot is skipped defensively -------------------------------------


def test_self_slot_gets_no_term() -> None:
    # Even if the self slot somehow held a non-neutral relation value, its bang
    # action must stay at exactly 0.0.
    relation = [0.0, 0.0, -1.0, 0.0, 0.0]
    result = _relation_target_contribution(relation, 3.0, 2)
    # seat 2 is our own seat -> action index 3 stays 0.0
    assert result[3] == 0.0


def test_none_player_idx_applies_all_five_seats() -> None:
    relation = [-1.0, -1.0, -1.0, -1.0, -1.0]
    result = _relation_target_contribution(relation, 1.0, None)
    # No seat is skipped; every bang index gets a positive bump.
    assert result == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
