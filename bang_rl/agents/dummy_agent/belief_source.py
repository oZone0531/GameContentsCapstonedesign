"""Where a dummy's role beliefs come from.

Every consumer of a dummy's role beliefs (intent policy, emotion formula,
faction strength) reads ``DummyAgent.role_beliefs``. That property starts from
the public rule-based ``RoleBeliefEstimator`` and, when a belief source is
attached, passes the matrix through it. This is the single place where a
different inference can be plugged in without touching the policy or the
emotion formula.

Sources
-------
``rule`` (no source attached)
    The public rule-based estimator, unchanged. Default everywhere.
``hint`` (:class:`HintBeliefSource`)
    Warm start that can be wrong. At episode start each dummy gets one hinted
    role per other seat: the true role with probability ``accuracy``, otherwise
    a wrong role. Beliefs are ``(1 - strength) * rule + strength * hint``.
    This raises the chance of leaning toward the right answer without making
    the dummy omniscient, so mistakes (and friendly fire) remain.
``oracle`` (:class:`OracleMixBeliefSource`)
    Mixes the exact true roles: ``(1 - alpha) * rule + alpha * truth``. Even a
    tiny alpha breaks every tie of the rule estimator toward the truth, so the
    dummies play as if they knew every role. Kept as an upper-bound reference.

Both warm starts read privileged roles. They are for data generation only and
must never be given to the main agent.
``learned`` (``learned:PATH`` or ``learned@WEIGHT:PATH``)
    The main agent's trained intent inference from a Phase-2 artifact,
    run on the dummy's own public observation (no privileged input). See
    ``bang_rl.agents.main_agent.dummy_belief_adapter``.

Belief matrices are ``(5, 5)``: row = seat, column = canonical role
(sheriff, deputy, outlaw_a, outlaw_b, traitor). Outlaw A and B are the same
role in play, so a true outlaw is represented with its mass split evenly over
the two outlaw columns; the source never leaks which outlaw label a seat has.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

PLAYER_COUNT = 5
ROLE_COUNT = 5
OUTLAW_COLUMNS = (2, 3)

__all__ = [
    "BeliefSource",
    "HintBeliefSource",
    "OracleMixBeliefSource",
    "describe_belief_spec",
    "make_belief_source",
    "parse_belief_spec",
    "truth_matrix",
]


@runtime_checkable
class BeliefSource(Protocol):
    """Transforms the rule-based belief matrix of one dummy."""

    name: str

    def on_episode_start(self, player_idx: int, observation: Any | None) -> None:
        ...

    def beliefs(self, rule_beliefs: np.ndarray, observation: Any | None) -> np.ndarray:
        """Return the ``(5, 5)`` belief matrix the dummy should act on."""
        ...

    def describe(self) -> dict[str, Any]:
        ...


def truth_matrix(roles: Sequence[int]) -> np.ndarray:
    """One-hot role matrix with outlaw mass split over both outlaw columns."""

    roles = [int(role) for role in roles]
    if len(roles) != PLAYER_COUNT or any(not 0 <= role < ROLE_COUNT for role in roles):
        raise ValueError("roles must hold five canonical role indices")
    matrix = np.zeros((PLAYER_COUNT, ROLE_COUNT), dtype=np.float64)
    for seat, role in enumerate(roles):
        if role in OUTLAW_COLUMNS:
            matrix[seat, list(OUTLAW_COLUMNS)] = 0.5
        else:
            matrix[seat, role] = 1.0
    return matrix


class OracleMixBeliefSource:
    """Warm-start beliefs: ``(1 - alpha) * rule + alpha * truth``.

    The dummy's own row is left to the rule estimator (it already knows its
    own role exactly). Data generation only: holds privileged roles.
    """

    name = "oracle"

    def __init__(self, roles: Sequence[int], alpha: float) -> None:
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self._truth = truth_matrix(roles)
        self.player_idx: int | None = None

    def on_episode_start(self, player_idx: int, observation: Any | None) -> None:
        self.player_idx = int(player_idx)

    def beliefs(self, rule_beliefs: np.ndarray, observation: Any | None) -> np.ndarray:
        rule = np.asarray(rule_beliefs, dtype=np.float64)
        if rule.shape != (PLAYER_COUNT, ROLE_COUNT):
            raise ValueError("rule beliefs must have shape (5, 5)")
        mixed = (1.0 - self.alpha) * rule + self.alpha * self._truth
        if self.player_idx is not None:
            mixed[self.player_idx] = rule[self.player_idx]
        totals = mixed.sum(axis=1, keepdims=True)
        mixed = np.divide(mixed, totals, out=np.full_like(mixed, 1.0 / ROLE_COUNT), where=totals > 0)
        return mixed.astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {"source": self.name, "alpha": self.alpha}


class HintBeliefSource:
    """Warm start from a hint that is right with probability ``accuracy``.

    For every other seat the dummy draws one hinted role at episode start:
    the true role with probability ``accuracy``, otherwise a uniformly chosen
    wrong role. The sheriff is public, so a sheriff seat is always hinted
    correctly and a wrong hint never names the sheriff. Outlaw A/B are one
    role, so a wrong hint never swaps one outlaw label for the other.
    Beliefs are ``(1 - strength) * rule + strength * hint``; the dummy's own
    row stays with the rule estimator.
    """

    name = "hint"
    # Semantic roles a wrong hint can name (sheriff excluded: it is public).
    _HIDDEN_ROLES = ("deputy", "outlaw", "traitor")
    _ROLE_ROWS = {
        "deputy": (0.0, 1.0, 0.0, 0.0, 0.0),
        "outlaw": (0.0, 0.0, 0.5, 0.5, 0.0),
        "traitor": (0.0, 0.0, 0.0, 0.0, 1.0),
    }

    def __init__(
        self,
        roles: Sequence[int],
        accuracy: float,
        *,
        strength: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        accuracy, strength = float(accuracy), float(strength)
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("hint accuracy must be in [0, 1]")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("hint strength must be in [0, 1]")
        self.accuracy = accuracy
        self.strength = strength
        self._roles = [int(role) for role in roles]
        self._truth = truth_matrix(self._roles)
        self._rng = rng if rng is not None else np.random.default_rng()
        self.player_idx: int | None = None
        self.hint = self._truth.copy()
        self.hint_correct = np.ones(PLAYER_COUNT, dtype=bool)

    @staticmethod
    def _semantic(role: int) -> str:
        return {0: "sheriff", 1: "deputy", 2: "outlaw", 3: "outlaw", 4: "traitor"}[int(role)]

    def on_episode_start(self, player_idx: int, observation: Any | None) -> None:
        self.player_idx = int(player_idx)
        self.hint = self._truth.copy()
        self.hint_correct = np.ones(PLAYER_COUNT, dtype=bool)
        for seat, role in enumerate(self._roles):
            if seat == self.player_idx or role == 0:
                continue
            if self._rng.random() < self.accuracy:
                continue
            true_name = self._semantic(role)
            wrong = [name for name in self._HIDDEN_ROLES if name != true_name]
            name = wrong[int(self._rng.integers(len(wrong)))]
            self.hint[seat] = self._ROLE_ROWS[name]
            self.hint_correct[seat] = False

    def beliefs(self, rule_beliefs: np.ndarray, observation: Any | None) -> np.ndarray:
        rule = np.asarray(rule_beliefs, dtype=np.float64)
        if rule.shape != (PLAYER_COUNT, ROLE_COUNT):
            raise ValueError("rule beliefs must have shape (5, 5)")
        mixed = (1.0 - self.strength) * rule + self.strength * self.hint
        if self.player_idx is not None:
            mixed[self.player_idx] = rule[self.player_idx]
        # A role the rule estimator has ruled out (e.g. publicly revealed on
        # death) stays ruled out: public facts override the hint.
        known = rule.max(axis=1) >= 1.0 - 1e-6
        mixed[known] = rule[known]
        totals = mixed.sum(axis=1, keepdims=True)
        mixed = np.divide(mixed, totals, out=np.full_like(mixed, 1.0 / ROLE_COUNT), where=totals > 0)
        return mixed.astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {"source": self.name, "accuracy": self.accuracy, "strength": self.strength}


def _learned_parts(spec: str) -> tuple[float, str] | None:
    """``(weight, path)`` for ``learned:PATH`` / ``learned@W:PATH``, else None."""

    text = str(spec).strip()
    head, sep, path = text.partition(":")  # path may itself contain ':' (Windows drive)
    head = head.lower()
    if head != "learned" and not head.startswith("learned@"):
        return None
    if not sep or not path:
        raise ValueError("learned belief source needs a model path: 'learned:PATH'")
    weight = 1.0
    if "@" in head:
        try:
            weight = float(head.split("@", 1)[1])
        except ValueError as exc:
            raise ValueError(f"learned weight must be a number: {spec!r}") from exc
    if not 0.0 <= weight <= 1.0:
        raise ValueError("learned weight must be in [0, 1]")
    return weight, path


def describe_belief_spec(spec: str | None) -> str:
    """Short label for metadata, e.g. 'rule', 'hint:0.7:0.5', 'learned:1:model.phase2model'."""

    learned = None if spec is None else _learned_parts(spec)
    if learned is not None:
        from pathlib import Path

        return f"learned:{learned[0]:g}:{Path(learned[1]).name}"
    kind, params = parse_belief_spec(spec)
    return ":".join([kind, *(f"{value:g}" for value in params)])


def parse_belief_spec(spec: str | None) -> tuple[str, tuple[float, ...]]:
    """Parse a belief spec into ``(kind, params)``.

    ``"rule"`` -> ``("rule", ())``; ``"hint:Q"`` or ``"hint:Q:S"`` ->
    ``("hint", (accuracy, strength))`` (strength defaults to 0.5);
    ``"oracle:ALPHA"`` -> ``("oracle", (alpha,))``;
    ``"learned[@W]:PATH"`` -> ``("learned", (weight,))`` (path via
    :func:`describe_belief_spec` / :func:`make_belief_source`).
    """

    if spec is None or str(spec).strip().lower() in ("", "rule"):
        return "rule", ()
    learned = _learned_parts(spec)
    if learned is not None:
        return "learned", (learned[0],)
    parts = str(spec).strip().lower().split(":")
    kind, values = parts[0], parts[1:]
    try:
        numbers = tuple(float(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"belief source parameters must be numbers: {spec!r}") from exc
    if any(not 0.0 <= value <= 1.0 for value in numbers):
        raise ValueError("belief source parameters must be in [0, 1]")
    if kind == "hint" and len(numbers) in (1, 2):
        return "hint", (numbers[0], numbers[1] if len(numbers) == 2 else 0.5)
    if kind == "oracle" and len(numbers) == 1:
        return "oracle", numbers
    raise ValueError(
        f"unknown belief source {spec!r}; use 'rule', 'hint:Q[:S]', 'oracle:ALPHA' or 'learned:PATH'"
    )


def make_belief_source(
    spec: str | None, roles: Sequence[int], *, seed: int | None = None
) -> BeliefSource | None:
    """Belief source for one dummy from a spec string; ``None`` means rule."""

    kind, params = parse_belief_spec(spec)
    if kind == "rule":
        return None
    if kind == "learned":
        from pathlib import Path

        from bang_rl.agents.main_agent.dummy_belief_adapter import LearnedBeliefSource, load_learned_beliefs

        weight, path = _learned_parts(spec)
        return LearnedBeliefSource(load_learned_beliefs(path), weight=weight, label=Path(path).name)
    if kind == "hint":
        return HintBeliefSource(roles, params[0], strength=params[1], rng=np.random.default_rng(seed))
    return OracleMixBeliefSource(roles, params[0])
