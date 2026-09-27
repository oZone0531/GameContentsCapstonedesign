"""Public-observation-only faction strength and DummyAgent intent policy.

This module intentionally has no dependency on ``bang_rl.env``.  It consumes
only the canonical observation mapping and a RoleBeliefEstimator matrix.  The
policy is deterministic: masked actions are never considered and ties resolve
to the lowest action index.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite
from typing import Any, ClassVar

import numpy as np

from ..base import ACTION_SPACE_SIZE, validate_action_mask
from .role_belief import PLAYER_COUNT, ROLE_COUNT, RoleBeliefEstimator

SHERIFF_FACTION = "sheriff"
OUTLAW_FACTION = "outlaw"
TRAITOR_FACTION = "traitor"
POLICY_SCHEMA_VERSION = "1.0"

_ROLE_TO_FACTION = (
    SHERIFF_FACTION,
    SHERIFF_FACTION,
    OUTLAW_FACTION,
    OUTLAW_FACTION,
    TRAITOR_FACTION,
)


class IntentCategory(str, Enum):
    """Stable, serializable high-level intent labels."""

    PASS = "pass"
    ATTACK_SHERIFF_TEAM = "attack_sheriff_team"
    ATTACK_OUTLAW_TEAM = "attack_outlaw_team"
    ATTACK_ENEMY = "attack_enemy"
    ATTACK_STRONG_FACTION = "attack_strong_faction"
    ATTACK_SHERIFF_DUEL = "attack_sheriff_duel"


@dataclass(frozen=True)
class IntentPolicyConfig:
    """Configurable behavioral parameters for the deterministic intent policy.

    The mapping form is accepted as well, so a preset loaded from YAML/JSON
    can be passed directly to :class:`IntentModule`.  Values are policy
    weights, not environment rules; the common action mask remains authoritative.
    """

    schema_version: str = POLICY_SCHEMA_VERSION
    tie_policy: str = SHERIFF_FACTION
    pass_score: float = 0.0
    sheriff_team_attack_weight: float = 1.0
    outlaw_team_attack_weight: float = 1.0
    traitor_attack_weight: float = 1.0
    sheriff_direct_non_duel_weight: float = 0.25
    sheriff_direct_duel_weight: float = 2.0
    sheriff_role_bonus: float = 1.0
    deputy_role_bonus: float = 0.25
    target_health_weight: float = 0.0
    role_target_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("intent policy schema_version must not be blank")
        if self.tie_policy not in {SHERIFF_FACTION, OUTLAW_FACTION}:
            raise ValueError("tie_policy must be 'sheriff' or 'outlaw'")
        for name in (
            "pass_score",
            "sheriff_team_attack_weight",
            "outlaw_team_attack_weight",
            "traitor_attack_weight",
            "sheriff_direct_non_duel_weight",
            "sheriff_direct_duel_weight",
            "sheriff_role_bonus",
            "deputy_role_bonus",
            "target_health_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        weights = dict(self.role_target_weights)
        for role, weight in weights.items():
            if not isinstance(role, str) or not isinstance(weight, (int, float)) or not isfinite(float(weight)):
                raise ValueError("role_target_weights must contain finite numeric values")
        object.__setattr__(self, "role_target_weights", weights)

    @classmethod
    def from_value(cls, value: Any | None) -> "IntentPolicyConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="python")
        elif not isinstance(value, Mapping):
            value = {
                name: getattr(value, name)
                for name in cls.__dataclass_fields__
                if hasattr(value, name)
            }
        values = dict(value)
        # Accept the descriptive spelling used by some preset files.
        if "faction_tie_policy" in values and "tie_policy" not in values:
            values["tie_policy"] = values["faction_tie_policy"]
        # The Pydantic IntentPolicyConfig also contains architecture fields.
        # They are metadata for inference and not behavioral policy settings.
        accepted = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in values.items() if key in accepted})

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tie_policy": self.tie_policy,
            "pass_score": float(self.pass_score),
            "sheriff_team_attack_weight": float(self.sheriff_team_attack_weight),
            "outlaw_team_attack_weight": float(self.outlaw_team_attack_weight),
            "traitor_attack_weight": float(self.traitor_attack_weight),
            "sheriff_direct_non_duel_weight": float(self.sheriff_direct_non_duel_weight),
            "sheriff_direct_duel_weight": float(self.sheriff_direct_duel_weight),
            "sheriff_role_bonus": float(self.sheriff_role_bonus),
            "deputy_role_bonus": float(self.deputy_role_bonus),
            "target_health_weight": float(self.target_health_weight),
            "role_target_weights": dict(sorted(self.role_target_weights.items())),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "IntentPolicyConfig":
        if state.get("schema_version", POLICY_SCHEMA_VERSION) != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported intent policy schema")
        return cls.from_value(state)


@dataclass(frozen=True)
class FactionStrength:
    """Belief-weighted living count and public-health estimate for one faction."""

    faction: str
    survivor_count: float
    expected_health: float

    @property
    def estimated_survivor_count(self) -> float:
        return self.survivor_count

    @property
    def health(self) -> float:
        return self.expected_health

    def to_dict(self) -> dict[str, Any]:
        return {
            "faction": self.faction,
            "survivor_count": float(self.survivor_count),
            "expected_health": float(self.expected_health),
        }


@dataclass(frozen=True)
class FactionStrengthReport:
    """Comparison result derived from one public observation boundary."""

    factions: Mapping[str, FactionStrength]
    strong_faction: str
    tie_break_applied: bool = False

    def __getitem__(self, faction: str) -> FactionStrength:
        aliases = {
            "sheriff_team": SHERIFF_FACTION,
            "outlaw_team": OUTLAW_FACTION,
        }
        return self.factions[aliases.get(faction, faction)]

    def get(self, faction: str, default: Any = None) -> FactionStrength | Any:
        try:
            return self[faction]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "factions": {key: value.to_dict() for key, value in sorted(self.factions.items())},
            "strong_faction": self.strong_faction,
            "tie_break_applied": self.tie_break_applied,
        }


@dataclass(frozen=True)
class IntentDecision:
    """Deterministic policy output before ActionSelector chooses an index."""

    category: IntentCategory
    action_scores: tuple[float, ...]
    target_scores: tuple[float, ...]
    strong_faction: str | None = None
    one_on_one_sheriff_duel: bool = False

    @property
    def scores(self) -> np.ndarray:
        return np.asarray(self.action_scores, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "action_scores": list(self.action_scores),
            "target_scores": list(self.target_scores),
            "strong_faction": self.strong_faction,
            "one_on_one_sheriff_duel": self.one_on_one_sheriff_duel,
        }


ACTION_SELECTION_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ActionSelection:
    """Auditable result of combining intent, emotion, and the action mask.

    ``probabilities`` is the unmasked softmax of the combined scores and
    ``masked_probabilities`` is the same distribution after invalid actions
    have been removed and the remaining probability mass re-normalized.
    """

    action: int
    mode: str
    action_mask: tuple[int, ...]
    intent_scores: tuple[float, ...]
    emotion_contribution: tuple[float, ...]
    combined_scores: tuple[float, ...]
    masked_scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    masked_probabilities: tuple[float, ...]

    @property
    def selected_probability(self) -> float:
        return float(self.masked_probabilities[self.action])

    @property
    def final_scores(self) -> np.ndarray:
        return np.asarray(self.combined_scores, dtype=np.float64)

    @property
    def emotion_effects(self) -> np.ndarray:
        return np.asarray(self.emotion_contribution, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_SELECTION_SCHEMA_VERSION,
            "action": int(self.action),
            "mode": self.mode,
            "action_mask": list(self.action_mask),
            "intent_scores": list(self.intent_scores),
            "emotion_contribution": list(self.emotion_contribution),
            "emotion_effects": list(self.emotion_contribution),
            "combined_scores": list(self.combined_scores),
            "final_scores": list(self.combined_scores),
            "masked_scores": [
                None if not np.isfinite(value) else float(value) for value in self.masked_scores
            ],
            "probabilities": list(self.probabilities),
            "masked_probabilities": list(self.masked_probabilities),
            "selected_probability": self.selected_probability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionSelection":
        if value.get("schema_version", ACTION_SELECTION_SCHEMA_VERSION) != ACTION_SELECTION_SCHEMA_VERSION:
            raise ValueError("unsupported action selection schema")
        return cls(
            action=int(value["action"]),
            mode=str(value["mode"]),
            action_mask=tuple(int(item) for item in value["action_mask"]),
            intent_scores=tuple(float(item) for item in value["intent_scores"]),
            emotion_contribution=tuple(float(item) for item in value["emotion_contribution"]),
            combined_scores=tuple(float(item) for item in value["combined_scores"]),
            masked_scores=tuple(-np.inf if item is None else float(item) for item in value["masked_scores"]),
            probabilities=tuple(float(item) for item in value["probabilities"]),
            masked_probabilities=tuple(float(item) for item in value["masked_probabilities"]),
        )


class ActionSelector:
    """Combine intent/emotion scores and select a mask-valid action.

    Deterministic mode chooses the highest valid score and breaks ties by the
    lowest action index.  Stochastic mode samples the mask-renormalized
    softmax distribution using only the caller's private RNG.
    """

    schema_version: ClassVar[str] = ACTION_SELECTION_SCHEMA_VERSION
    MODES: ClassVar[frozenset[str]] = frozenset({"deterministic", "stochastic"})

    @classmethod
    def _mode(cls, mode: Any) -> str:
        if not isinstance(mode, str) or mode not in cls.MODES:
            raise ValueError("action mode must be 'deterministic' or 'stochastic'")
        return mode

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - float(np.max(values))
        exponentials = np.exp(shifted)
        total = float(exponentials.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("scores cannot produce a probability distribution")
        return exponentials / total

    @classmethod
    def select_with_info(
        cls,
        scores: Any,
        action_mask: Any,
        *,
        emotion_contribution: Any | None = None,
        mode: str = "deterministic",
        rng: np.random.Generator | None = None,
    ) -> ActionSelection:
        values = np.asarray(scores, dtype=np.float64)
        if values.shape != (ACTION_SPACE_SIZE,):
            raise ValueError(f"scores must have shape ({ACTION_SPACE_SIZE},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must contain finite values")
        if emotion_contribution is None:
            emotion = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        else:
            emotion = np.asarray(emotion_contribution, dtype=np.float64)
            if emotion.shape != (ACTION_SPACE_SIZE,):
                raise ValueError(f"emotion_contribution must have shape ({ACTION_SPACE_SIZE},)")
            if not np.all(np.isfinite(emotion)):
                raise ValueError("emotion_contribution must contain finite values")
        selected_mode = cls._mode(mode)
        mask = validate_action_mask(action_mask)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise ValueError("cannot select an action from an empty action mask")

        combined = values + emotion
        probabilities = cls._softmax(combined)
        masked = np.full(ACTION_SPACE_SIZE, -np.inf, dtype=np.float64)
        masked[valid] = combined[valid]
        masked_probabilities = probabilities * mask.astype(np.float64)
        mass = float(masked_probabilities.sum())
        if mass <= 0.0 or not np.isfinite(mass):
            # Recompute only over valid actions to remain stable for extreme
            # score gaps where an invalid action owns all unmasked mass.
            masked_probabilities[:] = 0.0
            masked_probabilities[valid] = cls._softmax(combined[valid])
        else:
            masked_probabilities /= mass

        if selected_mode == "deterministic":
            action = int(np.argmax(masked))
        else:
            if rng is None:
                rng = np.random.default_rng()
            if not isinstance(rng, np.random.Generator):
                raise TypeError("rng must be a numpy.random.Generator")
            action = int(rng.choice(valid, p=masked_probabilities[valid]))

        return ActionSelection(
            action=action,
            mode=selected_mode,
            action_mask=tuple(int(item) for item in mask),
            intent_scores=tuple(float(item) for item in values),
            emotion_contribution=tuple(float(item) for item in emotion),
            combined_scores=tuple(float(item) for item in combined),
            masked_scores=tuple(float(item) for item in masked),
            probabilities=tuple(float(item) for item in probabilities),
            masked_probabilities=tuple(float(item) for item in masked_probabilities),
        )

    @classmethod
    def select(
        cls,
        scores: Any,
        action_mask: Any,
        *,
        emotion_contribution: Any | None = None,
        mode: str = "deterministic",
        rng: np.random.Generator | None = None,
    ) -> int:
        return cls.select_with_info(
            scores,
            action_mask,
            emotion_contribution=emotion_contribution,
            mode=mode,
            rng=rng,
        ).action

    select_action = select


class FactionStrengthEstimator:
    """Compute faction strength from role beliefs and public board state."""

    schema_version: ClassVar[str] = POLICY_SCHEMA_VERSION

    def __init__(self, *, tie_policy: str = SHERIFF_FACTION) -> None:
        if tie_policy not in {SHERIFF_FACTION, OUTLAW_FACTION}:
            raise ValueError("tie_policy must be 'sheriff' or 'outlaw'")
        self.tie_policy = tie_policy

    @staticmethod
    def _field(observation: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(observation, Mapping) and name in observation:
                return observation[name]
            if hasattr(observation, name):
                return getattr(observation, name)
            if hasattr(observation, "get"):
                value = observation.get(name, None)
                if value is not None:
                    return value
        return default

    @classmethod
    def _board(cls, observation: Any) -> tuple[np.ndarray, np.ndarray]:
        health_values = cls._field(observation, "healths", "health", default=None)
        alive_values = cls._field(observation, "alive", default=None)
        if health_values is None or alive_values is None:
            raise ValueError("observation must include public healths and alive fields")
        health = np.asarray(health_values, dtype=np.float64).reshape(-1)
        alive = np.asarray(alive_values).reshape(-1)
        if health.shape != (PLAYER_COUNT,) or alive.shape != (PLAYER_COUNT,):
            raise ValueError("healths and alive must each contain five entries")
        if not np.all(np.isfinite(health)) or np.any(health < 0.0):
            raise ValueError("public healths must be finite and non-negative")
        if alive.dtype.kind not in "biu" or np.any((alive != 0) & (alive != 1)):
            raise ValueError("alive must contain binary values")
        return health, alive.astype(bool)

    @staticmethod
    def _normalise_beliefs(beliefs: Any) -> np.ndarray:
        values = np.asarray(beliefs, dtype=np.float64)
        if values.shape != (PLAYER_COUNT, ROLE_COUNT):
            raise ValueError("role beliefs must have shape (5, 5)")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("role beliefs must be finite and non-negative")
        row_sums = values.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError("each role-belief row must sum to one")
        return values

    def estimate(self, observation: Any, beliefs: Any) -> FactionStrengthReport:
        health, alive = self._board(observation)
        matrix = self._normalise_beliefs(beliefs)
        counts = {faction: 0.0 for faction in (SHERIFF_FACTION, OUTLAW_FACTION)}
        expected_health = {faction: 0.0 for faction in counts}
        for player_idx in range(PLAYER_COUNT):
            if not alive[player_idx]:
                continue
            for role, faction in enumerate(_ROLE_TO_FACTION):
                if faction in counts:
                    probability = float(matrix[player_idx, role])
                    counts[faction] += probability
                    expected_health[faction] += float(health[player_idx]) * probability
        sheriff = FactionStrength(SHERIFF_FACTION, counts[SHERIFF_FACTION], expected_health[SHERIFF_FACTION])
        outlaw = FactionStrength(OUTLAW_FACTION, counts[OUTLAW_FACTION], expected_health[OUTLAW_FACTION])
        count_delta = sheriff.survivor_count - outlaw.survivor_count
        if not np.isclose(count_delta, 0.0, atol=1e-7):
            strong = SHERIFF_FACTION if count_delta > 0 else OUTLAW_FACTION
            tied = False
        else:
            health_delta = sheriff.expected_health - outlaw.expected_health
            if not np.isclose(health_delta, 0.0, atol=1e-7):
                strong = SHERIFF_FACTION if health_delta > 0 else OUTLAW_FACTION
                tied = False
            else:
                strong = self.tie_policy
                tied = True
        return FactionStrengthReport(
            factions={SHERIFF_FACTION: sheriff, OUTLAW_FACTION: outlaw},
            strong_faction=strong,
            tie_break_applied=tied,
        )

    calculate = estimate
    evaluate = estimate


class IntentModule:
    """Generate role/faction-aware deterministic action priorities."""

    schema_version: ClassVar[str] = POLICY_SCHEMA_VERSION

    def __init__(
        self,
        *,
        player_idx: int | None = None,
        config: IntentPolicyConfig | Mapping[str, Any] | Any | None = None,
    ) -> None:
        if player_idx is not None and (isinstance(player_idx, bool) or not 0 <= int(player_idx) < PLAYER_COUNT):
            raise ValueError("player_idx must be in [0, 4] or None")
        self.player_idx = None if player_idx is None else int(player_idx)
        self.config = IntentPolicyConfig.from_value(config)
        self.faction_estimator = FactionStrengthEstimator(tie_policy=self.config.tie_policy)
        self.last_decision: IntentDecision | None = None

    @staticmethod
    def _role_name(role: int) -> str:
        return ("sheriff", "deputy", "outlaw_a", "outlaw_b", "traitor")[role]

    @staticmethod
    def _field(observation: Any, *names: str, default: Any = None) -> Any:
        return FactionStrengthEstimator._field(observation, *names, default=default)

    def _alive_and_health(self, observation: Any) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            return FactionStrengthEstimator._board(observation)
        except ValueError:
            # Legacy unit fixtures may only test role belief updates.  In that
            # case policy generation remains safely PASS-only until board state
            # is present, without inventing private information.
            return None

    def _self_role(self, beliefs: np.ndarray) -> int | None:
        if self.player_idx is None:
            return None
        row = beliefs[self.player_idx]
        if np.isclose(row.max(), 1.0, atol=1e-6) and np.count_nonzero(np.isclose(row, 1.0, atol=1e-6)) == 1:
            return int(np.argmax(row))
        return int(np.argmax(row))

    def update_config(self, config: IntentPolicyConfig | Mapping[str, Any] | Any) -> None:
        """Apply a new non-structural policy preset at the next decision."""

        self.config = IntentPolicyConfig.from_value(config)
        self.faction_estimator = FactionStrengthEstimator(tie_policy=self.config.tie_policy)

    def decision(self, observation: Any, beliefs: Any) -> IntentDecision:
        matrix = FactionStrengthEstimator._normalise_beliefs(beliefs)
        board = self._alive_and_health(observation)
        if board is None or self.player_idx is None:
            decision = IntentDecision(IntentCategory.PASS, (self.config.pass_score,) + (0.0,) * 5, (0.0,) * 5)
            self.last_decision = decision
            return decision
        health, alive = board
        self_role = self._self_role(matrix)
        if self_role is None:
            decision = IntentDecision(IntentCategory.PASS, (self.config.pass_score,) + (0.0,) * 5, (0.0,) * 5)
            self.last_decision = decision
            return decision

        report = self.faction_estimator.estimate(observation, matrix)
        living = [idx for idx in range(PLAYER_COUNT) if alive[idx]]
        duel = len(living) == 2 and alive[self.player_idx] and alive[int(np.asarray(self._field(observation, "sheriff_idx", default=0)).reshape(-1)[0])]
        sheriff_idx = int(np.asarray(self._field(observation, "sheriff_idx", default=0)).reshape(-1)[0])
        target_scores = np.zeros(PLAYER_COUNT, dtype=np.float64)
        category = IntentCategory.ATTACK_ENEMY

        if self_role in (0, 1):  # Sheriff team pursues Outlaws and Traitor.
            category = IntentCategory.ATTACK_ENEMY
            for target in living:
                if target == self.player_idx:
                    continue
                enemy_probability = float(matrix[target, 2] + matrix[target, 3] + matrix[target, 4])
                target_scores[target] = enemy_probability * self.config.sheriff_team_attack_weight
                target_scores[target] += float(matrix[target, 4]) * self.config.traitor_attack_weight
                target_scores[target] += float(matrix[target, 2] + matrix[target, 3]) * self.config.outlaw_team_attack_weight
                target_scores[target] += float(matrix[target, 0]) * self.config.sheriff_role_bonus
                target_scores[target] += float(matrix[target, 1]) * self.config.deputy_role_bonus
        elif self_role in (2, 3):  # Outlaws prioritize the Sheriff victory goal.
            category = IntentCategory.ATTACK_SHERIFF_TEAM
            for target in living:
                if target == self.player_idx:
                    continue
                target_scores[target] = float(matrix[target, 0]) * self.config.sheriff_role_bonus
                target_scores[target] += float(matrix[target, 1]) * self.config.deputy_role_bonus
                target_scores[target] *= self.config.outlaw_team_attack_weight
        else:  # Traitor attacks the currently stronger public-estimated faction.
            category = IntentCategory.ATTACK_STRONG_FACTION
            if duel and sheriff_idx != self.player_idx:
                category = IntentCategory.ATTACK_SHERIFF_DUEL
            for target in living:
                if target == self.player_idx:
                    continue
                sheriff_probability = float(matrix[target, 0] + matrix[target, 1])
                outlaw_probability = float(matrix[target, 2] + matrix[target, 3])
                target_faction_probability = sheriff_probability if report.strong_faction == SHERIFF_FACTION else outlaw_probability
                target_scores[target] = target_faction_probability * self.config.traitor_attack_weight
                target_scores[target] += float(matrix[target, 0]) * (
                    self.config.sheriff_direct_duel_weight if duel else self.config.sheriff_direct_non_duel_weight
                )
                target_scores[target] += float(matrix[target, 1]) * self.config.deputy_role_bonus

        for target in living:
            role_weight = self.config.role_target_weights.get(self._role_name(int(np.argmax(matrix[target]))), 0.0)
            target_scores[target] += role_weight
            target_scores[target] += self.config.target_health_weight * float(health[target])
        action_scores = np.full(ACTION_SPACE_SIZE, self.config.pass_score, dtype=np.float64)
        action_scores[1:] = target_scores
        decision = IntentDecision(
            category=category,
            action_scores=tuple(float(value) for value in action_scores),
            target_scores=tuple(float(value) for value in target_scores),
            strong_faction=report.strong_faction if self_role == 4 else None,
            one_on_one_sheriff_duel=bool(duel and self_role == 4),
        )
        self.last_decision = decision
        return decision

    generate = decision
    policy = decision

    def select_action(self, observation: Any, beliefs: Any, action_mask: Any) -> int:
        decision = self.decision(observation, beliefs)
        return ActionSelector.select(decision.scores, action_mask)

    act = select_action

    def faction_strength(self, observation: Any, beliefs: Any) -> FactionStrengthReport:
        return self.faction_estimator.estimate(observation, beliefs)

    faction_strengths = faction_strength

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "player_idx": self.player_idx,
            "config": self.config.state_dict(),
            "last_decision": None if self.last_decision is None else self.last_decision.to_dict(),
        }

    to_dict = state_dict

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "IntentModule":
        if state.get("schema_version", POLICY_SCHEMA_VERSION) != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported intent module state schema")
        module = cls(player_idx=state.get("player_idx"), config=state.get("config"))
        value = state.get("last_decision")
        if value is not None:
            module.last_decision = IntentDecision(
                category=IntentCategory(value["category"]),
                action_scores=tuple(float(item) for item in value["action_scores"]),
                target_scores=tuple(float(item) for item in value["target_scores"]),
                strong_faction=value.get("strong_faction"),
                one_on_one_sheriff_duel=bool(value.get("one_on_one_sheriff_duel", False)),
            )
        return module

    from_dict = from_state_dict


# Descriptive alias for integrations that call this layer an IntentPolicy.
IntentPolicy = IntentModule


__all__ = [
    "ACTION_SELECTION_SCHEMA_VERSION",
    "ActionSelection",
    "ActionSelector",
    "FactionStrength",
    "FactionStrengthEstimator",
    "FactionStrengthReport",
    "IntentCategory",
    "IntentDecision",
    "IntentModule",
    "IntentPolicy",
    "IntentPolicyConfig",
    "OUTLAW_FACTION",
    "POLICY_SCHEMA_VERSION",
    "SHERIFF_FACTION",
    "TRAITOR_FACTION",
]
