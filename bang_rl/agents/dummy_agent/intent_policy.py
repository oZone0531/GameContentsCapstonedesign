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
from .role_belief import (
    DEPUTY_ROLE,
    OUTLAW_A_ROLE,
    OUTLAW_B_ROLE,
    PLAYER_COUNT,
    ROLE_COUNT,
    SHERIFF_ROLE,
    TRAITOR_ROLE,
    RoleBeliefEstimator,
)

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


class RoleGoal(str, Enum):
    """Ultimate role objective that owns the tactical Intent layer."""

    SHERIFF_ELIMINATE_ENEMIES = "sheriff_eliminate_enemies"
    DEPUTY_PROTECT_SHERIFF = "deputy_protect_sheriff"
    OUTLAW_ELIMINATE_SHERIFF = "outlaw_eliminate_sheriff"
    TRAITOR_REACH_DUEL_THEN_ELIMINATE_SHERIFF = "traitor_reach_duel_then_eliminate_sheriff"


class IntentCategory(str, Enum):
    """Stable, serializable high-level intent labels."""

    PASS = "pass"
    ATTACK_SHERIFF_TEAM = "attack_sheriff_team"
    ATTACK_OUTLAW_TEAM = "attack_outlaw_team"
    ATTACK_ENEMY = "attack_enemy"
    ATTACK_STRONG_FACTION = "attack_strong_faction"
    ATTACK_SHERIFF_DUEL = "attack_sheriff_duel"
    ATTACK_OUTLAW_SUSPECT = "attack_outlaw_suspect"
    ATTACK_DEPUTY_SUSPECT = "attack_deputy_suspect"


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
    relation_target_weight: float = 0.0
    role_target_weights: Mapping[str, float] = field(default_factory=dict)
    # v1 role-intent rules.  These are policy settings, not game rules.
    pass_confidence_score: float = 0.45
    outlaw_target_threshold: float = 0.60
    deputy_target_threshold: float = 0.40
    secondary_traitor_weight: float = 0.35
    finish_bonus: float = 0.10
    main_target_score: float = 1.0
    outlaw_pressure_per_survivor: float = 2.0
    role_mass_epsilon: float = 0.05
    # Role-inconsistent actions stay legal but receive a very low finite score.
    # In deterministic mode they normally lose unless Emotion overturns them;
    # in stochastic mode they retain a tiny non-zero probability.
    deputy_sheriff_intent_score: float = -5.0
    sheriff_known_deputy_intent_score: float = -5.0
    one_on_one_pass_score: float = -5.0

    # v5 goal-aware Intent weights.  The role goal owns these tactical scores;
    # Emotion is applied only after this module returns them.
    sheriff_outlaw_weight: float = 4.0
    sheriff_traitor_weight: float = 2.0
    sheriff_deputy_penalty: float = 4.0
    sheriff_pass_score: float = 1.5
    deputy_outlaw_weight: float = 4.0
    deputy_traitor_weight: float = 2.0
    deputy_pass_score: float = -1.0
    outlaw_sheriff_finish_score: float = 5.0
    outlaw_sheriff_setup_score: float = 1.0
    # Strategic identity-exposure cost for an Outlaw shooting the public Sheriff.
    # Opening combat carries the highest cost; once public combat has started the
    # cost falls, and after this Outlaw has already shot Sheriff it becomes zero.
    outlaw_identity_reveal_cost_opening: float = 1.0
    outlaw_identity_reveal_cost_after_sheriff_attacked: float = 0.3
    outlaw_deputy_weight: float = 4.0
    outlaw_ally_penalty: float = 3.0
    outlaw_pass_score: float = 0.0
    traitor_outlaw_weight: float = 5.0
    traitor_secondary_target_weight: float = 1.0
    traitor_deputy_weight: float = 5.0
    traitor_sheriff_avoid_score: float = -5.0
    traitor_duel_sheriff_score: float = 5.0
    traitor_pass_score: float = 0.0
    goal_finish_bonus: float = 1.0

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
            "relation_target_weight",
            "pass_confidence_score",
            "outlaw_target_threshold",
            "deputy_target_threshold",
            "secondary_traitor_weight",
            "finish_bonus",
            "main_target_score",
            "outlaw_pressure_per_survivor",
            "role_mass_epsilon",
            "deputy_sheriff_intent_score",
            "sheriff_known_deputy_intent_score",
            "one_on_one_pass_score",
            "sheriff_outlaw_weight",
            "sheriff_traitor_weight",
            "sheriff_deputy_penalty",
            "sheriff_pass_score",
            "deputy_outlaw_weight",
            "deputy_traitor_weight",
            "deputy_pass_score",
            "outlaw_sheriff_finish_score",
            "outlaw_sheriff_setup_score",
            "outlaw_identity_reveal_cost_opening",
            "outlaw_identity_reveal_cost_after_sheriff_attacked",
            "outlaw_deputy_weight",
            "outlaw_ally_penalty",
            "outlaw_pass_score",
            "traitor_outlaw_weight",
            "traitor_secondary_target_weight",
            "traitor_deputy_weight",
            "traitor_sheriff_avoid_score",
            "traitor_duel_sheriff_score",
            "traitor_pass_score",
            "goal_finish_bonus",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        weights = dict(self.role_target_weights)
        for role, weight in weights.items():
            if not isinstance(role, str) or not isinstance(weight, (int, float)) or not isfinite(float(weight)):
                raise ValueError("role_target_weights must contain finite numeric values")
        object.__setattr__(self, "role_target_weights", weights)
        for name in ("pass_confidence_score", "outlaw_target_threshold", "deputy_target_threshold", "role_mass_epsilon"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "secondary_traitor_weight",
            "finish_bonus",
            "main_target_score",
            "outlaw_pressure_per_survivor",
            "outlaw_identity_reveal_cost_opening",
            "outlaw_identity_reveal_cost_after_sheriff_attacked",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

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
        # v10 renamed the exposure-cost trigger: ordinary combat no longer
        # lowers the cost; only a public attack on Sheriff does.  Accept the
        # older key when loading a v9-style preset.
        if (
            "outlaw_identity_reveal_cost_after_combat" in values
            and "outlaw_identity_reveal_cost_after_sheriff_attacked" not in values
        ):
            values["outlaw_identity_reveal_cost_after_sheriff_attacked"] = values[
                "outlaw_identity_reveal_cost_after_combat"
            ]
        if (
            "traitor_deputy_secondary_weight" in values
            and "traitor_secondary_target_weight" not in values
        ):
            values["traitor_secondary_target_weight"] = values["traitor_deputy_secondary_weight"]
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
            "relation_target_weight": float(self.relation_target_weight),
            "role_target_weights": dict(sorted(self.role_target_weights.items())),
            "pass_confidence_score": float(self.pass_confidence_score),
            "outlaw_target_threshold": float(self.outlaw_target_threshold),
            "deputy_target_threshold": float(self.deputy_target_threshold),
            "secondary_traitor_weight": float(self.secondary_traitor_weight),
            "finish_bonus": float(self.finish_bonus),
            "main_target_score": float(self.main_target_score),
            "outlaw_pressure_per_survivor": float(self.outlaw_pressure_per_survivor),
            "role_mass_epsilon": float(self.role_mass_epsilon),
            "deputy_sheriff_intent_score": float(self.deputy_sheriff_intent_score),
            "sheriff_known_deputy_intent_score": float(self.sheriff_known_deputy_intent_score),
            "one_on_one_pass_score": float(self.one_on_one_pass_score),
            "sheriff_outlaw_weight": float(self.sheriff_outlaw_weight),
            "sheriff_traitor_weight": float(self.sheriff_traitor_weight),
            "sheriff_deputy_penalty": float(self.sheriff_deputy_penalty),
            "sheriff_pass_score": float(self.sheriff_pass_score),
            "deputy_outlaw_weight": float(self.deputy_outlaw_weight),
            "deputy_traitor_weight": float(self.deputy_traitor_weight),
            "deputy_pass_score": float(self.deputy_pass_score),
            "outlaw_sheriff_finish_score": float(self.outlaw_sheriff_finish_score),
            "outlaw_sheriff_setup_score": float(self.outlaw_sheriff_setup_score),
            "outlaw_identity_reveal_cost_opening": float(self.outlaw_identity_reveal_cost_opening),
            "outlaw_identity_reveal_cost_after_sheriff_attacked": float(self.outlaw_identity_reveal_cost_after_sheriff_attacked),
            "outlaw_deputy_weight": float(self.outlaw_deputy_weight),
            "outlaw_ally_penalty": float(self.outlaw_ally_penalty),
            "outlaw_pass_score": float(self.outlaw_pass_score),
            "traitor_outlaw_weight": float(self.traitor_outlaw_weight),
            "traitor_secondary_target_weight": float(self.traitor_secondary_target_weight),
            "traitor_deputy_weight": float(self.traitor_deputy_weight),
            "traitor_sheriff_avoid_score": float(self.traitor_sheriff_avoid_score),
            "traitor_duel_sheriff_score": float(self.traitor_duel_sheriff_score),
            "traitor_pass_score": float(self.traitor_pass_score),
            "goal_finish_bonus": float(self.goal_finish_bonus),
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
    ultimate_goal: RoleGoal | None = None
    goal_stage: str | None = None

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
            "ultimate_goal": None if self.ultimate_goal is None else self.ultimate_goal.value,
            "goal_stage": self.goal_stage,
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

    @classmethod
    def _alive_outlaw_count(cls, observation: Any, alive: np.ndarray) -> int:
        """Return the exact publicly knowable number of living Outlaws.

        The game always starts with exactly two Outlaws.  A dead player's role
        is public, so the count can be derived without using hidden role labels:
        ``2 - number of dead players publicly revealed as Outlaw``.  The
        viewer's own living role may also appear in ``revealed_roles``; checking
        ``alive`` prevents that private self-knowledge from being mistaken for
        an eliminated Outlaw.
        """

        revealed_values = cls._field(observation, "revealed_roles", default=None)
        if revealed_values is None:
            return 2
        revealed = np.asarray(revealed_values).reshape(-1)
        if revealed.shape != (PLAYER_COUNT,):
            raise ValueError("revealed_roles must contain five entries")
        dead_outlaws = 0
        for idx in range(PLAYER_COUNT):
            if bool(alive[idx]):
                continue
            try:
                role = int(revealed[idx])
            except (TypeError, ValueError):
                continue
            if role in (OUTLAW_A_ROLE, OUTLAW_B_ROLE):
                dead_outlaws += 1
        return max(0, 2 - dead_outlaws)

    @classmethod
    def _outlaw_reveal_context(
        cls,
        observation: Any,
        *,
        player_idx: int,
        sheriff_idx: int,
    ) -> tuple[int, bool]:
        """Return public Sheriff-attack count and whether this Outlaw shot Sheriff.

        IdentityRevealCost is a tactical Intent cost, not hidden information.
        It uses only PUBLIC Bang history.  Ordinary crossfire does not reduce
        the cost.  The cost falls only after Sheriff has actually been attacked,
        and becomes zero once this Outlaw has personally exposed itself by
        attacking Sheriff.
        """

        events = cls._field(observation, "event_sequence", "event_history", default=())
        if events is None:
            return 0, False
        try:
            iterator = iter(events)
        except TypeError:
            return 0, False

        sheriff_attack_count = 0
        self_attacked_sheriff = False
        for event in iterator:
            if isinstance(event, Mapping):
                value = event
            elif hasattr(event, "to_public_dict"):
                value = event.to_public_dict()
                if not isinstance(value, Mapping):
                    continue
            elif hasattr(event, "to_dict"):
                try:
                    value = event.to_dict(privileged=False)
                except TypeError:
                    value = event.to_dict()
                if not isinstance(value, Mapping):
                    continue
            else:
                continue

            event_type = str(
                getattr(value.get("event_type", ""), "value", value.get("event_type", ""))
            ).rsplit(".", 1)[-1].lower()
            if event_type != "bang":
                continue

            actor = value.get("actor_idx")
            target = value.get("target_idx")
            try:
                actor_value = int(np.asarray(actor).reshape(-1)[0])
                target_value = int(np.asarray(target).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                continue
            if target_value == sheriff_idx:
                sheriff_attack_count += 1
                if actor_value == player_idx:
                    self_attacked_sheriff = True

        return sheriff_attack_count, self_attacked_sheriff

    def update_config(self, config: IntentPolicyConfig | Mapping[str, Any] | Any) -> None:
        """Apply a new non-structural policy preset at the next decision."""

        self.config = IntentPolicyConfig.from_value(config)
        self.faction_estimator = FactionStrengthEstimator(tie_policy=self.config.tie_policy)

    def decision(self, observation: Any, beliefs: Any) -> IntentDecision:
        """Return goal-aware role Intent scores from public state and role beliefs.

        The decision is intentionally split into two conceptual layers.

        1. ``RoleGoal`` states what the role ultimately needs in order to win.
        2. Tactical Intent scores state which currently legal-looking action best
           advances that goal under the agent's *beliefs*, never true hidden roles.

        Emotion is not used here.  It is applied later by ``ActionSelector``.
        """

        matrix = FactionStrengthEstimator._normalise_beliefs(beliefs)
        board = self._alive_and_health(observation)
        if board is None or self.player_idx is None:
            decision = IntentDecision(
                IntentCategory.PASS,
                (self.config.pass_confidence_score,) + (0.0,) * 5,
                (0.0,) * 5,
                ultimate_goal=None,
                goal_stage="insufficient_public_state",
            )
            self.last_decision = decision
            return decision

        health, alive = board
        self_role = self._self_role(matrix)
        if self_role is None:
            decision = IntentDecision(
                IntentCategory.PASS,
                (self.config.pass_confidence_score,) + (0.0,) * 5,
                (0.0,) * 5,
                ultimate_goal=None,
                goal_stage="unknown_self_role",
            )
            self.last_decision = decision
            return decision

        sheriff_idx = int(np.asarray(self._field(observation, "sheriff_idx", default=0)).reshape(-1)[0])
        living = [idx for idx in range(PLAYER_COUNT) if bool(alive[idx])]
        opponents = [idx for idx in living if idx != self.player_idx]

        p_outlaw = matrix[:, OUTLAW_A_ROLE] + matrix[:, OUTLAW_B_ROLE]
        p_deputy = matrix[:, DEPUTY_ROLE]
        p_traitor = matrix[:, TRAITOR_ROLE]
        p_sheriff = matrix[:, SHERIFF_ROLE]

        alive_outlaw_count = self._alive_outlaw_count(observation, alive)
        # Publicly knowable living non-Outlaw count.  In the Outlaw policy this
        # is named AliveTeamCount by design: Sheriff + Deputy + Traitor who are
        # still alive.  Because the game always starts with exactly two
        # Outlaws and eliminated roles are public, this does not inspect hidden
        # role labels.
        alive_team_count = max(0, len(living) - alive_outlaw_count)
        estimated_alive_deputies = float(sum(float(p_deputy[idx]) for idx in living))
        sheriff_hp = float(health[sheriff_idx]) if bool(alive[sheriff_idx]) else 0.0

        target_scores = np.zeros(PLAYER_COUNT, dtype=np.float64)
        pass_score = float(self.config.pass_confidence_score)
        category = IntentCategory.PASS
        goal_stage = "unknown"
        duel = (
            self_role == TRAITOR_ROLE
            and len(living) == 2
            and self.player_idx in living
            and sheriff_idx in living
        )

        goals = {
            SHERIFF_ROLE: RoleGoal.SHERIFF_ELIMINATE_ENEMIES,
            DEPUTY_ROLE: RoleGoal.DEPUTY_PROTECT_SHERIFF,
            OUTLAW_A_ROLE: RoleGoal.OUTLAW_ELIMINATE_SHERIFF,
            OUTLAW_B_ROLE: RoleGoal.OUTLAW_ELIMINATE_SHERIFF,
            TRAITOR_ROLE: RoleGoal.TRAITOR_REACH_DUEL_THEN_ELIMINATE_SHERIFF,
        }
        ultimate_goal = goals[self_role]

        def finish_bonus(target: int) -> float:
            return self.config.goal_finish_bonus if int(health[target]) == 1 else 0.0

        def best_attack_score() -> float:
            return max((float(target_scores[t]) for t in opponents), default=-np.inf)

        # ------------------------------------------------------------------
        # SHERIFF
        # Ultimate goal: survive while eliminating both Outlaws and Traitor.
        # The Sheriff therefore values enemy probability, avoids likely Deputy,
        # and may wait when all targets remain too ambiguous.
        # ------------------------------------------------------------------
        if self_role == SHERIFF_ROLE:
            goal_stage = "identify_and_eliminate_enemies"
            pass_score = float(self.config.sheriff_pass_score)

            # If a non-terminal game has only Sheriff and one other player alive,
            # the remaining player should normally be an enemy.  Do not let an
            # equal PASS score stall the duel.  A nearly-certain Deputy is kept
            # as the exception because Sheriff+Deputy should normally have
            # already terminated as a Sheriff-team win.
            if len(living) == 2 and len(opponents) == 1:
                last_target = opponents[0]
                if float(p_deputy[last_target]) < 0.99:
                    goal_stage = "finish_last_enemy"
                    pass_score = float(self.config.one_on_one_pass_score)

            for target in opponents:
                score = (
                    self.config.sheriff_outlaw_weight * float(p_outlaw[target])
                    + self.config.sheriff_traitor_weight * float(p_traitor[target])
                    - self.config.sheriff_deputy_penalty * float(p_deputy[target])
                )
                target_scores[target] = score

            if best_attack_score() > pass_score:
                category = IntentCategory.ATTACK_OUTLAW_SUSPECT
            else:
                category = IntentCategory.PASS

        # ------------------------------------------------------------------
        # DEPUTY
        # Ultimate goal: keep Sheriff alive and remove threats to Sheriff.
        # Sheriff is still a legal target, but receives an extremely poor
        # role-intent score rather than a hard ban.
        # ------------------------------------------------------------------
        elif self_role == DEPUTY_ROLE:
            goal_stage = "protect_sheriff_and_remove_threats"
            pass_score = float(self.config.deputy_pass_score)
            for target in opponents:
                if target == sheriff_idx:
                    target_scores[target] = self.config.deputy_sheriff_intent_score
                    continue
                target_scores[target] = (
                    self.config.deputy_outlaw_weight * float(p_outlaw[target])
                    + self.config.deputy_traitor_weight * float(p_traitor[target])
                    + finish_bonus(target)
                )
            category = (
                IntentCategory.ATTACK_OUTLAW_SUSPECT
                if best_attack_score() > pass_score
                else IntentCategory.PASS
            )

        # ------------------------------------------------------------------
        # OUTLAW
        # Ultimate goal: eliminate Sheriff.  ``alive_outlaw_count`` starts at
        # two and decreases only when an eliminated Outlaw is publicly revealed.
        # This keeps the pressure rule based on public facts instead of hidden
        # roles or belief-mass estimates.
        # ------------------------------------------------------------------
        elif self_role in (OUTLAW_A_ROLE, OUTLAW_B_ROLE):
            pressure_limit = alive_outlaw_count * self.config.outlaw_pressure_per_survivor
            deputy_gone = estimated_alive_deputies <= self.config.role_mass_epsilon
            numerical_advantage = alive_outlaw_count >= alive_team_count
            can_finish_sheriff = bool(alive[sheriff_idx]) and (
                numerical_advantage
                or sheriff_hp <= pressure_limit
                or deputy_gone
            )

            if can_finish_sheriff:
                goal_stage = "finish_sheriff"
                pass_score = float(self.config.outlaw_pass_score)
                if sheriff_idx != self.player_idx and bool(alive[sheriff_idx]):
                    target_scores[sheriff_idx] = self.config.outlaw_sheriff_finish_score
                for target in opponents:
                    if target == sheriff_idx:
                        continue
                    target_scores[target] = (
                        self.config.outlaw_deputy_weight * float(p_deputy[target])
                        - self.config.outlaw_ally_penalty * float(p_outlaw[target])
                        + finish_bonus(target)
                    )
                category = IntentCategory.ATTACK_SHERIFF_TEAM
            else:
                goal_stage = "weaken_sheriff_team_before_finish"
                pass_score = float(self.config.outlaw_pass_score)

                # Shooting the public Sheriff is strategically informative: at
                # the opening it effectively announces hostile alignment.  Keep
                # that as an Intent cost rather than randomizing the action.
                # Ordinary crossfire does NOT reduce the cost.  It falls only
                # after Sheriff has actually been attacked, and disappears once
                # this Outlaw has personally exposed itself by shooting Sheriff.
                sheriff_attack_count, self_attacked_sheriff = self._outlaw_reveal_context(
                    observation,
                    player_idx=self.player_idx,
                    sheriff_idx=sheriff_idx,
                )
                if self_attacked_sheriff:
                    identity_reveal_cost = 0.0
                elif sheriff_attack_count == 0:
                    identity_reveal_cost = float(self.config.outlaw_identity_reveal_cost_opening)
                else:
                    identity_reveal_cost = float(
                        self.config.outlaw_identity_reveal_cost_after_sheriff_attacked
                    )

                if sheriff_idx != self.player_idx and bool(alive[sheriff_idx]):
                    target_scores[sheriff_idx] = (
                        self.config.outlaw_sheriff_setup_score - identity_reveal_cost
                    )
                for target in opponents:
                    if target == sheriff_idx:
                        continue
                    target_scores[target] = (
                        self.config.outlaw_deputy_weight * float(p_deputy[target])
                        - self.config.outlaw_ally_penalty * float(p_outlaw[target])
                        + finish_bonus(target)
                    )
                best_non_sheriff = max(
                    (float(target_scores[t]) for t in opponents if t != sheriff_idx),
                    default=-np.inf,
                )
                category = (
                    IntentCategory.ATTACK_DEPUTY_SUSPECT
                    if best_non_sheriff > max(pass_score, float(target_scores[sheriff_idx]))
                    else IntentCategory.ATTACK_SHERIFF_TEAM
                )

        # ------------------------------------------------------------------
        # TRAITOR
        # Ultimate goal: preserve Sheriff until the final duel while keeping
        # Sheriff-team and Outlaw-team strength in balance.  If Outlaws are
        # stronger, pressure likely Outlaws.  If Sheriff+Deputy are stronger,
        # pressure the likely Deputy.  This avoids turning Traitor into a fixed
        # third member of Sheriff team.
        # ------------------------------------------------------------------
        else:
            pass_score = float(self.config.traitor_pass_score)
            strength_report = self.faction_estimator.estimate(observation, matrix)

            if duel:
                goal_stage = "final_sheriff_duel"
                pass_score = float(self.config.one_on_one_pass_score)
                if sheriff_idx != self.player_idx and bool(alive[sheriff_idx]):
                    target_scores[sheriff_idx] = self.config.traitor_duel_sheriff_score
                category = IntentCategory.ATTACK_SHERIFF_DUEL

            elif alive_outlaw_count == 0:
                # No Outlaws remain publicly alive: Deputy is now the only
                # non-Sheriff obstacle to the final duel.
                goal_stage = "remove_deputy_preserve_sheriff"
                for target in opponents:
                    if target == sheriff_idx:
                        target_scores[target] = self.config.traitor_sheriff_avoid_score
                        continue
                    target_scores[target] = (
                        self.config.traitor_deputy_weight * float(p_deputy[target])
                        + finish_bonus(target)
                    )
                category = (
                    IntentCategory.ATTACK_DEPUTY_SUSPECT
                    if best_attack_score() > pass_score
                    else IntentCategory.PASS
                )

            elif strength_report.strong_faction == OUTLAW_FACTION:
                goal_stage = "balance_against_outlaw_strength"
                for target in opponents:
                    if target == sheriff_idx:
                        target_scores[target] = self.config.traitor_sheriff_avoid_score
                        continue
                    target_scores[target] = (
                        self.config.traitor_outlaw_weight * float(p_outlaw[target])
                        + self.config.traitor_secondary_target_weight * float(p_deputy[target])
                        + finish_bonus(target)
                    )
                category = (
                    IntentCategory.ATTACK_OUTLAW_SUSPECT
                    if best_attack_score() > pass_score
                    else IntentCategory.PASS
                )

            else:
                goal_stage = "balance_against_sheriff_strength"
                for target in opponents:
                    if target == sheriff_idx:
                        target_scores[target] = self.config.traitor_sheriff_avoid_score
                        continue
                    target_scores[target] = (
                        self.config.traitor_deputy_weight * float(p_deputy[target])
                        + self.config.traitor_secondary_target_weight * float(p_outlaw[target])
                        + finish_bonus(target)
                    )
                category = (
                    IntentCategory.ATTACK_DEPUTY_SUSPECT
                    if best_attack_score() > pass_score
                    else IntentCategory.PASS
                )

        action_scores = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        action_scores[0] = pass_score
        action_scores[1:] = target_scores

        report = self.faction_estimator.estimate(observation, matrix)
        decision = IntentDecision(
            category=category,
            action_scores=tuple(float(value) for value in action_scores),
            target_scores=tuple(float(value) for value in target_scores),
            strong_faction=report.strong_faction if self_role == TRAITOR_ROLE else None,
            one_on_one_sheriff_duel=bool(duel),
            ultimate_goal=ultimate_goal,
            goal_stage=goal_stage,
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
                ultimate_goal=(
                    None
                    if value.get("ultimate_goal") is None
                    else RoleGoal(value["ultimate_goal"])
                ),
                goal_stage=value.get("goal_stage"),
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
    "RoleGoal",
    "IntentDecision",
    "IntentModule",
    "IntentPolicy",
    "IntentPolicyConfig",
    "OUTLAW_FACTION",
    "POLICY_SCHEMA_VERSION",
    "SHERIFF_FACTION",
    "TRAITOR_FACTION",
]
