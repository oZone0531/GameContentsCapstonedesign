"""Validated configuration models for the Bang RL system.

The models in this module are the configuration boundary for environment,
training, reward, hardware, and artifact settings.  Runtime code should read
values from these models (or a job snapshot), rather than embedding policy or
reward constants in implementation code.
"""

from __future__ import annotations

from enum import Enum
from hashlib import sha256
from math import isfinite
import json
from typing import Annotated, Literal, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)


class ConfigModel(BaseModel):
    """Common validation policy for all persisted configuration objects."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class Role(str, Enum):
    SHERIFF = "sheriff"
    DEPUTY = "deputy"
    OUTLAW_A = "outlaw_a"
    OUTLAW_B = "outlaw_b"
    TRAITOR = "traitor"


class Team(str, Enum):
    SHERIFF_TEAM = "sheriff_team"
    OUTLAW_TEAM = "outlaw_team"
    TRAITOR_TEAM = "traitor_team"


class EventHistoryConfig(ConfigModel):
    """Public event history retention; ``None`` means unlimited."""

    max_events: int | None = None

    @field_validator("max_events", mode="before")
    @classmethod
    def validate_max_events(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("event_history.max_events must be null or a positive integer")
        return value


class EnvConfig(ConfigModel):
    """Environment execution and episode boundary settings."""

    max_episode_steps: StrictInt = Field(default=2000, ge=1)
    warn_turn_threshold: StrictInt = Field(default=200, ge=1)
    critical_turn_threshold: StrictInt = Field(default=500, ge=1)
    event_history: EventHistoryConfig = Field(default_factory=EventHistoryConfig)
    action_space_size: Literal[6] = 6

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "EnvConfig":
        if not (
            0 < self.warn_turn_threshold < self.critical_turn_threshold < self.max_episode_steps
        ):
            raise ValueError(
                "thresholds must satisfy "
                "0 < warn_turn_threshold < critical_turn_threshold < max_episode_steps"
            )
        return self


class TerminalResultCondition(ConfigModel):
    type: Literal["terminal_result"] = "terminal_result"
    result: Literal["win", "loss"]


class EventCondition(ConfigModel):
    type: Literal["event"] = "event"
    event: Literal["bang_hit", "kill_enemy", "pass"]


class RoleAttackCondition(ConfigModel):
    type: Literal["role_attack"] = "role_attack"
    actor_role: Role
    target_role: Role


class FactionAttackCondition(ConfigModel):
    type: Literal["faction_attack"] = "faction_attack"
    actor_faction: Team
    target_faction: Team


class PassCondition(ConfigModel):
    type: Literal["pass"] = "pass"


RewardCondition = Annotated[
    Union[
        TerminalResultCondition,
        EventCondition,
        RoleAttackCondition,
        FactionAttackCondition,
        PassCondition,
    ],
    Field(discriminator="type"),
]


class TerminalRewardComponent(ConfigModel):
    id: Literal["terminal_win", "terminal_loss"]
    enabled: StrictBool
    condition: TerminalResultCondition
    value: float

    @field_validator("value", mode="before")
    @classmethod
    def value_must_be_finite_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("reward component value must be a finite number")
        if not isfinite(float(value)):
            raise ValueError("reward component value must be a finite number")
        return value

    @model_validator(mode="after")
    def condition_matches_id(self) -> "TerminalRewardComponent":
        expected = "win" if self.id == "terminal_win" else "loss"
        if self.condition.result != expected:
            raise ValueError(f"{self.id} requires terminal_result result={expected!r}")
        return self


class ShapingRewardComponent(ConfigModel):
    id: Literal[
        "bang_hit",
        "kill_enemy",
        "attack_sheriff_outlaw",
        "attack_sheriff_deputy",
        "role_or_faction_attack_terms",
        "pass_penalty",
    ]
    enabled: StrictBool
    condition: RewardCondition
    value: float

    @field_validator("value", mode="before")
    @classmethod
    def value_must_be_finite_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("reward component value must be a finite number")
        if not isfinite(float(value)):
            raise ValueError("reward component value must be a finite number")
        return value

    @model_validator(mode="after")
    def condition_matches_id(self) -> "ShapingRewardComponent":
        condition_type = self.condition.type
        if self.id == "bang_hit" and not (
            condition_type == "event" and self.condition.event == "bang_hit"
        ):
            raise ValueError("bang_hit requires an event/bang_hit condition")
        if self.id == "kill_enemy" and not (
            condition_type == "event" and self.condition.event == "kill_enemy"
        ):
            raise ValueError("kill_enemy requires an event/kill_enemy condition")
        if self.id == "pass_penalty" and condition_type != "pass":
            raise ValueError("pass_penalty requires a pass condition")
        if self.id in {"attack_sheriff_outlaw", "attack_sheriff_deputy"}:
            if condition_type != "role_attack":
                raise ValueError(f"{self.id} requires a role_attack condition")
            if self.condition.target_role != Role.SHERIFF:
                raise ValueError(f"{self.id} requires target_role=sheriff")
            if self.id == "attack_sheriff_outlaw" and self.condition.actor_role not in {
                Role.OUTLAW_A,
                Role.OUTLAW_B,
            }:
                raise ValueError(
                    "attack_sheriff_outlaw requires actor_role=outlaw_a or outlaw_b"
                )
            if self.id == "attack_sheriff_deputy" and self.condition.actor_role != Role.DEPUTY:
                raise ValueError("attack_sheriff_deputy requires actor_role=deputy")
        if self.id == "role_or_faction_attack_terms" and condition_type not in {
            "role_attack",
            "faction_attack",
        }:
            raise ValueError(
                "role_or_faction_attack_terms requires role_attack or faction_attack"
            )
        return self


class RewardNamespace(ConfigModel):
    components: list[TerminalRewardComponent] = Field(default_factory=list)


class ShapingRewardNamespace(ConfigModel):
    components: list[ShapingRewardComponent] = Field(default_factory=list)


class RewardClippingConfig(ConfigModel):
    """Optional post-sum clipping policy for calculated rewards.

    Clipping is deliberately disabled by default.  Bounds are configuration
    data rather than implementation constants and are retained in snapshots.
    ``min``/``max`` are accepted as compact YAML spellings for the persisted
    ``min_value``/``max_value`` fields.
    """

    enabled: StrictBool = False
    min_value: float | None = Field(
        default=None,
        validation_alias=AliasChoices("min_value", "min"),
    )
    max_value: float | None = Field(
        default=None,
        validation_alias=AliasChoices("max_value", "max"),
    )

    @field_validator("min_value", "max_value", mode="before")
    @classmethod
    def bounds_must_be_finite(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("reward clipping bounds must be finite numbers")
        if not isfinite(float(value)):
            raise ValueError("reward clipping bounds must be finite numbers")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> "RewardClippingConfig":
        if self.enabled and (self.min_value is None or self.max_value is None):
            raise ValueError("enabled reward clipping requires min_value and max_value")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("reward clipping min_value must not exceed max_value")
        return self


class RewardConfig(ConfigModel):
    """Versioned reward DSL with independent terminal and shaping namespaces."""

    version: str = Field(min_length=1)
    terminal: RewardNamespace = Field(default_factory=RewardNamespace)
    shaping: ShapingRewardNamespace = Field(default_factory=ShapingRewardNamespace)
    clipping: RewardClippingConfig = Field(
        default_factory=RewardClippingConfig,
        validation_alias=AliasChoices("clipping", "clip"),
    )

    @field_validator("version")
    @classmethod
    def version_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reward_config.version must not be blank")
        return value

    @model_validator(mode="after")
    def reject_duplicate_component_ids(self) -> "RewardConfig":
        for namespace_name, components in (
            ("terminal", self.terminal.components),
            ("shaping", self.shaping.components),
        ):
            ids = [component.id for component in components]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate component id in {namespace_name} namespace")
        return self

    @property
    def config_identifier(self) -> str:
        """Stable identifier for the exact persisted reward configuration."""

        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    identifier = config_identifier


class FixedOptimizerConfig(ConfigModel):
    """Optimizer contract recorded in snapshots and artifact metadata."""

    name: Literal["Adam"] = "Adam"
    learning_rate: Literal[0.001, 0.0003]
    betas: tuple[Literal[0.9], Literal[0.999]] = (0.9, 0.999)
    eps: Literal[1e-08] = 1e-8
    weight_decay: Literal[0.0] = 0.0


class IntentPolicyConfig(ConfigModel):
    """IntentInference architecture plus DummyAgent policy parameters.

    The behavioral fields are consumed by DummyAgent only; they do not alter
    the fixed MainAgent inference architecture recorded in snapshots.
    """

    encoder: Literal["GRU"] = "GRU"
    hidden_size: Literal[128] = 128
    num_layers: Literal[1] = 1
    batch_first: Literal[True] = True
    dropout: Literal[0.0] = 0.0
    role_head_hidden_size: Literal[64] = 64
    role_output_size: Literal[5] = 5
    tie_policy: Literal["sheriff", "outlaw"] = "sheriff"
    pass_score: float = 0.0
    sheriff_team_attack_weight: float = 1.0
    outlaw_team_attack_weight: float = 1.0
    traitor_attack_weight: float = 1.0
    sheriff_direct_non_duel_weight: float = 0.25
    sheriff_direct_duel_weight: float = 2.0
    sheriff_role_bonus: float = 1.0
    deputy_role_bonus: float = 0.25
    target_health_weight: float = 0.0
    role_target_weights: dict[str, float] = Field(default_factory=dict)
    optimizer: FixedOptimizerConfig = Field(
        default_factory=lambda: FixedOptimizerConfig(learning_rate=0.001)
    )

    @field_validator("optimizer")
    @classmethod
    def require_supervised_learning_rate(cls, value: FixedOptimizerConfig) -> FixedOptimizerConfig:
        if value.learning_rate != 0.001:
            raise ValueError("IntentInference optimizer learning_rate is fixed at 1e-3")
        return value


class EmotionDefinition(ConfigModel):
    """One configurable DummyAgent emotion and its public update rules."""

    name: str = Field(min_length=1)
    initial_value: float | None = None
    change_rate: float = 0.0
    decay_rate: float = 0.0
    event_updates: dict[str, float] = Field(default_factory=dict)
    action_updates: dict[str, float] = Field(default_factory=dict)
    reward_updates: dict[str, float] = Field(default_factory=dict)
    terminal_updates: dict[str, float] = Field(default_factory=dict)
    effects: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_preset_spellings(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        values = dict(value)
        aliases = {
            "initial": "initial_value",
            "baseline": "initial_value",
            "change": "change_rate",
            "delta": "change_rate",
            "decay": "decay_rate",
            "event_rules": "event_updates",
            "action_rules": "action_updates",
            "reward_rules": "reward_updates",
            "terminal_rules": "terminal_updates",
            "action_effects": "effects",
        }
        for source, target in aliases.items():
            if source in values and target not in values:
                values[target] = values.pop(source)
        return values

    @field_validator(
        "initial_value", "change_rate", "decay_rate", mode="before"
    )
    @classmethod
    def numeric_values_must_be_finite(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
            raise ValueError("emotion numeric parameters must be finite numbers")
        return value

    @field_validator("initial_value")
    @classmethod
    def initial_value_must_be_normalized(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("emotion initial_value must be in [0, 1]")
        return value

    @field_validator("event_updates", "action_updates", "reward_updates", "terminal_updates", "effects")
    @classmethod
    def update_rules_must_be_finite(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("emotion update rule keys must be non-empty strings")
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(float(item)):
                raise ValueError("emotion update rule values must be finite numbers")
            normalized[key] = float(item)
        return normalized


class EmotionConfig(ConfigModel):
    """Emotion schema, baseline, and configurable DummyAgent behavior rules."""

    schema_version: str = Field(default="1.0", min_length=1)
    emotion_count: StrictInt = Field(default=1, ge=1)
    initial_baseline: float = Field(default=0.0, ge=0.0, le=1.0)
    emotions: tuple[EmotionDefinition, ...] = Field(default_factory=tuple)
    # Per-opponent relation vector configuration (EmotionModule schema "1.1").
    # ``relation_baseline`` is the neutral value every seat starts at; the
    # relation vector is SIGNED and clamped to [-1, 1] by the module (a grudge is
    # negative, trust is positive).  ``relation_updates`` maps a relationship
    # kind (for example ``shot_by``, ``shot_at``, ``eliminated_by``) to a finite
    # signed delta.  Both DEFAULT so pre-existing configs behave identically:
    # with an empty ``relation_updates`` the relation vector never moves.
    relation_baseline: float = Field(default=0.0, ge=-1.0, le=1.0)
    relation_updates: dict[str, float] = Field(default_factory=dict)
    # Public-HP control thresholds (EmotionModule schema "1.1").  The emotion
    # module derives HP-context bucket keys (``target_low_hp`` / ``target_lethal``
    # / ``self_low_hp`` / ``self_lethal``) from the PUBLIC per-event ``hp_after``.
    # Both thresholds DEFAULT to ``None`` so no HP bucket key ever fires and
    # behaviour is byte-for-byte identical to before this feature.  ``hp_after``
    # at or below ``hp_lethal_threshold`` yields the ``*_lethal`` bucket; at or
    # below ``hp_low_threshold`` yields the ``*_low_hp`` bucket.  Configure the
    # matching keys in an emotion definition's ``event_updates`` to react.
    hp_low_threshold: int | None = Field(default=None, ge=0)
    hp_lethal_threshold: int | None = Field(default=None, ge=0)
    encoder: Literal["GRU"] = "GRU"
    hidden_size: Literal[128] = 128
    num_layers: Literal[1] = 1
    batch_first: Literal[True] = True
    dropout: Literal[0.0] = 0.0
    emotion_head_hidden_size: Literal[64] = 64
    output_activation: Literal["Sigmoid"] = "Sigmoid"
    loss: Literal["MSE"] = "MSE"
    optimizer: FixedOptimizerConfig = Field(
        default_factory=lambda: FixedOptimizerConfig(learning_rate=0.001)
    )

    @field_validator("emotions", mode="before")
    @classmethod
    def normalize_emotion_collection(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, dict):
            return tuple(
                {"name": name, **(definition if isinstance(definition, dict) else {})}
                for name, definition in value.items()
            )
        if isinstance(value, (list, tuple)):
            normalized = []
            for index, definition in enumerate(value):
                if isinstance(definition, str):
                    normalized.append({"name": definition})
                elif isinstance(definition, dict):
                    item = dict(definition)
                    item.setdefault("name", f"emotion_{index}")
                    normalized.append(item)
                else:
                    normalized.append(definition)
            return tuple(normalized)
        return value

    @field_validator("initial_baseline", mode="before")
    @classmethod
    def baseline_must_be_finite_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("initial_baseline must be a finite number in [0, 1]")
        if not isfinite(float(value)):
            raise ValueError("initial_baseline must be a finite number in [0, 1]")
        return value

    @field_validator("relation_baseline", mode="before")
    @classmethod
    def relation_baseline_must_be_finite_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("relation_baseline must be a finite number in [-1, 1]")
        if not isfinite(float(value)):
            raise ValueError("relation_baseline must be a finite number in [-1, 1]")
        return value

    @field_validator("hp_low_threshold", "hp_lethal_threshold", mode="before")
    @classmethod
    def hp_threshold_must_be_non_negative_int_or_none(cls, value: object) -> object:
        # ``None`` keeps the HP buckets a strict no-op (the shipped default).  When
        # a threshold is provided it must be a genuine non-negative integer HP
        # value; booleans are rejected so ``True``/``False`` never masquerade as 1/0.
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("HP thresholds must be a non-negative integer or None")
        return value

    @field_validator("relation_updates")
    @classmethod
    def relation_updates_must_be_finite(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("relation update keys must be non-empty strings")
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(float(item)):
                raise ValueError("relation update values must be finite numbers")
            normalized[key] = float(item)
        return normalized

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("emotion.schema_version must not be blank")
        return value

    @model_validator(mode="after")
    def normalize_default_emotions(self) -> "EmotionConfig":
        definitions = self.emotions
        if not definitions:
            definitions = tuple(
                EmotionDefinition(name=f"emotion_{index}")
                for index in range(self.emotion_count)
            )
            object.__setattr__(self, "emotions", definitions)
        else:
            names = [item.name for item in definitions]
            if len(names) != len(set(names)):
                raise ValueError("emotion names must be unique")
            object.__setattr__(self, "emotion_count", len(definitions))
        return self

    @field_validator("optimizer")
    @classmethod
    def require_emotion_learning_rate(cls, value: FixedOptimizerConfig) -> FixedOptimizerConfig:
        if value.learning_rate != 0.001:
            raise ValueError("EmotionInference optimizer learning_rate is fixed at 1e-3")
        return value


class PPOConfig(ConfigModel):
    """PPO hyperparameters and the fixed PPOModule architecture contract."""

    rollout_transitions_per_worker: StrictInt = Field(default=256, ge=1)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    ppo_epochs: StrictInt = Field(default=4, ge=1)
    minibatch_size: StrictInt = Field(default=64, ge=1)
    clip_epsilon: float = Field(default=0.2, gt=0.0, lt=1.0)
    max_grad_norm: float | None = Field(default=0.5, gt=0.0)
    value_loss_coef: float = Field(default=0.5, ge=0.0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    shared_trunk_hidden_size: Literal[256] = 256
    shared_trunk_layers: Literal[2] = 2
    policy_output_size: Literal[6] = 6
    value_output_size: Literal[1] = 1
    activation: Literal["ReLU"] = "ReLU"
    optimizer: FixedOptimizerConfig = Field(
        default_factory=lambda: FixedOptimizerConfig(learning_rate=0.0003)
    )

    @field_validator("optimizer")
    @classmethod
    def require_ppo_learning_rate(cls, value: FixedOptimizerConfig) -> FixedOptimizerConfig:
        if value.learning_rate != 0.0003:
            raise ValueError("PPOModule optimizer learning_rate is fixed at 3e-4")
        return value


class PresetConfig(ConfigModel):
    """Dummy-agent preset data, versioned for export/import and snapshots."""

    preset_id: str = Field(default="default", min_length=1)
    version: str = Field(default="1.0", min_length=1)
    intent_policy: IntentPolicyConfig = Field(default_factory=IntentPolicyConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)


class SLTrainingConfig(ConfigModel):
    """Phase 2 supervised-learning settings."""

    epochs: StrictInt = Field(default=1, ge=1)
    batch_size: StrictInt = Field(default=64, ge=1)
    intent_policy: IntentPolicyConfig = Field(default_factory=IntentPolicyConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)


class RLTrainingConfig(ConfigModel):
    """Phase 3 PPO settings and the inference baseline used by RL."""

    ppo: PPOConfig = Field(default_factory=PPOConfig)
    emotion_inference: EmotionConfig = Field(default_factory=EmotionConfig)


class EvaluationConfig(ConfigModel):
    """Evaluation protocol inputs; policy values are explicit and recorded.

    ``deterministic_v1`` is the immutable primary protocol.  ``custom_v1``
    and ``stochastic_v1`` are intentionally separate run identities so their
    results cannot be mistaken for a continuation of the primary benchmark.
    """

    protocol_id: Literal["deterministic_v1", "custom_v1", "stochastic_v1"] = "deterministic_v1"
    opponent: str = Field(default="default", min_length=1)
    opponent_preset: str = Field(default="default", min_length=1)
    opponent_presets: dict[str, "PresetConfig"] = Field(default_factory=dict)
    games: int = Field(default=500, ge=1)
    games_per_seat: int = Field(default=100, ge=1)
    base_seed: int = 0
    seed_formula: Literal["base_seed_plus_evaluation_index", "explicit"] = (
        "base_seed_plus_evaluation_index"
    )
    seeds: list[int] = Field(default_factory=list)
    deterministic: bool = True
    action_mode: Literal["deterministic", "stochastic"] = "deterministic"
    seat_policy: Literal["balanced", "cyclic", "fixed", "custom"] = "balanced"
    seats: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action_mode(self) -> "EvaluationConfig":
        if self.deterministic != (self.action_mode == "deterministic"):
            raise ValueError("deterministic must agree with action_mode")
        if self.seed_formula == "explicit" and len(self.seeds) != self.games:
            raise ValueError("explicit seed_formula requires one seed per game")
        if self.seeds and len(self.seeds) != self.games:
            raise ValueError("seeds must contain exactly one seed per game")
        if any(isinstance(seat, bool) or seat < 0 or seat > 4 for seat in self.seats):
            raise ValueError("evaluation seats must be player indices in [0, 4]")
        if len(set(self.seats)) != len(self.seats):
            raise ValueError("evaluation seats must not contain duplicates")
        if self.seat_policy == "fixed" and len(self.seats) != 1:
            raise ValueError("fixed seat_policy requires exactly one seat")
        return self


class PauseConfig(ConfigModel):
    """In-process pause handshake settings."""

    ack_timeout_seconds: float = Field(default=30.0, gt=0.0)


class HardwareConfig(ConfigModel):
    """User-supplied resource limits; no benchmark or auto-tuning is performed."""

    cpu_cores: StrictInt = Field(default=1, ge=1, le=4096)
    ram_gb: float = Field(default=1.0, gt=0.0, le=1_048_576.0)
    use_gpu: StrictBool = False
    workers: StrictInt = Field(default=1, ge=1, le=4096)
    batch_size: StrictInt = Field(default=64, ge=1, le=1_000_000)
    buffer_size: StrictInt = Field(default=256, ge=1, le=1_000_000_000)

    @field_validator("cpu_cores", "workers", "batch_size", "buffer_size", mode="before")
    @classmethod
    def reject_boolean_integer_values(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("hardware integer settings must be integers, not booleans")
        return value
    @field_validator("ram_gb", mode="before")
    @classmethod
    def reject_invalid_ram(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("ram_gb must be a positive finite number")
        if not isfinite(float(value)):
            raise ValueError("ram_gb must be a positive finite number")
        return value


class DatasetConfig(ConfigModel):
    """Canonical Parquet shard and derived-cache locations and limits."""

    canonical_shard_max_episodes: StrictInt = Field(default=10_000, ge=1)
    canonical_shard_max_bytes: StrictInt = Field(default=256_000_000, ge=1)
    canonical_path: str = "datasets/canonical"
    cache_path: str = "datasets/cache"

    @field_validator("canonical_shard_max_episodes", "canonical_shard_max_bytes", mode="before")
    @classmethod
    def reject_boolean_limits(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("dataset shard limits must be integers, not booleans")
        return value


class ModelArchitectureSnapshot(ConfigModel):
    """Fixed architecture values that must be copied into job/artifact metadata."""

    intent_encoder: Literal["GRU"] = "GRU"
    intent_hidden_size: Literal[128] = 128
    intent_num_layers: Literal[1] = 1
    intent_batch_first: Literal[True] = True
    intent_dropout: Literal[0.0] = 0.0
    intent_role_head: Literal["Linear(128,64)->ReLU->Linear(64,5)"] = (
        "Linear(128,64)->ReLU->Linear(64,5)"
    )
    emotion_encoder: Literal["GRU"] = "GRU"
    emotion_hidden_size: Literal[128] = 128
    emotion_num_layers: Literal[1] = 1
    emotion_batch_first: Literal[True] = True
    emotion_dropout: Literal[0.0] = 0.0
    emotion_head: Literal["Linear(128,64)->ReLU->Linear(64,E)->Sigmoid"] = (
        "Linear(128,64)->ReLU->Linear(64,E)->Sigmoid"
    )
    ppo_shared_trunk: Literal[
        "Linear(input_dim,256)->ReLU->Linear(256,256)->ReLU"
    ] = "Linear(input_dim,256)->ReLU->Linear(256,256)->ReLU"
    ppo_policy_head: Literal["Linear(256,6)"] = "Linear(256,6)"
    ppo_value_head: Literal["Linear(256,1)"] = "Linear(256,1)"


class OptimizerSnapshot(ConfigModel):
    """Fixed optimizer values for all trainable modules."""

    intent: FixedOptimizerConfig = Field(
        default_factory=lambda: FixedOptimizerConfig(learning_rate=0.001)
    )
    emotion: FixedOptimizerConfig = Field(
        default_factory=lambda: FixedOptimizerConfig(learning_rate=0.001)
    )
    ppo: FixedOptimizerConfig = Field(
        default_factory=lambda: FixedOptimizerConfig(learning_rate=0.0003)
    )

    @model_validator(mode="after")
    def validate_module_learning_rates(self) -> "OptimizerSnapshot":
        if self.intent.learning_rate != 0.001 or self.emotion.learning_rate != 0.001:
            raise ValueError("Intent and Emotion optimizers must use learning_rate=1e-3")
        if self.ppo.learning_rate != 0.0003:
            raise ValueError("PPO optimizer must use learning_rate=3e-4")
        return self


class ArtifactConfig(ConfigModel):
    """Final inference artifact format and metadata contract."""

    # ``artifact_format_version`` is retained as a compatibility field for
    # early configuration files; the Design §15.3 name is canonical.
    artifact_format_version: Literal["1.0"] = "1.0"
    inference_model_artifact_format_version: Literal["1.0"] = Field(
        default="1.0",
        validation_alias=AliasChoices(
            "inference_model_artifact_format_version", "artifact_format_version"
        ),
    )
    artifact_type: Literal["inference"] = "inference"
    output_directory: str = "models"
    extension: Literal[".bangmodel"] = ".bangmodel"
    weights_file: Literal["model.safetensors"] = "model.safetensors"
    metadata_file: Literal["metadata.json"] = "metadata.json"
    manifest_file: Literal["manifest.json"] = "manifest.json"

    @model_validator(mode="after")
    def format_versions_must_match(self) -> "ArtifactConfig":
        if self.artifact_format_version != self.inference_model_artifact_format_version:
            raise ValueError("artifact format version fields must match")
        return self


class SystemConfig(ConfigModel):
    """Top-level configuration and the source for immutable job snapshots."""

    config_version: str = Field(default="1.0", min_length=1)
    seed: int = 0
    env: EnvConfig = Field(default_factory=EnvConfig)
    reward_config: RewardConfig = Field(
        default_factory=lambda: RewardConfig(version="1.0"),
        validation_alias=AliasChoices("reward_config", "reward"),
    )
    intent_policy: IntentPolicyConfig = Field(default_factory=IntentPolicyConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    preset: PresetConfig = Field(default_factory=PresetConfig)
    sl_training: SLTrainingConfig = Field(default_factory=SLTrainingConfig)
    rl_training: RLTrainingConfig = Field(
        default_factory=RLTrainingConfig,
        validation_alias=AliasChoices("rl_training", "training"),
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    pause: PauseConfig = Field(default_factory=PauseConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    artifact: ArtifactConfig = Field(default_factory=ArtifactConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model_architecture: ModelArchitectureSnapshot = Field(
        default_factory=ModelArchitectureSnapshot
    )
    optimizer_snapshot: OptimizerSnapshot = Field(default_factory=OptimizerSnapshot)

    @field_validator("config_version")
    @classmethod
    def config_version_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("config_version must not be blank")
        return value

    def job_snapshot(self) -> dict[str, object]:
        """Return a JSON-ready immutable-at-capture configuration snapshot."""

        return {
            "config_version": self.config_version,
            "reward_config_version": self.reward_config.version,
            "reward_config_id": self.reward_config.config_identifier,
            "config": self.model_dump(mode="json"),
            "model_architecture": self.model_architecture.model_dump(mode="json"),
            "optimizers": self.optimizer_snapshot.model_dump(mode="json"),
        }

    snapshot = job_snapshot

    def artifact_metadata(self) -> dict[str, object]:
        """Return metadata fields required to verify fixed model contracts."""

        return {
            "inference_model_artifact_format_version": (
                self.artifact.inference_model_artifact_format_version
            ),
            # Retain the short key for consumers of the initial schema draft.
            "artifact_format_version": self.artifact.artifact_format_version,
            "model_architecture": self.model_architecture.model_dump(mode="json"),
            "optimizers": self.optimizer_snapshot.model_dump(mode="json"),
            "reward_config_version": self.reward_config.version,
            "reward_config_id": self.reward_config.config_identifier,
            "emotion_schema_version": self.emotion.schema_version,
            "training": {
                "ppo": self.rl_training.ppo.model_dump(mode="json"),
                "emotion_initial_baseline": self.rl_training.emotion_inference.initial_baseline,
            },
        }


__all__ = [
    "ArtifactConfig",
    "EmotionConfig",
    "EmotionDefinition",
    "EnvConfig",
    "EvaluationConfig",
    "EventCondition",
    "EventHistoryConfig",
    "FactionAttackCondition",
    "FixedOptimizerConfig",
    "HardwareConfig",
    "IntentPolicyConfig",
    "ModelArchitectureSnapshot",
    "OptimizerSnapshot",
    "PassCondition",
    "PauseConfig",
    "PPOConfig",
    "PresetConfig",
    "RLTrainingConfig",
    "RewardClippingConfig",
    "RewardConfig",
    "RewardCondition",
    "Role",
    "RoleAttackCondition",
    "SLTrainingConfig",
    "ShapingRewardComponent",
    "SystemConfig",
    "Team",
    "TerminalResultCondition",
    "TerminalRewardComponent",
    "DatasetConfig",
]
