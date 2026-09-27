"""Masked PPO policy/value module for MainAgent decisions.

The module is the final decision boundary after public observation processing,
RoleFusion, and EmotionInference.  It accepts either the five public feature
blocks described by Design §7.4 or an already-concatenated feature tensor.  It
never accepts labels, rewards, hidden roles/emotions, or privileged logs.

This module owns inference-time action selection and PPO-compatible policy
statistics only.  Rollout buffers and optimizer state are intentionally not
persisted by its versioned module artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from ...config import PPOConfig
from .preprocessor import (
    PLAYER_FEATURE_SIZE,
    STATE_VECTOR_SIZE,
)
from .role_fusion import ROLE_COUNT, TARGET_PLAYER_COUNT


PPO_MODULE_FORMAT_VERSION = "1.0"
PPO_MODULE_ARCHITECTURE_VERSION = "1.0"
PPO_OPTIMIZER_METADATA = {
    "type": "Adam",
    "learning_rate": 3e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
}
POLICY_OUTPUT_SIZE = 6
VALUE_OUTPUT_SIZE = 1
SHARED_TRUNK_HIDDEN_SIZE = 256

# These names are rejected before conversion so a caller cannot silently pass
# training labels or privileged diagnostics as an auxiliary feature.
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "roles",
        "role",
        "role_label",
        "role_labels",
        "hidden_role",
        "hidden_roles",
        "true_role",
        "true_roles",
        "actual_role",
        "actual_roles",
        "emotion_label",
        "emotion_labels",
        "emotion_state",
        "emotion_states",
        "true_emotion",
        "true_emotions",
        "actual_emotion",
        "actual_emotions",
        "reward",
        "rewards",
        "reward_breakdown",
        "privileged_state",
        "privileged_observation",
        "private_observation",
        "privileged_diagnostics",
        "raw_intent",
        "raw_role_distribution",
    }
)


@dataclass(frozen=True, slots=True)
class PPOOutput:
    """Policy/value statistics for one state or a batch of states.

    ``logits`` and ``value`` preserve the input's unbatched/batched form:
    ``(6,)``/``(1,)`` for one state and ``(B, 6)``/``(B, 1)`` for a batch.
    ``distribution`` always contains the mask-normalized categorical policy.
    ``action``, ``log_prob`` and ``entropy`` are populated by ``act`` or when
    ``forward(..., sample=True)`` is requested.
    """

    logits: Tensor
    value: Tensor
    distribution: Categorical
    action: Tensor | None = None
    log_prob: Tensor | None = None
    entropy: Tensor | None = None

    @property
    def values(self) -> Tensor:
        return self.value

    @property
    def probs(self) -> Tensor:
        return self.distribution.probs

    @property
    def probabilities(self) -> Tensor:
        return self.distribution.probs

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "logits": self.logits,
            "masked_logits": self.logits,
            "value": self.value,
            "values": self.value,
            "distribution": self.distribution,
            "probs": self.probs,
            "probabilities": self.probs,
            "action": self.action,
            "actions": self.action,
            "log_prob": self.log_prob,
            "log_probs": self.log_prob,
            "entropy": self.entropy,
        }
        if key not in aliases:
            raise KeyError(key)
        return aliases[key]

    def keys(self) -> tuple[str, ...]:
        return (
            "logits",
            "value",
            "distribution",
            "action",
            "log_prob",
            "entropy",
        )

    def __iter__(self):
        # Allows the conventional ``logits, value = module(...)`` usage.
        yield self.logits
        yield self.value


@dataclass(frozen=True, slots=True)
class PPOAction:
    """Sampled action and PPO statistics for the selected action."""

    action: Tensor
    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    logits: Tensor
    distribution: Categorical

    @property
    def values(self) -> Tensor:
        return self.value

    @property
    def probs(self) -> Tensor:
        return self.distribution.probs

    def __iter__(self):
        # Convenient for callers that want action, log-prob, entropy, value.
        yield self.action
        yield self.log_prob
        yield self.entropy
        yield self.value


class PPOModule(nn.Module):
    """Shared-trunk categorical policy and scalar value estimator.

    Structured input dimensions default to the actual MainAgent preprocessor
    contract: state vector (9), player board (5x8), confirmed roles (5x5),
    fused intent (4x5), and emotion (4xE).  ``emotion_dim`` defaults to one
    emotion per target player; callers with a different Emotion Schema provide
    its flattened dimension explicitly.  ``input_dim`` can be supplied for a
    direct concatenated feature contract, or is derived from the component
    dimensions when structured inputs are used.
    """

    module_type = "PPOModule"
    format_version = PPO_MODULE_FORMAT_VERSION
    architecture_version = PPO_MODULE_ARCHITECTURE_VERSION

    def __init__(
        self,
        input_dim: int | None = None,
        *,
        self_state_dim: int = STATE_VECTOR_SIZE,
        board_state_dim: int = 5 * PLAYER_FEATURE_SIZE,
        confirmed_role_dim: int = 5 * ROLE_COUNT,
        intent_dim: int = TARGET_PLAYER_COUNT * ROLE_COUNT,
        emotion_dim: int = TARGET_PLAYER_COUNT,
        ppo_config: PPOConfig | Mapping[str, Any] | None = None,
        config: PPOConfig | Mapping[str, Any] | None = None,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if ppo_config is not None and config is not None:
            raise ValueError("provide only one of ppo_config or config")
        raw_config = ppo_config if ppo_config is not None else config
        self.config = self._coerce_config(raw_config)
        self._component_dims = {
            "self_state": self._positive_int(self_state_dim, "self_state_dim"),
            "board_state": self._positive_int(board_state_dim, "board_state_dim"),
            "confirmed_role": self._positive_int(confirmed_role_dim, "confirmed_role_dim"),
            "fused_intent": self._positive_int(intent_dim, "intent_dim"),
            "emotion": self._positive_int(emotion_dim, "emotion_dim"),
        }
        calculated_input_dim = sum(self._component_dims.values())
        if input_dim is None:
            input_dim = calculated_input_dim
        self._input_dim = self._positive_int(input_dim, "input_dim")
        if input_dim != calculated_input_dim:
            # A direct tensor contract can use an independently calculated
            # dimension, but structured blocks must never be silently padded.
            self._structured_input_dim = calculated_input_dim
        else:
            self._structured_input_dim = calculated_input_dim
        self._seed = self._validate_seed(seed)

        layers: list[nn.Module] = [
            nn.Linear(self.input_dim, SHARED_TRUNK_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(SHARED_TRUNK_HIDDEN_SIZE, SHARED_TRUNK_HIDDEN_SIZE),
            nn.ReLU(),
        ]
        self.shared_trunk = self._initialise(layers, seed)
        self.policy_head = self._initialise(
            [nn.Linear(SHARED_TRUNK_HIDDEN_SIZE, POLICY_OUTPUT_SIZE)],
            None if seed is None else seed + 1,
        )
        self.value_head = self._initialise(
            [nn.Linear(SHARED_TRUNK_HIDDEN_SIZE, VALUE_OUTPUT_SIZE)],
            None if seed is None else seed + 2,
        )

    @staticmethod
    def _coerce_config(config: PPOConfig | Mapping[str, Any] | None) -> PPOConfig:
        if config is None:
            return PPOConfig()
        if isinstance(config, PPOConfig):
            return config
        if isinstance(config, Mapping):
            return PPOConfig.model_validate(dict(config))
        raise TypeError("config must be a PPOConfig, mapping, or None")

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _validate_seed(value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise TypeError("seed must be an integer or None")
        return value

    @staticmethod
    def _initialise(layers: list[nn.Module], seed: int | None) -> nn.Sequential:
        if seed is None:
            return nn.Sequential(*layers)
        # Linear modules initialize themselves at construction time.  Reset
        # them inside the forked seeded RNG as well, otherwise two modules
        # constructed with the same seed would still receive different weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for layer in layers:
                reset = getattr(layer, "reset_parameters", None)
                if reset is not None:
                    reset()
            return nn.Sequential(*layers)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return POLICY_OUTPUT_SIZE

    @property
    def value_dim(self) -> int:
        return VALUE_OUTPUT_SIZE

    @property
    def component_dims(self) -> dict[str, int]:
        return dict(self._component_dims)

    @property
    def optimizer_metadata(self) -> dict[str, Any]:
        return dict(PPO_OPTIMIZER_METADATA)

    @property
    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "module_type": self.module_type,
            "architecture_version": self.architecture_version,
            "ppo_module_format_version": self.format_version,
            "input_dim": self.input_dim,
            "initialization_seed": self._seed,
            "structured_input_dim": self._structured_input_dim,
            "component_dims": self.component_dims,
            "shared_trunk": {
                "layers": [
                    {"type": "Linear", "in_features": self.input_dim, "out_features": 256},
                    {"type": "ReLU"},
                    {"type": "Linear", "in_features": 256, "out_features": 256},
                    {"type": "ReLU"},
                ]
            },
            "policy_head": {
                "layers": [{"type": "Linear", "in_features": 256, "out_features": 6}],
                "output": "masked categorical logits",
            },
            "value_head": {
                "layers": [{"type": "Linear", "in_features": 256, "out_features": 1}],
                "output": "scalar value estimate",
            },
            "activation": "ReLU",
            "action_space_size": POLICY_OUTPUT_SIZE,
            "optimizer": self.optimizer_metadata,
            "ppo_config": self.config.model_dump(mode="json"),
            "input_contract": {
                "order": [
                    "self_state",
                    "board_state",
                    "confirmed_role_features",
                    "fused_intent",
                    "emotion_features",
                ],
                "intent_source": "RoleFusion output only; raw IntentInference output is forbidden",
                "role_label": "forbidden",
                "emotion_label": "forbidden",
                "reward": "forbidden",
                "privileged_log": "forbidden",
                "action_mask": "required for action selection and policy statistics",
            },
        }

    def metadata(self) -> dict[str, Any]:
        return self._json_ready(self.architecture_metadata)

    get_metadata = metadata

    def create_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.parameters(),
            lr=PPO_OPTIMIZER_METADATA["learning_rate"],
            betas=tuple(PPO_OPTIMIZER_METADATA["betas"]),
            eps=PPO_OPTIMIZER_METADATA["eps"],
            weight_decay=PPO_OPTIMIZER_METADATA["weight_decay"],
        )

    def freeze_for_inference(self) -> "PPOModule":
        """Switch to inference mode and disable gradients for every weight."""

        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    freeze = freeze_for_inference

    def forward(
        self,
        inputs: Any,
        action_mask: Any | None = None,
        *,
        deterministic: bool = False,
        sample: bool = False,
        fused_intent: Any | None = None,
        emotion_features: Any | None = None,
        confirmed_role_features: Any | None = None,
        self_state: Any | None = None,
        board_state: Any | None = None,
    ) -> PPOOutput:
        """Return masked policy statistics and value estimates.

        ``inputs`` may be a direct concatenated tensor, a public processed
        observation (with the remaining inference outputs supplied as kwargs),
        or a mapping containing the five named feature blocks.  Passing
        ``sample=True`` also selects an action; ``deterministic=True`` uses
        masked argmax and otherwise samples from the masked categorical policy.
        """

        features, mask, unbatched = self._extract_features(
            inputs,
            action_mask=action_mask,
            self_state=self_state,
            board_state=board_state,
            confirmed_role_features=confirmed_role_features,
            fused_intent=fused_intent,
            emotion_features=emotion_features,
        )
        trunk = self.shared_trunk(features)
        raw_logits = self.policy_head(trunk)
        masked_logits = self.masked_logits(raw_logits, mask)
        result_logits = masked_logits[0] if unbatched else masked_logits
        distribution = Categorical(logits=result_logits)
        values = self.value_head(trunk)
        result_values = values[0] if unbatched else values
        result = PPOOutput(result_logits, result_values, distribution)
        if sample:
            action = self._select_action(distribution, masked_logits, deterministic)
            action_log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()
            if unbatched:
                action = action.reshape(())
                action_log_prob = action_log_prob.reshape(())
                entropy = entropy.reshape(())
            result = PPOOutput(
                result_logits,
                result_values,
                distribution,
                action,
                action_log_prob,
                entropy,
            )
        return result

    def forward_features(
        self,
        self_state: Any,
        board_state: Any,
        confirmed_role_features: Any,
        fused_intent: Any,
        emotion_features: Any,
        action_mask: Any | None = None,
        *,
        deterministic: bool = False,
        sample: bool = False,
    ) -> PPOOutput:
        """Explicit structured-feature entry point."""

        return self.forward(
            {
                "self_state": self_state,
                "board_state": board_state,
                "confirmed_role_features": confirmed_role_features,
                "fused_intent": fused_intent,
                "emotion_features": emotion_features,
                "action_mask": action_mask,
            },
            deterministic=deterministic,
            sample=sample,
        )

    def masked_logits(self, logits: Tensor, action_mask: Any) -> Tensor:
        """Set unavailable actions to ``-inf`` after validating the mask."""

        values = logits if isinstance(logits, Tensor) else torch.as_tensor(logits)
        if values.shape[-1] != POLICY_OUTPUT_SIZE:
            raise ValueError("policy logits must have final dimension 6")
        mask = self._normalise_mask(action_mask, values.shape[:-1], values.device)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("action_mask must enable at least one action per state")
        invalid = ~mask
        return values.masked_fill(invalid, -torch.inf)

    def action_distribution(self, inputs: Any, action_mask: Any | None = None, **kwargs: Any) -> Categorical:
        return self.forward(inputs, action_mask, **kwargs).distribution

    distribution = action_distribution

    def act(
        self,
        inputs: Any,
        action_mask: Any | None = None,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> PPOAction:
        """Choose a valid action and return all PPO rollout statistics."""

        result = self.forward(inputs, action_mask, **kwargs)
        mask = self._extract_mask_for_distribution(result.distribution, action_mask, inputs)
        del mask  # validation is already performed by ``forward``.
        # Reconstruct masked logits from the distribution so this method also
        # works with custom callers that supplied a direct feature tensor.
        logits = result.logits
        batched_logits = logits.unsqueeze(0) if logits.ndim == 1 else logits
        if generator is None:
            action = (
                batched_logits.argmax(dim=-1)
                if deterministic
                else torch.multinomial(result.distribution.probs, 1).squeeze(-1)
            )
        else:
            if deterministic:
                action = batched_logits.argmax(dim=-1)
            else:
                action = torch.multinomial(result.distribution.probs, 1, generator=generator).squeeze(-1)
        log_prob = result.distribution.log_prob(action)
        entropy = result.distribution.entropy()
        if logits.ndim == 1:
            action = action.reshape(())
            log_prob = log_prob.reshape(())
            entropy = entropy.reshape(())
        return PPOAction(action, log_prob, entropy, result.value, result.logits, result.distribution)

    sample_action = act
    select_action = act
    get_action = act

    def evaluate_actions(
        self,
        inputs: Any,
        actions: Any,
        action_mask: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Recompute log-probability, entropy, and value with the same mask."""

        result = self.forward(inputs, action_mask, **kwargs)
        action = torch.as_tensor(actions, device=result.logits.device, dtype=torch.long)
        expected = result.distribution.probs.shape[:-1]
        if tuple(action.shape) != tuple(expected):
            raise ValueError(f"actions must have shape {tuple(expected)}")
        if bool(((action < 0) | (action >= POLICY_OUTPUT_SIZE)).any()):
            raise ValueError("actions must be in [0, 5]")
        return result.distribution.log_prob(action), result.distribution.entropy(), result.value

    def log_prob(self, inputs: Any, actions: Any, action_mask: Any | None = None, **kwargs: Any) -> Tensor:
        return self.evaluate_actions(inputs, actions, action_mask, **kwargs)[0]

    def entropy(self, inputs: Any, action_mask: Any | None = None, **kwargs: Any) -> Tensor:
        return self.forward(inputs, action_mask, **kwargs).distribution.entropy()

    def value(self, inputs: Any, action_mask: Any | None = None, **kwargs: Any) -> Tensor:
        return self.forward(inputs, action_mask, **kwargs).value

    def _extract_features(
        self,
        inputs: Any,
        *,
        action_mask: Any | None,
        self_state: Any | None,
        board_state: Any | None,
        confirmed_role_features: Any | None,
        fused_intent: Any | None,
        emotion_features: Any | None,
    ) -> tuple[Tensor, Tensor, bool]:
        values: Mapping[str, Any] | None = None
        if isinstance(inputs, Mapping) or (hasattr(inputs, "keys") and hasattr(inputs, "__getitem__")):
            values = self._mapping(inputs)
            self._reject_privileged(values)
            if "observation" in values:
                observation = values["observation"]
                if not isinstance(observation, Mapping):
                    raise TypeError("observation must be a public mapping")
                observation_values = self._mapping(observation)
                self._reject_privileged(observation_values)
                values = {**observation_values, **{key: item for key, item in values.items() if key != "observation"}}
        elif hasattr(inputs, "state_vector") and hasattr(inputs, "player_state_matrix"):
            values = {
                "state_vector": inputs.state_vector,
                "player_state_matrix": inputs.player_state_matrix,
                "role_beliefs": inputs.role_beliefs,
                "action_mask": inputs.action_mask,
            }
        if values is not None:
            # A mapping may carry an already-concatenated public feature
            # tensor alongside its mask.
            if all(item is None for item in (self_state, board_state, confirmed_role_features, fused_intent, emotion_features)) and "features" in values:
                inputs = values["features"]
            else:
                inputs = values
            self_state = self_state if self_state is not None else values.get("self_state", values.get("state_vector"))
            board_state = board_state if board_state is not None else values.get("board_state", values.get("player_state_matrix"))
            confirmed_role_features = (
                confirmed_role_features
                if confirmed_role_features is not None
                else values.get("confirmed_role_features", values.get("confirmed_role", values.get("role_beliefs")))
            )
            fused_intent = fused_intent if fused_intent is not None else values.get("fused_intent", values.get("intent_features"))
            emotion_features = emotion_features if emotion_features is not None else values.get("emotion_features", values.get("emotion"))
            action_mask = action_mask if action_mask is not None else values.get("action_mask")
        # A direct tensor is an already-concatenated public feature vector.
        if self_state is None and board_state is None and confirmed_role_features is None and fused_intent is None and emotion_features is None:
            features, unbatched = self._direct_features(inputs)
        else:
            missing = [
                name
                for name, item in (
                    ("self_state", self_state),
                    ("board_state", board_state),
                    ("confirmed_role_features", confirmed_role_features),
                    ("fused_intent", fused_intent),
                    ("emotion_features", emotion_features),
                )
                if item is None
            ]
            if missing:
                raise ValueError("missing PPO feature blocks: " + ", ".join(missing))
            blocks = [
                self._feature_block(self_state, "self_state", self._component_dims["self_state"], (STATE_VECTOR_SIZE,)),
                self._feature_block(board_state, "board_state", self._component_dims["board_state"], (5, PLAYER_FEATURE_SIZE)),
                self._feature_block(confirmed_role_features, "confirmed_role_features", self._component_dims["confirmed_role"], (5, ROLE_COUNT)),
                self._feature_block(fused_intent, "fused_intent", self._component_dims["fused_intent"], (TARGET_PLAYER_COUNT, ROLE_COUNT)),
                self._feature_block(emotion_features, "emotion_features", self._component_dims["emotion"], (TARGET_PLAYER_COUNT,)),
            ]
            batch_sizes = {int(block.shape[0]) for block in blocks}
            if len(batch_sizes) != 1:
                raise ValueError("PPO feature blocks must share a batch dimension")
            features = torch.cat(blocks, dim=-1)
            unbatched = features.shape[0] == 1 and all(self._is_unbatched_block(item, shape) for item, shape in zip(
                (self_state, board_state, confirmed_role_features, fused_intent, emotion_features),
                ((STATE_VECTOR_SIZE,), (5, PLAYER_FEATURE_SIZE), (5, ROLE_COUNT), (TARGET_PLAYER_COUNT, ROLE_COUNT), (TARGET_PLAYER_COUNT,)),
            ))
            if features.shape[-1] != self.input_dim:
                raise ValueError(
                    f"structured PPO features concatenate to {features.shape[-1]}, expected input_dim {self.input_dim}"
                )
            if unbatched:
                features = features[:1]
        mask = self._normalise_mask(action_mask, features.shape[:-1], features.device)
        return features, mask, unbatched

    def _direct_features(self, inputs: Any) -> tuple[Tensor, bool]:
        try:
            features = inputs if isinstance(inputs, Tensor) else torch.as_tensor(inputs)
        except (TypeError, ValueError) as exc:
            raise TypeError("PPO inputs must be numeric feature tensors or public mappings") from exc
        if features.ndim not in {1, 2}:
            raise ValueError("direct PPO features must have shape (input_dim,) or (B, input_dim)")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"PPO feature dimension must be {self.input_dim}")
        features = self._numeric_tensor(features, "features")
        return (features.unsqueeze(0) if features.ndim == 1 else features), features.ndim == 1

    def _feature_block(self, value: Any, name: str, expected_dim: int, unbatched_shape: tuple[int, ...]) -> Tensor:
        try:
            tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric") from exc
        is_structured_unbatched = tuple(tensor.shape) == unbatched_shape
        if is_structured_unbatched:
            tensor = tensor.reshape(1, expected_dim)
        elif tensor.ndim == 1:
            if tensor.shape[0] != expected_dim:
                raise ValueError(f"{name} must contain {expected_dim} values")
            tensor = tensor.unsqueeze(0)
        elif tuple(tensor.shape[1:]) == unbatched_shape:
            tensor = tensor.reshape(tensor.shape[0], expected_dim)
        elif tensor.shape[-1] == expected_dim:
            tensor = tensor.reshape(-1, expected_dim)
        else:
            raise ValueError(f"{name} has incompatible shape {tuple(tensor.shape)}")
        return self._numeric_tensor(tensor, name)

    @staticmethod
    def _is_unbatched_block(value: Any, expected_shape: tuple[int, ...]) -> bool:
        try:
            return tuple(torch.as_tensor(value).shape) == expected_shape
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _numeric_tensor(value: Tensor, name: str) -> Tensor:
        if value.dtype == torch.bool or not (value.dtype.is_floating_point or value.dtype.is_complex or value.dtype in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }):
            raise TypeError(f"{name} must contain numeric values")
        value = value.to(dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")
        return value

    @staticmethod
    def _normalise_mask(value: Any, batch_shape: torch.Size | tuple[int, ...], device: torch.device) -> Tensor:
        if value is None:
            raise ValueError("action_mask is required for PPO policy selection")
        try:
            mask = value if isinstance(value, Tensor) else torch.as_tensor(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("action_mask must be a boolean or binary integer array") from exc
        expected = (*tuple(batch_shape), POLICY_OUTPUT_SIZE)
        if tuple(mask.shape) == (POLICY_OUTPUT_SIZE,):
            mask = mask.reshape((1,) * len(tuple(batch_shape)) + (POLICY_OUTPUT_SIZE,)).expand(expected)
        if tuple(mask.shape) != expected:
            raise ValueError(f"action_mask must have shape {expected}, got {tuple(mask.shape)}")
        if mask.dtype != torch.bool and mask.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }:
            raise TypeError("action_mask must be bool or an integer array")
        if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("action_mask must contain only 0 or 1")
        mask = mask.to(device=device, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("action_mask must enable at least one action per state")
        return mask

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        return {key: value[key] for key in value.keys()}

    @staticmethod
    def _reject_privileged(values: Mapping[str, Any]) -> None:
        leaked = []
        for key in values:
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_INPUT_KEYS or normalized.endswith("_true_role"):
                leaked.append(str(key))
        if leaked:
            raise ValueError("privileged PPO inputs are not accepted: " + ", ".join(sorted(leaked)))

    @staticmethod
    def _select_action(distribution: Categorical, logits: Tensor, deterministic: bool) -> Tensor:
        if deterministic:
            return logits.argmax(dim=-1)
        return torch.multinomial(distribution.probs, 1).squeeze(-1)

    @staticmethod
    def _extract_mask_for_distribution(distribution: Categorical, action_mask: Any, inputs: Any) -> Tensor:
        del action_mask, inputs
        return distribution.probs > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.format_version,
            "module_type": self.module_type,
            "architecture": self.metadata(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    serialize = to_json

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PPOModule":
        if not isinstance(value, Mapping):
            raise TypeError("PPO metadata must be a mapping")
        version = value.get("schema_version", value.get("ppo_module_format_version"))
        if version != cls.format_version:
            raise ValueError(f"unsupported PPOModule schema version: {version!r}")
        architecture = value.get("architecture", value)
        if not isinstance(architecture, Mapping) or architecture.get("module_type", cls.module_type) != cls.module_type:
            raise ValueError("metadata does not describe PPOModule")
        component_dims = architecture.get("component_dims", {})
        if not isinstance(component_dims, Mapping):
            raise ValueError("PPO metadata component_dims must be a mapping")
        config_data = architecture.get("ppo_config", {})
        config = PPOConfig.model_validate(config_data)
        return cls(
            input_dim=architecture.get("input_dim"),
            self_state_dim=component_dims.get("self_state", STATE_VECTOR_SIZE),
            board_state_dim=component_dims.get("board_state", 5 * PLAYER_FEATURE_SIZE),
            confirmed_role_dim=component_dims.get("confirmed_role", 5 * ROLE_COUNT),
            intent_dim=component_dims.get("fused_intent", TARGET_PLAYER_COUNT * ROLE_COUNT),
            emotion_dim=component_dims.get("emotion", TARGET_PLAYER_COUNT),
            ppo_config=config,
            seed=architecture.get("initialization_seed", 0),
        )

    from_metadata = from_dict

    def save(self, path: str | Path) -> None:
        """Persist weights and versioned metadata, never optimizer state."""

        payload = {
            "schema_version": self.format_version,
            "module_type": self.module_type,
            "metadata": self.to_dict(),
            "state_dict": self.state_dict(),
            "training_mode": self.training,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, map_location: Any = "cpu") -> "PPOModule":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != cls.format_version:
            raise ValueError("unsupported or malformed PPOModule artifact")
        if payload.get("module_type") != cls.module_type:
            raise ValueError("artifact does not contain PPOModule weights")
        module = cls.from_dict(payload["metadata"])
        module.load_state_dict(payload["state_dict"])
        if bool(payload.get("training_mode", False)):
            module.train()
        else:
            module.eval()
        return module

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, Mapping):
            return {str(key): PPOModule._json_ready(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [PPOModule._json_ready(item) for item in value]
        return value


PPO = PPOModule

__all__ = [
    "POLICY_OUTPUT_SIZE",
    "PPO",
    "PPOAction",
    "PPO_MODULE_ARCHITECTURE_VERSION",
    "PPO_MODULE_FORMAT_VERSION",
    "PPO_OPTIMIZER_METADATA",
    "PPOOutput",
    "PPOModule",
    "SHARED_TRUNK_HIDDEN_SIZE",
    "VALUE_OUTPUT_SIZE",
]
