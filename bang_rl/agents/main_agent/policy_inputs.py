"""Single source of truth for the PPO policy's input features.

Evaluation (``PrimaryEvaluationProtocol``), the GUI artifact agent and the
main-AI training loop must feed the PPO module byte-identical features, or a
model trained in one place would silently see different inputs in another.
``build_policy_inputs`` composes the five public feature blocks from the frozen
intent / emotion modules, RoleFusion and the observation preprocessor. It only
reads the public observation; privileged game state never enters this path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .role_fusion import TARGET_PLAYER_COUNT

POLICY_FEATURE_KEYS = (
    "self_state",
    "board_state",
    "confirmed_role_features",
    "fused_intent",
    "emotion_features",
)


@dataclass(frozen=True, slots=True)
class PolicyInputs:
    """The PPO feature blocks plus the intermediate estimates used by metrics.

    ``features`` holds the five float32 blocks in ``POLICY_FEATURE_KEYS`` order
    with ``emotion_features`` already flattened to ``(4 * emotion_count,)``.
    ``estimated_emotion`` keeps the unflattened ``(4, emotion_count)`` estimate
    that the emotion MSE metric compares with the opponents' labels.
    """

    features: Mapping[str, np.ndarray]
    action_mask: np.ndarray
    raw_intent: np.ndarray
    fused_intent: np.ndarray
    estimated_emotion: np.ndarray
    processed: Any

    def ppo_inputs(self) -> dict[str, np.ndarray]:
        """Mapping accepted by ``PPOModule.act``/``forward`` (blocks + mask)."""

        return {**self.features, "action_mask": self.action_mask}


def policy_emotion_dim(emotion_module: Any) -> int:
    """Flattened emotion block size the PPO module needs for this emotion module."""

    return int(TARGET_PLAYER_COUNT * int(emotion_module.emotion_count))


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _observation_values(observation: Any) -> Mapping[str, Any]:
    if isinstance(observation, Mapping):
        return observation
    if hasattr(observation, "to_observation_dict"):
        return observation.to_observation_dict()
    return {key: observation[key] for key in observation.keys()}


def build_policy_inputs(
    observation: Any,
    *,
    preprocessor: Any,
    intent: Any,
    emotion: Any,
    role_fusion: Any,
    viewer_idx: int,
) -> PolicyInputs:
    """Compute the PPO input blocks for ``viewer_idx`` from a public observation."""

    processed = preprocessor.preprocess(observation)
    with torch.inference_mode():
        raw_intent = intent.predict_proba(processed)
        estimated_emotion = emotion.predict_proba(processed)
    raw_intent_array = _to_numpy(raw_intent)
    estimated_emotion_array = _to_numpy(estimated_emotion)
    values = _observation_values(observation)
    fused = role_fusion.fuse(
        raw_intent_array,
        role_visibility=values["role_visibility"],
        revealed_roles=values["revealed_roles"],
        confirmed_role_distribution=values["confirmed_role_distribution"],
        self_role=values["self_role"],
        sheriff_idx=values["sheriff_idx"],
        viewer_idx=int(viewer_idx),
    )
    fused_array = np.asarray(fused, dtype=np.float32)
    features = {
        "self_state": np.asarray(processed.state_vector, dtype=np.float32),
        "board_state": np.asarray(processed.player_state_matrix, dtype=np.float32),
        "confirmed_role_features": np.asarray(processed.role_beliefs, dtype=np.float32),
        "fused_intent": fused_array,
        "emotion_features": estimated_emotion_array.reshape(-1).astype(np.float32),
    }
    return PolicyInputs(
        features=features,
        action_mask=np.asarray(processed.action_mask, dtype=np.uint8),
        raw_intent=raw_intent_array,
        fused_intent=fused_array,
        estimated_emotion=estimated_emotion_array,
        processed=processed,
    )


__all__ = [
    "POLICY_FEATURE_KEYS",
    "PolicyInputs",
    "build_policy_inputs",
    "policy_emotion_dim",
]
