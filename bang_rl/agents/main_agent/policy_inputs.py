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

from .preprocessor import STATE_VECTOR_SIZE
from .role_fusion import PLAYER_COUNT, ROLE_COUNT, TARGET_PLAYER_COUNT

# self_state appends a one-hot of the viewer's own role and a one-hot of the
# viewer's own seat to the preprocessor's state vector.
SELF_STATE_DIM = STATE_VECTOR_SIZE + ROLE_COUNT + PLAYER_COUNT

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
    with the ``fused_intent`` and ``emotion_features`` blocks laid out per
    absolute seat and flattened to ``(5 * 5,)`` and ``(5 * emotion_width,)``,
    where ``emotion_width`` is ``emotion_count`` plus the number of personality
    presets when the emotion module has a personality head (see
    :func:`policy_emotion_width`).
    The ``fused_intent`` and ``estimated_emotion`` attributes keep the
    four-opponent ``(4, ...)`` estimates that the metrics compare with the
    opponents' labels.
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


def policy_personality_count(emotion_module: Any) -> int:
    """Number of personality classes the emotion module predicts (0 = no personality head)."""

    return len(tuple(getattr(emotion_module, "personality_names", ()) or ()))


def policy_emotion_width(emotion_module: Any) -> int:
    """Per-seat emotion row width: emotion values plus personality probabilities."""

    return int(emotion_module.emotion_count) + policy_personality_count(emotion_module)


def policy_emotion_dim(emotion_module: Any) -> int:
    """Flattened emotion block size the PPO module needs for this emotion module.

    The block has one row per absolute seat (the viewer's own row is zero). Each
    row holds the estimated emotions followed by the estimated personality
    probabilities when the emotion module has a personality head.
    """

    return int(PLAYER_COUNT * policy_emotion_width(emotion_module))


def policy_ppo_dims(emotion_module: Any) -> dict[str, int]:
    """Keyword arguments that size a ``PPOModule`` for :func:`build_policy_inputs`."""

    return {
        "self_state_dim": SELF_STATE_DIM,
        "intent_dim": PLAYER_COUNT * ROLE_COUNT,
        "emotion_dim": policy_emotion_dim(emotion_module),
    }


def _one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vector[index] = 1.0
    return vector


def _seat_rows(block: np.ndarray, viewer_idx: int) -> np.ndarray:
    """Place the four other-player rows at their absolute seats (own row zero)."""

    rows = np.zeros((PLAYER_COUNT, block.shape[-1]), dtype=np.float32)
    others = [seat for seat in range(PLAYER_COUNT) if seat != viewer_idx]
    rows[others] = block.reshape(TARGET_PLAYER_COUNT, -1)
    return rows


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
    """Compute the PPO input blocks for ``viewer_idx`` from a public observation.

    Other-player intent/emotion rows sit at their absolute seats, the same
    order as the PPO action slots (the viewer's own row is zero), and the
    viewer's own role and seat are appended to ``self_state`` as one-hot vectors.
    """

    processed = preprocessor.preprocess(observation)
    with torch.inference_mode():
        raw_intent = intent.predict_proba(processed)
        if policy_personality_count(emotion):
            estimated_emotion, estimated_personality = emotion.predict_emotion_and_personality(processed)
        else:
            estimated_emotion, estimated_personality = emotion.predict_proba(processed), None
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
    viewer = int(viewer_idx)
    self_role = int(np.asarray(values["self_role"]).reshape(-1)[0])
    self_state = np.concatenate(
        [
            np.asarray(processed.state_vector, dtype=np.float32).reshape(-1),
            _one_hot(self_role, ROLE_COUNT),
            _one_hot(viewer, PLAYER_COUNT),
        ]
    )
    intent_block = _seat_rows(fused_array, viewer).reshape(-1)
    emotion_rows = estimated_emotion_array.astype(np.float32)
    if estimated_personality is not None:
        # Personality probabilities ride along in each seat's emotion row.
        emotion_rows = np.concatenate(
            [emotion_rows, _to_numpy(estimated_personality).astype(np.float32).reshape(emotion_rows.shape[0], -1)],
            axis=-1,
        )
    emotion_block = _seat_rows(emotion_rows, viewer).reshape(-1)
    features = {
        "self_state": self_state,
        "board_state": np.asarray(processed.player_state_matrix, dtype=np.float32),
        "confirmed_role_features": np.asarray(processed.role_beliefs, dtype=np.float32),
        "fused_intent": intent_block,
        "emotion_features": emotion_block,
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
    "SELF_STATE_DIM",
    "PolicyInputs",
    "build_policy_inputs",
    "policy_emotion_dim",
    "policy_emotion_width",
    "policy_personality_count",
    "policy_ppo_dims",
]
