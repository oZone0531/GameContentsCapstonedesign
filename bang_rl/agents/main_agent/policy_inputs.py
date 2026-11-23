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

RELATIVE_LAYOUT = "relative"
SEAT_ALIGNED_LAYOUT = "seat_aligned"
POLICY_INPUT_LAYOUTS = (RELATIVE_LAYOUT, SEAT_ALIGNED_LAYOUT)
# Seat-aligned self_state appends one-hot own role and one-hot own seat.
SEAT_ALIGNED_SELF_STATE_DIM = STATE_VECTOR_SIZE + ROLE_COUNT + PLAYER_COUNT

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


def policy_emotion_dim(emotion_module: Any, layout: str = RELATIVE_LAYOUT) -> int:
    """Flattened emotion block size the PPO module needs for this emotion module."""

    rows = PLAYER_COUNT if layout == SEAT_ALIGNED_LAYOUT else TARGET_PLAYER_COUNT
    return int(rows * int(emotion_module.emotion_count))


def ppo_dims_for_layout(emotion_module: Any, layout: str) -> dict[str, int]:
    """Keyword arguments for ``PPOModule`` that match ``layout``."""

    if layout not in POLICY_INPUT_LAYOUTS:
        raise ValueError(f"unknown policy input layout: {layout!r}")
    if layout == RELATIVE_LAYOUT:
        return {"emotion_dim": policy_emotion_dim(emotion_module)}
    return {
        "self_state_dim": SEAT_ALIGNED_SELF_STATE_DIM,
        "intent_dim": PLAYER_COUNT * ROLE_COUNT,
        "emotion_dim": policy_emotion_dim(emotion_module, SEAT_ALIGNED_LAYOUT),
    }


def policy_input_layout(ppo: Any) -> str:
    """Infer the layout a PPO module was built for from its block sizes."""

    dims = getattr(ppo, "_component_dims", None) or {}
    if int(dims.get("fused_intent", 0)) == PLAYER_COUNT * ROLE_COUNT:
        return SEAT_ALIGNED_LAYOUT
    return RELATIVE_LAYOUT


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
    layout: str = RELATIVE_LAYOUT,
) -> PolicyInputs:
    """Compute the PPO input blocks for ``viewer_idx`` from a public observation.

    ``layout="seat_aligned"`` puts other-player intent/emotion rows at their
    absolute seats, the same order as the PPO action slots (own row zero), and
    appends the viewer's own role and seat as one-hot vectors to ``self_state``.
    """

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
    self_state = np.asarray(processed.state_vector, dtype=np.float32)
    intent_block = fused_array
    emotion_block = estimated_emotion_array.reshape(-1).astype(np.float32)
    if layout == SEAT_ALIGNED_LAYOUT:
        viewer = int(viewer_idx)
        self_role = int(np.asarray(values["self_role"]).reshape(-1)[0])
        self_state = np.concatenate(
            [self_state.reshape(-1), _one_hot(self_role, ROLE_COUNT), _one_hot(viewer, PLAYER_COUNT)]
        )
        intent_block = _seat_rows(fused_array, viewer).reshape(-1)
        emotion_block = _seat_rows(estimated_emotion_array.astype(np.float32), viewer).reshape(-1)
    elif layout != RELATIVE_LAYOUT:
        raise ValueError(f"unknown policy input layout: {layout!r}")
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
    "POLICY_INPUT_LAYOUTS",
    "RELATIVE_LAYOUT",
    "SEAT_ALIGNED_LAYOUT",
    "policy_input_layout",
    "ppo_dims_for_layout",
    "PolicyInputs",
    "build_policy_inputs",
    "policy_emotion_dim",
]
