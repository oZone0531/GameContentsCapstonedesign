"""Runtime adapter that turns MainAgent actions into PPO transitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bang_rl.agents.main_agent.preprocessor import ObservationPreprocessor

from .ppo_trainer import PPOTransition


class PPOTrainingAgent:
    """Use a PPOModule for one seat and record committed step transitions."""

    def __init__(self, module: Any, *, player_idx: int = 0, seed: int = 0, phase2_artifact: Any = None) -> None:
        self.module = module
        self.phase2_artifact = phase2_artifact
        self.player_idx = int(player_idx)
        self.seed = int(seed)
        self.worker_id = 0
        self.agent_id = f"main_player_{self.player_idx}"
        self.preprocessor = ObservationPreprocessor()
        self._pending: dict[str, Any] | None = None
        self._episode_id = ""
        self._transition_id = 0

    def on_episode_start(self, context: Any) -> None:
        self._episode_id = str(getattr(context, "episode_id", ""))
        self._transition_id = 0
        self._pending = None

    def on_episode_end(self, _result: Any) -> None:
        self._pending = None

    def _features(self, observation: Any) -> tuple[dict[str, Any], np.ndarray]:
        processed = self.phase2_artifact.preprocessor.preprocess(observation) if self.phase2_artifact is not None else self.preprocessor.preprocess(observation)
        if self.phase2_artifact is not None:
            intent = self.phase2_artifact.intent.predict_proba(processed)
            emotion = self.phase2_artifact.emotion.predict_proba(processed)
            fused = self.phase2_artifact.role_fusion.fuse(
                intent,
                observation.get("role_visibility"),
                observation.get("revealed_roles"),
                observation.get("confirmed_role_distribution"),
                viewer_idx=observation.get("viewer_idx", observation.get("current_player_idx")),
            )
            fused = fused.detach().cpu().numpy() if hasattr(fused, "detach") else fused
            emotion = emotion.detach().cpu().numpy() if hasattr(emotion, "detach") else emotion
            fused = np.asarray(fused, dtype=np.float32).reshape(4, 4)
            emotion = np.asarray(emotion, dtype=np.float32).reshape(-1)
        else:
            fused = np.zeros((4, 4), dtype=np.float32)
            emotion = np.zeros((4,), dtype=np.float32)
        features = {
            "self_state": processed.state_vector,
            "board_state": processed.player_state_matrix,
            "confirmed_role_features": processed.role_beliefs,
            "fused_intent": fused,
            "emotion_features": emotion,
        }
        mask = np.asarray(observation["action_mask"], dtype=np.uint8).reshape(6).copy()
        return features, mask

    def act(self, observation: Any, action_mask: Any) -> int:
        features, mask = self._features(observation)
        output = self.module.act(features, mask, deterministic=False)
        self._pending = {
            "features": features,
            "action": int(output.action.detach().cpu().item()),
            "action_mask": mask,
            "old_log_prob": float(output.log_prob.detach().cpu().item()),
            "value": float(output.value.detach().cpu().reshape(-1)[0].item()),
        }
        return int(self._pending["action"])

    def on_step(
        self,
        observation: Any,
        action: int,
        action_mask: Any,
        result: Any,
        next_observation: Any,
    ) -> PPOTransition | None:
        if self._pending is None:
            return None
        features = self._pending["features"]
        next_features, next_mask = self._features(next_observation)
        terminated = bool(getattr(result, "terminated", False))
        truncated = bool(getattr(result, "truncated", False))
        next_value = None
        if not terminated and not truncated:
            next_value = float(self.module.value(next_features, next_mask).detach().cpu().reshape(-1)[0].item())
        transition = PPOTransition(
            features=features,
            action=int(action),
            action_mask=np.asarray(action_mask, dtype=np.uint8).copy(),
            old_log_prob=self._pending["old_log_prob"],
            value=self._pending["value"],
            reward=float(getattr(result, "reward", 0.0)),
            next_features=next_features,
            next_action_mask=next_mask,
            next_value=next_value,
            terminated=terminated,
            truncated=truncated,
            trainable=not truncated,
            worker_id=int(self.worker_id),
            episode_id=self._episode_id,
            transition_id=self._transition_id,
            environment_step=int(getattr(result, "environment_step", 0)),
            environment_steps_spanned=1,
        )
        self._transition_id += 1
        self._pending = None
        return transition

    def sync_from(self, source: Any) -> None:
        self.module.load_state_dict(source.state_dict())


__all__ = ["PPOTrainingAgent"]
