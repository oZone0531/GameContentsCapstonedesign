"""Use the main agent's trained role inference as a dummy's beliefs.

A dummy's role beliefs normally come from the public rule-based estimator.
:class:`LearnedBeliefSource` replaces them with the main agent's trained
intent inference (a Phase-2 ``.phase2model`` artifact): at each boundary the
dummy's *own public observation* is preprocessed exactly as for the main agent,
the intent module predicts a role distribution for each of the four other
seats, public confirmed roles are fused in, and the result becomes the rows of
the dummy's belief matrix. Nothing privileged is read.

This closes the loop the team planned: dummies generate data -> the main agent
learns to infer roles -> that inference is plugged back into the dummies ->
the dummies generate the next, better-informed dataset.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

PLAYER_COUNT = 5
ROLE_COUNT = 5

__all__ = ["LearnedBeliefSource", "load_learned_beliefs"]


@lru_cache(maxsize=8)
def _load(path: str) -> Any:
    from bang_rl.training.inference_artifact import load_phase2_inference_artifact

    return load_phase2_inference_artifact(path)


def load_learned_beliefs(path: str | Path) -> Any:
    """Load (once per process) a Phase-2 artifact for use as dummy beliefs."""

    return _load(str(Path(path).resolve()))


def _to_canonical(row: np.ndarray) -> np.ndarray:
    """A 5-way canonical row from a 5-way or 4-way (outlaws merged) row."""

    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size == ROLE_COUNT:
        return row
    if row.size == 4:
        return np.array([row[0], row[1], row[2] / 2.0, row[2] / 2.0, row[3]])
    raise ValueError("role distribution must have 4 or 5 entries")


class LearnedBeliefSource:
    """Beliefs from a trained intent inference module.

    ``weight`` blends with the rule estimator: 1.0 replaces it, 0.5 averages
    the two. The dummy's own row and publicly settled rows stay with the rule
    estimator, and roles the rule estimator rules out (public facts and the
    dummy's own role) stay impossible. Predictions are cached per environment
    step because one boundary reads the beliefs several times.
    """

    name = "learned"

    def __init__(self, artifact: Any, *, weight: float = 1.0, label: str = "") -> None:
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("learned belief weight must be in [0, 1]")
        self.artifact = artifact
        self.weight = weight
        self.label = label
        self.player_idx: int | None = None
        self._cached_step: int | None = None
        self._cached_rows: np.ndarray | None = None

    def on_episode_start(self, player_idx: int, observation: Any | None) -> None:
        self.player_idx = int(player_idx)
        self._cached_step = None
        self._cached_rows = None

    @staticmethod
    def _step_of(observation: Any) -> int | None:
        try:
            return int(np.asarray(observation["environment_step"]).reshape(-1)[0])
        except (KeyError, TypeError, IndexError):
            return None

    def _predict(self, observation: Any) -> np.ndarray:
        processed = self.artifact.preprocessor.preprocess(observation)
        with torch.inference_mode():
            raw = self.artifact.intent.predict_proba(processed)
        raw = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
        values = {key: observation[key] for key in observation.keys()}
        fused = self.artifact.role_fusion.fuse(
            raw,
            role_visibility=values["role_visibility"],
            revealed_roles=values["revealed_roles"],
            confirmed_role_distribution=values["confirmed_role_distribution"],
            self_role=values["self_role"],
            sheriff_idx=values["sheriff_idx"],
            viewer_idx=self.player_idx,
        )
        streams = np.asarray(fused, dtype=np.float64).reshape(4, -1)
        rows = np.zeros((PLAYER_COUNT, ROLE_COUNT))
        others = [seat for seat in range(PLAYER_COUNT) if seat != self.player_idx]
        for stream, seat in enumerate(others):
            rows[seat] = _to_canonical(streams[stream])
        return rows

    def beliefs(self, rule_beliefs: np.ndarray, observation: Any | None) -> np.ndarray:
        rule = np.asarray(rule_beliefs, dtype=np.float64)
        if rule.shape != (PLAYER_COUNT, ROLE_COUNT):
            raise ValueError("rule beliefs must have shape (5, 5)")
        if observation is None or self.player_idx is None:
            return rule.astype(np.float32)
        step = self._step_of(observation)
        if step is None or step != self._cached_step or self._cached_rows is None:
            self._cached_rows = self._predict(observation)
            self._cached_step = step
        mixed = (1.0 - self.weight) * rule + self.weight * self._cached_rows
        # The intent model reads only the event history, so it can give mass
        # to roles the dummy already knows are impossible (the public sheriff's
        # role on another seat, a second deputy when the dummy is the deputy).
        # The rule estimator's zeros are exactly those public and own-role
        # facts, so they are kept.
        mixed = np.where(rule > 0.0, mixed, 0.0)
        mixed[self.player_idx] = rule[self.player_idx]
        settled = rule.max(axis=1) >= 1.0 - 1e-6  # public facts win
        mixed[settled] = rule[settled]
        totals = mixed.sum(axis=1, keepdims=True)
        mixed = np.divide(mixed, totals, out=np.full_like(mixed, 1.0 / ROLE_COUNT), where=totals > 0)
        return mixed.astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {"source": self.name, "weight": self.weight, "model": self.label}
