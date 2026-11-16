"""Inference accuracy of intent/emotion module pairs under a fixed PPO.

The main AI's purpose is to infer the dummies' roles and emotions while it plays
among four dummies. This module measures exactly that: the PPO of a ``.bangmodel``
stays fixed, a module pair is plugged in, and the deterministic policy plays the
fixed main-AI evaluation deals (``MAIN_AI_EVAL_BASE_SEED + i``, seat ``i % 5``),
the same deals main-AI training evaluates on.

Swapping the modules changes the inputs the PPO receives, so its play may shift.
That is the real deployment effect and is intended; the win rate is reported as
a secondary number.

Reported per pair: ``role_accuracy``, ``hidden_role_accuracy``, ``emotion_mse``
(all from :class:`InferenceMetrics`), plus the accuracy per target role and the
MSE per emotion entry, and the win rate.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
from bang_rl.agents.main_agent import policy_emotion_dim
from bang_rl.config import SystemConfig
from bang_rl.data.emotion_slots import emotion_value_mask
from bang_rl.evaluation.protocol import (
    PRIMARY_SEATS,
    SEMANTIC_ROLE_NAMES,
    SEMANTIC_ROLE_OF,
    InferenceMetrics,
    opponent_emotion_target,
    semantic_role_prediction,
)
from bang_rl.exceptions import TrainingError

from . import module_artifacts
from .main_ai_training import (
    MAIN_AI_EVAL_BASE_SEED,
    MAIN_AI_EVAL_GAMES,
    EnvironmentFactory,
    PolicyModules,
    _freeze,
    evaluate_policy,
    validate_emotion_contract,
)

INFERENCE_SCALAR_KEYS = ("role_accuracy", "hidden_role_accuracy", "emotion_mse", "win_rate")


@dataclass(slots=True)
class DetailedInferenceMetrics(InferenceMetrics):
    """:class:`InferenceMetrics` plus per-target-role accuracy and per-emotion MSE."""

    role_correct_by_target: list[int] = field(default_factory=lambda: [0] * len(SEMANTIC_ROLE_NAMES))
    role_total_by_target: list[int] = field(default_factory=lambda: [0] * len(SEMANTIC_ROLE_NAMES))
    hidden_correct_by_target: list[int] = field(default_factory=lambda: [0] * len(SEMANTIC_ROLE_NAMES))
    hidden_total_by_target: list[int] = field(default_factory=lambda: [0] * len(SEMANTIC_ROLE_NAMES))
    emotion_squared_error_by_entry: np.ndarray = field(default_factory=lambda: np.zeros(len(LABEL_NAMES)))
    emotion_count_by_entry: np.ndarray = field(default_factory=lambda: np.zeros(len(LABEL_NAMES)))

    def observe_step(
        self,
        *,
        env: Any,
        observation: Any,
        seat: int,
        fused_intent: Any,
        estimated_emotion: Any,
        opponents: Mapping[int, Any],
    ) -> None:
        # ``super()`` does not work in a slots dataclass; call the base directly.
        InferenceMetrics.observe_step(
            self,
            env=env,
            observation=observation,
            seat=seat,
            fused_intent=fused_intent,
            estimated_emotion=estimated_emotion,
            opponents=opponents,
        )
        targets = [index for index in range(5) if index != seat]
        roles = env.state.roles  # metrics-only privileged read
        visibility = np.asarray(observation["role_visibility"]).reshape(-1)
        for stream, target in enumerate(targets):
            actual = SEMANTIC_ROLE_OF[int(roles[target])]
            correct = int(semantic_role_prediction(fused_intent[stream]) == actual)
            self.role_total_by_target[actual] += 1
            self.role_correct_by_target[actual] += correct
            if int(visibility[target]) == 0:
                self.hidden_total_by_target[actual] += 1
                self.hidden_correct_by_target[actual] += correct
        alive_state = getattr(env.state, "alive", None)
        alive = np.ones(5, dtype=bool) if alive_state is None else np.asarray(alive_state, dtype=bool)
        for stream, target in enumerate(targets):
            if not alive[target]:
                continue
            entry = opponent_emotion_target(opponents[target])
            if entry is None:
                continue
            values, names = entry
            predicted = np.asarray(estimated_emotion[stream], dtype=np.float64).reshape(-1)
            if values.shape != predicted.shape or values.size != len(LABEL_NAMES):
                continue
            used = emotion_value_mask(names, np.array(target), alive)
            error = np.square(predicted - values)
            self.emotion_squared_error_by_entry += np.where(used, error, 0.0)
            self.emotion_count_by_entry += used.astype(np.float64)

    def detail(self) -> dict[str, Any]:
        def rate(correct: list[int], total: list[int]) -> dict[str, float | None]:
            return {
                name: (correct[index] / total[index]) if total[index] else None
                for index, name in enumerate(SEMANTIC_ROLE_NAMES)
            }

        counts = self.emotion_count_by_entry
        return {
            "role_accuracy_by_target_role": rate(self.role_correct_by_target, self.role_total_by_target),
            "hidden_role_accuracy_by_target_role": rate(self.hidden_correct_by_target, self.hidden_total_by_target),
            "role_counts_by_target_role": dict(zip(SEMANTIC_ROLE_NAMES, self.role_total_by_target)),
            "emotion_mse_by_entry": {
                name: (float(self.emotion_squared_error_by_entry[index] / counts[index]) if counts[index] else None)
                for index, name in enumerate(LABEL_NAMES)
            },
        }


def check_modules_fit_ppo(ppo: Any, emotion: Any) -> None:
    """The PPO was built for one emotion input size; a swapped pair must match it."""

    expected = int(ppo.component_dims["emotion"])
    actual = int(policy_emotion_dim(emotion))
    if expected != actual:
        raise TrainingError(
            f"감정 모듈의 출력 크기({actual})가 PPO 입력({expected})과 다릅니다. "
            f"The emotion module output size ({actual}) does not match the PPO input ({expected})."
        )


def evaluate_inference(
    policy: PolicyModules,
    config: SystemConfig,
    *,
    games: int = MAIN_AI_EVAL_GAMES,
    base_seed: int = MAIN_AI_EVAL_BASE_SEED,
    seats: Sequence[int] = PRIMARY_SEATS,
    environment_factory: EnvironmentFactory | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """Play the fixed deals with ``policy`` and report inference accuracy (None if stopped)."""

    check_modules_fit_ppo(policy.ppo, policy.emotion)
    return evaluate_policy(
        policy,
        config,
        games=games,
        base_seed=base_seed,
        seats=seats,
        environment_factory=environment_factory,
        should_stop=should_stop,
        metrics_factory=DetailedInferenceMetrics,
    )


def inference_scalars(evaluation: Mapping[str, Any]) -> dict[str, float | None]:
    """The headline numbers of one evaluation (None when a metric had no samples)."""

    return {key: evaluation.get(key) for key in INFERENCE_SCALAR_KEYS}


@dataclass(frozen=True, slots=True)
class ModulePair:
    """One intent + emotion module pair to evaluate."""

    label: str
    intent: Any
    emotion: Any
    intent_path: str | None = None
    emotion_path: str | None = None


def load_module_pair(label: str, intent_path: str | Path, emotion_path: str | Path) -> ModulePair:
    """Load two ``.phase2model`` files as a frozen pair (emotion contract checked)."""

    intent, _ = module_artifacts.load_intent_for_resume(intent_path)
    emotion, _ = module_artifacts.load_emotion_for_resume(emotion_path)
    validate_emotion_contract(emotion)
    _freeze(intent)
    _freeze(emotion)
    return ModulePair(label, intent, emotion, str(intent_path), str(emotion_path))


def evaluate_module_pairs(
    base: Any,
    pairs: Sequence[ModulePair],
    config: SystemConfig,
    *,
    games: int = MAIN_AI_EVAL_GAMES,
    base_seed: int = MAIN_AI_EVAL_BASE_SEED,
    include_base: bool = True,
    environment_factory: EnvironmentFactory | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate the base model's own modules and each pair under the base PPO.

    ``base`` is a loaded ``.bangmodel`` (``load_final_inference_artifact``). The
    PPO, role fusion and preprocessor are the base model's and never change.
    Returns one row per evaluated pair (stops early when ``should_stop`` fires).
    """

    candidates: list[ModulePair] = []
    if include_base:
        candidates.append(ModulePair("base", base.intent, base.emotion))
    candidates.extend(pairs)
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(candidates):
        if progress is not None:
            progress(index, len(candidates), pair.label)
        policy = PolicyModules(pair.intent, pair.emotion, base.role_fusion, base.preprocessor, base.ppo)
        evaluation = evaluate_inference(
            policy, config, games=games, base_seed=base_seed,
            environment_factory=environment_factory, should_stop=should_stop,
        )
        if evaluation is None:
            break
        rows.append(
            {
                "label": pair.label,
                "intent_path": pair.intent_path,
                "emotion_path": pair.emotion_path,
                "games": evaluation["games"],
                "truncated_games": evaluation["truncated_games"],
                **inference_scalars(evaluation),
                **dict(evaluation.get("inference_detail", {})),
                "by_role_win_rate": {name: values["win_rate"] for name, values in evaluation["by_role"].items()},
            }
        )
    if progress is not None:
        progress(len(rows), len(candidates), "")
    return rows


__all__ = [
    "DetailedInferenceMetrics",
    "INFERENCE_SCALAR_KEYS",
    "ModulePair",
    "check_modules_fit_ppo",
    "evaluate_inference",
    "evaluate_module_pairs",
    "inference_scalars",
    "load_module_pair",
]
