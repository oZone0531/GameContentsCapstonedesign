"""PPO rollout collection and in-memory policy updates.

This module is deliberately downstream of :mod:`pending_transition` and the
vector worker runtime.  It consumes committed MainAgent transitions, never
creates transitions for cleanup/faults, and keeps rollout/optimizer state in
memory only.  Normal terminal transitions are trainable without a bootstrap;
truncated transitions are retained by the ledger for diagnostics but are
excluded from every PPO calculation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np
import torch
from torch import Tensor

from bang_rl.config import JobSnapshot, PPOConfig
from bang_rl.exceptions import TrainingError
from bang_rl.runtime.pause_coordinator import WorkerCancelled
from bang_rl.runtime.metadata import build_run_metadata

from .pending_transition import MainAgentTransition, PendingTransitionLedger

PPO_TRAINER_FORMAT_VERSION = "1.0"
ACTION_SPACE_SIZE = 6


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mask(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.uint8).reshape(-1)
    if result.shape != (ACTION_SPACE_SIZE,):
        raise ValueError("action_mask must have shape (6,)")
    if not np.isin(result, (0, 1)).all() or not result.any():
        raise ValueError("action_mask must contain binary values and enable an action")
    return result.copy()


def _action(value: Any) -> int:
    result = int(value)
    if result < 0 or result >= ACTION_SPACE_SIZE:
        raise ValueError("action must be in [0, 5]")
    return result


def _normalize_action(value: Any, *, actor_idx: int) -> int:
    """Decode scalar and domain Action mappings into Discrete(6).

    Ledger records normally contain ``Action.to_dict()``.  A scalar ``action``
    field remains supported for older callers, but when both representations
    are present they must describe the same fixed-space action.
    """

    if not isinstance(value, Mapping):
        return _action(value)

    scalar = _action(value["action"]) if "action" in value else None
    mapped_actor = value.get("actor_idx")
    if mapped_actor is not None:
        if isinstance(mapped_actor, (bool, np.bool_)) or int(mapped_actor) != int(actor_idx):
            raise ValueError("action actor_idx is inconsistent with the transition actor")

    if "action_type" not in value:
        if scalar is None:
            raise ValueError("action mapping requires action or action_type")
        return scalar

    action_type = value["action_type"]
    if isinstance(action_type, str):
        normalized_type = action_type.strip().upper()
        if normalized_type in {"PASS", "0"}:
            action_kind = 0
        elif normalized_type in {"BANG", "1"}:
            action_kind = 1
        else:
            raise ValueError("action_type must be PASS or BANG")
    else:
        if isinstance(action_type, (bool, np.bool_)):
            raise TypeError("action_type must be PASS or BANG")
        try:
            action_kind = int(action_type)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("action_type must be PASS or BANG") from exc
        if action_kind not in (0, 1):
            raise ValueError("action_type must be PASS or BANG")

    target = value.get("target_idx")
    if action_kind == 0:
        if target is not None:
            raise ValueError("PASS action must have target_idx=None")
        decoded = 0
    else:
        if target is None or isinstance(target, (bool, np.bool_)):
            raise ValueError("BANG action requires an integer target_idx")
        try:
            target_index = int(target)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("BANG target_idx must be an integer") from exc
        if target_index < 0 or target_index >= ACTION_SPACE_SIZE - 1:
            raise ValueError("BANG target_idx must be in [0, 4]")
        decoded = target_index + 1

    if scalar is not None and scalar != decoded:
        raise ValueError("scalar action and domain Action mapping are inconsistent")
    return decoded


def _detached(value: Any) -> Any:
    """Copy feature containers without exposing mutable rollout storage."""

    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_detached(item) for item in value)
    return value


def _as_features(value: Any, *, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        # The PPOModule accepts public structured feature blocks.  Do not
        # permit reward/private labels to be smuggled through this boundary.
        forbidden = {
            "reward", "rewards", "reward_breakdown", "role", "roles",
            "true_role", "true_roles", "emotion_label", "emotion_labels",
            "true_emotion", "privileged_state", "private_observation",
        }
        leaked = sorted(str(key) for key in value if str(key).lower() in forbidden)
        if leaked:
            raise TrainingError(f"{name} contains forbidden PPO feature fields: {', '.join(leaked)}")
        return _detached(value)
    try:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric PPO features") from exc
    if tensor.ndim == 0 or tensor.dtype == torch.bool:
        raise ValueError(f"{name} must be a feature vector")
    if not torch.isfinite(tensor.to(dtype=torch.float32)).all():
        raise ValueError(f"{name} must contain finite values")
    return tensor.detach().clone()


@dataclass(frozen=True, slots=True)
class PPOTransition:
    """PPO-ready view of one committed MainAgent transition.

    ``features`` and ``next_features`` are the public, already-processed PPO
    inputs.  ``next_value`` is an optional caller-supplied value useful at a
    rollout boundary; it is never used for terminated or truncated records.
    """

    features: Any
    action: int
    action_mask: np.ndarray
    old_log_prob: float
    value: float
    reward: float
    next_features: Any = None
    next_action_mask: np.ndarray | None = None
    next_value: float | None = None
    terminated: bool = False
    truncated: bool = False
    trainable: bool = True
    worker_id: int = 0
    episode_id: str = ""
    transition_id: int = 0
    environment_step: int = 0
    rollout_id: str = ""
    main_agent_eliminated: bool = False
    environment_steps_spanned: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _as_features(self.features, name="features"))
        object.__setattr__(self, "next_features", _as_features(self.next_features, name="next_features"))
        object.__setattr__(self, "action", _action(self.action))
        object.__setattr__(self, "action_mask", _mask(self.action_mask))
        if self.next_action_mask is not None:
            object.__setattr__(self, "next_action_mask", _mask(self.next_action_mask))
        object.__setattr__(self, "old_log_prob", _finite(self.old_log_prob, "old_log_prob"))
        object.__setattr__(self, "value", _finite(self.value, "value"))
        object.__setattr__(self, "reward", _finite(self.reward, "reward"))
        if self.next_value is not None:
            object.__setattr__(self, "next_value", _finite(self.next_value, "next_value"))
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated cannot both be true")
        if self.truncated and self.trainable:
            object.__setattr__(self, "trainable", False)

    @property
    def log_prob(self) -> float:
        return self.old_log_prob

    @property
    def old_value(self) -> float:
        return self.value

    @classmethod
    def from_committed(cls, transition: MainAgentTransition) -> "PPOTransition":
        if not isinstance(transition, MainAgentTransition):
            raise TypeError("from_committed requires a MainAgentTransition")
        state = transition.state
        if state is None:
            if isinstance(transition.observation, Mapping):
                state = transition.observation.get("features")
                if state is None and any(key in transition.observation for key in ("self_state", "state_vector")):
                    state = transition.observation
            else:
                state = transition.observation
        next_state = None
        if transition.next_observation is not None:
            if isinstance(transition.next_observation, Mapping):
                next_state = transition.next_observation.get("features")
                if next_state is None and any(key in transition.next_observation for key in ("self_state", "state_vector")):
                    next_state = transition.next_observation
            else:
                next_state = transition.next_observation
        metadata = dict(transition.metadata)
        next_state = metadata.get("next_features", next_state)
        next_mask = metadata.get("next_action_mask")
        next_value = metadata.get("next_value")
        if state is None:
            raise TrainingError("committed transition has no PPO feature state")
        if transition.log_prob is None or transition.value is None:
            raise TrainingError("committed transition is missing rollout log_prob/value")
        return cls(
            features=state,
            action=_normalize_action(transition.action, actor_idx=transition.actor_idx),
            action_mask=transition.action_mask,
            old_log_prob=transition.log_prob,
            value=transition.value,
            reward=transition.accumulated_reward,
            next_features=next_state,
            next_action_mask=next_mask,
            next_value=next_value,
            terminated=transition.terminated,
            truncated=transition.truncated,
            trainable=transition.trainable,
            worker_id=transition.worker_id,
            episode_id=transition.episode_id,
            transition_id=transition.transition_id,
            environment_step=transition.environment_step,
            rollout_id=str(metadata.get("rollout_id", "")),
            main_agent_eliminated=transition.main_agent_eliminated,
            environment_steps_spanned=transition.environment_steps_spanned,
        )

    from_ledger = from_committed


RolloutTransition = PPOTransition


class RolloutBuffer:
    """Deterministically ordered, bounded in-memory PPO transition buffer."""

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and (isinstance(capacity, bool) or int(capacity) < 1):
            raise ValueError("capacity must be a positive integer")
        self.capacity = None if capacity is None else int(capacity)
        self._items: list[PPOTransition] = []

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self.ordered())

    @property
    def transitions(self) -> tuple[PPOTransition, ...]:
        return self.ordered()

    @property
    def trainable(self) -> tuple[PPOTransition, ...]:
        return tuple(item for item in self.ordered() if item.trainable and not item.truncated)

    def add(self, item: PPOTransition | MainAgentTransition) -> None:
        transition = item if isinstance(item, PPOTransition) else PPOTransition.from_committed(item)
        if self.capacity is not None and len(self._items) >= self.capacity:
            raise TrainingError("PPO rollout buffer capacity reached")
        self._items.append(transition)

    append = add

    def extend(self, items: Iterable[PPOTransition | MainAgentTransition]) -> None:
        for item in items:
            self.add(item)

    def ordered(self) -> tuple[PPOTransition, ...]:
        return tuple(sorted(self._items, key=lambda item: (item.worker_id, item.episode_id, item.environment_step, item.transition_id)))

    def clear(self) -> None:
        self._items.clear()

    def clear_trainable(self) -> None:
        trainable = {id(item) for item in self.trainable}
        self._items = [item for item in self._items if id(item) not in trainable]

    def collect_from_ledgers(self, ledgers: PendingTransitionLedger | Mapping[int, PendingTransitionLedger] | Sequence[PendingTransitionLedger]) -> tuple[PPOTransition, ...]:
        if isinstance(ledgers, PendingTransitionLedger):
            values = ledgers.committed
        elif isinstance(ledgers, Mapping):
            values = tuple(item for key in sorted(ledgers) for item in ledgers[key].committed)
        else:
            values = tuple(item for ledger in ledgers for item in ledger.committed)
        result = tuple(PPOTransition.from_committed(item) for item in values if item.trainable and not item.truncated)
        self.extend(result)
        return result


# Public aliases used by callers that name the buffer after the update phase.
PPORolloutBuffer = RolloutBuffer


def _array(value: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=dtype)


def compute_gae(
    rewards: Sequence[float] | Tensor,
    values: Sequence[float] | Tensor,
    next_values: Sequence[float] | Tensor,
    terminated: Sequence[bool] | Tensor,
    truncated: Sequence[bool] | Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    trainable: Sequence[bool] | Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute transition-unit GAE and returns.

    Terminated and truncated transitions both have a zero bootstrap.  A
    truncated transition is not trainable, so its advantage/return is zero;
    this makes accidental inclusion visible while keeping the function safe
    for mixed diagnostic batches.  ``next_values`` is supplied explicitly so
    rollout boundaries can bootstrap from the actual next MainAgent state.
    """

    gamma = float(gamma)
    gae_lambda = float(gae_lambda)
    if not 0.0 < gamma <= 1.0 or not 0.0 < gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in (0, 1]")
    reward = _array(rewards).reshape(-1)
    value = _array(values).reshape(-1)
    following = _array(next_values).reshape(-1)
    done = torch.as_tensor(terminated, dtype=torch.bool).reshape(-1)
    limited = torch.as_tensor(truncated, dtype=torch.bool).reshape(-1)
    if not (reward.numel() == value.numel() == following.numel() == done.numel() == limited.numel()):
        raise ValueError("GAE inputs must have equal lengths")
    eligible = torch.ones_like(done)
    if trainable is not None:
        eligible = torch.as_tensor(trainable, dtype=torch.bool).reshape(-1)
        if eligible.numel() != done.numel():
            raise ValueError("trainable must match GAE input length")
    advantages = torch.zeros_like(reward)
    running = torch.zeros((), dtype=reward.dtype, device=reward.device)
    for index in range(reward.numel() - 1, -1, -1):
        if not bool(eligible[index]) or bool(limited[index]):
            running = torch.zeros_like(running)
            continue
        continuation = not bool(done[index]) and not bool(limited[index])
        bootstrap = following[index] if continuation else torch.zeros_like(following[index])
        delta = reward[index] + gamma * bootstrap - value[index]
        running = delta + (gamma * gae_lambda * running if continuation else torch.zeros_like(running))
        advantages[index] = running
    returns = advantages + value
    returns = torch.where(eligible & ~limited, returns, torch.zeros_like(returns))
    return advantages, returns


def gae_and_returns(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
    return compute_gae(*args, **kwargs)


def clipped_surrogate_loss(
    old_log_prob: Tensor | Sequence[float],
    current_log_prob: Tensor | Sequence[float],
    advantages: Tensor | Sequence[float],
    clip_epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return PPO policy loss, ratios, and clip fraction."""

    epsilon = float(clip_epsilon)
    if epsilon <= 0.0 or epsilon >= 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")
    old = _array(old_log_prob).reshape(-1)
    current = _array(current_log_prob).reshape(-1)
    advantage = _array(advantages).reshape(-1)
    if not (old.numel() == current.numel() == advantage.numel()):
        raise ValueError("PPO loss inputs must have equal lengths")
    ratio = torch.exp(current - old)
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    objective = torch.minimum(ratio * advantage, clipped * advantage)
    loss = -objective.mean()
    fraction = (torch.abs(ratio - clipped) > 1e-8).to(torch.float32).mean()
    return loss, ratio, fraction


def value_function_loss(values: Tensor | Sequence[float], returns: Tensor | Sequence[float]) -> Tensor:
    prediction = _array(values).reshape(-1)
    target = _array(returns).reshape(-1)
    if prediction.numel() != target.numel():
        raise ValueError("value loss inputs must have equal lengths")
    return 0.5 * torch.mean((prediction - target) ** 2)


def entropy_bonus(entropy: Tensor | Sequence[float]) -> Tensor:
    return _array(entropy).reshape(-1).mean()


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    grad_norm: float = 0.0
    samples: int = 0
    epochs: int = 0
    minibatches: int = 0
    ppo_update: int = 0
    environment_steps: int = 0
    main_agent_transitions: int = 0
    cancelled: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "metadata"}
        result["metadata"] = dict(self.metadata)
        return result


class PPOTrainer:
    """Collect fixed rollouts and optimize only the supplied PPOModule."""

    def __init__(
        self,
        module: Any,
        *,
        config: PPOConfig | Mapping[str, Any] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        snapshot: JobSnapshot | Mapping[str, Any] | None = None,
        worker_count: int = 1,
        seed: int = 0,
        logger: Any | None = None,
        control: Any | None = None,
        cancel_event: Any | None = None,
        value_loss_coef: float | None = None,
        entropy_coef: float | None = None,
        phase2_artifact: Any | None = None,
        intent_module: torch.nn.Module | None = None,
        emotion_module: torch.nn.Module | None = None,
    ) -> None:
        self.module = module
        self.phase2_artifact = phase2_artifact
        if phase2_artifact is not None:
            intent_module = intent_module or getattr(phase2_artifact, "intent", getattr(phase2_artifact, "intent_inference", None))
            emotion_module = emotion_module or getattr(phase2_artifact, "emotion", getattr(phase2_artifact, "emotion_inference", None))
        self.inference_modules = tuple(item for item in (intent_module, emotion_module) if item is not None)
        for inference_module in self.inference_modules:
            if not isinstance(inference_module, torch.nn.Module):
                raise TypeError("inference modules must be torch.nn.Module instances")
            inference_module.eval()
            for parameter in inference_module.parameters():
                parameter.requires_grad_(False)
        self._snapshot = snapshot
        if config is None and snapshot is not None:
            snapshot_value = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
            snapshot_config = snapshot_value.get("config_snapshot", {})
            if isinstance(snapshot_config, Mapping):
                training_config = snapshot_config.get("rl_training", snapshot_config.get("training", {}))
                if isinstance(training_config, Mapping) and isinstance(training_config.get("ppo"), Mapping):
                    config = training_config["ppo"]
        if config is None:
            config = getattr(module, "config", None) or PPOConfig()
        self.config = config if isinstance(config, PPOConfig) else PPOConfig.model_validate(dict(config))
        if isinstance(worker_count, bool) or int(worker_count) < 1:
            raise ValueError("worker_count must be positive")
        self.worker_count = int(worker_count)
        self.seed = int(seed)
        self.logger = logger
        self.control = control
        self.cancel_event = cancel_event
        self.value_loss_coef = _finite(self.config.value_loss_coef if value_loss_coef is None else value_loss_coef, "value_loss_coef")
        self.entropy_coef = _finite(self.config.entropy_coef if entropy_coef is None else entropy_coef, "entropy_coef")
        if self.value_loss_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("loss coefficients must be non-negative")
        self.optimizer = optimizer or module.create_optimizer()
        module_parameter_ids = {id(parameter) for parameter in module.parameters()}
        optimizer_parameter_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group.get("params", ())
        }
        if not optimizer_parameter_ids.issubset(module_parameter_ids):
            raise TrainingError("PPO optimizer may update PPOModule parameters only")
        self.buffer = RolloutBuffer(self.config.rollout_transitions_per_worker * self.worker_count)
        self.environment_steps = 0
        self.main_agent_transitions = 0
        self.ppo_updates = 0
        self.rollout_index = 0

    @property
    def rollout_buffer(self) -> RolloutBuffer:
        return self.buffer

    def _safe_point(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise WorkerCancelled("PPO cancellation requested")
        if self.control is not None and hasattr(self.control, "safe_point"):
            self.control.safe_point()
        if self.control is not None and bool(getattr(self.control, "cancel_requested", False)):
            raise WorkerCancelled("PPO cancellation requested")

    def collect_rollout(
        self,
        transitions: Iterable[PPOTransition | MainAgentTransition] | None = None,
        *,
        ledgers: PendingTransitionLedger | Mapping[int, PendingTransitionLedger] | Sequence[PendingTransitionLedger] | None = None,
        max_transitions: int | None = None,
    ) -> tuple[PPOTransition, ...]:
        """Collect at most the fixed per-worker rollout size in stable order."""

        self._safe_point()
        if transitions is None and ledgers is None:
            raise ValueError("collect_rollout requires transitions or ledgers")
        if transitions is None:
            if isinstance(ledgers, PendingTransitionLedger):
                transitions = ledgers.committed
            elif isinstance(ledgers, Mapping):
                transitions = (item for key in sorted(ledgers) for item in ledgers[key].committed)
            else:
                transitions = (item for ledger in ledgers or () for item in ledger.committed)
        converted = [item if isinstance(item, PPOTransition) else PPOTransition.from_committed(item) for item in transitions]
        converted = [item for item in converted if item.trainable and not item.truncated]
        converted.sort(key=lambda item: (item.worker_id, item.episode_id, item.environment_step, item.transition_id))
        limit = self.config.rollout_transitions_per_worker * self.worker_count if max_transitions is None else int(max_transitions)
        if limit < 1:
            raise ValueError("max_transitions must be positive")
        selected = tuple(converted[:limit])
        rollout_id = f"rollout-{self.rollout_index}"
        self.rollout_index += 1
        selected = tuple(
            PPOTransition(
                features=item.features,
                action=item.action,
                action_mask=item.action_mask,
                old_log_prob=item.old_log_prob,
                value=item.value,
                reward=item.reward,
                next_features=item.next_features,
                next_action_mask=item.next_action_mask,
                next_value=item.next_value,
                terminated=item.terminated,
                truncated=item.truncated,
                trainable=item.trainable,
                worker_id=item.worker_id,
                episode_id=item.episode_id,
                transition_id=item.transition_id,
                environment_step=item.environment_step,
                rollout_id=rollout_id,
                main_agent_eliminated=item.main_agent_eliminated,
                environment_steps_spanned=item.environment_steps_spanned,
            )
            for item in selected
        )
        self.buffer.extend(selected)
        self.main_agent_transitions += len(selected)
        self.environment_steps += sum(item.environment_steps_spanned for item in selected)
        return selected

    def collect_from_vector_workers(
        self,
        vector_env: Any,
        *,
        ledgers: PendingTransitionLedger | Mapping[int, PendingTransitionLedger] | Sequence[PendingTransitionLedger] | None = None,
        max_transitions: int | None = None,
    ) -> tuple[PPOTransition, ...]:
        """Collect committed records exposed by vector workers in worker order.

        WorkerRuntime remains execution-only; integrations may attach a ledger
        as ``worker.ledger`` or expose committed records as ``worker.transitions``.
        This adapter intentionally does not manufacture a transition when a
        worker has faulted, been cancelled, or has no committed record.
        """

        if ledgers is not None:
            return self.collect_rollout(ledgers=ledgers, max_transitions=max_transitions)
        workers = getattr(vector_env, "workers", None)
        if workers is None:
            workers = (vector_env,)
        records: list[Any] = []
        for worker in workers:
            source = getattr(worker, "ledger", None)
            if source is not None:
                records.extend(source.committed)
                continue
            source = getattr(worker, "transitions", None)
            if source is not None:
                records.extend(source)
        return self.collect_rollout(records, max_transitions=max_transitions)

    # Backward-compatible name for callers that pass a worker/ledger source.
    collect_from_workers = collect_from_vector_workers

    def _next_values(self, items: Sequence[PPOTransition]) -> Tensor:
        values: list[float] = []
        for item in items:
            if item.terminated or item.truncated:
                values.append(0.0)
            elif item.next_value is not None:
                values.append(item.next_value)
            elif item.next_features is not None:
                mask = item.next_action_mask if item.next_action_mask is not None else item.action_mask
                with torch.no_grad():
                    result = self.module(item.next_features, mask)
                    values.append(float(result.value.reshape(-1)[0]))
            else:
                raise TrainingError("non-terminal PPO transition has no next state for bootstrap")
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _stack(values: Sequence[Any]) -> Any:
        if not values:
            raise ValueError("cannot stack an empty sequence")
        if all(isinstance(value, Mapping) for value in values):
            keys = tuple(values[0].keys())
            if any(tuple(value.keys()) != keys for value in values):
                raise ValueError("batched PPO mappings must have identical keys")
            return {key: PPOTrainer._stack([value[key] for value in values]) for key in keys}
        if all(isinstance(value, Tensor) for value in values):
            return torch.stack([value.to(dtype=torch.float32) for value in values])
        try:
            return torch.stack([torch.as_tensor(value, dtype=torch.float32) for value in values])
        except (TypeError, ValueError) as exc:
            raise TrainingError("PPO features cannot be batched") from exc

    def _segmented_gae(
        self,
        items: Sequence[PPOTransition],
        old_values: Tensor,
        next_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute GAE independently for each worker/episode/rollout segment."""

        advantages = torch.zeros_like(old_values)
        returns = torch.zeros_like(old_values)
        start = 0
        while start < len(items):
            segment = (
                items[start].worker_id,
                items[start].episode_id,
                items[start].rollout_id,
            )
            end = start + 1
            while end < len(items) and (
                items[end].worker_id,
                items[end].episode_id,
                items[end].rollout_id,
            ) == segment:
                end += 1
            segment_advantages, segment_returns = compute_gae(
                [item.reward for item in items[start:end]],
                old_values[start:end],
                next_values[start:end],
                [item.terminated for item in items[start:end]],
                [item.truncated for item in items[start:end]],
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                trainable=[item.trainable for item in items[start:end]],
            )
            advantages[start:end] = segment_advantages
            returns[start:end] = segment_returns
            start = end
        return advantages, returns

    def update(self, transitions: Iterable[PPOTransition | MainAgentTransition] | None = None) -> PPOUpdateMetrics:
        """Run configured epochs/minibatches over eligible rollout records."""

        self._safe_point()
        if transitions is None:
            items = list(self.buffer.trainable)
        else:
            items = [item if isinstance(item, PPOTransition) else PPOTransition.from_committed(item) for item in transitions]
            items = [item for item in items if item.trainable and not item.truncated]
        items.sort(key=lambda item: (item.worker_id, item.episode_id, item.rollout_id, item.environment_step, item.transition_id))
        if not items:
            return PPOUpdateMetrics(ppo_update=self.ppo_updates, metadata=self.metadata())
        next_values = self._next_values(items)
        old_values = torch.tensor([item.value for item in items], dtype=torch.float32)
        advantages, returns = self._segmented_gae(items, old_values, next_values)
        features = self._stack([item.features for item in items])
        masks = torch.tensor(np.stack([item.action_mask for item in items]), dtype=torch.uint8)
        actions = torch.tensor([item.action for item in items], dtype=torch.long)
        old_log_probs = torch.tensor([item.old_log_prob for item in items], dtype=torch.float32)
        rng = np.random.default_rng(self.seed + self.ppo_updates)
        policy_total = value_total = entropy_total = total_total = kl_total = clip_total = grad_total = 0.0
        minibatches = 0
        sample_count = 0
        for _epoch in range(self.config.ppo_epochs):
            self._safe_point()
            order = np.arange(len(items))
            rng.shuffle(order)
            for start in range(0, len(items), self.config.minibatch_size):
                self._safe_point()
                indices = torch.as_tensor(order[start:start + self.config.minibatch_size], dtype=torch.long)
                batch_features = self._index(features, indices)
                current_log_prob, entropy, current_value = self.module.evaluate_actions(
                    batch_features, self._index(actions, indices), self._index(masks, indices)
                )
                batch_old = self._index(old_log_probs, indices)
                batch_advantage = self._index(advantages, indices)
                batch_returns = self._index(returns, indices)
                policy_loss, ratio, clip_fraction = clipped_surrogate_loss(
                    batch_old, current_log_prob, batch_advantage, self.config.clip_epsilon
                )
                value_loss = value_function_loss(current_value, batch_returns)
                entropy = entropy_bonus(entropy)
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                configured_limit = getattr(self.config, "max_grad_norm", 0.5)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), configured_limit) if configured_limit is not None else torch.tensor(0.0)
                self.optimizer.step()
                count = len(indices)
                policy_total += float(policy_loss.detach()) * count
                value_total += float(value_loss.detach()) * count
                entropy_total += float(entropy.detach()) * count
                total_total += float(loss.detach()) * count
                kl_total += float((batch_old - current_log_prob).mean().detach()) * count
                clip_total += float(clip_fraction.detach()) * count
                grad_total += float(torch.as_tensor(grad_norm).detach()) * count
                sample_count += count
                minibatches += 1
        self.ppo_updates += 1
        metadata = self.metadata()
        result = PPOUpdateMetrics(
            policy_loss=policy_total / sample_count,
            value_loss=value_total / sample_count,
            entropy=entropy_total / sample_count,
            total_loss=total_total / sample_count,
            approx_kl=kl_total / sample_count,
            clip_fraction=clip_total / sample_count,
            grad_norm=grad_total / sample_count,
            samples=sample_count,
            epochs=self.config.ppo_epochs,
            minibatches=minibatches,
            ppo_update=self.ppo_updates,
            environment_steps=self.environment_steps,
            main_agent_transitions=self.main_agent_transitions,
            metadata=metadata,
        )
        if self.logger is not None:
            if hasattr(self.logger, "event"):
                self.logger.event("ppo_update", message="PPO update", **result.to_dict())
            elif hasattr(self.logger, "info"):
                self.logger.info("PPO update", **result.to_dict())
        self.buffer.clear_trainable()
        return result

    @staticmethod
    def _index(value: Any, indices: Tensor) -> Any:
        if isinstance(value, Mapping):
            return {key: PPOTrainer._index(item, indices) for key, item in value.items()}
        return value[indices]

    def metadata(self) -> dict[str, Any]:
        snapshot = self.snapshot_metadata()
        return {
            "trainer_format_version": PPO_TRAINER_FORMAT_VERSION,
            "ppo_config": self.config.model_dump(mode="json"),
            "optimizer": getattr(self.module, "optimizer_metadata", {}),
            "counters": {
                "environment_steps": self.environment_steps,
                "main_agent_transitions": self.main_agent_transitions,
                "ppo_updates": self.ppo_updates,
            },
            "rollout": {
                "worker_count": self.worker_count,
                "buffer_capacity": self.buffer.capacity,
                "deterministic_seed": self.seed,
            },
            "inference_modules": {
                "count": len(self.inference_modules),
                "frozen": all(
                    not parameter.requires_grad
                    for inference_module in self.inference_modules
                    for parameter in inference_module.parameters()
                ),
            },
            "checkpoint": {
                "persistent": False,
                "resume_supported": False,
                "resume_state_fields": (),
            },
            "snapshot": snapshot,
            "run_metadata": build_run_metadata(
                run_id=str((snapshot or {}).get("job_id") or f"ppo-{self.seed}"),
                job_id=(snapshot or {}).get("job_id"),
                component="ppo_trainer",
                snapshot=snapshot or None,
                counters={
                    "environment_steps": self.environment_steps,
                    "main_agent_transitions": self.main_agent_transitions,
                    "ppo_updates": self.ppo_updates,
                },
                hardware=((snapshot or {}).get("config_snapshot", {}) or {}).get("hardware", {}) if isinstance((snapshot or {}).get("config_snapshot", {}), Mapping) else {},
            ),
        }

    checkpoint_metadata = metadata

    def snapshot_metadata(self) -> dict[str, Any]:
        if self._snapshot is None:
            return {}
        return self._snapshot.to_dict() if hasattr(self._snapshot, "to_dict") else dict(self._snapshot)

    @property
    def _snapshot(self) -> JobSnapshot | Mapping[str, Any] | None:
        return getattr(self, "_PPOTrainer__snapshot", None)

    @_snapshot.setter
    def _snapshot(self, value: JobSnapshot | Mapping[str, Any] | None) -> None:
        self.__snapshot = value

    def validate_checkpoint_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Validate compatibility metadata without accepting resume state."""

        checkpoint = metadata.get("checkpoint", {})
        if checkpoint.get("persistent") or checkpoint.get("resume_supported"):
            raise TrainingError("PPO checkpoint/resume state is not supported")
        configured = metadata.get("ppo_config")
        if configured is not None and configured != self.config.model_dump(mode="json"):
            raise TrainingError("PPO metadata is incompatible with the current JobSnapshot")

    validate_resume_metadata = validate_checkpoint_metadata

    def freeze_inference_modules(self) -> tuple[torch.nn.Module, ...]:
        """Freeze the Phase-2 modules while leaving PPOModule trainable."""

        for inference_module in self.inference_modules:
            inference_module.eval()
            for parameter in inference_module.parameters():
                parameter.requires_grad_(False)
        return self.inference_modules

    def save_inference_artifact(self, path: str, **kwargs: Any):
        """Save the trained policy as a final inference-only ``.bangmodel``."""

        from .final_inference_artifact import save_final_inference_artifact

        self.freeze_inference_modules()
        artifact_kwargs = dict(kwargs)
        phase2_artifact = artifact_kwargs.pop("phase2_artifact", self.phase2_artifact)
        if phase2_artifact is None:
            if len(self.inference_modules) != 2:
                raise TrainingError("Phase-2 modules are required to save a final inference artifact")
            artifact_kwargs.setdefault("intent", self.inference_modules[0])
            artifact_kwargs.setdefault("emotion", self.inference_modules[1])
        return save_final_inference_artifact(
            path,
            self.module,
            phase2_artifact=phase2_artifact,
            trainer=self,
            logger=artifact_kwargs.pop("logger", self.logger),
            **artifact_kwargs,
        )

    save_final_artifact = save_inference_artifact


__all__ = [
    "ACTION_SPACE_SIZE", "PPO_TRAINER_FORMAT_VERSION", "PPOTransition", "RolloutTransition",
    "RolloutBuffer", "PPORolloutBuffer", "compute_gae", "gae_and_returns",
    "clipped_surrogate_loss", "value_function_loss", "entropy_bonus",
    "PPOUpdateMetrics", "PPOTrainer",
]
