"""Declarative reward evaluation at validated environment transitions.

The environment owns game rules and terminal-condition evaluation.  This module
only consumes the resulting transition plus privileged role context supplied by
the training/runtime layer.  It never changes the game state or feeds reward
fields back into an agent observation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
import json
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar

from bang_rl.config import RewardConfig
from bang_rl.runtime.logging import LogChannel, StructuredLogger, get_component_logger


REWARD_RUNTIME_STATE_VERSION = "1.0"
PLAYER_COUNT = 5


class RewardCalculationError(ValueError):
    """Raised when a transition cannot produce a finite deterministic reward."""


def _json_safe(value: Any) -> Any:
    if hasattr(value, "value"):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RewardCalculationError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise RewardCalculationError(f"{label} must be finite")
    return result


def _index(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RewardCalculationError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RewardCalculationError(f"{label} must be an integer") from exc
    if result < 0 or result >= PLAYER_COUNT:
        raise RewardCalculationError(f"{label} must be in [0, 4]")
    return result


def _token(value: Any) -> str:
    """Normalize enum, integer, and YAML string role/team values."""

    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str):
        return value.rsplit(".", 1)[-1].strip().lower()
    try:
        return str(int(value))
    except (TypeError, ValueError, OverflowError):
        return str(value).strip().lower()


def _role_name(value: Any) -> str:
    token = _token(value)
    numeric = {"0": "sheriff", "1": "deputy", "2": "outlaw_a", "3": "outlaw_b", "4": "traitor"}
    return numeric.get(token, token)


def _team_name(value: Any) -> str:
    token = _token(value)
    numeric = {"0": "sheriff_team", "1": "outlaw_team", "2": "traitor_team"}
    return numeric.get(token, token)


def _role_team(role: Any) -> str:
    role_name = _role_name(role)
    if role_name in {"sheriff", "deputy"}:
        return "sheriff_team"
    if role_name in {"outlaw_a", "outlaw_b"}:
        return "outlaw_team"
    if role_name == "traitor":
        return "traitor_team"
    return ""


def _event_kind(event: Any) -> str:
    if isinstance(event, Mapping):
        value = event.get("event_type", event.get("type", event.get("event")))
    else:
        value = getattr(event, "event_type", None)
    if hasattr(value, "value"):
        value = value.value
    return str(value).rsplit(".", 1)[-1].lower() if value is not None else ""


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _action_values(result: Any) -> tuple[int, int | None]:
    action = result.action if hasattr(result, "action") else result.get("action")
    if isinstance(action, Mapping):
        action_type = action.get("action_type", action.get("type"))
        actor = action.get("actor_idx")
        target = action.get("target_idx")
    else:
        action_type = getattr(action, "action_type", None)
        actor = getattr(action, "actor_idx", None)
        target = getattr(action, "target_idx", None)
    if hasattr(action_type, "value"):
        action_type = action_type.value
    if isinstance(action_type, str):
        action_type = action_type.rsplit(".", 1)[-1].lower()
        action_index = 0 if action_type in {"pass", "0"} else 1
    else:
        action_index = int(action_type)
    return _index(actor, "action.actor_idx"), (None if action_index == 0 else _index(target, "action.target_idx"))


def _result_values(result: Any) -> dict[str, Any]:
    if hasattr(result, "action") and hasattr(result, "environment_step"):
        return {
            "action": result.action,
            "environment_step": result.environment_step,
            "events": result.events,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "info": dict(result.info),
        }
    if isinstance(result, Mapping):
        values = dict(result)
        if "action" not in values or "environment_step" not in values:
            raise RewardCalculationError("transition requires action and environment_step")
        values.setdefault("events", ())
        values.setdefault("terminated", False)
        values.setdefault("truncated", False)
        values.setdefault("info", {})
        return values
    if hasattr(result, "to_dict"):
        return _result_values(result.to_dict())
    raise TypeError("transition must be an ActionResult or mapping")


def _terminal_values(value: Any) -> tuple[set[int], set[int]]:
    if value is None:
        return set(), set()
    if hasattr(value, "winners"):
        return {int(item) for item in value.winners}, {int(item) for item in value.losers}
    if isinstance(value, Mapping):
        return (
            {int(item) for item in value.get("winners", ())},
            {int(item) for item in value.get("losers", ())},
        )
    raise RewardCalculationError("terminal_result must be serializable")


def _roles_from_context(state: Any, roles: Any) -> dict[int, Any]:
    if roles is None and state is not None:
        roles = getattr(state, "roles", None)
    if roles is None:
        return {}
    if isinstance(roles, Mapping):
        return {_index(key, "role player"): value for key, value in roles.items()}
    try:
        return {_index(index, "role player"): value for index, value in enumerate(roles)}
    except TypeError as exc:
        raise RewardCalculationError("roles must be a mapping or sequence") from exc


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """One agent's immutable, serializable reward signal."""

    agent_id: int
    environment_step: int
    main_agent_transition_id: Any | None
    terminal_reward: float
    shaping: Mapping[str, float]
    total: float
    reward_config_version: str
    reward_config_id: str
    unclipped_total: float
    clipped: bool = False
    terminal_components: Mapping[str, float] = field(default_factory=dict)

    @property
    def shaping_reward(self) -> float:
        return _finite(sum(self.shaping.values()), "shaping reward")

    @property
    def reward(self) -> float:
        return self.total

    @property
    def config_version(self) -> str:
        return self.reward_config_version

    @property
    def config_identifier(self) -> str:
        return self.reward_config_id

    @property
    def components(self) -> dict[str, float]:
        return {
            "terminal_reward": self.terminal_reward,
            **dict(self.terminal_components),
            **dict(self.shaping),
        }

    def __float__(self) -> float:
        return self.total

    def __getitem__(self, key: str) -> Any:
        if key in self.components:
            return self.components[key]
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "environment_step": self.environment_step,
            "main_agent_transition_id": self.main_agent_transition_id,
            "terminal_reward": self.terminal_reward,
            "terminal_components": dict(self.terminal_components),
            "shaping": dict(self.shaping),
            "components": self.components,
            "unclipped_total": self.unclipped_total,
            "total": self.total,
            "clipped": self.clipped,
            "reward_config_version": self.reward_config_version,
            "reward_config_id": self.reward_config_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RewardBreakdown":
        return cls(
            agent_id=_index(value["agent_id"], "reward agent_id"),
            environment_step=int(value["environment_step"]),
            main_agent_transition_id=value.get("main_agent_transition_id"),
            terminal_reward=_finite(value.get("terminal_reward", 0.0), "terminal reward"),
            terminal_components={str(key): _finite(item, f"terminal component {key}") for key, item in value.get("terminal_components", {}).items()},
            shaping={str(key): _finite(item, f"shaping component {key}") for key, item in value.get("shaping", {}).items()},
            total=_finite(value.get("total", 0.0), "total reward"),
            reward_config_version=str(value["reward_config_version"]),
            reward_config_id=str(value.get("reward_config_id", "")),
            unclipped_total=_finite(value.get("unclipped_total", value.get("total", 0.0)), "unclipped reward"),
            clipped=bool(value.get("clipped", False)),
        )


@dataclass(frozen=True, slots=True)
class RewardBatch:
    """Per-agent rewards calculated for one validated environment transition."""

    rewards: Mapping[int, RewardBreakdown]
    environment_step: int
    reward_config_version: str
    reward_config_id: str

    @property
    def totals(self) -> dict[int, float]:
        return {index: breakdown.total for index, breakdown in self.rewards.items()}

    @property
    def breakdowns(self) -> Mapping[int, RewardBreakdown]:
        return self.rewards

    @property
    def config_version(self) -> str:
        return self.reward_config_version

    @property
    def config_identifier(self) -> str:
        return self.reward_config_id

    def __getitem__(self, agent_id: int) -> RewardBreakdown:
        return self.rewards[agent_id]

    def __iter__(self):
        return iter(self.rewards)

    def __len__(self) -> int:
        return len(self.rewards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_step": self.environment_step,
            "reward_config_version": self.reward_config_version,
            "reward_config_id": self.reward_config_id,
            "rewards": {str(index): value.to_dict() for index, value in self.rewards.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RewardBatch":
        rewards = {
            _index(index, "reward agent_id"): RewardBreakdown.from_dict(item)
            for index, item in value.get("rewards", {}).items()
        }
        return cls(
            rewards=rewards,
            environment_step=int(value["environment_step"]),
            reward_config_version=str(value["reward_config_version"]),
            reward_config_id=str(value.get("reward_config_id", "")),
        )


class RewardCalculator:
    """Evaluate the configured terminal and shaping reward namespaces.

    ``calculate`` accepts only a completed valid transition.  Role context is
    explicit because it is privileged training data and must never be inferred
    from an agent observation.  The method is pure with respect to the supplied
    transition and returns a reward for every requested agent.
    """

    state_version: ClassVar[str] = REWARD_RUNTIME_STATE_VERSION

    def __init__(
        self,
        config: RewardConfig | Mapping[str, Any],
        *,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.config = config if isinstance(config, RewardConfig) else RewardConfig.model_validate(config)
        self.logger = logger
        self.reward_config_id = self.config.config_identifier

    @property
    def config_version(self) -> str:
        return self.config.version

    @property
    def config_identifier(self) -> str:
        return self.reward_config_id

    @classmethod
    def from_yaml(cls, source: str | Path, *, logger: StructuredLogger | None = None) -> "RewardCalculator":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is declared by the project
            raise RuntimeError("PyYAML is required to load the reward DSL") from exc
        path = Path(source)
        try:
            is_file = path.exists()
        except OSError:
            is_file = False
        text = path.read_text(encoding="utf-8") if is_file else str(source)
        loaded = yaml.safe_load(text) or {}
        if "reward_config" in loaded and len(loaded) == 1:
            loaded = loaded["reward_config"]
        return cls(loaded, logger=logger)

    from_yaml_file = from_yaml

    def to_dict(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json")

    def to_yaml(self) -> str:
        import yaml
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def _logger(self) -> StructuredLogger:
        return self.logger or get_component_logger("reward-calculator", channel=LogChannel.REWARD_BREAKDOWN)

    def _component_matches(
        self,
        component: Any,
        *,
        action_actor: int,
        action_target: int | None,
        events: tuple[Any, ...],
        roles: Mapping[int, Any],
    ) -> int:
        condition = component.condition
        condition_type = condition.type
        if condition_type == "event":
            expected = condition.event
            if expected == "pass":
                return int(action_target is None)
            if expected == "bang_hit":
                return sum(_event_kind(event) in {"bang", "bang_hit"} for event in events)
            if expected == "kill_enemy":
                killed_targets: set[int | str] = set()
                for event in events:
                    if _event_kind(event) not in {"bang", "kill_enemy", "player_eliminated"}:
                        continue
                    eliminated = _event_value(event, "eliminated_idx")
                    if eliminated is None:
                        continue
                    key = str(eliminated)
                    killed_targets.add(key)
                return len(killed_targets)
            return 0
        if condition_type == "pass":
            return int(action_target is None)
        if action_target is None:
            return 0
        if condition_type == "role_attack":
            return int(
                _role_name(roles.get(action_actor)) == _role_name(condition.actor_role)
                and _role_name(roles.get(action_target)) == _role_name(condition.target_role)
            )
        if condition_type == "faction_attack":
            return int(
                _role_team(roles.get(action_actor)) == _team_name(condition.actor_faction)
                and _role_team(roles.get(action_target)) == _team_name(condition.target_faction)
            )
        return 0

    def _breakdown_for_agent(
        self,
        agent_id: int,
        values: Mapping[str, Any],
        *,
        roles: Mapping[int, Any],
        terminal_result: Any,
        main_agent_transition_id: Any | None,
    ) -> RewardBreakdown:
        step = int(values["environment_step"])
        actor, target = _action_values(values)
        events = tuple(values.get("events") or ())
        shaping: dict[str, float] = {}
        for component in self.config.shaping.components:
            if agent_id != actor or not component.enabled:
                shaping[component.id] = 0.0
                continue
            count = self._component_matches(
                component,
                action_actor=actor,
                action_target=target,
                events=events,
                roles=roles,
            )
            shaping[component.id] = _finite(component.value * count, f"shaping component {component.id}")

        terminal_components = {component.id: 0.0 for component in self.config.terminal.components}
        terminal_reward = 0.0
        terminated = bool(values.get("terminated", False))
        if terminated and not bool(values.get("truncated", False)):
            winners, losers = _terminal_values(terminal_result)
            outcome = "win" if agent_id in winners else "loss" if agent_id in losers else None
            if outcome is not None:
                for component in self.config.terminal.components:
                    if component.enabled and component.condition.result == outcome:
                        contribution = _finite(component.value, f"terminal component {component.id}")
                        terminal_components[component.id] = contribution
                        terminal_reward += contribution
        terminal_reward = _finite(terminal_reward, "terminal reward")
        shaping_total = _finite(sum(shaping.values()), "shaping reward")
        unclipped = _finite(terminal_reward + shaping_total, "unclipped total reward")
        clipping = self.config.clipping
        clipped = False
        total = unclipped
        if clipping.enabled:
            assert clipping.min_value is not None and clipping.max_value is not None
            total = min(max(unclipped, clipping.min_value), clipping.max_value)
            clipped = total != unclipped
        total = _finite(total, "total reward")
        return RewardBreakdown(
            agent_id=agent_id,
            environment_step=step,
            main_agent_transition_id=main_agent_transition_id,
            terminal_reward=terminal_reward,
            terminal_components=terminal_components,
            shaping=shaping,
            total=total,
            reward_config_version=self.config.version,
            reward_config_id=self.reward_config_id,
            unclipped_total=unclipped,
            clipped=clipped,
        )

    def calculate(
        self,
        result: Any,
        *,
        state: Any | None = None,
        roles: Any | None = None,
        terminal_result: Any | None = None,
        agent_indices: Iterable[int] | None = None,
        agent_idx: int | None = None,
        agent_id: int | None = None,
        main_agent_transition_id: Any | None = None,
        transition_id: Any | None = None,
        logger: StructuredLogger | None = None,
    ) -> RewardBatch | RewardBreakdown:
        """Calculate rewards without mutating the action result or game state."""

        if agent_idx is not None and agent_id is not None and int(agent_idx) != int(agent_id):
            raise ValueError("agent_idx and agent_id disagree")
        selected = agent_idx if agent_idx is not None else agent_id
        values = _result_values(result)
        terminated = bool(values.get("terminated", False))
        truncated = bool(values.get("truncated", False))
        if terminated and truncated:
            raise RewardCalculationError("terminated and truncated cannot both be true")
        step = int(values["environment_step"])
        if step < 0:
            raise RewardCalculationError("environment_step must be non-negative")
        resolved_roles = _roles_from_context(state, roles)
        info = values.get("info")
        info = info if isinstance(info, Mapping) else {}
        resolved_terminal = terminal_result
        if resolved_terminal is None and state is not None:
            resolved_terminal = getattr(state, "terminal_result", None)
        if resolved_terminal is None:
            resolved_terminal = info.get("terminal_result")
        if selected is not None:
            indices = (_index(selected, "agent_idx"),)
        elif agent_indices is not None:
            indices = tuple(dict.fromkeys(_index(value, "agent_idx") for value in agent_indices))
        elif resolved_roles:
            indices = tuple(sorted(resolved_roles))
        else:
            indices = tuple(range(PLAYER_COUNT))
        breakdowns = {
            index: self._breakdown_for_agent(
                index,
                values,
                roles=resolved_roles,
                terminal_result=resolved_terminal,
                main_agent_transition_id=(
                    main_agent_transition_id if main_agent_transition_id is not None else transition_id
                ),
            )
            for index in indices
        }
        batch = RewardBatch(
            rewards=breakdowns,
            environment_step=step,
            reward_config_version=self.config.version,
            reward_config_id=self.reward_config_id,
        )
        emit = logger or self.logger
        if emit is not None:
            for breakdown in breakdowns.values():
                emit.event(
                    "reward_breakdown",
                    "reward calculated",
                    step=step,
                    agent_id=breakdown.agent_id,
                    environment_step=step,
                    main_agent_transition_id=breakdown.main_agent_transition_id,
                    terminal_reward=breakdown.terminal_reward,
                    terminal_components=dict(breakdown.terminal_components),
                    shaping=dict(breakdown.shaping),
                    components=breakdown.components,
                    total=breakdown.total,
                    reward_config_version=breakdown.reward_config_version,
                    reward_config_id=breakdown.reward_config_id,
                )
        return breakdowns[selected] if selected is not None else batch

    compute = calculate
    calculate_rewards = calculate

    def calculate_for_agent(self, result: Any, agent_idx: int, **kwargs: Any) -> RewardBreakdown:
        value = self.calculate(result, agent_idx=agent_idx, **kwargs)
        assert isinstance(value, RewardBreakdown)
        return value

    def calculate_all(self, result: Any, **kwargs: Any) -> RewardBatch:
        value = self.calculate(result, **kwargs)
        if isinstance(value, RewardBreakdown):
            return RewardBatch(
                rewards={value.agent_id: value},
                environment_step=value.environment_step,
                reward_config_version=value.reward_config_version,
                reward_config_id=value.reward_config_id,
            )
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "reward_config_version": self.config.version,
            "reward_config_id": self.reward_config_id,
            "reward_config": self.to_dict(),
        }

    to_dict_state = state_dict

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any], *, logger: StructuredLogger | None = None) -> "RewardCalculator":
        if state.get("state_version", cls.state_version) != cls.state_version:
            raise ValueError("unsupported reward runtime state schema")
        calculator = cls(state["reward_config"], logger=logger)
        if state.get("reward_config_id", calculator.reward_config_id) != calculator.reward_config_id:
            raise ValueError("reward config identifier does not match serialized config")
        return calculator

    from_dict = from_state_dict

    def to_json(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str, *, logger: StructuredLogger | None = None) -> "RewardCalculator":
        return cls.from_state_dict(json.loads(value), logger=logger)


def attach_rewards(result: ActionResult, batch: RewardBatch) -> ActionResult:
    """Attach training-only per-agent rewards at the transition boundary.

    The public observation remains unchanged.  The returned immutable result is
    suitable for EmotionRuntime/PPO handoff, where ``info['rewards']`` is a
    separate learning signal and ``reward`` is the acting agent's scalar value.
    """

    if not hasattr(result, "info") or not hasattr(result, "action"):
        raise TypeError("result must be an ActionResult-like immutable result")
    info = dict(result.info)
    info["rewards"] = {index: breakdown.total for index, breakdown in batch.rewards.items()}
    info["reward_breakdown"] = batch.to_dict()
    actor = result.action.actor_idx
    actor_reward = batch.rewards.get(actor)
    return replace(result, reward=0.0 if actor_reward is None else actor_reward.total, info=info)


class RewardRuntime:
    """Small transition-boundary coordinator with replay-safe deduplication."""

    state_version: ClassVar[str] = REWARD_RUNTIME_STATE_VERSION

    def __init__(self, calculator: RewardCalculator) -> None:
        if not isinstance(calculator, RewardCalculator):
            raise TypeError("calculator must be a RewardCalculator")
        self.calculator = calculator
        self.episode_id: str | None = None
        self._committed: dict[tuple[str | None, int], RewardBatch] = {}

    def reset(self, context: Any | None = None, *, episode_id: Any = None) -> None:
        if context is not None and episode_id is not None:
            raise ValueError("provide context or episode_id, not both")
        if context is not None:
            episode_id = context.get("episode_id") if isinstance(context, Mapping) else getattr(context, "episode_id", None)
        self.episode_id = None if episode_id is None else str(episode_id)
        self._committed = {}

    def on_invalid_action(self, *_: Any, **__: Any) -> None:
        return None

    def on_action_result(self, result: Any, *, valid: bool = True, episode_id: Any = None, **kwargs: Any) -> RewardBatch | None:
        if not valid:
            return None
        values = _result_values(result)
        step = int(values["environment_step"])
        key = (self.episode_id if episode_id is None else str(episode_id), step)
        if key in self._committed:
            return self._committed[key]
        batch = self.calculator.calculate(result, **kwargs)
        if isinstance(batch, RewardBreakdown):
            batch = RewardBatch(
                rewards={batch.agent_id: batch},
                environment_step=batch.environment_step,
                reward_config_version=batch.reward_config_version,
                reward_config_id=batch.reward_config_id,
            )
        self._committed[key] = batch
        return batch

    commit = on_action_result
    process_action_result = on_action_result

    @property
    def records(self) -> tuple[RewardBatch, ...]:
        return tuple(self._committed.values())

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "episode_id": self.episode_id,
            "calculator": self.calculator.state_dict(),
            "records": [batch.to_dict() for batch in self._committed.values()],
        }

    to_dict = state_dict

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RewardRuntime":
        if state.get("state_version", cls.state_version) != cls.state_version:
            raise ValueError("unsupported reward runtime state schema")
        runtime = cls(RewardCalculator.from_state_dict(state["calculator"]))
        runtime.episode_id = state.get("episode_id")
        for raw in state.get("records", ()):
            batch = RewardBatch.from_dict(raw)
            runtime._committed[(runtime.episode_id, batch.environment_step)] = batch
        return runtime

    from_dict = from_state_dict

    def to_json(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "RewardRuntime":
        return cls.from_state_dict(json.loads(value))


__all__ = [
    "REWARD_RUNTIME_STATE_VERSION",
    "RewardBatch",
    "RewardBreakdown",
    "RewardCalculationError",
    "RewardCalculator",
    "RewardRuntime",
    "attach_rewards",
]
