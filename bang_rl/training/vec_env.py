"""Vectorized BangEnvironment orchestration and cooperative worker runtime.

The module owns execution concerns only.  It never passes ``EnvironmentState``
or inference labels to an action provider: action selection receives a detached
public observation and its six-entry mask.  ``DummyVecEnv`` is deterministic
and lightweight for local execution; ``SubprocVecEnv`` uses one spawned process
per configured worker and propagates child faults to the caller.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import multiprocessing as mp
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from bang_rl.agents.base import EpisodeContext, EpisodeResult
from bang_rl.config import EnvConfig, JobSnapshot, SystemConfig
from bang_rl.env import BangEnvironment
from bang_rl.exceptions import TrainingError
from bang_rl.runtime.logging import LogChannel, StructuredLogger
from bang_rl.runtime.pause_coordinator import (
    PauseCoordinator,
    PauseState,
    WorkerCancelled,
    WorkerControl,
)
from bang_rl.runtime.seed import SeedManager, derive_seed

from .resource_guard import ResourceBudget, ResourceGuard


PublicActionProvider = Callable[[Any, np.ndarray], int]
EnvironmentFactory = Callable[..., Any]


class WorkerFault(TrainingError):
    """A worker failed; the worker identity and episode are retained."""

    error_code = "worker_fault"

    def __init__(self, worker_id: int, message: str, *, episode_id: str | None = None, cause: BaseException | None = None) -> None:
        self.worker_id = int(worker_id)
        self.episode_id = episode_id
        self.cause = cause
        super().__init__(
            f"worker {self.worker_id} failed{f' in {episode_id}' if episode_id else ''}: {message}",
            context={"worker_id": self.worker_id, "episode_id": episode_id},
        )


WorkerRuntimeError = WorkerFault


@dataclass(frozen=True, slots=True)
class EpisodeAssignment:
    """Stable global episode-to-worker assignment and derived seed."""

    episode_index: int
    worker_id: int
    episode_id: str
    seed: int


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Public execution record; no hidden role array or emotion label appears."""

    worker_id: int
    episode_index: int
    episode_id: str
    seed: int
    terminated: bool
    truncated: bool
    termination_status: str
    environment_steps: int
    initial_observation: Mapping[str, Any]
    final_observation: Mapping[str, Any]
    actions: tuple[Mapping[str, Any], ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    terminal_result: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def public(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_status": self.termination_status,
            "environment_steps": self.environment_steps,
            "initial_observation": deepcopy(dict(self.initial_observation)),
            "final_observation": deepcopy(dict(self.final_observation)),
            "actions": [dict(item) for item in self.actions],
            "events": [dict(item) for item in self.events],
            "terminal_result": None if self.terminal_result is None else dict(self.terminal_result),
            "metadata": deepcopy(dict(self.metadata)),
            "cancelled": self.cancelled,
        }


@dataclass(frozen=True, slots=True)
class WorkerStep:
    """One public vectorized worker result."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    result: Any | None = None


@dataclass(frozen=True, slots=True)
class WorkerRuntimeMetadata:
    worker_id: int
    worker_seed: int
    batch_size: int
    buffer_size: int
    hardware: Mapping[str, Any]
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "worker_seed": self.worker_seed,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "hardware": dict(self.hardware),
            "job_id": self.job_id,
        }


def _mapping_observation(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "to_dict"):
        value = observation.to_dict()
    elif isinstance(observation, Mapping):
        value = dict(observation)
    else:
        value = {"observation": observation}
    return deepcopy(value)


def _public_observation(environment: Any) -> Any:
    """Detach the canonical public observation without reading privileged state."""

    observation = environment.observe()
    # PublicEnvironmentState is immutable, but its event payloads are copied so
    # one worker cannot mutate another worker's returned container.
    return deepcopy(observation)


def _episode_result_from_environment(environment: Any, episode_id: str) -> EpisodeResult:
    return EpisodeResult(
        episode_id=episode_id,
        terminated=bool(getattr(environment.state, "terminated", False)),
        truncated=bool(getattr(environment.state, "truncated", False)),
        info={
            "termination_status": getattr(getattr(environment.state, "termination_status", None), "value", None),
            "environment_step": int(getattr(environment, "environment_step", 0)),
        },
    )


def _invoke_factory(factory: EnvironmentFactory, worker_id: int, seed: int, episode_id: str) -> Any:
    """Call factories with their supported keyword subset, including no-arg test factories."""

    try:
        signature = inspect.signature(factory)
        parameters = signature.parameters
        kwargs = {
            name: value
            for name, value in (("worker_id", worker_id), ("seed", seed), ("episode_id", episode_id))
            if name in parameters or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        }
        return factory(**kwargs)
    except (TypeError, ValueError):
        # Some extension callables do not expose a signature.  The fallback is
        # intentionally narrow so an internal factory TypeError is not hidden.
        return factory()


def _reset_environment(environment: Any, seed: int, episode_id: str) -> Any:
    reset = environment.reset
    try:
        return reset(seed=seed, options={"episode_id": episode_id}, episode_id=episode_id)
    except TypeError:
        try:
            return reset(seed=seed, options={"episode_id": episode_id})
        except TypeError:
            return reset(seed=seed)


def _action_from_provider(provider: Any, observation: Any, mask: np.ndarray) -> int:
    if provider is None:
        return 0  # PASS is always legal.
    if hasattr(provider, "act"):
        return int(provider.act(observation, mask.copy()))
    return int(provider(observation, mask.copy()))


def _clone_agent_graph(agents: Sequence[Any] | Mapping[Any, Any] | None, *, worker_id: int, seed_manager: SeedManager) -> Sequence[Any] | Mapping[Any, Any] | None:
    """Clone every mutable agent graph and give each clone a worker stream."""

    if agents is None:
        return None

    def clone(index: int, template: Any) -> Any:
        try:
            value = deepcopy(template)
        except Exception as exc:
            raise TrainingError(
                "vector workers require deepcopy-able agent graphs; refusing to share mutable agent state"
            ) from exc
        agent_id = getattr(value, "agent_id", index)
        stream_seed = derive_seed(seed_manager.worker_seed(worker_id), "agent", index, agent_id)
        if hasattr(value, "seed"):
            value.seed = stream_seed
        # Built-in agents keep their action stream in this private field.  A
        # fresh generator is required even when deepcopy already detached it,
        # otherwise workers begin from identical copied RNG state.
        if hasattr(value, "_rng"):
            value._rng = np.random.default_rng(stream_seed)
        return value

    if isinstance(agents, Mapping):
        return {key: clone(index, agents[key]) for index, key in enumerate(agents)}
    return tuple(clone(index, template) for index, template in enumerate(agents))


class WorkerRuntime:
    """Own one environment, its derived streams, and cooperative controls."""

    fixed_agent_order = tuple(f"player_{index}" for index in range(5))

    def __init__(
        self,
        worker_id: int,
        *,
        seed_manager: SeedManager,
        environment_factory: EnvironmentFactory | None = None,
        action_provider: Any = None,
        agents: Sequence[Any] | Mapping[Any, Any] | None = None,
        env_config: EnvConfig | None = None,
        resource_guard: ResourceGuard | None = None,
        batch_size: int | None = None,
        buffer_size: int | None = None,
        job_id: str | None = None,
        logger: StructuredLogger | None = None,
        privileged_logger: StructuredLogger | None = None,
        inference_artifact: Any = None,
        control: WorkerControl | None = None,
    ) -> None:
        if isinstance(worker_id, bool) or int(worker_id) < 0:
            raise ValueError("worker_id must be a non-negative integer")
        self.worker_id = int(worker_id)
        self.seed_manager = seed_manager
        self.environment_factory = environment_factory
        self.action_provider = action_provider
        self.agents = agents
        self.env_config = env_config
        self.resource_guard = resource_guard
        if resource_guard is not None:
            resource_guard.validate_batch_buffer(batch_size, buffer_size)
            budget = resource_guard.budget
        else:
            budget = ResourceBudget.from_value(None)
        self.batch_size = budget.batch_size if batch_size is None else int(batch_size)
        self.buffer_size = budget.buffer_size if buffer_size is None else int(buffer_size)
        self.job_id = job_id
        self.logger = logger
        self.privileged_logger = privileged_logger
        self.inference_artifact = inference_artifact
        if inference_artifact is not None:
            for module_name in ("intent", "emotion", "intent_inference", "emotion_inference"):
                module = getattr(inference_artifact, module_name, None)
                if module is not None and hasattr(module, "parameters") and any(parameter.requires_grad for parameter in module.parameters()):
                    raise ValueError("inference artifact modules must be frozen before worker execution")
        if privileged_logger is not None and privileged_logger.channel != LogChannel.PRIVILEGED_DIAGNOSTICS.value:
            raise ValueError("privileged_logger must use the privileged_diagnostics channel")
        self.control = control or WorkerControl()
        self.environment: Any | None = None
        self.current_assignment: EpisodeAssignment | None = None
        self._initial_observation: Any | None = None
        self._actions: list[Mapping[str, Any]] = []
        self._events: list[Mapping[str, Any]] = []
        self._active = False
        self._started_at = 0.0
        self._paused_seconds = 0.0

    @property
    def worker_seed(self) -> int:
        return self.seed_manager.worker_seed(self.worker_id)

    @property
    def metadata(self) -> WorkerRuntimeMetadata:
        return WorkerRuntimeMetadata(
            worker_id=self.worker_id,
            worker_seed=self.worker_seed,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            hardware=self.resource_guard.budget.to_dict() if self.resource_guard is not None else ResourceBudget.from_value(None).to_dict(),
            job_id=self.job_id,
        )

    @property
    def environment_step(self) -> int:
        return int(getattr(self.environment, "environment_step", 0)) if self.environment is not None else 0

    @property
    def paused(self) -> bool:
        return self.control.paused

    @property
    def cancelled(self) -> bool:
        return self.control.cancel_requested

    def assignment(self, episode_index: int, *, episode_id: str | None = None) -> EpisodeAssignment:
        if isinstance(episode_index, bool) or int(episode_index) < 0:
            raise ValueError("episode_index must be a non-negative integer")
        resolved_id = episode_id or f"episode-{int(episode_index)}"
        return EpisodeAssignment(
            episode_index=int(episode_index),
            worker_id=self.worker_id,
            episode_id=resolved_id,
            seed=self.seed_manager.environment_seed(resolved_id, worker_id=self.worker_id),
        )

    def _new_environment(self, assignment: EpisodeAssignment) -> Any:
        if self.environment is None:
            if self.environment_factory is None:
                self.environment = BangEnvironment(
                    episode_id=assignment.episode_id,
                    seed=assignment.seed,
                    config=self.env_config,
                )
            else:
                self.environment = _invoke_factory(self.environment_factory, self.worker_id, assignment.seed, assignment.episode_id)
        _reset_environment(self.environment, assignment.seed, assignment.episode_id)
        return self.environment

    def reset(self, assignment: EpisodeAssignment | None = None, *, episode_index: int = 0, episode_id: str | None = None) -> Any:
        assignment = assignment or self.assignment(episode_index, episode_id=episode_id)
        try:
            self.control.safe_point()
            environment = self._new_environment(assignment)
            self.current_assignment = assignment
            self._initial_observation = _public_observation(environment)
            self._actions = []
            self._events = []
            self._started_at = time.monotonic()
            self._active = True
            self.control.begin()
            if self.logger is not None:
                self.logger.event("worker_episode_started", worker_id=self.worker_id, worker_seed=self.worker_seed, episode_seed=assignment.seed, episode_index=assignment.episode_index)
            self._notify_agents_start(assignment)
            return deepcopy(self._initial_observation)
        except WorkerCancelled:
            raise
        except Exception as exc:
            raise self._fault(exc) from exc

    def _notify_agents_start(self, assignment: EpisodeAssignment) -> None:
        context = EpisodeContext(assignment.episode_id, assignment.seed, self.worker_id, {"worker_id": self.worker_id})
        if self.agents is None:
            return
        values = self.agents.values() if isinstance(self.agents, Mapping) else self.agents
        for agent in values:
            callback = getattr(agent, "on_episode_start", None)
            if callback is not None:
                callback(context)

    def _agent_for(self, player_index: int) -> Any:
        if isinstance(self.agents, Mapping):
            return self.agents.get(player_index, self.agents.get(f"player_{player_index}"))
        if self.agents is not None and 0 <= player_index < len(self.agents):
            return self.agents[player_index]
        return None

    def _choose_action(self, observation: Any, mask: np.ndarray) -> int:
        selected = getattr(self.environment, "agent_selection", None)
        player_index = int(getattr(selected, "index", int(str(selected).rsplit("_", 1)[-1]))) if selected is not None else 0
        agent = self._agent_for(player_index)
        if agent is not None:
            return _action_from_provider(agent, observation, mask)
        return _action_from_provider(self.action_provider, observation, mask)

    def _fault(self, exc: BaseException) -> WorkerFault:
        assignment = self.current_assignment
        if isinstance(exc, WorkerFault):
            return exc
        if self.logger is not None:
            self.logger.error("worker fault", worker_id=self.worker_id, episode_id=None if assignment is None else assignment.episode_id, error_type=type(exc).__name__, error=str(exc))
        return WorkerFault(self.worker_id, str(exc), episode_id=None if assignment is None else assignment.episode_id, cause=exc)

    def step(self, action: Any | None = None, *, action_provider: Any = None) -> WorkerStep:
        if self.environment is None or self.current_assignment is None or not self._active:
            raise WorkerFault(self.worker_id, "step requested before reset")
        try:
            self.control.safe_point()
            observation = _public_observation(self.environment)
            observed_mask = observation["action_mask"] if isinstance(observation, Mapping) else observation.action_mask
            mask = np.asarray(observed_mask, dtype=np.uint8).copy()
            selected_action = self._choose_action(observation, mask) if action is None and action_provider is None else _action_from_provider(action_provider, observation, mask) if action is None else int(action)
            self.environment.step(selected_action)
            result = getattr(self.environment, "last_result", None)
            if result is None:
                raise RuntimeError("environment did not expose last_result after step")
            next_observation = _public_observation(self.environment)
            action_dict = result.action.to_dict() if hasattr(result.action, "to_dict") else {"action": int(selected_action)}
            action_dict.update({"worker_id": self.worker_id, "episode_id": self.current_assignment.episode_id, "environment_step": int(result.environment_step)})
            self._actions.append(deepcopy(action_dict))
            for event in getattr(result, "events", ()):
                public_event = event.to_public_dict() if hasattr(event, "to_public_dict") else event.to_dict() if hasattr(event, "to_dict") else dict(event)
                self._events.append(deepcopy(public_event))
                if self.logger is not None:
                    self.logger.event(str(public_event.get("event_type", "game_event")), worker_id=self.worker_id, episode_id=self.current_assignment.episode_id, environment_step=int(result.environment_step), event=public_event)
            terminated = bool(result.terminated)
            truncated = bool(result.truncated)
            if terminated or truncated:
                self._active = False
                self.control.end()
            info = dict(getattr(result, "info", {}))
            info.update({"worker_id": self.worker_id, "episode_id": self.current_assignment.episode_id, "environment_step": int(result.environment_step), "termination_status": "normal" if terminated else "truncated" if truncated else "running"})
            return WorkerStep(next_observation, float(getattr(result, "reward", 0.0)), terminated, truncated, info, result)
        except WorkerCancelled:
            raise
        except Exception as exc:
            raise self._fault(exc) from exc

    def _artifact_metadata(self) -> Mapping[str, Any] | None:
        if self.inference_artifact is None:
            return None
        metadata = getattr(self.inference_artifact, "metadata", None)
        if isinstance(metadata, Mapping):
            return {key: deepcopy(value) for key, value in metadata.items() if key in {"artifact_type", "artifact_format_version", "inference_model_artifact_format_version", "phase"}}
        return {"artifact_type": type(self.inference_artifact).__name__}

    def _notify_agents_end(self, record: EpisodeRecord) -> None:
        if self.agents is None:
            return
        result = EpisodeResult(record.episode_id, record.terminated, record.truncated, record.metadata)
        values = self.agents.values() if isinstance(self.agents, Mapping) else self.agents
        for agent in values:
            callback = getattr(agent, "on_episode_end", None)
            if callback is not None:
                callback(result)

    def run_episode(self, assignment: EpisodeAssignment | None = None, *, episode_index: int = 0, episode_id: str | None = None, action_provider: Any = None) -> EpisodeRecord:
        assignment = assignment or self.assignment(episode_index, episode_id=episode_id)
        self.reset(assignment)
        try:
            while True:
                step = self.step(action_provider=action_provider)
                if step.terminated or step.truncated:
                    break
            assert self._initial_observation is not None
            result = step.result
            terminal = getattr(getattr(self.environment, "state", None), "terminal_result", None)
            terminal_dict = terminal.to_dict() if terminal is not None and hasattr(terminal, "to_dict") else None
            record = EpisodeRecord(
                self.worker_id,
                assignment.episode_index,
                assignment.episode_id,
                assignment.seed,
                step.terminated,
                step.truncated,
                "normal" if step.terminated else "truncated",
                int(step.info.get("environment_step", 0)),
                _mapping_observation(self._initial_observation),
                _mapping_observation(step.observation),
                tuple(deepcopy(self._actions)),
                tuple(deepcopy(self._events)),
                terminal_dict,
                {**self.metadata.to_dict(), "inference_artifact": self._artifact_metadata(), "termination_reason": step.info.get("termination_reason")},
            )
            self._notify_agents_end(record)
            if self.logger is not None:
                channel_event = "worker_episode_terminated" if record.terminated else "worker_episode_truncated"
                self.logger.event(channel_event, worker_id=self.worker_id, episode_id=record.episode_id, environment_steps=record.environment_steps, termination_status=record.termination_status)
            if self.privileged_logger is not None and self.environment is not None:
                self.privileged_logger.info("worker privileged episode boundary", worker_id=self.worker_id, episode_id=record.episode_id, roles=[int(role) for role in self.environment.state.roles], termination_status=record.termination_status)
            return record
        except WorkerCancelled:
            self._active = False
            self.control.end()
            raise
        except Exception as exc:
            raise self._fault(exc) from exc

    def pause(self) -> None:
        self.control.request_pause()

    def resume(self) -> None:
        self.control.resume()

    def cancel(self) -> None:
        self.control.cancel()

    def close(self) -> None:
        self._active = False
        self.control.end()
        if self.environment is not None and hasattr(self.environment, "close"):
            self.environment.close()
        self.environment = None


class DummyVecEnv:
    """Synchronous vectorized environment with one independent WorkerRuntime per slot."""

    def __init__(
        self,
        env_fns: Sequence[EnvironmentFactory] | None = None,
        *,
        num_workers: int | None = None,
        seed: int = 0,
        snapshot: JobSnapshot | Mapping[str, Any] | None = None,
        config: SystemConfig | None = None,
        environment_factory: EnvironmentFactory | None = None,
        action_provider: Any = None,
        agents: Sequence[Any] | None = None,
        hardware: Any = None,
        batch_size: int | None = None,
        buffer_size: int | None = None,
        job_id: str | None = None,
        logger: StructuredLogger | None = None,
        inference_artifact: Any = None,
        pause_timeout_seconds: float | None = None,
    ) -> None:
        if env_fns is None:
            count = int(num_workers if num_workers is not None else ResourceBudget.from_value(hardware).workers if hardware is not None else 1)
            if count < 1:
                raise ValueError("num_workers must be positive")
            env_fns = [environment_factory or (lambda **_: BangEnvironment(config=config.env if config else None)) for _ in range(count)]
        if not env_fns:
            raise ValueError("at least one environment factory is required")
        if num_workers is not None and int(num_workers) != len(env_fns):
            raise ValueError("num_workers must match the number of environment factories")
        self.n_envs = len(env_fns)
        self.num_workers = self.n_envs
        self.job_id = job_id
        self.seed_manager = SeedManager(seed, snapshot=snapshot)
        self.resource_guard = ResourceGuard(hardware if hardware is not None else config if config is not None else None) if hardware is not None or config is not None else None
        if self.resource_guard is not None:
            self.resource_guard.validate_workers(self.n_envs)
            self.resource_guard.validate_batch_buffer(batch_size, buffer_size)
        self.batch_size = self.resource_guard.budget.batch_size if self.resource_guard is not None and batch_size is None else int(batch_size or 64)
        self.buffer_size = self.resource_guard.budget.buffer_size if self.resource_guard is not None and buffer_size is None else int(buffer_size or 256)
        self._controls = [WorkerControl() for _ in range(self.n_envs)]
        pause_timeout = pause_timeout_seconds if pause_timeout_seconds is not None else config.pause.ack_timeout_seconds if config else 30.0
        self.pause_coordinator = PauseCoordinator(self._controls, ack_timeout_seconds=pause_timeout)
        self.workers = [
            WorkerRuntime(
                index,
                seed_manager=self.seed_manager,
                environment_factory=factory,
                action_provider=action_provider,
                agents=_clone_agent_graph(agents, worker_id=index, seed_manager=self.seed_manager),
                env_config=config.env if config else None,
                resource_guard=self.resource_guard,
                batch_size=self.batch_size if self.resource_guard is not None else None,
                buffer_size=self.buffer_size if self.resource_guard is not None else None,
                job_id=job_id,
                logger=logger,
                inference_artifact=inference_artifact,
                control=self._controls[index],
            )
            for index, factory in enumerate(env_fns)
        ]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "worker_count": self.n_envs,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "hardware": self.resource_guard.budget.to_dict() if self.resource_guard is not None else None,
            "worker_seeds": [worker.worker_seed for worker in self.workers],
            "job_id": self.job_id,
            "benchmark_executed": False,
            "auto_optimization": False,
        }

    def assignment_for_episode(self, episode_index: int) -> EpisodeAssignment:
        if isinstance(episode_index, bool) or int(episode_index) < 0:
            raise ValueError("episode_index must be a non-negative integer")
        worker_id = int(episode_index) % self.n_envs
        episode_id = f"{self.job_id + '-' if self.job_id else ''}episode-{int(episode_index)}"
        return self.workers[worker_id].assignment(int(episode_index), episode_id=episode_id)

    def reset(self, *, seed: int | None = None, episode_index: int = 0, indices: Iterable[int] | None = None) -> list[Any]:
        selected = list(range(self.n_envs)) if indices is None else [int(index) for index in indices]
        observations: list[Any] = [None] * self.n_envs
        for index in selected:
            assignment = self.assignment_for_episode(episode_index + index if indices is None else episode_index)
            if seed is not None:
                assignment = EpisodeAssignment(assignment.episode_index, assignment.worker_id, assignment.episode_id, self.seed_manager.environment_seed(assignment.episode_id, worker_id=index) if seed == self.seed_manager.root_seed else int(seed) + index)
            observations[index] = self.workers[index].reset(assignment)
        return observations

    def step(self, actions: Sequence[int]) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if len(actions) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} actions, got {len(actions)}")
        steps = [worker.step(action) for worker, action in zip(self.workers, actions)]
        observations = [step.observation for step in steps]
        rewards = np.asarray([step.reward for step in steps], dtype=np.float32)
        terminated = np.asarray([step.terminated for step in steps], dtype=np.bool_)
        truncated = np.asarray([step.truncated for step in steps], dtype=np.bool_)
        infos = [dict(step.info) for step in steps]
        return observations, rewards, terminated, truncated, infos

    def run_episodes(self, count: int, *, action_provider: Any = None) -> list[EpisodeRecord]:
        if isinstance(count, bool) or int(count) < 0:
            raise ValueError("count must be a non-negative integer")
        records: list[EpisodeRecord] = []
        for episode_index in range(int(count)):
            assignment = self.assignment_for_episode(episode_index)
            try:
                records.append(self.workers[assignment.worker_id].run_episode(assignment, action_provider=action_provider))
            except WorkerCancelled:
                break
        return records

    def pause(self, timeout: float | None = None) -> bool:
        acknowledged = self.pause_coordinator.pause(timeout)
        if not acknowledged and any(worker.logger is not None for worker in self.workers):
            for worker in self.workers:
                if worker.logger is not None:
                    worker.logger.error("pause failed", worker_id=worker.worker_id, state=PauseState.PAUSE_FAILED.value)
        return acknowledged

    def resume(self) -> None:
        self.pause_coordinator.resume()

    def cancel(self) -> None:
        self.pause_coordinator.cancel()

    def close(self) -> None:
        for worker in self.workers:
            worker.close()


# The public name used by training callers that do not care which backend is selected.
VectorizedEnvironment = DummyVecEnv
VecEnv = DummyVecEnv


def _subprocess_worker(connection: Any, worker_id: int, seed: int, environment_factory: EnvironmentFactory | None, env_config: EnvConfig | None, snapshot: Any, job_id: str | None) -> None:
    manager = SeedManager(seed, snapshot=snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot)
    worker = WorkerRuntime(worker_id, seed_manager=manager, environment_factory=environment_factory, env_config=env_config, job_id=job_id)
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                worker.close()
                connection.send((True, None))
                return
            if command == "reset":
                assignment = EpisodeAssignment(payload["episode_index"], worker_id, payload["episode_id"], payload["seed"])
                connection.send((True, worker.reset(assignment)))
            elif command == "step":
                value = worker.step(payload)
                connection.send((True, value))
            elif command == "pause":
                worker.pause()
                connection.send((True, True))
            elif command == "resume":
                worker.resume()
                connection.send((True, True))
            elif command == "cancel":
                worker.cancel()
                connection.send((True, True))
            else:
                raise ValueError(f"unknown worker command {command!r}")
    except Exception as exc:
        try:
            connection.send((False, {"worker_id": worker_id, "error": str(exc), "traceback": traceback.format_exc()}))
        except Exception:
            pass
    finally:
        worker.close()
        connection.close()


class SubprocVecEnv:
    """Spawned-process vector backend with the same public reset/step contract."""

    def __init__(self, env_fns: Sequence[EnvironmentFactory], *, seed: int = 0, snapshot: JobSnapshot | Mapping[str, Any] | None = None, config: SystemConfig | None = None, hardware: Any = None, batch_size: int | None = None, buffer_size: int | None = None, job_id: str | None = None, start_method: str = "spawn") -> None:
        if not env_fns:
            raise ValueError("at least one environment factory is required")
        self.n_envs = len(env_fns)
        self.num_workers = self.n_envs
        self.job_id = job_id
        self.seed_manager = SeedManager(seed, snapshot=snapshot)
        self.resource_guard = ResourceGuard(hardware if hardware is not None else config if config is not None else None) if hardware is not None or config is not None else None
        if self.resource_guard is not None:
            self.resource_guard.validate_workers(self.n_envs)
            self.resource_guard.validate_batch_buffer(batch_size, buffer_size)
        budget = self.resource_guard.budget if self.resource_guard is not None else ResourceBudget.from_value(None)
        self.batch_size = budget.batch_size if batch_size is None else int(batch_size)
        self.buffer_size = budget.buffer_size if buffer_size is None else int(buffer_size)
        context = mp.get_context(start_method)
        snapshot_payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot
        self._parents: list[Any] = []
        self._processes: list[Any] = []
        env_config = config.env if config else None
        for worker_id, factory in enumerate(env_fns):
            parent, child = context.Pipe()
            process = context.Process(target=_subprocess_worker, args=(child, worker_id, seed, factory, env_config, snapshot_payload, job_id), daemon=True)
            process.start()
            child.close()
            self._parents.append(parent)
            self._processes.append(process)
        self._closed = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "worker_count": self.n_envs,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "hardware": self.resource_guard.budget.to_dict() if self.resource_guard is not None else None,
            "worker_seeds": [self.seed_manager.worker_seed(index) for index in range(self.n_envs)],
            "job_id": self.job_id,
            "benchmark_executed": False,
            "auto_optimization": False,
        }

    def assignment_for_episode(self, episode_index: int) -> EpisodeAssignment:
        worker_id = int(episode_index) % self.n_envs
        episode_id = f"{self.job_id + '-' if self.job_id else ''}episode-{int(episode_index)}"
        return EpisodeAssignment(int(episode_index), worker_id, episode_id, self.seed_manager.environment_seed(episode_id, worker_id=worker_id))

    def _request_all(self, command: str, payloads: Sequence[Any]) -> list[Any]:
        if self._closed:
            raise WorkerFault(-1, "vector environment is closed")
        for connection, payload in zip(self._parents, payloads):
            connection.send((command, payload))
        values = []
        for worker_id, connection in enumerate(self._parents):
            try:
                ok, value = connection.recv()
            except (EOFError, OSError) as exc:
                raise WorkerFault(worker_id, "worker process disconnected", cause=exc) from exc
            if not ok:
                raise WorkerFault(worker_id, value.get("error", "unknown worker fault"), cause=RuntimeError(value.get("traceback", "")))
            values.append(value)
        return values

    def _request_one(self, worker_id: int, command: str, payload: Any) -> Any:
        if self._closed:
            raise WorkerFault(worker_id, "vector environment is closed")
        connection = self._parents[worker_id]
        try:
            connection.send((command, payload))
            ok, value = connection.recv()
        except (EOFError, OSError) as exc:
            raise WorkerFault(worker_id, "worker process disconnected", cause=exc) from exc
        if not ok:
            raise WorkerFault(worker_id, value.get("error", "unknown worker fault"), cause=RuntimeError(value.get("traceback", "")))
        return value

    def reset(self, *, episode_index: int = 0) -> list[Any]:
        assignments = [self.assignment_for_episode(episode_index + index) for index in range(self.n_envs)]
        values = self._request_all("reset", [{"episode_index": a.episode_index, "episode_id": a.episode_id, "seed": a.seed} for a in assignments])
        return values

    def step(self, actions: Sequence[int]) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if len(actions) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} actions, got {len(actions)}")
        values = self._request_all("step", [int(action) for action in actions])
        observations = [value.observation for value in values]
        rewards = np.asarray([value.reward for value in values], dtype=np.float32)
        terminated = np.asarray([value.terminated for value in values], dtype=np.bool_)
        truncated = np.asarray([value.truncated for value in values], dtype=np.bool_)
        infos = [dict(value.info) for value in values]
        return observations, rewards, terminated, truncated, infos

    def run_episodes(self, count: int, *, action_provider: Any = None) -> list[EpisodeRecord]:
        if isinstance(count, bool) or int(count) < 0:
            raise ValueError("count must be a non-negative integer")
        records: list[EpisodeRecord] = []
        for episode_index in range(int(count)):
            assignment = self.assignment_for_episode(episode_index)
            initial = self._request_one(
                assignment.worker_id,
                "reset",
                {"episode_index": assignment.episode_index, "episode_id": assignment.episode_id, "seed": assignment.seed},
            )
            observation = initial
            actions: list[Mapping[str, Any]] = []
            events: list[Mapping[str, Any]] = []
            while True:
                mask = np.asarray(observation["action_mask"] if isinstance(observation, Mapping) else observation.action_mask, dtype=np.uint8)
                action = _action_from_provider(action_provider, observation, mask)
                step = self._request_one(assignment.worker_id, "step", action)
                observation = step.observation
                result = step.result
                action_dict = result.action.to_dict() if result is not None and hasattr(result.action, "to_dict") else {"action": int(action)}
                action_dict.update({"worker_id": assignment.worker_id, "episode_id": assignment.episode_id, "environment_step": int(step.info.get("environment_step", 0))})
                actions.append(action_dict)
                for event in getattr(result, "events", ()):
                    events.append(event.to_public_dict() if hasattr(event, "to_public_dict") else event.to_dict())
                if step.terminated or step.truncated:
                    records.append(EpisodeRecord(
                        assignment.worker_id,
                        assignment.episode_index,
                        assignment.episode_id,
                        assignment.seed,
                        step.terminated,
                        step.truncated,
                        "normal" if step.terminated else "truncated",
                        int(step.info.get("environment_step", 0)),
                        _mapping_observation(initial),
                        _mapping_observation(observation),
                        tuple(actions),
                        tuple(events),
                        None,
                        dict(self.metadata),
                    ))
                    break
        return records

    def pause(self, timeout: float | None = None) -> bool:
        del timeout
        try:
            self._request_all("pause", [None] * self.n_envs)
            return True
        except WorkerFault:
            return False

    def resume(self) -> None:
        self._request_all("resume", [None] * self.n_envs)

    def cancel(self) -> None:
        try:
            self._request_all("cancel", [None] * self.n_envs)
        except WorkerFault:
            pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._request_all("close", [None] * self.n_envs)
        except (WorkerFault, EOFError, OSError):
            pass
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
        for connection in self._parents:
            connection.close()
        self._closed = True


__all__ = [
    "DummyVecEnv",
    "EnvironmentFactory",
    "EpisodeAssignment",
    "EpisodeRecord",
    "PauseCoordinator",
    "PauseState",
    "SubprocVecEnv",
    "VecEnv",
    "VectorizedEnvironment",
    "WorkerFault",
    "WorkerRuntime",
    "WorkerRuntimeError",
    "WorkerRuntimeMetadata",
    "WorkerStep",
]
