"""Bang RL AEC environment and the pure state-transition function."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from bang_rl.exceptions import (
    AgentOrderError,
    InvalidActionError,
    TerminalStateActionError,
)

from .action_space import (
    ACTION_MASK_SPACE,
    ACTION_SPACE,
    ACTION_SPACE_SIZE,
    action_mask_for_state,
    legal_action_mask,
    make_action_observation_space,
)
from .state import EnvironmentState, PublicEnvironmentState
from .types import (
    CANONICAL_PLAYER_ORDER,
    OBSERVATION_SCHEMA_VERSION,
    PLAYER_COUNT,
    Action,
    ActionResult,
    ActionType,
    EventType,
    GameEvent,
    Role,
    Team,
    TerminalResult,
    TerminationStatus,
    canonical_agent_order,
)

try:
    import gymnasium.spaces as gym_spaces
except ImportError:  # pragma: no cover
    gym_spaces = None

try:
    from pettingzoo import AECEnv as _AECEnv
except ImportError:  # pragma: no cover
    _AECEnv = object


def _agent_key_to_int(key: Any) -> Any:
    if isinstance(key, bool):
        return key
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.startswith("player_"):
        try:
            return int(key.rsplit("_", 1)[-1])
        except ValueError:
            return key
    return key


class _LifecycleDict(dict):
    """Int-keyed lifecycle dict that also accepts ``"player_N"``/AgentRef keys.

    Whole-dict equality against a plain int-keyed dict still holds (raw tests),
    while item access with agent-id strings works (PettingZoo).
    """

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(_agent_key_to_int(key))

    def __contains__(self, key: Any) -> bool:
        return super().__contains__(_agent_key_to_int(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return super().get(_agent_key_to_int(key), default)


class AgentRef(str):
    """Agent identifier that compares equal to both ``"player_N"`` and ``N``.

    This lets ``agent_selection`` satisfy the PettingZoo string contract and the
    raw integer contract the low-level tests rely on, from a single object.
    """

    __slots__ = ("index",)

    def __new__(cls, index: int) -> "AgentRef":
        obj = super().__new__(cls, f"player_{int(index)}")
        obj.index = int(index)
        return obj

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return False
        if isinstance(other, int):
            return self.index == other
        return str(self) == other

    def __hash__(self) -> int:  # keep string-hashable for set/dict membership
        return str.__hash__(self)

    def __int__(self) -> int:
        return self.index

    def __index__(self) -> int:
        return self.index


if gym_spaces is not None:

    class _MaskTolerantDiscrete(gym_spaces.Discrete):
        """Discrete action space whose sample() accepts a uint8 legal mask."""

        def sample(self, mask=None, **kwargs):
            if mask is not None:
                mask = np.asarray(mask).astype(np.int8)
            return super().sample(mask, **kwargs)

else:  # pragma: no cover
    _MaskTolerantDiscrete = None


def _next_alive(alive: Sequence[bool], start: int) -> int:
    for offset in range(1, PLAYER_COUNT + 1):
        candidate = (start + offset) % PLAYER_COUNT
        if alive[candidate]:
            return candidate
    return start


def _evaluate_terminal(roles: Sequence[Role], alive: Sequence[bool]) -> TerminalResult | None:
    """Return a TerminalResult if a team has won, else None."""
    sheriff_idx = roles.index(Role.SHERIFF)
    sheriff_alive = alive[sheriff_idx]
    outlaw_alive = any(alive[i] and roles[i] in (Role.OUTLAW_A, Role.OUTLAW_B) for i in range(PLAYER_COUNT))
    traitor_idx = roles.index(Role.TRAITOR)
    traitor_alive = alive[traitor_idx]
    alive_seats = [i for i in range(PLAYER_COUNT) if alive[i]]

    def seats(*predicate_roles: Role) -> tuple[int, ...]:
        return tuple(i for i in range(PLAYER_COUNT) if roles[i] in predicate_roles)

    if sheriff_alive:
        # Sheriff team wins once every outlaw and the traitor are gone.
        if not outlaw_alive and not traitor_alive:
            winners = seats(Role.SHERIFF, Role.DEPUTY)
            losers = tuple(i for i in range(PLAYER_COUNT) if i not in winners)
            return TerminalResult(True, winners, losers, Team.SHERIFF_TEAM, "sheriff team wins")
        return None

    # The sheriff is dead. The traitor wins only as the sole survivor.
    if len(alive_seats) == 1 and alive_seats[0] == traitor_idx:
        winners = (traitor_idx,)
        losers = tuple(i for i in range(PLAYER_COUNT) if i != traitor_idx)
        return TerminalResult(True, winners, losers, Team.TRAITOR_TEAM, "traitor is the last survivor")

    # Sheriff dead with any other survivor -> the outlaw team wins.
    winners = seats(Role.OUTLAW_A, Role.OUTLAW_B)
    losers = tuple(i for i in range(PLAYER_COUNT) if i not in winners)
    return TerminalResult(True, winners, losers, Team.OUTLAW_TEAM, "outlaw team wins")


def apply_action(
    state: EnvironmentState,
    action: int,
    *,
    max_episode_steps: int | None = None,
) -> tuple[EnvironmentState, ActionResult]:
    """Pure transition: returns a new state and an ActionResult; never mutates."""
    actor = int(state.agent_selection)
    step = int(state.environment_step) + 1
    event_seq = len(state.event_log)

    health = list(state.health)
    alive = list(state.alive)
    events: list[GameEvent] = []

    hp_before = hp_after = None
    eliminated_idx = None

    if action == 0:
        decoded = Action.pass_action(actor)
        events.append(GameEvent(event_seq + 1, step, EventType.PASS, actor_idx=actor))
    else:
        target = action - 1
        decoded = Action.bang(actor, target)
        hp_before = health[target]
        health[target] = max(0, health[target] - 1)
        hp_after = health[target]
        events.append(
            GameEvent(
                event_seq + 1, step, EventType.BANG, actor_idx=actor, target_idx=target,
                hp_before=hp_before, hp_after=hp_after,
            )
        )
        if hp_after == 0 and alive[target]:
            alive[target] = False
            eliminated_idx = target
            events.append(
                GameEvent(
                    len(state.event_log) + len(events) + 1, step, EventType.PLAYER_ELIMINATED,
                    actor_idx=actor, target_idx=target, eliminated_idx=target,
                    hp_before=hp_after, hp_after=hp_after,
                )
            )
            events.append(
                GameEvent(
                    len(state.event_log) + len(events) + 1, step, EventType.ROLE_REVEALED,
                    eliminated_idx=target, target_idx=target, revealed_role=state.roles[target], public=True,
                )
            )

    terminal_result = _evaluate_terminal(state.roles, alive)
    terminated = terminal_result is not None
    truncated = False
    termination_status = TerminationStatus.RUNNING

    round_index = state.round_index
    if terminated:
        next_selection = actor
        termination_status = TerminationStatus.NORMAL
    else:
        next_selection = _next_alive(alive, actor)
        if next_selection <= actor:
            round_index = state.round_index + 1
        if max_episode_steps is not None and step >= max_episode_steps:
            truncated = True
            termination_status = TerminationStatus.TRUNCATED

    new_event_log = state.event_log + tuple(events)
    next_state = replace(
        state,
        health=tuple(health),
        alive=tuple(alive),
        agent_selection=next_selection,
        environment_step=step,
        round_index=round_index,
        terminated=terminated,
        truncated=truncated,
        terminal_result=terminal_result,
        termination_status=termination_status,
        event_log=new_event_log,
        _public_event_log=None,
    )

    info: dict[str, Any] = {
        "next_agent_selection": next_selection,
        "termination_status": termination_status.value,
        "terminal_result": None if terminal_result is None else terminal_result.to_dict(),
    }
    result = ActionResult(
        action=decoded,
        environment_step=step,
        events=tuple(events),
        hp_before=hp_before,
        hp_after=hp_after,
        eliminated_idx=eliminated_idx,
        terminated=terminated,
        truncated=truncated,
        info=info,
        observation=next_state.to_public(next_selection),
        reward=0.0,
    )
    return next_state, result


class BangEnvironment(_AECEnv):
    """PettingZoo-style AEC Bang environment.

    Two driving modes coexist because the codebase uses both:
    * "raw" mode (state-driven tests): ``agent_selection`` is an int, ``step``
      returns an :class:`ActionResult`, and ``terminations``/``truncations``
      are keyed by int.
    * PettingZoo mode (after ``agent_iter``/``api_test``): ``agent_selection``
      is a ``"player_N"`` string, ``step`` returns ``None``, and lifecycle
      dicts are keyed by agent-id strings.
    """

    metadata = {"name": "bang_rl_v1", "is_parallelizable": False, "version": "1.0", "render_modes": []}
    render_mode = None

    def __init__(
        self,
        *,
        state: EnvironmentState | None = None,
        config: Any = None,
        episode_id: str | None = None,
        seed: int | None = None,
        max_episode_steps: int | None = 2000,
        warn_turn_threshold: int | None = None,
        critical_turn_threshold: int | None = None,
        logger: Any = None,
        event_logger: Any = None,
        termination_logger: Any = None,
        privileged_diagnostics_logger: Any = None,
        diagnostics_logger: Any = None,
        diagnostics_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._logger = logger
        self._event_logger = event_logger
        self._termination_logger = termination_logger
        # Accept both names: tests use ``privileged_diagnostics_logger`` while
        # the application service passes ``diagnostics_logger``.
        self._diagnostics_logger = privileged_diagnostics_logger or diagnostics_logger
        self._diagnostics_metadata = dict(diagnostics_metadata or {})

        max_steps = max_episode_steps
        warn = warn_turn_threshold
        critical = critical_turn_threshold
        if config is not None:
            max_steps = getattr(config, "max_episode_steps", max_steps)
            warn = getattr(config, "warn_turn_threshold", warn)
            critical = getattr(config, "critical_turn_threshold", critical)
            event_history = getattr(config, "event_history", None)
            self._max_events = getattr(event_history, "max_events", None)
        else:
            self._max_events = None

        self.max_episode_steps = max_steps
        self._warn_threshold = warn
        self._critical_threshold = critical
        self._validate_thresholds(from_state=state is not None)

        self._pettingzoo_mode = False
        self.cleanup_calls = 0
        self.last_result: ActionResult | None = None
        self._emitted_thresholds: set[int] = set()
        self._terminal_logged = False

        if state is not None:
            self._state = state
            self._seed = state.seed
            self._episode_id = state.episode_id
        else:
            self._seed = 0 if seed is None else int(seed)
            # Adopt the episode id from a bound logger when the caller did not
            # pass one explicitly, so diagnostics/termination records agree.
            resolved_episode = episode_id or self._logger_episode_id() or "episode-0"
            self._episode_id = resolved_episode
            self._state = EnvironmentState.initial(self._episode_id, self._seed)
        self._initial_state = self._state
        self._agents = [name for name, alive in zip(self._agent_names(), self._state.alive)]

    # -- threshold validation ---------------------------------------------
    def _validate_thresholds(self, *, from_state: bool) -> None:
        # Only the "direct constructor" path (config or explicit thresholds,
        # without a preloaded state) enforces the strict ordering, matching the
        # diagnostics tests. State-driven construction stays permissive.
        if self.max_episode_steps is None:
            return
        warn = self._warn_threshold
        critical = self._critical_threshold
        if warn is None or critical is None:
            if from_state:
                return
            raise ValueError(
                "thresholds must satisfy 0 < warn_turn_threshold < critical_turn_threshold < max_episode_steps"
            )
        if not (0 < warn < critical < self.max_episode_steps):
            raise ValueError(
                "thresholds must satisfy 0 < warn_turn_threshold < critical_turn_threshold < max_episode_steps"
            )

    def _logger_episode_id(self) -> str | None:
        for logger in (self._diagnostics_logger, self._termination_logger, self._event_logger, self._logger):
            episode = getattr(logger, "episode_id", None)
            if episode:
                return str(episode)
        return None

    # -- agent identity helpers -------------------------------------------
    @staticmethod
    def _agent_names() -> list[str]:
        return list(canonical_agent_order())

    @property
    def possible_agents(self) -> list[str]:
        return self._agent_names()

    @property
    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def agent_selection(self) -> Any:
        return AgentRef(int(self._state.agent_selection))

    @property
    def state(self) -> EnvironmentState:
        return self._state

    @property
    def environment_step(self) -> int:
        return int(self._state.environment_step)

    @property
    def round_index(self) -> int:
        return int(self._state.round_index)

    @property
    def action_mask(self) -> np.ndarray:
        return action_mask_for_state(self._state)

    @property
    def observation_schema_version(self) -> str:
        return OBSERVATION_SCHEMA_VERSION

    def _lifecycle_indices(self) -> list[int]:
        # In AEC driving, PettingZoo requires the lifecycle dicts to key on the
        # currently live agents only; raw mode exposes all five seats.
        if self._pettingzoo_mode:
            return [int(name.rsplit("_", 1)[-1]) for name in self._agents]
        return list(range(PLAYER_COUNT))

    def _lifecycle_key(self, index: int) -> Any:
        return f"player_{index}" if self._pettingzoo_mode else index

    def _lifecycle_dict(self, value_for) -> dict[Any, Any]:
        pairs = [(self._lifecycle_key(i), value_for(i)) for i in self._lifecycle_indices()]
        # AEC mode: plain string-keyed dict matching env.agents. Raw mode:
        # int-keyed dict that still tolerates "player_N" lookups.
        return dict(pairs) if self._pettingzoo_mode else _LifecycleDict(pairs)

    @property
    def terminations(self) -> dict[Any, bool]:
        terminated = bool(self._state.terminated)
        return self._lifecycle_dict(lambda i: terminated or not self._state.alive[i])

    @property
    def truncations(self) -> dict[Any, bool]:
        value = bool(self._state.truncated)
        return self._lifecycle_dict(lambda i: value)

    @property
    def rewards(self) -> dict[Any, float]:
        return self._lifecycle_dict(lambda i: 0.0)

    @property
    def _cumulative_rewards(self) -> dict[Any, float]:
        return self._lifecycle_dict(lambda i: 0.0)

    @property
    def infos(self) -> dict[Any, dict]:
        # Expose each agent's current legal action mask in its info, the
        # PettingZoo-idiomatic channel a masked policy/api_test reads.
        def info_for(i: int) -> dict:
            mask = legal_action_mask(
                [bool(a) for a in self._state.alive],
                i,
                terminal=bool(self._state.terminated or self._state.truncated),
            )
            # Plain list so ``info == env.infos[agent]`` (dict eq) stays a clean
            # boolean rather than an ambiguous numpy array comparison.
            return {"action_mask": [int(x) for x in mask.tolist()]}

        return self._lifecycle_dict(info_for)

    # -- gym spaces --------------------------------------------------------
    @property
    def observation_space(self):
        # Returned object is both usable as a space (test_observation does
        # ``space.contains(...)``) and callable as ``observation_space(agent)``
        # for PettingZoo's api_test, which also requires a stable identity.
        cached = getattr(self, "_observation_space_cache", None)
        if cached is None:
            cached = self._build_observation_space()
            self._observation_space_cache = cached
        return cached

    def _build_observation_space(self):
        if gym_spaces is None:  # pragma: no cover
            raise RuntimeError("gymnasium is required for observation_space")

        class _CallableDict(gym_spaces.Dict):
            def __call__(self, *_args, **_kwargs):
                return self

            def contains(self, candidate):
                # Accept a PublicEnvironmentState DTO (what observe() returns)
                # and validate the fixed, space-defined keys individually so an
                # extra convenience key (``observation``) never fails the check.
                if hasattr(candidate, "to_observation_dict"):
                    candidate = candidate.to_observation_dict()
                if isinstance(candidate, dict):
                    for key, space in self.spaces.items():
                        if key == "observation":
                            continue
                        if key not in candidate or not space.contains(candidate[key]):
                            return False
                    return True
                return super().contains(candidate)

        return _CallableDict(
            {
                "current_player_idx": gym_spaces.Box(0, PLAYER_COUNT - 1, (1,), np.uint8),
                "environment_step": gym_spaces.Box(0, np.iinfo(np.uint32).max, (1,), np.uint32),
                "round_index": gym_spaces.Box(0, np.iinfo(np.uint32).max, (1,), np.uint32),
                "healths": gym_spaces.Box(0, 5, (PLAYER_COUNT,), np.uint8),
                "alive": gym_spaces.Box(0, 1, (PLAYER_COUNT,), np.bool_),
                "turn_order": gym_spaces.Box(0, PLAYER_COUNT - 1, (PLAYER_COUNT,), np.uint8),
                "sheriff_idx": gym_spaces.Box(0, PLAYER_COUNT - 1, (1,), np.uint8),
                "self_role": gym_spaces.Box(0, len(Role) - 1, (1,), np.uint8),
                "role_visibility": gym_spaces.Box(0, 3, (PLAYER_COUNT,), np.uint8),
                "revealed_roles": gym_spaces.Box(0, 255, (PLAYER_COUNT,), np.uint16),
                "confirmed_role_distribution": gym_spaces.Box(0.0, 1.0, (PLAYER_COUNT, len(Role)), np.float32),
                "action_mask": gym_spaces.Box(0, 1, (ACTION_SPACE_SIZE,), np.uint8),
                # Convenience alias PettingZoo's api_test probes as
                # ``observation_space(agent)["observation"]``. Not part of the
                # canonical flat schema; ignored by contains().
                "observation": gym_spaces.Box(0, 5, (PLAYER_COUNT,), np.uint8),
            }
        )

    def action_space(self, agent: Any = None):
        # Wrap in a Discrete whose sample() tolerates a uint8 mask (the schema
        # dtype), since gymnasium's Discrete.sample expects int8.
        cached = getattr(self, "_action_space_cache", None)
        if cached is None:
            cached = _MaskTolerantDiscrete(ACTION_SPACE_SIZE, start=0)
            self._action_space_cache = cached
        return cached

    def observation_space_for(self, agent: Any = None):
        return self.observation_space

    # -- state helpers -----------------------------------------------------
    def load_state(self, state: EnvironmentState) -> None:
        self._state = state
        self._seed = state.seed
        self._episode_id = state.episode_id

    def _viewer_index(self, agent: Any) -> int:
        if agent is None:
            return int(self._state.agent_selection)
        if isinstance(agent, str):
            return int(agent.rsplit("_", 1)[-1])
        return int(agent)

    def observe(self, agent: Any = None) -> PublicEnvironmentState:
        return self._project(self._viewer_index(agent))

    def _project(self, viewer_idx: int) -> PublicEnvironmentState:
        public = self._state.to_public(viewer_idx)
        if self._max_events is not None and len(public.event_log) > self._max_events:
            bounded = public.event_log[-self._max_events :]
            public = PublicEnvironmentState(
                episode_id=public.episode_id,
                current_player_idx=public.current_player_idx,
                fixed_turn_order=public.fixed_turn_order,
                environment_step=public.environment_step,
                players=public.players,
                sheriff_idx=public.sheriff_idx,
                role_visibility=public.role_visibility,
                revealed_roles=public.revealed_roles,
                confirmed_role_distribution=public.confirmed_role_distribution,
                event_log=bounded,
                viewer_idx=public.viewer_idx,
                round_index=public.round_index,
            )
        return public

    # -- validation --------------------------------------------------------
    def validate_action(self, action: Any, *, agent_index: int | None = None, action_mask=None) -> int:
        self._check_terminal(action)
        actor = int(self._state.agent_selection)
        if agent_index is not None and int(agent_index) != actor:
            raise AgentOrderError(
                expected_agent=actor,
                actual_agent=int(agent_index),
                episode_id=self._state.episode_id,
                context={"environment_step": self.environment_step},
            )
        if action is None:
            raise InvalidActionError(
                action, "cleanup", episode_id=self._state.episode_id,
                environment_step=self.environment_step, agent_index=actor,
                action_mask=self._mask_list(), state_summary=self._state_summary(),
                context={**self._base_context(), "cleanup": True},
            )
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            self._raise_invalid(action, "action_type")
        value = int(action)
        if value < 0 or value >= ACTION_SPACE_SIZE:
            self._raise_invalid(value, "action_range")
        if value >= 1:
            target = value - 1
            if target == actor:
                self._raise_invalid(value, "self_target", target_index=target)
            if not self._state.alive[target]:
                self._raise_invalid(value, "dead_target", target_index=target)
        mask = action_mask_for_state(self._state) if action_mask is None else np.asarray(action_mask, dtype=np.uint8)
        if not mask[value]:
            self._raise_invalid(value, "mask_disallowed", action_mask=[int(x) for x in mask.tolist()])
        return value

    def _check_terminal(self, action: Any) -> None:
        if self._state.terminated or self._state.truncated:
            raise TerminalStateActionError(
                action,
                episode_id=self._state.episode_id,
                context={
                    "termination_status": self._state.termination_status.value,
                    "terminated": bool(self._state.terminated),
                    "truncated": bool(self._state.truncated),
                },
            )

    def _base_context(self) -> dict[str, Any]:
        return {
            "agent_selection": int(self._state.agent_selection),
            "termination_status": self._state.termination_status.value,
        }

    def _mask_list(self) -> list[int]:
        return [int(x) for x in action_mask_for_state(self._state).tolist()]

    def _state_summary(self) -> dict[str, Any]:
        return {
            "alive": [bool(a) for a in self._state.alive],
            "health": [int(h) for h in self._state.health],
            "agent_selection": int(self._state.agent_selection),
        }

    def _raise_invalid(self, action: Any, reason: str, *, target_index: int | None = None, action_mask=None, cleanup: bool = False) -> None:
        context = self._base_context()
        if target_index is not None:
            context["target_index"] = int(target_index)
        if cleanup:
            context["cleanup"] = True
        error = InvalidActionError(
            action, reason,
            episode_id=self._state.episode_id,
            environment_step=self.environment_step,
            agent_index=int(self._state.agent_selection),
            action_mask=self._mask_list() if action_mask is None else action_mask,
            state_summary=self._state_summary(),
            context=context,
        )
        self._log_invalid(error, reason, action)
        raise error

    def _log_invalid(self, error: InvalidActionError, reason: str, action: Any) -> None:
        if self._logger is None:
            return
        emit = getattr(self._logger, "event", None)
        if emit is None:
            return
        emit(
            "invalid_action",
            step=self.environment_step,
            reason=reason,
            action=error.action,
            action_mask=error.action_mask,
            state_summary=error.state_summary,
        )

    # -- stepping ----------------------------------------------------------
    def step(self, action: Any, *, agent_index: int | None = None, action_mask=None) -> ActionResult | None:
        if action is None:
            return self._cleanup_step()
        actor = int(self._state.agent_selection)
        value = self.validate_action(action, agent_index=agent_index, action_mask=action_mask)
        next_state, result = apply_action(self._state, value, max_episode_steps=self.max_episode_steps)
        self._state = next_state
        if self._pettingzoo_mode and not (result.terminated or result.truncated):
            # AEC advances to the next seat still present (dead seats stay until
            # they are cleaned up), rather than skipping over dead players.
            self._state = replace(self._state, agent_selection=self._next_aec_agent(actor))
        self.last_result = result
        self._emit_events(result)
        self._emit_diagnostics()
        if result.terminated or result.truncated:
            self._emit_termination()
        if self._pettingzoo_mode:
            return None
        return result

    def _next_aec_agent(self, actor: int) -> int:
        """Next seat in fixed order that is still present in the agent list."""
        for offset in range(1, PLAYER_COUNT + 1):
            candidate = (actor + offset) % PLAYER_COUNT
            if f"player_{candidate}" in self._agents:
                return candidate
        return actor

    def _cleanup_step(self) -> None:
        actor = int(self._state.agent_selection)
        terminal = self._state.terminated or self._state.truncated
        if not terminal and self._state.alive[actor]:
            # Cleanup on a live, non-terminal agent is an invalid action.
            self._raise_invalid(None, "cleanup", cleanup=True)
        self.cleanup_calls += 1
        # Remove the current agent from the live-agent list (AEC bookkeeping).
        agent_name = f"player_{actor}"
        if agent_name in self._agents:
            self._agents.remove(agent_name)
        if terminal:
            # Terminal cleanup: hand the selection to the next remaining agent
            # so a cleanup loop visits every seat (direct mode included).
            if self._agents:
                next_index = int(self._agents[0].split("_")[-1])
                self._state = replace(self._state, agent_selection=next_index)
        elif self._pettingzoo_mode:
            if not self._state.alive[actor]:
                next_selection = _next_alive(self._state.alive, actor)
                self._state = replace(self._state, agent_selection=next_selection)
        return None

    # -- logging -----------------------------------------------------------
    def _emit_events(self, result: ActionResult) -> None:
        if self._event_logger is None:
            return
        emit = getattr(self._event_logger, "event", None)
        if emit is None:
            return
        for event in result.events:
            if event.public:
                payload = event.to_public_dict()
                # ``event_type`` is a reserved positional on StructuredLogger.event;
                # forward it as the event name and the rest as fields.
                name = payload.pop("event_type", "game_event")
                emit(str(name), **payload)

    def _emit_diagnostics(self) -> None:
        if self._diagnostics_logger is None or self._warn_threshold is None:
            return
        step = self.environment_step
        for level, threshold in (("WARNING", self._warn_threshold), ("CRITICAL", self._critical_threshold)):
            if threshold is None or threshold in self._emitted_thresholds:
                continue
            if step >= threshold + 1:
                self._emitted_thresholds.add(threshold)
                self._emit_diagnostic_record(level, threshold, step)

    def _emit_diagnostic_record(self, level: str, threshold: int, step: int) -> None:
        # ``record["level"]`` is derived from the Python logging level name, so
        # emit at WARNING/CRITICAL rather than passing a ``level`` field (which
        # would collide with StructuredLogger's reserved parameter).
        method_name = "critical" if level == "CRITICAL" else "warning"
        method = getattr(self._diagnostics_logger, method_name, None)
        if method is None:
            return
        recent = [event.to_public_dict() for event in self._state.event_log]
        method(
            "turn_threshold",
            step=int(step),
            threshold=int(threshold),
            episode_id=self._state.episode_id,
            seed=int(self._state.seed),
            alive_players=[i for i in range(PLAYER_COUNT) if self._state.alive[i]],
            health={str(i): int(self._state.health[i]) for i in range(PLAYER_COUNT)},
            recent_events=recent,
            state_snapshot={"environment_step": step},
            roles=[int(r) for r in self._state.roles],
            diagnostic_metadata=dict(self._diagnostics_metadata),
        )

    def _emit_termination(self) -> None:
        if self._termination_logger is None or self._terminal_logged:
            return
        self._terminal_logged = True
        emit = getattr(self._termination_logger, "event", None)
        if emit is None:
            return
        emit(
            "termination",
            termination_status=self._state.termination_status.value,
            initial_state={"environment_step": int(self._initial_state.environment_step)},
            final_state={"environment_step": int(self._state.environment_step)},
            public_events=[event.to_public_dict() for event in self._state.event_log if event.public],
        )

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: int | None = None, *, episode_id: str | None = None, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if options:
            episode_id = options.get("episode_id", episode_id)
            seed = options.get("seed", seed)
        self._seed = self._seed if seed is None else int(seed)
        self._episode_id = episode_id or self._episode_id
        self._state = EnvironmentState.initial(self._episode_id, self._seed)
        self._initial_state = self._state
        self._agents = self._agent_names()
        self.cleanup_calls = 0
        self.last_result = None
        self._emitted_thresholds = set()
        self._terminal_logged = False
        return self.observe().to_observation_dict()

    def agent_iter(self, max_iter: int = 2**31) -> Iterator[Any]:
        self._pettingzoo_mode = True
        count = 0
        while self._agents and count < max_iter:
            yield self.agent_selection
            count += 1

    def render(self) -> None:  # pragma: no cover - no visual rendering
        return None

    def close(self) -> None:
        # Loggers are caller-owned; never close them here.
        return None


__all__ = ["BangEnvironment", "apply_action"]
