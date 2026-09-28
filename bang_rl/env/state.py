"""Privileged and public game state DTOs with the role-privacy boundary."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .action_space import legal_action_mask
from .types import (
    ACTION_SPACE_SIZE,
    CANONICAL_PLAYER_ORDER,
    HIDDEN_ROLE_SENTINEL,
    OBSERVATION_SCHEMA_VERSION,
    PLAYER_COUNT,
    GameEvent,
    Role,
    RoleVisibility,
    TerminalResult,
    TerminationStatus,
    _as_role,
    _as_termination_status,
    assign_roles,
    initial_health,
)

ROLE_COUNT = len(Role)


def _public_projection(event_log: Sequence[GameEvent]) -> tuple[GameEvent, ...]:
    return tuple(event for event in event_log if event.public)


@dataclass(frozen=True)
class PublicPlayerState:
    player_idx: int
    health: int
    alive: bool
    role: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_idx": int(self.player_idx),
            "health": int(self.health),
            "alive": bool(self.alive),
            "role": None if self.role is None else int(self.role),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PublicPlayerState":
        role = data.get("role")
        return cls(
            player_idx=int(data["player_idx"]),
            health=int(data["health"]),
            alive=bool(data["alive"]),
            role=None if role is None else int(role),
        )


@dataclass(frozen=True)
class EnvironmentState:
    episode_id: str
    seed: int
    roles: tuple[Role, ...]
    health: tuple[int, ...]
    alive: tuple[bool, ...]
    agent_selection: int = 0
    environment_step: int = 0
    round_index: int = 0
    fixed_turn_order: tuple[int, ...] = CANONICAL_PLAYER_ORDER
    terminated: bool = False
    truncated: bool = False
    terminal_result: TerminalResult | None = None
    termination_status: TerminationStatus = TerminationStatus.RUNNING
    event_log: tuple[GameEvent, ...] = ()
    _public_event_log: tuple[GameEvent, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(_as_role(role) for role in self.roles))
        object.__setattr__(self, "health", tuple(int(h) for h in self.health))
        object.__setattr__(self, "alive", tuple(bool(a) for a in self.alive))
        object.__setattr__(self, "fixed_turn_order", tuple(int(i) for i in self.fixed_turn_order))
        object.__setattr__(self, "termination_status", _as_termination_status(self.termination_status))
        object.__setattr__(self, "event_log", tuple(self.event_log))
        if len(self.roles) != PLAYER_COUNT:
            raise ValueError("state requires exactly five roles")
        if sorted(self.roles) != sorted(Role):
            raise ValueError("roles must be a permutation of the five canonical roles")
        if len(self.health) != PLAYER_COUNT or len(self.alive) != PLAYER_COUNT:
            raise ValueError("health and alive must have five entries")
        # Validate/populate the cached public event projection.
        expected_public = _public_projection(self.event_log)
        if self._public_event_log is None:
            object.__setattr__(self, "_public_event_log", expected_public)
        else:
            cache = tuple(self._public_event_log)
            if any(not event.public for event in cache):
                raise ValueError("cached public event projection must contain only public events")
            if cache != expected_public:
                raise ValueError("cached public event projection does not match the event log")
            object.__setattr__(self, "_public_event_log", cache)
        # Serialize the public projection eagerly (once) so later viewer DTOs
        # consume this trusted payload instead of reserializing events.
        object.__setattr__(
            self,
            "_cachedevent_dicts",
            tuple(event.to_public_dict() for event in self._public_event_log),
        )

    # -- convenience -------------------------------------------------------
    @property
    def sheriff_idx(self) -> int:
        return int(self.roles.index(Role.SHERIFF))

    @property
    def players(self) -> tuple[PublicPlayerState, ...]:
        return tuple(
            PublicPlayerState(index, self.health[index], self.alive[index], int(self.roles[index]))
            for index in range(PLAYER_COUNT)
        )

    @classmethod
    def initial(cls, episode_id: str, seed: int) -> "EnvironmentState":
        roles = assign_roles(seed)
        return cls(
            episode_id=str(episode_id),
            seed=int(seed),
            roles=roles,
            health=initial_health(roles),
            alive=tuple(True for _ in range(PLAYER_COUNT)),
            agent_selection=0,
        )

    # -- privileged serialization -----------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": int(self.seed),
            "roles": [int(role) for role in self.roles],
            "health": [int(h) for h in self.health],
            "alive": [bool(a) for a in self.alive],
            "agent_selection": int(self.agent_selection),
            "environment_step": int(self.environment_step),
            "round_index": int(self.round_index),
            "fixed_turn_order": [int(i) for i in self.fixed_turn_order],
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "terminal_result": None if self.terminal_result is None else self.terminal_result.to_dict(),
            "termination_status": self.termination_status.value,
            "event_log": [event.to_dict() for event in self.event_log],
        }

    to_privileged_dict = to_dict

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentState":
        terminal = data.get("terminal_result")
        return cls(
            episode_id=str(data["episode_id"]),
            seed=int(data["seed"]),
            roles=tuple(_as_role(role) for role in data["roles"]),
            health=tuple(int(h) for h in data["health"]),
            alive=tuple(bool(a) for a in data["alive"]),
            agent_selection=int(data.get("agent_selection", 0)),
            environment_step=int(data.get("environment_step", 0)),
            round_index=int(data.get("round_index", 0)),
            fixed_turn_order=tuple(int(i) for i in data.get("fixed_turn_order", CANONICAL_PLAYER_ORDER)),
            terminated=bool(data.get("terminated", False)),
            truncated=bool(data.get("truncated", False)),
            terminal_result=None if terminal is None else TerminalResult.from_dict(terminal),
            termination_status=_as_termination_status(data.get("termination_status", "running")),
            event_log=tuple(GameEvent.from_dict(event) for event in data.get("event_log", ())),
        )

    # -- public projection -------------------------------------------------
    def _publicevent_dicts(self) -> tuple[dict[str, Any], ...]:
        """Return the public event projection serialized once at construction."""
        return getattr(self, "_cachedevent_dicts", ())

    def _known_roles(self) -> dict[int, Role]:
        # The Sheriff's identity is always public, and every eliminated (dead)
        # player's role is publicly revealed. Still-living non-sheriff roles
        # stay hidden regardless of any injected reveal event.
        known: dict[int, Role] = {self.sheriff_idx: Role.SHERIFF}
        for index in range(PLAYER_COUNT):
            if not self.alive[index]:
                known[index] = self.roles[index]
        return known

    def to_public(self, viewer_idx: int) -> "PublicEnvironmentState":
        viewer_idx = int(viewer_idx)
        public_events = self._public_event_log or ()

        known = self._known_roles()
        known[viewer_idx] = self.roles[viewer_idx]

        # By game rule only the Sheriff (and eliminated players) are ever
        # public: a living non-sheriff, non-self role must never be inferred
        # for the viewer. The former "logical deduction" step could reveal a
        # living teammate (e.g. a deputy's teammate) by elimination of
        # alternatives, so it is intentionally not performed here.

        revealed_roles: list[int] = []
        visibility: list[RoleVisibility] = []
        distribution: list[tuple[float, ...]] = []
        for index in range(PLAYER_COUNT):
            role = known.get(index)
            if role is None:
                revealed_roles.append(HIDDEN_ROLE_SENTINEL)
                visibility.append(RoleVisibility.HIDDEN)
                distribution.append(tuple(0.0 for _ in range(ROLE_COUNT)))
                continue
            revealed_roles.append(int(role))
            if index == viewer_idx:
                visibility.append(RoleVisibility.SELF)
            else:
                visibility.append(RoleVisibility.REVEALED)
            row = [0.0] * ROLE_COUNT
            row[int(role)] = 1.0
            distribution.append(tuple(row))

        players = tuple(
            PublicPlayerState(
                index,
                self.health[index],
                self.alive[index],
                revealed_roles[index] if revealed_roles[index] != HIDDEN_ROLE_SENTINEL else None,
            )
            for index in range(PLAYER_COUNT)
        )

        return PublicEnvironmentState(
            episode_id=self.episode_id,
            current_player_idx=int(self.agent_selection),
            fixed_turn_order=self.fixed_turn_order,
            environment_step=int(self.environment_step),
            players=players,
            sheriff_idx=self.sheriff_idx,
            role_visibility=tuple(visibility),
            revealed_roles=tuple(revealed_roles),
            confirmed_role_distribution=tuple(distribution),
            event_log=public_events,
            viewer_idx=viewer_idx,
            round_index=int(self.round_index),
            event_dicts=self._publicevent_dicts(),
        )


@dataclass(frozen=True)
class PublicEnvironmentState:
    episode_id: str
    current_player_idx: int
    fixed_turn_order: tuple[int, ...]
    environment_step: int
    players: tuple[PublicPlayerState, ...]
    sheriff_idx: int
    role_visibility: tuple[RoleVisibility, ...]
    revealed_roles: tuple[int, ...]
    confirmed_role_distribution: tuple[tuple[float, ...], ...]
    event_log: tuple[GameEvent, ...]
    viewer_idx: int
    round_index: int = 0
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    event_dicts: tuple[dict[str, Any], ...] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_log", tuple(self.event_log))
        if any(not event.public for event in self.event_log):
            raise ValueError("public state must not contain private events")
        # Consume the caller's pre-serialized projection when provided; only
        # serialize here as a fallback (e.g. direct construction in tests).
        if self.event_dicts is None:
            object.__setattr__(self, "event_dicts", tuple(event.to_public_dict() for event in self.event_log))
        else:
            object.__setattr__(self, "event_dicts", tuple(self.event_dicts))

    # -- mapping-like access ----------------------------------------------
    @property
    def event_sequence(self) -> tuple[GameEvent, ...]:
        return self.event_log

    def keys(self) -> tuple[str, ...]:
        return tuple(self.to_observation_dict().keys())

    def __getitem__(self, key: str) -> Any:
        return self.to_observation_dict()[key]

    def __contains__(self, key: object) -> bool:
        # Behave like a mapping so ``"action_mask" in observation`` works and
        # never falls back to positional __getitem__ probing (which raises
        # KeyError on integer indices).
        return isinstance(key, str) and key in self.to_observation_dict()

    def get(self, key: str, default: Any = None) -> Any:
        values = self.to_observation_dict()
        return values.get(key, default)

    def __iter__(self):
        return iter(self.to_observation_dict())

    # -- observation schema -----------------------------------------------
    def to_observation_dict(self) -> dict[str, Any]:
        healths = np.asarray([p.health for p in self.players], dtype=np.uint8)
        alive = np.asarray([p.alive for p in self.players], dtype=np.bool_)
        self_role = int(self.revealed_roles[self.viewer_idx])
        mask = legal_action_mask(
            [bool(a) for a in alive.tolist()],
            self.current_player_idx,
        )
        events = tuple(copy.deepcopy(dict(item)) for item in self.event_dicts)
        return {
            "current_player_idx": np.asarray([self.current_player_idx], dtype=np.uint8),
            "environment_step": np.asarray([self.environment_step], dtype=np.uint32),
            "round_index": np.asarray([self.round_index], dtype=np.uint32),
            "healths": healths,
            "alive": alive,
            "turn_order": np.asarray(self.fixed_turn_order, dtype=np.uint8),
            "sheriff_idx": np.asarray([self.sheriff_idx], dtype=np.uint8),
            "self_role": np.asarray([self_role], dtype=np.uint8),
            "role_visibility": np.asarray([int(v) for v in self.role_visibility], dtype=np.uint8),
            "revealed_roles": np.asarray([int(r) for r in self.revealed_roles], dtype=np.uint16),
            "confirmed_role_distribution": np.asarray(self.confirmed_role_distribution, dtype=np.float32),
            "action_mask": mask.astype(np.uint8),
            # Alias probed by PettingZoo's api_test; mirrors public healths.
            "observation": healths,
            "event_sequence": events,
            "event_history": tuple(copy.deepcopy(dict(item)) for item in self.event_dicts),
            "observation_schema_version": self.observation_schema_version,
        }

    def to_dict(self) -> dict[str, Any]:
        event_dicts = [copy.deepcopy(dict(item)) for item in self.event_dicts]
        return {
            "episode_id": self.episode_id,
            "current_player_idx": int(self.current_player_idx),
            "fixed_turn_order": [int(i) for i in self.fixed_turn_order],
            "environment_step": int(self.environment_step),
            "round_index": int(self.round_index),
            "players": [p.to_dict() for p in self.players],
            "sheriff_idx": int(self.sheriff_idx),
            "role_visibility": [int(v) for v in self.role_visibility],
            "revealed_roles": [int(r) for r in self.revealed_roles],
            "confirmed_role_distribution": [list(row) for row in self.confirmed_role_distribution],
            "event_log": event_dicts,
            "event_sequence": [copy.deepcopy(dict(item)) for item in self.event_dicts],
            "viewer_idx": int(self.viewer_idx),
            "observation_schema_version": self.observation_schema_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PublicEnvironmentState":
        return cls(
            episode_id=str(data["episode_id"]),
            current_player_idx=int(data["current_player_idx"]),
            fixed_turn_order=tuple(int(i) for i in data["fixed_turn_order"]),
            environment_step=int(data["environment_step"]),
            players=tuple(PublicPlayerState.from_dict(p) for p in data["players"]),
            sheriff_idx=int(data["sheriff_idx"]),
            role_visibility=tuple(RoleVisibility(int(v)) for v in data["role_visibility"]),
            revealed_roles=tuple(int(r) for r in data["revealed_roles"]),
            confirmed_role_distribution=tuple(tuple(float(x) for x in row) for row in data["confirmed_role_distribution"]),
            event_log=tuple(GameEvent.from_dict(e) for e in data.get("event_log", ())),
            viewer_idx=int(data["viewer_idx"]),
            round_index=int(data.get("round_index", 0)),
        )


__all__ = ["EnvironmentState", "PublicEnvironmentState", "PublicPlayerState"]

