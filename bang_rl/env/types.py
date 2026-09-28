"""Core domain types for the Bang RL environment.

This module holds the pure, immutable domain vocabulary: role/team/event enums,
the ``Action``/``GameEvent``/``ActionResult``/``TerminalResult`` dataclasses,
canonical constants, and deterministic role helpers.  Nothing here imports Qt,
torch, or the environment runtime, so agents/data/training can depend on it
cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
PLAYER_COUNT = 5
CANONICAL_PLAYER_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4)
ACTION_SPACE_SIZE = 6
PASS_ACTION = 0
HIDDEN_ROLE_SENTINEL = 255
PASS_TARGET_SENTINEL = 255
STATE_VECTOR_SIZE = 9
ACTION_SPACE_VERSION = "discrete-6-v1"
OBSERVATION_SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
class Team(IntEnum):
    SHERIFF_TEAM = 0
    OUTLAW_TEAM = 1
    TRAITOR_TEAM = 2


class Role(IntEnum):
    SHERIFF = 0
    DEPUTY = 1
    OUTLAW_A = 2
    OUTLAW_B = 3
    TRAITOR = 4

    @property
    def team(self) -> Team:
        return _ROLE_TEAM[self]


_ROLE_TEAM = {
    Role.SHERIFF: Team.SHERIFF_TEAM,
    Role.DEPUTY: Team.SHERIFF_TEAM,
    Role.OUTLAW_A: Team.OUTLAW_TEAM,
    Role.OUTLAW_B: Team.OUTLAW_TEAM,
    Role.TRAITOR: Team.TRAITOR_TEAM,
}


class ActionType(IntEnum):
    PASS = 0
    BANG = 1


class EventType(str, Enum):
    PASS = "pass"
    BANG = "bang"
    PLAYER_ELIMINATED = "player_eliminated"
    ROLE_REVEALED = "role_revealed"


class RoleVisibility(IntEnum):
    HIDDEN = 0
    SELF = 1
    REVEALED = 2
    CONFIRMED = 3


class TerminationStatus(str, Enum):
    RUNNING = "running"
    NORMAL = "normal"
    TRUNCATED = "truncated"


# --------------------------------------------------------------------------
# Small coercion helpers
# --------------------------------------------------------------------------
def _as_role(value: Any) -> Role:
    if isinstance(value, Role):
        return value
    return Role(int(value))


def _as_team(value: Any) -> Team | None:
    if value is None:
        return None
    if isinstance(value, Team):
        return value
    if isinstance(value, str):
        return Team[value.upper()] if value.upper() in Team.__members__ else Team(int(value))
    return Team(int(value))


def _as_termination_status(value: Any) -> TerminationStatus:
    if isinstance(value, TerminationStatus):
        return value
    return TerminationStatus(str(value))


def _as_event_type(value: Any) -> EventType:
    if isinstance(value, EventType):
        return value
    return EventType(str(value))


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


# --------------------------------------------------------------------------
# Action
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    action_type: ActionType
    actor_idx: int
    target_idx: int | None = None

    @staticmethod
    def bang(actor_idx: int, target_idx: int) -> "Action":
        return Action(ActionType.BANG, int(actor_idx), int(target_idx))

    @staticmethod
    def pass_action(actor_idx: int) -> "Action":
        return Action(ActionType.PASS, int(actor_idx), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "actor_idx": int(self.actor_idx),
            "target_idx": None if self.target_idx is None else int(self.target_idx),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Action":
        raw = data.get("action_type", data.get("type"))
        if isinstance(raw, ActionType):
            action_type = raw
        else:
            token = str(getattr(raw, "value", raw)).rsplit(".", 1)[-1].lower()
            action_type = ActionType.PASS if token in {"0", "pass"} else ActionType.BANG
        return cls(action_type, int(data["actor_idx"]), _opt_int(data.get("target_idx")))


# --------------------------------------------------------------------------
# GameEvent
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GameEvent:
    event_id: int
    environment_step: int
    event_type: EventType
    actor_idx: int | None = None
    target_idx: int | None = None
    hp_before: int | None = None
    hp_after: int | None = None
    eliminated_idx: int | None = None
    revealed_role: Role | None = None
    public: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", _as_event_type(self.event_type))
        if self.revealed_role is not None:
            object.__setattr__(self, "revealed_role", _as_role(self.revealed_role))

    def to_dict(self) -> dict[str, Any]:
        """Privileged serialization; retains revealed_role even when private."""
        return {
            "event_id": int(self.event_id),
            "environment_step": int(self.environment_step),
            "event_type": self.event_type.value,
            "actor_idx": _opt_int(self.actor_idx),
            "target_idx": _opt_int(self.target_idx),
            "hp_before": _opt_int(self.hp_before),
            "hp_after": _opt_int(self.hp_after),
            "eliminated_idx": _opt_int(self.eliminated_idx),
            "revealed_role": None if self.revealed_role is None else int(self.revealed_role),
            "public": bool(self.public),
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Public projection; a private event never leaks its revealed role."""
        payload = self.to_dict()
        if not self.public:
            payload["revealed_role"] = None
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GameEvent":
        revealed = data.get("revealed_role")
        return cls(
            event_id=int(data["event_id"]),
            environment_step=int(data["environment_step"]),
            event_type=_as_event_type(data["event_type"]),
            actor_idx=_opt_int(data.get("actor_idx")),
            target_idx=_opt_int(data.get("target_idx")),
            hp_before=_opt_int(data.get("hp_before")),
            hp_after=_opt_int(data.get("hp_after")),
            eliminated_idx=_opt_int(data.get("eliminated_idx")),
            revealed_role=None if revealed is None else _as_role(revealed),
            public=bool(data.get("public", True)),
        )


# --------------------------------------------------------------------------
# TerminalResult
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TerminalResult:
    terminated: bool
    winners: tuple[int, ...]
    losers: tuple[int, ...]
    winning_team: Team | None
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "winners", tuple(int(x) for x in self.winners))
        object.__setattr__(self, "losers", tuple(int(x) for x in self.losers))
        object.__setattr__(self, "winning_team", _as_team(self.winning_team))
        if not self.terminated and not self.winners and not self.losers and self.winning_team is None:
            raise ValueError("a non-terminal result is not a valid TerminalResult")

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminated": bool(self.terminated),
            "winners": [int(x) for x in self.winners],
            "losers": [int(x) for x in self.losers],
            "winning_team": None if self.winning_team is None else int(self.winning_team),
            "reason": str(self.reason),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TerminalResult":
        return cls(
            terminated=bool(data["terminated"]),
            winners=tuple(int(x) for x in data.get("winners", ())),
            losers=tuple(int(x) for x in data.get("losers", ())),
            winning_team=_as_team(data.get("winning_team")),
            reason=str(data.get("reason", "")),
        )


# --------------------------------------------------------------------------
# ActionResult
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionResult:
    action: Action
    environment_step: int
    events: tuple[GameEvent, ...] = ()
    hp_before: int | None = None
    hp_after: int | None = None
    eliminated_idx: int | None = None
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)
    observation: Any = None
    reward: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "environment_step": int(self.environment_step),
            "events": [event.to_dict() for event in self.events],
            "hp_before": _opt_int(self.hp_before),
            "hp_after": _opt_int(self.hp_after),
            "eliminated_idx": _opt_int(self.eliminated_idx),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "info": dict(self.info),
            "reward": float(self.reward),
        }


# --------------------------------------------------------------------------
# Role helpers
# --------------------------------------------------------------------------
def canonical_player_order() -> tuple[int, ...]:
    return CANONICAL_PLAYER_ORDER


def canonical_agent_order() -> tuple[str, ...]:
    return tuple(f"player_{index}" for index in CANONICAL_PLAYER_ORDER)


def assign_roles(seed: int) -> tuple[Role, ...]:
    """Deterministic role permutation seeded by ``seed``."""
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    permutation = generator.permutation(
        np.asarray(tuple(role.value for role in Role), dtype=np.uint8)
    )
    return tuple(Role(int(value)) for value in permutation)


def initial_health(roles: Sequence[Any]) -> tuple[int, ...]:
    """Sheriff starts with 5 HP; every other seat starts with 4."""
    return tuple(5 if _as_role(role) is Role.SHERIFF else 4 for role in roles)


__all__ = [
    "ACTION_SPACE_SIZE",
    "ACTION_SPACE_VERSION",
    "Action",
    "ActionResult",
    "ActionType",
    "CANONICAL_PLAYER_ORDER",
    "EventType",
    "GameEvent",
    "HIDDEN_ROLE_SENTINEL",
    "OBSERVATION_SCHEMA_VERSION",
    "PASS_ACTION",
    "PASS_TARGET_SENTINEL",
    "PLAYER_COUNT",
    "Role",
    "RoleVisibility",
    "STATE_VECTOR_SIZE",
    "Team",
    "TerminalResult",
    "TerminationStatus",
    "assign_roles",
    "canonical_agent_order",
    "canonical_player_order",
    "initial_health",
]
