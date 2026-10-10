"""Public-information role belief estimation for :class:`DummyAgent`.

The estimator intentionally operates on the canonical observation mapping (or
its mapping-compatible public state object).  It never accepts or inspects the
privileged environment role array.  Role indices are kept local to this module
so the agent layer does not depend on the environment implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from typing import Any, ClassVar

import numpy as np


PLAYER_COUNT = 5
ROLE_COUNT = 5
HIDDEN_ROLE_SENTINEL = 255
SHERIFF_ROLE = 0
DEPUTY_ROLE = 1
OUTLAW_A_ROLE = 2
OUTLAW_B_ROLE = 3
TRAITOR_ROLE = 4
STATE_SCHEMA_VERSION = "1.0"

DEFAULT_EVIDENCE_MULTIPLIERS = {
    "attack_sheriff_outlaw": 2.0,
    "crossfire_traitor": 1.5,
    "crossfire_target_outlaw": 1.25,
    "attack_outlaw_self_deputy": 1.5,
    "attack_outlaw_self_traitor": 1.5,
    "protect_sheriff_deputy": 1.5,
}
ROLE_NAMES = ("sheriff", "deputy", "outlaw_a", "outlaw_b", "traitor")

_PRIVILEGED_KEYS = frozenset(
    {
        "roles",
        "true_roles",
        "actual_roles",
        "privileged_state",
        "privileged_observation",
        "private_state",
        "private_observation",
    }
)


def _as_public_mapping(observation: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Return public fields and non-role metadata from a canonical observation.

    ``PublicEnvironmentState.to_observation_dict`` is used instead of
    ``to_dict`` because the latter is a persistence representation and may
    contain viewer-visible player objects.  Mapping callers are accepted for
    lightweight tests and non-environment integrations.
    """

    if observation is None:
        return {}, {}
    if isinstance(observation, Mapping):
        values: Mapping[str, Any] = observation
        metadata: dict[str, Any] = {}
    elif hasattr(observation, "to_observation_dict"):
        values = observation.to_observation_dict()
        metadata = {}
        for name in ("episode_id", "viewer_idx", "current_player_idx", "environment_step"):
            if hasattr(observation, name):
                metadata[name] = getattr(observation, name)
    elif hasattr(observation, "keys") and hasattr(observation, "__getitem__"):
        values = {key: observation[key] for key in observation.keys()}
        metadata = {}
    else:
        raise TypeError("observation must be a canonical public observation mapping")

    leaked = sorted(_PRIVILEGED_KEYS.intersection(values.keys()))
    if leaked:
        raise ValueError(f"privileged observation fields are not accepted: {', '.join(leaked)}")
    return values, metadata


def _scalar(value: Any, name: str) -> int | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    item = array.reshape(-1)[0].item()
    if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
        raise TypeError(f"{name} must contain an integer")
    return int(item)


def _player_index(value: Any, name: str) -> int:
    index = _scalar(value, name)
    if index is None or not 0 <= index < PLAYER_COUNT:
        raise ValueError(f"{name} must be an integer in [0, {PLAYER_COUNT - 1}]")
    return index


def _role(value: Any, name: str = "role") -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ROLE_NAMES:
            return ROLE_NAMES.index(normalized)
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a role index or role name") from exc
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer role index")
    normalized = int(value)
    if normalized == HIDDEN_ROLE_SENTINEL:
        return None
    if not 0 <= normalized < ROLE_COUNT:
        raise ValueError(f"{name} must be a role index in [0, {ROLE_COUNT - 1}]")
    return normalized


def _role_vector(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (ROLE_COUNT,):
        raise ValueError(f"role distribution must have shape ({ROLE_COUNT},)")
    try:
        vector = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("role distribution must be numeric") from exc
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("role distribution must contain finite non-negative values")
    return vector


def _event_mapping(event: Any) -> Mapping[str, Any] | None:
    if isinstance(event, Mapping):
        return event
    if hasattr(event, "to_public_dict"):
        # The public conversion is important: a private GameEvent must not
        # disclose its role through a raw object attribute.
        value = event.to_public_dict()
        return value if isinstance(value, Mapping) else None
    if hasattr(event, "to_dict"):
        value = event.to_dict(privileged=False)
        return value if isinstance(value, Mapping) else None
    return None


class RoleBeliefEstimator:
    """Estimate each slot's distribution over the five canonical roles.

    The estimator is deliberately non-probabilistic and deterministic.  It
    represents every known role as one-hot and distributes each still-available
    role count uniformly over unresolved slots.  Public role-reveal events are
    accumulated exactly once by event ID; repeating an event is harmless.
    """

    ROLE_COUNT: ClassVar[int] = ROLE_COUNT
    PLAYER_COUNT: ClassVar[int] = PLAYER_COUNT
    role_names: ClassVar[tuple[str, ...]] = ROLE_NAMES

    def __init__(
        self,
        player_idx: int | None = None,
        *,
        self_idx: int | None = None,
        evidence_multipliers: Mapping[str, float] | None = None,
    ) -> None:
        if player_idx is not None and self_idx is not None and int(player_idx) != int(self_idx):
            raise ValueError("player_idx and self_idx must identify the same slot")
        resolved = player_idx if player_idx is not None else self_idx
        self.player_idx: int | None = None if resolved is None else _player_index(resolved, "player_idx")
        self.episode_id: str | None = None
        self.last_event_id: int | None = None
        self.last_environment_step: int | None = None
        self._known_roles: dict[int, int] = {}
        self._processed_event_ids: set[int] = set()
        values = dict(DEFAULT_EVIDENCE_MULTIPLIERS)
        if evidence_multipliers is not None:
            incoming = dict(evidence_multipliers)
            # Intent v1 serialized this now-removed heuristic.  Ignore the
            # legacy key so old development snapshots can still be inspected,
            # but never apply the rule again.
            incoming.pop("attacked_twice_deputy", None)
            for key, value in incoming.items():
                if key not in values:
                    raise ValueError(f"unknown role-belief evidence multiplier: {key}")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError("role-belief evidence multipliers must be positive finite numbers")
                values[key] = float(value)
        self.evidence_multipliers = values
        self._evidence = np.ones((PLAYER_COUNT, ROLE_COUNT), dtype=np.float64)
        self._attack_received_counts = np.zeros(PLAYER_COUNT, dtype=np.int64)
        self._sheriff_attackers: set[int] = set()
        self._beliefs = np.full((PLAYER_COUNT, ROLE_COUNT), 1.0 / ROLE_COUNT, dtype=np.float32)

    @property
    def beliefs(self) -> np.ndarray:
        """Return a detached ``(5, 5)`` float32 belief matrix."""

        return self._beliefs.copy()

    @property
    def belief_vector(self) -> np.ndarray:
        """Alias for callers that use singular vector terminology."""

        return self.beliefs

    @property
    def role_distributions(self) -> np.ndarray:
        return self.beliefs

    @property
    def known_roles(self) -> dict[int, int]:
        return dict(self._known_roles)

    def reset(self, observation: Any | None = None) -> np.ndarray:
        """Reset all public evidence and optionally initialize from observation."""

        self.episode_id = None
        self.last_event_id = None
        self.last_environment_step = None
        self._known_roles.clear()
        self._processed_event_ids.clear()
        self._evidence = np.ones((PLAYER_COUNT, ROLE_COUNT), dtype=np.float64)
        self._attack_received_counts = np.zeros(PLAYER_COUNT, dtype=np.int64)
        self._sheriff_attackers.clear()
        self._beliefs = np.full((PLAYER_COUNT, ROLE_COUNT), 1.0 / ROLE_COUNT, dtype=np.float32)
        if observation is not None:
            return self.initialize(observation)
        return self.beliefs

    def initialize(self, observation: Any) -> np.ndarray:
        """Start a public evidence stream from ``observation``."""

        self.reset()
        return self.update(observation)

    def update(self, observation: Any | None = None, *, events: Iterable[Any] | None = None) -> np.ndarray:
        """Consume public observation and/or public events deterministically.

        Evidence is monotonic within an episode: a later partial observation
        cannot erase a role already confirmed by an earlier public event.  A
        changed ``episode_id`` starts a fresh stream when that metadata is
        available.
        """

        values: Mapping[str, Any] = {}
        metadata: dict[str, Any] = {}
        if observation is not None:
            values, metadata = _as_public_mapping(observation)
            episode_id = values.get("episode_id", metadata.get("episode_id"))
            if episode_id is not None:
                episode_id = str(episode_id)
                if self.episode_id is not None and episode_id != self.episode_id:
                    self.reset()
                self.episode_id = episode_id
            inferred_player = values.get("viewer_idx", metadata.get("viewer_idx"))
            if inferred_player is None and self.player_idx is None:
                inferred_player = values.get("self_idx", values.get("current_player_idx", metadata.get("current_player_idx")))
            if inferred_player is not None and self.player_idx is None:
                self.player_idx = _player_index(inferred_player, "player_idx")
            step_value = values.get("environment_step", metadata.get("environment_step"))
            if step_value is not None:
                self.last_environment_step = _scalar(step_value, "environment_step")

            self._consume_observation_constraints(values)
            observation_events = values.get("event_sequence", values.get("event_history", ()))
            if observation_events is not None:
                self._consume_events(observation_events)
        if events is not None:
            self._consume_events(events)
        self._recompute_beliefs()
        return self.beliefs

    update_from_observation = update
    update_public_observation = update

    def update_from_events(self, events: Iterable[Any]) -> np.ndarray:
        """Apply only public events, preserving previously known constraints."""

        return self.update(events=events)

    update_from_public_events = update_from_events
    observe = update
    estimate = update

    def _consume_observation_constraints(self, values: Mapping[str, Any]) -> None:
        self_idx = self.player_idx
        if self_idx is None:
            candidate = values.get("self_idx", values.get("viewer_idx", values.get("current_player_idx")))
            if candidate is not None:
                self_idx = _player_index(candidate, "self_idx")
                self.player_idx = self_idx
        if self_idx is not None and "self_role" in values:
            self._confirm(self_idx, _role(_scalar(values["self_role"], "self_role"), "self_role"))

        if "sheriff_idx" in values:
            sheriff_idx = _player_index(values["sheriff_idx"], "sheriff_idx")
            self._confirm(sheriff_idx, SHERIFF_ROLE)

        if "revealed_roles" in values:
            revealed = np.asarray(values["revealed_roles"]).reshape(-1)
            if revealed.size != PLAYER_COUNT:
                raise ValueError("revealed_roles must have five entries")
            for index, value in enumerate(revealed):
                role = _role(value.item() if hasattr(value, "item") else value, f"revealed_roles[{index}]")
                if role is not None:
                    self._confirm(index, role)

        if "confirmed_role_distribution" in values:
            matrix = np.asarray(values["confirmed_role_distribution"])
            if matrix.shape != (PLAYER_COUNT, ROLE_COUNT):
                raise ValueError("confirmed_role_distribution must have shape (5, 5)")
            for index, row in enumerate(matrix):
                vector = _role_vector(row)
                if np.isclose(float(vector.sum()), 0.0):
                    continue
                if np.isclose(float(vector.sum()), 1.0) and np.count_nonzero(~np.isclose(vector, 0.0)) == 1:
                    self._confirm(index, int(np.argmax(vector)))

    def _consume_events(self, events: Iterable[Any]) -> None:
        for event in events:
            value = _event_mapping(event)
            if value is None:
                continue
            public = value.get("public", True)
            if public is False:
                continue
            event_id = value.get("event_id")
            if event_id is not None:
                event_id = _scalar(event_id, "event_id")
                if event_id is not None:
                    if event_id in self._processed_event_ids:
                        continue
                    self._processed_event_ids.add(event_id)
                    self.last_event_id = event_id if self.last_event_id is None else max(self.last_event_id, event_id)

            event_role = value.get("revealed_role")
            role = _role(event_role, "revealed_role") if event_role is not None else None
            if role is not None:
                slot = value.get("eliminated_idx", value.get("target_idx"))
                if slot is not None:
                    self._confirm(_player_index(slot, "event player index"), role)

            event_type = str(getattr(value.get("event_type", ""), "value", value.get("event_type", ""))).rsplit(".", 1)[-1].lower()
            if event_type != "bang":
                continue
            actor = value.get("actor_idx")
            target = value.get("target_idx")
            if actor is None or target is None:
                continue
            self._apply_bang_evidence(
                _player_index(actor, "actor_idx"),
                _player_index(target, "target_idx"),
            )

    def _boost(self, player_idx: int, roles: Iterable[int], multiplier_key: str) -> None:
        if player_idx in self._known_roles:
            return
        factor = float(self.evidence_multipliers[multiplier_key])
        for role in roles:
            self._evidence[player_idx, int(role)] *= factor

    def _apply_bang_evidence(self, actor: int, target: int) -> None:
        """Update soft role evidence from one PUBLIC Bang event.

        The rules deliberately stop at role-belief updates.  They never add an
        attack score here; that belongs to IntentModule.
        """

        self._attack_received_counts[target] += 1
        sheriff_idx = next((idx for idx, role in self._known_roles.items() if role == SHERIFF_ROLE), None)
        if sheriff_idx is None:
            return
        if target == sheriff_idx:
            self._sheriff_attackers.add(actor)

        self_role = self._known_roles.get(self.player_idx) if self.player_idx is not None else None
        if self_role is None:
            return

        # Common strong evidence: intentionally attacking the public Sheriff is
        # most compatible with an Outlaw objective.
        if target == sheriff_idx and actor != sheriff_idx:
            self._boost(actor, (OUTLAW_A_ROLE, OUTLAW_B_ROLE), "attack_sheriff_outlaw")

        if self_role == SHERIFF_ROLE:
            # Being attacked repeatedly is NOT role evidence anymore.  The
            # Sheriff handles that public situation as an Intent preference
            # (temporary target de-prioritisation) instead of changing the
            # target's Deputy probability.
            pass

        elif self_role == DEPUTY_ROLE:
            # From the Deputy's perspective the three hidden players are
            # Outlaw, Outlaw, Traitor.  If one hidden player shoots another,
            # the actor looks more Traitor-like (an Outlaw should normally
            # pressure the public Sheriff), while the target becomes modestly
            # more Outlaw-like.
            if actor not in {self.player_idx, sheriff_idx} and target not in {self.player_idx, sheriff_idx}:
                self._boost(actor, (TRAITOR_ROLE,), "crossfire_traitor")
                self._boost(target, (OUTLAW_A_ROLE, OUTLAW_B_ROLE), "crossfire_target_outlaw")

        elif self_role in (OUTLAW_A_ROLE, OUTLAW_B_ROLE):
            # An unknown non-Sheriff who attacks the Outlaw observer is more
            # compatible with Sheriff-team protection or Traitor survival play.
            if target == self.player_idx and actor != sheriff_idx:
                self._boost(actor, (DEPUTY_ROLE,), "attack_outlaw_self_deputy")
                self._boost(actor, (TRAITOR_ROLE,), "attack_outlaw_self_traitor")

        elif self_role == TRAITOR_ROLE:
            # The user had not yet specified a Traitor-specific belief rule.
            # As a provisional public-only rule, a player who attacks someone
            # already observed attacking the Sheriff looks more Deputy-like.
            if target in self._sheriff_attackers and actor not in {self.player_idx, sheriff_idx}:
                self._boost(actor, (DEPUTY_ROLE,), "protect_sheriff_deputy")

    def _confirm(self, player_idx: int, role: int | None) -> None:
        if role is None:
            return
        previous = self._known_roles.get(player_idx)
        if previous is not None and previous != role:
            raise ValueError(f"conflicting public roles for player {player_idx}")
        for other_idx, other_role in self._known_roles.items():
            if other_idx != player_idx and other_role == role:
                raise ValueError(f"role {role} is confirmed for multiple players")
        self._known_roles[player_idx] = role

    def _recompute_beliefs(self) -> None:
        beliefs = np.zeros((PLAYER_COUNT, ROLE_COUNT), dtype=np.float64)
        for player_idx, role in self._known_roles.items():
            beliefs[player_idx, role] = 1.0

        unresolved = [index for index in range(PLAYER_COUNT) if index not in self._known_roles]
        remaining = np.ones(ROLE_COUNT, dtype=np.float64)
        for role in self._known_roles.values():
            remaining[role] -= 1.0
        if np.any(remaining < -1e-9) or len(unresolved) != int(round(float(remaining.sum()))):
            raise ValueError("public role constraints cannot form the five-role assignment")
        if unresolved:
            # Start from soft public evidence, then iteratively scale rows and
            # columns.  This keeps each player's row normalized while also
            # respecting the exact remaining role counts (one Sheriff, one
            # Deputy, two distinct Outlaw slots, one Traitor globally).
            matrix = np.maximum(self._evidence[unresolved, :].copy(), 1e-12)
            matrix[:, remaining <= 0.0] = 0.0
            for _ in range(128):
                for role in range(ROLE_COUNT):
                    if remaining[role] <= 0.0:
                        matrix[:, role] = 0.0
                        continue
                    total = float(matrix[:, role].sum())
                    if total <= 0.0:
                        matrix[:, role] = remaining[role] / float(len(unresolved))
                    else:
                        matrix[:, role] *= remaining[role] / total
                row_sums = matrix.sum(axis=1, keepdims=True)
                if np.any(row_sums <= 0.0):
                    raise ValueError("public role evidence produced an empty role row")
                matrix /= row_sums
                column_error = float(np.max(np.abs(matrix.sum(axis=0) - remaining)))
                if column_error < 1e-10:
                    break
            # One final row normalization protects callers from tiny floating
            # error while preserving the column-count constraint closely.
            matrix /= matrix.sum(axis=1, keepdims=True)
            beliefs[unresolved, :] = matrix
        self._beliefs = beliefs.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-safe estimator state without privileged role data."""

        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "player_idx": self.player_idx,
            "episode_id": self.episode_id,
            "last_event_id": self.last_event_id,
            "last_environment_step": self.last_environment_step,
            "known_roles": {str(index): int(role) for index, role in sorted(self._known_roles.items())},
            "processed_event_ids": sorted(self._processed_event_ids),
            "evidence_multipliers": dict(sorted(self.evidence_multipliers.items())),
            "evidence": self._evidence.astype(float).tolist(),
            "attack_received_counts": self._attack_received_counts.astype(int).tolist(),
            "sheriff_attackers": sorted(self._sheriff_attackers),
            "beliefs": self._beliefs.astype(float).tolist(),
        }

    to_dict = state_dict
    serialize = state_dict

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RoleBeliefEstimator":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("schema_version", STATE_SCHEMA_VERSION) != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported RoleBeliefEstimator state schema")
        estimator = cls(
            player_idx=state.get("player_idx"),
            evidence_multipliers=state.get("evidence_multipliers"),
        )
        estimator.episode_id = state.get("episode_id")
        estimator.last_event_id = state.get("last_event_id")
        estimator.last_environment_step = state.get("last_environment_step")
        known = state.get("known_roles", {})
        if not isinstance(known, Mapping):
            raise TypeError("known_roles must be a mapping")
        for index, role in known.items():
            estimator._confirm(_player_index(int(index), "known role player index"), _role(role, "known role"))
        estimator._processed_event_ids = {int(event_id) for event_id in state.get("processed_event_ids", [])}
        evidence = np.asarray(state.get("evidence", estimator._evidence), dtype=np.float64)
        if evidence.shape != (PLAYER_COUNT, ROLE_COUNT) or not np.all(np.isfinite(evidence)) or np.any(evidence <= 0.0):
            raise ValueError("serialized role evidence must be a positive finite (5, 5) matrix")
        estimator._evidence = evidence
        counts = np.asarray(state.get("attack_received_counts", estimator._attack_received_counts), dtype=np.int64)
        if counts.shape != (PLAYER_COUNT,) or np.any(counts < 0):
            raise ValueError("serialized attack_received_counts must contain five non-negative values")
        estimator._attack_received_counts = counts
        estimator._sheriff_attackers = {int(index) for index in state.get("sheriff_attackers", [])}
        if any(index < 0 or index >= PLAYER_COUNT for index in estimator._sheriff_attackers):
            raise ValueError("serialized sheriff_attackers contains an invalid player index")
        estimator._recompute_beliefs()
        expected = np.asarray(state.get("beliefs", estimator._beliefs), dtype=np.float32)
        if expected.shape != (PLAYER_COUNT, ROLE_COUNT) or not np.allclose(expected, estimator._beliefs, atol=1e-6):
            raise ValueError("serialized beliefs do not match serialized public constraints")
        return estimator

    from_dict = from_state_dict
    deserialize = from_state_dict

    def to_json(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "RoleBeliefEstimator":
        if not isinstance(value, str):
            raise TypeError("serialized state must be a JSON string")
        return cls.from_state_dict(json.loads(value))


__all__ = [
    "HIDDEN_ROLE_SENTINEL",
    "PLAYER_COUNT",
    "ROLE_COUNT",
    "ROLE_NAMES",
    "RoleBeliefEstimator",
    "SHERIFF_ROLE",
    "STATE_SCHEMA_VERSION",
]
