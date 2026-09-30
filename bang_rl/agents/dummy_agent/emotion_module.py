"""Deterministic, public-input-only emotion state for ``DummyAgent``.

The module is deliberately independent from ``bang_rl.env``.  The execution
layer feeds it public events and valid actions; the environment never reaches
into this private state.  Emotion effects are exposed for the later action
selector task, but are not applied to intent selection here.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any, ClassVar

import numpy as np

from bang_rl.config import EmotionConfig


EMOTION_MODULE_STATE_VERSION = "1.1"

# Number of seats in a game.  The per-opponent relation vector has one slot per
# seat; the slot whose index equals the module's own ``player_idx`` stays at the
# neutral baseline and is never mutated.
PLAYER_COUNT = 5

# Documented clamping range for the signed per-opponent relation vector.  Unlike
# the flat emotion values (which are normalized to [0, 1]), relation is SIGNED so
# a single scalar can represent both a grudge (negative) and trust (positive)
# toward another seat.  The relation vector is kept in a SEPARATE array so the
# emotion [0, 1] clamping is untouched.
RELATION_MIN = -1.0
RELATION_MAX = 1.0

# Version-compatibility policy for ``from_state_dict``:
#   * "1.0" states predate the relation vector.  They are ACCEPTED and upgraded
#     in memory by defaulting the relation vector (and its baseline) to neutral.
#   * "1.1" states carry ``relation`` / ``relation_baseline`` and round-trip.
# ``state_dict`` always writes the current schema version ("1.1").
_SUPPORTED_STATE_VERSIONS = frozenset({"1.0", "1.1"})

# These names are rejected at the agent boundary.  Public role-reveal fields
# such as ``revealed_role`` are intentionally not included in this list.
_PRIVATE_FIELDS = {
    "roles",
    "true_roles",
    "actual_roles",
    "hidden_roles",
    "privileged_state",
    "privileged_observation",
    "private_observation",
    "emotion_label",
    "reward_breakdown",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Convert a public event/action-like value without importing the env."""

    if isinstance(value, Mapping):
        result = value
    elif hasattr(value, "to_public_dict"):
        result = value.to_public_dict()
    elif hasattr(value, "to_dict"):
        result = value.to_dict()
    else:
        result = {
            name: getattr(value, name)
            for name in ("event_id", "event_type", "public", "action", "action_type", "valid")
            if hasattr(value, name)
        }
    if not isinstance(result, Mapping):
        raise TypeError(f"{label} must be a public mapping or public event/action value")
    leaked = sorted(_PRIVATE_FIELDS.intersection(result.keys()))
    if leaked:
        raise ValueError(f"privileged fields are not accepted: {', '.join(leaked)}")
    return result


def _text(value: Any) -> str:
    """Return the stable wire spelling for strings and enum-like values."""

    if hasattr(value, "value"):
        value = value.value
    return str(value).rsplit(".", 1)[-1].strip().lower()


def _seat_index(value: Any) -> int | None:
    """Return a genuine integer seat index, or ``None`` when unknown.

    Booleans are rejected (mirroring the ``isinstance`` guards used elsewhere in
    this module) and any non-integer / ``None`` value is treated as unknown.
    The helper intentionally avoids importing numpy so it can be reused by pure,
    dependency-free unit tests.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None


def _relationship_event_keys(
    event_type: str,
    actor_idx: Any,
    target_idx: Any,
    player_idx: Any,
) -> tuple[str, ...]:
    """Derive relationship-qualified event keys from public actor/target seats.

    The keys are ordered most-specific-first and are meant to be tried BEFORE
    the plain ``event_type`` key.  They are built solely from the PUBLIC
    ``actor_idx`` / ``target_idx`` relative to the module's own ``player_idx``;
    nothing private is consulted.

    Vocabulary (prefix is the already-lowercased ``event_type`` wire spelling):
    * actor-relative (``_by_*``): ``{event_type}_by_me`` when the actor is our
      own seat, ``{event_type}_by_other`` when the actor is a known other seat.
    * target-relative (``_at_*``): ``{event_type}_at_me`` when the target is our
      own seat, ``{event_type}_at_other`` when the target is a known other seat.

    The actor-relative (``_by_*``) key is emitted first so it takes precedence
    over the target-relative (``_at_*``) key when both are present.  A side
    whose index is ``None`` / unknown is omitted entirely.  When ``player_idx``
    is ``None`` an empty tuple is returned, so relationship keys never fire and
    the pre-existing plain-key behaviour is preserved exactly.

    Concretely this yields, for example: ``bang_by_me``, ``bang_at_me``,
    ``bang_by_other``, ``bang_at_other``; ``player_eliminated_by_me``,
    ``player_eliminated_at_me`` (i.e. "I was eliminated"),
    ``player_eliminated_by_other``, ``player_eliminated_at_other``;
    ``role_revealed_at_me`` and ``role_revealed_at_other`` (``role_revealed``
    has no actor so no ``_by_*`` keys are produced).
    """

    seat = _seat_index(player_idx)
    if seat is None:
        return ()
    keys: list[str] = []
    actor = _seat_index(actor_idx)
    if actor is not None:
        if actor == seat:
            keys.append(f"{event_type}_by_me")
        else:
            keys.append(f"{event_type}_by_other")
    target = _seat_index(target_idx)
    if target is not None:
        if target == seat:
            keys.append(f"{event_type}_at_me")
        else:
            keys.append(f"{event_type}_at_other")
    return tuple(keys)


def _relation_event_seats(
    event_type: str,
    actor_idx: Any,
    target_idx: Any,
    player_idx: Any,
) -> tuple[tuple[int, str], ...]:
    """Map a public event to the OTHER seat(s) it implicates and a relation kind.

    This is the per-opponent counterpart of :func:`_relationship_event_keys`.
    Where the relationship keys collapse "some other seat" into a single
    ``_by_other`` / ``_at_other`` bucket, this helper preserves WHICH other seat
    is involved so a per-seat relation vector can be updated.

    Returns a tuple of ``(seat, kind)`` pairs where ``seat`` is a genuine other
    seat index (never the module's own ``player_idx``) and ``kind`` is a stable,
    lowercase relationship kind that is looked up in the configured
    ``relation_updates`` map.  The kinds are built solely from the PUBLIC
    ``actor_idx`` / ``target_idx`` relative to ``player_idx``; nothing private is
    consulted.

    Vocabulary (``{event_type}`` is the already-lowercased wire spelling, with
    ``player_eliminated`` shortened to ``eliminated`` for readability of the
    common case, and other event types using their own spelling):
    * ``{kind}_by`` when a DIFFERENT seat is the actor and WE are the target
      (for example ``shot_by`` when seat s banged us, ``eliminated_by`` when
      seat s eliminated us).  Direction: increase grudge toward seat s.
    * ``{kind}_at`` when WE are the actor and a DIFFERENT seat is the target
      (for example ``shot_at`` when we banged seat s).  Direction is left to the
      configuration (a positive delta reads as satisfaction/dominance, a
      negative one as remorse).
    * ``{kind}_by_other`` / ``{kind}_at_other`` when a DIFFERENT seat acts on
      ANOTHER different seat (neither is us); this lets the config react to
      third-party interactions.  Only the acting/affected OTHER seat is
      implicated, never our own.

    ``bang`` maps to the ``shot`` kind and ``player_eliminated`` to the
    ``eliminated`` kind; every other event type uses its own spelling.  When our
    own seat is unknown (``player_idx`` is ``None``) an empty tuple is returned,
    so relation is never mutated without a known seat.  A self-directed event
    (actor == target == us) implicates no other seat.  The helper avoids
    importing numpy so it can be reused by pure, dependency-free unit tests.
    """

    seat = _seat_index(player_idx)
    if seat is None:
        return ()
    kind_aliases = {
        "bang": "shot",
        "player_eliminated": "eliminated",
    }
    kind = kind_aliases.get(event_type, event_type)
    actor = _seat_index(actor_idx)
    target = _seat_index(target_idx)
    result: list[tuple[int, str]] = []
    if actor is not None and actor != seat and target is not None and target == seat:
        # A different seat acted on us -> "{kind}_by" that seat.
        result.append((actor, f"{kind}_by"))
    elif actor is not None and actor == seat and target is not None and target != seat:
        # We acted on a different seat -> "{kind}_at" that seat.
        result.append((target, f"{kind}_at"))
    else:
        # Third-party interactions: neither the actor nor the target is us.
        if actor is not None and actor != seat and (target is None or target != seat):
            result.append((actor, f"{kind}_by_other"))
        if target is not None and target != seat and (actor is None or actor != seat):
            result.append((target, f"{kind}_at_other"))
    return tuple(result)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


class EmotionModule:
    """Manage N configurable normalized emotion values deterministically.

    ``event_id`` is the idempotency key for public events.  ``advance_step``
    applies configured recovery/decay at most once for a supplied environment
    step.  Decay is a linear amount toward the configured initial value; all
    mutations are clamped to ``[0, 1]``.
    """

    schema_version: ClassVar[str] = EMOTION_MODULE_STATE_VERSION

    def __init__(
        self,
        config: EmotionConfig | Mapping[str, Any] | Any | None = None,
        *,
        player_idx: int | None = None,
    ) -> None:
        self.config = self._coerce_config(config)
        if player_idx is not None:
            if isinstance(player_idx, (bool, np.bool_)) or not 0 <= int(player_idx) < 5:
                raise ValueError("player_idx must be in [0, 4] or None")
            player_idx = int(player_idx)
        self.player_idx = player_idx
        self._definitions = tuple(self.config.emotions)
        self._names = tuple(definition.name for definition in self._definitions)
        self._baselines = np.asarray(
            [
                self.config.initial_baseline
                if definition.initial_value is None
                else definition.initial_value
                for definition in self._definitions
            ],
            dtype=np.float32,
        )
        self._values = self._baselines.copy()
        # Per-opponent relation vector (one signed scalar per seat).  It is held
        # in a SEPARATE float32 array clamped to [RELATION_MIN, RELATION_MAX] so
        # the flat emotion [0, 1] clamping is untouched.  The self slot stays at
        # the neutral baseline and is never mutated.
        self._relation_baseline = float(self.config.relation_baseline)
        self._relation = np.full(PLAYER_COUNT, self._relation_baseline, dtype=np.float32)
        self.episode_id: str | None = None
        self.episode_seed: int | None = None
        self.last_event_id: int | None = None
        self.last_environment_step: int | None = None
        self.terminated = False
        self.truncated = False
        self._processed_event_ids: set[int] = set()

    @staticmethod
    def _coerce_config(config: EmotionConfig | Mapping[str, Any] | Any | None) -> EmotionConfig:
        if config is None:
            return EmotionConfig()
        if isinstance(config, EmotionConfig):
            return config
        if hasattr(config, "model_dump"):
            config = config.model_dump(mode="python")
        if isinstance(config, Mapping):
            return EmotionConfig.model_validate(dict(config))
        raise TypeError("config must be EmotionConfig, a mapping, or None")

    def update_config(self, config: EmotionConfig | Mapping[str, Any] | Any) -> None:
        """Apply a preset at a decision boundary without changing current state."""

        updated = self._coerce_config(config)
        old_names = self.emotion_names
        new_names = tuple(item.name for item in updated.emotions)
        if old_names != new_names and updated.schema_version == self.config.schema_version:
            raise ValueError("emotion schema changes require a new schema_version")
        old_values = dict(zip(old_names, self._values))
        self.config = updated
        self._definitions = tuple(updated.emotions)
        self._names = new_names
        self._baselines = np.asarray(
            [
                updated.initial_baseline
                if definition.initial_value is None
                else definition.initial_value
                for definition in self._definitions
            ],
            dtype=np.float32,
        )
        self._values = np.asarray(
            [old_values.get(name, baseline) for name, baseline in zip(self._names, self._baselines)],
            dtype=np.float32,
        )
        self._values = np.clip(self._values, 0.0, 1.0).astype(np.float32)

    @property
    def emotion_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def emotion_count(self) -> int:
        return len(self._names)

    @property
    def values(self) -> np.ndarray:
        """Return a detached float32 vector in schema order."""

        return self._values.copy()

    @property
    def state(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self._names, self._values)}

    @property
    def baseline(self) -> np.ndarray:
        return self._baselines.copy()

    @property
    def relation(self) -> np.ndarray:
        """Return a detached copy of the per-seat relation vector.

        The array has length ``PLAYER_COUNT``; index ``s`` is this agent's signed
        relation toward seat ``s`` (negative is grudge, positive is trust) clamped
        to ``[RELATION_MIN, RELATION_MAX]``.  The self slot (index ``player_idx``)
        stays at the neutral baseline.  Callers such as the action selector / GUI
        read this without mutating internal state; wiring relation into action
        selection is intentionally deferred to a later task.
        """

        return self._relation.copy()

    @property
    def relation_baseline(self) -> float:
        """Neutral baseline value for the relation vector."""

        return self._relation_baseline

    @property
    def processed_event_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._processed_event_ids))

    @property
    def effects(self) -> dict[str, dict[str, float]]:
        return {
            definition.name: dict(sorted(definition.effects.items()))
            for definition in self._definitions
        }

    def effects_for_action(self, action: Any) -> dict[str, float]:
        """Return configured action effects without mutating state.

        Action-selection integration is intentionally deferred to TASK-2-5.
        """

        key, action_type = self._action_keys(action)
        result: dict[str, float] = {}
        for definition in self._definitions:
            for candidate in (key, action_type, "action", "*"):
                if candidate is not None and candidate in definition.effects:
                    result[definition.name] = float(definition.effects[candidate])
                    break
        return result

    def reset(self, context: Any | None = None) -> np.ndarray:
        """Reset values, event deduplication, step clock, and lifecycle flags."""

        self._values = self._baselines.copy()
        self._relation = np.full(PLAYER_COUNT, self._relation_baseline, dtype=np.float32)
        self.last_event_id = None
        self.last_environment_step = None
        self._processed_event_ids.clear()
        self.terminated = False
        self.truncated = False
        self.episode_id = None
        self.episode_seed = None
        if context is not None:
            if isinstance(context, Mapping):
                self.episode_id = context.get("episode_id")
                seed = context.get("seed", context.get("episode_seed"))
            else:
                self.episode_id = getattr(context, "episode_id", None)
                seed = getattr(context, "seed", getattr(context, "episode_seed", None))
            if self.episode_id is not None:
                self.episode_id = str(self.episode_id)
            if seed is not None:
                self.episode_seed = int(seed)
        return self.values

    def on_episode_start(self, context: Any | None = None) -> np.ndarray:
        return self.reset(context)

    def _change(self, deltas: list[float]) -> np.ndarray:
        self._values = np.clip(
            self._values + np.asarray(deltas, dtype=np.float32), 0.0, 1.0
        ).astype(np.float32)
        return self.values

    @staticmethod
    def _rule_value(rules: Mapping[str, float], keys: tuple[str | None, ...]) -> float | None:
        for key in keys:
            if key is not None and key in rules:
                return float(rules[key])
        return None

    def _event_value(
        self,
        definition: Any,
        event_type: str,
        actor_idx: Any = None,
        target_idx: Any = None,
    ) -> float:
        aliases = {
            "elimination": "player_eliminated",
            "role_reveal": "role_revealed",
        }
        # Relationship keys are built from the actual (already-lowercased)
        # event_type wire spelling, not the alias.
        normalized = aliases.get(event_type, event_type)
        # Precedence (first match wins): most-specific relationship key(s) ->
        # plain event_type -> normalized alias -> "*" -> change_rate.  Among the
        # relationship keys the actor-relative (_by_*) key is tried before the
        # target-relative (_at_*) key.  When self.player_idx is None the helper
        # returns (), so relationship keys are skipped entirely.
        relationship_keys = _relationship_event_keys(
            event_type, actor_idx, target_idx, self.player_idx
        )
        value = self._rule_value(
            definition.event_updates,
            relationship_keys + (event_type, normalized, "*"),
        )
        return definition.change_rate if value is None else value

    def apply_event(self, event: Any) -> np.ndarray:
        """Apply one public event once; duplicate IDs are a no-op."""

        values = _mapping(event, label="event")
        if values.get("public", True) is False:
            return self.values
        event_id = values.get("event_id")
        if isinstance(event_id, (bool, np.bool_)) or not isinstance(event_id, (int, np.integer)):
            raise ValueError("public emotion events require an integer event_id")
        event_id = int(event_id)
        if event_id < 0:
            raise ValueError("event_id must be non-negative")
        if event_id in self._processed_event_ids:
            return self.values
        event_type = _text(values.get("event_type", values.get("type", values.get("event", ""))))
        if not event_type:
            raise ValueError("public emotion events require an event_type")
        # actor_idx/target_idx come from the already-validated public mapping;
        # no _PRIVATE_FIELDS are read here.  The same public indices are passed
        # to every definition.
        actor_idx = values.get("actor_idx")
        target_idx = values.get("target_idx")
        deltas = [
            self._event_value(definition, event_type, actor_idx, target_idx)
            for definition in self._definitions
        ]
        # Per-opponent relation update is guarded by the SAME dedup gate below
        # (we only reach here for a not-yet-processed event_id), so a duplicated
        # event never double-counts relation either.
        self._apply_relation_event(event_type, actor_idx, target_idx)
        self._processed_event_ids.add(event_id)
        self.last_event_id = event_id if self.last_event_id is None else max(self.last_event_id, event_id)
        return self._change(deltas)

    def _apply_relation_event(
        self,
        event_type: str,
        actor_idx: Any,
        target_idx: Any,
    ) -> None:
        """Update the per-seat relation vector for the implicated other seat(s).

        No-op when ``player_idx`` is unknown (the helper returns no seats) or when
        no ``relation_updates`` key matches the derived kind.  The self slot is
        never touched because the helper never yields our own seat.  Mutations are
        clamped to ``[RELATION_MIN, RELATION_MAX]`` in the separate relation array.
        """

        rules = self.config.relation_updates
        if not rules:
            return
        for seat, kind in _relation_event_seats(
            event_type, actor_idx, target_idx, self.player_idx
        ):
            if seat == self.player_idx or not 0 <= seat < PLAYER_COUNT:
                continue
            delta = self._rule_value(rules, (kind, "*"))
            if delta is None:
                continue
            updated = float(self._relation[seat]) + float(delta)
            self._relation[seat] = np.float32(
                min(RELATION_MAX, max(RELATION_MIN, updated))
            )

    update_from_event = apply_event
    apply_public_event = apply_event

    def apply_events(self, events: Any) -> np.ndarray:
        if events is None:
            return self.values
        if isinstance(events, Mapping) or hasattr(events, "event_id"):
            events = (events,)
        for event in events:
            self.apply_event(event)
        return self.values

    update_from_events = apply_events

    def advance_step(self, environment_step: Any) -> np.ndarray:
        """Apply decay/recovery once for this environment step."""

        if isinstance(environment_step, (bool, np.bool_)) or not isinstance(environment_step, (int, np.integer)):
            raise TypeError("environment_step must be a non-negative integer")
        step = int(environment_step)
        if step < 0:
            raise ValueError("environment_step must be non-negative")
        if self.last_environment_step is not None and step < self.last_environment_step:
            raise ValueError("environment_step cannot move backwards")
        if self.last_environment_step == step:
            return self.values
        if self.last_environment_step is not None or step > 0:
            deltas = []
            for definition, current, baseline in zip(self._definitions, self._values, self._baselines):
                amount = max(0.0, float(definition.decay_rate))
                distance = float(baseline) - float(current)
                deltas.append(np.sign(distance) * min(abs(distance), amount))
            self._change(deltas)
        self.last_environment_step = step
        return self.values

    decay = advance_step
    apply_decay = advance_step

    def prepare_action(self, environment_step: int, events: Any = ()) -> np.ndarray:
        """Apply pending public events and step decay, then snapshot the label."""

        self.apply_events(events)
        self.advance_step(environment_step)
        return self.values

    label_snapshot = prepare_action

    @staticmethod
    def _action_keys(action: Any) -> tuple[str, str]:
        if isinstance(action, Mapping):
            if "action" in action and not isinstance(action["action"], Mapping):
                raw = action["action"]
            else:
                raw = action.get("action_type", action.get("type", action.get("index")))
            if action.get("valid", True) is False:
                return "invalid", "invalid"
        else:
            raw = action
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError("action must be a public integer or action mapping")
        if isinstance(raw, (int, np.integer)):
            number = int(raw)
            if number < 0:
                raise ValueError("action must be non-negative")
            kind = "pass" if number == 0 else "bang" if number == 1 else "bang"
            return str(number), kind
        text = _text(raw)
        return text, text

    def apply_action(
        self,
        action: Any,
        *,
        valid: bool = True,
        action_mask: Any | None = None,
    ) -> np.ndarray:
        """Apply a configured action rule only for a validated game action."""

        if not valid:
            return self.values
        if action_mask is not None:
            mask = np.asarray(action_mask)
            if mask.shape != (6,) or mask.dtype.kind not in "biu" or np.any((mask != 0) & (mask != 1)):
                raise ValueError("action_mask must be a six-entry binary mask")
            key, _ = self._action_keys(action)
            try:
                action_index = int(key)
            except ValueError:
                action_index = None
            if action_index is not None and (action_index < 0 or action_index >= 6 or not bool(mask[action_index])):
                return self.values
        key, action_type = self._action_keys(action)
        deltas = []
        for definition in self._definitions:
            value = self._rule_value(
                definition.action_updates,
                (key, action_type, "*"),
            )
            deltas.append(0.0 if value is None else value)
        return self._change(deltas)

    update_from_action = apply_action
    record_action = apply_action

    def apply_reward(self, reward: Any, *, valid: bool = True) -> np.ndarray:
        """Apply a configured reward rule; invalid/retried actions are no-ops."""

        if not valid:
            return self.values
        amount = _finite_number(reward, "reward")
        sign = "positive" if amount > 0 else "negative" if amount < 0 else "zero"
        exact = str(amount).rstrip("0").rstrip(".") if "." in str(amount) else str(amount)
        deltas = []
        for definition in self._definitions:
            value = self._rule_value(definition.reward_updates, (exact, sign, "*"))
            deltas.append(0.0 if value is None else value)
        return self._change(deltas)

    update_from_reward = apply_reward

    def apply_terminal(self, result: Any) -> np.ndarray:
        """Apply terminal/truncated rules from a public lifecycle result."""

        values = _mapping(result, label="terminal result") if result is not None else {}
        terminated = bool(values.get("terminated", False))
        truncated = bool(values.get("truncated", False))
        if terminated and truncated:
            raise ValueError("terminated and truncated cannot both be true")
        self.terminated = terminated
        self.truncated = truncated
        status = "truncated" if truncated else "terminated" if terminated else "running"
        result_name = values.get("result", values.get("outcome"))
        if isinstance(result_name, Mapping):
            result_name = result_name.get("result", result_name.get("outcome"))
        result_name = None if result_name is None else _text(result_name)
        deltas = []
        for definition in self._definitions:
            value = self._rule_value(
                definition.terminal_updates,
                (result_name, status, "normal" if terminated else None, "*")
            )
            deltas.append(0.0 if value is None else value)
        return self._change(deltas)

    update_from_terminal = apply_terminal

    def on_episode_end(self, result: Any | None = None) -> np.ndarray:
        return self.apply_terminal(result or {})

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "emotion_schema_version": self.config.schema_version,
            "player_idx": self.player_idx,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "config": self.config.model_dump(mode="json"),
            "emotion_names": list(self._names),
            "values": [float(value) for value in self._values],
            "baseline": [float(value) for value in self._baselines],
            "relation": [float(value) for value in self._relation],
            "relation_baseline": float(self._relation_baseline),
            "last_event_id": self.last_event_id,
            "processed_event_ids": list(self.processed_event_ids),
            "last_environment_step": self.last_environment_step,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    to_dict = state_dict
    serialize = state_dict

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = self.from_state_dict(state)
        self.__dict__.update(restored.__dict__)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EmotionModule":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        # Backward-compatible version gate: accept both the pre-relation "1.0"
        # states and current "1.1" states.  "1.0" states have no relation field
        # and are upgraded in memory to a neutral relation vector below.
        version = state.get("schema_version", cls.schema_version)
        if version not in _SUPPORTED_STATE_VERSIONS:
            raise ValueError("unsupported emotion module state schema")
        module = cls(state.get("config"), player_idx=state.get("player_idx"))
        names = tuple(state.get("emotion_names", module.emotion_names))
        if names != module.emotion_names:
            raise ValueError("emotion state names do not match its config schema")
        values = np.asarray(state.get("values", module.values), dtype=np.float32)
        if values.shape != (module.emotion_count,) or not np.all(np.isfinite(values)):
            raise ValueError("emotion state values have the wrong shape")
        module._values = np.clip(values, 0.0, 1.0).astype(np.float32)
        # Relation vector: absent in "1.0" states -> default to the neutral
        # baseline the constructor already installed.  When present ("1.1"),
        # validate shape/finiteness and clamp to the documented signed range.
        if "relation_baseline" in state:
            module._relation_baseline = _finite_number(
                state.get("relation_baseline"), "relation_baseline"
            )
        if "relation" in state:
            relation = np.asarray(state.get("relation"), dtype=np.float32)
            if relation.shape != (PLAYER_COUNT,) or not np.all(np.isfinite(relation)):
                raise ValueError("relation state values have the wrong shape")
            module._relation = np.clip(
                relation, RELATION_MIN, RELATION_MAX
            ).astype(np.float32)
        else:
            module._relation = np.full(
                PLAYER_COUNT, module._relation_baseline, dtype=np.float32
            )
        module.episode_id = state.get("episode_id")
        module.episode_seed = state.get("episode_seed")
        module.last_event_id = state.get("last_event_id")
        module.last_environment_step = state.get("last_environment_step")
        module.terminated = bool(state.get("terminated", False))
        module.truncated = bool(state.get("truncated", False))
        module._processed_event_ids = {int(item) for item in state.get("processed_event_ids", [])}
        return module

    from_dict = from_state_dict
    deserialize = from_state_dict

    def to_json(self) -> str:
        import json

        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "EmotionModule":
        import json

        if not isinstance(value, str):
            raise TypeError("serialized state must be a JSON string")
        return cls.from_state_dict(json.loads(value))


__all__ = [
    "EMOTION_MODULE_STATE_VERSION",
    "PLAYER_COUNT",
    "RELATION_MIN",
    "RELATION_MAX",
    "EmotionModule",
]
