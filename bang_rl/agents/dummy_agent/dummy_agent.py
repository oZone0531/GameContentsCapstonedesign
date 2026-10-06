"""Deterministic public-information DummyAgent implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..base import BaseAgent
from .emotion_module import EmotionModule
from .intent_policy import ActionSelection, ActionSelector, IntentModule, IntentPolicyConfig
from .role_belief import RoleBeliefEstimator


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


def _relation_target_contribution(
    relation: Any,
    weight: float,
    player_idx: int | None,
) -> list[float]:
    """Return the per-target relation term over the length-6 action space.

    The action space is: index ``0`` = pass; indices ``1..5`` = bang at seat
    ``s`` where ``s = action_index - 1``.  Pass gets no term.  For each bang
    action the term is ``-weight * relation[s]``.

    Sign convention (consistent with FEAT-002): ``relation[s]`` is signed, with
    negative meaning grudge and positive meaning trust.  The leading minus sign
    means that, with ``weight > 0``, a grudge (negative relation) toward seat
    ``s`` produces a POSITIVE bump to the "bang seat ``s``" score, so the agent
    is biased to shoot the opponent it resents; trust (positive relation)
    lowers that score.  With ``weight == 0.0`` (the default) every entry is
    exactly ``0.0`` so action selection is byte-for-byte unchanged.

    The self slot (``s == player_idx``) is neutral in the relation vector, so
    its term is ~0; it is skipped defensively regardless.  This helper is pure
    (builtins only, no numpy) so it is unit-testable in the restricted sandbox.
    """

    contribution = [0.0] * 6
    weight_value = float(weight)
    if weight_value == 0.0:
        return contribution
    for seat in range(5):
        if player_idx is not None and seat == player_idx:
            continue
        contribution[seat + 1] = -weight_value * float(relation[seat])
    return contribution


def _json_safe(value: Any) -> Any:
    """Convert NumPy RNG state values to stable JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


class DummyAgent(BaseAgent):
    """Dummy agent composed of public role beliefs and an intent policy.

    The policy receives only the canonical observation and the estimator's
    public-information belief matrix. It never receives the environment's
    hidden role assignment.
    """

    def __init__(
        self,
        agent_id: str | int = "dummy",
        seed: int | None = None,
        *,
        player_idx: int | None = None,
        role_estimator: RoleBeliefEstimator | None = None,
        intent_module: IntentModule | None = None,
        emotion_module: EmotionModule | None = None,
        emotion_config: Any | None = None,
        emotion: Any | None = None,
        policy_config: IntentPolicyConfig | Mapping[str, Any] | Any | None = None,
        intent_config: IntentPolicyConfig | Mapping[str, Any] | Any | None = None,
        action_mode: str = "deterministic",
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, seed=seed)
        if action_mode not in ActionSelector.MODES:
            raise ValueError("action_mode must be 'deterministic' or 'stochastic'")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator or None")
        if rng is not None and seed is not None:
            raise ValueError("provide either seed or rng, not both")
        if role_estimator is not None and not isinstance(role_estimator, RoleBeliefEstimator):
            raise TypeError("role_estimator must be a RoleBeliefEstimator or None")
        if intent_module is not None and not isinstance(intent_module, IntentModule):
            raise TypeError("intent_module must be an IntentModule or None")
        if player_idx is None and role_estimator is not None:
            player_idx = role_estimator.player_idx
        if player_idx is None and intent_module is not None:
            player_idx = intent_module.player_idx
        if player_idx is None and isinstance(agent_id, str) and agent_id.startswith("player_"):
            suffix = agent_id.removeprefix("player_")
            if suffix.isdigit():
                player_idx = int(suffix)
        self.role_estimator = role_estimator or RoleBeliefEstimator(player_idx=player_idx)
        if intent_module is not None:
            self.intent_module = intent_module
        else:
            self.intent_module = IntentModule(
                player_idx=player_idx,
                config=policy_config if policy_config is not None else intent_config,
            )
        if emotion_module is not None and not isinstance(emotion_module, EmotionModule):
            raise TypeError("emotion_module must be an EmotionModule or None")
        if emotion_config is not None and emotion is not None:
            raise ValueError("provide only one of emotion_config or emotion")
        self.emotion_module = emotion_module or EmotionModule(
            emotion_config if emotion_config is not None else emotion,
            player_idx=player_idx,
        )
        self._action_mode = action_mode
        self._rng = rng if rng is not None else np.random.default_rng(self.action_seed())
        self._last_selection: ActionSelection | None = None

    @property
    def role_beliefs(self) -> np.ndarray:
        return self.role_estimator.beliefs

    @property
    def belief_estimator(self) -> RoleBeliefEstimator:
        """Descriptive alias used by integrations that avoid implementation names."""

        return self.role_estimator

    @property
    def intent(self) -> IntentModule:
        return self.intent_module

    @property
    def emotion(self) -> EmotionModule:
        return self.emotion_module

    @property
    def emotion_state(self) -> np.ndarray:
        """Return the detached private DummyAgent emotion vector."""

        return self.emotion_module.values

    @property
    def policy_config(self) -> IntentPolicyConfig:
        return self.intent_module.config

    @property
    def action_mode(self) -> str:
        return self._action_mode

    @property
    def rng(self) -> np.random.Generator:
        """Return the private action RNG used only by stochastic selection."""

        return self._rng

    @property
    def last_decision_info(self) -> ActionSelection | None:
        """Return the latest structured, public-input decision result."""

        return self._last_selection

    @property
    def decision_info(self) -> dict[str, Any] | None:
        """Return a detached JSON-safe view of the latest decision."""

        return None if self._last_selection is None else self._last_selection.to_dict()

    def set_action_mode(self, mode: str) -> None:
        if mode not in ActionSelector.MODES:
            raise ValueError("action_mode must be 'deterministic' or 'stochastic'")
        self._action_mode = mode

    def reset_beliefs(self, observation: Any | None = None) -> np.ndarray:
        return self.role_estimator.reset(observation)

    def update_beliefs(self, observation: Any | None = None, *, events: Any | None = None) -> np.ndarray:
        return self.role_estimator.update(observation, events=events)

    def faction_strength(self, observation: Any):
        """Return public-information faction strength for the current board."""

        return self.intent_module.faction_strength(observation, self.role_beliefs)

    faction_strengths = faction_strength

    def on_episode_start(self, context: Any) -> None:
        super().on_episode_start(context)
        self.role_estimator.reset()
        context_observation = None
        if isinstance(context, Mapping):
            context_observation = context.get("observation")
        elif hasattr(context, "observation"):
            context_observation = context.observation
        if context_observation is not None:
            self.role_estimator.update(context_observation)
        self.emotion_module.on_episode_start(context)
        self._last_selection = None
        if self.episode_seed is not None:
            self._rng = np.random.default_rng(self.action_seed(self.episode_seed))

    def on_episode_end(self, result: Any) -> None:
        super().on_episode_end(result)
        self.emotion_module.on_episode_end(result)

    @staticmethod
    def _field(observation: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(observation, Mapping) and name in observation:
                return observation[name]
            if hasattr(observation, name):
                return getattr(observation, name)
            if hasattr(observation, "get"):
                value = observation.get(name, None)
                if value is not None:
                    return value
        return default

    @classmethod
    def _public_events(cls, observation: Any) -> tuple[Any, ...]:
        events = cls._field(observation, "event_sequence", "event_history", default=())
        if events is None:
            return ()
        if isinstance(events, Mapping) or hasattr(events, "event_id"):
            events = (events,)
        try:
            values = tuple(events)
        except TypeError as exc:
            raise TypeError("event_sequence must be iterable") from exc
        for event in values:
            if isinstance(event, Mapping):
                leaked = _PRIVATE_FIELDS.intersection(event.keys())
                if leaked:
                    raise ValueError(
                        "privileged fields are not accepted: " + ", ".join(sorted(leaked))
                    )
        return values

    @classmethod
    def _environment_step(cls, observation: Any) -> int | None:
        raw = cls._field(observation, "environment_step", default=None)
        if raw is None:
            return None
        values = np.asarray(raw).reshape(-1)
        if values.size != 1:
            raise ValueError("environment_step must contain one non-negative integer")
        value = values[0].item()
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("environment_step must be a non-negative integer")
        if int(value) < 0:
            raise ValueError("environment_step must be a non-negative integer")
        return int(value)

    @classmethod
    def _board_state(cls, observation: Any) -> tuple[Any, Any]:
        """Return the PUBLIC per-seat (healths, alive) from the observation.

        Reads only the public ``healths`` / ``alive`` fields through ``_field``
        (never a private field).  A fixture that omits either field yields
        ``(None, None)`` so the emotion module treats it as "no board observed"
        (a safe no-op), exactly like ``FactionStrengthEstimator._board`` /
        ``IntentModule._alive_and_health`` tolerate a missing board.
        """

        healths = cls._field(observation, "healths", "health", default=None)
        alive = cls._field(observation, "alive", default=None)
        if healths is None or alive is None:
            return None, None
        return healths, alive

    def _prepare_emotion(self, observation: Any) -> None:
        healths, alive = self._board_state(observation)
        self._prepare_emotion_values(
            self._public_events(observation),
            self._environment_step(observation),
            healths,
            alive,
        )

    def _prepare_emotion_values(
        self,
        events: tuple[Any, ...],
        step: int | None,
        healths: Any = None,
        alive: Any = None,
    ) -> None:
        # Feed the latest PUBLIC board snapshot into the emotion module before
        # applying events so control judgments can read it.  A missing board
        # (None) clears the snapshot to a safe no-op.
        self.emotion_module.observe_public_state(healths, alive)
        if step is None:
            # Small standalone fixtures may omit the environment clock.  They
            # can still deliver a valid public event boundary, but no decay is
            # inferred without an explicit environment step.
            self.emotion_module.apply_events(events)
        else:
            self.emotion_module.prepare_action(step, events)

    def _emotion_contribution(self) -> np.ndarray:
        values = self.emotion_module.values
        by_name = dict(zip(self.emotion_module.emotion_names, values))
        contribution = np.zeros(6, dtype=np.float64)
        for action in range(6):
            effects = self.emotion_module.effects_for_action(action)
            contribution[action] = sum(
                float(by_name[name]) * float(effect)
                for name, effect in effects.items()
                if name in by_name
            )
        # Additive per-target relation term (FEAT-004): bias banging a resented
        # seat when relation_target_weight > 0.  Defaults to exactly 0.0 so
        # existing configs and determinism are preserved.  Read relation via the
        # read-only property (a detached copy); never mutate emotion/relation
        # state during action selection.
        weight = float(self.intent_module.config.relation_target_weight)
        if weight != 0.0:
            relation = self.emotion_module.relation
            target_term = _relation_target_contribution(
                relation, weight, self.emotion_module.player_idx
            )
            contribution += np.asarray(target_term, dtype=np.float64)
        return contribution

    def act(self, observation: Any, action_mask: Any, *, mode: str | None = None) -> int:
        """Choose an action from intent plus emotion using only public state."""

        mask = self._validate_act_inputs(observation, action_mask)
        if self.emotion_module.terminated or self.emotion_module.truncated:
            raise RuntimeError("cannot act after the episode has ended")
        effective_mode = self._action_mode if mode is None else mode
        if effective_mode not in ActionSelector.MODES:
            raise ValueError("action mode must be 'deterministic' or 'stochastic'")
        # Validate public event fields and the environment boundary before
        # mutating either private module.
        public_events = self._public_events(observation)
        environment_step = self._environment_step(observation)
        board_healths, board_alive = self._board_state(observation)
        self.role_estimator.update(observation)
        self._prepare_emotion_values(
            public_events, environment_step, board_healths, board_alive
        )
        intent = self.intent_module.decision(observation, self.role_beliefs)
        selection = ActionSelector.select_with_info(
            intent.scores,
            mask,
            emotion_contribution=self._emotion_contribution(),
            mode=effective_mode,
            rng=self._rng,
        )
        self._last_selection = selection
        return self._validate_selected_action(selection.action, mask)

    def act_with_info(
        self, observation: Any, action_mask: Any, *, mode: str | None = None
    ) -> ActionSelection:
        """Select an action and return the complete structured decision."""

        self.act(observation, action_mask, mode=mode)
        assert self._last_selection is not None
        return self._last_selection

    def select_action(self, observation: Any, action_mask: Any, *, mode: str | None = None) -> int:
        return self.act(observation, action_mask, mode=mode)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "agent_id": self.agent_id,
            "seed": self.seed,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "role_estimator": self.role_estimator.state_dict(),
            "intent_module": self.intent_module.state_dict(),
            "emotion_module": self.emotion_module.state_dict(),
            "action_selector": {
                "schema_version": ActionSelector.schema_version,
                "mode": self._action_mode,
                "rng_state": _json_safe(self._rng.bit_generator.state),
            },
            "last_decision": None if self._last_selection is None else self._last_selection.to_dict(),
        }

    to_dict = state_dict
    serialize = state_dict

    def to_json(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        restored = self.from_state_dict(state)
        self.__dict__.update(restored.__dict__)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DummyAgent":
        if not isinstance(state, dict):
            raise TypeError("state must be a mapping")
        if state.get("schema_version", "2.0") != "2.0":
            raise ValueError("unsupported DummyAgent state schema")
        estimator = RoleBeliefEstimator.from_state_dict(state["role_estimator"])
        intent_state = state.get("intent_module")
        intent_module = IntentModule.from_state_dict(intent_state) if intent_state is not None else None
        emotion_state = state.get("emotion_module")
        emotion_module = EmotionModule.from_state_dict(emotion_state) if emotion_state is not None else None
        selector_state = state.get("action_selector", {})
        mode = selector_state.get("mode", "deterministic")
        agent = cls(
            agent_id=state.get("agent_id", "dummy"),
            seed=state.get("seed"),
            role_estimator=estimator,
            intent_module=intent_module,
            emotion_module=emotion_module,
            action_mode=mode,
        )
        agent.episode_id = state.get("episode_id")
        agent.episode_seed = state.get("episode_seed")
        rng_state = selector_state.get("rng_state")
        if isinstance(rng_state, Mapping):
            agent._rng.bit_generator.state = dict(rng_state)
        last_decision = state.get("last_decision")
        agent._last_selection = (
            None if last_decision is None else ActionSelection.from_dict(last_decision)
        )
        return agent

    @classmethod
    def from_json(cls, value: str) -> "DummyAgent":
        if not isinstance(value, str):
            raise TypeError("serialized state must be a JSON string")
        return cls.from_state_dict(json.loads(value))

    from_dict = from_state_dict
    deserialize = from_state_dict


__all__ = ["DummyAgent"]
