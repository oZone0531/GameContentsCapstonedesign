"""Emotion formula module for the DummyAgent (alternative to ``EmotionModule``).

Spec: emotion_formula_spec.md (SWCON367 emotion design).

- Three global emotion states (S confidence, A aggression, F anxiety) + derived calm P.
- Per-opponent relation R[5] in [-1, 1] (negative = grudge, positive = liking).
- Every coefficient is an ``EmotionParams`` variable (no hard-coded constants).

Selection: a DummyAgent uses this module only when its ``EmotionConfig`` has a
``formula`` block. Without it the existing ``EmotionModule`` is used unchanged.

Range convention for relations (team decision 2026-10-06):
- internal value ``R`` is signed in [-1, 1];
- ``relation_01`` / the label use ``(R + 1) / 2`` in [0, 1] (0.5 = neutral);
- converting back: ``R = 2 * label - 1``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

PLAYER_COUNT = 5
ACTION_COUNT = 6
SHERIFF, DEPUTY, OUTLAW_A, OUTLAW_B, TRAITOR = range(5)

FORMULA_STATE_SCHEMA = "formula-1.0"

# My role -> roles treated as my team. Traitor uses a special rule (see ally_prob).
TEAM_ROLES: dict[int, tuple[int, ...]] = {
    SHERIFF: (SHERIFF, DEPUTY),
    DEPUTY: (SHERIFF, DEPUTY),
    OUTLAW_A: (OUTLAW_A, OUTLAW_B),
    OUTLAW_B: (OUTLAW_A, OUTLAW_B),
    TRAITOR: (),
}

# Role index -> key used in "personality_by_role".
ROLE_KEYS = {SHERIFF: "sheriff", DEPUTY: "deputy", OUTLAW_A: "outlaw", OUTLAW_B: "outlaw", TRAITOR: "traitor"}
PERSONALITY_KEYS = ("sheriff", "deputy", "outlaw", "traitor")

# Only these events move global / relation emotions.
HARM_EVENTS = ("bang", "player_eliminated")

# Global emotions shown as bars in the GUI (values are in [0, 1]).
EMOTION_NAMES = ("calm", "confidence", "aggression", "anxiety")
# 9-entry training label: 4 global + 5 relations mapped to [0, 1].
LABEL_NAMES = EMOTION_NAMES + tuple(f"relation_{seat}" for seat in range(PLAYER_COUNT))

DEFAULT_DESCRIPTIONS = {
    "calm": "Derived: 1 - max(confidence, aggression, anxiety).",
    "confidence": "Rises when good things happen to my side (an enemy is hit or eliminated).",
    "aggression": "Share of bad events while I still feel in control (HP-based control appraisal).",
    "anxiety": "Share of bad events while I feel out of control (HP-based control appraisal).",
}


@dataclass
class EmotionParams:
    # --- event magnitude ---
    beta_bang: float = 0.15        # base magnitude of one bang
    beta_elim: float = 0.25        # extra magnitude of an elimination (added on top of bang)
    nu_hp: float = 0.0             # larger magnitude when the target has low HP (0 = off)
    # --- good / bad event weights ---
    omega_self: float = 1.0        # I was hit
    omega_ally: float = 0.6        # an ally was hit (times a_tau)
    omega_enemy: float = 0.5       # an enemy was hit (times 1 - a_tau)
    # --- cross inhibition ---
    lambda_S: float = 0.3          # how much bad events reduce confidence
    lambda_A: float = 0.0          # how much good events reduce aggression
    lambda_F: float = 0.3          # how much good events reduce anxiety
    # --- control appraisal ---
    w_hp: float = 0.5              # weight of my own HP in D (the rest is team HP)
    theta: float = 0.5             # aggression / anxiety threshold
    k: float = 8.0                 # sharpness of the threshold
    # --- relation ---
    eta_hurt: float = 1.0          # grudge toward whoever shot me
    eta_ally: float = 0.5          # grudge toward whoever shot my ally
    eta_avenge: float = 0.6        # liking toward whoever shot someone I hold a grudge against
    eta_enemy: float = 0.0         # liking toward whoever shot a (likely) enemy (0 = off)
    traitor_sheriff_ally: float = 1.0  # how much the traitor treats the sheriff as "keep alive" (until the 1v1)
    # --- personality (sensitivity p, baseline b, recovery d) ---
    p_S: float = 1.0
    p_A: float = 1.0
    p_F: float = 1.0
    p_R: float = 1.0
    b_S: float = 0.1
    b_A: float = 0.1
    b_F: float = 0.1
    d_S: float = 0.01
    d_A: float = 0.01
    d_F: float = 0.01
    d_R: float = 0.005
    # --- noise ---
    sigma: float = 0.01            # per-step noise std for S, A, F
    sigma_R: float = 0.005         # per-step noise std for R
    # --- action contribution ---
    kappa_S: float = 1.0
    kappa_A: float = 1.0
    kappa_F: float = 1.0
    kappa_R: float = 1.0
    lambda_E: float = 1.0          # overall scale of the emotion contribution (added to intent scores)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "EmotionParams":
        if values is None:
            return cls()
        unknown = set(values) - set(cls.names())
        if unknown:
            raise ValueError(f"unknown emotion formula params: {sorted(unknown)}")
        out = cls(**{key: float(value) for key, value in values.items()})
        for name, value in asdict(out).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return out

    @classmethod
    def for_role(
        cls,
        params: Mapping[str, Any] | None,
        personality_by_role: Mapping[str, Mapping[str, Any]] | None,
        role: int | None,
    ) -> "EmotionParams":
        """Base params with the role's personality overrides applied on top."""

        values = dict(params or {})
        if role is not None and personality_by_role:
            values.update(personality_by_role.get(ROLE_KEYS[int(role)], {}) or {})
        return cls.from_mapping(values)


def validate_formula_block(params: Mapping[str, Any], personality_by_role: Mapping[str, Any]) -> None:
    """Raise ValueError for unknown parameter names or role keys (used by config validation)."""

    EmotionParams.from_mapping(params)
    for role_key, overrides in (personality_by_role or {}).items():
        if role_key not in PERSONALITY_KEYS:
            raise ValueError(f"unknown personality role {role_key!r}; expected one of {PERSONALITY_KEYS}")
        EmotionParams.from_mapping({**dict(params), **dict(overrides or {})})


# ---------------------------------------------------------------------------
# 1. Input conversion
# ---------------------------------------------------------------------------
def ally_prob(beliefs: np.ndarray, self_idx: int, self_role: int, alive: Sequence[bool],
              params: EmotionParams) -> np.ndarray:
    """a_j = sum_{r in Team(me)} B[j, r],  a_me = 1.

    beliefs: role_belief matrix (5x5, B[player, role]).
    Traitor: if 3+ players are alive, a_j = traitor_sheriff_ally * B[j, SHERIFF]; otherwise 0.
    """
    B = np.asarray(beliefs, dtype=np.float64)
    if self_role == TRAITOR:
        living = int(np.sum(np.asarray(alive, dtype=bool)))
        a = params.traitor_sheriff_ally * B[:, SHERIFF] if living > 2 else np.zeros(PLAYER_COUNT)
    else:
        a = B[:, list(TEAM_ROLES[self_role])].sum(axis=1)
    a = np.clip(a, 0.0, 1.0)
    a[self_idx] = 1.0
    return a


def max_health(sheriff_idx: int) -> np.ndarray:
    """Game rule: sheriff 5, others 4 (same as bang_rl.env.types.initial_health)."""
    hp = np.full(PLAYER_COUNT, 4.0)
    hp[sheriff_idx] = 5.0
    return hp


def control(self_idx: int, a: np.ndarray, healths: Sequence[float], alive: Sequence[bool],
            hp_max: np.ndarray, params: EmotionParams) -> tuple[float, float]:
    """Control appraisal D in [0,1] and split ratio g = sigmoid(k (D - theta))."""
    h = np.asarray(healths, dtype=np.float64)
    live = np.asarray(alive, dtype=bool)
    own = h[self_idx] / hp_max[self_idx]
    h_ally = float(np.sum(a[live] * h[live]))
    h_enemy = float(np.sum((1.0 - a[live]) * h[live]))
    team = 0.5 if h_ally + h_enemy <= 0 else h_ally / (h_ally + h_enemy)
    D = params.w_hp * own + (1.0 - params.w_hp) * team
    g = 1.0 / (1.0 + np.exp(-params.k * (D - params.theta)))
    return float(D), float(g)


def _event_fields(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    if hasattr(event, "to_public_dict"):
        return dict(event.to_public_dict())
    return dict(event.to_dict())


def _event_type(values: Mapping[str, Any]) -> str:
    raw = values.get("event_type")
    raw = getattr(raw, "value", raw)
    return str(raw).rsplit(".", 1)[-1].lower()


# ---------------------------------------------------------------------------
# 2. State and updates (reference implementation, unchanged math)
# ---------------------------------------------------------------------------
class EmotionState:
    """Emotion state of one dummy."""

    def __init__(self, self_idx: int, params: EmotionParams | None = None,
                 rng: np.random.Generator | None = None) -> None:
        self.i = int(self_idx)
        self.p = params or EmotionParams()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        p = self.p
        self.S, self.A, self.F = p.b_S, p.b_A, p.b_F
        self.R = np.zeros(PLAYER_COUNT)
        self.R_active = np.ones(PLAYER_COUNT, dtype=bool)
        self.R_active[self.i] = False
        self.last_step: int | None = None
        self.processed: set[int] = set()
        self.last_debug: dict[str, float] = {}

    # -- step 1: decay -----------------------------------------------------
    def _decay(self, dt: int) -> None:
        if dt <= 0:
            return
        p = self.p

        def toward(x: float, target: float, rate: float) -> float:
            dist = target - x
            return float(x + np.sign(dist) * min(abs(dist), max(rate, 0.0) * dt))

        self.S = toward(self.S, p.b_S, p.d_S)
        self.A = toward(self.A, p.b_A, p.d_A)
        self.F = toward(self.F, p.b_F, p.d_F)
        step = np.minimum(np.abs(self.R), max(p.d_R, 0.0) * dt)
        self.R = np.where(self.R_active, self.R - np.sign(self.R) * step, self.R)

    # -- step 2: events ----------------------------------------------------
    def _apply_events(self, events: Iterable[Mapping[str, Any]], a: np.ndarray,
                      healths: Sequence[float], alive: Sequence[bool], hp_max: np.ndarray) -> None:
        p, i = self.p, self.i
        R_old = self.R.copy()   # all events in this step use R from the start of the step (order-independent)
        G = Bd = 0.0
        dR = np.zeros(PLAYER_COUNT)
        for ev in events:
            etype = _event_type(ev)
            if etype not in HARM_EVENTS:
                continue
            tau = ev.get("target_idx")
            if tau is None:
                tau = ev.get("eliminated_idx")
            if tau is None:
                continue
            tau = int(tau)
            s = ev.get("actor_idx")
            m = p.beta_bang if etype == "bang" else p.beta_elim
            if p.nu_hp != 0.0 and ev.get("hp_after") is not None:
                m *= 1.0 + p.nu_hp * (1.0 - float(ev["hp_after"]) / hp_max[tau])
            if tau == i:
                Bd += m * p.omega_self
            else:
                Bd += m * p.omega_ally * a[tau]
                G += m * p.omega_enemy * (1.0 - a[tau])
            if s is not None and int(s) != i:
                s = int(s)
                if tau == i:
                    dR[s] -= p.eta_hurt * m
                else:
                    dR[s] -= p.eta_ally * m * a[tau]
                    dR[s] += p.eta_avenge * m * max(0.0, -R_old[tau])
                    dR[s] += p.eta_enemy * m * (1.0 - a[tau])
        D, g = control(i, a, healths, alive, hp_max, p)
        self.S = float(np.clip(self.S + p.p_S * (G - p.lambda_S * Bd), 0.0, 1.0))
        self.A = float(np.clip(self.A + p.p_A * (Bd * g - p.lambda_A * G), 0.0, 1.0))
        self.F = float(np.clip(self.F + p.p_F * (Bd * (1.0 - g) - p.lambda_F * G), 0.0, 1.0))
        self.R = np.where(self.R_active, np.clip(self.R + p.p_R * dR, -1.0, 1.0), self.R)
        self.last_debug = {"G": float(G), "Bd": float(Bd), "D": D, "g": g}

    # -- step 3: noise -----------------------------------------------------
    def _noise(self) -> None:
        p = self.p
        if p.sigma > 0:
            eps = self.rng.normal(0.0, p.sigma, 3)
            self.S = float(np.clip(self.S + eps[0], 0.0, 1.0))
            self.A = float(np.clip(self.A + eps[1], 0.0, 1.0))
            self.F = float(np.clip(self.F + eps[2], 0.0, 1.0))
        if p.sigma_R > 0:
            eps = self.rng.normal(0.0, p.sigma_R, PLAYER_COUNT)
            self.R = np.where(self.R_active, np.clip(self.R + eps, -1.0, 1.0), self.R)

    # -- full step ---------------------------------------------------------
    def step(self, environment_step: int, events: Iterable[Any], *, beliefs: np.ndarray,
             self_role: int, sheriff_idx: int, healths: Sequence[float],
             alive: Sequence[bool]) -> np.ndarray:
        """Call once at each pre-action boundary. Returns the label vector."""
        step = int(environment_step)
        dt = 0 if self.last_step is None else step - self.last_step
        if dt < 0:
            raise ValueError("environment_step cannot move backwards")
        new_events = []
        for raw in events:
            ev = _event_fields(raw)
            if ev.get("public", True) is False:
                continue
            eid = int(ev["event_id"])
            if eid in self.processed:
                continue
            self.processed.add(eid)
            new_events.append(ev)
        hp_max = max_health(sheriff_idx)
        a = ally_prob(beliefs, self.i, self_role, alive, self.p)
        self._decay(dt)
        self._apply_events(new_events, a, healths, alive, hp_max)
        # freeze relations to dead players (masked as 0.5 in the label)
        dead = ~np.asarray(alive, dtype=bool)
        self.R_active &= ~dead
        self.R[dead] = 0.0
        if dt > 0:
            self._noise()
        self.last_step = step
        return self.label()

    # -- outputs -----------------------------------------------------------
    @property
    def P(self) -> float:
        return float(np.clip(1.0 - max(self.S, self.A, self.F), 0.0, 1.0))

    def relation_01(self) -> np.ndarray:
        return np.where(self.R_active, (self.R + 1.0) / 2.0, 0.5)

    def label(self) -> np.ndarray:
        """[calm, confidence, aggression, anxiety, relation_0..4]. Relations mapped to [0,1] via (R+1)/2; self/dead = 0.5."""
        return np.concatenate([[self.P, self.S, self.A, self.F], self.relation_01()]).astype(np.float32)

    def label_mask(self) -> np.ndarray:
        return np.concatenate([np.ones(4, dtype=bool), self.R_active])

    def action_contribution(self) -> np.ndarray:
        """6-entry vector to pass as ActionSelector.select_with_info(emotion_contribution=...)."""
        p = self.p
        u = np.zeros(ACTION_COUNT)
        u[0] = p.kappa_F * self.F - p.kappa_S * self.S - p.kappa_A * self.A
        for j in range(PLAYER_COUNT):
            if j == self.i:
                continue
            u[1 + j] = p.kappa_S * self.S + p.kappa_A * self.A - p.kappa_R * self.R[j]
        return p.lambda_E * u


# ---------------------------------------------------------------------------
# 3. DummyAgent-facing module
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _get(observation: Any, name: str, default: Any = None) -> Any:
    if isinstance(observation, Mapping):
        return observation.get(name, default)
    getter = getattr(observation, "get", None)
    if callable(getter):
        value = getter(name, None)
        if value is not None:
            return value
    return getattr(observation, name, default)


def _scalar(value: Any) -> int | None:
    if value is None:
        return None
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise ValueError("expected a single integer value")
    return int(values[0])


def formula_block(config: Any) -> Mapping[str, Any] | None:
    """Return the ``formula`` block of an EmotionConfig-like value, or None."""

    if config is None:
        return None
    raw = config.get("formula") if isinstance(config, Mapping) else getattr(config, "formula", None)
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    return dict(raw)


class FormulaEmotionModule:
    """Per-dummy emotion state driven by the emotion formula.

    The DummyAgent calls :meth:`observe` with the dummy's OWN public observation
    and its own role-belief matrix at every pre-action boundary, then reads
    :meth:`action_contribution` on its turn.
    """

    kind = "emotion_formula"
    schema_version = FORMULA_STATE_SCHEMA

    def __init__(
        self,
        config: Any = None,
        *,
        player_idx: int | None = None,
        seed: int | None = None,
    ) -> None:
        if player_idx is None:
            raise ValueError("FormulaEmotionModule requires player_idx")
        self.player_idx = int(player_idx)
        self._config_dump = self._dump_config(config)
        block = formula_block(config) or {}
        self.formula_schema_version = str(block.get("schema_version", FORMULA_STATE_SCHEMA))
        self.base_params: dict[str, float] = {k: float(v) for k, v in dict(block.get("params", {}) or {}).items()}
        self.personality_by_role: dict[str, dict[str, float]] = {
            str(role): {k: float(v) for k, v in dict(values or {}).items()}
            for role, values in dict(block.get("personality_by_role", {}) or {}).items()
        }
        validate_formula_block(self.base_params, self.personality_by_role)
        self.self_role: int | None = None
        self.episode_id: str | None = None
        self.episode_seed: int | None = None
        self.terminated = False
        self.truncated = False
        self._seed = seed
        self._state = EmotionState(
            self.player_idx,
            EmotionParams.for_role(self.base_params, self.personality_by_role, None),
            np.random.default_rng(seed),
        )

    # -- config ------------------------------------------------------------
    @staticmethod
    def _dump_config(config: Any) -> dict[str, Any]:
        if config is None:
            return {}
        if hasattr(config, "model_dump"):
            return dict(config.model_dump(mode="json"))
        return dict(_json_safe(dict(config)))

    def _descriptions(self) -> list[dict[str, str]]:
        given = {}
        for item in self._config_dump.get("emotions", []) or []:
            if isinstance(item, Mapping) and item.get("name"):
                given[str(item["name"])] = str(item.get("description") or "")
        return [{"name": n, "description": given.get(n) or DEFAULT_DESCRIPTIONS[n]} for n in EMOTION_NAMES]

    @property
    def params(self) -> EmotionParams:
        return self._state.p

    # -- read-only views ---------------------------------------------------
    @property
    def emotion_names(self) -> tuple[str, ...]:
        return EMOTION_NAMES

    @property
    def emotion_count(self) -> int:
        return len(EMOTION_NAMES)

    @property
    def values(self) -> np.ndarray:
        st = self._state
        return np.asarray([st.P, st.S, st.A, st.F], dtype=np.float32)

    @property
    def relation(self) -> np.ndarray:
        """Signed relation vector in [-1, 1] (detached copy)."""
        return np.asarray(self._state.R, dtype=np.float64).copy()

    @property
    def relation_01(self) -> np.ndarray:
        return np.asarray(self._state.relation_01(), dtype=np.float64)

    @property
    def last_environment_step(self) -> int | None:
        return self._state.last_step

    def label(self) -> np.ndarray:
        return self._state.label()

    def label_mask(self) -> np.ndarray:
        return self._state.label_mask()

    def action_contribution(self) -> np.ndarray:
        return np.asarray(self._state.action_contribution(), dtype=np.float64)

    # -- lifecycle ---------------------------------------------------------
    def _set_role(self, role: int) -> None:
        """Apply the role's personality. Resets to the new baselines if nothing happened yet."""

        role = int(role)
        if self.self_role == role:
            return
        self.self_role = role
        st = self._state
        untouched = not st.processed and st.last_step is None
        st.p = EmotionParams.for_role(self.base_params, self.personality_by_role, role)
        if untouched:
            rng = st.rng
            st.reset()
            st.rng = rng

    def on_episode_start(self, context: Any = None, *, seed: int | None = None) -> None:
        self.episode_id = None
        self.terminated = False
        self.truncated = False
        if isinstance(context, Mapping):
            self.episode_id = None if context.get("episode_id") is None else str(context.get("episode_id"))
            self.episode_seed = context.get("seed", self.episode_seed)
            observation = context.get("observation")
        else:
            observation = getattr(context, "observation", None)
        if seed is not None:
            self._seed = int(seed)
        self.self_role = None
        self._state = EmotionState(
            self.player_idx,
            EmotionParams.for_role(self.base_params, self.personality_by_role, None),
            np.random.default_rng(self._seed),
        )
        if observation is not None:
            role = _scalar(_get(observation, "self_role"))
            if role is not None:
                self._set_role(role)

    def on_episode_end(self, result: Any = None) -> None:
        self.terminated = True

    def observe(self, observation: Any, beliefs: np.ndarray) -> np.ndarray | None:
        """Advance to this observation's boundary. Returns the label, or None if the
        observation lacks the public fields the formula needs."""

        step = _scalar(_get(observation, "environment_step"))
        role = _scalar(_get(observation, "self_role"))
        sheriff = _scalar(_get(observation, "sheriff_idx"))
        healths = _get(observation, "healths")
        alive = _get(observation, "alive")
        if step is None or role is None or sheriff is None or healths is None or alive is None:
            return None
        if self._state.last_step is not None and step < self._state.last_step:
            # A stale observation (e.g. a re-read of an earlier boundary): ignore it.
            return self._state.label()
        self._set_role(role)
        events = _get(observation, "event_sequence", ()) or ()
        return self._state.step(
            step,
            events,
            beliefs=np.asarray(beliefs, dtype=np.float64),
            self_role=role,
            sheriff_idx=sheriff,
            healths=np.asarray(healths, dtype=np.float64),
            alive=np.asarray(alive, dtype=bool),
        )

    # -- serialization -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        st = self._state
        config = dict(self._config_dump)
        config["emotions"] = self._descriptions()
        return {
            "schema_version": FORMULA_STATE_SCHEMA,
            "kind": self.kind,
            "player_idx": self.player_idx,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "self_role": self.self_role,
            "config": config,
            # GUI emotion bars read these three keys.
            "emotion_names": list(EMOTION_NAMES),
            "values": [float(v) for v in self.values],
            # Relations: signed [-1, 1] and the [0, 1] label form.
            "relation": [float(v) for v in st.R],
            "relation_01": [float(v) for v in st.relation_01()],
            "label_names": list(LABEL_NAMES),
            "label": [float(v) for v in st.label()],
            "label_mask": [bool(v) for v in st.label_mask()],
            "params": asdict(st.p),
            "formula": {
                "schema_version": self.formula_schema_version,
                "params": dict(self.base_params),
                "personality_by_role": {k: dict(v) for k, v in self.personality_by_role.items()},
            },
            "internal": {
                "S": float(st.S),
                "A": float(st.A),
                "F": float(st.F),
                "R": [float(v) for v in st.R],
                "R_active": [bool(v) for v in st.R_active],
                "last_step": st.last_step,
                "processed_event_ids": sorted(int(e) for e in st.processed),
                "last_debug": dict(st.last_debug),
            },
            "seed": self._seed,
            "rng_state": _json_safe(st.rng.bit_generator.state),
            "last_environment_step": st.last_step,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    to_dict = state_dict

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = self.from_state_dict(state)
        self.__dict__.update(restored.__dict__)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FormulaEmotionModule":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("kind") != cls.kind or state.get("schema_version") != FORMULA_STATE_SCHEMA:
            raise ValueError("not an emotion formula state")
        config = dict(state.get("config") or {})
        config["formula"] = dict(state.get("formula") or {})
        module = cls(config, player_idx=int(state["player_idx"]), seed=state.get("seed"))
        module.episode_id = state.get("episode_id")
        module.episode_seed = state.get("episode_seed")
        module.terminated = bool(state.get("terminated", False))
        module.truncated = bool(state.get("truncated", False))
        role = state.get("self_role")
        st = module._state
        if role is not None:
            module.self_role = int(role)
            st.p = EmotionParams.for_role(module.base_params, module.personality_by_role, int(role))
        internal = dict(state.get("internal") or {})
        st.S = float(internal.get("S", st.S))
        st.A = float(internal.get("A", st.A))
        st.F = float(internal.get("F", st.F))
        st.R = np.asarray(internal.get("R", st.R), dtype=np.float64)
        st.R_active = np.asarray(internal.get("R_active", st.R_active), dtype=bool)
        last = internal.get("last_step")
        st.last_step = None if last is None else int(last)
        st.processed = {int(e) for e in internal.get("processed_event_ids", [])}
        st.last_debug = dict(internal.get("last_debug", {}))
        rng_state = state.get("rng_state")
        if isinstance(rng_state, Mapping):
            st.rng.bit_generator.state = dict(rng_state)
        return module

    @staticmethod
    def is_formula_state(state: Any) -> bool:
        return isinstance(state, Mapping) and state.get("kind") == FormulaEmotionModule.kind


__all__ = [
    "EMOTION_NAMES",
    "LABEL_NAMES",
    "EmotionParams",
    "EmotionState",
    "FormulaEmotionModule",
    "ally_prob",
    "control",
    "formula_block",
    "max_health",
    "validate_formula_block",
]
