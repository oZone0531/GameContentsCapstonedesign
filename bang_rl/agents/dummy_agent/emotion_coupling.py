"""Coupling the emotion formula to a learned (non rule-based) intent policy.

The rule-based dummy picks ``argmax(intent_scores + emotion_contribution)``.
The emotion contribution is calibrated in *rule-score units* (lambda_E and
kappa were tuned against intent scores whose typical gap is ~2). A learned
policy (e.g. a reinforcement-learning intent module) returns logits on a
different, policy-specific scale, so adding the same contribution would make
emotion either negligible or dominant.

The coupling therefore keeps the emotion formula unchanged and adds one
conversion factor per policy:

    scores = policy_logits + emotion_scale * emotion_contribution

``emotion_scale`` is chosen so that emotion changes the same share of
decisions as it does for the rule-based dummy (the "changed by emotion" rate,
about 9% with the tuned formula once random tie-breaks are excluded): same emotional influence, different policy.

Plugging a policy in
--------------------
``DummyAgent.intent_source`` takes any callable

    source(observation, policy_mask, agent, rule_scores) -> 6 scores

(observation = the dummy's own public observation; rule_scores = what the rule
module would have scored, for policies that refine it). Ready-made sources:

``RuleSoftmaxSource(T)``
    log-softmax of the rule scores at temperature T. Same preferences as the
    rule dummy on a different scale: a sanity check for the calibration (the
    right scale is known to be about 1/T).
``IntentModelSource(path)``
    the dummy intent policy learned by reinforcement learning (.intentmodel,
    train_intent.py / the GUI intent training screen): the intended use.
``BangModelSource(path)``
    a trained main-agent policy (.bangmodel) run on the dummy's own public
    observation (another learned policy, mainly for testing).

Calibration: ``calibrate_scale`` (or the ``calibrate_emotion_coupling.py``
script) plays games with every dummy using the source, records each decision,
solves for the scale offline (kept inside the bracket of scales already
played) and replays, until the changed-by-emotion rate matches the rule-based
dummy's.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

IntentSource = Callable[[Any, np.ndarray, Any, np.ndarray], Any]

ROLE_NAMES = ("sheriff", "deputy", "outlaw", "outlaw", "traitor")


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def _log_softmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, -1e9, dtype=np.float64)
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return out
    part = values[valid] - float(np.max(values[valid]))
    out[valid] = part - math.log(float(np.exp(part).sum()))
    return out


class RuleSoftmaxSource:
    """Rule preferences rescaled to log-probabilities at temperature T."""

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def __call__(self, observation: Any, policy_mask: Any, agent: Any, rule_scores: Any) -> np.ndarray:
        scores = np.asarray(rule_scores, dtype=np.float64) / self.temperature
        return _log_softmax(scores, np.asarray(policy_mask) > 0)

    def describe(self) -> str:
        return f"rule-softmax:{self.temperature:g}"


class BangModelSource:
    """A trained main-agent policy (.bangmodel) used as a dummy's intent policy."""

    def __init__(self, path: str) -> None:
        import torch  # noqa: F401  (fail early if torch is missing)

        from bang_rl.training.final_inference_artifact import load_final_inference_artifact

        self.path = str(path)
        self.artifact = load_final_inference_artifact(path)

    def _inputs(self, observation: Any, seat: int) -> dict[str, Any]:
        import torch

        art = self.artifact
        processed = art.preprocessor.preprocess(observation)
        with torch.inference_mode():
            raw = art.intent.predict_proba(processed)
            emotion = art.emotion.predict_proba(processed)
        raw = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
        emotion = emotion.detach().cpu().numpy() if torch.is_tensor(emotion) else np.asarray(emotion)
        values = {key: observation[key] for key in observation.keys()}
        fused = art.role_fusion.fuse(
            raw,
            role_visibility=values["role_visibility"],
            revealed_roles=values["revealed_roles"],
            confirmed_role_distribution=values["confirmed_role_distribution"],
            self_role=values["self_role"],
            sheriff_idx=values["sheriff_idx"],
            viewer_idx=seat,
        )
        return {
            "self_state": torch.as_tensor(np.asarray(processed.state_vector, dtype=np.float32)),
            "board_state": torch.as_tensor(np.asarray(processed.player_state_matrix, dtype=np.float32)),
            "confirmed_role_features": torch.as_tensor(np.asarray(processed.role_beliefs, dtype=np.float32)),
            "fused_intent": torch.as_tensor(np.asarray(fused, dtype=np.float32)),
            "emotion_features": torch.as_tensor(emotion.reshape(-1).astype(np.float32)),
            "action_mask": torch.as_tensor(np.asarray(processed.action_mask, dtype=np.float32)),
        }

    def __call__(self, observation: Any, policy_mask: Any, agent: Any, rule_scores: Any) -> np.ndarray:
        import torch

        mask = np.asarray(policy_mask, dtype=np.uint8)
        inputs = self._inputs(observation, int(agent.emotion_module.player_idx))
        with torch.inference_mode():
            result = self.artifact.ppo.forward(inputs, torch.as_tensor(mask))
        logits = result.logits.detach().cpu().numpy().reshape(-1).astype(np.float64)
        return _log_softmax(logits, mask > 0)

    def describe(self) -> str:
        return f"bangmodel:{self.path}"


class IntentModelSource:
    """A learned dummy intent policy (.intentmodel from train_intent.py / the GUI).

    It sees the same inputs as the learned intent agent: the dummy's own public
    observation and its role beliefs (``agent.role_beliefs``, i.e. the rule
    estimator or the configured belief source).
    """

    def __init__(self, path: str) -> None:
        from bang_rl.intent_training import LearnedIntentPolicy

        self.path = str(path)
        self.policy = LearnedIntentPolicy.load(path)

    def __call__(self, observation: Any, policy_mask: Any, agent: Any, rule_scores: Any) -> np.ndarray:
        mask = np.asarray(policy_mask) > 0
        logits = np.asarray(self.policy.action_logits(observation, agent.role_beliefs), dtype=np.float64)
        return _log_softmax(logits.reshape(-1), mask)

    def describe(self) -> str:
        return f"intentmodel:{self.path}"


def make_intent_source(spec: str | None) -> Any | None:
    """``None``/``rule`` -> None, ``rule-softmax:T``, ``bangmodel:PATH``, ``intentmodel:PATH``."""

    if spec is None or spec in ("", "rule"):
        return None
    kind, _, arg = spec.partition(":")
    if kind == "rule-softmax":
        return RuleSoftmaxSource(float(arg or 1.0))
    if kind == "bangmodel":
        if not arg:
            raise ValueError("bangmodel:PATH needs a path")
        return BangModelSource(arg)
    if kind == "intentmodel":
        if not arg:
            raise ValueError("intentmodel:PATH needs a path")
        return IntentModelSource(arg)
    raise ValueError(
        f"unknown intent source {spec!r}; use rule, rule-softmax:T, bangmodel:PATH or intentmodel:PATH"
    )


# ---------------------------------------------------------------------------
# measuring
# ---------------------------------------------------------------------------
@dataclass
class CouplingStats:
    """Behaviour of dummies with one (source, scale) setting."""

    scale: float
    games: int = 0
    decisions: int = 0
    changed: int = 0
    passes: int = 0
    friendly: int = 0
    wins: Counter = field(default_factory=Counter)

    @property
    def changed_rate(self) -> float:
        return self.changed / max(self.decisions, 1)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["wins"] = {k: round(v / max(self.games, 1), 3) for k, v in sorted(self.wins.items())}
        out["changed_by_emotion"] = round(self.changed_rate, 3)
        out["pass_rate"] = round(self.passes / max(self.decisions, 1), 3)
        out["friendly_fire"] = round(self.friendly / max(self.decisions, 1), 3)
        for key in ("changed", "passes", "friendly"):
            out.pop(key)
        return out


def _same_team(a: int, b: int) -> bool:
    return (a < 2 and b < 2) or (a in (2, 3) and b in (2, 3))


def _best_actions(values: Sequence[float], mask: Sequence[int], tol: float = 1e-9) -> set[int]:
    """Every valid action tied for the top score (ties are broken at random)."""

    valid = [index for index, ok in enumerate(mask) if ok]
    if not valid:
        return set()
    top = max(float(values[index]) for index in valid)
    return {index for index in valid if float(values[index]) >= top - tol}


def measure(
    config: Any,
    source: Any | None,
    scale: float,
    *,
    games: int,
    seed0: int = 0,
    seats: Sequence[int] = (0, 1, 2, 3, 4),
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> CouplingStats:
    """Play ``games`` all-dummy games; ``seats`` use ``source`` with ``scale``.

    "Changed by emotion" counts decisions of those seats whose chosen action
    is not one of the best valid actions of the scores alone (exact ties are
    broken at random, so picking any tied best action is not a change).

    ``records`` (optional list) receives (scores, unscaled emotion
    contribution, mask) for every counted decision, for offline_scale().
    """

    from bang_rl.agents import DummyAgent
    from bang_rl.data.game_collection import build_game

    stats = CouplingStats(scale=float(scale))
    chosen = set(int(seat) for seat in seats)
    for index in range(games):
        env, agents, _ = build_game(config, seed=seed0 + index)
        for seat, agent in enumerate(agents):
            if seat in chosen:
                agent.intent_source = source
                agent.emotion_scale = float(scale)
        roles = [int(r) for r in env.state.roles]
        while not (env.state.terminated or env.state.truncated):
            actor = int(env.state.agent_selection)
            if not bool(env.state.alive[actor]):
                env.step(None)
                continue
            for other, agent in enumerate(agents):
                if (
                    other != actor
                    and isinstance(agent, DummyAgent)
                    and getattr(agent, "uses_emotion_formula", True)
                    and bool(env.state.alive[other])
                ):
                    agent.observe_public_boundary(env.observe(other))
            observation = env.observe()
            mask = np.asarray(env.action_mask, dtype=np.uint8).copy()
            action = agents[actor].act(observation, mask)
            info = agents[actor].last_decision_info
            if actor in chosen and info is not None:
                stats.decisions += 1
                # Changed = the chosen action is not among the policy's own
                # best actions (a random pick among tied best actions is not
                # an emotion effect).
                stats.changed += int(info.action not in _best_actions(info.intent_scores, info.action_mask))
                stats.passes += int(info.action == 0)
                if records is not None and scale > 0:
                    records.append(
                        (
                            np.asarray(info.intent_scores, dtype=np.float64),
                            np.asarray(info.emotion_contribution, dtype=np.float64) / float(scale),
                            np.asarray(info.action_mask, dtype=np.uint8),
                        )
                    )
                if info.action > 0:
                    stats.friendly += int(_same_team(roles[actor], roles[info.action - 1]))
            env.step(env.validate_action(action))
        stats.games += 1
        result = env.state.terminal_result
        team = "TRUNCATED" if result is None or result.winning_team is None else result.winning_team.name
        stats.wins[team] += 1
    return stats


def changed_share(records: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], scale: float) -> float:
    """Share of recorded decisions whose best action would change at ``scale``."""

    if not records:
        return 0.0
    changed = 0
    for scores, emotion, mask in records:
        best = _best_actions(scores, mask)
        combined = scores + scale * emotion
        top = _best_actions(combined, mask)
        changed += int(not (top & best))
    return changed / len(records)


def offline_scale(
    records: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    target: float,
    *,
    low: float = 1e-4,
    high: float = 1e4,
    iterations: int = 40,
) -> float:
    """Scale where changed_share(records, scale) ~= target (log bisection, no replay)."""

    lo, hi = math.log(low), math.log(high)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if changed_share(records, math.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    return math.exp((lo + hi) / 2)


def calibrate_scale(
    config: Any,
    source: Any,
    *,
    target: float,
    games: int = 60,
    seed0: int = 0,
    start: float = 1.0,
    rounds: int = 8,
    tolerance: float = 0.01,
    log: Callable[[str], None] | None = None,
) -> tuple[float, list[CouplingStats]]:
    """Find the scale whose played changed-by-emotion rate ~= target.

    Each round plays ``games`` games at the current scale and records every
    decision (policy scores, emotion contribution, mask). The next scale is
    solved offline on those records (no replay per candidate), but kept inside
    the bracket of scales already played below/above the target: emotion also
    changes the games themselves (a strongly emotional table plays different,
    shorter games), so the offline answer can overshoot. When it falls outside
    the bracket the geometric middle of the bracket is used instead. The same
    seeds are used every round. Returns (scale, every played measurement).
    """

    history: list[CouplingStats] = []
    scale = float(start)
    below: float | None = None  # largest scale played with rate < target
    above: float | None = None  # smallest scale played with rate > target
    best: CouplingStats | None = None
    for _ in range(rounds):
        records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        stats = measure(config, source, scale, games=games, seed0=seed0, records=records)
        history.append(stats)
        if log:
            log(f"  played scale {scale:9.4f} -> changed {stats.changed_rate:.3f}")
        if best is None or abs(stats.changed_rate - target) < abs(best.changed_rate - target):
            best = stats
        if abs(stats.changed_rate - target) <= tolerance:
            break
        if stats.changed_rate < target:
            below = scale if below is None else max(below, scale)
        else:
            above = scale if above is None else min(above, scale)
        guess = offline_scale(records, target)
        low = below if below is not None else 0.0
        high = above if above is not None else math.inf
        if not (low < guess < high):
            if below is not None and above is not None:
                guess = math.sqrt(below * above)
            elif above is not None:
                guess = above / 4.0
            else:
                guess = (below or 1.0) * 4.0
        elif below is not None and above is not None:
            # Inside the bracket: do not hug an edge (avoid tiny steps).
            lo, hi = math.log(below), math.log(above)
            g = math.log(guess)
            guess = math.exp(min(max(g, lo + 0.2 * (hi - lo)), hi - 0.2 * (hi - lo)))
        scale = guess
        if log:
            log(f"  next scale {scale:.4f} (offline estimate on {len(records)} decisions, bracket {below} .. {above})")
    assert best is not None
    return best.scale, history


__all__ = [
    "BangModelSource",
    "CouplingStats",
    "IntentModelSource",
    "RuleSoftmaxSource",
    "calibrate_scale",
    "changed_share",
    "make_intent_source",
    "measure",
    "offline_scale",
]
