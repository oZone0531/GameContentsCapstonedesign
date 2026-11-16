"""Measure how emotion personality presets change dummy emotions and behaviour.

Usage (from the repository root, next to the bang_rl folder):
    python measure_personality.py --games 500
    python measure_personality.py --games 500 --emotion-file emotion_formula.personality.json
    python measure_personality.py --games 500 --compare emotion_formula.example.json --out personality.json

Games run through bang_rl.app.GameService in "dummy_five" mode (the GUI code
path), so seeds, update order and emotion handling match the GUI. Every number
is grouped by the personality preset of the dummy it belongs to:

  emotion_mean / emotion_p90   global emotions of living dummies, every step
  changed_by_emotion           decisions that differ from the intent-only argmax
  pass / friendly_fire         share of that preset's decisions
  team_win / survived          share of that preset's seats whose team won / who lived
  eta_squared                  share of each emotion's variance explained by the
                               preset (0 = presets do not change it, 1 = fully)

With --compare, the same seeds are played with a second emotion file (usually
the presets-off file) and both results are printed side by side.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bang_rl.agents import DummyAgent  # noqa: E402
from bang_rl.agents.dummy_agent.intent_policy import ActionSelector  # noqa: E402
from bang_rl.app import GameService, SessionConfig  # noqa: E402
from bang_rl.config import SystemConfig  # noqa: E402

EMOTIONS = ("calm", "confidence", "aggression", "anxiety")
ROLE_NAMES = ("sheriff", "deputy", "outlaw_a", "outlaw_b", "traitor")
NO_PRESET = "(none)"


def same_team(a: int, b: int) -> bool:
    return (a < 2 and b < 2) or (a in (2, 3) and b in (2, 3))


def load_emotion(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(centre - half, 3), round(centre + half, 3)]


def eta_squared(values: np.ndarray, groups: list[str]) -> list[float]:
    """Between-group share of total variance, per column."""

    if len(values) == 0 or len(set(groups)) < 2:
        return [0.0] * values.shape[1] if values.ndim == 2 else []
    labels = np.asarray(groups)
    total = ((values - values.mean(axis=0)) ** 2).sum(axis=0)
    between = np.zeros(values.shape[1])
    for name in set(groups):
        part = values[labels == name]
        between += len(part) * (part.mean(axis=0) - values.mean(axis=0)) ** 2
    return [round(float(b / t), 3) if t > 0 else 0.0 for b, t in zip(between, total)]


def personality_of(agent: object) -> str:
    module = getattr(agent, "emotion_module", None)
    name = getattr(module, "personality", None)
    return NO_PRESET if name is None else str(name)


def run(games: int, emotion: dict, *, seed0: int = 0, mode: str = "deterministic") -> dict:
    cfg = SystemConfig.model_validate({"preset": {"emotion": emotion}})
    svc = GameService()
    wins: Counter = Counter()
    lengths: list[int] = []
    decisions: dict[str, Counter] = defaultdict(Counter)
    seats: dict[str, Counter] = defaultdict(Counter)
    values: list[np.ndarray] = []
    groups: list[str] = []
    assignment: Counter = Counter()
    by_role: dict[str, Counter] = defaultdict(Counter)
    for g in range(games):
        handle = svc.create_session(
            SessionConfig(mode="dummy_five", config=cfg, seed=seed0 + g, action_mode=mode)
        )
        session = svc._session(handle)
        env = session.environment
        roles = [int(r) for r in env.state.roles]
        names = [personality_of(agent) for agent in session.agents]
        for seat, name in enumerate(names):
            assignment[name] += 1
            by_role[ROLE_NAMES[roles[seat]]][name] += 1
        while not (env.state.terminated or env.state.truncated):
            actor = int(env.state.agent_selection)
            if not env.state.alive[actor]:
                svc.step(handle)
                continue
            agent = session.agents[actor]
            svc.step(handle)
            info = agent.last_decision_info if isinstance(agent, DummyAgent) else None
            if info is not None:
                base = ActionSelector.select_with_info(
                    np.asarray(info.intent_scores), np.asarray(info.action_mask), mode="deterministic"
                ).action
                counter = decisions[names[actor]]
                counter["decisions"] += 1
                counter["changed"] += int(info.action != base)
                counter["pass"] += int(info.action == 0)
                if info.action > 0:
                    counter["friendly"] += int(same_team(roles[actor], roles[info.action - 1]))
            for seat, seat_agent in enumerate(session.agents):
                if env.state.alive[seat] and isinstance(seat_agent, DummyAgent):
                    values.append(np.asarray(seat_agent.emotion_module.values, dtype=float)[: len(EMOTIONS)])
                    groups.append(names[seat])
        result = env.state.terminal_result
        wins[result.winning_team.name if result is not None and result.winning_team is not None else "TRUNCATED"] += 1
        winners = set(result.winners) if result is not None else set()
        for seat, name in enumerate(names):
            seats[name]["seats"] += 1
            seats[name]["team_win"] += int(seat in winners)
            seats[name]["survived"] += int(bool(env.state.alive[seat]))
        lengths.append(int(env.state.environment_step))
        svc.close_session(handle)

    vals = np.asarray(values) if values else np.zeros((0, len(EMOTIONS)))
    labels = np.asarray(groups)
    presets = {}
    for name in sorted(set(groups) | set(seats)):
        part = vals[labels == name] if len(vals) else np.zeros((0, len(EMOTIONS)))
        d, s = decisions[name], seats[name]
        presets[name] = {
            "seats": s["seats"],
            "decisions": d["decisions"],
            "changed_by_emotion": round(d["changed"] / max(d["decisions"], 1), 3),
            "pass": round(d["pass"] / max(d["decisions"], 1), 3),
            "friendly_fire": round(d["friendly"] / max(d["decisions"], 1), 3),
            "team_win": round(s["team_win"] / max(s["seats"], 1), 3),
            "team_win_ci95": wilson(s["team_win"], s["seats"]),
            "survived": round(s["survived"] / max(s["seats"], 1), 3),
            "emotion_mean": dict(zip(EMOTIONS, np.round(part.mean(axis=0), 3).tolist())) if len(part) else {},
            "emotion_p90": dict(zip(EMOTIONS, np.round(np.percentile(part, 90, axis=0), 3).tolist())) if len(part) else {},
        }
    total = Counter()
    for counter in decisions.values():
        total.update(counter)
    return {
        "games": games,
        "seed0": seed0,
        "win_rate": {k: round(v / games, 3) for k, v in sorted(wins.items())},
        "win_ci95": {k: wilson(v, games) for k, v in sorted(wins.items())},
        "mean_steps": round(float(np.mean(lengths)), 1) if lengths else 0.0,
        "changed_by_emotion": round(total["changed"] / max(total["decisions"], 1), 3),
        "friendly_fire": round(total["friendly"] / max(total["decisions"], 1), 3),
        "pass": round(total["pass"] / max(total["decisions"], 1), 3),
        "emotion_mean": dict(zip(EMOTIONS, np.round(vals.mean(axis=0), 3).tolist())) if len(vals) else {},
        "emotion_p90": dict(zip(EMOTIONS, np.round(np.percentile(vals, 90, axis=0), 3).tolist())) if len(vals) else {},
        "eta_squared": dict(zip(EMOTIONS, eta_squared(vals, groups))),
        "assignment": dict(sorted(assignment.items())),
        "assignment_by_role": {role: dict(sorted(c.items())) for role, c in sorted(by_role.items())},
        "by_personality": presets,
    }


def print_report(title: str, result: dict) -> None:
    print(f"\n== {title} ({result['games']} games)")
    print("win rate:", result["win_rate"], " mean steps:", result["mean_steps"])
    print(
        f"changed by emotion {result['changed_by_emotion']:.3f}  pass {result['pass']:.3f}  "
        f"friendly fire {result['friendly_fire']:.3f}"
    )
    print("variance explained by preset (eta^2):", result["eta_squared"])
    header = f"{'preset':<10}{'seats':>6}{'chg':>7}{'pass':>7}{'ff':>7}{'win':>7}{'alive':>7}  " + "  ".join(
        f"{e[:5]:>11}" for e in EMOTIONS
    )
    print(header)
    for name, row in result["by_personality"].items():
        means = row["emotion_mean"]
        p90 = row["emotion_p90"]
        cells = "  ".join(f"{means.get(e, 0):.2f}/{p90.get(e, 0):.2f}".rjust(11) for e in EMOTIONS)
        print(
            f"{name:<10}{row['seats']:>6}{row['changed_by_emotion']:>7.3f}{row['pass']:>7.3f}"
            f"{row['friendly_fire']:>7.3f}{row['team_win']:>7.3f}{row['survived']:>7.3f}  {cells}"
        )
    print("(emotion cells: mean/p90)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0, help="base seed; game i uses seed + i")
    parser.add_argument("--emotion-file", default="emotion_formula.personality.json")
    parser.add_argument("--compare", default=None, help="second emotion file played on the same seeds")
    parser.add_argument("--mode", default="deterministic", choices=("deterministic", "stochastic"))
    parser.add_argument("--out", default=None, help="write results as JSON here")
    args = parser.parse_args(argv)

    results = {"main": {"emotion_file": args.emotion_file,
                        **run(args.games, load_emotion(args.emotion_file), seed0=args.seed, mode=args.mode)}}
    print_report(args.emotion_file, results["main"])
    if args.compare:
        results["compare"] = {"emotion_file": args.compare,
                              **run(args.games, load_emotion(args.compare), seed0=args.seed, mode=args.mode)}
        print_report(args.compare, results["compare"])
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
