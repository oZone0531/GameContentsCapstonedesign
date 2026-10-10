"""Measure how the emotion formula changes dummy behaviour inside the real game.

Usage (from the repository root, next to the bang_rl folder):
    python measure_emotion.py --games 200
    python measure_emotion.py --games 200 --emotion-file emotion_formula.example.json
    python measure_emotion.py --games 3 --save-replays out_replays

Every game is played through bang_rl.app.GameService in "dummy_five" mode, so the
numbers come from the same code path the GUI uses. "Changed by emotion" compares
each decision with the action the intent scores alone would pick (deterministic
argmax over legal actions); the intent scores are read from the dummy's own
last_decision record.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from bang_rl.agents import DummyAgent  # noqa: E402
from bang_rl.agents.dummy_agent.intent_policy import ActionSelector  # noqa: E402
from bang_rl.app import GameService, SessionConfig  # noqa: E402
from bang_rl.config import SystemConfig  # noqa: E402

ROLE_NAMES = ("sheriff", "deputy", "outlaw_a", "outlaw_b", "traitor")
EMOTIONS = ["calm", "confidence", "aggression", "anxiety"]


def same_team(a: int, b: int) -> bool:
    return (a < 2 and b < 2) or (a in (2, 3) and b in (2, 3))


def emotion_config(emotion_file: str | None, overrides: dict) -> dict:
    if emotion_file:
        block = json.loads(Path(emotion_file).read_text(encoding="utf-8"))
    else:
        block = {"emotions": [{"name": n} for n in EMOTIONS], "formula": {"params": {}}}
    block = json.loads(json.dumps(block))
    if overrides:
        block.setdefault("formula", {}).setdefault("params", {}).update(overrides)
    return block


def run(games: int, emotion: dict | None, mode: str = "deterministic", seed0: int = 0) -> dict:
    cfg = SystemConfig.model_validate({"preset": {"emotion": emotion}} if emotion else {})
    svc = GameService()
    dec = changed = passes = friendly = 0
    by_role = defaultdict(lambda: [0, 0])
    wins: Counter = Counter()
    lengths = []
    values = []
    for g in range(games):
        h = svc.create_session(SessionConfig(mode="dummy_five", config=cfg, seed=seed0 + g, action_mode=mode))
        sess = svc._session(h)
        env = sess.environment
        roles = [int(r) for r in env.state.roles]
        while not (env.state.terminated or env.state.truncated):
            actor = int(env.state.agent_selection)
            if not env.state.alive[actor]:
                svc.step(h)
                continue
            agent = sess.agents[actor]
            svc.step(h)
            sel = agent.last_decision_info if isinstance(agent, DummyAgent) else None
            if sel is None:
                continue
            base = ActionSelector.select_with_info(
                np.asarray(sel.intent_scores), np.asarray(sel.action_mask), mode="deterministic"
            ).action
            dec += 1
            diff = int(sel.action != base)
            changed += diff
            by_role[roles[actor]][0] += 1
            by_role[roles[actor]][1] += diff
            passes += int(sel.action == 0)
            if sel.action > 0:
                friendly += int(same_team(roles[actor], roles[sel.action - 1]))
            for i, a in enumerate(sess.agents):
                if env.state.alive[i] and isinstance(a, DummyAgent):
                    values.append(np.asarray(a.emotion_module.values, dtype=float)[: len(EMOTIONS)])
        result = env.state.terminal_result
        wins[result.winning_team.name if result else "TRUNCATED"] += 1
        lengths.append(int(env.state.environment_step))
        svc.close_session(h)
    vals = np.asarray(values) if values else np.zeros((1, 4))
    return {
        "games": games,
        "win_rate": {k: round(v / games, 3) for k, v in sorted(wins.items())},
        "mean_steps": round(float(np.mean(lengths)), 1),
        "changed_by_emotion": round(changed / max(dec, 1), 3),
        "friendly_fire": round(friendly / max(dec, 1), 3),
        "pass": round(passes / max(dec, 1), 3),
        "decisions": dec,
        "changed_count": changed,
        "by_role": {
            ROLE_NAMES[r]: {
                "decisions": by_role[r][0],
                "changed": by_role[r][1],
                "changed_rate": round(by_role[r][1] / max(by_role[r][0], 1), 3),
                "share_of_changed": round(by_role[r][1] / max(changed, 1), 3),
            }
            for r in range(5)
        },
        "emotion_mean": dict(zip(EMOTIONS, np.round(vals.mean(axis=0), 3).tolist())),
        "emotion_p90": dict(zip(EMOTIONS, np.round(np.percentile(vals, 90, axis=0), 3).tolist())),
    }


def save_replays(games: int, emotion: dict, out_dir: Path) -> list[Path]:
    """Play spectator games through the GUI controller and save FULL replays to out_dir."""

    import bang_rl.paths as paths
    from bang_rl.gui.game import GameSessionController

    out_dir.mkdir(parents=True, exist_ok=True)
    paths.REPLAYS_DIR = out_dir          # save_replay reads this at call time
    cfg = SystemConfig.model_validate({"preset": {"emotion": emotion}})
    before = set(out_dir.glob("*.json"))
    for g in range(games):
        ctl = GameSessionController(GameService(), config=cfg, seed=1000 + g)
        ctl.start("dummy_five")
        guard = 0
        while ctl.active and ctl.is_ai_turn() and guard < 500:
            ctl.step_single_ai_turn()
            guard += 1
        ctl.close()
    return sorted(set(out_dir.glob("*.json")) - before)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--emotion-file", default=None, help="EmotionConfig JSON with a 'formula' block")
    ap.add_argument("--mode", default="deterministic", choices=["deterministic", "stochastic"])
    ap.add_argument("--save-replays", default=None, help="folder for a few full replays (spectator path)")
    ap.add_argument("--json", default=None, help="write all results to this file")
    args = ap.parse_args()

    emotion = emotion_config(args.emotion_file, {})
    off = emotion_config(args.emotion_file, {"lambda_E": 0.0, "sigma": 0.0, "sigma_R": 0.0})
    results = {
        "default_config_rule_table_module": run(args.games, None, args.mode),
        "formula_intent_only": run(args.games, off, args.mode),
        "formula_on": run(args.games, emotion, args.mode),
    }
    for name, res in results.items():
        print(name, json.dumps(res, ensure_ascii=False))
    if args.save_replays:
        saved = save_replays(3, emotion, Path(args.save_replays))
        results["replays"] = [str(p) for p in saved]
        print("saved replays:", *saved, sep="\n  ")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
