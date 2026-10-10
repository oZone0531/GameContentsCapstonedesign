"""Show, turn by turn, how emotion changes each dummy decision in one game.

Usage (from the repository root, next to the bang_rl folder):
    python trace_emotion.py
    python trace_emotion.py --emotion-file emotion_formula.example.json --seed 3
    python trace_emotion.py --only-changed            # print only turns emotion changed
    python trace_emotion.py --out my_trace.txt        # choose the text file name

The trace is printed AND saved as a text file. By default the text file sits next
to the saved replay with the same name (bang_rl/replays/<replay>.txt), so each
replay has its own trace. The GUI replay list only shows .json files, and the
bang_rl/replays folder is ignored by git.

What it does
- Plays one "dummy_five" spectator game with the emotion formula switched on.
- For every dummy decision prints: the emotion state used, then for each action
  intent score + emotion score = total, the action intent alone would pick,
  the action actually picked, and why they differ.
- Saves the same game as a FULL replay in bang_rl/replays, so it can be opened
  in the GUI (Replay) and compared turn by turn (hover a seat for emotion bars).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from bang_rl.agents import DummyAgent  # noqa: E402
from bang_rl.agents.dummy_agent.intent_policy import ActionSelector  # noqa: E402
from bang_rl.app import GameService  # noqa: E402
from bang_rl.config import SystemConfig  # noqa: E402
from bang_rl.gui.game import GameSessionController  # noqa: E402

ROLE = ("sheriff", "deputy", "outlaw", "outlaw", "traitor")
TEAM = ("S-team", "S-team", "O-team", "O-team", "traitor")


def action_name(a: int) -> str:
    return "pass" if a == 0 else f"bang seat{a - 1}"


def why_changed(base: int, chosen: int, intent: np.ndarray, emo: np.ndarray, rel: np.ndarray, p) -> str:
    """Name the emotion term that flipped the choice."""

    intent_gap = intent[base] - intent[chosen]          # how much intent preferred `base`
    emo_gap = emo[chosen] - emo[base]                   # how much emotion preferred `chosen`
    tie = "intent tie" if abs(intent_gap) < 1e-9 else f"intent gap {intent_gap:.3f}"
    if base > 0 and chosen > 0:
        rc, rb = rel[chosen - 1], rel[base - 1]
        rel_term = -p.kappa_R * p.lambda_E * (rc - rb)
        if abs(rel_term) >= 0.02:
            feel = "grudge" if rc < rb else "liking"
            return (f"{tie}; relation decided it ({feel}: seat{chosen - 1} R={rc:+.3f} "
                    f"vs seat{base - 1} R={rb:+.3f}, +{emo_gap:.3f})")
        return f"{tie}; small noise in relation broke it (+{emo_gap:.3f})"
    if chosen == 0:
        return f"{tie}; anxiety pushed PASS over shooting (+{emo_gap:.3f})"
    return f"{tie}; confidence/aggression pushed shooting over PASS (+{emo_gap:.3f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion-file", default="emotion_formula.example.json")
    ap.add_argument("--seed", type=int, default=0, help="session seed (the game itself still varies)")
    ap.add_argument("--only-changed", action="store_true")
    ap.add_argument("--no-replay", action="store_true", help="do not save a replay")
    ap.add_argument("--out", default=None, help="text file for the trace (default: next to the replay)")
    args = ap.parse_args()

    lines: list[str] = []

    def out(*parts: object) -> None:
        text = " ".join(str(part) for part in parts)
        print(text)
        lines.append(text)

    emotion = json.loads(Path(args.emotion_file).read_text(encoding="utf-8"))
    if "formula" not in emotion:
        sys.exit("This emotion file has no 'formula' block, so the emotion formula is OFF.")
    cfg = SystemConfig.model_validate({"preset": {"emotion": emotion}})

    from bang_rl.paths import REPLAYS_DIR
    before = set(Path(REPLAYS_DIR).glob("*.json")) if Path(REPLAYS_DIR).exists() else set()

    ctl = GameSessionController(GameService(), config=cfg, seed=args.seed)
    ctl.start("dummy_five")
    if args.no_replay:
        ctl._replay_saved = True  # the controller skips its end-of-game save when this is set
    session = ctl.service._session(ctl.handle)
    env = session.environment
    roles = [int(r) for r in env.state.roles]
    out("Roles (hidden in game):", ", ".join(f"seat{i}={ROLE[r]}" for i, r in enumerate(roles)))
    out("Emotion module per seat:", ", ".join(type(a.emotion_module).__name__ for a in session.agents))
    out()

    decisions = changed = 0
    guard = 0
    while ctl.active and ctl.is_ai_turn() and guard < 500:
        guard += 1
        actor = int(env.state.agent_selection)
        hp_before = [int(h) for h in env.state.health]
        step = int(env.state.environment_step)
        ctl.step_single_ai_turn()
        agent = session.agents[actor]
        sel = agent.last_decision_info if isinstance(agent, DummyAgent) else None
        if sel is None:
            continue
        intent = np.asarray(sel.intent_scores, dtype=float)
        emo = np.asarray(sel.emotion_contribution, dtype=float)
        total = np.asarray(sel.combined_scores, dtype=float)
        mask = np.asarray(sel.action_mask)
        base = ActionSelector.select_with_info(intent, mask, mode="deterministic").action
        chosen = int(sel.action)
        decisions += 1
        flipped = chosen != base
        changed += int(flipped)
        if args.only_changed and not flipped:
            continue
        module = agent.emotion_module
        P, S, A, F = (float(v) for v in module.values)
        rel = module.relation
        params = module.params
        tag = "  <== CHANGED BY EMOTION" if flipped else ""
        out(f"step {step:>2} | seat{actor} {ROLE[roles[actor]]:<7} | HP {hp_before}{tag}")
        out(f"  emotion  calm {P:.2f}  confidence {S:.2f}  aggression {A:.2f}  anxiety {F:.2f}")
        out("  relation " + "  ".join(
            f"seat{j} {rel[j]:+.3f}" for j in range(5) if j != actor))
        out("  action        intent   +emotion   =total")
        for a in range(6):
            if not mask[a]:
                continue
            marks = ("  <- intent alone" if a == base else "") + ("  <- picked" if a == chosen else "")
            target = "" if a == 0 else f" ({TEAM[roles[a - 1]]})"
            out(f"  {action_name(a):<11}{target:<10} {intent[a]:7.3f}   {emo[a]:+7.3f}   {total[a]:7.3f}{marks}")
        if flipped:
            out("  why: " + why_changed(base, chosen, intent, emo, rel, params))
        out()

    st = env.state
    winner = st.terminal_result.winning_team.name if st.terminal_result else "TRUNCATED"
    out(f"Game over at step {st.environment_step}: {winner}. "
          f"Emotion changed {changed} of {decisions} decisions ({changed / max(decisions, 1):.0%}).")
    target = Path(args.out) if args.out else None
    if not args.no_replay:
        after = set(Path(REPLAYS_DIR).glob("*.json")) if Path(REPLAYS_DIR).exists() else set()
        new = sorted(after - before)
        if new:
            out(f"Replay saved: {new[-1]}  (GUI > Replay, hover a seat to see its emotion bars)")
            if target is None:
                target = new[-1].with_suffix(".txt")
    if target is None:
        Path(REPLAYS_DIR).mkdir(parents=True, exist_ok=True)
        target = Path(REPLAYS_DIR) / f"emotion-trace-{datetime.now():%Y%m%d-%H%M%S}.txt"
    out(f"Trace saved: {target}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
