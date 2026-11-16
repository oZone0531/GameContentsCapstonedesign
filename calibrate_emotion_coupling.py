"""Find the emotion scale for a learned dummy intent policy.

Usage (from the repository root, next to the bang_rl folder):
    python calibrate_emotion_coupling.py --source rule-softmax:4 --games 150
    python calibrate_emotion_coupling.py --source bangmodel:outputs/main.bangmodel --games 150 --out coupling.json
    python calibrate_emotion_coupling.py --source bangmodel:outputs/main.bangmodel --emotion-file emotion_formula.personality.json

Steps
  1. reference: rule-based dummies (scale 1) -> their changed-by-emotion rate
     (unless --target is given);
  2. search: every dummy uses --source; play, record every decision, solve
     the emotion scale offline, replay to confirm (2-3 rounds) until the
     changed-by-emotion rate matches the reference;
  3. report: reference vs calibrated vs no-emotion (scale 0) for the source,
     with win rates, pass rate and friendly fire, on the same seeds.

Sources (see bang_rl/agents/dummy_agent/emotion_coupling.py)
  intentmodel:PATH   the learned dummy intent policy (.intentmodel, train_intent.py)
  rule-softmax:T     the rule scores on another scale (check: answer ~ 1/T)
  bangmodel:PATH     a trained main-agent policy run as a dummy policy
Any other policy plugs in the same way: a callable
(observation, policy_mask, agent, rule_scores) -> 6 scores set as
DummyAgent.intent_source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bang_rl.agents.dummy_agent.emotion_coupling import (  # noqa: E402
    calibrate_scale,
    make_intent_source,
    measure,
)
from bang_rl.data.game_collection import DEFAULT_EMOTION_FILE, load_collection_config  # noqa: E402


def _row(name: str, stats) -> str:
    d = stats.to_dict()
    wins = d["wins"]
    return (
        f"{name:<26}{d['scale']:>9.4f}{d['changed_by_emotion']:>9.3f}{d['pass_rate']:>8.3f}"
        f"{d['friendly_fire']:>8.3f}   sheriff {wins.get('SHERIFF_TEAM', 0):.3f}  "
        f"outlaw {wins.get('OUTLAW_TEAM', 0):.3f}  traitor {wins.get('TRAITOR_TEAM', 0):.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="intentmodel:PATH, rule-softmax:T or bangmodel:PATH")
    parser.add_argument("--games", type=int, default=60, help="games per measurement")
    parser.add_argument("--report-games", type=int, default=None, help="games for the final report (default: --games)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=None, help="changed-by-emotion rate to match")
    parser.add_argument("--emotion-file", default=DEFAULT_EMOTION_FILE)
    parser.add_argument("--out", default=None, help="write results as JSON here")
    args = parser.parse_args(argv)

    config = load_collection_config(emotion_file=args.emotion_file)
    source = make_intent_source(args.source)
    if source is None:
        parser.error("--source must be a learned/rescaled source, not rule")
    report_games = args.report_games or args.games

    if args.target is None:
        print(f"reference: rule-based dummies ({args.games} games)", flush=True)
        reference = measure(config, None, 1.0, games=args.games, seed0=args.seed)
        target = reference.changed_rate
    else:
        reference = None
        target = float(args.target)
    print(f"target changed-by-emotion rate: {target:.3f}", flush=True)
    print(f"searching scale for {args.source}", flush=True)
    scale, history = calibrate_scale(
        config, source, target=target, games=args.games, seed0=args.seed, log=lambda line: print(line, flush=True)
    )
    print(f"calibrated emotion_scale = {scale:.4f}", flush=True)

    print(f"\nreport ({report_games} games, same seeds)")
    rows = {}
    if reference is not None and report_games == args.games:
        rows["rule dummy (reference)"] = reference
    elif args.target is None:
        rows["rule dummy (reference)"] = measure(config, None, 1.0, games=report_games, seed0=args.seed)
    rows["source, no emotion"] = measure(config, source, 0.0, games=report_games, seed0=args.seed)
    rows["source, calibrated"] = measure(config, source, scale, games=report_games, seed0=args.seed)
    print(f"{'setting':<26}{'scale':>9}{'changed':>9}{'pass':>8}{'ff':>8}   win rates")
    for name, stats in rows.items():
        print(_row(name, stats))

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "source": args.source,
                    "emotion_file": args.emotion_file,
                    "target": target,
                    "emotion_scale": scale,
                    "search": [stats.to_dict() for stats in history],
                    "report": {name: stats.to_dict() for name, stats in rows.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
