"""Compare the main agent against baselines on the same evaluation games.

Usage (from the repository root, next to the bang_rl folder):
    python evaluate_baselines.py --games 500
    python evaluate_baselines.py --games 500 --artifact outputs/final.bangmodel
    python evaluate_baselines.py --games 500 --baselines random,dummy,hint:0.7 --out eval.json

Every row plays the same games: seed = --seed + i, seat cycles 0..4, the other
four seats are emotion-formula dummies. Roles depend only on the seed, so game i
deals the same roles to every row. Rows:
    random    random legal actions (floor)
    dummy     the rule-based dummy itself (what the main agent has to beat)
    hint:Q    a dummy with a privileged role hint (reference ceiling, cheats)
    main      the --artifact model (role accuracy and emotion error included)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bang_rl.data.game_collection import DEFAULT_EMOTION_FILE, load_collection_config  # noqa: E402
from bang_rl.evaluation import (  # noqa: E402
    CustomEvaluationProtocol,
    baseline_factory,
    run_seat_evaluation,
    summarize_protocol_report,
)
from bang_rl.config import EvaluationConfig  # noqa: E402

ROLE_ORDER = ("sheriff", "deputy", "outlaw", "traitor")


def _percent(value) -> str:
    return "  -  " if value is None else f"{100 * value:5.1f}"


def _row(result: dict) -> str:
    ci = result.get("win_rate_ci") or {}
    interval = f"[{_percent(ci.get('lower'))},{_percent(ci.get('upper'))}]" if ci else ""
    roles = " ".join(_percent(result["by_role"][name]["win_rate"]) for name in ROLE_ORDER)
    extra = ""
    if "role_accuracy" in result:
        extra = (
            f"  role acc {_percent(result.get('role_accuracy'))}"
            f"  hidden acc {_percent(result.get('hidden_role_accuracy'))}"
            f"  emotion mse {result.get('emotion_mse')}"
        )
    return f"{result['agent']:<12} {_percent(result['win_rate'])} {interval:<16} {roles}{extra}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Main agent vs baselines on identical evaluation games.")
    parser.add_argument("--games", type=int, default=500, help="games per row (default 500)")
    parser.add_argument("--seed", type=int, default=0, help="base seed; game i uses seed + i")
    parser.add_argument("--baselines", default="random,dummy", help="comma list: random, dummy, hint:Q, oracle:A")
    parser.add_argument("--artifact", default=None, help="trained main agent (.bangmodel) to add as a row")
    parser.add_argument("--config", default=None, help="optional SystemConfig file (.yaml/.json)")
    parser.add_argument("--emotion-file", default=DEFAULT_EMOTION_FILE, help="dummy emotion formula file")
    parser.add_argument("--out", default=None, help="write all results as JSON here")
    args = parser.parse_args(argv)

    config = load_collection_config(args.config, emotion_file=args.emotion_file)
    results = []
    for name in [item.strip() for item in args.baselines.split(",") if item.strip()]:
        print(f"evaluating {name} ({args.games} games)...", flush=True)
        results.append(
            run_seat_evaluation(
                baseline_factory(name, config), config, games=args.games, base_seed=args.seed, label=name
            )
        )
    if args.artifact:
        print(f"evaluating main agent {args.artifact} ({args.games} games)...", flush=True)
        evaluation = EvaluationConfig(
            protocol_id="custom_v1",
            games=args.games,
            games_per_seat=max(1, args.games // 5),
            base_seed=args.seed,
        )
        report = CustomEvaluationProtocol(args.artifact, config=config).run(evaluation)
        results.append(summarize_protocol_report(report, label="main"))

    print()
    print(f"{'agent':<12} {'win%':>5} {'95% CI':<16} " + " ".join(f"{name[:5]:>5}" for name in ROLE_ORDER))
    for result in results:
        print(_row(result))
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
