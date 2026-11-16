"""Optional headless training entry point; the GUI uses the same trainer.

Example: python train_intent.py --games-per-role 100 --max-steps 200
"""
from __future__ import annotations

import argparse

from bang_rl.intent_training import IntentTrainer, IntentTrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Train independent role-balanced intent policies")
    parser.add_argument("--games-per-role", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--random-ratio", type=float, default=0.35)
    parser.add_argument("--batch-games", type=int, default=16)
    parser.add_argument("--eval-games-per-role", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--output", default="bang_rl/models/intent/trained.intentmodel")
    args = parser.parse_args()
    cfg = IntentTrainingConfig(
        episodes_per_role=args.games_per_role,
        max_steps=args.max_steps,
        seed=args.seed,
        random_opponent_probability=args.random_ratio,
        batch_games=args.batch_games,
        eval_games_per_role=args.eval_games_per_role,
        checkpoint_every=args.checkpoint_every,
        output_path=args.output,
    )
    def report(event: dict) -> None:
        phase = event.get("phase")
        if phase == "training" and (event["completed"] % max(1, event["total"] // 20) == 0 or event["completed"] == event["total"]):
            print(f"Progress {event['completed']}/{event['total']}  role={event['role']}  updates={event['updates']}", flush=True)
        elif phase == "checkpoint":
            print("Checkpoint:", event["path"], flush=True)
        elif phase == "baseline":
            print("Evaluating rule baseline...", flush=True)
        elif phase == "done":
            print("Saved model:", event["path"], "Report:", event["report_path"], flush=True)
    result = IntentTrainer(cfg, callback=report).run()
    for role, before in result["baseline_evaluation"].items():
        after = result["learned_evaluation"][role]
        def fmt(x):
            return "N/A" if x is None else f"{x:.1%}"
        print(f"{role}: baseline={fmt(before['win_rate'])} trained={fmt(after['win_rate'])}")
    for label, key in (("old_five", "all_five_rule_evaluation"),
                       ("learned_five", "all_five_learned_evaluation")):
        result5 = result[key]
        print(label, "traitor_wins=", result5["winning_teams"]["TRAITOR_TEAM"],
              "rate=", result5["team_win_rates"]["TRAITOR_TEAM"],
              "truncated=", result5["truncated"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
