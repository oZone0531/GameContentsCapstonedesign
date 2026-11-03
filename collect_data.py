"""Collect a canonical training dataset from all-dummy games (Phase 1).

Usage (from the repository root, next to the bang_rl folder):
    python collect_data.py --games 1000 --out bang_rl/datasets/canonical/formula_v1
    python collect_data.py --games 1000 --out <folder> --emotion-file my_formula.json
    python collect_data.py --games 1000 --out <folder> --rule-table
    python collect_data.py --games 1000 --out <folder> --config config.example.yaml
    python collect_data.py --games 1000 --out <folder> --mode stochastic --seed 5000

By default the dummies run the emotion formula from emotion_formula.example.json,
so every step carries the 9-entry emotion label
[calm, confidence, aggression, anxiety, relation_0..relation_4] for every living
dummy. --rule-table keeps the emotion block of the loaded config instead.

Game i uses seed (--seed + i); the same seed always replays the same game.
The output folder gets the Parquet shard(s) plus collection_summary.json and
must be new or empty.

Next step (Phase 2): split -> NPZ tensor cache -> supervised training. The main
agent's emotion inference must use the 9-entry schema:
    from bang_rl.data.game_collection import formula_label_emotion_config
    EmotionInferenceSLTrainer(..., emotion_config=formula_label_emotion_config())
See bang_rl/tests/test_game_collection.py for a full example.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bang_rl.data.game_collection import (  # noqa: E402
    ACTION_MODES,
    DEFAULT_EMOTION_FILE,
    collect_games,
    load_collection_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect all-dummy games into canonical Parquet shards.")
    parser.add_argument("--games", type=int, required=True, help="number of games to play")
    parser.add_argument("--out", required=True, help="output folder (new or empty)")
    parser.add_argument("--seed", type=int, default=0, help="base seed; game i uses seed + i (default 0)")
    parser.add_argument("--mode", default="deterministic", choices=ACTION_MODES, help="dummy action mode")
    parser.add_argument("--config", default=None, help="optional SystemConfig file (.yaml/.json)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--emotion-file",
        default=DEFAULT_EMOTION_FILE,
        help=f"emotion block with a 'formula' section (default {DEFAULT_EMOTION_FILE})",
    )
    group.add_argument(
        "--rule-table",
        action="store_true",
        help="do not apply an emotion file; keep the config's own emotion block",
    )
    args = parser.parse_args(argv)

    emotion_file = None if args.rule_table else args.emotion_file
    try:
        config = load_collection_config(args.config, emotion_file=emotion_file)
    except FileNotFoundError as exc:
        parser.error(f"file not found: {exc.filename}")

    step = max(1, args.games // 20)

    def progress(done: int, total: int) -> None:
        if done == total or done % step == 0:
            print(f"  {done}/{total} games", flush=True)

    print(f"collecting {args.games} games -> {args.out}", flush=True)
    try:
        summary = collect_games(
            config,
            args.games,
            args.out,
            base_seed=args.seed,
            action_mode=args.mode,
            progress=progress,
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
