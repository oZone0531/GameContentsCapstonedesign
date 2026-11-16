"""Collect a canonical training dataset from all-dummy games (Phase 1).

Usage (from the repository root, next to the bang_rl folder):
    python collect_data.py --games 1000 --out bang_rl/datasets/canonical/formula_v1
    python collect_data.py --games 1000 --out <folder> --emotion-file my_formula.json
    python collect_data.py --games 1000 --out <folder> --config config.example.yaml
    python collect_data.py --games 1000 --out <folder> --mode stochastic --seed 5000
    python collect_data.py --games 1000 --out <folder> --belief hint:0.7
    python collect_data.py --games 1000 --out <folder> --players main1-dummy4 --model artifacts/x.bangmodel

With --players main1-dummy4 a main AI (.bangmodel) plays one seat (game i: seat
i % 5) among four dummies and only the main AI's own decisions are stored, with
the dummies' emotions and all roles as labels (the data for refining the
inference modules, see USER_GUIDE.md).

By default the dummies run the emotion formula from emotion_formula.example.json,
so every step carries the 9-entry emotion label
[calm, confidence, aggression, anxiety, relation_0..relation_4] for every living
dummy.

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
    load_collection_config,
)
from bang_rl.data.parallel_collection import collect_games_parallel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect all-dummy games into canonical Parquet shards.")
    parser.add_argument("--games", type=int, required=True, help="number of games to play")
    parser.add_argument("--out", required=True, help="output folder (new or empty)")
    parser.add_argument("--seed", type=int, default=0, help="base seed; game i uses seed + i (default 0)")
    parser.add_argument("--mode", default="deterministic", choices=ACTION_MODES, help="dummy action mode")
    parser.add_argument("--config", default=None, help="optional SystemConfig file (.yaml/.json)")
    parser.add_argument(
        "--belief",
        default="rule",
        help=(
            "dummy role beliefs: 'rule' (default), 'hint:Q[:S]' (start from a hint that is right "
            "with probability Q, strength S=0.5), 'learned:PATH' (main agent inference from a "
            ".phase2model) or 'oracle:ALPHA' (exact roles, reference only)"
        ),
    )
    parser.add_argument(
        "--emotion-file",
        default=DEFAULT_EMOTION_FILE,
        help=f"emotion block with a 'formula' section (default {DEFAULT_EMOTION_FILE})",
    )
    parser.add_argument(
        "--players",
        default="dummy5",
        choices=("dummy5", "main1-dummy4"),
        help=(
            "dummy5 (default): five dummies. main1-dummy4: the --model .bangmodel plays one seat "
            "(game i: seat i %% 5) among four dummies; only the main AI's own decisions are stored"
        ),
    )
    parser.add_argument("--model", default=None, help="main AI .bangmodel file (required for --players main1-dummy4)")
    parser.add_argument(
        "--main-mode",
        default="deterministic",
        choices=ACTION_MODES,
        help="main AI action mode for --players main1-dummy4 (default deterministic, like the GUI game and evaluation)",
    )
    parser.add_argument("--workers", type=int, default=1, help="CPU processes (1=original serial collector)")
    parser.add_argument("--batch-games", type=int, default=None, help="games per worker task (default: auto)")
    parser.add_argument("--shard-games", type=int, default=20, help="flush Parquet every N games (default 20)")
    args = parser.parse_args(argv)

    try:
        config = load_collection_config(args.config, emotion_file=args.emotion_file)
    except FileNotFoundError as exc:
        parser.error(f"file not found: {exc.filename}")

    step = max(1, args.games // 20)
    if args.players == "main1-dummy4":
        return _collect_main_ai(args, config, step)

    def progress(done: int, total: int) -> None:
        if done == total or done % step == 0:
            print(f"  {done}/{total} games", flush=True)

    print(f"collecting {args.games} games -> {args.out}", flush=True)
    try:
        summary = collect_games_parallel(
            config,
            args.games,
            args.out,
            base_seed=args.seed,
            action_mode=args.mode,
            progress=progress,
            belief=args.belief,
            workers=args.workers, batch_games=args.batch_games, shard_games=args.shard_games,
            created_by="cli:collect_data",
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _collect_main_ai(args: argparse.Namespace, config, step: int) -> int:
    """Collect main1+dummy4 games with a .bangmodel (Phase 4 refinement data)."""

    from bang_rl.training.final_inference_artifact import load_final_inference_artifact
    from bang_rl.training.main_ai_collection import (
        collect_main_ai_games,
        main_ai_dataset_info,
        policy_from_artifact,
    )

    if not args.model or not Path(args.model).is_file():
        print("error: --model must be an existing .bangmodel file for --players main1-dummy4", file=sys.stderr)
        return 2
    policy = policy_from_artifact(load_final_inference_artifact(args.model))
    info = main_ai_dataset_info(
        games=args.games,
        base_seed=args.seed,
        created_by="cli:collect_data",
        created_at=None,
        action_mode=args.main_mode,
        model_path=args.model,
    )

    def progress(done: int, total: int) -> None:
        if done == total or done % step == 0:
            print(f"  {done}/{total} games", flush=True)

    print(f"collecting {args.games} main AI + 4 dummy games with {args.model} -> {args.out}", flush=True)
    try:
        summary = collect_main_ai_games(
            config,
            policy,
            args.games,
            args.out,
            base_seed=args.seed,
            action_mode=args.main_mode,
            progress=progress,
            info=info,
            source_model_path=args.model,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
