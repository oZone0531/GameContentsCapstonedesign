"""Measure genuine Phase 1 data collection throughput (not PPO training).

Use an identical number of games, config and seeds for 1 and N processes.
Both runs write canonical Parquet files to their own folders.  Comparing
seeds/actions/emotion labels guards against bogus "speedups" that change data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from bang_rl.data.game_collection import DEFAULT_EMOTION_FILE, load_collection_config
from bang_rl.data.parallel_collection import collect_games_parallel
from bang_rl.data.parquet_writer import ParquetCanonicalShardWriter


def _episode_hashes(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for shard in sorted(directory.glob("shard-*.parquet")):
        for episode in ParquetCanonicalShardWriter.read_shard(shard):
            # Batch metadata includes the batch starting seed by design. It
            # isn't part of gameplay and is excluded from this equivalence test.
            payload = {
                "seed": episode.seed,
                "termination_status": episode.termination_status,
                "role_labels": episode.role_labels,
                "steps": [
                    {"action": step.action, "actor_idx": step.actor_idx,
                     "emotion_labels": step.emotion_labels,
                     "pre_observation": step.pre_observation,
                     "post_observation": step.post_observation}
                    for step in episode.steps
                ],
            }
            raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
            if episode.episode_id in result:
                raise ValueError("duplicate episode_id: " + episode.episode_id)
            result[episode.episode_id] = hashlib.sha256(raw).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark one vs several CPU data-collection workers.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("benchmark_collection"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--emotion-file", default=DEFAULT_EMOTION_FILE)
    parser.add_argument("--rule-table", action="store_true")
    parser.add_argument("--belief", default="rule")
    args = parser.parse_args(argv)
    if args.workers < 2:
        parser.error("--workers must be at least 2 for a meaningful comparison")
    if args.games < 1:
        parser.error("--games must be positive")
    out = args.out
    if out.exists() and any(out.iterdir()):
        parser.error(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = load_collection_config(args.config, emotion_file=None if args.rule_table else args.emotion_file)
    reports = {}
    for name, workers in (("serial", 1), ("parallel", args.workers)):
        print(f"{name}: collecting {args.games} games using {workers} worker(s)...", flush=True)
        reports[name] = collect_games_parallel(
            config, args.games, out / name, workers=workers,
            base_seed=args.seed, action_mode=args.mode, belief=args.belief,
        )
    serial_signatures = _episode_hashes(out / "serial")
    parallel_signatures = _episode_hashes(out / "parallel")
    same = serial_signatures == parallel_signatures
    serial_s = float(reports["serial"]["seconds"])
    parallel_s = float(reports["parallel"]["seconds"])
    result = {
        "description": "canonical Phase 1 dummy game data generation; PPO not executed",
        "games": args.games, "seed": args.seed, "workers": args.workers,
        "serial_seconds": serial_s,
        "parallel_seconds": parallel_s,
        "serial_games_per_second": reports["serial"]["games_per_second"],
        "parallel_games_per_second": reports["parallel"]["games_per_second"],
        "speedup": round(serial_s / parallel_s, 3) if parallel_s else None,
        "same_gameplay": same,
        "serial_total_steps": reports["serial"]["total_steps"],
        "parallel_total_steps": reports["parallel"]["total_steps"],
        "same_seed_set": serial_signatures.keys() == parallel_signatures.keys(),
    }
    (out / "benchmark_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not same:
        print("ERROR: same seeds produced different episode data", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
