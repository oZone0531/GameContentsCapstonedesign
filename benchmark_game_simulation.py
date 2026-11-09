"""Runner-free CPU simulation benchmark (no Parquet serialization).

Measures how fast real DummyAgent games can generate observation/action/emotion
records, using the same play_and_collect_game API as canonical data collection.
For full disk-inclusive collection throughput, use benchmark_data_collection.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

from bang_rl.data.collector import EpisodeCollector
from bang_rl.data.game_collection import ACTION_MODES, DEFAULT_EMOTION_FILE, load_collection_config, play_and_collect_game
from bang_rl.data.parallel_collection import _worker_setup
from bang_rl.config import SystemConfig


def _simulate_chunk(config: SystemConfig, first_seed: int, count: int,
                    mode: str, belief: str | None) -> list[tuple[int, int, str]]:
    records = []
    for seed in range(first_seed, first_seed + count):
        episode, _, _ = play_and_collect_game(config, EpisodeCollector(),
                                             seed=seed, action_mode=mode, belief=belief)
        payload = {
            "seed": episode.seed,
            "status": episode.termination_status,
            "roles": episode.role_labels,
            "steps": [
                (step.actor_idx, step.action, step.pre_observation, step.emotion_labels)
                for step in episode.steps
            ],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                            default=str).encode("utf-8")).hexdigest()
        records.append((seed, len(episode.steps), digest))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure true game-simulation CPU throughput, independent of PPO runner")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mode", choices=ACTION_MODES, default="deterministic")
    parser.add_argument("--config", default=None)
    parser.add_argument("--emotion-file", default=DEFAULT_EMOTION_FILE)
    parser.add_argument("--rule-table", action="store_true")
    parser.add_argument("--belief", default="rule")
    parser.add_argument("--report", type=Path, default=Path("benchmark_simulation_report.json"))
    args = parser.parse_args(argv)
    if args.games < 1 or args.workers < 2:
        parser.error("--games must be >= 1 and --workers must be >= 2")
    config = load_collection_config(args.config, emotion_file=None if args.rule_table else args.emotion_file)
    print(f"Serial: {args.games} real games (same seed/config in both runs)...", flush=True)
    before = time.perf_counter()
    serial = _simulate_chunk(config, args.seed, args.games, args.mode, args.belief)
    serial_seconds = time.perf_counter() - before

    actual_workers = min(args.workers, args.games)
    base, remainder = divmod(args.games, actual_workers)
    chunks = []
    position = 0
    for i in range(actual_workers):
        count = base + (1 if i < remainder else 0)
        chunks.append((args.seed + position, count))
        position += count
    print(f"Parallel: {args.games} real games on {actual_workers} CPU processes...", flush=True)
    before = time.perf_counter()
    with ProcessPoolExecutor(max_workers=actual_workers,
                             mp_context=mp.get_context("spawn"),
                             initializer=_worker_setup) as executor:
        futures = [executor.submit(_simulate_chunk, config, first, count, args.mode, args.belief)
                   for first, count in chunks]
        parallel = [record for future in futures for record in future.result()]
    parallel_seconds = time.perf_counter() - before
    same = serial == parallel
    report = {
        "measurement": "real DummyAgent game simulation including episode collection; excludes Parquet IO and PPO",
        "games": args.games, "workers": actual_workers,
        "serial_seconds": round(serial_seconds, 3),
        "parallel_seconds": round(parallel_seconds, 3),
        "speedup": round(serial_seconds / parallel_seconds, 3),
        "serial_games_per_second": round(args.games / serial_seconds, 3),
        "parallel_games_per_second": round(args.games / parallel_seconds, 3),
        "same_gameplay": same,
        "environment_steps": sum(entry[1] for entry in serial),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
