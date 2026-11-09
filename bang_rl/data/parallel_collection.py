"""Runner-independent CPU-parallel canonical dataset collection.

The processes play real five-DummyAgent games using game_collection.collect_games.
Workers write Parquet locally (no per-step interprocess traffic); the parent
moves finished shards into a flat, legacy-compatible canonical dataset folder.

This is Phase 1 *data generation* for intent/emotion supervised learning.  It
is not a PPO optimizer and does not pretend to train the MainAgent policy.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from bang_rl.config import SystemConfig
from bang_rl.data.game_collection import (
    ACTION_MODES, COLLECTION_FORMAT_VERSION, collect_games,
    game_seed,
)
from bang_rl.agents.dummy_agent.belief_source import parse_belief_spec


@dataclass(frozen=True)
class CollectionBatch:
    index: int
    first_game: int
    games: int


def _worker_setup() -> None:
    """Avoid N workers each launching N CPU threads for incidental Torch ops."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def _collect_batch(
    batch: CollectionBatch,
    config: SystemConfig,
    root: str,
    seed: int,
    mode: str,
    belief: str | None,
    shard_games: int,
) -> tuple[int, dict[str, Any]]:
    destination = Path(root) / f"batch-{batch.index:06d}"
    summary = collect_games(
        config, batch.games, destination,
        base_seed=game_seed(seed, batch.first_game),
        action_mode=mode, belief=belief, shard_games=shard_games,
    )
    return batch.index, summary


def _partition(games: int, batch_games: int) -> list[CollectionBatch]:
    return [CollectionBatch(index, start, min(batch_games, games - start))
            for index, start in enumerate(range(0, games, batch_games))]


def _aggregate(
    pieces: list[dict[str, Any]], games: int, base_seed: int, mode: str,
    workers: int, elapsed: float, shard_paths: list[Path], batch_games: int,
) -> dict[str, Any]:
    first = pieces[0]
    wins: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    total_steps = 0
    labels: set[str] = set()
    label_dims: set[int] = set()
    modes: set[str] = set()
    for entry in pieces:
        count = int(entry["games"])
        if "win_counts" in entry:
            wins.update({str(key): int(value) for key, value in entry["win_counts"].items()})
        else:
            # Compatibility with datasets created before exact counts existed.
            for key, value in entry["win_rate"].items():
                wins[str(key)] += round(float(value) * count)
        statuses.update({key: int(value) for key, value in entry["termination_status"].items()})
        total_steps += int(entry["total_steps"])
        labels.update(str(item) for item in entry.get("emotion_label_schema", []))
        dim = entry.get("emotion_label_dimension")
        if isinstance(dim, list):
            label_dims.update(int(value) for value in dim)
        elif dim is not None:
            label_dims.add(int(dim))
        modes.update(entry.get("emotion_mode", []))
    result: dict[str, Any] = {
        "collection_format_version": COLLECTION_FORMAT_VERSION,
        "games": games,
        "base_seed": base_seed,
        "seeds": [base_seed, base_seed + games - 1],
        "action_mode": mode,
        "belief_source": first["belief_source"],
        "emotion_mode": sorted(modes),
        "emotion_label_schema": sorted(labels),
        "emotion_label_dimension": next(iter(label_dims)) if len(label_dims) == 1 else sorted(label_dims) or None,
        "emotion_label_names": first.get("emotion_label_names"),
        "termination_status": dict(sorted(statuses.items())),
        "win_counts": dict(sorted(wins.items())),
        "win_rate": {key: round(value / games, 4) for key, value in sorted(wins.items())},
        "mean_steps": round(total_steps / games, 2),
        "total_steps": total_steps,
        "shards": [path.name for path in shard_paths],
        "seconds": round(elapsed, 3),
        "workers": workers,
        "batches": len(pieces),
        "batch_games": batch_games,
        "games_per_second": round(games / max(elapsed, 1e-9), 3),
        "environment_steps_per_second": round(total_steps / max(elapsed, 1e-9), 3),
    }
    if sum(statuses.values()) != games:
        raise RuntimeError("collected game count does not match requested games")
    return result


def collect_games_parallel(
    config: SystemConfig, games: int, output_dir: str | Path,
    *, workers: int = 4, base_seed: int = 0,
    action_mode: str = "deterministic", belief: str | None = "rule",
    batch_games: int | None = None, shard_games: int = 20,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Collect the original canonical dataset in separate CPU processes.

    A fixed seed always generates the same game, independent of worker count.
    Windows-safe spawn, atomic existing shard writer, flat output filenames.
    The output directory must be new/empty, like ``collect_games``.

    When ``workers=1`` the original sequential collector runs unchanged, so it
    can serve as a baseline for a legitimate acceleration measurement.
    """
    games = int(games)
    workers = int(workers)
    shard_games = int(shard_games)
    if games < 1 or workers < 1 or shard_games < 1:
        raise ValueError("games, workers and shard_games must be positive integers")
    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}")
    parse_belief_spec(belief)
    if not isinstance(config, SystemConfig):
        raise TypeError("config must be SystemConfig")
    if batch_games is not None and int(batch_games) < 1:
        raise ValueError("batch_games must be positive")
    path = Path(output_dir)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path} is not an empty output folder")

    start = time.perf_counter()
    if workers == 1:
        result = collect_games(
            config, games, path, base_seed=base_seed,
            action_mode=action_mode, belief=belief,
            progress=progress, shard_games=shard_games,
        )
        elapsed = time.perf_counter() - start
        result.update(
            workers=1, batches=1, batch_games=games,
            seconds=round(elapsed, 3),
            games_per_second=round(games / max(elapsed, 1e-9), 3),
            environment_steps_per_second=round(result["total_steps"] / max(elapsed, 1e-9), 3),
        )
        (path / "collection_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # Keep tasks small enough for load balancing but not so small that process
    # dispatch and repeated Parquet setup eclipse the actual game simulation.
    effective_workers = min(workers, games)
    if batch_games is None:
        batch_games = max(1, min(shard_games, (games + effective_workers * 4 - 1) // (effective_workers * 4)))
    batch_games = int(batch_games)
    tasks = _partition(games, batch_games)
    path.mkdir(parents=True, exist_ok=True)
    staging = path / ".parallel_work"
    staging.mkdir()
    done = 0
    results: dict[int, dict[str, Any]] = {}
    try:
        with ProcessPoolExecutor(max_workers=effective_workers,
                                 mp_context=mp.get_context("spawn"),
                                 initializer=_worker_setup) as executor:
            futures = {
                executor.submit(_collect_batch, batch, config, str(staging),
                                base_seed, action_mode, belief, shard_games): batch
                for batch in tasks
            }
            for future in as_completed(futures):
                batch = futures[future]
                index, result = future.result()  # Propagate worker error, never mark success.
                if index != batch.index or int(result["games"]) != batch.games:
                    raise RuntimeError("parallel batch result mismatch")
                results[index] = result
                done += batch.games
                if progress is not None:
                    progress(done, games)
        # Commit in global seed order, not process completion order.  The final
        # dataset layout is identical to the regular flat canonical layout.
        shards: list[Path] = []
        for batch in tasks:
            local = staging / f"batch-{batch.index:06d}"
            discovered = sorted(local.glob("shard-*.parquet"))
            expected = sorted(str(name) for name in results[batch.index]["shards"])
            if not discovered or [file.name for file in discovered] != expected:
                raise RuntimeError(f"batch {batch.index} is missing one or more Parquet shards")
            for source in discovered:
                target = path / f"shard-{len(shards):06d}.parquet"
                source.replace(target)
                shards.append(target)
        elapsed = time.perf_counter() - start
        summary = _aggregate(
            [results[batch.index] for batch in tasks], games, base_seed,
            action_mode, effective_workers, elapsed, shards, batch_games,
        )
        (path / "collection_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
    finally:
        shutil.rmtree(staging, ignore_errors=True)
