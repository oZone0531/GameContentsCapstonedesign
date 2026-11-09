"""Runner-free collection acceleration tests.

The fake collector validates deterministic output merging without Arrow I/O;
existing test_game_collection.py validates actual canonical Parquet encoding.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import pytest

from bang_rl.data.game_collection import game_seed, load_collection_config
from bang_rl.data.parallel_collection import _partition, collect_games_parallel
from bang_rl.data import parallel_collection as parallel


def _fake_collect(config, games, destination, *, base_seed=0, action_mode="deterministic",
                  belief=None, shard_games=20, progress=None):
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    for idx in range(games):
        # Real collector stores canonical episodes, not JSON: the dummy bytes
        # here let this test isolate file ordering and seed bookkeeping.
        (folder / f"shard-{idx:06d}.parquet").write_text(str(base_seed + idx))
        if progress:
            progress(idx + 1, games)
    report = {
        "games": games, "base_seed": base_seed,
        "win_counts": {"SHERIFF": games},
        "win_rate": {"SHERIFF": 1.0},
        "termination_status": {"normal": games},
        "total_steps": 10 * games, "mean_steps": 10,
        "belief_source": "rule", "emotion_label_schema": ["formula-1.0+label9"],
        "emotion_label_dimension": 9, "emotion_label_names": ["calm"],
        "emotion_mode": ["formula"],
        "shards": [f"shard-{i:06d}.parquet" for i in range(games)],
    }
    (folder / "collection_summary.json").write_text(json.dumps(report))
    return report


def test_partition_has_no_duplicate_or_missing_games():
    tasks = _partition(57, 7)
    assert sum(batch.games for batch in tasks) == 57
    assert [x for batch in tasks for x in range(batch.first_game, batch.first_game + batch.games)] == list(range(57))


class _TestThreadExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers, mp_context=None, initializer=None):
        super().__init__(max_workers=max_workers)


def test_parallel_merges_flat_shards_in_seed_order(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "collect_games", _fake_collect)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", _TestThreadExecutor)
    conf = load_collection_config()
    output = tmp_path / "parallel"
    counts = []
    result = collect_games_parallel(conf, 13, output, workers=4, base_seed=400,
                                    batch_games=3, progress=lambda n, total: counts.append(n))
    assert result["workers"] == 4
    assert result["games"] == 13
    assert result["total_steps"] == 130
    assert result["win_counts"] == {"SHERIFF": 13}
    assert result["seeds"] == [400, 412]
    assert counts[-1] == 13
    assert len(result["shards"]) == 13
    assert sorted(int((output / name).read_text()) for name in result["shards"]) == list(range(400, 413))
    assert [int((output / name).read_text()) for name in result["shards"]] == list(range(400, 413))
    assert not (output / ".parallel_work").exists()
    assert json.loads((output / "collection_summary.json").read_text())["games"] == 13


def test_serial_remains_legacy_collector(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "collect_games", _fake_collect)
    result = collect_games_parallel(load_collection_config(), 3, tmp_path / "serial", workers=1)
    assert result["workers"] == 1
    assert result["total_steps"] == 30
    assert result["games_per_second"] > 0


def test_rejects_existing_dataset_and_invalid_arguments(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "something.txt").write_text("do not clobber")
    config = load_collection_config()
    with pytest.raises(FileExistsError):
        collect_games_parallel(config, 3, out, workers=2)
    with pytest.raises(ValueError):
        collect_games_parallel(config, 3, tmp_path / "bad", workers=0)
    assert (out / "something.txt").read_text() == "do not clobber"


def _check_spawn_configuration(config):
    return config.env.max_episode_steps


def test_real_spawn_environment_accepts_config(tmp_path):
    # A pickled real config and a top-level function run in the Windows-style
    # spawn context.  This is important: notebook-local lambdas would fail.
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(_check_spawn_configuration, load_collection_config()) for _ in range(4)]
        assert [future.result(timeout=30) for future in futures] == [2000] * 4
