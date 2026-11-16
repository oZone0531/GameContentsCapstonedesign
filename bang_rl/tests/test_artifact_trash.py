"""Library delete: entries move to a trash folder (with a record) and can be restored."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bang_rl.artifact_trash import (
    TRASH_RECORD,
    TrashRefusedError,
    move_to_trash,
    restore_from_trash,
)


def test_files_sidecars_and_replay_traces_move_to_trash_and_restore(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    replay = data / "replay.json"
    replay.write_text("{}", encoding="utf-8")
    (data / "replay.json.about.md").write_text("about", encoding="utf-8")
    (data / "replay.txt").write_text("trace", encoding="utf-8")
    folder = data / "dataset"
    folder.mkdir()
    (folder / "shard-000000.parquet").write_bytes(b"x")

    result = move_to_trash(
        [replay, folder],
        name="my entry",
        include_replay_trace=True,
        trash_dir=tmp_path / "trash",
        allowed_roots=[data],
    )
    assert not replay.exists() and not folder.exists()
    assert not (data / "replay.txt").exists() and not (data / "replay.json.about.md").exists()
    record = json.loads((result.folder / TRASH_RECORD).read_text(encoding="utf-8"))
    assert len(record["items"]) == 4

    restored = restore_from_trash(result.folder)
    assert len(restored) == 4
    assert replay.is_file() and (folder / "shard-000000.parquet").is_file()


def test_paths_outside_the_data_folders_are_refused(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(TrashRefusedError):
        move_to_trash([outside], name="x", trash_dir=tmp_path / "trash", allowed_roots=[tmp_path / "data"])
    assert outside.is_file()


def test_missing_files_raise_instead_of_creating_an_empty_trash(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        move_to_trash([tmp_path / "gone"], name="x", trash_dir=tmp_path / "trash", allowed_roots=[tmp_path])
    assert not (tmp_path / "trash").exists()
