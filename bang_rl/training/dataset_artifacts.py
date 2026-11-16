"""Collected datasets (canonical Parquet folders): discovery for dataset selection.

Headless (no Qt), shared by the training screen and the training workflows. A
dataset is a folder holding ``shard-*.parquet`` files, found by that pattern, so
renamed folders work. The readable name, kind and conditions come from the
folder's ``ABOUT.md`` (:class:`~bang_rl.artifact_info.ArtifactInfo`); a legacy
folder without it counts as an all-dummy ("dummy5") dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bang_rl.artifact_info import read_info
from bang_rl.paths import PROJECT_ROOT, SAVED_DATASETS_DIR

DEFAULT_CANONICAL_ROOT = PROJECT_ROOT / "datasets" / "canonical"
PLAYERS_DUMMY_FIVE = "dummy5"
PLAYERS_MAIN_DUMMY_FOUR = "main1-dummy4"


@dataclass(frozen=True, slots=True)
class SavedDataset:
    """One dataset folder found on disk."""

    path: Path
    modified: float
    name: str = ""
    kind_label: str = ""
    players: str = PLAYERS_DUMMY_FIVE
    games: int | None = None
    rows: int | None = None
    created_at: str = ""
    notes: str = ""
    run_id: str = ""
    saved: bool = False

    @property
    def is_main_ai(self) -> bool:
        return self.players == PLAYERS_MAIN_DUMMY_FOUR

    @property
    def display_name(self) -> str:
        return self.name or self.path.name

    @property
    def label(self) -> str:
        """One line: readable name, players and number of games."""

        games = "?" if self.games is None else str(self.games)
        return f"{self.display_name} · {self.players} · {games}판"

    @property
    def tooltip(self) -> str:
        lines = [self.display_name, self.created_at or datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")]
        lines.append(f"{self.kind_label or 'dataset'} · {self.players} · games={self.games} · rows={self.rows}")
        if self.notes.strip():
            lines += ["", self.notes.strip()]
        return "\n".join(lines)


def canonical_root_for_config(config: Any) -> tuple[Path, bool]:
    """Canonical dataset folder named by a config (mapping or model), and whether it is the default.

    A relative ``dataset.canonical_path`` is resolved against the project root.
    """

    if hasattr(config, "model_dump"):
        config = config.model_dump(mode="json")
    mapping = config if isinstance(config, dict) else {}
    dataset = mapping.get("dataset") if isinstance(mapping.get("dataset"), dict) else {}
    text = dataset.get("canonical_path") or "datasets/canonical"
    path = Path(str(text)).expanduser()
    path = path if path.is_absolute() else PROJECT_ROOT / path
    return path, path == DEFAULT_CANONICAL_ROOT


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _dataset_folders(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted({shard.parent for shard in root.rglob("shard-*.parquet") if shard.is_file()})


def describe_dataset(folder: str | Path, *, saved: bool = False) -> SavedDataset:
    """Readable description of one dataset folder (works without an info block)."""

    path = Path(folder)
    info = read_info(path) if path.is_dir() else None
    conditions = info.conditions if info is not None else {}
    summary = info.summary if info is not None else {}
    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = 0.0
    return SavedDataset(
        path=path,
        modified=modified,
        name=info.name if info is not None else "",
        kind_label=info.kind_label if info is not None else "",
        players=str(conditions.get("players") or PLAYERS_DUMMY_FIVE),
        games=_int_or_none(conditions.get("games", summary.get("games"))),
        rows=_int_or_none(summary.get("rows")),
        created_at=info.created_at if info is not None else "",
        notes=info.notes if info is not None else "",
        run_id=info.id if info is not None else "",
        saved=saved,
    )


def list_datasets(
    canonical_root: str | Path | None = None,
    *,
    saved_root: str | Path | None = None,
) -> tuple[SavedDataset, ...]:
    """Dataset folders under the canonical root and the user-kept ``saved/datasets``.

    Newest first. When the same dataset (same info id) exists locally and as a
    saved copy, the local original is listed.
    """

    root = Path(canonical_root) if canonical_root is not None else DEFAULT_CANONICAL_ROOT
    saved = Path(saved_root) if saved_root is not None else (SAVED_DATASETS_DIR if canonical_root is None else None)
    found: list[SavedDataset] = []
    for folder in _dataset_folders(root):
        found.append(describe_dataset(folder))
    seen_ids = {item.run_id for item in found if item.run_id}
    if saved is not None:
        for folder in _dataset_folders(saved):
            item = describe_dataset(folder, saved=True)
            if item.run_id and item.run_id in seen_ids:
                continue
            found.append(item)
    return tuple(sorted(found, key=lambda item: item.modified, reverse=True))


__all__ = [
    "DEFAULT_CANONICAL_ROOT",
    "PLAYERS_DUMMY_FIVE",
    "PLAYERS_MAIN_DUMMY_FOUR",
    "SavedDataset",
    "canonical_root_for_config",
    "describe_dataset",
    "list_datasets",
]
