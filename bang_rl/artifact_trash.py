"""Move library entries to a trash folder instead of deleting them (headless, no Qt).

"Delete" in the data library never removes data permanently: every file of an
entry (and its ``.about.md`` sidecar, and a replay's ``.txt`` trace) is moved
into ``artifacts/trash/<time>_<name>/`` together with ``trash-record.json``
listing the original locations, so a mistake can be undone by moving the
files back. ``artifacts/`` is gitignored, so the trash never reaches git.

Only paths inside the project's data folders can be trashed; anything else is
refused (:class:`TrashRefusedError`).
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from bang_rl.artifact_info import now_iso, sidecar_path
from bang_rl.paths import (
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    REPLAYS_DIR,
    REPORTS_DIR,
    SAVED_DIR,
)

TRASH_DIR = ARTIFACTS_DIR / "trash"
TRASH_RECORD = "trash-record.json"
DEFAULT_ALLOWED_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "bang_rl" / "models",
    PROJECT_ROOT / "bang_rl" / "datasets",
    PROJECT_ROOT / "datasets",
    REPORTS_DIR,
    OUTPUTS_DIR,
    REPLAYS_DIR,
    SAVED_DIR,
    ARTIFACTS_DIR,
)


class TrashRefusedError(PermissionError):
    """A path is outside the folders the library may delete from."""


@dataclass(frozen=True, slots=True)
class TrashResult:
    folder: Path
    moved: tuple[tuple[Path, Path], ...]  # (original, location in the trash)


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
        except (OSError, ValueError):
            continue
        return True
    return False


def trash_paths(paths: Sequence[Path], *, include_replay_trace: bool = False) -> list[Path]:
    """Existing files/folders to move: the paths plus their sidecars (and replay traces)."""

    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        candidates = [path]
        if path.is_file():
            candidates.append(sidecar_path(path))
            if include_replay_trace:
                candidates.append(path.with_suffix(".txt"))
        for candidate in candidates:
            if candidate.exists() and candidate not in found:
                found.append(candidate)
    return found


def check_allowed(paths: Sequence[Path], allowed_roots: Sequence[Path] | None = None) -> None:
    roots = tuple(allowed_roots) if allowed_roots is not None else DEFAULT_ALLOWED_ROOTS
    for path in paths:
        if not _inside(Path(path), roots):
            raise TrashRefusedError(f"refusing to delete outside the data folders: {path}")


def move_to_trash(
    paths: Sequence[Path],
    *,
    name: str,
    include_replay_trace: bool = False,
    trash_dir: Path | None = None,
    allowed_roots: Sequence[Path] | None = None,
) -> TrashResult:
    """Move ``paths`` (plus sidecars) into a new trash folder and record where they came from."""

    targets = trash_paths(paths, include_replay_trace=include_replay_trace)
    if not targets:
        raise FileNotFoundError("nothing to delete: the files no longer exist")
    check_allowed(targets, allowed_roots)
    root = Path(trash_dir) if trash_dir is not None else TRASH_DIR
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", str(name)).strip("._")[:60] or "entry"
    folder = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{safe}"
    counter = 1
    while folder.exists():
        folder = root / f"{folder.name}-{counter}"
        counter += 1
    folder.mkdir(parents=True)
    moved: list[tuple[Path, Path]] = []
    for index, source in enumerate(targets):
        destination = folder / f"{index:02d}_{source.name}"
        shutil.move(str(source), str(destination))
        moved.append((source, destination))
    record = {
        "deleted_at": now_iso(),
        "name": str(name),
        "items": [{"original": str(old), "trashed": new.name} for old, new in moved],
        "note": "To restore, move each item back to its 'original' path (drop the NN_ prefix).",
    }
    (folder / TRASH_RECORD).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrashResult(folder, tuple(moved))


def restore_from_trash(folder: Path) -> list[Path]:
    """Move every item of a trash folder back to its original place (if that is free)."""

    record = json.loads((Path(folder) / TRASH_RECORD).read_text(encoding="utf-8"))
    restored: list[Path] = []
    for item in record.get("items", []):
        original = Path(item["original"])
        source = Path(folder) / item["trashed"]
        if source.exists() and not original.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(original))
            restored.append(original)
    return restored


__all__ = [
    "DEFAULT_ALLOWED_ROOTS",
    "TRASH_DIR",
    "TRASH_RECORD",
    "TrashRefusedError",
    "TrashResult",
    "check_allowed",
    "move_to_trash",
    "restore_from_trash",
    "trash_paths",
]
