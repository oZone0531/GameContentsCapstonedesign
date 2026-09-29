"""Qt-free, numpy-free replay persistence for human GUI games.

Every human-played game is auto-saved as a JSON replay record containing
everything needed to reproduce and re-watch it: the game type / mode, the
win/loss outcome, the AI version(s) used, and the per-turn PUBLIC observations
(each already carrying its cumulative ``event_sequence``).

The record shape is one :class:`bang_rl.gui.replay.ReplayModel` already accepts:
a mapping carrying an ``observations`` list falls through to
``ReplayModel._from_mapping`` -> ``_from_observations``, so a saved replay
renders on the SAME graphical board + log as live play. The extra metadata keys
(``schema_version``, ``mode``, ``ai_versions`` ...) sit alongside ``observations``
and are simply ignored by ReplayModel, while the browser reads them for display.

This module imports NO Qt and NO numpy: it accepts already-JSON-safe dicts (the
application ``GameService`` already returns JSON-safe observations), so it is
importable and unit-testable in a headless CI where PyQt6/numpy are absent.

Privacy: only the PUBLIC observation projection is stored. The recorded
observations use the same public keys the live board reads, so hidden living
roles stay hidden. No private ``role_labels`` are ever written here.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

# Bumped when the record layout changes in a way that affects loaders.
SCHEMA_VERSION = "1.0"


def _json_safe(value: Any) -> Any:
    """Return a deep, JSON-safe copy of ``value`` (dicts/lists/scalars only).

    The application service already returns JSON-safe observations, but this
    stays defensive: tuples become lists, mappings become plain dicts, and any
    non-JSON scalar falls back to ``str`` so a single odd value never breaks a
    save. No numpy import: numpy scalars expose ``.item()`` which is used when
    present.
    """

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def build_replay_record(
    *,
    mode: str,
    mode_label: str,
    ai_versions: Any,
    winning_team: str | None,
    winners: Sequence[int] | None,
    human_player: int,
    seed: int | None,
    observations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]] | None = None,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, Any]:
    """Build a JSON-safe replay record for a finished human game.

    The returned mapping carries an ``observations`` list (one public
    observation dict per turn, each with its cumulative ``event_sequence``)
    plus reproduction metadata. Feeding it to
    :class:`bang_rl.gui.replay.ReplayModel` yields one snapshot per observation.
    """

    winner_list = [int(seat) for seat in (winners or ())]
    human_won = human_player in winner_list
    outcome = "truncated" if truncated else "normal"
    safe_observations = [_json_safe(obs) for obs in (observations or ()) if isinstance(obs, Mapping)]

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        # Stable enum value (e.g. 'human_random') for programmatic use, and a
        # raw english label; the viewer localizes separately.
        "game_type": mode,
        "mode": mode,
        "mode_label": mode_label,
        "ai_versions": _json_safe(ai_versions),
        "winning_team": winning_team,
        "winners": winner_list,
        "human_player": int(human_player),
        "human_won": bool(human_won),
        "outcome": outcome,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "seed": seed,
        "observations": safe_observations,
    }
    if events:
        record["events"] = [_json_safe(event) for event in events if isinstance(event, Mapping)]
    return record


def _timestamped_filename(session: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if session:
        safe = "".join(ch for ch in str(session) if ch.isalnum() or ch in ("-", "_"))
        if safe:
            return f"replay-{safe}-{stamp}.json"
    return f"replay-{stamp}.json"


def save_replay(
    record: Mapping[str, Any],
    directory: str | Path | None = None,
    *,
    filename: str | None = None,
    session: str | None = None,
) -> Path:
    """Write ``record`` as JSON into ``directory`` and return the file path.

    ``directory`` defaults to :data:`bang_rl.paths.REPLAYS_DIR`. The directory
    is created if missing. A unique timestamped filename is used when
    ``filename`` is not given.
    """

    if directory is None:
        from bang_rl.paths import REPLAYS_DIR

        directory = REPLAYS_DIR
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    name = filename or _timestamped_filename(session)
    path = target_dir / name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    return path


def list_replays(directory: str | Path | None = None) -> list[Path]:
    """Return replay files (``*.json`` / ``*.jsonl``) newest-first.

    Missing directories yield an empty list instead of raising, so a browser
    can be opened before any replay has been saved.
    """

    if directory is None:
        from bang_rl.paths import REPLAYS_DIR

        directory = REPLAYS_DIR
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        return []
    files = [
        path
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".json", ".jsonl")
    ]
    files.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return files


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load a replay JSON file and return its parsed mapping."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def replay_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight display header (mode_label / outcome / winner).

    Used by the browser to label each list entry without reconstructing the
    whole ReplayModel. Tolerant of missing keys.
    """

    return {
        "mode": record.get("mode"),
        "mode_label": record.get("mode_label"),
        "outcome": record.get("outcome"),
        "winning_team": record.get("winning_team"),
        "human_won": record.get("human_won"),
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_replay_record",
    "list_replays",
    "load_replay",
    "replay_summary",
    "save_replay",
]
