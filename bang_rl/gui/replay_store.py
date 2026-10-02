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

# ITEM 2 (spectator replay): the all-AI game modes. A replay recorded in one of
# these has no human seat, so the replay board must render no "나"/"You" seat.
# Mirrors GameService._SPECTATOR_MODES; kept here (Qt-free) so the replay screen
# can decide the board layout without importing the service layer.
SPECTATOR_MODES = frozenset({"random_five", "dummy_five", "main_one_dummy_four"})


def is_spectator_record(record: Mapping[str, Any] | None) -> bool:
    """Return True when ``record`` is an all-AI (spectator) replay.

    Reads the record's ``mode`` and checks it against :data:`SPECTATOR_MODES`.
    Tolerant of a missing/None record or mode. Pure and Qt-free.
    """

    if not isinstance(record, Mapping):
        return False
    mode = record.get("mode")
    return isinstance(mode, str) and mode in SPECTATOR_MODES


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
    final_roles: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe replay record for a finished human game.

    The returned mapping carries an ``observations`` list (one public
    observation dict per turn, each with its cumulative ``event_sequence``)
    plus reproduction metadata. Feeding it to
    :class:`bang_rl.gui.replay.ReplayModel` yields one snapshot per observation.

    Item 1: ``final_roles`` is the finished game's terminal role of every seat
    (from the terminal :class:`~bang_rl.app.services.PublicGameState`). It is a
    TERMINAL-ONLY reveal: :class:`~bang_rl.gui.replay.ReplayModel` surfaces it
    on the final snapshot only, so a replay reveals all roles at its END without
    leaking any living role mid-replay. It is OPTIONAL -- when absent/empty the
    record omits the field entirely, so an existing record (and any replay built
    without it) loads and renders exactly as before (legacy, no reveal).
    """

    winner_list = [int(seat) for seat in (winners or ())]
    human_won = human_player in winner_list
    outcome = "truncated" if truncated else "normal"
    safe_observations = [_json_safe(obs) for obs in (observations or ()) if isinstance(obs, Mapping)]
    role_list = [int(role) for role in (final_roles or ())]

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        # 'public' == the training-safe observations-only format (no hidden
        # state). Distinguished from the 'full' format below so a loader knows
        # whether authoritative reproduction state is available.
        "replay_format": "public",
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
    # Item 1: only write final_roles when non-empty so legacy records (and any
    # unfinished / mock save) stay byte-for-byte unchanged and loaders that do
    # not know the field keep working.
    if role_list:
        record["final_roles"] = role_list
    return record


def build_full_replay_record(
    *,
    mode: str,
    mode_label: str,
    ai_versions: Any,
    winning_team: str | None,
    winners: Sequence[int] | None,
    human_player: int,
    seed: int | None,
    states: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    internals: Sequence[Any] | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    terminated: bool = False,
    truncated: bool = False,
    final_roles: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe FULL-reproduction-state replay record (item 5).

    Unlike :func:`build_replay_record` (the training-safe ``public`` format),
    this stores a top-level ``states`` list of authoritative
    :meth:`bang_rl.env.state.EnvironmentState.to_dict` dicts (one per recorded
    turn) plus the ``seed``, so a chosen moment can be restored deterministically
    for rewind-and-continue. It intentionally carries hidden roles / deck / seat
    order and MUST only be written for human-played and AI-spectator games, never
    for AI-training datasets.

    The record ALSO carries the same public ``observations`` list a public
    record would, so a ``full`` replay renders on the graphical board / log
    exactly like a public one (``ReplayModel`` reads ``observations`` and simply
    ignores the extra ``states`` key).

    Item 4/6: when ``internals`` is provided it is stored as a top-level
    per-step ``internals`` list, each entry the JSON-safe per-step internal
    snapshot (every player's HP plus each dummy / main AI's emotion and role
    belief / guess). Like ``states`` this carries hidden information, so it is
    only ever written into the ``full`` format for human / spectator games,
    never into the public / training format (:func:`build_replay_record` does
    not accept it). A loaded full replay therefore has actions and situation
    (from ``observations``), the authoritative moment (``states``), and per-step
    HP + AI internals (``internals``), so it can be re-watched and resumed.
    """

    base = build_replay_record(
        mode=mode,
        mode_label=mode_label,
        ai_versions=ai_versions,
        winning_team=winning_team,
        winners=winners,
        human_player=human_player,
        seed=seed,
        observations=observations,
        events=events,
        terminated=terminated,
        truncated=truncated,
        final_roles=final_roles,
    )
    base["replay_format"] = "full"
    base["states"] = [
        _json_safe(state) for state in (states or ()) if isinstance(state, Mapping)
    ]
    if internals is not None:
        # Preserve the index-parallel invariant with ``states`` / ``observations``:
        # a per-turn export miss is recorded as a ``None`` placeholder by the
        # controller, so keep those entries (as JSON null) rather than dropping
        # them, which would offset every later index.
        base["internals"] = [
            None if entry is None else _json_safe(entry) for entry in internals
        ]
    return base


def _timestamped_filename(session: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if session:
        safe = "".join(ch for ch in str(session) if ch.isalnum() or ch in ("-", "_"))
        if safe:
            return f"replay-{safe}-{stamp}.json"
    return f"replay-{stamp}.json"


def _sanitize_mode_token(mode: Any) -> str:
    """Return a filename-safe, lowercased token for a game mode.

    Keeps only alnum / ``-`` / ``_`` (the same rule :func:`_sanitize_replay_name`
    uses), which strips any path separator so a malformed mode can never escape
    the replay directory. Returns ``""`` when nothing usable remains, so the
    caller falls back to the mode-less name. Pure and Qt-free.
    """

    if mode is None:
        return ""
    text = str(mode).strip().lower()
    return "".join(ch for ch in text if ch.isalnum() or ch in ("-", "_"))


def build_replay_filename(
    mode: str | None,
    *,
    session: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """Build the auto-generated replay filename, encoding the game mode (item 5).

    Returns ``'replay-<mode>-<stamp>.json'`` where ``<mode>`` is the sanitized,
    lowercased mode token (e.g. ``human_random``, ``random_five``). When ``mode``
    is None / empty / has no usable characters, falls back to the prior mode-less
    :func:`_timestamped_filename` behaviour (which still honours ``session``).
    Pure and Qt-free so it is unit-testable headlessly.
    """

    stamp = (timestamp or datetime.now()).strftime("%Y%m%d-%H%M%S-%f")
    token = _sanitize_mode_token(mode)
    if not token:
        return _timestamped_filename(session)
    if session:
        safe = "".join(ch for ch in str(session) if ch.isalnum() or ch in ("-", "_"))
        if safe:
            return f"replay-{token}-{safe}-{stamp}.json"
    return f"replay-{token}-{stamp}.json"


def describe_ai_versions(ai_versions: Any) -> list[tuple[str, str, str, str | None]]:
    """Return a localization-friendly, Qt-free breakdown of ``ai_versions``.

    ``ai_versions`` is a seat -> descriptor mapping where a descriptor is either
    the string ``'random'`` / ``'dummy'`` or a dict
    ``{'kind': 'main', 'version': ..., 'metadata': {...}}`` (see
    :meth:`bang_rl.gui.game.GameSessionController._derive_ai_versions`).

    Each returned entry is a tuple ``('seat', seat, kind, version)`` where:

    * ``seat`` is the seat key as a string,
    * ``kind`` is one of ``'random'`` / ``'dummy'`` / ``'main'`` / ``'unknown'``,
    * ``version`` is the main-AI version string (or None) and is None for the
      other kinds.

    Entries are sorted by numeric seat when possible so the screen can render
    ``Seat 0 ... Seat 4`` in order. No localized text is produced here; the
    screen turns each tuple into user-facing text via ``tr()``. Tolerant of a
    missing / non-mapping input (returns an empty list). Pure and Qt-free.
    """

    if not isinstance(ai_versions, Mapping):
        return []

    def _sort_key(item: tuple[str, Any]) -> tuple[int, int, str]:
        seat = item[0]
        try:
            return (0, int(seat), "")
        except (TypeError, ValueError):
            return (1, 0, str(seat))

    entries: list[tuple[str, str, str, str | None]] = []
    for seat, descriptor in sorted(ai_versions.items(), key=_sort_key):
        seat_key = str(seat)
        if isinstance(descriptor, str):
            kind = descriptor.strip().lower()
            if kind in ("random", "dummy"):
                entries.append(("seat", seat_key, kind, None))
            else:
                entries.append(("seat", seat_key, "unknown", None))
        elif isinstance(descriptor, Mapping):
            kind = str(descriptor.get("kind", "")).strip().lower()
            if kind == "main":
                version = descriptor.get("version")
                entries.append(
                    ("seat", seat_key, "main", None if version is None else str(version))
                )
            elif kind in ("random", "dummy"):
                entries.append(("seat", seat_key, kind, None))
            else:
                entries.append(("seat", seat_key, "unknown", None))
        else:
            entries.append(("seat", seat_key, "unknown", None))
    return entries


def save_replay(
    record: Mapping[str, Any],
    directory: str | Path | None = None,
    *,
    filename: str | None = None,
    session: str | None = None,
    mode: str | None = None,
) -> Path:
    """Write ``record`` as JSON into ``directory`` and return the file path.

    ``directory`` defaults to :data:`bang_rl.paths.REPLAYS_DIR`. The directory
    is created if missing. When ``filename`` is not given the name is generated:
    if ``mode`` is provided (or the record carries a ``mode``), the auto name
    encodes it via :func:`build_replay_filename` (item 5); otherwise the prior
    mode-less :func:`_timestamped_filename` is used. A bad mode is sanitized and
    can never escape the directory.
    """

    if directory is None:
        from bang_rl.paths import REPLAYS_DIR

        directory = REPLAYS_DIR
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        name = filename
    else:
        effective_mode = mode if mode is not None else record.get("mode")
        name = build_replay_filename(effective_mode, session=session)
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


def _sanitize_replay_name(new_name: str, *, default_suffix: str) -> str:
    """Return a safe single-segment file name preserving a .json/.jsonl suffix.

    Strips any directory component (defends against path traversal), keeps only
    filename-safe characters, and preserves a ``.json``/``.jsonl`` suffix,
    defaulting to ``default_suffix`` when the caller supplied none. Raises
    ``ValueError`` when nothing usable remains.
    """

    # Drop any directory component so the rename can never escape the folder.
    raw = Path(str(new_name)).name.strip()
    if not raw:
        raise ValueError("replay name must not be empty")
    suffix = ""
    lowered = raw.lower()
    for candidate in (".jsonl", ".json"):
        if lowered.endswith(candidate):
            suffix = raw[-len(candidate):]
            raw = raw[: -len(candidate)]
            break
    stem = "".join(ch for ch in raw if ch.isalnum() or ch in (" ", "-", "_", ".")).strip()
    stem = stem.replace(" ", "_")
    if not stem:
        raise ValueError("replay name has no usable characters")
    if not suffix:
        suffix = default_suffix
    return f"{stem}{suffix}"


def rename_replay(path: str | Path, new_name: str) -> Path:
    """Rename the replay file at ``path`` to ``new_name`` within its folder.

    Qt-free and defensive (item 10): the new name is sanitized to a single
    filename-segment and its ``.json``/``.jsonl`` suffix is preserved, so the
    renamed file always stays inside the original replay directory. Returns the
    new path. Raises ``FileNotFoundError`` when the source is missing,
    ``ValueError`` for an unusable name, and ``FileExistsError`` when the target
    already exists.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"replay file not found: {source}")
    default_suffix = source.suffix if source.suffix.lower() in (".json", ".jsonl") else ".json"
    safe_name = _sanitize_replay_name(new_name, default_suffix=default_suffix)
    target = source.parent / safe_name
    # Confirm the resolved target really stays inside the source directory.
    if target.parent.resolve() != source.parent.resolve():
        raise ValueError("renamed replay must stay in the same directory")
    if target == source:
        return source
    if target.exists():
        raise FileExistsError(f"a replay named {safe_name!r} already exists")
    source.rename(target)
    return target


def delete_replay(path: str | Path) -> None:
    """Delete the replay file at ``path`` (item 10).

    Qt-free and defensive: a missing file is tolerated (no error), and only a
    regular file is removed so a directory can never be deleted here.
    """

    target = Path(path)
    try:
        if target.is_file():
            target.unlink()
    except FileNotFoundError:
        # Already gone -- treat as success so a double-delete never raises.
        return


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
        # 'public' (training-safe observations-only) vs 'full' (adds a top-level
        # authoritative 'states' list). Default to 'public' for legacy records
        # written before the format tag existed.
        "replay_format": record.get("replay_format", "public"),
        "has_full_state": bool(record.get("states")),
        # Item 4/6: True when the full record carries per-step internal state
        # (HP + AI emotion / belief) so a step can be re-watched / resumed.
        "has_internals": bool(record.get("internals")),
    }


__all__ = [
    "SCHEMA_VERSION",
    "SPECTATOR_MODES",
    "build_full_replay_record",
    "build_replay_filename",
    "build_replay_record",
    "delete_replay",
    "describe_ai_versions",
    "is_spectator_record",
    "list_replays",
    "load_replay",
    "rename_replay",
    "replay_summary",
    "save_replay",
]
