"""Data library: one searchable list of everything stored on disk (headless, no Qt).

The GUI's "데이터 보관함" shows collected datasets, module pairs (intent plus
emotion), main-AI models, training reports and replays in a single table. This
module only *aggregates*: discovery stays in the existing listers
(:func:`~bang_rl.training.dataset_artifacts.list_datasets`,
:func:`~bang_rl.training.module_artifacts.list_saved_module_runs`,
:func:`~bang_rl.training.main_ai_artifacts.list_saved_main_ai_models`,
:func:`~bang_rl.gui.replay_store.list_replays` and a plain scan of the report
folders), so renamed files and the local-versus-saved rules stay identical to
the training screen.

Speed: info blocks and module pair facts come from the readers' stamp caches.
Model weights are never loaded and folder sizes are computed lazily
(:meth:`LibraryEntry.size_bytes`).

Entries describe themselves with *facts*: ``(key, text)`` pairs such as
``("games", "100")``. The GUI turns each key into a translated label, so this
module has no UI text.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bang_rl.artifact_info import (
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI as INFO_KIND_MAIN_AI,
    KIND_REPORT as INFO_KIND_REPORT,
    ArtifactInfo,
    read_info,
    sidecar_path,
)
from bang_rl.artifact_info import KIND_DATASET as INFO_KIND_DATASET
from bang_rl.paths import (
    REPLAYS_DIR,
    REPORTS_DIR,
    SAVED_DATASETS_DIR,
    SAVED_MAIN_AI_DIR,
    SAVED_MODULES_DIR,
    SAVED_REPLAYS_DIR,
    SAVED_REPORTS_DIR,
)

KIND_DATASET = "dataset"
KIND_MODULE_PAIR = "module_pair"
KIND_MAIN_AI = "main_ai"
KIND_REPORT = "report"
KIND_REPLAY = "replay"
LIBRARY_KINDS: tuple[str, ...] = (KIND_DATASET, KIND_MODULE_PAIR, KIND_MAIN_AI, KIND_REPORT, KIND_REPLAY)

LOCATION_LOCAL = "local"
LOCATION_SAVED = "saved"

Fact = tuple[str, str]

# Source references name an artifact by its info kind; the library groups the
# two module kinds into one "module pair" entry.
_SOURCE_GROUP = {
    INFO_KIND_DATASET: KIND_DATASET,
    KIND_INTENT: KIND_MODULE_PAIR,
    KIND_EMOTION: KIND_MODULE_PAIR,
    "module_pair": KIND_MODULE_PAIR,
    INFO_KIND_MAIN_AI: KIND_MAIN_AI,
    INFO_KIND_REPORT: KIND_REPORT,
    "replay": KIND_REPLAY,
}


@dataclass(frozen=True, slots=True)
class LibraryLocations:
    """Folders the library reads. ``None`` fields use the project defaults."""

    model_root: Path | None = None
    saved_modules: Path | None = None
    main_ai_dirs: tuple[Path, ...] | None = None
    saved_main_ai: Path | None = None
    canonical_root: Path | None = None
    saved_datasets: Path | None = None
    report_dirs: tuple[Path, ...] | None = None
    saved_reports: Path | None = None
    replay_dirs: tuple[Path, ...] | None = None
    saved_replays: Path | None = None

    def resolved(self, config: Any = None) -> "LibraryLocations":
        """Every field filled in (defaults from the project paths and the config)."""

        from bang_rl.training.dataset_artifacts import canonical_root_for_config
        from bang_rl.training.main_ai_artifacts import MAIN_AI_MODEL_DIRS
        from bang_rl.training.module_artifacts import MODULE_MODEL_ROOT

        # Like the listers: choosing a folder explicitly drops the matching
        # ``saved/`` folder unless that one is named too.
        canonical = self.canonical_root
        saved_datasets = self.saved_datasets
        if canonical is None:
            canonical, is_default = canonical_root_for_config(config)
            if saved_datasets is None and is_default:
                saved_datasets = SAVED_DATASETS_DIR
        main_ai = self.main_ai_dirs
        saved_main_ai = self.saved_main_ai
        if main_ai is None:
            main_ai = tuple(path for path in MAIN_AI_MODEL_DIRS if path != SAVED_MAIN_AI_DIR)
            if saved_main_ai is None:
                saved_main_ai = SAVED_MAIN_AI_DIR
        model_root = self.model_root
        saved_modules = self.saved_modules
        if model_root is None:
            model_root = MODULE_MODEL_ROOT
            if saved_modules is None:
                saved_modules = SAVED_MODULES_DIR
        report_dirs = self.report_dirs
        saved_reports = self.saved_reports
        if report_dirs is None:
            report_dirs = (REPORTS_DIR,)
            if saved_reports is None:
                saved_reports = SAVED_REPORTS_DIR
        replay_dirs = self.replay_dirs
        saved_replays = self.saved_replays
        if replay_dirs is None:
            replay_dirs = (REPLAYS_DIR,)
            if saved_replays is None:
                saved_replays = SAVED_REPLAYS_DIR
        return LibraryLocations(
            model_root=Path(model_root),
            saved_modules=None if saved_modules is None else Path(saved_modules),
            main_ai_dirs=tuple(Path(item) for item in main_ai),
            saved_main_ai=None if saved_main_ai is None else Path(saved_main_ai),
            canonical_root=Path(canonical),
            saved_datasets=None if saved_datasets is None else Path(saved_datasets),
            report_dirs=tuple(Path(item) for item in report_dirs),
            saved_reports=None if saved_reports is None else Path(saved_reports),
            replay_dirs=tuple(Path(item) for item in replay_dirs),
            saved_replays=None if saved_replays is None else Path(saved_replays),
        )

    @classmethod
    def under(cls, root: str | Path) -> "LibraryLocations":
        """Every folder below one root (tests, other installs)."""

        base = Path(root)
        return cls(
            model_root=base / "models",
            saved_modules=base / "saved" / "modules",
            main_ai_dirs=(base / "models" / "main_ai", base / "artifacts"),
            saved_main_ai=base / "saved" / "main_ai",
            canonical_root=base / "canonical",
            saved_datasets=base / "saved" / "datasets",
            report_dirs=(base / "reports",),
            saved_reports=base / "saved" / "reports",
            replay_dirs=(base / "replays",),
            saved_replays=base / "saved" / "replays",
        )


@dataclass(eq=False)
class LibraryEntry:
    """One row of the library (identity is the object, :attr:`key` is its stable id)."""

    kind: str
    key: str
    id: str
    name: str
    created_epoch: float
    location: str
    path: Path
    paths: tuple[Path, ...] = ()
    kind_label: str = ""
    created_by: str = ""
    tags: tuple[str, ...] = ()
    condition_facts: tuple[Fact, ...] = ()
    summary_facts: tuple[Fact, ...] = ()
    notes: str = ""
    sources: tuple[dict[str, Any], ...] = ()
    info: ArtifactInfo | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Filled by :func:`build_library`.
    source_links: list[tuple[dict[str, Any], "LibraryEntry | None"]] = field(default_factory=list)
    used_by: list["LibraryEntry"] = field(default_factory=list)
    related: list["LibraryEntry"] = field(default_factory=list)
    _size: int | None = field(default=None, repr=False)
    _compat: tuple[bool, str | None] | None = field(default=None, repr=False)

    # -- basic views -----------------------------------------------------------
    @property
    def saved(self) -> bool:
        return self.location == LOCATION_SAVED

    @property
    def created_text(self) -> str:
        if not self.created_epoch:
            return ""
        return datetime.fromtimestamp(self.created_epoch).strftime("%Y-%m-%d %H:%M")

    @property
    def first_note_line(self) -> str:
        for line in str(self.notes or "").splitlines():
            if line.strip():
                return line.strip()
        return ""

    @property
    def intent_path(self) -> Path | None:
        return self.extra.get("intent_path")

    @property
    def emotion_path(self) -> Path | None:
        return self.extra.get("emotion_path")

    @property
    def folder(self) -> Path:
        """Folder to show in the file explorer (the dataset folder itself, else the parent)."""

        return self.path if self.path.is_dir() else self.path.parent

    @property
    def info_paths(self) -> tuple[Path, ...]:
        """Files holding this entry's info block (notes are edited there)."""

        if self.kind == KIND_MODULE_PAIR:
            return tuple(path for path in (self.intent_path, self.emotion_path) if path is not None)
        return (self.path,)

    def search_text(self) -> str:
        parts = [self.name, self.id, self.kind_label, self.notes, self.created_by, self.path.name, *self.tags]
        parts += [value for _key, value in (*self.condition_facts, *self.summary_facts)]
        return "\n".join(str(part) for part in parts if part).lower()

    # -- lazy facts ---------------------------------------------------------------
    def size_bytes(self) -> int:
        """Total size of the entry's files (and sidecars); computed once, on demand."""

        if self._size is None:
            total = 0
            for target in self.paths or (self.path,):
                total += _path_size(target)
            self._size = total
        return self._size

    def compatibility(self) -> tuple[bool, str | None]:
        """``(usable, reason)`` for main-AI use: a module pair feeding main-AI training,
        or a ``.bangmodel`` that can be played, collected with and refined.
        Other kinds are always ``(True, None)``."""

        if self._compat is None:
            self._compat = _compatibility_of(self)
        return self._compat


def _path_size(target: Path) -> int:
    total = 0
    try:
        if target.is_dir():
            for root, _dirs, files in os.walk(target):
                for name in files:
                    try:
                        total += (Path(root) / name).stat().st_size
                    except OSError:
                        continue
            return total
        total += target.stat().st_size
        side = sidecar_path(target)
        if side != target and side.is_file():
            total += side.stat().st_size
    except OSError:
        pass
    return total


def _compatibility_of(entry: LibraryEntry) -> tuple[bool, str | None]:
    if entry.kind == KIND_MODULE_PAIR:
        run = entry.extra.get("run")
        if run is None:
            return False, "unreadable"
        from bang_rl.training.module_artifacts import main_ai_compatibility

        return main_ai_compatibility(run)
    if entry.kind == KIND_MAIN_AI:
        from bang_rl.training.main_ai_artifacts import main_ai_model_compatibility

        return main_ai_model_compatibility(entry.path)
    return True, None


# --------------------------------------------------------------------- helpers


def _epoch(created_at: str | None, fallback: float) -> float:
    if created_at:
        try:
            return datetime.fromisoformat(str(created_at)).timestamp()
        except ValueError:
            pass
    return float(fallback or 0.0)


def _num(value: Any, digits: int = 3) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value:  # NaN
        return None
    return f"{float(value):.{digits}f}"


def _int_text(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return str(int(value))


def _facts(*pairs: tuple[str, str | None]) -> tuple[Fact, ...]:
    return tuple((key, text) for key, text in pairs if text not in (None, ""))


def _under(path: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        path.resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_tuple(info: ArtifactInfo | None) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in (info.sources if info is not None else ()))


def _unique_key(seen: set[str], key: str, path: Path) -> str:
    if key in seen:
        key = f"{key}@{path}"
    seen.add(key)
    return key


# --------------------------------------------------------------------- builders


def _dataset_entries(locations: LibraryLocations, seen: set[str]) -> list[LibraryEntry]:
    from bang_rl.training.dataset_artifacts import list_datasets

    entries: list[LibraryEntry] = []
    try:
        found = list_datasets(locations.canonical_root, saved_root=locations.saved_datasets)
    except OSError:
        return entries
    for dataset in found:
        info = read_info(dataset.path)
        conditions = _mapping(info.conditions if info else None)
        summary = _mapping(info.summary if info else None)
        win_rate = summary.get("win_rate")
        entries.append(
            LibraryEntry(
                kind=KIND_DATASET,
                key=_unique_key(seen, f"{KIND_DATASET}:{dataset.run_id or dataset.path}", dataset.path),
                id=dataset.run_id,
                name=dataset.display_name,
                created_epoch=_epoch(dataset.created_at, dataset.modified),
                location=LOCATION_SAVED if dataset.saved else LOCATION_LOCAL,
                path=dataset.path,
                paths=(dataset.path,),
                kind_label=dataset.kind_label,
                created_by=info.created_by if info else "",
                tags=(dataset.players,),
                condition_facts=_facts(
                    ("players", dataset.players),
                    ("games", _int_text(dataset.games)),
                    ("action_mode", str(conditions["action_mode"]) if conditions.get("action_mode") else None),
                ),
                summary_facts=_facts(
                    ("rows", _int_text(dataset.rows)),
                    ("win_rate", _num(win_rate, 2)),
                    ("truncated", _int_text(summary.get("truncated_games"))),
                ),
                notes=dataset.notes,
                sources=_source_tuple(info),
                info=info,
                extra={"saved_dataset": dataset},
            )
        )
    return entries


def _module_entries(locations: LibraryLocations, seen: set[str]) -> list[LibraryEntry]:
    from bang_rl.training.module_artifacts import (
        list_saved_module_runs,
        main_ai_compatibility,
        module_run_infos,
    )

    entries: list[LibraryEntry] = []
    try:
        runs = list_saved_module_runs(locations.model_root, saved_root=locations.saved_modules)
    except OSError:
        return entries
    for run in runs:
        infos = module_run_infos(run)
        intent, emotion = infos["intent"], infos["emotion"]
        primary = intent or emotion
        conditions = _mapping(primary.conditions if primary else None)
        epochs = (intent.conditions.get("total_epochs") if intent else None) or conditions.get("total_epochs")
        datasets = [item for item in (primary.sources if primary else ()) if item.get("kind") == INFO_KIND_DATASET]
        isummary = _mapping(intent.summary if intent else None)
        esummary = _mapping(emotion.summary if emotion else None)
        ok, reason = main_ai_compatibility(run)
        parts = tuple(name for name, path in (("intent", run.intent_path), ("emotion", run.emotion_path)) if path)
        paths = tuple(path for path in (run.intent_path, run.emotion_path) if path is not None)
        entries.append(
            LibraryEntry(
                kind=KIND_MODULE_PAIR,
                key=_unique_key(seen, f"{KIND_MODULE_PAIR}:{run.job_id}", paths[0]),
                id=run.job_id,
                name=run.display_name,
                created_epoch=run.modified,
                location=LOCATION_SAVED if run.saved else LOCATION_LOCAL,
                path=paths[0] if paths else Path(),
                paths=paths,
                kind_label=run.kind_label,
                created_by=primary.created_by if primary else "",
                tags=parts,
                condition_facts=_facts(
                    ("epochs", _int_text(epochs)),
                    ("datasets", str(len(datasets)) if datasets else None),
                    ("parts", "+".join(parts) if len(parts) < 2 else None),
                ),
                summary_facts=_facts(
                    ("intent_accuracy", _num(isummary.get("test_accuracy"))),
                    ("emotion_accuracy", _num(esummary.get("validation_accuracy"))),
                    ("emotion_mse", _num(esummary.get("test_mse"), 4)),
                    ("main_ai_ready", "ok" if ok else str(reason or "no")),
                ),
                notes=run.notes,
                sources=_source_tuple(primary),
                info=primary,
                extra={
                    "run": run,
                    "intent_path": run.intent_path,
                    "emotion_path": run.emotion_path,
                    "job_id": run.job_id,
                },
                _compat=(ok, reason),
            )
        )
    return entries


def _main_ai_entries(locations: LibraryLocations, seen: set[str]) -> list[LibraryEntry]:
    from bang_rl.training.main_ai_artifacts import (
        list_saved_main_ai_models,
        summarize_main_ai_artifact,
    )

    from bang_rl.training.main_ai_artifacts import MAIN_AI_MODEL_DIRS

    entries: list[LibraryEntry] = []
    directories = [*(locations.main_ai_dirs or ()), locations.saved_main_ai]
    default_local = tuple(path for path in MAIN_AI_MODEL_DIRS if path != SAVED_MAIN_AI_DIR)
    is_default = tuple(locations.main_ai_dirs or ()) == default_local and locations.saved_main_ai == SAVED_MAIN_AI_DIR
    try:
        # The default folders use the lister's own default mode (it also searches inside saved/main_ai).
        models = list_saved_main_ai_models(None if is_default else [path for path in directories if path is not None])
    except OSError:
        return entries
    for model in models:
        info = read_info(model.path)
        summary = _mapping(info.summary if info else None)
        conditions = _mapping(info.conditions if info else None)
        games = conditions.get("training_games")
        legacy: Mapping[str, Any] = {}
        if info is None or not summary:
            # An old model keeps its numbers in the bundle metadata (zip, no weights).
            legacy = summarize_main_ai_artifact(model.path)
        refined = bool(summary.get("refined")) or bool(info and info.created_by == "inference_refinement")
        base = next((item for item in (info.sources if info else ()) if item.get("kind") == INFO_KIND_MAIN_AI), None)
        win = summary.get("win_rate", legacy.get("eval_win_rate"))
        score = summary.get("normalized_score", legacy.get("normalized_mean"))
        hidden = summary.get("hidden_role_accuracy")
        role_accuracy = summary.get("role_accuracy", legacy.get("role_accuracy"))
        mse = summary.get("emotion_mse", legacy.get("emotion_mse"))
        if games is None:
            games = legacy.get("training_games")
        saved = model.saved or _under(model.path, locations.saved_main_ai)
        entries.append(
            LibraryEntry(
                kind=KIND_MAIN_AI,
                key=_unique_key(seen, f"{KIND_MAIN_AI}:{model.run_id or model.path}", model.path),
                id=model.run_id,
                name=model.display_name,
                created_epoch=model.modified,
                location=LOCATION_SAVED if saved else LOCATION_LOCAL,
                path=model.path,
                paths=(model.path,),
                kind_label=info.kind_label if info else "",
                created_by=info.created_by if info else "",
                tags=("refined",) if refined else (),
                condition_facts=_facts(
                    ("games", None if refined else _int_text(games)),
                    ("base_model", str(base.get("name") or base.get("id")) if base and refined else None),
                ),
                summary_facts=_facts(
                    ("win_rate", _num(win)),
                    ("score", _num(score)),
                    ("inference_accuracy", _num(hidden if hidden is not None else role_accuracy)),
                    ("emotion_mse", _num(mse, 4)),
                ),
                notes=model.notes,
                sources=_source_tuple(info),
                info=info,
                extra={"saved_model": model},
            )
        )
    return entries


def _report_files(locations: LibraryLocations) -> list[tuple[Path, bool]]:
    found: list[tuple[Path, bool]] = []
    for directory in locations.report_dirs or ():
        if Path(directory).is_dir():
            found += [(path, False) for path in sorted(Path(directory).glob("*.json")) if path.is_file()]
    saved = locations.saved_reports
    if saved is not None and Path(saved).is_dir():
        found += [(path, True) for path in sorted(Path(saved).rglob("*.json")) if path.is_file()]
    return found


def _report_entries(locations: LibraryLocations, seen: set[str]) -> list[LibraryEntry]:
    entries: list[LibraryEntry] = []
    local_ids: set[str] = set()
    for path, saved in _report_files(locations):
        info = read_info(path)
        if info is not None and info.kind != INFO_KIND_REPORT:
            continue  # another JSON kind (baseline, evaluation) that lives in the same folder
        if info is not None and info.id and not saved:
            local_ids.add(info.id)
        elif info is not None and info.id in local_ids:
            continue  # the local original wins over its saved copy
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        conditions = _mapping(info.conditions if info else None)
        summary = _mapping(info.summary if info else None)
        workflow = str(conditions.get("workflow") or (info.created_by if info else "") or "")
        entries.append(
            LibraryEntry(
                kind=KIND_REPORT,
                key=_unique_key(seen, f"{KIND_REPORT}:{info.id if info and info.id else path}", path),
                id=info.id if info else "",
                name=(info.name if info and info.name else path.stem),
                created_epoch=_epoch(info.created_at if info else None, stamp),
                location=LOCATION_SAVED if saved else LOCATION_LOCAL,
                path=path,
                paths=(path,),
                kind_label=info.kind_label if info else "",
                created_by=info.created_by if info else "",
                tags=(workflow,) if workflow else (),
                condition_facts=_facts(("workflow", workflow or None)),
                summary_facts=_facts(
                    ("status", str(summary["status"]) if summary.get("status") else None),
                    ("games", _int_text(summary.get("games_completed"))),
                    ("win_rate", _num(summary.get("win_rate"), 2)),
                ),
                notes=info.notes if info else "",
                sources=_source_tuple(info),
                info=info,
            )
        )
    return entries


def _replay_entries(locations: LibraryLocations, seen: set[str]) -> list[LibraryEntry]:
    from bang_rl.gui import replay_store

    entries: list[LibraryEntry] = []
    sources: list[tuple[Path, bool]] = [(Path(item), False) for item in (locations.replay_dirs or ())]
    if locations.saved_replays is not None:
        sources.append((Path(locations.saved_replays), True))
    local_ids: set[str] = set()
    for directory, saved in sources:
        try:
            paths = replay_store.list_replays(directory)
        except OSError:
            continue
        for path in paths:
            info = read_info(path)
            if info is not None and info.id and not saved:
                local_ids.add(info.id)
            elif info is not None and info.id in local_ids:
                continue
            try:
                stamp = path.stat().st_mtime
            except OSError:
                continue
            conditions = _mapping(info.conditions if info else None)
            summary = _mapping(info.summary if info else None)
            if info is None:
                # A replay recorded before info blocks existed: read its small header.
                try:
                    record = replay_store.load_replay(path)
                except (OSError, ValueError):
                    record = {}
                header = replay_store.replay_summary(record) if isinstance(record, Mapping) else {}
                conditions = {"mode": header.get("mode"), "mode_label": header.get("mode_label"), "players": header.get("mode")}
                summary = {"outcome": header.get("outcome"), "winning_team": header.get("winning_team"), "human_won": header.get("human_won")}
            players = str(conditions.get("players") or conditions.get("mode") or "")
            human_won = summary.get("human_won")
            entries.append(
                LibraryEntry(
                    kind=KIND_REPLAY,
                    key=_unique_key(seen, f"{KIND_REPLAY}:{info.id if info and info.id else path}", path),
                    id=info.id if info else "",
                    name=(info.name if info and info.name else path.stem),
                    created_epoch=_epoch(info.created_at if info else None, stamp),
                    location=LOCATION_SAVED if saved else LOCATION_LOCAL,
                    path=path,
                    paths=(path,),
                    kind_label=info.kind_label if info else "",
                    created_by=info.created_by if info else "",
                    tags=(players,) if players else (),
                    condition_facts=_facts(
                        ("mode", str(conditions.get("mode_label") or conditions.get("mode") or "") or None),
                        ("players", players or None),
                    ),
                    summary_facts=_facts(
                        ("winner", str(summary["winning_team"]) if summary.get("winning_team") else None),
                        ("outcome", str(summary["outcome"]) if summary.get("outcome") else None),
                        ("human_won", ("yes" if human_won else "no") if isinstance(human_won, bool) else None),
                    ),
                    notes=info.notes if info else "",
                    sources=_source_tuple(info),
                    info=info,
                )
            )
    return entries


# --------------------------------------------------------------------- the library


@dataclass(eq=False)
class Library:
    """All entries (newest first) with source and used-by links resolved."""

    entries: list[LibraryEntry]
    _by_key: dict[str, LibraryEntry] = field(default_factory=dict, repr=False)
    _by_group_id: dict[tuple[str, str], LibraryEntry] = field(default_factory=dict, repr=False)

    def get(self, key: str) -> LibraryEntry | None:
        return self._by_key.get(key)

    def resolve(self, ref: Mapping[str, Any]) -> LibraryEntry | None:
        """Entry a source reference (``{"kind", "id", "name"}``) points to, if stored here."""

        group = _SOURCE_GROUP.get(str(ref.get("kind") or ""))
        identifier = str(ref.get("id") or "")
        if group is None or not identifier:
            return None
        return self._by_group_id.get((group, identifier))

    def counts(self) -> dict[str, int]:
        result = {kind: 0 for kind in LIBRARY_KINDS}
        for entry in self.entries:
            result[entry.kind] += 1
        return result

    def filtered(
        self,
        *,
        kinds: str | Iterable[str] | None = None,
        location: str | None = None,
        text: str = "",
    ) -> list[LibraryEntry]:
        return filter_entries(self.entries, kinds=kinds, location=location, text=text)

    def _index(self) -> None:
        self._by_key = {entry.key: entry for entry in self.entries}
        self._by_group_id = {}
        for entry in self.entries:
            if entry.id:
                # A local original is listed before a saved copy of the same id, so keep the first.
                self._by_group_id.setdefault((entry.kind, entry.id), entry)

    def _link(self) -> None:
        for entry in self.entries:
            entry.source_links = []
            entry.used_by = []
            entry.related = []
        for entry in self.entries:
            seen_targets: set[int] = set()
            for ref in entry.sources:
                target = self.resolve(ref)
                entry.source_links.append((dict(ref), target))
                if target is not None and target is not entry and id(target) not in seen_targets:
                    seen_targets.add(id(target))
                    target.used_by.append(entry)
            # A job's report shares its id with the modules or model the job produced.
            if entry.kind in (KIND_MODULE_PAIR, KIND_MAIN_AI) and entry.id:
                report = self._by_group_id.get((KIND_REPORT, entry.id))
                if report is not None:
                    entry.related.append(report)
                    report.related.append(entry)


def filter_entries(
    entries: Iterable[LibraryEntry],
    *,
    kinds: str | Iterable[str] | None = None,
    location: str | None = None,
    text: str = "",
) -> list[LibraryEntry]:
    """Entries of the given kind(s) and location whose text holds every search word."""

    wanted_kinds = {kinds} if isinstance(kinds, str) else (set(kinds) if kinds else None)
    words = [word for word in str(text or "").lower().split() if word]
    result: list[LibraryEntry] = []
    for entry in entries:
        if wanted_kinds is not None and entry.kind not in wanted_kinds:
            continue
        if location and entry.location != location:
            continue
        if words:
            haystack = entry.search_text()
            if not all(word in haystack for word in words):
                continue
        result.append(entry)
    return result


def build_library(locations: LibraryLocations | None = None, *, config: Any = None) -> Library:
    """Scan every folder once and return the linked library, newest first."""

    resolved = (locations or LibraryLocations()).resolved(config)
    seen: set[str] = set()
    entries: list[LibraryEntry] = []
    for builder in (_dataset_entries, _module_entries, _main_ai_entries, _report_entries, _replay_entries):
        entries.extend(builder(resolved, seen))
    entries.sort(key=lambda item: (-item.created_epoch, item.name, item.key))
    library = Library(entries)
    library._index()
    library._link()
    return library


def list_library_entries(
    locations: LibraryLocations | None = None,
    *,
    config: Any = None,
    kinds: str | Iterable[str] | None = None,
    location: str | None = None,
    text: str = "",
) -> list[LibraryEntry]:
    """Convenience: build the library and return the filtered entries."""

    return build_library(locations, config=config).filtered(kinds=kinds, location=location, text=text)


__all__ = [
    "KIND_DATASET",
    "KIND_MAIN_AI",
    "KIND_MODULE_PAIR",
    "KIND_REPLAY",
    "KIND_REPORT",
    "LIBRARY_KINDS",
    "LOCATION_LOCAL",
    "LOCATION_SAVED",
    "Fact",
    "Library",
    "LibraryEntry",
    "LibraryLocations",
    "build_library",
    "filter_entries",
    "list_library_entries",
]
