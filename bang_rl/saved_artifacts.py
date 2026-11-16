"""Keep a copy of training outputs under ``saved/`` so they can be shared (headless).

Generated data lives in gitignored folders. The training screen's "Save
outputs" button (and the saved-model dialog) copies selected outputs into
``saved/`` instead. ``saved/`` is NOT gitignored, so teammates get those copies
with a normal ``git pull``. There is no Git LFS: GitHub rejects files above
100 MiB and warns above 50 MiB, so :func:`size_level` lets the UI warn first.

All logic here is Qt free; the dialog only renders a :class:`SavePlan`.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from bang_rl.artifact_info import (
    ArtifactInfo,
    FILE_SIDECAR_SUFFIX,
    FOLDER_SIDECAR,
    KIND_DATASET,
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI,
    KIND_REPORT,
    default_artifact_name,
    find_json_by_id,
    new_info,
    read_info,
    sanitize_user_name,
    sidecar_path,
    write_info,
)
from bang_rl.paths import REPORTS_DIR, SAVED_DIR, SAVED_REPORTS_DIR

MIB = 1024 * 1024
GITHUB_WARN_BYTES = 50 * MIB
GITHUB_LIMIT_BYTES = 100 * MIB

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_STRONG = "strong"

CATEGORY_DATASET = "datasets"
CATEGORY_MODULES = "modules"
CATEGORY_MAIN_AI = "main_ai"


class SaveNameExistsError(FileExistsError):
    """The chosen name already exists in ``saved/``; ask for another one."""


def size_level(sizes: Iterable[int]) -> str:
    """``ok`` up to 50 MiB, ``warn`` above 50 MiB, ``strong`` above 100 MiB (per file)."""

    largest = max(sizes, default=0)
    if largest > GITHUB_LIMIT_BYTES:
        return LEVEL_STRONG
    if largest > GITHUB_WARN_BYTES:
        return LEVEL_WARN
    return LEVEL_OK


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


@dataclass(frozen=True, slots=True)
class SaveItem:
    """One thing to copy: a file or a folder, and where it goes under the target."""

    source: Path
    relative: str  # name inside the destination ("" = the destination itself)
    kind: str
    label: str = ""


@dataclass(slots=True)
class SavePlan:
    category: str
    items: list[SaveItem]
    default_name: str
    notes: str = ""
    # Test seam: replaces os.path.getsize so sizes above 50/100 MiB can be faked.
    size_of: Callable[[Path], int] = field(default=lambda path: path.stat().st_size, repr=False)
    # Root of the user-kept copies; tests point this at a temporary folder.
    saved_root: Path = SAVED_DIR

    @property
    def target_root(self) -> Path:
        return self.saved_root / {
            CATEGORY_DATASET: "datasets",
            CATEGORY_MODULES: "modules",
            CATEGORY_MAIN_AI: "main_ai",
        }[self.category]

    @property
    def reports_root(self) -> Path:
        return self.saved_root / "reports"

    def files(self) -> list[tuple[Path, int]]:
        """Every file that will be copied, with its size."""

        found: list[tuple[Path, int]] = []
        for item in self.items:
            if item.source.is_dir():
                found += [(path, self.size_of(path)) for path in sorted(item.source.rglob("*")) if path.is_file()]
            elif item.source.is_file():
                found.append((item.source, self.size_of(item.source)))
                side = sidecar_path(item.source)
                if side.is_file():
                    found.append((side, self.size_of(side)))
        return found

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size in self.files())

    @property
    def largest(self) -> tuple[Path | None, int]:
        files = self.files()
        if not files:
            return None, 0
        return max(files, key=lambda pair: pair[1])

    @property
    def level(self) -> str:
        return size_level(size for _, size in self.files())

    def destination(self, name: str) -> Path:
        """Folder (datasets, modules) or base path (main AI) of the saved copy."""

        return self.target_root / sanitize_user_name(name)

    def conflicts(self, name: str) -> list[Path]:
        clean = sanitize_user_name(name)
        if self.category == CATEGORY_MAIN_AI:
            main = next((item for item in self.items if item.kind == KIND_MAIN_AI), None)
            suffix = main.source.suffix if main is not None else ".bangmodel"
            candidates = [self.target_root / f"{clean}{suffix}"]
        else:
            candidates = [self.target_root / clean]
        if any(item.kind == KIND_REPORT for item in self.items):
            candidates.append(self.reports_root / f"{clean}.json")
        return [path for path in candidates if path.exists()]


@dataclass(frozen=True, slots=True)
class SaveResult:
    category: str
    name: str
    paths: tuple[Path, ...]
    total_bytes: int


# ----------------------------------------------------------------- plan builders


def _info_or_none(path: Path) -> ArtifactInfo | None:
    return read_info(path) if path.exists() else None


def _fallback_info(kind: str, path: Path) -> ArtifactInfo:
    """Info for a legacy file without one (so a copy can still be described)."""

    return new_info(kind, id=path.stem if path.is_file() else path.name, created_by="legacy_file")


def _pair_default_name(intent: ArtifactInfo | None, emotion: ArtifactInfo | None) -> str:
    info = intent or emotion
    if info is None:
        return "modules"
    name = default_artifact_name(info)
    return name.replace("_intent_", "_modules_", 1).replace("_emotion_", "_modules_", 1)


def find_report_for(info: ArtifactInfo | None, reports_dirs: Sequence[Path] | None = None) -> Path | None:
    if info is None or not info.id:
        return None
    roots = list(reports_dirs) if reports_dirs is not None else [REPORTS_DIR, SAVED_REPORTS_DIR]
    return find_json_by_id(roots, info.id)


def plan_dataset(folder: str | Path, *, saved_root: Path | None = None) -> SavePlan | None:
    source = Path(folder)
    if not source.is_dir():
        return None
    info = _info_or_none(source)
    name = default_artifact_name(info) if info is not None else source.name
    return SavePlan(
        CATEGORY_DATASET,
        [SaveItem(source, "", KIND_DATASET, source.name)],
        default_name=name,
        notes=info.notes if info is not None else "",
        saved_root=saved_root or SAVED_DIR,
    )


def plan_modules(
    intent_path: str | Path | None,
    emotion_path: str | Path | None,
    *,
    report_path: str | Path | None = None,
    reports_dirs: Sequence[Path] | None = None,
    saved_root: Path | None = None,
) -> SavePlan | None:
    items: list[SaveItem] = []
    intent = Path(intent_path) if intent_path else None
    emotion = Path(emotion_path) if emotion_path else None
    infos: dict[str, ArtifactInfo | None] = {}
    if intent is not None and intent.is_file():
        items.append(SaveItem(intent, "intent.phase2model", KIND_INTENT, intent.name))
        infos[KIND_INTENT] = _info_or_none(intent)
    if emotion is not None and emotion.is_file():
        items.append(SaveItem(emotion, "emotion.phase2model", KIND_EMOTION, emotion.name))
        infos[KIND_EMOTION] = _info_or_none(emotion)
    if not items:
        return None
    report = Path(report_path) if report_path and Path(report_path).is_file() else None
    if report is None:
        report = find_report_for(infos.get(KIND_INTENT) or infos.get(KIND_EMOTION), reports_dirs)
    if report is not None:
        items.append(SaveItem(report, "report.json", KIND_REPORT, report.name))
    primary = infos.get(KIND_INTENT) or infos.get(KIND_EMOTION)
    return SavePlan(
        CATEGORY_MODULES,
        items,
        default_name=_pair_default_name(infos.get(KIND_INTENT), infos.get(KIND_EMOTION)),
        notes=primary.notes if primary is not None else "",
        saved_root=saved_root or SAVED_DIR,
    )


def plan_main_ai(
    model_path: str | Path,
    *,
    report_path: str | Path | None = None,
    reports_dirs: Sequence[Path] | None = None,
    saved_root: Path | None = None,
) -> SavePlan | None:
    model = Path(model_path)
    if not model.is_file():
        return None
    info = _info_or_none(model)
    items = [SaveItem(model, model.name, KIND_MAIN_AI, model.name)]
    report = Path(report_path) if report_path and Path(report_path).is_file() else None
    if report is None:
        report = find_report_for(info, reports_dirs)
    if report is not None:
        items.append(SaveItem(report, "report.json", KIND_REPORT, report.name))
    return SavePlan(
        CATEGORY_MAIN_AI,
        items,
        default_name=default_artifact_name(info) if info is not None else model.stem,
        notes=info.notes if info is not None else "",
        saved_root=saved_root or SAVED_DIR,
    )


def plans_for_training_result(
    mode: str,
    *,
    output_path: str | None = None,
    artifact_path: str | None = None,
    report_path: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    saved_root: Path | None = None,
) -> list[SavePlan]:
    """Save plans for what a finished (or stopped) training job produced."""

    metrics = metrics or {}
    plans: list[SavePlan | None] = []
    if mode == "data_collection":
        plans.append(plan_dataset(metrics.get("dataset_path") or output_path or "", saved_root=saved_root))
    elif mode == "module_training":
        plans.append(
            plan_modules(
                metrics.get("intent_artifact_path"),
                metrics.get("emotion_artifact_path"),
                report_path=report_path,
                saved_root=saved_root,
            )
        )
    elif mode == "main_ai_training":
        if artifact_path:
            plans.append(plan_main_ai(artifact_path, report_path=report_path, saved_root=saved_root))
    elif mode == "inference_refinement":
        # The refined model (with the run's report) and, when a round was
        # accepted, its module pair. The collected main1+dummy4 datasets stay in
        # the local dataset folder and can be saved from there.
        if artifact_path:
            plans.append(plan_main_ai(artifact_path, report_path=report_path, saved_root=saved_root))
        plans.append(
            plan_modules(
                metrics.get("intent_artifact_path"), metrics.get("emotion_artifact_path"), saved_root=saved_root
            )
        )
    return [plan for plan in plans if plan is not None]


# --------------------------------------------------------------------- executing


def _retag(info: ArtifactInfo, name: str, notes: str) -> ArtifactInfo:
    info.name = name
    info.notes = notes
    return info


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _json_with_info(path: Path, info: ArtifactInfo) -> None:
    write_info(path, info)


def execute_save(plan: SavePlan, name: str, notes: str | None = None) -> SaveResult:
    """Copy (never move) the planned items into ``saved/`` under ``name``.

    The copy gets the chosen ``name`` and ``notes`` in its info block and a
    fresh sidecar. An existing target is never overwritten
    (:class:`SaveNameExistsError`). Pair copies (intent + emotion) keep the same
    info id so they stay paired and compatible.
    """

    clean = sanitize_user_name(name)
    existing = plan.conflicts(clean)
    if existing:
        raise SaveNameExistsError(str(existing[0]))
    note_text = plan.notes if notes is None else notes
    written: list[Path] = []
    staging: list[Path] = []
    try:
        if plan.category == CATEGORY_DATASET:
            item = plan.items[0]
            final = plan.target_root / clean
            temporary = plan.target_root / f".{clean}.partial"
            staging.append(temporary)
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(item.source, temporary, ignore=shutil.ignore_patterns("*.tmp", ".*.tmp"))
            info = _retag(read_info(item.source) or _fallback_info(KIND_DATASET, item.source), clean, note_text)
            write_info(temporary, info)
            summary = temporary / "collection_summary.json"
            if summary.is_file():
                write_info(summary, info)
            os.replace(temporary, final)
            written.append(final)
        elif plan.category == CATEGORY_MODULES:
            final = plan.target_root / clean
            temporary = plan.target_root / f".{clean}.partial"
            staging.append(temporary)
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            report_item = None
            for item in plan.items:
                if item.kind == KIND_REPORT:
                    report_item = item
                    continue
                target = temporary / item.relative
                _copy_file(item.source, target)
                info = read_info(item.source) or _fallback_info(item.kind, item.source)
                write_info(target, _retag(info, clean, note_text))
            os.replace(temporary, final)
            written.append(final)
            if report_item is not None:
                written.append(_save_report(report_item, clean, note_text, plan.reports_root))
        else:
            for item in plan.items:
                if item.kind == KIND_REPORT:
                    written.append(_save_report(item, clean, note_text, plan.reports_root))
                    continue
                final = plan.target_root / f"{clean}{item.source.suffix}"
                temporary = plan.target_root / f".{clean}{item.source.suffix}.partial"
                staging.append(temporary)
                _copy_file(item.source, temporary)
                os.replace(temporary, final)
                info = read_info(item.source) or _fallback_info(item.kind, item.source)
                write_info(final, _retag(info, clean, note_text))
                written.append(final)
    except BaseException:
        for path in staging:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        raise
    total = 0
    for path in written:
        total += sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
    return SaveResult(plan.category, clean, tuple(written), total)


def _save_report(item: SaveItem, name: str, notes: str, reports_root: Path) -> Path:
    target = reports_root / f"{name}.json"
    _copy_file(item.source, target)
    info = read_info(item.source) or _fallback_info(KIND_REPORT, item.source)
    write_info(target, _retag(info, name, notes))
    return target


__all__ = [
    "CATEGORY_DATASET",
    "CATEGORY_MAIN_AI",
    "CATEGORY_MODULES",
    "GITHUB_LIMIT_BYTES",
    "GITHUB_WARN_BYTES",
    "LEVEL_OK",
    "LEVEL_STRONG",
    "LEVEL_WARN",
    "SaveItem",
    "SaveNameExistsError",
    "SavePlan",
    "SaveResult",
    "execute_save",
    "find_report_for",
    "human_size",
    "plan_dataset",
    "plan_main_ai",
    "plan_modules",
    "plans_for_training_result",
    "size_level",
]
