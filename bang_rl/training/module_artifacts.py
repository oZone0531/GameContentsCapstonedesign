"""Saved Phase-2 inference modules: discovery, performance summary, resume loading.

Module training saves one ``.phase2model`` per module and job under
``bang_rl/models/intent/intent-<job_id>.phase2model`` and
``bang_rl/models/emotion/emotion-<job_id>.phase2model``. Each file holds the
module weights plus metadata with the per-epoch history and the final test
metrics, so its performance can be shown without any report file.

This module is headless (no Qt) so the GUI and tests share the same logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from bang_rl.artifact_info import (
    ArtifactInfo,
    find_json_by_id,
    info_from_embedded,
    info_from_payload,
)
from bang_rl.exceptions import TrainingError
from bang_rl.paths import PROJECT_ROOT, REPORTS_DIR, SAVED_MODULES_DIR, SAVED_REPORTS_DIR

MODULE_MODEL_ROOT = PROJECT_ROOT / "bang_rl" / "models"
ARTIFACT_SUFFIX = ".phase2model"
_KIND_PREFIX = {"intent": "intent-", "emotion": "emotion-"}
_KIND_TYPE = {
    "phase2_intent_inference": "intent",
    "phase2_emotion_inference": "emotion",
}


@dataclass(frozen=True, slots=True)
class SavedModuleRun:
    """The intent and/or emotion module saved by one module-training job."""

    job_id: str
    intent_path: Path | None
    emotion_path: Path | None
    modified: float
    # Readable name (naming rule) and the user's free-text note (비고). Both are
    # empty for a legacy file that has no embedded info.
    name: str = ""
    notes: str = ""
    kind_label: str = ""
    created_at: str = ""
    saved: bool = False

    @property
    def modified_text(self) -> str:
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")

    @property
    def short_id(self) -> str:
        return self.job_id.split("-")[0]

    @property
    def display_name(self) -> str:
        return self.name or self.job_id

    @property
    def tooltip(self) -> str:
        lines = [self.display_name, self.created_at or self.modified_text]
        if self.notes.strip():
            lines += ["", self.notes.strip()]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _ModuleFile:
    path: Path
    kind: str
    run_id: str
    info: ArtifactInfo | None
    mtime: float
    saved: bool


_identity_cache: dict[tuple[str, int, int], tuple[str | None, str, ArtifactInfo | None]] = {}


def _legacy_identity(path: Path) -> tuple[str | None, str]:
    """Kind/job id guessed from a legacy ``intent-<id>.phase2model`` file name."""

    stem = path.name[: -len(ARTIFACT_SUFFIX)]
    for kind, prefix in _KIND_PREFIX.items():
        if stem.startswith(prefix) and len(stem) > len(prefix):
            return kind, stem[len(prefix):]
    return None, stem


def _module_identity(path: Path) -> tuple[str | None, str, ArtifactInfo | None]:
    """``(kind, run id, info)`` read from inside the file, never from its name.

    The name is only a last resort for files that cannot be read.
    """

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _identity_cache.get(key)
    if cached is not None:
        return cached
    legacy_kind, legacy_id = _legacy_identity(path)
    kind: str | None = legacy_kind
    run_id = legacy_id
    info: ArtifactInfo | None = None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - listed anyway so the summary can report the error
        payload = None
    if isinstance(payload, Mapping):
        kind = _KIND_TYPE.get(str(payload.get("artifact_type")), kind)
        info = info_from_embedded(path, info_from_payload(payload))
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        run_meta = metadata.get("run_metadata") if isinstance(metadata.get("run_metadata"), Mapping) else {}
        snapshot = metadata.get("job_snapshot") if isinstance(metadata.get("job_snapshot"), Mapping) else {}
        found = (info.id if info and info.id else None) or run_meta.get("job_id") or snapshot.get("job_id")
        if found:
            run_id = str(found)
    else:
        info = info_from_embedded(path, None)
        if info is not None and info.id:
            run_id = info.id
    if len(_identity_cache) > 512:
        _identity_cache.clear()
    result = (kind, run_id, info)
    _identity_cache[key] = result
    return result


def _created_epoch(info: ArtifactInfo | None, fallback: float) -> float:
    if info is not None and info.created_at:
        try:
            return datetime.fromisoformat(info.created_at).timestamp()
        except ValueError:
            pass
    return fallback


def _module_files(model_root: Path, saved_root: Path | None) -> list[_ModuleFile]:
    """Every ``*.phase2model`` under the model folders, whatever it is called."""

    candidates: list[tuple[Path, bool]] = []
    for folder in (model_root, model_root / "intent", model_root / "emotion"):
        if folder.is_dir():
            candidates += [(path, False) for path in sorted(folder.glob(f"*{ARTIFACT_SUFFIX}"))]
    if saved_root is not None and saved_root.is_dir():
        candidates += [(path, True) for path in sorted(saved_root.rglob(f"*{ARTIFACT_SUFFIX}"))]
    found: list[_ModuleFile] = []
    for path, saved in candidates:
        if not path.is_file():
            continue
        kind, run_id, info = _module_identity(path)
        if kind is None or not run_id:
            continue
        found.append(_ModuleFile(path, kind, run_id, info, path.stat().st_mtime, saved))
    return found


def list_saved_module_runs(
    model_root: str | Path | None = None,
    *,
    saved_root: str | Path | None = None,
) -> tuple[SavedModuleRun, ...]:
    """Return saved module runs grouped by run id, newest first.

    Files are found by extension and paired by the run id stored *inside* them
    (``info.id``, falling back to the training job id in the metadata), so
    renaming a file in the file explorer changes nothing. Copies kept in
    ``saved/modules`` are included; when the same run exists in both places the
    local original wins.
    """

    root = Path(model_root) if model_root is not None else MODULE_MODEL_ROOT
    if saved_root is None and model_root is None:
        saved_root = SAVED_MODULES_DIR
    saved = Path(saved_root) if saved_root is not None else None
    groups: dict[str, dict[str, _ModuleFile]] = {}
    for item in _module_files(root, saved):
        slot = groups.setdefault(item.run_id, {})
        current = slot.get(item.kind)
        # Prefer the local original over a saved copy, then the newest file.
        if current is None or (item.saved, -item.mtime) < (current.saved, -current.mtime):
            slot[item.kind] = item
    runs: list[SavedModuleRun] = []
    for run_id, parts in groups.items():
        files = list(parts.values())
        primary = parts.get("intent") or parts.get("emotion")
        info = primary.info if primary is not None else None
        name = info.name if info is not None else ""
        if name:
            name = name.replace("_intent_", "_modules_", 1).replace("_emotion_", "_modules_", 1)
        runs.append(
            SavedModuleRun(
                job_id=run_id,
                intent_path=parts["intent"].path if "intent" in parts else None,
                emotion_path=parts["emotion"].path if "emotion" in parts else None,
                modified=max(_created_epoch(item.info, item.mtime) for item in files),
                name=name,
                notes=info.notes if info is not None else "",
                kind_label=info.kind_label if info is not None else "",
                created_at=info.created_at if info is not None else "",
                saved=all(item.saved for item in files),
            )
        )
    return tuple(sorted(runs, key=lambda run: run.modified, reverse=True))


def module_run_infos(run: SavedModuleRun) -> dict[str, ArtifactInfo | None]:
    """Info block of each saved module of a run (cached, no second file read).

    Keys are ``"intent"`` and ``"emotion"``; a missing module or a legacy file
    without info maps to ``None``.
    """

    infos: dict[str, ArtifactInfo | None] = {"intent": None, "emotion": None}
    for kind, path in (("intent", run.intent_path), ("emotion", run.emotion_path)):
        if path is None:
            continue
        try:
            infos[kind] = _module_identity(Path(path))[2]
        except OSError:
            infos[kind] = None
    return infos


def find_report_path(job_id: str, reports_dir: str | Path | None = None) -> Path | None:
    """Report of a job: by ``info.id`` first, then the legacy ``<job_id>.json`` name."""

    directory = Path(reports_dir if reports_dir is not None else REPORTS_DIR)
    roots = [directory]
    if reports_dir is None:
        roots.append(SAVED_REPORTS_DIR)
    found = find_json_by_id(roots, job_id)
    if found is not None:
        return found
    legacy = directory / f"{job_id}.json"
    return legacy if legacy.is_file() else None


def _read_artifact(path: str | Path) -> tuple[str, Mapping[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise TrainingError(f"module artifact not found: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TrainingError(f"cannot read module artifact {source}: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
        raise TrainingError(f"{source.name} is not a Phase-2 module artifact")
    kind = _KIND_TYPE.get(str(payload.get("artifact_type")))
    if kind is None:
        raise TrainingError(
            f"{source.name} has unsupported artifact type {payload.get('artifact_type')!r}"
        )
    return kind, payload["metadata"]


def _record_at(history: list[Mapping[str, Any]], epoch: int) -> Mapping[str, Any]:
    for record in history:
        if int(record.get("epoch", -1)) == epoch:
            return record
    return history[-1] if history else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _numbers(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    return [float(item) for item in value if _number(item) is not None]


def summarize_module_artifact(path: str | Path) -> dict[str, Any]:
    """Summarize one saved module's training and final performance."""

    kind, metadata = _read_artifact(path)
    training = metadata.get("training") if isinstance(metadata.get("training"), Mapping) else {}
    history = [item for item in training.get("metrics", ()) if isinstance(item, Mapping)]
    epochs = int(training.get("epochs_completed", len(history)) or 0)
    prior = int(training.get("prior_epochs", 0) or 0)
    best_epoch = int(training.get("best_epoch", 0) or 0)
    test = training.get("test_metrics") if isinstance(training.get("test_metrics"), Mapping) else {}
    # The saved weights are the best-validation weights, so report validation
    # numbers from that epoch. Best epoch 0 means the resumed starting model
    # was never beaten, so there is no new validation record to show.
    best = _record_at(history, best_epoch) if best_epoch > 0 else {}
    split = metadata.get("split") if isinstance(metadata.get("split"), Mapping) else {}
    summary: dict[str, Any] = {
        "kind": kind,
        "path": str(path),
        "epochs_completed": epochs,
        "prior_epochs": prior,
        "total_epochs": int(training.get("total_epochs", prior + epochs) or 0),
        "best_epoch": best_epoch,
        "best_validation_loss": _number(training.get("best_validation_loss")),
        "resumed_from": training.get("resumed_from"),
        "batch_size": training.get("batch_size"),
        "train_episodes": split.get("train_episode_count"),
        "validation_episodes": split.get("validation_episode_count"),
        "test_episodes": split.get("test_episode_count"),
        "test_accuracy": _number(test.get("accuracy")),
        "test_loss": _number(test.get("loss")),
    }
    if kind == "intent":
        names = test.get("role_names") or best.get("validation_role_names") or ()
        summary.update(
            names=[str(name) for name in names],
            test_by_class=_numbers(test.get("accuracy_by_role")),
            validation_accuracy=_number(best.get("validation_accuracy")),
            validation_by_class=_numbers(best.get("validation_accuracy_by_role")),
        )
    else:
        schema = metadata.get("emotion_schema") if isinstance(metadata.get("emotion_schema"), Mapping) else {}
        names = schema.get("names") or training.get("emotion_dimension_names") or ()
        summary.update(
            names=[str(name) for name in names],
            schema_version=schema.get("version") or metadata.get("emotion_schema_version"),
            tolerance=_number(training.get("accuracy_tolerance")),
            test_mse=_number(test.get("mse")),
            test_by_class=_numbers(test.get("accuracy_by_dimension")),
            validation_accuracy=_number(best.get("validation_emotion_accuracy")),
            validation_mse=_number(best.get("validation_mse")),
            validation_by_class=_numbers(best.get("validation_emotion_accuracy_by_dimension")),
        )
    return summary


def summarize_module_run(
    run: SavedModuleRun, *, reports_dir: str | Path | None = None
) -> dict[str, Any]:
    """Summarize both modules of a run. A broken file is reported, not raised."""

    report = find_report_path(run.job_id, reports_dir)
    result: dict[str, Any] = {
        "job_id": run.job_id,
        "name": run.display_name,
        "notes": run.notes,
        "modified": run.modified_text,
        "report_path": str(report) if report is not None else None,
    }
    for kind, path in (("intent", run.intent_path), ("emotion", run.emotion_path)):
        if path is None:
            result[kind] = None
            continue
        try:
            result[kind] = summarize_module_artifact(path)
        except TrainingError as exc:
            result[kind] = {"kind": kind, "path": str(path), "error": str(exc)}
    return result


MAIN_AI_INTENT_ROLE_COUNT = 4
# Reasons returned by :func:`main_ai_compatibility` when a run cannot be used.
MAIN_AI_REASON_MISSING = "missing_module"
MAIN_AI_REASON_INTENT_ROLES = "intent_roles"
MAIN_AI_REASON_EMOTION_SCHEMA = "emotion_schema"
MAIN_AI_REASON_UNREADABLE = "unreadable"

_compat_cache: dict[tuple[str, int, int], tuple[Any, ...]] = {}


def _architecture(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    value = metadata.get("architecture")
    return value if isinstance(value, Mapping) else {}


def _module_compat(path: Path, kind: str) -> tuple[Any, ...]:
    """Cheap per-file facts for the main-AI check, cached by file stamp.

    Weights are inside the same pickle as the metadata, so the file is read
    once per (path, mtime) and the facts are kept.
    """

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _compat_cache.get(key)
    if cached is not None:
        return cached
    try:
        found_kind, metadata = _read_artifact(path)
        if found_kind != kind:
            raise TrainingError(f"{path.name} is a {found_kind} module")
        architecture = _architecture(metadata)
        if kind == "intent":
            facts: tuple[Any, ...] = (True, architecture.get("role_count"))
        else:
            schema = metadata.get("emotion_schema") if isinstance(metadata.get("emotion_schema"), Mapping) else {}
            version = schema.get("version") or architecture.get("emotion_schema_version") or metadata.get("emotion_schema_version")
            count = schema.get("count") if schema.get("count") is not None else architecture.get("emotion_count")
            facts = (True, version, count)
    except TrainingError:
        facts = (False,)
    if len(_compat_cache) > 256:
        _compat_cache.clear()
    _compat_cache[key] = facts
    return facts


def main_ai_compatibility(run: SavedModuleRun) -> tuple[bool, str | None]:
    """Whether a saved module run can feed main-AI training.

    The intent module must predict 4 role classes and the emotion module must
    use the dummies' formula label schema with 9 entries (the same contract
    ``main_ai_training.validate_emotion_contract`` enforces after loading).
    Returns ``(True, None)`` or ``(False, reason_code)``.
    """

    from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
    from bang_rl.data.game_collection import FORMULA_LABEL_SCHEMA

    if run.intent_path is None or run.emotion_path is None:
        return False, MAIN_AI_REASON_MISSING
    try:
        intent = _module_compat(Path(run.intent_path), "intent")
        emotion = _module_compat(Path(run.emotion_path), "emotion")
    except OSError:
        return False, MAIN_AI_REASON_UNREADABLE
    if not intent[0] or not emotion[0]:
        return False, MAIN_AI_REASON_UNREADABLE
    if intent[1] != MAIN_AI_INTENT_ROLE_COUNT:
        return False, MAIN_AI_REASON_INTENT_ROLES
    if str(emotion[1]) != FORMULA_LABEL_SCHEMA or emotion[2] != len(LABEL_NAMES):
        return False, MAIN_AI_REASON_EMOTION_SCHEMA
    return True, None


def artifact_total_epochs(metadata: Mapping[str, Any]) -> int:
    """Cumulative epochs already behind a saved module (for resumed runs)."""

    training = metadata.get("training") if isinstance(metadata.get("training"), Mapping) else {}
    if "total_epochs" in training:
        return int(training.get("total_epochs") or 0)
    return int(training.get("epochs_completed", 0) or 0)


def load_intent_for_resume(path: str | Path) -> tuple[Any, Mapping[str, Any]]:
    """Load a saved intent module to continue training (weights only)."""

    from bang_rl.training.sl_trainer import load_intent_artifact

    source = Path(path)
    if not source.is_file():
        raise TrainingError(f"intent module to continue from was not found: {source}")
    module, metadata = load_intent_artifact(source)
    if int(module.role_count) != 4:
        raise TrainingError(
            f"{source.name} predicts {module.role_count} role classes; module training uses 4"
        )
    return module, metadata


def load_emotion_for_resume(path: str | Path) -> tuple[Any, Mapping[str, Any]]:
    """Load a saved emotion module to continue training (weights only).

    The emotion trainer rejects a module whose schema or dimension does not
    match the current cache, so an old incompatible model fails with a clear
    message instead of training on mismatched targets.
    """

    from bang_rl.training.emotion_sl_trainer import load_emotion_artifact

    source = Path(path)
    if not source.is_file():
        raise TrainingError(f"emotion module to continue from was not found: {source}")
    return load_emotion_artifact(source)


__all__ = [
    "MODULE_MODEL_ROOT",
    "MAIN_AI_REASON_EMOTION_SCHEMA",
    "MAIN_AI_REASON_INTENT_ROLES",
    "MAIN_AI_REASON_MISSING",
    "MAIN_AI_REASON_UNREADABLE",
    "SavedModuleRun",
    "artifact_total_epochs",
    "find_report_path",
    "list_saved_module_runs",
    "main_ai_compatibility",
    "module_run_infos",
    "load_emotion_for_resume",
    "load_intent_for_resume",
    "summarize_module_artifact",
    "summarize_module_run",
]
