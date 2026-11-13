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

from bang_rl.exceptions import TrainingError
from bang_rl.paths import PROJECT_ROOT, REPORTS_DIR

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

    @property
    def modified_text(self) -> str:
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")

    @property
    def short_id(self) -> str:
        return self.job_id.split("-")[0]


def list_saved_module_runs(model_root: str | Path | None = None) -> tuple[SavedModuleRun, ...]:
    """Return saved module runs grouped by job id, newest first."""

    root = Path(model_root) if model_root is not None else MODULE_MODEL_ROOT
    found: dict[str, dict[str, Path]] = {}
    for kind, prefix in _KIND_PREFIX.items():
        folder = root / kind
        if not folder.is_dir():
            continue
        for path in folder.glob(f"{prefix}*{ARTIFACT_SUFFIX}"):
            if not path.is_file():
                continue
            job_id = path.name[len(prefix) : -len(ARTIFACT_SUFFIX)]
            if job_id:
                found.setdefault(job_id, {})[kind] = path
    runs = [
        SavedModuleRun(
            job_id=job_id,
            intent_path=paths.get("intent"),
            emotion_path=paths.get("emotion"),
            modified=max(path.stat().st_mtime for path in paths.values()),
        )
        for job_id, paths in found.items()
    ]
    return tuple(sorted(runs, key=lambda run: run.modified, reverse=True))


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

    report = Path(reports_dir if reports_dir is not None else REPORTS_DIR) / f"{run.job_id}.json"
    result: dict[str, Any] = {
        "job_id": run.job_id,
        "modified": run.modified_text,
        "report_path": str(report) if report.is_file() else None,
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
    "SavedModuleRun",
    "artifact_total_epochs",
    "list_saved_module_runs",
    "load_emotion_for_resume",
    "load_intent_for_resume",
    "summarize_module_artifact",
    "summarize_module_run",
]
