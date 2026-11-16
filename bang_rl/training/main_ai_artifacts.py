"""Saved main-AI models (.bangmodel): discovery and performance summary.

Main-AI training saves its best evaluated model to ``<ARTIFACTS_DIR>/<job_id>.bangmodel``
(``bang_rl/models/main_ai`` is scanned too). The evaluation result, baselines,
normalized scores and training counters are stored in the artifact metadata, so
the performance can be shown by reading only the manifest and metadata, never
the weights.

This module is headless (no Qt) so the GUI and tests share the same logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from bang_rl.artifact_info import read_info
from bang_rl.paths import ARTIFACTS_DIR, SAVED_MAIN_AI_DIR

from .final_inference_artifact import FINAL_ARTIFACT_EXTENSION, read_final_artifact_metadata
from .module_artifacts import MODULE_MODEL_ROOT

MAIN_AI_MODEL_DIRS: tuple[Path, ...] = (ARTIFACTS_DIR, MODULE_MODEL_ROOT / "main_ai", SAVED_MAIN_AI_DIR)
ROLE_NAMES = ("sheriff", "deputy", "outlaw", "traitor")
BASELINE_NAMES = ("dummy", "random")


@dataclass(frozen=True, slots=True)
class SavedMainAiModel:
    """One ``.bangmodel`` file found on disk."""

    path: Path
    modified: float
    # Filled from the embedded info block; empty for a legacy file.
    run_id: str = ""
    name: str = ""
    notes: str = ""
    created_at: str = ""
    saved: bool = False

    @property
    def job_id(self) -> str:
        return self.run_id or self.path.stem

    @property
    def short_id(self) -> str:
        return self.job_id.split("-")[0]

    @property
    def display_name(self) -> str:
        return self.name or self.path.stem

    @property
    def modified_text(self) -> str:
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")

    @property
    def tooltip(self) -> str:
        lines = [self.display_name, self.created_at or self.modified_text]
        if self.notes.strip():
            lines += ["", self.notes.strip()]
        return "\n".join(lines)


def _created_epoch(created_at: str, fallback: float) -> float:
    try:
        return datetime.fromisoformat(created_at).timestamp()
    except (TypeError, ValueError):
        return fallback


def list_saved_main_ai_models(
    directories: Sequence[str | Path] | None = None,
) -> tuple[SavedMainAiModel, ...]:
    """Return ``.bangmodel`` files from the model folders, newest first.

    Files are found by extension (any name). The default folders include the
    user-kept ``saved/main_ai`` folder; when the same run exists as a local
    original and as a saved copy, only the local original is listed.
    """

    use_default = directories is None
    roots = [Path(item) for item in (directories if directories is not None else MAIN_AI_MODEL_DIRS)]
    seen: set[Path] = set()
    found: dict[str, SavedMainAiModel] = {}
    anonymous: list[SavedMainAiModel] = []
    for root in roots:
        if not root.is_dir():
            continue
        saved_root = use_default and root == SAVED_MAIN_AI_DIR
        pattern = f"*{FINAL_ARTIFACT_EXTENSION}"
        for path in sorted(root.rglob(pattern) if saved_root else root.glob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            stamp = path.stat().st_mtime
            info = read_info(path)
            model = SavedMainAiModel(
                path=path,
                modified=_created_epoch(info.created_at, stamp) if info else stamp,
                run_id=info.id if info else "",
                name=info.name if info else "",
                notes=info.notes if info else "",
                created_at=info.created_at if info else "",
                saved=saved_root,
            )
            if not model.run_id:
                anonymous.append(model)
            elif model.run_id not in found or (found[model.run_id].saved and not model.saved):
                found[model.run_id] = model
    return tuple(sorted([*found.values(), *anonymous], key=lambda item: item.modified, reverse=True))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main_ai_model_compatibility(path: str | Path) -> tuple[bool, str | None]:
    """Whether a ``.bangmodel`` can be played, collected with and refined.

    Reads only the manifest and metadata. The reason is ``"unreadable"`` or
    ``"emotion_schema"`` (its emotion module does not use the dummies' 9-entry
    labels), or None when usable.
    """

    from bang_rl.data.game_collection import FORMULA_LABEL_SCHEMA

    try:
        _manifest, metadata = read_final_artifact_metadata(path)
    except Exception:  # noqa: BLE001 - a broken file is simply not usable
        return False, "unreadable"
    schema = metadata.get("emotion_schema_version") or _mapping(metadata.get("versions")).get("emotion_schema_version")
    if schema != FORMULA_LABEL_SCHEMA:
        return False, "emotion_schema"
    return True, None


def _role_rates(summary: Mapping[str, Any]) -> dict[str, float | None]:
    by_role = _mapping(summary.get("by_role"))
    return {name: _number(_mapping(by_role.get(name)).get("win_rate")) for name in ROLE_NAMES}


def summarize_main_ai_artifact(path: str | Path) -> dict[str, Any]:
    """Summarize one saved main-AI model from its metadata only.

    An unreadable or incompatible file is reported in ``error``, not raised.
    """

    source = Path(path)
    info = read_info(source) if source.is_file() else None
    summary: dict[str, Any] = {
        "path": str(source),
        "job_id": info.id if info and info.id else source.stem,
        "name": info.name if info and info.name else source.stem,
        "notes": info.notes if info else "",
    }
    try:
        stamp = source.stat().st_mtime
        if info is not None:
            stamp = _created_epoch(info.created_at, stamp)
        summary["modified"] = datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")
        _manifest, metadata = read_final_artifact_metadata(source)
    except Exception as exc:  # noqa: BLE001 - a broken file must not break the list
        summary["error"] = str(exc)
        return summary
    main = _mapping(metadata.get("main_ai_training"))
    evaluation = _mapping(main.get("evaluation")) or _mapping(_mapping(metadata.get("evaluation")).get("result"))
    scores = _mapping(main.get("normalized_scores"))
    baselines = _mapping(main.get("baselines"))
    source_info = _mapping(main.get("source"))
    counters = _mapping(_mapping(metadata.get("training")).get("counters"))
    summary.update(
        is_main_ai_training=bool(main),
        created_at=metadata.get("created_at"),
        training_games=main.get("training_games"),
        ppo_updates=main.get("ppo_updates", counters.get("ppo_updates")),
        training_seconds=_number(metadata.get("total_training_time_seconds")),
        source_origin=source_info.get("origin"),
        source_module_job_id=source_info.get("module_job_id"),
        source_intent_path=source_info.get("intent_path"),
        source_emotion_path=source_info.get("emotion_path"),
        eval_games=evaluation.get("games"),
        eval_win_rate=_number(evaluation.get("win_rate")),
        eval_by_role=_role_rates(evaluation),
        role_accuracy=_number(evaluation.get("role_accuracy")),
        emotion_mse=_number(evaluation.get("emotion_mse")),
        truncated_games=evaluation.get("truncated_games"),
        normalized_mean=_number(scores.get("mean")),
        normalized_by_role={
            name: _number(_mapping(scores.get("by_role")).get(name)) for name in ROLE_NAMES
        },
        baselines={
            name: {
                "win_rate": _number(_mapping(baselines.get(name)).get("win_rate")),
                "by_role": _role_rates(_mapping(baselines.get(name))),
            }
            for name in BASELINE_NAMES
        },
    )
    return summary


__all__ = [
    "BASELINE_NAMES",
    "MAIN_AI_MODEL_DIRS",
    "ROLE_NAMES",
    "SavedMainAiModel",
    "list_saved_main_ai_models",
    "main_ai_model_compatibility",
    "summarize_main_ai_artifact",
]
