"""Saved model browser: Phase-2 module performance ('continue training') and
saved main-AI (.bangmodel) training results."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..i18n import tr
from ..qt_compat import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.{digits}f}"


def _by_class(names: Sequence[str], values: Sequence[float]) -> str:
    if not values:
        return ""
    labels = list(names) + [f"#{index}" for index in range(len(names), len(values))]
    return " · ".join(f"{label} {_fmt(value)}" for label, value in zip(labels, values))


def format_module_summary(summary: Mapping[str, Any] | None) -> list[str]:
    """Readable lines for one module summary (see summarize_module_artifact)."""

    if summary is None:
        return ["  " + tr("perf.not_saved")]
    if summary.get("error"):
        return ["  " + tr("perf.error", error=summary["error"])]
    lines = ["  " + tr("perf.epochs", run=summary.get("epochs_completed", 0), total=summary.get("total_epochs", 0))]
    if summary.get("resumed_from"):
        lines.append("  " + tr("perf.resumed_from", name=Path(str(summary["resumed_from"])).name))
    if int(summary.get("best_epoch") or 0) == 0 and summary.get("resumed_from"):
        lines.append("  " + tr("perf.best_is_start"))
    else:
        lines.append("  " + tr("perf.best_epoch", epoch=summary.get("best_epoch", 0)))
    names = list(summary.get("names") or ())
    if summary.get("kind") == "emotion":
        lines.append("  " + tr(
            "perf.test_emotion",
            accuracy=_fmt(summary.get("test_accuracy")),
            tolerance=_fmt(summary.get("tolerance"), 2),
            mse=_fmt(summary.get("test_mse"), 4),
        ))
    else:
        lines.append("  " + tr(
            "perf.test_intent",
            accuracy=_fmt(summary.get("test_accuracy")),
            loss=_fmt(summary.get("test_loss")),
        ))
    by_class = _by_class(names, list(summary.get("test_by_class") or ()))
    if by_class:
        lines.append("    " + by_class)
    if summary.get("validation_accuracy") is not None:
        lines.append("  " + tr("perf.validation", accuracy=_fmt(summary.get("validation_accuracy"))))
        by_class = _by_class(names, list(summary.get("validation_by_class") or ()))
        if by_class:
            lines.append("    " + by_class)
    lines.append("  " + tr(
        "perf.split",
        train=summary.get("train_episodes", "—"),
        validation=summary.get("validation_episodes", "—"),
        test=summary.get("test_episodes", "—"),
    ))
    return lines


def _name_notes_lines(summary: Mapping[str, Any]) -> list[str]:
    """Readable name and the user's note (if any) from an artifact summary."""

    lines: list[str] = []
    name = summary.get("name")
    if name and name != summary.get("job_id"):
        lines.append(tr("perf.name", name=name))
    notes = str(summary.get("notes") or "").strip()
    if notes:
        lines.append(tr("perf.notes_line", notes=notes))
    return lines


def _note_snippet(notes: str, limit: int = 28) -> str:
    first = " ".join(str(notes or "").split())
    if not first:
        return ""
    return " · " + (first if len(first) <= limit else first[: limit - 1] + "…")


def format_module_run_summary(run_summary: Mapping[str, Any]) -> str:
    """Full text shown for one saved run (see summarize_module_run)."""

    lines = [
        tr("perf.job", job=run_summary.get("job_id", "—")),
        tr("perf.saved_at", time=run_summary.get("modified", "—")),
    ]
    lines += _name_notes_lines(run_summary)
    if run_summary.get("report_path"):
        lines.append(tr("perf.report", path=run_summary["report_path"]))
    lines += ["", tr("perf.intent_section")]
    lines += format_module_summary(run_summary.get("intent"))
    lines += ["", tr("perf.emotion_section")]
    lines += format_module_summary(run_summary.get("emotion"))
    return "\n".join(lines)


def format_main_ai_model_summary(summary: Mapping[str, Any]) -> str:
    """Full text for one saved main-AI model (see summarize_main_ai_artifact)."""

    from ..main_ai_view import ROLE_NAMES, fmt, role_label

    lines = [
        tr("perf.job", job=summary.get("job_id", "—")),
        tr("perf.saved_at", time=summary.get("modified", "—")),
    ]
    lines += _name_notes_lines(summary)
    lines += [
        tr("perf.main_ai_file", path=summary.get("path", "—")),
        "",
        tr("perf.main_ai_section"),
    ]
    if summary.get("error"):
        lines.append("  " + tr("perf.error", error=summary["error"]))
        return "\n".join(lines)
    if not summary.get("is_main_ai_training"):
        lines.append("  " + tr("perf.main_ai_not_trained"))
        return "\n".join(lines)
    lines.append("  " + tr(
        "perf.main_ai_training",
        games=summary.get("training_games", "—"),
        updates=summary.get("ppo_updates", "—"),
        seconds=fmt(summary.get("training_seconds"), 0),
    ))
    origin = summary.get("source_origin")
    job = summary.get("source_module_job_id")
    lines.append("  " + tr(
        "perf.main_ai_source",
        origin=tr(f"perf.main_ai_origin.{origin}") if origin else "—",
        job=tr("perf.main_ai_source_job", job=job) if job else "",
    ))
    baselines = summary.get("baselines") or {}
    dummy = baselines.get("dummy") or {}
    random = baselines.get("random") or {}
    lines.append("  " + tr(
        "perf.main_ai_eval",
        games=summary.get("eval_games", "—"),
        win=fmt(summary.get("eval_win_rate")),
        score=fmt(summary.get("normalized_mean")),
    ))
    lines.append("  " + tr(
        "perf.main_ai_eval_extra",
        accuracy=fmt(summary.get("role_accuracy")),
        mse=fmt(summary.get("emotion_mse"), 4),
        truncated=summary.get("truncated_games", "—"),
    ))
    lines.append("  " + tr(
        "perf.main_ai_baselines", dummy=fmt(dummy.get("win_rate")), random=fmt(random.get("win_rate"))
    ))
    by_role = summary.get("eval_by_role") or {}
    dummy_roles = dummy.get("by_role") or {}
    random_roles = random.get("by_role") or {}
    parts = [
        f"{role_label(name)} {fmt(by_role.get(name), 2)} / {fmt(dummy_roles.get(name), 2)} / {fmt(random_roles.get(name), 2)}"
        for name in ROLE_NAMES
    ]
    lines.append("  " + tr("mainai.by_role") + " " + " · ".join(parts))
    normalized = summary.get("normalized_by_role") or {}
    lines.append("  " + tr("perf.main_ai_normalized"))
    lines.append("    " + " · ".join(f"{role_label(name)} {fmt(normalized.get(name), 2)}" for name in ROLE_NAMES))
    return "\n".join(lines)


def main_ai_model_label(model: Any) -> str:
    """One-line list label for a SavedMainAiModel."""

    return tr("perf.main_ai_item", time=model.modified_text, id=model.short_id) + _note_snippet(
        getattr(model, "notes", "")
    )


def _module_parts(run: Any) -> str:
    if run.intent_path is not None and run.emotion_path is not None:
        return tr("perf.parts_both")
    if run.intent_path is not None:
        return tr("perf.parts_intent")
    return tr("perf.parts_emotion")


def module_run_label(run: Any) -> str:
    """One-line list label for a SavedModuleRun."""

    return f"{run.modified_text} · {run.short_id} · {_module_parts(run)}" + _note_snippet(
        getattr(run, "notes", "")
    )


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class ModulePerformanceDialog(QDialog):
        """List saved module runs and show their final performance.

        "Continue training from this model" closes the dialog with
        ``selected_job_id`` set so the training screen can preselect it.
        """

        def __init__(
            self,
            *,
            model_root: str | Path | None = None,
            reports_dir: str | Path | None = None,
            parent: QWidget | None = None,
            list_runs: Callable[..., Sequence[Any]] | None = None,
            summarize: Callable[..., Mapping[str, Any]] | None = None,
            list_main_ai: Callable[..., Sequence[Any]] | None = None,
            summarize_main_ai: Callable[..., Mapping[str, Any]] | None = None,
            main_ai_dirs: Sequence[str | Path] | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            from bang_rl.training.main_ai_artifacts import (
                list_saved_main_ai_models,
                summarize_main_ai_artifact,
            )
            from bang_rl.training.module_artifacts import (
                list_saved_module_runs,
                summarize_module_run,
            )

            self._model_root = model_root
            self._reports_dir = reports_dir
            # A custom model root (tests, other installs) scopes main-AI models too.
            if main_ai_dirs is None and model_root is not None:
                main_ai_dirs = [Path(model_root) / "main_ai"]
            self._main_ai_dirs = main_ai_dirs
            self._list_runs = list_runs or list_saved_module_runs
            self._summarize = summarize or summarize_module_run
            self._list_main_ai = list_main_ai or list_saved_main_ai_models
            self._summarize_main_ai = summarize_main_ai or summarize_main_ai_artifact
            self._cache: dict[str, str] = {}
            self._runs: list[Any] = []
            self._main_ai_models: list[Any] = []
            self.selected_job_id: str | None = None

            self.setWindowTitle(tr("perf.title"))
            self.resize(900, 560)
            self.list = QListWidget(self)
            self.list.setMinimumWidth(300)
            self.details = QPlainTextEdit(self)
            self.details.setReadOnly(True)
            self.empty_label = QLabel(tr("perf.empty"), self)
            self.empty_label.setWordWrap(True)
            self.refresh_button = QPushButton(tr("perf.refresh"), self)
            self.resume_button = QPushButton(tr("perf.use_for_resume"), self)
            self.resume_button.setToolTip(tr("tooltip.perf_use_for_resume"))
            self.close_button = QPushButton(tr("perf.close"), self)
            self.notes_button = QPushButton(tr("perf.edit_notes"), self)
            self.notes_button.setToolTip(tr("tooltip.perf_edit_notes"))
            self.save_copy_button = QPushButton(tr("perf.save_copy"), self)
            self.save_copy_button.setToolTip(tr("tooltip.perf_save_copy"))

            body = QHBoxLayout()
            body.addWidget(self.list, 1)
            body.addWidget(self.details, 2)
            buttons = QHBoxLayout()
            buttons.addWidget(self.refresh_button)
            buttons.addWidget(self.notes_button)
            buttons.addWidget(self.save_copy_button)
            buttons.addStretch(1)
            buttons.addWidget(self.resume_button)
            buttons.addWidget(self.close_button)
            layout = QVBoxLayout(self)
            layout.addWidget(self.empty_label)
            layout.addLayout(body, 1)
            layout.addLayout(buttons)

            self.list.currentRowChanged.connect(self._show_row)
            self.refresh_button.clicked.connect(self.refresh)
            self.resume_button.clicked.connect(self._use_for_resume)
            self.close_button.clicked.connect(self.reject)
            self.notes_button.clicked.connect(self._edit_notes)
            self.save_copy_button.clicked.connect(self._save_copy)
            self.refresh()

        def refresh(self) -> None:
            self._cache.clear()
            self._runs = list(self._list_runs(self._model_root))
            try:
                self._main_ai_models = list(self._list_main_ai(self._main_ai_dirs))
            except OSError:
                self._main_ai_models = []
            self.list.clear()
            # Saved module runs first, then saved main-AI models.
            for run in self._runs:
                item = QListWidgetItem(module_run_label(run))
                item.setToolTip(getattr(run, "tooltip", "") or module_run_label(run))
                self.list.addItem(item)
            for model in self._main_ai_models:
                item = QListWidgetItem(main_ai_model_label(model))
                item.setToolTip(getattr(model, "tooltip", "") or main_ai_model_label(model))
                self.list.addItem(item)
            has_items = bool(self._runs or self._main_ai_models)
            self.notes_button.setEnabled(has_items)
            self.save_copy_button.setEnabled(has_items)
            self.empty_label.setVisible(not has_items)
            self.resume_button.setEnabled(bool(self._runs))
            if has_items:
                self.list.setCurrentRow(0)
            else:
                self.details.setPlainText("")

        def _show_row(self, row: int) -> None:
            module_count = len(self._runs)
            if 0 <= row < module_count:
                run = self._runs[row]
                text = self._cache.get(f"module:{run.job_id}")
                if text is None:
                    text = format_module_run_summary(self._summarize(run, reports_dir=self._reports_dir))
                    self._cache[f"module:{run.job_id}"] = text
                self.resume_button.setEnabled(True)
            elif module_count <= row < module_count + len(self._main_ai_models):
                model = self._main_ai_models[row - module_count]
                text = self._cache.get(f"main:{model.path}")
                if text is None:
                    text = format_main_ai_model_summary(self._summarize_main_ai(model.path))
                    self._cache[f"main:{model.path}"] = text
                # A main-AI model is not a module run to continue from.
                self.resume_button.setEnabled(False)
            else:
                self.details.setPlainText("")
                return
            self.details.setPlainText(text)

        def _selected(self) -> tuple[str, Any] | None:
            row = self.list.currentRow()
            if 0 <= row < len(self._runs):
                return "module", self._runs[row]
            index = row - len(self._runs)
            if 0 <= index < len(self._main_ai_models):
                return "main", self._main_ai_models[index]
            return None

        def _edit_notes(self) -> None:
            from bang_rl.artifact_info import read_info, update_notes

            from .save_outputs import NotesDialog

            selected = self._selected()
            if selected is None:
                return
            kind, item = selected
            paths = (
                [p for p in (item.intent_path, item.emotion_path) if p is not None]
                if kind == "module"
                else [item.path]
            )
            first = next((read_info(p) for p in paths if read_info(p) is not None), None)
            title = item.display_name
            dialog = NotesDialog(title, first.notes if first is not None else "", self)
            if not dialog.exec():
                return
            for path in paths:
                if read_info(path) is not None:
                    update_notes(path, dialog.text)
            self.refresh()

        def _save_copy(self) -> None:
            from bang_rl.saved_artifacts import plan_main_ai, plan_modules

            from .save_outputs import SaveOutputsDialog

            selected = self._selected()
            if selected is None:
                return
            kind, item = selected
            plan = (
                plan_modules(item.intent_path, item.emotion_path)
                if kind == "module"
                else plan_main_ai(item.path)
            )
            if plan is None:
                return
            SaveOutputsDialog(plan, self).exec()
            self.refresh()

        def _use_for_resume(self) -> None:
            row = self.list.currentRow()
            if 0 <= row < len(self._runs):
                self.selected_job_id = self._runs[row].job_id
                self.accept()

else:

    class ModulePerformanceDialog:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = [
    "ModulePerformanceDialog",
    "format_main_ai_model_summary",
    "format_module_run_summary",
    "format_module_summary",
    "main_ai_model_label",
    "module_run_label",
]
