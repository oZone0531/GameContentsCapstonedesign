"""Saved Phase-2 module browser: final performance and 'continue training'."""
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


def format_module_run_summary(run_summary: Mapping[str, Any]) -> str:
    """Full text shown for one saved run (see summarize_module_run)."""

    lines = [
        tr("perf.job", job=run_summary.get("job_id", "—")),
        tr("perf.saved_at", time=run_summary.get("modified", "—")),
    ]
    if run_summary.get("report_path"):
        lines.append(tr("perf.report", path=run_summary["report_path"]))
    lines += ["", tr("perf.intent_section")]
    lines += format_module_summary(run_summary.get("intent"))
    lines += ["", tr("perf.emotion_section")]
    lines += format_module_summary(run_summary.get("emotion"))
    return "\n".join(lines)


def _module_parts(run: Any) -> str:
    if run.intent_path is not None and run.emotion_path is not None:
        return tr("perf.parts_both")
    if run.intent_path is not None:
        return tr("perf.parts_intent")
    return tr("perf.parts_emotion")


def module_run_label(run: Any) -> str:
    """One-line list label for a SavedModuleRun."""

    return f"{run.modified_text} · {run.short_id} · {_module_parts(run)}"


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
        ) -> None:
            require_qt()
            super().__init__(parent)
            from bang_rl.training.module_artifacts import (
                list_saved_module_runs,
                summarize_module_run,
            )

            self._model_root = model_root
            self._reports_dir = reports_dir
            self._list_runs = list_runs or list_saved_module_runs
            self._summarize = summarize or summarize_module_run
            self._cache: dict[str, str] = {}
            self._runs: list[Any] = []
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

            body = QHBoxLayout()
            body.addWidget(self.list, 1)
            body.addWidget(self.details, 2)
            buttons = QHBoxLayout()
            buttons.addWidget(self.refresh_button)
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
            self.refresh()

        def refresh(self) -> None:
            self._cache.clear()
            self._runs = list(self._list_runs(self._model_root))
            self.list.clear()
            for run in self._runs:
                self.list.addItem(QListWidgetItem(module_run_label(run)))
            has_runs = bool(self._runs)
            self.empty_label.setVisible(not has_runs)
            self.resume_button.setEnabled(has_runs)
            if has_runs:
                self.list.setCurrentRow(0)
            else:
                self.details.setPlainText("")

        def _show_row(self, row: int) -> None:
            if not 0 <= row < len(self._runs):
                self.details.setPlainText("")
                return
            run = self._runs[row]
            text = self._cache.get(run.job_id)
            if text is None:
                text = format_module_run_summary(self._summarize(run, reports_dir=self._reports_dir))
                self._cache[run.job_id] = text
            self.details.setPlainText(text)

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
    "format_module_run_summary",
    "format_module_summary",
    "module_run_label",
]
