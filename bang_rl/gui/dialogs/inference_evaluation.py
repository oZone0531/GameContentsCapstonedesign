"""Inference accuracy evaluation dialog ("추론 정확도 평가").

Compares intent + emotion module pairs under the fixed PPO of one ``.bangmodel``
on the fixed main-AI evaluation games (hidden-role accuracy, role accuracy,
emotion MSE and, as a secondary number, the win rate). The evaluation runs on a
background thread; the dialog polls it.

:class:`InferenceEvaluationJob` is Qt-free so it can be tested without a display.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import main_ai_view
from ..i18n import tr
from ..qt_compat import (
    QCheckBox,
    QDialog,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTimer,
    Qt,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)


@dataclass(slots=True)
class InferenceEvaluationJob:
    """Evaluate module pairs under one model's fixed PPO on a background thread.

    ``pairs`` are ``(label, intent_path, emotion_path)`` tuples. Rows appear in
    ``rows`` as each pair finishes; ``status`` is ``idle``, ``running``,
    ``finished``, ``stopped`` or ``failed`` (then ``error`` is set).
    """

    model_path: str | Path
    pairs: Sequence[tuple[str, str, str]]
    games: int = 250
    include_base: bool = True
    config: Any = None
    status: str = "idle"
    error: str | None = None
    done: int = 0
    total: int = 0
    current: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the evaluation was already started")
        self.total = len(self.pairs) + (1 if self.include_base else 0)
        self.status = "running"
        self._thread = threading.Thread(target=self._run, name="bang-rl-inference-eval", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> tuple[str, int, int, str, list[dict[str, Any]], str | None]:
        with self._lock:
            return self.status, self.done, self.total, self.current, list(self.rows), self.error

    def _run(self) -> None:
        try:
            from bang_rl.config import SystemConfig
            from bang_rl.training.final_inference_artifact import load_final_inference_artifact
            from bang_rl.training.inference_evaluation import (
                ModulePair,
                evaluate_module_pairs,
                load_module_pair,
            )

            config = self.config if self.config is not None else SystemConfig()
            base = load_final_inference_artifact(self.model_path)
            candidates: list[ModulePair] = []
            if self.include_base:
                candidates.append(ModulePair("base", base.intent, base.emotion))
            for label, intent_path, emotion_path in self.pairs:
                candidates.append(load_module_pair(label, intent_path, emotion_path))
            stopped = False
            for pair in candidates:
                with self._lock:
                    self.current = pair.label
                rows = evaluate_module_pairs(
                    base, [pair], config, games=int(self.games), include_base=False, should_stop=self._stop.is_set
                )
                if not rows:  # stopped in the middle of this pair
                    stopped = True
                    break
                with self._lock:
                    self.rows.extend(rows)
                    self.done += 1
            with self._lock:
                self.status = "stopped" if stopped or self._stop.is_set() and self.done < self.total else "finished"
        except Exception as exc:  # noqa: BLE001 - shown to the user, never raised into Qt
            with self._lock:
                self.status = "failed"
                self.error = str(exc) or type(exc).__name__


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class InferenceEvaluationDialog(QDialog):
        """Choose a main AI model and module pairs, run the evaluation, read the table."""

        COLUMNS = ("infer.col.name", "infer.col.hidden", "infer.col.role", "infer.col.mse", "infer.col.win", "infer.col.by_role")

        def __init__(
            self,
            *,
            config: Any = None,
            model_root: str | Path | None = None,
            main_ai_dirs: Sequence[str | Path] | None = None,
            parent: QWidget | None = None,
            select_model: str | Path | None = None,
            select_pair: tuple[str | Path | None, str | Path | None] | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self._config = config
            # Preselection handed over by the data library: a model and/or a module pair.
            self._select_model = None if select_model is None else str(select_model)
            self._select_pair = (
                None if select_pair is None else tuple(None if item is None else str(item) for item in select_pair)
            )
            self._model_root = model_root
            self._main_ai_dirs = main_ai_dirs
            self._job: InferenceEvaluationJob | None = None
            self._modules: dict[int, Any] = {}
            self.setWindowTitle(tr("infer.title"))
            self.resize(980, 560)

            self.model = QComboBox(self)
            self.pairs = QListWidget(self)
            self.pairs.setMaximumHeight(130)
            self.include_base = QCheckBox(tr("infer.include_base"), self)
            self.include_base.setChecked(True)
            self.games = QSpinBox(self)
            self.games.setRange(1, 1_000_000)
            self.games.setValue(250)
            self.note = QLabel(tr("infer.note"), self)
            self.note.setWordWrap(True)
            self.status = QLabel("", self)
            self.status.setWordWrap(True)
            self.progress = QProgressBar(self)
            self.progress.setRange(0, 1)
            self.table = QTableWidget(0, len(self.COLUMNS), self)
            self.table.setHorizontalHeaderLabels([tr(key) for key in self.COLUMNS])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.run_button = QPushButton(tr("infer.run"), self)
            self.stop_button = QPushButton(tr("infer.stop"), self)
            self.stop_button.setEnabled(False)
            self.close_button = QPushButton(tr("perf.close"), self)
            self.refresh_button = QPushButton(tr("perf.refresh"), self)
            self.run_button.clicked.connect(self.run)
            self.stop_button.clicked.connect(self.stop)
            self.close_button.clicked.connect(self.close)
            self.refresh_button.clicked.connect(self.refresh)

            row = QHBoxLayout()
            for widget in (self.run_button, self.stop_button, self.refresh_button, self.close_button):
                row.addWidget(widget)
            games_row = QHBoxLayout()
            games_row.addWidget(QLabel(tr("infer.games"), self))
            games_row.addWidget(self.games)
            games_row.addWidget(self.include_base)
            games_row.addStretch(1)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(tr("infer.model"), self))
            layout.addWidget(self.model)
            layout.addWidget(QLabel(tr("infer.pairs"), self))
            layout.addWidget(self.pairs)
            layout.addLayout(games_row)
            layout.addWidget(self.note)
            layout.addLayout(row)
            layout.addWidget(self.progress)
            layout.addWidget(self.status)
            layout.addWidget(self.table, 1)
            self._timer = QTimer(self)
            self._timer.setInterval(300)
            self._timer.timeout.connect(self._poll)
            self.refresh()

        def refresh(self) -> None:
            """Reload the model and module lists."""

            from bang_rl.training.main_ai_artifacts import list_saved_main_ai_models, main_ai_model_compatibility
            from bang_rl.training.module_artifacts import list_saved_module_runs, main_ai_compatibility
            from .module_performance import module_run_label

            self.model.clear()
            try:
                models = list_saved_main_ai_models(self._main_ai_dirs)
            except OSError:
                models = ()
            for model in models:
                ok, _reason = main_ai_model_compatibility(model.path)
                if ok:
                    self.model.addItem(model.display_name, str(model.path))
                    self.model.setItemData(self.model.count() - 1, model.tooltip, Qt.ItemDataRole.ToolTipRole)
            self.pairs.clear()
            try:
                runs = list_saved_module_runs(self._model_root)
            except OSError:
                runs = ()
            for run in runs:
                if run.intent_path is None or run.emotion_path is None or not main_ai_compatibility(run)[0]:
                    continue
                item = QListWidgetItem(module_run_label(run))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, (run.display_name, str(run.intent_path), str(run.emotion_path)))
                item.setToolTip(run.tooltip)
                self.pairs.addItem(item)
                if self._select_pair is not None and (str(run.intent_path), str(run.emotion_path)) == self._select_pair:
                    item.setCheckState(Qt.CheckState.Checked)
            if self._select_model is not None:
                index = self.model.findData(self._select_model)
                if index >= 0:
                    self.model.setCurrentIndex(index)

        def selected_pairs(self) -> list[tuple[str, str, str]]:
            chosen: list[tuple[str, str, str]] = []
            for index in range(self.pairs.count()):
                item = self.pairs.item(index)
                if item.checkState() == Qt.CheckState.Checked:
                    chosen.append(tuple(item.data(Qt.ItemDataRole.UserRole)))  # type: ignore[arg-type]
            return chosen

        def run(self) -> None:
            if self._job is not None and self._job.status == "running":
                return
            path = self.model.currentData()
            if not path:
                self.status.setText(tr("infer.no_model"))
                return
            pairs = self.selected_pairs()
            if not pairs and not self.include_base.isChecked():
                self.status.setText(tr("infer.nothing"))
                return
            self.table.setRowCount(0)
            self._job = InferenceEvaluationJob(
                path, pairs, games=int(self.games.value()), include_base=self.include_base.isChecked(), config=self._config
            )
            self._job.start()
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self._timer.start()
            self._poll()

        def stop(self) -> None:
            if self._job is not None:
                self._job.stop()

        def _poll(self) -> None:
            if self._job is None:
                return
            status, done, total, current, rows, error = self._job.snapshot()
            self.progress.setRange(0, max(total, 1))
            self.progress.setValue(done)
            self.table.setRowCount(0)
            for cells in main_ai_view.inference_rows(rows):
                index = self.table.rowCount()
                self.table.insertRow(index)
                for column, text in enumerate(cells):
                    self.table.setItem(index, column, QTableWidgetItem(text))
            if status == "running":
                self.status.setText(tr("infer.running", done=done, total=total, label=current))
                return
            self._timer.stop()
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status.setText(
                tr("infer.failed", error=error) if status == "failed" else tr("infer.finished" if status == "finished" else "infer.stopped")
            )

        def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt signature
            if self._job is not None:
                self._job.stop()
            self._timer.stop()
            super().closeEvent(event)

else:

    class InferenceEvaluationDialog:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["InferenceEvaluationDialog", "InferenceEvaluationJob"]
