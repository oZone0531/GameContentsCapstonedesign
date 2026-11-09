"""Optional PyQt6 Training Setup and Training Monitor screens."""
from __future__ import annotations

import json
from typing import Any

from collections.abc import Callable

from bang_rl.paths import ARTIFACTS_DIR, OUTPUTS_DIR, REPORTS_DIR

from ..qt_compat import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QPushButton,
    QSpinBox,
    QComboBox,
    QApplication,
    QStackedWidget,
    QTimer,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from ..i18n import tr
from ..training import (
    TRAINING_MODE_DATA_COLLECTION,
    TRAINING_MODE_MAIN_AI,
    TRAINING_MODE_MODULE_TRAINING,
    TrainingMonitorController,
    TrainingMonitorSnapshot,
    TrainingSetup,
    public_training_metadata,
)

try:  # pyqtgraph is optional even when PyQt6 is available.
    if QT_AVAILABLE:
        from pyqtgraph import PlotWidget  # type: ignore[import-not-found]

        PYQTGRAPH_AVAILABLE = True
    else:  # pragma: no cover - depends on optional dependency
        PlotWidget = None
        PYQTGRAPH_AVAILABLE = False
except ImportError:  # pragma: no cover - depends on optional dependency
    PlotWidget = None
    PYQTGRAPH_AVAILABLE = False


def metrics_json(value: Any) -> str:
    """Return the complete public Metrics payload copied by the GUI."""

    return json.dumps(public_training_metadata(value or {}), ensure_ascii=False, sort_keys=True)


def metrics_preview(value: Any, max_chars: int = 220) -> str:
    """Return a one-line Metrics preview while preserving the full copy value."""

    text = metrics_json(value)
    if len(text) <= int(max_chars):
        return text
    return text[: max(1, int(max_chars) - 3)] + "..."
def _set_log_text_preserving_scroll(editor: Any, text: str) -> None:
    """Replace log text without stealing the user's viewport position."""

    if editor.toPlainText() == text:
        return
    scrollbar = editor.verticalScrollBar()
    old_value = scrollbar.value()
    was_at_bottom = old_value >= scrollbar.maximum()
    editor.setPlainText(text)
    if was_at_bottom:
        scrollbar.setValue(scrollbar.maximum())
    else:
        scrollbar.setValue(min(old_value, scrollbar.maximum()))


if QT_AVAILABLE:  # pragma: no cover - display dependent; controller is headless tested

    class TrainingSetupPanel(QWidget):
        """Small setup form; values are passed to the service unchanged."""

        def __init__(self, config: Any = None, parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(parent)
            self.config = config
            self.mode = QComboBox(self)
            self.mode.addItem(tr("training.mode.data_collection"), TRAINING_MODE_DATA_COLLECTION)
            self.mode.addItem(tr("training.mode.module_training"), TRAINING_MODE_MODULE_TRAINING)
            self.mode.addItem(tr("training.mode.main_ai_training"), TRAINING_MODE_MAIN_AI)
            self.episodes = QSpinBox(self)
            self.episodes.setRange(1, 2_147_483_647)
            self.episodes.setValue(1)
            self.target = QSpinBox(self)
            self.target.setRange(0, 2_147_483_647)
            self.target.setSpecialValueText("Use episodes")
            self.seed = QSpinBox(self)
            self.seed.setRange(-2_147_483_648, 2_147_483_647)
            self.report = QLineEdit(str(REPORTS_DIR), self)
            self.output = QLineEdit(str(OUTPUTS_DIR), self)
            self.artifact = QLineEdit(str(ARTIFACTS_DIR), self)
            self.resume_artifact = QLineEdit(str(ARTIFACTS_DIR), self)
            form = QFormLayout(self)
            form.addRow(tr("training.mode"), self.mode)
            form.addRow(tr("training.episodes"), self.episodes)
            form.addRow(tr("training.target"), self.target)
            form.addRow(tr("training.seed"), self.seed)
            form.addRow(tr("training.report_path"), self.report)
            form.addRow(tr("training.output_path"), self.output)
            form.addRow(tr("training.artifact_path"), self.artifact)
            form.addRow(tr("training.resume_artifact_path"), self.resume_artifact)
            # Item 3: hover help on every field AND its QFormLayout label so the
            # AI-training settings menu gets mouse-over tooltips too.
            field_tooltips = (
                (self.mode, "tooltip.training_mode"),
                (self.episodes, "tooltip.training_episodes"),
                (self.target, "tooltip.training_target"),
                (self.seed, "tooltip.training_seed"),
                (self.report, "tooltip.training_report_path"),
                (self.output, "tooltip.training_output_path"),
                (self.artifact, "tooltip.training_artifact_path"),
                (self.resume_artifact, "tooltip.training_resume_artifact_path"),
            )
            for field, key in field_tooltips:
                field.setToolTip(tr(key))
                label = form.labelForField(field)
                if label is not None:
                    label.setToolTip(tr(key))

        def setup(self) -> TrainingSetup:
            target = self.target.value() or None
            return TrainingSetup(
                config=self.config,
                mode=str(self.mode.currentData() or TRAINING_MODE_DATA_COLLECTION),
                episodes=self.episodes.value(),
                target_transitions=target,
                seed=self.seed.value(),
                report_path=self.report.text().strip() or None,
                output_path=self.output.text().strip() or None,
                artifact_path=self.artifact.text().strip() or None,
                resume_artifact_path=self.resume_artifact.text().strip() or None,
            )


    class TrainingMonitor(QWidget):
        """Live monitor for public counters, metrics, logs, and lifecycle state."""

        def __init__(
            self,
            controller: TrainingMonitorController | None = None,
            *,
            setup_panel: TrainingSetupPanel | None = None,
            parent: QWidget | None = None,
            on_back: Callable[[], None] | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.controller = controller or TrainingMonitorController()
            self.setup_panel = setup_panel or TrainingSetupPanel(parent=self)
            self.on_back = on_back
            self.controller.on_update = self._render
            self.progress_bar = QProgressBar(self)
            self.progress_bar.setFormat("Collected games: %v / %m")
            self.collection_progress = self.progress_bar
            self.environment_steps = QLabel("Environment steps: 0", self)
            self.truncated = QLabel("Truncated: 0 (0.0%)", self)
            self.metrics = QLabel("Metrics: —", self)
            self.metrics.setWordWrap(False)
            self.metrics.setMaximumHeight(28)
            self.metrics_copy_button = QPushButton("복사", self)
            self.metrics_copy_button.setToolTip("Metrics 전체 내용을 클립보드에 복사")
            self.metrics_copy_button.clicked.connect(self._copy_metrics)
            self._metrics_full_text = "{}"
            self.validation_summary = QLabel("Validation: —", self)
            self.validation_summary.setWordWrap(True)
            self.status = QLabel(tr("training.no_job"), self)
            self.snapshot = QLabel(tr("training.snapshot_not_started"), self)
            self.paths = QLabel("Report: — · Output: — · Artifact: —", self)
            self.paths.setWordWrap(True)
            self.logs = QPlainTextEdit(self)
            self.logs.setReadOnly(True)
            self.logs.setMinimumHeight(70)
            self.logs.setMaximumHeight(110)
            self.logs.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            self.message = QLabel(self)
            self.message.setWordWrap(True)
            self.start_button = QPushButton(tr("training.start"), self)
            self.pause_button = QPushButton(tr("training.pause"), self)
            self.resume_button = QPushButton(tr("training.resume"), self)
            self.stop_button = QPushButton(tr("training.stop"), self)
            buttons = QHBoxLayout()
            for button in (self.start_button, self.pause_button, self.resume_button, self.stop_button):
                buttons.addWidget(button)
            self.start_button.clicked.connect(self._start)
            self.pause_button.clicked.connect(self._pause)
            self.resume_button.clicked.connect(self._resume)
            self.stop_button.clicked.connect(self._stop)
            self.back_button = QPushButton(tr("common.back"), self)
            self.back_button.clicked.connect(self._go_back)

            self._graph_mode = "accuracy"
            self.role_graph_button = QPushButton("역할추론", self)
            self.emotion_graph_button = QPushButton("감정추론", self)
            self.accuracy_graph_button = QPushButton("정확도", self)
            self.accuracy_graph_button.setVisible(False)
            self.role_graph_button.clicked.connect(lambda: self._set_graph_mode("roles"))
            self.emotion_graph_button.clicked.connect(lambda: self._set_graph_mode("emotions"))
            self.accuracy_graph_button.clicked.connect(lambda: self._set_graph_mode("accuracy"))
            graph_controls = QHBoxLayout()
            graph_controls.addWidget(self.role_graph_button)
            graph_controls.addWidget(self.emotion_graph_button)
            graph_controls.addWidget(self.accuracy_graph_button)
            graph_controls.addStretch(1)

            self.graph_stack = QStackedWidget(self)
            self.graph_stack.setMinimumHeight(420)
            self.graph_stack.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.collection_graph = self.progress_bar
            self.collection_graph.setRange(0, 1)
            self.collection_graph.setFormat("Collected games: %v / %m")
            self.graph_stack.addWidget(self.collection_graph)
            self._collection_graph_index = 0
            if PYQTGRAPH_AVAILABLE:
                self.plot = PlotWidget(self)
                self.plot.setLabel("left", "Accuracy")
                self.plot.setLabel("bottom", "Batch")
                self.plot.addLegend()
                self._intent_curve = self.plot.plot([], [], pen="y", name="Intent")
                self._emotion_curve = self.plot.plot([], [], pen="c", name="Emotion")
                self._role_curves = []
                self._emotion_curves = []
                self.graph_stack.addWidget(self.plot)
                self._module_graph_index = 1
            else:
                self.plot = None
                self._role_curves = []
                self._emotion_curves = []
                graph_help = QLabel(tr("training.graph_needs_pyqtgraph"), self)
                graph_help.setWordWrap(True)
                self.graph_stack.addWidget(graph_help)
                self._module_graph_index = 1
            self._main_ai_graph_index = self._module_graph_index

            layout = QVBoxLayout(self)
            layout.addWidget(self.back_button)
            columns = QHBoxLayout()
            left_panel = QVBoxLayout()
            left_panel.addWidget(self.setup_panel)
            left_panel.addLayout(buttons)
            left_panel.addStretch(1)
            right_panel = QVBoxLayout()
            right_panel.addWidget(self.environment_steps)
            right_panel.addWidget(self.truncated)
            metrics_row = QHBoxLayout()
            metrics_row.addWidget(self.metrics, 1)
            metrics_row.addWidget(self.metrics_copy_button)
            right_panel.addLayout(metrics_row)
            right_panel.addWidget(self.validation_summary)
            right_panel.addWidget(self.status)
            right_panel.addWidget(self.snapshot)
            right_panel.addWidget(self.paths)
            right_panel.addWidget(self.graph_stack, 6)
            right_panel.addLayout(graph_controls)
            right_panel.addWidget(self.logs, 1)
            right_panel.addWidget(self.message)
            columns.addLayout(left_panel, 1)
            columns.addLayout(right_panel, 2)
            layout.addLayout(columns, 1)
            self._timer = QTimer(self)
            self._timer.setInterval(250)
            self._timer.timeout.connect(self.controller.refresh)
            self._timer.start()
            self._render(self.controller.snapshot)

        def _go_back(self) -> None:
            self._timer.stop()
            if self.on_back is not None:
                self.on_back()

        def _start(self) -> None:
            self.controller.start(self.setup_panel.setup())

        def _pause(self) -> None:
            self.controller.pause(timeout=30.0)

        def _resume(self) -> None:
            self.controller.resume()

        def _stop(self) -> None:
            self.controller.cancel()

        def _copy_metrics(self) -> None:
            if QApplication is not None:
                QApplication.clipboard().setText(self._metrics_full_text)

        def _set_graph_mode(self, mode: str) -> None:
            self._graph_mode = mode if mode in {"accuracy", "roles", "emotions"} else "accuracy"
            is_default = self._graph_mode == "accuracy"
            self.role_graph_button.setVisible(is_default)
            self.emotion_graph_button.setVisible(is_default)
            self.accuracy_graph_button.setVisible(not is_default)
            self._render(self.controller.snapshot)

        def _plot_matrix(self, points: Any, names: Any, colors: tuple[str, ...]) -> None:
            if self.plot is None:
                return
            self.plot.clear()
            self.plot.addLegend()
            rows = [row for row in points if isinstance(row, (list, tuple))]
            labels = list(names or ())
            for index, name in enumerate(labels):
                values = [float(row[index]) for row in rows if len(row) > index]
                pen = colors[index % len(colors)]
                self.plot.plot(
                    list(range(1, len(values) + 1)),
                    values,
                    pen=pen,
                    name=str(name),
                )

        def _render_graph(self, state: TrainingMonitorSnapshot) -> None:
            if self.plot is None:
                return
            if self._graph_mode == "roles":
                self._plot_matrix(
                    getattr(state, "intent_accuracy_by_role_points", ()),
                    getattr(state, "intent_role_names", ()),
                    ("#f6c344", "#5bc0eb", "#e85d75", "#8bd450"),
                )
                return
            if self._graph_mode == "emotions":
                self._plot_matrix(
                    getattr(state, "emotion_accuracy_by_dimension_points", ()),
                    getattr(state, "emotion_dimension_names", ()),
                    (
                        "#f6c344", "#5bc0eb", "#e85d75", "#8bd450", "#b084cc",
                        "#f28f3b", "#4d9de0", "#e15554", "#3bb273",
                    ),
                )
                return
            self.plot.clear()
            self.plot.addLegend()
            intent_values = getattr(state, "intent_accuracy_points", ())
            emotion_values = getattr(state, "emotion_accuracy_points", ())
            self.plot.plot(
                list(range(1, len(intent_values) + 1)),
                list(intent_values),
                pen="y",
                name="Intent",
            )
            self.plot.plot(
                list(range(1, len(emotion_values) + 1)),
                list(emotion_values),
                pen="c",
                name="Emotion",
            )

        def _render(self, state: TrainingMonitorSnapshot) -> None:
            progress = state.progress
            if progress is None:
                self.progress_bar.setRange(0, 1)
                self.progress_bar.setValue(0)
                self.status.setText(tr("training.no_job"))
                self.snapshot.setText(tr("training.snapshot_not_started"))
                self._metrics_full_text = "{}"
                self.metrics.setToolTip("{}")
                self.metrics.setText("Metrics: —")
                self.role_graph_button.setVisible(False)
                self.emotion_graph_button.setVisible(False)
                self.accuracy_graph_button.setVisible(False)
                return
            target_config = getattr(state.snapshot, "config_snapshot", {}) if state.snapshot is not None else {}
            workflow = getattr(state.snapshot, "mode", None) if state.snapshot is not None else None
            if workflow == "data_collection":
                collection_metrics = getattr(progress, "metrics", {}) or {}
                maximum = int(collection_metrics.get("collection_total", 0) or 1)
                current = int(collection_metrics.get("collection_completed", progress.episodes_completed) or 0)
                self.progress_bar.setFormat("Collected games: %v / %m")
                self.progress_bar.setRange(0, max(maximum, current, 1))
                self.progress_bar.setValue(current)
            else:
                training = target_config.get("rl_training", {}) if isinstance(target_config, dict) else {}
                ppo = training.get("ppo", {}) if isinstance(training, dict) else {}
                maximum = int(getattr(progress, "main_agent_transitions", 0))
                requested = int(ppo.get("rollout_transitions_per_worker", maximum) or maximum or 1)
                self.progress_bar.setFormat("MainAgent transitions: %v / %m")
                self.progress_bar.setRange(0, max(requested, maximum, 1))
                self.progress_bar.setValue(maximum)
            self.environment_steps.setText(f"Environment steps: {progress.environment_steps}")
            count = int(getattr(progress, "truncated_episodes", 0))
            rate = getattr(progress, "truncated_rate", None)
            self.truncated.setText(f"Truncated: {count} ({'N/A' if rate is None else f'{rate:.1%}'})")
            metrics_value = public_training_metadata(getattr(progress, "metrics", {}) or {})
            self._metrics_full_text = metrics_json(metrics_value)
            self.metrics.setToolTip(self._metrics_full_text)
            self.metrics.setText("Metrics: " + metrics_preview(metrics_value))
            if workflow == "module_training":
                intent_validation = metrics_value.get("intent_validation_accuracy")
                emotion_validation = metrics_value.get("emotion_validation_accuracy")
                emotion_mse = metrics_value.get("emotion_validation_mse")
                self.validation_summary.setText(
                    "Validation: Intent accuracy="
                    + ("—" if intent_validation is None else f"{float(intent_validation):.3f}")
                    + " · Emotion accuracy="
                    + ("—" if emotion_validation is None else f"{float(emotion_validation):.3f}")
                    + " · Emotion MSE="
                    + ("—" if emotion_mse is None else f"{float(emotion_mse):.4f}")
                )
            else:
                self.validation_summary.setText("Validation: —")
            self.status.setText(f"Status: {progress.status} · PPO updates: {progress.ppo_updates}")
            if state.snapshot is not None:
                self.snapshot.setText(
                    f"Snapshot: {state.snapshot.job_id} · config {state.snapshot.config_version} "
                    f"· seed {state.snapshot.seed}"
                )
            self.paths.setText(
                f"Report: {progress.report_path or '—'} · Output: {getattr(progress, 'output_path', None) or '—'} · "
                f"Artifact: {progress.artifact_path or '—'}"
            )
            _set_log_text_preserving_scroll(self.logs, "\n".join(state.logs))
            self.message.setText(state.message or "")
            if self.plot is not None:
                if workflow == "module_training":
                    self.graph_stack.setCurrentIndex(self._module_graph_index)
                    is_default = self._graph_mode == "accuracy"
                    self.role_graph_button.setVisible(is_default)
                    self.emotion_graph_button.setVisible(is_default)
                    self.accuracy_graph_button.setVisible(not is_default)
                    self._render_graph(state)
                elif workflow == "data_collection":
                    self.role_graph_button.setVisible(False)
                    self.emotion_graph_button.setVisible(False)
                    self.accuracy_graph_button.setVisible(False)
                    self.graph_stack.setCurrentIndex(self._collection_graph_index)
                else:
                    self.role_graph_button.setVisible(False)
                    self.emotion_graph_button.setVisible(False)
                    self.accuracy_graph_button.setVisible(False)
                    self.graph_stack.setCurrentIndex(self._main_ai_graph_index)
            elif workflow != "data_collection":
                self.graph_stack.setCurrentIndex(self._module_graph_index)


else:

    class TrainingSetupPanel:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

    class TrainingMonitor:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["PYQTGRAPH_AVAILABLE", "TrainingMonitor", "TrainingSetupPanel"]
