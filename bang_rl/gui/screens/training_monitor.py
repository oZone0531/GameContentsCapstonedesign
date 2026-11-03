"""Optional PyQt6 Training Setup and Training Monitor screens."""
from __future__ import annotations

import json
from typing import Any

from collections.abc import Callable, Mapping

from bang_rl.paths import ARTIFACTS_DIR, OUTPUTS_DIR, REPORTS_DIR, REPLAYS_DIR

from ..qt_compat import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTimer,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from ..i18n import tr
from ..training import (
    TrainingMonitorController,
    TrainingMonitorSnapshot,
    TrainingSetup,
    _public_value,
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


if QT_AVAILABLE:  # pragma: no cover - display dependent; controller is headless tested

    class TrainingSetupPanel(QWidget):
        """Small setup form; values are passed to the service unchanged."""

        def __init__(self, config: Any = None, parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(parent)
            self.config = config
            self.mode = QComboBox(self)
            self.mode.addItem(tr("training.mode_data_collection"), "data_collection")
            self.mode.addItem(tr("training.mode_inference"), "inference_sl")
            self.mode.addItem(tr("training.mode_policy"), "policy_ppo")
            self.episodes = QSpinBox(self)
            self.episodes.setRange(0, 2_147_483_647)
            self.episodes.setValue(1)
            self.target = QSpinBox(self)
            self.target.setRange(0, 2_147_483_647)
            self.target.setSpecialValueText("Use episodes")
            self.update_mode = QComboBox(self)
            self.update_mode.addItem(tr("training.update_online"), "online")
            self.update_mode.addItem(tr("training.update_batch"), "batch")
            self.update_mode.addItem(tr("training.update_semibatch"), "semibatch")
            self.update_mode.setCurrentIndex(1)
            self.ppo_batch_size = QSpinBox(self)
            self.ppo_batch_size.setRange(1, 1_000_000)
            self.ppo_batch_size.setValue(64)
            self.ppo_batch_size.setEnabled(False)
            self.sl_epochs = QSpinBox(self)
            self.sl_epochs.setRange(1, 1_000)
            self.sl_epochs.setValue(10)
            self.sl_epochs.setEnabled(False)
            self.update_mode.currentIndexChanged.connect(
                lambda _index: self._update_training_fields()
            )
            self.mode.currentIndexChanged.connect(lambda _index: self._update_training_fields())
            self.seed = QSpinBox(self)
            self.seed.setRange(-2_147_483_648, 2_147_483_647)
            self.report = QLineEdit(str(REPORTS_DIR), self)
            self.output = QLineEdit(str(OUTPUTS_DIR), self)
            self.artifact = QLineEdit(str(ARTIFACTS_DIR), self)
            self.resume_artifact = QLineEdit(str(ARTIFACTS_DIR), self)
            self.replay_source = QLineEdit(str(REPLAYS_DIR), self)
            self.phase2_artifact = QLineEdit("", self)
            form = QFormLayout(self)
            form.addRow(tr("training.mode"), self.mode)
            form.addRow(tr("training.episodes"), self.episodes)
            form.addRow(tr("training.target"), self.target)
            form.addRow(tr("training.update_mode"), self.update_mode)
            form.addRow(tr("training.ppo_batch_size"), self.ppo_batch_size)
            form.addRow(tr("training.sl_epochs"), self.sl_epochs)
            form.addRow(tr("training.seed"), self.seed)
            form.addRow(tr("training.report_path"), self.report)
            form.addRow(tr("training.output_path"), self.output)
            form.addRow(tr("training.artifact_path"), self.artifact)
            form.addRow(tr("training.resume_artifact_path"), self.resume_artifact)
            form.addRow(tr("training.replay_source_path"), self.replay_source)
            form.addRow(tr("training.phase2_artifact_path"), self.phase2_artifact)
            # Item 3: hover help on every field AND its QFormLayout label so the
            # AI-training settings menu gets mouse-over tooltips too.
            field_tooltips = (
                (self.mode, "tooltip.training_mode"),
                (self.episodes, "tooltip.training_episodes"),
                (self.target, "tooltip.training_target"),
                (self.update_mode, "tooltip.training_update_mode"),
                (self.ppo_batch_size, "tooltip.training_ppo_batch_size"),
                (self.sl_epochs, "tooltip.training_sl_epochs"),
                (self.seed, "tooltip.training_seed"),
                (self.report, "tooltip.training_report_path"),
                (self.output, "tooltip.training_output_path"),
                (self.artifact, "tooltip.training_artifact_path"),
                (self.resume_artifact, "tooltip.training_resume_artifact_path"),
                (self.replay_source, "tooltip.training_replay_source_path"),
                (self.phase2_artifact, "tooltip.training_phase2_artifact_path"),
            )
            for field, key in field_tooltips:
                field.setToolTip(tr(key))
                label = form.labelForField(field)
                if label is not None:
                    label.setToolTip(tr(key))
            self._update_training_fields()

        def _update_training_fields(self) -> None:
            mode = self.mode.currentData()
            policy = mode == "policy_ppo"
            collection = mode == "data_collection"
            inference = mode == "inference_sl"
            self.episodes.setEnabled(collection or policy)
            self.target.setEnabled(policy)
            self.update_mode.setEnabled(policy)
            self.ppo_batch_size.setEnabled(policy and self.update_mode.currentData() == "semibatch")
            self.replay_source.setEnabled(collection or inference)
            self.phase2_artifact.setEnabled(policy)
            self.sl_epochs.setEnabled(inference)

        def setup(self) -> TrainingSetup:
            target = self.target.value() or None
            return TrainingSetup(
                config=self.config,
                mode=str(self.mode.currentData() or "rl"),
                episodes=self.episodes.value(),
                target_transitions=target,
                seed=self.seed.value(),
                report_path=self.report.text().strip() or None,
                output_path=self.output.text().strip() or None,
                artifact_path=self.artifact.text().strip() or None,
                resume_artifact_path=self.resume_artifact.text().strip() or None,
                replay_source_path=self.replay_source.text().strip() or None,
                phase2_artifact_path=self.phase2_artifact.text().strip() or None,
                sl_epochs=self.sl_epochs.value(),
                ppo_update_mode=str(self.update_mode.currentData() or "batch"),
                ppo_batch_size=self.ppo_batch_size.value(),
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
            self.progress_bar.setFormat("MainAgent transitions: %v / %m")
            self.environment_steps = QLabel("Environment steps: 0", self)
            self.truncated = QLabel("Truncated: 0 (0.0%)", self)
            self.metrics = QLabel("Metrics: —", self)
            self.status = QLabel(tr("training.no_job"), self)
            self.snapshot = QLabel(tr("training.snapshot_not_started"), self)
            self.paths = QLabel("Report: — · Output: — · Artifact: —", self)
            self.logs = QPlainTextEdit(self)
            self.logs.setReadOnly(True)
            self.message = QLabel(self)
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
            layout = QVBoxLayout(self)
            layout.addWidget(self.back_button)
            columns = QHBoxLayout()
            left_panel = QWidget(self)
            left_panel.setMinimumWidth(300)
            left_panel.setMaximumWidth(370)
            left = QVBoxLayout(left_panel)
            left.setContentsMargins(0, 0, 8, 0)
            left.addWidget(self.setup_panel)
            left.addLayout(buttons)
            left.addWidget(self.progress_bar)
            left.addWidget(self.environment_steps)
            left.addWidget(self.truncated)
            left.addWidget(self.metrics)
            left.addWidget(self.status)
            left.addWidget(self.snapshot)
            left.addWidget(self.paths)
            columns.addWidget(left_panel, 0)
            right = QVBoxLayout()
            if PYQTGRAPH_AVAILABLE:
                self.plot = PlotWidget(self)
                self.plot.addLegend()
                self.plot.setLabel("left", "Win rate")
                self.plot.setLabel("bottom", "Monitor sample")
                self._curve = self.plot.plot(
                    [], [], name="Win rate / Role accuracy", pen="#4C9AFF", symbol="o", symbolSize=8, symbolBrush="#4C9AFF"
                )
                self._emotion_curve = self.plot.plot(
                    [], [], name="Emotion score (1-MSE)", pen="#F39C12", symbol="o", symbolSize=8, symbolBrush="#F39C12"
                )
                right.addWidget(self.plot)
            else:
                self.plot = None
                right.addWidget(QLabel(tr("training.graph_needs_pyqtgraph"), self))
            columns.addLayout(right, 1)
            layout.addLayout(columns)
            layout.addWidget(self.logs)
            layout.addWidget(self.message)
            self._timer = QTimer(self)
            self._timer.setInterval(250)
            self._timer.timeout.connect(self.controller.refresh)
            self._timer.start()
            self._render(self.controller.snapshot)

        def _go_back(self) -> None:
            self._timer.stop()
            # The central widget can be replaced while a job is still running.
            # Do not let a later refresh call a deleted widget; a new monitor
            # binds the controller again when the training screen is reopened.
            self.controller.on_update = None
            if self.on_back is not None:
                self.on_back()

        def _start(self) -> None:
            # Starting a job must also recover the polling loop when this
            # monitor was previously hidden/reused.
            self._timer.start()
            state = self.controller.start(self.setup_panel.setup())
            # The controller normally publishes through on_update. Render the
            # returned snapshot explicitly as well so the first click is
            # visible even when a short job completes before the first timer
            # tick or the callback was replaced by screen navigation.
            self._render(state)

        def _pause(self) -> None:
            self.controller.pause(timeout=30.0)

        def _resume(self) -> None:
            self.controller.resume()

        def _stop(self) -> None:
            self.controller.cancel()

        def _render(self, state: TrainingMonitorSnapshot) -> None:
            progress = state.progress
            if progress is None:
                self.progress_bar.setRange(0, 1)
                self.progress_bar.setValue(0)
                self.status.setText(tr("training.no_job"))
                self.snapshot.setText(tr("training.snapshot_not_started"))
                self.paths.setText("Report: — · Output: — · Artifact: —")
                self.message.setText(state.message or "")
                return
            target = getattr(state.handle, "snapshot", None)
            target_config = getattr(target, "config_snapshot", {}) if target is not None else {}
            training = target_config.get("rl_training", {}) if isinstance(target_config, dict) else {}
            ppo = training.get("ppo", {}) if isinstance(training, dict) else {}
            maximum = int(getattr(progress, "main_agent_transitions", 0))
            requested = int(ppo.get("rollout_transitions_per_worker", maximum) or maximum or 1)
            self.progress_bar.setRange(0, max(requested, maximum, 1))
            self.progress_bar.setValue(maximum)
            self.environment_steps.setText(f"Environment steps: {progress.environment_steps}")
            count = int(getattr(progress, "truncated_episodes", 0))
            rate = getattr(progress, "truncated_rate", None)
            self.truncated.setText(f"Truncated: {count} ({'N/A' if rate is None else f'{rate:.1%}'})")
            metrics = _public_value(getattr(progress, "metrics", {}) or {})
            self.metrics.setText(f"Metrics: {json.dumps(metrics, sort_keys=True)}")
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
            log_text = "\n".join(state.logs)
            if self.logs.toPlainText() != log_text:
                scrollbar = self.logs.verticalScrollBar()
                at_bottom = scrollbar.value() >= scrollbar.maximum() - 2
                previous_value = scrollbar.value()
                self.logs.setPlainText(log_text)
                if at_bottom:
                    scrollbar.setValue(scrollbar.maximum())
                else:
                    scrollbar.setValue(min(previous_value, scrollbar.maximum()))
            self.message.setText(state.message or "")
            if self.plot is not None:
                mode = state.public_metadata.get("mode") if isinstance(state.public_metadata, Mapping) else None
                if mode == "data_collection":
                    self.plot.setVisible(False)
                    self._curve.setData([], [])
                    self._emotion_curve.setData([], [])
                else:
                    self.plot.setVisible(True)
                    metrics = _public_value(getattr(progress, "metrics", {}) or {})
                    if isinstance(metrics, dict) and metrics.get("phase") == "phase2_supervised":
                        self.plot.setLabel("left", "Validation score")
                        self.plot.setLabel("bottom", "Epoch")
                        self._curve.setData(
                            list(range(1, len(state.role_accuracy_points) + 1)),
                            list(state.role_accuracy_points),
                        )
                        self._emotion_curve.setData(
                            list(range(1, len(state.emotion_score_points) + 1)),
                            list(state.emotion_score_points),
                        )
                    else:
                        self.plot.setLabel("left", "Win rate")
                        self.plot.setLabel("bottom", "Monitor sample")
                        self._curve.setData(list(range(len(state.win_rate_points))), list(state.win_rate_points))
                        self._emotion_curve.setData([], [])


else:

    class TrainingSetupPanel:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

    class TrainingMonitor:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["PYQTGRAPH_AVAILABLE", "TrainingMonitor", "TrainingSetupPanel"]
