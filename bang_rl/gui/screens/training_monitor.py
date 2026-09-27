"""Optional PyQt6 Training Setup and Training Monitor screens."""
from __future__ import annotations

import json
from typing import Any

from ..qt_compat import (
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
from ..training import TrainingMonitorController, TrainingMonitorSnapshot, TrainingSetup

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
            self.mode = QLineEdit("rl", self)
            self.episodes = QSpinBox(self)
            self.episodes.setRange(0, 2_147_483_647)
            self.episodes.setValue(1)
            self.target = QSpinBox(self)
            self.target.setRange(0, 2_147_483_647)
            self.target.setSpecialValueText("Use episodes")
            self.seed = QSpinBox(self)
            self.seed.setRange(-2_147_483_648, 2_147_483_647)
            self.report = QLineEdit(self)
            self.output = QLineEdit(self)
            self.artifact = QLineEdit(self)
            form = QFormLayout(self)
            form.addRow("Mode", self.mode)
            form.addRow("Episodes", self.episodes)
            form.addRow("Target MainAgent transitions", self.target)
            form.addRow("Seed", self.seed)
            form.addRow("Report path", self.report)
            form.addRow("Output path", self.output)
            form.addRow("Artifact path", self.artifact)

        def setup(self) -> TrainingSetup:
            target = self.target.value() or None
            return TrainingSetup(
                config=self.config,
                mode=self.mode.text().strip() or "rl",
                episodes=self.episodes.value(),
                target_transitions=target,
                seed=self.seed.value(),
                report_path=self.report.text().strip() or None,
                output_path=self.output.text().strip() or None,
                artifact_path=self.artifact.text().strip() or None,
            )


    class TrainingMonitor(QWidget):
        """Live monitor for public counters, metrics, logs, and lifecycle state."""

        def __init__(
            self,
            controller: TrainingMonitorController | None = None,
            *,
            setup_panel: TrainingSetupPanel | None = None,
            parent: QWidget | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.controller = controller or TrainingMonitorController()
            self.setup_panel = setup_panel or TrainingSetupPanel(parent=self)
            self.controller.on_update = self._render
            self.progress_bar = QProgressBar(self)
            self.progress_bar.setFormat("MainAgent transitions: %v / %m")
            self.environment_steps = QLabel("Environment steps: 0", self)
            self.truncated = QLabel("Truncated: 0 (0.0%)", self)
            self.metrics = QLabel("Metrics: —", self)
            self.status = QLabel("No training job", self)
            self.snapshot = QLabel("Snapshot: not started", self)
            self.paths = QLabel("Report: — · Artifact: —", self)
            self.logs = QPlainTextEdit(self)
            self.logs.setReadOnly(True)
            self.message = QLabel(self)
            self.start_button = QPushButton("Start", self)
            self.pause_button = QPushButton("Pause", self)
            self.resume_button = QPushButton("Resume", self)
            self.stop_button = QPushButton("Stop", self)
            buttons = QHBoxLayout()
            for button in (self.start_button, self.pause_button, self.resume_button, self.stop_button):
                buttons.addWidget(button)
            self.start_button.clicked.connect(self._start)
            self.pause_button.clicked.connect(self._pause)
            self.resume_button.clicked.connect(self._resume)
            self.stop_button.clicked.connect(self._stop)
            layout = QVBoxLayout(self)
            layout.addWidget(self.setup_panel)
            layout.addLayout(buttons)
            layout.addWidget(self.progress_bar)
            layout.addWidget(self.environment_steps)
            layout.addWidget(self.truncated)
            layout.addWidget(self.metrics)
            layout.addWidget(self.status)
            layout.addWidget(self.snapshot)
            layout.addWidget(self.paths)
            if PYQTGRAPH_AVAILABLE:
                self.plot = PlotWidget(self)
                self.plot.setLabel("left", "Win rate")
                self.plot.setLabel("bottom", "Monitor sample")
                self._curve = self.plot.plot([], [])
                layout.addWidget(self.plot)
            else:
                self.plot = None
                layout.addWidget(QLabel("Win-rate graph requires optional pyqtgraph.", self))
            layout.addWidget(self.logs)
            layout.addWidget(self.message)
            self._timer = QTimer(self)
            self._timer.setInterval(250)
            self._timer.timeout.connect(self.controller.refresh)
            self._timer.start()
            self._render(self.controller.snapshot)

        def _start(self) -> None:
            self.controller.start(self.setup_panel.setup())

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
                self.status.setText("No training job")
                self.snapshot.setText("Snapshot: not started")
                self.paths.setText("Report: — · Artifact: —")
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
            self.metrics.setText(f"Metrics: {json.dumps(getattr(progress, 'metrics', {}) or {}, sort_keys=True)}")
            self.status.setText(f"Status: {progress.status} · PPO updates: {progress.ppo_updates}")
            if state.snapshot is not None:
                self.snapshot.setText(
                    f"Snapshot: {state.snapshot.job_id} · config {state.snapshot.config_version} "
                    f"· seed {state.snapshot.seed}"
                )
            self.paths.setText(
                f"Report: {progress.report_path or '—'} · Artifact: {progress.artifact_path or '—'}"
            )
            self.logs.setPlainText("\n".join(state.logs))
            self.message.setText(state.message or "")
            if self.plot is not None:
                self._curve.setData(list(range(len(state.win_rate_points))), list(state.win_rate_points))


else:

    class TrainingSetupPanel:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

    class TrainingMonitor:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["PYQTGRAPH_AVAILABLE", "TrainingMonitor", "TrainingSetupPanel"]
