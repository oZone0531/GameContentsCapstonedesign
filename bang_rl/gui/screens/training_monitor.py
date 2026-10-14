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
    QPushButton,
    QSpinBox,
    QTimer,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from ..i18n import tr
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
                mode=self.mode.text().strip() or "rl",
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
            self.progress_bar.setFormat("MainAgent transitions: %v / %m")
            self.environment_steps = QLabel("Environment steps: 0", self)
            self.truncated = QLabel("Truncated: 0 (0.0%)", self)
            self.metrics = QLabel("Metrics: —", self)
            self.status = QLabel(tr("training.no_job"), self)
            self.snapshot = QLabel(tr("training.snapshot_not_started"), self)
            self.paths = QLabel("Report: — · Output: — · Artifact: —", self)
            self.logs = QPlainTextEdit(self)
            self.logs.setReadOnly(True)
            # Dedicated status / error area. Styled so a start failure, a
            # faulted run, or a missing-dependency notice stands out from the
            # normal counter labels, mirroring the game screen's error-surfacing
            # area. Only the label's own appearance changes; the layout contract
            # (a single QLabel appended at the end) is untouched.
            self.message = QLabel(self)
            self.message.setWordWrap(True)
            self.message.setStyleSheet("color: #b00020; font-weight: bold;")
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
                layout.addWidget(QLabel(tr("training.graph_needs_pyqtgraph"), self))
            layout.addWidget(self.logs)
            layout.addWidget(self.message)
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

        def _render(self, state: TrainingMonitorSnapshot) -> None:
            progress = state.progress
            if progress is None:
                self.progress_bar.setRange(0, 1)
                self.progress_bar.setValue(0)
                self.status.setText(tr("training.no_job"))
                self.snapshot.setText(tr("training.snapshot_not_started"))
                self.paths.setText("Report: — · Output: — · Artifact: —")
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
            status_text = f"Status: {progress.status} · PPO updates: {progress.ppo_updates}"
            if progress.status == "running":
                status_text = f"{tr('training.in_progress')} · {status_text}"
            self.status.setText(status_text)
            if state.snapshot is not None:
                self.snapshot.setText(
                    f"Snapshot: {state.snapshot.job_id} · config {state.snapshot.config_version} "
                    f"· seed {state.snapshot.seed}"
                )
            self.paths.setText(
                f"Report: {progress.report_path or '—'} · Output: {getattr(progress, 'output_path', None) or '—'} · "
                f"Artifact: {progress.artifact_path or '—'}"
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
