"""Optional PyQt6 Training Setup and Training Monitor screens."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from collections.abc import Callable, Mapping

from bang_rl.paths import ARTIFACTS_DIR, OUTPUTS_DIR, REPORTS_DIR

from ..qt_compat import (
    QCheckBox,
    QColor,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
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

    class _LegendSwatch(QLabel):
        """A single legend entry that highlights its curve on mouse hover.

        Hovering reports the entry name to ``on_hover``; leaving reports None so
        the monitor can brighten the hovered curve and dim the others.
        """

        def __init__(
            self,
            name: str,
            color: str,
            on_hover: Callable[[str | None], None],
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(f"\u25cf {name}", parent)
            self._name = name
            self._on_hover = on_hover
            self.setStyleSheet(f"color: {color}; padding: 0px 6px;")

        def enterEvent(self, event: Any) -> None:  # noqa: N802 - Qt signature
            self._on_hover(self._name)
            super().enterEvent(event)

        def leaveEvent(self, event: Any) -> None:  # noqa: N802 - Qt signature
            self._on_hover(None)
            super().leaveEvent(event)

    class TrainingSetupPanel(QWidget):
        """Small setup form; values are passed to the service unchanged."""

        def __init__(
            self,
            config: Any = None,
            parent: QWidget | None = None,
            *,
            model_root: str | Path | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.config = config
            # Where saved .phase2model modules are looked up (None = project default).
            self.model_root = model_root
            self.mode = QComboBox(self)
            self.mode.addItem(tr("training.mode.data_collection"), TRAINING_MODE_DATA_COLLECTION)
            self.mode.addItem(tr("training.mode.module_training"), TRAINING_MODE_MODULE_TRAINING)
            self.mode.addItem(tr("training.mode.main_ai_training"), TRAINING_MODE_MAIN_AI)
            self.episodes = QSpinBox(self)
            self.episodes.setRange(1, 2_147_483_647)
            self.episodes.setValue(1)
            # Epochs and training time are two exclusive ways to bound module
            # training. A number of epochs trains exactly that many; 0 (shown as
            # "blank") switches to time-limited training instead.
            self.epochs = QSpinBox(self)
            self.epochs.setRange(0, 2_147_483_647)
            self.epochs.setSpecialValueText(tr("training.epochs_blank"))
            self.epochs.setValue(self._config_epochs())
            # Max training time in minutes, used only while epochs is blank.
            # Blank = unlimited (train until the user presses Stop).
            self.max_minutes = QLineEdit(self)
            self.max_minutes.setPlaceholderText(tr("training.max_minutes_placeholder"))
            self.epochs.valueChanged.connect(self._sync_limit_mode)
            self._sync_limit_mode()
            # One graph point is stored per this many training batches.
            self.graph_interval = QSpinBox(self)
            self.graph_interval.setRange(1, 10_000_000)
            self.graph_interval.setValue(1000)
            self.target = QSpinBox(self)
            self.target.setRange(0, 2_147_483_647)
            self.target.setSpecialValueText("Use episodes")
            self.seed = QSpinBox(self)
            self.seed.setRange(-2_147_483_648, 2_147_483_647)
            self.report = QLineEdit(str(REPORTS_DIR), self)
            self.output = QLineEdit(str(OUTPUTS_DIR), self)
            self.artifact = QLineEdit(str(ARTIFACTS_DIR), self)
            self.resume_artifact = QLineEdit(str(ARTIFACTS_DIR), self)
            # Saved intent/emotion modules to continue module training from.
            self.resume_module = QComboBox(self)
            self.resume_module_refresh = QPushButton(tr("perf.refresh"), self)
            self.resume_module_refresh.clicked.connect(self.refresh_resume_runs)
            resume_row = QWidget(self)
            resume_row_layout = QHBoxLayout(resume_row)
            resume_row_layout.setContentsMargins(0, 0, 0, 0)
            resume_row_layout.addWidget(self.resume_module, 1)
            resume_row_layout.addWidget(self.resume_module_refresh)
            self._resume_runs: dict[str, Any] = {}
            self.refresh_resume_runs()
            # Build the form unparented; ``outer`` below is the panel's only
            # layout and holds the form plus the hardware group.
            form = QFormLayout()
            form.addRow(tr("training.mode"), self.mode)
            form.addRow(tr("training.episodes"), self.episodes)
            form.addRow(tr("training.epochs"), self.epochs)
            form.addRow(tr("training.max_minutes"), self.max_minutes)
            form.addRow(tr("training.graph_interval"), self.graph_interval)
            form.addRow(tr("training.target"), self.target)
            form.addRow(tr("training.seed"), self.seed)
            form.addRow(tr("training.report_path"), self.report)
            form.addRow(tr("training.output_path"), self.output)
            form.addRow(tr("training.artifact_path"), self.artifact)
            form.addRow(tr("training.resume_module"), resume_row)
            form.addRow(tr("training.resume_artifact_path"), self.resume_artifact)

            # Hardware / AI-training controls, moved here from the Settings
            # screen so they sit next to the job they configure. Their values
            # are injected into the SystemConfig in setup().
            hw = self._config_hardware()
            self.clip = QDoubleSpinBox(self)
            self.clip.setRange(0.000001, 0.999999)
            self.clip.setDecimals(6)
            self.clip.setValue(self._config_clip())
            self.cpu = QSpinBox(self)
            self.cpu.setRange(1, 4096)
            self.cpu.setValue(int(hw.get("cpu_cores", 1)))
            self.ram = QDoubleSpinBox(self)
            self.ram.setRange(0.001, 1_048_576.0)
            self.ram.setValue(float(hw.get("ram_gb", 1.0)))
            self.workers = QSpinBox(self)
            self.workers.setRange(1, 4096)
            self.workers.setValue(int(hw.get("workers", 1)))
            self.batch = QSpinBox(self)
            self.batch.setRange(1, 1_000_000)
            self.batch.setValue(int(hw.get("batch_size", 64)))
            self.buffer = QSpinBox(self)
            self.buffer.setRange(1, 1_000_000_000)
            self.buffer.setValue(int(hw.get("buffer_size", 256)))
            self.gpu = QCheckBox(tr("settings.gpu_check"), self)
            self.gpu.setChecked(bool(hw.get("use_gpu", False)))
            self.hardware_group = QGroupBox(tr("training.hardware_group"), self)
            hardware_form = QFormLayout(self.hardware_group)
            hardware_form.addRow(tr("settings.ppo_clip"), self.clip)
            hardware_form.addRow(tr("settings.cpu"), self.cpu)
            hardware_form.addRow(tr("settings.ram"), self.ram)
            hardware_form.addRow(tr("settings.workers"), self.workers)
            hardware_form.addRow(tr("settings.batch"), self.batch)
            hardware_form.addRow(tr("settings.buffer"), self.buffer)
            hardware_form.addRow(tr("settings.gpu"), self.gpu)
            self.hardware_form = hardware_form

            outer = QVBoxLayout(self)
            outer.addLayout(form)
            outer.addWidget(self.hardware_group)

            # Item 3: hover help on every field AND its QFormLayout label so the
            # AI-training settings menu gets mouse-over tooltips too.
            field_tooltips = (
                (self.mode, "tooltip.training_mode"),
                (self.episodes, "tooltip.training_episodes"),
                (self.epochs, "tooltip.training_epochs"),
                (self.max_minutes, "tooltip.training_max_minutes"),
                (self.graph_interval, "tooltip.training_graph_interval"),
                (self.target, "tooltip.training_target"),
                (self.seed, "tooltip.training_seed"),
                (self.report, "tooltip.training_report_path"),
                (self.output, "tooltip.training_output_path"),
                (self.artifact, "tooltip.training_artifact_path"),
                (self.resume_artifact, "tooltip.training_resume_artifact_path"),
                (self.resume_module, "tooltip.training_resume_module"),
            )
            label = form.labelForField(resume_row)
            if label is not None:
                label.setToolTip(tr("tooltip.training_resume_module"))
            for field, key in field_tooltips:
                field.setToolTip(tr(key))
                label = form.labelForField(field)
                if label is not None:
                    label.setToolTip(tr(key))
            hardware_tooltips = (
                (self.clip, "tooltip.ppo_clip"),
                (self.cpu, "tooltip.cpu"),
                (self.ram, "tooltip.ram"),
                (self.workers, "tooltip.workers"),
                (self.batch, "tooltip.batch"),
                (self.buffer, "tooltip.buffer"),
                (self.gpu, "tooltip.gpu"),
            )
            for field, key in hardware_tooltips:
                field.setToolTip(tr(key))
                label = hardware_form.labelForField(field)
                if label is not None:
                    label.setToolTip(tr(key))

        def _config_mapping(self) -> dict[str, Any]:
            """Return the panel's config as a plain dict (JSON-safe), or {}."""

            config = self.config
            if config is None:
                return {}
            if hasattr(config, "model_dump"):
                return dict(config.model_dump(mode="json"))
            if isinstance(config, Mapping):
                return dict(config)
            return {}

        def _config_epochs(self) -> int:
            sl = self._config_mapping().get("sl_training") or {}
            try:
                return max(1, int(sl.get("epochs", 1)))
            except (TypeError, ValueError):
                return 1

        def _config_hardware(self) -> dict[str, Any]:
            hardware = self._config_mapping().get("hardware")
            return dict(hardware) if isinstance(hardware, Mapping) else {}

        def _config_clip(self) -> float:
            rl = self._config_mapping().get("rl_training") or {}
            ppo = rl.get("ppo") or {}
            try:
                return float(ppo.get("clip_epsilon", 0.2))
            except (TypeError, ValueError):
                return 0.2

        def _config_with_overrides(self) -> Any:
            """Return the config with the panel's hardware/epoch edits applied.

            The GUI form is the source of truth for these fields at launch, so
            the edited values are merged into a fresh SystemConfig that is handed
            to the training service unchanged.
            """

            payload = self._config_mapping()
            sl = dict(payload.get("sl_training") or {})
            # The schema requires epochs >= 1; in time mode (epochs blank) the
            # epoch count is ignored by the runner, so 1 is just a placeholder.
            sl["epochs"] = max(1, int(self.epochs.value()))
            payload["sl_training"] = sl

            hardware = dict(payload.get("hardware") or {})
            hardware.update(
                {
                    "cpu_cores": int(self.cpu.value()),
                    "ram_gb": float(self.ram.value()),
                    "use_gpu": bool(self.gpu.isChecked()),
                    "workers": int(self.workers.value()),
                    "batch_size": int(self.batch.value()),
                    "buffer_size": int(self.buffer.value()),
                }
            )
            payload["hardware"] = hardware

            rl = dict(payload.get("rl_training") or {})
            ppo = dict(rl.get("ppo") or {})
            ppo["clip_epsilon"] = float(self.clip.value())
            rl["ppo"] = ppo
            payload["rl_training"] = rl

            if not payload:
                return self.config
            from bang_rl.config import SystemConfig

            return SystemConfig.model_validate(payload)

        def refresh_resume_runs(self) -> None:
            """Reload saved module runs into the 'continue from' list."""

            from bang_rl.training.module_artifacts import list_saved_module_runs
            from ..dialogs.module_performance import module_run_label

            previous = self.resume_module.currentData()
            self.resume_module.blockSignals(True)
            self.resume_module.clear()
            self.resume_module.addItem(tr("training.resume_module_none"), None)
            try:
                runs = list_saved_module_runs(self.model_root)
            except OSError:
                runs = ()
            self._resume_runs = {run.job_id: run for run in runs}
            for run in runs:
                self.resume_module.addItem(module_run_label(run), run.job_id)
            index = self.resume_module.findData(previous) if previous else 0
            self.resume_module.setCurrentIndex(max(0, index))
            self.resume_module.blockSignals(False)

        def select_resume_job(self, job_id: str | None) -> bool:
            """Preselect a saved run to continue from; False if it is unknown."""

            if job_id is not None and job_id not in self._resume_runs:
                self.refresh_resume_runs()
            index = self.resume_module.findData(job_id) if job_id else 0
            if index < 0:
                return False
            self.resume_module.setCurrentIndex(index)
            return True

        def _resume_paths(self) -> tuple[str | None, str | None]:
            run = self._resume_runs.get(self.resume_module.currentData() or "")
            if run is None:
                return None, None
            return (
                None if run.intent_path is None else str(run.intent_path),
                None if run.emotion_path is None else str(run.emotion_path),
            )

        def _train_by_time(self) -> bool:
            """True when the epochs field is blank (0): bound training by time."""

            return int(self.epochs.value()) == 0

        def _sync_limit_mode(self, _value: int | None = None) -> None:
            """Enable the time field only while epochs is blank."""

            self.max_minutes.setEnabled(self._train_by_time())

        def _max_training_seconds(self) -> float | None:
            """Parse the 'max minutes' field: blank/0/invalid => unlimited."""

            if not self._train_by_time():
                return None
            text = self.max_minutes.text().strip()
            if not text:
                return None
            try:
                minutes = float(text)
            except ValueError:
                return None
            if minutes <= 0:
                return None
            return minutes * 60.0

        def setup(self) -> TrainingSetup:
            target = self.target.value() or None
            resume_intent, resume_emotion = self._resume_paths()
            return TrainingSetup(
                resume_intent_artifact_path=resume_intent,
                resume_emotion_artifact_path=resume_emotion,
                config=self._config_with_overrides(),
                mode=str(self.mode.currentData() or TRAINING_MODE_DATA_COLLECTION),
                episodes=self.episodes.value(),
                target_transitions=target,
                seed=self.seed.value(),
                report_path=self.report.text().strip() or None,
                output_path=self.output.text().strip() or None,
                artifact_path=self.artifact.text().strip() or None,
                resume_artifact_path=self.resume_artifact.text().strip() or None,
                train_by_time=self._train_by_time(),
                max_training_seconds=self._max_training_seconds(),
                graph_point_interval=int(self.graph_interval.value()),
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
            self.performance_button = QPushButton(tr("perf.open"), self)
            self.performance_button.setToolTip(tr("tooltip.perf_open"))
            self.performance_button.clicked.connect(self._open_performance)
            self._last_status: str | None = None

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
            # Curves currently drawn, as (name, curve_item, base_color_hex).
            # Used to rebuild the external legend and to brighten/dim on hover.
            self._active_curves: list[tuple[str, Any, str]] = []
            self._legend_swatches: list[QWidget] = []
            self._graph_signature_cache: tuple[Any, ...] | None = None
            # External legend row, kept OUTSIDE the plot axes (below the graph),
            # so curve lines are never hidden behind an overlapping legend box.
            self.legend_bar = QWidget(self)
            self._legend_layout = QHBoxLayout(self.legend_bar)
            self._legend_layout.setContentsMargins(4, 0, 4, 0)
            self._legend_layout.addStretch(1)
            if PYQTGRAPH_AVAILABLE:
                self.plot = PlotWidget(self)
                self.plot.setLabel("left", "Accuracy")
                self.plot.setLabel("bottom", "Batch")
                # No in-plot addLegend(): the legend lives in self.legend_bar,
                # outside the axes, with hover highlighting.
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
            left_panel.addWidget(self.performance_button)
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
            right_panel.addWidget(self.legend_bar)
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

        def _open_performance(self) -> None:
            from ..dialogs.module_performance import ModulePerformanceDialog

            dialog = ModulePerformanceDialog(
                model_root=getattr(self.setup_panel, "model_root", None), parent=self
            )
            if dialog.exec() and dialog.selected_job_id:
                if self.setup_panel.select_resume_job(dialog.selected_job_id):
                    self.message.setText(tr("perf.resume_selected", job=dialog.selected_job_id))

        def _refresh_resume_on_finish(self, status: str) -> None:
            """Newly saved modules appear in the 'continue from' list."""

            if status != self._last_status and status in {"finished", "cancelled"}:
                self.setup_panel.refresh_resume_runs()
            self._last_status = status

        def _copy_metrics(self) -> None:
            if QApplication is not None:
                QApplication.clipboard().setText(self._metrics_full_text)

        def _set_graph_mode(self, mode: str) -> None:
            self._graph_mode = mode if mode in {"accuracy", "roles", "emotions"} else "accuracy"
            self._graph_signature_cache = None  # force a redraw for the new mode
            is_default = self._graph_mode == "accuracy"
            self.role_graph_button.setVisible(is_default)
            self.emotion_graph_button.setVisible(is_default)
            self.accuracy_graph_button.setVisible(not is_default)
            self._render(self.controller.snapshot)

        @staticmethod
        def _dim_color(color_hex: str, *, dim: bool) -> QColor:
            """Return the curve colour, darkened/faded when it is not hovered."""

            qcolor = QColor(color_hex)
            if not dim:
                return qcolor
            faded = QColor(qcolor)
            faded.setAlpha(60)
            return faded

        def _apply_curve_pen(self, curve: Any, color_hex: str, *, hovered: bool, width: int) -> None:
            import pyqtgraph as pg

            curve.setPen(pg.mkPen(self._dim_color(color_hex, dim=not hovered), width=width))

        def _highlight(self, name: str | None) -> None:
            """Brighten the hovered curve and dim the rest.

            ``name is None`` (no hover) restores every curve to full colour.
            """

            for entry_name, curve, color_hex in self._active_curves:
                hovered = name is None or entry_name == name
                width = 3 if (name is not None and entry_name == name) else 2
                self._apply_curve_pen(curve, color_hex, hovered=hovered, width=width)

        def _clear_legend(self) -> None:
            for widget in self._legend_swatches:
                widget.setParent(None)
                widget.deleteLater()
            self._legend_swatches = []

        def _rebuild_legend(self) -> None:
            self._clear_legend()
            for name, _curve, color_hex in self._active_curves:
                swatch = _LegendSwatch(name, color_hex, self._highlight, self.legend_bar)
                # Insert before the trailing stretch so entries pack to the left.
                self._legend_layout.insertWidget(self._legend_layout.count() - 1, swatch)
                self._legend_swatches.append(swatch)

        def _plot_matrix(self, points: Any, names: Any, colors: tuple[str, ...]) -> None:
            if self.plot is None:
                return
            self.plot.clear()
            self._active_curves = []
            rows = [row for row in points if isinstance(row, (list, tuple))]
            labels = list(names or ())
            for index, name in enumerate(labels):
                values = [float(row[index]) for row in rows if len(row) > index]
                color_hex = colors[index % len(colors)]
                curve = self.plot.plot(
                    list(range(1, len(values) + 1)),
                    values,
                    pen=color_hex,
                    name=str(name),
                )
                self._active_curves.append((str(name), curve, color_hex))
            self._highlight(None)
            self._rebuild_legend()

        def _graph_signature(self, state: TrainingMonitorSnapshot) -> tuple[Any, ...]:
            """Cheap fingerprint of what the current graph mode would draw."""

            job = getattr(getattr(state, "snapshot", None), "job_id", None)
            if self._graph_mode == "roles":
                return (
                    "roles", job,
                    len(getattr(state, "intent_accuracy_by_role_points", ())),
                    tuple(getattr(state, "intent_role_names", ())),
                )
            if self._graph_mode == "emotions":
                return (
                    "emotions", job,
                    len(getattr(state, "emotion_accuracy_by_dimension_points", ())),
                    tuple(getattr(state, "emotion_dimension_names", ())),
                )
            return (
                "accuracy", job,
                len(getattr(state, "intent_accuracy_points", ())),
                len(getattr(state, "emotion_accuracy_points", ())),
            )

        def _render_graph(self, state: TrainingMonitorSnapshot) -> None:
            if self.plot is None:
                return
            # The monitor re-renders every poll (250 ms); only redraw the plot
            # when new points arrived, since clearing and re-plotting is costly.
            signature = self._graph_signature(state)
            if signature == self._graph_signature_cache:
                return
            self._graph_signature_cache = signature
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
            self._active_curves = []
            intent_values = getattr(state, "intent_accuracy_points", ())
            emotion_values = getattr(state, "emotion_accuracy_points", ())
            intent_curve = self.plot.plot(
                list(range(1, len(intent_values) + 1)),
                list(intent_values),
                pen="#f6c344",
                name="Intent",
            )
            emotion_curve = self.plot.plot(
                list(range(1, len(emotion_values) + 1)),
                list(emotion_values),
                pen="#5bc0eb",
                name="Emotion",
            )
            self._active_curves = [
                ("Intent", intent_curve, "#f6c344"),
                ("Emotion", emotion_curve, "#5bc0eb"),
            ]
            self._highlight(None)
            self._rebuild_legend()

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
            self._refresh_resume_on_finish(str(progress.status))
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
