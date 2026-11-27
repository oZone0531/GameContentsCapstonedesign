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
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QPushButton,
    QSpinBox,
    QComboBox,
    QApplication,
    QStackedWidget,
    QTimer,
    Qt,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from ..i18n import tr
from .. import main_ai_view
from ..widgets.artifact_picker import ArtifactPicker, MultiArtifactPicker, PathField, open_folder

# The setup column keeps this width in every mode; the graphs take the rest.
SETUP_PANEL_WIDTH = 480
from ..training import (
    TRAINING_MODE_DATA_COLLECTION,
    TRAINING_MODE_INFERENCE_REFINEMENT,
    TRAINING_MODE_MAIN_AI,
    TRAINING_MODE_MODULE_TRAINING,
    PresetResult,
    TrainingMonitorController,
    TrainingMonitorSnapshot,
    TrainingPreset,
    TrainingSetup,
    controller_status,
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
def _norm_path(value: Any) -> str:
    """Comparable form of a file or folder path (case and separators normalized)."""

    import os

    return os.path.normcase(os.path.abspath(str(value)))


def _lock_plot(plot: Any) -> Any:
    """Make a pyqtgraph plot view-only: no mouse/wheel zoom or pan, no menu, no auto-range button.

    The view follows the data by itself (auto range on both axes; a fixed Y
    range set afterwards with ``setYRange`` stays fixed).
    """

    plot.setMouseEnabled(x=False, y=False)
    plot.setMenuEnabled(False)
    plot.hideButtons()
    plot.enableAutoRange(x=True, y=True)
    return plot


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
            main_ai_dirs: Any = None,
            dataset_root: str | Path | None = None,
            settings_editor: Any = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.config = config
            # When given, the hardware group is written to the local (git-ignored)
            # settings file whenever it changes, so it survives a restart.
            self._settings_editor = settings_editor
            # Where saved .phase2model modules are looked up (None = project default).
            self.model_root = model_root
            # A custom model root scopes the main AI models too (as in the performance dialog).
            if main_ai_dirs is None and model_root is not None:
                main_ai_dirs = [Path(model_root) / "main_ai"]
            self._main_ai_dirs = main_ai_dirs
            self._dataset_root_override = dataset_root
            self.mode = QComboBox(self)
            self.mode.addItem(tr("training.mode.data_collection"), TRAINING_MODE_DATA_COLLECTION)
            self.mode.addItem(tr("training.mode.module_training"), TRAINING_MODE_MODULE_TRAINING)
            self.mode.addItem(tr("training.mode.main_ai_training"), TRAINING_MODE_MAIN_AI)
            self.mode.addItem(tr("training.mode.inference_refinement"), TRAINING_MODE_INFERENCE_REFINEMENT)
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
            # Main AI training counts games instead of epochs; 0 also means
            # "limit by time" there, with the same max-time field.
            self.games = QSpinBox(self)
            self.games.setRange(0, 2_147_483_647)
            self.games.setSpecialValueText(tr("training.epochs_blank"))
            self.games.setValue(100)
            self.epochs.valueChanged.connect(self._sync_limit_mode)
            self.games.valueChanged.connect(self._sync_limit_mode)
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
            # Where results are written: folder fields with Browse/Open buttons,
            # grouped in the collapsible "Save locations" box built below.
            self.report = PathField(str(REPORTS_DIR), parent=self)
            self.output = PathField(str(OUTPUTS_DIR), parent=self)
            self.artifact = PathField(str(ARTIFACTS_DIR), parent=self)
            # Main AI training: the saved .bangmodel to continue from (or none).
            self.resume_main_ai_row = ArtifactPicker(
                on_refresh=self.refresh_resume_main_ai_models,
                folder=lambda: ARTIFACTS_DIR,
                browse_filter=tr("picker.bangmodel_filter"),
                parent=self,
            )
            self.resume_main_ai = self.resume_main_ai_row.combo
            self.refresh_resume_main_ai_models()
            # Saved intent/emotion modules to continue module training from.
            resume_row = self.resume_row = ArtifactPicker(
                on_refresh=self.refresh_resume_runs,
                folder=self._module_folder,
                parent=self,
            )
            self.resume_module = resume_row.combo
            self.resume_module_refresh = resume_row.refresh_button
            self._resume_runs: dict[str, Any] = {}
            # Main-AI compatibility of each listed run: job id -> (ok, reason).
            self._resume_compat: dict[str, tuple[bool, str | None]] = {}
            # Last choice per kind ("module" or "main"), so switching the mode
            # back and forth keeps each selection.
            self._resume_selection: dict[str, str | None] = {}
            self._resume_kind_shown = self._resume_kind()
            self.refresh_resume_runs()
            # Data collection: dummies only, or the chosen main AI among four dummies.
            self.players = QComboBox(self)
            self.players.addItem(tr("training.collection_players.dummy5"), "dummy5")
            self.players.addItem(tr("training.collection_players.main1_dummy4"), "main1-dummy4")
            # Data collection: give every dummy a random personality preset per game.
            self.personality = QCheckBox(tr("training.personality_check"), self)
            self.personality.setChecked(self._config_personality())
            # The main AI model that plays (collection) or whose PPO stays fixed (refinement).
            base_model_row = self.base_model_row = ArtifactPicker(
                on_refresh=self.refresh_base_models,
                folder=lambda: ARTIFACTS_DIR,
                parent=self,
            )
            self.base_model = base_model_row.combo
            self.base_model_refresh = base_model_row.refresh_button
            self.action_mode = QComboBox(self)
            self.action_mode.addItem(tr("training.action_mode.deterministic"), "deterministic")
            self.action_mode.addItem(tr("training.action_mode.stochastic"), "stochastic")
            # Datasets (dummy5 and main1+dummy4, local and saved) to train the modules on.
            # Same row as the single pickers; the list opens as a drop-down.
            datasets_row = self.datasets_row = MultiArtifactPicker(
                on_refresh=self.refresh_datasets,
                folder=lambda: self._dataset_root()[0],
                none_text=tr("training.datasets_all"),
                parent=self,
            )
            self.datasets = datasets_row.list
            self.datasets_refresh = datasets_row.refresh_button
            # Inference refinement settings.
            self.refine_games = QSpinBox(self)
            self.refine_games.setRange(1, 2_147_483_647)
            self.refine_games.setValue(5000)
            self.refine_rounds = QSpinBox(self)
            self.refine_rounds.setRange(1, 1000)
            self.refine_rounds.setValue(5)
            self.refine_epochs = QSpinBox(self)
            self.refine_epochs.setRange(1, 100_000)
            self.refine_epochs.setValue(5)
            self.refine_patience = QSpinBox(self)
            self.refine_patience.setRange(1, 1000)
            self.refine_patience.setValue(1)
            self.refine_eval_games = QSpinBox(self)
            self.refine_eval_games.setRange(1, 1_000_000)
            self.refine_eval_games.setValue(250)
            self.max_games = QSpinBox(self)
            self.max_games.setRange(0, 10_000_000)
            self.max_games.setSingleStep(1000)
            self.max_games.setSpecialValueText(tr("training.max_games_unlimited"))
            self.max_games.setValue(10_000)
            self.refresh_base_models()
            self.refresh_datasets()
            # Build the form unparented; ``outer`` below is the panel's only
            # layout and holds the form plus the hardware group.
            form = QFormLayout()
            self._form = form
            form.addRow(tr("training.mode"), self.mode)
            form.addRow(tr("training.collection_players"), self.players)
            form.addRow(tr("training.personality"), self.personality)
            form.addRow(tr("training.episodes"), self.episodes)
            form.addRow(tr("training.epochs"), self.epochs)
            form.addRow(tr("training.games"), self.games)
            form.addRow(tr("training.games_per_round"), self.refine_games)
            form.addRow(tr("training.refine_rounds"), self.refine_rounds)
            form.addRow(tr("training.refine_epochs"), self.refine_epochs)
            form.addRow(tr("training.refine_patience"), self.refine_patience)
            form.addRow(tr("training.refine_eval_games"), self.refine_eval_games)
            form.addRow(tr("training.base_model"), base_model_row)
            form.addRow(tr("training.main_action_mode"), self.action_mode)
            form.addRow(tr("training.datasets"), datasets_row)
            form.addRow(tr("training.max_games"), self.max_games)
            form.addRow(tr("training.max_minutes"), self.max_minutes)
            form.addRow(tr("training.graph_interval"), self.graph_interval)
            form.addRow(tr("training.target"), self.target)
            form.addRow(tr("training.seed"), self.seed)
            form.addRow(tr("training.resume_module"), resume_row)
            form.addRow(tr("training.resume_artifact_path"), self.resume_main_ai_row)

            # Save locations: collapsed to a one-line summary by default.
            self.storage_toggle = QPushButton(self)
            self.storage_toggle.setCheckable(True)
            self.storage_toggle.setFlat(True)
            self.storage_toggle.setStyleSheet("text-align: left; padding: 2px 0px;")
            self.storage_toggle.setToolTip(tr("tooltip.training_storage"))
            self.storage_toggle.toggled.connect(self._sync_storage)
            self.storage_body = QWidget(self)
            storage_form = self._storage_form = QFormLayout(self.storage_body)
            storage_form.setContentsMargins(0, 0, 0, 0)
            storage_form.addRow(tr("training.report_path"), self.report)
            storage_form.addRow(tr("training.output_path"), self.output)
            storage_form.addRow(tr("training.artifact_path"), self.artifact)
            naming = QLabel(tr("training.storage_naming"), self.storage_body)
            naming.setWordWrap(True)
            naming.setStyleSheet("color: gray;")
            storage_form.addRow(naming)
            for path_field in (self.report, self.output, self.artifact):
                path_field.edit.editingFinished.connect(self._sync_storage)

            # Hardware / AI-training controls, moved here from the Settings
            # screen so they sit next to the job they configure. Their values
            # are injected into the SystemConfig in setup().
            hw = self._config_hardware()
            self.clip = QDoubleSpinBox(self)
            self.clip.setRange(0.000001, 0.999999)
            self.clip.setDecimals(6)
            self.clip.setValue(self._config_clip())
            # Main AI training: one policy/value head per own role (PPO role_heads).
            self.role_heads = QCheckBox(tr("settings.role_heads_check"), self)
            self.role_heads.setChecked(self._config_role_heads())
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
            hardware_form.addRow(tr("settings.role_heads"), self.role_heads)
            hardware_form.addRow(tr("settings.cpu"), self.cpu)
            hardware_form.addRow(tr("settings.ram"), self.ram)
            hardware_form.addRow(tr("settings.workers"), self.workers)
            hardware_form.addRow(tr("settings.batch"), self.batch)
            hardware_form.addRow(tr("settings.buffer"), self.buffer)
            hardware_form.addRow(tr("settings.gpu"), self.gpu)
            self.hardware_form = hardware_form
            # Save the hardware group shortly after the last change (one write
            # per burst of spin-box clicks, not one per click).
            self._hardware_save_timer = QTimer(self)
            self._hardware_save_timer.setSingleShot(True)
            self._hardware_save_timer.setInterval(500)
            self._hardware_save_timer.timeout.connect(self.save_hardware_settings)
            for spin in (self.clip, self.cpu, self.ram, self.workers, self.batch, self.buffer):
                spin.valueChanged.connect(lambda _value: self._hardware_save_timer.start())
            self.gpu.toggled.connect(lambda _checked: self._hardware_save_timer.start())
            self.hardware_status = QLabel("", self.hardware_group)
            self.hardware_status.setWordWrap(True)
            self.hardware_status.setStyleSheet("color: gray;")
            hardware_form.addRow(self.hardware_status)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.addLayout(form)
            outer.addWidget(self.storage_toggle)
            outer.addWidget(self.storage_body)
            outer.addWidget(self.hardware_group)
            outer.addStretch(1)
            self._fix_column_widths()

            # Item 3: hover help on every field AND its QFormLayout label so the
            # AI-training settings menu gets mouse-over tooltips too.
            field_tooltips = (
                (self.mode, "tooltip.training_mode"),
                (self.episodes, "tooltip.training_episodes"),
                (self.epochs, "tooltip.training_epochs"),
                (self.games, "tooltip.training_games"),
                (self.max_minutes, "tooltip.training_max_minutes"),
                (self.graph_interval, "tooltip.training_graph_interval"),
                (self.target, "tooltip.training_target"),
                (self.seed, "tooltip.training_seed"),
                (self.report, "tooltip.training_report_path"),
                (self.output, "tooltip.training_output_path"),
                (self.artifact, "tooltip.training_artifact_path"),
                (self.resume_main_ai, "tooltip.training_resume_artifact_path"),
                (self.resume_main_ai_row, "tooltip.training_resume_artifact_path"),
                (self.resume_row, "tooltip.training_resume_module"),
                (self.resume_module, "tooltip.training_resume_module"),
                (self.players, "tooltip.training_collection_players"),
                (self.personality, "tooltip.training_personality"),
                (self.base_model, "tooltip.training_base_model"),
                (self.base_model_row, "tooltip.training_base_model"),
                (self.action_mode, "tooltip.training_main_action_mode"),
                (self.datasets, "tooltip.training_datasets"),
                (self.datasets_row, "tooltip.training_datasets"),
                (self.max_games, "tooltip.training_max_games"),
                (self.refine_games, "tooltip.training_games_per_round"),
                (self.refine_rounds, "tooltip.training_refine_rounds"),
                (self.refine_epochs, "tooltip.training_refine_epochs"),
                (self.refine_patience, "tooltip.training_refine_patience"),
                (self.refine_eval_games, "tooltip.training_refine_eval_games"),
            )
            label = form.labelForField(resume_row)
            if label is not None:
                label.setToolTip(tr("tooltip.training_resume_module"))
            for field, key in field_tooltips:
                text = tr(key)
                if field is self.artifact:
                    text += "\n\n" + tr("training.storage_naming")
                field.setToolTip(text)
                label = form.labelForField(field) or storage_form.labelForField(field)
                if label is not None:
                    label.setToolTip(text)
            hardware_tooltips = (
                (self.clip, "tooltip.ppo_clip"),
                (self.role_heads, "tooltip.role_heads"),
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

            self.mode.currentIndexChanged.connect(self._on_mode_changed)
            self.players.currentIndexChanged.connect(lambda _index: self._apply_mode())
            self.resume_main_ai.currentIndexChanged.connect(lambda _index: self._sync_main_ai_sources())
            self.resume_main_ai.currentIndexChanged.connect(
                lambda _index: self.role_heads.setEnabled(self._main_ai_resume_file() is None)
            )
            self._apply_mode()

        # --- mode-dependent form -------------------------------------------

        def _is_main_ai(self) -> bool:
            return self.mode.currentData() == TRAINING_MODE_MAIN_AI

        def _resume_kind(self) -> str:
            return "main" if self._is_main_ai() else "module"

        def _label_for(self, field: QWidget) -> Any:
            return self._form.labelForField(field) or self._storage_form.labelForField(field)

        def _set_row_visible(self, field: QWidget, visible: bool) -> None:
            field.setVisible(visible)
            label = self._label_for(field)
            if label is not None:
                label.setVisible(visible)

        # Every label text a row can show in any mode; the label column is as
        # wide as the longest of them, so switching modes never moves a field.
        _ALTERNATE_LABEL_KEYS = (
            "training.graph_interval_main",
            "training.resume_module_main",
        )
        _LABEL_COLUMN_MAX = 170

        def _fix_column_widths(self) -> None:
            """Give the label column of every form one fixed width.

            QFormLayout sizes its label column to the longest *visible* label,
            so hiding rows per mode used to move the fields sideways. Labels
            longer than the cap wrap onto a second line.
            """

            forms = (self._form, self._storage_form, self.hardware_form)
            labels: list[QLabel] = []
            for layout in forms:
                layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
                layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                for row in range(layout.rowCount()):
                    item = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
                    widget = item.widget() if item is not None else None
                    if isinstance(widget, QLabel):
                        labels.append(widget)
            metrics = self.fontMetrics()
            texts = [label.text() for label in labels] + [tr(key) for key in self._ALTERNATE_LABEL_KEYS]
            widest = max((metrics.horizontalAdvance(text) for text in texts), default=0) + 8
            width = min(widest, self._LABEL_COLUMN_MAX)
            for label in labels:
                label.setWordWrap(True)
                label.setFixedWidth(width)
            # Hardware group labels sit inside a frame; keep the same column.
            self._label_column_width = width
            for combo in (self.mode, self.players, self.action_mode):
                combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(12)
            for spin in self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox):
                policy = spin.sizePolicy()
                policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
                spin.setSizePolicy(policy)

        def _sync_storage(self, _value: Any = None) -> None:
            """Show or hide the save-location fields and refresh the summary line."""

            expanded = self.storage_toggle.isChecked()
            self.storage_body.setVisible(expanded)
            visible = [field for field in (self.report, self.artifact) if not field.isHidden()]
            changed = any(not field.is_default() for field in visible)
            names = " · ".join(tr(key) for key, field in (
                ("training.storage_report", self.report), ("training.storage_artifact", self.artifact),
            ) if field in visible)
            state = tr("training.storage_changed" if changed else "training.storage_default")
            # U+25BC / U+25B6 exist in Korean UI fonts (Malgun Gothic); ▾/▸ do not.
            arrow = "\u25bc" if expanded else "\u25b6"
            self.storage_toggle.setText(f"{arrow} {tr('training.storage_group')}: {names} ({state})")

        def _module_folder(self) -> Path:
            from bang_rl.training.module_artifacts import MODULE_MODEL_ROOT

            run = self._resume_runs.get(self.resume_module.currentData() or "") if hasattr(self, "_resume_runs") else None
            path = getattr(run, "intent_path", None) or getattr(run, "emotion_path", None)
            if path:
                return Path(path).parent
            return Path(self.model_root) if self.model_root is not None else MODULE_MODEL_ROOT

        def _set_row_text(self, field: QWidget, label_key: str | None, tooltip_key: str) -> None:
            """Set a row's label text (when given) and the tooltip on both widgets."""

            label = self._label_for(field)
            if label is not None:
                if label_key is not None:
                    label.setText(tr(label_key))
                label.setToolTip(tr(tooltip_key))
            field.setToolTip(tr(tooltip_key))

        def _mode_value(self) -> str:
            return str(self.mode.currentData() or TRAINING_MODE_DATA_COLLECTION)

        def _is_refinement(self) -> bool:
            return self._mode_value() == TRAINING_MODE_INFERENCE_REFINEMENT

        def _collects_main_ai(self) -> bool:
            """Data collection of main AI + 4 dummy games (needs a model)."""

            return (
                self._mode_value() == TRAINING_MODE_DATA_COLLECTION
                and self.players.currentData() == "main1-dummy4"
            )

        def _apply_mode(self) -> None:
            """Show only the inputs that apply to the selected mode."""

            main = self._is_main_ai()
            refine = self._is_refinement()
            mode = self._mode_value()
            collect = mode == TRAINING_MODE_DATA_COLLECTION
            module = mode == TRAINING_MODE_MODULE_TRAINING
            # Each row is shown only for the workflows that read it
            # (pipeline_runner / main_ai_training / inference_refinement).
            self._set_row_visible(self.players, collect)
            self._set_row_visible(self.personality, collect)
            self._set_row_visible(self.episodes, collect)  # games to collect
            self._set_row_visible(self.epochs, module)
            self._set_row_visible(self.max_games, module)
            self._set_row_visible(self.games, main)
            # Legacy RL-runner field; none of the four GUI workflows reads it.
            self._set_row_visible(self.target, False)
            for field in (
                self.refine_games, self.refine_rounds, self.refine_epochs,
                self.refine_patience, self.refine_eval_games,
            ):
                self._set_row_visible(field, refine)
            self._set_row_visible(self.base_model_row, refine or self._collects_main_ai())
            self._set_row_visible(self.action_mode, refine or self._collects_main_ai())
            self._set_row_visible(self.datasets_row, refine or module)
            # Collection has no time limit and no batch graphs.
            self._set_row_visible(self.max_minutes, not collect)
            self._set_row_visible(self.graph_interval, module or main)
            self._set_row_visible(self.resume_row, module or main)
            self._set_row_visible(self.resume_main_ai_row, main)
            # Every workflow writes its data under folders derived from the
            # config, so the generic output path is never read by them.
            self._set_row_visible(self.output, False)
            # Only main-AI training and refinement write a .bangmodel.
            self._set_row_visible(self.artifact, main or refine)
            # The PPO clip only matters where the PPO trains.
            self.clip.setVisible(main)
            clip_label = self.hardware_form.labelForField(self.clip)
            if clip_label is not None:
                clip_label.setVisible(main)
            self.role_heads.setVisible(main)
            role_heads_label = self.hardware_form.labelForField(self.role_heads)
            if role_heads_label is not None:
                role_heads_label.setVisible(main)
            # A saved .bangmodel fixes the PPO layout, so the choice only applies to a fresh PPO.
            self.role_heads.setEnabled(self._main_ai_resume_file() is None)
            self._set_row_text(
                self.graph_interval,
                "training.graph_interval_main" if main else "training.graph_interval",
                "tooltip.training_graph_interval_main" if main else "tooltip.training_graph_interval",
            )
            self._set_row_text(
                self.max_minutes,
                None,
                "tooltip.training_max_minutes_refine" if refine
                else "tooltip.training_max_minutes_main" if main
                else "tooltip.training_max_minutes",
            )
            self._set_row_text(
                self.resume_main_ai_row,
                None,
                "tooltip.training_resume_artifact_path_main" if main else "tooltip.training_resume_artifact_path",
            )
            self.resume_main_ai.setToolTip(self.resume_main_ai_row.toolTip())
            resume_label = self._form.labelForField(self.resume_row)
            if resume_label is not None:
                resume_label.setText(tr("training.resume_module_main" if main else "training.resume_module"))
            self._sync_limit_mode()
            self._sync_main_ai_sources()
            self._sync_storage()

        def _on_mode_changed(self, _index: int | None = None) -> None:
            self._apply_mode()
            self.refresh_resume_runs()

        def _main_ai_resume_file(self) -> str | None:
            """The selected .bangmodel file to continue from, or None.

            Only an existing file counts: a folder would silently pick its
            newest model, so "none" never sends a folder to the service.
            """

            value = self.resume_main_ai.currentData()
            if not isinstance(value, str) or not value:
                return None
            index = self.resume_main_ai.currentIndex()
            item = self.resume_main_ai.model().item(index)
            if item is not None and not item.isEnabled():
                return None
            try:
                path = Path(value).expanduser()
                return str(path) if path.is_file() else None
            except OSError:
                return None

        def refresh_resume_main_ai_models(self) -> None:
            """Reload the main AI models that main-AI training can continue from."""

            self._fill_main_ai_combo(
                self.resume_main_ai_row, tr("training.resume_main_ai_none"), keep_first=True
            )

        def _fill_main_ai_combo(self, picker: Any, none_text: str, *, keep_first: bool) -> None:
            """List saved .bangmodel files in ``picker``; incompatible ones are disabled.

            ``keep_first`` keeps the "none" entry selected by default (continue
            from nothing); otherwise the newest usable model is preselected.
            """

            from bang_rl.training.main_ai_artifacts import (
                list_saved_main_ai_models,
                main_ai_model_compatibility,
            )

            combo = picker.combo
            previous = combo.currentData()
            # Files added with "Browse..." stay listed after a refresh.
            extra = [
                (combo.itemData(i), combo.itemText(i), combo.itemData(i, Qt.ItemDataRole.ToolTipRole))
                for i in range(combo.count())
                if isinstance(combo.itemData(i), str) and combo.itemData(i).endswith(".bangmodel")
            ]
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(none_text, None)
            try:
                models = list_saved_main_ai_models(self._main_ai_dirs)
            except OSError:
                models = ()
            first_usable = -1
            for model in models:
                ok, reason = main_ai_model_compatibility(model.path)
                label = model.display_name + ("" if ok else tr("training.resume_module_incompatible"))
                combo.addItem(label, str(model.path))
                index = combo.count() - 1
                tip = model.tooltip if ok else tr(f"training.base_model_reason.{reason}") + "\n\n" + model.tooltip
                combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
                if not ok:
                    item = combo.model().item(index)
                    if item is not None:
                        item.setEnabled(False)
                elif first_usable < 0:
                    first_usable = index
            for path, text, tip in extra:
                if combo.findData(path) < 0 and Path(path).is_file():
                    picker.add_path_item(path, text, str(tip or ""))
            picker.add_browse_item()
            index = combo.findData(previous) if previous else -1
            if index < 0:
                index = 0 if keep_first else max(first_usable, 0)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
            picker.update_summary()

        def set_resume_main_ai(self, path: str | Path | None) -> bool:
            """Choose the .bangmodel to continue from; anything but an existing file selects "none"."""

            if path and Path(str(path)).is_file() and self._select_main_ai_file(self.resume_main_ai_row, str(path)):
                return True
            self.resume_main_ai.setCurrentIndex(0)
            return False

        def _select_main_ai_file(self, picker: Any, path: str) -> bool:
            """Select ``path`` in a main-AI picker, adding it when it is outside the listed folders."""

            combo = picker.combo
            index = combo.findData(str(path))
            if index < 0:
                if not Path(path).is_file():
                    return False
                from bang_rl.artifact_info import display_name

                index = picker.add_path_item(path, display_name(path))
            item = combo.model().item(index)
            if item is not None and not item.isEnabled():
                return False
            combo.setCurrentIndex(index)
            picker.update_summary()
            return True

        def _sync_main_ai_sources(self, _text: str | None = None) -> None:
            """A .bangmodel to continue from overrides the module choice."""

            overridden = self._is_main_ai() and self._main_ai_resume_file() is not None
            self.resume_module.setEnabled(not overridden)
            key = "tooltip.training_resume_module_main" if self._is_main_ai() else "tooltip.training_resume_module"
            tip = tr(key)
            if overridden:
                tip += "\n" + tr("training.main_ai_bangmodel_active")
            self.resume_module.setToolTip(tip)
            label = self._form.labelForField(self.resume_row)
            if label is not None:
                label.setToolTip(tip)

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

        def _config_personality(self) -> bool:
            preset = self._config_mapping().get("preset") or {}
            formula = (preset.get("emotion") or {}).get("formula") or {}
            return str(formula.get("personality_mode", "role")) == "random"

        def _config_role_heads(self) -> bool:
            rl = self._config_mapping().get("rl_training") or {}
            return bool((rl.get("ppo") or {}).get("role_heads", False))

        def _apply_personality_override(self, payload: dict[str, Any]) -> None:
            """Turn the dummies' random personality presets on or off for collection."""

            preset = dict(payload.get("preset") or {})
            emotion = dict(preset.get("emotion") or {})
            formula = dict(emotion.get("formula") or {})
            if self.personality.isChecked():
                if not formula.get("personality_presets"):
                    # The config has no presets of its own: use the shipped ones.
                    from bang_rl.config.schema import DEFAULT_EMOTION_FORMULA_FILE, default_emotion_formula

                    shipped = default_emotion_formula(
                        DEFAULT_EMOTION_FORMULA_FILE.with_name("emotion_formula.personality.json")
                    )
                    formula["personality_presets"] = dict(shipped.personality_presets)
                formula["personality_mode"] = "random"
            else:
                formula["personality_mode"] = "role"
            emotion["formula"] = formula
            preset["emotion"] = emotion
            payload["preset"] = preset

        def _hardware_values(self) -> dict[str, Any]:
            return {
                "cpu_cores": int(self.cpu.value()),
                "ram_gb": float(self.ram.value()),
                "use_gpu": bool(self.gpu.isChecked()),
                "workers": int(self.workers.value()),
                "batch_size": int(self.batch.value()),
                "buffer_size": int(self.buffer.value()),
            }

        def save_hardware_settings(self) -> bool:
            """Write the hardware group to the local settings file; False when there is no editor or it fails."""

            self._hardware_save_timer.stop()
            if self._settings_editor is None:
                return False
            try:
                snapshot = self._settings_editor.save_training_resources(
                    self._hardware_values(), clip_epsilon=float(self.clip.value())
                )
            except Exception as exc:  # noqa: BLE001 - a failed save must not break the form
                self.hardware_status.setText(tr("training.hardware_save_failed", error=str(exc)))
                return False
            # Later setup() calls start from the saved values too.
            self.config = snapshot.config
            self.hardware_status.setText(tr("training.hardware_saved"))
            return True

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
            if self._is_main_ai() and self.role_heads.isEnabled():
                ppo["role_heads"] = bool(self.role_heads.isChecked())
            rl["ppo"] = ppo
            payload["rl_training"] = rl

            if self._mode_value() == TRAINING_MODE_DATA_COLLECTION:
                self._apply_personality_override(payload)

            if not payload:
                return self.config
            from bang_rl.config import SystemConfig

            return SystemConfig.model_validate(payload)

        def refresh_resume_runs(self) -> None:
            """Reload saved module runs into the 'continue from' list."""

            from bang_rl.training.module_artifacts import list_saved_module_runs, main_ai_compatibility
            from ..dialogs.module_performance import module_run_label

            # Remember what was chosen for the kind the combo currently shows.
            if self.resume_module.count():
                self._resume_selection[self._resume_kind_shown] = self.resume_module.currentData()
            kind = self._resume_kind()
            main = kind == "main"
            self.resume_module.blockSignals(True)
            self.resume_module.clear()
            self.resume_module.addItem(
                tr("training.resume_module_auto" if main else "training.resume_module_none"), None
            )
            try:
                runs = list_saved_module_runs(self.model_root)
            except OSError:
                runs = ()
            self._resume_runs = {run.job_id: run for run in runs}
            self._resume_compat = {}
            newest_compatible: str | None = None
            for run in runs:
                label = module_run_label(run)
                ok, reason = (True, None)
                if main:
                    ok, reason = main_ai_compatibility(run)
                    self._resume_compat[run.job_id] = (ok, reason)
                    if not ok:
                        label += tr("training.resume_module_incompatible")
                self.resume_module.addItem(label, run.job_id)
                index = self.resume_module.count() - 1
                # Readable name, creation time and the user's note on hover.
                self.resume_module.setItemData(
                    index, getattr(run, "tooltip", "") or label, Qt.ItemDataRole.ToolTipRole
                )
                if main and not ok:
                    item = self.resume_module.model().item(index)
                    if item is not None:
                        item.setEnabled(False)
                    self.resume_module.setItemData(
                        index,
                        tr(f"training.resume_module_reason.{reason}") + "\n\n" + (getattr(run, "tooltip", "") or ""),
                        Qt.ItemDataRole.ToolTipRole,
                    )
                elif main and newest_compatible is None:
                    newest_compatible = run.job_id
            # Main AI defaults to the newest compatible run; an explicit earlier
            # choice (including "auto") is kept while it is still usable.
            if kind in self._resume_selection:
                wanted = self._resume_selection[kind]
            else:
                wanted = newest_compatible if main else None
            index = self.resume_module.findData(wanted) if wanted else 0
            if index < 0 or (main and wanted and not self._resume_compat.get(wanted, (False, None))[0]):
                index = self.resume_module.findData(newest_compatible) if newest_compatible else 0
            self.resume_module.setCurrentIndex(max(0, index))
            self._resume_kind_shown = kind
            self.resume_module.blockSignals(False)
            self.resume_row.update_summary()

        def select_resume_job(self, job_id: str | None) -> bool:
            """Preselect a saved run in the module list; False if it is unknown or unusable."""

            if job_id is not None and job_id not in self._resume_runs:
                self.refresh_resume_runs()
            if job_id is not None and self._is_main_ai() and not self._resume_compat.get(job_id, (False, None))[0]:
                return False
            index = self.resume_module.findData(job_id) if job_id else 0
            if index < 0:
                return False
            self.resume_module.setCurrentIndex(index)
            return True

        # --- presets (data library "apply to training") ---------------------------

        def set_checked_datasets(self, paths: Any) -> list[str]:
            """Check exactly ``paths`` in the dataset list; returns the paths that were not listed.

            A dataset outside the listed folders (for example one picked from the
            library) is added to the list so it can still be used.
            """

            paths = list(paths)
            wanted = [_norm_path(path) for path in paths]
            missing: list[str] = []
            listed = {
                _norm_path(self.datasets.item(index).data(Qt.ItemDataRole.UserRole)): index
                for index in range(self.datasets.count())
                if self.datasets.item(index).data(Qt.ItemDataRole.UserRole)
            }
            for key in wanted:
                if key not in listed:
                    missing.append(key)
            for original, key in zip(paths, wanted):
                if key in listed:
                    continue
                from bang_rl.training.dataset_artifacts import describe_dataset

                if not Path(str(original)).is_dir():
                    continue
                dataset = describe_dataset(original)
                item = QListWidgetItem(dataset.label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setData(Qt.ItemDataRole.UserRole, str(dataset.path))
                item.setToolTip(dataset.tooltip)
                # Drop the "no datasets" placeholder before the first real item.
                if self.datasets.count() == 1 and not self.datasets.item(0).data(Qt.ItemDataRole.UserRole):
                    self.datasets.takeItem(0)
                self.datasets.addItem(item)
                listed[key] = self.datasets.count() - 1
            for index in range(self.datasets.count()):
                item = self.datasets.item(index)
                value = item.data(Qt.ItemDataRole.UserRole)
                if not value:
                    continue
                checked = _norm_path(value) in wanted
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            return [key for key in missing if key not in listed]

        def _select_base_model(self, path: str) -> bool:
            """Select ``path`` in the base-model combo (adding it when it is outside the listed folders)."""

            if self.base_model.findData(str(path)) < 0:
                self.refresh_base_models()
            return self._select_main_ai_file(self.base_model_row, str(path))

        def apply_preset(self, preset: TrainingPreset) -> PresetResult:
            """Switch the form to ``preset`` (mode, datasets, module pair, model, players).

            The lists are reloaded first so items created since the screen opened
            are found. Nothing starts: the resulting :meth:`setup` simply reflects
            the preset. Problems (an unknown or unusable item) are reported in
            the result and the rest of the preset is still applied.
            """

            index = self.mode.findData(preset.mode)
            if index < 0:
                return PresetResult(False, tr("library.apply_unknown_mode", mode=preset.mode))
            self.mode.setCurrentIndex(index)
            self._apply_mode()
            self.refresh_datasets()
            self.refresh_base_models()
            self.refresh_resume_main_ai_models()
            self.refresh_resume_runs()
            problems: list[str] = []
            parts: list[str] = []
            mode = preset.mode

            if preset.players is not None and mode == TRAINING_MODE_DATA_COLLECTION:
                player_index = self.players.findData(preset.players)
                if player_index >= 0:
                    self.players.setCurrentIndex(player_index)
                    self._apply_mode()
            if preset.dataset_paths is not None and mode in (
                TRAINING_MODE_MODULE_TRAINING,
                TRAINING_MODE_INFERENCE_REFINEMENT,
            ):
                missing = self.set_checked_datasets(preset.dataset_paths)
                problems += [tr("library.apply_dataset_missing", path=path) for path in missing]
                parts.append(tr("library.applied_datasets", count=len(self.checked_dataset_paths())))
            if preset.module_run_id is not None and mode in (TRAINING_MODE_MODULE_TRAINING, TRAINING_MODE_MAIN_AI):
                if mode == TRAINING_MODE_MAIN_AI and preset.bangmodel_path is None:
                    # A .bangmodel chosen earlier would override the chosen module pair.
                    self.resume_main_ai.setCurrentIndex(0)
                if self.select_resume_job(preset.module_run_id):
                    parts.append(tr("library.applied_module", job=str(preset.module_run_id)[:8]))
                else:
                    problems.append(tr("library.apply_module_unusable", job=preset.module_run_id))
            if preset.bangmodel_path is not None:
                if mode == TRAINING_MODE_MAIN_AI:
                    if not Path(preset.bangmodel_path).is_file():
                        problems.append(tr("library.apply_model_missing", path=preset.bangmodel_path))
                    elif self._select_main_ai_file(self.resume_main_ai_row, str(preset.bangmodel_path)):
                        parts.append(tr("library.applied_continue", name=Path(preset.bangmodel_path).name))
                    else:
                        problems.append(tr("library.apply_model_unusable", path=preset.bangmodel_path))
                elif mode in (TRAINING_MODE_INFERENCE_REFINEMENT, TRAINING_MODE_DATA_COLLECTION):
                    if self._select_base_model(preset.bangmodel_path):
                        parts.append(tr("library.applied_base", name=Path(preset.bangmodel_path).name))
                    else:
                        problems.append(tr("library.apply_model_unusable", path=preset.bangmodel_path))
            mode_label = self.mode.itemText(self.mode.currentIndex())
            text = tr("library.applied", mode=mode_label, details=" / ".join(parts) if parts else "-")
            if problems:
                text += "\n" + "\n".join(problems)
            return PresetResult(not problems, text, tuple(problems))

        def _resume_paths(self) -> tuple[str | None, str | None]:
            if self._is_main_ai() and self._main_ai_resume_file() is not None:
                # The .bangmodel carries its own frozen modules.
                return None, None
            run = self._resume_runs.get(self.resume_module.currentData() or "")
            if run is None:
                return None, None
            if self._is_main_ai() and not self._resume_compat.get(run.job_id, (False, None))[0]:
                return None, None
            return (
                None if run.intent_path is None else str(run.intent_path),
                None if run.emotion_path is None else str(run.emotion_path),
            )

        def _train_by_time(self) -> bool:
            """True when the epochs (main AI: games) field is blank (0): bound training by time."""

            if self._is_refinement():
                return False  # rounds bound refinement; the time field is an optional extra limit
            field = self.games if self._is_main_ai() else self.epochs
            return int(field.value()) == 0

        def _sync_limit_mode(self, _value: int | None = None) -> None:
            """Enable the time field only while epochs is blank (always for refinement)."""

            self.max_minutes.setEnabled(self._train_by_time() or self._is_refinement())

        def _max_training_seconds(self) -> float | None:
            """Parse the 'max minutes' field: blank/0/invalid => unlimited."""

            if not self._train_by_time() and not self._is_refinement():
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

        # --- main AI model and dataset lists ---------------------------------

        def refresh_base_models(self) -> None:
            """Reload the main AI models (compatible ones are selectable)."""

            self._fill_main_ai_combo(self.base_model_row, tr("training.base_model_none"), keep_first=False)

        def _dataset_root(self) -> tuple[Path, bool]:
            """Canonical dataset folder of the config, and whether it is the default one."""

            from bang_rl.training.dataset_artifacts import canonical_root_for_config

            if self._dataset_root_override is not None:
                return Path(self._dataset_root_override), False
            return canonical_root_for_config(self._config_mapping())

        def refresh_datasets(self) -> None:
            """Reload the dataset list; checked datasets stay checked."""

            from bang_rl.paths import SAVED_DATASETS_DIR
            from bang_rl.training.dataset_artifacts import list_datasets

            checked = set(self.checked_dataset_paths())
            self.datasets.clear()
            root, is_default = self._dataset_root()
            try:
                found = list_datasets(root, saved_root=SAVED_DATASETS_DIR if is_default else None)
            except OSError:
                found = ()
            for dataset in found:
                item = QListWidgetItem(dataset.label + ("  [saved]" if dataset.saved else ""))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if str(dataset.path) in checked else Qt.CheckState.Unchecked
                )
                item.setData(Qt.ItemDataRole.UserRole, str(dataset.path))
                item.setToolTip(dataset.tooltip)
                self.datasets.addItem(item)
            if not found:
                placeholder = QListWidgetItem(tr("training.datasets_empty"))
                placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
                self.datasets.addItem(placeholder)
            self.datasets_row.update_summary()

        def checked_dataset_paths(self) -> tuple[str, ...]:
            """Folders of the checked datasets (empty = use every dataset found)."""

            paths: list[str] = []
            for index in range(self.datasets.count()):
                item = self.datasets.item(index)
                value = item.data(Qt.ItemDataRole.UserRole)
                if value and item.checkState() == Qt.CheckState.Checked:
                    paths.append(str(value))
            return tuple(paths)

        def _selected_base_model(self) -> str | None:
            value = self.base_model.currentData()
            return str(value) if value else None

        def flush_hardware_settings(self) -> None:
            """Save a pending hardware change now (before a run starts or the screen closes)."""

            if self._hardware_save_timer.isActive():
                self.save_hardware_settings()

        def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt signature
            self.flush_hardware_settings()
            super().hideEvent(event)

        def setup(self) -> TrainingSetup:
            self.flush_hardware_settings()
            main = self._is_main_ai()
            refine = self._is_refinement()
            mode = self._mode_value()
            target = None if main or refine else (self.target.value() or None)
            resume_intent, resume_emotion = (None, None) if refine else self._resume_paths()
            if main:
                # Games (not episodes/epochs) bound main-AI training; only an
                # existing .bangmodel file counts as "continue from".
                episodes = self.games.value()
                resume_artifact_path = self._main_ai_resume_file()
            elif refine:
                # Games per collection round; the base model is chosen explicitly.
                episodes = self.refine_games.value()
                resume_artifact_path = None
            else:
                episodes = self.episodes.value()
                # Data collection and module training never continue from a
                # .bangmodel; sending one would only make the service load it.
                resume_artifact_path = None
            uses_model = refine or self._collects_main_ai()
            max_seconds = self._max_training_seconds()
            return TrainingSetup(
                resume_intent_artifact_path=resume_intent,
                resume_emotion_artifact_path=resume_emotion,
                config=self._config_with_overrides(),
                mode=mode,
                episodes=episodes,
                target_transitions=target,
                seed=self.seed.value(),
                report_path=self.report.text().strip() or None,
                output_path=self.output.text().strip() or None,
                artifact_path=self.artifact.text().strip() or None,
                resume_artifact_path=resume_artifact_path,
                train_by_time=self._train_by_time() or (refine and max_seconds is not None),
                max_training_seconds=max_seconds,
                graph_point_interval=int(self.graph_interval.value()),
                collection_players=str(self.players.currentData() or "dummy5")
                if mode == TRAINING_MODE_DATA_COLLECTION
                else "dummy5",
                main_action_mode=str(self.action_mode.currentData() or "deterministic"),
                base_model_path=self._selected_base_model() if uses_model else None,
                dataset_paths=(self.checked_dataset_paths() or None)
                if refine or mode == TRAINING_MODE_MODULE_TRAINING
                else None,
                refine_rounds=int(self.refine_rounds.value()),
                refine_epochs=int(self.refine_epochs.value()),
                refine_patience=int(self.refine_patience.value()),
                refine_eval_games=int(self.refine_eval_games.value()),
                max_games=(int(self.max_games.value()) or None)
                if mode == TRAINING_MODE_MODULE_TRAINING
                else None,
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
            on_library_choice: Callable[[Any], None] | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.controller = controller or TrainingMonitorController()
            self.setup_panel = setup_panel or TrainingSetupPanel(parent=self)
            self.on_back = on_back
            # Called with a library choice this screen cannot carry out itself
            # (start a game, open a replay); applying to the setup form is done here.
            self.on_library_choice = on_library_choice
            # A confirmation (for example "library item applied") that stays
            # visible until a job reports its own message or is started.
            self._notice = ""
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
            # Big coloured banner: job state + mode on top, current stage and
            # progress below. Separate from the log so it is seen at a glance.
            self.status_banner = QWidget(self)
            self.status_banner.setObjectName("statusBanner")
            self.status_banner.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            banner_layout = QVBoxLayout(self.status_banner)
            banner_layout.setContentsMargins(12, 8, 12, 8)
            banner_layout.setSpacing(2)
            self.status_title = QLabel("", self.status_banner)
            title_font = self.status_title.font()
            title_font.setPointSize(max(title_font.pointSize() + 5, 15))
            title_font.setBold(True)
            self.status_title.setFont(title_font)
            self.status_detail = QLabel("", self.status_banner)
            detail_font = self.status_detail.font()
            detail_font.setPointSize(max(detail_font.pointSize() + 1, 11))
            self.status_detail.setFont(detail_font)
            self.status_detail.setWordWrap(True)
            banner_layout.addWidget(self.status_title)
            banner_layout.addWidget(self.status_detail)
            self._banner_state: str | None = None
            self.status = QLabel(tr("training.no_job"), self)
            self.snapshot = QLabel(tr("training.snapshot_not_started"), self)
            self.paths = QLabel("Report: —\nOutput: —\nArtifact: —", self)
            self.paths.setWordWrap(True)
            # Where this run's result is (or will be) saved, with Open/Copy.
            self.result_bar = QWidget(self)
            result_layout = QHBoxLayout(self.result_bar)
            result_layout.setContentsMargins(0, 0, 0, 0)
            self.result_label = QLabel("", self.result_bar)
            self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            result_policy = self.result_label.sizePolicy()
            result_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            self.result_label.setSizePolicy(result_policy)
            self.result_open_button = QPushButton(tr("picker.open_folder"), self.result_bar)
            self.result_open_button.setToolTip(tr("tooltip.picker_open_folder"))
            self.result_open_button.clicked.connect(lambda: open_folder(self._result_path))
            self.result_copy_button = QPushButton(tr("training.result_copy"), self.result_bar)
            self.result_copy_button.setToolTip(tr("tooltip.training_result_copy"))
            self.result_copy_button.clicked.connect(self._copy_result_path)
            result_layout.addWidget(self.result_label, 1)
            result_layout.addWidget(self.result_open_button)
            result_layout.addWidget(self.result_copy_button)
            self._result_path = ""
            self.result_bar.setVisible(False)
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
            # The data library: every stored dataset, module pair, model, report and
            # replay in one place, with buttons that apply the selection to this form.
            self.library_button = QPushButton(tr("library.open"), self)
            self.library_button.setToolTip(tr("tooltip.library_open"))
            self.library_button.clicked.connect(lambda: self._open_library())
            self.inference_button = QPushButton(tr("infer.open"), self)
            self.inference_button.setToolTip(tr("tooltip.infer_open"))
            self.inference_button.clicked.connect(self._open_inference_evaluation)
            # Copies this run's outputs into the shared (not git-ignored) saved/
            # folder under a name the user picks. Enabled once a run has outputs.
            self.save_button = QPushButton(tr("save.button"), self)
            self.save_button.setToolTip(tr("tooltip.save_button"))
            self.save_button.setEnabled(False)
            self.save_button.clicked.connect(self._open_save_dialog)
            self._save_plans: list[Any] = []
            self._save_status: str | None = None
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
                self.plot = _lock_plot(PlotWidget(self))
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
            # Main AI training: win rates on top, PPO losses below, one legend.
            self.main_ai_win_plot = None
            self.main_ai_loss_plot = None
            self._main_ai_signature: tuple[Any, ...] | None = None
            self._dashed_curves: set[str] = set()
            if PYQTGRAPH_AVAILABLE:
                self.main_ai_win_plot = _lock_plot(PlotWidget(self))
                self.main_ai_win_plot.setLabel("left", tr("mainai.axis.win_rate"))
                self.main_ai_win_plot.setLabel("bottom", tr("mainai.axis.games"))
                self.main_ai_win_plot.setYRange(0.0, 1.0)
                self.main_ai_loss_plot = _lock_plot(PlotWidget(self))
                self.main_ai_loss_plot.setLabel("left", tr("mainai.axis.loss"))
                self.main_ai_loss_plot.setLabel("bottom", tr("mainai.axis.updates"))
                main_ai_graphs = QWidget(self)
                main_ai_graphs_layout = QVBoxLayout(main_ai_graphs)
                main_ai_graphs_layout.setContentsMargins(0, 0, 0, 0)
                main_ai_graphs_layout.addWidget(self.main_ai_win_plot, 1)
                main_ai_graphs_layout.addWidget(self.main_ai_loss_plot, 1)
                self.graph_stack.addWidget(main_ai_graphs)
                self._main_ai_graph_index = self.graph_stack.count() - 1
            else:
                self._main_ai_graph_index = self._module_graph_index
            # Inference refinement: accuracies per round on top, emotion MSE below.
            self.refine_acc_plot = None
            self.refine_mse_plot = None
            self._refine_signature: tuple[Any, ...] | None = None
            if PYQTGRAPH_AVAILABLE:
                self.refine_acc_plot = _lock_plot(PlotWidget(self))
                self.refine_acc_plot.setLabel("left", tr("refine.axis.accuracy"))
                self.refine_acc_plot.setLabel("bottom", tr("refine.axis.round"))
                self.refine_acc_plot.setYRange(0.0, 1.0)
                self.refine_mse_plot = _lock_plot(PlotWidget(self))
                self.refine_mse_plot.setLabel("left", tr("refine.axis.mse"))
                self.refine_mse_plot.setLabel("bottom", tr("refine.axis.round"))
                refine_graphs = QWidget(self)
                refine_graphs_layout = QVBoxLayout(refine_graphs)
                refine_graphs_layout.setContentsMargins(0, 0, 0, 0)
                refine_graphs_layout.addWidget(self.refine_acc_plot, 1)
                refine_graphs_layout.addWidget(self.refine_mse_plot, 1)
                self.graph_stack.addWidget(refine_graphs)
                self._refine_graph_index = self.graph_stack.count() - 1
            else:
                self._refine_graph_index = self._module_graph_index
            # Text summary shown only for inference refinement.
            self.refine_panel = QWidget(self)
            refine_layout = QVBoxLayout(self.refine_panel)
            refine_layout.setContentsMargins(0, 0, 0, 0)
            self.refine_labels: dict[str, QLabel] = {}
            for key in ("progress", "base", "best", "rounds", "rule", "saved"):
                label = QLabel("", self.refine_panel)
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                refine_layout.addWidget(label)
                self.refine_labels[key] = label
            self.refine_panel.setVisible(False)
            # Text summary shown only for main AI training.
            self.main_ai_panel = QWidget(self)
            main_ai_layout = QVBoxLayout(self.main_ai_panel)
            main_ai_layout.setContentsMargins(0, 0, 0, 0)
            self.main_ai_labels: dict[str, QLabel] = {}
            for key in ("progress", "baselines", "latest", "best", "saved"):
                label = QLabel("", self.main_ai_panel)
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                main_ai_layout.addWidget(label)
                self.main_ai_labels[key] = label
            self.main_ai_warning = QLabel("", self.main_ai_panel)
            self.main_ai_warning.setWordWrap(True)
            self.main_ai_warning.setStyleSheet("color: #e08a00; font-weight: bold;")
            main_ai_layout.addWidget(self.main_ai_warning)
            self.main_ai_panel.setVisible(False)

            layout = QVBoxLayout(self)
            layout.addWidget(self.back_button)
            columns = QHBoxLayout()
            left_panel = QVBoxLayout()
            # The form scrolls vertically when it is taller than the window
            # (for example with the save locations expanded); never sideways.
            self.setup_scroll = QScrollArea(self)
            self.setup_scroll.setWidgetResizable(True)
            self.setup_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            self.setup_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setup_scroll.setWidget(self.setup_panel)
            left_panel.addWidget(self.setup_scroll, 1)
            left_panel.addLayout(buttons)
            left_panel.addWidget(self.library_button)
            left_panel.addWidget(self.performance_button)
            left_panel.addWidget(self.inference_button)
            left_panel.addWidget(self.save_button)
            right_panel = QVBoxLayout()
            right_panel.addWidget(self.status_banner)
            right_panel.addWidget(self.environment_steps)
            right_panel.addWidget(self.truncated)
            metrics_row = QHBoxLayout()
            metrics_row.addWidget(self.metrics, 1)
            metrics_row.addWidget(self.metrics_copy_button)
            right_panel.addLayout(metrics_row)
            right_panel.addWidget(self.validation_summary)
            right_panel.addWidget(self.main_ai_panel)
            right_panel.addWidget(self.refine_panel)
            right_panel.addWidget(self.status)
            right_panel.addWidget(self.snapshot)
            right_panel.addWidget(self.paths)
            right_panel.addWidget(self.result_bar)
            right_panel.addWidget(self.graph_stack, 6)
            right_panel.addLayout(graph_controls)
            right_panel.addWidget(self.legend_bar)
            right_panel.addWidget(self.logs, 1)
            right_panel.addWidget(self.message)
            # A fixed-width setup column: switching modes or long item names
            # never resizes it (or the graphs next to it).
            left_container = self.setup_column = QWidget(self)
            left_container.setLayout(left_panel)
            left_panel.setContentsMargins(0, 0, 0, 0)
            left_container.setFixedWidth(SETUP_PANEL_WIDTH)
            columns.addWidget(left_container)
            columns.addLayout(right_panel, 1)
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
            self.set_notice("")
            self.controller.start(self.setup_panel.setup())

        def set_notice(self, text: str) -> None:
            """Show ``text`` in the message line until the next job message or start."""

            self._notice = str(text or "")
            self.message.setText(self._notice)

        def _pause(self) -> None:
            self.controller.pause(timeout=30.0)

        def _resume(self) -> None:
            self.controller.resume()

        def _stop(self) -> None:
            self.controller.cancel()

        def _open_performance(self) -> None:
            """"Saved model performance" is the data library filtered to modules and main-AI models."""

            from ..library_view import KIND_FILTER_MODELS

            self._open_library(KIND_FILTER_MODELS)

        def _library_locations(self) -> Any:
            """Folders the library reads: the ones this setup form lists from."""

            from bang_rl.data_library import LibraryLocations
            from bang_rl.paths import SAVED_DATASETS_DIR

            panel = self.setup_panel
            canonical, is_default = panel._dataset_root()
            main_ai_dirs = getattr(panel, "_main_ai_dirs", None)
            model_root = getattr(panel, "model_root", None)
            return LibraryLocations(
                model_root=None if model_root is None else Path(model_root),
                main_ai_dirs=None if main_ai_dirs is None else tuple(Path(item) for item in main_ai_dirs),
                canonical_root=canonical,
                saved_datasets=SAVED_DATASETS_DIR if is_default else None,
            )

        def _open_library(self, kind_filter: str | None = None) -> None:
            """Open the data library; applying an item fills this form (only while no job is active)."""

            from ..dialogs.data_library import DataLibraryDialog
            from ..training import training_is_active

            dialog = DataLibraryDialog(
                locations=self._library_locations(),
                config=self.setup_panel.config,
                kind_filter=kind_filter if isinstance(kind_filter, str) else None,
                training_busy=lambda: training_is_active(self.controller),
                status_provider=lambda: controller_status(self.controller),
                parent=self,
            )
            dialog.exec()
            if dialog.chosen is not None:
                self.handle_library_choice(dialog.chosen)

        def handle_library_choice(self, choice: Any) -> None:
            """Apply a library choice to this screen's setup form, or pass it on."""

            from ..library_view import build_preset
            from ..training import training_is_active

            preset = build_preset(choice.action, choice.entries)
            if preset is None:
                if self.on_library_choice is not None:
                    self.on_library_choice(choice)
                return
            if training_is_active(self.controller):
                self.set_notice(tr("library.busy"))
                return
            self.set_notice(self.setup_panel.apply_preset(preset).message)

        def _open_inference_evaluation(self) -> None:
            from ..dialogs.inference_evaluation import InferenceEvaluationDialog

            dialog = InferenceEvaluationDialog(
                config=self.setup_panel.config,
                model_root=getattr(self.setup_panel, "model_root", None),
                main_ai_dirs=getattr(self.setup_panel, "_main_ai_dirs", None),
                parent=self,
            )
            dialog.exec()

        def _update_save_button(self, progress: Any, workflow: str | None) -> None:
            """Enable 'Save outputs' when a finished or stopped run left outputs."""

            status = str(getattr(progress, "status", ""))
            if status == self._save_status:
                return
            self._save_status = status
            plans: list[Any] = []
            if status in {"finished", "cancelled"} and workflow:
                from bang_rl.saved_artifacts import plans_for_training_result

                try:
                    plans = plans_for_training_result(
                        str(workflow),
                        output_path=getattr(progress, "output_path", None),
                        artifact_path=getattr(progress, "artifact_path", None),
                        report_path=getattr(progress, "report_path", None),
                        metrics=getattr(progress, "metrics", None) or {},
                    )
                except OSError:
                    plans = []
            self._save_plans = plans
            self.save_button.setEnabled(bool(plans))

        def _open_save_dialog(self) -> None:
            from ..dialogs.save_outputs import SaveOutputsDialog

            if not self._save_plans:
                self.message.setText(tr("save.nothing"))
                return
            # Refresh sizes/notes from disk (the user may have edited notes since).
            for plan in self._save_plans:
                dialog = SaveOutputsDialog(plan, self)
                if dialog.exec():
                    self.message.setText(dialog.controller.done_text())

        def _refresh_resume_on_finish(self, status: str) -> None:
            """Newly saved modules appear in the 'continue from' list."""

            if status != self._last_status and status in {"finished", "cancelled"}:
                self.setup_panel.refresh_resume_runs()
                self.setup_panel.refresh_base_models()
                if hasattr(self.setup_panel, "refresh_resume_main_ai_models"):
                    self.setup_panel.refresh_resume_main_ai_models()
                self.setup_panel.refresh_datasets()
            self._last_status = status

        def _update_status_banner(self, progress: Any, workflow: Any, metrics: Mapping[str, Any] | None) -> None:
            state, title, detail = main_ai_view.status_banner(
                None if progress is None else str(getattr(progress, "status", "")),
                None if workflow is None else str(workflow),
                metrics,
                ppo_updates=int(getattr(progress, "ppo_updates", 0) or 0) if progress is not None else 0,
                error=getattr(progress, "error", None) if progress is not None else None,
            )
            if state != self._banner_state:
                self._banner_state = state
                color = main_ai_view.STATUS_COLORS.get(state, main_ai_view.STATUS_COLORS["running"])
                self.status_banner.setStyleSheet(
                    f"#statusBanner {{ background-color: {color}; border-radius: 6px; }}"
                    " #statusBanner QLabel { color: white; background: transparent; }"
                )
            self.status_title.setText(("\u25cf " if state == "running" else "") + title)
            self.status_detail.setText(detail)

        def _update_result_bar(self, progress: Any) -> None:
            """Show the main result path: "Saved: ..." once the run ended, "Saving to: ..." before."""

            path = ""
            if progress is not None:
                for key in ("artifact_path", "output_path", "report_path"):
                    value = getattr(progress, key, None)
                    if value:
                        path = str(value)
                        break
            self._result_path = path
            self.result_bar.setVisible(bool(path))
            if not path:
                return
            done = str(getattr(progress, "status", "")) in {"finished", "cancelled"}
            prefix = tr("training.result_saved" if done else "training.result_saving")
            text = f"{prefix} {path}"
            width = max(60, self.result_label.width())
            self.result_label.setText(self.result_label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideMiddle, width))
            self.result_label.setToolTip(text)

        def _copy_result_path(self) -> None:
            if QApplication is not None and self._result_path:
                QApplication.clipboard().setText(self._result_path)

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

        def _apply_curve_pen(
            self, curve: Any, color_hex: str, *, hovered: bool, width: int, dashed: bool = False
        ) -> None:
            import pyqtgraph as pg

            style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
            curve.setPen(pg.mkPen(self._dim_color(color_hex, dim=not hovered), width=width, style=style))

        def _highlight(self, name: str | None) -> None:
            """Brighten the hovered curve and dim the rest.

            ``name is None`` (no hover) restores every curve to full colour.
            """

            for entry_name, curve, color_hex in self._active_curves:
                hovered = name is None or entry_name == name
                width = 3 if (name is not None and entry_name == name) else 2
                self._apply_curve_pen(
                    curve, color_hex, hovered=hovered, width=width, dashed=entry_name in self._dashed_curves
                )

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
                len(getattr(state, "personality_accuracy_points", ())),
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
            # Shown only when the emotion module also learns the personality.
            personality_values = getattr(state, "personality_accuracy_points", ())
            if personality_values:
                personality_curve = self.plot.plot(
                    list(range(1, len(personality_values) + 1)),
                    list(personality_values),
                    pen="#b084cc",
                    name="Personality",
                )
                self._active_curves.append(("Personality", personality_curve, "#b084cc"))
            self._highlight(None)
            self._rebuild_legend()

        def _render_main_ai_graph(self, metrics: Mapping[str, Any], job_id: Any = None) -> None:
            """Win rates (training, evaluation, dummy/random baselines) and PPO losses."""

            win_plot, loss_plot = self.main_ai_win_plot, self.main_ai_loss_plot
            if win_plot is None or loss_plot is None:
                return
            interval = main_ai_view.number(metrics.get("main_graph_point_interval")) or 1
            train = main_ai_view.series(metrics, "main_win_rate_points")
            evaluation = main_ai_view.series(metrics, "main_eval_win_rate_points")
            eval_games = main_ai_view.series(metrics, "main_eval_games")
            policy = main_ai_view.series(metrics, "main_policy_loss_points")
            value = main_ai_view.series(metrics, "main_value_loss_points")
            entropy = main_ai_view.series(metrics, "main_entropy_points")
            baselines = main_ai_view.baseline_win_rates(metrics)
            prior = main_ai_view.prior_win_rate(metrics)
            # The monitor re-renders every poll; redraw only when something new arrived.
            signature = (
                job_id, interval, len(train), len(evaluation), len(eval_games),
                len(policy), len(value), len(entropy), tuple(baselines.values()), prior,
            )
            if signature == self._main_ai_signature:
                return
            self._main_ai_signature = signature
            win_plot.clear()
            loss_plot.clear()
            train_x = main_ai_view.sampled_axis(len(train), interval)
            eval_x = [int(item) for item in eval_games] if len(eval_games) == len(evaluation) else (
                list(range(1, len(evaluation) + 1))
            )
            update_x = main_ai_view.sampled_axis(len(policy), interval)
            # Continuing a saved model: both win-rate curves start at game 0
            # from its saved evaluation instead of from no data.
            train_x, train = main_ai_view.with_prior_point(train_x, train, prior)
            eval_x, evaluation = main_ai_view.with_prior_point(eval_x, evaluation, prior)
            x_end = max([2, *train_x[-1:], *eval_x[-1:]])
            specs: list[tuple[Any, str, str, list[Any], list[float], bool]] = [
                (win_plot, tr("mainai.curve.train_win"), "#5bc0eb", train_x, train, False),
                (win_plot, tr("mainai.curve.eval_win"), "#8bd450", eval_x, evaluation, False),
            ]
            for name, color, key in (
                ("dummy", "#f6c344", "mainai.curve.dummy"),
                ("random", "#e85d75", "mainai.curve.random"),
            ):
                rate = baselines.get(name)
                if rate is not None:
                    specs.append((win_plot, tr(key), color, [0 if prior is not None else 1, x_end], [rate, rate], True))
            specs += [
                (loss_plot, tr("mainai.curve.policy_loss"), "#b084cc", update_x, policy, False),
                (loss_plot, tr("mainai.curve.value_loss"), "#f28f3b", update_x, value, False),
                (loss_plot, tr("mainai.curve.entropy"), "#4d9de0", update_x, entropy, False),
            ]
            self._active_curves = []
            self._dashed_curves = set()
            for plot, name, color, xs, ys, dashed in specs:
                count = min(len(xs), len(ys))
                curve = plot.plot(list(xs[:count]), list(ys[:count]), pen=color, name=name)
                self._active_curves.append((name, curve, color))
                if dashed:
                    self._dashed_curves.add(name)
            if prior is not None:
                # A line through one point draws nothing: mark the saved
                # evaluation at game 0 so it is visible before new points arrive.
                win_plot.plot([0], [prior], pen=None, symbol="o", symbolSize=10, symbolBrush="#8bd450", symbolPen="w")
            self._highlight(None)
            self._rebuild_legend()

        def _render_refinement_graph(self, metrics: Mapping[str, Any], job_id: Any = None) -> None:
            """Per-round hidden-role accuracy, role accuracy, win rate (dashed) and emotion MSE."""

            acc_plot, mse_plot = self.refine_acc_plot, self.refine_mse_plot
            if acc_plot is None or mse_plot is None:
                return
            series = main_ai_view.refine_series(metrics)
            signature = (job_id, *(tuple(series[key]) for key in main_ai_view.REFINE_SERIES))
            if signature == self._refine_signature:
                return
            self._refine_signature = signature
            acc_plot.clear()
            mse_plot.clear()
            specs = (
                (acc_plot, tr("refine.curve.hidden"), "#f6c344", "hidden_role_accuracy", False),
                (acc_plot, tr("refine.curve.role"), "#5bc0eb", "role_accuracy", False),
                (acc_plot, tr("refine.curve.win"), "#8bd450", "win_rate", True),
                (mse_plot, tr("refine.curve.mse"), "#e85d75", "emotion_mse", False),
            )
            self._active_curves = []
            self._dashed_curves = set()
            for plot, name, color, key, dashed in specs:
                points = [(index, value) for index, value in enumerate(series[key]) if value is not None]
                curve = plot.plot([x for x, _ in points], [y for _, y in points], pen=color, name=name, symbol="o")
                self._active_curves.append((name, curve, color))
                if dashed:
                    self._dashed_curves.add(name)
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
                self.main_ai_panel.setVisible(False)
                self.refine_panel.setVisible(False)
                self.validation_summary.setVisible(True)
                self._save_status = None
                self._save_plans = []
                self.save_button.setEnabled(False)
                self._update_result_bar(None)
                self._update_status_banner(None, None, None)
                return
            self._refresh_resume_on_finish(str(progress.status))
            target_config = getattr(state.snapshot, "config_snapshot", {}) if state.snapshot is not None else {}
            workflow = getattr(state.snapshot, "mode", None) if state.snapshot is not None else None
            self._update_save_button(progress, workflow)
            if workflow == "data_collection":
                collection_metrics = getattr(progress, "metrics", {}) or {}
                maximum = int(collection_metrics.get("collection_total", 0) or 1)
                current = int(collection_metrics.get("collection_completed", progress.episodes_completed) or 0)
                self.progress_bar.setFormat("Collected games: %v / %m")
                self.progress_bar.setRange(0, max(maximum, current, 1))
                self.progress_bar.setValue(current)
            elif workflow == TRAINING_MODE_MAIN_AI:
                main_metrics = getattr(progress, "metrics", {}) or {}
                games_done = int(main_ai_view.number(main_metrics.get("main_games_completed")) or 0)
                games_total = int(main_ai_view.number(main_metrics.get("main_games_total")) or 0)
                if games_total > 0:
                    self.progress_bar.setFormat(tr("mainai.progress_bar"))
                    self.progress_bar.setRange(0, max(games_total, games_done, 1))
                else:
                    self.progress_bar.setFormat(tr("mainai.progress_bar_time"))
                    self.progress_bar.setRange(0, max(games_done, 1))
                self.progress_bar.setValue(games_done)
            elif workflow == TRAINING_MODE_INFERENCE_REFINEMENT:
                refine_progress = getattr(progress, "metrics", {}) or {}
                rounds_total = int(main_ai_view.number(refine_progress.get("refine_total_rounds")) or 0)
                finished_rounds = max(0, len(refine_progress.get("refine_history") or ()) - 1)
                self.progress_bar.setFormat(tr("refine.progress_bar"))
                self.progress_bar.setRange(0, max(rounds_total, finished_rounds, 1))
                self.progress_bar.setValue(finished_rounds)
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
            is_main_ai = workflow == TRAINING_MODE_MAIN_AI
            truncation_high = is_main_ai and main_ai_view.truncation_is_high(rate)
            self.truncated.setStyleSheet("color: #e08a00; font-weight: bold;" if truncation_high else "")
            metrics_value = public_training_metadata(getattr(progress, "metrics", {}) or {})
            self._metrics_full_text = metrics_json(metrics_value)
            self._update_status_banner(progress, workflow, metrics_value)
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
            is_refine = workflow == TRAINING_MODE_INFERENCE_REFINEMENT
            self.validation_summary.setVisible(not is_main_ai and not is_refine)
            self.main_ai_panel.setVisible(is_main_ai)
            self.refine_panel.setVisible(is_refine)
            if is_refine:
                texts = main_ai_view.format_refinement_summary(
                    metrics_value, artifact_path=getattr(progress, "artifact_path", None)
                )
                for key, label in self.refine_labels.items():
                    label.setText(texts[key])
            if is_main_ai:
                texts = main_ai_view.format_main_ai_summary(
                    metrics_value,
                    ppo_updates=int(getattr(progress, "ppo_updates", 0) or 0),
                    artifact_path=getattr(progress, "artifact_path", None),
                )
                for key, label in self.main_ai_labels.items():
                    label.setText(texts[key])
                self.main_ai_warning.setText(tr("mainai.truncated_warning") if truncation_high else "")
                self.main_ai_warning.setVisible(bool(truncation_high))
            self.status.setText(f"Status: {progress.status} · PPO updates: {progress.ppo_updates}")
            if state.snapshot is not None:
                self.snapshot.setText(
                    f"Snapshot: {state.snapshot.job_id} · config {state.snapshot.config_version} "
                    f"· seed {state.snapshot.seed}"
                )
            # One line per path so each kind starts at the left edge.
            self.paths.setText(
                f"Report: {progress.report_path or '—'}\n"
                f"Output: {getattr(progress, 'output_path', None) or '—'}\n"
                f"Artifact: {progress.artifact_path or '—'}"
            )
            self.paths.setToolTip(self.paths.text())
            self._update_result_bar(progress)
            _set_log_text_preserving_scroll(self.logs, "\n".join(state.logs))
            self.message.setText(state.message or self._notice or "")
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
                elif is_refine:
                    self.role_graph_button.setVisible(False)
                    self.emotion_graph_button.setVisible(False)
                    self.accuracy_graph_button.setVisible(False)
                    self.graph_stack.setCurrentIndex(self._refine_graph_index)
                    self._render_refinement_graph(metrics_value, getattr(state.snapshot, "job_id", None))
                else:
                    self.role_graph_button.setVisible(False)
                    self.emotion_graph_button.setVisible(False)
                    self.accuracy_graph_button.setVisible(False)
                    self.graph_stack.setCurrentIndex(self._main_ai_graph_index)
                    if is_main_ai:
                        self._render_main_ai_graph(metrics_value, getattr(state.snapshot, "job_id", None))
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
