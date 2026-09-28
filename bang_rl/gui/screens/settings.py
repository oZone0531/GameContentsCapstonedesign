"""Optional PyQt6 editor for validated training and reward settings."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..i18n import LANGUAGE_LABELS, LANGUAGES, current_language, set_language, tr
from ..qt_compat import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from ..settings import SettingsEditor, SettingsSnapshot, SettingsValidationError


if QT_AVAILABLE:  # pragma: no cover - display dependent; editor is headless tested

    class SettingsScreen(QWidget):
        """Settings form with validation before values are staged or saved."""

        def __init__(
            self,
            editor: SettingsEditor | None = None,
            *,
            parent: QWidget | None = None,
            on_back: Callable[[], None] | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.editor = editor or SettingsEditor()
            self.on_back = on_back
            # ``snapshot.hardware`` is a plain, JSON-safe dict; the underlying
            # ``config.hardware`` is a Pydantic model and is not subscriptable.
            hardware = self.editor.snapshot.hardware

            self.language = QComboBox(self)
            for code in LANGUAGES:
                self.language.addItem(LANGUAGE_LABELS.get(code, code), code)
            current_index = self.language.findData(current_language())
            if current_index >= 0:
                self.language.setCurrentIndex(current_index)
            self.language.currentIndexChanged.connect(self._change_language)

            self.clip = QDoubleSpinBox(self)
            self.clip.setRange(0.000001, 0.999999)
            self.clip.setDecimals(6)
            self.clip.setValue(self.editor.snapshot.ppo_clip_epsilon)
            self.baseline = QDoubleSpinBox(self)
            self.baseline.setRange(0.0, 1.0)
            self.baseline.setDecimals(4)
            self.baseline.setValue(self.editor.snapshot.emotion_initial_baseline)
            self.cpu = QSpinBox(self)
            self.cpu.setRange(1, 4096)
            self.cpu.setValue(hardware["cpu_cores"])
            self.ram = QDoubleSpinBox(self)
            self.ram.setRange(0.001, 1_048_576.0)
            self.ram.setValue(hardware["ram_gb"])
            self.workers = QSpinBox(self)
            self.workers.setRange(1, 4096)
            self.workers.setValue(hardware["workers"])
            self.batch = QSpinBox(self)
            self.batch.setRange(1, 1_000_000)
            self.batch.setValue(hardware["batch_size"])
            self.buffer = QSpinBox(self)
            self.buffer.setRange(1, 1_000_000_000)
            self.buffer.setValue(hardware["buffer_size"])
            self.gpu = QCheckBox(tr("settings.gpu_check"), self)
            self.gpu.setChecked(bool(hardware["use_gpu"]))
            self.reward = QPlainTextEdit(self)
            self.reward.setPlainText(json.dumps(self.editor.snapshot.reward_config, indent=2))
            self.error = QLabel(self)
            self.help = QLabel(tr("settings.resource_help"), self)
            self.help.setWordWrap(True)
            self.save_button = QPushButton(tr("settings.save"), self)
            self.save_button.clicked.connect(self._save)
            self.back_button = QPushButton(tr("common.back"), self)
            self.back_button.clicked.connect(self._go_back)
            self.reward_label = QLabel(tr("settings.reward_json"), self)

            form = QFormLayout()
            form.addRow(tr("settings.language"), self.language)
            form.addRow(tr("settings.ppo_clip"), self.clip)
            form.addRow(tr("settings.emotion_baseline"), self.baseline)
            form.addRow(tr("settings.cpu"), self.cpu)
            form.addRow(tr("settings.ram"), self.ram)
            form.addRow(tr("settings.workers"), self.workers)
            form.addRow(tr("settings.batch"), self.batch)
            form.addRow(tr("settings.buffer"), self.buffer)
            form.addRow(tr("settings.gpu"), self.gpu)
            layout = QVBoxLayout(self)
            layout.addWidget(self.back_button)
            layout.addLayout(form)
            layout.addWidget(self.reward_label)
            layout.addWidget(self.reward)
            layout.addWidget(self.help)
            layout.addWidget(self.save_button)
            layout.addWidget(self.error)

        def _change_language(self, _index: int) -> None:
            code = self.language.currentData()
            if code:
                set_language(str(code))
                self._retranslate()
                self.error.setText(tr("settings.language_saved"))

        def _retranslate(self) -> None:
            """Update this screen's visible strings after a language change."""

            self.gpu.setText(tr("settings.gpu_check"))
            self.help.setText(tr("settings.resource_help"))
            self.save_button.setText(tr("settings.save"))
            self.back_button.setText(tr("common.back"))
            self.reward_label.setText(tr("settings.reward_json"))

        def _go_back(self) -> None:
            if self.on_back is not None:
                self.on_back()

        def _values(self) -> dict[str, Any]:
            try:
                reward = json.loads(self.reward.toPlainText())
            except json.JSONDecodeError as exc:
                raise SettingsValidationError(({"loc": ("reward_config",), "msg": str(exc)},)) from exc
            return {
                "ppo_clip_epsilon": self.clip.value(),
                "emotion_initial_baseline": self.baseline.value(),
                "hardware": {
                    "cpu_cores": self.cpu.value(),
                    "ram_gb": self.ram.value(),
                    "use_gpu": self.gpu.isChecked(),
                    "workers": self.workers.value(),
                    "batch_size": self.batch.value(),
                    "buffer_size": self.buffer.value(),
                },
                "reward_config": reward,
            }

        def _save(self) -> None:
            try:
                self.editor.apply(self._values())
                self.editor.save()
                self.error.setText(tr("settings.saved"))
            except SettingsValidationError as exc:
                self.error.setText(str(exc))


else:

    class SettingsScreen:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["SettingsScreen"]
