"""Optional PyQt6 editor for validated training and reward settings."""
from __future__ import annotations

import json
from typing import Any

from ..qt_compat import (
    QCheckBox,
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
from ..settings import RESOURCE_HELP, SettingsEditor, SettingsSnapshot, SettingsValidationError


if QT_AVAILABLE:  # pragma: no cover - display dependent; editor is headless tested

    class SettingsScreen(QWidget):
        """Settings form with validation before values are staged or saved."""

        def __init__(
            self,
            editor: SettingsEditor | None = None,
            *,
            parent: QWidget | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.editor = editor or SettingsEditor()
            hardware = self.editor.snapshot.config.hardware
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
            self.gpu = QCheckBox("Use GPU/CUDA", self)
            self.gpu.setChecked(bool(hardware["use_gpu"]))
            self.reward = QPlainTextEdit(self)
            self.reward.setPlainText(json.dumps(self.editor.snapshot.reward_config, indent=2))
            self.error = QLabel(self)
            self.help = QLabel(RESOURCE_HELP, self)
            self.help.setWordWrap(True)
            self.save_button = QPushButton("Validate and save", self)
            self.save_button.clicked.connect(self._save)
            form = QFormLayout()
            form.addRow("PPO clip epsilon", self.clip)
            form.addRow("Emotion initial baseline", self.baseline)
            form.addRow("CPU cores", self.cpu)
            form.addRow("RAM (GB)", self.ram)
            form.addRow("Workers", self.workers)
            form.addRow("Batch size", self.batch)
            form.addRow("Buffer size", self.buffer)
            form.addRow("GPU", self.gpu)
            layout = QVBoxLayout(self)
            layout.addLayout(form)
            layout.addWidget(QLabel("Reward Config (JSON)", self))
            layout.addWidget(self.reward)
            layout.addWidget(self.help)
            layout.addWidget(self.save_button)
            layout.addWidget(self.error)

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
                self.error.setText("Settings saved for the next training job.")
            except SettingsValidationError as exc:
                self.error.setText(str(exc))


else:

    class SettingsScreen:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["SettingsScreen"]
