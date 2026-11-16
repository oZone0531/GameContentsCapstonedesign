"""Optional PyQt6 editor for validated training and reward settings."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..dialogs import DummyAISettingsDialog
from ..i18n import LANGUAGE_LABELS, LANGUAGES, current_language, set_language, tr
from ..qt_compat import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
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

            self.language = QComboBox(self)
            for code in LANGUAGES:
                self.language.addItem(LANGUAGE_LABELS.get(code, code), code)
            current_index = self.language.findData(current_language())
            if current_index >= 0:
                self.language.setCurrentIndex(current_index)
            self.language.currentIndexChanged.connect(self._change_language)

            self.dummy_ai_button = QPushButton(tr("settings.dummy_ai"), self)
            self.dummy_ai_button.clicked.connect(self._open_dummy_ai)
            self.emotion_save_button = QPushButton(tr("settings.emotion_save_file"), self)
            self.emotion_save_button.clicked.connect(self._save_emotion_file)
            self.emotion_load_button = QPushButton(tr("settings.emotion_load_file"), self)
            self.emotion_load_button.clicked.connect(self._load_emotion_file)
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

            self._apply_tooltips()

            form = QFormLayout()
            form.addRow(tr("settings.language"), self.language)
            # Keep the form so _apply_tooltips() (also called on language change)
            # can attach tooltips to the menu LABELS, not just the fields.
            self.form = form
            self._apply_label_tooltips()
            layout = QVBoxLayout(self)
            layout.addWidget(self.back_button)
            layout.addLayout(form)
            layout.addWidget(self.dummy_ai_button)
            emotion_file_row = QHBoxLayout()
            emotion_file_row.addWidget(self.emotion_save_button)
            emotion_file_row.addWidget(self.emotion_load_button)
            layout.addLayout(emotion_file_row)
            layout.addWidget(self.reward_label)
            layout.addWidget(self.reward)
            layout.addWidget(self.help)
            layout.addWidget(self.save_button)
            layout.addWidget(self.error)

        def _apply_tooltips(self) -> None:
            """Set localized mouse-over help on every settings field.

            Called once on build and again from _retranslate() so a language
            switch updates the help text.
            """

            self.language.setToolTip(tr("tooltip.language"))
            self.reward.setToolTip(tr("tooltip.reward_json"))
            self.reward_label.setToolTip(tr("tooltip.reward_json"))
            self.save_button.setToolTip(tr("tooltip.settings_save"))
            self.dummy_ai_button.setToolTip(tr("tooltip.dummy_ai"))
            self.emotion_save_button.setToolTip(tr("tooltip.emotion_save_file"))
            self.emotion_load_button.setToolTip(tr("tooltip.emotion_load_file"))

        def _apply_label_tooltips(self) -> None:
            """Attach mouse-over help to each QFormLayout menu LABEL.

            Item 3: hover help must cover the setting menu labels (Language,
            PPO clip, CPU cores, RAM, Workers, Batch, Buffer, GPU) themselves,
            not only the input fields. Called after the form is built and again
            from _retranslate() so a language switch updates the label tooltips.
            The form may not exist yet on the first _apply_tooltips() call, so
            this is a separate method guarded by a getattr check.
            """

            form = getattr(self, "form", None)
            if form is None:
                return
            label_tooltips = (
                (self.language, "tooltip.language"),
            )
            for field, key in label_tooltips:
                label = form.labelForField(field)
                if label is not None:
                    label.setToolTip(tr(key))

        def _open_dummy_ai(self) -> None:
            dialog = DummyAISettingsDialog(self.editor, parent=self, on_applied=self._on_dummy_applied)
            dialog.exec()

        def _on_dummy_applied(self) -> None:
            self.error.setText(tr("settings.dummy_ai_applied"))

        def _save_emotion_file(self) -> None:
            path, _ = QFileDialog.getSaveFileName(
                self, tr("settings.emotion_save_file"), "", tr("settings.emotion_file_filter")
            )
            if not path:
                return
            try:
                self.editor.export_dummy_emotion(path)
            except SettingsValidationError as exc:
                self.error.setText(str(exc))
                return
            self.error.setText(tr("settings.emotion_file_saved", path=path))

        def _load_emotion_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, tr("settings.emotion_load_file"), "", tr("settings.emotion_file_filter")
            )
            if not path:
                return
            try:
                self.editor.import_dummy_emotion(path)
            except SettingsValidationError as exc:
                self.error.setText(str(exc))
                return
            self.error.setText(tr("settings.emotion_file_loaded", path=path))

        def _change_language(self, _index: int) -> None:
            code = self.language.currentData()
            if code:
                set_language(str(code))
                self._retranslate()
                self.error.setText(tr("settings.language_saved"))

        def _retranslate(self) -> None:
            """Update this screen's visible strings after a language change."""

            self.help.setText(tr("settings.resource_help"))
            self.save_button.setText(tr("settings.save"))
            self.back_button.setText(tr("common.back"))
            self.reward_label.setText(tr("settings.reward_json"))
            self.dummy_ai_button.setText(tr("settings.dummy_ai"))
            self.emotion_save_button.setText(tr("settings.emotion_save_file"))
            self.emotion_load_button.setText(tr("settings.emotion_load_file"))
            self._apply_tooltips()
            self._apply_label_tooltips()

        def _go_back(self) -> None:
            if self.on_back is not None:
                self.on_back()

        def _values(self) -> dict[str, Any]:
            try:
                reward = json.loads(self.reward.toPlainText())
            except json.JSONDecodeError as exc:
                raise SettingsValidationError(({"loc": ("reward_config",), "msg": str(exc)},)) from exc
            return {
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
