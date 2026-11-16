"""Optional PyQt6 dialog to configure the dummy AI's nine emotions.

The dummy AI always acts on nine emotions (calm, confidence, aggression,
anxiety and one relation value per seat) computed by the emotion formula.  The
dialog edits ``config.preset.emotion`` (an ``EmotionConfig``): the description
shown for each emotion and the ``formula`` block (``params`` and
``personality_by_role``).  Every edit is staged through
:class:`~bang_rl.gui.settings.SettingsEditor`, so Pydantic validation (unique
names, known formula parameters) surfaces as a localized
:class:`SettingsValidationError` message.

Importing :mod:`bang_rl.gui` stays safe without PyQt6 through the
``QT_AVAILABLE`` real-implementation / ``else`` ``require_qt()`` stub pattern.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..i18n import tr
from ..qt_compat import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from ..settings import SettingsEditor, SettingsValidationError


def _loads_object(text: str, loc: str) -> dict[str, Any]:
    """Parse a JSON object from a text field."""

    stripped = text.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SettingsValidationError(({"loc": (loc,), "msg": str(exc)},)) from exc
    if not isinstance(parsed, dict):
        raise SettingsValidationError(
            ({"loc": (loc,), "msg": "expected a JSON object"},)
        )
    return parsed


if QT_AVAILABLE:  # pragma: no cover - display dependent; editor is headless tested

    class DummyAISettingsDialog(QDialog):
        """Edit the dummy AI's emotion descriptions and formula; stage via the editor."""

        def __init__(
            self,
            editor: SettingsEditor,
            *,
            parent: QWidget | None = None,
            on_applied: Callable[[], None] | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.editor = editor
            self.on_applied = on_applied
            self.setWindowTitle(tr("settings.dummy_ai_title"))

            self.help = QLabel(tr("settings.dummy_ai_help"), self)
            self.help.setWordWrap(True)

            self.descriptions: list[QLineEdit] = []
            self._names: list[str] = []
            self._rows_host = QWidget(self)
            self._rows_form = QFormLayout(self._rows_host)
            self._scroll = QScrollArea(self)
            self._scroll.setWidgetResizable(True)
            self._scroll.setWidget(self._rows_host)

            self.formula = QPlainTextEdit(self)
            self.formula.setToolTip(tr("tooltip.emotion_formula"))
            self.formula_label = QLabel(tr("settings.emotion_formula_json"), self)
            self.formula_label.setToolTip(tr("tooltip.emotion_formula"))

            self.error = QLabel(self)
            self.error.setWordWrap(True)
            self.buttons = QDialogButtonBox(self)
            self.save_file_button = self.buttons.addButton(
                tr("settings.emotion_save_file"), QDialogButtonBox.ButtonRole.ActionRole
            )
            self.load_file_button = self.buttons.addButton(
                tr("settings.emotion_load_file"), QDialogButtonBox.ButtonRole.ActionRole
            )
            self.apply_button = self.buttons.addButton(
                tr("settings.dummy_ai_apply"), QDialogButtonBox.ButtonRole.AcceptRole
            )
            self.cancel_button = self.buttons.addButton(
                tr("settings.dummy_ai_cancel"), QDialogButtonBox.ButtonRole.RejectRole
            )
            self.save_file_button.setToolTip(tr("tooltip.emotion_save_file"))
            self.load_file_button.setToolTip(tr("tooltip.emotion_load_file"))
            self.save_file_button.clicked.connect(self._save_to_file)
            self.load_file_button.clicked.connect(self._load_from_file)
            self.apply_button.clicked.connect(self._apply)
            self.cancel_button.clicked.connect(self.reject)

            layout = QVBoxLayout(self)
            layout.addWidget(self.help)
            layout.addWidget(self._scroll)
            layout.addWidget(self.formula_label)
            layout.addWidget(self.formula)
            layout.addWidget(self.error)
            layout.addWidget(self.buttons)

            self._populate_from_editor()

        def _populate_from_editor(self) -> None:
            """(Re)build the description rows and formula text from the staged config."""

            emotion = dict(self.editor.snapshot.dummy_emotion)
            for field in self.descriptions:
                label = self._rows_form.labelForField(field)
                if label is not None:
                    label.setParent(None)
                    label.deleteLater()
                field.setParent(None)
                field.deleteLater()
            self.descriptions = []
            self._names = []
            for definition in emotion.get("emotions") or []:
                name = str(definition.get("name", ""))
                field = QLineEdit(str(definition.get("description", "")), self._rows_host)
                field.setToolTip(tr("tooltip.emotion_description"))
                self._rows_form.addRow(name, field)
                self.descriptions.append(field)
                self._names.append(name)
            self.formula.setPlainText(
                json.dumps(emotion.get("formula") or {}, ensure_ascii=False, indent=2)
            )

        def _collect_emotion(self) -> dict[str, Any]:
            """Build the emotion mapping from the current fields (may raise)."""

            emotion = dict(self.editor.snapshot.dummy_emotion)
            emotion["emotions"] = [
                {"name": name, "description": field.text()}
                for name, field in zip(self._names, self.descriptions)
            ]
            emotion["formula"] = _loads_object(self.formula.toPlainText(), "formula")
            return emotion

        def _save_to_file(self) -> None:
            path, _ = QFileDialog.getSaveFileName(
                self, tr("settings.emotion_save_file"), "", tr("settings.emotion_file_filter")
            )
            if not path:
                return
            try:
                self.editor.apply_dummy_emotion(self._collect_emotion())
                self.editor.export_dummy_emotion(path)
            except SettingsValidationError as exc:
                self.error.setText(str(exc))
                return
            self.error.setText(tr("settings.emotion_file_saved", path=path))

        def _load_from_file(self) -> None:
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
            self._populate_from_editor()
            self.error.setText(tr("settings.emotion_file_loaded", path=path))

        def _apply(self) -> None:
            try:
                self.editor.apply_dummy_emotion(self._collect_emotion())
            except SettingsValidationError as exc:
                self.error.setText(str(exc))
                return
            if self.on_applied is not None:
                self.on_applied()
            self.accept()


else:

    class DummyAISettingsDialog:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["DummyAISettingsDialog"]
