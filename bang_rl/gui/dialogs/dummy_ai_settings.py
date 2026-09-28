"""Optional PyQt6 dialog to configure the dummy AI's emotions.

The dialog edits ``config.preset.emotion`` (an ``EmotionConfig``): the number
of emotions, their names, how they increase/decrease (``change_rate`` /
``decay_rate`` plus the ``event_updates`` / ``action_updates`` rule maps), and
their effects when high or low (the ``effects`` map).  Every edit is staged
through :class:`~bang_rl.gui.settings.SettingsEditor`, so Pydantic validation
(unique names, ``initial_value`` in ``[0, 1]``, finite rules, ``emotion_count``
``>= 1``) surfaces as a localized :class:`SettingsValidationError` message.

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
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from ..settings import SettingsEditor, SettingsValidationError


def _dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, indent=2)


def _loads_map(text: str, loc: str) -> dict[str, float]:
    """Parse a JSON object of ``str -> number`` from a text field."""

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

    class _EmotionRow(QGroupBox):
        """One editable emotion definition."""

        def __init__(self, index: int, definition: dict[str, Any], parent: QWidget) -> None:
            super().__init__(tr("settings.emotion_number", index=index + 1), parent)
            self.name = QLineEdit(str(definition.get("name", f"emotion_{index}")), self)
            self.initial = QDoubleSpinBox(self)
            self.initial.setRange(0.0, 1.0)
            self.initial.setDecimals(4)
            initial_value = definition.get("initial_value")
            self.initial.setValue(float(initial_value) if initial_value is not None else 0.0)
            self.change_rate = QDoubleSpinBox(self)
            self.change_rate.setRange(-1_000_000.0, 1_000_000.0)
            self.change_rate.setDecimals(4)
            self.change_rate.setValue(float(definition.get("change_rate", 0.0)))
            self.decay_rate = QDoubleSpinBox(self)
            self.decay_rate.setRange(-1_000_000.0, 1_000_000.0)
            self.decay_rate.setDecimals(4)
            self.decay_rate.setValue(float(definition.get("decay_rate", 0.0)))
            self.event_updates = QPlainTextEdit(_dumps(definition.get("event_updates")), self)
            self.action_updates = QPlainTextEdit(_dumps(definition.get("action_updates")), self)
            self.reward_updates = QPlainTextEdit(_dumps(definition.get("reward_updates")), self)
            self.terminal_updates = QPlainTextEdit(_dumps(definition.get("terminal_updates")), self)
            self.effects = QPlainTextEdit(_dumps(definition.get("effects")), self)

            form = QFormLayout(self)
            form.addRow(tr("settings.emotion_name"), self.name)
            form.addRow(tr("settings.emotion_initial"), self.initial)
            form.addRow(tr("settings.emotion_change_rate"), self.change_rate)
            form.addRow(tr("settings.emotion_decay_rate"), self.decay_rate)
            form.addRow(tr("settings.emotion_event_updates"), self.event_updates)
            form.addRow(tr("settings.emotion_action_updates"), self.action_updates)
            form.addRow(tr("settings.emotion_reward_updates"), self.reward_updates)
            form.addRow(tr("settings.emotion_terminal_updates"), self.terminal_updates)
            form.addRow(tr("settings.emotion_effects"), self.effects)

        def to_definition(self, index: int) -> dict[str, Any]:
            return {
                "name": self.name.text().strip(),
                "initial_value": self.initial.value(),
                "change_rate": self.change_rate.value(),
                "decay_rate": self.decay_rate.value(),
                "event_updates": _loads_map(
                    self.event_updates.toPlainText(), f"emotions.{index}.event_updates"
                ),
                "action_updates": _loads_map(
                    self.action_updates.toPlainText(), f"emotions.{index}.action_updates"
                ),
                "reward_updates": _loads_map(
                    self.reward_updates.toPlainText(), f"emotions.{index}.reward_updates"
                ),
                "terminal_updates": _loads_map(
                    self.terminal_updates.toPlainText(), f"emotions.{index}.terminal_updates"
                ),
                "effects": _loads_map(self.effects.toPlainText(), f"emotions.{index}.effects"),
            }

    class DummyAISettingsDialog(QDialog):
        """Edit the dummy AI's emotion configuration and stage it via the editor."""

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

            emotion = dict(self.editor.snapshot.dummy_emotion)
            definitions = list(emotion.get("emotions") or [])

            self.help = QLabel(tr("settings.dummy_ai_help"), self)
            self.help.setWordWrap(True)
            self.schema_version = QLineEdit(str(emotion.get("schema_version", "1.0")), self)
            self.count = QSpinBox(self)
            self.count.setRange(1, 64)
            self.count.setValue(max(1, int(emotion.get("emotion_count", len(definitions) or 1))))
            self.count.valueChanged.connect(self._rebuild_rows)
            self.initial_baseline = QDoubleSpinBox(self)
            self.initial_baseline.setRange(-1_000_000.0, 1_000_000.0)
            self.initial_baseline.setDecimals(4)
            self.initial_baseline.setValue(float(emotion.get("initial_baseline", 0.0) or 0.0))
            self.initial_baseline.setToolTip(tr("tooltip.emotion_initial_baseline_field"))

            header = QFormLayout()
            header.addRow(tr("settings.emotion_schema_version"), self.schema_version)
            header.addRow(tr("settings.emotion_count"), self.count)
            header.addRow(tr("settings.emotion_initial_baseline"), self.initial_baseline)

            self._rows: list[_EmotionRow] = []
            self._rows_host = QWidget(self)
            self._rows_layout = QVBoxLayout(self._rows_host)
            self._scroll = QScrollArea(self)
            self._scroll.setWidgetResizable(True)
            self._scroll.setWidget(self._rows_host)

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
            layout.addLayout(header)
            layout.addWidget(self._scroll)
            layout.addWidget(self.error)
            layout.addWidget(self.buttons)

            self._definitions_cache = definitions
            self._populate_rows(definitions)

        def _clear_rows(self) -> None:
            for row in self._rows:
                row.setParent(None)
                row.deleteLater()
            self._rows = []

        def _populate_rows(self, definitions: list[dict[str, Any]]) -> None:
            self._clear_rows()
            count = self.count.value()
            for index in range(count):
                definition = definitions[index] if index < len(definitions) else {
                    "name": f"emotion_{index}"
                }
                row = _EmotionRow(index, dict(definition), self._rows_host)
                self._rows.append(row)
                self._rows_layout.addWidget(row)

        def _rebuild_rows(self, _count: int) -> None:
            # Preserve any edits the user already made before changing the count.
            current = self._collect_definitions_safe()
            self._populate_rows(current)

        def _collect_definitions_safe(self) -> list[dict[str, Any]]:
            collected: list[dict[str, Any]] = []
            for index, row in enumerate(self._rows):
                try:
                    collected.append(row.to_definition(index))
                except SettingsValidationError:
                    collected.append({"name": row.name.text().strip() or f"emotion_{index}"})
            return collected

        def _collect_emotion(self) -> dict[str, Any]:
            """Build the emotion mapping from the current rows (may raise)."""

            emotions = [row.to_definition(index) for index, row in enumerate(self._rows)]
            return {
                "schema_version": self.schema_version.text().strip() or "1.0",
                "emotion_count": self.count.value(),
                # Carry the editable baseline so a non-zero initial_baseline is
                # never silently reset to EmotionConfig's 0.0 default on Apply
                # or Save. The dialog exposes it as its own spin field.
                "initial_baseline": self.initial_baseline.value(),
                "emotions": emotions,
            }

        def _reload_rows_from_editor(self) -> None:
            """Repopulate the header and rows from the staged emotion config."""

            emotion = dict(self.editor.snapshot.dummy_emotion)
            definitions = list(emotion.get("emotions") or [])
            self.schema_version.setText(str(emotion.get("schema_version", "1.0")))
            self.initial_baseline.setValue(float(emotion.get("initial_baseline", 0.0) or 0.0))
            # Setting the count fires valueChanged -> _rebuild_rows; block it so
            # the freshly loaded definitions are used instead of the old rows.
            self.count.blockSignals(True)
            self.count.setValue(
                max(1, int(emotion.get("emotion_count", len(definitions) or 1)))
            )
            self.count.blockSignals(False)
            self._populate_rows(definitions)

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
            self._reload_rows_from_editor()
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
