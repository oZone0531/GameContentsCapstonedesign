"""Dialogs to save outputs to ``saved/`` and to edit an artifact's notes.

The decisions (what is listed, size warnings, name checks, the copy itself)
live in :class:`SaveOutputsController`, which does not need Qt, so they are
unit tested directly. The dialog classes only render that controller.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bang_rl.artifact_info import sanitize_user_name
from bang_rl.saved_artifacts import (
    LEVEL_OK,
    LEVEL_STRONG,
    LEVEL_WARN,
    SaveNameExistsError,
    SavePlan,
    SaveResult,
    execute_save,
    human_size,
)

from ..i18n import tr
from ..qt_compat import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)


@dataclass(frozen=True, slots=True)
class FileRow:
    name: str
    size_text: str
    size: int


class SaveOutputsController:
    """State and actions behind the save dialog (no Qt)."""

    def __init__(self, plan: SavePlan) -> None:
        self.plan = plan
        self.name = plan.default_name
        self.notes = plan.notes
        self.last_error: str | None = None
        self.result: SaveResult | None = None

    # -- display -----------------------------------------------------------
    @property
    def title(self) -> str:
        return tr(f"save.category.{self.plan.category}")

    def file_rows(self) -> list[FileRow]:
        rows = []
        for path, size in self.plan.files():
            rows.append(FileRow(path.name, human_size(size), size))
        return rows

    def total_text(self) -> str:
        rows = self.file_rows()
        return tr("save.total", size=human_size(sum(row.size for row in rows)), count=len(rows))

    def target_text(self) -> str:
        return tr("save.target", path=str(self.plan.target_root))

    def warning_text(self) -> str:
        """Empty when every file is within the limits, else the size warning."""

        level = self.plan.level
        path, size = self.plan.largest
        if level == LEVEL_OK or path is None:
            return ""
        key = "save.strong_100" if level == LEVEL_STRONG else "save.warn_50"
        return tr(key, file=path.name, size=human_size(size))

    @property
    def needs_confirmation(self) -> bool:
        """Files over the 100 MB GitHub limit need an explicit OK before saving."""

        return self.plan.level == LEVEL_STRONG

    def confirmation_text(self) -> str:
        path, size = self.plan.largest
        return tr("save.confirm_strong", file=path.name if path else "", size=human_size(size))

    # -- action -----------------------------------------------------------
    def save(self, name: str, notes: str, confirm: Callable[[str], bool] | None = None) -> SaveResult | None:
        """Copy the outputs. Returns the result, or None (see ``last_error``)."""

        self.last_error = None
        try:
            clean = sanitize_user_name(name)
        except ValueError:
            self.last_error = tr("save.name_invalid")
            return None
        if self.plan.conflicts(clean):
            self.last_error = tr("save.name_exists", name=clean)
            return None
        if self.needs_confirmation and (confirm is None or not confirm(self.confirmation_text())):
            self.last_error = ""  # declined: stay in the dialog, no message
            return None
        try:
            self.result = execute_save(self.plan, clean, notes)
        except SaveNameExistsError:
            self.last_error = tr("save.name_exists", name=clean)
            return None
        except OSError as exc:
            self.last_error = tr("save.failed", error=str(exc))
            return None
        return self.result

    def done_text(self) -> str:
        if self.result is None:
            return ""
        return tr("save.done", path=str(self.result.paths[0]) if self.result.paths else "")


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class SaveOutputsDialog(QDialog):
        def __init__(self, plan: SavePlan, parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(parent)
            self.controller = SaveOutputsController(plan)
            self.setWindowTitle(tr("save.title"))
            self.resize(640, 520)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(f"<b>{self.controller.title}</b>", self))
            layout.addWidget(QLabel(tr("save.files"), self))
            self.files = QPlainTextEdit(self)
            self.files.setReadOnly(True)
            self.files.setMaximumHeight(130)
            rows = self.controller.file_rows()
            shown = rows[:40]
            lines = [f"{row.name}   {row.size_text}" for row in shown]
            if len(rows) > len(shown):
                lines.append(f"... (+{len(rows) - len(shown)})")
            self.files.setPlainText("\n".join(lines))
            layout.addWidget(self.files)
            layout.addWidget(QLabel(self.controller.total_text(), self))
            self.warning = QLabel(self.controller.warning_text(), self)
            self.warning.setWordWrap(True)
            color = "#d32f2f" if plan.level == LEVEL_STRONG else "#e08a00"
            self.warning.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.warning.setVisible(bool(self.warning.text()))
            layout.addWidget(self.warning)
            layout.addWidget(QLabel(tr("save.name"), self))
            self.name_edit = QLineEdit(self.controller.name, self)
            self.name_edit.setToolTip(tr("save.name_hint"))
            layout.addWidget(self.name_edit)
            layout.addWidget(QLabel(tr("save.name_hint"), self))
            layout.addWidget(QLabel(tr("save.notes"), self))
            self.notes_edit = QPlainTextEdit(self)
            self.notes_edit.setPlainText(self.controller.notes)
            layout.addWidget(self.notes_edit, 1)
            layout.addWidget(QLabel(self.controller.target_text(), self))
            self.message = QLabel("", self)
            self.message.setWordWrap(True)
            self.message.setStyleSheet("color: #d32f2f;")
            layout.addWidget(self.message)
            buttons = QHBoxLayout()
            buttons.addStretch(1)
            self.save_button = QPushButton(tr("save.save"), self)
            self.cancel_button = QPushButton(tr("save.cancel"), self)
            buttons.addWidget(self.save_button)
            buttons.addWidget(self.cancel_button)
            layout.addLayout(buttons)
            self.save_button.clicked.connect(self._save)
            self.cancel_button.clicked.connect(self.reject)

        def _confirm(self, text: str) -> bool:
            answer = QMessageBox.question(
                self,
                tr("save.confirm_title"),
                text,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return answer == QMessageBox.StandardButton.Yes

        def _save(self) -> None:
            result = self.controller.save(
                self.name_edit.text(), self.notes_edit.toPlainText(), confirm=self._confirm
            )
            if result is None:
                self.message.setText(self.controller.last_error or "")
                return
            self.accept()

    class NotesDialog(QDialog):
        """Edit the free-text notes (비고) of one artifact."""

        def __init__(self, title: str, notes: str, parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(parent)
            self.setWindowTitle(tr("notes.title"))
            self.resize(520, 320)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(f"<b>{title}</b>", self))
            prompt = QLabel(tr("notes.prompt"), self)
            prompt.setWordWrap(True)
            layout.addWidget(prompt)
            self.edit = QPlainTextEdit(self)
            self.edit.setPlainText(notes)
            layout.addWidget(self.edit, 1)
            buttons = QHBoxLayout()
            buttons.addStretch(1)
            ok = QPushButton(tr("save.save"), self)
            cancel = QPushButton(tr("save.cancel"), self)
            buttons.addWidget(ok)
            buttons.addWidget(cancel)
            layout.addLayout(buttons)
            ok.clicked.connect(self.accept)
            cancel.clicked.connect(self.reject)

        @property
        def text(self) -> str:
            return self.edit.toPlainText()

else:

    class SaveOutputsDialog:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()

    class NotesDialog:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["FileRow", "NotesDialog", "SaveOutputsController", "SaveOutputsDialog"]
