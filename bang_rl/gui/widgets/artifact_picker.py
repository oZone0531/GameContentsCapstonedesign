"""Shared pickers for the AI-training form.

Every "choose one saved item" field (module pair, main AI model, the model to
continue from) and the "choose datasets" field use the same row layout:

    [selector                         ] [Refresh] [Open folder]
    one-line summary of the current selection

:class:`ArtifactPicker` wraps a ``QComboBox`` (``.combo``) and
:class:`MultiArtifactPicker` wraps a checkable ``QListWidget`` (``.list``)
shown in a drop-down popup. The inner widgets are exposed unchanged so callers
fill them with the usual Qt item APIs. :class:`PathField` is a folder path
input with Browse/Open buttons whose text is shortened in the middle while it
is not being edited.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from collections.abc import Callable

from ..i18n import tr
from ..qt_compat import (
    QComboBox,
    QDesktopServices,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QT_AVAILABLE,
    Qt,
    QUrl,
    QVBoxLayout,
    QWidget,
)

# Characters a closed selector reserves, so its width never follows its items.
SELECTOR_MIN_CHARS = 10
# Lines of an item tooltip copied into the one-line summary.
SUMMARY_LINES = 3


def summary_from_tooltip(text: str, max_lines: int = SUMMARY_LINES) -> str:
    """First non-empty tooltip lines joined with " · " (pure, for tests)."""

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return " · ".join(lines[:max_lines])


def existing_folder(path: str | Path | None) -> Path | None:
    """The folder to open for ``path``: itself, its parent for a file, or the nearest existing parent."""

    if not path:
        return None
    target = Path(str(path)).expanduser()
    if target.is_file():
        target = target.parent
    while not target.is_dir():
        if target.parent == target:
            return None
        target = target.parent
    return target


def open_folder(path: str | Path | None) -> bool:
    """Open the folder of ``path`` in the file explorer; False when there is none."""

    folder = existing_folder(path)
    if folder is None or QDesktopServices is None:
        return False
    return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))))


if QT_AVAILABLE:  # pragma: no cover - display dependent

    def _compact(button: Any) -> None:
        """Size a button to its text (plus padding) instead of the default minimum width."""

        button.setFixedWidth(button.fontMetrics().horizontalAdvance(button.text()) + 18)

    def _expanding(widget: QWidget) -> None:
        policy = widget.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        widget.setSizePolicy(policy)

    class _ElidedLabel(QLabel):
        """One-line label that shortens its text to fit; the full text is in the tooltip."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._full = ""
            self.setMinimumWidth(10)
            policy = self.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            self.setSizePolicy(policy)
            self.setStyleSheet("color: gray;")

        def set_full_text(self, text: str) -> None:
            self._full = str(text or "")
            self.setToolTip(self._full)
            self._elide()

        def full_text(self) -> str:
            return self._full

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt signature
            super().resizeEvent(event)
            self._elide()

        def _elide(self) -> None:
            width = max(10, self.width())
            super().setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, width))

    class _PickerRow(QWidget):
        """Common frame: selector, Refresh and Open-folder buttons, summary line."""

        def __init__(
            self,
            selector: QWidget,
            *,
            on_refresh: Callable[[], None] | None,
            folder: Callable[[], Any] | None,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._folder = folder
            # Compact buttons leave the selector most of the row.
            self.refresh_button = QPushButton("\u21bb", self)
            self.refresh_button.setToolTip(tr("picker.refresh") + ": " + tr("tooltip.picker_refresh"))
            _compact(self.refresh_button)
            if on_refresh is not None:
                self.refresh_button.clicked.connect(on_refresh)
            self.refresh_button.setVisible(on_refresh is not None)
            self.folder_button = QPushButton(tr("picker.open_folder"), self)
            self.folder_button.setToolTip(tr("tooltip.picker_open_folder"))
            _compact(self.folder_button)
            self.folder_button.clicked.connect(self.open_folder)
            self.folder_button.setVisible(folder is not None)
            self.summary = _ElidedLabel(self)
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            top.addWidget(selector, 1)
            top.addWidget(self.refresh_button)
            top.addWidget(self.folder_button)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            layout.addLayout(top)
            layout.addWidget(self.summary)

        def folder_path(self) -> Path | None:
            return existing_folder(self._folder() if self._folder is not None else None)

        def open_folder(self) -> bool:
            return open_folder(self.folder_path())

        def set_summary(self, text: str) -> None:
            self.summary.set_full_text(text)

    class ArtifactPicker(_PickerRow):
        """Pick one saved item from ``.combo`` (items carry their path/id as data).

        Disabled items (incompatible) stay listed with their reason in the item
        tooltip. With ``browse_filter`` a last "Browse..." entry opens a file
        dialog and adds the chosen file as an item.
        """

        def __init__(
            self,
            *,
            on_refresh: Callable[[], None] | None = None,
            folder: Callable[[], Any] | None = None,
            browse_filter: str | None = None,
            parent: QWidget | None = None,
        ) -> None:
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(SELECTOR_MIN_CHARS)
            _expanding(combo)
            super().__init__(combo, on_refresh=on_refresh, folder=folder, parent=parent)
            combo.setParent(self)
            self.combo = combo
            self._browse_filter = browse_filter
            self._last_index = 0
            self._custom_folder = folder
            combo.currentIndexChanged.connect(self._on_index_changed)

        # The folder button opens the selected file's folder when there is one.
        def folder_path(self) -> Path | None:
            data = self.combo.currentData()
            if isinstance(data, str) and data and Path(data).exists():
                return existing_folder(data)
            return super().folder_path()

        def add_browse_item(self) -> None:
            """Append the "Browse..." entry (call after filling the combo)."""

            if self._browse_filter is not None:
                self.combo.addItem(tr("picker.browse"), _BROWSE)
                self.combo.setItemData(self.combo.count() - 1, tr("tooltip.picker_browse"), Qt.ItemDataRole.ToolTipRole)

        def add_path_item(self, path: str | Path, label: str, tooltip: str = "") -> int:
            """Insert ``path`` before the "Browse..." entry and return its index."""

            index = self.combo.count()
            if index and self.combo.itemData(index - 1) == _BROWSE:
                index -= 1
            self.combo.insertItem(index, label, str(path))
            self.combo.setItemData(index, tooltip or str(path), Qt.ItemDataRole.ToolTipRole)
            return index

        def update_summary(self) -> None:
            index = self.combo.currentIndex()
            tip = self.combo.itemData(index, Qt.ItemDataRole.ToolTipRole) if index >= 0 else ""
            text = summary_from_tooltip(str(tip or ""))
            self.set_summary(text if self.combo.currentData() else "")

        def _on_index_changed(self, index: int) -> None:
            if self.combo.itemData(index) == _BROWSE:
                self._browse()
                return
            self._last_index = index
            self.update_summary()

        def _browse(self) -> None:
            start = str(super().folder_path() or "")
            path, _ = QFileDialog.getOpenFileName(self, tr("picker.browse_title"), start, self._browse_filter or "")
            if not path:
                self.combo.setCurrentIndex(min(self._last_index, self.combo.count() - 1))
                return
            index = self.combo.findData(str(Path(path)))
            if index < 0:
                index = self.add_path_item(Path(path), Path(path).name)
            self.combo.setCurrentIndex(index)

    class MultiArtifactPicker(_PickerRow):
        """Pick several items: a drop-down button holding a checkable ``.list``."""

        def __init__(
            self,
            *,
            on_refresh: Callable[[], None] | None = None,
            folder: Callable[[], Any] | None = None,
            none_text: str = "",
            parent: QWidget | None = None,
        ) -> None:
            button = QPushButton()
            button.setStyleSheet("text-align: left;")
            _expanding(button)
            # Same height as the single-choice combos so the rows line up.
            button.setMinimumHeight(QComboBox().sizeHint().height())
            button.setMinimumWidth(button.fontMetrics().averageCharWidth() * 10)
            super().__init__(button, on_refresh=on_refresh, folder=folder, parent=parent)
            button.setParent(self)
            self.button = button
            self._none_text = none_text
            self.popup = QWidget(self, Qt.WindowType.Popup)
            popup_layout = QVBoxLayout(self.popup)
            popup_layout.setContentsMargins(2, 2, 2, 2)
            self.list = QListWidget(self.popup)
            popup_layout.addWidget(self.list)
            self.list.itemChanged.connect(lambda _item: self.update_summary())
            self.list.itemClicked.connect(self._toggle)
            button.clicked.connect(self.show_popup)
            self.update_summary()

        def _toggle(self, item: Any) -> None:
            # A click anywhere on the row (not only the box) toggles it.
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                checked = item.checkState() == Qt.CheckState.Checked
                item.setCheckState(Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked)

        def show_popup(self) -> None:
            width = max(self.button.width(), 320)
            rows = max(1, min(self.list.count(), 10))
            height = self.list.sizeHintForRow(0) * rows + 12 if self.list.count() else 40
            self.popup.resize(width, max(40, height))
            self.popup.move(self.button.mapToGlobal(self.button.rect().bottomLeft()))
            self.popup.show()

        def checked_items(self) -> list[Any]:
            return [
                self.list.item(index)
                for index in range(self.list.count())
                if self.list.item(index).checkState() == Qt.CheckState.Checked
                and self.list.item(index).flags() & Qt.ItemFlag.ItemIsUserCheckable
            ]

        def update_summary(self) -> None:
            checked = self.checked_items()
            arrow = "  \u25bc"
            if not checked:
                self.button.setText(self._none_text + arrow)
                self.set_summary("")
                return
            self.button.setText(tr("picker.selected_count", count=len(checked)) + arrow)
            self.set_summary(", ".join(item.text() for item in checked))

    class PathField(QWidget):
        """Folder path input with Browse and Open buttons.

        While the field has no focus the path is shortened in the middle
        (``C:\\Users\\…\\artifacts``); :meth:`text` always returns the full path.
        """

        def __init__(self, text: str = "", default: str | None = None, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._default = default if default is not None else text
            self.edit = _ElidedPathEdit(text, self)
            self.browse_button = QPushButton("\u2026", self)
            self.browse_button.setToolTip(tr("tooltip.path_browse"))
            _compact(self.browse_button)
            self.browse_button.clicked.connect(self._browse)
            self.open_button = QPushButton(tr("picker.open_folder"), self)
            self.open_button.setToolTip(tr("tooltip.picker_open_folder"))
            _compact(self.open_button)
            self.open_button.clicked.connect(lambda: open_folder(self.text()))
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.edit, 1)
            layout.addWidget(self.browse_button)
            layout.addWidget(self.open_button)

        def text(self) -> str:
            return self.edit.full_text()

        def setText(self, text: str) -> None:  # noqa: N802 - mirrors QLineEdit
            self.edit.set_full_text(text)

        def is_default(self) -> bool:
            from os.path import normcase, normpath

            return normcase(normpath(self.text().strip() or ".")) == normcase(normpath(str(self._default) or "."))

        def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt signature
            super().setToolTip(text)
            self.edit.set_hint(text)

        def _browse(self) -> None:
            start = str(existing_folder(self.text()) or "")
            folder = QFileDialog.getExistingDirectory(self, tr("picker.browse_folder_title"), start)
            if folder:
                self.setText(str(Path(folder)))

    class _ElidedPathEdit(QLineEdit):
        def __init__(self, text: str, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._full = str(text)
            self._hint = ""
            _expanding(self)
            self.setMinimumWidth(80)
            self._show()

        def full_text(self) -> str:
            return super().text() if self.hasFocus() else self._full

        def set_full_text(self, text: str) -> None:
            self._full = str(text)
            if self.hasFocus():
                super().setText(self._full)
            else:
                self._show()

        def set_hint(self, text: str) -> None:
            self._hint = str(text or "")
            self._update_tooltip()

        def _update_tooltip(self) -> None:
            super().setToolTip(self._full + ("\n\n" + self._hint if self._hint else ""))

        def _show(self) -> None:
            width = max(20, self.width() - 12)
            super().setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, width))
            self.setCursorPosition(0)
            self._update_tooltip()

        def focusInEvent(self, event: Any) -> None:  # noqa: N802 - Qt signature
            super().setText(self._full)
            super().focusInEvent(event)

        def focusOutEvent(self, event: Any) -> None:  # noqa: N802 - Qt signature
            self._full = super().text()
            super().focusOutEvent(event)
            self._show()

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt signature
            super().resizeEvent(event)
            if not self.hasFocus():
                self._show()

else:

    class ArtifactPicker:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            from ..qt_compat import require_qt

            require_qt()

    MultiArtifactPicker = PathField = ArtifactPicker  # type: ignore[misc,assignment]


# Item data of the "Browse..." combo entry.
_BROWSE = "\u0000browse"

__all__ = [
    "ArtifactPicker",
    "MultiArtifactPicker",
    "PathField",
    "existing_folder",
    "open_folder",
    "summary_from_tooltip",
]
