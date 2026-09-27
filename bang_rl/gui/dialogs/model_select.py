"""Inference artifact selection with compatibility gating."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..qt_compat import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)


@dataclass(frozen=True, slots=True)
class ArtifactOption:
    path: str
    compatible: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[Mapping[str, Any], ...] = ()

    @property
    def display_name(self) -> str:
        return Path(self.path).name

    @property
    def compatibility_text(self) -> str:
        return "Compatible" if self.compatible else "Not compatible"


class ModelSelection:
    """Discover and filter artifacts without requiring a Qt application."""

    def __init__(self, compatibility_service: Any | None = None) -> None:
        if compatibility_service is None:
            from bang_rl.app import CompatibilityService

            compatibility_service = CompatibilityService()
        self.compatibility_service = compatibility_service

    def discover(self, paths: Iterable[str | Path], *, config: Any = None) -> tuple[ArtifactOption, ...]:
        options: list[ArtifactOption] = []
        for path in paths:
            artifact_path = str(path)
            try:
                kwargs = {} if config is None else {"config": config}
                result = self.compatibility_service.check(artifact_path, **kwargs)
                options.append(
                    ArtifactOption(
                        artifact_path,
                        bool(result.compatible),
                        dict(result.metadata or {}),
                        tuple(result.errors or ()),
                    )
                )
            except Exception as exc:
                options.append(
                    ArtifactOption(
                        artifact_path,
                        False,
                        {},
                        ({"error_type": type(exc).__name__, "message": str(exc)},),
                    )
                )
        return tuple(options)

    @staticmethod
    def selectable(options: Iterable[ArtifactOption]) -> tuple[ArtifactOption, ...]:
        return tuple(option for option in options if option.compatible)


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class ModelSelectionDialog(QDialog):
        """Modal list that never permits an incompatible artifact to be chosen."""

        def __init__(self, options: Iterable[ArtifactOption], parent: QWidget | None = None) -> None:
            require_qt()
            super().__init__(parent)
            self.setWindowTitle("Select inference model")
            self._options = tuple(options)
            self._selected: ArtifactOption | None = None
            layout = QVBoxLayout(self)
            self._list = QListWidget(self)
            self._details = QLabel("Select a compatible model.", self)
            self._details.setWordWrap(True)
            layout.addWidget(self._list)
            layout.addWidget(self._details)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                parent=self,
            )
            buttons.accepted.connect(self._accept_selection)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)
            for option in self._options:
                item = QListWidgetItem(f"{option.display_name} — {option.compatibility_text}")
                item.setData(0x0100, option)
                item.setFlags(item.flags() | 0x0001)
                if not option.compatible:
                    item.setFlags(item.flags() & ~0x0001)
                self._list.addItem(item)
            self._list.currentItemChanged.connect(self._show_details)

        @property
        def selected_option(self) -> ArtifactOption | None:
            return self._selected

        def _show_details(self, item: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
            option = None if item is None else item.data(0x0100)
            if option is None:
                self._details.setText("Select a compatible model.")
                return
            metadata = ", ".join(f"{key}: {value}" for key, value in option.metadata.items())
            errors = "; ".join(str(error.get("message", error)) for error in option.errors)
            suffix = f" Errors: {errors}" if errors else ""
            self._details.setText(f"{metadata or 'No metadata available.'}{suffix}")

        def _accept_selection(self) -> None:
            item = self._list.currentItem()
            option = None if item is None else item.data(0x0100)
            if option is None or not option.compatible:
                self._details.setText("This model is incompatible and cannot be executed.")
                return
            self._selected = option
            self.accept()

else:

    class ModelSelectionDialog:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["ArtifactOption", "ModelSelection", "ModelSelectionDialog"]
