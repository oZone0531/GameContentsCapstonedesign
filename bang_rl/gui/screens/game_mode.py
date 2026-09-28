"""Game mode choices for the main menu."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..game import GameMode
from ..i18n import tr
from ..qt_compat import (
    QGroupBox,
    QVBoxLayout,
    QPushButton,
    QT_AVAILABLE,
    require_qt,
)


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class GameModeSelect(QGroupBox):
        """Three TASK-9-2 modes; selection emits the stable enum value."""

        def __init__(self, on_selected: Callable[[GameMode], None] | None = None, parent: Any = None) -> None:
            require_qt()
            super().__init__(tr("mode.group"), parent)
            self.on_selected = on_selected
            layout = QVBoxLayout(self)
            self.buttons: dict[GameMode, QPushButton] = {}
            for mode in GameMode:
                button = QPushButton(tr(f"mode.{mode.value}"), self)
                button.clicked.connect(lambda _checked=False, choice=mode: self._select(choice))
                layout.addWidget(button)
                self.buttons[mode] = button

        def _select(self, mode: GameMode) -> None:
            if self.on_selected is not None:
                self.on_selected(mode)

else:

    class GameModeSelect:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["GameModeSelect"]
