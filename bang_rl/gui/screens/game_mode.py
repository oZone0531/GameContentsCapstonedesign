"""Game mode choices for the main menu."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..game import GameMode
from ..i18n import tr
from ..qt_compat import (
    Qt,
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
            # Deferred import avoids a circular import at module load time
            # (main_window imports this screen); the shared helper sizes each
            # mode button to about half the window width with a larger font.
            from ..main_window import style_menu_button, style_menu_column
            layout = QVBoxLayout(self)
            # Same gap and margins as the main menu's other column, so the
            # button rows of both columns line up.
            style_menu_column(layout)
            self.buttons: dict[GameMode, QPushButton] = {}
            for mode in GameMode:
                button = QPushButton(tr(f"mode.{mode.value}"), self)
                button.clicked.connect(lambda _checked=False, choice=mode: self._select(choice))
                # Same size and font as the main-menu buttons.
                style_menu_button(button)
                layout.addWidget(button)
                self.buttons[mode] = button
            layout.addStretch(1)

        def _select(self, mode: GameMode) -> None:
            if self.on_selected is not None:
                self.on_selected(mode)

else:

    class GameModeSelect:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["GameModeSelect"]
