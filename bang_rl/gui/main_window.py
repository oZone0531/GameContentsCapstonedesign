"""Main menu and game/training/settings entry point."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .i18n import tr
from .qt_compat import (
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QT_AVAILABLE,
    require_qt,
)
from .dialogs.model_select import ModelSelection, ModelSelectionDialog
from .game import GameGuiError, GameMode, GameSessionController
from .screens.game_mode import GameModeSelect
from .screens.game_screen import GameScreen
from .screens.settings import SettingsScreen
from .screens.training_monitor import TrainingMonitor
from .settings import SettingsEditor
from .training import TrainingMonitorController


if QT_AVAILABLE:  # pragma: no cover - UI rendering is host/display dependent

    class MainWindow(QMainWindow):
        """Menu → game, training monitor, or settings for the three modes."""

        def __init__(
            self,
            *,
            service: Any = None,
            training_service: Any = None,
            artifact_paths: Iterable[str | Path] = (),
            config: Any = None,
            seed: int | None = None,
            parent: QWidget | None = None,
        ) -> None:
            require_qt()
            super().__init__(parent)
            self.setWindowTitle(tr("app.title"))
            self.service = service
            self.training_service = training_service
            self.artifact_paths = tuple(str(path) for path in artifact_paths)
            self.config = config
            self.seed = seed
            self.controller = GameSessionController(service, config=config, seed=seed)
            self.training_controller = TrainingMonitorController(training_service)
            self.settings_editor = SettingsEditor(config)
            self._game_screen: GameScreen | None = None
            self._training_monitor: TrainingMonitor | None = None
            self._settings_screen: SettingsScreen | None = None
            self._selection: ModelSelection | None = None
            self._show_menu()

        def _set_screen(self, widget: QWidget) -> None:
            """Swap the central widget and resize the window to fit it.

            Screens vary a lot in size (the settings form is much taller than
            the menu), so without this the window keeps the largest screen's
            geometry and the smaller menu ends up stranded in a big window.
            """

            self.setCentralWidget(widget)
            widget.adjustSize()
            self.adjustSize()

        def _show_menu(self) -> None:
            menu = QWidget(self)
            layout = QVBoxLayout(menu)
            title = QLabel(tr("app.title"), menu)
            layout.addWidget(title)
            mode_select = GameModeSelect(self._select_mode, menu)
            layout.addWidget(mode_select)
            training_button = QPushButton(tr("menu.training"), menu)
            training_button.clicked.connect(self._show_training)
            layout.addWidget(training_button)
            settings_button = QPushButton(tr("menu.settings"), menu)
            settings_button.clicked.connect(self._show_settings)
            layout.addWidget(settings_button)
            exit_button = QPushButton(tr("menu.exit"), menu)
            exit_button.clicked.connect(self.close)
            layout.addWidget(exit_button)
            self._set_screen(menu)

        def _show_training(self) -> None:
            self._training_monitor = TrainingMonitor(
                self.training_controller, parent=self, on_back=self._show_menu
            )
            self._set_screen(self._training_monitor)

        def _show_settings(self) -> None:
            self._settings_screen = SettingsScreen(
                self.settings_editor, parent=self, on_back=self._show_menu
            )
            self._set_screen(self._settings_screen)

        def _select_mode(self, mode: GameMode) -> None:
            artifact_path: str | None = None
            if mode is GameMode.HUMAN_MAIN_DUMMY:
                self._selection = self._selection or ModelSelection()
                options = self._selection.discover(self.artifact_paths, config=self.config)
                dialog = ModelSelectionDialog(options, self)
                if dialog.exec() != dialog.DialogCode.Accepted or dialog.selected_option is None:
                    return
                artifact_path = dialog.selected_option.path
            # Detach any callback still bound to a previously-shown (and now
            # torn-down) GameScreen so controller.start()'s internal publish
            # cannot render into deleted C++ widgets.
            self.controller.on_update = None
            try:
                self.controller.start(mode, artifact_path=artifact_path)
            except GameGuiError as exc:
                # showMessage's default timeout auto-clears the text; passing 0
                # keeps the error visible in the always-present status bar until
                # the next action replaces it. Fall back to a localized generic
                # message when the exception carries no text.
                message = str(exc) or tr("game.error_generic")
                self.statusBar().showMessage(message, 0)
                return
            # GameScreen.__init__ re-binds controller.on_update to its own
            # _render and renders the current snapshot itself, so building it
            # after start() keeps the callback pointed at a live screen.
            self._game_screen = GameScreen(self.controller, self, on_back=self._show_menu)
            self._set_screen(self._game_screen)

        def closeEvent(self, event: Any) -> None:
            self.controller.close()
            event.accept()

else:

    class MainWindow:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["MainWindow"]
