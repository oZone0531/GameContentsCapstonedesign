"""Main menu and game/training/settings entry point."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
            self.setWindowTitle("Bang RL")
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

        def _show_menu(self) -> None:
            menu = QWidget(self)
            layout = QVBoxLayout(menu)
            title = QLabel("Bang RL", menu)
            layout.addWidget(title)
            mode_select = GameModeSelect(self._select_mode, menu)
            layout.addWidget(mode_select)
            training_button = QPushButton("AI Training", menu)
            training_button.clicked.connect(self._show_training)
            layout.addWidget(training_button)
            settings_button = QPushButton("Settings", menu)
            settings_button.clicked.connect(self._show_settings)
            layout.addWidget(settings_button)
            exit_button = QPushButton("Exit", menu)
            exit_button.clicked.connect(self.close)
            layout.addWidget(exit_button)
            self.setCentralWidget(menu)

        def _show_training(self) -> None:
            self._training_monitor = TrainingMonitor(self.training_controller, parent=self)
            self.setCentralWidget(self._training_monitor)

        def _show_settings(self) -> None:
            self._settings_screen = SettingsScreen(self.settings_editor, parent=self)
            self.setCentralWidget(self._settings_screen)

        def _select_mode(self, mode: GameMode) -> None:
            artifact_path: str | None = None
            if mode is GameMode.HUMAN_MAIN_DUMMY:
                self._selection = self._selection or ModelSelection()
                options = self._selection.discover(self.artifact_paths, config=self.config)
                dialog = ModelSelectionDialog(options, self)
                if dialog.exec() != dialog.DialogCode.Accepted or dialog.selected_option is None:
                    return
                artifact_path = dialog.selected_option.path
            try:
                self.controller.start(mode, artifact_path=artifact_path)
            except GameGuiError as exc:
                self.statusBar().showMessage(str(exc))
                return
            self._game_screen = GameScreen(self.controller, self)
            self.setCentralWidget(self._game_screen)

        def closeEvent(self, event: Any) -> None:
            self.controller.close()
            event.accept()

else:

    class MainWindow:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["MainWindow"]
