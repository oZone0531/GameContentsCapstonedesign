"""Optional PyQt6 game, training, and settings GUI with service seams."""

from .game import GameControllerSnapshot, GameGuiError, GameMode, GameSessionController
from .main_window import MainWindow
from .qt_compat import GuiDependencyError, QT_AVAILABLE
from .settings import RESOURCE_HELP, SettingsEditor, SettingsSnapshot, SettingsValidationError
from .training import (
    TrainingMonitorController,
    TrainingMonitorSnapshot,
    TrainingServicePort,
    TrainingSetup,
    public_training_metadata,
)

__all__ = [
    "GameControllerSnapshot",
    "GameGuiError",
    "GameMode",
    "GameSessionController",
    "GuiDependencyError",
    "MainWindow",
    "QT_AVAILABLE",
    "RESOURCE_HELP",
    "SettingsEditor",
    "SettingsSnapshot",
    "SettingsValidationError",
    "TrainingMonitorController",
    "TrainingMonitorSnapshot",
    "TrainingServicePort",
    "TrainingSetup",
    "public_training_metadata",
]
