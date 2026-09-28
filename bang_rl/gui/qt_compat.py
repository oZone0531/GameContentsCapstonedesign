"""Optional PyQt6 boundary for the game, training, and settings GUI.

Importing :mod:`bang_rl.gui` remains safe in headless installations.  Qt and
pyqtgraph are imported only by concrete widgets and a recoverable error is
raised when a widget is actually constructed.
"""
from __future__ import annotations

from typing import Any


class GuiDependencyError(RuntimeError):
    """The optional GUI dependencies are not installed."""


try:  # pragma: no cover - availability depends on the host environment
    from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
    QT_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised when Qt is absent
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = exc
    QObject = QTimer = Qt = pyqtSignal = QApplication = None
    QComboBox = None
    QCheckBox = QDialog = QDialogButtonBox = QDoubleSpinBox = None
    QFormLayout = QGroupBox = QHBoxLayout = QHeaderView = QLabel = QLineEdit = None
    QListWidget = QListWidgetItem = QMainWindow = QPlainTextEdit = None
    QProgressBar = QPushButton = QScrollArea = QSpinBox = QStackedWidget = None
    QTableWidget = QTableWidgetItem = QTabWidget = QVBoxLayout = QWidget = None


def require_qt() -> None:
    """Raise an actionable error instead of failing during module import."""

    if not QT_AVAILABLE:
        raise GuiDependencyError(
            "Game/training/settings GUI requires optional PyQt6 dependencies; "
            "install with `pip install 'bang-rl[gui]'`."
        ) from QT_IMPORT_ERROR


__all__ = [
    "GuiDependencyError",
    "QT_AVAILABLE",
    "QApplication",
    "QCheckBox",
    "QComboBox",
    "QDialog",
    "QDialogButtonBox",
    "QDoubleSpinBox",
    "QFormLayout",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QLabel",
    "QLineEdit",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QPlainTextEdit",
    "QProgressBar",
    "QPushButton",
    "QScrollArea",
    "QSpinBox",
    "QStackedWidget",
    "QTableWidget",
    "QTableWidgetItem",
    "QTabWidget",
    "QTimer",
    "QVBoxLayout",
    "QWidget",
    "require_qt",
]
