"""Runnable entry point for the optional Bang RL GUI.

Launch with::

    python -m bang_rl.gui

The optional PyQt6 dependencies must be installed first (see ``install_gui.bat``
or ``pip install -e ".[gui]"``).  When they are missing this prints an
actionable message instead of a raw traceback.
"""
from __future__ import annotations

import sys

from .qt_compat import QApplication, GuiDependencyError, QFont, QT_AVAILABLE, QT_IMPORT_ERROR
from .main_window import MIN_FONT_POINT_SIZE, MainWindow


def main(argv: list[str] | None = None) -> int:
    """Create the Qt application, show the main window, and run the loop."""

    if not QT_AVAILABLE:
        detail = f" ({QT_IMPORT_ERROR})" if QT_IMPORT_ERROR else ""
        print(
            "The Bang RL GUI needs the optional PyQt6 dependencies." + detail + "\n"
            "Install them first, for example by running install_gui.bat or:\n"
            '    python -m pip install -e ".[gui]"',
            file=sys.stderr,
        )
        return 1

    args = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(args)
    # Enlarge the whole GUI's text to at least MIN_FONT_POINT_SIZE (>=10pt) so
    # every screen is comfortably readable. Keep any larger platform default.
    font: QFont = app.font()
    font.setPointSize(max(MIN_FONT_POINT_SIZE, font.pointSize()))
    app.setFont(font)
    try:
        window = MainWindow()
    except GuiDependencyError as exc:  # pragma: no cover - defensive
        print(str(exc), file=sys.stderr)
        return 1
    window.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
