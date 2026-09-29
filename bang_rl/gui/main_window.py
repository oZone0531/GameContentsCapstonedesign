"""Main menu and game/training/settings entry point."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .i18n import tr
from .qt_compat import (
    Qt,
    QFileDialog,
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
from .replay import ReplayModel
from .screens.game_mode import GameModeSelect
from .screens.game_screen import GameScreen
from .screens.replay_screen import ReplayScreen
from .screens.settings import SettingsScreen
from .screens.training_monitor import TrainingMonitor
from .settings import SettingsEditor
from .training import TrainingMonitorController


# Fixed, large window geometry shared by every screen. These live outside the
# ``if QT_AVAILABLE:`` gate so they can be imported and unit-tested headlessly.
# The window is sized once (see MainWindow.__init__) and never resized when the
# user switches between the menu, game, training, and settings screens, so its
# size stays constant and every screen's content fits without scrolling.
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 860

# Minimum application font point size (item: make GUI text >= 10pt). Shared with
# __main__.main() which sets the global QApplication font.
MIN_FONT_POINT_SIZE = 10

# Menu button geometry (user item 1). Each main-menu button and each game-mode
# button is sized to about half the window width, with horizontal padding and a
# larger font, then centered in its vertical layout. These live outside the
# ``if QT_AVAILABLE:`` gate so they can be imported and unit-tested headlessly
# (mirroring WINDOW_WIDTH / MIN_FONT_POINT_SIZE above). GameModeSelect imports
# them so the whole menu stays visually consistent.
MENU_BUTTON_WIDTH = WINDOW_WIDTH // 2
MENU_BUTTON_MIN_HEIGHT = 48
# Point size (>= 14pt) and horizontal/vertical padding applied via a stylesheet
# so the button text is large relative to the wide window.
MENU_BUTTON_FONT_POINT_SIZE = 16
MENU_BUTTON_STYLE = "padding: 12px 40px; font-size: 16pt;"


def style_menu_button(button: Any) -> None:
    """Apply the shared half-width + large-font menu-button geometry.

    Shared by the main menu (main_window) and the game-mode selector
    (game_mode.py) so the whole menu is visually consistent. Defined outside
    any ``if QT_AVAILABLE:`` gate but only ever called on real ``QPushButton``
    instances, which exist only when Qt is available.
    """

    button.setFixedWidth(MENU_BUTTON_WIDTH)
    button.setMinimumHeight(MENU_BUTTON_MIN_HEIGHT)
    button.setStyleSheet(MENU_BUTTON_STYLE)
    font = button.font()
    if font.pointSize() < MENU_BUTTON_FONT_POINT_SIZE:
        font.setPointSize(MENU_BUTTON_FONT_POINT_SIZE)
        button.setFont(font)


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
            # Size the window once to a large fixed geometry and keep it there.
            # A matching minimum size stops any screen from shrinking it, and
            # _set_screen no longer calls adjustSize(), so switching screens
            # never changes the window size (item 1).
            self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
            self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
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
            self._replay_screen: ReplayScreen | None = None
            self._selection: ModelSelection | None = None
            # An in-memory replay episode may be supplied by an embedder; when
            # set it is used instead of prompting for a file.
            self.replay_episode: Any = None
            self._show_menu()

        def _set_screen(self, widget: QWidget) -> None:
            """Swap the central widget while keeping the window size fixed.

            The window is sized once in __init__ to a large fixed geometry, so
            this must NOT call adjustSize() on the widget or the window.
            Otherwise the window would resize to each screen's preferred size
            and jump around when the user moves between the menu, game,
            training, and settings screens (item 1).
            """

            self.setCentralWidget(widget)

        def _style_menu_button(self, button: QPushButton, layout: QVBoxLayout) -> None:
            """Size a menu button to about half the window with a larger font.

            Applies the shared MENU_BUTTON_* geometry (half-width, minimum
            height, horizontal padding + larger font via a stylesheet) and
            centers the button horizontally in its vertical layout so the
            half-width buttons sit centered (user item 1).
            """

            style_menu_button(button)
            layout.setAlignment(button, Qt.AlignmentFlag.AlignHCenter)

        def _show_menu(self) -> None:
            menu = QWidget(self)
            layout = QVBoxLayout(menu)
            title = QLabel(tr("app.title"), menu)
            layout.addWidget(title)
            layout.setAlignment(title, Qt.AlignmentFlag.AlignHCenter)
            mode_select = GameModeSelect(self._select_mode, menu)
            layout.addWidget(mode_select)
            layout.setAlignment(mode_select, Qt.AlignmentFlag.AlignHCenter)
            training_button = QPushButton(tr("menu.training"), menu)
            training_button.clicked.connect(self._show_training)
            layout.addWidget(training_button)
            self._style_menu_button(training_button, layout)
            settings_button = QPushButton(tr("menu.settings"), menu)
            settings_button.clicked.connect(self._show_settings)
            layout.addWidget(settings_button)
            self._style_menu_button(settings_button, layout)
            replay_button = QPushButton(tr("menu.replay"), menu)
            replay_button.clicked.connect(self._show_replay)
            layout.addWidget(replay_button)
            self._style_menu_button(replay_button, layout)
            exit_button = QPushButton(tr("menu.exit"), menu)
            exit_button.clicked.connect(self.close)
            layout.addWidget(exit_button)
            self._style_menu_button(exit_button, layout)
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

        def _show_replay(self) -> None:
            """Open the graphical replay viewer for a recorded episode.

            An in-memory episode (``self.replay_episode``) is used when present;
            otherwise the user is prompted for a recorded episode file. If no
            source is available the screen still opens with an empty model, so a
            missing recorder never blocks reaching the replay UI.
            """

            episode = self.replay_episode
            model = None
            if episode is not None:
                try:
                    model = ReplayModel(episode)
                except (TypeError, ValueError, KeyError) as exc:
                    self.statusBar().showMessage(str(exc) or tr("replay.no_episode"), 0)
                    model = None
            # Open the folder-based browser rooted at the in-project replay
            # folder so every auto-saved human game is listed. An in-memory
            # episode (if supplied) is still shown first via the model argument.
            from ..paths import REPLAYS_DIR

            self._replay_screen = ReplayScreen(
                model,
                parent=self,
                on_back=self._show_menu,
                replay_directory=REPLAYS_DIR,
            )
            self._set_screen(self._replay_screen)

        def _prompt_replay_episode(self) -> Any:
            """Prompt for a recorded episode file and load it as a dict.

            Returns the parsed mapping, or ``None`` when the user cancels or the
            file cannot be read. Kept tolerant so the replay screen still opens
            (empty) rather than raising when no valid file is chosen.

            Privacy: the standard recorded episode shape is already a public
            projection. ``bang_rl.data.collector`` stores only public data --
            every observation goes through ``_observation`` which strips the
            private role keys, and every event goes through ``_event_mapping``
            which rejects non-public events. The only private field a dump can
            carry is the top-level ``role_labels`` list (written only under
            ``include_private=True``), and :class:`~bang_rl.gui.replay.ReplayModel`
            never reads it: it copies just the public observation keys, so hidden
            (living, non-self, non-sheriff) roles stay hidden on the board. See
            the note in ``bang_rl/gui/replay.py``.
            """

            path, _selected = QFileDialog.getOpenFileName(
                self, tr("replay.open_file"), "", tr("replay.file_filter")
            )
            if not path:
                return None
            try:
                text = Path(path).read_text(encoding="utf-8")
                return json.loads(text)
            except (OSError, ValueError) as exc:
                self.statusBar().showMessage(str(exc) or tr("replay.no_episode"), 0)
                return None

        def _select_mode(self, mode: GameMode) -> None:
            artifact_path: str | None = None
            # Clear any prior Main-AI metadata so a non-Main mode does not carry
            # a stale AI version into its saved replay.
            self.controller.set_ai_metadata(None)
            if mode is GameMode.HUMAN_MAIN_DUMMY:
                self._selection = self._selection or ModelSelection()
                options = self._selection.discover(self.artifact_paths, config=self.config)
                dialog = ModelSelectionDialog(options, self)
                if dialog.exec() != dialog.DialogCode.Accepted or dialog.selected_option is None:
                    return
                artifact_path = dialog.selected_option.path
                # Record the chosen artifact's metadata (incl. version) so the
                # auto-saved replay captures which AI version was used.
                self.controller.set_ai_metadata(dialog.selected_option.metadata)
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
