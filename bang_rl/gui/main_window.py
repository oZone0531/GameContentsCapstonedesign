"""Main menu and game/training/settings entry point."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .i18n import tr
from .intent_preferences import IntentSeatPreferences
from .qt_compat import (
    Qt,
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
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
from .screens.intent_training import IntentTrainingScreen
from .screens.game_screen import GameScreen
from .screens.replay_screen import ReplayScreen
from .screens.settings import SettingsScreen
from .screens.training_monitor import TrainingMonitor, TrainingSetupPanel
from .settings import SettingsEditor
from .library_view import (
    ACTION_PLAY,
    ACTION_VIEW_REPLAY,
    LibraryChoice,
    build_preset,
)
from .training import (
    PresetResult,
    TrainingMonitorController,
    TrainingPreset,
    controller_status,
    training_is_active,
)


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
# The main menu has two columns: game modes on the left, everything else
# (training, library, settings, ...) on the right. Both columns use the same
# button size, the same vertical gap between buttons and the same frame
# margins, so their rows line up; MENU_COLUMN_GAP is the one horizontal gap.
MENU_BUTTON_WIDTH = 440
MENU_BUTTON_MIN_HEIGHT = 48
MENU_BUTTON_SPACING = 16
MENU_COLUMN_GAP = 48
MENU_COLUMN_MARGIN = 16
# Point size (>= 14pt) and horizontal/vertical padding applied via a stylesheet
# so the button text is large relative to the wide window.
MENU_BUTTON_FONT_POINT_SIZE = 16
MENU_BUTTON_STYLE = "padding: 12px 16px; font-size: 16pt;"


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


def style_menu_column(layout: Any) -> None:
    """Give a menu column's button layout the shared gap and margins.

    Used by both columns (game modes and the other menu items) so the vertical
    gap between buttons is the same everywhere and the rows of the two
    columns line up.
    """

    layout.setSpacing(MENU_BUTTON_SPACING)
    layout.setContentsMargins(MENU_COLUMN_MARGIN, MENU_COLUMN_MARGIN, MENU_COLUMN_MARGIN, MENU_COLUMN_MARGIN)


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
            self._intent_training_screen: IntentTrainingScreen | None = None
            self._settings_screen: SettingsScreen | None = None
            self._replay_screen: ReplayScreen | None = None
            self._selection: ModelSelection | None = None
            # Item 7: remember the last started game so Restart re-runs it.
            self._last_mode: GameMode | None = None
            self._last_artifact: str | None = None
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
            # Item 8: the title is the largest element at the top, with the
            # buttons evenly spaced below. A stretch above the title and below
            # the last button balances the whole group vertically.
            layout.setSpacing(16)
            layout.addStretch(1)
            title = QLabel(tr("app.title"), menu)
            title_font = title.font()
            # Clearly bigger than the menu buttons (~2x the button font).
            title_font.setPointSize(MENU_BUTTON_FONT_POINT_SIZE * 2)
            title_font.setBold(True)
            title.setFont(title_font)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(title)
            layout.setAlignment(title, Qt.AlignmentFlag.AlignHCenter)
            layout.addSpacing(8)
            # Two columns: game modes (left) and the other menu items (right).
            # Both are group boxes with the same margins and button gap, so
            # the button rows line up across the columns.
            mode_select = GameModeSelect(self._select_mode, menu)
            other = QGroupBox(tr("menu.group"), menu)
            other_layout = QVBoxLayout(other)
            style_menu_column(other_layout)
            self._menu_columns = (mode_select, other)
            training_button = QPushButton(tr("menu.training"), other)
            training_button.clicked.connect(self._show_training)
            library_button = QPushButton(tr("menu.library"), other)
            library_button.setToolTip(tr("tooltip.library_open"))
            library_button.clicked.connect(lambda: self._open_library())
            self._library_button = library_button
            intent_button = QPushButton(tr("menu.intent_training"), other)
            intent_button.clicked.connect(self._show_intent_training)
            settings_button = QPushButton(tr("menu.settings"), other)
            settings_button.clicked.connect(self._show_settings)
            replay_button = QPushButton(tr("menu.replay"), other)
            replay_button.clicked.connect(self._show_replay)
            exit_button = QPushButton(tr("menu.exit"), other)
            exit_button.clicked.connect(self.close)
            self._menu_buttons = (
                training_button, library_button, intent_button, settings_button, replay_button, exit_button,
            )
            for button in self._menu_buttons:
                other_layout.addWidget(button)
                style_menu_button(button)
            # A column with fewer buttons keeps its buttons at the top.
            other_layout.addStretch(1)
            columns = QHBoxLayout()
            columns.setSpacing(MENU_COLUMN_GAP)
            columns.addStretch(1)
            columns.addWidget(mode_select, 0, Qt.AlignmentFlag.AlignTop)
            columns.addWidget(other, 0, Qt.AlignmentFlag.AlignTop)
            columns.addStretch(1)
            layout.addLayout(columns)
            layout.addStretch(1)
            self._set_screen(menu)

        def _show_loading(self) -> None:
            """Show a minimal centered 'Loading...' screen (item 5, FEAT-003).

            Pressing a game-play mode button runs the blocking
            controller.start() (session creation / env build) before the
            GameScreen can appear, which reads as a frozen UI. This intermediate
            screen is set BEFORE that blocking work (see _start_game) and, once
            painted via QApplication.processEvents(), tells the user the wait is
            a normal loading step rather than an error. Mirrors the menu title
            styling lightly (bold, larger point size, centered).
            """

            loading = QWidget(self)
            layout = QVBoxLayout(loading)
            layout.addStretch(1)
            label = QLabel(tr("game.loading"), loading)
            font = label.font()
            font.setPointSize(MENU_BUTTON_FONT_POINT_SIZE * 2)
            font.setBold(True)
            label.setFont(font)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            layout.setAlignment(label, Qt.AlignmentFlag.AlignHCenter)
            layout.addStretch(1)
            self._set_screen(loading)

        def _show_intent_training(self) -> None:
            self._intent_training_screen = IntentTrainingScreen(
                on_back=self._show_menu,
                rule_policy_config=self.settings_editor.snapshot.config.preset.intent_policy,
                parent=self,
            )
            self._set_screen(self._intent_training_screen)

        def _show_training(self) -> None:
            # Building the training screen (saved-model scan, plot widgets) can
            # take a moment, so show the same 'Loading...' screen as game start
            # and let Qt paint it before the blocking construction.
            self._show_loading()
            if QApplication is not None:
                QApplication.processEvents()
            try:
                setup_panel = TrainingSetupPanel(
                    config=self.settings_editor.snapshot.config,
                    parent=self,
                    # Hardware / AI-training values are saved locally on change.
                    settings_editor=self.settings_editor,
                )
                self._training_monitor = TrainingMonitor(
                    self.training_controller,
                    setup_panel=setup_panel,
                    parent=self,
                    on_back=self._show_menu,
                    on_library_choice=self._handle_library_choice,
                )
            except Exception as exc:  # noqa: BLE001 - never strand the user on 'Loading...'
                self._training_monitor = None
                self.statusBar().showMessage(str(exc) or tr("game.error_generic"), 0)
                self._show_menu()
                return
            self._set_screen(self._training_monitor)

        def open_training(self, preset: TrainingPreset | None = None) -> PresetResult | None:
            """Open the training screen, optionally with a preset applied (data library).

            A running job is never touched: its monitor is shown again, and a
            preset is only applied while no job is active (otherwise a message
            says so and the setup form stays as it was).
            """

            busy = training_is_active(self.training_controller)
            self._show_training()
            monitor = self._training_monitor
            if monitor is None or preset is None:
                return None
            if busy:
                text = tr("library.busy")
                monitor.set_notice(text)
                self.statusBar().showMessage(text, 0)
                return PresetResult(False, text)
            result = monitor.setup_panel.apply_preset(preset)
            monitor.set_notice(result.message)
            self.statusBar().showMessage(result.message.splitlines()[0] if result.message else "", 0)
            return result

        def _open_library(self, kind_filter: str | None = None) -> None:
            """Open the data library; a chosen action is carried out after it closes."""

            from .dialogs.data_library import DataLibraryDialog

            dialog = DataLibraryDialog(
                config=self.settings_editor.snapshot.config,
                kind_filter=kind_filter if isinstance(kind_filter, str) else None,
                training_busy=lambda: training_is_active(self.training_controller),
                status_provider=lambda: controller_status(self.training_controller),
                parent=self,
            )
            dialog.exec()
            if dialog.chosen is not None:
                self._handle_library_choice(dialog.chosen)

        def _handle_library_choice(self, choice: LibraryChoice) -> None:
            """Carry out what the data library handed back (apply to training, play, replay)."""

            preset = build_preset(choice.action, choice.entries)
            if preset is not None:
                self.open_training(preset)
                return
            entry = choice.entries[0] if choice.entries else None
            if entry is None:
                return
            if choice.action == ACTION_PLAY:
                self._play_with_model(GameMode(str(choice.params.get("mode"))), str(entry.path))
            elif choice.action == ACTION_VIEW_REPLAY:
                self._show_replay_file(entry.path)

        def _play_with_model(self, mode: GameMode, path: str) -> None:
            """Start ``mode`` with the chosen ``.bangmodel`` (the same checks as the mode menu)."""

            self.controller.set_ai_metadata(None)
            self._selection = self._selection or ModelSelection()
            option = self._selection.discover([path], config=self.config)[0]
            if not option.compatible:
                self.statusBar().showMessage(tr("library.play_incompatible", name=Path(path).name), 0)
                return
            self.controller.set_ai_metadata(option.metadata)
            self._start_game(mode, option.path)

        def _show_replay_file(self, path: str | Path) -> None:
            """Open the replay screen on one replay file."""

            self._show_replay()
            if self._replay_screen is not None:
                self._replay_screen.open_path(path)

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
            # Item 6: both the human Main-AI mode and the Main-AI spectator mode
            # need a compatible .bangmodel, chosen through the same dialog.
            if mode in (GameMode.HUMAN_MAIN_DUMMY, GameMode.MAIN_ONE_DUMMY_FOUR):
                self._selection = self._selection or ModelSelection()
                paths = self.artifact_paths
                if not paths:
                    # No explicit list: find every .bangmodel by extension in the
                    # model folders (any file name), including saved/main_ai.
                    from bang_rl.training.main_ai_artifacts import list_saved_main_ai_models

                    try:
                        paths = tuple(str(model.path) for model in list_saved_main_ai_models())
                    except OSError:
                        paths = ()
                options = self._selection.discover(paths, config=self.config)
                dialog = ModelSelectionDialog(options, self)
                if dialog.exec() != dialog.DialogCode.Accepted or dialog.selected_option is None:
                    return
                artifact_path = dialog.selected_option.path
                # Record the chosen artifact's metadata (incl. version) so the
                # auto-saved replay captures which AI version was used.
                self.controller.set_ai_metadata(dialog.selected_option.metadata)
            self._start_game(mode, artifact_path)

        def _start_game(self, mode: GameMode, artifact_path: str | None) -> None:
            """Start ``mode`` on the shared controller and open a GameScreen.

            Shared by the mode menu and the item-7 Restart button. The last
            started mode/artifact are remembered so Restart can re-run the same
            game (spectator or human).
            """

            # Item 5 (FEAT-003): show a 'Loading...' screen BEFORE the blocking
            # controller.start() (session creation / env build) so the UI does
            # not look frozen. processEvents() lets Qt actually paint the
            # loading widget before start() blocks the event loop. This runs
            # inside _start_game, i.e. AFTER any ModelSelectionDialog in
            # _select_mode has been accepted, so the dialog is never hidden
            # behind the loading screen; on_restart funnels here too.
            self._show_loading()
            if QApplication is not None:
                QApplication.processEvents()
            # Detach any callback still bound to a previously-shown (and now
            # torn-down) GameScreen so controller.start()'s internal publish
            # cannot render into deleted C++ widgets.
            self.controller.on_update = None
            # Start the game with the settings staged in the Settings screen
            # (e.g. Dummy AI emotion). Previously games always ran on the
            # config given at window construction (the default SystemConfig).
            self.config = self.settings_editor.snapshot.config
            self.controller.config = self.config



            try:
                intent_preferences = IntentSeatPreferences.load()
                self.controller.intent_selections = intent_preferences.resolved
                self.controller.intent_model_path = intent_preferences.model_path

                from .game import GameMode as _Mode
                dummy_seats = {
                    _Mode.DUMMY_FIVE: range(5),
                    _Mode.HUMAN_DUMMY: range(1, 5),
                    _Mode.HUMAN_MAIN_DUMMY: range(2, 5),
                    _Mode.MAIN_ONE_DUMMY_FOUR: range(1, 5),
                }.get(mode, ())
                if any(intent_preferences.resolved[i] == "learned" for i in dummy_seats):
                    intent_preferences.validate_model()
            except Exception as exc:
                self.statusBar().showMessage(str(exc), 0)
                self._show_menu()
                return







            advance_ai = not bool(getattr(mode, "is_human", False))
            try:
                self.controller.start(
                    mode, artifact_path=artifact_path, advance_ai=advance_ai
                )
            except GameGuiError as exc:
                # showMessage's default timeout auto-clears the text; passing 0
                # keeps the error visible in the always-present status bar until
                # the next action replaces it. Fall back to a localized generic
                # message when the exception carries no text.
                message = str(exc) or tr("game.error_generic")
                self.statusBar().showMessage(message, 0)
                # Item 5 (FEAT-003): do not strand the user on the 'Loading...'
                # screen when start() fails; restore the menu so the status-bar
                # error is shown on a sensible screen they can act from.
                self._show_menu()
                return
            self._last_mode = mode
            self._last_artifact = artifact_path
            # GameScreen.__init__ re-binds controller.on_update to its own
            # _render and renders the current snapshot itself, so building it
            # after start() keeps the callback pointed at a live screen. Item 7
            # wires the Restart / Return-to-menu callbacks the screen shows when
            # a game is over.
            self._game_screen = GameScreen(
                self.controller, self, on_back=self._show_menu, on_restart=self.on_restart
            )
            self._set_screen(self._game_screen)

        def on_restart(self) -> None:
            """Item 7: restart the last game in the SAME mode + rebuild screen."""

            mode = getattr(self, "_last_mode", None)
            if mode is None:
                self._show_menu()
                return
            self._start_game(mode, getattr(self, "_last_artifact", None))

        def closeEvent(self, event: Any) -> None:
            self.controller.close()
            event.accept()

else:

    class MainWindow:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_qt()


__all__ = ["MainWindow"]
