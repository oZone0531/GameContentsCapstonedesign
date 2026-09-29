"""Headless-testable game-session orchestration for the GUI.

This module intentionally knows only the application service DTO boundary.  It
never imports or inspects an Environment, Agent, or private role state.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from bang_rl.app import GameService, PublicGameState, SessionConfig, SessionHandle
else:
    GameService = Any
    PublicGameState = Any
    SessionConfig = Any
    SessionHandle = Any


@dataclass(frozen=True, slots=True)
class _GuiSessionConfig:
    """Fallback request shape used by headless/mock service seams."""

    mode: str
    config: Any = None
    seed: int | None = None
    artifact_path: str | None = None


class GameGuiError(RuntimeError):
    """A user-facing game GUI operation failed."""


class GameMode(str, Enum):
    HUMAN_RANDOM = "human_random"
    HUMAN_DUMMY = "human_dummy"
    HUMAN_MAIN_DUMMY = "human_main_dummy"

    @property
    def label(self) -> str:
        return {
            self.HUMAN_RANDOM: "Human 1 vs Random AI 4",
            self.HUMAN_DUMMY: "Human 1 vs Dummy AI 4",
            self.HUMAN_MAIN_DUMMY: "Human 1 vs Main AI 1 + Dummy AI 3",
        }[self]


class GameServicePort(Protocol):
    """Minimal service seam used by the controller and headless tests."""

    def create_session(self, request: SessionConfig) -> SessionHandle: ...

    def get_state(self, handle: SessionHandle | str) -> PublicGameState: ...

    def step(self, handle: SessionHandle | str, action: Any | None = None) -> Any: ...

    def close_session(self, handle: SessionHandle | str) -> None: ...


@dataclass(frozen=True, slots=True)
class GameControllerSnapshot:
    state: PublicGameState | None
    mode: GameMode | None
    paused: bool
    active: bool
    message: str | None


class GameSessionController:
    """Coordinate a human game while remaining independent of Qt.

    ``service`` may be the real :class:`GameService` or a small mock implementing
    :class:`GameServicePort`.  AI turns are advanced through ``service.step``
    until the human seat (slot 0) is active again.  The controller closes a
    terminal/truncated session exactly once, after retaining its final DTO for
    display.
    """

    HUMAN_PLAYER = 0
    ACTION_COUNT = 6

    def __init__(
        self,
        service: GameServicePort | None = None,
        *,
        config: Any = None,
        seed: int | None = None,
        on_update: Callable[[GameControllerSnapshot], None] | None = None,
    ) -> None:
        self.service = service
        self.config = config
        self.seed = seed
        self.on_update = on_update
        self.handle: SessionHandle | None = None
        self.state: PublicGameState | None = None
        self.mode: GameMode | None = None
        self.paused = False
        self.active = False
        self._closed = False
        self.message: str | None = None
        # Replay capture (FEAT-003). Every human game is auto-saved on terminal.
        # ai_metadata is optional artifact metadata (incl. version) supplied by
        # main_window when a Main-AI model is chosen; mode_ai_versions maps each
        # non-human seat to its AI descriptor. _replay_observations accumulates a
        # deep-copyable public observation per turn and _replay_saved guards the
        # one-save-per-episode rule.
        self.ai_metadata: Mapping[str, Any] | None = None
        self.mode_ai_versions: dict[str, Any] = {}
        self._replay_observations: list[dict[str, Any]] = []
        self._replay_saved = False
        # Time-travel (FEAT-004 item 8). While previewing a past moment the
        # live session state is stashed here so resume_live() can restore it.
        # ``previewing`` marks that self.state currently holds a historic,
        # read-only snapshot rather than the live session state.
        self.previewing = False
        self._live_state: PublicGameState | None = None
        self._live_active = False
        self._live_paused = False

    def set_ai_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        """Record artifact metadata (incl. version) for the Main-AI seat.

        Called by main_window when a Main-AI model is selected so the saved
        replay can record which AI version was used. Kept Qt-free.
        """

        self.ai_metadata = dict(metadata) if metadata else None

    @property
    def snapshot(self) -> GameControllerSnapshot:
        return GameControllerSnapshot(self.state, self.mode, self.paused, self.active, self.message)

    def _publish(self) -> GameControllerSnapshot:
        snapshot = self.snapshot
        if self.on_update is not None:
            self.on_update(snapshot)
        return snapshot

    def start(self, mode: GameMode | str, *, artifact_path: str | None = None) -> GameControllerSnapshot:
        selected = mode if isinstance(mode, GameMode) else GameMode(str(mode))
        if selected is GameMode.HUMAN_MAIN_DUMMY and not artifact_path:
            raise GameGuiError("Select a compatible .bangmodel before starting Main AI mode.")
        if self.active:
            self.close()
        try:
            from bang_rl.app import SessionConfig as ConcreteSessionConfig
        except (ImportError, SyntaxError):
            # A mock service should remain usable even when optional/runtime
            # application modules are unavailable in a headless test install.
            ConcreteSessionConfig = _GuiSessionConfig

        request = ConcreteSessionConfig(
            mode=selected.value,
            config=self.config,
            seed=self.seed,
            artifact_path=artifact_path,
        )
        try:
            if self.service is None:
                from bang_rl.app import GameService as ConcreteGameService

                self.service = ConcreteGameService()
            self.handle = self.service.create_session(request)
            self.state = self.service.get_state(self.handle)
            self.mode = selected
            self.paused = False
            self.active = True
            self._closed = False
            self.message = None
            # Reset replay capture for the new episode and derive the AI
            # descriptor for each non-human seat from the mode (+ optional
            # artifact metadata for the Main-AI seat).
            self._replay_observations = []
            self._replay_saved = False
            self.mode_ai_versions = self._derive_ai_versions(selected)
            self._record_replay_snapshot()
            self._advance_to_human_turn()
            return self._publish()
        except Exception as exc:
            if self.handle is not None and not self._closed:
                try:
                    self.service.close_session(self.handle)
                finally:
                    self._closed = True
            self.handle = None
            self.state = None
            self.mode = None
            self.active = False
            self.message = self._error_message(exc)
            self._publish()
            raise GameGuiError(self.message) from exc

    def submit_human_action(self, action: int, *, advance_ai: bool = True) -> GameControllerSnapshot:
        if not self.active or self.handle is None or self.state is None:
            raise GameGuiError("No active game session.")
        if self.paused:
            self.message = "Game is paused. Resume before choosing an action."
            return self._publish()
        try:
            normalized = int(action)
        except (TypeError, ValueError) as exc:
            self.message = "Action must be an integer in the current action set."
            self._publish()
            raise GameGuiError(self.message) from exc
        mask = tuple(int(value) for value in self.state.action_mask)
        if normalized < 0 or normalized >= self.ACTION_COUNT or not mask[normalized]:
            self.message = f"Action {normalized} is not legal in the current action mask."
            return self._publish()
        try:
            result = self.service.step(self.handle, normalized)
            self.state = result.state
            self.message = None
            self._record_replay_snapshot()
            self._finish_if_terminal()
            # When ``advance_ai`` is False the caller (the graphical screen)
            # drives the AI cadence one turn at a time via step_single_ai_turn
            # on a QTimer, so the controller must not fast-forward here.
            if self.active and advance_ai:
                self._advance_to_human_turn()
            return self._publish()
        except Exception as exc:
            self.message = self._error_message(exc)
            self._publish()
            raise GameGuiError(self.message) from exc

    def is_ai_turn(self) -> bool:
        """Return ``True`` when it is an AI seat's turn on an active session."""

        return bool(
            self.active
            and not self.paused
            and self.state is not None
            and not (self.state.terminated or self.state.truncated)
            and self.state.current_player != self.HUMAN_PLAYER
        )

    def step_single_ai_turn(self) -> GameControllerSnapshot:
        """Advance exactly ONE AI turn, then publish the new snapshot.

        Unlike :meth:`_advance_to_human_turn`, which loops until the human seat
        is reached, this performs at most a single ``service.step`` so the GUI
        can animate AI turns one at a time on a QTimer.  It is a no-op when it
        is not an AI seat's turn (human active, paused, terminal, or inactive),
        which lets the caller poll it safely.  Kept Qt-free and covered by the
        FakeService tests.
        """

        if not self.is_ai_turn() or self.handle is None:
            return self._publish()
        try:
            result = self.service.step(self.handle)
            self.state = result.state
            self.message = None
            self._record_replay_snapshot()
            self._finish_if_terminal()
            return self._publish()
        except Exception as exc:
            self.message = self._error_message(exc)
            self._publish()
            raise GameGuiError(self.message) from exc

    def time_travel_to(self, moment: Any) -> GameControllerSnapshot:
        """Rewind the board to a recorded ``moment`` (item 8).

        ``moment`` may be a log-row index, an ``environment_step`` int, or an
        event mapping. The recorded per-turn public observations
        (``self._replay_observations``, populated by :meth:`_record_replay_snapshot`)
        are wrapped in a Qt-free :class:`~bang_rl.gui.replay.ReplayModel`, which
        maps the moment to the snapshot representing the game *as of* that
        moment; that historic snapshot is published so the board and log show
        exactly that point in time.

        Behaviour chosen (documented in FEAT-004 findings): a REVIEW-then-continue
        rewind. ``GameService`` sessions are seeded and deterministic, but the
        public observations recorded here do NOT capture the human's individual
        chosen actions unambiguously (a PASS produces no public event, and the
        BANG action index cannot always be recovered from the public event
        projection alone), so a faithful deterministic re-seek is not achievable
        through the current service API. Instead the controller seeks the board
        to the chosen moment as a clearly-labelled read-only preview
        (``previewing`` True, ``active`` False) while stashing the live session
        state; :meth:`resume_live` restores the live session so the human can
        continue playing the LIVE game (not a new line forked from the past
        moment). This keeps all branching logic Qt-free and never reads private
        role state.

        When the live game is already over (terminal, so ``self.active`` is
        False), there is no live turn to resume, so the rewind is a pure
        read-only review: the message uses ``game.time_travel_hint_terminal``
        and leaving the log simply clears the preview.

        Returns the current snapshot; a no-op (unchanged snapshot) when there is
        nothing recorded to rewind to.
        """

        if not self._replay_observations:
            return self._publish()
        try:
            from bang_rl.gui.replay import ReplayModel
        except (ImportError, SyntaxError):
            return self._publish()
        try:
            model = ReplayModel(list(self._replay_observations))
        except (TypeError, ValueError):
            return self._publish()
        if model.is_empty():
            return self._publish()
        snapshot = self._resolve_moment(model, moment)
        if snapshot is None:
            return self._publish()
        # Stash the live session state exactly once (the first rewind) so a
        # later resume restores it even after several previews. Remember whether
        # that stashed state is actually a resumable LIVE turn (active) so the
        # hint does not over-promise "continue" on an already-finished game.
        if not self.previewing:
            self._live_state = self.state
            self._live_active = self.active
            self._live_paused = self.paused
        self.previewing = True
        self.state = snapshot
        # A previewed historic moment is read-only: it is not the live turn.
        self.active = False
        self.paused = False
        step = getattr(snapshot, "environment_step", None)
        self.message = self._time_travel_message(step, resumable=self._live_active)
        return self._publish()

    @staticmethod
    def _resolve_moment(model: Any, moment: Any) -> Any:
        """Map a log-row / step / event ``moment`` to a replay snapshot."""

        if isinstance(moment, Mapping):
            index = model.index_for_event(moment)
            return None if index is None else model.snapshot_at(index)
        # Prefer a log-row interpretation (the GUI passes the clicked row), and
        # fall back to treating the value as an environment_step.
        try:
            row = int(moment)
        except (TypeError, ValueError):
            return model.snapshot_for_moment(moment)
        index = model.index_for_log_row(row)
        if index is not None:
            return model.snapshot_at(index)
        return model.snapshot_for_moment(row)

    @staticmethod
    def _time_travel_message(step: Any, *, resumable: bool = True) -> str:
        from bang_rl.gui.i18n import tr

        try:
            step_int = int(step)
        except (TypeError, ValueError):
            step_int = 0
        # When the live game is still in progress the preview is followed by a
        # return to the LIVE game; when it is already over there is nothing to
        # resume, so use the review-only wording.
        key = "game.time_travel_hint" if resumable else "game.time_travel_hint_terminal"
        return tr(key, step=step_int)

    def resume_live(self) -> GameControllerSnapshot:
        """Leave a time-travel preview and restore the live session (item 8).

        Restores the stashed live state/active/paused flags so the human can
        continue playing from where the game actually is. A no-op when the
        controller is not currently previewing.
        """

        if not self.previewing:
            return self._publish()
        self.previewing = False
        self.state = self._live_state
        self.active = self._live_active
        self.paused = self._live_paused
        self._live_state = None
        self.message = None
        return self._publish()

    def refresh(self) -> GameControllerSnapshot:
        if self.handle is not None and self.active:
            self.state = self.service.get_state(self.handle)
            self._finish_if_terminal()
        return self._publish()

    def pause(self) -> GameControllerSnapshot:
        if self.active:
            self.paused = True
            self.message = "Game paused."
        return self._publish()

    def resume(self) -> GameControllerSnapshot:
        if self.active:
            self.paused = False
            self.message = None
            self._advance_to_human_turn()
        return self._publish()

    def close(self) -> GameControllerSnapshot:
        if self.handle is not None and not self._closed:
            try:
                self.service.close_session(self.handle)
            finally:
                self._closed = True
        self.active = False
        self.paused = False
        return self._publish()

    def _advance_to_human_turn(self) -> None:
        if self.paused or self.handle is None or self.state is None:
            return
        # An application service owns agent turns. The bound prevents a broken
        # mock service from hanging a headless test or the UI thread forever.
        for _ in range(10000):
            if not self.active or self.state.terminated or self.state.truncated:
                self._finish_if_terminal()
                return
            if self.state.current_player == self.HUMAN_PLAYER:
                return
            result = self.service.step(self.handle)
            self.state = result.state
            self._record_replay_snapshot()
            self._finish_if_terminal()
        raise GameGuiError("AI turn did not reach the human seat.")

    def _derive_ai_versions(self, mode: GameMode) -> dict[str, Any]:
        """Map each non-human seat to its AI descriptor for the replay record.

        Random/dummy seats are described by the mode; the Main-AI seat (seat 1
        in HUMAN_MAIN_DUMMY) uses the artifact metadata version when available.
        Seat keys are strings so the record round-trips through JSON unchanged.
        Qt-free and defensive.
        """

        seats = range(1, 5)
        if mode is GameMode.HUMAN_RANDOM:
            return {str(seat): "random" for seat in seats}
        if mode is GameMode.HUMAN_DUMMY:
            return {str(seat): "dummy" for seat in seats}
        if mode is GameMode.HUMAN_MAIN_DUMMY:
            version = None
            if self.ai_metadata:
                version = (
                    self.ai_metadata.get("version")
                    or self.ai_metadata.get("model_version")
                    or self.ai_metadata.get("schema_version")
                )
            main_descriptor: Any = {"kind": "main", "version": version}
            if self.ai_metadata:
                main_descriptor["metadata"] = dict(self.ai_metadata)
            versions: dict[str, Any] = {"1": main_descriptor}
            for seat in range(2, 5):
                versions[str(seat)] = "dummy"
            return versions
        return {str(seat): mode.value for seat in seats}

    def _record_replay_snapshot(self) -> None:
        """Append the current public observation to the replay buffer.

        Stores a deep-copyable copy of ``self.state.observation`` (the public
        projection, already carrying the cumulative ``event_sequence``).
        Defensive: a missing state or observation is skipped, and consecutive
        identical observations are de-duplicated. Never raises. Qt-free.
        """

        state = self.state
        if state is None:
            return
        observation = getattr(state, "observation", None)
        if not isinstance(observation, Mapping):
            return
        try:
            import copy

            snapshot = copy.deepcopy(dict(observation))
        except Exception:
            return
        if self._replay_observations and self._replay_observations[-1] == snapshot:
            return
        self._replay_observations.append(snapshot)

    def _save_replay_if_needed(self) -> None:
        """Best-effort save of the finished human game as a replay JSON.

        Runs exactly once per episode (``_replay_saved`` guard), only when a
        mode is set and at least one observation was recorded. Imports the store
        and paths lazily to keep import-time coupling minimal, and swallows any
        failure so a save error never crashes the game.
        """

        if self._replay_saved or self.mode is None or not self._replay_observations:
            return
        self._replay_saved = True
        state = self.state
        if state is None:
            return
        try:
            from bang_rl.gui import replay_store

            record = replay_store.build_replay_record(
                mode=self.mode.value,
                mode_label=self.mode.label,
                ai_versions=self.mode_ai_versions,
                winning_team=getattr(state, "winning_team", None),
                winners=getattr(state, "winners", ()) or (),
                human_player=self.HUMAN_PLAYER,
                seed=self.seed,
                observations=self._replay_observations,
                terminated=bool(getattr(state, "terminated", False)),
                truncated=bool(getattr(state, "truncated", False)),
            )
            replay_store.save_replay(record)
        except Exception:
            # Never let a replay-save failure crash or interrupt the game.
            return

    def _finish_if_terminal(self) -> None:
        if self.state is None or not (self.state.terminated or self.state.truncated):
            return
        self.active = False
        self.paused = False
        self.message = self._terminal_message()
        self._save_replay_if_needed()
        if self.handle is not None and not self._closed:
            try:
                self.service.close_session(self.handle)
            finally:
                self._closed = True

    def _terminal_message(self) -> str:
        # i18n.tr and roles.faction_name are Qt-free, so importing them in this
        # headless controller keeps game-over text localized without pulling in
        # PyQt. Imported lazily to avoid any import-time coupling.
        from bang_rl.gui.i18n import tr
        from bang_rl.gui.roles import faction_name

        if self.state is not None and self.state.truncated:
            return tr("game.finished_truncated")
        faction = getattr(self.state, "winning_team", None) if self.state is not None else None
        if faction:
            return tr("game.finished_normally_faction", faction=faction_name(faction))
        return tr("game.finished_normally")

    @staticmethod
    def _error_message(exc: Exception) -> str:
        if hasattr(exc, "to_dict"):
            details = exc.to_dict()
            reason = details.get("message") or details.get("code")
            if reason:
                return str(reason)
        text = str(exc)
        if text:
            return text
        # tr is Qt-free, so this keeps the fallback localized without pulling
        # in PyQt from this headless controller. Imported lazily to avoid any
        # import-time coupling.
        from bang_rl.gui.i18n import tr

        return tr("game.error_generic")


__all__ = [
    "GameControllerSnapshot",
    "GameGuiError",
    "GameMode",
    "GameServicePort",
    "GameSessionController",
]
