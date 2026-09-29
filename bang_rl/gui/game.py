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
    # Item 6: all-AI spectator modes. The human cannot influence these games;
    # every seat is AI-driven and the human only watches (speed slider + log).
    RANDOM_FIVE = "random_five"
    DUMMY_FIVE = "dummy_five"
    MAIN_ONE_DUMMY_FOUR = "main_one_dummy_four"

    @property
    def label(self) -> str:
        return {
            self.HUMAN_RANDOM: "Human 1 vs Random AI 4",
            self.HUMAN_DUMMY: "Human 1 vs Dummy AI 4",
            self.HUMAN_MAIN_DUMMY: "Human 1 vs Main AI 1 + Dummy AI 3",
            self.RANDOM_FIVE: "Random AI x5",
            self.DUMMY_FIVE: "Dummy AI x5",
            self.MAIN_ONE_DUMMY_FOUR: "Main AI 1 vs Dummy AI 4",
        }[self]

    @property
    def is_spectator(self) -> bool:
        """``True`` for the all-AI modes the human only watches (item 6)."""

        return self in (
            GameMode.RANDOM_FIVE,
            GameMode.DUMMY_FIVE,
            GameMode.MAIN_ONE_DUMMY_FOUR,
        )

    @property
    def is_human(self) -> bool:
        """``True`` for the modes a human actually plays a seat in."""

        return self in (
            GameMode.HUMAN_RANDOM,
            GameMode.HUMAN_DUMMY,
            GameMode.HUMAN_MAIN_DUMMY,
        )


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
        # Item 5: authoritative per-turn EnvironmentState dicts captured from the
        # service (when it exposes them) so a moment can be restored to a live,
        # continuable session. Parallel to _replay_observations (one per turn).
        self._replay_states: list[dict[str, Any]] = []
        self._replay_saved = False
        # Item 5/6: human-played and AI-spectator games persist the FULL
        # reproduction-state replay format; AI-training lines stay public-only.
        self._save_full_replay = False
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
        if (
            selected in (GameMode.HUMAN_MAIN_DUMMY, GameMode.MAIN_ONE_DUMMY_FOUR)
            and not artifact_path
        ):
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
            self._replay_states = []
            self._replay_saved = False
            # Human-played and AI-spectator games both save the FULL replay
            # format (item 5/6); a public-only save is used elsewhere.
            self._save_full_replay = selected.is_human or selected.is_spectator
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
        if self.mode is not None and self.mode.is_spectator:
            # Item 6: in a spectator (all-AI) game the human can never influence
            # play. A submitted action is a no-op that only surfaces a localized
            # notice; the game is NEVER stepped from a human action here.
            from bang_rl.gui.i18n import tr

            self.message = tr("game.spectator_no_actions")
            return self._publish()
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

        if not (
            self.active
            and not self.paused
            and self.state is not None
            and not (self.state.terminated or self.state.truncated)
        ):
            return False
        # Item 6: in a spectator (all-AI) game EVERY seat -- including seat 0 --
        # is AI-driven, so the single-step AI cadence drives all five seats.
        if self.mode is not None and self.mode.is_spectator:
            return True
        return self.state.current_player != self.HUMAN_PLAYER

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

    def restore_to_moment(self, moment: Any) -> GameControllerSnapshot:
        """Fork a live, continuable session from a recorded ``moment`` (item 5).

        Maps ``moment`` (a log-row index, an ``environment_step`` int, or an
        event mapping) to the recorded authoritative :class:`EnvironmentState`
        dict captured in :attr:`_replay_states`, then asks the service to rebuild
        a live session from that state (``restore_session`` /
        ``create_session_from_state`` -- added in FEAT-003). On success the human
        can CONTINUE playing from that point: ``active`` becomes True,
        ``previewing`` False, and the replay buffers are reset to the restored
        prefix so the continued line can be saved as a fresh replay.

        Falls back to the read-only :meth:`time_travel_to` review path when the
        service lacks restore support or no full state was captured, and NEVER
        raises so a hovered/clicked log row is always safe.
        """

        if not self._replay_states or self.service is None:
            return self.time_travel_to(moment)
        restorer = None
        for name in ("restore_session", "create_session_from_state"):
            candidate = getattr(self.service, name, None)
            if callable(candidate):
                restorer = candidate
                break
        if restorer is None:
            return self.time_travel_to(moment)
        index = self._moment_state_index(moment)
        if index is None:
            return self.time_travel_to(moment)
        state_dict = self._replay_states[index]
        if not isinstance(state_dict, Mapping):
            # Defensive: a None/placeholder entry means no authoritative state
            # was captured for that turn, so we must not fork from a wrong or
            # absent state. Fall back to the read-only review.
            return self.time_travel_to(moment)
        try:
            handle = restorer(state_dict)
            new_state = self.service.get_state(handle)
        except Exception:
            return self.time_travel_to(moment)
        # Close the previous live session (best effort) before adopting the fork.
        if self.handle is not None and not self._closed:
            try:
                self.service.close_session(self.handle)
            except Exception:
                pass
        self.handle = handle
        self.state = new_state
        self.active = True
        self.paused = False
        self.previewing = False
        self._closed = False
        self._live_state = None
        # The continued line is a new episode: keep the restored prefix so the
        # saved replay reflects everything up to the fork, and re-arm the save.
        self._replay_observations = self._replay_observations[: index + 1]
        self._replay_states = self._replay_states[: index + 1]
        self._replay_saved = False
        self.message = None
        self._record_replay_snapshot()
        if self.active:
            self._advance_to_human_turn()
        return self._publish()

    def _moment_state_index(self, moment: Any) -> int | None:
        """Map a moment to an index into :attr:`_replay_states` (or ``None``)."""

        if not self._replay_states:
            return None
        try:
            from bang_rl.gui.replay import ReplayModel
        except (ImportError, SyntaxError):
            return None
        try:
            model = ReplayModel(list(self._replay_observations))
        except (TypeError, ValueError):
            return None
        if model.is_empty():
            return None
        if isinstance(moment, Mapping):
            index = model.index_for_event(moment)
        else:
            try:
                row = int(moment)
            except (TypeError, ValueError):
                index = None
            else:
                index = model.index_for_log_row(row)
                if index is None:
                    index = model.index_for_step(row)
        if index is None:
            return None
        # Review fix: only trust an index into _replay_states when the two
        # buffers are still perfectly index-parallel. A single export miss now
        # appends a None placeholder (see _record_full_state), so the lengths
        # stay equal; if they ever diverge, or the selected entry is a None
        # placeholder (no authoritative state for that turn), refuse to fork so
        # restore_to_moment falls back to the read-only review path instead of
        # rewinding from a misaligned/absent state.
        if len(self._replay_states) != len(self._replay_observations):
            return None
        clamped = max(0, min(int(index), len(self._replay_states) - 1))
        if self._replay_states[clamped] is None:
            return None
        return clamped

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
        # Item 6: a spectator game has no human seat to advance to; the screen
        # drives every AI turn one at a time via step_single_ai_turn, so this
        # fast-forward loop must not run (it would never terminate).
        if self.mode is not None and self.mode.is_spectator:
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
        # Item 6 spectator modes: map ALL FIVE seats (seat 0 is AI too).
        if mode is GameMode.RANDOM_FIVE:
            return {str(seat): "random" for seat in range(0, 5)}
        if mode is GameMode.DUMMY_FIVE:
            return {str(seat): "dummy" for seat in range(0, 5)}
        if mode in (GameMode.HUMAN_MAIN_DUMMY, GameMode.MAIN_ONE_DUMMY_FOUR):
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
            # HUMAN_MAIN_DUMMY: human seat 0, Main AI seat 1, dummies 2..4.
            # MAIN_ONE_DUMMY_FOUR (spectator): Main AI seat 0, dummies 1..4.
            main_seat = 0 if mode is GameMode.MAIN_ONE_DUMMY_FOUR else 1
            versions: dict[str, Any] = {str(main_seat): main_descriptor}
            for seat in range(0, 5):
                if seat == main_seat:
                    continue
                if mode is GameMode.HUMAN_MAIN_DUMMY and seat == self.HUMAN_PLAYER:
                    # Seat 0 is the human in HUMAN_MAIN_DUMMY -> not an AI seat.
                    continue
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
        self._record_full_state()

    def _record_full_state(self) -> None:
        """Capture the authoritative EnvironmentState dict for the current turn.

        Item 5: for human / spectator games we persist a FULL reproduction-state
        replay so a moment can later be restored to a live, continuable session.
        The service exposes the current authoritative state via one of a few
        method names (``full_state`` / ``authoritative_state`` / ``export_state``)
        added in FEAT-003.

        INDEX-PARALLEL INVARIANT (review fix): this is called exactly once per
        appended observation in :meth:`_record_replay_snapshot`, so for a
        full-replay game ``_replay_states`` MUST stay one-to-one with
        ``_replay_observations``. If the exporter is missing, raises, or returns
        a non-mapping, we append a ``None`` placeholder rather than nothing, so a
        single transient export miss can never permanently offset every later
        index (which would fork a rewind from the WRONG turn). ``restore_to_moment``
        / :meth:`_moment_state_index` then refuse to fork on a ``None`` entry (or a
        length mismatch) and safely fall back to the read-only review path.

        For a public-only line (``_save_full_replay`` False) this is a no-op and
        ``_replay_states`` stays empty, so restore falls back to review as before.
        Never raises. Qt-free.
        """

        if not self._save_full_replay:
            return
        state_dict: Any = None
        if self.handle is not None and self.service is not None:
            exporter = None
            for name in ("full_state", "authoritative_state", "export_state"):
                candidate = getattr(self.service, name, None)
                if callable(candidate):
                    exporter = candidate
                    break
            if exporter is not None:
                try:
                    exported = exporter(self.handle)
                except Exception:
                    exported = None
                if isinstance(exported, Mapping):
                    try:
                        import copy

                        state_dict = copy.deepcopy(dict(exported))
                    except Exception:
                        state_dict = None
        # Always append (possibly ``None``) to preserve the index-parallel
        # invariant with ``_replay_observations``.
        self._replay_states.append(state_dict)

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

            common = dict(
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
            # Item 5/6: human and spectator games persist the FULL format when
            # authoritative per-turn state was captured; otherwise fall back to
            # the training-safe public format.
            if self._save_full_replay and self._replay_states:
                record = replay_store.build_full_replay_record(
                    states=self._replay_states, **common
                )
            else:
                record = replay_store.build_replay_record(**common)
            # Item 5: encode the game mode in the auto-generated filename via
            # the Qt-free helper. Best-effort/defensive: build_replay_filename
            # sanitizes the mode so a bad value never crashes the save (and
            # falls back to the mode-less name).
            replay_store.save_replay(
                record, filename=replay_store.build_replay_filename(self.mode.value)
            )
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
