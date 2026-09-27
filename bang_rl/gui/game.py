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

    def submit_human_action(self, action: int) -> GameControllerSnapshot:
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
            self._finish_if_terminal()
            if self.active:
                self._advance_to_human_turn()
            return self._publish()
        except Exception as exc:
            self.message = self._error_message(exc)
            self._publish()
            raise GameGuiError(self.message) from exc

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
            self._finish_if_terminal()
        raise GameGuiError("AI turn did not reach the human seat.")

    def _finish_if_terminal(self) -> None:
        if self.state is None or not (self.state.terminated or self.state.truncated):
            return
        self.active = False
        self.paused = False
        self.message = "Game finished normally." if self.state.terminated else "Game stopped at the episode limit."
        if self.handle is not None and not self._closed:
            try:
                self.service.close_session(self.handle)
            finally:
                self._closed = True

    @staticmethod
    def _error_message(exc: Exception) -> str:
        if hasattr(exc, "to_dict"):
            details = exc.to_dict()
            reason = details.get("message") or details.get("code")
            if reason:
                return str(reason)
        return str(exc) or type(exc).__name__


__all__ = [
    "GameControllerSnapshot",
    "GameGuiError",
    "GameMode",
    "GameServicePort",
    "GameSessionController",
]
