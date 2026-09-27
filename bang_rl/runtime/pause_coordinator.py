"""In-memory pause/cancel coordination for vectorized workers."""

from __future__ import annotations

from enum import Enum
from threading import Event, Lock
import time


class PauseState(str, Enum):
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    PAUSE_FAILED = "pause_failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class WorkerControl:
    """A cooperative control gate checked only at safe environment boundaries."""

    def __init__(self) -> None:
        self._pause_requested = Event()
        self._resume = Event()
        self._resume.set()
        self._pause_ack = Event()
        self._cancel_requested = Event()
        self._lock = Lock()
        self._active = False

    @property
    def paused(self) -> bool:
        return self._pause_requested.is_set() and self._pause_ack.is_set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    def begin(self) -> None:
        with self._lock:
            self._active = True
            if not self._pause_requested.is_set():
                self._pause_ack.clear()

    def end(self) -> None:
        with self._lock:
            self._active = False
            if self._pause_requested.is_set():
                self._pause_ack.set()

    def request_pause(self) -> None:
        self._pause_requested.set()
        self._resume.clear()
        with self._lock:
            if not self._active:
                self._pause_ack.set()

    def resume(self) -> None:
        self._pause_requested.clear()
        self._pause_ack.clear()
        self._resume.set()

    def cancel(self) -> None:
        self._cancel_requested.set()
        self._resume.set()

    def safe_point(self, poll_seconds: float = 0.01) -> None:
        """Acknowledge pause and wait without interrupting an active action."""

        if self.cancel_requested:
            raise WorkerCancelled("worker cancellation requested")
        if not self._pause_requested.is_set():
            return
        self._pause_ack.set()
        while self._pause_requested.is_set():
            if self.cancel_requested:
                raise WorkerCancelled("worker cancellation requested")
            self._resume.wait(timeout=poll_seconds)
        self._pause_ack.clear()

    def wait_for_ack(self, timeout: float) -> bool:
        return self._pause_ack.wait(timeout=max(0.0, float(timeout)))


class WorkerCancelled(RuntimeError):
    """Raised at a safe worker boundary after cancellation was requested."""


class PauseCoordinator:
    """Barrier for all workers; pause state exists only in process memory."""

    def __init__(self, controls: list[WorkerControl], *, ack_timeout_seconds: float = 30.0) -> None:
        if ack_timeout_seconds <= 0:
            raise ValueError("ack_timeout_seconds must be positive")
        self.controls = tuple(controls)
        self.ack_timeout_seconds = float(ack_timeout_seconds)
        self.state = PauseState.RUNNING
        self._lock = Lock()

    def pause(self, timeout: float | None = None) -> bool:
        with self._lock:
            if self.state is PauseState.PAUSED:
                return True
            self.state = PauseState.PAUSING
            for control in self.controls:
                control.request_pause()
        deadline = time.monotonic() + (self.ack_timeout_seconds if timeout is None else float(timeout))
        acknowledged = all(
            control.wait_for_ack(max(0.0, deadline - time.monotonic())) for control in self.controls
        )
        with self._lock:
            self.state = PauseState.PAUSED if acknowledged else PauseState.PAUSE_FAILED
        return acknowledged

    def resume(self) -> None:
        with self._lock:
            for control in self.controls:
                control.resume()
            self.state = PauseState.RUNNING

    def cancel(self) -> None:
        with self._lock:
            self.state = PauseState.CANCELLING
            for control in self.controls:
                control.cancel()
            self.state = PauseState.CANCELLED


__all__ = ["PauseCoordinator", "PauseState", "WorkerCancelled", "WorkerControl"]
