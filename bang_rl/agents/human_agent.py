"""Queue-backed human agent with pause and interrupt control."""

from __future__ import annotations

import time
from queue import Empty, Queue
from threading import Event
from typing import Any

from .base import AgentInterruptedError, BaseAgent


class HumanAgent(BaseAgent):
    """Read actions from a thread-safe queue without importing GUI code.

    ``act`` blocks until an action is submitted, the agent is resumed after a
    pause, or the wait is interrupted.  A queued action is never consumed while
    paused, which lets a GUI safely queue an intended click before resuming.
    """

    def __init__(
        self,
        agent_id: str | int = "human",
        seed: int | None = None,
        *,
        input_queue: Queue[Any] | None = None,
        poll_interval: float = 0.05,
    ) -> None:
        super().__init__(agent_id=agent_id, seed=seed)
        if input_queue is not None and not hasattr(input_queue, "get"):
            raise TypeError("input_queue must provide a queue get method")
        if isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float)) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.input_queue: Queue[Any] = input_queue if input_queue is not None else Queue()
        self.poll_interval = float(poll_interval)
        self._paused = Event()
        self._interrupted = Event()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def submit_action(self, action: Any) -> None:
        """Queue one raw action for validation at the current action boundary."""

        self.input_queue.put(action)

    put_action = submit_action
    enqueue_action = submit_action

    def pause(self) -> None:
        """Pause consumption; queued actions remain pending."""

        self._paused.set()

    def resume(self) -> None:
        """Resume consumption of queued actions."""

        self._paused.clear()

    def interrupt(self) -> None:
        """Interrupt a currently waiting ``act`` call."""

        self._interrupted.set()

    def clear_interrupt(self) -> None:
        self._interrupted.clear()

    def _check_control_state(self) -> None:
        if self._interrupted.is_set():
            self._interrupted.clear()
            raise AgentInterruptedError(f"human agent {self.agent_id!r} action wait interrupted")

    def act(self, observation: Any, action_mask: Any, *, timeout: float | None = None) -> int:
        mask = self._validate_act_inputs(observation, action_mask)
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0
        ):
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            self._check_control_state()
            if self.paused:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("timed out while HumanAgent was paused")
                time.sleep(self.poll_interval)
                continue
            wait_for = self.poll_interval
            if deadline is not None:
                wait_for = min(wait_for, max(0.0, deadline - time.monotonic()))
                if wait_for == 0.0:
                    raise TimeoutError("timed out waiting for a human action")
            try:
                selected = self.input_queue.get(timeout=wait_for)
            except Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for a human action")
                continue
            # Validate only after dequeue so invalid input is surfaced to the
            # caller and cannot be silently converted to PASS.
            return self._validate_selected_action(selected, mask)

    select_action = act

    def on_episode_start(self, context: Any) -> None:
        super().on_episode_start(context)
        self.clear_interrupt()


__all__ = ["HumanAgent"]
