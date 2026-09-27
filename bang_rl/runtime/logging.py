"""Structured JSON Lines logging and diagnostic helpers.

The logger keeps public game telemetry and privileged diagnostics on distinct
channels/files.  Every record carries run/job/episode context, while secret-
like fields and common secret assignments in messages are redacted before the
record reaches a stream or file.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable, Mapping
from threading import RLock
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

REDACTED = "[REDACTED]"
_SECRET_KEY = re.compile(
    r"(?:password|passphrase|secret|token|api[_-]?key|authorization|credential|"
    r"private[_-]?key|access[_-]?token|refresh[_-]?token|cookie)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?P<key>password|passphrase|secret|token|api[_-]?key|authorization|credential|"
    r"private[_-]?key|access[_-]?token|refresh[_-]?token)"
    r"(\s*[:=]\s*)(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)
_PRIVATE_PUBLIC_KEY = re.compile(
    r"(?:^|_)(?:[a-z0-9]+_)*(?:role|roles|emotion|emotions|label|labels)$|"
    r"(?:^|_)(?:true_role|actual_role|hidden_role|role_label|emotion_label|"
    r"privileged_state|private_state|state_snapshot|role_specific)(?:_|$)",
    re.IGNORECASE,
)


def new_id(prefix: str) -> str:
    """Create a human-searchable opaque execution identifier."""

    return f"{prefix}-{uuid4().hex}"


class LogChannel(str, Enum):
    EPISODE_EVENTS = "episode_events"
    TERMINATION = "termination"
    REWARD_BREAKDOWN = "reward_breakdown"
    INVALID_ACTION = "invalid_action"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PRIVILEGED_DIAGNOSTICS = "privileged_diagnostics"


LOG_CHANNELS: tuple[str, ...] = tuple(channel.value for channel in LogChannel)


def _redact_text(value: str) -> str:
    return _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group('key')}={REDACTED}", value)


def sanitize_for_log(
    value: Any, *, key: str | None = None, allow_privileged: bool = False
) -> Any:
    """Return JSON-compatible data with secrets and public-private fields masked.

    Public channels drop role/label/private-state keys entirely.  The
    privileged diagnostics channel may retain them, but secret-like keys are
    always redacted regardless of channel.
    """

    if key is not None and _SECRET_KEY.search(key):
        return REDACTED
    if isinstance(value, Mapping):
        result = {}
        for item_key, item_value in value.items():
            name = str(item_key)
            if not allow_privileged and _PRIVATE_PUBLIC_KEY.search(name):
                continue
            result[name] = sanitize_for_log(item_value, key=name, allow_privileged=allow_privileged)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_for_log(item, allow_privileged=allow_privileged) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (datetime, Path)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return sanitize_for_log(value.item(), allow_privileged=allow_privileged)
        except (TypeError, ValueError):
            pass
    return _redact_text(str(value))


class JsonLinesFormatter(logging.Formatter):
    """Serialize each standard LogRecord as one safe JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "structured_fields", {})
        if not isinstance(fields, Mapping):
            fields = {"fields": fields}
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(dict(fields))
        # Core fields are always present and cannot be overridden by user data.
        payload.update(
            {
                "timestamp": payload["timestamp"],
                "level": payload["level"],
                "logger": payload["logger"],
                "message": payload["message"],
            }
        )
        return json.dumps(
            sanitize_for_log(
                payload,
                allow_privileged=payload.get("channel") == LogChannel.PRIVILEGED_DIAGNOSTICS.value,
            ),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


class JsonLinesHandler(logging.Handler):
    """Append-only handler with complete-line writes safe for resumed jobs."""

    _write_lock = RLock()

    def __init__(self, *, stream: TextIO | None = None, path: str | Path | None = None) -> None:
        if stream is not None and path is not None:
            raise ValueError("provide either stream or path, not both")
        super().__init__()
        self._owns_stream = path is not None
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.stream: TextIO = destination.open("a", encoding="utf-8")
        else:
            self.stream = stream or sys.stderr
        self.setFormatter(JsonLinesFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + "\n"
            with self._write_lock:
                self.stream.write(line)
                self.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self.flush()
            if self._owns_stream:
                self.stream.close()
        finally:
            super().close()


class StructuredLogger:
    """Context-bound component logger for one JSON Lines channel."""

    def __init__(
        self,
        channel: str | LogChannel,
        *,
        run_id: str | None = None,
        job_id: str | None = None,
        episode_id: str | None = None,
        worker_id: str | int | None = None,
        environment_step: int | None = None,
        main_agent_transition: int | None = None,
        ppo_update: int | None = None,
        agent_id: str | int | None = None,
        component: str | None = None,
        path: str | Path | None = None,
        stream: TextIO | None = None,
        logger_name: str | None = None,
    ) -> None:
        self.channel = LogChannel(channel).value
        self.run_id = run_id or new_id("run")
        self.job_id = job_id or new_id("job")
        self.episode_id = episode_id or new_id("episode")
        self.worker_id = worker_id
        self.environment_step = environment_step
        self.main_agent_transition = main_agent_transition
        self.ppo_update = ppo_update
        self.agent_id = agent_id
        self.component = component or self.channel
        self._logger = logging.getLogger(logger_name or f"bang_rl.{self.component}.{uuid4().hex}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._handler = JsonLinesHandler(stream=stream, path=path)
        self._logger.addHandler(self._handler)

    @property
    def logger(self) -> logging.Logger:
        """Expose the component logger for integrations needing stdlib APIs."""

        return self._logger

    def _emit(self, level: int, message: str, *, step: int | None = None, **fields: Any) -> None:
        resolved_step = fields.pop("environment_step", self.environment_step if step is None else step)
        context: dict[str, Any] = {
            "channel": self.channel,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "episode_id": self.episode_id,
            "worker_id": fields.pop("worker_id", self.worker_id),
            "step": resolved_step,
            "environment_step": resolved_step,
            "main_agent_transition": fields.pop("main_agent_transition", self.main_agent_transition),
            "ppo_update": fields.pop("ppo_update", self.ppo_update),
            "agent_id": fields.pop("agent_id", self.agent_id),
            "agent": fields.pop("agent", self.agent_id),
            "component": self.component,
        }
        context.update(fields)
        self._logger.log(level, message, extra={"structured_fields": context})

    def log(self, level: int, message: str, *, step: int | None = None, **fields: Any) -> None:
        self._emit(level, message, step=step, **fields)

    def debug(self, message: str, *, step: int | None = None, **fields: Any) -> None:
        self._emit(logging.DEBUG, message, step=step, **fields)

    def info(self, message: str, *, step: int | None = None, **fields: Any) -> None:
        self._emit(logging.INFO, message, step=step, **fields)

    def warning(self, message: str, *, step: int | None = None, **fields: Any) -> None:
        self._emit(logging.WARNING, message, step=step, **fields)

    warn = warning

    def error(self, message: str, *, step: int | None = None, **fields: Any) -> None:
        self._emit(logging.ERROR, message, step=step, **fields)

    def critical(self, message: str, *, step: int | None = None, **fields: Any) -> None:
        self._emit(logging.CRITICAL, message, step=step, **fields)

    def event(self, event_type: str, message: str = "event", *, step: int | None = None, **fields: Any) -> None:
        self._emit(logging.INFO, message, step=step, event_type=event_type, **fields)

    def close(self) -> None:
        self._logger.removeHandler(self._handler)
        self._handler.close()

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class LogManager:
    """Create channel-specific loggers with privileged output in a separate tree."""

    def __init__(self, directory: str | Path, *, run_id: str | None = None, job_id: str | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or new_id("run")
        self.job_id = job_id or new_id("job")
        self._loggers: list[StructuredLogger] = []

    def logger(
        self,
        channel: str | LogChannel,
        *,
        episode_id: str | None = None,
        worker_id: str | int | None = None,
        environment_step: int | None = None,
        main_agent_transition: int | None = None,
        ppo_update: int | None = None,
        agent_id: str | int | None = None,
        component: str | None = None,
    ) -> StructuredLogger:
        resolved = LogChannel(channel)
        if resolved is LogChannel.PRIVILEGED_DIAGNOSTICS:
            path = self.directory / "privileged_diagnostics" / f"{resolved.value}.jsonl"
        else:
            path = self.directory / f"{resolved.value}.jsonl"
        logger = StructuredLogger(
            resolved,
            run_id=self.run_id,
            job_id=self.job_id,
            episode_id=episode_id,
            worker_id=worker_id,
            environment_step=environment_step,
            main_agent_transition=main_agent_transition,
            ppo_update=ppo_update,
            agent_id=agent_id,
            component=component,
            path=path,
        )
        self._loggers.append(logger)
        return logger

    get_logger = logger

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()
        self._loggers.clear()

    def __enter__(self) -> "LogManager":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def read_jsonl(path: str | Path, *, tolerate_trailing_partial: bool = True) -> list[dict[str, Any]]:
    """Read append-only JSONL, ignoring a torn final line from a crash/resume."""

    records: list[dict[str, Any]] = []
    # Read the append-only log once.  In particular, malformed-line handling
    # must not reread the complete file for each record.
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_trailing_partial and line_number == len(lines):
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record {line_number} is not an object")
        records.append(value)
    return records


def get_component_logger(
    component: str,
    *,
    channel: str | LogChannel = LogChannel.TRAINING,
    manager: LogManager | None = None,
    **context: Any,
) -> StructuredLogger:
    """Return a component-scoped logger while preserving channel boundaries."""

    if manager is not None:
        return manager.logger(channel, component=component, **context)
    return StructuredLogger(channel, component=component, **context)


component_logger = get_component_logger


class LongGameDiagnostics:
    """Emit one warning and one critical record per episode/threshold."""

    def __init__(
        self,
        logger: StructuredLogger,
        *,
        warn_turn_threshold: int = 200,
        critical_turn_threshold: int = 500,
        max_episode_steps: int = 2_000,
        recent_event_limit: int = 10,
    ) -> None:
        if not 0 < warn_turn_threshold < critical_turn_threshold < max_episode_steps:
            raise ValueError(
                "thresholds must satisfy 0 < warn_turn_threshold < "
                "critical_turn_threshold < max_episode_steps"
            )
        if recent_event_limit < 0:
            raise ValueError("recent_event_limit must be non-negative")
        self.logger = logger
        self.warn_turn_threshold = warn_turn_threshold
        self.critical_turn_threshold = critical_turn_threshold
        self.max_episode_steps = max_episode_steps
        self.recent_event_limit = recent_event_limit
        self._emitted: set[tuple[str, int]] = set()

    def observe(
        self,
        step: int,
        *,
        episode_id: str | None = None,
        alive_players: Iterable[Any] = (),
        health: Mapping[Any, Any] | Iterable[Any] = (),
        recent_events: Iterable[Any] = (),
        seed: int | None = None,
        roles: Any | None = None,
        state_snapshot: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Observe a state and emit newly crossed thresholds at most once.

        ``recent_events`` is deliberately supplied by the environment as a
        public projection.  ``roles`` and ``state_snapshot`` may contain
        privileged data, so they are only attached when the logger is using
        the physically separate privileged channel.
        """

        resolved_episode = episode_id or self.logger.episode_id
        thresholds = (
            (self.warn_turn_threshold, logging.WARNING, "long_game_warning"),
            (self.critical_turn_threshold, logging.CRITICAL, "long_game_critical"),
        )
        event_summary = list(recent_events)
        for threshold, level, diagnostic_type in thresholds:
            if step <= threshold or (resolved_episode, threshold) in self._emitted:
                continue
            self._emitted.add((resolved_episode, threshold))
            if isinstance(health, Mapping):
                health_summary: Any = dict(health)
            else:
                health_summary = list(health)
            fields: dict[str, Any] = {
                "episode_id": resolved_episode,
                "threshold": threshold,
                "max_episode_steps": self.max_episode_steps,
                "alive_players": list(alive_players),
                "health": health_summary,
                "recent_events": event_summary[-self.recent_event_limit :]
                if self.recent_event_limit
                else [],
                "seed": seed,
                "diagnostic_type": diagnostic_type,
            }
            privileged_channel = self.logger.channel == LogChannel.PRIVILEGED_DIAGNOSTICS.value
            if privileged_channel:
                if roles is not None:
                    fields["roles"] = roles
                if state_snapshot is not None:
                    fields["state_snapshot"] = dict(state_snapshot)
            if metadata:
                fields["diagnostic_metadata"] = dict(metadata)
            self.logger.log(level, diagnostic_type, step=step, **fields)


LongGameDiagnosticLogger = LongGameDiagnostics


__all__ = [
    "LOG_CHANNELS",
    "LogChannel",
    "LogManager",
    "JsonLinesFormatter",
    "JsonLinesHandler",
    "LongGameDiagnosticLogger",
    "LongGameDiagnostics",
    "REDACTED",
    "StructuredLogger",
    "component_logger",
    "get_component_logger",
    "new_id",
    "read_jsonl",
    "sanitize_for_log",
]
