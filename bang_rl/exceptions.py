"""Shared, serializable exception hierarchy for Bang RL.

Exceptions are part of the public API.  They deliberately contain only
JSON-safe values so callers can log or transport an error without serializing
an arbitrary exception object or a framework-specific validation error.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


def _json_safe(value: Any) -> Any:
    """Convert common runtime values into deterministic JSON-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    # Do not call an arbitrary object's repr in a log message.  The type name
    # is sufficient context and cannot contain unserializable object state.
    return f"<{type(value).__name__}>"


def _display(value: Any) -> str:
    """Return a compact, stable representation for an exception message."""

    safe = _json_safe(value)
    if isinstance(safe, str):
        return safe
    return json.dumps(safe, sort_keys=True, separators=(",", ":"))


_UNSET = object()


class BangRLError(Exception):
    """Base class for expected Bang RL domain and operational failures."""

    error_code = "bang_rl_error"

    def __init__(
        self,
        message: str = "Bang RL error",
        *,
        code: str | None = None,
        field: str | None = None,
        context: Mapping[str, Any] | None = None,
        current_value: Any = None,
        expected_value: Any = None,
    ) -> None:
        self.message = str(message)
        self.code = code or self.error_code
        self.field = field
        self.current_value = _json_safe(current_value) if current_value is not None else None
        self.expected_value = _json_safe(expected_value) if expected_value is not None else None
        self.context = _json_safe(dict(context or {}))
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible error representation."""

        result: dict[str, Any] = {
            "error_type": type(self).__name__,
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            result["field"] = self.field
        if self.current_value is not None:
            result["current_value"] = self.current_value
        if self.expected_value is not None:
            result["expected_value"] = self.expected_value
        if self.context:
            result["context"] = dict(self.context)
        return result

    as_dict = to_dict

    @property
    def details(self) -> Mapping[str, Any]:
        """Backward-friendly name for structured contextual details."""

        return self.context

    def to_json(self) -> str:
        """Serialize the error without depending on exception pickling."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def __str__(self) -> str:
        return self.message


class DomainError(BangRLError):
    """Base class for errors raised by the game/domain execution layer."""

    error_code = "domain_error"


class EnvironmentError(DomainError):
    """Base class for environment and AEC lifecycle errors."""

    error_code = "environment_error"


class InvalidActionError(EnvironmentError):
    """An action is malformed, out of range, or disallowed by the mask."""

    error_code = "invalid_action"

    def __init__(
        self,
        action: Any,
        reason: str = "action is not allowed",
        *,
        episode_id: str | None = None,
        environment_step: int | None = None,
        agent_index: int | None = None,
        action_mask: Sequence[int] | None = None,
        state_summary: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.action = _json_safe(action)
        self.reason = str(reason)
        self.episode_id = episode_id
        self.environment_step = environment_step
        self.agent_index = agent_index
        self.action_mask = _json_safe(action_mask) if action_mask is not None else None
        self.state_summary = _json_safe(state_summary) if state_summary is not None else None
        details: dict[str, Any] = {
            "action": self.action,
            "reason": self.reason,
        }
        for name, value in (
            ("episode_id", episode_id),
            ("environment_step", environment_step),
            ("agent_index", agent_index),
            ("action_mask", self.action_mask),
            ("state_summary", self.state_summary),
        ):
            if value is not None:
                details[name] = value
        details.update(context or {})
        super().__init__(
            f"Invalid action {_display(self.action)}: {self.reason}",
            context=details,
        )


class TerminalStateActionError(EnvironmentError):
    """An action was submitted after the episode reached a terminal state."""

    error_code = "terminal_state_action"

    def __init__(self, action: Any = None, *, episode_id: str | None = None, context=None) -> None:
        self.action = _json_safe(action)
        self.episode_id = episode_id
        details = {"action": self.action}
        if episode_id is not None:
            details["episode_id"] = episode_id
        details.update(context or {})
        super().__init__(
            f"Cannot perform action {_display(self.action)} after the episode has terminated",
            context=details,
        )


class AgentOrderError(EnvironmentError):
    """An action was submitted by an agent other than the selected agent."""

    error_code = "agent_order"

    def __init__(
        self,
        expected_agent: Any = None,
        actual_agent: Any = None,
        *,
        episode_id: str | None = None,
        context=None,
    ) -> None:
        self.expected_agent = _json_safe(expected_agent)
        self.actual_agent = _json_safe(actual_agent)
        self.episode_id = episode_id
        details = {
            "expected_agent": self.expected_agent,
            "actual_agent": self.actual_agent,
        }
        if episode_id is not None:
            details["episode_id"] = episode_id
        details.update(context or {})
        super().__init__(
            "Agent order violation: "
            f"expected {_display(self.expected_agent)}, got {_display(self.actual_agent)}",
            context=details,
        )


class ConfigError(BangRLError, ValueError):
    """Base class for configuration resolution, loading, and validation errors."""

    error_code = "config_error"


class ConfigLoadError(ConfigError):
    """A configuration source could not be read, parsed, or written."""

    error_code = "config_load_error"


class ConfigFileError(ConfigLoadError):
    """A requested configuration file is missing, unreadable, or malformed."""

    error_code = "config_file_error"


class InvalidConfigError(ConfigError):
    """A configuration value does not satisfy the configured schema."""

    error_code = "invalid_config"

    def __init__(
        self,
        message: str = "Invalid configuration",
        *,
        field: str | None = None,
        value: Any = None,
        expected: Any = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.value = _json_safe(value) if value is not None else None
        self.expected = _json_safe(expected) if expected is not None else None
        super().__init__(
            message,
            field=field,
            current_value=self.value,
            expected_value=self.expected,
            context=context,
        )



class ConfigValidationError(InvalidConfigError):
    """A Pydantic validation failure with stable field-oriented details."""

    error_code = "config_validation_error"

    def __init__(self, path: Path | str | None, errors: Sequence[Mapping[str, Any]]) -> None:
        self.path = str(path) if path is not None else None
        self.errors = tuple(_json_safe(dict(error)) for error in errors)
        location = self.path or "configuration defaults"
        details = "; ".join(_format_validation_error(error) for error in self.errors)
        super().__init__(
            f"Invalid configuration in {location}: {details}",
            field=location if self.path else None,
            context={"path": self.path, "errors": self.errors},
        )


class MissingConfigKeyError(InvalidConfigError):
    """A required configuration key is absent."""

    error_code = "missing_config_key"

    def __init__(self, key: str | Sequence[str], *, context=None) -> None:
        self.key = list(key) if isinstance(key, (list, tuple, set, frozenset)) else str(key)
        super().__init__(
            f"Missing required configuration key: {_display(self.key)}",
            field=str(self.key) if isinstance(self.key, str) else None,
            context={"key": self.key, **(context or {})},
        )


class SchemaChangedError(ConfigError):
    """A schema version or structure changed incompatibly."""

    error_code = "schema_changed"

    def __init__(
        self,
        schema: str,
        current_version: Any,
        required_version: Any,
        *,
        context=None,
    ) -> None:
        self.schema = str(schema)
        self.current_version = _json_safe(current_version)
        self.required_version = _json_safe(required_version)
        super().__init__(
            f"Schema {self.schema!r} changed: current version "
            f"{_display(self.current_version)} does not match required version "
            f"{_display(self.required_version)}",
            field=self.schema,
            current_value=self.current_version,
            expected_value=self.required_version,
            context={"schema": self.schema, **(context or {})},
        )


class CompatibilityError(BangRLError):
    """A persisted artifact is incompatible with the running system."""

    error_code = "compatibility_error"

    def __init__(
        self,
        field: str,
        current_value: Any = None,
        expected_value: Any = None,
        *,
        current: Any = _UNSET,
        required: Any = _UNSET,
        artifact: str | None = None,
        context=None,
    ) -> None:
        if current is not _UNSET:
            current_value = current
        if required is not _UNSET:
            expected_value = required
        self.artifact = artifact
        super().__init__(
            f"Incompatible {field}: current {_display(current_value)}, "
            f"required {_display(expected_value)}",
            field=field,
            current_value=current_value,
            expected_value=expected_value,
            context={"artifact": artifact, **(context or {})} if artifact is not None else context,
        )


class ArtifactCompatibilityError(CompatibilityError):
    """Compatibility failure specifically associated with an artifact bundle."""

    error_code = "artifact_compatibility_error"


class DataError(BangRLError):
    """Base class for dataset and artifact data read/write failures."""

    error_code = "data_error"


class TrainingError(BangRLError):
    """Base class for training and execution-pipeline failures."""

    error_code = "training_error"


class ResourceLimitError(TrainingError):
    """A configured resource limit prevents an operation from proceeding."""

    error_code = "resource_limit"

    def __init__(
        self,
        resource: str,
        requested: Any,
        limit: Any,
        *,
        context=None,
    ) -> None:
        self.resource = str(resource)
        self.requested = _json_safe(requested)
        self.limit = _json_safe(limit)
        super().__init__(
            f"Resource limit exceeded for {self.resource}: requested "
            f"{_display(self.requested)}, limit {_display(self.limit)}",
            field=self.resource,
            current_value=self.requested,
            expected_value=self.limit,
            context=context,
        )


def _format_validation_error(error: Mapping[str, Any]) -> str:
    loc = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
    message = str(error.get("msg", "invalid value"))
    return f"{loc}: {message}"


# Compatibility aliases for callers that use the more explicit domain names.
BaseError = BangRLError
EnvironmentDomainError = EnvironmentError


__all__ = [
    "AgentOrderError",
    "ArtifactCompatibilityError",
    "BangRLError",
    "BaseError",
    "CompatibilityError",
    "ConfigError",
    "ConfigFileError",
    "ConfigLoadError",
    "ConfigValidationError",
    "DataError",
    "DomainError",
    "EnvironmentError",
    "EnvironmentDomainError",
    "InvalidActionError",
    "InvalidConfigError",
    "MissingConfigKeyError",
    "ResourceLimitError",
    "SchemaChangedError",
    "TerminalStateActionError",
    "TrainingError",
]
