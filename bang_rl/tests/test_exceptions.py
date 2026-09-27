"""Focused tests for TASK-0-6's public exception contracts."""

import json

import pytest

import bang_rl
from bang_rl.config import (
    ConfigError,
    ConfigFileError,
    ConfigLoadError,
    ConfigLoader,
    ConfigValidationError,
)
from bang_rl.exceptions import (
    AgentOrderError,
    ArtifactCompatibilityError,
    BangRLError,
    CompatibilityError,
    DataError,
    DomainError,
    EnvironmentError,
    InvalidActionError,
    InvalidConfigError,
    MissingConfigKeyError,
    ResourceLimitError,
    SchemaChangedError,
    TerminalStateActionError,
    TrainingError,
)


def test_domain_exception_hierarchy_and_invalid_action_context() -> None:
    error = InvalidActionError(
        7,
        "out of range",
        episode_id="episode-1",
        environment_step=12,
        agent_index=3,
        action_mask=[1, 0, 0, 0, 0, 0],
        state_summary={"alive": [True, False]},
    )

    assert isinstance(error, BangRLError)
    assert isinstance(error, DomainError)
    assert isinstance(error, EnvironmentError)
    assert error.action == 7
    assert error.reason == "out of range"
    assert error.episode_id == "episode-1"
    assert error.environment_step == 12
    assert error.agent_index == 3
    assert error.action_mask == [1, 0, 0, 0, 0, 0]
    assert "7" in str(error)
    assert "out of range" in str(error)


def test_aec_errors_are_distinct_from_invalid_action() -> None:
    assert isinstance(TerminalStateActionError(0), EnvironmentError)
    assert isinstance(AgentOrderError(expected_agent=1, actual_agent=2), EnvironmentError)
    assert not isinstance(TerminalStateActionError(0), InvalidActionError)
    assert not isinstance(AgentOrderError(1, 2), InvalidActionError)


def test_config_loader_exceptions_keep_legacy_identity_and_new_hierarchy(tmp_path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigFileError) as captured:
        ConfigLoader().load(missing)

    error = captured.value
    assert isinstance(error, ConfigLoadError)
    assert isinstance(error, ConfigError)
    assert "does not exist" in str(error)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "env:\n  warn_turn_threshold: 100\n  critical_turn_threshold: 100\n  max_episode_steps: 200\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError) as captured:
        ConfigLoader().load(invalid)
    assert isinstance(captured.value, InvalidConfigError)
    assert "env:" in str(captured.value)
    assert captured.value.path == str(invalid)
    assert captured.value.errors


def test_config_schema_and_compatibility_errors_expose_validation_context() -> None:
    missing = MissingConfigKeyError("emotion.schema_version")
    assert isinstance(missing, InvalidConfigError)
    assert missing.key == "emotion.schema_version"
    assert missing.context["key"] == "emotion.schema_version"

    changed = SchemaChangedError("emotion", "1.0", "2.0")
    assert changed.field == "emotion"
    assert changed.current_version == "1.0"
    assert changed.required_version == "2.0"
    assert "1.0" in str(changed) and "2.0" in str(changed)

    compatible = ArtifactCompatibilityError(
        "emotion_schema_version", "1.0", "2.0", artifact="model.bangmodel"
    )
    assert isinstance(compatible, CompatibilityError)
    assert compatible.artifact == "model.bangmodel"
    assert compatible.field == "emotion_schema_version"
    assert compatible.current_value == "1.0"
    assert compatible.expected_value == "2.0"


def test_all_exception_messages_and_context_are_json_serializable() -> None:
    errors = [
        InvalidActionError(object(), context={"payload": {"mask": (1, 0)}}),
        ConfigError("config failed", context={"path": object()}),
        DataError("dataset failed", context={"path": object()}),
        TrainingError("training failed", context={"job": object()}),
        ResourceLimitError("ram_gb", 16, 8),
    ]

    for error in errors:
        encoded = error.to_json()
        decoded = json.loads(encoded)
        assert decoded["message"] == str(error)
        assert decoded["error_type"] == type(error).__name__
        assert json.dumps(error.to_dict())


def test_exceptions_are_public_from_package_and_config_exports() -> None:
    for name in (
        "BangRLError",
        "InvalidActionError",
        "TerminalStateActionError",
        "AgentOrderError",
        "CompatibilityError",
        "ResourceLimitError",
    ):
        assert getattr(bang_rl, name) is not None

    assert bang_rl.ConfigFileError is ConfigFileError
    assert bang_rl.ConfigValidationError is ConfigValidationError
    assert ConfigFileError.__module__ == "bang_rl.exceptions"
    assert ConfigValidationError.__module__ == "bang_rl.exceptions"
