"""Configuration file loading and validation services.

The loader is deliberately small and side-effect free: it resolves a YAML or
JSON document into a Pydantic model and reports file/parse/validation failures
with field-oriented messages.  A caller may omit the path to use the first
standard config path found in the current working directory; if none exists,
the schema defaults are used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from ..exceptions import (
    ConfigError,
    ConfigFileError,
    ConfigValidationError,
)
from .schema import SystemConfig

ModelT = TypeVar("ModelT", bound=BaseModel)

DEFAULT_CONFIG_PATHS: tuple[Path, ...] = (
    Path("config/default.yaml"),
    Path("config/default.yml"),
    Path("config/default.json"),
    Path("config.yaml"),
    Path("config.yml"),
    Path("config.json"),
)


def _default_path(default_path: str | Path | None = None) -> Path | None:
    if default_path is not None:
        return Path(default_path)
    return next((candidate for candidate in DEFAULT_CONFIG_PATHS if candidate.is_file()), None)


def _read_document(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigFileError(f"Unable to read configuration file {path}: {exc}") from exc

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(text)
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigFileError(f"Unable to parse configuration file {path}: {exc}") from exc

    raise ConfigFileError(
        f"Unsupported configuration format {path.suffix!r} for {path}; use .yaml, .yml, or .json"
    )


def _validate_document(
    document: Any, schema: type[ModelT], path: Path | None
) -> ModelT:
    if document is None:
        raise ConfigFileError(f"Configuration document {path or '<defaults>'} is empty")
    if not isinstance(document, dict):
        raise ConfigFileError(
            f"Configuration document {path or '<defaults>'} must contain a mapping at its root"
        )
    try:
        return schema.model_validate(document)
    except ValidationError as exc:
        raise ConfigValidationError(path, exc.errors()) from exc


class ConfigLoader:
    """Load and validate YAML/JSON config documents.

    ``default_path`` is only used when ``load()`` receives no path.  Explicit
    paths never silently fall back to defaults, which makes deployment and
    command-line mistakes visible immediately.
    """

    def __init__(self, default_path: str | Path | None = None) -> None:
        self.default_path = Path(default_path) if default_path is not None else None

    def resolve_path(self, path: str | Path | None = None) -> Path | None:
        if path is not None:
            return Path(path)
        return _default_path(self.default_path)

    def load(
        self,
        path: str | Path | None = None,
        schema: type[ModelT] = SystemConfig,
    ) -> ModelT:
        resolved = self.resolve_path(path)
        if resolved is None:
            # Pydantic defaults are the supported built-in configuration.
            return schema.model_validate({})
        if not resolved.is_file():
            raise ConfigFileError(f"Configuration file does not exist: {resolved}")
        return _validate_document(_read_document(resolved), schema, resolved)

    def validate(self, value: Any, schema: type[ModelT] = SystemConfig) -> ModelT:
        """Validate an in-memory mapping using the same errors as file loading."""

        return _validate_document(value, schema, None)

    def save(
        self,
        path: str | Path,
        value: BaseModel | dict[str, Any],
        schema: type[ModelT] = SystemConfig,
    ) -> Path:
        """Validate and persist a model/mapping as YAML or JSON."""

        destination = Path(path)
        if isinstance(value, BaseModel):
            document: Any = value.model_dump(mode="json")
        else:
            document = self.validate(value, schema).model_dump(mode="json")
        if not isinstance(document, dict):
            raise ConfigFileError("Configuration to save must be a mapping")

        suffix = destination.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml"}:
            raise ConfigFileError(
                f"Unsupported configuration format {destination.suffix!r}; use .yaml, .yml, or .json"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if suffix == ".json":
                text = json.dumps(document, indent=2, sort_keys=True) + "\n"
            else:
                text = yaml.safe_dump(document, sort_keys=False)
            destination.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise ConfigFileError(f"Unable to write configuration file {destination}: {exc}") from exc
        return destination


class ConfigService:
    """Application-facing facade matching the Design §14 ConfigService API."""

    def __init__(self, loader: ConfigLoader | None = None) -> None:
        self.loader = loader or ConfigLoader()

    def load(self, path: str | Path | None, schema: type[ModelT] = SystemConfig) -> ModelT:
        return self.loader.load(path, schema)

    def validate(self, value: Any, schema: type[ModelT] = SystemConfig) -> ModelT:
        return self.loader.validate(value, schema)

    def save(
        self,
        path: str | Path,
        value: BaseModel | dict[str, Any],
        schema: type[ModelT] = SystemConfig,
    ) -> Path:
        return self.loader.save(path, value, schema)

    def snapshot(
        self,
        config: SystemConfig | None = None,
        *,
        mode: str = "rl",
        seed: int | None = None,
        split_seed: int | None = None,
        job_id: str | None = None,
    ):
        """Capture the immutable JobSnapshot used by application/GUI seams."""

        from .snapshot import TrainingJob

        return TrainingJob.start(
            config or SystemConfig(),
            mode=mode,
            seed=seed,
            split_seed=split_seed,
            job_id=job_id,
        ).snapshot

def load_config(
    path: str | Path | None = None,
    *,
    schema: type[ModelT] = SystemConfig,
    default_path: str | Path | None = None,
) -> ModelT:
    """Convenience function for loading the default or explicit config path."""

    return ConfigLoader(default_path=default_path).load(path, schema)


def save_config(
    path: str | Path,
    value: BaseModel | dict[str, Any],
    *,
    schema: type[ModelT] = SystemConfig,
) -> Path:
    """Convenience function for validating and persisting configuration."""

    return ConfigLoader().save(path, value, schema)


__all__ = [
    "ConfigError",
    "ConfigFileError",
    "ConfigLoader",
    "ConfigService",
    "ConfigValidationError",
    "DEFAULT_CONFIG_PATHS",
    "load_config",
    "save_config",
]
