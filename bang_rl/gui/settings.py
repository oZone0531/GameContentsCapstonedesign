"""Headless settings editor used by the optional Settings GUI.

All edits are staged in a detached Pydantic model.  Validation happens before
an edit can be committed, and a running TrainingJob continues to use its own
immutable snapshot; saving this editor never mutates an active job.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from bang_rl.config import ConfigFileError, ConfigService, EmotionConfig, SystemConfig


RESOURCE_HELP = (
    "Set resources manually. Leaving 1–2 CPU cores available usually keeps the "
    "operating system and GUI responsive. No benchmark, auto-tuning, or OOM search "
    "is performed."
)


class SettingsValidationError(ValueError):
    """Validation error with field-level messages suitable for a form."""

    def __init__(self, errors: tuple[Mapping[str, Any], ...]) -> None:
        self.errors = errors
        message = "; ".join(
            f"{'.'.join(str(part) for part in error.get('loc', ())) or 'config'}: "
            f"{error.get('msg', 'invalid value')}"
            for error in errors
        )
        super().__init__(message or "Invalid settings")


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    """Detached, validated settings state shown in the Settings UI."""

    config: SystemConfig
    source_path: str | None = None
    dirty: bool = False

    @property
    def ppo_clip_epsilon(self) -> float:
        return float(self.config.rl_training.ppo.clip_epsilon)

    @property
    def emotion_initial_baseline(self) -> float:
        return float(self.config.rl_training.emotion_inference.initial_baseline)

    @property
    def hardware(self) -> Mapping[str, Any]:
        return self.config.hardware.model_dump(mode="json")

    @property
    def reward_config(self) -> Mapping[str, Any]:
        return self.config.reward_config.model_dump(mode="json")

    @property
    def dummy_emotion(self) -> Mapping[str, Any]:
        """The dummy AI's full emotion config as a JSON-safe mapping."""

        return self.config.preset.emotion.model_dump(mode="json")

    @property
    def dummy_emotions(self) -> tuple[Mapping[str, Any], ...]:
        """One JSON-safe mapping per configured dummy-AI emotion."""

        return tuple(
            definition.model_dump(mode="json")
            for definition in self.config.preset.emotion.emotions
        )

    @property
    def dummy_emotion_count(self) -> int:
        return int(self.config.preset.emotion.emotion_count)


class SettingsEditor:
    """Stage, validate, save, and commit GUI-editable configuration values."""

    def __init__(
        self,
        config: SystemConfig | Mapping[str, Any] | None = None,
        *,
        config_service: ConfigService | None = None,
        on_update: Callable[[SettingsSnapshot], None] | None = None,
    ) -> None:
        self.config_service = config_service or ConfigService()
        self.on_update = on_update
        self._config = self._validate_config(config or SystemConfig())
        self._source_path: str | None = None
        self._dirty = False

    @property
    def snapshot(self) -> SettingsSnapshot:
        # model_copy prevents a caller from changing the editor through the
        # previously returned Pydantic object.
        return SettingsSnapshot(self._config.model_copy(deep=True), self._source_path, self._dirty)

    def _publish(self) -> SettingsSnapshot:
        state = self.snapshot
        if self.on_update is not None:
            self.on_update(state)
        return state

    @staticmethod
    def _validate_config(value: SystemConfig | Mapping[str, Any]) -> SystemConfig:
        try:
            return value.model_copy(deep=True) if isinstance(value, SystemConfig) else SystemConfig.model_validate(dict(value))
        except ValidationError as exc:
            raise SettingsValidationError(tuple(exc.errors())) from exc

    @staticmethod
    def _error_list(exc: ValidationError) -> tuple[Mapping[str, Any], ...]:
        return tuple(exc.errors())

    def validate(self, values: SystemConfig | Mapping[str, Any]) -> SystemConfig:
        """Validate a complete config or GUI-friendly flattened values."""
        if isinstance(values, SystemConfig):
            return self._validate_config(values)
        payload = self._payload(values)
        try:
            return SystemConfig.model_validate(payload)
        except ValidationError as exc:
            raise SettingsValidationError(self._error_list(exc)) from exc

    def apply(self, values: SystemConfig | Mapping[str, Any]) -> SettingsSnapshot:
        """Validate and stage settings; a running service is not changed."""
        self._config = self.validate(values)
        self._dirty = True
        return self._publish()

    def update(self, **values: Any) -> SettingsSnapshot:
        return self.apply(values)

    def apply_dummy_emotion(self, emotion: Mapping[str, Any]) -> SettingsSnapshot:
        """Stage an edited dummy-AI emotion config into ``config.preset.emotion``.

        The full preset dict is rebuilt from the current draft and only its
        ``emotion`` entry is replaced, so the edit flows through the existing
        ``preset`` key in :meth:`_payload` and is validated by Pydantic exactly
        like every other staged change (unique names, ``initial_value`` in
        ``[0, 1]``, finite rules, ``emotion_count >= 1``).
        """

        preset = self._config.preset.model_dump(mode="python")
        preset["emotion"] = dict(emotion)
        return self.apply({"preset": preset})

    def export_dummy_emotion(self, path: str | Path) -> Path:
        """Write the staged ``config.preset.emotion`` to a standalone file.

        The file contains ONLY the emotion config (``schema_version``,
        ``emotion_count``, ``initial_baseline``, ``emotions[...]``) as YAML or
        JSON (dispatched on the path suffix by :class:`ConfigService`), so it
        can be edited by hand outside the GUI and re-loaded with
        :meth:`import_dummy_emotion`.
        """

        destination = Path(path)
        try:
            self.config_service.save(destination, self._config.preset.emotion, EmotionConfig)
        except ValidationError as exc:
            raise SettingsValidationError(self._error_list(exc)) from exc
        except ConfigFileError as exc:
            raise SettingsValidationError(({"loc": ("emotion",), "msg": str(exc)},)) from exc
        return destination

    def import_dummy_emotion(self, path: str | Path) -> SettingsSnapshot:
        """Load a standalone emotion file and stage it as the dummy AI emotion.

        The document is validated into an :class:`EmotionConfig` and staged
        through :meth:`apply_dummy_emotion`, so Pydantic rules (unique names,
        ``initial_value`` in ``[0, 1]``, finite rules, ``emotion_count >= 1``)
        surface as a readable :class:`SettingsValidationError` rather than a raw
        traceback.
        """

        try:
            loaded = self.config_service.load(path, EmotionConfig)
        except ValidationError as exc:
            raise SettingsValidationError(self._error_list(exc)) from exc
        except ConfigFileError as exc:
            raise SettingsValidationError(({"loc": ("emotion",), "msg": str(exc)},)) from exc
        return self.apply_dummy_emotion(loaded.model_dump(mode="json"))

    def save(self, path: str | Path | None = None) -> SettingsSnapshot:
        destination = Path(path or self._source_path or "config.json")
        try:
            self.config_service.save(destination, self._config, SystemConfig)
        except ValidationError as exc:
            raise SettingsValidationError(self._error_list(exc)) from exc
        self._source_path = str(destination)
        self._dirty = False
        return self._publish()

    def load(self, path: str | Path) -> SettingsSnapshot:
        try:
            loaded = self.config_service.load(path, SystemConfig)
        except ValidationError as exc:
            raise SettingsValidationError(self._error_list(exc)) from exc
        self._config = self._validate_config(loaded)
        self._source_path = str(path)
        self._dirty = False
        return self._publish()

    def _payload(self, values: Mapping[str, Any]) -> dict[str, Any]:
        payload = self._config.model_dump(mode="python")
        source = dict(values)
        if "config" in source and isinstance(source["config"], Mapping):
            source = dict(source["config"], **{key: value for key, value in source.items() if key != "config"})

        # Accept a complete nested object as well as the compact form used by
        # the form widgets.
        for key in ("config_version", "seed", "env", "intent_policy", "emotion", "preset", "sl_training", "evaluation", "pause", "artifact", "dataset", "model_architecture", "optimizer_snapshot"):
            if key in source:
                payload[key] = source[key]
        if "reward_config" in source:
            payload["reward_config"] = source["reward_config"]
        elif "reward" in source:
            payload["reward_config"] = source["reward"]
        if "hardware" in source:
            # Nested GUI edits may contain only one resource field; preserve
            # the other validated values from the current draft.
            hardware = dict(payload.get("hardware", {}))
            hardware.update(dict(source["hardware"]))
            payload["hardware"] = hardware

        rl = dict(payload.get("rl_training", {}))
        ppo = dict(rl.get("ppo", {}))
        emotion_inference = dict(rl.get("emotion_inference", {}))
        if "rl_training" in source:
            rl.update(dict(source["rl_training"]))
            ppo.update(dict(rl.get("ppo", {})))
            emotion_inference.update(dict(rl.get("emotion_inference", {})))
        if "ppo_clip_epsilon" in source:
            ppo["clip_epsilon"] = source["ppo_clip_epsilon"]
        if "clip_epsilon" in source:
            ppo["clip_epsilon"] = source["clip_epsilon"]
        if "emotion_initial_baseline" in source:
            emotion_inference["initial_baseline"] = source["emotion_initial_baseline"]
        if "initial_baseline" in source:
            emotion_inference["initial_baseline"] = source["initial_baseline"]
        rl["ppo"] = ppo
        rl["emotion_inference"] = emotion_inference
        payload["rl_training"] = rl

        hardware = dict(payload.get("hardware", {}))
        for key in ("cpu_cores", "ram_gb", "use_gpu", "workers", "batch_size", "buffer_size"):
            if key in source:
                hardware[key] = source[key]
        payload["hardware"] = hardware
        return payload


__all__ = [
    "RESOURCE_HELP",
    "SettingsEditor",
    "SettingsSnapshot",
    "SettingsValidationError",
]
