"""Headless training setup and monitor controllers.

The controllers in this module deliberately depend on the application service
DTO boundary only.  They can be exercised without Qt (or a display server),
while :mod:`bang_rl.gui.screens.training_monitor` supplies the optional Qt
presentation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

from bang_rl.paths import ARTIFACTS_DIR, OUTPUTS_DIR, REPORTS_DIR, REPLAYS_DIR

if TYPE_CHECKING:
    from bang_rl.app import TrainingConfig, TrainingHandle, TrainingProgress
    from bang_rl.config import JobSnapshot, SystemConfig
else:
    TrainingConfig = Any
    TrainingHandle = Any
    TrainingProgress = Any
    JobSnapshot = Any
    SystemConfig = Any


_PRIVATE_KEYS = frozenset(
    {
        "roles",
        "true_roles",
        "actual_roles",
        "hidden_roles",
        "private_roles",
        "role_labels",
        "emotion_labels",
        "actual_emotion",
        "actual_emotions",
        "private_emotion",
        "privileged_diagnostics",
        "reward_breakdown",
    }
)


def _public_value(value: Any, key: str | None = None) -> Any:
    """Detach public display data and exclude privileged training fields."""

    normalized = key.lower() if key else ""
    if normalized in _PRIVATE_KEYS:
        return None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for item_key, item_value in value.items():
            cleaned = _public_value(item_value, str(item_key))
            if cleaned is not None:
                result[str(item_key)] = cleaned
        return result
    if isinstance(value, (list, tuple)):
        return tuple(
            cleaned
            for cleaned in (_public_value(item) for item in value)
            if cleaned is not None
        )
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def public_training_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a detached, non-privileged mapping for the monitor."""

    cleaned = _public_value(value or {})
    return dict(cleaned) if isinstance(cleaned, Mapping) else {}


class TrainingServicePort(Protocol):
    def start(self, request: TrainingConfig) -> TrainingHandle: ...

    def progress(self, handle: TrainingHandle | str) -> TrainingProgress: ...

    def pause(self, handle: TrainingHandle | str, timeout: float | None = None) -> None: ...

    def resume(self, handle: TrainingHandle | str) -> None: ...

    def cancel(self, handle: TrainingHandle | str) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainingMonitorSnapshot:
    """Immutable public view rendered by the monitor."""

    handle: TrainingHandle | None
    progress: TrainingProgress | None
    snapshot: JobSnapshot | None
    win_rate_points: tuple[float, ...] = ()
    logs: tuple[str, ...] = ()
    message: str | None = None
    public_metadata: Mapping[str, Any] = field(default_factory=dict)
    role_accuracy_points: tuple[float, ...] = ()
    emotion_score_points: tuple[float, ...] = ()

    @property
    def active(self) -> bool:
        return self.progress is not None and self.progress.status in {
            "running",
            "paused",
            "pause_failed",
            "cancelling",
        }

    @property
    def paused(self) -> bool:
        return self.progress is not None and self.progress.status == "paused"


@dataclass(frozen=True, slots=True)
class TrainingSetup:
    """Validated inputs for starting one training job."""

    config: SystemConfig | Mapping[str, Any] | None = None
    mode: str = "rl"
    episodes: int = 1
    target_transitions: int | None = None
    seed: int | None = None
    split_seed: int | None = None
    report_path: str | Path | None = REPORTS_DIR
    output_path: str | Path | None = OUTPUTS_DIR
    artifact_path: str | Path | None = ARTIFACTS_DIR
    resume_artifact_path: str | Path | None = ARTIFACTS_DIR
    replay_source_path: str | Path | None = REPLAYS_DIR
    phase2_artifact_path: str | Path | None = None
    sl_epochs: int = 10
    ppo_update_mode: str = "batch"
    ppo_batch_size: int = 64

    def to_request(self) -> TrainingConfig:
        from bang_rl.app import TrainingConfig as ConcreteTrainingConfig

        return ConcreteTrainingConfig(
            config=self.config,
            mode=self.mode,
            episodes=self.episodes,
            target_transitions=self.target_transitions,
            seed=self.seed,
            split_seed=self.split_seed,
            report_path=self.report_path,
            output_path=self.output_path,
            artifact_path=self.artifact_path,
            resume_artifact_path=self.resume_artifact_path,
            replay_source_path=self.replay_source_path,
            phase2_artifact_path=self.phase2_artifact_path,
            sl_epochs=self.sl_epochs,
            ppo_update_mode=self.ppo_update_mode,
            ppo_batch_size=self.ppo_batch_size,
        )


class TrainingMonitorController:
    """Wire TrainingService lifecycle methods to a display-safe snapshot."""

    def __init__(
        self,
        service: TrainingServicePort | None = None,
        *,
        on_update: Callable[[TrainingMonitorSnapshot], None] | None = None,
    ) -> None:
        self.service = service
        self.on_update = on_update
        self.handle: TrainingHandle | None = None
        self._snapshot = TrainingMonitorSnapshot(None, None, None)
        self._win_rates: list[float] = []
        self._last_rate_key: tuple[Any, ...] | None = None
        self._logs: list[str] = []
        self._log_cursor = 0

    @property
    def snapshot(self) -> TrainingMonitorSnapshot:
        return self._snapshot

    def _publish(
        self,
        progress: TrainingProgress | None = None,
        *,
        message: str | None = None,
    ) -> TrainingMonitorSnapshot:
        if progress is None and self.handle is not None and self.service is not None:
            progress = self.service.progress(self.handle)
        if progress is not None:
            rate = getattr(progress, "win_rate", None)
            rate_key = (
                getattr(progress, "episodes_completed", 0),
                getattr(progress, "environment_steps", 0),
                getattr(progress, "main_agent_transitions", 0),
                getattr(progress, "ppo_updates", 0),
                getattr(progress, "status", None),
                rate,
            )
            if (
                isinstance(rate, (int, float))
                and not isinstance(rate, bool)
                and rate_key != self._last_rate_key
            ):
                self._win_rates.append(float(rate))
            self._last_rate_key = rate_key

            source_logs = tuple(getattr(progress, "logs", ()) or ())
            if self._log_cursor > len(source_logs):
                # A service implementation may replace its log buffer when a
                # job is restarted. Treat that as a new stream.
                self._log_cursor = 0
            for item in source_logs[self._log_cursor:]:
                cleaned = _public_value(item)
                if isinstance(cleaned, Mapping) and not cleaned:
                    continue
                text = str(cleaned.get("message", cleaned) if isinstance(cleaned, Mapping) else cleaned)
                if text:
                    self._logs.append(text)
            self._log_cursor = len(source_logs)
            if message is None and getattr(progress, "error", None):
                error = progress.error
                if isinstance(error, Mapping):
                    message = str(error.get("message") or error.get("error_type") or error)
                else:
                    message = str(error)
        role_accuracy_points: tuple[float, ...] = ()
        emotion_score_points: tuple[float, ...] = ()
        if progress is not None:
            metrics = _public_value(getattr(progress, "metrics", {}) or {})
            if isinstance(metrics, Mapping) and metrics.get("phase") == "phase2_supervised":
                intent_history = metrics.get("intent_history", ())
                emotion_history = metrics.get("emotion_history", ())
                if isinstance(intent_history, (list, tuple)):
                    role_accuracy_points = tuple(
                        float(item["validation_accuracy"])
                        for item in intent_history
                        if isinstance(item, Mapping) and item.get("validation_accuracy") is not None
                    )
                if isinstance(emotion_history, (list, tuple)):
                    # Emotion is a regression task, so graph 1 - validation MSE
                    # as a normalized higher-is-better score beside role accuracy.
                    emotion_score_points = tuple(
                        max(0.0, 1.0 - float(item["validation_mse"]))
                        for item in emotion_history
                        if isinstance(item, Mapping) and item.get("validation_mse") is not None
                    )
        handle = self.handle
        snapshot = None if handle is None else getattr(handle, "snapshot", None)
        metadata = {}
        if snapshot is not None:
            metadata = public_training_metadata(
                {
                    "job_id": getattr(snapshot, "job_id", None),
                    "mode": getattr(snapshot, "mode", None),
                    "config_version": getattr(snapshot, "config_version", None),
                    "seed": getattr(snapshot, "seed", None),
                    "split_seed": getattr(snapshot, "split_seed", None),
                    "versions": getattr(snapshot, "versions", {}),
                }
            )
        self._snapshot = TrainingMonitorSnapshot(
            handle,
            progress,
            snapshot,
            tuple(self._win_rates),
            tuple(self._logs),
            message,
            metadata,
            role_accuracy_points,
            emotion_score_points,
        )
        if self.on_update is not None:
            self.on_update(self._snapshot)
        return self._snapshot

    def start(self, setup: TrainingSetup | TrainingConfig | Mapping[str, Any]) -> TrainingMonitorSnapshot:
        if self.service is None:
            from bang_rl.app import TrainingService

            self.service = TrainingService()
        if isinstance(setup, TrainingSetup):
            request = setup.to_request()
        elif isinstance(setup, Mapping):
            from bang_rl.app import TrainingConfig as ConcreteTrainingConfig

            request = ConcreteTrainingConfig(**dict(setup))
        else:
            request = setup
        try:
            self._win_rates.clear()
            self._last_rate_key = None
            self._logs.clear()
            self._log_cursor = 0
            self.handle = self.service.start(request)
            return self._publish(message=None)
        except Exception as exc:
            self.handle = None
            return self._publish(message=self._error_message(exc))

    def refresh(self) -> TrainingMonitorSnapshot:
        if self.handle is None or self.service is None:
            return self._publish(message="No training job is running.")
        try:
            return self._publish()
        except Exception as exc:
            return self._publish(message=self._error_message(exc))

    def pause(self, timeout: float | None = None) -> TrainingMonitorSnapshot:
        return self._lifecycle("pause", timeout=timeout)

    def resume(self) -> TrainingMonitorSnapshot:
        return self._lifecycle("resume")

    def cancel(self) -> TrainingMonitorSnapshot:
        # This is an in-process stop request, never a persisted resume path.
        return self._lifecycle("cancel")

    stop = cancel

    def _lifecycle(self, action: str, **kwargs: Any) -> TrainingMonitorSnapshot:
        if self.handle is None or self.service is None:
            return self._publish(message="No training job is running.")
        try:
            getattr(self.service, action)(self.handle, **kwargs)
            return self._publish()
        except Exception as exc:
            return self._publish(message=self._error_message(exc))

    @staticmethod
    def _error_message(exc: Exception) -> str:
        if hasattr(exc, "to_dict"):
            details = exc.to_dict()
            return str(details.get("message") or details.get("code") or exc)
        return str(exc) or type(exc).__name__


__all__ = [
    "TrainingMonitorController",
    "TrainingMonitorSnapshot",
    "TrainingServicePort",
    "TrainingSetup",
    "public_training_metadata",
]
