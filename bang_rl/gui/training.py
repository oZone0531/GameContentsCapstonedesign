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

from bang_rl.gui.i18n import tr
from bang_rl.paths import ARTIFACTS_DIR, OUTPUTS_DIR, REPORTS_DIR

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
        self._logs: list[str] = []

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
            if isinstance(rate, (int, float)) and not isinstance(rate, bool):
                self._win_rates.append(float(rate))
            for item in getattr(progress, "logs", ()) or ():
                cleaned = _public_value(item)
                if isinstance(cleaned, Mapping) and not cleaned:
                    continue
                text = str(cleaned.get("message", cleaned) if isinstance(cleaned, Mapping) else cleaned)
                if text and (not self._logs or self._logs[-1] != text):
                    self._logs.append(text)
            status = getattr(progress, "status", None)
            error = getattr(progress, "error", None)
            # Surface a localized failure message whenever the run FAULTED, not
            # only when a structured error payload happens to be present.  A
            # faulted background run always sets ``status == 'faulted'`` (see
            # TrainingService._execute); the error Mapping is the usual detail
            # source, but if it is ever empty we still show a non-empty message
            # so a fault can never leave the monitor silent.
            if message is None and (error or status == "faulted"):
                if isinstance(error, Mapping):
                    detail = str(error.get("message") or error.get("error_type") or error)
                elif error:
                    detail = str(error)
                else:
                    detail = tr("training.run_failed_unknown")
                message = tr("training.run_failed", detail=detail)
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
            self.handle = self.service.start(request)
            self._win_rates.clear()
            self._logs.clear()
            # Seed a visible baseline point so the win-rate graph has a starting
            # point from the start of the run instead of being empty until the
            # first evaluation. Real evaluation win-rates append after this.
            self._win_rates.append(0.0)
            return self._publish(message=None)
        except Exception as exc:
            self.handle = None
            return self._publish(
                message=tr("training.start_failed", detail=self._error_message(exc))
            )

    def refresh(self) -> TrainingMonitorSnapshot:
        if self.handle is None or self.service is None:
            return self._publish(message=tr("training.no_job_running"))
        try:
            return self._publish()
        except Exception as exc:
            return self._publish(
                message=tr("training.unavailable", detail=self._error_message(exc))
            )

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
            return self._publish(message=tr("training.no_job_running"))
        try:
            getattr(self.service, action)(self.handle, **kwargs)
            return self._publish()
        except Exception as exc:
            return self._publish(
                message=tr("training.unavailable", detail=self._error_message(exc))
            )

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
