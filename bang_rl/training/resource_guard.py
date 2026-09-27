"""Explicit user-controlled resource validation for the worker runtime.

This module intentionally does not benchmark, auto-tune, or probe for OOM.  It
only validates the values supplied by ``HardwareConfig`` and records the exact
budget applied to a training job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from bang_rl.config import HardwareConfig, JobSnapshot, SystemConfig
from bang_rl.exceptions import ResourceLimitError


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """The immutable resource settings applied to one worker runtime."""

    cpu_cores: int
    ram_gb: float
    use_gpu: bool
    workers: int
    batch_size: int
    buffer_size: int

    @classmethod
    def from_value(cls, value: HardwareConfig | SystemConfig | JobSnapshot | Mapping[str, Any] | None) -> "ResourceBudget":
        if value is None:
            return cls(cpu_cores=1, ram_gb=1.0, use_gpu=False, workers=1, batch_size=64, buffer_size=256)
        if isinstance(value, HardwareConfig):
            hardware = value
        elif isinstance(value, SystemConfig):
            hardware = value.hardware
        elif isinstance(value, JobSnapshot):
            return cls.from_value(value.config_snapshot.get("hardware", {}))
        elif isinstance(value, Mapping):
            # A full config/snapshot mapping is accepted for runtime callers.
            source = value.get("hardware", value.get("config", value))
            if isinstance(source, Mapping) and "hardware" in source:
                source = source["hardware"]
            hardware = HardwareConfig.model_validate(dict(source))
        else:
            raise TypeError("resource settings must be HardwareConfig, SystemConfig, JobSnapshot, mapping, or None")
        return cls(
            cpu_cores=int(hardware.cpu_cores),
            ram_gb=float(hardware.ram_gb),
            use_gpu=bool(hardware.use_gpu),
            workers=int(hardware.workers),
            batch_size=int(hardware.batch_size),
            buffer_size=int(hardware.buffer_size),
        )

    def validate(self) -> "ResourceBudget":
        if self.cpu_cores < 1:
            raise ResourceLimitError("cpu_cores", self.cpu_cores, ">= 1")
        if self.ram_gb <= 0:
            raise ResourceLimitError("ram_gb", self.ram_gb, "> 0")
        for field in ("workers", "batch_size", "buffer_size"):
            if getattr(self, field) < 1:
                raise ResourceLimitError(field, getattr(self, field), ">= 1")
        return self

    def validate_worker_count(self, actual_workers: int) -> None:
        if isinstance(actual_workers, bool) or not isinstance(actual_workers, int) or actual_workers < 1:
            raise ResourceLimitError("workers", actual_workers, self.workers)
        if actual_workers != self.workers:
            raise ResourceLimitError("workers", actual_workers, self.workers)

    def validate_batch_and_buffer(self, batch_size: int | None, buffer_size: int | None) -> None:
        if batch_size is not None and int(batch_size) != self.batch_size:
            raise ResourceLimitError("batch_size", batch_size, self.batch_size)
        if buffer_size is not None and int(buffer_size) != self.buffer_size:
            raise ResourceLimitError("buffer_size", buffer_size, self.buffer_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "use_gpu": self.use_gpu,
            "workers": self.workers,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
        }


class ResourceGuard:
    """Validate and expose the user-selected resource budget."""

    def __init__(self, value: ResourceBudget | HardwareConfig | SystemConfig | JobSnapshot | Mapping[str, Any] | None = None) -> None:
        self.budget = value if isinstance(value, ResourceBudget) else ResourceBudget.from_value(value)
        self.budget.validate()

    def validate_workers(self, actual_workers: int) -> None:
        self.budget.validate_worker_count(actual_workers)

    def validate_batch_buffer(self, batch_size: int | None, buffer_size: int | None) -> None:
        self.budget.validate_batch_and_buffer(batch_size, buffer_size)

    def snapshot_metadata(self) -> dict[str, Any]:
        return {"hardware": self.budget.to_dict(), "auto_optimization": False, "benchmark_executed": False}


__all__ = ["ResourceBudget", "ResourceGuard"]
