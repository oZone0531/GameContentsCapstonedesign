"""Phase-2 supervised training for the public IntentInference module.

The trainer keeps labels on the training side of the boundary: only public
``event_history`` and ``event_valid_mask`` tensors are passed to the model.
Role labels are selected for the four non-viewer streams and are used solely as
CrossEntropy targets.  Cache manifests and the immutable JobSnapshot are
carried through to checkpoints, metrics, and the phase-2 inference artifact so
that a result can be checked before it is loaded elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from bang_rl.agents.main_agent import IntentInferenceModule
from bang_rl.config.snapshot import JobSnapshot
from bang_rl.data.tensor_cache import (
    CACHE_FORMAT_VERSION,
    CACHE_SCHEMA_VERSION,
    NPZTensorCacheLoader,
)
from bang_rl.exceptions import ArtifactCompatibilityError, DataError, TrainingError
from bang_rl.runtime.metadata import build_run_metadata


SL_ARTIFACT_FORMAT_VERSION = "1.0"
SL_CHECKPOINT_FORMAT_VERSION = "1.0"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise TrainingError(f"failed to write training artifact {path}") from exc


def _clone_state_dict(model: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all(value["cuda"])


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


@dataclass(frozen=True)
class IntentBatch:
    """One cache batch with labels kept outside the public model input mapping."""

    model_inputs: Mapping[str, Tensor]
    role_labels: Tensor
    role_label_mask: Tensor
    viewer_indices: Tensor
    row_indices: tuple[int, ...]

    @property
    def inputs(self) -> Mapping[str, Tensor]:
        return self.model_inputs

    @property
    def valid_mask(self) -> Tensor:
        return self.model_inputs["event_valid_mask"]

    @property
    def labels(self) -> Tensor:
        return self.role_labels

    def __getitem__(self, key: str) -> Any:
        if key in self.model_inputs:
            return self.model_inputs[key]
        if key in {"inputs", "model_inputs"}:
            return self.model_inputs
        if key in {"role_labels", "labels"}:
            return self.role_labels
        if key == "role_label_mask":
            return self.role_label_mask
        if key == "viewer_indices":
            return self.viewer_indices
        raise KeyError(key)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.model_inputs) + ("role_labels", "role_label_mask", "viewer_indices")


def _as_loader(cache: NPZTensorCacheLoader | str | os.PathLike[str]) -> NPZTensorCacheLoader:
    if isinstance(cache, NPZTensorCacheLoader):
        loader = cache
    else:
        loader = NPZTensorCacheLoader(cache)
    loader.validate()
    return loader


def _viewer_indices(state_vector: np.ndarray) -> np.ndarray:
    values = np.asarray(state_vector, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 9:
        raise DataError("intent cache state_vector must expose the public viewer index")
    scaled = values[:, 8] * 4.0
    viewers = np.rint(scaled).astype(np.int64)
    if not np.allclose(scaled, viewers, atol=1e-5) or np.any((viewers < 0) | (viewers >= 5)):
        raise DataError("intent cache contains an invalid public viewer index")
    return viewers


def build_intent_batch(
    cache: NPZTensorCacheLoader | str | os.PathLike[str],
    indices: Sequence[int] | np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> IntentBatch:
    """Build a deterministic batch from NPZ tensors.

    The returned ``model_inputs`` contains only public event streams and their
    valid masks.  The four target labels are selected from the all-player role
    target using the public viewer index encoded in ``state_vector``.
    """

    loader = _as_loader(cache)
    row_indices = tuple(int(item) for item in np.asarray(indices, dtype=np.int64).reshape(-1))
    if any(item < 0 or item >= len(loader) for item in row_indices):
        raise IndexError("intent batch row index is outside the cache")
    tensors = loader.model_inputs(np.asarray(row_indices, dtype=np.int64))
    labels = loader.labels(np.asarray(row_indices, dtype=np.int64))
    viewers = _viewer_indices(tensors["state_vector"])
    role_values = np.asarray(labels["role_labels"], dtype=np.int64)
    role_mask = np.asarray(labels["role_label_mask"], dtype=np.bool_)
    selected_labels = np.zeros((len(row_indices), 4), dtype=np.int64)
    selected_mask = np.zeros((len(row_indices), 4), dtype=np.bool_)
    for row, viewer in enumerate(viewers):
        targets = [player for player in range(5) if player != int(viewer)]
        selected_labels[row] = role_values[row, targets]
        selected_mask[row] = role_mask[row, targets]
    if np.any(selected_mask & ((selected_labels < 0) | (selected_labels >= 5))):
        raise DataError("role target contains an invalid role index")

    public_inputs = {
        "event_history": torch.as_tensor(tensors["event_history"], dtype=torch.float32, device=device),
        "event_valid_mask": torch.as_tensor(
            tensors["event_valid_mask"], dtype=torch.bool, device=device
        ),
    }
    return IntentBatch(
        model_inputs=public_inputs,
        role_labels=torch.as_tensor(selected_labels, dtype=torch.long, device=device),
        role_label_mask=torch.as_tensor(selected_mask, dtype=torch.bool, device=device),
        viewer_indices=torch.as_tensor(viewers, dtype=torch.long, device=device),
        row_indices=row_indices,
    )


@dataclass(frozen=True)
class IntentSLConfig:
    """Runtime settings for deterministic phase-2 intent training."""

    epochs: int = 1
    batch_size: int = 64
    seed: int = 0
    early_stopping_patience: int | None = None
    min_delta: float = 0.0
    checkpoint_path: str | Path | None = None
    metrics_path: str | Path | None = None
    device: str = "cpu"
    deterministic: bool = True
    resume_from: str | Path | None = None

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool) or self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be positive or None")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")

    @classmethod
    def from_value(cls, value: "IntentSLConfig | Mapping[str, Any] | None") -> "IntentSLConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("training config must be IntentSLConfig or a mapping")
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError(f"unknown IntentSLConfig fields: {', '.join(unknown)}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "early_stopping_patience": self.early_stopping_patience,
            "min_delta": self.min_delta,
            "device": self.device,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True)
class TrainingResult:
    model: IntentInferenceModule
    history: tuple[Mapping[str, Any], ...]
    best_epoch: int
    stopped_early: bool
    test_metrics: Mapping[str, Any] | None
    artifact_metadata: Mapping[str, Any]

    @property
    def metrics(self) -> tuple[Mapping[str, Any], ...]:
        return self.history


class IntentInferenceSLTrainer:
    """Train IntentInference from normal-only NPZ cache splits."""

    phase = "phase2_supervised_intent"

    def __init__(
        self,
        train_cache: NPZTensorCacheLoader | str | os.PathLike[str],
        validation_cache: NPZTensorCacheLoader | str | os.PathLike[str],
        *,
        test_cache: NPZTensorCacheLoader | str | os.PathLike[str] | None = None,
        config: IntentSLConfig | Mapping[str, Any] | None = None,
        model: IntentInferenceModule | None = None,
        job_snapshot: JobSnapshot | Mapping[str, Any] | None = None,
        config_snapshot: Mapping[str, Any] | Any | None = None,
        config_hash: str | None = None,
        resume_from: str | os.PathLike[str] | None = None,
        logger: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.config = IntentSLConfig.from_value(config)
        self.train_cache = _as_loader(train_cache)
        self.validation_cache = _as_loader(validation_cache)
        self.test_cache = None if test_cache is None else _as_loader(test_cache)
        self._validate_splits()
        self.job_snapshot = self._snapshot_dict(job_snapshot)
        self.config_snapshot = self._config_dict(config_snapshot)
        self.config_hash = config_hash or _hash(
            self.job_snapshot if self.job_snapshot is not None else self.config_snapshot or self.config.to_dict()
        )
        self.logger = logger
        self.device = torch.device(self.config.device)
        self._seed_everything(self.config.seed, deterministic=self.config.deterministic)
        self.model = model or IntentInferenceModule(seed=self.config.seed)
        self.model.to(self.device)
        self.optimizer = self.model.create_optimizer()
        self.history: list[dict[str, Any]] = []
        self.start_epoch = 1
        self.best_epoch = 0
        self.best_validation_loss = float("inf")
        self.best_state = _clone_state_dict(self.model)
        self._bad_epochs = 0
        self._stopped_early = False
        self._resume_path: Path | None = None

        resume_from = resume_from or self.config.resume_from
        if resume_from is not None:
            self.load_checkpoint(resume_from)

    @staticmethod
    def _seed_everything(seed: int, *, deterministic: bool) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def _snapshot_dict(value: JobSnapshot | Mapping[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, JobSnapshot):
            return value.to_dict()
        if isinstance(value, Mapping):
            return _json_safe(dict(value))
        raise TypeError("job_snapshot must be a JobSnapshot or mapping")

    @staticmethod
    def _config_dict(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return _json_safe(value.model_dump(mode="json"))
        if hasattr(value, "to_dict"):
            return _json_safe(value.to_dict())
        if isinstance(value, Mapping):
            return _json_safe(dict(value))
        raise TypeError("config_snapshot must be a mapping or serializable config")

    def _validate_splits(self) -> None:
        for expected, loader in (("train", self.train_cache), ("validation", self.validation_cache)):
            actual = loader.manifest.get("split")
            if actual != expected:
                raise DataError(f"{expected} trainer cache must have split={expected!r}, got {actual!r}")
        if self.test_cache is not None and self.test_cache.manifest.get("split") != "test":
            raise DataError("test trainer cache must have split='test'")
        manifests = [self.train_cache.manifest, self.validation_cache.manifest]
        if self.test_cache is not None:
            manifests.append(self.test_cache.manifest)
        manifests = [dict(manifest) for manifest in manifests]
        ids: dict[str, str] = {}
        for manifest in manifests:
            for episode_id in manifest.get("episode_ids", ()):
                prior = ids.get(str(episode_id))
                if prior is not None:
                    raise DataError(
                        f"episode {episode_id!r} occurs in both {prior} and another trainer split"
                    )
                ids[str(episode_id)] = str(manifest.get("split"))
        for manifest in manifests:
            if manifest.get("emotion_schema_version") is None:
                raise DataError("trainer cache manifest is missing emotion_schema_version")

    def _cache_metadata(self, loader: NPZTensorCacheLoader) -> dict[str, Any]:
        manifest = loader.manifest
        return {
            "split": manifest.get("split"),
            "cache_content_hash": manifest.get("cache_content_hash"),
            "npz_sha256": manifest.get("npz_sha256"),
            "source_content_hash": manifest.get("source_content_hash"),
            "source_shard_ids": list(manifest.get("source_shard_ids", ())),
            "split_seed": manifest.get("split_seed"),
            "preprocessing_version": manifest.get("preprocessing_version"),
            "preprocessor_config_hash": manifest.get("preprocessor_config_hash"),
            "observation_schema_version": manifest.get("observation_schema_version"),
            "emotion_schema_version": manifest.get("emotion_schema_version"),
            "schema_hash": manifest.get("schema_hash"),
            "episode_ids": list(manifest.get("episode_ids", ())),
        }

    def _data_metadata(self) -> dict[str, Any]:
        result = {"train": self._cache_metadata(self.train_cache), "validation": self._cache_metadata(self.validation_cache)}
        if self.test_cache is not None:
            result["test"] = self._cache_metadata(self.test_cache)
        return result

    def _epoch_indices(self, count: int, epoch: int) -> np.ndarray:
        # Epoch-derived generators make the order independent of batch count
        # and make a resumed epoch exactly match an uninterrupted run.
        return np.random.default_rng(self.config.seed + int(epoch)).permutation(count)

    @staticmethod
    def _loss_and_accuracy(
        model: IntentInferenceModule, batch: IntentBatch
    ) -> tuple[Tensor, int, int]:
        logits = model(
            batch.model_inputs["event_history"],
            batch.model_inputs["event_valid_mask"],
            output="logits",
        )
        valid = batch.role_label_mask
        if logits.ndim != 3 or logits.shape[:2] != batch.role_labels.shape:
            raise TrainingError("IntentInference output does not match four role targets")
        if not bool(valid.any()):
            raise TrainingError("intent batch contains no valid role labels")
        flat_logits = logits[valid]
        flat_labels = batch.role_labels[valid]
        loss = model.role_loss(flat_logits, flat_labels)
        correct = int((flat_logits.argmax(dim=-1) == flat_labels).sum().item())
        return loss, correct, int(flat_labels.numel())

    def _write_metric(self, record: Mapping[str, Any]) -> None:
        safe = _json_safe(dict(record))
        snapshot = self.job_snapshot or {}
        run_id = str(snapshot.get("job_id") or self.config_hash)
        metadata = build_run_metadata(
            run_id=run_id,
            job_id=snapshot.get("job_id"),
            component=self.phase,
            snapshot=snapshot or None,
            config=self.config_snapshot,
            counters={"environment_steps": 0, "main_agent_transitions": 0, "ppo_updates": 0},
        )
        safe["run_metadata"] = metadata
        if self.logger is not None:
            if hasattr(self.logger, "event"):
                self.logger.event("training_metric", phase=self.phase, **safe)
            else:
                self.logger(safe)
        if self.config.metrics_path is not None:
            destination = Path(self.config.metrics_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe, sort_keys=True, separators=(",", ":")) + "\n")

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        order = self._epoch_indices(len(self.train_cache), epoch)
        losses: list[float] = []
        gradients: list[float] = []
        correct = targets = 0
        for start in range(0, len(order), self.config.batch_size):
            indices = order[start : start + self.config.batch_size]
            batch = build_intent_batch(self.train_cache, indices, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss, batch_correct, batch_targets = self._loss_and_accuracy(self.model, batch)
            loss.backward()
            squared = 0.0
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    squared += float(parameter.grad.detach().norm(2).item()) ** 2
            gradient_norm = squared**0.5
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
            gradients.append(gradient_norm)
            correct += batch_correct
            targets += batch_targets
            self._write_metric(
                {
                    "phase": self.phase,
                    "split": "train",
                    "event": "batch",
                    "epoch": epoch,
                    "batch_start": int(start),
                    "batch_size": int(len(indices)),
                    "loss": float(loss.detach().item()),
                    "accuracy": batch_correct / batch_targets,
                    "valid_targets": batch_targets,
                    "gradient_norm": gradient_norm,
                }
            )
        return {
            "loss": float(np.average(losses, weights=None)) if losses else float("nan"),
            "accuracy": correct / targets if targets else float("nan"),
            "valid_targets": targets,
            "gradient_norm": float(np.mean(gradients)) if gradients else 0.0,
            "batches": len(losses),
        }

    def _evaluate(self, loader: NPZTensorCacheLoader, split: str, epoch: int) -> dict[str, Any]:
        self.model.eval()
        losses: list[float] = []
        correct = targets = 0
        with torch.no_grad():
            for start in range(0, len(loader), self.config.batch_size):
                indices = np.arange(start, min(start + self.config.batch_size, len(loader)))
                batch = build_intent_batch(loader, indices, device=self.device)
                loss, batch_correct, batch_targets = self._loss_and_accuracy(self.model, batch)
                losses.append(float(loss.item()))
                correct += batch_correct
                targets += batch_targets
        result = {
            "phase": self.phase,
            "split": split,
            "event": "epoch",
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "accuracy": correct / targets if targets else float("nan"),
            "valid_targets": targets,
            "batches": len(losses),
        }
        self._write_metric(result)
        return result

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_format_version": SL_CHECKPOINT_FORMAT_VERSION,
            "phase": self.phase,
            "config": self.config.to_dict(),
            "config_hash": self.config_hash,
            "model": self.model.metadata(),
            "data": self._data_metadata(),
            "job_snapshot": self.job_snapshot,
        }

    def save_checkpoint(self, path: str | os.PathLike[str] | None = None) -> Path:
        destination = Path(path or self.config.checkpoint_path or "intent-inference.checkpoint.pt")
        payload = {
            "metadata": self._checkpoint_metadata(),
            "model_state": _clone_state_dict(self.model),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
            "next_epoch": self.start_epoch,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "best_state": self.best_state,
            "bad_epochs": self._bad_epochs,
            "rng_state": _rng_state(),
        }
        _atomic_torch_save(payload, destination)
        self._write_metric(
            {"phase": self.phase, "event": "checkpoint_saved", "path": str(destination), "epoch": self.start_epoch - 1}
        )
        return destination

    def _check_checkpoint_compatibility(self, metadata: Mapping[str, Any]) -> None:
        if metadata.get("checkpoint_format_version") != SL_CHECKPOINT_FORMAT_VERSION:
            raise ArtifactCompatibilityError(
                "checkpoint_format_version", metadata.get("checkpoint_format_version"), SL_CHECKPOINT_FORMAT_VERSION
            )
        if metadata.get("phase") != self.phase:
            raise ArtifactCompatibilityError("phase", metadata.get("phase"), self.phase)
        if metadata.get("config_hash") != self.config_hash:
            raise ArtifactCompatibilityError("config_hash", metadata.get("config_hash"), self.config_hash)
        expected_model = self.model.metadata()
        actual_model = metadata.get("model")
        if actual_model != expected_model:
            raise ArtifactCompatibilityError("model_architecture", actual_model, expected_model)
        actual_data = metadata.get("data")
        if actual_data != self._data_metadata():
            raise ArtifactCompatibilityError("data_cache_metadata", actual_data, self._data_metadata())

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        source = Path(path)
        try:
            payload = torch.load(source, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise TrainingError(f"cannot load intent training checkpoint {source}") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
            raise TrainingError("malformed intent training checkpoint")
        self._check_checkpoint_compatibility(payload["metadata"])
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.history = [dict(item) for item in payload.get("history", ())]
        self.start_epoch = int(payload.get("next_epoch", 1))
        self.best_epoch = int(payload.get("best_epoch", 0))
        self.best_validation_loss = float(payload.get("best_validation_loss", float("inf")))
        self.best_state = {
            name: value.detach().cpu().clone() for name, value in payload.get("best_state", {}).items()
        }
        self._bad_epochs = int(payload.get("bad_epochs", 0))
        if not self.best_state:
            self.best_state = _clone_state_dict(self.model)
        if isinstance(payload.get("rng_state"), Mapping):
            _restore_rng_state(payload["rng_state"])
        self._resume_path = source

    def fit(self, *, stop_after_epoch: int | None = None) -> TrainingResult:
        if self.start_epoch > self.config.epochs:
            self._restore_best()
            return self._result()
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            validation = self._evaluate(self.validation_cache, "validation", epoch)
            record = {
                "phase": self.phase,
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_gradient_norm": train_metrics["gradient_norm"],
                "train_valid_targets": train_metrics["valid_targets"],
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_valid_targets": validation["valid_targets"],
            }
            self.history.append(record)
            self._write_metric(record)
            current_loss = float(validation["loss"])
            if current_loss < self.best_validation_loss - self.config.min_delta:
                self.best_validation_loss = current_loss
                self.best_epoch = epoch
                self.best_state = _clone_state_dict(self.model)
                self._bad_epochs = 0
            else:
                self._bad_epochs += 1
            self.start_epoch = epoch + 1
            if self.config.checkpoint_path is not None:
                self.save_checkpoint()
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break
            if (
                self.config.early_stopping_patience is not None
                and self._bad_epochs >= self.config.early_stopping_patience
            ):
                self._stopped_early = True
                break
        self._restore_best()
        return self._result()

    def _restore_best(self) -> None:
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)
            self.model.eval()

    def _result(self) -> TrainingResult:
        test_metrics = None
        if self.test_cache is not None and self.history:
            test_metrics = self._evaluate(self.test_cache, "test", self.best_epoch or self.start_epoch - 1)
        metadata = self.artifact_metadata(test_metrics=test_metrics)
        return TrainingResult(
            model=self.model,
            history=tuple(self.history),
            best_epoch=self.best_epoch,
            stopped_early=self._stopped_early,
            test_metrics=test_metrics,
            artifact_metadata=metadata,
        )

    def artifact_metadata(self, *, test_metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        versions = {}
        if self.job_snapshot is not None:
            versions.update(self.job_snapshot.get("versions", {}))
        versions.update(
            {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "intent_inference_artifact_format_version": SL_ARTIFACT_FORMAT_VERSION,
            }
        )
        return _json_safe(
            {
                "artifact_type": "phase2_intent_inference",
                "run_metadata": build_run_metadata(
                    run_id=str((self.job_snapshot or {}).get("job_id") or self.config_hash),
                    job_id=(self.job_snapshot or {}).get("job_id"),
                    component=self.phase,
                    snapshot=self.job_snapshot,
                    config=self.config_snapshot,
                    source_hashes={"train": self.train_cache.manifest.get("cache_content_hash"), "validation": self.validation_cache.manifest.get("cache_content_hash")},
                ),
                "artifact_format_version": SL_ARTIFACT_FORMAT_VERSION,
                "module_type": IntentInferenceModule.module_type,
                "module_metadata": self.model.metadata(),
                "architecture": self.model.architecture_metadata,
                "optimizer": self.model.optimizer_metadata,
                "schema_versions": versions,
                "config_hash": self.config_hash,
                "config": self.config_snapshot or self.config.to_dict(),
                "job_snapshot": self.job_snapshot,
                "source_hashes": {
                    split: {
                        "source_content_hash": data.get("source_content_hash"),
                        "cache_content_hash": data.get("cache_content_hash"),
                        "npz_sha256": data.get("npz_sha256"),
                    }
                    for split, data in self._data_metadata().items()
                },
                "cache_metadata": self._data_metadata(),
                "split": {
                    "train_episode_count": len(self.train_cache.manifest.get("episode_ids", ())),
                    "validation_episode_count": len(self.validation_cache.manifest.get("episode_ids", ())),
                    "test_episode_count": 0 if self.test_cache is None else len(self.test_cache.manifest.get("episode_ids", ())),
                    "split_seed": self.train_cache.manifest.get("split_seed"),
                },
                "training": {
                    "phase": self.phase,
                    "epochs_configured": self.config.epochs,
                    "epochs_completed": len(self.history),
                    "best_epoch": self.best_epoch,
                    "best_validation_loss": self.best_validation_loss,
                    "early_stopped": self._stopped_early,
                    "seed": self.config.seed,
                    "batch_size": self.config.batch_size,
                    "metrics": list(self.history),
                    "test_metrics": test_metrics,
                },
                "compatibility": {
                    "model_architecture": self.model.architecture_metadata,
                    "schema_versions": versions,
                    "config_hash": self.config_hash,
                },
            }
        )

    def save_artifact(self, path: str | os.PathLike[str], *, test_metrics: Mapping[str, Any] | None = None) -> Path:
        destination = Path(path)
        metadata = self.artifact_metadata(test_metrics=test_metrics)
        payload = {
            "artifact_type": "phase2_intent_inference",
            "artifact_format_version": SL_ARTIFACT_FORMAT_VERSION,
            "metadata": metadata,
            "module_metadata": self.model.to_dict(),
            "state_dict": _clone_state_dict(self.model),
            "training_mode": False,
        }
        _atomic_torch_save(payload, destination)
        self._write_metric({"phase": self.phase, "event": "artifact_saved", "path": str(destination)})
        return destination


def load_intent_artifact(
    path: str | os.PathLike[str],
    *,
    expected: Mapping[str, Any] | None = None,
    map_location: torch.device | str = "cpu",
) -> tuple[IntentInferenceModule, Mapping[str, Any]]:
    """Load a phase-2 module artifact and enforce compatibility fields."""

    source = Path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise TrainingError(f"cannot load IntentInference artifact {source}") from exc
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "phase2_intent_inference":
        raise ArtifactCompatibilityError("artifact_type", payload.get("artifact_type") if isinstance(payload, Mapping) else None, "phase2_intent_inference")
    if payload.get("artifact_format_version") != SL_ARTIFACT_FORMAT_VERSION:
        raise ArtifactCompatibilityError("artifact_format_version", payload.get("artifact_format_version"), SL_ARTIFACT_FORMAT_VERSION)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TrainingError("IntentInference artifact metadata is missing")
    if expected:
        for field_name, required in expected.items():
            current: Any = metadata
            for part in str(field_name).split("."):
                if not isinstance(current, Mapping) or part not in current:
                    raise ArtifactCompatibilityError(field_name, None, required, artifact=str(source))
                current = current[part]
            if current != required:
                raise ArtifactCompatibilityError(field_name, current, required, artifact=str(source))
    module_metadata = payload.get("module_metadata")
    if not isinstance(module_metadata, Mapping):
        raise TrainingError("IntentInference artifact module metadata is missing")
    try:
        module = IntentInferenceModule.from_dict(module_metadata)
        module.load_state_dict(payload["state_dict"])
    except Exception as exc:
        raise ArtifactCompatibilityError("module_metadata", "malformed", "IntentInferenceModule", artifact=str(source)) from exc
    module.eval()
    return module, metadata


# Friendly aliases for callers using the shorter task vocabulary.
IntentInferenceTrainer = IntentInferenceSLTrainer
SLTrainer = IntentInferenceSLTrainer
build_batch = build_intent_batch
load_artifact = load_intent_artifact


def __getattr__(name: str):
    """Lazily expose the emotion trainer from the historical trainer module."""

    if name in {
        "EmotionBatch",
        "EmotionInferenceSLTrainer",
        "EmotionInferenceTrainer",
        "EmotionSLConfig",
        "EmotionSLTrainer",
        "EmotionTrainingResult",
        "EmotionTrainer",
        "build_emotion_batch",
        "build_emotion_training_batch",
        "load_emotion_artifact",
        "load_emotion_inference_artifact",
    }:
        from . import emotion_sl_trainer

        return getattr(emotion_sl_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IntentBatch",
    "IntentInferenceSLTrainer",
    "IntentInferenceTrainer",
    "IntentSLConfig",
    "SLTrainer",
    "SL_ARTIFACT_FORMAT_VERSION",
    "SL_CHECKPOINT_FORMAT_VERSION",
    "TrainingResult",
    "build_batch",
    "build_intent_batch",
    "load_artifact",
    "load_intent_artifact",
    "EmotionBatch",
    "EmotionInferenceSLTrainer",
    "EmotionInferenceTrainer",
    "EmotionSLConfig",
    "EmotionSLTrainer",
    "EmotionTrainingResult",
    "EmotionTrainer",
    "build_emotion_batch",
    "build_emotion_training_batch",
    "load_emotion_artifact",
    "load_emotion_inference_artifact",
]
