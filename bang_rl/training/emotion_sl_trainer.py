"""Phase-2 supervised training for the public EmotionInference module.

The trainer consumes only normal-episode NPZ tensor caches.  Public event
history and its valid mask are the complete model input; emotion labels,
transition actions, rewards, and other target metadata remain on the training
side of the boundary.  Labels are selected for the four players other than
the public viewer and therefore remain aligned with the pre-action row that
produced them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import os

import numpy as np
import torch
from torch import Tensor

from bang_rl.agents.main_agent import EmotionInferenceModule
from bang_rl.config import EmotionConfig
from bang_rl.config.snapshot import JobSnapshot
from bang_rl.data.tensor_cache import (
    CACHE_FORMAT_VERSION,
    CACHE_SCHEMA_VERSION,
    NPZTensorCacheLoader,
)
from bang_rl.exceptions import ArtifactCompatibilityError, DataError, TrainingError
from bang_rl.runtime.metadata import build_run_metadata
from .sl_trainer import (
    SL_ARTIFACT_FORMAT_VERSION,
    SL_CHECKPOINT_FORMAT_VERSION,
    _atomic_torch_save,
    _clone_state_dict,
    _hash,
    _json_safe,
    _restore_rng_state,
    _rng_state,
)


@dataclass(frozen=True)
class EmotionBatch:
    """A cache batch with public inputs separated from emotion targets."""

    model_inputs: Mapping[str, Tensor]
    emotion_labels: Tensor
    emotion_label_mask: Tensor
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
        return self.emotion_labels

    def __getitem__(self, key: str) -> Any:
        if key in self.model_inputs:
            return self.model_inputs[key]
        if key in {"inputs", "model_inputs"}:
            return self.model_inputs
        if key in {"emotion_labels", "labels"}:
            return self.emotion_labels
        if key == "emotion_label_mask":
            return self.emotion_label_mask
        if key == "viewer_indices":
            return self.viewer_indices
        raise KeyError(key)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.model_inputs) + (
            "emotion_labels",
            "emotion_label_mask",
            "viewer_indices",
        )


def _as_loader(cache: NPZTensorCacheLoader | str | os.PathLike[str]) -> NPZTensorCacheLoader:
    loader = cache if isinstance(cache, NPZTensorCacheLoader) else NPZTensorCacheLoader(cache)
    loader.validate()
    return loader


def _viewer_indices(state_vector: np.ndarray) -> np.ndarray:
    values = np.asarray(state_vector, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 9:
        raise DataError("emotion cache state_vector must expose the public viewer index")
    scaled = values[:, 8] * 4.0
    viewers = np.rint(scaled).astype(np.int64)
    if not np.allclose(scaled, viewers, atol=1e-5) or np.any((viewers < 0) | (viewers >= 5)):
        raise DataError("emotion cache contains an invalid public viewer index")
    return viewers


def build_emotion_batch(
    cache: NPZTensorCacheLoader | str | os.PathLike[str],
    indices: Sequence[int] | np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> EmotionBatch:
    """Build a deterministic public-input/continuous-label batch.

    ``state_vector[..., 8]`` is the normalized public viewer index used by the
    cache contract.  The target for each row is reordered to the four other
    players in the same order as ``event_history``.  No action, reward, role,
    or emotion target is included in ``model_inputs``.
    """

    loader = _as_loader(cache)
    row_indices = tuple(int(item) for item in np.asarray(indices, dtype=np.int64).reshape(-1))
    if any(item < 0 or item >= len(loader) for item in row_indices):
        raise IndexError("emotion batch row index is outside the cache")
    tensors = loader.model_inputs(np.asarray(row_indices, dtype=np.int64))
    labels = loader.labels(np.asarray(row_indices, dtype=np.int64))
    viewers = _viewer_indices(tensors["state_vector"])
    values = np.asarray(labels["emotion_labels"], dtype=np.float32)
    masks = np.asarray(labels["emotion_label_mask"], dtype=np.bool_)
    if values.ndim != 3 or values.shape[1] != 5:
        raise DataError("emotion_labels must have shape (batch, 5, emotion_count)")
    if masks.shape != values.shape[:2]:
        raise DataError("emotion_label_mask must have shape (batch, 5)")
    emotion_count = values.shape[-1]
    if emotion_count < 1:
        raise DataError("emotion cache must contain at least one emotion dimension")
    selected_values = np.zeros((len(row_indices), 4, emotion_count), dtype=np.float32)
    selected_masks = np.zeros((len(row_indices), 4), dtype=np.bool_)
    for row, viewer in enumerate(viewers):
        targets = [player for player in range(5) if player != int(viewer)]
        selected_values[row] = values[row, targets]
        selected_masks[row] = masks[row, targets]
    if not np.all(np.isfinite(selected_values)) or np.any((selected_values < 0.0) | (selected_values > 1.0)):
        raise DataError("emotion targets must be finite values in [0, 1]")

    # Deliberately construct this mapping instead of forwarding all cache
    # inputs: action_mask/state/reward bookkeeping cannot become leakage.
    public_inputs = {
        "event_history": torch.as_tensor(
            tensors["event_history"], dtype=torch.float32, device=device
        ),
        "event_valid_mask": torch.as_tensor(
            tensors["event_valid_mask"], dtype=torch.bool, device=device
        ),
    }
    return EmotionBatch(
        model_inputs=public_inputs,
        emotion_labels=torch.as_tensor(selected_values, dtype=torch.float32, device=device),
        emotion_label_mask=torch.as_tensor(selected_masks, dtype=torch.bool, device=device),
        viewer_indices=torch.as_tensor(viewers, dtype=torch.long, device=device),
        row_indices=row_indices,
    )


@dataclass(frozen=True)
class EmotionSLConfig:
    """Runtime settings for deterministic EmotionInference training."""

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
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be positive or None")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")

    @classmethod
    def from_value(cls, value: "EmotionSLConfig | Mapping[str, Any] | None") -> "EmotionSLConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("training config must be EmotionSLConfig or a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError(f"unknown EmotionSLConfig fields: {', '.join(unknown)}")
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
class EmotionTrainingResult:
    model: EmotionInferenceModule
    history: tuple[Mapping[str, Any], ...]
    best_epoch: int
    stopped_early: bool
    test_metrics: Mapping[str, Any] | None
    artifact_metadata: Mapping[str, Any]

    @property
    def metrics(self) -> tuple[Mapping[str, Any], ...]:
        return self.history


class EmotionInferenceSLTrainer:
    """Train EmotionInference from normal-only NPZ cache splits."""

    phase = "phase2_supervised_emotion"

    def __init__(
        self,
        train_cache: NPZTensorCacheLoader | str | os.PathLike[str],
        validation_cache: NPZTensorCacheLoader | str | os.PathLike[str],
        *,
        test_cache: NPZTensorCacheLoader | str | os.PathLike[str] | None = None,
        config: EmotionSLConfig | Mapping[str, Any] | None = None,
        emotion_config: EmotionConfig | Mapping[str, Any] | None = None,
        model: EmotionInferenceModule | None = None,
        job_snapshot: JobSnapshot | Mapping[str, Any] | None = None,
        config_snapshot: Mapping[str, Any] | Any | None = None,
        config_hash: str | None = None,
        resume_from: str | os.PathLike[str] | None = None,
        logger: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.config = EmotionSLConfig.from_value(config)
        self.train_cache = _as_loader(train_cache)
        self.validation_cache = _as_loader(validation_cache)
        self.test_cache = None if test_cache is None else _as_loader(test_cache)
        self._validate_splits()
        self.job_snapshot = self._snapshot_dict(job_snapshot)
        self.config_snapshot = self._config_dict(config_snapshot)
        self.config_hash = config_hash or _hash(
            self.job_snapshot
            if self.job_snapshot is not None
            else self.config_snapshot or self.config.to_dict()
        )
        self.logger = logger
        self.device = torch.device(self.config.device)
        self._seed_everything(self.config.seed, deterministic=self.config.deterministic)

        cache_schema = str(self.train_cache.manifest["emotion_schema_version"])
        cache_dimension = int(self.train_cache.manifest.get("emotion_dimension", 0))
        if cache_dimension < 1:
            sample = self.train_cache.labels(slice(None))["emotion_labels"]
            cache_dimension = int(sample.shape[-1]) if sample.ndim == 3 else 0
        if cache_dimension < 1:
            raise DataError("emotion cache does not define a positive emotion dimension")
        if model is not None and emotion_config is not None:
            raise ValueError("provide either model or emotion_config, not both")
        if model is None:
            if emotion_config is None and self.job_snapshot is not None:
                snapshot_schema = self.job_snapshot.get("emotion_schema_snapshot")
                if isinstance(snapshot_schema, Mapping):
                    emotion_config = snapshot_schema
            if emotion_config is None and self.config_snapshot is not None:
                snapshot_schema = self.config_snapshot.get("emotion")
                if isinstance(snapshot_schema, Mapping):
                    emotion_config = snapshot_schema
            if emotion_config is None:
                emotion_config = EmotionConfig(
                    schema_version=cache_schema,
                    emotion_count=cache_dimension,
                )
            elif not isinstance(emotion_config, EmotionConfig):
                emotion_config = EmotionConfig.model_validate(dict(emotion_config))
            model = EmotionInferenceModule(config=emotion_config, seed=self.config.seed)
        if not isinstance(model, EmotionInferenceModule):
            raise TypeError("model must be an EmotionInferenceModule")
        if model.emotion_schema_version != cache_schema:
            raise DataError(
                "Emotion Schema Version mismatch: "
                f"model={model.emotion_schema_version!r}, cache={cache_schema!r}"
            )
        if model.emotion_count != cache_dimension:
            raise DataError(
                "Emotion Schema dimension mismatch: "
                f"model={model.emotion_count}, cache={cache_dimension}"
            )
        self._validate_snapshot_schema(cache_schema)
        for loader in (self.validation_cache, self.test_cache):
            if loader is not None and int(loader.manifest.get("emotion_dimension", cache_dimension)) != model.emotion_count:
                raise DataError("Emotion Schema dimension mismatch across cache splits")
        self.model = model.to(self.device)
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
        import random

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
            if len(loader) == 0:
                raise DataError(f"{expected} trainer cache must not be empty")
        if self.test_cache is not None and self.test_cache.manifest.get("split") != "test":
            raise DataError("test trainer cache must have split='test'")
        loaders = [self.train_cache, self.validation_cache]
        if self.test_cache is not None:
            loaders.append(self.test_cache)
        manifests = [dict(loader.manifest) for loader in loaders]
        ids: dict[str, str] = {}
        for manifest in manifests:
            for episode_id in manifest.get("episode_ids", ()):
                prior = ids.get(str(episode_id))
                if prior is not None:
                    raise DataError(f"episode {episode_id!r} occurs in both {prior} and another trainer split")
                ids[str(episode_id)] = str(manifest.get("split"))
            if not manifest.get("emotion_schema_version"):
                raise DataError("trainer cache manifest is missing emotion_schema_version")
        schemas = {str(manifest.get("emotion_schema_version")) for manifest in manifests}
        if len(schemas) != 1:
            raise DataError("Emotion Schema Version mismatch across trainer cache splits")
        for loader in loaders:
            if np.any(loader.transition()["truncated"]):
                raise DataError("EmotionInference trainer accepts normal-only caches; truncated rows are forbidden")
        dimensions = {int(manifest.get("emotion_dimension", 0)) for manifest in manifests}
        if len(dimensions) > 1:
            raise DataError("Emotion Schema dimension mismatch across trainer cache splits")

    def _validate_snapshot_schema(self, cache_schema: str) -> None:
        candidates: list[tuple[str, Any]] = []
        if self.job_snapshot is not None:
            versions = self.job_snapshot.get("versions", {})
            if isinstance(versions, Mapping) and versions.get("emotion_schema_version") is not None:
                candidates.append(("job_snapshot.versions.emotion_schema_version", versions["emotion_schema_version"]))
            schema_snapshot = self.job_snapshot.get("emotion_schema_snapshot", {})
            if isinstance(schema_snapshot, Mapping) and schema_snapshot.get("schema_version") is not None:
                candidates.append(("job_snapshot.emotion_schema_snapshot.schema_version", schema_snapshot["schema_version"]))
        if self.config_snapshot is not None:
            emotion = self.config_snapshot.get("emotion", {})
            if isinstance(emotion, Mapping) and emotion.get("schema_version") is not None:
                candidates.append(("config_snapshot.emotion.schema_version", emotion["schema_version"]))
        for field_name, value in candidates:
            if str(value) != cache_schema:
                raise DataError(
                    f"Emotion Schema Version mismatch: {field_name}={value!r}, cache={cache_schema!r}"
                )

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
            "emotion_dimension": manifest.get("emotion_dimension"),
            "schema_hash": manifest.get("schema_hash"),
            "episode_ids": list(manifest.get("episode_ids", ())),
        }

    def _data_metadata(self) -> dict[str, Any]:
        result = {
            "train": self._cache_metadata(self.train_cache),
            "validation": self._cache_metadata(self.validation_cache),
        }
        if self.test_cache is not None:
            result["test"] = self._cache_metadata(self.test_cache)
        return result

    def _epoch_indices(self, count: int, epoch: int) -> np.ndarray:
        return np.random.default_rng(self.config.seed + int(epoch)).permutation(count)

    def _loss_and_metrics(self, batch: EmotionBatch) -> tuple[Tensor, int, int]:
        predictions = self.model(
            batch.model_inputs["event_history"],
            batch.model_inputs["event_valid_mask"],
            output="values",
        )
        if predictions.ndim != 3 or tuple(predictions.shape[:2]) != tuple(batch.emotion_labels.shape[:2]):
            raise TrainingError("EmotionInference output does not match four emotion target streams")
        if predictions.shape[-1] != batch.emotion_labels.shape[-1]:
            raise TrainingError("EmotionInference output dimension does not match Emotion Schema")
        if not bool(torch.all((predictions >= 0.0) & (predictions <= 1.0))):
            raise TrainingError("EmotionInference output is outside [0, 1]")
        valid = batch.emotion_label_mask
        if not bool(valid.any()):
            raise TrainingError("emotion batch contains no valid emotion labels")
        flat_predictions = predictions[valid]
        flat_targets = batch.emotion_labels[valid]
        loss = self.model.emotion_loss(flat_predictions, flat_targets)
        return loss, int(flat_predictions.shape[0]), int(flat_predictions.numel())

    def _write_metric(self, record: Mapping[str, Any]) -> None:
        safe = _json_safe(dict(record))
        # Every epoch metric is a split-scoped record.  Keep the aggregate
        # train/validation summary compatible with consumers that require the
        # common split field, while preserving explicit validation/test values.
        if safe.get("event") == "epoch":
            safe.setdefault("split", "train")
        snapshot = self.job_snapshot or {}
        metadata = build_run_metadata(
            run_id=str(snapshot.get("job_id") or self.config_hash),
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
                import json
                handle.write(json.dumps(safe, sort_keys=True, separators=(",", ":")) + "\n")

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        order = self._epoch_indices(len(self.train_cache), epoch)
        losses: list[float] = []
        gradients: list[float] = []
        targets = values = 0
        for start in range(0, len(order), self.config.batch_size):
            indices = order[start : start + self.config.batch_size]
            batch = build_emotion_batch(self.train_cache, indices, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss, batch_targets, batch_values = self._loss_and_metrics(batch)
            loss.backward()
            squared = 0.0
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    squared += float(parameter.grad.detach().norm(2).item()) ** 2
            gradient_norm = squared**0.5
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
            gradients.append(gradient_norm)
            targets += batch_targets
            values += batch_values
            self._write_metric({
                "phase": self.phase,
                "split": "train",
                "event": "batch",
                "epoch": epoch,
                "batch_start": int(start),
                "batch_size": int(len(indices)),
                "loss": float(loss.detach().item()),
                "mse": float(loss.detach().item()),
                "valid_targets": batch_targets,
                "valid_values": batch_values,
                "gradient_norm": gradient_norm,
            })
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "mse": float(np.mean(losses)) if losses else float("nan"),
            "valid_targets": targets,
            "valid_values": values,
            "gradient_norm": float(np.mean(gradients)) if gradients else 0.0,
            "batches": len(losses),
        }

    def _evaluate(self, loader: NPZTensorCacheLoader, split: str, epoch: int) -> dict[str, Any]:
        self.model.eval()
        losses: list[float] = []
        targets = values = 0
        with torch.no_grad():
            for start in range(0, len(loader), self.config.batch_size):
                indices = np.arange(start, min(start + self.config.batch_size, len(loader)))
                batch = build_emotion_batch(loader, indices, device=self.device)
                loss, batch_targets, batch_values = self._loss_and_metrics(batch)
                losses.append(float(loss.item()))
                targets += batch_targets
                values += batch_values
        result = {
            "phase": self.phase,
            "split": split,
            "event": "epoch",
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "mse": float(np.mean(losses)) if losses else float("nan"),
            "valid_targets": targets,
            "valid_values": values,
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
        destination = Path(path or self.config.checkpoint_path or "emotion-inference.checkpoint.pt")
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
        self._write_metric({
            "phase": self.phase,
            "event": "checkpoint_saved",
            "path": str(destination),
            "epoch": self.start_epoch - 1,
        })
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
        if metadata.get("model") != self.model.metadata():
            raise ArtifactCompatibilityError("model_architecture", metadata.get("model"), self.model.metadata())
        if metadata.get("data") != self._data_metadata():
            raise ArtifactCompatibilityError("data_cache_metadata", metadata.get("data"), self._data_metadata())

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        source = Path(path)
        try:
            payload = torch.load(source, map_location=self.device, weights_only=False)
        except Exception as exc:
            raise TrainingError(f"cannot load emotion training checkpoint {source}") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
            raise TrainingError("malformed emotion training checkpoint")
        self._check_checkpoint_compatibility(payload["metadata"])
        try:
            self.model.load_state_dict(payload["model_state"])
            self.optimizer.load_state_dict(payload["optimizer_state"])
        except Exception as exc:
            raise ArtifactCompatibilityError("checkpoint_state", "malformed", "EmotionInferenceModule", artifact=str(source)) from exc
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

    def fit(self, *, stop_after_epoch: int | None = None) -> EmotionTrainingResult:
        if stop_after_epoch is not None and (isinstance(stop_after_epoch, bool) or stop_after_epoch < 1):
            raise ValueError("stop_after_epoch must be positive or None")
        if self.start_epoch > self.config.epochs:
            self._restore_best()
            return self._result()
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            validation = self._evaluate(self.validation_cache, "validation", epoch)
            record = {
                "phase": self.phase,
                "split": "train",
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_valid_targets": train_metrics["valid_targets"],
                "train_valid_values": train_metrics["valid_values"],
                "train_gradient_norm": train_metrics["gradient_norm"],
                "validation_loss": validation["loss"],
                "validation_mse": validation["mse"],
                "validation_valid_targets": validation["valid_targets"],
                "validation_valid_values": validation["valid_values"],
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

    def _result(self) -> EmotionTrainingResult:
        test_metrics = None
        if self.test_cache is not None and self.history:
            test_metrics = self._evaluate(self.test_cache, "test", self.best_epoch or self.start_epoch - 1)
        return EmotionTrainingResult(
            model=self.model,
            history=tuple(self.history),
            best_epoch=self.best_epoch,
            stopped_early=self._stopped_early,
            test_metrics=test_metrics,
            artifact_metadata=self.artifact_metadata(test_metrics=test_metrics),
        )

    def artifact_metadata(self, *, test_metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        versions: dict[str, Any] = {}
        if self.job_snapshot is not None:
            versions.update(self.job_snapshot.get("versions", {}))
        versions.update({
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "emotion_inference_artifact_format_version": SL_ARTIFACT_FORMAT_VERSION,
            "emotion_schema_version": self.model.emotion_schema_version,
        })
        data = self._data_metadata()
        return _json_safe({
            "artifact_type": "phase2_emotion_inference",
            "run_metadata": build_run_metadata(
                run_id=str((self.job_snapshot or {}).get("job_id") or self.config_hash),
                job_id=(self.job_snapshot or {}).get("job_id"),
                component=self.phase,
                snapshot=self.job_snapshot,
                config=self.config_snapshot,
                source_hashes={"train": self.train_cache.manifest.get("cache_content_hash"), "validation": self.validation_cache.manifest.get("cache_content_hash")},
            ),
            "artifact_format_version": SL_ARTIFACT_FORMAT_VERSION,
            "module_type": EmotionInferenceModule.module_type,
            "module_metadata": self.model.metadata(),
            "architecture": self.model.architecture_metadata,
            "emotion_schema": {
                "version": self.model.emotion_schema_version,
                "count": self.model.emotion_count,
                "names": list(self.model.emotion_names),
                "initial_baseline": self.model.initial_baseline,
            },
            "optimizer": self.model.optimizer_metadata,
            "loss": self.model.loss_name,
            "schema_versions": versions,
            "emotion_schema_version": self.model.emotion_schema_version,
            "config_hash": self.config_hash,
            "config": self.config_snapshot or self.config.to_dict(),
            "job_snapshot": self.job_snapshot,
            "source_hashes": {
                split: {
                    "source_content_hash": values.get("source_content_hash"),
                    "cache_content_hash": values.get("cache_content_hash"),
                    "npz_sha256": values.get("npz_sha256"),
                }
                for split, values in data.items()
            },
            "cache_metadata": data,
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
                "label_alignment": "pre-action emotion state; current/future events excluded",
                "model_inputs": ["event_history", "event_valid_mask"],
            },
            "compatibility": {
                "model_architecture": self.model.architecture_metadata,
                "schema_versions": versions,
                "config_hash": self.config_hash,
                "emotion_schema_version": self.model.emotion_schema_version,
            },
        })

    def save_artifact(
        self, path: str | os.PathLike[str], *, test_metrics: Mapping[str, Any] | None = None
    ) -> Path:
        """Save only the phase-2 module weights and metadata, never optimizer state."""

        destination = Path(path)
        payload = {
            "artifact_type": "phase2_emotion_inference",
            "artifact_format_version": SL_ARTIFACT_FORMAT_VERSION,
            "metadata": self.artifact_metadata(test_metrics=test_metrics),
            "module_metadata": self.model.to_dict(),
            "state_dict": _clone_state_dict(self.model),
            "training_mode": False,
        }
        _atomic_torch_save(payload, destination)
        self._write_metric({"phase": self.phase, "event": "artifact_saved", "path": str(destination)})
        return destination


def load_emotion_artifact(
    path: str | os.PathLike[str],
    *,
    expected: Mapping[str, Any] | None = None,
    map_location: torch.device | str = "cpu",
) -> tuple[EmotionInferenceModule, Mapping[str, Any]]:
    """Load a phase-2 EmotionInference artifact and check compatibility."""

    source = Path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise TrainingError(f"cannot load EmotionInference artifact {source}") from exc
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "phase2_emotion_inference":
        actual = payload.get("artifact_type") if isinstance(payload, Mapping) else None
        raise ArtifactCompatibilityError("artifact_type", actual, "phase2_emotion_inference")
    if payload.get("artifact_format_version") != SL_ARTIFACT_FORMAT_VERSION:
        raise ArtifactCompatibilityError("artifact_format_version", payload.get("artifact_format_version"), SL_ARTIFACT_FORMAT_VERSION)
    metadata = payload.get("metadata")
    module_metadata = payload.get("module_metadata")
    if not isinstance(metadata, Mapping):
        raise TrainingError("EmotionInference artifact metadata is missing")
    if expected:
        for field_name, required in expected.items():
            current: Any = metadata
            for part in str(field_name).split("."):
                if not isinstance(current, Mapping) or part not in current:
                    raise ArtifactCompatibilityError(field_name, None, required, artifact=str(source))
                current = current[part]
            if current != required:
                raise ArtifactCompatibilityError(field_name, current, required, artifact=str(source))
    if not isinstance(module_metadata, Mapping):
        raise TrainingError("EmotionInference artifact module metadata is missing")
    try:
        module = EmotionInferenceModule.from_dict(module_metadata)
        module.load_state_dict(payload["state_dict"])
    except Exception as exc:
        raise ArtifactCompatibilityError(
            "module_metadata", "malformed", "EmotionInferenceModule", artifact=str(source)
        ) from exc
    metadata_schema = metadata.get("emotion_schema_version")
    if metadata_schema is None and isinstance(metadata.get("emotion_schema"), Mapping):
        metadata_schema = metadata["emotion_schema"].get("version")
    if str(metadata_schema) != module.emotion_schema_version:
        raise ArtifactCompatibilityError(
            "emotion_schema_version", metadata_schema, module.emotion_schema_version, artifact=str(source)
        )
    module.eval()
    return module, metadata


EmotionInferenceTrainer = EmotionInferenceSLTrainer
EmotionSLTrainer = EmotionInferenceSLTrainer
EmotionTrainer = EmotionInferenceSLTrainer
build_emotion_training_batch = build_emotion_batch
load_emotion_inference_artifact = load_emotion_artifact
load_artifact = load_emotion_artifact

__all__ = [
    "EmotionBatch",
    "EmotionInferenceSLTrainer",
    "EmotionInferenceTrainer",
    "EmotionSLConfig",
    "EmotionSLTrainer",
    "EmotionTrainingResult",
    "EmotionTrainer",
    "SL_ARTIFACT_FORMAT_VERSION",
    "SL_CHECKPOINT_FORMAT_VERSION",
    "build_emotion_batch",
    "build_emotion_training_batch",
    "load_emotion_artifact",
    "load_emotion_inference_artifact",
]
