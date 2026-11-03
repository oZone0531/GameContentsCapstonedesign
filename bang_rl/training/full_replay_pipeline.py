"""Offline supervised-learning pipeline for full five-Dummy replays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from bang_rl.config import JobSnapshot, SystemConfig
from bang_rl.data import (
    EpisodeSplitResult,
    EpisodeSplitter,
    TensorCacheArtifact,
    NPZTensorCacheWriter,
    ParquetCanonicalShardWriter,
    full_replay_to_canonical,
)

from .emotion_sl_trainer import EmotionInferenceSLTrainer, EmotionSLConfig, EmotionTrainingResult
from .inference_artifact import save_phase2_inference_artifact
from .sl_trainer import IntentInferenceSLTrainer, IntentSLConfig, TrainingResult


@dataclass(frozen=True)
class FullReplayPipelineResult:
    """Artifacts and trained phase-2 modules produced by the offline flow."""

    episodes: tuple[Any, ...]
    split: EpisodeSplitResult
    caches: dict[str, TensorCacheArtifact]
    intent: TrainingResult
    emotion: EmotionTrainingResult
    canonical_paths: tuple[Path, ...]
    phase2_artifact_path: Path


def train_from_full_replays(
    replay_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    config: SystemConfig | None = None,
    split_seed: int = 0,
    sl_epochs: int | None = None,
    sl_batch_size: int | None = None,
    job_snapshot: JobSnapshot | dict[str, Any] | None = None,
) -> FullReplayPipelineResult:
    """Convert full replays, build caches, and train role/emotion inference.

    Full replay state and DummyAgent internals are trusted labels at this
    boundary.  The resulting trainers still receive only public event-history
    tensors.  At least one normal episode is required in each train,
    validation, and test split by the existing cache/trainer contracts.
    """

    system_config = config or SystemConfig()
    root = Path(output_dir)
    canonical_root = root / "canonical"
    cache_root = root / "cache"
    canonical_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    converted: list[Any] = []
    seen_episode_ids: set[str] = set()
    for path in replay_paths:
        episode = full_replay_to_canonical(path)
        if episode.episode_id in seen_episode_ids:
            continue
        seen_episode_ids.add(episode.episode_id)
        converted.append(episode)
    episodes = tuple(converted)
    if not episodes:
        raise ValueError("at least one full replay is required")

    writer = ParquetCanonicalShardWriter(canonical_root, dataset_config=system_config.dataset)
    for episode in episodes:
        writer.append(episode)
    canonical_paths = writer.finalize()

    manifest_path = root / "split-manifest.json"
    split = (
        EpisodeSplitter(seed=split_seed, counts=(len(episodes) - 2, 1, 1), manifest_path=manifest_path).split(episodes)
        if 3 <= len(episodes) < 10
        else EpisodeSplitter(seed=split_seed, manifest_path=manifest_path).split(episodes)
    )
    caches = NPZTensorCacheWriter(
        cache_root,
        emotion_schema_version=system_config.emotion.schema_version,
        config=system_config,
    ).write_all(split)
    intent = IntentInferenceSLTrainer(
        caches["train"].tensor_path,
        caches["validation"].tensor_path,
        test_cache=caches["test"].tensor_path,
        config=IntentSLConfig(
            epochs=sl_epochs or system_config.sl_training.epochs,
            batch_size=sl_batch_size or system_config.sl_training.batch_size,
            seed=system_config.seed,
        ),
        job_snapshot=job_snapshot,
        config_snapshot=system_config,
    ).fit()
    emotion = EmotionInferenceSLTrainer(
        caches["train"].tensor_path,
        caches["validation"].tensor_path,
        test_cache=caches["test"].tensor_path,
        config=EmotionSLConfig(
            epochs=sl_epochs or system_config.sl_training.epochs,
            batch_size=sl_batch_size or system_config.sl_training.batch_size,
            seed=system_config.seed,
        ),
        emotion_config=system_config.emotion,
        job_snapshot=job_snapshot,
        config_snapshot=system_config,
    ).fit()
    phase2_path = root / "phase2.phase2model"
    save_phase2_inference_artifact(
        phase2_path,
        intent.model,
        emotion.model,
        job_snapshot=job_snapshot,
        config_snapshot=system_config,
        source_hashes={name: artifact.cache_hash for name, artifact in caches.items()},
    )
    return FullReplayPipelineResult(
        episodes=episodes,
        split=split,
        caches=caches,
        intent=intent,
        emotion=emotion,
        canonical_paths=tuple(canonical_paths),
        phase2_artifact_path=phase2_path,
    )


__all__ = ["FullReplayPipelineResult", "train_from_full_replays"]
