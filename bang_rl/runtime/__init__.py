"""Runtime coordination, deterministic seeds, and structured telemetry."""

from .emotion import (
    EMOTION_RUNTIME_STATE_VERSION,
    EmotionLabelSnapshot,
    EmotionRuntime,
    EmotionRuntimeError,
    EmotionTransitionRecord,
)
from .job import JobSnapshot, TrainingJob
from .pause_coordinator import PauseCoordinator, PauseState, WorkerCancelled, WorkerControl
from .seed import RNGStreams, SeedBundle, SeedManager, derive_seed, seed_everything
from .logging import (
    LOG_CHANNELS,
    LogChannel,
    LogManager,
    LongGameDiagnostics,
    StructuredLogger,
    get_component_logger,
    read_jsonl,
)
from .metadata import RunMetadata, build_run_metadata, canonical_json, content_hash, file_hash, json_safe
from .reward import (
    REWARD_RUNTIME_STATE_VERSION,
    RewardBatch,
    RewardBreakdown,
    RewardCalculationError,
    RewardCalculator,
    RewardRuntime,
    attach_rewards,
)

__all__ = [
    "EMOTION_RUNTIME_STATE_VERSION",
    "EmotionLabelSnapshot",
    "EmotionRuntime",
    "EmotionRuntimeError",
    "EmotionTransitionRecord",
    "JobSnapshot",
    "TrainingJob",
    "PauseCoordinator",
    "PauseState",
    "WorkerCancelled",
    "WorkerControl",
    "LOG_CHANNELS",
    "LogChannel",
    "LogManager",
    "LongGameDiagnostics",
    "StructuredLogger",
    "get_component_logger",
    "read_jsonl",
    "RunMetadata",
    "build_run_metadata",
    "canonical_json",
    "content_hash",
    "file_hash",
    "json_safe",
    "RNGStreams",
    "SeedBundle",
    "SeedManager",
    "derive_seed",
    "REWARD_RUNTIME_STATE_VERSION",
    "RewardBatch",
    "RewardBreakdown",
    "RewardCalculationError",
    "RewardCalculator",
    "RewardRuntime",
    "attach_rewards",
]
