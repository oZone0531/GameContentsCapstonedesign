"""Deterministic random-number management for every Bang RL component.

The runtime must not create ad-hoc random generators.  ``SeedManager`` derives
named child seeds from a user-controlled root seed (and, optionally, a frozen
job snapshot), then exposes independent streams for environment and action
sampling.  Global Python, NumPy, and Torch RNGs can be seeded through the same
API for libraries that use process-global state.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np

MAX_UINT64 = (1 << 64) - 1
MAX_NUMPY_SEED = (1 << 32) - 1


def _normalise_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return int(seed) & MAX_UINT64


def _jsonable(value: Any) -> Any:
    """Convert snapshot-like values into stable JSON input for hashing."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    return value


def _snapshot_digest(snapshot: Any) -> str | None:
    if snapshot is None:
        return None
    if hasattr(snapshot, "to_dict"):
        snapshot = snapshot.to_dict()
    payload = json.dumps(
        _jsonable(snapshot), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_seed(
    base_seed: int,
    *components: Any,
    snapshot: Any | None = None,
) -> int:
    """Derive a stable 64-bit child seed from a root seed and namespace.

    SHA-256 is used instead of Python's process-randomised ``hash()`` so child
    seeds are identical across processes and machines.  Components should be
    semantic identifiers such as ``("worker", 2, "episode", 17)``.
    """

    payload = {
        "base_seed": _normalise_seed(base_seed),
        "components": _jsonable(components),
        "snapshot": _snapshot_digest(snapshot),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & MAX_UINT64


@dataclass(frozen=True)
class RNGStreams:
    """Independent streams for one named execution scope."""

    seed: int
    python: random.Random
    numpy: np.random.Generator
    environment: np.random.Generator
    action: np.random.Generator
    torch: Any | None = None


@dataclass(frozen=True)
class SeedBundle:
    """Seeds used by global libraries and the named runtime streams."""

    root_seed: int
    python_seed: int
    numpy_seed: int
    torch_seed: int
    environment_seed: int
    action_seed: int


class SeedManager:
    """Own root/config seed state and derive reproducible component streams."""

    def __init__(self, seed: int, *, snapshot: Any | None = None) -> None:
        self.root_seed = _normalise_seed(seed)
        self.snapshot = snapshot
        self.snapshot_digest = _snapshot_digest(snapshot)

    def derive(self, *components: Any) -> int:
        return derive_seed(self.root_seed, *components, snapshot=self.snapshot)

    def worker_seed(self, worker_id: int) -> int:
        return self.derive("worker", worker_id)

    def episode_seed(self, episode_id: str | int, *, worker_id: int = 0) -> int:
        return self.derive("worker", worker_id, "episode", episode_id)

    def agent_seed(
        self,
        agent_id: str | int,
        *,
        episode_id: str | int | None = None,
        worker_id: int = 0,
    ) -> int:
        return self.derive("worker", worker_id, "episode", episode_id, "agent", agent_id)

    def environment_seed(
        self, episode_id: str | int, *, worker_id: int = 0
    ) -> int:
        return self.derive("environment", "worker", worker_id, "episode", episode_id)

    def action_seed(
        self,
        agent_id: str | int,
        *,
        episode_id: str | int | None = None,
        worker_id: int = 0,
    ) -> int:
        return self.derive(
            "action", "worker", worker_id, "episode", episode_id, "agent", agent_id
        )

    def for_worker(self, worker_id: int) -> int:
        return self.worker_seed(worker_id)

    def for_episode(self, episode_id: str | int, *, worker_id: int = 0) -> int:
        return self.episode_seed(episode_id, worker_id=worker_id)

    def for_agent(
        self,
        agent_id: str | int,
        *,
        episode_id: str | int | None = None,
        worker_id: int = 0,
    ) -> int:
        return self.agent_seed(agent_id, episode_id=episode_id, worker_id=worker_id)

    def seed_all(self, *, deterministic_torch: bool = True) -> SeedBundle:
        """Seed process-global Python/NumPy/Torch state once.

        Environment and action sampling are intentionally represented by named
        child seeds in the returned bundle; callers should create their own
        ``numpy_rng``/``action_rng`` from those seeds rather than sharing a
        mutable global generator between workers.
        """

        # Global libraries receive the explicit user seed.  Named runtime
        # streams below still use snapshot-aware derived seeds so workers and
        # agents cannot accidentally share mutable global state.
        python_seed = self.root_seed
        numpy_seed = self.root_seed
        torch_seed = self.root_seed
        random.seed(python_seed)
        np.random.seed(numpy_seed % (MAX_NUMPY_SEED + 1))

        try:
            import torch
        except ImportError:  # pragma: no cover - optional for lightweight installs
            torch = None
        if torch is not None:
            torch.manual_seed(torch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(torch_seed)
                torch.backends.cudnn.deterministic = deterministic_torch
                torch.backends.cudnn.benchmark = not deterministic_torch
            if deterministic_torch:
                # ``warn_only`` keeps CPU-only and unsupported accelerator ops
                # usable while still selecting deterministic implementations.
                torch.use_deterministic_algorithms(True, warn_only=True)

        return SeedBundle(
            root_seed=self.root_seed,
            python_seed=python_seed,
            numpy_seed=numpy_seed,
            torch_seed=torch_seed,
            environment_seed=self.environment_seed("default"),
            action_seed=self.action_seed("default"),
        )

    def streams(
        self,
        *scope: Any,
        torch_device: str | None = None,
    ) -> RNGStreams:
        """Create independent Python, NumPy, environment, action, and Torch streams."""

        seed = self.derive("streams", *scope)
        python_seed = self.derive("streams", *scope, "python")
        numpy_seed = self.derive("streams", *scope, "numpy")
        environment_seed = self.derive("streams", *scope, "environment")
        action_seed = self.derive("streams", *scope, "action")
        torch_generator = None
        try:
            import torch
        except ImportError:  # pragma: no cover - optional for lightweight installs
            torch = None
        if torch is not None:
            try:
                torch_generator = torch.Generator(device=torch_device or "cpu")
                torch_generator.manual_seed(self.derive("streams", *scope, "torch"))
            except (RuntimeError, TypeError):
                # A CPU generator remains useful if a requested device is not
                # installed; global ``seed_all`` still handles CUDA when present.
                torch_generator = torch.Generator()
                torch_generator.manual_seed(self.derive("streams", *scope, "torch"))
        return RNGStreams(
            seed=seed,
            python=random.Random(python_seed),
            numpy=np.random.default_rng(numpy_seed),
            environment=np.random.default_rng(environment_seed),
            action=np.random.default_rng(action_seed),
            torch=torch_generator,
        )

    # Explicit aliases make the one public API easy to discover at call sites.
    rngs = streams

    def numpy_rng(self, *scope: Any) -> np.random.Generator:
        return self.streams(*scope).numpy

    def environment_rng(self, *scope: Any) -> np.random.Generator:
        return self.streams("environment", *scope).environment

    def action_rng(self, *scope: Any) -> np.random.Generator:
        return self.streams("action", *scope).action


def seed_everything(
    seed: int,
    *,
    snapshot: Any | None = None,
    deterministic_torch: bool = True,
) -> SeedManager:
    """Create a manager and seed all supported process-global RNG libraries."""

    manager = SeedManager(seed, snapshot=snapshot)
    manager.seed_all(deterministic_torch=deterministic_torch)
    return manager


# Common spelling used by training entry points.
set_global_seed = seed_everything


__all__ = [
    "MAX_NUMPY_SEED",
    "MAX_UINT64",
    "RNGStreams",
    "SeedBundle",
    "SeedManager",
    "derive_seed",
    "seed_everything",
    "set_global_seed",
]
