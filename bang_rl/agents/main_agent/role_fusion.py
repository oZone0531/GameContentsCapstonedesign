"""Public role-belief fusion with confirmed-role hard overrides.

``RoleFusion`` is intentionally stateless.  IntentInference produces a raw
role distribution from public event history; this boundary replaces only roles
that are already confirmed by the public observation.  It does not project
remaining role counts, construct role assignments, or inspect privileged truth.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # Torch is a project dependency, but keeping the boundary optional helps tooling.
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised only by documentation tooling.
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]


ROLE_FUSION_FORMAT_VERSION = "1.0"
ROLE_FUSION_ARCHITECTURE_VERSION = "1.0"
ROLE_COUNT = 5
PLAYER_COUNT = 5
TARGET_PLAYER_COUNT = 4
HIDDEN_ROLE_SENTINEL = 255
SHERIFF_ROLE = 0

_PRIVILEGED_KEYS = frozenset(
    {
        "roles",
        "role",
        "role_labels",
        "hidden_roles",
        "hidden_role",
        "hidden_role_labels",
        "true_roles",
        "true_role",
        "actual_roles",
        "actual_role",
        "private_roles",
        "privileged_roles",
        "privileged_state",
        "privileged_observation",
        "private_observation",
        "privileged_diagnostics",
        "emotion_label",
        "emotion_labels",
        "reward",
        "reward_breakdown",
    }
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


class RoleFusion:
    """Apply public confirmed-role constraints to IntentInference output.

    The canonical model output is ``(4, 5)`` for the four players other than
    the viewer.  Batched ``(B, 4, 5)`` streams are also accepted.  Full-slot
    ``(5, 5)``/``(B, 5, 5)`` inputs are supported when a caller needs to retain
    the viewer row; this does not change the canonical four-target contract.

    ``raw_role_distribution`` may be probabilities or logits.  Probabilities
    are returned normalized (without changing already-normalized rows), while
    logits are converted with a stable softmax by default.  Set
    ``output_type="logits"`` to retain unresolved logits exactly and represent
    a hard override as ``0`` for the confirmed role and ``-inf`` elsewhere.
    """

    module_type = "RoleFusion"
    format_version = ROLE_FUSION_FORMAT_VERSION
    architecture_version = ROLE_FUSION_ARCHITECTURE_VERSION

    def metadata(self) -> dict[str, Any]:
        """Return deterministic, JSON-safe compatibility metadata."""

        return {
            "module_type": self.module_type,
            "architecture_version": self.architecture_version,
            "role_fusion_format_version": self.format_version,
            "role_count": ROLE_COUNT,
            "player_count": PLAYER_COUNT,
            "target_player_count": TARGET_PLAYER_COUNT,
            "input_contract": {
                "raw_role_distribution": "IntentInference logits or probabilities",
                "public_fields": [
                    "self_role",
                    "sheriff_idx",
                    "role_visibility",
                    "revealed_roles",
                    "confirmed_role_distribution",
                ],
                "hidden_true_role": "forbidden",
                "privileged_role_inputs": "forbidden",
            },
            "override_sources": [
                "self_role",
                "sheriff_idx",
                "revealed_roles",
                "confirmed_role_distribution",
            ],
            "output_contract": {
                "canonical_shape": [TARGET_PLAYER_COUNT, ROLE_COUNT],
                "batched_shape": ["B", TARGET_PLAYER_COUNT, ROLE_COUNT],
                "unresolved": "raw distribution, normalized only",
                "confirmed": "role one-hot",
            },
            "role_count_projection": "not_implemented",
        }

    get_metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.format_version,
            "module_type": self.module_type,
            "architecture": self.metadata(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    serialize = to_json

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoleFusion":
        if not isinstance(value, Mapping):
            raise TypeError("RoleFusion metadata must be a mapping")
        version = value.get("schema_version", value.get("role_fusion_format_version"))
        if version != cls.format_version:
            raise ValueError(f"unsupported RoleFusion schema version: {version!r}")
        architecture = value.get("architecture", value)
        if not isinstance(architecture, Mapping) or architecture.get("module_type", cls.module_type) != cls.module_type:
            raise ValueError("metadata does not describe RoleFusion")
        if architecture.get("role_count", ROLE_COUNT) != ROLE_COUNT:
            raise ValueError("RoleFusion role_count is fixed at 5")
        if architecture.get("target_player_count", TARGET_PLAYER_COUNT) != TARGET_PLAYER_COUNT:
            raise ValueError("RoleFusion target_player_count is fixed at 4")
        return cls()

    @classmethod
    def from_json(cls, value: str) -> "RoleFusion":
        if not isinstance(value, str):
            raise TypeError("RoleFusion metadata JSON must be a string")
        return cls.from_dict(json.loads(value))

    def save(self, path: str | Path) -> None:
        """Persist only the stateless, versioned fusion contract."""

        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RoleFusion":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def fuse(
        self,
        raw_role_distribution: Any,
        role_visibility: Any | None = None,
        revealed_roles: Any | None = None,
        confirmed_role_distribution: Any | None = None,
        *,
        self_role: Any | None = None,
        sheriff_idx: Any | None = None,
        viewer_idx: Any | None = None,
        target_player_indices: Any | None = None,
        input_type: str = "auto",
        output_type: str = "probabilities",
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Fuse raw intent output with public confirmed roles.

        A canonical public observation mapping may be supplied as the first
        argument.  In that form it must contain ``raw_role_distribution`` (or
        ``intent_distribution``/``role_distribution``) and may contain the
        public role fields.  No privileged role field is accepted.
        """

        values: Mapping[str, Any] | None = None
        if isinstance(raw_role_distribution, Mapping):
            values = raw_role_distribution
            self._reject_privileged(values)
            raw_role_distribution = self._first_present(
                values, "raw_role_distribution", "intent_distribution", "role_distribution"
            )
            if raw_role_distribution is None:
                raise ValueError("public fusion mapping is missing raw_role_distribution")
        if metadata is not None:
            self._reject_privileged(metadata)
            values = {**(dict(values) if values is not None else {}), **dict(metadata)}

        def choose(explicit: Any | None, name: str) -> Any | None:
            if explicit is not None:
                return explicit
            return None if values is None else values.get(name)

        role_visibility = choose(role_visibility, "role_visibility")
        revealed_roles = choose(revealed_roles, "revealed_roles")
        confirmed_role_distribution = choose(
            confirmed_role_distribution, "confirmed_role_distribution"
        )
        self_role = choose(self_role, "self_role")
        sheriff_idx = choose(sheriff_idx, "sheriff_idx")
        viewer_idx = choose(viewer_idx, "viewer_idx")
        if viewer_idx is None and values is not None:
            viewer_idx = values.get("current_player_idx")

        raw, raw_kind, original_kind, was_torch = self._raw_array(raw_role_distribution, input_type)
        stream_count, batch_shape = self._stream_layout(raw.shape)
        public_shape = batch_shape + (PLAYER_COUNT,)
        visibility = self._public_int_array(
            role_visibility, public_shape, "role_visibility", default=0
        )
        revealed = self._public_int_array(
            revealed_roles,
            public_shape,
            "revealed_roles",
            default=HIDDEN_ROLE_SENTINEL,
        )
        confirmed = self._public_float_array(
            confirmed_role_distribution,
            batch_shape + (PLAYER_COUNT, ROLE_COUNT),
            "confirmed_role_distribution",
            default=0.0,
        )
        self._validate_visibility(visibility)
        self._validate_revealed(revealed)
        self._validate_confirmed(confirmed)

        viewers = self._batch_scalar_array(viewer_idx, batch_shape, "viewer_idx", default=0)
        sheriffs = self._batch_scalar_array(sheriff_idx, batch_shape, "sheriff_idx", default=None)
        self_roles = self._batch_role_array(self_role, batch_shape, "self_role")
        target_slots = self._target_slots(
            stream_count, batch_shape, viewers, target_player_indices
        )

        probabilities = self._as_probabilities(raw, raw_kind)
        fused = probabilities.copy()
        known_by_batch = self._known_roles(
            batch_shape, visibility, revealed, confirmed, viewers, sheriffs, self_roles
        )
        for batch_index in np.ndindex(batch_shape) if batch_shape else [()]:
            known = known_by_batch[batch_index]
            slots = target_slots[batch_index]
            for stream_index, slot in enumerate(slots):
                role = known[slot]
                if role < 0:
                    continue
                fused[batch_index + (stream_index,)] = self._one_hot(role)

        if output_type not in {"probabilities", "probability", "probs", "logits", "same"}:
            raise ValueError("output_type must be 'probabilities', 'logits', or 'same'")
        resolved_output = (
            original_kind if output_type == "same" else output_type
        )
        if resolved_output in {"probabilities", "probability", "probs"}:
            result = fused
        else:
            result = self._logit_output(raw, fused, raw_kind, known_by_batch, target_slots)

        if was_torch:
            assert torch is not None
            source = raw_role_distribution if isinstance(raw_role_distribution, Tensor) else None
            dtype = source.dtype if source is not None and source.dtype.is_floating_point else torch.float32
            return torch.as_tensor(result, dtype=dtype, device=source.device if source is not None else None)
        source_array = np.asarray(raw_role_distribution)
        dtype = source_array.dtype if np.issubdtype(source_array.dtype, np.floating) else np.dtype(np.float32)
        return np.asarray(result, dtype=dtype)

    # Alias used by callers that describe the layer as a confirmed-role gate.
    apply = fuse
    forward = fuse

    @staticmethod
    def _first_present(values: Mapping[str, Any], *names: str) -> Any | None:
        for name in names:
            if name in values:
                return values[name]
        return None

    @staticmethod
    def _reject_privileged(values: Mapping[str, Any]) -> None:
        leaked = []
        for key in values:
            normalized = str(key).lower()
            if normalized in _PRIVILEGED_KEYS or normalized.endswith("_true_role"):
                leaked.append(str(key))
        if leaked:
            raise ValueError(
                "privileged role inputs are not accepted: " + ", ".join(sorted(leaked))
            )

    @staticmethod
    def _raw_array(value: Any, input_type: str) -> tuple[np.ndarray, str, str, bool]:
        if input_type not in {"auto", "probabilities", "probability", "probs", "logits"}:
            raise ValueError("input_type must be 'auto', 'probabilities', or 'logits'")
        was_torch = torch is not None and isinstance(value, Tensor)
        if was_torch:
            array = value.detach().cpu().numpy()
        else:
            try:
                array = np.asarray(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("raw_role_distribution must be numeric") from exc
        if array.ndim not in {1, 2, 3} or array.shape[-1] != ROLE_COUNT:
            raise ValueError("raw_role_distribution must have shape (5,), (N,5), or (B,N,5)")
        if array.ndim == 3 and array.shape[-2] not in {TARGET_PLAYER_COUNT, PLAYER_COUNT}:
            raise ValueError("raw_role_distribution stream dimension must be 4 or 5")
        if array.ndim == 2 and array.shape[-2] not in {TARGET_PLAYER_COUNT, PLAYER_COUNT}:
            raise ValueError("raw_role_distribution stream dimension must be 4 or 5")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            raise TypeError("raw_role_distribution must contain numeric values")
        array = array.astype(np.float64, copy=False)
        if not np.all(np.isfinite(array)):
            raise ValueError("raw_role_distribution must contain finite values")
        if input_type == "auto":
            nonnegative = np.all(array >= 0.0, axis=-1)
            normalized = np.isclose(np.sum(array, axis=-1), 1.0, atol=1e-5)
            if np.all(nonnegative & normalized):
                kind = "probabilities"
            else:
                kind = "logits"
        else:
            kind = "probabilities" if input_type in {"probabilities", "probability", "probs"} else "logits"
        return array, kind, kind, was_torch

    @staticmethod
    def _stream_layout(shape: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        if len(shape) == 1:
            return 1, ()
        if len(shape) == 2:
            if shape[0] not in {TARGET_PLAYER_COUNT, PLAYER_COUNT}:
                return 1, (shape[0],)
            return shape[0], ()
        return shape[-2], shape[:-2]

    @staticmethod
    def _public_int_array(value: Any | None, shape: tuple[int, ...], name: str, *, default: int) -> np.ndarray:
        if value is None:
            return np.full(shape, default, dtype=np.int64)
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
            raise TypeError(f"{name} must contain integers")
        return array.astype(np.int64, copy=False)

    @staticmethod
    def _public_float_array(value: Any | None, shape: tuple[int, ...], name: str, *, default: float) -> np.ndarray:
        if value is None:
            return np.full(shape, default, dtype=np.float64)
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            raise TypeError(f"{name} must contain numbers")
        result = array.astype(np.float64, copy=False)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain finite values")
        return result

    @staticmethod
    def _validate_visibility(value: np.ndarray) -> None:
        if np.any((value < 0) | (value > 3)):
            raise ValueError("role_visibility must contain values in [0, 3]")

    @staticmethod
    def _validate_revealed(value: np.ndarray) -> None:
        if np.any((value != HIDDEN_ROLE_SENTINEL) & ((value < 0) | (value >= ROLE_COUNT))):
            raise ValueError("revealed_roles must contain role indices or 255")

    @staticmethod
    def _validate_confirmed(value: np.ndarray) -> None:
        if np.any((value < 0.0) | (value > 1.0)):
            raise ValueError("confirmed_role_distribution must be in [0, 1]")
        sums = value.sum(axis=-1)
        if not np.all(np.isclose(sums, 0.0) | np.isclose(sums, 1.0)):
            raise ValueError("confirmed_role_distribution rows must be zero or normalized")
        nonzero = np.sum(~np.isclose(value, 0.0), axis=-1)
        if np.any((sums > 1e-6) & (nonzero != 1)):
            raise ValueError("confirmed_role_distribution must be zero or one-hot")

    @staticmethod
    def _batch_scalar_array(value: Any | None, batch_shape: tuple[int, ...], name: str, *, default: int | None) -> np.ndarray:
        if not batch_shape:
            if value is None:
                return np.asarray(default if default is not None else -1, dtype=np.int64)
            array = np.asarray(value)
            if array.shape not in [(), (1,)]:
                raise ValueError(f"{name} must be a scalar")
            item = array.reshape(-1)[0].item()
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise TypeError(f"{name} must contain integers")
            return np.asarray(int(item), dtype=np.int64)
        if value is None:
            return np.full(batch_shape, default if default is not None else -1, dtype=np.int64)
        array = np.asarray(value)
        if array.shape == ():
            array = np.full(batch_shape, array.item())
        if array.shape != batch_shape:
            raise ValueError(f"{name} must have shape {batch_shape}, got {array.shape}")
        if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
            raise TypeError(f"{name} must contain integers")
        return array.astype(np.int64, copy=False)

    @staticmethod
    def _batch_role_array(value: Any | None, batch_shape: tuple[int, ...], name: str) -> np.ndarray:
        if value is None:
            return np.full(batch_shape, -1, dtype=np.int64)
        result = RoleFusion._batch_scalar_array(value, batch_shape, name, default=-1)
        if np.any((result != -1) & ((result < 0) | (result >= ROLE_COUNT))):
            raise ValueError(f"{name} must contain role indices in [0, 4]")
        return result

    @staticmethod
    def _target_slots(
        stream_count: int,
        batch_shape: tuple[int, ...],
        viewers: np.ndarray,
        explicit: Any | None,
    ) -> np.ndarray:
        if explicit is not None:
            array = np.asarray(explicit)
            expected = batch_shape + (stream_count,)
            if array.shape != expected:
                raise ValueError(f"target_player_indices must have shape {expected}, got {array.shape}")
            if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
                raise TypeError("target_player_indices must contain integers")
            slots = array.astype(np.int64, copy=False)
        else:
            slots = np.empty(batch_shape + (stream_count,), dtype=np.int64)
            for index in np.ndindex(batch_shape) if batch_shape else [()]:
                viewer = int(viewers[index])
                if not 0 <= viewer < PLAYER_COUNT:
                    raise ValueError("viewer_idx must be in [0, 4]")
                if stream_count == PLAYER_COUNT:
                    slots[index] = np.arange(PLAYER_COUNT)
                elif stream_count == TARGET_PLAYER_COUNT:
                    slots[index] = np.asarray(
                        [slot for slot in range(PLAYER_COUNT) if slot != viewer], dtype=np.int64
                    )
                else:
                    slots[index] = viewer
        if np.any((slots < 0) | (slots >= PLAYER_COUNT)):
            raise ValueError("target_player_indices must contain values in [0, 4]")
        for index in np.ndindex(batch_shape) if batch_shape else [()]:
            if len(set(slots[index].tolist())) != stream_count:
                raise ValueError("target_player_indices must not contain duplicate slots")
        return slots

    @staticmethod
    def _known_roles(
        batch_shape: tuple[int, ...],
        visibility: np.ndarray,
        revealed: np.ndarray,
        confirmed: np.ndarray,
        viewers: np.ndarray,
        sheriffs: np.ndarray,
        self_roles: np.ndarray,
    ) -> dict[tuple[int, ...], np.ndarray]:
        result: dict[tuple[int, ...], np.ndarray] = {}
        for index in np.ndindex(batch_shape) if batch_shape else [()]:
            known = np.full(PLAYER_COUNT, -1, dtype=np.int64)
            sources: dict[int, str] = {}

            def assign(slot: int, role: int, source: str) -> None:
                if not 0 <= slot < PLAYER_COUNT:
                    raise ValueError(f"{source} player index must be in [0, 4]")
                if not 0 <= role < ROLE_COUNT:
                    raise ValueError(f"{source} role must be in [0, 4]")
                if known[slot] >= 0 and known[slot] != role:
                    raise ValueError(
                        f"conflicting confirmed roles for player {slot}: "
                        f"{sources[slot]}={known[slot]} and {source}={role}"
                    )
                known[slot] = role
                sources[slot] = source

            sheriff = int(sheriffs[index])
            if sheriff >= 0:
                assign(sheriff, SHERIFF_ROLE, "sheriff_idx")
            viewer = int(viewers[index])
            self_role = int(self_roles[index])
            if self_role >= 0:
                assign(viewer, self_role, "self_role")
            for slot, role in enumerate(revealed[index]):
                role = int(role)
                if role != HIDDEN_ROLE_SENTINEL:
                    assign(slot, role, f"revealed_roles[{slot}]")
            for slot, row in enumerate(confirmed[index]):
                if np.isclose(float(row.sum()), 0.0):
                    continue
                assign(slot, int(np.argmax(row)), f"confirmed_role_distribution[{slot}]")

            confirmed_slots = [slot for slot, role in enumerate(known) if role >= 0]
            confirmed_roles = [int(known[slot]) for slot in confirmed_slots]
            if len(set(confirmed_roles)) != len(confirmed_roles):
                raise ValueError("conflicting confirmed roles: one role is assigned to multiple players")
            result[index] = known
        return result

    @staticmethod
    def _as_probabilities(raw: np.ndarray, kind: str) -> np.ndarray:
        if kind == "probabilities":
            if np.any(raw < 0.0):
                raise ValueError("probability role distributions must be non-negative")
            sums = raw.sum(axis=-1, keepdims=True)
            if np.any(sums <= 0.0):
                raise ValueError("probability role distributions must have positive row sums")
            normalized = raw / sums
            already = np.isclose(sums, 1.0, atol=1e-5)
            return np.where(already, raw, normalized)
        shifted = raw - np.max(raw, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=-1, keepdims=True)

    @staticmethod
    def _one_hot(role: int) -> np.ndarray:
        result = np.zeros(ROLE_COUNT, dtype=np.float64)
        result[role] = 1.0
        return result

    @staticmethod
    def _logit_output(
        raw: np.ndarray,
        probabilities: np.ndarray,
        raw_kind: str,
        known_by_batch: dict[tuple[int, ...], np.ndarray],
        target_slots: np.ndarray,
    ) -> np.ndarray:
        result = raw.copy() if raw_kind == "logits" else np.log(np.clip(probabilities, 1e-45, 1.0))
        for index in np.ndindex(target_slots.shape[:-1]) if target_slots.ndim > 1 else [()]:
            known = known_by_batch[index]
            for stream_index, slot in enumerate(target_slots[index]):
                role = int(known[int(slot)])
                if role >= 0:
                    result[index + (stream_index,)] = -np.inf
                    result[index + (stream_index, role)] = 0.0
        if target_slots.ndim == 1 and target_slots.shape[0] == 1 and raw.ndim == 1:
            role = int(known_by_batch[()][int(target_slots[0])])
            if role >= 0:
                result[:] = -np.inf
                result[role] = 0.0
        return result


__all__ = [
    "HIDDEN_ROLE_SENTINEL",
    "PLAYER_COUNT",
    "ROLE_COUNT",
    "ROLE_FUSION_ARCHITECTURE_VERSION",
    "ROLE_FUSION_FORMAT_VERSION",
    "RoleFusion",
    "SHERIFF_ROLE",
    "TARGET_PLAYER_COUNT",
]
