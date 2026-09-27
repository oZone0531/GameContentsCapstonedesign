"""Public-observation role inference for MainAgent.

The intent model is deliberately small and explicit: a replaceable
:class:`SequenceEncoder` (GRU by default) encodes one public event stream per
other player, and one shared role head maps the encoded representation to the
five role logits.  This module never accepts or derives hidden true roles.

``forward`` follows a useful train/inference boundary.  In training mode it
returns logits for ``CrossEntropyLoss``; in evaluation mode it returns softmax
probabilities.  Callers can explicitly request either representation with the
``output`` argument.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoder import (
    DEFAULT_ENCODER_SEED,
    DEFAULT_GRU_BATCH_FIRST,
    DEFAULT_GRU_DROPOUT,
    DEFAULT_GRU_HIDDEN_SIZE,
    DEFAULT_GRU_NUM_LAYERS,
    SequenceEncoder,
    create_sequence_encoder,
)
from .preprocessor import (
    EVENT_FEATURE_SIZE,
    OTHER_PLAYER_COUNT,
    ObservationPreprocessor,
    ProcessedObservation,
)
from .role_fusion import RoleFusion


INTENT_INFERENCE_FORMAT_VERSION = "1.0"
INTENT_INFERENCE_ARCHITECTURE_VERSION = "1.0"
ROLE_COUNT = 5
TARGET_PLAYER_COUNT = OTHER_PLAYER_COUNT
ROLE_HEAD_HIDDEN_SIZE = 64
INTENT_OPTIMIZER_METADATA = {
    "type": "Adam",
    "learning_rate": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
}

# These names are checked before any tensor conversion.  In particular, a
# caller cannot smuggle a privileged role label through an otherwise public
# mapping and rely on it being ignored by the network.
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "roles",
        "role",
        "role_labels",
        "hidden_roles",
        "hidden_role",
        "true_roles",
        "true_role",
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
    """Convert metadata values to deterministic JSON-compatible values."""

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


class IntentInferenceModule(nn.Module):
    """Infer a role distribution from four public player event streams.

    Accepted sequence shapes are:

    * ``(T, F)`` — one stream, returning ``(5,)``;
    * ``(4, T, F)`` — the four other-player streams, returning ``(4, 5)``;
    * ``(B, 4, T, F)`` — a batch of observations, returning ``(B, 4, 5)``.

    A ``(B, T, F)`` input is also accepted as a batch of independent streams
    and returns ``(B, 5)``.  ``valid_mask`` follows the sequence shape without
    its feature dimension.  Invalid positions are ignored by the encoder,
    including interior masked positions.
    """

    module_type = "IntentInferenceModule"
    format_version = INTENT_INFERENCE_FORMAT_VERSION
    architecture_version = INTENT_INFERENCE_ARCHITECTURE_VERSION

    def __init__(
        self,
        input_dim: int = EVENT_FEATURE_SIZE,
        *,
        encoder: SequenceEncoder | None = None,
        encoder_type: str = "GRU",
        seed: int | None = DEFAULT_ENCODER_SEED,
        target_player_count: int = TARGET_PLAYER_COUNT,
        role_count: int = ROLE_COUNT,
        preprocessor: ObservationPreprocessor | None = None,
    ) -> None:
        super().__init__()
        self._validate_positive_int(input_dim, "input_dim")
        self._validate_positive_int(target_player_count, "target_player_count")
        self._validate_positive_int(role_count, "role_count")
        if target_player_count != TARGET_PLAYER_COUNT:
            raise ValueError(f"target_player_count is fixed at {TARGET_PLAYER_COUNT}")
        if role_count != ROLE_COUNT:
            raise ValueError(f"role_count is fixed at {ROLE_COUNT}")
        if preprocessor is not None and not isinstance(preprocessor, ObservationPreprocessor):
            raise TypeError("preprocessor must be an ObservationPreprocessor")
        self._input_dim = input_dim
        self._target_player_count = target_player_count
        self._role_count = role_count
        self._seed = seed
        self.preprocessor = preprocessor or ObservationPreprocessor()

        if encoder is None:
            encoder = create_sequence_encoder(
                encoder_type,
                input_dim=input_dim,
                hidden_size=DEFAULT_GRU_HIDDEN_SIZE,
                num_layers=DEFAULT_GRU_NUM_LAYERS,
                batch_first=DEFAULT_GRU_BATCH_FIRST,
                dropout=DEFAULT_GRU_DROPOUT,
                seed=seed,
            )
        if not isinstance(encoder, SequenceEncoder):
            raise TypeError("encoder must implement SequenceEncoder")
        if encoder.input_dim != input_dim:
            raise ValueError("encoder input dimension does not match input_dim")
        if encoder.output_dim != DEFAULT_GRU_HIDDEN_SIZE:
            raise ValueError("IntentInference role head requires encoder output_dim=128")
        self.encoder = encoder

        # GRUEncoder isolates its own initialization.  Isolate the role-head
        # initialization as well, so two modules with the same seed are byte
        # reproducible without changing the caller's global RNG state.
        if seed is None:
            self.role_head = nn.Sequential(
                nn.Linear(DEFAULT_GRU_HIDDEN_SIZE, ROLE_HEAD_HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(ROLE_HEAD_HIDDEN_SIZE, role_count),
            )
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 1)
                self.role_head = nn.Sequential(
                    nn.Linear(DEFAULT_GRU_HIDDEN_SIZE, ROLE_HEAD_HIDDEN_SIZE),
                    nn.ReLU(),
                    nn.Linear(ROLE_HEAD_HIDDEN_SIZE, role_count),
                )

        self._stream_sequence: Tensor | None = None
        self._stream_valid_mask: Tensor | None = None

    @staticmethod
    def _validate_positive_int(value: Any, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def target_player_count(self) -> int:
        return self._target_player_count

    @property
    def role_count(self) -> int:
        return self._role_count

    @property
    def output_dim(self) -> int:
        return self._role_count

    @property
    def optimizer_metadata(self) -> dict[str, Any]:
        return _json_ready(INTENT_OPTIMIZER_METADATA)

    @property
    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "module_type": self.module_type,
            "architecture_version": self.architecture_version,
            "intent_inference_format_version": self.format_version,
            "input_dim": self.input_dim,
            "target_player_count": self.target_player_count,
            "role_count": self.role_count,
            "encoder": self.encoder.metadata(),
            "role_head": {
                "layers": [
                    {
                        "type": "Linear",
                        "in_features": DEFAULT_GRU_HIDDEN_SIZE,
                        "out_features": ROLE_HEAD_HIDDEN_SIZE,
                    },
                    {"type": "ReLU"},
                    {
                        "type": "Linear",
                        "in_features": ROLE_HEAD_HIDDEN_SIZE,
                        "out_features": self.role_count,
                    },
                ],
                "output": "logits in train mode; softmax probabilities in eval mode",
                "shared_across_target_players": True,
            },
            "optimizer": self.optimizer_metadata,
            "input_contract": {
                "source": "public event history only",
                "feature_count": self.input_dim,
                "valid_mask": "boolean/binary; false positions are ignored",
                "hidden_true_role": "forbidden",
                "confirmed_role_fusion": "deferred to RoleFusion",
            },
            "preprocessor": self.preprocessor.metadata(),
            "role_fusion": RoleFusion().metadata(),
        }

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe architecture metadata for a phase-2 artifact."""

        return _json_ready(self.architecture_metadata)

    get_metadata = metadata

    def create_optimizer(self) -> torch.optim.Adam:
        """Create the fixed phase-2 optimizer without storing its state."""

        return torch.optim.Adam(
            self.parameters(),
            lr=INTENT_OPTIMIZER_METADATA["learning_rate"],
            betas=tuple(INTENT_OPTIMIZER_METADATA["betas"]),
            eps=INTENT_OPTIMIZER_METADATA["eps"],
            weight_decay=INTENT_OPTIMIZER_METADATA["weight_decay"],
        )

    def forward(
        self,
        inputs: Any,
        valid_mask: Any | None = None,
        *,
        output: str | None = None,
        stateful: bool = False,
    ) -> Tensor:
        """Encode public streams and return logits or probabilities.

        By default, training mode returns logits and evaluation mode returns
        probabilities.  ``output`` may explicitly be ``"logits"`` or
        ``"probabilities"`` without changing the module's mode.
        """

        sequence, mask = self._extract_inputs(inputs, valid_mask)
        if stateful:
            sequence, mask = self._append_stream_state(sequence, mask)
        encoded = self.encoder(sequence, mask)
        logits = self.role_head(encoded)
        selected = "logits" if output is None and self.training else (output or "probabilities")
        if selected == "logits":
            return logits
        if selected in {"probabilities", "probs", "probability"}:
            return F.softmax(logits, dim=-1)
        raise ValueError("output must be 'logits' or 'probabilities'")

    def logits(self, inputs: Any, valid_mask: Any | None = None, *, stateful: bool = False) -> Tensor:
        """Return role logits regardless of train/eval mode."""

        return self.forward(inputs, valid_mask, output="logits", stateful=stateful)

    def predict_proba(
        self, inputs: Any, valid_mask: Any | None = None, *, stateful: bool = False
    ) -> Tensor:
        """Return role probabilities regardless of train/eval mode."""

        return self.forward(inputs, valid_mask, output="probabilities", stateful=stateful)

    predict = predict_proba
    inference = predict_proba

    def role_loss(self, logits: Tensor, role_labels: Any) -> Tensor:
        """Compute supervised role Cross-Entropy from logits.

        Labels are training targets, not model input.  They are deliberately
        accepted only by this explicit training helper, never by ``forward``.
        """

        values = logits if isinstance(logits, Tensor) else torch.as_tensor(logits)
        if values.shape[-1] != self.role_count:
            raise ValueError(f"logits final dimension must be {self.role_count}")
        labels = torch.as_tensor(role_labels, device=values.device)
        expected = values.shape[:-1]
        if tuple(labels.shape) != tuple(expected):
            raise ValueError(f"role_labels must have shape {tuple(expected)}")
        if labels.dtype == torch.bool or labels.dtype.is_floating_point:
            raise TypeError("role_labels must contain integer role indices")
        if torch.any((labels < 0) | (labels >= self.role_count)):
            raise ValueError("role_labels contain an invalid role index")
        return F.cross_entropy(values, labels.to(dtype=torch.long))

    def reset_state(self) -> None:
        """Discard accumulated stream input at an episode boundary."""

        self._stream_sequence = None
        self._stream_valid_mask = None

    reset = reset_state

    def get_state(self) -> dict[str, Tensor] | None:
        """Return a detached copy of the optional stateful stream buffer."""

        if self._stream_sequence is None or self._stream_valid_mask is None:
            return None
        return {
            "sequence": self._stream_sequence.detach().clone(),
            "valid_mask": self._stream_valid_mask.detach().clone(),
        }

    def set_state(self, state: Mapping[str, Any] | None) -> None:
        """Restore a stateful stream buffer after validating its public shape."""

        if state is None:
            self.reset_state()
            return
        if not isinstance(state, Mapping) or "sequence" not in state or "valid_mask" not in state:
            raise ValueError("state must contain sequence and valid_mask")
        sequence, mask = self._normalise_sequence_and_mask(state["sequence"], state["valid_mask"])
        if sequence.ndim != 3 or sequence.shape[0] not in {1, self.target_player_count}:
            raise ValueError("state must represent one stream or four target-player streams")
        self._stream_sequence = sequence.detach().clone()
        self._stream_valid_mask = mask.detach().clone()

    def _append_stream_state(self, sequence: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if sequence.ndim == 2:
            current = sequence.unsqueeze(0)
            current_mask = mask.unsqueeze(0)
        elif sequence.ndim == 3 and sequence.shape[0] in {1, self.target_player_count}:
            current = sequence
            current_mask = mask
        else:
            raise ValueError("stateful input must be (T,F) or (4,T,F)")
        if self._stream_sequence is None:
            combined = current
            combined_mask = current_mask
        else:
            if self._stream_sequence.shape[0] != current.shape[0]:
                raise ValueError("stateful stream count changed; call reset_state first")
            combined = torch.cat((self._stream_sequence, current), dim=1)
            combined_mask = torch.cat((self._stream_valid_mask, current_mask), dim=1)
        self._stream_sequence = combined.detach().clone()
        self._stream_valid_mask = combined_mask.detach().clone()
        return combined, combined_mask

    def _extract_inputs(self, inputs: Any, valid_mask: Any | None) -> tuple[Tensor, Tensor]:
        if isinstance(inputs, ProcessedObservation):
            sequence_value = inputs.event_history
            mask_value = inputs.event_valid_mask if valid_mask is None else valid_mask
        elif isinstance(inputs, Mapping) or (
            hasattr(inputs, "keys") and hasattr(inputs, "__getitem__")
        ):
            values = self._mapping(inputs)
            self._reject_privileged(values)
            if "event_history" in values or "action_history" in values:
                sequence_value = values.get("event_history", values.get("action_history"))
                mask_value = values.get("event_valid_mask", values.get("valid_mask"))
                if valid_mask is not None:
                    mask_value = valid_mask
            else:
                # A canonical environment observation is converted by the
                # preprocessor, which performs the complete public-field and
                # event validation before model input is formed.
                processed = self.preprocessor.preprocess(values)
                if not isinstance(processed, ProcessedObservation):
                    raise ValueError("batched canonical observations require explicit preprocessing")
                sequence_value = processed.event_history
                mask_value = processed.event_valid_mask if valid_mask is None else valid_mask
        else:
            sequence_value = inputs
            mask_value = valid_mask
        return self._normalise_sequence_and_mask(sequence_value, mask_value)

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        return {key: value[key] for key in value.keys()}

    @staticmethod
    def _reject_privileged(values: Mapping[str, Any]) -> None:
        leaked = []
        for key in values:
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_INPUT_KEYS or normalized.endswith("_true_role"):
                leaked.append(str(key))
        if leaked:
            raise ValueError(
                "privileged role/reward/emotion fields are not accepted: "
                + ", ".join(sorted(leaked))
            )

    def _normalise_sequence_and_mask(
        self, sequence: Any, valid_mask: Any | None
    ) -> tuple[Tensor, Tensor]:
        try:
            values = sequence if isinstance(sequence, Tensor) else torch.as_tensor(sequence)
        except (TypeError, ValueError) as exc:
            raise TypeError("event_history must be a numeric tensor or array") from exc
        if values.ndim not in {2, 3, 4}:
            raise ValueError("event_history must have shape (T,F), (B,T,F), (4,T,F), or (B,4,T,F)")
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"event_history feature dimension must be {self.input_dim}, got {values.shape[-1]}"
            )
        if values.ndim == 4 and values.shape[1] != self.target_player_count:
            raise ValueError(f"event_history player dimension must be {self.target_player_count}")
        if values.ndim == 3 and values.shape[0] == 0:
            raise ValueError("event_history must contain at least one stream")
        if valid_mask is None:
            mask = torch.ones(values.shape[:-1], dtype=torch.bool)
        else:
            try:
                mask = valid_mask if isinstance(valid_mask, Tensor) else torch.as_tensor(valid_mask)
            except (TypeError, ValueError) as exc:
                raise TypeError("valid_mask must be a boolean or binary tensor") from exc
            if tuple(mask.shape) != tuple(values.shape[:-1]):
                raise ValueError(
                    f"valid_mask must have shape {tuple(values.shape[:-1])}, got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                if mask.dtype.is_floating_point and not torch.isfinite(mask).all():
                    raise ValueError("valid_mask must contain only 0 or 1")
                if not (torch.is_floating_point(mask) or mask.dtype in {
                    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
                }):
                    raise TypeError("valid_mask must be boolean or numeric")
                if not bool(((mask == 0) | (mask == 1)).all()):
                    raise ValueError("valid_mask must contain only 0 or 1")
                mask = mask.to(dtype=torch.bool)
        return values, mask

    def to_dict(self) -> dict[str, Any]:
        """Return schema-versioned metadata, excluding weights and optimizer state."""

        return {
            "schema_version": self.format_version,
            "module_type": self.module_type,
            "architecture": self.metadata(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    serialize = to_json

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IntentInferenceModule":
        if not isinstance(value, Mapping):
            raise TypeError("intent metadata must be a mapping")
        version = value.get("schema_version", value.get("intent_inference_format_version"))
        if version != cls.format_version:
            raise ValueError(f"unsupported IntentInference schema version: {version!r}")
        architecture = value.get("architecture", value)
        if not isinstance(architecture, Mapping):
            raise ValueError("intent metadata architecture must be a mapping")
        if architecture.get("module_type", cls.module_type) != cls.module_type:
            raise ValueError("metadata does not describe IntentInferenceModule")
        encoder_metadata = architecture.get("encoder", {})
        encoder_type = str(encoder_metadata.get("encoder_type", encoder_metadata.get("architecture", "GRU")))
        encoder = create_sequence_encoder(encoder_type, **{
            "input_dim": architecture.get("input_dim", EVENT_FEATURE_SIZE),
            "hidden_size": encoder_metadata.get("hidden_size", DEFAULT_GRU_HIDDEN_SIZE),
            "num_layers": encoder_metadata.get("num_layers", DEFAULT_GRU_NUM_LAYERS),
            "batch_first": encoder_metadata.get("batch_first", DEFAULT_GRU_BATCH_FIRST),
            "dropout": encoder_metadata.get("dropout", DEFAULT_GRU_DROPOUT),
            "seed": encoder_metadata.get("initialization_seed", DEFAULT_ENCODER_SEED),
        })
        return cls(
            input_dim=architecture.get("input_dim", EVENT_FEATURE_SIZE),
            encoder=encoder,
            seed=encoder_metadata.get("initialization_seed", DEFAULT_ENCODER_SEED),
            target_player_count=architecture.get("target_player_count", TARGET_PLAYER_COUNT),
            role_count=architecture.get("role_count", ROLE_COUNT),
        )

    from_metadata = from_dict

    @classmethod
    def from_json(cls, value: str) -> "IntentInferenceModule":
        if not isinstance(value, str):
            raise TypeError("intent metadata JSON must be a string")
        return cls.from_dict(json.loads(value))

    def save(self, path: str | Path) -> None:
        """Save weights plus versioned architecture metadata, never optimizer state."""

        payload = {
            "schema_version": self.format_version,
            "module_type": self.module_type,
            "metadata": self.to_dict(),
            "state_dict": self.state_dict(),
            "training_mode": self.training,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, map_location: Any = "cpu") -> "IntentInferenceModule":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != cls.format_version:
            raise ValueError("unsupported or malformed IntentInference artifact")
        module = cls.from_dict(payload["metadata"])
        module.load_state_dict(payload["state_dict"])
        if bool(payload.get("training_mode", False)):
            module.train()
        else:
            module.eval()
        return module


# Naming aliases keep the module discoverable to consumers using either the
# design's full name or the shorter MainAgent terminology.
IntentModule = IntentInferenceModule

__all__ = [
    "INTENT_INFERENCE_ARCHITECTURE_VERSION",
    "INTENT_INFERENCE_FORMAT_VERSION",
    "INTENT_OPTIMIZER_METADATA",
    "IntentInferenceModule",
    "IntentModule",
    "ROLE_COUNT",
    "TARGET_PLAYER_COUNT",
]
