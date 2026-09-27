"""Public-observation emotion inference for MainAgent.

The module predicts the emotion state of the four players other than the
viewer.  Its input is deliberately limited to the public event streams made
by :class:`ObservationPreprocessor`; DummyAgent's private emotion state,
labels, roles, rewards, and future events never cross this boundary.
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

from ...config import EmotionConfig
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


EMOTION_INFERENCE_FORMAT_VERSION = "1.0"
EMOTION_INFERENCE_ARCHITECTURE_VERSION = "1.0"
EMOTION_HEAD_HIDDEN_SIZE = 64
EMOTION_OPTIMIZER_METADATA = {
    "type": "Adam",
    "learning_rate": 1e-3,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
}
EMOTION_LOSS = "MSE"

# The model boundary accepts only public feature tensors.  These names are
# rejected before tensor conversion so labels cannot be accidentally treated
# as ignored auxiliary inputs.
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "roles",
        "role",
        "role_labels",
        "hidden_roles",
        "hidden_role",
        "true_roles",
        "true_role",
        "actual_roles",
        "actual_role",
        "privileged_state",
        "privileged_observation",
        "private_observation",
        "privileged_diagnostics",
        "emotion",
        "emotions",
        "emotion_state",
        "emotion_states",
        "emotion_label",
        "emotion_labels",
        "true_emotion",
        "true_emotions",
        "actual_emotion",
        "actual_emotions",
        "reward",
        "reward_breakdown",
        "action",
        "current_action",
        "future_event",
        "future_events",
        "next_event",
        "next_observation",
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


class EmotionInferenceModule(nn.Module):
    """Infer normalized continuous emotions from four public event streams.

    Accepted sequence shapes mirror ``IntentInferenceModule``:

    * ``(T, F)`` -> ``(E,)`` for one stream;
    * ``(4, T, F)`` -> ``(4, E)`` for the four target players;
    * ``(B, 4, T, F)`` -> ``(B, 4, E)`` for batched observations;
    * ``(B, T, F)`` -> ``(B, E)`` for independent streams.

    ``valid_mask`` identifies public events.  All-invalid streams use the
    configured ``initial_baseline`` instead of a learned value, while valid
    streams use the shared GRU and emotion head.
    """

    module_type = "EmotionInferenceModule"
    format_version = EMOTION_INFERENCE_FORMAT_VERSION
    architecture_version = EMOTION_INFERENCE_ARCHITECTURE_VERSION

    def __init__(
        self,
        input_dim: int = EVENT_FEATURE_SIZE,
        *,
        config: EmotionConfig | Mapping[str, Any] | None = None,
        emotion_config: EmotionConfig | Mapping[str, Any] | None = None,
        encoder: SequenceEncoder | None = None,
        encoder_type: str = "GRU",
        seed: int | None = DEFAULT_ENCODER_SEED,
        target_player_count: int = OTHER_PLAYER_COUNT,
        preprocessor: ObservationPreprocessor | None = None,
    ) -> None:
        super().__init__()
        self._validate_positive_int(input_dim, "input_dim")
        self._validate_positive_int(target_player_count, "target_player_count")
        if target_player_count != OTHER_PLAYER_COUNT:
            raise ValueError(f"target_player_count is fixed at {OTHER_PLAYER_COUNT}")
        if config is not None and emotion_config is not None:
            raise ValueError("provide only one of config or emotion_config")
        raw_config = config if config is not None else emotion_config
        self.config = self._coerce_config(raw_config)
        if preprocessor is not None and not isinstance(preprocessor, ObservationPreprocessor):
            raise TypeError("preprocessor must be an ObservationPreprocessor")
        self.preprocessor = preprocessor or ObservationPreprocessor()
        self._input_dim = input_dim
        self._target_player_count = target_player_count
        self._seed = self._validate_seed(seed)

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
            raise ValueError("EmotionInference head requires encoder output_dim=128")
        self.encoder = encoder

        if seed is None:
            self.emotion_head = nn.Sequential(
                nn.Linear(DEFAULT_GRU_HIDDEN_SIZE, EMOTION_HEAD_HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(EMOTION_HEAD_HIDDEN_SIZE, self.emotion_count),
                nn.Sigmoid(),
            )
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 1)
                self.emotion_head = nn.Sequential(
                    nn.Linear(DEFAULT_GRU_HIDDEN_SIZE, EMOTION_HEAD_HIDDEN_SIZE),
                    nn.ReLU(),
                    nn.Linear(EMOTION_HEAD_HIDDEN_SIZE, self.emotion_count),
                    nn.Sigmoid(),
                )

        self.register_buffer(
            "_initial_baseline",
            torch.full((self.emotion_count,), self.initial_baseline, dtype=torch.float32),
            persistent=False,
        )
        self._stream_sequence: Tensor | None = None
        self._stream_valid_mask: Tensor | None = None

    @staticmethod
    def _validate_positive_int(value: Any, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_seed(value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise TypeError("seed must be an integer or None")
        return value

    @staticmethod
    def _coerce_config(config: EmotionConfig | Mapping[str, Any] | None) -> EmotionConfig:
        if config is None:
            return EmotionConfig()
        if isinstance(config, EmotionConfig):
            return config
        if isinstance(config, Mapping):
            return EmotionConfig.model_validate(dict(config))
        raise TypeError("config must be an EmotionConfig, mapping, or None")

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def target_player_count(self) -> int:
        return self._target_player_count

    @property
    def emotion_count(self) -> int:
        return int(self.config.emotion_count)

    @property
    def emotion_schema_version(self) -> str:
        return self.config.schema_version

    @property
    def schema_version(self) -> str:
        """Alias for the applied Emotion Schema Version."""

        return self.emotion_schema_version

    @property
    def emotion_names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.config.emotions)

    @property
    def initial_baseline(self) -> float:
        return float(self.config.initial_baseline)

    @property
    def output_dim(self) -> int:
        return self.emotion_count

    @property
    def optimizer_metadata(self) -> dict[str, Any]:
        return _json_ready(EMOTION_OPTIMIZER_METADATA)

    @property
    def loss_name(self) -> str:
        return EMOTION_LOSS

    @property
    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "module_type": self.module_type,
            "architecture_version": self.architecture_version,
            "emotion_inference_format_version": self.format_version,
            "emotion_schema_version": self.emotion_schema_version,
            "emotion_count": self.emotion_count,
            "emotion_names": list(self.emotion_names),
            "initial_baseline": self.initial_baseline,
            "input_dim": self.input_dim,
            "target_player_count": self.target_player_count,
            "encoder": self.encoder.metadata(),
            "emotion_head": {
                "layers": [
                    {
                        "type": "Linear",
                        "in_features": DEFAULT_GRU_HIDDEN_SIZE,
                        "out_features": EMOTION_HEAD_HIDDEN_SIZE,
                    },
                    {"type": "ReLU"},
                    {
                        "type": "Linear",
                        "in_features": EMOTION_HEAD_HIDDEN_SIZE,
                        "out_features": self.emotion_count,
                    },
                    {"type": "Sigmoid"},
                ],
                "output": "continuous values in [0, 1]",
                "shared_across_target_players": True,
            },
            "loss": self.loss_name,
            "optimizer": self.optimizer_metadata,
            "emotion_config": self.config.model_dump(mode="json"),
            "input_contract": {
                "source": "public action history, attack/being-hit records, health changes, and alive state",
                "feature_count": self.input_dim,
                "valid_mask": "boolean/binary; false positions are ignored",
                "label_alignment": "DummyAgent state immediately before the action",
                "current_action_and_future_events": "forbidden",
                "hidden_true_role": "forbidden",
                "actual_emotion_state": "forbidden",
                "reward": "forbidden",
                "empty_history": "configured initial_baseline",
            },
            "preprocessor": self.preprocessor.metadata(),
        }

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe architecture and schema metadata."""

        return _json_ready(self.architecture_metadata)

    get_metadata = metadata

    def create_optimizer(self) -> torch.optim.Adam:
        """Create the fixed Phase-2 Adam optimizer without storing its state."""

        return torch.optim.Adam(
            self.parameters(),
            lr=EMOTION_OPTIMIZER_METADATA["learning_rate"],
            betas=tuple(EMOTION_OPTIMIZER_METADATA["betas"]),
            eps=EMOTION_OPTIMIZER_METADATA["eps"],
            weight_decay=EMOTION_OPTIMIZER_METADATA["weight_decay"],
        )

    def forward(
        self,
        inputs: Any,
        valid_mask: Any | None = None,
        *,
        output: str | None = None,
        stateful: bool = False,
    ) -> Tensor:
        """Return continuous emotion predictions for public input only.

        Train and eval modes intentionally share the same sigmoid output
        boundary.  ``output`` is accepted as an explicit semantic check and
        supports ``values``, ``predictions``, and ``probabilities`` aliases.
        """

        if output is not None and output not in {
            "values",
            "predictions",
            "probabilities",
            "probability",
            "probs",
            "sigmoid",
        }:
            raise ValueError("output must be 'values' or 'probabilities'")
        sequence, mask = self._extract_inputs(inputs, valid_mask)
        if stateful:
            sequence, mask = self._append_stream_state(sequence, mask)
        encoded = self.encoder(sequence, mask)
        predictions = self.emotion_head(encoded)
        nonempty = mask.any(dim=-1)
        baseline = self._initial_baseline.to(device=predictions.device, dtype=predictions.dtype)
        baseline = baseline.expand_as(predictions)
        predictions = torch.where(nonempty.unsqueeze(-1), predictions, baseline)
        # Sigmoid guarantees this mathematically; clamp only protects callers
        # using unusual floating-point dtypes at the public boundary.
        return predictions.clamp(0.0, 1.0)

    def predict_proba(
        self, inputs: Any, valid_mask: Any | None = None, *, stateful: bool = False
    ) -> Tensor:
        """Return inference values without changing the module mode."""

        return self.forward(inputs, valid_mask, output="probabilities", stateful=stateful)

    predict = predict_proba
    inference = predict_proba

    def emotion_loss(self, predictions: Tensor, targets: Any) -> Tensor:
        """Compute MSE against a separate, normalized pre-action label tensor."""

        values = predictions if isinstance(predictions, Tensor) else torch.as_tensor(predictions)
        if values.shape[-1] != self.emotion_count:
            raise ValueError(f"predictions final dimension must be {self.emotion_count}")
        labels = torch.as_tensor(targets, device=values.device, dtype=values.dtype)
        if tuple(labels.shape) != tuple(values.shape):
            raise ValueError(f"emotion targets must have shape {tuple(values.shape)}")
        if not torch.isfinite(labels).all():
            raise ValueError("emotion targets must contain only finite values")
        if torch.any((labels < 0.0) | (labels > 1.0)):
            raise ValueError("emotion targets must be in [0, 1]")
        return F.mse_loss(values, labels)

    mse_loss = emotion_loss
    loss = emotion_loss

    def reset_state(self) -> None:
        """Discard stateful public streams at an episode boundary."""

        self._stream_sequence = None
        self._stream_valid_mask = None

    reset = reset_state

    def get_state(self) -> dict[str, Tensor] | None:
        """Return a detached copy of the optional public stream buffer."""

        if self._stream_sequence is None or self._stream_valid_mask is None:
            return None
        return {
            "sequence": self._stream_sequence.detach().clone(),
            "valid_mask": self._stream_valid_mask.detach().clone(),
        }

    def set_state(self, state: Mapping[str, Any] | None) -> None:
        """Restore a stateful stream buffer after public shape validation."""

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
                "privileged role/emotion/action fields are not accepted: "
                + ", ".join(sorted(leaked))
            )

    def _normalise_sequence_and_mask(
        self, sequence: Any, valid_mask: Any | None
    ) -> tuple[Tensor, Tensor]:
        try:
            values = sequence if isinstance(sequence, Tensor) else torch.as_tensor(sequence)
        except (TypeError, ValueError) as exc:
            raise TypeError("event_history must be a numeric tensor or array") from exc
        if values.ndim == 1 and values.numel() == 0:
            values = values.reshape(0, self.input_dim)
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
            "emotion_schema_version": self.emotion_schema_version,
            "architecture": self.metadata(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    serialize = to_json

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EmotionInferenceModule":
        if not isinstance(value, Mapping):
            raise TypeError("emotion metadata must be a mapping")
        version = value.get("schema_version", value.get("emotion_inference_format_version"))
        if version != cls.format_version:
            raise ValueError(f"unsupported EmotionInference schema version: {version!r}")
        architecture = value.get("architecture", value)
        if not isinstance(architecture, Mapping):
            raise ValueError("emotion metadata architecture must be a mapping")
        if architecture.get("module_type", cls.module_type) != cls.module_type:
            raise ValueError("metadata does not describe EmotionInferenceModule")
        config_data = architecture.get("emotion_config")
        if config_data is None:
            config_data = {
                "schema_version": architecture.get("emotion_schema_version", "1.0"),
                "emotion_count": architecture.get("emotion_count", 1),
                "initial_baseline": architecture.get("initial_baseline", 0.0),
            }
        config = EmotionConfig.model_validate(config_data)
        if architecture.get("emotion_schema_version", config.schema_version) != config.schema_version:
            raise ValueError("emotion schema metadata does not match emotion config")
        encoder_metadata = architecture.get("encoder", {})
        if not isinstance(encoder_metadata, Mapping):
            raise ValueError("emotion metadata encoder must be a mapping")
        encoder_type = str(encoder_metadata.get("encoder_type", encoder_metadata.get("architecture", "GRU")))
        encoder = create_sequence_encoder(
            encoder_type,
            input_dim=architecture.get("input_dim", EVENT_FEATURE_SIZE),
            hidden_size=encoder_metadata.get("hidden_size", DEFAULT_GRU_HIDDEN_SIZE),
            num_layers=encoder_metadata.get("num_layers", DEFAULT_GRU_NUM_LAYERS),
            batch_first=encoder_metadata.get("batch_first", DEFAULT_GRU_BATCH_FIRST),
            dropout=encoder_metadata.get("dropout", DEFAULT_GRU_DROPOUT),
            seed=encoder_metadata.get("initialization_seed", DEFAULT_ENCODER_SEED),
        )
        return cls(
            input_dim=architecture.get("input_dim", EVENT_FEATURE_SIZE),
            config=config,
            encoder=encoder,
            seed=encoder_metadata.get("initialization_seed", DEFAULT_ENCODER_SEED),
            target_player_count=architecture.get("target_player_count", OTHER_PLAYER_COUNT),
        )

    from_metadata = from_dict

    @classmethod
    def from_json(cls, value: str) -> "EmotionInferenceModule":
        if not isinstance(value, str):
            raise TypeError("emotion metadata JSON must be a string")
        return cls.from_dict(json.loads(value))

    def save(self, path: str | Path) -> None:
        """Save weights plus versioned metadata, never optimizer state."""

        payload = {
            "schema_version": self.format_version,
            "module_type": self.module_type,
            "emotion_schema_version": self.emotion_schema_version,
            "metadata": self.to_dict(),
            "state_dict": self.state_dict(),
            "training_mode": self.training,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls, path: str | Path, *, map_location: Any = "cpu"
    ) -> "EmotionInferenceModule":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != cls.format_version:
            raise ValueError("unsupported or malformed EmotionInference artifact")
        module = cls.from_dict(payload["metadata"])
        if payload.get("emotion_schema_version") != module.emotion_schema_version:
            raise ValueError("Emotion Schema Version does not match artifact metadata")
        module.load_state_dict(payload["state_dict"])
        if bool(payload.get("training_mode", False)):
            module.train()
        else:
            module.eval()
        return module


EmotionModule = EmotionInferenceModule

__all__ = [
    "EMOTION_INFERENCE_ARCHITECTURE_VERSION",
    "EMOTION_INFERENCE_FORMAT_VERSION",
    "EMOTION_LOSS",
    "EMOTION_OPTIMIZER_METADATA",
    "EmotionInferenceModule",
    "EmotionModule",
]
