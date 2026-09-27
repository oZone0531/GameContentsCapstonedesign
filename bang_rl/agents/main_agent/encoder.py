"""Mask-aware sequence encoders used by MainAgent inference modules.

The preprocessor emits left-padded, fixed-length public event streams.  This
module keeps the model boundary independent from that implementation detail:
``GRUEncoder`` consumes a sequence and a validity mask, removes invalid
positions before recurrent processing, and returns the final valid hidden state.
Only the public event feature tensor is accepted; role labels and other
privileged values have no path into the encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import json
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from .preprocessor import EVENT_FEATURE_SIZE


SEQUENCE_ENCODER_FORMAT_VERSION = "1.0"
GRU_ENCODER_ARCHITECTURE_VERSION = "1.0"
DEFAULT_GRU_HIDDEN_SIZE = 128
DEFAULT_GRU_NUM_LAYERS = 1
DEFAULT_GRU_BATCH_FIRST = True
DEFAULT_GRU_DROPOUT = 0.0
DEFAULT_ENCODER_SEED = 0


class SequenceEncoder(nn.Module, ABC):
    """Interface for replaceable sequence encoders.

    Implementations return one representation per input stream.  Inputs may be
    unbatched ``(T, F)``, batched ``(B, T, F)``, or have additional leading
    stream dimensions ``(..., T, F)``.  The matching validity mask has shape
    ``(..., T)``.  Invalid positions are padding and must not affect the
    returned representation.
    """

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Number of features in each timestep."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Size of one encoded stream representation."""

    @abstractmethod
    def forward(self, sequence: Any, valid_mask: Any | None = None) -> Tensor:
        """Encode a sequence and return its final valid hidden representation."""

    @property
    @abstractmethod
    def architecture_metadata(self) -> dict[str, Any]:
        """Return JSON-safe architecture and input-contract metadata."""

    def metadata(self) -> dict[str, Any]:
        """Return a detached, serializable description of this encoder."""

        return dict(self.architecture_metadata)

    get_metadata = metadata


class GRUEncoder(SequenceEncoder):
    """GRU encoder with padding/mask-safe final-hidden-state extraction.

    The default values are the fixed MainAgent architecture from Design §8.1.
    Constructor values remain explicit so a compatible future configuration or
    a replacement encoder can be tested without changing the interface.
    """

    encoder_type = "GRU"
    architecture_version = GRU_ENCODER_ARCHITECTURE_VERSION

    def __init__(
        self,
        input_dim: int = EVENT_FEATURE_SIZE,
        *,
        input_size: int | None = None,
        hidden_size: int = DEFAULT_GRU_HIDDEN_SIZE,
        num_layers: int = DEFAULT_GRU_NUM_LAYERS,
        batch_first: bool = DEFAULT_GRU_BATCH_FIRST,
        dropout: float = DEFAULT_GRU_DROPOUT,
        seed: int | None = DEFAULT_ENCODER_SEED,
        initialization_seed: int | None = None,
    ) -> None:
        super().__init__()

        if input_size is not None:
            if input_dim != EVENT_FEATURE_SIZE and input_dim != input_size:
                raise ValueError("input_dim and input_size disagree")
            input_dim = input_size
        self._input_dim = self._positive_int(input_dim, "input_dim")
        self._hidden_size = self._positive_int(hidden_size, "hidden_size")
        self._num_layers = self._positive_int(num_layers, "num_layers")
        if not isinstance(batch_first, bool):
            raise TypeError("batch_first must be a bool")
        self._batch_first = batch_first
        if isinstance(dropout, bool) or not isinstance(dropout, (float, int)):
            raise TypeError("dropout must be a number in [0, 1)")
        self._dropout = float(dropout)
        if not 0.0 <= self._dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if initialization_seed is not None:
            if seed not in (DEFAULT_ENCODER_SEED, initialization_seed):
                raise ValueError("seed and initialization_seed disagree")
            seed = initialization_seed
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")
        self._initialization_seed = seed

        # Isolate initialization from the caller's RNG.  A default seed makes
        # architecture construction reproducible, while callers can pass None
        # when they intentionally want ambient PyTorch RNG initialization.
        if seed is None:
            self.gru = nn.GRU(
                input_size=self._input_dim,
                hidden_size=self._hidden_size,
                num_layers=self._num_layers,
                batch_first=self._batch_first,
                dropout=self._dropout,
            )
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.gru = nn.GRU(
                    input_size=self._input_dim,
                    hidden_size=self._hidden_size,
                    num_layers=self._num_layers,
                    batch_first=self._batch_first,
                    dropout=self._dropout,
                )

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def input_size(self) -> int:
        """PyTorch-compatible alias for :attr:`input_dim`."""

        return self._input_dim

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def output_dim(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def batch_first(self) -> bool:
        return self._batch_first

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def initialization_seed(self) -> int | None:
        return self._initialization_seed

    @property
    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "encoder_type": self.encoder_type,
            "architecture": self.encoder_type,
            "architecture_version": self.architecture_version,
            "sequence_encoder_format_version": SEQUENCE_ENCODER_FORMAT_VERSION,
            "input_dim": self.input_dim,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "batch_first": self.batch_first,
            "dropout": self.dropout,
            "initialization_seed": self.initialization_seed,
            "input_feature_schema": {
                "feature_count": self.input_dim,
                "valid_mask": {
                    "dtype": "bool",
                    "semantics": "true positions are public valid events",
                },
                "padding": "invalid positions are ignored before GRU processing",
                "output": "last valid hidden state; all-invalid streams are zeros",
            },
        }

    @property
    def config(self) -> dict[str, Any]:
        """Alias for the constructor configuration used by artifact writers."""

        return self.metadata()

    def to_dict(self) -> dict[str, Any]:
        return self.metadata()

    def to_json(self) -> str:
        return json.dumps(self.metadata(), sort_keys=True, separators=(",", ":"))

    serialize = to_json

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GRUEncoder":
        if not isinstance(value, Mapping):
            raise TypeError("encoder metadata must be a mapping")
        expected = value.get("encoder_type", value.get("architecture", "GRU"))
        if str(expected).upper() != cls.encoder_type:
            raise ValueError(f"unsupported encoder type: {expected!r}")
        version = value.get("architecture_version", cls.architecture_version)
        if version != cls.architecture_version:
            raise ValueError(f"unsupported GRU architecture version: {version!r}")
        return cls(
            input_dim=value.get("input_dim", value.get("input_size", EVENT_FEATURE_SIZE)),
            hidden_size=value.get("hidden_size", DEFAULT_GRU_HIDDEN_SIZE),
            num_layers=value.get("num_layers", DEFAULT_GRU_NUM_LAYERS),
            batch_first=value.get("batch_first", DEFAULT_GRU_BATCH_FIRST),
            dropout=value.get("dropout", DEFAULT_GRU_DROPOUT),
            seed=value.get("initialization_seed", DEFAULT_ENCODER_SEED),
        )

    from_metadata = from_dict

    def forward(self, sequence: Any, valid_mask: Any | None = None) -> Tensor:
        """Return one final hidden state for each stream in ``sequence``.

        ``valid_mask`` is intentionally required by the model contract for
        padded input but defaults to all-valid for callers with compact
        sequences.  It may be boolean or binary integer/float data and must
        match the sequence's leading dimensions and time dimension.
        """

        values, original_leading_shape, was_unbatched = self._normalise_sequence(sequence)
        mask = self._normalise_mask(valid_mask, values, original_leading_shape, was_unbatched)
        values = values.to(device=self._module_device(), dtype=self._module_dtype())
        mask = mask.to(device=values.device)
        if not torch.isfinite(values).all():
            raise ValueError("sequence must contain only finite numbers")

        batch_size, sequence_length, _ = values.shape
        if sequence_length == 0:
            encoded = torch.zeros(
                (batch_size, self.output_dim), device=values.device, dtype=values.dtype
            )
        else:
            lengths = mask.sum(dim=1, dtype=torch.long)
            nonempty = lengths > 0
            if not bool(nonempty.any()):
                encoded = torch.zeros(
                    (batch_size, self.output_dim), device=values.device, dtype=values.dtype
                )
            else:
                max_valid_length = int(lengths.max().item())
                # Compact all streams with one indexed tensor assignment.  A
                # mask's cumulative valid-position rank preserves event order
                # for left padding and arbitrary interior padding without a
                # Python loop over the batch.
                compact = torch.zeros(
                    (batch_size, max_valid_length, self.input_dim),
                    device=values.device,
                    dtype=values.dtype,
                )
                valid_rows, valid_times = torch.nonzero(mask, as_tuple=True)
                valid_ranks = mask.cumsum(dim=1)[valid_rows, valid_times] - 1
                compact[valid_rows, valid_ranks] = values[valid_rows, valid_times]
                packed = pack_padded_sequence(
                    compact,
                    lengths.clamp_min(1).cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, hidden = self.gru(packed)
                encoded = hidden[-1]
                encoded = encoded * nonempty.to(dtype=encoded.dtype).unsqueeze(-1)

        if was_unbatched:
            return encoded[0]
        return encoded.reshape(*original_leading_shape, self.output_dim)

    def _normalise_sequence(self, sequence: Any) -> tuple[Tensor, tuple[int, ...], bool]:
        try:
            values = sequence if isinstance(sequence, Tensor) else torch.as_tensor(sequence)
        except (TypeError, ValueError) as exc:
            raise TypeError("sequence must be a numeric tensor or array") from exc
        if values.ndim < 2:
            if values.ndim == 1 and values.numel() == 0:
                return values.reshape(1, 0, self.input_dim), (), True
            raise ValueError("sequence must have shape (T, F) or (..., T, F)")
        if values.ndim == 2:
            if values.shape[-1] != self.input_dim:
                raise ValueError(
                    f"sequence feature dimension must be {self.input_dim}, got {values.shape[-1]}"
                )
            return values.unsqueeze(0), (), True
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"sequence feature dimension must be {self.input_dim}, got {values.shape[-1]}"
            )
        if self.batch_first:
            leading_shape = tuple(int(item) for item in values.shape[:-2])
            return values.reshape(-1, values.shape[-2], values.shape[-1]), leading_shape, False
        # With batch_first=False the public 3-D convention is (T, B, F).
        if values.ndim != 3:
            raise ValueError("batch_first=False accepts an unbatched (T, F) or 3-D (T, B, F) sequence")
        leading_shape = (int(values.shape[1]),)
        return values.transpose(0, 1), leading_shape, False

    def _normalise_mask(
        self,
        valid_mask: Any | None,
        values: Tensor,
        original_leading_shape: tuple[int, ...],
        was_unbatched: bool,
    ) -> Tensor:
        if valid_mask is None:
            return torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        try:
            mask = valid_mask if isinstance(valid_mask, Tensor) else torch.as_tensor(valid_mask)
        except (TypeError, ValueError) as exc:
            raise TypeError("valid_mask must be a boolean or binary numeric tensor") from exc
        if was_unbatched:
            if mask.ndim != 1 or mask.shape[0] != values.shape[1]:
                raise ValueError("valid_mask must have shape (T,) for an unbatched sequence")
            mask = mask.unsqueeze(0)
        elif self.batch_first:
            expected = (*original_leading_shape, values.shape[1])
            if tuple(mask.shape) != expected:
                raise ValueError(f"valid_mask must have shape {expected}, got {tuple(mask.shape)}")
            mask = mask.reshape(-1, values.shape[1])
        else:
            expected = (values.shape[1], original_leading_shape[0])
            if tuple(mask.shape) != expected:
                raise ValueError(f"valid_mask must have shape {expected}, got {tuple(mask.shape)}")
            mask = mask.transpose(0, 1)
        if mask.dtype == torch.bool:
            return mask.to(device=values.device)
        if not (
            torch.is_floating_point(mask)
            or mask.dtype
            in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        ):
            raise TypeError("valid_mask must be boolean or numeric")
        if torch.is_floating_point(mask) and not torch.isfinite(mask).all():
            raise ValueError("valid_mask must contain only 0 or 1")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("valid_mask must contain only 0 or 1")
        return mask.to(device=values.device, dtype=torch.bool)

    def _module_device(self) -> torch.device:
        return next(self.parameters()).device

    def _module_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


EncoderConstructor = Callable[..., SequenceEncoder]


class SequenceEncoderFactory:
    """Registry for selecting an encoder without coupling callers to GRU."""

    _constructors: dict[str, EncoderConstructor] = {"GRU": GRUEncoder}

    @classmethod
    def register(cls, name: str, constructor: EncoderConstructor) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("encoder name must be a non-empty string")
        if not callable(constructor):
            raise TypeError("encoder constructor must be callable")
        cls._constructors[name.upper()] = constructor

    @classmethod
    def create(cls, name: str = "GRU", **kwargs: Any) -> SequenceEncoder:
        if not isinstance(name, str):
            raise TypeError("encoder name must be a string")
        try:
            constructor = cls._constructors[name.upper()]
        except KeyError as exc:
            supported = ", ".join(sorted(cls._constructors))
            raise ValueError(f"unsupported sequence encoder {name!r}; choose from {supported}") from exc
        encoder = constructor(**kwargs)
        if not isinstance(encoder, SequenceEncoder):
            raise TypeError("registered encoder must implement SequenceEncoder")
        return encoder


create_sequence_encoder = SequenceEncoderFactory.create


__all__ = [
    "DEFAULT_ENCODER_SEED",
    "DEFAULT_GRU_BATCH_FIRST",
    "DEFAULT_GRU_DROPOUT",
    "DEFAULT_GRU_HIDDEN_SIZE",
    "DEFAULT_GRU_NUM_LAYERS",
    "GRU_ENCODER_ARCHITECTURE_VERSION",
    "GRUEncoder",
    "SEQUENCE_ENCODER_FORMAT_VERSION",
    "SequenceEncoder",
    "SequenceEncoderFactory",
    "create_sequence_encoder",
]
