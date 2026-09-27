"""Focused TASK-3-2 tests for the replaceable sequence encoder contract."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import (
    EVENT_FEATURE_SIZE,
    GRUEncoder,
    SequenceEncoder,
    SequenceEncoderFactory,
    create_sequence_encoder,
)


def test_gru_default_architecture_and_single_stream_shape() -> None:
    encoder = GRUEncoder()
    sequence = torch.randn(64, EVENT_FEATURE_SIZE)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[-3:] = True

    output = encoder(sequence, mask)

    assert isinstance(encoder, SequenceEncoder)
    assert output.shape == (128,)
    assert output.dtype == torch.float32
    assert encoder.input_dim == EVENT_FEATURE_SIZE
    assert encoder.hidden_size == 128
    assert encoder.num_layers == 1
    assert encoder.batch_first is True
    assert encoder.dropout == 0.0


def test_mask_ignores_left_padding_and_interior_invalid_values() -> None:
    torch.manual_seed(22)
    valid_events = torch.randn(3, EVENT_FEATURE_SIZE)
    padded = torch.zeros(8, EVENT_FEATURE_SIZE)
    padded[-3:] = valid_events
    mask = torch.tensor([False, False, False, False, False, True, True, True])

    altered_padding = padded.clone()
    altered_padding[:5] = torch.randn(5, EVENT_FEATURE_SIZE)
    encoder = GRUEncoder(seed=11)

    expected = encoder(padded, mask)
    actual = encoder(altered_padding, mask)
    torch.testing.assert_close(expected, actual)

    interior = torch.randn(5, EVENT_FEATURE_SIZE)
    interior_mask = torch.tensor([True, False, True, False, True])
    compact = torch.stack((interior[0], interior[2], interior[4]))
    compact_output = encoder(compact)
    masked_output = encoder(interior, interior_mask)
    torch.testing.assert_close(compact_output, masked_output)


def test_empty_history_returns_zero_and_does_not_use_padding() -> None:
    encoder = GRUEncoder(seed=7)
    sequence = torch.randn(0, EVENT_FEATURE_SIZE)
    empty = encoder(sequence, torch.zeros(0, dtype=torch.bool))
    assert empty.shape == (128,)
    assert torch.count_nonzero(empty) == 0
    assert torch.count_nonzero(encoder([], [])) == 0

    padded = torch.randn(16, EVENT_FEATURE_SIZE)
    no_events = encoder(padded, torch.zeros(16, dtype=torch.bool))
    assert no_events.shape == (128,)
    assert torch.count_nonzero(no_events) == 0


def test_batch_and_preprocessor_stream_dimensions_are_supported() -> None:
    encoder = GRUEncoder(input_dim=EVENT_FEATURE_SIZE, hidden_size=24, seed=3)
    batch = torch.randn(4, 64, EVENT_FEATURE_SIZE)
    mask = torch.zeros(4, 64, dtype=torch.bool)
    mask[0, -1] = True
    mask[1, -2:] = True
    mask[2, :4] = True

    output = encoder(batch, mask)
    assert output.shape == (4, 24)
    assert torch.count_nonzero(output[3]) == 0

    stream_batch = torch.randn(2, 4, 64, EVENT_FEATURE_SIZE)
    stream_mask = torch.zeros(2, 4, 64, dtype=torch.bool)
    stream_mask[:, :, -1] = True
    stream_output = encoder(stream_batch, stream_mask)
    assert stream_output.shape == (2, 4, 24)


def test_configured_layers_batch_first_and_serializable_metadata() -> None:
    encoder = GRUEncoder(
        input_dim=EVENT_FEATURE_SIZE,
        hidden_size=13,
        num_layers=2,
        batch_first=False,
        dropout=0.25,
        seed=19,
    )
    sequence = torch.randn(6, 3, EVENT_FEATURE_SIZE)
    mask = torch.ones(6, 3, dtype=torch.bool)
    output = encoder(sequence, mask)
    assert output.shape == (3, 13)

    metadata = encoder.metadata()
    assert metadata["encoder_type"] == "GRU"
    assert metadata["architecture_version"] == "1.0"
    assert metadata["input_dim"] == EVENT_FEATURE_SIZE
    assert metadata["hidden_size"] == 13
    assert metadata["num_layers"] == 2
    assert metadata["batch_first"] is False
    assert metadata["dropout"] == pytest.approx(0.25)
    assert "role" not in json.dumps(metadata).lower()

    restored = GRUEncoder.from_dict(json.loads(encoder.to_json()))
    assert restored.metadata() == metadata


def test_initialization_and_inference_are_deterministic() -> None:
    first = GRUEncoder(hidden_size=17, seed=123)
    second = GRUEncoder(hidden_size=17, seed=123)
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter)

    sequence = np.arange(5 * EVENT_FEATURE_SIZE, dtype=np.float32).reshape(5, EVENT_FEATURE_SIZE)
    mask = np.array([False, False, True, True, True])
    first_output = first(sequence, mask)
    second_output = first(sequence, mask)
    torch.testing.assert_close(first_output, second_output)


def test_batch_mask_compaction_preserves_mixed_padding_and_all_invalid_outputs() -> None:
    torch.manual_seed(31)
    encoder = GRUEncoder(hidden_size=19, seed=31).eval()
    sequence = torch.randn(4, 9, EVENT_FEATURE_SIZE)
    mask = torch.tensor(
        [
            [False, False, True, True, False, True, False, False, False],
            [True, False, True, False, False, False, False, False, False],
            [False] * 9,
            [False, False, False, True, True, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    output = encoder(sequence, mask)
    expected = torch.stack(
        (
            encoder(sequence[0, [2, 3, 5]]),
            encoder(sequence[1, [0, 2]]),
            torch.zeros(19),
            encoder(sequence[3, 3:7]),
        )
    )
    torch.testing.assert_close(output, expected)
    assert output.shape == (4, 19)



    encoder = create_sequence_encoder("gru", input_dim=EVENT_FEATURE_SIZE, hidden_size=9)
    assert isinstance(encoder, SequenceEncoder)
    assert isinstance(SequenceEncoderFactory.create("GRU"), GRUEncoder)

    with pytest.raises(ValueError, match="unsupported sequence encoder"):
        SequenceEncoderFactory.create("CNN")


def test_malformed_sequence_and_mask_are_rejected() -> None:
    encoder = GRUEncoder()
    with pytest.raises(ValueError, match="feature dimension"):
        encoder(torch.zeros(3, EVENT_FEATURE_SIZE + 1))
    with pytest.raises(ValueError, match="valid_mask must have shape"):
        encoder(torch.zeros(3, EVENT_FEATURE_SIZE), torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="only 0 or 1"):
        encoder(torch.zeros(3, EVENT_FEATURE_SIZE), torch.tensor([0, 2, 1]))
    with pytest.raises(ValueError, match="finite"):
        bad = torch.zeros(3, EVENT_FEATURE_SIZE)
        bad[0, 0] = float("nan")
        encoder(bad)
    with pytest.raises(ValueError, match="feature dimension"):
        encoder([[1.0, 2.0]])


def test_constructor_rejects_invalid_architecture_values() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GRUEncoder(hidden_size=0)
    with pytest.raises(ValueError, match="dropout"):
        GRUEncoder(dropout=1.0)
    with pytest.raises(ValueError, match="disagree"):
        GRUEncoder(input_dim=8, input_size=9)
