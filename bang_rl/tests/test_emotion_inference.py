"""Focused TASK-3-5 tests for public emotion inference."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import (
    EMOTION_LOSS,
    EVENT_FEATURE_SIZE,
    EmotionInferenceModule,
)
from bang_rl.config import EmotionConfig


def _streams(batch: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    shape = (4, 64, EVENT_FEATURE_SIZE) if batch is None else (batch, 4, 64, EVENT_FEATURE_SIZE)
    sequence = np.linspace(-1.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape[:-1], dtype=np.bool_)
    if batch is None:
        mask[:, -3:] = True
    else:
        mask[:, :, -3:] = True
    return sequence, mask


def test_configurable_emotion_count_outputs_normalized_four_player_vectors() -> None:
    sequence, mask = _streams()
    config = EmotionConfig(
        schema_version="2.1",
        emotion_count=3,
        initial_baseline=0.2,
        emotions=["calm", "fear", "anger"],
    )
    module = EmotionInferenceModule(config=config, seed=17).eval()

    output = module(sequence, mask)

    assert output.shape == (4, 3)
    assert output.dtype == torch.float32
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_empty_history_uses_configured_baseline_and_valid_mask() -> None:
    sequence, mask = _streams()
    mask[:] = False
    sequence[:] = 99.0
    module = EmotionInferenceModule(
        config=EmotionConfig(emotion_count=2, initial_baseline=0.35), seed=3
    ).eval()

    output = module(sequence, mask)

    torch.testing.assert_close(output, torch.full((4, 2), 0.35))


def test_train_and_eval_share_sigmoid_boundary_and_mse_loss() -> None:
    sequence, mask = _streams()
    module = EmotionInferenceModule(config=EmotionConfig(emotion_count=2), seed=3)
    module.train()
    training_output = module(sequence, mask)
    module.eval()
    inference_output = module(sequence, mask)

    assert torch.all((training_output >= 0.0) & (training_output <= 1.0))
    assert torch.all((inference_output >= 0.0) & (inference_output <= 1.0))
    torch.testing.assert_close(training_output, inference_output)
    target = torch.full_like(inference_output, 0.25)
    assert module.loss(inference_output, target).item() == pytest.approx(
        torch.mean((inference_output - target) ** 2).item()
    )
    assert module.loss_name == EMOTION_LOSS == "MSE"


def test_architecture_and_optimizer_metadata_use_fixed_literals() -> None:
    module = EmotionInferenceModule(config=EmotionConfig(emotion_count=3), seed=0)
    metadata = module.metadata()

    assert metadata["module_type"] == "EmotionInferenceModule"
    assert metadata["emotion_schema_version"] == "1.0"
    assert metadata["emotion_count"] == 3
    assert metadata["encoder"]["hidden_size"] == 128
    assert metadata["encoder"]["num_layers"] == 1
    assert metadata["encoder"]["batch_first"] is True
    assert metadata["encoder"]["dropout"] == 0.0
    assert metadata["emotion_head"]["layers"] == [
        {"type": "Linear", "in_features": 128, "out_features": 64},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 64, "out_features": 3},
        {"type": "Sigmoid"},
    ]
    assert metadata["optimizer"] == {
        "type": "Adam",
        "learning_rate": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    assert metadata["loss"] == "MSE"
    assert metadata["input_contract"]["actual_emotion_state"] == "forbidden"


def test_deterministic_initialization_and_inference_without_rng_side_effect() -> None:
    sequence, mask = _streams()
    torch.manual_seed(101)
    before = torch.random.get_rng_state()
    first = EmotionInferenceModule(config=EmotionConfig(emotion_count=2), seed=29).eval()
    first_output = first(sequence, mask)
    after = torch.random.get_rng_state()
    second_output = EmotionInferenceModule(config=EmotionConfig(emotion_count=2), seed=29).eval()(sequence, mask)

    torch.testing.assert_close(before, after)
    torch.testing.assert_close(first_output, second_output)


def test_batch_and_stateful_stream_shapes_are_supported() -> None:
    sequence, mask = _streams(batch=3)
    module = EmotionInferenceModule(config=EmotionConfig(emotion_count=2), seed=5).eval()

    batch = module(sequence, mask)
    one = module(sequence[0], mask[0])
    independent = module(sequence[:, 0], mask[:, 0])
    assert batch.shape == (3, 4, 2)
    assert one.shape == (4, 2)
    assert independent.shape == (3, 2)
    torch.testing.assert_close(batch[0], one)

    module.reset_state()
    isolated = module(sequence[0, :, -2:], mask[0, :, -2:], stateful=True)
    module.reset_state()
    module(sequence[0], mask[0], stateful=True)
    continued = module(sequence[0, :, -2:], mask[0, :, -2:], stateful=True)
    assert not torch.equal(isolated, continued)
    module.reset_state()
    replayed = module(sequence[0, :, -2:], mask[0, :, -2:], stateful=True)
    torch.testing.assert_close(isolated, replayed)


def test_public_mapping_rejects_labels_actions_and_future_or_private_state() -> None:
    sequence, mask = _streams()
    module = EmotionInferenceModule(seed=13).eval()
    public = {"event_history": sequence, "event_valid_mask": mask}
    expected = module(public)
    torch.testing.assert_close(expected, module(sequence, mask))

    for forbidden in ("emotion_label", "emotion_state", "true_role", "action", "future_events"):
        private = copy.deepcopy(public)
        private[forbidden] = np.asarray([0])
        with pytest.raises(ValueError, match="privileged|fields"):
            module(private)


def test_mse_rejects_misaligned_or_out_of_range_targets() -> None:
    sequence, mask = _streams()
    module = EmotionInferenceModule(config=EmotionConfig(emotion_count=2), seed=1).eval()
    predictions = module(sequence, mask)

    with pytest.raises(ValueError, match="shape"):
        module.loss(predictions, np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="\[0, 1\]"):
        module.loss(predictions, np.full((4, 2), 1.1, dtype=np.float32))


def test_schema_versioned_persistence_restores_weights_metadata_and_mode(tmp_path) -> None:
    sequence, mask = _streams()
    config = EmotionConfig(schema_version="3.0", emotion_count=2, initial_baseline=0.15)
    original = EmotionInferenceModule(config=config, seed=23).eval()
    expected = original(sequence, mask)
    artifact = tmp_path / "emotion.inference.pt"

    original.save(artifact)
    restored = EmotionInferenceModule.load(artifact)

    assert restored.training is False
    assert restored.to_dict() == original.to_dict()
    torch.testing.assert_close(restored(sequence, mask), expected)
    with pytest.raises(ValueError, match="schema version"):
        EmotionInferenceModule.from_dict({"schema_version": "0.0"})


def test_config_schema_changes_are_reflected_in_model_contract() -> None:
    first = EmotionInferenceModule(config=EmotionConfig(schema_version="1.0", emotion_count=1))
    second = EmotionInferenceModule(config=EmotionConfig(schema_version="2.0", emotion_count=4))

    assert first.emotion_schema_version == "1.0"
    assert second.emotion_schema_version == "2.0"
    assert first.output_dim == 1
    assert second.output_dim == 4
