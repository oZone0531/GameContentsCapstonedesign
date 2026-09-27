"""Focused TASK-3-3 tests for public role intent inference."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import (
    EVENT_FEATURE_SIZE,
    IntentInferenceModule,
    ObservationPreprocessor,
)


def _streams(batch: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    shape = (4, 64, EVENT_FEATURE_SIZE) if batch is None else (batch, 4, 64, EVENT_FEATURE_SIZE)
    sequence = np.linspace(-1.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    mask_shape = shape[:-1]
    mask = np.zeros(mask_shape, dtype=np.bool_)
    if batch is None:
        mask[:, -3:] = True
    else:
        mask[:, :, -3:] = True
    return sequence, mask


def test_eval_outputs_four_role_distributions_with_probability_normalization() -> None:
    sequence, mask = _streams()
    module = IntentInferenceModule(seed=17)
    module.eval()

    probabilities = module(sequence, mask)

    assert probabilities.shape == (4, 5)
    assert probabilities.dtype == torch.float32
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4))


def test_train_returns_logits_and_explicit_probability_boundary() -> None:
    sequence, mask = _streams()
    module = IntentInferenceModule(seed=3)
    module.train()

    logits = module(sequence, mask)
    probabilities = module.predict_proba(sequence, mask)

    assert logits.shape == (4, 5)
    assert probabilities.shape == (4, 5)
    torch.testing.assert_close(probabilities, torch.softmax(logits, dim=-1))
    assert module.role_head[0].in_features == 128
    assert module.role_head[0].out_features == 64
    assert module.role_head[2].in_features == 64
    assert module.role_head[2].out_features == 5


def test_same_seed_and_input_are_deterministic_without_global_rng_side_effect() -> None:
    sequence, mask = _streams()
    torch.manual_seed(101)
    before = torch.random.get_rng_state()
    first = IntentInferenceModule(seed=29).eval()(sequence, mask)
    after = torch.random.get_rng_state()
    second = IntentInferenceModule(seed=29).eval()(sequence, mask)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(before, after)


def test_batch_and_single_stream_shapes_are_supported() -> None:
    sequence, mask = _streams(batch=3)
    module = IntentInferenceModule(seed=5).eval()

    batch = module(sequence, mask)
    one = module(sequence[0], mask[0])
    independent = module(sequence[:, 0], mask[:, 0])

    assert batch.shape == (3, 4, 5)
    assert one.shape == (4, 5)
    assert independent.shape == (3, 5)
    torch.testing.assert_close(batch[0], one)


def test_masked_values_do_not_change_role_distribution() -> None:
    sequence, mask = _streams()
    altered = sequence.copy()
    altered[:, :61] = 99.0
    module = IntentInferenceModule(seed=11).eval()

    expected = module(sequence, mask)
    actual = module(altered, mask)

    torch.testing.assert_close(expected, actual)


def test_state_reset_prevents_cross_episode_stream_leakage() -> None:
    first, first_mask = _streams()
    second = first[:, -2:]
    second_mask = first_mask[:, -2:]
    module = IntentInferenceModule(seed=7).eval()

    module.reset_state()
    isolated = module(second, second_mask, stateful=True)
    module.reset_state()
    replayed = module(second, second_mask, stateful=True)
    torch.testing.assert_close(isolated, replayed)

    module.reset_state()
    module(first, first_mask, stateful=True)
    continued = module(second, second_mask, stateful=True)
    assert not torch.equal(isolated, continued)
    module.reset_state()
    reset_again = module(second, second_mask, stateful=True)
    torch.testing.assert_close(isolated, reset_again)


def test_architecture_optimizer_and_schema_metadata_are_literal_and_public_only() -> None:
    metadata = IntentInferenceModule(seed=0).metadata()

    assert metadata["module_type"] == "IntentInferenceModule"
    assert metadata["architecture_version"] == "1.0"
    assert metadata["encoder"]["hidden_size"] == 128
    assert metadata["encoder"]["num_layers"] == 1
    assert metadata["encoder"]["batch_first"] is True
    assert metadata["encoder"]["dropout"] == 0.0
    assert metadata["role_head"]["layers"] == [
        {"type": "Linear", "in_features": 128, "out_features": 64},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 64, "out_features": 5},
    ]
    assert metadata["optimizer"] == {
        "type": "Adam",
        "learning_rate": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    assert "role" not in metadata["input_contract"]["source"]
    assert metadata["input_contract"]["hidden_true_role"] == "forbidden"


def test_schema_versioned_persistence_restores_weights_and_mode(tmp_path) -> None:
    sequence, mask = _streams()
    original = IntentInferenceModule(seed=23).eval()
    expected = original(sequence, mask)
    artifact = tmp_path / "intent.inference.pt"

    original.save(artifact)
    restored = IntentInferenceModule.load(artifact)

    assert restored.training is False
    assert restored.to_dict() == original.to_dict()
    torch.testing.assert_close(restored(sequence, mask), expected)

    with pytest.raises(ValueError, match="schema version"):
        IntentInferenceModule.from_dict({"schema_version": "0.0"})


def test_public_mapping_and_privileged_role_inputs_are_enforced() -> None:
    sequence, mask = _streams()
    public = {"event_history": sequence, "event_valid_mask": mask}
    module = IntentInferenceModule(seed=13).eval()
    expected = module(public)
    torch.testing.assert_close(expected, module(sequence, mask))

    private = copy.deepcopy(public)
    private["true_roles"] = np.asarray([0, 1, 2, 3, 4], dtype=np.uint8)
    with pytest.raises(ValueError, match="privileged"):
        module(private)

    malformed = {"event_history": sequence, "event_valid_mask": np.ones((4, 63), dtype=np.bool_)}
    with pytest.raises(ValueError, match="valid_mask must have shape"):
        module(malformed)


def test_canonical_observation_uses_preprocessor_and_rejects_hidden_role() -> None:
    beliefs = np.zeros((5, 5), dtype=np.float32)
    beliefs[0, 0] = 1.0
    observation = {
        "current_player_idx": np.asarray([0], dtype=np.uint8),
        "environment_step": np.asarray([0], dtype=np.uint32),
        "round_index": np.asarray([0], dtype=np.uint32),
        "healths": np.asarray([5, 4, 4, 4, 4], dtype=np.uint8),
        "alive": np.asarray([True] * 5, dtype=np.bool_),
        "turn_order": np.arange(5, dtype=np.uint8),
        "sheriff_idx": np.asarray([0], dtype=np.uint8),
        "self_role": np.asarray([0], dtype=np.uint8),
        "role_visibility": np.asarray([2, 0, 0, 0, 0], dtype=np.uint8),
        "revealed_roles": np.asarray([0, 255, 255, 255, 255], dtype=np.uint8),
        "confirmed_role_distribution": beliefs,
        "event_sequence": (),
        "action_mask": np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8),
    }
    module = IntentInferenceModule(preprocessor=ObservationPreprocessor(), seed=31).eval()
    assert module(observation).shape == (4, 5)

    observation["roles"] = np.asarray([0, 1, 2, 3, 4], dtype=np.uint8)
    with pytest.raises(ValueError, match="privileged"):
        module(observation)
