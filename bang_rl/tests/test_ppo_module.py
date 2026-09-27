"""Focused TASK-3-6 tests for the masked PPO decision module."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import (
    PPO_OPTIMIZER_METADATA,
    PPOModule,
    RoleFusion,
)
from bang_rl.config import PPOConfig


def _direct_inputs(batch: int | None = None) -> tuple[torch.Tensor, np.ndarray]:
    shape = (98,) if batch is None else (batch, 98)
    features = torch.linspace(-1.0, 1.0, int(np.prod(shape)), dtype=torch.float32).reshape(shape)
    mask = np.asarray([1, 1, 0, 1, 0, 1], dtype=np.uint8)
    if batch is not None:
        mask = np.broadcast_to(mask, (batch, 6)).copy()
    return features, mask


def _structured_inputs(batch: int | None = None) -> dict[str, np.ndarray]:
    prefix = () if batch is None else (batch,)
    return {
        "self_state": np.zeros(prefix + (9,), dtype=np.float32),
        "board_state": np.zeros(prefix + (5, 8), dtype=np.float32),
        "confirmed_role_features": np.zeros(prefix + (5, 5), dtype=np.float32),
        "fused_intent": np.zeros(prefix + (4, 5), dtype=np.float32),
        "emotion_features": np.zeros(prefix + (4,), dtype=np.float32),
        "action_mask": np.asarray([1, 1, 1, 0, 0, 1], dtype=np.uint8),
    }


def test_policy_and_value_shapes_mask_and_distribution() -> None:
    features, mask = _direct_inputs()
    module = PPOModule(input_dim=98, seed=11).eval()

    result = module(features, mask, sample=True)

    assert result.logits.shape == (6,)
    assert result.value.shape == (1,)
    assert result.action.shape == ()
    assert result.log_prob.shape == ()
    assert result.entropy.shape == ()
    assert result.distribution.probs.shape == (6,)
    assert torch.all(result.distribution.probs[mask == 0] == 0.0)
    assert torch.isfinite(result.log_prob)
    assert torch.isfinite(result.entropy)
    assert 0 <= int(result.action) < 6
    assert mask[int(result.action)] == 1


def test_structured_features_concatenate_rolefusion_output_and_support_batches() -> None:
    values = _structured_inputs(batch=3)
    raw_intent = np.full((4, 5), 0.2, dtype=np.float32)
    fused = RoleFusion().fuse(raw_intent, viewer_idx=0)
    values["fused_intent"] = np.broadcast_to(fused, (3, 4, 5)).copy()
    values["action_mask"] = np.broadcast_to(values["action_mask"], (3, 6)).copy()
    module = PPOModule(emotion_dim=4, seed=7).eval()

    result = module(values)

    assert module.input_dim == 9 + 40 + 25 + 20 + 4
    assert result.logits.shape == (3, 6)
    assert result.value.shape == (3, 1)
    assert result.distribution.probs.shape == (3, 6)
    assert torch.all(result.distribution.probs[:, 3:5] == 0.0)


def test_deterministic_and_seeded_stochastic_action_modes_are_valid() -> None:
    features, mask = _direct_inputs()
    module = PPOModule(input_dim=98, seed=23).eval()

    deterministic = module.act(features, mask, deterministic=True)
    assert deterministic.action.item() == int(module(features, mask).logits.argmax())
    assert mask[deterministic.action.item()] == 1

    generator_a = torch.Generator().manual_seed(1234)
    generator_b = torch.Generator().manual_seed(1234)
    first = module.act(features, mask, generator=generator_a)
    second = module.act(features, mask, generator=generator_b)
    assert first.action.item() == second.action.item()
    assert first.log_prob.item() == pytest.approx(second.log_prob.item())


def test_log_prob_entropy_and_value_match_masked_distribution() -> None:
    features, mask = _direct_inputs(batch=4)
    module = PPOModule(input_dim=98, seed=2).eval()
    actions = torch.tensor([0, 1, 3, 5])

    log_prob, entropy, value = module.evaluate_actions(features, actions, mask)
    result = module(features, mask)

    torch.testing.assert_close(log_prob, result.distribution.log_prob(actions))
    torch.testing.assert_close(entropy, result.distribution.entropy())
    torch.testing.assert_close(value, result.value)
    assert torch.allclose(module.log_prob(features, actions, mask), log_prob)
    assert torch.allclose(module.entropy(features, mask), entropy)
    assert torch.allclose(module.value(features, mask), value)


def test_fixed_architecture_optimizer_metadata_and_config_literals() -> None:
    config = PPOConfig()
    module = PPOModule(input_dim=98, ppo_config=config, seed=0)
    metadata = module.metadata()

    assert config.rollout_transitions_per_worker == 256
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.ppo_epochs == 4
    assert config.minibatch_size == 64
    assert config.clip_epsilon == 0.2
    assert metadata["shared_trunk"]["layers"] == [
        {"type": "Linear", "in_features": 98, "out_features": 256},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 256, "out_features": 256},
        {"type": "ReLU"},
    ]
    assert metadata["policy_head"]["layers"] == [
        {"type": "Linear", "in_features": 256, "out_features": 6}
    ]
    assert metadata["value_head"]["layers"] == [
        {"type": "Linear", "in_features": 256, "out_features": 1}
    ]
    assert metadata["optimizer"] == PPO_OPTIMIZER_METADATA
    assert metadata["input_contract"]["intent_source"].startswith("RoleFusion")
    assert metadata["input_contract"]["role_label"] == "forbidden"
    assert metadata["input_contract"]["emotion_label"] == "forbidden"
    assert metadata["input_contract"]["reward"] == "forbidden"


def test_malformed_masks_features_and_privileged_inputs_are_rejected() -> None:
    features, mask = _direct_inputs()
    module = PPOModule(input_dim=98, seed=1).eval()

    with pytest.raises(ValueError, match="action_mask"):
        module(features, np.zeros(6, dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        module(features, np.ones(5, dtype=np.uint8))
    with pytest.raises(ValueError, match="feature dimension"):
        module(torch.zeros(97), mask)
    with pytest.raises(ValueError, match="finite"):
        module(torch.full((98,), float("nan")), mask)
    with pytest.raises(ValueError, match="at least one"):
        module(features, np.zeros(6, dtype=np.uint8))

    private = copy.deepcopy(_structured_inputs())
    private["reward"] = np.asarray([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="privileged"):
        module(private)


def test_schema_versioned_save_load_restores_mode_metadata_and_policy(tmp_path) -> None:
    features, mask = _direct_inputs()
    original = PPOModule(input_dim=98, seed=31).eval()
    expected = original(features, mask, sample=True)
    artifact = tmp_path / "ppo.inference.pt"

    original.save(artifact)
    restored = PPOModule.load(artifact)
    actual = restored(features, mask, sample=True)

    assert restored.training is False
    assert restored.to_dict() == original.to_dict()
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.value, expected.value)
    with pytest.raises(ValueError, match="schema version"):
        PPOModule.from_dict({"schema_version": "0.0"})
