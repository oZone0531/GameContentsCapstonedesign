"""Focused TASK-7-3 tests for PPO rollout math and in-memory updates."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import PPOModule
from bang_rl.config import PPOConfig
from bang_rl.env import Action
from bang_rl.training import (
    PPOTrainer,
    PPOTransition,
    PendingTransitionLedger,
    clipped_surrogate_loss,
    compute_gae,
)


def _transition(index: int, *, terminated: bool = True, truncated: bool = False) -> PPOTransition:
    mask = np.asarray([1, 1, 0, 1, 0, 1], dtype=np.uint8)
    return PPOTransition(
        features=np.full(4, index / 10, dtype=np.float32),
        action=(index % 2),
        action_mask=mask,
        old_log_prob=-0.4,
        value=0.1,
        reward=1.0,
        next_features=None if terminated or truncated else np.full(4, (index + 1) / 10, dtype=np.float32),
        next_action_mask=mask,
        terminated=terminated and not truncated,
        truncated=truncated,
        trainable=not truncated,
        worker_id=index % 2,
        episode_id=f"episode-{index // 2}",
        transition_id=index,
        environment_step=index + 1,
    )


def test_gae_uses_main_agent_transition_units_and_resets_on_termination() -> None:
    advantages, returns = compute_gae(
        [1.0, 3.0], [1.0, 2.0], [2.0, 0.0], [False, True], [False, False],
        gamma=0.99, gae_lambda=0.95,
    )
    assert advantages.tolist() == pytest.approx([2.9205, 1.0])
    assert returns.tolist() == pytest.approx([3.9205, 3.0])


def test_truncated_transition_never_bootstraps_or_enters_returns() -> None:
    advantages, returns = compute_gae(
        [2.0], [1.0], [100.0], [False], [True], gamma=0.99, gae_lambda=0.95,
        trainable=[False],
    )
    assert advantages.item() == 0.0
    assert returns.item() == 0.0


def test_committed_domain_actions_normalize_to_discrete_six_and_reject_conflicts() -> None:
    ledger = PendingTransitionLedger()
    ledger.start_episode(0, "domain-actions")
    for step, action, expected in (
        (0, Action.pass_action(0), 0),
        (1, Action.bang(0, 3), 4),
    ):
        ledger.begin_action(
            0,
            "domain-actions",
            step,
            0,
            {"state_vector": [0.0, 1.0]},
            [1] * 6,
            action,
            log_prob=-0.2,
            value=0.1,
            state={"state_vector": [0.0, 1.0]},
        )
        ledger.commit_result(
            0,
            "domain-actions",
            {
                "environment_step": step + 1,
                "action": action,
                "actor_idx": 0,
                "reward": 0.0,
            },
        )
        committed = ledger.finish_episode(
            0,
            "domain-actions",
            terminated=True,
        )[0]
        converted = PPOTransition.from_committed(committed)
        assert converted.action == expected
        np.testing.assert_array_equal(converted.action_mask, np.ones(6, dtype=np.uint8))
        ledger.start_episode(0, "domain-actions")

    ledger.start_episode(0, "inconsistent")
    ledger.begin_action(
        0,
        "inconsistent",
        0,
        0,
        {"state_vector": [0.0]},
        [1] * 6,
        {"action": 2, "action_type": "BANG", "actor_idx": 0, "target_idx": 1},
        log_prob=-0.2,
        value=0.1,
        state={"state_vector": [0.0]},
    )
    ledger.commit_result(
        0,
        "inconsistent",
        {"environment_step": 1, "action": {"action": 2, "action_type": "BANG", "actor_idx": 0, "target_idx": 1}, "actor_idx": 0},
    )
    committed = ledger.finish_episode(0, "inconsistent", terminated=True)[0]
    with pytest.raises(ValueError, match="inconsistent"):
        PPOTransition.from_committed(
            committed.__class__(**{**committed.to_dict(), "action": {"action": 2, "action_type": "BANG", "actor_idx": 0, "target_idx": 2}})
        )


def test_segmented_gae_resets_between_workers_and_episodes() -> None:
    module = PPOModule(input_dim=1, seed=1)
    trainer = PPOTrainer(module, config=PPOConfig(ppo_epochs=1, minibatch_size=2))
    mask = np.ones(6, dtype=np.uint8)
    first = PPOTransition(
        features=np.asarray([0.0], dtype=np.float32), action=0, action_mask=mask,
        old_log_prob=0.0, value=0.0, reward=1.0, next_value=5.0,
        worker_id=0, episode_id="episode-a", rollout_id="rollout-a",
    )
    second = PPOTransition(
        features=np.asarray([1.0], dtype=np.float32), action=0, action_mask=mask,
        old_log_prob=0.0, value=0.0, reward=10.0, next_value=7.0,
        worker_id=1, episode_id="episode-b", rollout_id="rollout-b",
    )
    values = torch.zeros(2)
    advantages, returns = trainer._segmented_gae([first, second], values, torch.tensor([5.0, 7.0]))
    expected_first = compute_gae([1.0], [0.0], [5.0], [False], [False])
    expected_second = compute_gae([10.0], [0.0], [7.0], [False], [False])
    torch.testing.assert_close(advantages, torch.cat([expected_first[0], expected_second[0]]))
    torch.testing.assert_close(returns, torch.cat([expected_first[1], expected_second[1]]))


def test_segmented_gae_rollout_cutoff_uses_only_its_transition_next_value() -> None:
    module = PPOModule(input_dim=1, seed=2)
    trainer = PPOTrainer(module, config=PPOConfig(ppo_epochs=1, minibatch_size=2))
    mask = np.ones(6, dtype=np.uint8)
    cutoff = PPOTransition(
        features=np.asarray([0.0], dtype=np.float32), action=0, action_mask=mask,
        old_log_prob=0.0, value=1.0, reward=2.0, next_value=3.0,
        worker_id=0, episode_id="episode", rollout_id="rollout-1",
    )
    next_rollout = PPOTransition(
        features=np.asarray([1.0], dtype=np.float32), action=0, action_mask=mask,
        old_log_prob=0.0, value=100.0, reward=4.0, next_value=0.0,
        terminated=True, worker_id=0, episode_id="episode", rollout_id="rollout-2",
    )
    advantages, _ = trainer._segmented_gae([cutoff, next_rollout], torch.tensor([1.0, 100.0]), torch.tensor([3.0, 0.0]))
    expected_cutoff, _ = compute_gae([2.0], [1.0], [3.0], [False], [False])
    assert advantages[0].item() == pytest.approx(expected_cutoff.item())


    loss, ratio, fraction = clipped_surrogate_loss(
        torch.tensor([0.0]), torch.tensor([np.log(1.5)]), torch.tensor([2.0]), 0.2,
    )
    assert ratio.item() == pytest.approx(1.5)
    assert loss.item() == pytest.approx(-2.4)
    assert fraction.item() == pytest.approx(1.0)


def test_rollout_keeps_only_mask_valid_actions_and_excludes_truncation() -> None:
    module = PPOModule(input_dim=4, seed=3)
    trainer = PPOTrainer(module, worker_count=1, config=PPOConfig(rollout_transitions_per_worker=2))
    selected = trainer.collect_rollout([_transition(0), _transition(1, truncated=True)], max_transitions=2)
    assert len(selected) == 1
    result = module(selected[0].features, selected[0].action_mask, sample=True, deterministic=False)
    assert selected[0].action_mask[int(result.action)] == 1
    assert torch.all(result.distribution.probs[torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.bool)] == 0)


def test_update_changes_parameters_and_records_separate_counters() -> None:
    module = PPOModule(input_dim=4, seed=7)
    trainer = PPOTrainer(
        module, worker_count=1,
        config=PPOConfig(rollout_transitions_per_worker=4, ppo_epochs=2, minibatch_size=2),
        seed=19,
    )
    transitions = [_transition(index) for index in range(4)]
    trainer.collect_rollout(transitions)
    before = [parameter.detach().clone() for parameter in module.parameters()]
    metrics = trainer.update()
    assert metrics.samples == 8
    assert metrics.ppo_update == 1
    assert metrics.main_agent_transitions == 4
    assert metrics.environment_steps == 4
    assert any(not torch.equal(old, new) for old, new in zip(before, module.parameters()))
    assert len(trainer.rollout_buffer.trainable) == 0


def test_deterministic_seed_replays_update_and_parameters() -> None:
    config = PPOConfig(rollout_transitions_per_worker=4, ppo_epochs=2, minibatch_size=2)
    first_module = PPOModule(input_dim=4, seed=11)
    second_module = PPOModule(input_dim=4, seed=11)
    first = PPOTrainer(first_module, config=config, seed=29)
    second = PPOTrainer(second_module, config=config, seed=29)
    records = [_transition(index) for index in range(4)]
    first.collect_rollout(records)
    second.collect_rollout(records)
    first_metrics = first.update()
    second_metrics = second.update()
    assert first_metrics.to_dict() == second_metrics.to_dict()
    for left, right in zip(first_module.parameters(), second_module.parameters()):
        torch.testing.assert_close(left, right)


def test_checkpoint_metadata_is_explicitly_non_resumable_and_snapshot_compatible() -> None:
    module = PPOModule(input_dim=4, seed=2)
    trainer = PPOTrainer(module, snapshot={"job_id": "job-1"})
    metadata = trainer.checkpoint_metadata()
    assert metadata["checkpoint"]["persistent"] is False
    assert metadata["checkpoint"]["resume_supported"] is False
    assert metadata["snapshot"]["job_id"] == "job-1"
    trainer.validate_checkpoint_metadata(metadata)
    with pytest.raises(Exception, match="not supported"):
        trainer.validate_checkpoint_metadata({"checkpoint": {"resume_supported": True}})
