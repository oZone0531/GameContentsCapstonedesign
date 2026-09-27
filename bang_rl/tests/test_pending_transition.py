"""Focused TASK-7-2 tests for the pending transition boundary."""

from __future__ import annotations

import json

import pytest

from bang_rl.training import PendingTransitionLedger, PendingTransitionLedgerError


def observation(step: int, *, marker: str = "public") -> dict:
    return {
        "environment_step": step,
        "current_player_idx": 0,
        "marker": marker,
        "action_mask": [1, 1, 1, 1, 1, 1],
    }


def result(step: int, *, reward: float = 0.0, actor_idx: int = 0, terminated: bool = False, truncated: bool = False, events=(), breakdown=None) -> dict:
    value = {
        "environment_step": step,
        "action": {"actor_idx": actor_idx, "action_type": "pass", "target_idx": None},
        "reward": reward,
        "events": list(events),
        "terminated": terminated,
        "truncated": truncated,
    }
    if breakdown is not None:
        value["info"] = {"reward_breakdown": breakdown}
    return value


def open_ledger(ledger: PendingTransitionLedger, *, worker: int = 0, episode: str = "episode") -> None:
    ledger.start_episode(worker, episode)
    ledger.begin_action(worker, episode, 0, 0, observation(0), [1] * 6, 2, log_prob=-0.25, value=0.5, state={"healths": [4, 4, 4, 4, 4]}, emotion_labels={"0": {"values": [0.4]}})


def test_pre_post_alignment_accumulates_environment_steps_and_emits_at_next_agent_action() -> None:
    ledger = PendingTransitionLedger()
    open_ledger(ledger)
    ledger.commit_result(0, "episode", result(1, reward=0.5, events=({"event_id": 1, "event_type": "pass", "public": True},)))
    ledger.commit_result(0, "episode", result(2, reward=1.25, actor_idx=1, events=({"event_id": 2, "event_type": "pass", "public": True},)))

    previous = ledger.begin_action(0, "episode", 2, 0, observation(2, marker="next"), [1] * 6, 0)
    assert previous is not None
    assert previous.environment_step == 0
    assert previous.environment_steps_spanned == 2
    assert previous.accumulated_reward == pytest.approx(1.75)
    assert previous.next_observation["marker"] == "next"
    assert previous.terminated is False
    assert previous.trainable is True
    assert previous.log_prob == pytest.approx(-0.25)


def test_retry_invalid_and_duplicate_results_do_not_duplicate_transitions() -> None:
    ledger = PendingTransitionLedger()
    open_ledger(ledger)
    ledger.begin_action(0, "episode", 0, 0, observation(0), [1] * 6, 2, log_prob=-0.25, value=0.5, state={"healths": [4]}, emotion_labels={"0": {"values": [0.4]}})
    ledger.record_invalid_action(0, "episode", 0, 0, 99, reason="out_of_range", action_mask=[1] * 6)
    ledger.commit_result(0, "episode", result(1, reward=1.0))
    ledger.commit_result(0, "episode", result(1, reward=1.0))
    assert len(ledger.pending) == 1
    assert len(ledger.committed) == 0
    assert {item["kind"] for item in ledger.diagnostics} >= {"invalid_action", "duplicate_result_ignored"}

    with pytest.raises(PendingTransitionLedgerError):
        ledger.commit_result(0, "episode", result(1, reward=2.0))


def test_reward_breakdown_and_emotion_labels_are_separate_from_public_observation() -> None:
    ledger = PendingTransitionLedger()
    open_ledger(ledger)
    ledger.commit_result(
        0,
        "episode",
        result(1, reward=1.5, breakdown={"total": 1.5, "terminal_reward": 0.0, "shaping": {"bang_hit": 1.5}, "reward_config_version": "r1"}),
    )
    transitions = ledger.finish_episode(0, "episode", terminated=True, final_observation=observation(1), terminal_reward=2.0)
    assert transitions[0].terminal_reward == pytest.approx(2.0)
    assert transitions[0].accumulated_reward == pytest.approx(3.5)
    assert transitions[0].emotion_labels["0"]["values"] == [0.4]
    assert "reward" not in transitions[0].observation
    with pytest.raises(PendingTransitionLedgerError):
        ledger.begin_action(1, "leak", 0, 0, {"reward": 1}, [1] * 6, 0)


def test_workers_and_actors_are_isolated_and_cleanup_does_not_create_fake_pass() -> None:
    ledger = PendingTransitionLedger()
    open_ledger(ledger, worker=0, episode="same")
    open_ledger(ledger, worker=1, episode="same")
    ledger.commit_result(0, "same", result(1, reward=1.0))
    ledger.commit_result(1, "same", result(1, reward=2.0))
    finished = ledger.finish_episode(0, "same", terminated=True)
    assert len(finished) == 1 and finished[0].worker_id == 0
    ledger.cleanup(1, "same", reason="worker_cancelled", status="cancelled")
    assert len(ledger.committed) == 1
    assert all(item.worker_id == 0 for item in ledger.committed)
    assert any(item["reason"] == "worker_cancelled" for item in ledger.diagnostics)


def test_main_agent_elimination_parks_real_transition_until_environment_termination() -> None:
    ledger = PendingTransitionLedger()
    open_ledger(ledger)
    ledger.commit_result(0, "episode", result(1, reward=0.75, actor_idx=0))

    assert ledger.eliminate_agent(0, "episode") == ()
    assert len(ledger.committed) == 0
    assert ledger.pending[0].main_agent_eliminated is True
    assert ledger.pending[0].terminated is False
    assert ledger.pending[0].next_observation is None

    finished = ledger.finish_episode(
        0,
        "episode",
        terminated=True,
        final_observation=observation(9, marker="environment-final"),
        terminal_reward=2.0,
    )
    assert len(finished) == 1
    transition = finished[0]
    assert transition.main_agent_eliminated is True
    assert transition.terminated is True
    assert transition.truncated is False
    assert transition.next_observation is None
    assert transition.terminal_reward == pytest.approx(2.0)
    assert transition.accumulated_reward == pytest.approx(2.75)
    assert transition.trainable is True
    assert len(ledger.committed) == 1


    normal = PendingTransitionLedger()
    open_ledger(normal)
    normal.commit_result(0, "episode", result(1, reward=1.0, terminated=True))
    normal_transition = normal.finish_episode(0, "episode", terminated=True)[0]
    assert normal_transition.terminated is True
    assert normal_transition.truncated is False
    assert normal_transition.trainable is True
    assert normal_transition.next_observation is None

    truncated = PendingTransitionLedger()
    open_ledger(truncated, episode="truncated")
    truncated.commit_result(0, "truncated", result(1, reward=1.0, truncated=True))
    truncated_transition = truncated.finish_episode(0, "truncated", truncated=True, terminal_reward=99.0)[0]
    assert truncated_transition.truncated is True
    assert truncated_transition.terminated is False
    assert truncated_transition.trainable is False
    assert truncated_transition.terminal_reward == 0.0


def test_persistence_recovery_preserves_pending_boundary_and_deterministic_order() -> None:
    ledger = PendingTransitionLedger()
    open_ledger(ledger)
    ledger.commit_result(0, "episode", result(1, reward=0.75))
    encoded = ledger.to_json()
    assert json.loads(encoded)["state_version"] == "1.0"
    restored = PendingTransitionLedger.from_json(encoded)
    assert restored.to_json() == encoded
    restored.finish_episode(0, "episode", terminated=True)
    assert restored.committed[0].accumulated_reward == pytest.approx(0.75)
    assert restored.committed[0].transition_id == 0


def test_fault_timeout_and_cancel_cleanup_discard_unvalidated_pending_actions() -> None:
    ledger = PendingTransitionLedger()
    for status, method in (("fault", ledger.fault), ("timeout", ledger.timeout), ("cancelled", ledger.cancel)):
        episode = f"{status}-episode"
        open_ledger(ledger, episode=episode)
        method(0, episode, reason=status, status=status)
    assert not ledger.pending
    assert not ledger.committed
    assert {item["status"] for item in ledger.diagnostics} == {"fault", "timeout", "cancelled"}
