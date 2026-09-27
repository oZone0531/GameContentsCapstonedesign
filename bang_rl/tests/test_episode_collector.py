"""Focused TASK-5-1 tests for canonical episode collection."""

from __future__ import annotations

import numpy as np
import pytest

from bang_rl.agents import EmotionModule
from bang_rl.config import EmotionConfig, RewardConfig
from bang_rl.data import EpisodeCollector, NPZTensorCacheLoader, NPZTensorCacheWriter
from bang_rl.env import Action, ActionResult, BangEnvironment, EnvironmentState, EventType, GameEvent, Team
from bang_rl.runtime import EmotionRuntime, RewardCalculator


ROLES = (0, 1, 2, 3, 4)


def pass_result(step: int, *, truncated: bool = False) -> ActionResult:
    return ActionResult(
        action=Action.pass_action(0),
        environment_step=step,
        events=(GameEvent(step, step, EventType.PASS, actor_idx=0),),
        reward=0.25,
        truncated=truncated,
    )


def test_valid_step_preserves_pre_action_alignment_and_stores_events_once() -> None:
    collector = EpisodeCollector()
    collector.start_episode(episode_id="alignment", seed=7, roles=ROLES)
    pre = {
        "environment_step": np.asarray([0], dtype=np.uint32),
        "current_player_idx": np.asarray([0], dtype=np.uint8),
        "action_mask": np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8),
        "event_sequence": (),
    }
    collector.before_action(pre, emotion_record=None)
    step = collector.record_step(pass_result(1), post_observation={"environment_step": [1], "event_sequence": [{"event_id": 1}]})
    episode = collector.finish_episode(result={"terminated": False, "truncated": True, "environment_step": 1, "action": Action.pass_action(0), "events": ()}, final_state={"environment_step": 1})

    assert step.environment_step == 1
    assert step.pre_observation["environment_step"] == [0]
    assert step.pre_event_ids == ()
    assert step.event_ids == (1,)
    assert len(episode.events) == 1
    assert episode.events[0]["event_id"] == 1
    assert episode.termination_status == "truncated"
    assert episode.steps[0].truncated is False


def test_invalid_action_is_diagnostic_only_and_does_not_create_step() -> None:
    environment = BangEnvironment(episode_id="invalid", seed=4, max_episode_steps=3, warn_turn_threshold=1, critical_turn_threshold=2)
    collector = EpisodeCollector()
    collector.start_episode(environment)
    collector.before_action(environment.observe())
    environment.validate_action(0)
    collector.record_invalid_action(99, reason="out_of_range")
    collector.record_step(environment.step(0), observation=collector._active["pending"]["observation"])
    collector.before_action(environment.observe())
    collector.record_step(environment.step(0), observation=collector._active["pending"]["observation"])
    collector.before_action(environment.observe())
    collector.record_step(environment.step(0), observation=collector._active["pending"]["observation"])

    episode = collector.episodes[0]
    assert len(episode.steps) == 3
    assert len(episode.invalid_actions) == 1
    assert episode.invalid_actions[0]["environment_step"] == 0
    assert [step.environment_step for step in episode.steps] == [1, 2, 3]


def test_emotion_label_and_reward_breakdown_align_to_same_pre_action_step() -> None:
    config = EmotionConfig(
        schema_version="collector-emotion-v1",
        emotions=[{"name": "fear", "initial_value": 0.4}],
    )
    runtime = EmotionRuntime(emotion_modules={index: EmotionModule(config, player_idx=index) for index in range(5)})
    runtime.reset({"episode_id": "runtime", "seed": 8})
    labels = runtime.before_action(0, actor_idx=0)
    result = pass_result(1)
    reward = RewardCalculator(RewardConfig(version="collector-reward-v1")).calculate_all(result, roles=ROLES)
    emotion_record = runtime.on_action_result(result)

    collector = EpisodeCollector()
    collector.start_episode(episode_id="runtime", seed=8, roles=ROLES)
    collector.before_action(labels[0].public_observation, emotion_record=type("Boundary", (), {"labels": labels})())
    step = collector.record_step(result, emotion_record=emotion_record, reward_batch=reward)
    collector.finish_episode(result={**result.to_dict(), "truncated": True, "terminated": False}, final_state={"environment_step": 1})

    assert step.emotion_labels["0"]["environment_step"] == 0
    assert step.emotion_labels["0"]["input_event_ids"] == []
    assert step.reward_breakdowns["0"]["reward_config_version"] == "collector-reward-v1"
    assert step.event_ids == (1,)


def test_calculated_reward_becomes_canonical_step_reward_and_truncation_has_no_terminal_reward() -> None:
    config = RewardConfig(
        version="collector-reward-regression-v1",
        terminal={
            "components": [
                {"id": "terminal_win", "enabled": True, "condition": {"type": "terminal_result", "result": "win"}, "value": 2.0},
            ]
        },
        shaping={
            "components": [
                {"id": "pass_penalty", "enabled": True, "condition": {"type": "pass"}, "value": -0.25},
            ]
        },
    )
    normal_result = ActionResult(
        action=Action.pass_action(0),
        environment_step=1,
        events=(GameEvent(1, 1, EventType.PASS, actor_idx=0),),
        reward=0.0,
        terminated=True,
        info={"terminal_result": {"winners": [0, 1], "losers": [2, 3, 4], "winning_team": int(Team.SHERIFF_TEAM)}},
    )
    normal = EpisodeCollector(reward_calculator=RewardCalculator(config))
    normal.start_episode(episode_id="reward-normal", seed=1, roles=ROLES)
    normal.before_action({"environment_step": 0, "current_player_idx": 0, "action_mask": [1] * 6})
    step = normal.record_step(normal_result)
    assert step.reward == pytest.approx(1.75)
    assert step.reward_breakdowns["0"]["total"] == pytest.approx(1.75)

    truncated_result = ActionResult(
        action=Action.pass_action(0),
        environment_step=1,
        events=(GameEvent(1, 1, EventType.PASS, actor_idx=0),),
        reward=0.0,
        truncated=True,
    )
    truncated = EpisodeCollector(reward_calculator=RewardCalculator(config))
    truncated.start_episode(episode_id="reward-truncated", seed=1, roles=ROLES)
    truncated.before_action({"environment_step": 0, "current_player_idx": 0, "action_mask": [1] * 6})
    truncated_step = truncated.record_step(truncated_result)
    assert truncated_step.reward == pytest.approx(-0.25)
    assert truncated_step.reward_breakdowns["0"]["terminal_reward"] == 0.0


    first = EpisodeCollector(metadata={"preset_id": "dummy-a"})
    first.start_episode(episode_id="deterministic", seed=33, roles=ROLES)
    first.before_action({"environment_step": 0, "current_player_idx": 0, "action_mask": [1, 1, 1, 1, 1, 1]})
    first.record_step(pass_result(1))
    episode = first.finish_episode(result={**pass_result(1).to_dict(), "truncated": True}, final_state={"environment_step": 1})

    encoded = first.to_json()
    restored = EpisodeCollector.from_json(encoded)
    assert restored.to_json() == encoded
    assert episode.metadata["preset_id"] == "dummy-a"
    public_value = episode.to_public_dict()
    assert "role_labels" not in public_value
    assert "roles" not in public_value.get("initial_state", {})
    assert "roles" not in public_value.get("final_state", {})


def test_flush_and_finalize_send_completed_episodes_without_parquet_side_effects() -> None:
    class Writer:
        def __init__(self) -> None:
            self.written = []
            self.flushed = 0
            self.finalized = 0

        def write_episode(self, episode):
            self.written.append(episode.episode_id)

        def flush(self):
            self.flushed += 1

        def finalize(self):
            self.finalized += 1

    writer = Writer()
    collector = EpisodeCollector(writer=writer)
    collector.start_episode(episode_id="persist", seed=1, roles=ROLES)
    collector.before_action({"environment_step": 0, "current_player_idx": 0, "action_mask": [1] * 6})
    collector.record_step(pass_result(1))
    collector.finish_episode(result={**pass_result(1).to_dict(), "truncated": True}, final_state={"environment_step": 1})

    collector.flush()
    collector.finalize()
    collector.finalize()
    assert writer.written == ["persist"]
    assert writer.flushed == 1
    assert writer.finalized == 1
