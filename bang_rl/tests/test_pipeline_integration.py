"""TASK-10-3 end-to-end integration coverage.

These tests intentionally compose the public seams instead of reaching through
private state: environment -> agents/runtime -> canonical data -> ML ->
artifact/evaluation/application.  Lower-level invariants remain covered by the
focused TASK-0 through TASK-9 tests.
"""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import torch

from bang_rl.agents import DummyAgent, RandomAgent
from bang_rl.agents.dummy_agent.emotion_module import EmotionModule as DummyEmotionModule
from bang_rl.agents.main_agent import EmotionInferenceModule, IntentInferenceModule, PPOModule, RoleFusion
from bang_rl.app import EvaluationRunConfig, EvaluationService, GameService, SessionConfig, TrainingConfig, TrainingService
from bang_rl.config import EmotionConfig, EvaluationConfig, PPOConfig, RewardConfig, SystemConfig
from bang_rl.data import (
    CanonicalEpisode,
    CanonicalStep,
    EpisodeSplitter,
    NPZTensorCacheLoader,
    NPZTensorCacheWriter,
    ParquetCanonicalShardWriter,
)
from bang_rl.env import BangEnvironment, EnvironmentState, Role
from bang_rl.exceptions import InvalidActionError
from bang_rl.gui import GameMode, GameSessionController
from bang_rl.runtime import EmotionRuntime, RewardCalculator, RewardRuntime, attach_rewards
from bang_rl.runtime.logging import LogChannel, StructuredLogger
from bang_rl.runtime.pause_coordinator import PauseCoordinator, PauseState, WorkerControl
from bang_rl.training import (
    EmotionInferenceSLTrainer,
    EmotionSLConfig,
    IntentInferenceSLTrainer,
    IntentSLConfig,
    PPOTrainer,
    PPOTransition,
    load_final_inference_artifact,
    load_phase2_inference_artifact,
    save_final_inference_artifact,
    save_phase2_inference_artifact,
)


ROLES = (Role.SHERIFF, Role.DEPUTY, Role.OUTLAW_A, Role.OUTLAW_B, Role.TRAITOR)


def _public_trace(seed: int, *, max_episode_steps: int = 8):
    """Run deterministic Dummy/Random agents only through public boundaries."""
    env = BangEnvironment(seed=seed, episode_id=f"trace-{seed}", max_episode_steps=max_episode_steps, warn_turn_threshold=2, critical_turn_threshold=4)
    agents = tuple(
        DummyAgent(f"player_{idx}", seed=100 + idx, player_idx=idx, action_mode="deterministic")
        if idx % 2 == 0
        else RandomAgent(f"player_{idx}", seed=200 + idx)
        for idx in range(5)
    )
    for idx, agent in enumerate(agents):
        agent.on_episode_start({"episode_id": env.state.episode_id, "seed": seed, "observation": env.observe(idx)})
    trace = []
    try:
        while not (env.state.terminated or env.state.truncated):
            actor = int(env.state.agent_selection)
            observation = env.observe()
            assert "roles" not in observation
            action = agents[actor].act(observation, env.action_mask)
            assert env.action_mask[action] == 1
            env.step(action)
            result = env.last_result
            assert result is not None
            trace.append(
                (
                    result.environment_step,
                    result.action.to_dict(),
                    tuple(event.to_public_dict() for event in result.events),
                    tuple(env.state.health),
                    tuple(env.state.alive),
                    env.state.termination_status.value,
                )
            )
        return trace, env.state.termination_status.value, tuple(env.state.alive)
    finally:
        env.close()


def _observation(*, viewer: int = 0, step: int = 0, events=()) -> dict:
    beliefs = np.zeros((5, 5), dtype=np.float32)
    beliefs[0, 0] = 1.0
    return {
        "current_player_idx": np.asarray([viewer], dtype=np.uint8),
        "viewer_idx": viewer,
        "environment_step": np.asarray([step], dtype=np.uint32),
        "round_index": np.asarray([step], dtype=np.uint32),
        "healths": np.asarray([5, 4, 4, 4, 4], dtype=np.uint8),
        "alive": np.asarray([True] * 5, dtype=np.bool_),
        "turn_order": np.arange(5, dtype=np.uint8),
        "sheriff_idx": np.asarray([0], dtype=np.uint8),
        "self_role": np.asarray([0], dtype=np.uint8),
        "role_visibility": np.asarray([1, 0, 0, 0, 0], dtype=np.uint8),
        "revealed_roles": np.asarray([0, 255, 255, 255, 255], dtype=np.uint8),
        "confirmed_role_distribution": beliefs,
        "event_sequence": tuple(events),
        "action_mask": np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8),
        "observation_schema_version": "1.0",
    }


def _episode(index: int, *, status: str = "normal") -> CanonicalEpisode:
    before = _observation()
    event = {
        "event_id": index + 1,
        "environment_step": 1,
        "event_type": "pass",
        "actor_idx": 0,
        "target_idx": None,
        "hp_before": None,
        "hp_after": None,
        "eliminated_idx": None,
        "revealed_role": None,
        "public": True,
    }
    label = {
        "episode_id": f"episode-{index:02d}",
        "environment_step": 0,
        "player_idx": 0,
        "values": [0.25],
        "emotion_names": ["fear"],
        "emotion_schema_version": "emotion-v1",
        "input_event_ids": [],
        "public_observation": before,
    }
    step = CanonicalStep(
        environment_step=1,
        actor_idx=0,
        action=0,
        action_type="pass",
        target_idx=None,
        action_mask=(1, 0, 1, 1, 1, 1),
        pre_observation=before,
        post_observation=_observation(step=1),
        pre_event_ids=(),
        event_ids=(index + 1,),
        hp_before=None,
        hp_after=None,
        eliminated_idx=None,
        reward=0.0,
        emotion_labels={str(player): {**label, "player_idx": player} for player in range(5)},
        terminated=status == "normal",
        truncated=status == "truncated",
    )
    return CanonicalEpisode(
        episode_id=f"episode-{index:02d}",
        seed=index,
        agent_order=(0, 1, 2, 3, 4),
        metadata={
            "dataset_schema_version": "1.0",
            "environment_version": "1.0",
            "observation_schema_version": "1.0",
            "action_space_version": "1.0",
            "emotion_schema_version": "emotion-v1",
            "reward_config_version": "reward-v1",
            "preset_version": "preset-v1",
            "job_id": "job-integration",
        },
        initial_state=before,
        steps=(step,),
        events=(event,),
        termination_status=status,
        termination_reason="finished" if status == "normal" else "max_episode_steps_reached",
        terminal_result=(
            {"terminated": True, "winners": [0, 1], "losers": [2, 3, 4], "winning_team": 0, "reason": "finished"}
            if status == "normal" else None
        ),
        final_state=_observation(step=1),
        role_labels=(0, 1, 2, 3, 4),
    )


def test_mixed_agents_replay_deterministically_from_seed_and_public_observations() -> None:
    first = _public_trace(71)
    second = _public_trace(71)
    assert first == second
    assert first[0]
    assert first[1] in {"normal", "truncated"}


def test_normal_precedes_truncation_invalid_action_is_immutable_and_cleanup_is_not_a_step() -> None:
    state = EnvironmentState(
        episode_id="normal-priority",
        seed=3,
        roles=ROLES,
        health=(5, 1, 0, 0, 0),
        alive=(True, True, False, False, False),
        agent_selection=0,
    )
    env = BangEnvironment(state=state, max_episode_steps=3, warn_turn_threshold=1, critical_turn_threshold=2)
    before = (env.state, env.agent_selection, env.environment_step, env.cleanup_calls)
    with pytest.raises(InvalidActionError):
        env.step(1)  # player_0 attacking itself
    assert (env.state, env.agent_selection, env.environment_step, env.cleanup_calls) == before
    env.step(2)  # sheriff eliminates the deputy; normal termination wins the tie
    assert env.state.terminated and not env.state.truncated
    assert env.state.terminal_result is not None
    terminal_step = env.environment_step
    while env.agents:
        env.step(None)
    assert env.cleanup_calls == 5
    assert env.environment_step == terminal_step
    env.close()

    truncated = BangEnvironment(episode_id="truncated", seed=4, max_episode_steps=3, warn_turn_threshold=1, critical_turn_threshold=2)
    truncated.step(0)
    truncated.step(0)
    truncated.step(0)
    assert truncated.state.truncated and not truncated.state.terminated
    assert truncated.state.terminal_result is None
    truncated.close()


def test_role_reveal_role_fusion_and_privileged_separation() -> None:
    state = EnvironmentState(
        episode_id="reveal",
        seed=5,
        roles=ROLES,
        health=(5, 1, 0, 0, 0),
        alive=(True, True, False, False, False),
        agent_selection=0,
    )
    env = BangEnvironment(state=state, max_episode_steps=20, warn_turn_threshold=5, critical_turn_threshold=10)
    env.step(2)
    assert any(event.event_type.value == "role_revealed" for event in env.state.event_log)
    observation = env.observe()
    assert "roles" not in observation
    revealed = observation["revealed_roles"]
    assert revealed[1] == int(Role.DEPUTY)
    raw = np.full((5, 5), 0.2, dtype=np.float32)
    confirmed = np.asarray(observation["confirmed_role_distribution"], dtype=np.float32)
    fused = RoleFusion().fuse(
        raw, observation["role_visibility"], revealed, confirmed,
        sheriff_idx=observation["sheriff_idx"],
        viewer_idx=observation["current_player_idx"],
        self_role=observation["self_role"],
        target_player_indices=np.arange(5),
    )
    np.testing.assert_allclose(fused[0], np.eye(5, dtype=np.float32)[int(Role.SHERIFF)])
    np.testing.assert_allclose(fused[1], np.eye(5, dtype=np.float32)[int(Role.DEPUTY)])
    env.close()


def test_emotion_reward_and_transition_records_share_the_pre_action_boundary() -> None:
    emotion_config = EmotionConfig(
        schema_version="emotion-v1",
        emotions=[{"name": "fear", "initial_value": 0.4, "decay_rate": 0.1}],
    )
    modules = {idx: DummyEmotionModule(emotion_config, player_idx=idx) for idx in range(5)}
    emotion = EmotionRuntime(emotion_modules=modules)
    reward = RewardRuntime(RewardCalculator(RewardConfig(version="reward-v1")))
    env = BangEnvironment(episode_id="alignment", seed=8, max_episode_steps=20, warn_turn_threshold=5, critical_turn_threshold=10)
    emotion.reset({"episode_id": env.state.episode_id, "seed": env.state.seed})
    reward.reset(episode_id=env.state.episode_id)
    labels = emotion.before_action(0, actor_idx=0, observation=env.observe())
    assert labels[0].environment_step == 0
    assert labels[0].input_event_ids == ()
    before_invalid = (emotion.records, emotion.processed_event_ids, emotion.environment_step)
    with pytest.raises(InvalidActionError):
        env.step(1)
    assert (emotion.records, emotion.processed_event_ids, emotion.environment_step) == before_invalid
    env.step(0)
    result = env.last_result
    assert result is not None
    batch = reward.on_action_result(result, roles=env.state.roles)
    attached = attach_rewards(result, batch)
    record = emotion.on_action_result(attached)
    assert record is not None
    assert record.label_step == 0
    assert record.post_action_event_ids == (1,)
    assert batch is reward.on_action_result(result, roles=env.state.roles)
    next_labels = emotion.before_action(1, events=result.events, actor_idx=1, observation=env.observe())
    assert next_labels[1].environment_step == 1
    assert next_labels[1].input_event_ids == (1,)
    assert all("emotion" not in key.lower() for key in env.observe().keys())
    env.close()


def test_canonical_parquet_split_npz_and_both_sl_trainers_exclude_truncated(tmp_path: Path) -> None:
    episodes = [_episode(index) for index in range(13)] + [_episode(13, status="truncated")]
    parquet = ParquetCanonicalShardWriter(tmp_path / "canonical")
    for episode in episodes:
        parquet.append(episode)
    paths = parquet.finalize()
    restored = ParquetCanonicalShardWriter.read_shard(paths[0])
    assert {episode.termination_status for episode in restored} == {"normal", "truncated"}

    split = EpisodeSplitter(counts=(10, 2, 2), seed=23, manifest_path=tmp_path / "split.json").split(restored)
    assert not (set(split.train.episode_ids) & set(split.validation.episode_ids))
    assert not (set(split.train.episode_ids) & set(split.test.episode_ids))
    cache_artifacts = NPZTensorCacheWriter(
        tmp_path / "cache", emotion_schema_version="emotion-v1", emotion_dimension=1
    ).write_all(split)
    assert sum(len(artifact.manifest["episode_ids"]) for artifact in cache_artifacts.values()) == 13
    assert sum(artifact.manifest["excluded_truncated_episode_count"] for artifact in cache_artifacts.values()) == 1
    assert all("episode-13" not in artifact.manifest["episode_ids"] for artifact in cache_artifacts.values())
    train = NPZTensorCacheLoader(cache_artifacts["train"].tensor_path)
    validation = NPZTensorCacheLoader(cache_artifacts["validation"].tensor_path)
    test = NPZTensorCacheLoader(cache_artifacts["test"].tensor_path)
    intent = IntentInferenceSLTrainer(train, validation, test_cache=test, config=IntentSLConfig(epochs=1, batch_size=2, seed=7)).fit()
    emotion = EmotionInferenceSLTrainer(
        train, validation, test_cache=test,
        config=EmotionSLConfig(epochs=1, batch_size=2, seed=7),
        emotion_config=emotion_config_for_test(),
    ).fit()
    assert intent.history and emotion.history
    assert intent.test_metrics is not None and intent.test_metrics["split"] == "test"
    assert emotion.test_metrics is not None and emotion.test_metrics["split"] == "test"


def emotion_config_for_test() -> EmotionConfig:
    return EmotionConfig(schema_version="emotion-v1", emotion_count=1)


def test_ppo_uses_phase2_artifact_freezes_inference_and_round_trips_final_model(tmp_path: Path) -> None:
    intent = IntentInferenceModule(seed=13).eval()
    emotion = EmotionInferenceModule(config=emotion_config_for_test(), seed=13).eval()
    phase2_path = save_phase2_inference_artifact(tmp_path / "phase2.phase2model", intent, emotion)
    phase2 = load_phase2_inference_artifact(phase2_path)
    ppo = PPOModule(input_dim=4, seed=17)
    trainer = PPOTrainer(
        ppo,
        phase2_artifact=phase2,
        config=PPOConfig(rollout_transitions_per_worker=2, ppo_epochs=1, minibatch_size=2),
        seed=19,
    )
    mask = np.asarray([1, 1, 0, 1, 0, 1], dtype=np.uint8)
    transitions = [
        PPOTransition(
            features=np.ones(4, dtype=np.float32), action=0, action_mask=mask,
            old_log_prob=-0.2, value=0.1, reward=1.0, terminated=True,
            episode_id="normal", transition_id=0, environment_step=1,
        ),
        PPOTransition(
            features=np.zeros(4, dtype=np.float32), action=1, action_mask=mask,
            old_log_prob=-0.2, value=0.1, reward=0.0, truncated=True,
            trainable=False, episode_id="truncated", transition_id=0, environment_step=1,
        ),
    ]
    selected = trainer.collect_rollout(transitions)
    assert len(selected) == 1
    trainer.update()
    assert all(not parameter.requires_grad for module in (phase2.intent, phase2.emotion) for parameter in module.parameters())
    artifact_path = save_final_inference_artifact(
        tmp_path / "final.bangmodel", ppo, phase2_artifact=phase2,
        job_snapshot={"job_id": "job-integration", "seed": 7, "split_seed": 23, "config_snapshot": {}, "versions": {"environment_version": "1.0", "observation_schema_version": "1.0", "action_space_version": "1.0", "reward_config_version": "reward-v1", "emotion_schema_version": "emotion-v1", "preprocessing_version": "1.0"}},
        config_snapshot={},
        training_metadata={"counters": {"environment_steps": trainer.environment_steps, "main_agent_transitions": trainer.main_agent_transitions}},
    )
    loaded = load_final_inference_artifact(artifact_path)
    result = loaded.act(np.ones(4, dtype=np.float32), mask, deterministic=True)
    assert mask[int(result.action)] == 1
    assert loaded.metadata["training"]["total_environment_steps"] == 1


def test_evaluation_application_pause_fault_cancel_and_metadata_linkage(tmp_path: Path) -> None:
    evaluation = EvaluationService().run_diagnostic(
        EvaluationRunConfig(
            config=SystemConfig(evaluation=EvaluationConfig(protocol_id="custom_v1", games=1, games_per_seat=1, seats=[0])),
            games=1,
            runner=lambda options, seeds: {"normal_games": 0, "truncated_games": 1, "normal_wins": 0},
            report_path=tmp_path / "evaluation.json",
        )
    )
    assert evaluation.win_rate == "N/A"
    assert evaluation.truncated_rate == 1.0
    assert evaluation.report_path and Path(evaluation.report_path).is_file()
    payload = json.loads(Path(evaluation.report_path).read_text(encoding="utf-8"))
    assert payload["seed_set"] == [0]

    controls = [WorkerControl()]
    coordinator = PauseCoordinator(controls, ack_timeout_seconds=0.05)
    controls[0].begin()
    controls[0].request_pause()
    assert coordinator.pause(timeout=0.01) is False
    assert coordinator.state is PauseState.PAUSE_FAILED
    controls[0].end()
    assert coordinator.pause(timeout=0.1) is True
    coordinator.resume()
    assert coordinator.state is PauseState.RUNNING
    coordinator.cancel()
    assert coordinator.state is PauseState.CANCELLED

    def runner(handle, run):
        del handle, run
        time.sleep(0.02)
        return {"environment_steps": 3, "main_agent_transitions": 1, "logs": [{"run_id": "job"}]}

    training = TrainingService()
    handle = training.start(TrainingConfig(config=SystemConfig(), runner=runner, report_path=tmp_path / "training.json"))
    report = training.wait(handle, timeout=2)
    assert report.status == "finished"
    assert report.progress.environment_steps == 3
    assert report.report_path and Path(report.report_path).is_file()

    def fault(_handle, _run):
        raise RuntimeError("integration fault")

    failed = training.start(TrainingConfig(config=SystemConfig(), runner=fault))
    failed_report = training.wait(failed, timeout=2)
    assert failed_report.status == "faulted"
    assert failed_report.progress.error["error_type"] == "RuntimeError"


def test_application_and_gui_controller_seams_have_no_cross_layer_gui_imports() -> None:
    root = Path(__file__).parents[1]
    for relative in ("env", "agents", "training", "app"):
        for source in (root / relative).rglob("*.py"):
            text = source.read_text(encoding="utf-8").lower()
            assert "pyqt6" not in text
            assert "pyqtgraph" not in text
    assert {mode.value for mode in GameMode} == {"human_random", "human_dummy", "human_main_dummy"}
    assert GameSessionController is not None
    assert (root / "app" / "services.py").is_file()
