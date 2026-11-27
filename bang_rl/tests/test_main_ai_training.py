"""Main-AI training: shared policy inputs, training games, PPO options, eval."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bang_rl.agents.main_agent import (
    EmotionInferenceModule,
    IntentInferenceModule,
    ObservationPreprocessor,
    PPOModule,
    RoleFusion,
    build_policy_inputs,
    policy_emotion_dim,
    policy_ppo_dims,
)
from bang_rl.app import TrainingConfig, TrainingService
from bang_rl.app.services import _ArtifactAgent
from bang_rl.config import EnvConfig, PPOConfig, SystemConfig
from bang_rl.env import BangEnvironment
from bang_rl.evaluation.protocol import PrimaryEvaluationProtocol
from bang_rl.exceptions import TrainingError
from bang_rl.training import (
    SL_ARTIFACT_FORMAT_VERSION,
    PPOTrainer,
    PPOTransition,
    compute_gae,
    load_final_inference_artifact,
)
from bang_rl.training import module_artifacts, ppo_trainer
from bang_rl.training.main_ai_training import (
    MAIN_AI_EVAL_BASE_SEED,
    PolicyModules,
    baseline_cache_key,
    load_or_compute_baselines,
    main_ai_reward_config,
    main_seat_for_game,
    normalized_scores,
    play_training_game,
    resolve_main_ai_source,
    train_main_ai,
    training_game_seed,
)


# --- fixtures -------------------------------------------------------------------


def _policy(seed: int = 1) -> PolicyModules:
    intent = IntentInferenceModule(role_count=4, seed=seed).eval()
    emotion = EmotionInferenceModule(seed=seed).eval()
    for module in (intent, emotion):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    ppo = PPOModule(**policy_ppo_dims(emotion), seed=seed)
    return PolicyModules(intent, emotion, RoleFusion(), ObservationPreprocessor(), ppo)


@pytest.fixture(scope="module")
def policy() -> PolicyModules:
    return _policy()


def _small_config(**ppo: int) -> SystemConfig:
    settings = {"rollout_transitions_per_worker": 8, "minibatch_size": 8, "ppo_epochs": 1}
    settings.update(ppo)
    return SystemConfig.model_validate({"rl_training": {"ppo": settings}})


def _write_phase2(path: Path, module: torch.nn.Module, artifact_type: str, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": artifact_type,
            "artifact_format_version": SL_ARTIFACT_FORMAT_VERSION,
            "metadata": metadata,
            "module_metadata": module.to_dict(),
            "state_dict": module.state_dict(),
            "training_mode": False,
        },
        path,
    )


@pytest.fixture(scope="module")
def saved_modules(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("models")
    modules = _policy(seed=5)
    intent_path = root / "intent" / "intent-job-a.phase2model"
    emotion_path = root / "emotion" / "emotion-job-a.phase2model"
    _write_phase2(intent_path, modules.intent, "phase2_intent_inference", {"training": {"total_epochs": 1}})
    _write_phase2(
        emotion_path,
        modules.emotion,
        "phase2_emotion_inference",
        {"emotion_schema_version": modules.emotion.emotion_schema_version, "training": {"total_epochs": 1}},
    )
    return root, intent_path, emotion_path


# --- shared policy inputs -------------------------------------------------------


def _first_observation(seed: int = 3):
    env = BangEnvironment(episode_id="inputs", seed=seed, config=EnvConfig())
    env.reset(seed=seed, episode_id="inputs")
    seat = int(str(env.agent_selection).removeprefix("player_"))
    return env, seat, env.observe(seat)


def test_policy_inputs_are_identical_in_evaluation_and_gui_paths(policy: PolicyModules) -> None:
    env, seat, observation = _first_observation()
    mask = np.asarray(env.action_mask, dtype=np.uint8)
    captured: list[dict[str, np.ndarray]] = []
    original_act = policy.ppo.act

    def recording_act(inputs, action_mask=None, **kwargs):
        captured.append({key: np.array(value, copy=True) for key, value in inputs.items()})
        return original_act(inputs, action_mask, **kwargs)

    policy.ppo.act = recording_act  # type: ignore[method-assign]
    try:
        protocol = PrimaryEvaluationProtocol.__new__(PrimaryEvaluationProtocol)
        protocol.artifact = SimpleNamespace(
            intent=policy.intent,
            emotion=policy.emotion,
            role_fusion=policy.role_fusion,
            preprocessor=policy.preprocessor,
            ppo=policy.ppo,
        )
        evaluation_action = protocol._main_action(observation, seat)[0]
        gui_action = _ArtifactAgent(protocol.artifact, f"player_{seat}", 0).act(observation, mask)
    finally:
        del policy.ppo.act

    assert evaluation_action == gui_action
    assert len(captured) == 2
    assert captured[0].keys() == captured[1].keys()
    for key in captured[0]:
        assert captured[0][key].dtype == captured[1][key].dtype, key
        assert captured[0][key].tobytes() == captured[1][key].tobytes(), key


def test_emotion_block_is_flat_and_accepted_by_ppo(policy: PolicyModules) -> None:
    _, seat, observation = _first_observation()
    inputs = policy.inputs(observation, seat)
    assert policy.emotion.emotion_count == 9
    assert policy_emotion_dim(policy.emotion) == 45  # one row per seat, own row zero
    assert inputs.features["emotion_features"].shape == (45,)
    assert inputs.estimated_emotion.shape == (4, 9)
    assert all(block.dtype == np.float32 for block in inputs.features.values())
    chosen = policy.ppo.act(inputs.ppo_inputs(), inputs.action_mask, deterministic=True)
    assert inputs.action_mask[int(chosen.action.reshape(-1)[0])] == 1
    direct = build_policy_inputs(
        observation,
        preprocessor=policy.preprocessor,
        intent=policy.intent,
        emotion=policy.emotion,
        role_fusion=policy.role_fusion,
        viewer_idx=seat,
    )
    for key, block in inputs.features.items():
        assert np.array_equal(block, direct.features[key])


# --- reward, seeds, seats -------------------------------------------------------


def test_reward_config_is_terminal_team_result_only() -> None:
    config = main_ai_reward_config()
    values = {component.id: component.value for component in config.terminal.components}
    assert values == {"terminal_win": 1.0, "terminal_loss": -1.0}
    assert config.shaping.components == []
    assert not config.clipping.enabled


def test_training_seeds_are_deterministic_and_leave_the_evaluation_range() -> None:
    assert training_game_seed(7, 3) == training_game_seed(7, 3)
    assert training_game_seed(7, 3) != training_game_seed(7, 4)
    assert training_game_seed(7, 3) != training_game_seed(8, 3)
    seeds = {training_game_seed(0, index) for index in range(500)}
    assert not any(MAIN_AI_EVAL_BASE_SEED <= seed < MAIN_AI_EVAL_BASE_SEED + 1_000_000 for seed in seeds)
    assert [main_seat_for_game(index) for index in range(7)] == [0, 1, 2, 3, 4, 0, 1]


# --- one training game ----------------------------------------------------------


def test_training_game_has_valid_transitions_and_team_terminal_reward(policy: PolicyModules) -> None:
    config = SystemConfig()
    roles = []
    for index in range(6):
        game = play_training_game(index, run_seed=11, config=config, policy=policy)
        summary = game.summary
        roles.append(summary.role)
        assert summary.won is not None, "default environment should not truncate"
        assert summary.main_decisions == len(game.transitions)
        if not game.transitions:
            continue
        for transition in game.transitions:
            assert transition.action_mask[transition.action] == 1
            assert transition.trainable and not transition.truncated
            assert transition.features["emotion_features"].__len__() == 45
        *earlier, last = game.transitions
        assert all(item.reward == 0.0 and not item.terminated and item.next_value is not None for item in earlier)
        assert last.terminated
        assert last.reward == (1.0 if summary.won else -1.0)
        assert summary.total_reward == last.reward
    # The main agent plays sheriff, deputy, outlaw and traitor in turn.
    assert roles == ["sheriff", "deputy", "outlaw", "traitor", "sheriff", "deputy"]


def test_training_game_is_deterministic_for_the_same_seed(policy: PolicyModules) -> None:
    config = SystemConfig()
    first = play_training_game(2, run_seed=21, config=config, policy=policy)
    second = play_training_game(2, run_seed=21, config=config, policy=policy)
    assert first.summary == second.summary
    assert [(t.action, t.old_log_prob, t.reward) for t in first.transitions] == [
        (t.action, t.old_log_prob, t.reward) for t in second.transitions
    ]
    other = play_training_game(2, run_seed=22, config=config, policy=policy)
    assert other.summary.seed != first.summary.seed


def _short_config() -> SystemConfig:
    env = EnvConfig(max_episode_steps=8, warn_turn_threshold=2, critical_turn_threshold=4)
    return SystemConfig(env=env)


def _truncated_game(policy: PolicyModules):
    config = _short_config()
    for index in range(10):
        game = play_training_game(index, run_seed=3, config=config, policy=policy)
        if game.summary.truncated and len(game.transitions) >= 2:
            return game
    pytest.fail("no truncated game with two decisions found")


def test_truncated_game_bootstraps_the_final_transition(policy: PolicyModules) -> None:
    game = _truncated_game(policy)
    *earlier, last = game.transitions
    assert game.summary.won is None
    assert all(item.trainable and not item.truncated for item in earlier)
    assert last.truncated and last.time_limit_bootstrap and last.trainable
    assert last.reward == 0.0
    assert last.next_value is not None and np.isfinite(last.next_value)
    assert all(item.reward == 0.0 for item in game.transitions)


def test_trainer_bootstraps_truncated_only_when_asked(policy: PolicyModules) -> None:
    game = _truncated_game(policy)
    count = len(game.transitions)
    config = PPOConfig(rollout_transitions_per_worker=8, minibatch_size=8, ppo_epochs=1)
    default = PPOTrainer(PPOModule(**policy_ppo_dims(policy.emotion), seed=2), config=config)
    assert default.update(game.transitions).samples == count - 1
    opted_in = PPOTrainer(PPOModule(**policy_ppo_dims(policy.emotion), seed=2), config=config, bootstrap_truncated=True)
    assert opted_in.update(game.transitions).samples == count


def test_gae_bootstraps_a_truncated_transition_only_on_request() -> None:
    args = ([0.0, 0.0], [0.5, 0.4], [0.4, 0.3], [False, False], [False, True])
    plain_adv, plain_ret = compute_gae(*args, gamma=0.9, gae_lambda=1.0, trainable=[True, False])
    assert plain_adv[1].item() == 0.0 and plain_ret[1].item() == 0.0
    adv, ret = compute_gae(*args, gamma=0.9, gae_lambda=1.0, trainable=[True, True], bootstrap_truncated=True)
    assert adv[1].item() == pytest.approx(0.9 * 0.3 - 0.4)
    assert ret[1].item() == pytest.approx(0.9 * 0.3)
    # The truncated record ends the chain: the first advantage uses its own value.
    assert adv[0].item() == pytest.approx(0.9 * 0.4 - 0.5 + 0.9 * adv[1].item())


# --- advantage normalization -----------------------------------------------------


def _vector_transitions(count: int) -> list[PPOTransition]:
    mask = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    rng = np.random.default_rng(0)
    return [
        PPOTransition(
            features=rng.normal(size=4).astype(np.float32),
            action=index % 3,
            action_mask=mask,
            old_log_prob=-1.0,
            value=0.0,
            reward=float(index % 4) * 3.0 if index < count - 1 else 5.0,
            next_value=0.0,
            terminated=index == count - 1,
            episode_id="e",
            transition_id=index,
            environment_step=index,
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("normalize", [False, True])
def test_advantage_normalization_is_opt_in(monkeypatch: pytest.MonkeyPatch, normalize: bool) -> None:
    seen: list[torch.Tensor] = []
    real = ppo_trainer.clipped_surrogate_loss

    def spy(old, current, advantages, epsilon):
        seen.append(torch.as_tensor(advantages).clone())
        return real(old, current, advantages, epsilon)

    monkeypatch.setattr(ppo_trainer, "clipped_surrogate_loss", spy)
    config = PPOConfig(rollout_transitions_per_worker=16, minibatch_size=16, ppo_epochs=1)
    trainer = PPOTrainer(PPOModule(input_dim=4, seed=0), config=config, normalize_advantages=normalize)
    trainer.update(_vector_transitions(16))
    advantages = seen[0]
    if normalize:
        assert advantages.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert advantages.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-4)
    else:
        assert abs(advantages.mean().item()) > 0.1


# --- baselines and scores ---------------------------------------------------------


def test_baselines_are_cached_by_deal_set_and_configuration(tmp_path: Path) -> None:
    config = SystemConfig()
    first = load_or_compute_baselines(config, games=5, cache_dir=tmp_path)
    assert first is not None and first[1] is False
    assert set(first[0]) == {"dummy", "random"}
    second = load_or_compute_baselines(config, games=5, cache_dir=tmp_path)
    assert second is not None and second[1] is True and second[0] == first[0]
    files = list(tmp_path.glob("*_baseline_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["info"]["kind"] == "baseline"
    # The cache is found by the key stored inside it, so renaming the file is safe.
    renamed = files[0].with_name("my own baseline name.json")
    files[0].rename(renamed)
    third = load_or_compute_baselines(config, games=5, cache_dir=tmp_path)
    assert third is not None and third[1] is True
    assert len(list(tmp_path.glob("*.json"))) == 1
    different = baseline_cache_key(config, games=6, base_seed=MAIN_AI_EVAL_BASE_SEED, seats=range(5))
    assert different != baseline_cache_key(config, games=5, base_seed=MAIN_AI_EVAL_BASE_SEED, seats=range(5))


def _table(**rates: float | None) -> dict:
    return {"by_role": {name: {"win_rate": rates.get(name)} for name in ("sheriff", "deputy", "outlaw", "traitor")}}


def test_normalized_score_is_zero_at_random_and_one_at_dummy() -> None:
    baselines = {
        "dummy": _table(sheriff=0.6, deputy=0.6, outlaw=0.8, traitor=0.4),
        "random": _table(sheriff=0.2, deputy=0.2, outlaw=0.4, traitor=0.4),
    }
    main = _table(sheriff=0.2, deputy=0.6, outlaw=1.0, traitor=0.5)
    scores = normalized_scores(main, baselines, eps=0.01)
    assert scores["by_role"]["sheriff"] == pytest.approx(0.0)
    assert scores["by_role"]["deputy"] == pytest.approx(1.0)
    assert scores["by_role"]["outlaw"] == pytest.approx(1.5)
    assert scores["by_role"]["traitor"] == pytest.approx(10.0)  # eps guards dummy == random
    missing = normalized_scores(_table(sheriff=0.6), baselines)
    assert missing["by_role"]["deputy"] is None and missing["mean"] == pytest.approx(1.0)


# --- module sources -------------------------------------------------------------


def test_missing_modules_raise_a_clear_training_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module_artifacts, "MODULE_MODEL_ROOT", tmp_path)
    run = SimpleNamespace(config=SimpleNamespace(), system_config=SystemConfig(), resume_artifact=None)
    with pytest.raises(TrainingError, match="학습"):
        resolve_main_ai_source(run, ppo_seed=0)
    with pytest.raises(TrainingError, match="inference modules first"):
        resolve_main_ai_source(run, ppo_seed=0)
    half = SimpleNamespace(
        config=SimpleNamespace(resume_intent_artifact_path=tmp_path / "x.phase2model"),
        system_config=SystemConfig(),
        resume_artifact=None,
    )
    with pytest.raises(TrainingError, match="Both an intent"):
        resolve_main_ai_source(half, ppo_seed=0)


def test_incompatible_newer_module_runs_are_skipped(
    tmp_path: Path, saved_modules: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, intent_path, emotion_path = saved_modules
    for kind, source in (("intent", intent_path), ("emotion", emotion_path)):
        target = tmp_path / kind / source.name
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        broken = tmp_path / kind / f"{kind}-job-b.phase2model"
        torch.save({"artifact_type": "something_else"}, broken)
        os.utime(broken, (time.time() + 60, time.time() + 60))
    monkeypatch.setattr(module_artifacts, "MODULE_MODEL_ROOT", tmp_path)
    run = SimpleNamespace(config=SimpleNamespace(), system_config=SystemConfig(), resume_artifact=None)
    source = resolve_main_ai_source(run, ppo_seed=0)
    assert source.description["module_job_id"] == "job-a"
    assert source.description["skipped_incompatible_runs"] == 1

    for broken in tmp_path.glob("*/*-job-a.phase2model"):
        broken.unlink()
    with pytest.raises(TrainingError, match="incompatible"):
        resolve_main_ai_source(run, ppo_seed=0)


def test_latest_saved_module_run_is_used_with_a_fresh_ppo(
    saved_modules: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = saved_modules
    monkeypatch.setattr(module_artifacts, "MODULE_MODEL_ROOT", root)
    run = SimpleNamespace(config=SimpleNamespace(), system_config=SystemConfig(), resume_artifact=None)
    source = resolve_main_ai_source(run, ppo_seed=4)
    assert source.origin == "latest_saved_modules"
    assert source.description["module_job_id"] == "job-a"
    policy = source.policy
    assert policy.ppo.component_dims["emotion"] == 45
    for module in (policy.intent, policy.emotion):
        assert not module.training and not any(p.requires_grad for p in module.parameters())
    assert all(p.requires_grad for p in policy.ppo.parameters())


# --- end to end through the training service --------------------------------------


def _request(config: SystemConfig, tmp_path: Path, saved: tuple[Path, Path, Path], **kwargs) -> TrainingConfig:
    _, intent_path, emotion_path = saved
    runner_options = kwargs.pop("runner_options", {})

    def runner(handle, run):
        return train_main_ai(
            handle,
            run,
            eval_games=5,
            eval_every_updates=1,
            baseline_cache_dir=tmp_path / "baselines",
            **runner_options,
        )

    return TrainingConfig(
        config=config,
        mode="main_ai_training",
        runner=runner,
        report_path=tmp_path / "report.json",
        output_path=tmp_path / "out",
        artifact_path=tmp_path / "main.bangmodel",
        resume_artifact_path=None,
        resume_intent_artifact_path=intent_path,
        resume_emotion_artifact_path=emotion_path,
        seed=9,
        **kwargs,
    )


def test_main_ai_training_end_to_end_saves_a_loadable_best_model(
    tmp_path: Path, saved_modules: tuple[Path, Path, Path]
) -> None:
    service = TrainingService()
    handle = service.start(_request(_small_config(), tmp_path, saved_modules, episodes=6))
    report = service.wait(handle, timeout=600)
    assert report.status == "finished", report.progress.error

    progress = report.progress
    metrics = progress.metrics
    assert progress.episodes_completed == 6
    assert progress.ppo_updates >= 1
    assert set(metrics["baselines"]) == {"dummy", "random"}
    assert metrics["baselines_from_cache"] is False
    assert metrics["best_eval"] is not None
    assert "normalized_score" in metrics["best_eval"]
    assert metrics["eval_history"] and metrics["main_eval_win_rate_points"]
    assert metrics["truncated_rate"] == 0.0
    assert metrics["main_policy_loss_points"] and metrics["main_win_rate_points"]
    for key in ("role_accuracy", "hidden_role_accuracy", "emotion_mse"):
        assert key in metrics["best_eval"]

    path = tmp_path / "main.bangmodel"
    assert progress.artifact_path == str(path)
    artifact = load_final_inference_artifact(path)
    assert artifact.metadata["main_ai_training"]["reward"]["terminal_win"] == 1.0
    assert artifact.metadata["main_ai_training"]["source"]["origin"] == "phase2_pair"
    assert artifact.ppo.component_dims["emotion"] == 45
    assert artifact.metadata["main_ai_training"]["baselines"]["dummy"]["games"] == 5

    # The same deals and configuration reuse the cached baselines.
    again = service.wait(service.start(_request(_small_config(), tmp_path, saved_modules, episodes=2)), timeout=600)
    assert again.status == "finished", again.progress.error
    assert again.progress.metrics["baselines_from_cache"] is True


def test_main_ai_training_can_continue_from_a_bangmodel_and_stops_on_cancel(
    tmp_path: Path, saved_modules: tuple[Path, Path, Path]
) -> None:
    service = TrainingService()
    first = service.wait(
        service.start(_request(_small_config(), tmp_path, saved_modules, episodes=4)), timeout=600
    )
    assert first.status == "finished", first.progress.error
    seed_model = tmp_path / "main.bangmodel"
    assert seed_model.is_file()
    continued = tmp_path / "continued"
    request = _request(_small_config(), tmp_path, saved_modules, train_by_time=True)
    request = replace(
        request,
        artifact_path=continued / "again.bangmodel",
        resume_artifact_path=seed_model,
        resume_intent_artifact_path=None,
        resume_emotion_artifact_path=None,
    )
    handle = service.start(request)
    deadline = time.monotonic() + 300
    while True:
        progress = service.progress(handle)
        assert progress.status != "faulted", progress.error
        if progress.metrics.get("main_eval_win_rate_points"):
            break
        assert time.monotonic() < deadline, "no evaluation finished in time"
        time.sleep(0.1)
    service.cancel(handle)
    report = service.wait(handle, timeout=300)
    assert report.status == "cancelled"
    assert report.progress.metrics["stopped_early"] is True
    assert report.progress.metrics["source"]["origin"] == "bangmodel"
    assert load_final_inference_artifact(continued / "again.bangmodel") is not None


def test_main_ai_training_requires_a_budget(tmp_path: Path, saved_modules: tuple[Path, Path, Path]) -> None:
    service = TrainingService()
    handle = service.start(_request(_small_config(), tmp_path, saved_modules, episodes=0))
    report = service.wait(handle, timeout=120)
    assert report.status == "faulted"
    assert "episodes" in str(report.progress.error)
