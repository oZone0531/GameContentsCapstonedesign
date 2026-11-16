"""main1+dummy4 collection, mixed-data module training, inference evaluation and
the inference refinement loop (PPO fixed)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
from bang_rl.agents.main_agent import (
    EmotionInferenceModule,
    IntentInferenceModule,
    ObservationPreprocessor,
    PPOModule,
    RoleFusion,
    policy_emotion_dim,
)
from bang_rl.app import TrainingConfig, TrainingService
from bang_rl.artifact_info import KIND_DATASET, KIND_MAIN_AI, new_info, read_info
from bang_rl.config import SystemConfig
from bang_rl.data.collector import CanonicalEpisode
from bang_rl.data.game_collection import collect_games
from bang_rl.data.parquet_writer import ParquetCanonicalShardWriter
from bang_rl.data.splitter import EpisodeSplitter
from bang_rl.data.tensor_cache import NPZTensorCacheWriter
from bang_rl.evaluation.protocol import PRIMARY_SEATS
from bang_rl.exceptions import TrainingError
from bang_rl.training import inference_refinement, pipeline_runner
from bang_rl.training.dataset_artifacts import list_datasets
from bang_rl.training.final_inference_artifact import (
    load_final_artifact_for_training,
    load_final_inference_artifact,
    save_final_inference_artifact,
)
from bang_rl.training.inference_evaluation import (
    ModulePair,
    check_modules_fit_ppo,
    evaluate_inference,
    evaluate_module_pairs,
)
from bang_rl.training.inference_refinement import (
    ACCEPT_ACCURACY_TOLERANCE,
    accept_candidate,
    module_fingerprint,
    run_inference_refinement,
)
from bang_rl.training.main_ai_collection import (
    collect_main_ai_games,
    collection_game_seed,
    main_ai_dataset_info,
    play_collection_game,
    policy_from_artifact,
)
from bang_rl.training.main_ai_training import (
    MAIN_AI_EVAL_BASE_SEED,
    PolicyModules,
    main_seat_for_game,
    train_main_ai,
)


# --- fixtures -------------------------------------------------------------------


def _policy(seed: int = 1) -> PolicyModules:
    intent = IntentInferenceModule(role_count=4, seed=seed).eval()
    emotion = EmotionInferenceModule(seed=seed).eval()
    for module in (intent, emotion):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    ppo = PPOModule(emotion_dim=policy_emotion_dim(emotion), seed=seed)
    return PolicyModules(intent, emotion, RoleFusion(), ObservationPreprocessor(), ppo)


def _save_base_model(path: Path, seed: int = 1) -> Path:
    policy = _policy(seed)
    info = new_info(KIND_MAIN_AI, id=f"base-model-{seed}", created_by="test", conditions={"training_games": 10})
    save_final_inference_artifact(
        path,
        policy.ppo,
        info=info,
        intent=policy.intent,
        emotion=policy.emotion,
        role_fusion=policy.role_fusion,
        preprocessor=policy.preprocessor,
        config_snapshot=SystemConfig(),
        training_metadata={"counters": {"ppo_updates": 3}},
        metadata={"main_ai_training": {"training_games": 10, "source": {"origin": "test"}}},
    )
    return path


@pytest.fixture(scope="module")
def base_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _save_base_model(tmp_path_factory.mktemp("base") / "base.bangmodel")


@pytest.fixture(scope="module")
def dummy_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("dummy5") / "dummy-dataset"
    collect_games(SystemConfig(), 12, folder, base_seed=100, episode_id_prefix="d5")
    return folder


# --- acceptance rule ------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "accepted", "reason"),
    [
        ({"hidden_role_accuracy": 0.60, "emotion_mse": 0.050}, True, "improved_hidden_role_accuracy"),
        ({"hidden_role_accuracy": 0.50, "emotion_mse": 0.040}, True, "improved_emotion_mse"),
        ({"hidden_role_accuracy": 0.60, "emotion_mse": 0.040}, True, "improved_both"),
        # Within the tolerance on both: no meaningful gain.
        ({"hidden_role_accuracy": 0.503, "emotion_mse": 0.0499}, False, "no_meaningful_improvement"),
        # Exactly at the accuracy tolerance is not an improvement (strict >).
        ({"hidden_role_accuracy": 0.505, "emotion_mse": 0.050}, False, "no_meaningful_improvement"),
        # Accuracy up a lot but MSE meaningfully worse: rejected.
        ({"hidden_role_accuracy": 0.70, "emotion_mse": 0.060}, False, "emotion_mse_worse"),
        # MSE down a lot but accuracy meaningfully worse: rejected.
        ({"hidden_role_accuracy": 0.45, "emotion_mse": 0.020}, False, "hidden_role_accuracy_worse"),
        # Slightly worse MSE (<1%) with a real accuracy gain is allowed.
        ({"hidden_role_accuracy": 0.60, "emotion_mse": 0.0504}, True, "improved_hidden_role_accuracy"),
        ({"hidden_role_accuracy": None, "emotion_mse": 0.040}, False, "metric_missing"),
    ],
)
def test_acceptance_rule_cases(candidate: dict, accepted: bool, reason: str) -> None:
    best = {"hidden_role_accuracy": 0.50, "emotion_mse": 0.050}
    assert accept_candidate(best, candidate) == (accepted, reason)


def test_acceptance_rule_missing_best_metric_rejects() -> None:
    assert accept_candidate({"hidden_role_accuracy": None, "emotion_mse": 0.05}, {"hidden_role_accuracy": 0.9, "emotion_mse": 0.01}) == (
        False,
        "metric_missing",
    )
    assert ACCEPT_ACCURACY_TOLERANCE == 0.005


# --- main1+dummy4 collection ----------------------------------------------------


def test_collection_game_keeps_only_main_seat_rows_with_dummy_labels() -> None:
    policy = _policy()
    config = SystemConfig()
    for seat in range(5):
        episode, summary = play_collection_game(
            config, policy, seed=500 + seat, seat=seat, episode_id=f"g-{seat}"
        )
        assert isinstance(episode, CanonicalEpisode)
        assert episode.metadata["viewpoint"] == "main_only"
        assert episode.metadata["viewpoint_seat"] == seat
        assert episode.metadata["players"] == "main1-dummy4"
        assert episode.metadata["main_decisions"] == len(episode.steps) == summary["main_decisions"]
        assert summary["environment_steps"] > len(episode.steps), "dummy steps must be dropped"
        assert episode.role_labels is not None and len(episode.role_labels) == 5
        assert episode.steps, "the main AI acts at least once"
        for step in episode.steps:
            assert step.actor_idx == seat
            # The viewer's (main AI's) own emotion is never a target; the
            # other living dummies are, with nine entries each.
            assert str(seat) not in step.emotion_labels
            assert step.emotion_labels, "four dummies are alive at least until the first elimination"
            for key, label in step.emotion_labels.items():
                assert int(key) != seat
                assert len(label["values"]) == 9
                assert label["emotion_names"] == list(LABEL_NAMES)
            # The observation is the main seat's own view.
            assert int(np.asarray(step.pre_observation["current_player_idx"]).reshape(-1)[0]) == seat


def test_collection_game_is_reproducible_and_stochastic_mode_differs_in_policy_only() -> None:
    policy = _policy()
    config = SystemConfig()
    first, _ = play_collection_game(config, policy, seed=77, seat=2, episode_id="a")
    second, _ = play_collection_game(config, policy, seed=77, seat=2, episode_id="a")
    assert [step.action for step in first.steps] == [step.action for step in second.steps]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    stochastic, _ = play_collection_game(
        config, policy, seed=77, seat=2, episode_id="a", action_mode="stochastic", generator=generator
    )
    assert stochastic.metadata["main_action_mode"] == "stochastic"
    with pytest.raises(ValueError):
        play_collection_game(config, policy, seed=77, seat=2, episode_id="a", action_mode="greedy")


def test_collection_seeds_avoid_the_evaluation_deals() -> None:
    seeds = {collection_game_seed(base, index) for base in (0, 1, MAIN_AI_EVAL_BASE_SEED) for index in range(300)}
    assert not any(MAIN_AI_EVAL_BASE_SEED <= seed < MAIN_AI_EVAL_BASE_SEED + 1_000_000 for seed in seeds)
    assert collection_game_seed(3, 4) == collection_game_seed(3, 4)
    assert collection_game_seed(3, 4) != collection_game_seed(3, 5)
    assert collection_game_seed(3, 4) != collection_game_seed(4, 4)


@pytest.fixture(scope="module")
def main_dataset(tmp_path_factory: pytest.TempPathFactory, base_model: Path) -> tuple[Path, dict]:
    policy = policy_from_artifact(load_final_inference_artifact(base_model))
    info = main_ai_dataset_info(
        games=10, base_seed=5, created_by="test", created_at=None, action_mode="deterministic", model_path=base_model
    )
    folder = tmp_path_factory.mktemp("main1") / info.name  # the naming rule names the folder
    summary = collect_main_ai_games(SystemConfig(), policy, 10, folder, base_seed=5, info=info, source_model_path=base_model)
    return folder, summary


def test_main_dataset_summary_info_and_rows(main_dataset: tuple[Path, dict], base_model: Path) -> None:
    folder, summary = main_dataset
    episodes = ParquetCanonicalShardWriter.read_dataset(folder)
    rows = sum(len(episode.steps) for episode in episodes)
    assert summary["rows"] == summary["main_decisions"] == rows > 0
    assert summary["games"] == 10 and summary["viewpoint"] == "main_only"
    assert sum(summary["main_seat_games"].values()) == 10
    assert all(count == 2 for count in summary["main_seat_games"].values())  # seat = game_index % 5
    assert summary["total_environment_steps"] > rows
    for episode in episodes:
        seat = episode.metadata["viewpoint_seat"]
        assert all(step.actor_idx == seat for step in episode.steps)
        assert episode.seed not in range(MAIN_AI_EVAL_BASE_SEED, MAIN_AI_EVAL_BASE_SEED + 1_000_000)

    info = read_info(folder)
    assert info is not None and info.kind == KIND_DATASET
    assert info.conditions["players"] == "main1-dummy4"
    assert info.conditions["viewpoint"] == "main_only"
    assert info.conditions["main_action_mode"] == "deterministic"
    assert info.conditions["games"] == 10 and info.conditions["base_seed"] == 5
    assert "main1-dummy4" in info.name
    assert any(ref["kind"] == KIND_MAIN_AI and ref["id"] == "base-model-1" for ref in info.sources)
    assert info.summary["rows"] == rows and info.summary["main_decisions"] == rows
    assert json.loads((folder / "collection_summary.json").read_text(encoding="utf-8"))["info"]["kind"] == KIND_DATASET


def test_collection_requires_an_empty_folder_and_honours_stop(tmp_path: Path, base_model: Path) -> None:
    policy = policy_from_artifact(load_final_inference_artifact(base_model))
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        collect_main_ai_games(SystemConfig(), policy, 2, taken)
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    summary = collect_main_ai_games(
        SystemConfig(), policy, 6, tmp_path / "stopped", base_seed=1, should_stop=stop_after_two
    )
    assert summary["stopped"] is True and summary["games"] == 2


# --- mixed datasets: selection, cache, training ------------------------------------


def test_dataset_listing_describes_players_and_games(dummy_dataset: Path, main_dataset: tuple[Path, dict]) -> None:
    main_folder, _ = main_dataset
    root = dummy_dataset.parent
    found = {item.path: item for item in list_datasets(root, saved_root=None)}
    assert found[dummy_dataset].players == "dummy5" and found[dummy_dataset].games == 12
    mixed = {item.path: item for item in list_datasets(main_folder.parent, saved_root=None)}
    assert mixed[main_folder].players == "main1-dummy4" and mixed[main_folder].is_main_ai
    assert mixed[main_folder].games == 10 and mixed[main_folder].rows > 0
    assert "main1-dummy4" in mixed[main_folder].label


def test_mixed_datasets_build_one_cache_with_main_rows_from_the_main_seat(
    dummy_dataset: Path, main_dataset: tuple[Path, dict], tmp_path: Path
) -> None:
    main_folder, summary = main_dataset
    episodes, shards, duplicates, selection = pipeline_runner._read_canonical_dataset(
        tmp_path, only_folders=[dummy_dataset, main_folder]
    )
    assert selection["selected_episodes_by_players"] == {"dummy5": 12, "main1-dummy4": 10}
    assert len(episodes) == 22 and duplicates == 0

    only_dummy, _, _, only_selection = pipeline_runner._read_canonical_dataset(tmp_path, only_folders=[dummy_dataset])
    assert len(only_dummy) == 12 and only_selection["selected_episodes_by_players"] == {"dummy5": 12}
    # Reading the same folder twice must not duplicate episodes.
    twice, _, again, _ = pipeline_runner._read_canonical_dataset(tmp_path, only_folders=[dummy_dataset, dummy_dataset])
    assert len(twice) == 12

    split = EpisodeSplitter(seed=3, manifest_path=tmp_path / "cache" / "split_manifest.json").split(episodes)
    artifacts = NPZTensorCacheWriter(
        tmp_path / "cache", emotion_schema_version="formula-1.0+label9", emotion_dimension=9, split_manifest=split.manifest
    ).write_all(split)
    by_id = {episode.episode_id: episode for episode in episodes}
    total_main_rows = 0
    for artifact in artifacts.values():
        ids = artifact.manifest["episode_ids"]
        with np.load(artifact.tensor_path) as data:
            actors = np.asarray(data["actor_indices"])
        # One row per stored step: main episodes contribute only the main seat's rows.
        assert len(actors) == sum(len(by_id[item].steps) for item in ids)
        total_main_rows += sum(
            len(by_id[item].steps) for item in ids if by_id[item].metadata.get("players") == "main1-dummy4"
        )
    normal_main = sum(
        len(e.steps) for e in episodes if e.metadata.get("players") == "main1-dummy4" and e.termination_status == "normal"
    )
    assert total_main_rows == normal_main and 0 < normal_main <= summary["rows"]


# --- inference evaluation ---------------------------------------------------------


def test_inference_evaluation_reports_primary_and_detail_metrics() -> None:
    policy = _policy()
    result = evaluate_inference(policy, SystemConfig(), games=6)
    assert result is not None and result["games"] == 6
    for key in ("role_accuracy", "hidden_role_accuracy", "emotion_mse", "win_rate"):
        assert key in result
    assert 0.0 <= result["role_accuracy"] <= 1.0
    assert result["emotion_mse"] >= 0.0
    detail = result["inference_detail"]
    assert set(detail["role_accuracy_by_target_role"]) == {"sheriff", "deputy", "outlaw", "traitor"}
    assert set(detail["emotion_mse_by_entry"]) == set(LABEL_NAMES)
    assert sum(detail["role_counts_by_target_role"].values()) > 0
    # Fixed deals + deterministic policy: the evaluation repeats exactly.
    again = evaluate_inference(policy, SystemConfig(), games=6)
    assert again["role_accuracy"] == result["role_accuracy"] and again["emotion_mse"] == result["emotion_mse"]
    assert evaluate_inference(policy, SystemConfig(), games=6, should_stop=lambda: True) is None
    assert PRIMARY_SEATS == (0, 1, 2, 3, 4)


def test_module_pairs_are_evaluated_under_the_fixed_ppo(base_model: Path) -> None:
    base = load_final_inference_artifact(base_model)
    other = _policy(seed=9)
    before = module_fingerprint(base.ppo)
    progress: list[tuple[int, int, str]] = []
    rows = evaluate_module_pairs(
        base,
        [ModulePair("other", other.intent, other.emotion)],
        SystemConfig(),
        games=4,
        progress=lambda done, total, label: progress.append((done, total, label)),
    )
    assert [row["label"] for row in rows] == ["base", "other"]
    assert module_fingerprint(base.ppo) == before
    assert all("hidden_role_accuracy" in row and "emotion_mse_by_entry" in row for row in rows)
    assert progress[-1][:2] == (2, 2)
    wrong = PPOModule(emotion_dim=policy_emotion_dim(other.emotion) + 4, seed=1)
    with pytest.raises(TrainingError):
        check_modules_fit_ppo(wrong, other.emotion)


# --- refinement loop --------------------------------------------------------------


def _refine_request(tmp_path: Path, base: Path, dataset: Path, runner, **kwargs) -> TrainingConfig:
    values = dict(
        config=SystemConfig.model_validate({"sl_training": {"epochs": 1, "batch_size": 32}}),
        mode="inference_refinement",
        runner=runner,
        episodes=5,
        refine_rounds=2,
        refine_epochs=1,
        refine_patience=5,
        refine_eval_games=5,
        base_model_path=base,
        dataset_paths=[dataset],
        report_path=tmp_path / "report.json",
        output_path=tmp_path / "out",
        artifact_path=tmp_path / "refined.bangmodel",
        resume_artifact_path=None,
        seed=4,
    )
    values.update(kwargs)
    return TrainingConfig(**values)


def _runner(tmp_path: Path):
    def runner(handle, run):
        return run_inference_refinement(
            handle,
            run,
            dataset_root=tmp_path / "datasets",
            cache_root=tmp_path / "cache",
            model_root=tmp_path / "models",
        )

    return runner


def _state_equal(first: torch.nn.Module, second: torch.nn.Module) -> bool:
    return module_fingerprint(first) == module_fingerprint(second)


def test_refinement_loop_with_accepted_rounds_keeps_ppo_and_changes_modules(
    tmp_path: Path, base_model: Path, dummy_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inference_refinement, "accept_candidate", lambda best, candidate: (True, "improved_both"))
    service = TrainingService()
    report = service.wait(service.start(_refine_request(tmp_path, base_model, dummy_dataset, _runner(tmp_path))), timeout=900)
    assert report.status == "finished", report.progress.error
    metrics = report.progress.metrics
    assert [item["round"] for item in metrics["history"]] == [0, 1, 2]
    assert metrics["history"][0]["reason"] == "base_reference"
    assert metrics["best_round"] == 2 and metrics["stop_reason"] == "max_rounds"
    assert metrics["ppo_updated"] is False and metrics["rounds_run"] == 2
    for key in ("role_accuracy", "hidden_role_accuracy", "emotion_mse", "win_rate"):
        assert len(metrics[f"refine_{key}_points"]) == 3
    assert list(metrics["refine_accepted_points"]) == [1, 1, 1]
    assert metrics["refine_round"] == 2 and metrics["refine_best_round"] == 2

    base = load_final_inference_artifact(base_model)
    refined_path = tmp_path / "refined.bangmodel"
    assert report.progress.artifact_path == str(refined_path)
    refined = load_final_inference_artifact(refined_path)
    # The PPO is exactly the base PPO; the modules were refined.
    assert _state_equal(refined.ppo, base.ppo)
    assert not _state_equal(refined.intent, base.intent)
    assert not _state_equal(refined.emotion, base.emotion)
    assert module_fingerprint(base.ppo) == metrics["ppo_fingerprint"]

    info = read_info(refined_path)
    assert info is not None and info.kind == KIND_MAIN_AI and info.created_by == "inference_refinement"
    assert "refined" in info.name
    kinds = {ref["kind"] for ref in info.sources}
    assert {"main_ai", "dataset", "intent_module", "emotion_module"} <= kinds
    assert any(ref["id"] == "base-model-1" for ref in info.sources)
    assert info.conditions["ppo_updated"] is False and info.conditions["games_per_round"] == 5
    assert len(info.summary["rounds"]) == 3 and info.summary["best_round"] == 2
    assert refined.metadata["inference_refinement"]["best_round"] == 2
    assert refined.metadata["main_ai_training"]["training_games"] == 10  # PPO games are unchanged

    # One main1+dummy4 dataset per round plus module pair files with info.
    datasets = list_datasets(tmp_path / "datasets", saved_root=None)
    assert len(datasets) == 2 and all(item.players == "main1-dummy4" and item.games == 5 for item in datasets)
    assert all(read_info(Path(path)) is not None for path in (metrics["intent_artifact_path"], metrics["emotion_artifact_path"]))
    assert (tmp_path / "cache").is_dir()
    # The saved module files exist for the accepted rounds.
    assert len(metrics["saved_module_pairs"]) == 2


def test_refinement_loop_keeps_the_base_modules_when_every_round_is_rejected(
    tmp_path: Path, base_model: Path, dummy_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inference_refinement, "accept_candidate", lambda best, candidate: (False, "no_meaningful_improvement"))
    service = TrainingService()
    report = service.wait(
        service.start(_refine_request(tmp_path, base_model, dummy_dataset, _runner(tmp_path), refine_rounds=3, refine_patience=1)),
        timeout=900,
    )
    assert report.status == "finished", report.progress.error
    metrics = report.progress.metrics
    # Patience 1: stops after the first rejected round.
    assert metrics["stop_reason"] == "patience" and metrics["rounds_run"] == 1 and metrics["best_round"] == 0
    assert [item["accepted"] for item in metrics["history"]] == [True, False]
    assert metrics["history"][1]["reason"] == "no_meaningful_improvement"
    base = load_final_inference_artifact(base_model)
    refined = load_final_inference_artifact(tmp_path / "refined.bangmodel")
    assert _state_equal(refined.ppo, base.ppo)
    assert _state_equal(refined.intent, base.intent) and _state_equal(refined.emotion, base.emotion)
    assert not metrics["saved_module_pairs"]
    assert len(list_datasets(tmp_path / "datasets", saved_root=None)) == 1  # data is kept


def test_refinement_can_be_cancelled_and_still_leaves_the_best_model(
    tmp_path: Path, base_model: Path, dummy_dataset: Path
) -> None:
    service = TrainingService()
    request = _refine_request(tmp_path, base_model, dummy_dataset, _runner(tmp_path), refine_rounds=3, episodes=200)
    handle = service.start(request)
    deadline = time.monotonic() + 300
    while service.progress(handle).metrics.get("refine_stage") != "refine_collecting":
        assert service.progress(handle).status != "faulted", service.progress(handle).error
        assert time.monotonic() < deadline
        time.sleep(0.05)
    service.cancel(handle)
    report = service.wait(handle, timeout=300)
    assert report.status == "cancelled"
    assert report.progress.metrics["stopped_early"] is True
    refined = load_final_inference_artifact(tmp_path / "refined.bangmodel")
    base = load_final_inference_artifact(base_model)
    assert _state_equal(refined.ppo, base.ppo) and _state_equal(refined.intent, base.intent)


def test_refinement_needs_a_base_model_and_datasets(tmp_path: Path, base_model: Path, dummy_dataset: Path) -> None:
    service = TrainingService()
    missing = service.wait(
        service.start(_refine_request(tmp_path, tmp_path / "nope.bangmodel", dummy_dataset, _runner(tmp_path))), timeout=60
    )
    assert missing.status == "faulted" and ".bangmodel" in str(missing.progress.error)
    empty = tmp_path / "empty"
    empty.mkdir()
    no_data = service.wait(
        service.start(_refine_request(tmp_path, base_model, empty, _runner(tmp_path))), timeout=60
    )
    assert no_data.status == "faulted"
    bad = service.wait(
        service.start(_refine_request(tmp_path, base_model, dummy_dataset, _runner(tmp_path), refine_patience=0)), timeout=60
    )
    assert bad.status == "faulted" and "patience" in str(bad.progress.error)


def test_refined_model_can_continue_ppo_training_with_frozen_modules(
    tmp_path: Path, base_model: Path, dummy_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inference_refinement, "accept_candidate", lambda best, candidate: (True, "improved_both"))
    service = TrainingService()
    first = service.wait(
        service.start(_refine_request(tmp_path, base_model, dummy_dataset, _runner(tmp_path), refine_rounds=1)), timeout=900
    )
    assert first.status == "finished", first.progress.error
    refined_path = tmp_path / "refined.bangmodel"
    loaded = load_final_artifact_for_training(refined_path)
    assert loaded.training_ready
    assert all(not p.requires_grad for p in loaded.intent.parameters())
    assert all(not p.requires_grad for p in loaded.emotion.parameters())

    def main_runner(handle, run):
        return train_main_ai(handle, run, eval_games=4, eval_every_updates=1, baseline_cache_dir=tmp_path / "baselines")

    continued = tmp_path / "continued.bangmodel"
    report = service.wait(
        service.start(
            TrainingConfig(
                config=SystemConfig.model_validate(
                    {"rl_training": {"ppo": {"rollout_transitions_per_worker": 8, "minibatch_size": 8, "ppo_epochs": 1}}}
                ),
                mode="main_ai_training",
                runner=main_runner,
                episodes=4,
                report_path=tmp_path / "continue-report.json",
                output_path=tmp_path / "continue-out",
                artifact_path=continued,
                resume_artifact_path=refined_path,
                seed=2,
            )
        ),
        timeout=900,
    )
    assert report.status == "finished", report.progress.error
    assert report.progress.metrics["source"]["origin"] == "bangmodel"
    after = load_final_inference_artifact(continued)
    refined = load_final_inference_artifact(refined_path)
    assert _state_equal(after.intent, refined.intent) and _state_equal(after.emotion, refined.emotion)
    assert not _state_equal(after.ppo, refined.ppo)


# --- data collection / module training through the service -------------------------


def test_data_collection_service_collects_main1_dummy4_games(tmp_path: Path, base_model: Path) -> None:
    config = SystemConfig.model_validate({"dataset": {"canonical_path": str(tmp_path / "canonical")}})
    service = TrainingService()

    def request(players: str = "main1-dummy4", **kwargs) -> TrainingConfig:
        return TrainingConfig(
            config=config,
            mode="data_collection",
            episodes=5,
            collection_players=players,
            report_path=tmp_path / "collect.json",
            output_path=tmp_path / "out",
            resume_artifact_path=None,
            seed=3,
            **kwargs,
        )

    report = service.wait(service.start(request(base_model_path=base_model, main_action_mode="stochastic")), timeout=300)
    assert report.status == "finished", report.progress.error
    metrics = report.progress.metrics
    assert metrics["players"] == "main1-dummy4" and metrics["viewpoint"] == "main_only"
    assert metrics["collection_rows"] == metrics["collection_main_decisions"] > 0
    assert report.progress.episodes_completed == 5
    folder = Path(metrics["dataset_path"])
    info = read_info(folder)
    assert info is not None and info.conditions["players"] == "main1-dummy4"
    assert info.conditions["main_action_mode"] == "stochastic"
    assert [item["id"] for item in info.sources if item["kind"] == KIND_MAIN_AI] == ["base-model-1"]
    assert folder.parent == tmp_path / "canonical"

    # No model chosen: a clear error (a folder would silently pick the newest model).
    failed = service.wait(service.start(request(base_model_path=tmp_path)), timeout=60)
    assert failed.status == "faulted" and ".bangmodel" in str(failed.progress.error)
    unknown = service.wait(service.start(request(players="all", base_model_path=base_model)), timeout=60)
    assert unknown.status == "faulted"


def test_module_training_trains_on_selected_mixed_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_dataset: Path, main_dataset: tuple[Path, dict]
) -> None:
    monkeypatch.setattr(pipeline_runner, "PROJECT_ROOT", tmp_path)
    config = SystemConfig.model_validate(
        {
            "dataset": {"canonical_path": str(tmp_path / "empty-canonical"), "cache_path": str(tmp_path / "cache")},
            "sl_training": {"epochs": 1, "batch_size": 32},
        }
    )
    service = TrainingService()
    report = service.wait(
        service.start(
            TrainingConfig(
                config=config,
                mode="module_training",
                episodes=1,
                dataset_paths=[dummy_dataset, main_dataset[0]],
                report_path=tmp_path / "modules.json",
                output_path=tmp_path / "out",
                resume_artifact_path=None,
            )
        ),
        timeout=600,
    )
    assert report.status == "finished", report.progress.error
    selection = report.progress.metrics["dataset_selection"]
    assert selection["selected_episodes_by_players"] == {"dummy5": 12, "main1-dummy4": 10}
    assert len(selection["chosen_dataset_folders"]) == 2
    intent_info = read_info(Path(report.progress.metrics["intent_artifact_path"]))
    assert intent_info is not None
    assert intent_info.conditions["dataset_episodes_by_players"] == {"dummy5": 12, "main1-dummy4": 10}
    assert len([ref for ref in intent_info.sources if ref["kind"] == KIND_DATASET]) == 2


def test_dummy_episode_cap_scales_with_main_games() -> None:
    from bang_rl.training.inference_refinement import REFINE_MIN_DUMMY_EPISODES, refine_dummy_episode_cap

    assert refine_dummy_episode_cap(1000) == 4000
    assert refine_dummy_episode_cap(2000, 2.0) == 4000
    assert refine_dummy_episode_cap(10) == REFINE_MIN_DUMMY_EPISODES


def test_limited_folders_are_read_only_up_to_the_cap(tmp_path) -> None:
    from bang_rl.config import SystemConfig
    from bang_rl.data.game_collection import collect_games
    from bang_rl.training.pipeline_runner import _read_canonical_dataset

    big, small = tmp_path / "big", tmp_path / "small"
    collect_games(SystemConfig(), 45, big, base_seed=0, episode_id_prefix="big")
    collect_games(SystemConfig(), 5, small, base_seed=100, episode_id_prefix="small")
    episodes, _shards, _dup, selection = _read_canonical_dataset(
        tmp_path, only_folders=[big, small], limited_folders=[big], limit_episodes=12, limit_seed=3
    )
    from_big = [item for item in episodes if str(item.episode_id).startswith("big")]
    assert len(from_big) == 12
    assert len(episodes) == 12 + 5  # the uncapped folder is read completely
    assert selection["limited_episodes_taken"] == 12
    again, *_ = _read_canonical_dataset(
        tmp_path, only_folders=[big, small], limited_folders=[big], limit_episodes=12, limit_seed=4
    )
    assert {item.episode_id for item in again} != {item.episode_id for item in episodes}


def test_module_training_max_games_caps_episodes_and_records_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_dataset: Path, main_dataset: tuple[Path, dict]
) -> None:
    monkeypatch.setattr(pipeline_runner, "PROJECT_ROOT", tmp_path)
    config = SystemConfig.model_validate(
        {
            "dataset": {"canonical_path": str(tmp_path / "empty-canonical"), "cache_path": str(tmp_path / "cache")},
            "sl_training": {"epochs": 1, "batch_size": 32},
        }
    )
    service = TrainingService()
    report = service.wait(
        service.start(
            TrainingConfig(
                config=config,
                mode="module_training",
                episodes=1,
                dataset_paths=[dummy_dataset, main_dataset[0]],
                max_games=8,
                subset_seed=5,
                report_path=tmp_path / "modules.json",
                output_path=tmp_path / "out",
                resume_artifact_path=None,
            )
        ),
        timeout=600,
    )
    assert report.status == "finished", report.progress.error
    metrics = report.progress.metrics
    assert metrics["max_games"] == 8 and metrics["subset_seed"] == 5 and metrics["games_used"] == 8
    assert sum(metrics["dataset_selection"]["selected_episodes_by_players"].values()) <= 8
    info = read_info(Path(metrics["intent_artifact_path"]))
    assert info is not None
    assert info.conditions["max_games"] == 8 and info.conditions["games_used"] == 8


def test_module_subset_seed_differs_per_job_and_honors_explicit_seed() -> None:
    from types import SimpleNamespace

    def seed(job_id: str, subset_seed=None) -> int:
        run = SimpleNamespace(config=SimpleNamespace(subset_seed=subset_seed))
        return pipeline_runner._module_subset_seed(run, SimpleNamespace(job_id=job_id))

    assert seed("job-a") != seed("job-b")
    assert seed("job-a") == seed("job-a")
    assert seed("job-a", 42) == 42
    for value, expected in ((None, None), (0, None), (-1, None), (7, 7)):
        run = SimpleNamespace(config=SimpleNamespace(max_games=value))
        assert pipeline_runner._module_max_games(run) == expected
