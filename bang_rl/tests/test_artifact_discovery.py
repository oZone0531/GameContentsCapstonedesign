"""Renaming generated files in the file explorer must not break anything."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bang_rl.artifact_info import (
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI,
    KIND_REPORT,
    new_info,
    read_info,
    sidecar_path,
    write_info,
)
from bang_rl.gui import replay_store
from bang_rl.training.main_ai_artifacts import list_saved_main_ai_models, summarize_main_ai_artifact
from bang_rl.training.module_artifacts import (
    find_report_path,
    list_saved_module_runs,
    summarize_module_run,
)

RUN_A = "11111111-aaaa-4aaa-8aaa-000000000001"
RUN_B = "22222222-bbbb-4bbb-8bbb-000000000002"


def _module(path: Path, kind: str, run_id: str, *, with_info: bool = True, epochs: int = 3) -> Path:
    artifact_type = {"intent": "phase2_intent_inference", "emotion": "phase2_emotion_inference"}[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": artifact_type,
            "artifact_format_version": "1",
            "metadata": {
                "run_metadata": {"job_id": run_id},
                "training": {"epochs_completed": epochs, "total_epochs": epochs, "best_epoch": 1, "metrics": []},
                "architecture": {"role_count": 4},
            },
            "module_metadata": {},
            "state_dict": {"w": torch.zeros(2)},
            "training_mode": False,
        },
        path,
    )
    if with_info:
        info_kind = KIND_INTENT if kind == "intent" else KIND_EMOTION
        write_info(
            path,
            new_info(
                info_kind, id=run_id, created_by="module_training",
                created_at="2026-11-15T10:00:00+09:00", conditions={"total_epochs": epochs}, notes="메모",
            ),
        )
    return path


def test_module_pairs_survive_arbitrary_renames(tmp_path: Path) -> None:
    root = tmp_path / "models"
    intent = _module(root / "intent" / "intent-x.phase2model", "intent", RUN_A)
    emotion = _module(root / "emotion" / "emotion-x.phase2model", "emotion", RUN_A)
    before = list_saved_module_runs(root, saved_root=tmp_path / "none")
    assert [run.job_id for run in before] == [RUN_A] and before[0].intent_path and before[0].emotion_path

    # Rename both files (and their sidecars) to names that follow no rule at all.
    new_intent = intent.rename(intent.with_name("내 의도 모델 최종.phase2model"))
    new_emotion = emotion.rename(emotion.with_name("zzz.phase2model"))
    sidecar_path(Path(root / "intent" / "intent-x.phase2model")).rename(sidecar_path(new_intent))
    sidecar_path(Path(root / "emotion" / "emotion-x.phase2model")).rename(sidecar_path(new_emotion))
    after = list_saved_module_runs(root, saved_root=tmp_path / "none")
    assert [run.job_id for run in after] == [RUN_A]
    assert after[0].intent_path == new_intent and after[0].emotion_path == new_emotion
    assert after[0].notes == "메모"
    # Even moving a file into the other folder or deleting every sidecar changes nothing.
    for sidecar in root.rglob("*.about.md"):
        sidecar.unlink()
    moved = new_emotion.rename(root / "intent" / new_emotion.name)
    again = list_saved_module_runs(root, saved_root=tmp_path / "none")
    assert again[0].emotion_path == moved and again[0].intent_path == new_intent
    # The performance summary still reads both renamed files.
    summary = summarize_module_run(again[0], reports_dir=tmp_path / "reports")
    assert summary["intent"]["kind"] == "intent" and summary["emotion"]["kind"] == "emotion"


def test_legacy_files_without_info_still_pair_by_job_id_in_metadata(tmp_path: Path) -> None:
    root = tmp_path / "models"
    _module(root / "intent" / "intent-old.phase2model", "intent", RUN_B, with_info=False)
    _module(root / "emotion" / "totally-renamed.phase2model", "emotion", RUN_B, with_info=False)
    runs = list_saved_module_runs(root, saved_root=tmp_path / "none")
    assert len(runs) == 1 and runs[0].job_id == RUN_B
    assert runs[0].intent_path is not None and runs[0].emotion_path is not None
    assert runs[0].display_name == RUN_B  # no readable name until migrated


def test_saved_copy_is_listed_but_local_original_wins(tmp_path: Path) -> None:
    root = tmp_path / "models"
    saved = tmp_path / "saved" / "modules"
    _module(root / "intent" / "a.phase2model", "intent", RUN_A)
    _module(saved / "copy" / "intent.phase2model", "intent", RUN_A)
    _module(saved / "copy" / "emotion.phase2model", "emotion", RUN_A)
    _module(saved / "only" / "intent.phase2model", "intent", RUN_B)
    runs = {run.job_id: run for run in list_saved_module_runs(root, saved_root=saved)}
    assert set(runs) == {RUN_A, RUN_B}
    assert runs[RUN_A].intent_path == root / "intent" / "a.phase2model"
    assert runs[RUN_A].emotion_path == saved / "copy" / "emotion.phase2model"
    assert runs[RUN_B].saved is True


def test_report_is_linked_by_id_not_by_file_name(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    legacy = reports / f"{RUN_A}.json"
    legacy.write_text("{}", encoding="utf-8")
    assert find_report_path(RUN_A, reports) == legacy
    renamed = legacy.rename(reports / "내 보고서.json")
    write_info(renamed, new_info(KIND_REPORT, id=RUN_A, created_by="module_training"))
    assert find_report_path(RUN_A, reports) == renamed
    assert find_report_path(RUN_B, reports) is None
    run = list_saved_module_runs(tmp_path / "none", saved_root=tmp_path / "none")
    assert run == ()


def test_main_ai_models_are_found_by_extension_and_id(tmp_path: Path) -> None:
    from bang_rl.tests.test_final_inference_artifact import _save

    path, _ = _save(tmp_path)
    write_info(path, new_info(KIND_MAIN_AI, id=RUN_A, created_by="main_ai_training", notes="좋은 모델"))
    renamed = path.rename(tmp_path / "아무 이름이나.bangmodel")
    models = list_saved_main_ai_models([tmp_path])
    assert [model.path for model in models] == [renamed]
    assert models[0].job_id == RUN_A and models[0].notes == "좋은 모델"
    summary = summarize_main_ai_artifact(renamed)
    assert "error" not in summary and summary["job_id"] == RUN_A and summary["notes"] == "좋은 모델"


def test_replay_with_info_loads_and_lists_after_rename(tmp_path: Path) -> None:
    from bang_rl.artifact_builders import replay_info

    record = replay_store.build_replay_record(
        mode="human_dummy", mode_label="Human vs Dummy", ai_versions=None, winning_team="OUTLAW",
        winners=[1], human_player=0, seed=7, observations=[{"event_sequence": 0}], terminated=True,
    )
    info = replay_info(record, id="abc123def456", created_by="gui_game")
    path = replay_store.save_replay(record, tmp_path, info=info)
    assert next(iter(json.loads(path.read_text(encoding="utf-8")))) == "info"
    assert path.name == f"{info.name}.json" and info.name.endswith("_abc123")
    renamed = replay_store.rename_replay(path, "내 리플레이")
    assert replay_store.list_replays(tmp_path) == [renamed]
    loaded = replay_store.load_replay(renamed)
    assert loaded["seed"] == 7 and loaded["info"]["id"] == "abc123def456"
    assert replay_store.replay_summary(loaded)["mode"] == "human_dummy"
    assert read_info(renamed).id == "abc123def456"
    # The record without info stays byte compatible.
    assert replay_store.load_replay(replay_store.save_replay(record, tmp_path, filename="plain.json")) == record


def test_dataset_folder_and_cache_hashes_survive_info_and_rename(tmp_path: Path) -> None:
    from bang_rl.data.game_collection import FORMULA_LABEL_SCHEMA, collect_games, load_collection_config
    from bang_rl.data.parquet_writer import ParquetCanonicalShardWriter
    from bang_rl.data.splitter import EpisodeSplitter
    from bang_rl.data.tensor_cache import NPZTensorCacheLoader, NPZTensorCacheWriter
    from bang_rl.training.pipeline_runner import _read_canonical_dataset

    config = load_collection_config()
    canonical = tmp_path / "canonical"
    summary = collect_games(config, 12, canonical / "first", base_seed=5)
    info = read_info(canonical / "first")
    assert info is not None and info.kind == "dataset" and info.conditions["games"] == 12
    assert info.summary["games"] == 12 and (canonical / "first" / "ABOUT.md").is_file()
    assert json.loads((canonical / "first" / "collection_summary.json").read_text(encoding="utf-8"))["info"]["id"] == info.id
    assert summary["games"] == 12

    renamed = (canonical / "first").rename(canonical / "더미끼리 열두판")
    episodes, shards, duplicates, selection = _read_canonical_dataset(canonical)
    assert len(episodes) == 12 and duplicates == 0 and len(shards) >= 1
    assert read_info(renamed).id == info.id
    assert selection["selected_episode_count"] == 12

    # Hash verified files are never touched by the info block.
    split = EpisodeSplitter(counts=(8, 2, 2), seed=0, manifest_path=tmp_path / "cache" / "split_manifest.json").split(
        ParquetCanonicalShardWriter.read_dataset(renamed)
    )
    writer = NPZTensorCacheWriter(
        tmp_path / "cache", emotion_schema_version=FORMULA_LABEL_SCHEMA, emotion_dimension=9,
        split_manifest=split.manifest,
    )
    artifacts = writer.write_all(split)
    before = {path.name: path.read_bytes() for path in (tmp_path / "cache").iterdir() if path.is_file()}
    cache_info = new_info("tensor_cache", id="cache-1", created_by="module_training", conditions={"x": 1})
    write_info(tmp_path / "cache", cache_info)
    after = {path.name: path.read_bytes() for path in (tmp_path / "cache").iterdir() if path.is_file() and path.name != "ABOUT.md"}
    assert after == before
    assert (tmp_path / "cache" / "ABOUT.md").is_file()
    for name, artifact in artifacts.items():
        loader = NPZTensorCacheLoader(artifact.tensor_path, manifest_path=artifact.manifest_path)
        loader.validate()
    again = writer.write_all(split)
    assert {name: item.manifest["cache_content_hash"] for name, item in again.items()} == {
        name: item.manifest["cache_content_hash"] for name, item in artifacts.items()
    }


@pytest.mark.parametrize("name", ["a", "한글 이름", "x" * 80])
def test_report_naming_does_not_break_json_loading(tmp_path: Path, name: str) -> None:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"z": 1, "a": {"y": 2, "b": 3}}), encoding="utf-8")
    write_info(path, new_info(KIND_REPORT, id=RUN_A, created_by="x"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["a"] == {"y": 2, "b": 3} and data["info"]["id"] == RUN_A
