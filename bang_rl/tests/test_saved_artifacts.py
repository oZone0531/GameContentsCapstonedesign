"""The 'Save outputs' logic: plans, size warnings, name rules and the copy itself."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bang_rl.artifact_info import (
    KIND_DATASET,
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI,
    KIND_REPORT,
    new_info,
    read_info,
    write_info,
)
from bang_rl.gui.dialogs.save_outputs import SaveOutputsController
from bang_rl.saved_artifacts import (
    GITHUB_LIMIT_BYTES,
    GITHUB_WARN_BYTES,
    LEVEL_OK,
    LEVEL_STRONG,
    LEVEL_WARN,
    SaveNameExistsError,
    execute_save,
    human_size,
    plan_dataset,
    plan_main_ai,
    plan_modules,
    plans_for_training_result,
    size_level,
)
from bang_rl.training.module_artifacts import list_saved_module_runs
from bang_rl.tests.test_artifact_discovery import RUN_A, _module

MIB = 1024 * 1024


def test_size_thresholds_are_50_and_100_mib() -> None:
    assert GITHUB_WARN_BYTES == 50 * MIB and GITHUB_LIMIT_BYTES == 100 * MIB
    assert size_level([]) == LEVEL_OK
    assert size_level([50 * MIB]) == LEVEL_OK
    assert size_level([50 * MIB + 1]) == LEVEL_WARN
    assert size_level([100 * MIB]) == LEVEL_WARN
    assert size_level([100 * MIB + 1]) == LEVEL_STRONG
    assert size_level([1, 60 * MIB, 3]) == LEVEL_WARN  # the largest single file decides
    assert human_size(1536) == "1.5 KB"


def _dataset(tmp_path: Path) -> Path:
    folder = tmp_path / "gen" / "20261115-1432_dataset_dummy5_50000g_3f9a1c"
    folder.mkdir(parents=True)
    (folder / "shard-0000.parquet").write_bytes(b"p" * 100)
    (folder / "collection_summary.json").write_text("{}", encoding="utf-8")
    write_info(
        folder,
        new_info(
            KIND_DATASET, id="3f9a1c77-0000-4000-8000-000000000001", created_by="data_collection",
            created_at="2026-11-15T14:32:00+09:00", conditions={"players": "dummy5", "games": 50000}, notes="원본 메모",
        ),
    )
    write_info(folder / "collection_summary.json", read_info(folder))
    return folder


def test_dataset_save_copies_tree_with_chosen_name_and_notes(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    plan = plan_dataset(source, saved_root=tmp_path / "saved")
    assert plan is not None and plan.default_name == source.name  # default follows the naming rule
    assert plan.notes == "원본 메모" and plan.level == LEVEL_OK
    result = execute_save(plan, "더미끼리 5만판 11월 15일", "복사본 메모")
    target = tmp_path / "saved" / "datasets" / "더미끼리 5만판 11월 15일"
    assert result.paths == (target,)
    assert (target / "shard-0000.parquet").read_bytes() == b"p" * 100
    copy_info = read_info(target)
    assert copy_info.name == "더미끼리 5만판 11월 15일" and copy_info.notes == "복사본 메모"
    assert copy_info.id == read_info(source).id
    assert read_info(source).notes == "원본 메모" and read_info(source).name == source.name  # source untouched
    assert (source / "shard-0000.parquet").exists()  # copy, not move
    with pytest.raises(SaveNameExistsError):
        execute_save(plan, "더미끼리 5만판 11월 15일")


def test_module_pair_and_report_are_saved_together_and_stay_paired(tmp_path: Path) -> None:
    models = tmp_path / "models"
    intent = _module(models / "intent" / "i.phase2model", "intent", RUN_A, epochs=12)
    emotion = _module(models / "emotion" / "e.phase2model", "emotion", RUN_A, epochs=12)
    reports = tmp_path / "reports"
    reports.mkdir()
    report = reports / "r.json"
    report.write_text('{"job_id": "x"}', encoding="utf-8")
    write_info(report, new_info(KIND_REPORT, id=RUN_A, created_by="module_training"))
    plan = plan_modules(intent, emotion, reports_dirs=[reports], saved_root=tmp_path / "saved")
    assert plan is not None
    assert [item.kind for item in plan.items] == [KIND_INTENT, KIND_EMOTION, KIND_REPORT]
    assert "_modules_" in plan.default_name and plan.default_name.endswith(RUN_A[:6])
    controller = SaveOutputsController(plan)
    assert controller.name == plan.default_name
    assert {row.name for row in controller.file_rows()} >= {"i.phase2model", "e.phase2model", "r.json"}
    result = controller.save("내 모듈", "비고 내용")
    assert result is not None
    folder = tmp_path / "saved" / "modules" / "내 모듈"
    assert (folder / "intent.phase2model").is_file() and (folder / "emotion.phase2model").is_file()
    assert (tmp_path / "saved" / "reports" / "내 모듈.json").is_file()
    assert read_info(folder / "intent.phase2model").id == read_info(folder / "emotion.phase2model").id == RUN_A
    runs = list_saved_module_runs(tmp_path / "no-models", saved_root=tmp_path / "saved" / "modules")
    assert len(runs) == 1 and runs[0].job_id == RUN_A and runs[0].saved
    assert runs[0].intent_path and runs[0].emotion_path and runs[0].notes == "비고 내용"
    # Same name again is refused, a different one works.
    again = SaveOutputsController(plan)
    assert again.save("내 모듈", "") is None and "내 모듈" in (again.last_error or "")
    assert again.save("다른 이름", "") is not None


def test_main_ai_save_keeps_model_loadable_and_checksum_valid(tmp_path: Path) -> None:
    from bang_rl.tests.test_final_inference_artifact import _save
    from bang_rl.training import load_final_inference_artifact
    from bang_rl.training.final_inference_artifact import read_final_artifact_metadata

    model, _ = _save(tmp_path)
    write_info(model, new_info(KIND_MAIN_AI, id=RUN_A, created_by="main_ai_training", conditions={"training_games": 80000}))
    plan = plan_main_ai(model, saved_root=tmp_path / "saved")
    assert plan is not None and "_mainai_80000g_" in plan.default_name
    result = execute_save(plan, "메인 AI 8만판")
    saved = tmp_path / "saved" / "main_ai" / "메인 AI 8만판.bangmodel"
    assert result.paths == (saved,) and model.exists()
    read_final_artifact_metadata(saved)  # checksum still valid after the info rewrite
    load_final_inference_artifact(saved)
    assert read_info(saved).name == "메인 AI 8만판" and read_info(model).name != "메인 AI 8만판"


def test_size_warning_texts_and_confirmation(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    plan = plan_dataset(source, saved_root=tmp_path / "saved")
    assert plan is not None
    controller = SaveOutputsController(plan)
    assert controller.warning_text() == "" and not controller.needs_confirmation

    big = {"shard-0000.parquet": 60 * MIB}
    plan.size_of = lambda path: big.get(path.name, path.stat().st_size)
    warn = SaveOutputsController(plan)
    assert plan.level == LEVEL_WARN and "50" in warn.warning_text() and "shard-0000.parquet" in warn.warning_text()
    assert not warn.needs_confirmation

    big["shard-0000.parquet"] = 150 * MIB
    strong = SaveOutputsController(plan)
    assert plan.level == LEVEL_STRONG and strong.needs_confirmation
    assert "100" in strong.warning_text() and "GitHub" in strong.warning_text()
    asked: list[str] = []
    # Declined: nothing is written. Confirmed: it is still saved (warning only).
    assert strong.save("크다", "", confirm=lambda text: asked.append(text) or False) is None
    assert not (tmp_path / "saved" / "datasets" / "크다").exists() and len(asked) == 1
    assert strong.save("크다", "", confirm=lambda text: True) is not None
    assert (tmp_path / "saved" / "datasets" / "크다" / "shard-0000.parquet").is_file()
    # Without any confirm callback a strong warning blocks the save.
    assert strong.save("또 크다", "") is None


def test_names_with_forbidden_characters_are_cleaned_and_empty_rejected(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    plan = plan_dataset(source, saved_root=tmp_path / "saved")
    controller = SaveOutputsController(plan)
    assert controller.save("   ", "") is None and controller.last_error
    result = controller.save('a:b/c?', "")
    assert result is not None and result.name == "a_b_c_"
    assert (tmp_path / "saved" / "datasets" / "a_b_c_").is_dir()


def test_failed_copy_leaves_no_partial_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _dataset(tmp_path)
    plan = plan_dataset(source, saved_root=tmp_path / "saved")
    import bang_rl.saved_artifacts as module

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module, "write_info", boom)
    controller = SaveOutputsController(plan)
    assert controller.save("x", "") is None and "disk full" in controller.last_error
    assert list((tmp_path / "saved" / "datasets").iterdir()) == []


def test_plans_for_training_results(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    plans = plans_for_training_result(
        "data_collection", output_path=str(source), metrics={"dataset_path": str(source)}, saved_root=tmp_path / "s"
    )
    assert len(plans) == 1 and plans[0].category == "datasets"
    assert plans_for_training_result("data_collection", output_path=str(tmp_path / "missing")) == []
    assert plans_for_training_result("module_training", metrics={}) == []
    assert plans_for_training_result("main_ai_training", artifact_path=str(tmp_path / "none.bangmodel")) == []
    intent = _module(tmp_path / "m" / "i.phase2model", "intent", RUN_A)
    only_intent = plans_for_training_result(
        "module_training", metrics={"intent_artifact_path": str(intent), "emotion_artifact_path": None},
        saved_root=tmp_path / "s",
    )
    assert len(only_intent) == 1 and [item.kind for item in only_intent[0].items] == [KIND_INTENT]
