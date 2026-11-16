"""Artifact metadata (info block), naming rule and per-format round trips."""
from __future__ import annotations

import json
import re
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from bang_rl.artifact_info import (
    ArtifactInfo,
    KIND_DATASET,
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI,
    KIND_REPLAY,
    KIND_REPORT,
    default_artifact_name,
    dump_json_with_info,
    ensure_sidecar,
    find_json_by_id,
    new_info,
    read_info,
    sanitize_user_name,
    sidecar_path,
    source_ref,
    update_notes,
    write_info,
)
from bang_rl.training import load_final_inference_artifact
from bang_rl.training.final_inference_artifact import read_final_artifact_metadata

JOB = "3f9a1c77-0000-4000-8000-000000000001"
JOB2 = "7b2d0499-0000-4000-8000-000000000002"


def _dataset_info(**overrides) -> ArtifactInfo:
    values = dict(
        id=JOB,
        created_by="data_collection",
        created_at="2026-11-15T14:32:10+09:00",
        conditions={"players": "dummy5", "games": 50000},
        summary={"games": 50000},
        kind_label="학습 데이터(더미 5인)",
    )
    values.update(overrides)
    return new_info(KIND_DATASET, **values)


# ---------------------------------------------------------------- naming rule


def test_dataset_name_follows_rule() -> None:
    assert _dataset_info().name == "20261115-1432_dataset_dummy5_50000g_3f9a1c"


def test_module_and_main_ai_names_use_sources() -> None:
    dataset = source_ref(KIND_DATASET, JOB, "x")
    intent = new_info(
        KIND_INTENT, id=JOB2, created_by="module_training", created_at="2026-11-15T15:30:00+09:00",
        conditions={"total_epochs": 12}, sources=[dataset],
    )
    emotion = new_info(
        KIND_EMOTION, id=JOB2, created_by="module_training", created_at="2026-11-15T15:30:00+09:00",
        conditions={"total_epochs": 12}, sources=[dataset],
    )
    assert intent.name == "20261115-1530_intent_e12_from-3f9a1c_7b2d04"
    assert emotion.name == "20261115-1530_emotion_e12_from-3f9a1c_7b2d04"
    main = new_info(
        KIND_MAIN_AI, id="c81e5500-0000-4000-8000-000000000003", created_by="main_ai_training",
        created_at="2026-11-16T09:10:00+09:00", conditions={"training_games": 80000},
        sources=[source_ref(KIND_INTENT, JOB2, intent.name)],
    )
    assert main.name == "20261116-0910_mainai_80000g_mod-7b2d04_c81e55"
    report = new_info(KIND_REPORT, id=JOB2, created_by="module_training", created_at="2026-11-15T15:30:00+09:00")
    assert report.name == "20261115-1530_report_module-training_7b2d04"
    replay = new_info(
        KIND_REPLAY, id="a1b2c3d4", created_at="2026-11-16T10:12:00+09:00",
        created_by="gui_game", conditions={"players": "main1-dummy4"},
    )
    assert replay.name == "20261116-1012_replay_main1-dummy4_a1b2c3"


def test_names_are_ascii_sortable_and_short() -> None:
    names = []
    for index in range(5):
        info = _dataset_info(
            id=f"{index:02d}abcdef",
            created_at=f"2026-11-1{index}T10:00:00+09:00",
            conditions={"players": "한글 players/with:odd*chars " * 8, "games": 10 ** 12},
        )
        assert info.name.isascii()
        assert len(info.name) <= 90
        assert re.fullmatch(r"[a-z0-9._+-]+", info.name)
        assert info.name.endswith(f"_{index:02d}abcd")
        names.append(info.name)
    assert names == sorted(names)


def test_sanitize_user_name_keeps_korean_and_blocks_windows_forbidden() -> None:
    assert sanitize_user_name("더미끼리 5만판 11월 15일") == "더미끼리 5만판 11월 15일"
    assert sanitize_user_name('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"
    assert sanitize_user_name("name. ") == "name"
    assert sanitize_user_name("CON") == "_CON"
    assert len(sanitize_user_name("가" * 500)) <= 120
    with pytest.raises(ValueError):
        sanitize_user_name("  ... ")


def test_default_name_helper_matches_info_name() -> None:
    info = _dataset_info()
    assert default_artifact_name(info) == info.name


# --------------------------------------------------------------- round trips


def test_info_dict_round_trip_and_order() -> None:
    info = _dataset_info(notes="메모")
    data = info.to_dict()
    assert list(data)[:6] == ["kind", "kind_label", "name", "created_at", "created_by", "notes"]
    assert ArtifactInfo.from_dict(json.loads(json.dumps(data, ensure_ascii=False))).to_dict() == data


def test_json_file_info_is_first_key_and_readable(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"zeta": 1, "alpha": 2}), encoding="utf-8")
    info = new_info(KIND_REPORT, id=JOB, created_by="module_training")
    write_info(path, info)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert next(iter(data)) == "info"
    assert data["zeta"] == 1 and data["alpha"] == 2
    assert read_info(path).id == JOB
    update_notes(path, "새 비고")
    assert read_info(path).notes == "새 비고"
    assert json.loads(path.read_text(encoding="utf-8"))["info"]["notes"] == "새 비고"
    assert not sidecar_path(path).exists()


def test_dump_json_with_info_sorts_rest_deterministically() -> None:
    info = new_info(KIND_REPORT, id=JOB, created_by="x", created_at="2026-01-01T00:00:00+09:00")
    text = dump_json_with_info({"b": 1, "a": 2}, info, sort_keys=True)
    assert list(json.loads(text)) == ["info", "a", "b"]
    assert text == dump_json_with_info({"a": 2, "b": 1}, info, sort_keys=True)


def test_long_notes_still_found_in_json(tmp_path: Path) -> None:
    path = tmp_path / "long.json"
    path.write_text("{}", encoding="utf-8")
    write_info(path, new_info(KIND_REPORT, id=JOB, created_by="x", notes="가" * 100_000))
    assert read_info(path).notes == "가" * 100_000


def test_folder_info_uses_about_md(tmp_path: Path) -> None:
    folder = tmp_path / "ds"
    folder.mkdir()
    (folder / "shard-0000.parquet").write_bytes(b"x")
    info = _dataset_info(notes="처음 메모")
    write_info(folder, info)
    about = folder / "ABOUT.md"
    text = about.read_text(encoding="utf-8")
    assert "비고" in text and "처음 메모" in text and "```json" in text
    assert read_info(folder).id == JOB
    update_notes(folder, "고친 메모")
    assert read_info(folder).notes == "고친 메모"
    # The user may also edit the visible note section by hand.
    about.write_text(about.read_text(encoding="utf-8").replace("고친 메모", "손으로 쓴 메모"), encoding="utf-8")
    assert read_info(folder).notes == "손으로 쓴 메모"
    assert (folder / "shard-0000.parquet").read_bytes() == b"x"


def _module_file(path: Path, run_id: str) -> None:
    torch.save(
        {
            "artifact_type": "phase2_intent_inference",
            "artifact_format_version": "1",
            "metadata": {"run_metadata": {"job_id": run_id}},
            "module_metadata": {},
            "state_dict": {"w": torch.arange(4.0)},
            "training_mode": False,
        },
        path,
    )


def test_phase2model_info_is_embedded_and_payload_untouched(tmp_path: Path) -> None:
    path = tmp_path / "anything-at-all.phase2model"
    _module_file(path, JOB2)
    info = new_info(KIND_INTENT, id=JOB2, created_by="module_training", conditions={"total_epochs": 3})
    write_info(path, info)
    payload = torch.load(path, weights_only=False)
    assert next(iter(payload)) == "info"
    assert payload["state_dict"]["w"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert payload["artifact_type"] == "phase2_intent_inference"
    assert read_info(path).id == JOB2
    side = sidecar_path(path)
    assert side.name == "anything-at-all.phase2model.about.md"
    assert "의도 추론 모듈" in side.read_text(encoding="utf-8")


def test_sidecar_regenerates_from_embedded_info(tmp_path: Path) -> None:
    path = tmp_path / "m.phase2model"
    _module_file(path, JOB2)
    write_info(path, new_info(KIND_INTENT, id=JOB2, created_by="module_training", notes="keep"))
    sidecar_path(path).unlink()
    assert read_info(path).notes == "keep"
    assert ensure_sidecar(path) == sidecar_path(path)
    assert "keep" in sidecar_path(path).read_text(encoding="utf-8")


def test_update_notes_updates_embedded_and_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "m.phase2model"
    _module_file(path, JOB2)
    write_info(path, new_info(KIND_INTENT, id=JOB2, created_by="module_training"))
    update_notes(path, "첫 메모")
    assert torch.load(path, weights_only=False)["info"]["notes"] == "첫 메모"
    assert "첫 메모" in sidecar_path(path).read_text(encoding="utf-8")
    # A file copied without its sidecar still reports the embedded note.
    lonely = tmp_path / "copy.phase2model"
    lonely.write_bytes(path.read_bytes())
    assert read_info(lonely).notes == "첫 메모"


def test_bangmodel_info_keeps_checksum_and_loads(tmp_path: Path) -> None:
    from bang_rl.tests.test_final_inference_artifact import _save

    path, _ = _save(tmp_path)
    before = read_final_artifact_metadata(path)[1]
    info = new_info(KIND_MAIN_AI, id=JOB, created_by="main_ai_training", conditions={"training_games": 10})
    write_info(path, info)
    manifest, metadata = read_final_artifact_metadata(path)  # verifies metadata_sha256
    assert next(iter(metadata)) == "info"
    assert {key: value for key, value in metadata.items() if key != "info"} == before
    with ZipFile(path) as bundle:
        assert json.loads(bundle.read("metadata.json"))["info"]["id"] == JOB
        assert next(iter(json.loads(bundle.read("metadata.json")))) == "info"
    assert load_final_inference_artifact(path).metadata["info"]["id"] == JOB
    assert read_info(path).id == JOB
    update_notes(path, "비고 수정")
    read_final_artifact_metadata(path)
    assert read_info(path).notes == "비고 수정"
    assert load_final_inference_artifact(path) is not None


def test_save_final_artifact_with_info_argument(tmp_path: Path, monkeypatch) -> None:
    from bang_rl.tests import test_final_inference_artifact as helper
    from bang_rl.training import save_final_inference_artifact

    info = new_info(KIND_MAIN_AI, id=JOB, created_by="main_ai_training")
    monkeypatch.setattr(
        helper,
        "save_final_inference_artifact",
        lambda *args, **kwargs: save_final_inference_artifact(*args, info=info, **kwargs),
    )
    path, _ = helper._save(tmp_path)
    _, metadata = read_final_artifact_metadata(path)
    assert next(iter(metadata)) == "info"
    assert read_info(path).id == JOB
    assert sidecar_path(path).is_file()
    load_final_inference_artifact(path)


def test_find_json_by_id_ignores_file_name(tmp_path: Path) -> None:
    report = tmp_path / "completely renamed 보고서.json"
    report.write_text("{}", encoding="utf-8")
    write_info(report, new_info(KIND_REPORT, id=JOB, created_by="x"))
    (tmp_path / "other.json").write_text('{"a": 1}', encoding="utf-8")
    assert find_json_by_id([tmp_path], JOB) == report
    assert find_json_by_id([tmp_path], "missing") is None


def test_notes_update_is_seen_even_when_file_times_do_not_change(tmp_path, monkeypatch) -> None:
    """Coarse file clocks can give two writes the same mtime; reads must not go stale."""

    import os as _os

    from bang_rl import artifact_info as module

    folder = tmp_path / "dataset"
    folder.mkdir()
    module.write_info(folder, module.new_info(module.KIND_DATASET, id="frozen-clock", created_by="test"))
    assert module.read_info(folder).notes == ""
    stamp = (folder / module.FOLDER_SIDECAR).stat().st_mtime_ns
    real_replace = _os.replace

    def replace_keeping_mtime(source, target):
        real_replace(source, target)
        if str(target).endswith(module.FOLDER_SIDECAR):
            _os.utime(target, ns=(stamp, stamp))

    monkeypatch.setattr(module.os, "replace", replace_keeping_mtime)
    module.update_notes(folder, "first")
    assert module.read_info(folder).notes == "first"
    module.update_notes(folder, "abcde")  # same length as before, same mtime
    assert module.read_info(folder).notes == "abcde"
