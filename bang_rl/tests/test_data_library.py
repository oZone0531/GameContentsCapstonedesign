"""Headless tests of the data library model and its Qt-free view logic."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bang_rl import data_library as lib
from bang_rl.artifact_builders import dataset_info, replay_info
from bang_rl.artifact_info import (
    KIND_DATASET,
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI,
    KIND_REPORT,
    new_info,
    read_info,
    sidecar_path,
    source_ref,
    write_info,
)
from bang_rl.data.game_collection import FORMULA_LABEL_SCHEMA
from bang_rl.gui import library_view as view
from bang_rl.gui import replay_store
from bang_rl.gui.i18n import current_language, set_language, tr
from bang_rl.gui.training import TrainingPreset
from bang_rl.tests.test_inference_refinement import _save_base_model

DATASET_ID = "d0000000-aaaa-4aaa-8aaa-000000000001"
RUN_GOOD = "a1111111-aaaa-4aaa-8aaa-000000000001"
RUN_BAD = "b2222222-bbbb-4bbb-8bbb-000000000002"
RUN_OLD = "c3333333-cccc-4ccc-8ccc-000000000003"
MODEL_ID = "e4444444-eeee-4eee-8eee-000000000004"


# --- fake stored data ---------------------------------------------------------------


def _dataset(folder: Path, *, id: str = DATASET_ID, games: int = 40, players: str = "dummy5", notes: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "shard-00000.parquet").write_bytes(b"x" * 100)
    info = dataset_info(
        id=id, games=games, base_seed=1, created_by="data_collection", players=players,
        created_at="2026-11-10T10:00:00+09:00", summary={"games": games, "win_rate": 0.4, "truncated_games": 1},
        notes=notes,
    )
    info.summary["rows"] = games * 10
    write_info(folder, info)
    return folder


def _module(path: Path, kind: str, run_id: str, *, compatible: bool = True, with_info: bool = True,
            sources: list | None = None, created_at: str = "2026-11-11T10:00:00+09:00", notes: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "intent":
        metadata = {"run_metadata": {"job_id": run_id}, "architecture": {"role_count": 4 if compatible else 5},
                    "training": {"epochs_completed": 3, "total_epochs": 3, "best_epoch": 2, "metrics": []}}
    else:
        metadata = {"run_metadata": {"job_id": run_id}, "emotion_schema": {"version": FORMULA_LABEL_SCHEMA, "count": 9},
                    "training": {"epochs_completed": 3, "total_epochs": 3, "best_epoch": 2, "metrics": []}}
    torch.save(
        {"artifact_type": f"phase2_{kind}_inference", "artifact_format_version": "1", "metadata": metadata,
         "module_metadata": {}, "state_dict": {"w": torch.zeros(2)}, "training_mode": False},
        path,
    )
    if with_info:
        info_kind = KIND_INTENT if kind == "intent" else KIND_EMOTION
        summary = {"test_accuracy": 0.81} if kind == "intent" else {"test_mse": 0.0123, "validation_accuracy": 0.7}
        write_info(
            path,
            new_info(info_kind, id=run_id, created_by="module_training", created_at=created_at,
                     conditions={"total_epochs": 3}, summary=summary, notes=notes,
                     sources=sources if sources is not None else [source_ref(KIND_DATASET, DATASET_ID, "ds")]),
        )
    return path


def _pair(root: Path, run_id: str, **kwargs) -> tuple[Path, Path]:
    return (
        _module(root / "intent" / f"intent-{run_id[:4]}.phase2model", "intent", run_id, **kwargs),
        _module(root / "emotion" / f"emotion-{run_id[:4]}.phase2model", "emotion", run_id, **kwargs),
    )


def _report(path: Path, run_id: str, *, created_at: str = "2026-11-11T10:05:00+09:00") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "finished"}), encoding="utf-8")
    write_info(
        path,
        new_info(KIND_REPORT, id=run_id, created_by="module_training", created_at=created_at,
                 conditions={"workflow": "module_training"}, summary={"status": "finished", "games_completed": 40, "win_rate": 0.4}),
    )
    return path


def _replay(directory: Path, *, id: str = "r" * 12, mode: str = "human_dummy") -> Path:
    record = replay_store.build_replay_record(
        mode=mode, mode_label="Human vs Dummy", ai_versions=None, winning_team="OUTLAW", winners=[1],
        human_player=0, seed=7, observations=[{"event_sequence": 0}], terminated=True,
    )
    info = replay_info(record, id=id, created_at="2026-11-12T09:00:00+09:00")
    return replay_store.save_replay(record, directory, info=info)


@pytest.fixture(scope="module")
def bangmodel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _save_base_model(tmp_path_factory.mktemp("lib-model") / "base.bangmodel")


def _main_ai(path: Path, source: Path, *, id: str = MODEL_ID, created_at: str = "2026-11-12T08:00:00+09:00",
             refined: bool = False, notes: str = "좋은 모델") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source.read_bytes())
    info = new_info(
        KIND_MAIN_AI, id=id, created_by="inference_refinement" if refined else "main_ai_training",
        created_at=created_at, notes=notes, conditions={"training_games": 120},
        summary={"win_rate": 0.55, "normalized_score": 1.1, "refined": refined, "hidden_role_accuracy": 0.62 if refined else None},
        sources=[source_ref(KIND_INTENT, RUN_GOOD, "pair"), source_ref(KIND_EMOTION, RUN_GOOD, "pair")],
    )
    write_info(path, info)
    return path


@pytest.fixture
def stored(tmp_path: Path, bangmodel: Path) -> SimpleNamespace:
    loc = lib.LibraryLocations.under(tmp_path)
    dataset = _dataset(loc.canonical_root / "ds-one", notes="첫 번째 데이터\n둘째 줄")
    good = _pair(loc.model_root, RUN_GOOD)
    bad = _pair(loc.model_root, RUN_BAD, compatible=False)
    report = _report(loc.report_dirs[0] / "report-good.json", RUN_GOOD)
    replay = _replay(loc.replay_dirs[0])
    model = _main_ai(loc.main_ai_dirs[0] / "model.bangmodel", bangmodel)
    return SimpleNamespace(loc=loc, root=tmp_path, dataset=dataset, good=good, bad=bad, report=report,
                           replay=replay, model=model)


def _by_kind(library: lib.Library, kind: str) -> list[lib.LibraryEntry]:
    return [entry for entry in library.entries if entry.kind == kind]


# --- aggregation -------------------------------------------------------------------------


def test_library_aggregates_every_kind_with_facts(stored: SimpleNamespace) -> None:
    library = lib.build_library(stored.loc)
    assert library.counts() == {"dataset": 1, "module_pair": 2, "main_ai": 1, "report": 1, "replay": 1}

    dataset = _by_kind(library, "dataset")[0]
    assert dataset.id == DATASET_ID and dataset.location == lib.LOCATION_LOCAL and dataset.path == stored.dataset
    assert ("players", "dummy5") in dataset.condition_facts and ("games", "40") in dataset.condition_facts
    assert ("rows", "400") in dataset.summary_facts and ("win_rate", "0.40") in dataset.summary_facts
    assert dataset.first_note_line == "첫 번째 데이터"

    pairs = {entry.id: entry for entry in _by_kind(library, "module_pair")}
    good = pairs[RUN_GOOD]
    assert (good.intent_path, good.emotion_path) == stored.good
    assert ("epochs", "3") in good.condition_facts and ("datasets", "1") in good.condition_facts
    assert ("intent_accuracy", "0.810") in good.summary_facts and ("emotion_mse", "0.0123") in good.summary_facts
    assert ("main_ai_ready", "ok") in good.summary_facts and good.compatibility() == (True, None)
    assert pairs[RUN_BAD].compatibility() == (False, "intent_roles")
    assert dict(pairs[RUN_BAD].summary_facts)["main_ai_ready"] == "intent_roles"

    model = _by_kind(library, "main_ai")[0]
    assert model.id == MODEL_ID and ("games", "120") in model.condition_facts
    assert ("win_rate", "0.550") in model.summary_facts and ("score", "1.100") in model.summary_facts
    assert model.compatibility() == (True, None) and "refined" not in model.tags

    report = _by_kind(library, "report")[0]
    assert ("workflow", "module_training") in report.condition_facts and ("status", "finished") in report.summary_facts
    replay = _by_kind(library, "replay")[0]
    assert ("players", "human1-dummy4") in replay.condition_facts and ("winner", "OUTLAW") in replay.summary_facts


def test_entries_are_sorted_newest_first(stored: SimpleNamespace) -> None:
    library = lib.build_library(stored.loc)
    epochs = [entry.created_epoch for entry in library.entries]
    assert epochs == sorted(epochs, reverse=True)
    assert library.entries[0].kind == "replay"  # 2026-11-12 09:00 is the newest
    assert library.entries[-1].kind == "dataset"  # 2026-11-10 is the oldest


def test_sizes_are_lazy_and_include_sidecars(stored: SimpleNamespace) -> None:
    library = lib.build_library(stored.loc)
    dataset = _by_kind(library, "dataset")[0]
    assert dataset._size is None
    expected = sum(path.stat().st_size for path in stored.dataset.rglob("*") if path.is_file())
    assert dataset.size_bytes() == expected and dataset._size == expected
    pair = next(entry for entry in _by_kind(library, "module_pair") if entry.id == RUN_GOOD)
    files = [path.stat().st_size for path in stored.good] + [sidecar_path(path).stat().st_size for path in stored.good]
    assert pair.size_bytes() == sum(files)


def test_empty_locations_give_an_empty_library(tmp_path: Path) -> None:
    library = lib.build_library(lib.LibraryLocations.under(tmp_path))
    assert library.entries == [] and all(count == 0 for count in library.counts().values())


# --- filters ------------------------------------------------------------------------------------


def test_filters_by_kind_location_and_text(stored: SimpleNamespace) -> None:
    saved_ds = _dataset(stored.loc.saved_datasets / "kept", id="d9999999-saved", games=5, notes="보관본")
    assert saved_ds.is_dir()
    library = lib.build_library(stored.loc)
    assert len(library.filtered(kinds="dataset")) == 2
    assert {entry.location for entry in library.filtered(kinds="dataset")} == {"local", "saved"}
    assert [entry.id for entry in library.filtered(kinds=("dataset",), location="saved")] == ["d9999999-saved"]
    assert len(library.filtered(kinds=("module_pair", "main_ai"))) == 3
    assert [entry.kind for entry in library.filtered(text="좋은 모델")] == ["main_ai"]
    assert [entry.id for entry in library.filtered(text="DUMMY5 40games")] == []  # every word must match
    assert [entry.id for entry in library.filtered(text="dummy5 보관본")] == ["d9999999-saved"]
    assert library.filtered(text="nothing-matches-this") == []
    assert len(lib.list_library_entries(stored.loc, kinds="replay")) == 1
    assert len(lib.filter_entries(library.entries, kinds=None, location=None, text="")) == len(library.entries)


def test_saved_copy_is_flagged_and_local_original_wins(stored: SimpleNamespace, bangmodel: Path) -> None:
    # The same dataset kept in saved/ is not listed twice; a different one is flagged "saved".
    _dataset(stored.loc.saved_datasets / "copy", id=DATASET_ID)
    _pair(stored.loc.saved_modules / "copy", RUN_GOOD)
    only_saved = _pair(stored.loc.saved_modules / "other", RUN_OLD)
    _main_ai(stored.loc.saved_main_ai / "kept.bangmodel", bangmodel, id="f5555555-saved")
    _report(stored.loc.saved_reports / "kept.json", RUN_GOOD)
    _report(stored.loc.saved_reports / "only.json", "g6666666-saved-report")
    library = lib.build_library(stored.loc)
    assert library.counts() == {"dataset": 1, "module_pair": 3, "main_ai": 2, "report": 2, "replay": 1}
    saved = {entry.id: entry for entry in library.entries if entry.saved}
    assert set(saved) == {RUN_OLD, "f5555555-saved", "g6666666-saved-report"}
    assert saved[RUN_OLD].intent_path == only_saved[0]
    assert next(e for e in library.entries if e.id == DATASET_ID).location == "local"


def test_renamed_files_are_still_listed(stored: SimpleNamespace) -> None:
    renamed_dataset = stored.dataset.rename(stored.dataset.with_name("내 더미 데이터"))
    intent, emotion = stored.good
    new_intent = intent.rename(intent.with_name("zzz-의도.phase2model"))
    new_emotion = emotion.rename(emotion.with_name("qqq.phase2model"))
    for old, new in ((intent, new_intent), (emotion, new_emotion)):
        sidecar_path(old).rename(sidecar_path(new))
    renamed_model = stored.model.rename(stored.model.with_name("아무 이름.bangmodel"))
    renamed_report = stored.report.rename(stored.report.with_name("보고서 이름.json"))
    library = lib.build_library(stored.loc)
    assert library.counts() == {"dataset": 1, "module_pair": 2, "main_ai": 1, "report": 1, "replay": 1}
    assert _by_kind(library, "dataset")[0].path == renamed_dataset
    pair = next(entry for entry in _by_kind(library, "module_pair") if entry.id == RUN_GOOD)
    assert (pair.intent_path, pair.emotion_path) == (new_intent, new_emotion)
    assert _by_kind(library, "main_ai")[0].path == renamed_model
    assert _by_kind(library, "report")[0].path == renamed_report
    # Links still resolve by id, not by file name.
    model = _by_kind(library, "main_ai")[0]
    assert pair in [target for _ref, target in model.source_links]


def test_legacy_files_without_info_are_listed(tmp_path: Path, bangmodel: Path) -> None:
    loc = lib.LibraryLocations.under(tmp_path)
    _pair(loc.model_root, RUN_OLD, with_info=False)
    (loc.report_dirs[0]).mkdir(parents=True)
    (loc.report_dirs[0] / "old-report.json").write_text(json.dumps({"status": "finished"}), encoding="utf-8")
    (loc.main_ai_dirs[0]).mkdir(parents=True)
    (loc.main_ai_dirs[0] / "legacy.bangmodel").write_bytes(bangmodel.read_bytes())
    loc.replay_dirs[0].mkdir(parents=True)
    record = replay_store.build_replay_record(
        mode="human_random", mode_label="Human vs Random", ai_versions=None, winning_team="SHERIFF", winners=[0],
        human_player=0, seed=1, observations=[{"event_sequence": 0}],
    )
    replay_store.save_replay(record, loc.replay_dirs[0], filename="plain.json")
    library = lib.build_library(loc)
    assert library.counts() == {"dataset": 0, "module_pair": 1, "main_ai": 1, "report": 1, "replay": 1}
    pair = _by_kind(library, "module_pair")[0]
    assert pair.id == RUN_OLD and pair.name == RUN_OLD and pair.info is None
    assert _by_kind(library, "report")[0].name == "old-report" and _by_kind(library, "report")[0].info is None
    legacy_replay = _by_kind(library, "replay")[0]
    assert dict(legacy_replay.condition_facts)["mode"] == "Human vs Random"
    assert dict(legacy_replay.summary_facts)["winner"] == "SHERIFF"


def test_other_json_kinds_in_the_reports_folder_are_not_reports(stored: SimpleNamespace) -> None:
    other = stored.loc.report_dirs[0] / "baseline.json"
    other.write_text("{}", encoding="utf-8")
    write_info(other, new_info("baseline", id="base-1", created_by="main_ai_training"))
    assert len(_by_kind(lib.build_library(stored.loc), "report")) == 1


# --- links ----------------------------------------------------------------------------------------


def test_source_and_used_by_links_resolve_by_id(stored: SimpleNamespace) -> None:
    library = lib.build_library(stored.loc)
    dataset = _by_kind(library, "dataset")[0]
    good = next(e for e in _by_kind(library, "module_pair") if e.id == RUN_GOOD)
    bad = next(e for e in _by_kind(library, "module_pair") if e.id == RUN_BAD)
    model = _by_kind(library, "main_ai")[0]
    report = _by_kind(library, "report")[0]

    assert [target for _ref, target in good.source_links] == [dataset]
    assert [target for _ref, target in model.source_links] == [good, good]  # intent and emotion refs
    assert dataset.used_by == [good, bad] or set(dataset.used_by) == {good, bad}
    assert good.used_by == [model]  # listed once although two references point at it
    assert model.used_by == []
    # A job's report shares the id of what it produced.
    assert good.related == [report] and report.related == [good]
    assert library.resolve(source_ref(KIND_DATASET, DATASET_ID)) is dataset
    assert library.resolve(source_ref(KIND_DATASET, "unknown")) is None
    assert library.resolve({"kind": "weird", "id": DATASET_ID}) is None


def test_unresolved_sources_are_kept_as_plain_references(stored: SimpleNamespace) -> None:
    import shutil

    shutil.rmtree(stored.dataset)
    library = lib.build_library(stored.loc)
    good = next(e for e in _by_kind(library, "module_pair") if e.id == RUN_GOOD)
    assert len(good.source_links) == 1 and good.source_links[0][1] is None
    assert good.source_links[0][0]["id"] == DATASET_ID
    html = view.entry_details_html(good)
    assert tr("library.detail.not_stored") in html


def test_refined_model_shows_base_model(stored: SimpleNamespace, bangmodel: Path) -> None:
    refined = _main_ai(stored.loc.main_ai_dirs[0] / "refined.bangmodel", bangmodel, id="h7777777-refined", refined=True,
                       created_at="2026-11-13T08:00:00+09:00")
    info = read_info(refined)
    info.sources = [source_ref(KIND_MAIN_AI, MODEL_ID, "model")]
    write_info(refined, info)
    library = lib.build_library(stored.loc)
    entry = next(e for e in library.entries if e.id == "h7777777-refined")
    assert entry.tags == ("refined",) and ("base_model", "model") in entry.condition_facts
    assert ("inference_accuracy", "0.620") in entry.summary_facts
    base = next(e for e in library.entries if e.id == MODEL_ID)
    assert entry in base.used_by and entry.source_links[0][1] is base


# --- view logic -----------------------------------------------------------------------------------------


def _entry(library: lib.Library, kind: str, id: str | None = None) -> lib.LibraryEntry:
    return next(e for e in _by_kind(library, kind) if id is None or e.id == id)


def test_action_states_per_kind(stored: SimpleNamespace) -> None:
    library = lib.build_library(stored.loc)
    dataset, model = _entry(library, "dataset"), _entry(library, "main_ai")
    good, bad = _entry(library, "module_pair", RUN_GOOD), _entry(library, "module_pair", RUN_BAD)
    replay, report = _entry(library, "replay"), _entry(library, "report")

    states = view.action_states([dataset])
    assert states[view.ACTION_MODULE_TRAINING_DATA].enabled and states[view.ACTION_REFINEMENT_DATA].enabled
    assert not states[view.ACTION_PLAY].enabled and states[view.ACTION_SAVE_COPY].enabled
    assert states[view.ACTION_EDIT_NOTES].enabled and states[view.ACTION_OPEN_FOLDER].enabled

    states = view.action_states([good])
    assert all(states[a].enabled for a in (view.ACTION_MAIN_AI_TRAIN, view.ACTION_RESUME_MODULES, view.ACTION_EVALUATE))
    states = view.action_states([bad])
    assert not states[view.ACTION_MAIN_AI_TRAIN].enabled and states[view.ACTION_MAIN_AI_TRAIN].reason
    assert states[view.ACTION_RESUME_MODULES].enabled  # an incompatible pair can still be trained further

    states = view.action_states([model])
    assert all(states[a].enabled for a in (view.ACTION_PLAY, view.ACTION_CONTINUE_PPO, view.ACTION_REFINE_BASE,
                                           view.ACTION_COLLECT, view.ACTION_EVALUATE))
    assert view.action_states([replay])[view.ACTION_VIEW_REPLAY].enabled
    assert not view.action_states([replay])[view.ACTION_SAVE_COPY].enabled  # no save plan for replays
    assert view.action_states([report])[view.ACTION_OPEN_REPORT].enabled

    saved = lib.LibraryEntry(kind="dataset", key="k", id="x", name="n", created_epoch=0, location="saved", path=Path("."))
    assert not view.action_states([saved])[view.ACTION_SAVE_COPY].enabled


def test_action_states_for_selections(stored: SimpleNamespace) -> None:
    library = lib.build_library(stored.loc)
    dataset = _entry(library, "dataset")
    second = lib.LibraryEntry(kind="dataset", key="k2", id="x", name="n", created_epoch=0, location="local", path=Path("."))
    states = view.action_states([dataset, second])
    assert states[view.ACTION_MODULE_TRAINING_DATA].enabled and states[view.ACTION_REFINEMENT_DATA].enabled
    assert not states[view.ACTION_EDIT_NOTES].enabled and not states[view.ACTION_OPEN_FOLDER].enabled
    mixed = view.action_states([dataset, _entry(library, "replay")])
    # Only delete works on a mixed selection (everything else needs one kind).
    assert [action for action, state in mixed.items() if state.enabled] == [view.ACTION_DELETE]
    nothing = view.action_states([])
    assert not any(state.enabled for state in nothing.values()) and nothing[view.ACTION_PLAY].reason


def test_default_actions_per_kind(stored: SimpleNamespace) -> None:
    library = lib.build_library(stored.loc)
    assert view.default_action([_entry(library, "dataset")]) == view.ACTION_MODULE_TRAINING_DATA
    assert view.default_action([_entry(library, "module_pair", RUN_GOOD)]) == view.ACTION_MAIN_AI_TRAIN
    # An incompatible pair falls back to continuing module training.
    assert view.default_action([_entry(library, "module_pair", RUN_BAD)]) == view.ACTION_RESUME_MODULES
    assert view.default_action([_entry(library, "main_ai")]) == view.ACTION_PLAY
    assert view.default_action([_entry(library, "replay")]) == view.ACTION_VIEW_REPLAY
    assert view.default_action([_entry(library, "report")]) == view.ACTION_OPEN_REPORT
    assert view.default_action([]) is None
    assert view.default_action([_entry(library, "replay"), _entry(library, "report")]) is None


def test_presets_for_every_apply_action(stored: SimpleNamespace) -> None:
    library = lib.build_library(stored.loc)
    dataset, good, model = _entry(library, "dataset"), _entry(library, "module_pair", RUN_GOOD), _entry(library, "main_ai")
    assert view.build_preset(view.ACTION_MODULE_TRAINING_DATA, [dataset]) == TrainingPreset(
        "module_training", dataset_paths=(str(dataset.path),))
    assert view.build_preset(view.ACTION_REFINEMENT_DATA, [dataset]) == TrainingPreset(
        "inference_refinement", dataset_paths=(str(dataset.path),))
    assert view.build_preset(view.ACTION_RESUME_MODULES, [good]) == TrainingPreset("module_training", module_run_id=RUN_GOOD)
    assert view.build_preset(view.ACTION_MAIN_AI_TRAIN, [good]) == TrainingPreset("main_ai_training", module_run_id=RUN_GOOD)
    assert view.build_preset(view.ACTION_CONTINUE_PPO, [model]) == TrainingPreset(
        "main_ai_training", bangmodel_path=str(model.path))
    assert view.build_preset(view.ACTION_REFINE_BASE, [model]) == TrainingPreset(
        "inference_refinement", bangmodel_path=str(model.path))
    assert view.build_preset(view.ACTION_COLLECT, [model]) == TrainingPreset(
        "data_collection", players="main1-dummy4", bangmodel_path=str(model.path))
    for action in (view.ACTION_PLAY, view.ACTION_VIEW_REPLAY, view.ACTION_OPEN_REPORT, view.ACTION_EDIT_NOTES):
        assert view.build_preset(action, [model]) is None
    assert view.build_preset(view.ACTION_MAIN_AI_TRAIN, []) is None


def test_kind_filter_values() -> None:
    assert view.kinds_for_filter(None) is None and view.kinds_for_filter("all") is None
    assert view.kinds_for_filter("models") == ("module_pair", "main_ai")
    assert view.kinds_for_filter("replay") == ("replay",)


@pytest.mark.parametrize("language", ["en", "ko"])
def test_table_rows_and_details_text(stored: SimpleNamespace, language: str) -> None:
    original = current_language()
    set_language(language)
    try:
        library = lib.build_library(stored.loc)
        for entry in library.entries:
            row = view.table_row(entry)
            assert len(row.cells) == len(view.TABLE_COLUMNS) and row.key == entry.key
            assert row.cells[0] == tr(f"library.kind.{entry.kind}") and row.cells[1] == entry.name
            assert not any(text.startswith("library.") for text in row.cells)  # every key has a translation
            html = view.entry_details_html(entry, performance="perf text")
            assert entry.name in html and "perf text" in html and "library.detail" not in html
            assert "\u2014" not in html and "library.fact" not in html
        model = _entry(library, "main_ai")
        html = view.entry_details_html(model)
        good = _entry(library, "module_pair", RUN_GOOD)
        assert f'href="entry:{good.key}"' in html  # source link
        assert f'href="entry:{model.key}"' in view.entry_details_html(good)  # used-by link
        assert view.multi_details_html([model, good]).count("<li>") == 2
        text = view.facts_text(good.summary_facts)
        assert "0.810" in text and not text.startswith("library.")
    finally:
        set_language(original)


def test_details_html_escapes_untrusted_text(stored: SimpleNamespace) -> None:
    dataset = _entry(lib.build_library(stored.loc), "dataset")
    dataset.name = "<script>alert(1)</script>"
    dataset.notes = "<b>x</b>"
    html = view.entry_details_html(dataset)
    assert "<script>" not in html and "&lt;script&gt;" in html and "<b>x</b>" not in html


def test_report_text_and_save_plans(stored: SimpleNamespace, tmp_path: Path) -> None:
    library = lib.build_library(stored.loc)
    text = view.report_text(_entry(library, "report").path)
    assert "[info]" in text and "[status]" in text and "finished" in text
    assert view.report_text(tmp_path / "missing.json")  # the error text, not an exception
    saved_root = tmp_path / "copy-target"
    assert view.save_plan_for(_entry(library, "dataset"), saved_root=saved_root).category == "datasets"
    assert view.save_plan_for(_entry(library, "module_pair", RUN_GOOD), saved_root=saved_root).category == "modules"
    assert view.save_plan_for(_entry(library, "main_ai"), saved_root=saved_root).category == "main_ai"
    assert view.save_plan_for(_entry(library, "replay")) is None


def test_default_locations_use_project_folders() -> None:
    resolved = lib.LibraryLocations().resolved()
    from bang_rl.paths import SAVED_MAIN_AI_DIR, SAVED_MODULES_DIR, SAVED_REPORTS_DIR, SAVED_REPLAYS_DIR

    assert resolved.saved_modules == SAVED_MODULES_DIR and resolved.saved_main_ai == SAVED_MAIN_AI_DIR
    assert resolved.saved_reports == SAVED_REPORTS_DIR and resolved.saved_replays == SAVED_REPLAYS_DIR
    assert SAVED_MAIN_AI_DIR not in resolved.main_ai_dirs
    # Naming a folder explicitly drops the matching saved/ folder, as the listers do.
    custom = lib.LibraryLocations(model_root=Path("x"), report_dirs=(Path("r"),)).resolved()
    assert custom.saved_modules is None and custom.saved_reports is None and custom.saved_replays is not None
