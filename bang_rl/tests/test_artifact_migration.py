"""Retroactive info + naming for files generated before the naming rule."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from bang_rl.artifact_info import read_info
from bang_rl.artifact_migration import apply_plan, plan_migration, undo_migration
from bang_rl.gui import replay_store
from bang_rl.training.module_artifacts import (
    list_saved_module_runs,
    main_ai_compatibility,
    summarize_module_run,
)

JOB = "b8ecacce-beb6-4008-add2-5a518e263155"
JOB2 = "03acca08-e47b-44ad-96d4-a438ee254e2b"
REPLAY_NAME = "replay-human_dummy-20261109-124346-421378.json"


def _legacy_module(path: Path, kind: str, job: str, epochs: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_type": f"phase2_{kind}_inference",
        "run_metadata": {"job_id": job},
        "job_snapshot": {"job_id": job},
        "config_hash": "abcdef0123456789",
        "architecture": {"role_count": 4},
        "training": {
            "epochs_completed": epochs, "total_epochs": epochs, "best_epoch": 2, "batch_size": 64, "seed": 3,
            "metrics": [{"epoch": 2, "validation_accuracy": 0.9, "validation_emotion_accuracy": 0.8}],
            "test_metrics": {"accuracy": 0.7, "mse": 0.01},
        },
        "split": {"train_episode_count": 8, "validation_episode_count": 2, "test_episode_count": 2},
    }
    torch.save(
        {
            "artifact_type": f"phase2_{kind}_inference",
            "artifact_format_version": "1",
            "metadata": metadata,
            "module_metadata": {},
            "state_dict": {"w": torch.arange(3.0)},
            "training_mode": False,
        },
        path,
    )


@pytest.fixture()
def legacy_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    _legacy_module(root / "bang_rl" / "models" / "intent" / f"intent-{JOB}.phase2model", "intent", JOB)
    _legacy_module(root / "bang_rl" / "models" / "emotion" / f"emotion-{JOB}.phase2model", "emotion", JOB)
    _legacy_module(root / "bang_rl" / "models" / "intent" / f"intent-{JOB2}.phase2model", "intent", JOB2, epochs=1)
    _legacy_module(root / "bang_rl" / "models" / "emotion" / f"emotion-{JOB2}.phase2model", "emotion", JOB2, epochs=1)
    reports = root / "reports"
    reports.mkdir(parents=True)
    (reports / f"{JOB}.json").write_text(
        json.dumps(
            {"job_id": JOB, "status": "finished", "snapshot": {"mode": "module_training", "seed": 3, "captured_at": "2026-11-10T02:40:00+09:00"},
             "progress": {"episodes_completed": 12, "win_rate": None, "active_training_seconds": 61.5}, "metadata": {"workflow": "module_training"}},
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    (reports / "baselines").mkdir()
    (reports / "baselines" / "baselines-abcdef.json").write_text(
        json.dumps({"key": "abcdef0123456789abcdef", "games": 250, "base_seed": 9, "baselines": {"dummy": {"win_rate": 0.5}, "random": {"win_rate": 0.4}}}),
        encoding="utf-8",
    )
    replays = root / "bang_rl" / "replays"
    replays.mkdir(parents=True)
    record = replay_store.build_replay_record(
        mode="human_dummy", mode_label="Human vs Dummy", ai_versions=None, winning_team="OUTLAW", winners=[1],
        human_player=0, seed=3, observations=[{"event_sequence": 0}], terminated=True,
    )
    (replays / REPLAY_NAME).write_text(json.dumps(record), encoding="utf-8")
    (replays / REPLAY_NAME.replace(".json", ".txt")).write_text("trace", encoding="utf-8")
    dataset = root / "datasets" / "canonical" / "9a8b7c6d-0000-4000-8000-00000000abcd"
    dataset.mkdir(parents=True)
    (dataset / "shard-0000.parquet").write_bytes(b"x")
    (dataset / "collection_summary.json").write_text(
        json.dumps({"games": 40, "base_seed": 11, "action_mode": "deterministic", "belief_source": "rule", "win_rate": {}}), encoding="utf-8"
    )
    cache = root / "datasets" / "cache" / "5e4d3c2b-0000-4000-8000-00000000dcba"
    cache.mkdir(parents=True)
    (cache / "cache-train.manifest.json").write_text(json.dumps({"split": "train", "episode_ids": ["a", "b"]}), encoding="utf-8")
    (cache / "cache-train.npz").write_bytes(b"npz")
    (cache / "split_manifest.json").write_text("{}", encoding="utf-8")
    return root


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_dry_run_plans_everything_and_changes_nothing(legacy_project: Path) -> None:
    before = _tree(legacy_project)
    plan = plan_migration(legacy_project)
    assert plan.counts() == {"module": 4, "report": 1, "baseline": 1, "replay": 1, "dataset": 1, "tensor_cache": 1}
    text = plan.render()
    assert "MIGRATE module" in text and "20261109-1243_replay_human1-dummy4_" in text
    assert _tree(legacy_project) == before


def test_apply_migrates_names_info_and_keeps_everything_working(legacy_project: Path) -> None:
    plan = plan_migration(legacy_project)
    log = apply_plan(plan)
    assert log.parent == legacy_project / "artifacts" / "migrations" and log.is_file()

    models = legacy_project / "bang_rl" / "models"
    runs = list_saved_module_runs(models, saved_root=legacy_project / "none")
    assert {run.job_id for run in runs} == {JOB, JOB2}
    by_id = {run.job_id: run for run in runs}
    run = by_id[JOB]
    assert run.intent_path and run.emotion_path and JOB not in run.intent_path.name
    assert run.intent_path.name.startswith("20261110-0240_intent_e5_") and run.intent_path.name.endswith("b8ecac.phase2model")
    assert run.emotion_path.name.startswith("20261110-0240_emotion_e5_")
    assert run.created_at.startswith("2026-11-10T02:40")  # time taken from the old report
    assert main_ai_compatibility(run)[1] in (None, "emotion_schema")  # still evaluated on the renamed pair
    info = read_info(run.intent_path)
    assert info.kind == "intent_module" and info.conditions["total_epochs"] == 5 and info.summary["best_epoch"] == 2
    assert info.conditions["migrated_from"] == f"intent-{JOB}.phase2model"
    assert (run.intent_path.with_name(run.intent_path.name + ".about.md")).is_file()
    assert not list(models.rglob(f"*{JOB}*"))  # old names are gone
    summary = summarize_module_run(run, reports_dir=legacy_project / "reports")
    assert summary["intent"]["epochs_completed"] == 5 and summary["emotion"]["kind"] == "emotion"
    assert summary["report_path"] is not None and JOB not in Path(summary["report_path"]).name

    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    assert next(iter(report)) == "info" and report["job_id"] == JOB and report["info"]["id"] == JOB
    assert report["info"]["created_at"] == "2026-11-10T02:40:00+09:00"

    replays = list((legacy_project / "bang_rl" / "replays").glob("*.json"))
    assert len(replays) == 1 and replays[0].name.startswith("20261109-1243_replay_human1-dummy4_")
    loaded = replay_store.load_replay(replays[0])
    assert loaded["seed"] == 3 and read_info(replays[0]).conditions["players"] == "human1-dummy4"
    assert replays[0].with_suffix(".txt").read_text(encoding="utf-8") == "trace"  # trace follows its replay

    dataset = next((legacy_project / "datasets" / "canonical").iterdir())
    assert "_dataset_dummy5_40g_" in dataset.name
    assert read_info(dataset).conditions["games"] == 40 and (dataset / "ABOUT.md").is_file()
    assert (dataset / "shard-0000.parquet").read_bytes() == b"x"
    cache = next((legacy_project / "datasets" / "cache").iterdir())
    assert "_cache_" in cache.name and (cache / "cache-train.npz").read_bytes() == b"npz"
    assert (cache / "cache-train.manifest.json").read_text(encoding="utf-8").startswith('{"split"')
    baseline = next((legacy_project / "reports" / "baselines").iterdir())
    assert "_baseline_250g_" in baseline.name and json.loads(baseline.read_text(encoding="utf-8"))["key"].startswith("abcdef")

    entries = json.loads(log.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 9 and all(entry["moves"] for entry in entries)


def test_second_run_is_a_noop_and_unreadable_files_are_skipped(legacy_project: Path) -> None:
    apply_plan(plan_migration(legacy_project))
    again = plan_migration(legacy_project)
    assert again.to_migrate == []
    assert all(entry.reason == "already has info" for entry in again.skipped)
    snapshot = _tree(legacy_project)
    log = apply_plan(again)
    assert json.loads(log.read_text(encoding="utf-8"))["entries"] == []
    broken = legacy_project / "bang_rl" / "models" / "intent" / "broken.phase2model"
    broken.write_bytes(b"not a pickle")
    plan = plan_migration(legacy_project)
    entry = next(item for item in plan.entries if item.source == broken)
    assert entry.action == "skip" and "unreadable" in entry.reason
    assert broken.read_bytes() == b"not a pickle"
    assert {k: v for k, v in _tree(legacy_project).items() if "migrations" not in k and "broken" not in k} == {
        k: v for k, v in snapshot.items() if "migrations" not in k
    }


def test_undo_restores_old_names(legacy_project: Path) -> None:
    before = sorted(str(path.relative_to(legacy_project)) for path in legacy_project.rglob("*") if path.is_file())
    log = apply_plan(plan_migration(legacy_project))
    assert sorted(str(path.relative_to(legacy_project)) for path in legacy_project.rglob("*") if path.is_file()) != before
    restored = undo_migration(log)
    assert restored
    after = sorted(
        str(path.relative_to(legacy_project))
        for path in legacy_project.rglob("*")
        if path.is_file() and "migrations" not in path.parts and not path.name.endswith((".about.md", "ABOUT.md"))
    )
    assert after == [name for name in before]
    # Info stays embedded, so the old names still pair by id.
    runs = list_saved_module_runs(legacy_project / "bang_rl" / "models", saved_root=legacy_project / "none")
    assert {run.job_id for run in runs} == {JOB, JOB2}


def test_target_collisions_are_skipped_not_overwritten(legacy_project: Path) -> None:
    plan = plan_migration(legacy_project)
    entry = next(item for item in plan.to_migrate if item.kind == "replay")
    entry.target.write_text("occupied", encoding="utf-8")
    again = plan_migration(legacy_project)
    skipped = next(item for item in again.entries if item.kind == "replay" and item.source.name == REPLAY_NAME)
    assert skipped.action == "skip" and "already exists" in skipped.reason
    assert entry.target.read_text(encoding="utf-8") == "occupied"


def test_cli_dry_run_and_apply(legacy_project: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "migrate_artifact_names.py"
    import sys

    dry = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--root", str(legacy_project)], capture_output=True, text=True, encoding="utf-8"
    )
    assert dry.returncode == 0 and "MIGRATE" in dry.stdout
    assert (legacy_project / "reports" / f"{JOB}.json").is_file()
    done = subprocess.run(
        [sys.executable, str(script), "--root", str(legacy_project)], capture_output=True, text=True, encoding="utf-8"
    )
    assert done.returncode == 0 and "Undo log" in done.stdout
    assert not (legacy_project / "reports" / f"{JOB}.json").exists()
    log = next((legacy_project / "artifacts" / "migrations").glob("migration-*.json"))
    undone = subprocess.run(
        [sys.executable, str(script), "--undo", str(log)], capture_output=True, text=True, encoding="utf-8"
    )
    assert undone.returncode == 0 and (legacy_project / "reports" / f"{JOB}.json").is_file()
