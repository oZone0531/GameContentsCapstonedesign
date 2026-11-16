"""Give files that were generated before the info/naming rule an info block and a readable name.

``plan_migration`` only looks (dry run); ``apply_plan`` embeds the info, writes
sidecars and renames; ``undo_migration`` renames everything back using the log
written under ``artifacts/migrations/``. Nothing is ever deleted, files that
already carry an info block are skipped (idempotent), and anything that cannot
be read is reported and left as it is.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from bang_rl.artifact_builders import baseline_info, dataset_info, replay_info
from bang_rl.artifact_info import (
    ArtifactInfo,
    KIND_DATASET,
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI,
    KIND_REPORT,
    KIND_TENSOR_CACHE,
    new_info,
    now_iso,
    read_info,
    sidecar_path,
    source_ref,
    write_info,
)
from bang_rl.paths import PROJECT_ROOT

LOG_VERSION = 1
_UUID_NAMESPACE = uuid.UUID("6f1c2f58-4a34-4e0b-9d41-5a3d0f2f7b11")
_REPLAY_STAMP = re.compile(r"(\d{8})-(\d{6})(?:-(\d{6}))?")


@dataclass(slots=True)
class MigrationEntry:
    kind: str
    source: Path
    target: Path | None
    action: str  # "migrate" or "skip"
    reason: str = ""
    info: ArtifactInfo | None = field(default=None, repr=False)
    companions: list[tuple[Path, Path]] = field(default_factory=list)

    def line(self, root: Path) -> str:
        def rel(path: Path) -> str:
            try:
                return str(path.relative_to(root))
            except ValueError:
                return str(path)

        if self.action == "skip":
            return f"SKIP    {self.kind:<13} {rel(self.source)}  ({self.reason})"
        return f"MIGRATE {self.kind:<13} {rel(self.source)}  ->  {self.target.name if self.target else '?'}"


@dataclass(slots=True)
class MigrationPlan:
    root: Path
    entries: list[MigrationEntry] = field(default_factory=list)

    @property
    def to_migrate(self) -> list[MigrationEntry]:
        return [entry for entry in self.entries if entry.action == "migrate"]

    @property
    def skipped(self) -> list[MigrationEntry]:
        return [entry for entry in self.entries if entry.action == "skip"]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.to_migrate:
            counts[entry.kind] = counts.get(entry.kind, 0) + 1
        return counts

    def render(self) -> str:
        lines = [entry.line(self.root) for entry in self.entries]
        summary = ", ".join(f"{kind}: {count}" for kind, count in sorted(self.counts().items())) or "nothing to migrate"
        lines.append(f"-- {len(self.to_migrate)} to migrate ({summary}), {len(self.skipped)} skipped")
        return "\n".join(lines)


# --------------------------------------------------------------------- helpers


def _iso_from_mtime(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return now_iso()
    return datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stable_id(*parts: str) -> str:
    return uuid.uuid5(_UUID_NAMESPACE, "|".join(parts)).hex


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _skip(kind: str, source: Path, reason: str) -> MigrationEntry:
    return MigrationEntry(kind=kind, source=source, target=None, action="skip", reason=reason)


def _target_for(entry_info: ArtifactInfo, source: Path, suffix: str) -> Path:
    return source.with_name(f"{entry_info.name}{suffix}")


def _finish(entry: MigrationEntry, info: ArtifactInfo, suffix: str, *, sidecar: bool) -> MigrationEntry:
    target = _target_for(info, entry.source, suffix)
    info.name = target.name[: -len(suffix)] if suffix else target.name
    entry.info = info
    entry.target = target
    if target.exists() and target != entry.source:
        entry.action, entry.reason = "skip", f"target already exists: {target.name}"
    if sidecar:
        entry.companions.append((sidecar_path(entry.source), sidecar_path(target)))
    return entry


def _report_for(job_id: str, reports: list[tuple[Path, Any]]) -> Mapping[str, Any] | None:
    for _path, data in reports:
        if isinstance(data, Mapping) and str(data.get("job_id") or "") == job_id:
            return data
    return None


# --------------------------------------------------------------------- modules


def _module_infos(root: Path, reports: list[tuple[Path, Any]], dataset_ids: dict[str, str]) -> list[MigrationEntry]:
    import torch

    entries: list[MigrationEntry] = []
    model_root = root / "bang_rl" / "models"
    files = [path for folder in (model_root / "intent", model_root / "emotion") if folder.is_dir() for path in sorted(folder.glob("*.phase2model"))]
    for path in files:
        if read_info(path) is not None:
            entries.append(_skip("module", path, "already has info"))
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = _mapping(_mapping(payload).get("metadata"))
            artifact_type = str(_mapping(payload).get("artifact_type"))
        except Exception as exc:  # noqa: BLE001 - reported, file left as it is
            entries.append(_skip("module", path, f"unreadable: {exc}"[:120]))
            continue
        kind = {"phase2_intent_inference": KIND_INTENT, "phase2_emotion_inference": KIND_EMOTION}.get(artifact_type)
        if kind is None:
            entries.append(_skip("module", path, f"unsupported artifact type {artifact_type}"))
            continue
        run_meta = _mapping(metadata.get("run_metadata"))
        snapshot = _mapping(metadata.get("job_snapshot"))
        job_id = str(run_meta.get("job_id") or snapshot.get("job_id") or "")
        if not job_id:
            stem = path.stem
            job_id = stem.split("-", 1)[1] if "-" in stem else stem
        report = _report_for(job_id, reports)
        report_snapshot = _mapping(report.get("snapshot")) if report is not None else {}
        created = report_snapshot.get("captured_at") or snapshot.get("captured_at") or _iso_from_mtime(path)
        training = _mapping(metadata.get("training"))
        test = _mapping(training.get("test_metrics"))
        split = _mapping(metadata.get("split"))
        best = {}
        for record in training.get("metrics", ()) or ():
            if isinstance(record, Mapping) and int(record.get("epoch", -1)) == int(training.get("best_epoch", 0) or 0):
                best = record
        conditions: dict[str, Any] = {
            "total_epochs": training.get("total_epochs", training.get("epochs_completed")),
            "epochs_configured": training.get("epochs_configured"),
            "batch_size": training.get("batch_size"),
            "seed": training.get("seed"),
            "resumed_from": training.get("resumed_from"),
            "config_hash": str(metadata.get("config_hash") or "")[:12] or None,
            "train_episodes": split.get("train_episode_count"),
            "validation_episodes": split.get("validation_episode_count"),
            "test_episodes": split.get("test_episode_count"),
            "emotion_schema": metadata.get("emotion_schema_version"),
        }
        summary: dict[str, Any] = {
            "epochs_trained": training.get("epochs_completed"),
            "best_epoch": training.get("best_epoch"),
            "best_validation_loss": training.get("best_validation_loss"),
        }
        if kind == KIND_INTENT:
            summary.update(validation_accuracy=best.get("validation_accuracy"), test_accuracy=test.get("accuracy"))
        else:
            summary.update(
                validation_accuracy=best.get("validation_emotion_accuracy"),
                validation_mse=best.get("validation_mse"),
                test_mse=test.get("mse"),
            )
        sources: list[dict[str, Any]] = []
        selection = _mapping(_mapping(_mapping(report).get("progress")).get("metrics")).get("dataset_selection") if report else None
        for shard in _mapping(selection).get("selected_source_shard_ids", ()) or ():
            folder = Path(str(shard)).parts[0] if len(Path(str(shard)).parts) > 1 else ""
            if folder and folder in dataset_ids:
                ref = source_ref(KIND_DATASET, dataset_ids[folder], folder)
                if ref not in sources:
                    sources.append(ref)
        info = new_info(
            kind,
            id=job_id,
            created_by="module_training",
            created_at=str(created),
            conditions={key: value for key, value in conditions.items() if value is not None},
            summary={key: value for key, value in summary.items() if value is not None},
            sources=sources,
            notes="",
        )
        info.conditions["migrated_from"] = path.name
        entry = MigrationEntry(kind="module", source=path, target=None, action="migrate")
        entries.append(_finish(entry, info, ".phase2model", sidecar=True))
    return entries


# --------------------------------------------------------------------- reports


def _report_entries(root: Path, reports: list[tuple[Path, Any]]) -> list[MigrationEntry]:
    entries: list[MigrationEntry] = []
    for path, data in reports:
        if not isinstance(data, Mapping):
            entries.append(_skip("report", path, "not a JSON object"))
            continue
        if isinstance(data.get("info"), Mapping):
            entries.append(_skip("report", path, "already has info"))
            continue
        job_id = str(data.get("job_id") or path.stem)
        snapshot = _mapping(data.get("snapshot"))
        progress = _mapping(data.get("progress"))
        metadata = _mapping(data.get("metadata"))
        workflow = str(metadata.get("workflow") or snapshot.get("mode") or "training")
        created = snapshot.get("captured_at") or _iso_from_mtime(path)
        hashes = _mapping(metadata.get("hashes"))
        conditions = {
            "workflow": workflow,
            "seed": snapshot.get("seed"),
            "split_seed": snapshot.get("split_seed"),
            "config_hash": str(hashes.get("config_snapshot_sha256") or "")[:12] or None,
            "workers": _mapping(metadata.get("hardware")).get("workers"),
            "migrated_from": path.name,
        }
        summary = {
            "status": data.get("status"),
            "games_completed": progress.get("episodes_completed"),
            "win_rate": progress.get("win_rate"),
            "active_training_seconds": progress.get("active_training_seconds"),
        }
        info = new_info(
            KIND_REPORT,
            id=job_id,
            created_by=workflow,
            created_at=str(created),
            conditions={key: value for key, value in conditions.items() if value is not None},
            summary={key: value for key, value in summary.items() if value is not None},
        )
        entries.append(_finish(MigrationEntry(kind="report", source=path, target=None, action="migrate"), info, ".json", sidecar=False))
    # Baseline caches.
    baselines = root / "reports" / "baselines"
    if baselines.is_dir():
        for path in sorted(baselines.glob("*.json")):
            data = _load_json(path)
            if not isinstance(data, Mapping):
                entries.append(_skip("baseline", path, "not a JSON object"))
            elif isinstance(data.get("info"), Mapping):
                entries.append(_skip("baseline", path, "already has info"))
            else:
                key = str(data.get("key") or _stable_id(path.name))
                info = baseline_info(
                    id=key,
                    games=int(data.get("games") or 0),
                    label="dummy+random",
                    conditions={"base_seed": data.get("base_seed"), "migrated_from": path.name},
                    created_at=_iso_from_mtime(path),
                )
                info.summary = {
                    name: {"win_rate": _mapping(values).get("win_rate")}
                    for name, values in _mapping(data.get("baselines")).items()
                }
                entries.append(_finish(MigrationEntry(kind="baseline", source=path, target=None, action="migrate"), info, ".json", sidecar=False))
    return entries


# --------------------------------------------------------------------- replays


def _replay_created_at(path: Path) -> str:
    match = _REPLAY_STAMP.search(path.stem)
    if match:
        try:
            moment = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
            return moment.astimezone().isoformat(timespec="seconds")
        except ValueError:
            pass
    return _iso_from_mtime(path)


def _replay_entries(root: Path) -> list[MigrationEntry]:
    entries: list[MigrationEntry] = []
    folder = root / "bang_rl" / "replays"
    if not folder.is_dir():
        return entries
    for path in sorted(folder.glob("*.json")):
        data = _load_json(path)
        if not isinstance(data, dict):
            entries.append(_skip("replay", path, "not a replay JSON object"))
            continue
        if isinstance(data.get("info"), Mapping):
            entries.append(_skip("replay", path, "already has info"))
            continue
        info = replay_info(
            data,
            id=_stable_id("replay", path.name),
            created_at=_replay_created_at(path),
            created_by="gui_game",
        )
        info.conditions["migrated_from"] = path.name
        entry = MigrationEntry(kind="replay", source=path, target=None, action="migrate")
        _finish(entry, info, ".json", sidecar=False)
        trace = path.with_suffix(".txt")
        if trace.is_file() and entry.target is not None:
            entry.companions.append((trace, entry.target.with_suffix(".txt")))
        entries.append(entry)
    return entries


# ------------------------------------------------------------------ main AI


def _main_ai_entries(root: Path) -> list[MigrationEntry]:
    from bang_rl.training.final_inference_artifact import read_final_artifact_metadata

    entries: list[MigrationEntry] = []
    for folder in (root / "artifacts", root / "bang_rl" / "models" / "main_ai"):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.bangmodel")):
            if read_info(path) is not None:
                entries.append(_skip("main_ai", path, "already has info"))
                continue
            try:
                _manifest, metadata = read_final_artifact_metadata(path)
            except Exception as exc:  # noqa: BLE001
                entries.append(_skip("main_ai", path, f"unreadable: {exc}"[:120]))
                continue
            main = _mapping(metadata.get("main_ai_training"))
            run_meta = _mapping(metadata.get("run_metadata"))
            evaluation = _mapping(main.get("evaluation"))
            scores = _mapping(main.get("normalized_scores"))
            source = _mapping(main.get("source"))
            sources = []
            for key, kind in (("intent_path", KIND_INTENT), ("emotion_path", KIND_EMOTION)):
                if source.get(key):
                    sources.append(source_ref(kind, str(source.get("module_job_id") or Path(str(source[key])).stem), Path(str(source[key])).name))
            info = new_info(
                KIND_MAIN_AI,
                id=str(run_meta.get("job_id") or path.stem),
                created_by="main_ai_training",
                created_at=str(metadata.get("created_at") or _iso_from_mtime(path)),
                conditions={
                    "training_games": main.get("training_games"),
                    "source_origin": source.get("origin"),
                    "migrated_from": path.name,
                },
                summary={
                    "win_rate": evaluation.get("win_rate"),
                    "normalized_score": scores.get("mean"),
                    "ppo_updates": main.get("ppo_updates"),
                },
                sources=sources,
            )
            entries.append(_finish(MigrationEntry(kind="main_ai", source=path, target=None, action="migrate"), info, ".bangmodel", sidecar=True))
    return entries


# ----------------------------------------------------------------- datasets/caches


def _dataset_entries(root: Path) -> tuple[list[MigrationEntry], dict[str, str]]:
    entries: list[MigrationEntry] = []
    ids: dict[str, str] = {}
    for base in (root / "datasets" / "canonical", root / "bang_rl" / "datasets" / "canonical"):
        if not base.is_dir():
            continue
        for folder in sorted(path for path in base.iterdir() if path.is_dir()):
            shards = sorted(folder.rglob("shard-*.parquet"))
            if not shards:
                continue
            existing = read_info(folder)
            if existing is not None:
                ids[folder.name] = existing.id
                entries.append(_skip("dataset", folder, "already has info"))
                continue
            summary = _mapping(_load_json(folder / "collection_summary.json"))
            identifier = folder.name if re.fullmatch(r"[0-9a-f-]{32,36}", folder.name) else _stable_id("dataset", folder.name)
            ids[folder.name] = identifier
            games = int(summary.get("games") or 0)
            if not games:
                games = len(shards)
            info = dataset_info(
                id=identifier,
                games=games,
                base_seed=int(summary.get("base_seed") or 0),
                created_by="data_collection",
                created_at=_iso_from_mtime(shards[0]),
                action_mode=str(summary.get("action_mode") or "deterministic"),
                belief=summary.get("belief_source"),
                extra_conditions={"migrated_from": folder.name},
            )
            if summary:
                from bang_rl.artifact_builders import dataset_summary

                info.summary = dataset_summary(summary)
            entry = MigrationEntry(kind="dataset", source=folder, target=None, action="migrate")
            entries.append(_finish(entry, info, "", sidecar=False))
    return entries, ids


def _cache_entries(root: Path, dataset_ids: dict[str, str]) -> list[MigrationEntry]:
    entries: list[MigrationEntry] = []
    for base in (root / "datasets" / "cache", root / "bang_rl" / "datasets" / "cache"):
        if not base.is_dir():
            continue
        for folder in sorted(path for path in base.iterdir() if path.is_dir()):
            manifests = sorted(folder.glob("cache-*.manifest.json"))
            if not manifests:
                continue
            if read_info(folder) is not None:
                entries.append(_skip("tensor_cache", folder, "already has info"))
                continue
            splits: dict[str, int] = {}
            for manifest in manifests:
                data = _mapping(_load_json(manifest))
                splits[str(data.get("split") or manifest.name)] = len(data.get("episode_ids") or ())
            identifier = folder.name if re.fullmatch(r"[0-9a-f-]{32,36}", folder.name) else _stable_id("cache", folder.name)
            info = new_info(
                KIND_TENSOR_CACHE,
                id=identifier,
                created_by="module_training",
                created_at=_iso_from_mtime(manifests[0]),
                conditions={"migrated_from": folder.name},
                summary={"splits": splits},
            )
            entries.append(_finish(MigrationEntry(kind="tensor_cache", source=folder, target=None, action="migrate"), info, "", sidecar=False))
    return entries


# ------------------------------------------------------------------ plan / apply


def plan_migration(root: str | Path | None = None) -> MigrationPlan:
    """Look at the existing data and list what would change (changes nothing)."""

    base = Path(root) if root is not None else PROJECT_ROOT
    reports_dir = base / "reports"
    reports = [
        (path, _load_json(path)) for path in sorted(reports_dir.glob("*.json"))
    ] if reports_dir.is_dir() else []
    plan = MigrationPlan(root=base)
    datasets, dataset_ids = _dataset_entries(base)
    plan.entries += datasets
    plan.entries += _cache_entries(base, dataset_ids)
    plan.entries += _module_infos(base, reports, dataset_ids)
    plan.entries += _main_ai_entries(base)
    plan.entries += _report_entries(base, reports)
    plan.entries += _replay_entries(base)
    return plan


def _embed(entry: MigrationEntry) -> None:
    assert entry.info is not None
    write_info(entry.source, entry.info)
    if entry.kind == "dataset":
        summary = entry.source / "collection_summary.json"
        if summary.is_file():
            write_info(summary, entry.info)


def apply_plan(plan: MigrationPlan, log_dir: str | Path | None = None) -> Path:
    """Embed info, write sidecars and rename. Returns the path of the undo log."""

    directory = Path(log_dir) if log_dir is not None else plan.root / "artifacts" / "migrations"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = directory / f"migration-{stamp}.json"
    counter = 1
    while log_path.exists():
        log_path = directory / f"migration-{stamp}-{counter}.json"
        counter += 1
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in plan.entries:
        if entry.action != "migrate" or entry.info is None or entry.target is None:
            continue
        try:
            _embed(entry)
            moves = [(entry.source, entry.target)] + [
                pair for pair in entry.companions if pair[0].exists()
            ]
            done: list[tuple[Path, Path]] = []
            try:
                for old, new in moves:
                    if old != new:
                        os.rename(old, new)
                        done.append((old, new))
            except OSError:
                for old, new in reversed(done):
                    os.rename(new, old)
                raise
            records.append(
                {
                    "kind": entry.kind,
                    "id": entry.info.id,
                    "moves": [{"old": str(old), "new": str(new)} for old, new in done],
                }
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
            errors.append({"source": str(entry.source), "error": str(exc)})
    log = {
        "log_version": LOG_VERSION,
        "created_at": now_iso(),
        "root": str(plan.root),
        "entries": records,
        "errors": errors,
        "skipped": [{"source": str(item.source), "reason": item.reason} for item in plan.skipped],
        "note": "Undo with: python migrate_artifact_names.py --undo <this file>. Embedded info blocks stay in the files; only names are restored.",
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log_path


def undo_migration(log_path: str | Path) -> list[tuple[Path, Path]]:
    """Rename everything back to the names recorded in ``log_path``."""

    log = json.loads(Path(log_path).read_text(encoding="utf-8"))
    restored: list[tuple[Path, Path]] = []
    for record in reversed(log.get("entries", [])):
        for move in reversed(record.get("moves", [])):
            old, new = Path(move["old"]), Path(move["new"])
            if new.exists() and not old.exists():
                os.rename(new, old)
                restored.append((new, old))
    return restored


__all__ = [
    "MigrationEntry",
    "MigrationPlan",
    "apply_plan",
    "plan_migration",
    "undo_migration",
]
