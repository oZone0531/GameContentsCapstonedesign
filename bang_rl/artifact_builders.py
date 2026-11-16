"""Build :class:`~bang_rl.artifact_info.ArtifactInfo` blocks for each workflow output.

Headless helpers shared by the training workflows, the CLI tools and the
migration script, so every producer describes its output the same way.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from bang_rl.artifact_info import (
    ArtifactInfo,
    default_artifact_name,
    KIND_BASELINE,
    KIND_DATASET,
    KIND_EMOTION,
    KIND_INTENT,
    KIND_MAIN_AI,
    KIND_REPLAY,
    KIND_REPORT,
    KIND_TENSOR_CACHE,
    new_info,
    now_iso,
    read_info,
    source_ref,
)

WORKFLOW_LABELS = {
    "data_collection": "data_collection",
    "module_training": "module_training",
    "main_ai_training": "main_ai_training",
    "inference_refinement": "inference_refinement",
}


def config_hash(value: Any) -> str:
    """Short stable hash of a configuration snapshot (for the conditions list)."""

    try:
        text = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return sha256(text.encode("utf-8")).hexdigest()[:12]


def run_created_at(run: Any) -> str:
    return str(getattr(run, "created_at", None) or now_iso())


def run_conditions(run: Any) -> dict[str, Any]:
    """Conditions every training workflow shares (seed, hardware, limits)."""

    options = getattr(run, "config", None)
    system = getattr(run, "system_config", None)
    snapshot = getattr(getattr(run, "handle", None), "snapshot", None)
    conditions: dict[str, Any] = {}
    if options is not None:
        conditions.update(
            seed=getattr(options, "seed", None),
            split_seed=getattr(options, "split_seed", None),
            requested_games=getattr(options, "episodes", None),
            train_by_time=bool(getattr(options, "train_by_time", False)),
            max_training_seconds=getattr(options, "max_training_seconds", None),
        )
    if system is not None:
        conditions["config_seed"] = getattr(system, "seed", None)
        hardware = getattr(system, "hardware", None)
        if hardware is not None:
            conditions["workers"] = getattr(hardware, "workers", None)
    if snapshot is not None:
        conditions["config_hash"] = config_hash(getattr(snapshot, "config_snapshot", None))
    return {key: value for key, value in conditions.items() if value is not None}


# ------------------------------------------------------------------ datasets


def dataset_info(
    *,
    id: str,
    games: int,
    base_seed: int,
    created_by: str,
    created_at: str | None = None,
    players: str = "dummy5",
    action_mode: str = "deterministic",
    belief: str | None = None,
    extra_conditions: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
    notes: str = "",
) -> ArtifactInfo:
    players_text = {"dummy5": "더미 5인", "main1-dummy4": "메인 AI 1 + 더미 4"}.get(players, players)
    conditions: dict[str, Any] = {
        "players": players,
        "players_label": players_text,
        "games": int(games),
        "base_seed": int(base_seed),
        "action_mode": action_mode,
        "belief_source": belief or "default",
    }
    conditions.update(extra_conditions or {})
    return new_info(
        KIND_DATASET,
        id=id,
        created_by=created_by,
        created_at=created_at,
        conditions=conditions,
        summary=summary,
        notes=notes,
        kind_label=f"학습 데이터({players_text})",
    )


def dataset_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Short result block from a ``collection_summary``."""

    status = summary.get("termination_status") or {}
    return {
        "games": summary.get("games"),
        "normal_games": status.get("normal"),
        "truncated_games": status.get("truncated"),
        "win_rate": summary.get("win_rate"),
        "mean_steps": summary.get("mean_steps"),
        "seconds": summary.get("seconds"),
        "emotion_label_schema": summary.get("emotion_label_schema"),
        "shard_count": len(summary.get("shards") or ()),
    }


# ------------------------------------------------------------------- modules


def info_ref(path: str | Path, kind: str) -> dict[str, Any]:
    """Reference to an input artifact: its info id/name when it has one."""

    target = Path(path)
    info = read_info(target) if target.exists() else None
    if info is not None:
        return source_ref(info.kind, info.id, info.name)
    return source_ref(kind, target.stem if target.is_file() else target.name, target.name)


def module_infos(
    run: Any,
    handle: Any,
    *,
    sources: Sequence[Mapping[str, Any]],
    intent_total_epochs: int,
    emotion_total_epochs: int,
    intent_summary: Mapping[str, Any],
    emotion_summary: Mapping[str, Any],
    extra_conditions: Mapping[str, Any] | None = None,
    resumed_from: Mapping[str, Any] | None = None,
) -> tuple[ArtifactInfo, ArtifactInfo]:
    """Info for the intent and emotion module of one training job."""

    created = run_created_at(run)
    base = run_conditions(run)
    base.update(extra_conditions or {})
    out: list[ArtifactInfo] = []
    for kind, total, summary in (
        (KIND_INTENT, intent_total_epochs, intent_summary),
        (KIND_EMOTION, emotion_total_epochs, emotion_summary),
    ):
        conditions = dict(base)
        conditions["total_epochs"] = int(total)
        refs = [dict(item) for item in sources]
        if resumed_from and resumed_from.get(kind):
            conditions["resumed_from"] = resumed_from[kind]
        out.append(
            new_info(
                kind,
                id=str(handle.job_id),
                created_by="module_training",
                created_at=created,
                conditions=conditions,
                summary=summary,
                sources=refs,
            )
        )
    return out[0], out[1]


def cache_info(run: Any, handle: Any, *, sources: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> ArtifactInfo:
    return new_info(
        KIND_TENSOR_CACHE,
        id=str(handle.job_id),
        created_by="module_training",
        created_at=run_created_at(run),
        conditions=run_conditions(run),
        summary=summary,
        sources=[dict(item) for item in sources],
    )


# ------------------------------------------------------------------- reports


def report_info(run: Any, *, sources: Sequence[Mapping[str, Any]] = (), summary: Mapping[str, Any] | None = None) -> ArtifactInfo:
    workflow = str(getattr(getattr(run, "config", None), "mode", "training"))
    conditions = run_conditions(run)
    conditions["workflow"] = workflow
    return new_info(
        KIND_REPORT,
        id=str(run.handle.job_id),
        created_by=workflow,
        created_at=run_created_at(run),
        conditions=conditions,
        summary=summary,
        sources=[dict(item) for item in sources],
    )


def report_summary(run: Any) -> dict[str, Any]:
    return {
        "status": getattr(run, "status", None),
        "games_completed": getattr(run, "episodes_completed", None),
        "win_rate": getattr(run, "win_rate", None),
        "active_training_seconds": round(float(getattr(run, "active_training_seconds", 0.0) or 0.0), 1),
        "error": (getattr(run, "error", None) or {}).get("message") if getattr(run, "error", None) else None,
    }


def report_sources(run: Any) -> list[dict[str, Any]]:
    """Input/output artifacts the run reported in its metrics (by info id)."""

    metrics = getattr(run, "metrics", None) or {}
    refs: list[dict[str, Any]] = []
    for key, kind in (
        ("dataset_path", KIND_DATASET),
        ("canonical_path", KIND_DATASET),
        ("intent_artifact_path", KIND_INTENT),
        ("emotion_artifact_path", KIND_EMOTION),
    ):
        value = metrics.get(key)
        if value:
            path = Path(str(value))
            if path.exists() and read_info(path) is not None:
                refs.append(info_ref(path, kind))
    artifact = getattr(run, "artifact_path", None)
    if artifact and Path(str(artifact)).is_file() and read_info(artifact) is not None:
        refs.append(info_ref(artifact, KIND_MAIN_AI))
    return refs


# ------------------------------------------------------------- main AI model


def main_ai_info(
    run: Any,
    handle: Any,
    *,
    name: str | None,
    training_games: int,
    sources: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    extra_conditions: Mapping[str, Any] | None = None,
    notes: str = "",
    created_by: str = "main_ai_training",
) -> ArtifactInfo:
    conditions = run_conditions(run)
    conditions["training_games"] = int(training_games)
    conditions.update(extra_conditions or {})
    return new_info(
        KIND_MAIN_AI,
        id=str(handle.job_id),
        created_by=created_by,
        created_at=run_created_at(run),
        conditions=conditions,
        summary=summary,
        sources=[dict(item) for item in sources],
        name=name,
        notes=notes,
    )


def main_ai_pending_info(run: Any, handle: Any) -> ArtifactInfo:
    """Info known when the job starts, used to pick the default file name."""

    options = getattr(run, "config", None)
    games = int(getattr(options, "episodes", 0) or 0)
    by_time = bool(getattr(options, "train_by_time", False))
    if str(getattr(options, "mode", "")) == "inference_refinement":
        # A refined model is not trained for ``episodes`` games (that number is
        # games per collection round), so its name carries "refined" instead.
        info = main_ai_info(run, handle, name=None, training_games=0, sources=(), summary={}, created_by="inference_refinement")
        info.conditions.pop("training_games", None)
        info.name = default_artifact_name(info)
        return info
    return main_ai_info(
        run,
        handle,
        name=None,
        training_games=0 if by_time else games,
        sources=(),
        summary={},
    )


# ------------------------------------------------------------------- replays


def replay_info(
    record: Mapping[str, Any],
    *,
    id: str,
    created_at: str | None = None,
    created_by: str = "gui_game",
    notes: str = "",
    name: str | None = None,
) -> ArtifactInfo:
    """Info for a replay record (mode, who played, result)."""

    mode = str(record.get("mode") or "game")
    human = record.get("human_player")
    spectator = mode in {"main_one_dummy_four", "random_five", "dummy_five"}
    players = {
        "human_main_dummy": "human1-main1-dummy3",
        "human_random": "human1-random4",
        "human_dummy": "human1-dummy4",
        "main_one_dummy_four": "main1-dummy4",
        "random_five": "random5",
        "dummy_five": "dummy5",
    }.get(mode, mode)
    conditions: dict[str, Any] = {
        "mode": mode,
        "mode_label": record.get("mode_label"),
        "players": players,
        "seed": record.get("seed"),
        "replay_format": record.get("replay_format", "public"),
        "human_player": None if spectator else human,
        "observations": len(record.get("observations") or ()),
        "spectator": bool(spectator),
    }
    ai_versions = record.get("ai_versions")
    if ai_versions:
        conditions["ai_versions"] = ai_versions
    summary = {
        "winning_team": record.get("winning_team"),
        "winners": record.get("winners"),
        "human_won": record.get("human_won"),
        "outcome": record.get("outcome"),
    }
    return new_info(
        KIND_REPLAY,
        id=id,
        created_by=created_by,
        created_at=created_at,
        conditions=conditions,
        summary=summary,
        notes=notes,
        name=name,
    )


def baseline_info(*, id: str, games: int, label: str = "", conditions: Mapping[str, Any] | None = None, created_at: str | None = None) -> ArtifactInfo:
    merged = {"games": int(games), "label": label}
    merged.update(conditions or {})
    return new_info(
        KIND_BASELINE,
        id=id,
        created_by="main_ai_training",
        created_at=created_at,
        conditions=merged,
    )


__all__ = [
    "baseline_info",
    "cache_info",
    "config_hash",
    "dataset_info",
    "dataset_summary",
    "info_ref",
    "main_ai_info",
    "main_ai_pending_info",
    "module_infos",
    "replay_info",
    "report_info",
    "report_sources",
    "report_summary",
    "run_conditions",
    "run_created_at",
]
