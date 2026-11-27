"""Application-facing training workflows.

The GUI only selects a workflow.  This module keeps dataset collection and
Phase-2 supervised training out of the Qt layer.  Main-Agent PPO training lives
in :mod:`bang_rl.training.main_ai_training`.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from bang_rl.artifact_builders import (
    cache_info,
    dataset_info,
    info_ref,
    module_infos,
    run_created_at,
)
from bang_rl.artifact_info import KIND_DATASET, KIND_EMOTION, KIND_INTENT, write_info
from bang_rl.config import SystemConfig
from bang_rl.data.game_collection import (
    FORMULA_LABEL_SCHEMA,
    collect_games,
    formula_label_emotion_config,
)
from bang_rl.agents.dummy_agent.emotion_formula import LABEL_NAMES
from bang_rl.data.parquet_writer import ParquetCanonicalShardWriter
from bang_rl.data.splitter import EpisodeSplitter
from bang_rl.data.tensor_cache import NPZTensorCacheLoader, NPZTensorCacheWriter
from bang_rl.exceptions import DataError, TrainingError
from bang_rl.paths import PROJECT_ROOT, SAVED_DATASETS_DIR
from bang_rl.training.emotion_sl_trainer import EmotionInferenceSLTrainer, EmotionSLConfig
from bang_rl.training.sl_trainer import IntentInferenceSLTrainer, IntentSLConfig, ROLE_TARGET_NAMES
from bang_rl.agents.main_agent.intent_inference import IntentInferenceModule
from bang_rl.training.module_artifacts import (
    artifact_total_epochs,
    load_emotion_for_resume,
    load_intent_for_resume,
)

DEFAULT_CANONICAL_PATH = "datasets/canonical"
TRAINING_MODE_DATA_COLLECTION = "data_collection"
TRAINING_MODE_MODULE_TRAINING = "module_training"
TRAINING_MODE_MAIN_AI = "main_ai_training"
TRAINING_MODE_INFERENCE_REFINEMENT = "inference_refinement"
TRAINING_MODES = (
    TRAINING_MODE_DATA_COLLECTION,
    TRAINING_MODE_MODULE_TRAINING,
    TRAINING_MODE_MAIN_AI,
    TRAINING_MODE_INFERENCE_REFINEMENT,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _canonical_shards(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("shard-*.parquet") if path.is_file()))


def _episode_emotion_contract(episode: Any) -> str:
    """Return the strict emotion contract carried by one canonical episode."""

    schemas: set[str] = set()
    dimensions: set[int] = set()
    names: set[tuple[str, ...]] = set()
    for step in getattr(episode, "steps", ()):
        for label in getattr(step, "emotion_labels", {}).values():
            if not isinstance(label, Mapping):
                continue
            values = label.get("values")
            if values is None:
                continue
            try:
                dimensions.add(len(values))
            except TypeError:
                dimensions.add(-1)
            schemas.add(str(label.get("emotion_schema_version", "unknown")))
            raw_names = label.get("emotion_names", ())
            if isinstance(raw_names, (list, tuple)):
                names.add(tuple(str(name) for name in raw_names))
            else:
                names.add(())
    if (
        schemas == {FORMULA_LABEL_SCHEMA}
        and dimensions == {9}
        and names.issubset({tuple(LABEL_NAMES)})
    ):
        return f"{FORMULA_LABEL_SCHEMA}/9"
    if not schemas and not dimensions:
        return "unknown/0"
    schema_text = ",".join(sorted(schemas)) or "unknown"
    dimension_text = ",".join(str(value) for value in sorted(dimensions)) or "0"
    return f"{schema_text}/{dimension_text}"


def _read_canonical_dataset(
    root: Path,
    extra_roots: Sequence[Path] = (),
    only_folders: Sequence[Path] | None = None,
    *,
    limited_folders: Sequence[Path] = (),
    limit_episodes: int | None = None,
    limit_seed: int = 0,
):
    """Read only nine-emotion formula episodes from a canonical root.

    ``only_folders`` (explicitly chosen dataset folders) replaces the scan of
    ``root`` and ``extra_roots``: only shards inside those folders are read.
    Dummy-only and main1+dummy4 datasets can be mixed freely; the union is
    deduplicated by episode id.


    A canonical parent can contain historical collection-job directories. Jobs
    with an incompatible emotion contract are excluded from module training;
    mixing different label layouts would create an invalid supervised task.
    The returned selection diagnostics make that exclusion visible in reports.

    ``extra_roots`` (the user-kept ``saved/datasets`` folder) are scanned too.
    Shards are found by file pattern, so renamed dataset folders work, and an
    episode that exists in two places is used once (deduplicated by episode id).

    ``limit_episodes`` caps how many episodes are taken from the shards inside
    ``limited_folders`` (together). Those shards are read in an order shuffled
    by ``limit_seed`` and reading stops at the cap, so a huge dataset is never
    loaded whole into memory; other folders are read completely.
    """

    def relative(path: Path) -> str:
        return _shard_id(path, root, extra_roots)

    if only_folders is not None:
        shards = ()
        for folder in only_folders:
            if not Path(folder).is_dir():
                raise DataError(f"selected dataset folder does not exist: {folder}")
            shards = tuple(sorted({*shards, *_canonical_shards(Path(folder))}))
        if not shards:
            raise DataError("no canonical Parquet shards found in the selected datasets")
    else:
        shards = _canonical_shards(root)
        for extra in extra_roots:
            if extra.is_dir():
                shards = tuple(sorted({*shards, *_canonical_shards(extra)}))
        if not shards:
            raise DataError(f"no canonical Parquet shards found under {root}")
    episodes_by_id: dict[str, Any] = {}
    players_count: dict[str, int] = {}
    selected_shards: set[Path] = set()
    duplicate_count = 0
    excluded_count = 0
    excluded_contracts: dict[str, int] = {}
    limited_roots = [Path(folder).resolve() for folder in limited_folders]

    def is_limited(shard: Path) -> bool:
        resolved = shard.resolve()
        return any(folder in resolved.parents for folder in limited_roots)

    limited_taken = 0
    skipped_limited = 0
    if limit_episodes is not None and limited_roots:
        import random

        free = [shard for shard in shards if not is_limited(shard)]
        capped = [shard for shard in shards if is_limited(shard)]
        random.Random(int(limit_seed)).shuffle(capped)
        shards = (*free, *capped)
    for shard in shards:
        limited_shard = limit_episodes is not None and is_limited(shard)
        if limited_shard and limited_taken >= limit_episodes:
            skipped_limited += 1
            continue
        for episode in ParquetCanonicalShardWriter.read_shard(shard):
            if limited_shard:
                if limited_taken >= limit_episodes:
                    break
                limited_taken += 1
            contract = _episode_emotion_contract(episode)
            if contract != f"{FORMULA_LABEL_SCHEMA}/9":
                excluded_count += 1
                excluded_contracts[contract] = excluded_contracts.get(contract, 0) + 1
                continue
            selected_shards.add(shard)
            episode_id = str(episode.episode_id)
            prior = episodes_by_id.get(episode_id)
            if prior is None:
                episodes_by_id[episode_id] = episode
                players = str((getattr(episode, "metadata", None) or {}).get("players") or "dummy5")
                players_count[players] = players_count.get(players, 0) + 1
                continue
            if prior.to_dict(include_private=True) != episode.to_dict(include_private=True):
                raise DataError(
                    f"conflicting duplicate episode_id {episode_id!r} in canonical dataset; "
                    "the same episode id contains different data"
                )
            # Repeated deterministic collection jobs can contain the same
            # episode content. Keep one copy rather than training on it twice.
            duplicate_count += 1
    if not episodes_by_id:
        contracts = ", ".join(
            f"{contract}: {count}" for contract, count in sorted(excluded_contracts.items())
        ) or "no readable emotion labels"
        raise DataError(
            f"no nine-emotion formula episodes found under {root}; "
            f"excluded incompatible data ({contracts}). Set dataset.canonical_path "
            "to a collection folder with emotion_schema_version="
            f"{FORMULA_LABEL_SCHEMA} and nine values, or run data collection again."
        )
    selection = {
        "requested_emotion_schema": FORMULA_LABEL_SCHEMA,
        "requested_emotion_dimension": 9,
        "selected_episode_count": len(episodes_by_id),
        "selected_episodes_by_players": dict(sorted(players_count.items())),
        "excluded_incompatible_episode_count": excluded_count,
        "excluded_incompatible_contracts": dict(sorted(excluded_contracts.items())),
        "selected_source_shard_ids": [relative(path) for path in sorted(selected_shards)],
        "scanned_source_shard_count": len(shards),
    }
    if limit_episodes is not None and limited_roots:
        selection["limited_episode_cap"] = int(limit_episodes)
        selection["limited_episodes_taken"] = limited_taken
        selection["limited_shards_skipped"] = skipped_limited
    return tuple(episodes_by_id.values()), tuple(sorted(selected_shards)), duplicate_count, selection


SAVED_SHARD_PREFIX = "saved:"


def _shard_id(path: Path, root: Path, extra_roots: Sequence[Path] = ()) -> str:
    """Portable shard id: relative to ``root``, or ``saved:<relative>`` for an extra root."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        pass
    for extra in extra_roots:
        try:
            return SAVED_SHARD_PREFIX + str(path.relative_to(extra))
        except ValueError:
            continue
    return str(path)


def _dataset_source_refs(root: Path, selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Datasets (folders under ``root``) that contributed shards, as info references."""

    folders: dict[str, Path] = {}
    for shard_id in selection.get("selected_source_shard_ids", ()):
        text = str(shard_id)
        if text.startswith(SAVED_SHARD_PREFIX):
            shard = SAVED_DATASETS_DIR / text[len(SAVED_SHARD_PREFIX):]
            folder = shard.parent
        elif Path(text).is_absolute():
            folder = Path(text).parent
        else:
            shard = Path(text)
            folder = root / shard.parts[0] if len(shard.parts) > 1 else root
        folders.setdefault(str(folder), folder)
    return [info_ref(folder, KIND_DATASET) for folder in folders.values()]


def _training_seed(run: Any) -> int:
    return int(run.config.seed if run.config.seed is not None else run.system_config.seed)


def _publish_collection_progress(run: Any, done: int, total: int) -> None:
    """Publish collection progress to the live TrainingProgress snapshot."""

    message = f"Collected {int(done)}/{int(total)} games"
    with run.lock:
        run.episodes_completed = int(done)
        run.metrics["collection_completed"] = int(done)
        run.metrics["collection_total"] = int(total)
        run.logs.append({"message": message})
    run.logger.event(
        "data_collection_progress",
        completed=int(done),
        total=int(total),
        job_id=run.handle.job_id,
    )


def run_data_collection(handle: Any, run: Any) -> Mapping[str, Any]:
    """Collect formula-labelled all-Dummy games into an ignored dataset folder."""

    games = int(run.config.episodes)
    if games < 1:
        raise ValueError("data collection requires at least one game")
    config = run.system_config
    root = _project_path(config.dataset.canonical_path)
    players = str(getattr(run.config, "collection_players", "dummy5") or "dummy5")
    if players == "main1-dummy4":
        return _run_main_ai_data_collection(handle, run, games, config, root)
    if players != "dummy5":
        raise TrainingError(f"unsupported collection players: {players!r} (dummy5 or main1-dummy4)")
    # The folder is named by the naming rule (time, kind, players, games, id).
    info = dataset_info(
        id=str(handle.job_id),
        games=games,
        base_seed=_training_seed(run),
        created_by=TRAINING_MODE_DATA_COLLECTION,
        created_at=run_created_at(run),
        extra_conditions={"workers": getattr(getattr(config, "hardware", None), "workers", None)},
    )
    output = root / info.name
    summary = collect_games(
        config,
        games,
        output,
        base_seed=_training_seed(run),
        action_mode="deterministic",
        episode_id_prefix=f"job-{handle.job_id}",
        progress=lambda done, total: _publish_collection_progress(run, done, total),
        info=info,
    )
    status = summary.get("termination_status", {})
    normal = int(status.get("normal", 0))
    truncated = int(status.get("truncated", 0))
    return {
        "output_path": str(output),
        "environment_steps": int(summary.get("total_steps", 0)),
        "episodes_completed": games,
        "normal_episodes": normal,
        "truncated_episodes": truncated,
        "win_rate": None,
        "metrics": {
            "workflow": TRAINING_MODE_DATA_COLLECTION,
            "dataset_path": str(output),
            "emotion_label_schema": summary.get("emotion_label_schema"),
            "emotion_label_dimension": summary.get("emotion_label_dimension"),
            "role_class_count": 4,
            "raw_role_count": 5,
            "emotion_dimension": 9,
            "termination_status": status,
        },
        "logs": ({"message": f"dataset collected: {output}"},),
    }


def _base_model_file(run: Any) -> Path:
    """The explicitly chosen ``.bangmodel`` (a folder would silently pick the newest one)."""

    value = getattr(run.config, "base_model_path", None)
    path = Path(str(value)).expanduser() if value else None
    if path is None or not path.is_file():
        raise TrainingError(
            "메인 AI 모델(.bangmodel) 파일을 선택하세요. "
            "Choose a main AI model (.bangmodel) file."
        )
    return path


def _run_main_ai_data_collection(handle: Any, run: Any, games: int, config: SystemConfig, root: Path) -> Mapping[str, Any]:
    """Collect "main1+dummy4" games: the chosen model plays among four dummies."""

    from bang_rl.training.final_inference_artifact import load_final_inference_artifact
    from bang_rl.training.main_ai_collection import (
        MAIN_ACTION_MODES,
        collect_main_ai_games,
        main_ai_dataset_info,
        policy_from_artifact,
    )

    model_path = _base_model_file(run)
    action_mode = str(getattr(run.config, "main_action_mode", "deterministic") or "deterministic")
    if action_mode not in MAIN_ACTION_MODES:
        raise TrainingError(f"main_action_mode must be one of {MAIN_ACTION_MODES}")
    policy = policy_from_artifact(load_final_inference_artifact(model_path))
    info = main_ai_dataset_info(
        games=games,
        base_seed=_training_seed(run),
        created_by=TRAINING_MODE_DATA_COLLECTION,
        created_at=run_created_at(run),
        action_mode=action_mode,
        model_path=model_path,
        identifier=str(handle.job_id),
        extra_conditions={"workers": getattr(getattr(config, "hardware", None), "workers", None)},
    )
    output = root / info.name
    controller = _TrainingStopController(run)
    summary = collect_main_ai_games(
        config,
        policy,
        games,
        output,
        base_seed=_training_seed(run),
        action_mode=action_mode,
        progress=lambda done, total: _publish_collection_progress(run, done, total),
        should_stop=controller.should_stop,
        info=info,
        source_model_path=model_path,
        episode_id_prefix=f"job-{handle.job_id}-main1",
    )
    status = summary.get("termination_status", {})
    return {
        "output_path": str(output),
        "environment_steps": int(summary.get("total_steps", 0)),
        "episodes_completed": int(summary.get("games", 0)),
        "normal_episodes": int(status.get("normal", 0)),
        "truncated_episodes": int(status.get("truncated", 0)),
        "win_rate": summary.get("main_win_rate"),
        "metrics": {
            "workflow": TRAINING_MODE_DATA_COLLECTION,
            "dataset_path": str(output),
            "players": "main1-dummy4",
            "viewpoint": summary.get("viewpoint"),
            "main_action_mode": action_mode,
            "collection_rows": summary.get("rows"),
            "collection_main_decisions": summary.get("main_decisions"),
            "main_win_rate": summary.get("main_win_rate"),
            "emotion_label_schema": summary.get("emotion_label_schema"),
            "emotion_label_dimension": summary.get("emotion_label_dimension"),
            "role_class_count": 4,
            "raw_role_count": 5,
            "emotion_dimension": 9,
            "termination_status": status,
            "source_model": str(model_path),
        },
        "logs": (
            {
                "message": (
                    f"main1+dummy4 dataset collected: {output} "
                    f"(games {summary.get('games')}, rows {summary.get('rows')}, "
                    f"main decisions {summary.get('main_decisions')})"
                )
            },
        ),
    }


def _publish_module_stage(run: Any, stage: str, message: str, **fields: Any) -> None:
    """Publish a preprocessing-stage checkpoint for module training.

    Preprocessing (canonical read, split, cache build) can run for a long time
    before the first training metric arrives. Without these events the GUI shows
    no activity and the job looks stalled even though it is working. This mirrors
    the data-collection progress contract so the live snapshot stays current.
    """

    with run.lock:
        run.metrics["module_preprocessing_stage"] = stage
        run.logs.append({"message": message})
    run.logger.event(
        "module_training_progress",
        stage=stage,
        job_id=run.handle.job_id,
        **fields,
    )


class _TrainingStopController:
    """Cooperative stop signal for module training, polled once per batch.

    ``should_stop()`` is True when the user pressed Stop (job status
    ``cancelling``/``cancelled``) or when the optional time budget has elapsed.
    While the job is paused it blocks until resumed (or cancelled) and pushes
    the deadline back by the paused time, so Pause never counts against the
    budget and never ends training.
    """

    _CANCEL_STATES = frozenset({"cancelling", "cancelled"})

    def __init__(self, run: Any, *, budget_seconds: float | None = None) -> None:
        self._run = run
        self._deadline = None if budget_seconds is None else time.monotonic() + float(budget_seconds)

    def _status(self) -> str:
        lock = getattr(self._run, "lock", None)
        if lock is None:
            return str(getattr(self._run, "status", "running"))
        with lock:
            return str(getattr(self._run, "status", "running"))

    def should_stop(self) -> bool:
        paused_at: float | None = None
        while True:
            status = self._status()
            if status in self._CANCEL_STATES:
                return True
            if status != "paused":
                break
            if paused_at is None:
                paused_at = time.monotonic()
            time.sleep(0.1)
        if paused_at is not None and self._deadline is not None:
            self._deadline += time.monotonic() - paused_at
        return self._deadline is not None and time.monotonic() >= self._deadline


def _keep_graph_point(run: Any, key: str) -> bool:
    """Decide whether this batch becomes a graph point.

    Every batch is counted, but only every ``graph_point_interval``-th one (the
    first included) is stored, so the live metrics and the plot stay small no
    matter how many batches a long run trains. Without a configured interval
    (e.g. direct calls in tests) every batch is kept.
    """

    config = getattr(run, "config", None)
    try:
        interval = max(1, int(getattr(config, "graph_point_interval", 1)))
    except (TypeError, ValueError):
        interval = 1
    counts = getattr(run, "_graph_batch_counts", None)
    if counts is None:
        counts = {}
        setattr(run, "_graph_batch_counts", counts)
    index = counts.get(key, 0)
    counts[key] = index + 1
    return index % interval == 0


def _publish_module_metric(run: Any, record: Mapping[str, Any]) -> None:
    """Keep lightweight batch accuracy points available to the GUI."""

    if str(record.get("event")) == "epoch":
        with run.lock:
            if "intent" in str(record.get("phase", "")):
                if isinstance(record.get("validation_accuracy"), (int, float)):
                    run.metrics["intent_validation_accuracy"] = float(record["validation_accuracy"])
                if isinstance(record.get("validation_accuracy_by_role"), (list, tuple)):
                    run.metrics["intent_validation_accuracy_by_role"] = list(record["validation_accuracy_by_role"])
                if isinstance(record.get("validation_role_names"), (list, tuple)):
                    run.metrics["intent_role_names"] = list(record["validation_role_names"])
            if "emotion" in str(record.get("phase", "")):
                if isinstance(record.get("validation_emotion_accuracy"), (int, float)):
                    run.metrics["emotion_validation_accuracy"] = float(record["validation_emotion_accuracy"])
                if isinstance(record.get("validation_emotion_accuracy_by_dimension"), (list, tuple)):
                    run.metrics["emotion_validation_accuracy_by_dimension"] = list(record["validation_emotion_accuracy_by_dimension"])
                if isinstance(record.get("validation_mse"), (int, float)):
                    run.metrics["emotion_validation_mse"] = float(record["validation_mse"])
                if isinstance(record.get("validation_personality_accuracy"), (int, float)):
                    run.metrics["personality_validation_accuracy"] = float(record["validation_personality_accuracy"])
                if isinstance(record.get("personality_names"), (list, tuple)):
                    run.metrics["personality_names"] = list(record["personality_names"])
        return
    if str(record.get("event")) != "batch":
        return
    phase = str(record.get("phase", ""))
    with run.lock:
        if "intent" in phase and isinstance(record.get("accuracy"), (int, float)):
            if not _keep_graph_point(run, "intent"):
                return
            run.metrics.setdefault("intent_accuracy_points", []).append(
                float(record["accuracy"])
            )
            if isinstance(record.get("accuracy_by_role"), (list, tuple)):
                run.metrics.setdefault("intent_accuracy_by_role_points", []).append(
                    [float(value) for value in record["accuracy_by_role"]]
                )
            if isinstance(record.get("role_names"), (list, tuple)):
                run.metrics["intent_role_names"] = list(record["role_names"])
        elif "emotion" in phase and isinstance(record.get("accuracy"), (int, float)):
            if not _keep_graph_point(run, "emotion"):
                return
            run.metrics.setdefault("emotion_accuracy_points", []).append(float(record["accuracy"]))
            if isinstance(record.get("personality_accuracy"), (int, float)):
                run.metrics.setdefault("personality_accuracy_points", []).append(
                    float(record["personality_accuracy"])
                )
            if isinstance(record.get("accuracy_by_dimension"), (list, tuple)):
                run.metrics.setdefault("emotion_accuracy_by_dimension_points", []).append(
                    [float(value) for value in record["accuracy_by_dimension"]]
                )


def _history_metric(history: Any, field: str) -> list[float]:
    values: list[float] = []
    for record in history or ():
        if isinstance(record, Mapping) and isinstance(record.get(field), (int, float)):
            values.append(float(record[field]))
    return values


def _last_history_metric(history: Any, field: str) -> float | None:
    values = _history_metric(history, field)
    return values[-1] if values else None


def _cache_diagnostics(loaders: Mapping[str, NPZTensorCacheLoader]) -> dict[str, Any]:
    return {
        split: dict(loader.manifest.get("diagnostics", {}))
        for split, loader in loaders.items()
    }


def _module_max_games(run: Any) -> int | None:
    """Configured cap on games read for module training (None or 0 = no limit)."""

    value = getattr(run.config, "max_games", None)
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _module_subset_seed(run: Any, handle: Any) -> int:
    """Seed for the random game subset: explicit ``subset_seed`` or derived from the job id.

    Different job ids give different subsets, so repeated continue-training
    runs cover the whole dataset over time.
    """

    explicit = getattr(run.config, "subset_seed", None)
    if explicit is not None:
        return int(explicit)
    import hashlib

    return int(hashlib.sha256(str(handle.job_id).encode("utf-8")).hexdigest()[:8], 16)


def run_module_training(handle: Any, run: Any) -> Mapping[str, Any]:
    """Split the canonical dataset, build caches, and train both Phase-2 modules."""

    config = run.system_config
    canonical_root = _project_path(config.dataset.canonical_path)
    _publish_module_stage(
        run,
        "reading_canonical_dataset",
        f"Reading canonical dataset from {canonical_root}",
        canonical_path=str(canonical_root),
    )
    # With the default dataset location the user-kept saved/datasets copies
    # (for example datasets a teammate committed) are used as well.
    # ``dataset_paths`` (chosen dataset folders, dummy5 and/or main1-dummy4,
    # local or saved) replaces the scan of everything; the union is
    # deduplicated by episode id and split by episode.
    chosen = [_project_path(item) for item in (getattr(run.config, "dataset_paths", None) or ())]
    extra_roots: list[Path] = []
    if (chosen or canonical_root == _project_path(DEFAULT_CANONICAL_PATH)) and SAVED_DATASETS_DIR.is_dir():
        extra_roots.append(SAVED_DATASETS_DIR)
    max_games = _module_max_games(run)
    subset_seed = _module_subset_seed(run, handle) if max_games is not None else None
    limit_kwargs: dict[str, Any] = {}
    if max_games is not None:
        limit_kwargs = {
            "limited_folders": tuple(chosen) if chosen else (canonical_root, *extra_roots),
            "limit_episodes": max_games,
            "limit_seed": subset_seed,
        }
    episodes, shards, duplicate_count, dataset_selection = _read_canonical_dataset(
        canonical_root, extra_roots, only_folders=chosen or None, **limit_kwargs
    )
    games_used = int(dataset_selection.get("limited_episodes_taken", len(episodes))) if max_games is not None else None
    dataset_selection["max_games"] = max_games
    dataset_selection["subset_seed"] = subset_seed
    dataset_selection["games_used"] = games_used
    if chosen:
        dataset_selection["chosen_dataset_folders"] = [str(item) for item in chosen]
    _publish_module_stage(
        run,
        "selected_canonical_dataset",
        (
            f"Selected {len(episodes)} episodes from {len(shards)} shard(s) "
            f"(excluded {int(dataset_selection.get('excluded_incompatible_episode_count', 0))}, "
            f"deduplicated {int(duplicate_count)})"
        ),
        selected_episode_count=len(episodes),
        selected_shard_count=len(shards),
        excluded_incompatible_episode_count=int(
            dataset_selection.get("excluded_incompatible_episode_count", 0)
        ),
        deduplicated_episode_count=int(duplicate_count),
    )
    source_refs = _dataset_source_refs(canonical_root, dataset_selection)
    cache_meta = cache_info(run, handle, sources=source_refs, summary={})
    cache_root = _project_path(config.dataset.cache_path) / cache_meta.name
    _publish_module_stage(
        run,
        "splitting_dataset",
        f"Splitting {len(episodes)} episodes into train/validation/test",
        episode_count=len(episodes),
    )
    split = EpisodeSplitter(
        seed=int(run.config.split_seed if run.config.split_seed is not None else _training_seed(run)),
        manifest_path=cache_root / "split_manifest.json",
    ).split(
        episodes,
        source_metadata={"source_shard_ids": [_shard_id(path, canonical_root, extra_roots) for path in shards]},
    )
    _publish_module_stage(
        run,
        "building_tensor_cache",
        f"Building tensor cache at {cache_root}",
        cache_path=str(cache_root),
    )
    cache_artifacts = NPZTensorCacheWriter(
        cache_root,
        emotion_schema_version=FORMULA_LABEL_SCHEMA,
        emotion_dimension=9,
        split_manifest=split.manifest,
        config=config,
    ).write_all(split)
    # ABOUT.md only: the cache's npz/manifest files are hash verified and stay untouched.
    cache_meta.summary = {
        "episodes": len(episodes),
        "splits": {name: len(artifact.manifest.get("episode_ids", ())) for name, artifact in cache_artifacts.items()},
        "deduplicated_episodes": int(duplicate_count),
    }
    write_info(cache_root, cache_meta)
    _publish_module_stage(
        run,
        "cache_ready",
        "Tensor cache ready; starting Phase-2 module training",
        cache_path=str(cache_root),
    )
    loaders = {
        name: NPZTensorCacheLoader(artifact.tensor_path, manifest_path=artifact.manifest_path)
        for name, artifact in cache_artifacts.items()
    }
    sl_epochs = int(config.sl_training.epochs)
    sl_batch = int(config.sl_training.batch_size)

    # Two exclusive ways to bound training:
    #   * epoch mode (default): train exactly ``sl_epochs`` epochs.
    #   * time mode (``train_by_time``): ignore the epoch count and keep
    #     training until the time budget elapses (``max_training_seconds``;
    #     None = unlimited, i.e. until the user presses Stop).
    # In both modes Stop is honoured, even in the middle of an epoch.
    train_by_time = bool(getattr(run.config, "train_by_time", False))
    budget = getattr(run.config, "max_training_seconds", None) if train_by_time else None
    try:
        budget = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None
    if budget is not None and budget <= 0:
        budget = None
    controller = _TrainingStopController(run, budget_seconds=budget)
    if train_by_time:
        _publish_module_stage(
            run,
            "training",
            "Training by time: "
            + (f"{budget / 60.0:g} min" if budget is not None else "unlimited (press Stop to finish)"),
            train_by_time=True,
        )
    else:
        _publish_module_stage(run, "training", f"Training for {sl_epochs} epoch(s)", train_by_time=False)

    # Optional continuation from previously saved modules (weights only; the
    # optimizer starts fresh because .phase2model files do not store it).
    resume_intent = getattr(run.config, "resume_intent_artifact_path", None) or None
    resume_emotion = getattr(run.config, "resume_emotion_artifact_path", None) or None
    intent_model = IntentInferenceModule(role_count=4, seed=_training_seed(run))
    intent_prior = 0
    if resume_intent is not None:
        intent_model, intent_meta = load_intent_for_resume(resume_intent)
        intent_prior = artifact_total_epochs(intent_meta)
        _publish_module_stage(
            run, "resume", f"Continuing intent module from {Path(resume_intent).name}",
            resume_intent_artifact_path=str(resume_intent),
        )
    emotion_kwargs: dict[str, Any] = {"emotion_config": formula_label_emotion_config()}
    emotion_prior = 0
    if resume_emotion is not None:
        emotion_model, emotion_meta = load_emotion_for_resume(resume_emotion)
        emotion_kwargs = {"model": emotion_model}
        emotion_prior = artifact_total_epochs(emotion_meta)
        _publish_module_stage(
            run, "resume", f"Continuing emotion module from {Path(resume_emotion).name}",
            resume_emotion_artifact_path=str(resume_emotion),
        )

    intent_trainer = IntentInferenceSLTrainer(
        loaders["train"],
        loaders["validation"],
        test_cache=loaders["test"],
        config=IntentSLConfig(epochs=sl_epochs, batch_size=sl_batch, seed=_training_seed(run), role_count=4),
        model=intent_model,
        job_snapshot=handle.snapshot,
        config_snapshot=config,
        logger=lambda record: _publish_module_metric(run, record),
    )
    intent_trainer.should_stop = controller.should_stop
    emotion_trainer = EmotionInferenceSLTrainer(
        loaders["train"],
        loaders["validation"],
        test_cache=loaders["test"],
        config=EmotionSLConfig(epochs=sl_epochs, batch_size=sl_batch, seed=_training_seed(run)),
        logger=lambda record: _publish_module_metric(run, record),
        **emotion_kwargs,
    )
    emotion_trainer.should_stop = controller.should_stop
    # A resumed module starts as the best-so-far model, so further training
    # only replaces it when validation loss actually improves.
    if resume_intent is not None:
        intent_trainer.resumed_from = str(resume_intent)
        intent_trainer.prior_epochs = intent_prior
        intent_trainer.adopt_current_weights_as_baseline()
    if resume_emotion is not None:
        emotion_trainer.resumed_from = str(resume_emotion)
        emotion_trainer.prior_epochs = emotion_prior
        emotion_trainer.adopt_current_weights_as_baseline()

    if train_by_time:
        # Alternate the two modules one epoch at a time so both keep improving
        # no matter how long training runs (an unlimited run would otherwise
        # never get past the first module). Stop/deadline is checked per batch.
        while True:
            if not intent_trainer.train_next_epoch() or controller.should_stop():
                break
            if not emotion_trainer.train_next_epoch() or controller.should_stop():
                break
        intent_result = intent_trainer.finalize()
        emotion_result = emotion_trainer.finalize()
    else:
        intent_result = intent_trainer.fit()
        emotion_result = emotion_trainer.fit()

    # Only save a module that actually trained at least one epoch; an immediate
    # Stop must not leave an untrained model that looks like a real result.
    model_root = PROJECT_ROOT / "bang_rl" / "models"
    intent_path: Path | None = None
    emotion_path: Path | None = None
    intent_test = intent_result.test_metrics or {}
    emotion_test = emotion_result.test_metrics or {}
    try:
        intent_meta, emotion_meta = module_infos(
            run,
            handle,
            sources=source_refs,
            intent_total_epochs=intent_prior + len(intent_result.history),
            emotion_total_epochs=emotion_prior + len(emotion_result.history),
            intent_summary={
                "epochs_trained": len(intent_result.history),
                "best_epoch": intent_result.best_epoch,
                "validation_accuracy": _last_history_metric(intent_result.history, "validation_accuracy"),
                "test_accuracy": intent_test.get("accuracy"),
            },
            emotion_summary={
                "epochs_trained": len(emotion_result.history),
                "best_epoch": emotion_result.best_epoch,
                "validation_accuracy": _last_history_metric(emotion_result.history, "validation_emotion_accuracy"),
                "validation_mse": _last_history_metric(emotion_result.history, "validation_mse"),
                "test_mse": emotion_test.get("mse"),
            },
            extra_conditions={
                "emotion_schema": FORMULA_LABEL_SCHEMA,
                "batch_size": sl_batch,
                "dataset_episodes": len(episodes),
                "max_games": max_games,
                "subset_seed": subset_seed,
                "games_used": games_used,
                "dataset_episodes_by_players": dataset_selection.get("selected_episodes_by_players"),
            },
            resumed_from={
                **({"intent": info_ref(resume_intent, KIND_INTENT)} if resume_intent is not None else {}),
                **({"emotion": info_ref(resume_emotion, KIND_EMOTION)} if resume_emotion is not None else {}),
            },
        )
    except Exception as exc:  # noqa: BLE001 - never lose trained models over their description
        run.logger.event("module_info_failed", error=str(exc))
        intent_meta = emotion_meta = None
    if intent_result.history:
        intent_path = model_root / "intent" / (
            f"{intent_meta.name}.phase2model" if intent_meta is not None else f"intent-{handle.job_id}.phase2model"
        )
        intent_trainer.save_artifact(intent_path, test_metrics=intent_result.test_metrics, info=intent_meta)
    if emotion_result.history:
        emotion_path = model_root / "emotion" / (
            f"{emotion_meta.name}.phase2model" if emotion_meta is not None else f"emotion-{handle.job_id}.phase2model"
        )
        emotion_trainer.save_artifact(emotion_path, test_metrics=emotion_result.test_metrics, info=emotion_meta)
    return {
        "output_path": str(cache_root),
        "environment_steps": int(sum(len(ep.steps) for ep in episodes)),
        "episodes_completed": len(episodes),
        "normal_episodes": sum(1 for ep in episodes if ep.termination_status == "normal"),
        "truncated_episodes": sum(1 for ep in episodes if ep.termination_status == "truncated"),
        "metrics": {
            "workflow": TRAINING_MODE_MODULE_TRAINING,
            "canonical_path": str(canonical_root),
            "cache_path": str(cache_root),
            "deduplicated_episode_count": duplicate_count,
            "dataset_selection": dataset_selection,
            "max_games": max_games,
            "subset_seed": subset_seed,
            "games_used": games_used,
            "intent_artifact_path": None if intent_path is None else str(intent_path),
            "emotion_artifact_path": None if emotion_path is None else str(emotion_path),
            "train_by_time": train_by_time,
            "intent_epochs_completed": len(intent_result.history),
            "emotion_epochs_completed": len(emotion_result.history),
            "intent_total_epochs": intent_prior + len(intent_result.history),
            "emotion_total_epochs": emotion_prior + len(emotion_result.history),
            "resumed_intent_from": None if resume_intent is None else str(resume_intent),
            "resumed_emotion_from": None if resume_emotion is None else str(resume_emotion),
            "intent_history": list(intent_result.history),
            "emotion_history": list(emotion_result.history),
            "role_class_count": 4,
            "raw_role_count": 5,
            "emotion_dimension": 9,
            "emotion_schema": FORMULA_LABEL_SCHEMA,
            "intent_final_validation_accuracy": _last_history_metric(intent_result.history, "validation_accuracy"),
            "intent_final_validation_accuracy_by_role": (
                list(intent_result.history[-1].get("validation_accuracy_by_role", ()))
                if intent_result.history else []
            ),
            "intent_role_names": list(intent_result.history[-1].get("validation_role_names", ROLE_TARGET_NAMES)) if intent_result.history else list(ROLE_TARGET_NAMES),
            "emotion_final_validation_mse": _last_history_metric(emotion_result.history, "validation_mse"),
            "emotion_final_validation_accuracy": _last_history_metric(emotion_result.history, "validation_emotion_accuracy"),
            "emotion_final_validation_accuracy_by_dimension": (
                list(emotion_result.history[-1].get("validation_emotion_accuracy_by_dimension", ()))
                if emotion_result.history else []
            ),
            "emotion_accuracy_tolerance": float(emotion_trainer.config.accuracy_tolerance),
            "emotion_dimension_names": list(emotion_trainer.model.emotion_names),
            "personality_trained": bool(emotion_trainer.personality_enabled),
            "personality_names": list(emotion_trainer.model.personality_names),
            "personality_final_validation_accuracy": _last_history_metric(
                emotion_result.history, "validation_personality_accuracy"
            ),
            "intent_gradient_norm_history": _history_metric(intent_result.history, "train_gradient_norm"),
            "emotion_gradient_norm_history": _history_metric(emotion_result.history, "train_gradient_norm"),
            "cache_diagnostics": _cache_diagnostics(loaders),
        },
        "logs": ({"message": f"module training finished: {cache_root}"},),
    }


def run_main_ai_training(handle: Any, run: Any) -> Mapping[str, Any]:
    """Phase-3 PPO training of the main agent against DummyAgents."""

    # Imported here because main_ai_training reuses this module's stop
    # controller and progress helpers.
    from bang_rl.training.main_ai_training import train_main_ai

    return train_main_ai(handle, run)


def run_inference_refinement_training(handle: Any, run: Any) -> Mapping[str, Any]:
    """Refine the inference modules under a fixed PPO (see inference_refinement)."""

    from bang_rl.training.inference_refinement import run_inference_refinement

    return run_inference_refinement(handle, run)


def run_training_workflow(handle: Any, run: Any) -> Mapping[str, Any]:
    mode = str(run.config.mode)
    if mode == TRAINING_MODE_DATA_COLLECTION:
        return run_data_collection(handle, run)
    if mode == TRAINING_MODE_MODULE_TRAINING:
        return run_module_training(handle, run)
    if mode == TRAINING_MODE_MAIN_AI:
        return run_main_ai_training(handle, run)
    if mode == TRAINING_MODE_INFERENCE_REFINEMENT:
        return run_inference_refinement_training(handle, run)
    raise TrainingError(f"unsupported training workflow: {mode!r}")


__all__ = [
    "TRAINING_MODE_DATA_COLLECTION",
    "TRAINING_MODE_MODULE_TRAINING",
    "TRAINING_MODE_MAIN_AI",
    "TRAINING_MODE_INFERENCE_REFINEMENT",
    "TRAINING_MODES",
    "run_training_workflow",
]
